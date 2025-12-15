import os
import math
import random
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ==========================================
# 1. 基础配置与工具函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True # 针对固定输入尺寸加速

# ==========================================
# 2. 损失函数与特殊层
# ==========================================
class ArcMarginProduct(nn.Module):
    """ArcFace Layer for Metric Learning"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: (B, D), label: (B)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        return output

class UncertaintyLoss(nn.Module):
    """Gaussian NLL Loss: 学习数据的不确定性 (Aleatoric Uncertainty)"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        # pred: (B, T, 4) -> [sbp_mu, sbp_logvar, dbp_mu, dbp_logvar]
        # target: (B, T, 2) -> [sbp, dbp]
        mu = pred[:, :, [0, 2]]      
        log_var = pred[:, :, [1, 3]] 
        
        # Loss = 0.5 * (log_var + (y-mu)^2 / e^log_var)
        loss = 0.5 * (log_var + (target - mu)**2 / torch.exp(log_var))
        
        if mask is not None:
            if mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            
            masked_loss = loss * mask

            loss = masked_loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss

# ==========================================
# 3. 模型组件 (精简版 VQVAE + Transformer)
# ==========================================
class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        # 计算注意力权重: T -> 1
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2) # 在时间维度(T)做Softmax
        )

    def forward(self, x):
        # x: (B, C, T)
        # alpha: (B, C, T) 每个通道每个时间点的权重
        alpha = self.attention(x)
        
        # 加权平均 (Mean)
        mean = torch.sum(alpha * x, dim=2) # (B, C)
        
        # 加权方差 (Std) - 捕捉波动的统计特征
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        
        # 拼接均值和方差
        return torch.cat([mean, std], dim=1) # (B, C*2)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, in_channel, 1),
        )

    def forward(self, input):
        return self.conv(input) + input

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 4:
            blocks = [nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv1d(channel // 2, channel, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv1d(channel, channel, 3, padding=1)]
        elif stride == 2:
            blocks = [nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                      nn.Conv1d(channel // 2, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([nn.ConvTranspose1d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                           nn.ConvTranspose1d(channel // 2, out_channel, 4, stride=2, padding=1)])
        elif stride == 2:
            blocks.append(nn.ConvTranspose1d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class VQVAE(nn.Module):
    def __init__(self, in_channel=12, out_channel=4, num_vecs=800, channel=64, n_res_block=2, n_res_channel=32, embed_dim=64):
        super().__init__()
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv1d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, num_vecs)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv1d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, num_vecs)
        self.upsample_t = nn.ConvTranspose1d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(embed_dim + embed_dim, out_channel, channel, n_res_block, n_res_channel, stride=4)

    def forward(self, input):
        # Encode
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 1)
        quant_t, diff_t, _ = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 2, 1)
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 1)
        quant_b, diff_b, _ = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 2, 1)
        
        # Decode
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        return dec, diff_t + diff_b

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(8000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class RadarEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(6, d_model, kernel_size=1)
        self.in_planes = 64
        self.proj_b = nn.Sequential(
            self._make_layer(BasicBlock, 64, 2, stride=2),
            self._make_layer(BasicBlock, 128, 2, stride=2),
        )
        self.quan_b = nn.Conv1d(128, d_model, kernel_size=1)
        self.proj_t = nn.Sequential(
            self._make_layer(BasicBlock, 256, 2, stride=2),
        )
        self.quan_t = nn.Conv1d(256, d_model, kernel_size=1)

    def forward(self, x):
        x_z = self.conv(x)
        x_b = self.proj_b(x_z)
        q_b = self.quan_b(x_b)
        x_t = self.proj_t(x_b)
        q_t = self.quan_t(x_t)
        return q_t.permute(0, 2, 1), q_b.permute(0, 2, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

# === MAIN WRAPPER ===
class RadarModel(nn.Module):
    def __init__(self, ntoken=800, d_model=64, nhead=4, d_hid=128, nlayers=6, dropout=0.1, num_classes=10):
        super().__init__()
        self.rdr_embed = RadarEncoder(d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.rdr_encoder_t = nn.TransformerEncoder(enc_layer, nlayers)
        self.rdr_encoder_b = nn.TransformerEncoder(enc_layer, nlayers)
        self.vqvae = VQVAE(out_channel=4) # SBP_mu, SBP_var, DBP_mu, DBP_var

        # self.id_pool = nn.AdaptiveAvgPool1d(1) 
        self.id_pool = AttentiveStatsPooling(in_dim=d_model * 2)
        self.id_head = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128))
        self.arcface = ArcMarginProduct(128, num_classes)

    def forward(self, rdr_data, labels=None):
        rdr_t, rdr_b = self.rdr_embed(rdr_data)
        rdr_t = self.rdr_encoder_t(self.pos_encoder(rdr_t))
        rdr_b = self.rdr_encoder_b(self.pos_encoder(rdr_b))
        rdr_t = rdr_t.permute(0, 2, 1)
        rdr_b = rdr_b.permute(0, 2, 1)
        
        # Latent Fusion
        rdr_t_up = self.vqvae.upsample_t(rdr_t)
        latent = torch.cat([rdr_t_up, rdr_b], dim=1)

        # Branch 1: ID
        id_feat = self.id_pool(latent).squeeze(-1) 
        id_embedding = self.id_head(id_feat) 
        id_logits = self.arcface(id_embedding, labels) if labels is not None else None

        # Branch 2: BP
        rec = self.vqvae.dec(latent).permute(0, 2, 1)
        return rec, id_logits, id_embedding

# ==========================================
# 4. 数据处理 (时序分割与Dataset)
# ==========================================
def split_dataset_by_session(file_path, train_ratio=0.7):
    """
    智能划分：每个Subject内部按时间排序，前70% Sessions训练，后30%测试。
    """
    print(">>> Analyzing Temporal Sessions...")
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        dirs = [x.decode('utf-8') for x in f["dir"][:]] # 必须包含 'dir' 键
    
    meta_data = {}
    for idx, (sub, d_str) in enumerate(zip(subjects, dirs)):
        if sub not in meta_data: meta_data[sub] = []
        try:
            # 格式: lqy_2025_10_30_15_18_27 -> 取最后6部分解析时间
            date_str = "_".join(d_str.split('_')[-6:]) 
            dt = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
        except:
            dt = datetime.fromtimestamp(idx) # Fallback
        meta_data[sub].append({'time': dt, 'idx': idx, 'dir': d_str})

    train_idxs, test_idxs = [], []
    for sub, sessions in meta_data.items():
        # 按 dir 分组 (Session Level)
        unique_dirs = {} 
        for item in sessions:
            if item['dir'] not in unique_dirs: unique_dirs[item['dir']] = {'time': item['time'], 'idxs': []}
            unique_dirs[item['dir']]['idxs'].append(item['idx'])
        
        # 按时间排序 Session
        sorted_sessions = sorted(unique_dirs.values(), key=lambda x: x['time'])
        n = len(sorted_sessions)
        
        if n == 1:
            train_idxs.extend(sorted_sessions[0]['idxs'])
        else:
            cut = max(1, int(n * train_ratio))
            for i in range(cut): train_idxs.extend(sorted_sessions[i]['idxs'])
            for i in range(cut, n): test_idxs.extend(sorted_sessions[i]['idxs'])
            
    return np.array(train_idxs), np.array(test_idxs)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, index, file_path, global_name_map):
        self.file = file_path
        self.index = index
        self.name_to_id = global_name_map
        self.num_classes = len(global_name_map)
        
        with h5py.File(self.file, "r") as f:
            stats = f['stats']
            self.stats = {
                'max': {k: v for k, v in stats['max_values'].attrs.items()},
                'min': {k: v for k, v in stats['min_values'].attrs.items()},
            }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        real_idx = self.index[idx]
        with h5py.File(self.file, "r") as f:
            # ID
            name = f["subject_name"][real_idx].decode('utf-8')
            label_id = self.name_to_id[name]

            # Data & Normalize
            def get_norm(key):
                d = f[key][real_idx]
                return (d - self.stats['min'][key]) / (self.stats['max'][key] - self.stats['min'][key] + 1e-8)

            rdr = np.stack([
                get_norm('chest_radar_diff'), get_norm('neck_radar_diff'),
                get_norm('chest_heart_interval'), get_norm('neck_heart_interval'),
                get_norm('chest_radar_respiration'), get_norm('pulse_time')
            ], axis=0) # (6, 800)
            sbp = torch.from_numpy(f["sbp"][real_idx]).view(800, 1).float()
            dbp = torch.from_numpy(f["dbp"][real_idx]).view(800, 1).float()
            mask = torch.ones_like(sbp)
            
        return torch.from_numpy(rdr).float(), torch.cat([sbp, dbp, mask], dim=1), torch.tensor(label_id).long()

# ==========================================
# 5. 训练与评估引擎
# ==========================================
def train_epoch(model, loader, opt, crit_bp, crit_id, scaler, device, epoch, writer):
    model.train()
    metrics = {'loss': 0, 'bp': 0, 'id': 0}
    
    pbar = tqdm(loader, desc=f"Ep {epoch} [Train]", ncols=80)
    for i, (x, y, labels) in enumerate(pbar):
        x, y, labels = x.to(device), y.to(device), labels.to(device)
        true_bp, mask = y[:, :, :2], y[:, :, 2:]

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            pred_bp, logits, _ = model(x, labels)
            loss_bp = crit_bp(pred_bp, true_bp, mask)
            loss_id = crit_id(logits, labels)
            loss = 1.0 * loss_bp + 0.5 * loss_id
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        metrics['loss'] += loss.item()
        metrics['bp'] += loss_bp.item()
        metrics['id'] += loss_id.item()
        
        if i % 20 == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(loader) + i)

    return {k: v / len(loader) for k, v in metrics.items()}

def evaluate_verification(model, loader, device):
    """计算 EER 和 ROC"""
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, _, label in tqdm(loader, desc="[Verify]", ncols=80):
            _, _, emb = model(x.to(device))
            embeddings.append(F.normalize(emb, p=2, dim=1).cpu().numpy())
            labels.append(label.cpu().numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # Cosine Sim Matrix
    sim_matrix = np.dot(embeddings, embeddings.T)
    # Ground Truth Matrix
    label_matrix = (labels[:, None] == labels[None, :])
    
    # Extract upper triangle
    tri_idx = np.triu_indices(len(labels), k=1)
    scores = sim_matrix[tri_idx]
    y_true = label_matrix[tri_idx]
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer, auc(fpr, tpr)

def evaluate_regression(model, loader, device, criterion):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    mae_fn = nn.L1Loss(reduction='none')
    
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="[Regress]", ncols=80):
            x, y = x.to(device), y.to(device)
            true_bp, mask = y[:, :, :2], y[:, :, 2:]
            
            pred_bp, _, _ = model(x)
            loss = criterion(pred_bp, true_bp, mask)
            
            # MAE (Using Mean channels 0 and 2)
            mae_sbp = (mae_fn(pred_bp[:, :, 0], true_bp[:, :, 0]) * mask[:,:,0]).sum() / mask[:,:,0].sum()
            mae_dbp = (mae_fn(pred_bp[:, :, 2], true_bp[:, :, 1]) * mask[:,:,0]).sum() / mask[:,:,0].sum()
            
            total_loss += loss.item()
            total_mae += (mae_sbp + mae_dbp).item() / 2

    return total_loss / len(loader), total_mae / len(loader)

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 配置
    CFG = {
        'seed': 42,
        'batch_size': 256,
        'lr': 5e-4,
        'epochs': 200, # 建议多跑一些epoch
        'file_path': "../mamba/dataset/find_peaks_55_070_240_no_abnormal_percent_max25_new_find_peak_no_020_max_energy_filter_200hz_max_autocal_noquality_4.0s_0_75_43_test_pool_diff_hcu.h5", # 修改这里！
        'log_dir': f"arcface/Final_Exp_{datetime.now().strftime('%m%d_%H%M')}"
    }

    set_seed(CFG['seed'])
    os.makedirs(CFG['log_dir'], exist_ok=True)
    writer = SummaryWriter(CFG['log_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 构建全局映射 & 时序划分
    print(f">>> Loading {CFG['file_path']}...")
    with h5py.File(CFG['file_path'], "r") as f:
        all_names = sorted(list(set([x.decode('utf-8') for x in f["subject_name"][:]])))
        GLOBAL_MAP = {name: i for i, name in enumerate(all_names)}

    train_idx, test_idx = split_dataset_by_session(CFG['file_path'], train_ratio=0.7)
    print(f"Dataset Split (Temporal): Train={len(train_idx)}, Test={len(test_idx)}")

    # 2. Loaders
    train_set = Dataset(train_idx, CFG['file_path'], GLOBAL_MAP)
    test_set = Dataset(test_idx, CFG['file_path'], GLOBAL_MAP)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model Setup
    model = RadarModel(num_classes=len(GLOBAL_MAP)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    scaler = torch.cuda.amp.GradScaler()
    
    crit_bp = UncertaintyLoss().to(device)
    crit_id = nn.CrossEntropyLoss().to(device)

    # 4. Loop
    best_eer = 1.0

    for epoch in range(CFG['epochs']):
        # Train
        m = train_epoch(model, train_loader, optimizer, crit_bp, crit_id, scaler, device, epoch, writer)
        print(f"Ep {epoch}: Loss={m['loss']:.4f} (BP={m['bp']:.4f}, ID={m['id']:.4f})")
        
        # Validation (Every 5 epochs)
        if epoch % 5 == 0 or epoch == CFG['epochs'] - 1:
            # BP Metrics
            val_loss, val_mae = evaluate_regression(model, test_loader, device, crit_bp)
            # ID Metrics
            eer, roc_auc = evaluate_verification(model, test_loader, device)
            
            writer.add_scalar('Val/BP_MAE', val_mae, epoch)
            writer.add_scalar('Val/ID_EER', eer, epoch)
            
            print(f"  >>> Val: BP_MAE={val_mae:.2f} | EER={eer:.2%} | AUC={roc_auc:.4f}")
            
            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), os.path.join(CFG['log_dir'], "best_model.pt"))
                print("  *** Best Model Saved ***")
        
        scheduler.step()

    writer.close()
    print("Done.")