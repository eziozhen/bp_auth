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
        torch.backends.cudnn.benchmark = True 

# ==========================================
# 2. 损失函数 (修改点：新增 SupConLoss，移除 ArcFace)
# ==========================================
class UncertaintyLoss(nn.Module):
    """Gaussian NLL Loss: 学习数据的不确定性"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        mu = pred[:, :, [0, 2]]      
        log_var = pred[:, :, [1, 3]] 
        loss = 0.5 * (log_var + (target - mu)**2 / torch.exp(log_var))
        
        if mask is not None:
            if mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            masked_loss = loss * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning (InfoNCE with labels)"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (batch_size, dim) - 已经归一化的特征
        labels: (batch_size)
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. 计算相似度矩阵 (B, B)
        # features 已经是 normalize 过的，所以 dot product 就是 cosine similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 2. 构建 Mask
        # mask: 同类为1，异类为0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # logits_mask: 对角线为0 (自己和自己不计算)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        
        # positive mask: 同类且非自己
        mask = mask * logits_mask

        # 3. 计算 Log-Prob
        exp_logits = torch.exp(logits) * logits_mask
        # 分母：sum(exp(所有其他样本))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 4. 计算 Mean Log-Likelihood
        # 只取正样本对的 log_prob
        # sum(log_prob * pos_mask) / num_positives
        mask_pos_pairs = mask.sum(1)
        # 防止除以0 (如果没有正样本，loss设为0)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# ==========================================
# 3. 模型组件 (保持不变)
# ==========================================
class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
    def forward(self, x):
        alpha = self.attention(x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(), nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv1d(channel, in_channel, 1),
        )
    def forward(self, input): return self.conv(input) + input

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
        for i in range(n_res_block): blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, input): return self.blocks(input)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        blocks = [nn.Conv1d(in_channel, channel, 3, padding=1)]
        for i in range(n_res_block): blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        if stride == 4:
            blocks.extend([nn.ConvTranspose1d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
                           nn.ConvTranspose1d(channel // 2, out_channel, 4, stride=2, padding=1)])
        elif stride == 2:
            blocks.append(nn.ConvTranspose1d(channel, out_channel, 4, stride=2, padding=1))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, input): return self.blocks(input)

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim; self.n_embed = n_embed; self.decay = decay; self.eps = eps
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed); self.register_buffer("cluster_size", torch.zeros(n_embed)); self.register_buffer("embed_avg", embed.clone())
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
    def embed_code(self, embed_id): return F.embedding(embed_id, self.embed.transpose(0, 1))

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
    def forward(self, x): return self.dropout(x + self.pe[:, : x.size(1)])

# ==========================================
# 4. 集成模型 (修改点：移除 ArcFace)
# ==========================================
class RadarModel(nn.Module):
    def __init__(self, ntoken=800, d_model=64, nhead=4, d_hid=128, nlayers=6, dropout=0.1, num_classes=10):
        super().__init__()
        self.rdr_embed = RadarEncoder(d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.rdr_encoder_t = nn.TransformerEncoder(enc_layer, nlayers)
        self.rdr_encoder_b = nn.TransformerEncoder(enc_layer, nlayers)
        self.vqvae = VQVAE(out_channel=4) # SBP_mu, SBP_var, DBP_mu, DBP_var

        self.id_pool = AttentiveStatsPooling(in_dim=d_model * 2)
        self.id_head = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128))
        
        # 移除 ArcFace
        # self.arcface = ArcMarginProduct(128, num_classes) 

    def forward(self, rdr_data):
        # 移除了 labels 参数
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
        id_embedding = F.normalize(id_embedding, p=2, dim=1) # 必须归一化

        # Branch 2: BP
        rec = self.vqvae.dec(latent).permute(0, 2, 1)
        
        # 直接返回 embedding，不再计算 logits
        return rec, id_embedding

# ==========================================
# 5. 数据处理 
# ==========================================

def split_dataset_smart(file_path, train_ratio=0.7, min_gap_min=30, max_gap_min=None):
    """
    智能数据集划分 V2 (全分钟粒度)
    Args:
        min_gap_min (float): 最小间隔分钟数 (防作弊/防泄漏)
        max_gap_min (float): 最大间隔分钟数 (防太难/防漂移)。None 表示不限制。
    """
    print(f">>> Smart Splitting (Min): Ratio={train_ratio}, Gap=[{min_gap_min}m, {max_gap_min}m]")
    
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        dirs = [x.decode('utf-8') for x in f["dir"][:]]
    
    meta = {}
    for idx, (sub, d_str) in enumerate(zip(subjects, dirs)):
        if sub not in meta: meta[sub] = []
        try:
            date_str = "_".join(d_str.split('_')[-6:]) 
            dt = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
        except:
            dt = datetime.fromtimestamp(idx)
        meta[sub].append({'time': dt, 'idx': idx})

    train_idxs, test_idxs = [], []
    ignored_count = 0

    for sub, sessions in meta.items():
        sessions.sort(key=lambda x: x['time'])
        cut = int(len(sessions) * train_ratio)
        if cut == 0: cut = 1 
        
        candidate_train = sessions[:cut]
        candidate_test = sessions[cut:]
        last_train_time = candidate_train[-1]['time']
        
        train_idxs.extend([s['idx'] for s in candidate_train])
        
        for s in candidate_test:
            # 计算分钟差
            diff_minutes = (s['time'] - last_train_time).total_seconds() / 60.0
            
            # 1. 最小限制
            if diff_minutes < min_gap_min:
                ignored_count += 1; continue
            
            # 2. 最大限制 (如果设置了的话)
            if max_gap_min is not None:
                if diff_minutes > max_gap_min:
                    ignored_count += 1; continue
            
            test_idxs.append(s['idx'])

    print(f"Dataset Split Result: Train={len(train_idxs)}, Test={len(test_idxs)} (Ignored {ignored_count})")
    return np.array(train_idxs), np.array(test_idxs)

def split_dataset_by_session(file_path, train_ratio=0.7):
    print(">>> Analyzing Temporal Sessions...")
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        dirs = [x.decode('utf-8') for x in f["dir"][:]]
    
    meta_data = {}
    for idx, (sub, d_str) in enumerate(zip(subjects, dirs)):
        if sub not in meta_data: meta_data[sub] = []
        try:
            date_str = "_".join(d_str.split('_')[-6:]) 
            dt = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
        except:
            dt = datetime.fromtimestamp(idx) 
        meta_data[sub].append({'time': dt, 'idx': idx, 'dir': d_str})

    train_idxs, test_idxs = [], []
    for sub, sessions in meta_data.items():
        unique_dirs = {} 
        for item in sessions:
            if item['dir'] not in unique_dirs: unique_dirs[item['dir']] = {'time': item['time'], 'idxs': []}
            unique_dirs[item['dir']]['idxs'].append(item['idx'])
        
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
            name = f["subject_name"][real_idx].decode('utf-8')
            label_id = self.name_to_id[name]

            def get_norm(key):
                d = f[key][real_idx]
                return (d - self.stats['min'][key]) / (self.stats['max'][key] - self.stats['min'][key] + 1e-8)

            rdr = np.stack([
                get_norm('chest_radar_diff'), get_norm('neck_radar_diff'),
                get_norm('chest_heart_interval'), get_norm('neck_heart_interval'),
                get_norm('chest_radar_respiration'), get_norm('pulse_time')
            ], axis=0)
            sbp = torch.from_numpy(f["sbp"][real_idx]).view(800, 1).float()
            dbp = torch.from_numpy(f["dbp"][real_idx]).view(800, 1).float()
            mask = torch.ones_like(sbp)
            
        return torch.from_numpy(rdr).float(), torch.cat([sbp, dbp, mask], dim=1), torch.tensor(label_id).long()

# ==========================================
# 6. 训练与评估引擎 (修改点：返回值解包)
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
            # 修改：model forward 不再需要 labels，返回 (bp, emb)
            pred_bp, emb = model(x)
            
            loss_bp = crit_bp(pred_bp, true_bp, mask)
            
            # InfoNCE loss: 传入 embedding 和 labels
            loss_id = crit_id(emb, labels)
            
            loss = 1.0 * loss_bp + 1.0 * loss_id
        
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
            # 修改：返回值解包
            _, emb = model(x.to(device))
            embeddings.append(emb.cpu().numpy()) # 已经在 forward 里归一化了
            labels.append(label.cpu().numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    sim_matrix = np.dot(embeddings, embeddings.T)
    label_matrix = (labels[:, None] == labels[None, :])
    
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
            
            # 修改：返回值解包
            pred_bp, _ = model(x)
            loss = criterion(pred_bp, true_bp, mask)
            
            mae_sbp = (mae_fn(pred_bp[:, :, 0], true_bp[:, :, 0]) * mask[:,:,0]).sum() / mask[:,:,0].sum()
            mae_dbp = (mae_fn(pred_bp[:, :, 2], true_bp[:, :, 1]) * mask[:,:,0]).sum() / mask[:,:,0].sum()
            
            total_loss += loss.item()
            total_mae += (mae_sbp + mae_dbp).item() / 2

    return total_loss / len(loader), total_mae / len(loader)

# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 配置
    CFG = {
        'seed': 42,
        'batch_size': 256, # InfoNCE 喜欢大 Batch Size，显存够的话越大越好
        'lr': 5e-4,
        'epochs': 200, 
        'file_path': "../mamba/dataset/find_peaks_55_070_240_no_abnormal_percent_max25_new_find_peak_no_020_max_energy_filter_200hz_max_autocal_noquality_4.0s_0_75_43_test_pool_diff_hcu.h5", 
        'log_dir': f"infonce/Final_Exp_{datetime.now().strftime('%m%d_%H%M')}"
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

    # train_idx, test_idx = split_dataset_by_session(CFG['file_path'], train_ratio=0.7)
    train_idx, test_idx = split_dataset_smart(
            CFG['file_path'], 
            train_ratio=0.6, 
            min_gap_min=1,      # 最小隔 30 分钟 (防作弊)
            max_gap_min=10  # 最大隔 3 天 (控制难度)
        )
    print(f"Dataset Split (Temporal): Train={len(train_idx)}, Test={len(test_idx)}")

    # 2. Loaders
    train_set = Dataset(train_idx, CFG['file_path'], GLOBAL_MAP)
    test_set = Dataset(test_idx, CFG['file_path'], GLOBAL_MAP)
    
    # 训练集 drop_last=True 推荐保留，避免 Batch Size 变化导致 InfoNCE 不稳定
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model Setup
    model = RadarModel(num_classes=len(GLOBAL_MAP)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    scaler = torch.cuda.amp.GradScaler()
    
    crit_bp = UncertaintyLoss().to(device)
    crit_id = SupConLoss(temperature=0.07).to(device) # 使用 SupConLoss

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