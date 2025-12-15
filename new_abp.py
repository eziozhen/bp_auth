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
# 2. 损失函数 (新增 CAC Loss)
# ==========================================
class NegPearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, preds, targets):
        if preds.dim() == 3:
            preds = preds.squeeze(-1)
            targets = targets.squeeze(-1)
        vx = preds - torch.mean(preds, dim=1, keepdim=True)
        vy = targets - torch.mean(targets, dim=1, keepdim=True)
        cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + self.eps)
        return 1 - torch.mean(cost)

class WaveformLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha; self.beta = beta
        self.l1 = nn.L1Loss()
        self.pearson = NegPearsonLoss()
    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_pearson = self.pearson(pred, target)
        return self.alpha * loss_l1 + self.beta * loss_pearson, loss_l1, loss_pearson

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        # 过滤掉标签为 -1 (Unknown) 的样本，它们不参与 SupCon 训练
        mask_valid = labels != -1
        if mask_valid.sum() == 0: return torch.tensor(0.0).to(features.device)
        features = features[mask_valid]
        labels = labels[mask_valid]
        
        device = features.device
        batch_size = features.shape[0]
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        return -mean_log_prob_pos.mean()

# === [新增] Fixed-CAC Loss ===
class FixedCACLoss(nn.Module):
    """
    Fixed Class Anchor Clustering Loss
    适用于开集识别：强制已知类样本靠近固定的锚点，远离其他锚点。
    """
    def __init__(self, num_classes, feature_dim, magnitude=12.0, lambda_val=0.1):
        super(FixedCACLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.magnitude = magnitude # Alpha value in paper
        
        # 初始化固定锚点 (Scaled One-Hot 或 正交初始化)
        # 这里假设 feature_dim >= num_classes，使用 Scaled One-Hot 简单有效
        self.anchors = nn.Parameter(torch.zeros(num_classes, feature_dim), requires_grad=False)
        
        # 简单策略：前 num_classes 维设为 magnitude，其余为 0 (类似 One-Hot)
        # 也可以用 nn.init.orthogonal_
        nn.init.eye_(self.anchors)
        self.anchors.data = self.anchors.data * magnitude

    def forward(self, features, labels):
        # 过滤掉 Unknown (-1)
        mask = labels != -1
        if mask.sum() == 0: return torch.tensor(0.0, requires_grad=True).to(features.device)
        
        features = features[mask]
        labels = labels[mask]

        # 1. 计算距离 (Euclidean Distance)
        # features: [B, Dim], Anchors: [C, Dim]
        # dists: [B, C]
        dists = torch.cdist(features, self.anchors, p=2)
        
        # 2. 获取到 Ground Truth 类锚点的距离
        # labels.view(-1, 1) -> [B, 1]
        pos_dists = torch.gather(dists, 1, labels.view(-1, 1)) # [B, 1]
        
        # 3. Anchor Loss (拉近)
        loss_anchor = torch.mean(pos_dists)
        
        # 4. Tuplet Loss (推远其他)
        # log(1 + sum(exp(d_pos - d_neg)))
        # 构造 mask 排除自身类
        c_mask = torch.ones_like(dists).scatter_(1, labels.view(-1, 1), 0.)
        
        # exp(d_pos - d_other)
        exp_term = torch.exp(pos_dists - dists) * c_mask
        loss_tuplet = torch.mean(torch.log(1 + torch.sum(exp_term, dim=1)))
        
        return loss_tuplet + self.lambda_val * loss_anchor

# ==========================================
# 3. 模型组件 (保持不变，VQVAE输出通道改为动态)
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
    def __init__(self, in_channel=12, out_channel=1, num_vecs=800, channel=64, n_res_block=2, n_res_channel=32, embed_dim=64):
        super().__init__()
        # out_channel 修改为 1 (预测 ABP 波形)
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv1d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, num_vecs)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv1d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, num_vecs)
        self.upsample_t = nn.ConvTranspose1d(embed_dim, embed_dim, 4, stride=2, padding=1)
        # dec 输出维度由 out_channel 控制
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

# ==========================================
# 4. 集成模型 (修改：输出波形)
# ==========================================
class RadarModel(nn.Module):
    def __init__(self, ntoken=800, d_model=64, nhead=4, d_hid=128, nlayers=6, dropout=0.1, out_channel_wave=1):
        super().__init__()
        # 假设上面的 RadarEncoder, PositionalEncoding, VQVAE 都已定义
        self.rdr_embed = RadarEncoder(d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.rdr_encoder_t = nn.TransformerEncoder(enc_layer, nlayers)
        self.rdr_encoder_b = nn.TransformerEncoder(enc_layer, nlayers)
        self.vqvae = VQVAE(out_channel=out_channel_wave) 

        self.id_pool = AttentiveStatsPooling(in_dim=d_model * 2)
        
        # 修改：ID Head 只需要输出特征，不需要 Linear 分类层 (如果是 SupCon/CAC)
        # CAC 需要特征维度与 Anchor 维度一致，这里设为 128
        self.id_head = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128) # 输出 128 维特征
        )

    def forward(self, rdr_data):
        rdr_t, rdr_b = self.rdr_embed(rdr_data)
        rdr_t = self.rdr_encoder_t(self.pos_encoder(rdr_t))
        rdr_b = self.rdr_encoder_b(self.pos_encoder(rdr_b))
        rdr_t, rdr_b = rdr_t.permute(0, 2, 1), rdr_b.permute(0, 2, 1)
        
        rdr_t_up = self.vqvae.upsample_t(rdr_t)
        latent = torch.cat([rdr_t_up, rdr_b], dim=1)

        # ID Branch
        id_feat = self.id_pool(latent).squeeze(-1) 
        id_embedding = self.id_head(id_feat) 
        # CAC 建议归一化特征，SupCon 也需要
        id_embedding = F.normalize(id_embedding, p=2, dim=1) 

        # Waveform Branch
        rec = self.vqvae.dec(latent).permute(0, 2, 1)
        return rec, id_embedding

# ==========================================
# 5. 数据处理 (修改：加载 pulse 字典)
# ==========================================

def split_open_set(file_path, known_ratio=0.7, train_time_ratio=0.7, min_gap_min=30):
    """
    开集划分逻辑：
    1. 先把所有人分为：Known (训练+测试) 和 Unknown (只测试)。
    2. 对于 Known 人，按时间划分为：Train Session 和 Test Session。
    3. 对于 Unknown 人，所有数据放入 Test Set。
    """
    print(f">>> Open-Set Splitting: Known Ratio={known_ratio}, Time Split={train_time_ratio}")
    
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        dirs = [x.decode('utf-8') for x in f["dir"][:]]
    
    unique_subjects = sorted(list(set(subjects)))
    # 随机打乱人名
    random.shuffle(unique_subjects)
    
    num_known = int(len(unique_subjects) * known_ratio)
    known_names = set(unique_subjects[:num_known])
    unknown_names = set(unique_subjects[num_known:])
    
    print(f"   Known Subjects: {len(known_names)} | Unknown Subjects: {len(unknown_names)}")

    # 整理每个人的 Session
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
    
    # 遍历所有人，分配索引
    for sub, sessions in meta.items():
        sessions.sort(key=lambda x: x['time'])
        
        if sub in known_names:
            # 已知人：切分时间
            cut = int(len(sessions) * train_time_ratio)
            if cut == 0: cut = 1
            
            # Train Part
            train_sessions = sessions[:cut]
            train_idxs.extend([s['idx'] for s in train_sessions])
            
            # Test Part (需要考虑 gap)
            if len(train_sessions) > 0:
                last_train_time = train_sessions[-1]['time']
                test_sessions = sessions[cut:]
                for s in test_sessions:
                    diff = (s['time'] - last_train_time).total_seconds() / 60.0
                    if diff >= min_gap_min:
                        test_idxs.append(s['idx'])
            else:
                # 极端情况
                test_idxs.extend([s['idx'] for s in sessions[cut:]])
                
        else:
            # 未知人：全部进测试集
            test_idxs.extend([s['idx'] for s in sessions])

    return np.array(train_idxs), np.array(test_idxs), list(known_names)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, index, file_path, known_name_map):
        self.file = file_path
        self.index = index
        self.known_map = known_name_map # 仅包含 Known 的 {name: id}
        
        with h5py.File(self.file, "r") as f:
            stats = f['stats']
            self.stats = {
                'max': {k: v for k, v in stats['max_values'].attrs.items()},
                'min': {k: v for k, v in stats['min_values'].attrs.items()},
            }
            self.has_pulse_stats = 'pulse' in self.stats['max']

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        real_idx = self.index[idx]
        with h5py.File(self.file, "r") as f:
            name = f["subject_name"][real_idx].decode('utf-8')
            
            # 核心逻辑：如果是 Known，返回 ID；如果是 Unknown，返回 -1
            if name in self.known_map:
                label_id = self.known_map[name]
            else:
                label_id = -1 

            def get_norm(key):
                d = f[key][real_idx]
                return (d - self.stats['min'][key]) / (self.stats['max'][key] - self.stats['min'][key] + 1e-8)

            rdr = np.stack([
                get_norm('chest_radar_diff'), get_norm('neck_radar_diff'),
                get_norm('chest_heart_interval'), get_norm('neck_heart_interval'),
                get_norm('chest_radar_respiration'), get_norm('pulse_time')
            ], axis=0)
            
            abp = f["pulse"][real_idx]
            if self.has_pulse_stats:
                 abp = (abp - self.stats['min']['pulse']) / (self.stats['max']['pulse'] - self.stats['min']['pulse'] + 1e-8)
            abp_tensor = torch.from_numpy(abp).float().view(-1, 1)
            
        return torch.from_numpy(rdr).float(), abp_tensor, torch.tensor(label_id).long()

# ==========================================
# 5. 训练与评估引擎
# ==========================================
def train_epoch(model, loader, opt, crit_wave, crit_id, scaler, device, epoch, writer):
    model.train()
    metrics = {'loss': 0, 'wave': 0, 'id': 0}
    
    pbar = tqdm(loader, desc=f"Ep {epoch} [Train]", ncols=80)
    for i, (x, y_wave, labels) in enumerate(pbar):
        x, y_wave, labels = x.to(device), y_wave.to(device), labels.to(device)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            pred_wave, emb = model(x)
            
            # Loss 1: Waveform
            loss_wave_total, _, _ = crit_wave(pred_wave, y_wave)
            
            # Loss 2: ID (CAC or SupCon)
            # 注意：Train Set 中理论上不应该有 -1，但以防万一过滤一下
            loss_id = crit_id(emb, labels)
            
            loss = 10.0 * loss_wave_total + 1.0 * loss_id
        
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        metrics['loss'] += loss.item()
        metrics['wave'] += loss_wave_total.item()
        metrics['id'] += loss_id.item()
        
        if i % 20 == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(loader) + i)

    return {k: v / len(loader) for k, v in metrics.items()}

def evaluate_openset_verification(model, loader, device):
    """
    开集验证评估：
    这里的 EER 计算的是：是否为同一个人的概率。
    对于开集：
    - Known A vs Known A -> Positive Pair
    - Known A vs Known B -> Negative Pair
    - Known A vs Unknown -> Negative Pair
    """
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, _, label in tqdm(loader, desc="[Verify]", ncols=80):
            _, emb = model(x.to(device))
            embeddings.append(emb.cpu().numpy())
            labels.append(label.cpu().numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # 构建相似度矩阵
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    # 标签矩阵：是否为同一人
    # 注意：两个 Unknown (-1) 不应该被视为同一人！(除非是攻击样本的聚类分析，但一般验证中 imposters 互不相同)
    # 简单的处理：如果 label == -1，它和任何人都不是同一人 (除了它自己，但在 Triu 中对角线被排除)
    # 更严谨的逻辑：
    # y_true = (L[i] == L[j]) AND (L[i] != -1)
    
    n = len(labels)
    # 取上三角索引
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    label_i = labels[idx_i]
    label_j = labels[idx_j]
    
    # 同一人：ID 相同 且 ID 不是 -1
    y_true = (label_i == label_j) & (label_i != -1)
    scores = sim_matrix[idx_i, idx_j]
    
    # 计算 EER
    if y_true.sum() == 0:
        return 1.0, 0.5 # 无正样本，无法计算
        
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer, auc(fpr, tpr)

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == '__main__':
    # 配置
    CFG = {
        'seed': 42,
        'batch_size': 256,
        'lr': 5e-4,
        'epochs': 100,
        'file_path': "../mamba/dataset/find_peaks_55_070_240_no_abnormal_percent_max25_new_find_peak_no_020_max_energy_filter_200hz_max_autocal_noquality_4.0s_0_75_43_test_pool_diff_hcu.h5",
        
        # === 新增配置项 ===
        'use_cac': True,           # 核心开关：True 使用 CAC，False 使用 SupCon
        'open_set_ratio': 0.7,     # 70% 的人作为 Known，30% 作为 Unknown
        'feature_dim': 128,        # ID Head 输出维度
        
        'log_dir': f"OpenSet_Exp_{datetime.now().strftime('%m%d_%H%M')}"
    }
    
    CFG['log_dir'] += "_CAC" if CFG['use_cac'] else "_SupCon"

    set_seed(CFG['seed'])
    os.makedirs(CFG['log_dir'], exist_ok=True)
    writer = SummaryWriter(CFG['log_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 开集数据划分
    train_idx, test_idx, known_names_list = split_open_set(
        CFG['file_path'], 
        known_ratio=CFG['open_set_ratio'],
        train_time_ratio=0.7,
        min_gap_min=10
    )
    
    # 构建 Known Map (Unknown 在 Dataset 中会被标记为 -1)
    KNOWN_MAP = {name: i for i, name in enumerate(known_names_list)}
    num_known_classes = len(known_names_list)
    print(f"Num Known Classes: {num_known_classes}")

    # 2. Loaders
    train_set = Dataset(train_idx, CFG['file_path'], KNOWN_MAP)
    test_set = Dataset(test_idx, CFG['file_path'], KNOWN_MAP) # Test Set 包含 -1
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model
    model = RadarModel(out_channel_wave=1).to(device) # feature_dim 硬编码在 model 里了 (128)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    scaler = torch.cuda.amp.GradScaler()

    # 4. Losses
    crit_wave = WaveformLoss().to(device)
    
    # === 核心逻辑：根据配置选择 ID Loss ===
    if CFG['use_cac']:
        print(">>> Using Fixed-CAC Loss")
        # magnitude=12.0 是论文推荐值
        crit_id = FixedCACLoss(num_classes=num_known_classes, feature_dim=CFG['feature_dim'], magnitude=12.0).to(device)
    else:
        print(">>> Using SupCon Loss")
        crit_id = SupConLoss(temperature=0.07).to(device)

    # 5. Loop
    best_eer = 1.0

    for epoch in range(CFG['epochs']):
        # Train
        m = train_epoch(model, train_loader, optimizer, crit_wave, crit_id, scaler, device, epoch, writer)
        print(f"Ep {epoch}: Loss={m['loss']:.4f} | ID_Loss={m['id']:.4f}")
        
        # Validation
        if epoch % 5 == 0 or epoch == CFG['epochs'] - 1:
            # 评估开集 EER (包含 Unknown 样本的干扰)
            eer, roc_auc = evaluate_openset_verification(model, test_loader, device)
            
            writer.add_scalar('Val/ID_EER', eer, epoch)
            print(f"  >>> Val OpenSet: EER={eer:.2%} | AUC={roc_auc:.4f}")
            
            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), os.path.join(CFG['log_dir'], "best_model.pt"))
                print("  *** Best Model Saved ***")
        
        scheduler.step()

    writer.close()
    print("Done.")
