# filename: model_utils.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import h5py
from datetime import datetime

# ==========================================
# 1. 基础工具
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 

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

# ==========================================
# 2. Loss Functions
# ==========================================
class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, mask=None):
        mu = pred[:, :, [0, 2]]      
        log_var = pred[:, :, [1, 3]] 
        loss = 0.5 * (log_var + (target - mu)**2 / torch.exp(log_var))
        if mask is not None:
            if mask.ndim < loss.ndim: mask = mask.unsqueeze(-1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    def forward(self, features, labels):
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
        return - mean_log_prob_pos.mean()

# ==========================================
# 3. Radar Model Components (简化版)
# ==========================================
class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, 1), nn.Tanh(),
            nn.Conv1d(bottleneck_dim, in_dim, 1), nn.Softmax(dim=2)
        )
    def forward(self, x):
        alpha = self.attention(x)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, planes, 1, stride=stride, bias=False), nn.BatchNorm1d(planes))
    def forward(self, x): return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class RadarEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(6, d_model, kernel_size=1)
        self.proj_b = nn.Sequential(BasicBlock(d_model, d_model*2, 2), BasicBlock(d_model*2, d_model*4, 2)) # 简化写法
        self.quan_b = nn.Conv1d(d_model*4, d_model, 1)
        self.proj_t = nn.Sequential(BasicBlock(d_model*4, d_model*8, 2))
        self.quan_t = nn.Conv1d(d_model*8, d_model, 1)
    def forward(self, x):
        x_z = self.conv(x)
        x_b = self.proj_b(x_z)
        q_b = self.quan_b(x_b)
        x_t = self.proj_t(x_b)
        q_t = self.quan_t(x_t)
        return q_t.permute(0, 2, 1), q_b.permute(0, 2, 1)

class VQVAE_Simple(nn.Module):
    # 占位符，仅为了不报错，实际使用你原本复杂的 VQVAE 代码
    # 这里为了篇幅省略了复杂的 Decoder/Quantize 细节，
    # 请在你的实际文件中把原本 VQVAE 完整的粘贴过来，这里只做示意
    def __init__(self, out_channel=4, embed_dim=64):
        super().__init__()
        self.upsample_t = nn.ConvTranspose1d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(embed_dim*2, 128, 4, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, out_channel, 3, padding=1)
        )
    def forward(self, input): return self.dec(input), 0 # dummy diff

class RadarModel(nn.Module):
    def __init__(self, num_classes, d_model=64):
        super().__init__()
        self.rdr_embed = RadarEncoder(d_model=d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, 4, 128, 0.1, batch_first=True)
        self.rdr_encoder_t = nn.TransformerEncoder(enc_layer, 2)
        self.rdr_encoder_b = nn.TransformerEncoder(enc_layer, 2)
        
        # 为了演示方便使用 Simple 版，请替换回你原来的 VQVAE 类
        self.vqvae = VQVAE_Simple(out_channel=4, embed_dim=d_model) 

        self.id_pool = AttentiveStatsPooling(in_dim=d_model * 2)
        self.id_head = nn.Sequential(
            nn.Linear(d_model * 4, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 128)
        )

    def forward(self, rdr_data):
        rdr_t, rdr_b = self.rdr_embed(rdr_data)
        rdr_t = self.rdr_encoder_t(rdr_t) # (B, Seq, D)
        rdr_b = self.rdr_encoder_b(rdr_b)
        
        # Latent Fusion
        rdr_t_up = self.vqvae.upsample_t(rdr_t.permute(0, 2, 1)).permute(0, 2, 1)
        latent = torch.cat([rdr_t_up, rdr_b], dim=2).permute(0, 2, 1) # (B, 2D, Seq)

        # ID Branch
        id_feat = self.id_pool(latent).squeeze(-1) 
        id_embedding = F.normalize(self.id_head(id_feat), p=2, dim=1)

        # BP Branch
        rec, _ = self.vqvae(latent) # (B, 4, Seq)
        return rec.permute(0, 2, 1), id_embedding

class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, index, file_path, global_name_map):
        self.file = file_path
        self.index = index
        self.name_to_id = global_name_map
        with h5py.File(self.file, "r") as f:
            stats = f['stats']
            self.stats = {'max': {k: v for k, v in stats['max_values'].attrs.items()},
                          'min': {k: v for k, v in stats['min_values'].attrs.items()}}
    def __len__(self): return len(self.index)
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