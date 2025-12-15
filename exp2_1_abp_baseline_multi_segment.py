import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from collections import defaultdict
from datetime import datetime

# ==========================================
# 1. 基础配置
# ==========================================
CFG = {
    'seed': 42,
    'batch_size': 256,
    'lr': 1e-3,
    'epochs': 20,
    'file_path': "../mamba/dataset/find_peaks_all_55_070_240_no_abnormal_percent_max25_new_find_peak_no_020_max_energy_filter_200hz_max_autocal_noquality_4.0s_0_75_43_test_pool_diff_hcu.h5",
    
    # 模式设置
    'mode': 'open_set',
    'save_path': "best_model.pt",
    
    # === 新增：融合配置 ===
    'fusion_k': 5  # 每次测试融合 5 个连续片段 (约20秒数据)
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

# ==========================================
# 2. 损失函数 (SupCon)
# ==========================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        
        # 过滤 Unknown (-1)
        mask_valid = labels != -1
        if mask_valid.sum() < 2: return torch.tensor(0.0, requires_grad=True).to(device)
        
        features = F.normalize(features[mask_valid], dim=1)
        labels = labels[mask_valid]

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_pos_pairs = mask.sum(1)
        # 防止除零
        has_pos = mask_pos_pairs > 0
        if has_pos.sum() == 0: return torch.tensor(0.0, requires_grad=True).to(device)
        
        mean_log_prob_pos = (mask * log_prob).sum(1)[has_pos] / mask_pos_pairs[has_pos]
        return -mean_log_prob_pos.mean()

# ==========================================
# 3. 网络定义 (ABPNet)
# ==========================================
class AttentiveStatsPooling(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, in_dim, 1),
            nn.Softmax(dim=2)
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
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=False) # Fix inplace
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(planes))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x) # Fix inplace
        return self.relu(out)

class ABPNet(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=False)
        
        self.layer1 = nn.Sequential(BasicBlock(d_model, d_model), BasicBlock(d_model, d_model))
        self.layer2 = nn.Sequential(BasicBlock(d_model, d_model*2, 2), BasicBlock(d_model*2, d_model*2))
        self.layer3 = nn.Sequential(BasicBlock(d_model*2, d_model*4, 2), BasicBlock(d_model*4, d_model*4))
        
        self.pool = AttentiveStatsPooling(in_dim=d_model*4)
        self.head = nn.Sequential(
            nn.Linear(d_model*8, 256), 
            nn.LayerNorm(256), nn.SiLU(), 
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x).squeeze(-1)
        emb = self.head(x)
        return F.normalize(emb, p=2, dim=1)

# ==========================================
# 4. 数据集与划分 (升级版：支持 Session)
# ==========================================
def get_split_indices_scenario_test(
    file_path,
    mode='open_set', 
    known_ratio=0.7, 
    train_ratio=0.7,         # 前 60% 必做训练 (保持不动)
    # === 仅针对测试集 (Test Set) 的筛选参数 ===
    time_interval_min=0,     # [筛选] 测试样本必须晚于训练结束 X 分钟
    time_interval_max=None,  # [筛选] 测试样本必须早于训练结束 X 分钟 (None=不限)
    target_scenes=None,      # [筛选] 测试集只留这些场景 (None=不限, e.g. ['work'])
    target_seasons=None      # [筛选] 测试集只留这些季节 (None=不限, e.g. ['winter'])
):
    print(f">>> Splitting Dataset (Stable Train | Filtered Test)...")
    
    # 1. 读取数据
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        dirs = [x.decode('utf-8') for x in f["dir"][:]]
        # 兼容 scene 字段
        if "scene" in f.keys():
            scenes = [x.decode('utf-8') for x in f["scene"][:]]
        else:
            scenes = ["unknown"] * len(subjects)
            
        print(np.unique(scenes))
        
        print(f.keys())

    unique_subs = sorted(list(set(subjects)))
    random.shuffle(unique_subs)
    
    # 2. 划分 Known / Unknown
    if mode == 'open_set':
        n_known = int(len(unique_subs) * known_ratio)
        known_subs = set(unique_subs[:n_known])
        print(f"    [Open Set] Known: {len(known_subs)} | Unknown: {len(unique_subs)-n_known}")
    else:
        known_subs = set(unique_subs)
        print(f"    [Closed Set] All {len(unique_subs)} subjects are Known.")

    # 3. 辅助解析函数
    def parse_meta(d_str):
        try:
            # 解析时间字符串, 格式根据您的文件名调整
            date_str = "_".join(d_str.split('_')[-6:]) 
            dt = datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
        except: dt = datetime.min
        # 简单季节定义
        season = 'summer' if 5 <= dt.month <= 10 else 'winter'
        return dt, season

    # 4. 数据结构化：按人归档所有样本 (携带元数据)
    # 结构: person_samples[sub] = [ {idx, time, scene, season}, ... ]
    person_samples = defaultdict(list)
    dir_map = {}
    unique_dirs_set = sorted(list(set(dirs)))
    for i, d in enumerate(unique_dirs_set): dir_map[d] = i

    for idx, (sub, d_str, scn) in enumerate(zip(subjects, dirs, scenes)):
        dt, season = parse_meta(d_str)
        person_samples[sub].append({
            'idx': idx, 
            'time': dt, 
            'scene': scn, 
            'season': season,
            'dir_str': d_str
        })

    train_idx, test_idx = [], []
    
    # 5. 核心切分逻辑
    for sub in unique_subs:
        # 按时间排序该人的所有样本
        samples = sorted(person_samples[sub], key=lambda x: x['time'])
        if not samples: continue
        
        # === A. Known Subject: 切分 Train / Test ===
        if sub in known_subs:
            # 1. 强制划分训练集 (Baseline)
            # 逻辑：取前 train_ratio 的数据，不做任何筛选！
            # 目的：保证底库是该人最早、最全的状态，数量充足。
            total_len = len(samples)
            cut_point = int(total_len * train_ratio)
            if cut_point == 0: cut_point = 1
            
            train_part = samples[:cut_point]
            train_idx.extend([s['idx'] for s in train_part])
            
            # 记录训练集结束时间 (作为 Test Gap 的基准)
            last_train_time = train_part[-1]['time']
            
            # 2. 筛选测试集 (Filtered Probe)
            # 逻辑：取剩下的数据，严格应用筛选条件
            test_candidates = samples[cut_point:]
            
            for s in test_candidates:
                # --- 筛选条件 1: 场景 ---
                if target_scenes is not None and s['scene'] not in target_scenes: 
                    continue
                
                # --- 筛选条件 2: 季节 ---
                if target_seasons is not None and s['season'] not in target_seasons: 
                    continue
                
                # --- 筛选条件 3: 时间间隔 (相对于训练集结束时间) ---
                # 计算该测试样本距离训练结束过去了多少分钟
                gap_minutes = (s['time'] - last_train_time).total_seconds() / 60.0
                
                # 必须晚于 min (防止泄露)
                if time_interval_min is not None and gap_minutes < time_interval_min:
                    continue
                # 必须早于 max (验证特定时间段内的衰减)
                if time_interval_max is not None and gap_minutes > time_interval_max:
                    continue
                
                test_idx.append(s['idx'])
                
        # === B. Unknown Subject: 全量筛选 ===
        else:
            # 陌生人全部进测试，但也要符合场景/季节设定
            # (比如：我们只想测试“工作场景下的陌生人攻击”)
            for s in samples:
                if target_scenes is not None and s['scene'] not in target_scenes: continue
                if target_seasons is not None and s['season'] not in target_seasons: continue
                
                # Unknown 不需要时间间隔筛选 (没有训练集)
                test_idx.append(s['idx'])

    known_map = {n: i for i, n in enumerate(sorted(list(known_subs)))}
    
    print(f"    [Filter Config] Scenes={target_scenes if target_scenes else 'ALL'} | Seasons={target_seasons if target_seasons else 'ALL'} | Gap=[{time_interval_min}, {time_interval_max}]m")
    print(f"    Train Samples (Stable): {len(train_idx)} | Test Samples (Filtered): {len(test_idx)}")
    
    return np.array(train_idx), np.array(test_idx), known_map, dir_map


def get_split_indices_smart(file_path, mode='open_set', known_ratio=0.7, train_ratio=0.7):
    """
    【修正版划分逻辑】
    1. 划分 Known / Unknown。
    2. 对于 Known 人员，不按文件夹切，而是按【时间索引】切分。
       - 将该人的所有数据按时间排序。
       - 前 70% -> 训练，后 30% -> 测试。
       - 这样既保证了数据量充足，又保证了测试集的连续性（用于融合）。
    """
    print(f">>> Splitting Dataset (Time-Continuous Mode)...")
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        dirs = [x.decode('utf-8') for x in f["dir"][:]] # 用于辅助排序

    unique_subs = sorted(list(set(subjects)))
    random.shuffle(unique_subs)
    
    # 1. 划分 Known / Unknown
    if mode == 'open_set':
        n_known = int(len(unique_subs) * known_ratio)
        known_subs = set(unique_subs[:n_known])
        print(f"    Known: {len(known_subs)} | Unknown: {len(unique_subs)-n_known}")
    else:
        known_subs = set(unique_subs)

    # 2. 建立目录映射 (用于Fusion分组)
    unique_dirs = sorted(list(set(dirs)))
    dir_map = {d: i for i, d in enumerate(unique_dirs)}

    # 3. 按人整理索引
    sub_indices = defaultdict(list)
    for idx, (sub, d_str) in enumerate(zip(subjects, dirs)):
        # 记录 idx 和 目录ID
        # 我们假设 h5 文件里的 idx 本身就是按时间顺序写入的
        # 如果不是，这里需要解析 d_str 里的时间字符串进行排序
        sub_indices[sub].append(idx)
    
    train_idx, test_idx = [], []
    
    for sub in unique_subs:
        # 获取该人的所有索引，确保按时间排序 (默认 idx 就是有序的)
        idxs = sorted(sub_indices[sub])
        
        if sub in known_subs:
            # === 修正点：按样本数量切分，而不是按文件切分 ===
            total_samples = len(idxs)
            cut = int(total_samples * train_ratio)
            
            # 保证至少有数据
            if cut == 0: cut = 1
            
            # 前 70% 进训练
            train_idx.extend(idxs[:cut])
            # 后 30% 进测试 (这部分也是连续的，可以做融合)
            test_idx.extend(idxs[cut:])
        else:
            # Unknown 全部进测试
            test_idx.extend(idxs)
            
    known_map = {n: i for i, n in enumerate(sorted(list(known_subs)))}
    print(f"    Train Samples: {len(train_idx)} | Test Samples: {len(test_idx)}")
    return np.array(train_idx), np.array(test_idx), known_map, dir_map

class ABPDataset(torch.utils.data.Dataset):
    def __init__(self, index, file_path, name_map, dir_map):
        self.file = file_path
        self.index = index
        self.name_map = name_map
        self.dir_map = dir_map
        self.abp_min, self.abp_max = 40.0, 180.0 
        
        # 预加载 metadata 避免频繁IO
        self.meta = []
        with h5py.File(self.file, "r") as f:
            all_names = f["subject_name"][:]
            all_dirs = f["dir"][:]
            
        for real_idx in index:
            name = all_names[real_idx].decode('utf-8')
            d_str = all_dirs[real_idx].decode('utf-8')
            label = self.name_map.get(name, -1)
            dir_id = self.dir_map.get(d_str, -1)
            self.meta.append((real_idx, label, dir_id))

    def __len__(self): return len(self.index)
    
    def __getitem__(self, idx):
        real_idx, label, dir_id = self.meta[idx]
        
        with h5py.File(self.file, "r") as f:
            if 'pulse' in f: raw = f['pulse'][real_idx]
            else: raw = f['sbp'][real_idx]
            
            val = (np.clip(raw, self.abp_min, self.abp_max) - self.abp_min) / (self.abp_max - self.abp_min)
            val = torch.from_numpy(val).float().view(1, -1)
            
        # [关键] 返回 dir_id 和 real_idx 用于融合
        return val, torch.tensor(label).long(), torch.tensor(dir_id).long(), torch.tensor(real_idx).long()

# ==========================================
# 5. 评估函数 (包含融合逻辑)
# ==========================================
def evaluate_metric(model, loader, device, mode='closed_set'):
    """单片段基准评估"""
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y, _, _ in loader: # 忽略 dir_id, idx
            emb = model(x.to(device))
            embeddings.append(emb.cpu().numpy())
            labels.append(y.numpy())
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    if mode == 'closed_set': return 0.0, embeddings, labels

    # 采样加速
    if len(labels) > 3000:
        idx = np.random.choice(len(labels), 3000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    sim = np.dot(embeddings, embeddings.T)
    idx_i, idx_j = np.triu_indices(len(labels), k=1)
    y_true = (labels[idx_i] == labels[idx_j]) & (labels[idx_i] != -1)
    scores = sim[idx_i, idx_j]
    
    if y_true.sum() == 0: return 1.0, embeddings, labels
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer, embeddings, labels

def evaluate_fusion(model, loader, device, k=5):
    """
    [新增] 多片段融合评估
    保证：同一个人 + 同一个Session (dir) + 时间连续 (index)
    """
    print(f"    >>> Running Multi-Segment Fusion (K={k})...")
    model.eval()
    
    # 1. 收集所有信息
    all_emb, all_lbl, all_dir, all_idx = [], [], [], []
    with torch.no_grad():
        for x, y, d, i in loader:
            emb = model(x.to(device))
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(y.numpy())
            all_dir.append(d.numpy())
            all_idx.append(i.numpy())
            
    all_emb = np.concatenate(all_emb)
    all_lbl = np.concatenate(all_lbl)
    all_dir = np.concatenate(all_dir)
    all_idx = np.concatenate(all_idx)
    
    fused_emb, fused_lbl = [], []
    
    # 2. 按 Session (dir_id) 分组
    unique_dirs = np.unique(all_dir)
    
    for u_d in unique_dirs:
        # 获取该 Session 下的所有样本
        mask = (all_dir == u_d)
        curr_emb = all_emb[mask]
        curr_lbl = all_lbl[mask]
        curr_idx = all_idx[mask]
        
        # [关键] 按时间索引排序，确保物理连续
        sort_args = np.argsort(curr_idx)
        curr_emb = curr_emb[sort_args]
        curr_lbl = curr_lbl[sort_args] # 同一个session标签肯定一样
        
        n_samples = len(curr_lbl)
        if n_samples < k: continue
        
        # 3. 滑动窗口融合 (非重叠)
        for i in range(0, n_samples - k + 1, k):
            # 取 K 个连续片段
            chunk = curr_emb[i : i+k]
            
            # 特征平均
            avg_emb = np.mean(chunk, axis=0)
            
            # 重新归一化 (必做！)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            
            fused_emb.append(avg_emb)
            fused_lbl.append(curr_lbl[0])
            
    if len(fused_lbl) == 0:
        print("    [Warning] 样本不足以进行 K={k} 融合")
        return 1.0
        
    fused_emb = np.array(fused_emb)
    fused_lbl = np.array(fused_lbl)
    print(f"    Fusion Result: {len(all_lbl)} -> {len(fused_lbl)} fused samples")
    
    # 4. 计算 EER
    if len(fused_lbl) > 3000:
        idx = np.random.choice(len(fused_lbl), 3000, replace=False)
        fused_emb, fused_lbl = fused_emb[idx], fused_lbl[idx]
        
    sim = np.dot(fused_emb, fused_emb.T)
    idx_i, idx_j = np.triu_indices(len(fused_lbl), k=1)
    y_true = (fused_lbl[idx_i] == fused_lbl[idx_j]) & (fused_lbl[idx_i] != -1)
    scores = sim[idx_i, idx_j]
    
    if y_true.sum() == 0: return 1.0
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def plot_tsne(embeddings, labels, save_path):
    print(">>> Plotting t-SNE...")
    if len(labels) > 2000:
        idx = np.random.choice(len(labels), 2000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1: plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c='lightgrey', label='Unknown', alpha=0.5, s=20)
        else: plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=f'ID {lbl}', alpha=0.8, s=30)
            
    plt.title("t-SNE Visualization")
    if len(unique_labels) <= 10: plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ==========================================
# 6. 主训练循环
# ==========================================
if __name__ == '__main__':
    set_seed(CFG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 数据准备
    train_idx, test_idx, known_map, dir_map = get_split_indices_scenario_test(CFG['file_path'], mode=CFG['mode'], time_interval_min=30)
    
    train_set = ABPDataset(train_idx, CFG['file_path'], known_map, dir_map)
    test_set = ABPDataset(test_idx, CFG['file_path'], known_map, dir_map)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False)
    
    # 2. 模型与优化
    model = ABPNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    criterion = SupConLoss(temperature=0.07).to(device)
    
    print(f">>> Start Training ({CFG['mode']})...")
    best_eer = 1.0
    
    for ep in range(CFG['epochs']):
        model.train()
        loss_meter = []
        pbar = tqdm(train_loader, desc=f"Ep {ep}", ncols=80)
        
        for x, label, _, _ in pbar: # 训练时不需要 dir_id
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            emb = model(x)
            loss = criterion(emb, label)
            loss.backward()
            optimizer.step()
            loss_meter.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(loss_meter):.4f}"})
        
        # 3. 验证 (保存 Best Model)
        if (ep + 1) % 5 == 0 or ep == CFG['epochs'] - 1:
            # 训练中只看单片段 EER，速度快
            eer, _, _ = evaluate_metric(model, test_loader, device, mode=CFG['mode'])
            print(f"Ep {ep}: Loss={np.mean(loss_meter):.4f} | Single-Seg EER={eer:.2%}")
            
            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), CFG['save_path'])
                print(f"    >>> New Best Model Saved!")

    # ==========================================
    # 4. 最终测试 (包含多片段融合)
    # ==========================================
    print("\n>>> Training Finished. Loading Best Model for Final Evaluation...")
    
    model.load_state_dict(torch.load(CFG['save_path']))
    model.eval()

    # 1. 单片段 EER
    final_eer_1, embeddings, labels = evaluate_metric(model, test_loader, device, mode=CFG['mode'])
    print(f"Final Single-Segment EER: {final_eer_1:.2%}")
    
    # 2. [新增] 多片段融合 EER
    final_eer_k = evaluate_fusion(model, test_loader, device, k=CFG['fusion_k'])
    print(f"Final Multi-Segment (K={CFG['fusion_k']}) EER: {final_eer_k:.2%}")
    
    # 3. 绘制 t-SNE
    tsne_filename = f"tsne_best_{CFG['mode']}.png"
    plot_tsne(embeddings, labels, tsne_filename)
    print("Done.")