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

# ==========================================
# 1. 基础组件与工具
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

# === 损失函数: Supervised Contrastive Loss ===
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        
        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask-out self-contrast cases
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask
        
        # Log-prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood for positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        return -mean_log_prob_pos.mean()

# ==========================================
# 2. 网络组件 (ABPNet)
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

class ABPNet(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        # Input: (B, 1, 800)
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = nn.Sequential(BasicBlock(d_model, d_model), BasicBlock(d_model, d_model))
        self.layer2 = nn.Sequential(BasicBlock(d_model, d_model*2, 2), BasicBlock(d_model*2, d_model*2))
        self.layer3 = nn.Sequential(BasicBlock(d_model*2, d_model*4, 2), BasicBlock(d_model*4, d_model*4))
        
        self.pool = AttentiveStatsPooling(in_dim=d_model*4)
        
        # Projection Head for Contrastive Learning
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
# 3. 数据集与划分逻辑
# ==========================================
class ABPDataset(torch.utils.data.Dataset):
    def __init__(self, index, file_path, name_to_id_map):
        self.file = file_path
        self.index = index
        self.name_to_id = name_to_id_map
        self.abp_min, self.abp_max = 40.0, 180.0 

    def __len__(self): return len(self.index)
    
    def __getitem__(self, idx):
        real_idx = self.index[idx]
        with h5py.File(self.file, "r") as f:
            name = f["subject_name"][real_idx].decode('utf-8')
            
            # 如果是 Unknown Subject，设置为 -1 (仅在测试集出现)
            label_id = self.name_to_id.get(name, -1)
            
            # 优先读取 pulse，没有则读取 sbp 模拟
            if 'pulse' in f:
                raw = f['pulse'][real_idx]
            else:
                raw = f['sbp'][real_idx] 
            
            # Min-Max 归一化
            val = (raw - self.abp_min) / (self.abp_max - self.abp_min)
            val = torch.from_numpy(val).float().view(1, -1) 
        return val, torch.tensor(label_id).long()

def get_split_indices(file_path, mode='closed_set', known_ratio=0.7, train_time_ratio=0.7):
    """
    智能划分函数
    """
    print(f">>> Splitting Dataset. Mode: {mode}")
    with h5py.File(file_path, "r") as f:
        subjects = [x.decode('utf-8') for x in f["subject_name"][:]]
        # 简单用索引作为时间戳模拟
        timestamps = list(range(len(subjects))) 
    
    unique_subs = sorted(list(set(subjects)))
    random.shuffle(unique_subs)
    
    # 1. 确定 Known / Unknown 集合
    if mode == 'open_set':
        n_known = int(len(unique_subs) * known_ratio)
        known_subs = set(unique_subs[:n_known])
        unknown_subs = set(unique_subs[n_known:])
        print(f"    Known: {len(known_subs)} | Unknown: {len(unknown_subs)}")
    else:
        known_subs = set(unique_subs)
        unknown_subs = set() # 闭集没有陌生人
        print(f"    Closed Set: {len(known_subs)} Subjects")

    # 2. 构建 Session 索引
    sub_indices = {s: [] for s in unique_subs}
    for i, sub in enumerate(subjects):
        sub_indices[sub].append(i)
    
    train_idx, test_idx = [], []
    
    for sub in unique_subs:
        idxs = sub_indices[sub] 
        
        if sub in known_subs:
            # 已知人：前X%进训练，后(1-X)%进测试
            cut = int(len(idxs) * train_time_ratio)
            if cut == 0: cut = 1
            train_idx.extend(idxs[:cut])
            test_idx.extend(idxs[cut:])
        else:
            # 未知人：全部进测试
            test_idx.extend(idxs)
            
    return np.array(train_idx), np.array(test_idx), {n: i for i, n in enumerate(sorted(list(known_subs)))}

# ==========================================
# 4. 评估与可视化函数
# ==========================================
def evaluate_metric(model, loader, device, mode='closed_set'):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)
            emb = model(x)
            embeddings.append(emb.cpu().numpy())
            labels.append(label.numpy())
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # --- 闭集评估 ---
    if mode == 'closed_set':
        return 0.0, embeddings, labels

    # --- 开集评估：EER ---
    # 随机采样以加速计算
    if len(labels) > 3000:
        idx = np.random.choice(len(labels), 3000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    sim_matrix = np.dot(embeddings, embeddings.T) 
    
    n = len(labels)
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    lbl_i = labels[idx_i]
    lbl_j = labels[idx_j]
    
    # 正样本定义：ID相同，且不为-1
    y_true = (lbl_i == lbl_j) & (lbl_i != -1)
    scores = sim_matrix[idx_i, idx_j]
    
    if y_true.sum() == 0:
        return 1.0, embeddings, labels # 无法计算
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    return eer, embeddings, labels

def plot_tsne(embeddings, labels, save_path):
    """绘制并保存 t-SNE 图"""
    print(">>> Plotting t-SNE...")
    # 随机采样最多 1000 个点避免过密
    if len(labels) > 1000:
        idx = np.random.choice(len(labels), 1000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            # Unknown 用灰色小点
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c='lightgrey', label='Unknown', alpha=0.5, s=20)
        else:
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=f'ID {lbl}', alpha=0.8, s=30)
            
    plt.title("t-SNE Visualization of ABP Features")
    if len(unique_labels) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"    Saved t-SNE plot to {save_path}")
    plt.close()

# ==========================================
# 5. 主训练循环 (保存最优模型版本)
# ==========================================
if __name__ == '__main__':
    # === 配置区域 ===
    CFG = {
        'seed': 42,
        'batch_size': 256,
        'lr': 1e-3,
        'epochs': 10, # 建议多跑几轮
        'file_path': "../mamba/dataset/find_peaks_55_070_240_no_abnormal_percent_max25_new_find_peak_no_020_max_energy_filter_200hz_max_autocal_noquality_4.0s_0_75_43_test_pool_diff_hcu.h5",
        
        # 模式切换: 'closed_set' (闭集/t-SNE) 或 'open_set' (开集/EER)
        'mode': 'open_set',  
        'save_path': "best_model.pt"
    }
    
    set_seed(CFG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 数据准备
    train_idx, test_idx, known_map = get_split_indices(
        CFG['file_path'], 
        mode=CFG['mode'], 
        known_ratio=0.7
    )
    
    train_set = ABPDataset(train_idx, CFG['file_path'], known_map)
    test_set = ABPDataset(test_idx, CFG['file_path'], known_map)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False)
    
    # 2. 模型与优化
    model = ABPNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    criterion = SupConLoss(temperature=0.07).to(device)
    
    print(f">>> Start Training ({CFG['mode']})...")
    
    # 初始化最优指标记录
    # 如果是开集，监控 EER (越小越好)；如果是闭集，简单起见监控 Loss (越小越好)
    best_metric = float('inf') 
    
    for ep in range(CFG['epochs']):
        model.train()
        loss_meter = []
        
        pbar = tqdm(train_loader, desc=f"Ep {ep}", ncols=80)
        for x, label in pbar:
            x, label = x.to(device), label.to(device)
            
            optimizer.zero_grad()
            emb = model(x)
            loss = criterion(emb, label)
            loss.backward()
            optimizer.step()
            
            loss_meter.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(loss_meter):.4f}"})
        
        avg_train_loss = np.mean(loss_meter)

        # 3. 验证阶段 (每5轮或最后一轮)
        if (ep + 1) % 5 == 0 or ep == CFG['epochs'] - 1:
            eer, _, _ = evaluate_metric(model, test_loader, device, mode=CFG['mode'])
            
            # 定义当前指标 current_metric
            if CFG['mode'] == 'open_set':
                current_metric = eer
                metric_name = "EER"
                print(f"Ep {ep}: Loss={avg_train_loss:.4f} | Open-Set EER={eer:.2%}")
            else:
                current_metric = avg_train_loss 
                metric_name = "Train_Loss"
                print(f"Ep {ep}: Loss={avg_train_loss:.4f} | (Closed-Set Monitor)")

            # === 核心：保存最优模型 ===
            if current_metric < best_metric:
                best_metric = current_metric
                torch.save(model.state_dict(), CFG['save_path'])
                print(f"    >>> New Best {metric_name}! Model Saved to {CFG['save_path']}")

    # ==========================================
    # 4. 最终测试与绘图 (使用最优模型)
    # ==========================================
    print("\n>>> Training Finished. Loading Best Model for Final Evaluation...")
    
    # 加载权重
    model.load_state_dict(torch.load(CFG['save_path']))
    model.eval()
    
    # 重新跑一遍评估
    final_eer, embeddings, labels = evaluate_metric(model, test_loader, device, mode=CFG['mode'])
    
    if CFG['mode'] == 'open_set':
        print(f"Final Best Model EER: {final_eer:.2%}")
    
    # 绘制 t-SNE
    # 文件名根据 Mode 动态生成
    tsne_filename = f"tsne_best_{CFG['mode']}.png"
    plot_tsne(embeddings, labels, tsne_filename)
    print("Done.")