# filename: exp2_2_closed_set.py
import torch
import h5py
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model_utils import (
    set_seed, split_dataset_by_session, RadarModel, 
    RadarDataset, UncertaintyLoss, SupConLoss
)

def evaluate_accuracy_knn(model, train_loader, test_loader, device):
    """KNN Accuracy for Metric Learning"""
    model.eval()
    train_feats, train_labels = [], []
    test_feats, test_labels = [], []
    
    with torch.no_grad():
        # 1. 提取库内特征
        for x, _, label in train_loader:
            _, emb = model(x.to(device))
            train_feats.append(emb.cpu())
            train_labels.append(label.cpu())
        # 2. 提取测试特征
        for x, _, label in test_loader:
            _, emb = model(x.to(device))
            test_feats.append(emb.cpu())
            test_labels.append(label.cpu())

    train_feats = torch.cat(train_feats)
    train_labels = torch.cat(train_labels)
    test_feats = torch.cat(test_feats)
    test_labels = torch.cat(test_labels)

    # Cosine Similarity
    sim = torch.mm(test_feats, train_feats.t())
    _, top_idx = sim.max(dim=1)
    pred = train_labels[top_idx]
    
    acc = (pred == test_labels).float().mean().item()
    return acc

if __name__ == '__main__':
    CFG = {
        'seed': 42, 'batch_size': 256, 'lr': 5e-4, 'epochs': 100,
        'file_path': "../mamba/dataset/your_dataset.h5",
        'log_dir': f"logs/Exp2_2_Closed_{datetime.now().strftime('%m%d_%H%M')}"
    }
    set_seed(CFG['seed'])
    writer = SummaryWriter(CFG['log_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    with h5py.File(CFG['file_path'], "r") as f:
        names = sorted(list(set([x.decode('utf-8') for x in f["subject_name"][:]])))
        GLOBAL_MAP = {n: i for i, n in enumerate(names)}

    train_idx, test_idx = split_dataset_by_session(CFG['file_path'])
    train_set = RadarDataset(train_idx, CFG['file_path'], GLOBAL_MAP)
    test_set = RadarDataset(test_idx, CFG['file_path'], GLOBAL_MAP)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)
    # 专门用于 KNN 推断的 loader (不 shuffle)
    train_loader_inf = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False)

    model = RadarModel(num_classes=len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    crit_bp = UncertaintyLoss().to(device)
    crit_id = SupConLoss().to(device)

    print(">>> [Exp 2.2] Start Closed-Set Training...")
    for ep in range(CFG['epochs']):
        model.train()
        for i, (x, y, label) in enumerate(tqdm(train_loader, ncols=80, desc=f"Ep {ep}")):
            x, y, label = x.to(device), y.to(device), label.to(device)
            pred_bp, emb = model(x)
            loss = crit_bp(pred_bp, y[:,:,:2], y[:,:,2:]) + crit_id(emb, label)
            opt.zero_grad(); loss.backward(); opt.step()
        
        if ep % 5 == 0:
            acc = evaluate_accuracy_knn(model, train_loader_inf, test_loader, device)
            writer.add_scalar('Acc/ClosedSet', acc, ep)
            print(f"Ep {ep}: Closed-Set Accuracy = {acc:.2%}")

    print("Done.")
