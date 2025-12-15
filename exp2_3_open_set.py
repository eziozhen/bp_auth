# filename: exp2_3_open_set.py
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from datetime import datetime

from model_utils import (
    set_seed, split_dataset_by_session, RadarModel, 
    RadarDataset, UncertaintyLoss, SupConLoss
)

def evaluate_verification(model, loader, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for x, _, label in loader:
            _, emb = model(x.to(device))
            embeddings.append(emb.cpu().numpy())
            labels.append(label.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # Random Sample for EER calc
    if len(labels) > 3000:
        idx = np.random.choice(len(labels), 3000, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]
        
    sim_matrix = np.dot(embeddings, embeddings.T)
    label_matrix = (labels[:, None] == labels[None, :])
    tri_idx = np.triu_indices(len(labels), k=1)
    
    fpr, tpr, _ = roc_curve(label_matrix[tri_idx], sim_matrix[tri_idx])
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer, auc(fpr, tpr)

def visualize_tsne(model, loader, device, save_path):
    model.eval()
    feats, labels = [], []
    count = 0
    with torch.no_grad():
        for x, _, y in loader:
            _, emb = model(x.to(device))
            feats.append(emb.cpu().numpy())
            labels.append(y.cpu().numpy())
            count += x.shape[0]
            if count > 1000: break
            
    feats = np.concatenate(feats)[:1000]
    labels = np.concatenate(labels)[:1000]
    
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(feats)
    
    plt.figure(figsize=(8, 8))
    # 只画前10个人
    uni_lbls = np.unique(labels)
    for lbl in uni_lbls[:10]:
        idx = labels == lbl
        plt.scatter(X_emb[idx, 0], X_emb[idx, 1], label=f"ID {lbl}", alpha=0.7)
    plt.legend()
    plt.title("Radar Feature t-SNE")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE to {save_path}")

if __name__ == '__main__':
    CFG = {
        'seed': 42, 'batch_size': 256, 'lr': 5e-4, 'epochs': 100,
        'file_path': "../mamba/dataset/your_dataset.h5",
        'log_dir': f"logs/Exp2_3_Open_{datetime.now().strftime('%m%d_%H%M')}"
    }
    set_seed(CFG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG['log_dir'], exist_ok=True)

    # Reuse Data Loading
    with h5py.File(CFG['file_path'], "r") as f:
        names = sorted(list(set([x.decode('utf-8') for x in f["subject_name"][:]])))
        GLOBAL_MAP = {n: i for i, n in enumerate(names)}
    
    train_idx, test_idx = split_dataset_by_session(CFG['file_path'])
    train_set = RadarDataset(train_idx, CFG['file_path'], GLOBAL_MAP)
    test_set = RadarDataset(test_idx, CFG['file_path'], GLOBAL_MAP)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=CFG['batch_size'], shuffle=False)

    model = RadarModel(num_classes=len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    crit_bp = UncertaintyLoss().to(device)
    crit_id = SupConLoss().to(device)

    best_eer = 1.0
    print(">>> [Exp 2.3] Start Open-Set Verification Training...")
    
    for ep in range(CFG['epochs']):
        model.train()
        for x, y, label in tqdm(train_loader, ncols=80, desc=f"Ep {ep}"):
            x, y, label = x.to(device), y.to(device), label.to(device)
            pred_bp, emb = model(x)
            loss = crit_bp(pred_bp, y[:,:,:2], y[:,:,2:]) + crit_id(emb, label)
            opt.zero_grad(); loss.backward(); opt.step()
        
        if ep % 5 == 0:
            eer, auc_score = evaluate_verification(model, test_loader, device)
            print(f"Ep {ep}: EER = {eer:.2%} | AUC = {auc_score:.4f}")
            
            if eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), f"{CFG['log_dir']}/best.pt")
                if ep > 10:
                    visualize_tsne(model, test_loader, device, f"{CFG['log_dir']}/tsne_{ep}.png")