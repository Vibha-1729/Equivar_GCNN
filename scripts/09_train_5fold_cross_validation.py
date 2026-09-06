import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

sys.path.append("/home/vibhan23/scripts")
from equivar_model import EquivarBECGNN

print("==================================================")
print("  STRICT 5-FOLD CROSS-VALIDATION (Kutana et al. 2025 Protocol)")
print("==================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")

# 1. Load Processed PyG Dataset
processed_pt = "/home/vibhan23/data/processed/mp_pyg_dataset.pt"
if not os.path.exists(processed_pt):
    print(f"ERROR: Dataset file not found at {processed_pt}")
    exit(1)

graphs = torch.load(processed_pt, weights_only=False)
print(f"Total Dataset Size: {len(graphs):,} PyG crystal graphs")

# 2. Setup 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_dir = "/home/vibhan23/models/5fold"
logs_dir = "/home/vibhan23/logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

fold_results = []

print("\n--- BEGINNING 5-FOLD CROSS-VALIDATION ---")

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(graphs), 1):
    print(f"\n================ FOLD {fold_idx} / 5 ================")
    
    # 80% Train/Val indices split into 80% Train, 10% Val, 10% Test
    np.random.seed(42 + fold_idx)
    perm_train_val = np.random.permutation(train_val_idx)
    n_val = int(len(perm_train_val) * 0.125)  # 10% of total
    
    val_idx = perm_train_val[:n_val]
    train_idx = perm_train_val[n_val:]
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    print(f"Fold {fold_idx} Dataset Partition:")
    print(f"  - Train Set (80%): {len(train_graphs):,} graphs")
    print(f"  - Validation Set (10%): {len(val_graphs):,} graphs")
    print(f"  - Held-out Test Set (10%): {len(test_graphs):,} graphs (UNSEEN)")
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Model & Optimizer
    model = EquivarBECGNN(num_species=95, hidden_dim=64, lmax=2, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    def compute_loss(pred, target):
        return torch.mean(torch.abs(pred - target))
        
    best_val_mae = float('inf')
    best_model_path = os.path.join(models_dir, f"equivar_fold_{fold_idx}_best.pt")
    
    for epoch in range(1, 101):  # 100 Epochs per fold
        model.train()
        t_loss, t_nodes = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
            loss = compute_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * batch.num_nodes
            t_nodes += batch.num_nodes
        train_mae = t_loss / t_nodes
        
        # Validation Pass
        model.eval()
        v_loss, v_nodes = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                loss = compute_loss(pred, batch.y)
                v_loss += loss.item() * batch.num_nodes
                v_nodes += batch.num_nodes
        val_mae = v_loss / v_nodes
        scheduler.step()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 25 == 0 or epoch == 1:
            print(f"Fold {fold_idx} | Epoch {epoch:03d}/100 | Train MAE: {train_mae:.4f} e | Val MAE: {val_mae:.4f} e")
            
    # Strictly Evaluate on UNSEEN 10% Test Set
    checkpoint_best = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint_best)
    model.eval()
    
    test_loss, test_nodes = 0.0, 0
    diag_errors, offdiag_errors = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
            diff = torch.abs(pred - batch.y)
            
            test_loss += torch.sum(diff).item()
            test_nodes += batch.num_nodes * 9
            
            for i in range(3):
                for j in range(3):
                    c_err = diff[:, i, j].cpu().numpy()
                    if i == j:
                        diag_errors.append(c_err)
                    else:
                        offdiag_errors.append(c_err)
                        
    test_mae = test_loss / test_nodes
    diag_mae = np.mean(np.concatenate(diag_errors))
    offdiag_mae = np.mean(np.concatenate(offdiag_errors))
    
    print(f"---> FOLD {fold_idx} UNSEEN TEST SET MAE: {test_mae:.4f} e (Diagonal: {diag_mae:.4f} e, Off-diagonal: {offdiag_mae:.4f} e)")
    
    fold_results.append({
        'fold': fold_idx,
        'best_val_mae': round(best_val_mae, 4),
        'test_mae_overall': round(test_mae, 4),
        'test_mae_diagonal': round(diag_mae, 4),
        'test_mae_offdiagonal': round(offdiag_mae, 4)
    })

# Summary Table
df_folds = pd.DataFrame(fold_results)
mean_test_mae = df_folds['test_mae_overall'].mean()
std_test_mae = df_folds['test_mae_overall'].std()

print("\n==================================================")
print("  5-FOLD CROSS-VALIDATION FINAL SUMMARY REPORT")
print("==================================================")
print(df_folds.to_string(index=False))
print(f"\nFinal Unseen Test Set MAE across 5 Folds: {mean_test_mae:.4f} +/- {std_test_mae:.4f} e")

df_folds.to_csv(os.path.join(logs_dir, "5fold_cross_validation_results.csv"), index=False)
print("Saved 5-fold cross-validation report to /home/vibhan23/logs/5fold_cross_validation_results.csv")
print("==================================================")
