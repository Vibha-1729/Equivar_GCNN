import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.append("/home/vibhan23/scripts")
from equivar_model import EquivarBECGNN

print("==================================================")
print("  PHASE 3: EQUIVAR GNN FULL 500-EPOCH TRAINING")
print("==================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# 1. Load Processed PyG Dataset
processed_pt = "/home/vibhan23/data/processed/mp_pyg_dataset.pt"
if not os.path.exists(processed_pt):
    print(f"ERROR: Processed dataset not found at {processed_pt}")
    exit(1)

print(f"Loading PyG dataset from {processed_pt}...")
graphs = torch.load(processed_pt, weights_only=False)
print(f"Loaded {len(graphs):,} PyG crystal graphs.")

# Train / Validation Split (80 / 20)
np.random.seed(42)
indices = np.random.permutation(len(graphs))
split_idx = int(0.8 * len(graphs))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

train_graphs = [graphs[i] for i in train_indices]
val_graphs = [graphs[i] for i in val_indices]

print(f"Train Set: {len(train_graphs):,} graphs | Validation Set: {len(val_graphs):,} graphs")

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

# 2. Instantiate Model with Exact Paper Change-of-Basis Matrix & Acoustic Sum Rule
model = EquivarBECGNN(num_species=95, hidden_dim=64, lmax=2, num_layers=3).to(device)
print(f"Model Parameters Count: {sum(p.numel() for p in model.parameters()):,}")

num_epochs = 500
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

def compute_loss(pred_y, target_y):
    return torch.mean(torch.abs(pred_y - target_y))

best_val_mae = float('inf')
models_dir = "/home/vibhan23/models"
logs_dir = "/home/vibhan23/logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

history = []

print(f"\nStarting Official 500-Epoch Full Model Training...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    total_nodes = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred_y = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        loss = compute_loss(pred_y, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
        
    train_mae = total_train_loss / total_nodes
    
    # Validation Pass
    model.eval()
    total_val_loss = 0.0
    total_val_nodes = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred_y = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
            loss = compute_loss(pred_y, batch.y)
            
            total_val_loss += loss.item() * batch.num_nodes
            total_val_nodes += batch.num_nodes
            
    val_mae = total_val_loss / total_val_nodes
    scheduler.step()
    
    history.append({
        'epoch': epoch,
        'train_mae': round(train_mae, 4),
        'val_mae': round(val_mae, 4),
        'lr': round(optimizer.param_groups[0]['lr'], 6)
    })
    
    # Log progress every 10 epochs or on best epoch
    if epoch % 10 == 0 or epoch == 1 or val_mae < best_val_mae:
        print(f"Epoch {epoch:03d}/{num_epochs:03d} | Train MAE: {train_mae:.4f} e | Val MAE: {val_mae:.4f} e | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model_path = os.path.join(models_dir, "equivar_generalized_best_500ep.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_mae': best_val_mae
        }, best_model_path)

elapsed_time = time.time() - start_time
print(f"\n500-Epoch Training Complete in {elapsed_time/60.0:.2f} minutes!")
print(f"Best Validation BEC MAE: {best_val_mae:.4f} e")

df_hist = pd.DataFrame(history)
hist_path = os.path.join(logs_dir, "training_history_500ep.csv")
df_hist.to_csv(hist_path, index=False)
print(f"Saved 500-epoch training history to: {hist_path}")

print("==================================================")
