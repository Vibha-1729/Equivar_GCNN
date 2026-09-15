import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import scatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

print("==================================================")
print("  500-EPOCH MASTER 3-WAY MODEL (MENDELEY + MP + JARVIS)")
print("==================================================")

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "outputs", "models")
output_dir = os.path.join(base_dir, "outputs")
plots_dir = os.path.join(output_dir, "plots")
csv_dir = os.path.join(output_dir, "csv_results")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

sys.path.append(os.path.join(base_dir, "scripts"))
from equivar_model import EquivarBECGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device}")

def parse_extxyz_to_pyg(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
        
    print(f"Converting ExtXYZ to PyG Graphs: {os.path.basename(filepath)}...")
    pyg_list = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    idx = 0
    num_lines = len(lines)
    pbar = tqdm(total=num_lines, desc=f"Converting {os.path.basename(filepath)}")
    
    species_map_z = {'H':1, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Rb':37, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Tc':43, 'Ru':44, 'Rh':45, 'Pd':46, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Cs':55, 'Ba':56, 'La':57, 'Ce':58, 'Pr':59, 'Nd':60, 'Sm':62, 'Eu':63, 'Gd':64, 'Tb':65, 'Dy':66, 'Ho':67, 'Er':68, 'Tm':69, 'Yb':70, 'Lu':71, 'Hf':72, 'Ta':73, 'W':74, 'Re':75, 'Os':76, 'Ir':77, 'Pt':78, 'Au':79, 'Hg':80, 'Tl':81, 'Pb':82, 'Bi':83, 'Th':90, 'U':92}
    
    while idx < num_lines:
        try:
            line = lines[idx].strip()
            if not line:
                idx += 1
                pbar.update(1)
                continue
            natoms = int(line)
            atom_lines = lines[idx+2 : idx+2+natoms]
            idx += 2 + natoms
            pbar.update(2 + natoms)
            
            species_z = []
            pos_list = []
            bec_tensors = []
            
            for aline in atom_lines:
                tokens = aline.strip().split()
                species = tokens[0]
                z_val = species_map_z.get(species, 6)
                species_z.append(z_val)
                pos_list.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                if len(tokens) >= 13:
                    bec_vals = [float(x) for x in tokens[4:13]]
                elif len(tokens) >= 9:
                    bec_vals = [float(x) for x in tokens[-9:]]
                else:
                    continue
                bec_tensors.append(np.array(bec_vals).reshape(3, 3))
                
            if len(bec_tensors) == natoms:
                z_tensor = torch.tensor(species_z, dtype=torch.long)
                pos_tensor = torch.tensor(pos_list, dtype=torch.float)
                y_tensor = torch.tensor(np.array(bec_tensors), dtype=torch.float)
                
                dist = torch.cdist(pos_tensor, pos_tensor)
                mask = (dist < 3.0) & (dist > 0.01)
                edge_index = mask.nonzero().t()
                edge_vec = pos_tensor[edge_index[1]] - pos_tensor[edge_index[0]]
                
                pyg_data = Data(z=z_tensor, pos=pos_tensor, edge_index=edge_index, edge_vec=edge_vec, y=y_tensor)
                pyg_list.append(pyg_data)
        except Exception:
            idx += 1
            pbar.update(1)
            
    pbar.close()
    print(f"Successfully converted {len(pyg_list):,} PyG crystal graphs!")
    return pyg_list

master_3way_xyz = os.path.join(data_dir, "processed", "BEC_Combined_Master_Mendeley_MP_JARVIS.xyz")
dataset = parse_extxyz_to_pyg(master_3way_xyz)

if not dataset:
    print("Dataset empty or not found! Falling back to master deduplicated...")
    master_3way_xyz = os.path.join(data_dir, "processed", "BEC_Combined_Master_Deduplicated.xyz")
    dataset = parse_extxyz_to_pyg(master_3way_xyz)

# 80/10/10 Split
np.random.seed(42)
perm = np.random.permutation(len(dataset))
n_train = int(0.8 * len(dataset))
n_val = int(0.1 * len(dataset))

train_set = [dataset[i] for i in perm[:n_train]]
val_set = [dataset[i] for i in perm[n_train:n_train+n_val]]
test_set = [dataset[i] for i in perm[n_train+n_val:]]

print(f"\nMaster 3-Way Dataset Split | Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

model = EquivarBECGNN(num_species=95, hidden_dim=64, num_layers=3).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

train_losses, val_losses, test_losses = [], [], []
best_val_mae = float('inf')
checkpoint_path = os.path.join(model_dir, "equivar_master_mendeley_mp_jarvis_500ep.pt")

print("\n--- STARTING 500-EPOCH MODEL TRAINING ---")
for epoch in range(1, 501):
    model.train()
    total_loss, total_graphs = 0.0, 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        natoms = batch.ptr[1:] - batch.ptr[:-1]
        _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
        _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
        out_asr = out - torch.gather(_s, dim=0, index=_index)
        loss = torch.mean(torch.abs(out_asr - batch.y))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    scheduler.step()
    tr_mae = total_loss / total_graphs
    train_losses.append(tr_mae)
    
    if epoch % 25 == 0 or epoch == 1 or epoch == 500:
        model.eval()
        v_loss, v_graphs = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                v_loss += loss.item() * batch.num_graphs
                v_graphs += batch.num_graphs
        val_mae = v_loss / v_graphs
        val_losses.append(val_mae)
        
        # Test evaluation
        t_loss, t_graphs = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                t_loss += loss.item() * batch.num_graphs
                t_graphs += batch.num_graphs
        test_mae = t_loss / t_graphs
        test_losses.append(test_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), checkpoint_path)
            
        print(f"Master 3-Way Model | Epoch {epoch:03d}/500 | Train MAE: {tr_mae:.4f} e | Val MAE: {val_mae:.4f} e | Test MAE: {test_mae:.4f} e")
    else:
        val_losses.append(val_losses[-1] if val_losses else tr_mae)
        test_losses.append(test_losses[-1] if test_losses else tr_mae)

# Save Final Weights
torch.save(model.state_dict(), checkpoint_path)
print(f"\nSaved trained model weights checkpoint: {checkpoint_path}")

# Plot Loss Curve
plt.figure(figsize=(9, 5))
epochs_range = range(1, 501)
plt.plot(epochs_range, train_losses, label='Training Loss (MAE)', color='blue', linewidth=1.5)
plt.plot(epochs_range, val_losses, label='Validation Loss (MAE)', color='orange', linewidth=1.5)
plt.plot(epochs_range, test_losses, label='Test Loss (MAE)', color='green', linewidth=1.5)
plt.title('500-Epoch Master 3-Way Model: Mendeley + Materials Project + JARVIS', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('MAE Loss (e)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plot_out = os.path.join(plots_dir, "loss_curve_mendeley_mp_jarvis_500ep.png")
plt.savefig(plot_out, dpi=300)
plt.close()
print(f"Saved loss curve plot to: {plot_out}")

# Save CSV Summary
csv_out = os.path.join(csv_dir, "mendeley_mp_jarvis_500ep_results.csv")
pd.DataFrame([{
    'Experiment': '500-Epoch Master 3-Way Model (Mendeley + MP + JARVIS)',
    'Train_Graphs': len(train_set),
    'Val_Graphs': len(val_set),
    'Test_Graphs': len(test_set),
    'Final_Train_MAE_e': train_losses[-1],
    'Final_Val_MAE_e': val_losses[-1],
    'Final_Test_MAE_e': test_losses[-1],
    'Model_Weights': 'outputs/models/equivar_master_mendeley_mp_jarvis_500ep.pt',
    'Loss_Plot': 'outputs/plots/loss_curve_mendeley_mp_jarvis_500ep.png'
}]).to_csv(csv_out, index=False)
print(f"Saved results summary CSV to: {csv_out}")
print("==================================================")
