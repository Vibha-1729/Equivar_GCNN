import os
import sys
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
print("  500-EPOCH DATASET MIXING: MP + MENDELEY (10% & 20%) -> HELD-OUT MENDELEY TEST")
print("==================================================")

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

sys.path.append(os.path.join(base_dir, "scripts"))
from equivar_model import EquivarBECGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device}")

# Helper function to plot and save 500-epoch loss curves
def save_500ep_loss_curve(train_losses, val_losses, title, filename):
    plt.figure(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training MAE Loss', color='blue', linewidth=1.5)
    if val_losses:
        plt.plot(epochs, val_losses, label='Held-Out Mendeley Test MAE Loss', color='red', linewidth=1.5)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('MAE Loss (e)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved 500-epoch loss curve plot: {out_path}")

# Helper function to parse ExtXYZ into PyG Data list
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
    
    species_map_z = {'H':1, 'Li':3, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'K':19, 'Ca':20, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Sr':38, 'Y':39, 'Zr':40, 'Nb':41, 'Mo':42, 'Ag':47, 'Cd':48, 'In':49, 'Sn':50, 'Sb':51, 'Te':52, 'I':53, 'Ba':56, 'La':57, 'Hf':72, 'Ta':73, 'W':74, 'Pb':82, 'Bi':83}
    
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

# Load Datasets
master_xyz = os.path.join(data_dir, "processed", "BEC_Combined_Master_Deduplicated.xyz")
master_dataset = parse_extxyz_to_pyg(master_xyz)
mendeley_dataset = master_dataset[:24327]
mp_dataset = master_dataset[24327:]

np.random.seed(42)
mendeley_perm = np.random.permutation(len(mendeley_dataset))

# -----------------------------------------------------------------------------
# EXPERIMENT A: MP + 10% Mendeley -> Test on remaining 90% Mendeley
# -----------------------------------------------------------------------------
idx_10 = int(0.10 * len(mendeley_dataset))
mendeley_10_mix = [mendeley_dataset[i] for i in mendeley_perm[:idx_10]]
mendeley_90_test = [mendeley_dataset[i] for i in mendeley_perm[idx_10:]]

train_set_10 = mp_dataset + mendeley_10_mix
train_loader_10 = DataLoader(train_set_10, batch_size=16, shuffle=True)
test_loader_90 = DataLoader(mendeley_90_test, batch_size=16, shuffle=False)

print(f"\n--- EXPERIMENT A: MP ({len(mp_dataset)}) + 10% Mendeley ({len(mendeley_10_mix)}) -> Test 90% Mendeley ({len(mendeley_90_test)}) ---")

model_10 = EquivarBECGNN(num_species=95, hidden_dim=64, num_layers=3).to(device)
optimizer_10 = AdamW(model_10.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_10 = CosineAnnealingLR(optimizer_10, T_max=500, eta_min=1e-5)

train_losses_10, test_losses_90 = [], []

for epoch in range(1, 501):
    model_10.train()
    total_loss, total_graphs = 0.0, 0
    for batch in train_loader_10:
        batch = batch.to(device)
        optimizer_10.zero_grad()
        out = model_10(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        natoms = batch.ptr[1:] - batch.ptr[:-1]
        _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
        _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
        out_asr = out - torch.gather(_s, dim=0, index=_index)
        loss = torch.mean(torch.abs(out_asr - batch.y))
        loss.backward()
        optimizer_10.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    scheduler_10.step()
    tr_mae = total_loss / total_graphs
    train_losses_10.append(tr_mae)
    
    if epoch % 50 == 0 or epoch == 1 or epoch == 500:
        model_10.eval()
        t_loss, t_graphs = 0.0, 0
        with torch.no_grad():
            for batch in test_loader_90:
                batch = batch.to(device)
                out = model_10(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                t_loss += loss.item() * batch.num_graphs
                t_graphs += batch.num_graphs
        te_mae = t_loss / t_graphs
        test_losses_90.append(te_mae)
        print(f"Mix 10% | Epoch {epoch:03d}/500 | Train MAE: {tr_mae:.4f} e | Held-Out 90% Mendeley MAE: {te_mae:.4f} e")
    else:
        test_losses_90.append(test_losses_90[-1] if test_losses_90 else tr_mae)

final_mix10_mae = test_losses_90[-1]
save_500ep_loss_curve(train_losses_10, test_losses_90, "MP + 10% Mendeley Mixing -> Test on 90% Held-Out Mendeley", "loss_curve_mix10_mendeley_transfer.png")

# -----------------------------------------------------------------------------
# EXPERIMENT B: MP + 20% Mendeley -> Test on remaining 80% Mendeley
# -----------------------------------------------------------------------------
idx_20 = int(0.20 * len(mendeley_dataset))
mendeley_20_mix = [mendeley_dataset[i] for i in mendeley_perm[:idx_20]]
mendeley_80_test = [mendeley_dataset[i] for i in mendeley_perm[idx_20:]]

train_set_20 = mp_dataset + mendeley_20_mix
train_loader_20 = DataLoader(train_set_20, batch_size=16, shuffle=True)
test_loader_80 = DataLoader(mendeley_80_test, batch_size=16, shuffle=False)

print(f"\n--- EXPERIMENT B: MP ({len(mp_dataset)}) + 20% Mendeley ({len(mendeley_20_mix)}) -> Test 80% Mendeley ({len(mendeley_80_test)}) ---")

model_20 = EquivarBECGNN(num_species=95, hidden_dim=64, num_layers=3).to(device)
optimizer_20 = AdamW(model_20.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_20 = CosineAnnealingLR(optimizer_20, T_max=500, eta_min=1e-5)

train_losses_20, test_losses_80 = [], []

for epoch in range(1, 501):
    model_20.train()
    total_loss, total_graphs = 0.0, 0
    for batch in train_loader_20:
        batch = batch.to(device)
        optimizer_20.zero_grad()
        out = model_20(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        natoms = batch.ptr[1:] - batch.ptr[:-1]
        _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
        _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
        out_asr = out - torch.gather(_s, dim=0, index=_index)
        loss = torch.mean(torch.abs(out_asr - batch.y))
        loss.backward()
        optimizer_20.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    scheduler_20.step()
    tr_mae = total_loss / total_graphs
    train_losses_20.append(tr_mae)
    
    if epoch % 50 == 0 or epoch == 1 or epoch == 500:
        model_20.eval()
        t_loss, t_graphs = 0.0, 0
        with torch.no_grad():
            for batch in test_loader_80:
                batch = batch.to(device)
                out = model_20(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                t_loss += loss.item() * batch.num_graphs
                t_graphs += batch.num_graphs
        te_mae = t_loss / t_graphs
        test_losses_80.append(te_mae)
        print(f"Mix 20% | Epoch {epoch:03d}/500 | Train MAE: {tr_mae:.4f} e | Held-Out 80% Mendeley MAE: {te_mae:.4f} e")
    else:
        test_losses_80.append(test_losses_80[-1] if test_losses_80 else tr_mae)

final_mix20_mae = test_losses_80[-1]
save_500ep_loss_curve(train_losses_20, test_losses_80, "MP + 20% Mendeley Mixing -> Test on 80% Held-Out Mendeley", "loss_curve_mix20_mendeley_transfer.png")

# Save Summary CSV
mixing_results_csv = os.path.join(output_dir, "mendeley_dataset_mixing_results.csv")
pd.DataFrame([
    {
        'Experiment': 'MP + 10% Mendeley Mixing',
        'Train_Graphs': len(train_set_10),
        'Test_Graphs': len(mendeley_90_test),
        'Epochs': 500,
        'Final_Train_MAE_e': train_losses_10[-1],
        'Final_Test_Mendeley_MAE_e': final_mix10_mae,
        'Loss_Plot': 'loss_curve_mix10_mendeley_transfer.png'
    },
    {
        'Experiment': 'MP + 20% Mendeley Mixing',
        'Train_Graphs': len(train_set_20),
        'Test_Graphs': len(mendeley_80_test),
        'Epochs': 500,
        'Final_Train_MAE_e': train_losses_20[-1],
        'Final_Test_Mendeley_MAE_e': final_mix20_mae,
        'Loss_Plot': 'loss_curve_mix20_mendeley_transfer.png'
    }
]).to_csv(mixing_results_csv, index=False)

print(f"\nSaved Mendeley dataset mixing results summary to: {mixing_results_csv}")
print("==================================================")
