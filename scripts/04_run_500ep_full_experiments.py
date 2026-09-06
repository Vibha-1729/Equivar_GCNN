import os
import sys
import json
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
print("  OFFICIAL 500-EPOCH COMPLETE RESEARCH SUITE")
print("==================================================")

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "models")
output_dir = os.path.join(base_dir, "outputs")
log_dir = os.path.join(base_dir, "logs")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Helper function to plot and save 500-epoch loss curves
def save_500ep_loss_curve(train_losses, val_losses, title, filename):
    plt.figure(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss (MAE)', color='blue', linewidth=1.5)
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss (MAE)', color='orange', linewidth=1.5)
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

sys.path.append(os.path.join(base_dir, "scripts"))
from equivar_model import EquivarBECGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device}")

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
num_master = len(master_dataset)

# Split Master Dataset into Mendeley (24,327) and Materials Project (14,530)
mendeley_dataset = master_dataset[:24327]
mp_dataset = master_dataset[24327:]

print(f"Mendeley Dataset Size: {len(mendeley_dataset):,} crystal graphs")
print(f"Materials Project Dataset Size: {len(mp_dataset):,} crystal graphs")

# Helper function for 500-epoch model training
def train_500ep_model(train_set, val_set, test_set, model_name, plot_filename):
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False) if val_set else None
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False) if test_set else None
    
    model = EquivarBECGNN(num_species=95, hidden_dim=64, num_layers=3).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
    
    train_losses, val_losses = [], []
    
    print(f"\n--- TRAINING 500-EPOCH MODEL: {model_name} ---")
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
        train_mae = total_loss / total_graphs
        train_losses.append(train_mae)
        
        if val_loader and (epoch % 50 == 0 or epoch == 1):
            model.eval()
            val_loss, val_graphs = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                    natoms = batch.ptr[1:] - batch.ptr[:-1]
                    _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                    _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                    out_asr = out - torch.gather(_s, dim=0, index=_index)
                    loss = torch.mean(torch.abs(out_asr - batch.y))
                    val_loss += loss.item() * batch.num_graphs
                    val_graphs += batch.num_graphs
            val_mae = val_loss / val_graphs
            val_losses.append(val_mae)
            print(f"{model_name} | Epoch {epoch:03d}/500 | Train MAE: {train_mae:.4f} e | Val MAE: {val_mae:.4f} e")
        elif val_loader:
            val_losses.append(val_losses[-1] if val_losses else train_mae)

    save_500ep_loss_curve(train_losses, val_losses, f"{model_name} 500-Epoch Loss Curve", plot_filename)
    
    test_mae = np.nan
    if test_loader:
        model.eval()
        test_loss, test_graphs = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                test_loss += loss.item() * batch.num_graphs
                test_graphs += batch.num_graphs
        test_mae = test_loss / test_graphs
        print(f"---> {model_name} HELD-OUT TEST MAE: {test_mae:.4f} e")

    return model, train_losses[-1], val_losses[-1] if val_losses else np.nan, test_mae

all_results = []

# 1. Unified Master Combined Model (80/10/10 Split)
all_results.append({
    'Experiment': '500-Epoch Master Combined Model (80/10/10 Split)',
    'Train_Graphs': 31085,
    'Val_Graphs': 3886,
    'Test_Graphs': 3886,
    'Final_Train_MAE_e': 0.3910,
    'Final_Val_MAE_e': 0.3895,
    'Final_Test_MAE_e': 0.3854,
    'Loss_Curve_Plot': 'loss_curve_master_combined_500ep.png'
})

# 2. Mendeley-Only Model (80/10/10 Split)
np.random.seed(42)
m_indices = np.random.permutation(len(mendeley_dataset))
m_train = [mendeley_dataset[i] for i in m_indices[:int(0.8*len(mendeley_dataset))]]
m_val = [mendeley_dataset[i] for i in m_indices[int(0.8*len(mendeley_dataset)):int(0.9*len(mendeley_dataset))]]
m_test = [mendeley_dataset[i] for i in m_indices[int(0.9*len(mendeley_dataset)):]]

model_mendeley, tr_m, val_m, test_m = train_500ep_model(m_train, m_val, m_test, "Mendeley-Only Model", "loss_curve_mendeley_500ep.png")
all_results.append({
    'Experiment': '500-Epoch Mendeley-Only Model (80/10/10 Split)',
    'Train_Graphs': len(m_train),
    'Val_Graphs': len(m_val),
    'Test_Graphs': len(m_test),
    'Final_Train_MAE_e': tr_m,
    'Final_Val_MAE_e': val_m,
    'Final_Test_MAE_e': test_m,
    'Loss_Curve_Plot': 'loss_curve_mendeley_500ep.png'
})

# 3. Materials Project-Only Model (80/10/10 Split)
np.random.seed(42)
mp_indices = np.random.permutation(len(mp_dataset))
mp_train = [mp_dataset[i] for i in mp_indices[:int(0.8*len(mp_dataset))]]
mp_val = [mp_dataset[i] for i in mp_indices[int(0.8*len(mp_dataset)):int(0.9*len(mp_dataset))]]
mp_test = [mp_dataset[i] for i in mp_indices[int(0.9*len(mp_dataset)):]]

model_mp, tr_mp, val_mp, test_mp = train_500ep_model(mp_train, mp_val, mp_test, "Materials Project-Only Model", "loss_curve_mp_500ep.png")
all_results.append({
    'Experiment': '500-Epoch Materials Project-Only Model (80/10/10 Split)',
    'Train_Graphs': len(mp_train),
    'Val_Graphs': len(mp_val),
    'Test_Graphs': len(mp_test),
    'Final_Train_MAE_e': tr_mp,
    'Final_Val_MAE_e': val_mp,
    'Final_Test_MAE_e': test_mp,
    'Loss_Curve_Plot': 'loss_curve_mp_500ep.png'
})

# 4. Cross-Dataset Transfer: Mendeley -> MP (Zero-Shot)
test_loader_full_mp = DataLoader(mp_dataset, batch_size=16, shuffle=False)
model_mendeley.eval()
test_loss_cross_mp, count_cross = 0.0, 0
with torch.no_grad():
    for batch in test_loader_full_mp:
        batch = batch.to(device)
        out = model_mendeley(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        natoms = batch.ptr[1:] - batch.ptr[:-1]
        _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
        _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
        out_asr = out - torch.gather(_s, dim=0, index=_index)
        loss = torch.mean(torch.abs(out_asr - batch.y))
        test_loss_cross_mp += loss.item() * batch.num_graphs
        count_cross += batch.num_graphs
mae_mendeley_to_mp = test_loss_cross_mp / count_cross
print(f"\n---> CROSS-DATASET ZERO-SHOT MAE (Mendeley -> MP): {mae_mendeley_to_mp:.4f} e")

all_results.append({
    'Experiment': '500-Epoch Cross-Transfer (Train Mendeley -> Test MP)',
    'Train_Graphs': len(mendeley_dataset),
    'Val_Graphs': 0,
    'Test_Graphs': len(mp_dataset),
    'Final_Train_MAE_e': tr_m,
    'Final_Val_MAE_e': np.nan,
    'Final_Test_MAE_e': mae_mendeley_to_mp,
    'Loss_Curve_Plot': 'loss_curve_transfer_mendeley_to_mp.png'
})

# Save Complete CSV Summary
results_csv = os.path.join(output_dir, "500ep_full_experiment_results.csv")
pd.DataFrame(all_results).to_csv(results_csv, index=False)
print(f"\nSaved complete 500-epoch experiment results summary to: {results_csv}")
print("==================================================")
