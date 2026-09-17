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
print("  TRAINING SIZE VARIATION STUDY (60%, 70%, 80%)")
print("==================================================")

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "outputs")
plots_dir = os.path.join(output_dir, "plots")
csv_dir = os.path.join(output_dir, "csv_results")

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

split_ratios = [0.60, 0.70, 0.80]
learning_results = []

for split_ratio in split_ratios:
    print(f"\n--------------------------------------------------")
    print(f"  RUNNING TRAIN SPLIT: {int(split_ratio * 100)}%")
    print(f"--------------------------------------------------")
    
    np.random.seed(42)
    perm = np.random.permutation(len(dataset))
    
    n_train = int(split_ratio * len(dataset))
    n_test = int((1.0 - split_ratio) / 2.0 * len(dataset))
    n_val = len(dataset) - n_train - n_test
    
    train_set = [dataset[i] for i in perm[:n_train]]
    val_set = [dataset[i] for i in perm[n_train:n_train+n_val]]
    test_set = [dataset[i] for i in perm[n_train+n_val:]]
    
    print(f"Counts -> Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    
    model = EquivarBECGNN(num_species=95, hidden_dim=64, num_layers=3).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
    
    train_losses, test_losses = [], []
    
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
        
        if epoch % 50 == 0 or epoch == 1 or epoch == 500:
            model.eval()
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
            te_mae = t_loss / t_graphs
            test_losses.append(te_mae)
            print(f"Split {int(split_ratio*100)}% | Epoch {epoch:03d}/500 | Train MAE: {tr_mae:.4f} e | Test MAE: {te_mae:.4f} e")
        else:
            test_losses.append(test_losses[-1] if test_losses else tr_mae)
            
    learning_results.append({
        'Train_Split_Percent': int(split_ratio * 100),
        'Train_Count': len(train_set),
        'Test_Count': len(test_set),
        'Final_Train_MAE_e': train_losses[-1],
        'Final_Test_MAE_e': test_losses[-1]
    })

# Save Learning Curve Table
df_lc = pd.DataFrame(learning_results)
lc_csv = os.path.join(csv_dir, "learning_curve_training_size_results.csv")
df_lc.to_csv(lc_csv, index=False)
print(f"\nSaved training size variation results: {lc_csv}")

# Plot Learning Curve vs Training Size
plt.figure(figsize=(8, 5))
train_percents = [r['Train_Split_Percent'] for r in learning_results]
train_maes = [r['Final_Train_MAE_e'] for r in learning_results]
test_maes = [r['Final_Test_MAE_e'] for r in learning_results]

plt.plot(train_percents, train_maes, 'o-', label='Training MAE', color='blue', linewidth=2, markersize=8)
plt.plot(train_percents, test_maes, 's-', label='Test MAE', color='red', linewidth=2, markersize=8)
plt.title('Learning Curve: MAE vs Training Split Size (60%, 70%, 80%)', fontsize=12, fontweight='bold')
plt.xlabel('Training Set Percentage (%)', fontsize=11)
plt.ylabel('MAE Loss (e)', fontsize=11)
plt.xticks([60, 70, 80])
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

lc_plot = os.path.join(plots_dir, "learning_curve_training_size.png")
plt.savefig(lc_plot, dpi=300)
plt.close()
print(f"Saved learning curve plot: {lc_plot}")
print("==================================================")
