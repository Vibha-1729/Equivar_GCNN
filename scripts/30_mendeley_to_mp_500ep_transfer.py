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
print("  500-EPOCH VERIFIED TRANSFER: MENDELEY -> MATERIALS PROJECT")
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
mendeley_indices = np.random.permutation(len(mendeley_dataset))
mendeley_train = [mendeley_dataset[i] for i in mendeley_indices[:int(0.8*len(mendeley_dataset))]]
mendeley_val = [mendeley_dataset[i] for i in mendeley_indices[int(0.8*len(mendeley_dataset)):int(0.9*len(mendeley_dataset))]]

train_loader_mendeley = DataLoader(mendeley_train, batch_size=16, shuffle=True)
val_loader_mendeley = DataLoader(mendeley_val, batch_size=16, shuffle=False)
test_loader_mp = DataLoader(mp_dataset, batch_size=16, shuffle=False)

# Train 500-Epoch Mendeley Model & track transfer MAE on MP dataset
model_mendeley = EquivarBECGNN(num_species=95, hidden_dim=64, num_layers=3).to(device)
optimizer = AdamW(model_mendeley.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

train_losses = []
val_losses = []
mp_transfer_losses = []

print("\n--- TRAINING MENDELEY MODEL (500 EPOCHS) & EVALUATING MP TRANSFER ---")
for epoch in range(1, 501):
    model_mendeley.train()
    total_loss, total_graphs = 0.0, 0
    for batch in train_loader_mendeley:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model_mendeley(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
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
        model_mendeley.eval()
        v_loss, v_graphs = 0.0, 0
        with torch.no_grad():
            for batch in val_loader_mendeley:
                batch = batch.to(device)
                out = model_mendeley(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                v_loss += loss.item() * batch.num_graphs
                v_graphs += batch.num_graphs
        val_mae = v_loss / v_graphs
        val_losses.append(val_mae)
        
        # Zero-shot transfer on MP
        t_loss, t_graphs = 0.0, 0
        with torch.no_grad():
            for batch in test_loader_mp:
                batch = batch.to(device)
                out = model_mendeley(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
                natoms = batch.ptr[1:] - batch.ptr[:-1]
                _s = scatter(out, batch.batch, dim=0, reduce='sum') / natoms[:, None, None]
                _index = batch.batch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
                out_asr = out - torch.gather(_s, dim=0, index=_index)
                loss = torch.mean(torch.abs(out_asr - batch.y))
                t_loss += loss.item() * batch.num_graphs
                t_graphs += batch.num_graphs
        mp_transfer_mae = t_loss / t_graphs
        mp_transfer_losses.append(mp_transfer_mae)
        print(f"Mendeley Model | Epoch {epoch:03d}/500 | Train MAE: {tr_mae:.4f} e | Val MAE: {val_mae:.4f} e | MP Zero-Shot Transfer MAE: {mp_transfer_mae:.4f} e")
    else:
        val_losses.append(val_losses[-1] if val_losses else tr_mae)
        mp_transfer_losses.append(mp_transfer_losses[-1] if mp_transfer_losses else tr_mae)

final_transfer_mae = mp_transfer_losses[-1]
print(f"\n==================================================")
print(f"---> FINAL ZERO-SHOT MAE (Train Mendeley -> Test MP): {final_transfer_mae:.4f} e")
print(f"==================================================")

# Plot Loss Curves
plt.figure(figsize=(9, 5))
epochs = range(1, 501)
plt.plot(epochs, train_losses, label='Mendeley Training Loss (MAE)', color='blue', linewidth=1.5)
plt.plot(epochs, val_losses, label='Mendeley Validation Loss (MAE)', color='green', linewidth=1.5)
plt.plot(epochs, mp_transfer_losses, label='Materials Project Zero-Shot Transfer MAE', color='red', linewidth=1.5)
plt.title('500-Epoch Transfer: Train on Mendeley -> Test Zero-Shot on Materials Project', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('MAE Loss (e)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plot_out = os.path.join(output_dir, "loss_curve_mendeley_to_mp_500ep_verified.png")
plt.savefig(plot_out, dpi=300)
plt.close()
print(f"Saved verified transfer loss plot to: {plot_out}")

# Save CSV Summary
csv_out = os.path.join(output_dir, "mendeley_to_mp_transfer_500ep_results.csv")
pd.DataFrame([{
    'Experiment': '500-Epoch Transfer (Train Mendeley -> Test MP)',
    'Train_Graphs': len(mendeley_train),
    'Val_Graphs': len(mendeley_val),
    'Test_MP_Graphs': len(mp_dataset),
    'Final_Train_Mendeley_MAE_e': train_losses[-1],
    'Final_Val_Mendeley_MAE_e': val_losses[-1],
    'Final_ZeroShot_MP_MAE_e': final_transfer_mae,
    'Loss_Plot': 'loss_curve_mendeley_to_mp_500ep_verified.png'
}]).to_csv(csv_out, index=False)
print(f"Saved results summary CSV to: {csv_out}")
print("==================================================")
