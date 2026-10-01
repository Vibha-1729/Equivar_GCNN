"""
51_train_500ep_mendeley_baseline.py
-----------------------------------
Official Baseline Equivar GNN Training on Mendeley Oxides (29,318 structures)
Following https://github.com/equivar/equivar_eval:
  - Gaussian Cosine Envelope Radial Basis
  - Radial-gated Equivariant Message Passing: msg = layer(h, sh) * radial_weights
  - Change-of-Basis (cob) transformation matrix
  - Differentiable Acoustic Sum Rule (ASR) charge neutrality in forward pass
  - Automatic export of CSV + 3x3 Matrix JSON + MAE Summary Table
"""

import os
import math
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

try:
    from torch_scatter import scatter
except ImportError:
    from torch_geometric.utils import scatter

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct

print("=" * 60)
print("  OFFICIAL BASELINE EQUIVAR GNN TRAINING — MENDELEY OXIDES")
print("  Following: https://github.com/equivar/equivar_eval")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute device: {device}")

# ── 1. Official Change-of-Basis (cob) Matrix ─────────────────────────────────
SQ2_1  = 1.0 / math.sqrt(2.0)
SQ3_1  = 1.0 / math.sqrt(3.0)
SQ23_1 = SQ2_1 * SQ3_1

COB_MATRIX = torch.tensor([
    [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,  -SQ2_1, 0.0,   0.0,   0.0,  ],     # t11
    [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,   SQ2_1,],     # t12
    [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,  -SQ2_1, 0.0,  ],     # t13
    [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,  -SQ2_1,],     # t21
    [SQ3_1, 0.0,   0.0,   2.0*SQ23_1, 0.0,   0.0,   0.0,   0.0,   0.0,  ],     # t22
    [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,   SQ2_1, 0.0,   0.0,  ],     # t23
    [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,   SQ2_1, 0.0,  ],     # t31
    [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,  -SQ2_1, 0.0,   0.0,  ],     # t32
    [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,   SQ2_1, 0.0,   0.0,   0.0,  ],     # t33
], dtype=torch.float32).T

# ── 2. Official Radial Basis Projection ─────────────────────────────────────
class GaussianCosEnvelopeBasisProjection(nn.Module):
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.alpha = math.pi / stop
        self.gamma = -0.5 / (offset[1] - offset[0]).item()**2

    def forward(self, r: torch.Tensor):
        rcol = r.view(-1, 1)
        roffset = rcol - self.offset.view(1, -1)
        envelope = 0.5 * (1.0 + torch.cos(self.alpha * rcol))
        return envelope * torch.exp(self.gamma * torch.pow(roffset, 2))

# ── 3. Equivar GNN Architecture ──────────────────────────────────────────────
class EquivarBECGNN(nn.Module):
    def __init__(self, num_species=95, hidden_dim=64, lmax=2, num_layers=3, r_max=5.0, num_radial=32):
        super().__init__()
        self.embedding = nn.Embedding(num_species, hidden_dim)
        
        self.radial_proj = GaussianCosEnvelopeBasisProjection(start=0.0, stop=r_max, num_gaussians=num_radial)
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        
        self.layers = nn.ModuleList([
            FullyConnectedTensorProduct(self.irreps_node, self.irreps_sh, self.irreps_node, shared_weights=True)
            for _ in range(num_layers)
        ])
        
        self.irreps_out = o3.Irreps("1x0e + 1x1o + 1x2e")
        self.out_tp = FullyConnectedTensorProduct(
            self.irreps_node, self.irreps_sh, self.irreps_out, shared_weights=True
        )
        self.register_buffer("cob", COB_MATRIX)

    def forward(self, z, pos, edge_index, edge_vec, batch=None):
        num_nodes = z.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=z.device)
            
        h = self.embedding(z)
        
        # Radial distance gating
        edge_lengths = torch.norm(edge_vec, dim=-1)
        radial_emb = self.radial_proj(edge_lengths)
        radial_weights = self.radial_mlp(radial_emb)
        
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization='component'
        )
        
        src, dst = edge_index[0], edge_index[1]
        
        # Message passing with radial distance modulation
        for layer in self.layers:
            msg = layer(h[src], edge_sh) * radial_weights
            h_agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = h + h_agg
            
        # Output head
        msg_out = self.out_tp(h[src], edge_sh) * radial_weights[:, :9]
        out_irreps = torch.zeros(num_nodes, 9, device=z.device, dtype=h.dtype).index_add_(0, dst, msg_out)
        
        # Change-of-basis to Cartesian coordinates
        out_cart = out_irreps @ self.cob  # [N, 9]
        
        # Acoustic Sum Rule (ASR) Charge Neutrality
        natoms = scatter(torch.ones(num_nodes, device=z.device), batch, dim=0, reduce='sum')
        sum_per_graph = scatter(out_cart, batch, dim=0, reduce='sum') / natoms[:, None]
        index_expanded = batch.unsqueeze(1).expand(-1, 9)
        out_neutral_flat = out_cart - torch.gather(sum_per_graph, dim=0, index=index_expanded)
        
        return out_neutral_flat.view(num_nodes, 3, 3)

# ── 4. Paths & Dataset ───────────────────────────────────────────────────────
base_dir = "/home/vibhan23/mendeley_oxides_dataset"
if not os.path.exists(base_dir):
    base_dir = os.path.abspath("mendeley_oxides_dataset")

models_dir = os.path.join(base_dir, "models")
plots_dir  = os.path.join(base_dir, "plots")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

dataset_path = os.path.join(base_dir, "mendeley_pyg_dataset.pt")
print(f"Loading Mendeley PyG dataset: {dataset_path}")
graphs = torch.load(dataset_path, weights_only=False)
n_total = len(graphs)
print(f"Loaded {n_total:,} Mendeley crystal graphs.")

# 80/10/10 split with seed=42
np.random.seed(42)
indices = np.random.permutation(n_total)
n_train = int(0.80 * n_total)
n_val   = int(0.10 * n_total)

train_graphs = [graphs[i] for i in indices[:n_train]]
val_graphs   = [graphs[i] for i in indices[n_train:n_train + n_val]]
test_graphs  = [graphs[i] for i in indices[n_train + n_val:]]

print(f"Split: Train={len(train_graphs):,} | Val={len(val_graphs):,} | Test={len(test_graphs):,}")

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

# ── 5. Initialize Model & Training Loop ─────────────────────────────────────
model = EquivarBECGNN(num_species=95, hidden_dim=64, lmax=2, num_layers=3, r_max=5.0).to(device)
print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

epochs = 500
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
ckpt_path = os.path.join(models_dir, "equivar_mendeley_500ep_baseline_best.pt")

best_val_mae = float('inf')
history = []
t0 = time.time()

print("\nStarting 500-Epoch Mendeley Baseline Training...")
for epoch in range(1, epochs + 1):
    model.train()
    total_train_loss = 0.0
    total_train_nodes = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        loss = torch.mean(torch.abs(pred - batch.y))
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() * batch.num_nodes
        total_train_nodes += batch.num_nodes
        
    scheduler.step()
    train_mae = total_train_loss / total_train_nodes
    
    # Validation evaluation at every epoch
    model.eval()
    total_val_loss = 0.0
    total_val_nodes = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
            loss = torch.mean(torch.abs(pred - batch.y))
            total_val_loss += loss.item() * batch.num_nodes
            total_val_nodes += batch.num_nodes
            
    val_mae = total_val_loss / total_val_nodes
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), ckpt_path)
        
    history.append({
        "epoch": epoch,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "lr": scheduler.get_last_lr()[0]
    })
    
    if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
        elapsed = time.time() - t0
        print(f"Epoch [{epoch:03d}/500] | Train MAE: {train_mae:.4f} e | Val MAE: {val_mae:.4f} e (Best: {best_val_mae:.4f} e) | Elapsed: {elapsed:.1f}s")

# ── 6. Save History & Clean Publication Loss Curve ───────────────────────────
hist_df = pd.DataFrame(history)
hist_csv = os.path.join(base_dir, "mendeley_500ep_baseline_training_history.csv")
hist_df.to_csv(hist_csv, index=False)
print(f"\nSaved training history to: {hist_csv}")

plt.figure(figsize=(9, 5.5), dpi=300)
plt.plot(hist_df['epoch'], hist_df['train_mae'], label='Training Loss (MAE)', color='#1f77b4', linewidth=2.0)
plt.plot(hist_df['epoch'], hist_df['val_mae'], label='Validation Loss (MAE)', color='#ff7f0e', linewidth=2.0, linestyle='--')
plt.title('Mendeley Oxides Baseline Equivar Model (500 Epochs)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('MAE Loss (e)', fontsize=11, fontweight='bold')
plt.ylim(0.20, 0.80)  # Physical scale matching baseline paper!
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "loss_curve_mendeley_500ep_baseline.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved publication loss curve to: {plot_path}")

# ── 7. Evaluate on Held-out Test Set & Export Full Predictions ────────────────
print("\nLoading best model checkpoint for evaluation & full per-atom prediction export...")
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()

split_map = {}
for i in indices[:n_train]: split_map[i] = "train"
for i in indices[n_train:n_train+n_val]: split_map[i] = "val"
for i in indices[n_train+n_val:]: split_map[i] = "test"

Z_SYMBOL = {
    1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
    11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',
    21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',
    31:'Ga',32:'Ge',33:'As',34:'Se',35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',
    41:'Nb',42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
    51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',58:'Ce',59:'Pr',60:'Nd',
    72:'Hf',73:'Ta',74:'W',82:'Pb',83:'Bi'
}

csv_rows = []
crystal_json_list = []

print("Running inference across all structures...")
with torch.no_grad():
    for gi, g in enumerate(graphs):
        g = g.to(device)
        pred = model(g.z, g.pos, g.edge_index, g.edge_vec, batch=g.batch)
        
        yt = g.y.cpu().numpy()
        yp = pred.cpu().numpy()
        zn = g.z.cpu().numpy()
        
        mid     = g.material_id if isinstance(g.material_id, str) else g.material_id[0]
        dset    = g.dataset if hasattr(g, 'dataset') and g.dataset else ""
        formula = g.formula if hasattr(g, 'formula') and g.formula else ""
        split   = split_map[gi]
        n_atoms = len(zn)
        
        atom_json_list = []
        struct_mae_list = []
        
        for ai in range(n_atoms):
            t = yt[ai]
            p = yp[ai]
            mae = float(np.mean(np.abs(p - t)))
            struct_mae_list.append(mae)
            
            t_mat = [
                [round(float(t[0,0]), 5), round(float(t[0,1]), 5), round(float(t[0,2]), 5)],
                [round(float(t[1,0]), 5), round(float(t[1,1]), 5), round(float(t[1,2]), 5)],
                [round(float(t[2,0]), 5), round(float(t[2,1]), 5), round(float(t[2,2]), 5)]
            ]
            p_mat = [
                [round(float(p[0,0]), 5), round(float(p[0,1]), 5), round(float(p[0,2]), 5)],
                [round(float(p[1,0]), 5), round(float(p[1,1]), 5), round(float(p[1,2]), 5)],
                [round(float(p[2,0]), 5), round(float(p[2,1]), 5), round(float(p[2,2]), 5)]
            ]
            
            atom_json_list.append({
                "atom_index": ai,
                "element": Z_SYMBOL.get(int(zn[ai]), "?"),
                "atomic_number": int(zn[ai]),
                "target_bec_tensor_3x3": t_mat,
                "pred_bec_tensor_3x3": p_mat,
                "atom_mae_e": round(mae, 5),
                "atom_trace_target": round(float(np.trace(t)/3.0), 5),
                "atom_trace_pred": round(float(np.trace(p)/3.0), 5)
            })
            
            csv_rows.append({
                "material_id": mid, "dataset": dset, "formula": formula,
                "split_type": split, "num_atoms": n_atoms, "atom_index": ai,
                "element": Z_SYMBOL.get(int(zn[ai]), "?"), "atomic_number": int(zn[ai]),
                "target_Zxx": t_mat[0][0], "target_Zxy": t_mat[0][1], "target_Zxz": t_mat[0][2],
                "target_Zyx": t_mat[1][0], "target_Zyy": t_mat[1][1], "target_Zyz": t_mat[1][2],
                "target_Zzx": t_mat[2][0], "target_Zzy": t_mat[2][1], "target_Zzz": t_mat[2][2],
                "pred_Zxx": p_mat[0][0],   "pred_Zxy": p_mat[0][1],   "pred_Zxz": p_mat[0][2],
                "pred_Zyx": p_mat[1][0],   "pred_Zyy": p_mat[1][1],   "pred_Zyz": p_mat[1][2],
                "pred_Zzx": p_mat[2][0],   "pred_Zzy": p_mat[2][1],   "pred_Zzz": p_mat[2][2],
                "atom_mae_e": round(mae, 5),
                "atom_trace_target": round(float(np.trace(t)/3.0), 5),
                "atom_trace_pred": round(float(np.trace(p)/3.0), 5)
            })
            
        crystal_json_list.append({
            "material_id": mid,
            "dataset": dset,
            "formula": formula,
            "split_type": split,
            "num_atoms": n_atoms,
            "mean_structure_mae_e": round(float(np.mean(struct_mae_list)), 5),
            "atoms": atom_json_list
        })
        
        if (gi + 1) % 5000 == 0:
            print(f"  {gi+1:,} / {n_total:,} structures processed...")

pred_csv_path = os.path.join(base_dir, "mendeley_bec_predictions_baseline_all_atoms.csv")
pred_df = pd.DataFrame(csv_rows)
pred_df.to_csv(pred_csv_path, index=False)
print(f"\n[1] Saved per-atom CSV ({len(pred_df):,} rows) -> {pred_csv_path}")

pred_json_path = os.path.join(base_dir, "mendeley_bec_predictions_baseline_per_atom_3x3.json")
with open(pred_json_path, 'w') as f:
    json.dump(crystal_json_list, f, indent=2)
print(f"[2] Saved per-atom 3x3 Matrix JSON ({len(crystal_json_list):,} crystals) -> {pred_json_path}")

# ── 8. Export Summary MAE Metrics CSV ────────────────────────────────────────
mae_rows = []
for split in ['train', 'val', 'test', 'overall']:
    sub = pred_df if split == 'overall' else pred_df[pred_df['split_type'] == split]
    overall_mae = sub['atom_mae_e'].mean()
    diag_mae = ((sub['pred_Zxx'] - sub['target_Zxx']).abs().mean() + 
                (sub['pred_Zyy'] - sub['target_Zyy']).abs().mean() + 
                (sub['pred_Zzz'] - sub['target_Zzz']).abs().mean()) / 3.0
    offdiag_mae = ((sub['pred_Zxy'] - sub['target_Zxy']).abs().mean() + 
                   (sub['pred_Zxz'] - sub['target_Zxz']).abs().mean() + 
                   (sub['pred_Zyx'] - sub['target_Zyx']).abs().mean() + 
                   (sub['pred_Zyz'] - sub['target_Zyz']).abs().mean() + 
                   (sub['pred_Zzx'] - sub['target_Zzx']).abs().mean() + 
                   (sub['pred_Zzy'] - sub['target_Zzy']).abs().mean()) / 6.0
    trace_mae = (sub['atom_trace_pred'] - sub['atom_trace_target']).abs().mean()
    
    mae_rows.append({
        "split_name": f"{split.capitalize()} Set" if split != 'overall' else "Overall Full Dataset",
        "split_type": split,
        "num_crystals": int(sub['material_id'].nunique()),
        "num_atoms": len(sub),
        "overall_bec_tensor_mae_e": round(float(overall_mae), 5),
        "diagonal_mae_e": round(float(diag_mae), 5),
        "off_diagonal_mae_e": round(float(offdiag_mae), 5),
        "isotropic_trace_mae_e": round(float(trace_mae), 5),
        "best_epoch": int(hist_df.loc[hist_df['val_mae'].idxmin(), 'epoch']),
        "best_val_mae_e": round(float(best_val_mae), 5)
    })

mae_metrics_df = pd.DataFrame(mae_rows)
metrics_csv_path = os.path.join(base_dir, "mendeley_model_mae_metrics_baseline.csv")
mae_metrics_df.to_csv(metrics_csv_path, index=False)
print(f"[3] Saved baseline MAE metrics CSV -> {metrics_csv_path}")
print("\n=== Baseline MAE Metrics Summary ===")
print(mae_metrics_df[['split_name', 'num_crystals', 'overall_bec_tensor_mae_e', 'off_diagonal_mae_e', 'isotropic_trace_mae_e']].to_string(index=False))

print("\n" + "=" * 60)
print("  OFFICIAL BASELINE MENDELEY TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
