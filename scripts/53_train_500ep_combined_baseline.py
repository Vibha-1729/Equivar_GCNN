"""
53_train_500ep_combined_baseline.py
-----------------------------------
500-Epoch GPU Baseline Training of Official Equivar GNN on the Unified 3-Way Dataset:
  - Materials Project (14,565 crystals)
  - Mendeley Oxides (29,318 crystals)
  - Authentic JARVIS-DFT (4,303 crystals)
Total: 48,186 crystal graphs with 100% genuine DFPT Born Effective Charges.
- Exact parameter-free Acoustic Sum Rule (ASR) projection layer.
- Exact stoichiometric formulas.
- GPU acceleration via NVIDIA GeForce RTX 5060 (CUDA).
- Full 80/10/10 evaluation, per-atom CSV export, per-crystal JSON export, and per-dataset metric breakdowns.
"""

import os
import sys
import time
import math
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from pymatgen.core import Element

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
args = parser.parse_args()

# ── 1. Device & Setup ────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("=" * 65)
print("  EQUIVAR GNN — 500-EPOCH UNIFIED 3-WAY MASTER BASELINE TRAINING")
print(f"  Materials Project + Mendeley Oxides + JARVIS-DFT")
print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 65)

# ── 2. Change of Basis Matrix: Spherical Irreps -> Cartesian 3x3 ─────────────
COB_MATRIX = torch.tensor([
    [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.577350,  0.000000, -0.408248],
    [ 0.000000,  0.000000,  0.707107,  0.000000,  0.000000,  0.000000,  0.000000,  0.707107,  0.000000],
    [ 0.000000, -0.707107,  0.000000,  0.000000,  0.707107,  0.000000,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  0.000000, -0.707107,  0.000000,  0.000000,  0.000000,  0.000000,  0.707107,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.577350,  0.000000, -0.408248],
    [ 0.707107,  0.000000,  0.000000,  0.707107,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  0.707107,  0.000000,  0.000000,  0.707107,  0.000000,  0.000000,  0.000000,  0.000000],
    [-0.707107,  0.000000,  0.000000,  0.707107,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.577350,  0.000000,  0.816497]
], dtype=torch.float32)

class GaussianCosEnvelopeBasisProjection(nn.Module):
    def __init__(self, r_min=0.0, r_max=5.0, num_basis=32):
        super().__init__()
        self.r_max = r_max
        means = torch.linspace(r_min, r_max, num_basis)
        self.register_buffer("means", means)
        self.sigma = (r_max - r_min) / (num_basis - 1)

    def forward(self, r):
        r = r.unsqueeze(-1)
        envelope = 0.5 * (torch.cos(math.pi * r / self.r_max) + 1.0)
        envelope = torch.where(r < self.r_max, envelope, torch.zeros_like(envelope))
        return torch.exp(-((r - self.means) ** 2) / (2 * self.sigma ** 2)) * envelope

class EquivarBECGNN(nn.Module):
    def __init__(self, num_species=95, hidden_dim=64, lmax=2, num_layers=3, r_max=5.0, num_radial=32):
        super().__init__()
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.radial_proj = GaussianCosEnvelopeBasisProjection(0.0, r_max, num_radial)
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_sh   = o3.Irreps.spherical_harmonics(lmax)
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
            
        # Output head (9 irreps)
        msg_out = self.out_tp(h[src], edge_sh) * radial_weights[:, :9]
        out_irreps = torch.zeros(num_nodes, 9, device=z.device, dtype=h.dtype).index_add_(0, dst, msg_out)
        
        # Change-of-basis to Cartesian coordinates
        out_cart = out_irreps @ self.cob  # [N, 9]
        
        # Acoustic Sum Rule (ASR) Exact Charge Neutrality Projection
        natoms = scatter(torch.ones(num_nodes, device=z.device), batch, dim=0, reduce='sum')
        sum_per_graph = scatter(out_cart, batch, dim=0, reduce='sum') / natoms[:, None]
        index_expanded = batch.unsqueeze(1).expand(-1, 9)
        out_neutral_flat = out_cart - torch.gather(sum_per_graph, dim=0, index=index_expanded)
        
        return out_neutral_flat.view(num_nodes, 3, 3)

# ── 3. Paths & Dataset Loading ───────────────────────────────────────────────
base_dir = "/home/vibhan23/combined_bec_dataset"
models_dir = os.path.join(base_dir, "models")
plots_dir  = os.path.join(base_dir, "plots")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

dataset_path = os.path.join(base_dir, "combined_pyg_dataset.pt")
print(f"Loading Unified PyG dataset: {dataset_path} ...")
graphs = torch.load(dataset_path, weights_only=False)
n_total = len(graphs)
print(f"Loaded {n_total:,} Unified crystal graphs.")

# 80/10/10 split with seed=42 (identical to dataset builder split)
np.random.seed(42)
indices = np.random.permutation(n_total)
n_train = int(0.80 * n_total)
n_val   = int(0.10 * n_total)

train_graphs = [graphs[i] for i in indices[:n_train]]
val_graphs   = [graphs[i] for i in indices[n_train:n_train + n_val]]
test_graphs  = [graphs[i] for i in indices[n_train + n_val:]]

print(f"Split: Train={len(train_graphs):,} | Val={len(val_graphs):,} | Test={len(test_graphs):,}")

train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

# ── 4. Initialize Model & Training Loop ─────────────────────────────────────
model = EquivarBECGNN(num_species=95, hidden_dim=64, lmax=2, num_layers=3, r_max=5.0).to(device)
print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

epochs = args.epochs
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
ckpt_path = os.path.join(models_dir, "equivar_combined_500ep_baseline_best.pt")

best_val_mae = float('inf')
history = []
t0 = time.time()

print(f"\nStarting {epochs}-Epoch Unified Master Baseline Training...")
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
    
    if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
        elapsed = time.time() - t0
        print(f"Epoch [{epoch:03d}/{epochs}] | Train MAE: {train_mae:.4f} e | Val MAE: {val_mae:.4f} e (Best: {best_val_mae:.4f} e) | Elapsed: {elapsed:.1f}s")

# ── 5. Save History & Clean Publication Loss Curve ───────────────────────────
hist_df = pd.DataFrame(history)
hist_csv = os.path.join(base_dir, "combined_500ep_baseline_training_history.csv")
hist_df.to_csv(hist_csv, index=False)
print(f"\nSaved training history to: {hist_csv}")

plt.figure(figsize=(9, 5.5), dpi=300)
plt.plot(hist_df['epoch'], hist_df['train_mae'], label='Training Loss (MAE)', color='#1f77b4', linewidth=2.0)
plt.plot(hist_df['epoch'], hist_df['val_mae'], label='Validation Loss (MAE)', color='#ff7f0e', linewidth=2.0, linestyle='--')
plt.title('Unified 3-Way Combined Baseline Equivar Model (500 Epochs)\nMaterials Project + Mendeley Oxides + JARVIS-DFT', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('MAE Loss (e)', fontsize=11, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plot_path = os.path.join(plots_dir, "loss_curve_combined_500ep_baseline.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved publication loss curve to: {plot_path}")

# Zoomed Plot
plt.figure(figsize=(9, 5.5), dpi=300)
plt.plot(hist_df['epoch'], hist_df['train_mae'], label='Training Loss (MAE)', color='#1f77b4', linewidth=2.0)
plt.plot(hist_df['epoch'], hist_df['val_mae'], label='Validation Loss (MAE)', color='#ff7f0e', linewidth=2.0, linestyle='--')
min_loss = min(hist_df['train_mae'].min(), hist_df['val_mae'].min())
max_loss_zoom = hist_df.iloc[10:]['val_mae'].max() * 1.15 if len(hist_df) > 10 else hist_df['val_mae'].max()
plt.ylim(max(0.0, min_loss - 0.05), max_loss_zoom)
plt.title('Unified 3-Way Combined Baseline Equivar Model — Convergence Zoom', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('MAE Loss (e)', fontsize=11, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plot_zoom_path = os.path.join(plots_dir, "loss_curve_combined_500ep_zoomed.png")
plt.savefig(plot_zoom_path)
plt.close()
print(f"Saved zoomed loss curve to: {plot_zoom_path}")

# ── 6. Full Inference & Predictions Export ───────────────────────────────────
print("\nLoading best model checkpoint for evaluation & full per-atom prediction export...")
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()

split_map = {}
for i in indices[:n_train]: split_map[i] = "train"
for i in indices[n_train:n_train+n_val]: split_map[i] = "val"
for i in indices[n_train+n_val:]: split_map[i] = "test"

Z_SYMBOL = {z: Element.from_Z(z).symbol for z in range(1, 96)}

csv_rows = []
crystal_json_list = []

print("Running inference across all 48,186 structures...")
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
            
            elem_sym = Z_SYMBOL.get(int(zn[ai]), f"Z{int(zn[ai])}")
            
            atom_json_list.append({
                "atom_index": ai,
                "element": elem_sym,
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
                "element": elem_sym, "atomic_number": int(zn[ai]),
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
        
        if (gi + 1) % 10000 == 0 or (gi + 1) == n_total:
            print(f"  {gi+1:,} / {n_total:,} structures processed...")

pred_csv_path = os.path.join(base_dir, "combined_bec_predictions_baseline_all_atoms.csv")
pred_df = pd.DataFrame(csv_rows)
pred_df.to_csv(pred_csv_path, index=False)
print(f"\n[1] Saved per-atom CSV ({len(pred_df):,} rows) -> {pred_csv_path}")

pred_json_path = os.path.join(base_dir, "combined_bec_predictions_baseline_per_atom_3x3.json")
with open(pred_json_path, 'w') as f:
    json.dump(crystal_json_list, f, indent=2)
print(f"[2] Saved per-crystal 3x3 JSON ({len(crystal_json_list):,} crystals) -> {pred_json_path}")

# ── 7. Calculate and Save Model Metrics (Train / Val / Test & Per-Dataset) ───
metrics_records = []

# Overall Split Metrics
for s_label in ["train", "val", "test"]:
    sub = pred_df[pred_df['split_type'] == s_label]
    if len(sub) == 0: continue
    
    diff_overall = sub['atom_mae_e'].values
    diag_diffs = np.mean(np.abs(np.array([
        sub['pred_Zxx'] - sub['target_Zxx'],
        sub['pred_Zyy'] - sub['target_Zyy'],
        sub['pred_Zzz'] - sub['target_Zzz']
    ])), axis=0)
    
    offdiag_diffs = np.mean(np.abs(np.array([
        sub['pred_Zxy'] - sub['target_Zxy'],
        sub['pred_Zxz'] - sub['target_Zxz'],
        sub['pred_Zyx'] - sub['target_Zyx'],
        sub['pred_Zyz'] - sub['target_Zyz'],
        sub['pred_Zzx'] - sub['target_Zzx'],
        sub['pred_Zzy'] - sub['target_Zzy']
    ])), axis=0)
    
    trace_diffs = np.abs(sub['atom_trace_pred'] - sub['atom_trace_target']).values
    
    metrics_records.append({
        "subset": f"Overall ({s_label})",
        "split": s_label,
        "num_structures": len(sub['material_id'].unique()),
        "num_atoms": len(sub),
        "overall_mae_e": round(float(np.mean(diff_overall)), 5),
        "diagonal_mae_e": round(float(np.mean(diag_diffs)), 5),
        "offdiagonal_mae_e": round(float(np.mean(offdiag_diffs)), 5),
        "trace_mae_e": round(float(np.mean(trace_diffs)), 5)
    })

# Per-Dataset Breakdown on Test Set
for dname in ["materials_project", "mendeley", "jarvis"]:
    sub = pred_df[(pred_df['split_type'] == 'test') & (pred_df['dataset'].str.contains(dname))]
    if len(sub) == 0: continue
    
    diff_overall = sub['atom_mae_e'].values
    diag_diffs = np.mean(np.abs(np.array([
        sub['pred_Zxx'] - sub['target_Zxx'],
        sub['pred_Zyy'] - sub['target_Zyy'],
        sub['pred_Zzz'] - sub['target_Zzz']
    ])), axis=0)
    
    offdiag_diffs = np.mean(np.abs(np.array([
        sub['pred_Zxy'] - sub['target_Zxy'],
        sub['pred_Zxz'] - sub['target_Zxz'],
        sub['pred_Zyx'] - sub['target_Zyx'],
        sub['pred_Zyz'] - sub['target_Zyz'],
        sub['pred_Zzx'] - sub['target_Zzx'],
        sub['pred_Zzy'] - sub['target_Zzy']
    ])), axis=0)
    
    trace_diffs = np.abs(sub['atom_trace_pred'] - sub['atom_trace_target']).values
    
    metrics_records.append({
        "subset": f"Test Set - {dname.upper()}",
        "split": "test",
        "num_structures": len(sub['material_id'].unique()),
        "num_atoms": len(sub),
        "overall_mae_e": round(float(np.mean(diff_overall)), 5),
        "diagonal_mae_e": round(float(np.mean(diag_diffs)), 5),
        "offdiagonal_mae_e": round(float(np.mean(offdiag_diffs)), 5),
        "trace_mae_e": round(float(np.mean(trace_diffs)), 5)
    })

metrics_df = pd.DataFrame(metrics_records)
metrics_csv = os.path.join(base_dir, "combined_model_mae_metrics_baseline.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"\n[3] Saved Model Metrics CSV -> {metrics_csv}")
print(metrics_df.to_string(index=False))

print("\n" + "=" * 65)
print("  500-EPOCH UNIFIED MASTER BASELINE TRAINING & EVALUATION COMPLETE!")
print("=" * 65)
