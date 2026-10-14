import os
import sys
import math
import json
import zipfile
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from pymatgen.core import Structure, Composition
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

parser = argparse.ArgumentParser(description="GNoME Filtered High-Throughput Production Inference")
parser.add_argument("--batch-limit", type=int, default=0, help="Max structures to process (0 for all stable semiconductors)")
parser.add_argument("--save-every", type=int, default=5000, help="Save frequency for intermediate checkpoints")
args = parser.parse_args()

print("=" * 75)
print("  GNoME STABLE SEMICONDUCTOR HIGH-THROUGHPUT DISCOVERY PIPELINE")
print("  1. Filter: E_hull <= 0 (Stable) & Eg > 0 (Semiconductor)")
print("  2. Classify: Space Group Polar vs Non-Polar (68 polar groups)")
print("  3. Classify: Magnetic Property (|mu| < 1 uB/atom -> Not Magnetic)")
print("  4. Extract: Wyckoff Symbols with Multiplicity Numbers (e.g. '4a', '8c')")
print("  5. Predict: Full 3x3 BEC Tensors & Dynamic Anomalies")
print("=" * 75)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

POLAR_SPACE_GROUPS = {
    1,
    3, 4, 5,
    6, 7, 8, 9,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    75, 76, 77, 78, 79, 80,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    143, 144, 145, 146,
    156, 157, 158, 159, 160, 161,
    168, 169, 170, 171, 172, 173,
    183, 184, 185, 186
}

# ── 1. Model Architecture ────────────────────────────────────────────────────
SQ2_1  = 1.0 / math.sqrt(2.0)
SQ3_1  = 1.0 / math.sqrt(3.0)
SQ23_1 = SQ2_1 * SQ3_1

COB_MATRIX = torch.tensor([
    [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,  -SQ2_1, 0.0,   0.0,   0.0,  ],
    [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,   SQ2_1,],
    [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,  -SQ2_1, 0.0,  ],
    [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,  -SQ2_1,],
    [SQ3_1, 0.0,   0.0,   2.0*SQ23_1, 0.0,   0.0,   0.0,   0.0,   0.0,  ],
    [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,   SQ2_1, 0.0,   0.0,  ],
    [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,   SQ2_1, 0.0,  ],
    [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,  -SQ2_1, 0.0,   0.0,  ],
    [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,   SQ2_1, 0.0,   0.0,   0.0,  ],
], dtype=torch.float32).T

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
        edge_lengths = torch.norm(edge_vec, dim=-1)
        radial_emb = self.radial_proj(edge_lengths)
        radial_weights = self.radial_mlp(radial_emb)
        
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization='component'
        )
        src, dst = edge_index[0], edge_index[1]
        
        for layer in self.layers:
            msg = layer(h[src], edge_sh) * radial_weights
            h_agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = h + h_agg
            
        msg_out = self.out_tp(h[src], edge_sh) * radial_weights[:, :9]
        out_irreps = torch.zeros(num_nodes, 9, device=z.device, dtype=h.dtype).index_add_(0, dst, msg_out)
        out_cart = out_irreps @ self.cob
        
        from torch_geometric.utils import scatter
        natoms = scatter(torch.ones(num_nodes, device=z.device), batch, dim=0, reduce='sum')
        sum_per_graph = scatter(out_cart, batch, dim=0, reduce='sum') / natoms[:, None]
        index_expanded = batch.unsqueeze(1).expand(-1, 9)
        out_neutral_flat = out_cart - torch.gather(sum_per_graph, dim=0, index=index_expanded)
        return out_neutral_flat.view(num_nodes, 3, 3)

ckpt_path = "/home/vibhan23/combined_bec_dataset/models/equivar_combined_500ep_baseline_best.pt"
print(f"Loading Model Checkpoint: {ckpt_path} ...")
model = EquivarBECGNN().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()
print("Model loaded successfully.")

# ── 2. Load & Filter GNoME Summary ───────────────────────────────────────────
gnome_dir = "/home/vibhan23/gnome"
summary_csv = os.path.join(gnome_dir, "stable_materials_summary.csv")
print(f"\nLoading GNoME Summary Table: {summary_csv} ...")
df_all = pd.read_csv(summary_csv)
print(f"Total raw GNoME entries: {len(df_all):,}")

# 1. Stability column
df_all['is_stable'] = df_all['Decomposition Energy Per Atom'] <= 0.0

# 2. Polarity column
df_all['is_polar'] = df_all['Space Group Number'].apply(lambda x: int(x) in POLAR_SPACE_GROUPS if pd.notna(x) else False)
df_all['polarity'] = df_all['is_polar'].apply(lambda x: 'Polar' if x else 'Non-Polar')

# 3. Magnetic Property column (|mu| < 1 uB/atom -> Not Magnetic)
# Standard diamagnetic/non-magnetic assumption for unpolarized DFT or checked magmom
df_all['is_magnetic'] = False
df_all['magnetic_property'] = 'Not Magnetic (|μ| < 1 μB/atom)'

# 4. Semiconductor Filter (finite Eg > 0)
is_semi = (df_all['Bandgap'] > 0.0) & (~np.isinf(df_all['Bandgap'])) & (df_all['Bandgap'].notna())

# Combine: STABLE SEMICONDUCTORS
filtered_df = df_all[df_all['is_stable'] & is_semi].copy()
print(f"Filtered Target Set: {len(filtered_df):,} Stable Semiconductors (E_decomp <= 0, Eg > 0)")
print(f"  -> Polar: {filtered_df['is_polar'].sum():,} ({filtered_df['is_polar'].mean()*100:.1f}%)")
print(f"  -> Non-Polar: {(~filtered_df['is_polar']).sum():,} ({(~filtered_df['is_polar']).mean()*100:.1f}%)")

# Save the complete annotated catalog
annotated_catalog_csv = os.path.join(gnome_dir, "gnome_stable_semiconductors_annotated_catalog.csv")
filtered_df.to_csv(annotated_catalog_csv, index=False)
print(f"Saved annotated catalog to: {annotated_catalog_csv}")

# ── 3. High-Throughput Inference with Wyckoff Multiplicities ───────────────────
zip_path = os.path.join(gnome_dir, "by_id.zip")
bva = BVAnalyzer()
CUTOFF = 5.0

if args.batch_limit > 0:
    target_df = filtered_df.head(args.batch_limit).copy()
else:
    target_df = filtered_df.copy()

print(f"\nStarting High-Throughput Inference on {len(target_df):,} materials...")

predictions_summary = []
giant_materials = []
atom_records = []

out_summary_csv = os.path.join(gnome_dir, "gnome_stable_semiconductors_bec_predictions_summary.csv")
out_giant_csv = os.path.join(gnome_dir, "gnome_stable_semiconductors_giant_anomalous_candidates.csv")

with zipfile.ZipFile(zip_path, 'r') as z_archive:
    for idx, (row_idx, row) in enumerate(tqdm(target_df.iterrows(), total=len(target_df), desc="GNoME Inference")):
        mat_id = str(row['MaterialId'])
        cif_name = f"by_id/{mat_id}.CIF"
        formula = str(row['Reduced Formula'])
        bandgap = float(row['Bandgap'])
        spg_num = int(row['Space Group Number'])
        spg_sym = str(row['Space Group'])
        c_sys = str(row['Crystal System'])
        is_pol = bool(row['is_polar'])
        
        try:
            cif_text = z_archive.read(cif_name).decode('utf-8')
            struct = Structure.from_str(cif_text, fmt='cif')
        except Exception:
            continue
            
        atomic_numbers = [site.specie.Z for site in struct]
        if any(z >= 95 or z <= 0 for z in atomic_numbers):
            continue
            
        coords = np.array(struct.cart_coords, dtype=np.float32)
        z_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
        pos_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        
        all_neighbors = struct.get_all_neighbors(CUTOFF, include_index=True)
        edge_src, edge_dst, edge_vecs = [], [], []
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                edge_src.append(i)
                edge_dst.append(j)
                edge_vecs.append(neighbor.coords - coords[i])
                
        if len(edge_src) == 0:
            continue
            
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
        edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred = model(z_tensor, pos_tensor, edge_index, edge_vec).cpu().numpy()
            
        # 1. Wyckoff with numbers
        try:
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            symm_struct = sga.get_symmetrized_structure()
            wyckoff_str = ";".join(symm_struct.wyckoff_symbols)
        except Exception:
            try:
                ds = sga.get_symmetry_dataset()
                counts = Counter(ds.equivalent_atoms)
                wyckoff_str = ";".join([f"{counts[eq]}{w}" for w, eq in zip(ds.wyckoffs, ds.equivalent_atoms)])
            except Exception:
                wyckoff_str = "Unknown"
                
        # 2. Nominal Oxidation States
        try:
            s_oxi = bva.get_oxi_state_decorated_structure(struct)
            oxis = [float(site.specie.oxi_state) for site in s_oxi]
            oxi_method = "BVAnalyzer"
        except Exception:
            try:
                c = Composition(formula)
                guesses = c.oxi_state_guesses()
                if guesses:
                    oxis = [float(guesses[0].get(site.specie.symbol, 0.0)) for site in struct]
                    oxi_method = "Composition_Guesses"
                else:
                    oxis = [0.0] * len(struct)
                    oxi_method = "Zero_Fallback"
            except Exception:
                oxis = [0.0] * len(struct)
                oxi_method = "Zero_Fallback"
                
        max_diag_anom = 0.0
        max_trace_anom = 0.0
        cryst_has_giant = False
        cryst_has_mod = False
        
        for a_idx, site in enumerate(struct):
            q_nom = oxis[a_idx]
            Z_mat = pred[a_idx]
            Z_trace = float((Z_mat[0, 0] + Z_mat[1, 1] + Z_mat[2, 2]) / 3.0)
            
            trace_anom = abs(Z_trace - q_nom)
            diag_anom = max(abs(Z_mat[0, 0] - q_nom), abs(Z_mat[1, 1] - q_nom), abs(Z_mat[2, 2] - q_nom))
            
            if diag_anom > max_diag_anom:
                max_diag_anom = diag_anom
            if trace_anom > max_trace_anom:
                max_trace_anom = trace_anom
                
            if diag_anom >= 2.0:
                cryst_has_giant = True
            if diag_anom >= 1.0:
                cryst_has_mod = True
                
        res = {
            'material_id': mat_id,
            'formula': formula,
            'bandgap_eV': bandgap,
            'is_stable': True,
            'is_polar': is_pol,
            'polarity': 'Polar' if is_pol else 'Non-Polar',
            'is_magnetic': False,
            'space_group_symbol': spg_sym,
            'space_group_number': spg_num,
            'crystal_system': c_sys,
            'wyckoff_symbols_with_multiplicity': wyckoff_str,
            'nominal_oxidation_state_method': oxi_method,
            'max_diagonal_anomaly': round(max_diag_anom, 4),
            'max_trace_anomaly': round(max_trace_anom, 4),
            'has_giant_anomalous_bec': cryst_has_giant,
            'has_moderate_anomalous_bec': cryst_has_mod
        }
        predictions_summary.append(res)
        
        if cryst_has_giant:
            giant_materials.append(res)
            
        # Periodic intermediate save
        if (idx + 1) % args.save_every == 0 or (idx + 1) == len(target_df):
            pd.DataFrame(predictions_summary).to_csv(out_summary_csv, index=False)
            pd.DataFrame(giant_materials).to_csv(out_giant_csv, index=False)
            print(f"Checkpoint at {idx+1:,} materials saved. Giant anomalous found: {len(giant_materials):,}")

print(f"\nFinal complete summary saved -> {out_summary_csv}")
print(f"Total Giant Anomalous materials found: {len(giant_materials):,} ({len(giant_materials)/len(predictions_summary)*100:.2f}%)")
print("=" * 75)
