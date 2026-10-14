import os
import sys
import glob
import math
import json
import bz2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

print("=" * 75)
print("  ALEXANDRIA HIGH-THROUGHPUT STABLE SEMICONDUCTOR INFERENCE")
print("  Filtering: e_above_hull <= 0.05 eV & Eg > 0 eV & |μ| < 1 μB/atom")
print("=" * 75)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

alexandria_dir = "/home/vibhan23/alexandria"
out_summary = os.path.join(alexandria_dir, "alexandria_stable_semiconductors_summary.csv")
out_giant = os.path.join(alexandria_dir, "alexandria_giant_anomalous_candidates.csv")

bva = BVAnalyzer()
CUTOFF = 5.0

bz2_files = sorted(glob.glob(os.path.join(alexandria_dir, "alexandria_*.json.bz2")))
print(f"Found {len(bz2_files)} Alexandria chunk files.")

predictions = []
giant_candidates = []

for f_idx, bz2_path in enumerate(bz2_files):
    chunk_name = os.path.basename(bz2_path)
    print(f"\n[{f_idx+1}/{len(bz2_files)}] Processing {chunk_name} ...")
    
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
        entries = data.get("entries", [])
        
    for entry in tqdm(entries, desc=f"Evaluating {chunk_name}"):
        entry_data = entry.get("data", {})
        e_hull = float(entry_data.get("e_above_hull", 999.0))
        gap = float(entry_data.get("band_gap_ind", 0.0))
        total_mag = float(entry_data.get("total_mag", 0.0))
        
        # Stability filter (e_above_hull <= 0.05 eV) & Semiconductor filter (gap > 0.0)
        if e_hull > 0.05 or gap <= 0.0 or np.isinf(gap):
            continue
            
        mat_id = str(entry_data.get("mat_id", entry.get("entry_id", "Unknown")))
        formula = str(entry_data.get("formula", ""))
        
        struct_dict = entry.get("structure")
        if not struct_dict:
            continue
            
        try:
            struct = Structure.from_dict(struct_dict)
        except Exception:
            continue
            
        nsites = len(struct)
        # Magnetic filter: |mu| per atom < 1.0 uB/atom -> Not magnetic
        is_mag = (abs(total_mag) / max(1, nsites)) >= 1.0
        
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
            
        # Symmetry & Wyckoff
        try:
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            spg_num = sga.get_space_group_number()
            spg_sym = sga.get_space_group_symbol()
            c_sys = sga.get_crystal_system().capitalize()
            symm_struct = sga.get_symmetrized_structure()
            wyckoff_str = ";".join(symm_struct.wyckoff_symbols)
        except Exception:
            spg_num = 1
            spg_sym = "P1"
            c_sys = "Triclinic"
            wyckoff_str = "Unknown"
            
        is_pol = spg_num in POLAR_SPACE_GROUPS
        
        # Nominal oxidation states
        try:
            s_oxi = bva.get_oxi_state_decorated_structure(struct)
            oxis = [float(site.specie.oxi_state) for site in s_oxi]
            oxi_method = "BVAnalyzer"
        except Exception:
            oxis = [0.0] * len(struct)
            oxi_method = "Zero_Fallback"
            
        max_diag_anom = 0.0
        max_trace_anom = 0.0
        has_giant = False
        
        for a_idx in range(len(struct)):
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
                has_giant = True
                
        rec = {
            'material_id': mat_id,
            'formula': formula,
            'bandgap_eV': gap,
            'e_above_hull': e_hull,
            'is_stable': True,
            'is_polar': is_pol,
            'polarity': 'Polar' if is_pol else 'Non-Polar',
            'is_magnetic': is_mag,
            'total_magnetization': total_mag,
            'space_group_symbol': spg_sym,
            'space_group_number': spg_num,
            'crystal_system': c_sys,
            'wyckoff_symbols_with_multiplicity': wyckoff_str,
            'nominal_oxidation_state_method': oxi_method,
            'max_diagonal_anomaly': round(max_diag_anom, 4),
            'max_trace_anomaly': round(max_trace_anom, 4),
            'has_giant_anomalous_bec': has_giant
        }
        predictions.append(rec)
        if has_giant:
            giant_candidates.append(rec)
            
    # Save checkpoint after each chunk
    pd.DataFrame(predictions).to_csv(out_summary, index=False)
    pd.DataFrame(giant_candidates).to_csv(out_giant, index=False)
    print(f"Saved checkpoint after chunk {chunk_name}. Total evaluated: {len(predictions):,}, Giant anomalous: {len(giant_candidates):,}")

print(f"\nAlexandria Inference Complete! Results saved to {out_summary}")
print("=" * 75)
