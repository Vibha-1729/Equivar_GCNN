import os
import sys
import zlib
import json
import lmdb
import math
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
print("  OMAT24 (META FAIRCHEM) HIGH-THROUGHPUT BEC TENSOR INFERENCE")
print("  Rattled-Relax Relaxed Trajectories (100,544 Structures)")
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

omat_dir = "/home/vibhan23/omat24"
lmdb_path = os.path.join(omat_dir, "extracted_1M/train/rattled-relax/train.aselmdb")
out_summary = os.path.join(omat_dir, "omat24_rattled_relax_bec_predictions_summary.csv")
out_giant = os.path.join(omat_dir, "omat24_giant_anomalous_candidates.csv")

bva = BVAnalyzer()
CUTOFF = 5.0

def parse_ndarray(d):
    if isinstance(d, dict) and '__ndarray__' in d:
        shape = d['__ndarray__'][0]
        dtype = d['__ndarray__'][1]
        flat = d['__ndarray__'][2]
        return np.array(flat, dtype=dtype).reshape(shape)
    return np.array(d)

print(f"\nOpening OMat24 LMDB: {lmdb_path} ...")
env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
total_entries = env.stat()['entries']
print(f"Total entries in LMDB: {total_entries:,}")

predictions = []
giant_candidates = []

with env.begin() as txn:
    cursor = txn.cursor()
    for idx, (key, value) in enumerate(tqdm(cursor, total=total_entries, desc="OMat24 Inference")):
        if key == b'__len__':
            continue
            
        try:
            decomp = zlib.decompress(value)
            obj = json.loads(decomp.decode('utf-8'))
            cell = parse_ndarray(obj['cell'])
            pos = parse_ndarray(obj['positions'])
            nums = [int(n) for n in parse_ndarray(obj['numbers'])]
            
            if any(z >= 95 or z <= 0 for z in nums):
                continue
                
            struct = Structure(cell, nums, pos, coords_are_cartesian=True)
        except Exception:
            continue
            
        formula = struct.composition.reduced_formula
        uid = str(obj.get('unique_id', key.decode('utf-8', errors='ignore')))
        energy = float(obj.get('energy', 0.0))
        
        coords = np.array(struct.cart_coords, dtype=np.float32)
        z_tensor = torch.tensor(nums, dtype=torch.long, device=device)
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
            'material_id': uid,
            'formula': formula,
            'dft_energy_eV': energy,
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
            'has_giant_anomalous_bec': has_giant
        }
        predictions.append(rec)
        if has_giant:
            giant_candidates.append(rec)
            
        if (idx + 1) % 5000 == 0 or (idx + 1) == total_entries:
            pd.DataFrame(predictions).to_csv(out_summary, index=False)
            pd.DataFrame(giant_candidates).to_csv(out_giant, index=False)
            print(f"Checkpoint at {idx+1:,} materials. Giant anomalous: {len(giant_candidates):,}")

env.close()
print(f"\nOMat24 Inference Complete! Saved -> {out_summary}")
print("=" * 75)
