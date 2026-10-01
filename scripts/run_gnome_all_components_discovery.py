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
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from pymatgen.core import Structure, Composition
from pymatgen.analysis.bond_valence import BVAnalyzer

parser = argparse.ArgumentParser(description="GNoME High-Throughput BEC Tensor Discovery")
parser.add_argument("--batch-size", type=int, default=15000, help="Number of crystals to process (0 for all)")
parser.add_argument("--out-suffix", type=str, default="full_tensor", help="Suffix for output filenames")
args = parser.parse_args()

print("=" * 70)
print("  GNoME FULL 3x3 BEC TENSOR DISCOVERY (DIAGONAL + OFF-DIAGONAL)")
print("  Equivar GNN Deployed on Google DeepMind GNoME Semiconductors (Eg > 0)")
print("=" * 70)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ── 1. Official Change-of-Basis Matrix ───────────────────────────────────────
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

# ── 2. Load Model Checkpoint ────────────────────────────────────────────────
ckpt_path = "/home/vibhan23/combined_bec_dataset/models/equivar_combined_500ep_baseline_best.pt"
print(f"Loading best Combined Equivar model checkpoint: {ckpt_path} ...")
model = EquivarBECGNN().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print("Model loaded successfully.")

# ── 3. Filter GNoME for Semiconductors (Eg > 0 eV) ──────────────────────────
summary_csv = "/home/vibhan23/gnome/stable_materials_summary.csv"
print(f"Reading GNoME summary CSV: {summary_csv} ...")
df = pd.read_csv(summary_csv)

is_semiconductor = (df['Bandgap'] > 0.0) & (~np.isinf(df['Bandgap'])) & (df['Bandgap'].notna())
semiconductors_df = df[is_semiconductor].copy()
total_avail = len(semiconductors_df)
print(f"Total GNoME Semiconductors (finite Eg > 0 eV): {total_avail:,}")

if args.batch_size > 0:
    sample_df = semiconductors_df.head(args.batch_size).copy()
    print(f"Target Batch Size: {len(sample_df):,} semiconductors")
else:
    sample_df = semiconductors_df.copy()
    print(f"Processing FULL Dataset: {len(sample_df):,} semiconductors")

CUTOFF = 5.0
zip_path = "/home/vibhan23/gnome/by_id.zip"
bva = BVAnalyzer()

all_atoms = []
giant_anomalous_materials = []
moderate_anomalous_materials = []

with zipfile.ZipFile(zip_path, 'r') as z_archive:
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Discovering Full-Tensor BEC Semiconductors"):
        mat_id = str(row['MaterialId'])
        cif_name = f"by_id/{mat_id}.CIF"
        formula = str(row['Reduced Formula'])
        bandgap = float(row['Bandgap'])
        spg = str(row['Space Group'])
        c_sys = str(row['Crystal System'])
        
        try:
            cif_text = z_archive.read(cif_name).decode('utf-8')
            struct = Structure.from_str(cif_text, fmt='cif')
        except Exception:
            continue
            
        coords = np.array(struct.cart_coords, dtype=np.float32)
        atomic_numbers = [site.specie.Z for site in struct]
        if any(z >= 95 or z <= 0 for z in atomic_numbers):
            continue
            
        z_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
        pos_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        
        all_neighbors = struct.get_all_neighbors(CUTOFF, include_index=True)
        edge_src, edge_dst, edge_vecs = [], [], []
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                vec = neighbor.coords - coords[i]
                edge_src.append(i)
                edge_dst.append(j)
                edge_vecs.append(vec)
                
        if len(edge_src) == 0:
            continue
            
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
        edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pred = model(z_tensor, pos_tensor, edge_index, edge_vec).cpu().numpy()
            
        # Nominal oxidation states via BVAnalyzer
        try:
            s_oxi = bva.get_oxi_state_decorated_structure(struct)
            oxis = [float(site.specie.oxi_state) for site in s_oxi]
        except Exception:
            try:
                c = Composition(formula)
                guesses = c.oxi_state_guesses()
                if guesses:
                    oxis = [float(guesses[0].get(site.specie.symbol, 0.0)) for site in struct]
                else:
                    oxis = [0.0] * len(struct)
            except Exception:
                oxis = [0.0] * len(struct)
                
        has_giant = False
        has_mod = False
        max_trace_anom = 0.0
        max_cryst_offdiag = 0.0
        
        for a_idx, site in enumerate(struct):
            el = site.specie.symbol
            z_num = site.specie.Z
            q_nom = oxis[a_idx] if a_idx < len(oxis) else 0.0
            
            p_tensor = pred[a_idx]
            # Diagonal components
            pxx = float(p_tensor[0, 0])
            pyy = float(p_tensor[1, 1])
            pzz = float(p_tensor[2, 2])
            p_mean = (pxx + pyy + pzz) / 3.0
            
            # Off-diagonal components
            pxy = float(p_tensor[0, 1])
            pxz = float(p_tensor[0, 2])
            pyx = float(p_tensor[1, 0])
            pyz = float(p_tensor[1, 2])
            pzx = float(p_tensor[2, 0])
            pzy = float(p_tensor[2, 1])
            
            # Off-diagonal Frobenius magnitude
            offdiag_norm = math.sqrt(pxy**2 + pxz**2 + pyx**2 + pyz**2 + pzx**2 + pzy**2)
            if offdiag_norm > max_cryst_offdiag:
                max_cryst_offdiag = offdiag_norm
                
            # Tensor asymmetry |Zij - Zji|
            asym_xy = abs(pxy - pyx)
            asym_xz = abs(pxz - pzx)
            asym_yz = abs(pyz - pzy)
            max_asym = max(asym_xy, asym_xz, asym_yz)
            
            delta_trace = abs(p_mean - q_nom)
            delta_max = max(abs(pxx - q_nom), abs(pyy - q_nom), abs(pzz - q_nom))
            
            if delta_trace > max_trace_anom:
                max_trace_anom = delta_trace
                
            is_giant = (delta_trace >= 2.0) or (delta_max >= 2.0)
            is_mod = (delta_trace >= 1.0) or (delta_max >= 1.0)
            
            if is_giant:
                has_giant = True
            if is_mod:
                has_mod = True
                
            all_atoms.append({
                'material_id': mat_id,
                'formula': formula,
                'bandgap_eV': round(bandgap, 4),
                'space_group': spg,
                'crystal_system': c_sys,
                'atom_index': a_idx,
                'element': el,
                'atomic_number': z_num,
                'nominal_oxidation_state': round(q_nom, 2),
                # 3x3 Tensor Components
                'pred_Zxx': round(pxx, 5),
                'pred_Zxy': round(pxy, 5),
                'pred_Zxz': round(pxz, 5),
                'pred_Zyx': round(pyx, 5),
                'pred_Zyy': round(pyy, 5),
                'pred_Zyz': round(pyz, 5),
                'pred_Zzx': round(pzx, 5),
                'pred_Zzy': round(pzy, 5),
                'pred_Zzz': round(pzz, 5),
                # Aggregates & Physical Metrics
                'pred_Z_mean': round(p_mean, 5),
                'offdiag_frobenius_norm': round(offdiag_norm, 5),
                'tensor_max_asymmetry': round(max_asym, 5),
                'delta_trace_anomaly': round(delta_trace, 5),
                'delta_max_component': round(delta_max, 5),
                'is_giant_anomalous': bool(is_giant),
                'is_moderate_anomalous': bool(is_mod)
            })
            
        cryst_entry = {
            'material_id': mat_id,
            'formula': formula,
            'bandgap_eV': round(bandgap, 4),
            'space_group': spg,
            'crystal_system': c_sys,
            'num_atoms': len(struct),
            'max_trace_anomaly_e': round(max_trace_anom, 4),
            'max_offdiag_norm': round(max_cryst_offdiag, 4)
        }
        if has_giant:
            giant_anomalous_materials.append(cryst_entry)
        if has_mod:
            moderate_anomalous_materials.append(cryst_entry)

out_dir = "/home/vibhan23/gnome"
suffix = args.out_suffix
pred_csv_out = os.path.join(out_dir, f"gnome_semiconductors_full_3x3_predictions_{suffix}.csv")
print(f"\nSaving Full 3x3 GNoME predictions ({len(all_atoms):,} atoms) to: {pred_csv_out} ...")
pd.DataFrame(all_atoms).to_csv(pred_csv_out, index=False)

giant_csv_out = os.path.join(out_dir, f"gnome_semiconductors_giant_anomalies_{suffix}.csv")
print(f"Saving Giant Anomalous Materials ({len(giant_anomalous_materials):,} crystals) to: {giant_csv_out} ...")
pd.DataFrame(giant_anomalous_materials).to_csv(giant_csv_out, index=False)

mod_csv_out = os.path.join(out_dir, f"gnome_semiconductors_moderate_anomalies_{suffix}.csv")
print(f"Saving Moderate Anomalous Materials ({len(moderate_anomalous_materials):,} crystals) to: {mod_csv_out} ...")
pd.DataFrame(moderate_anomalous_materials).to_csv(mod_csv_out, index=False)

summary_csv_out = os.path.join(out_dir, f"gnome_semiconductors_discovery_summary_{suffix}.csv")
n_proc = len(sample_df)
pd.DataFrame([
    {'Metric': 'Semiconductors Processed', 'Value': n_proc},
    {'Metric': 'Total Atoms Inferred', 'Value': len(all_atoms)},
    {'Metric': 'Crystals with Giant Anomalous BEC (|Δ|>=2e)', 'Value': len(giant_anomalous_materials)},
    {'Metric': 'Giant Anomalous Crystals Percentage', 'Value': f"{len(giant_anomalous_materials)/n_proc*100:.2f}%"},
    {'Metric': 'Crystals with Moderate Anomalous BEC (|Δ|>=1e)', 'Value': len(moderate_anomalous_materials)},
    {'Metric': 'Moderate Anomalous Crystals Percentage', 'Value': f"{len(moderate_anomalous_materials)/n_proc*100:.2f}%"}
]).to_csv(summary_csv_out, index=False)

print("\n" + "=" * 70)
print("  GNoME FULL 3x3 BEC TENSOR DISCOVERY COMPLETE")
print("=" * 70)
print(f"Semiconductors Analyzed            : {n_proc:,}")
print(f"Total Atoms Inferred               : {len(all_atoms):,}")
print(f"Giant Anomalous BEC Crystals (|Δ|>=2e): {len(giant_anomalous_materials):,} ({len(giant_anomalous_materials)/n_proc*100:.2f}%)")
print(f"Moderate Anomalous BEC Crystals (|Δ|>=1e): {len(moderate_anomalous_materials):,} ({len(moderate_anomalous_materials)/n_proc*100:.2f}%)")
print("======================================================================")
