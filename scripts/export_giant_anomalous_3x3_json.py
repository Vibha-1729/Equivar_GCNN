import os
import sys
import math
import json
import zlib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from pymatgen.core import Structure, Composition
from pymatgen.analysis.bond_valence import BVAnalyzer

# ── 1. Equivariant GNN Architecture ──────────────────────────────────────────
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
        self.out_tp = FullyConnectedTensorProduct(self.irreps_node, self.irreps_sh, self.irreps_out, shared_weights=True)
        self.register_buffer("cob", COB_MATRIX)

    def forward(self, z, pos, edge_index, edge_vec, batch):
        h = self.embedding(z)
        num_nodes = z.size(0)
        r = torch.norm(edge_vec, dim=-1)
        radial_emb = self.radial_proj(r)
        radial_weights = self.radial_mlp(radial_emb)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
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

def parse_ndarray(d):
    if isinstance(d, dict) and '__ndarray__' in d:
        shape, dtype, flat = d['__ndarray__'][0], d['__ndarray__'][1], d['__ndarray__'][2]
        return np.array(flat, dtype=dtype).reshape(shape)
    return np.array(d)

def compute_bec_for_structure(model, struct, device, cutoff=5.0):
    nums = [site.specie.number for site in struct]
    coords = np.array(struct.cart_coords, dtype=np.float32)
    z_tensor = torch.tensor(nums, dtype=torch.long, device=device)
    pos_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    all_neighbors = struct.get_all_neighbors(cutoff, include_index=True)
    edge_src, edge_dst, edge_vecs = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            edge_src.append(i)
            edge_dst.append(neighbor.index)
            edge_vecs.append(neighbor.coords - coords[i])
    if len(edge_src) == 0:
        return None
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32, device=device)
    batch = torch.zeros(len(struct), dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(z_tensor, pos_tensor, edge_index, edge_vec, batch).cpu().numpy()
    return pred

def build_atom_records(struct, bec_pred, bva):
    try:
        s_oxi = bva.get_oxi_state_decorated_structure(struct)
        oxis = [float(site.specie.oxi_state) for site in s_oxi]
        oxi_method = "BVAnalyzer"
    except Exception:
        try:
            c = Composition(struct.composition.reduced_formula)
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
            
    atom_records = []
    max_diag_anom = 0.0
    max_trace_anom = 0.0
    has_giant = False
    has_mod = False
    
    for i, site in enumerate(struct):
        Z_mat = bec_pred[i]
        Zxx, Zxy, Zxz = float(Z_mat[0, 0]), float(Z_mat[0, 1]), float(Z_mat[0, 2])
        Zyx, Zyy, Zyz = float(Z_mat[1, 0]), float(Z_mat[1, 1]), float(Z_mat[1, 2])
        Zzx, Zzy, Zzz = float(Z_mat[2, 0]), float(Z_mat[2, 1]), float(Z_mat[2, 2])
        q_nom = oxis[i]
        trace_mean = float((Zxx + Zyy + Zzz) / 3.0)
        trace_anom = abs(trace_mean - q_nom)
        diag_anom = max(abs(Zxx - q_nom), abs(Zyy - q_nom), abs(Zzz - q_nom))
        off_diag_frobenius = math.sqrt(Zxy**2 + Zxz**2 + Zyx**2 + Zyz**2 + Zzx**2 + Zzy**2)
        tensor_asym = math.sqrt((Zxy - Zyx)**2 + (Zxz - Zzx)**2 + (Zyz - Zzy)**2) / math.sqrt(2.0)
        
        if diag_anom > max_diag_anom: max_diag_anom = diag_anom
        if trace_anom > max_trace_anom: max_trace_anom = trace_anom
        if diag_anom >= 2.0: has_giant = True
        if diag_anom >= 1.0: has_mod = True
        
        atom_records.append({
            "atom_index": i,
            "element": site.specie.symbol,
            "atomic_number": site.specie.number,
            "fractional_coords": [round(float(c), 6) for c in site.frac_coords],
            "cartesian_coords": [round(float(c), 6) for c in site.coords],
            "nominal_oxidation_state": q_nom,
            "bec_tensor_3x3": [
                [round(Zxx, 4), round(Zxy, 4), round(Zxz, 4)],
                [round(Zyx, 4), round(Zyy, 4), round(Zyz, 4)],
                [round(Zzx, 4), round(Zzy, 4), round(Zzz, 4)]
            ],
            "Z_trace_mean": round(trace_mean, 4),
            "trace_anomaly": round(trace_anom, 4),
            "max_diagonal_anomaly": round(diag_anom, 4),
            "off_diagonal_frobenius_norm": round(off_diag_frobenius, 4),
            "tensor_asymmetry": round(tensor_asym, 4),
            "is_giant_anomalous": bool(diag_anom >= 2.0),
            "is_moderate_anomalous": bool(diag_anom >= 1.0)
        })
        
    return atom_records, oxi_method, max_diag_anom, max_trace_anom, has_giant, has_mod

# ── Main Function ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['omat24', 'alexandria', 'gnome'])
    parser.add_argument('--limit', type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ckpt_path = "/home/vibhan23/combined_bec_dataset/models/equivar_combined_500ep_baseline_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model = EquivarBECGNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    bva = BVAnalyzer()

    if args.dataset == 'omat24':
        import lmdb
        cand_csv = "/home/vibhan23/omat24/omat24_giant_anomalous_candidates.csv"
        out_json = "/home/vibhan23/omat24/omat24_giant_anomalous_3x3_bec.json"
        lmdb_path = "/home/vibhan23/omat24/extracted_1M/train/rattled-relax/train.aselmdb"

        df_cand = pd.read_csv(cand_csv)
        target_ids = set(df_cand['material_id'])
        meta_dict = df_cand.set_index('material_id').to_dict('index')
        print(f"OMat24 Giant Candidates to export: {len(target_ids):,}")

        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        total_entries = env.stat()['entries']
        
        results = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in tqdm(cursor, total=total_entries, desc="Exporting OMat24 Giant BEC"):
                if key == b'__len__': continue
                try:
                    decomp = zlib.decompress(value)
                    obj = json.loads(decomp.decode('utf-8'))
                    uid = str(obj.get('unique_id', key.decode('utf-8', errors='ignore')))
                    if uid not in target_ids:
                        continue
                        
                    cell = parse_ndarray(obj['cell'])
                    pos = parse_ndarray(obj['positions'])
                    nums = [int(n) for n in parse_ndarray(obj['numbers'])]
                    struct = Structure(cell, nums, pos, coords_are_cartesian=True)
                except Exception:
                    continue

                bec_pred = compute_bec_for_structure(model, struct, device)
                if bec_pred is None: continue

                meta = meta_dict.get(uid, {})
                atoms_rec, oxi_m, max_d, max_t, h_g, h_m = build_atom_records(struct, bec_pred, bva)

                results.append({
                    "material_id": uid,
                    "formula": struct.composition.reduced_formula,
                    "dft_energy_eV": float(obj.get('energy', 0.0)),
                    "is_polar": meta.get('is_polar', False),
                    "polarity": meta.get('polarity', 'Non-Polar'),
                    "is_magnetic": meta.get('is_magnetic', False),
                    "space_group_symbol": meta.get('space_group_symbol', 'P1'),
                    "space_group_number": meta.get('space_group_number', 1),
                    "crystal_system": meta.get('crystal_system', 'Unknown'),
                    "wyckoff_symbols_with_multiplicity": meta.get('wyckoff_symbols_with_multiplicity', ''),
                    "nominal_oxidation_state_method": oxi_m,
                    "crystal_max_diagonal_anomaly": round(max_d, 4),
                    "crystal_max_trace_anomaly": round(max_t, 4),
                    "has_giant_anomalous_bec": h_g,
                    "has_moderate_anomalous_bec": h_m,
                    "num_atoms": len(struct),
                    "predicted_atoms_bec": atoms_rec
                })

                if args.limit and len(results) >= args.limit:
                    break

        print(f"Saving {len(results):,} OMat24 entries to {out_json}...")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("OMat24 export complete!")

    elif args.dataset == 'alexandria':
        import bz2
        cand_csv = "/home/vibhan23/alexandria/alexandria_giant_anomalous_candidates.csv"
        out_json = "/home/vibhan23/alexandria/alexandria_giant_anomalous_3x3_bec.json"
        alex_dir = "/home/vibhan23/alexandria"

        df_cand = pd.read_csv(cand_csv)
        target_ids = set(df_cand['material_id'])
        meta_dict = df_cand.set_index('material_id').to_dict('index')
        print(f"Alexandria Giant Candidates to export: {len(target_ids):,}")

        bz2_files = sorted([os.path.join(alex_dir, f) for f in os.listdir(alex_dir) if f.startswith('alexandria_') and f.endswith('.json.bz2')])
        results = []

        for bz2_path in tqdm(bz2_files, desc="Scanning Alexandria Chunks"):
            try:
                with bz2.open(bz2_path, 'rt', encoding='utf-8') as f:
                    chunk_data = json.load(f)
            except Exception:
                continue

            entries = chunk_data.get('entries', chunk_data) if isinstance(chunk_data, dict) else chunk_data
            for entry in entries:
                entry_data = entry.get("data", {}) if isinstance(entry, dict) else {}
                mat_id = str(entry_data.get("mat_id", entry.get("entry_id", entry.get("mat_id", ""))))
                if mat_id not in target_ids:
                    continue

                struct_dict = entry.get('structure', entry.get('structure_relaxed', None))
                if not struct_dict: continue
                try:
                    struct = Structure.from_dict(struct_dict)
                except Exception:
                    continue

                bec_pred = compute_bec_for_structure(model, struct, device)
                if bec_pred is None: continue

                meta = meta_dict.get(mat_id, {})
                atoms_rec, oxi_m, max_d, max_t, h_g, h_m = build_atom_records(struct, bec_pred, bva)

                results.append({
                    "material_id": mat_id,
                    "formula": struct.composition.reduced_formula,
                    "bandgap_eV": float(entry.get('band_gap', entry.get('bandgap_eV', 0.0))),
                    "e_above_hull": float(entry.get('e_above_hull', 0.0)),
                    "is_stable": True,
                    "is_polar": meta.get('is_polar', False),
                    "polarity": meta.get('polarity', 'Non-Polar'),
                    "is_magnetic": meta.get('is_magnetic', False),
                    "space_group_symbol": meta.get('space_group_symbol', 'P1'),
                    "space_group_number": meta.get('space_group_number', 1),
                    "crystal_system": meta.get('crystal_system', 'Unknown'),
                    "wyckoff_symbols_with_multiplicity": meta.get('wyckoff_symbols_with_multiplicity', ''),
                    "nominal_oxidation_state_method": oxi_m,
                    "crystal_max_diagonal_anomaly": round(max_d, 4),
                    "crystal_max_trace_anomaly": round(max_t, 4),
                    "has_giant_anomalous_bec": h_g,
                    "has_moderate_anomalous_bec": h_m,
                    "num_atoms": len(struct),
                    "predicted_atoms_bec": atoms_rec
                })

                if args.limit and len(results) >= args.limit:
                    break
            if args.limit and len(results) >= args.limit:
                break

        print(f"Saving {len(results):,} Alexandria entries to {out_json}...")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Alexandria export complete!")

    elif args.dataset == 'gnome':
        import zipfile
        cand_csv = "/home/vibhan23/gnome/gnome_stable_semiconductors_giant_anomalous_candidates.csv"
        out_json = "/home/vibhan23/gnome/gnome_giant_anomalous_3x3_bec.json"
        zip_path = "/home/vibhan23/gnome/by_id.zip"

        df_cand = pd.read_csv(cand_csv)
        if args.limit:
            df_cand = df_cand.head(args.limit)
        print(f"GNoME Giant Candidates to export: {len(df_cand):,}")

        meta_dict = df_cand.set_index('material_id').to_dict('index')
        results = []

        with zipfile.ZipFile(zip_path, 'r') as z:
            for mat_id, meta in tqdm(meta_dict.items(), desc="Exporting GNoME Giant BEC"):
                cif_name = f"by_id/{mat_id}.CIF"
                try:
                    cif_str = z.read(cif_name).decode('utf-8')
                    struct = Structure.from_str(cif_str, fmt="cif")
                except Exception:
                    continue

                bec_pred = compute_bec_for_structure(model, struct, device)
                if bec_pred is None: continue

                atoms_rec, oxi_m, max_d, max_t, h_g, h_m = build_atom_records(struct, bec_pred, bva)

                results.append({
                    "material_id": mat_id,
                    "formula": meta.get('formula', struct.composition.reduced_formula),
                    "bandgap_eV": float(meta.get('bandgap_eV', 0.0)),
                    "is_stable": True,
                    "is_polar": meta.get('is_polar', False),
                    "polarity": meta.get('polarity', 'Non-Polar'),
                    "space_group_symbol": meta.get('space_group_symbol', 'P1'),
                    "space_group_number": meta.get('space_group_number', 1),
                    "crystal_system": meta.get('crystal_system', 'Unknown'),
                    "wyckoff_symbols_with_multiplicity": meta.get('wyckoff_symbols_with_multiplicity', ''),
                    "nominal_oxidation_state_method": oxi_m,
                    "crystal_max_diagonal_anomaly": round(max_d, 4),
                    "crystal_max_trace_anomaly": round(max_t, 4),
                    "has_giant_anomalous_bec": h_g,
                    "has_moderate_anomalous_bec": h_m,
                    "num_atoms": len(struct),
                    "predicted_atoms_bec": atoms_rec
                })

        print(f"Saving {len(results):,} GNoME entries to {out_json}...")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("GNoME export complete!")

if __name__ == '__main__':
    main()
