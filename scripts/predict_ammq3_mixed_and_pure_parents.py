import os
import sys
import math
import json
import zipfile
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer

print("=" * 75)
print("  AMMQ3 COMPLETE PREDICTION: MIXED COMPOUNDS + PURE PARENT COMPOUNDS")
print("  Equivariant GNN (Unified Master 3-Way Model, 48,186 Crystals)")
print("=" * 75)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ── 1. Change-of-Basis Matrix ───────────────────────────────────────────────
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

# ── 2. Locate Paths & Load Checkpoint ────────────────────────────────────────
ckpt_path = "combined_master/models/equivar_combined_500ep_baseline_best.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = "/home/vibhan23/combined_bec_dataset/models/equivar_combined_500ep_baseline_best.pt"

zip_path = "AMMQ3.zip"
if not os.path.exists(zip_path):
    zip_path = "/home/vibhan23/AMMQ3.zip"

out_dir = "ammq3_analysis"
os.makedirs(out_dir, exist_ok=True)
cif_child_dir = os.path.join(out_dir, "cif_structures", "mixed_child")
cif_parent_dir = os.path.join(out_dir, "cif_structures", "pure_parent")
os.makedirs(cif_child_dir, exist_ok=True)
os.makedirs(cif_parent_dir, exist_ok=True)

print(f"Loading Model Checkpoint: {ckpt_path} ...")
model = EquivarBECGNN().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()
print("Model loaded successfully.")

# ── 3. Read AMMQ3 JSON ───────────────────────────────────────────────────────
with zipfile.ZipFile(zip_path, 'r') as z:
    raw_json_bytes = z.read("mix_anion/standard_wyckoff_complete_weighted.json")
    compounds_data = json.loads(raw_json_bytes.decode('utf-8'))
    zip_namelist = set(z.namelist())

print(f"Loaded {len(compounds_data)} compound elements from JSON.")

CUTOFF = 5.0
bva = BVAnalyzer()

# Standard formal valence mappings for Au-based and related ternary/quaternary phases
AU_FORMAL_MAPS = {
    # Mixed Child
    "BaLaAuSe2S": {"Ba": 2.0, "La": 3.0, "Au": 1.0, "Se": -2.0, "S": -2.0},
    "CsPAuSe2S":  {"Cs": 1.0, "P": 4.0,  "Au": 1.0, "Se": -2.0, "S": -2.0},
    "KPAuSe2S":   {"K": 1.0,  "P": 4.0,  "Au": 1.0, "Se": -2.0, "S": -2.0},
    "RbPAuSe2S":  {"Rb": 1.0, "P": 4.0,  "Au": 1.0, "Se": -2.0, "S": -2.0},
    "TlPAuSe2S":  {"Tl": 1.0, "P": 4.0,  "Au": 1.0, "Se": -2.0, "S": -2.0},
    # Pure Parents
    "BaLaAuSe3":  {"Ba": 2.0, "La": 3.0, "Au": 1.0, "Se": -2.0},
    "BaLaAuS3":   {"Ba": 2.0, "La": 3.0, "Au": 1.0, "S": -2.0},
    "CsPAuSe3":   {"Cs": 1.0, "P": 4.0,  "Au": 1.0, "Se": -2.0},
    "CsPAuS3":    {"Cs": 1.0, "P": 4.0,  "Au": 1.0, "S": -2.0},
    "KPAuSe3":    {"K": 1.0,  "P": 4.0,  "Au": 1.0, "Se": -2.0},
    "KPAuS3":     {"K": 1.0,  "P": 4.0,  "Au": 1.0, "S": -2.0},
    "RbPAuSe3":   {"Rb": 1.0, "P": 4.0,  "Au": 1.0, "Se": -2.0},
    "RbPAuS3":    {"Rb": 1.0, "P": 4.0,  "Au": 1.0, "S": -2.0},
    "TlPAuSe3":   {"Tl": 1.0, "P": 4.0,  "Au": 1.0, "Se": -2.0},
    "TlPAuS3":    {"Tl": 1.0, "P": 4.0,  "Au": 1.0, "S": -2.0},
}

def predict_structure_bec(struct, formula, is_parent=False):
    """Computes nominal oxidation states and 3x3 BEC tensors for a structure."""
    coords = np.array(struct.cart_coords, dtype=np.float32)
    atomic_numbers = [site.specie.Z for site in struct]
    elements = [site.specie.symbol for site in struct]
    num_atoms = len(struct)

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

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_vec = torch.tensor(np.array(edge_vecs), dtype=torch.float32, device=device)

    with torch.no_grad():
        bec_pred = model(z_tensor, pos_tensor, edge_index, edge_vec).cpu().numpy()

    # ── Strict Oxidation State Determination ──
    oxis = None
    oxi_source = None
    try:
        s_oxi = bva.get_oxi_state_decorated_structure(struct)
        oxis = [float(site.specie.oxi_state) for site in s_oxi]
        oxi_source = "BVAnalyzer_Rigorous"
    except Exception:
        clean_formula = formula.replace(" ", "")
        if clean_formula in AU_FORMAL_MAPS:
            f_map = AU_FORMAL_MAPS[clean_formula]
            oxis = [f_map[el] for el in elements]
            oxi_source = "Formal_Electronegative"
        else:
            oxis = [None] * num_atoms
            oxi_source = "Undefined_Metallic_Actinide"

    atom_records = []
    crystal_max_diag_anomaly = None
    crystal_max_trace_anomaly = None

    for i in range(num_atoms):
        Z_mat = bec_pred[i]
        Zxx, Zxy, Zxz = float(Z_mat[0, 0]), float(Z_mat[0, 1]), float(Z_mat[0, 2])
        Zyx, Zyy, Zyz = float(Z_mat[1, 0]), float(Z_mat[1, 1]), float(Z_mat[1, 2])
        Zzx, Zzy, Zzz = float(Z_mat[2, 0]), float(Z_mat[2, 1]), float(Z_mat[2, 2])

        q_nom = oxis[i]
        trace_mean = float((Zxx + Zyy + Zzz) / 3.0)
        off_diag_frobenius = float(math.sqrt(Zxy**2 + Zxz**2 + Zyx**2 + Zyz**2 + Zzx**2 + Zzy**2))
        tensor_asymmetry = float(math.sqrt((Zxy - Zyx)**2 + (Zxz - Zzx)**2 + (Zyz - Zzy)**2) / math.sqrt(2.0))

        if q_nom is not None:
            trace_anomaly = float(abs(trace_mean - q_nom))
            diag_anomaly = float(max(abs(Zxx - q_nom), abs(Zyy - q_nom), abs(Zzz - q_nom)))
            is_giant = bool(diag_anomaly >= 2.0)
            is_moderate = bool(diag_anomaly >= 1.0)
            
            if crystal_max_diag_anomaly is None or diag_anomaly > crystal_max_diag_anomaly:
                crystal_max_diag_anomaly = diag_anomaly
            if crystal_max_trace_anomaly is None or trace_anomaly > crystal_max_trace_anomaly:
                crystal_max_trace_anomaly = trace_anomaly
        else:
            trace_anomaly = None
            diag_anomaly = None
            is_giant = None
            is_moderate = None

        atom_records.append({
            "atom_index": i,
            "element": elements[i],
            "atomic_number": atomic_numbers[i],
            "fractional_coords": [round(float(c), 6) for c in struct[i].frac_coords],
            "cartesian_coords": [round(float(c), 6) for c in struct[i].coords],
            "nominal_oxidation_state": q_nom,
            "bec_tensor_3x3": [
                [round(Zxx, 4), round(Zxy, 4), round(Zxz, 4)],
                [round(Zyx, 4), round(Zyy, 4), round(Zyz, 4)],
                [round(Zzx, 4), round(Zzy, 4), round(Zzz, 4)]
            ],
            "Z_trace_mean": round(trace_mean, 4),
            "trace_anomaly": round(trace_anomaly, 4) if trace_anomaly is not None else None,
            "max_diagonal_anomaly": round(diag_anomaly, 4) if diag_anomaly is not None else None,
            "off_diagonal_frobenius_norm": round(off_diag_frobenius, 4),
            "tensor_asymmetry": round(tensor_asymmetry, 4),
            "is_giant_anomalous": is_giant,
            "is_moderate_anomalous": is_moderate
        })

    summary_info = {
        "num_atoms": num_atoms,
        "nominal_oxidation_state_method": oxi_source,
        "crystal_max_diagonal_anomaly": round(crystal_max_diag_anomaly, 4) if crystal_max_diag_anomaly is not None else None,
        "crystal_max_trace_anomaly": round(crystal_max_trace_anomaly, 4) if crystal_max_trace_anomaly is not None else None,
        "has_giant_anomalous_bec": bool(crystal_max_diag_anomaly >= 2.0) if crystal_max_diag_anomaly is not None else None,
        "has_moderate_anomalous_bec": bool(crystal_max_diag_anomaly >= 1.0) if crystal_max_diag_anomaly is not None else None,
        "predicted_atoms_bec": atom_records
    }
    return summary_info

# Cache parent predictions so we don't recompute identical pure parents multiple times
parent_cache = {}

mixed_summaries = []
parent_summaries = []

print("\nProcessing all 94 list elements (Mixed child + Pure parents)...")
with zipfile.ZipFile(zip_path, 'r') as z:
    for idx, entry in enumerate(tqdm(compounds_data, desc="AMMQ3 Mixed + Parents")):
        oqmd_id = entry.get("OQMD_ID", f"compound_{idx}")
        formula = entry.get("formula", "Unknown")
        band_gap = float(entry.get("band_gap_eV", 0.0))
        spg_info = entry.get("space_group", {})
        spg_num = spg_info.get("number", "Unknown") if isinstance(spg_info, dict) else "Unknown"
        spg_sym = spg_info.get("symbol", "Unknown") if isinstance(spg_info, dict) else "Unknown"

        # ── 1. Predict Mixed Child Compound ──
        child_vasp_path = f"mix_anion/child_structures/{oqmd_id}.vasp"
        child_vasp_str = z.read(child_vasp_path).decode('utf-8')
        child_struct = Structure.from_str(child_vasp_str, fmt="poscar")
        
        # Save child CIF
        child_cif_path = os.path.join(cif_child_dir, f"{oqmd_id}.cif")
        child_struct.to(filename=child_cif_path, fmt="cif")

        child_bec_info = predict_structure_bec(child_struct, formula, is_parent=False)
        
        # Update top-level mixed entry
        entry["cif_relative_path"] = f"cif_structures/mixed_child/{oqmd_id}.cif"
        entry["nominal_oxidation_state_method"] = child_bec_info["nominal_oxidation_state_method"]
        entry["num_atoms_in_cell"] = child_bec_info["num_atoms"]
        entry["crystal_max_diagonal_anomaly"] = child_bec_info["crystal_max_diagonal_anomaly"]
        entry["crystal_max_trace_anomaly"] = child_bec_info["crystal_max_trace_anomaly"]
        entry["has_giant_anomalous_bec"] = child_bec_info["has_giant_anomalous_bec"]
        entry["has_moderate_anomalous_bec"] = child_bec_info["has_moderate_anomalous_bec"]
        entry["predicted_atoms_bec"] = child_bec_info["predicted_atoms_bec"]

        mixed_summaries.append({
            "compound_type": "mixed_child",
            "element_index": idx,
            "oqmd_id": oqmd_id,
            "formula": formula,
            "band_gap_eV": band_gap,
            "space_group_symbol": spg_sym,
            "space_group_number": spg_num,
            "num_atoms": child_bec_info["num_atoms"],
            "oxi_method": child_bec_info["nominal_oxidation_state_method"],
            "max_diagonal_anomaly": child_bec_info["crystal_max_diagonal_anomaly"],
            "max_trace_anomaly": child_bec_info["crystal_max_trace_anomaly"],
            "has_giant_anomalous_bec": child_bec_info["has_giant_anomalous_bec"],
            "has_moderate_anomalous_bec": child_bec_info["has_moderate_anomalous_bec"]
        })

        # ── 2. Predict Each Pure Parent Compound ──
        parents = entry.get("parent_compounds", [])
        for p_idx, p in enumerate(parents):
            p_oid = p.get("OQMD_ID")
            p_formula = p.get("formula", "Unknown")
            p_gap = float(p.get("band_gap_eV", 0.0))
            p_spg = p.get("space_group", {}).get("symbol", "Unknown") if isinstance(p.get("space_group"), dict) else "Unknown"

            if p_oid not in parent_cache:
                # Find vasp path
                candidates = [
                    f"mix_anion/parent_structures/OQMD_id_{p_oid}.vasp",
                    f"mix_anion/parent_structures/{p_oid}.vasp"
                ]
                match_p_path = None
                for c in candidates:
                    if c in zip_namelist:
                        match_p_path = c
                        break
                
                if match_p_path is None:
                    print(f"Warning: could not find parent structure for {p_oid}")
                    continue

                p_vasp_str = z.read(match_p_path).decode('utf-8')
                p_struct = Structure.from_str(p_vasp_str, fmt="poscar")
                
                # Save parent CIF
                p_cif_path = os.path.join(cif_parent_dir, f"OQMD_id_{p_oid}.cif")
                p_struct.to(filename=p_cif_path, fmt="cif")

                p_bec_info = predict_structure_bec(p_struct, p_formula, is_parent=True)
                p_bec_info["cif_relative_path"] = f"cif_structures/pure_parent/OQMD_id_{p_oid}.cif"
                parent_cache[p_oid] = p_bec_info

            # Retrieve from cache and populate parent dictionary in JSON
            cached = parent_cache[p_oid]
            p["cif_relative_path"] = cached["cif_relative_path"]
            p["nominal_oxidation_state_method"] = cached["nominal_oxidation_state_method"]
            p["num_atoms_in_cell"] = cached["num_atoms"]
            p["crystal_max_diagonal_anomaly"] = cached["crystal_max_diagonal_anomaly"]
            p["crystal_max_trace_anomaly"] = cached["crystal_max_trace_anomaly"]
            p["has_giant_anomalous_bec"] = cached["has_giant_anomalous_bec"]
            p["has_moderate_anomalous_bec"] = cached["has_moderate_anomalous_bec"]
            p["predicted_atoms_bec"] = cached["predicted_atoms_bec"]

            parent_summaries.append({
                "compound_type": "pure_parent",
                "parent_of_child_id": oqmd_id,
                "child_formula": formula,
                "oqmd_id": f"OQMD_id_{p_oid}",
                "formula": p_formula,
                "band_gap_eV": p_gap,
                "space_group_symbol": p_spg,
                "num_atoms": cached["num_atoms"],
                "oxi_method": cached["nominal_oxidation_state_method"],
                "max_diagonal_anomaly": cached["crystal_max_diagonal_anomaly"],
                "max_trace_anomaly": cached["crystal_max_trace_anomaly"],
                "has_giant_anomalous_bec": cached["has_giant_anomalous_bec"],
                "has_moderate_anomalous_bec": cached["has_moderate_anomalous_bec"]
            })

# ── 4. Save JSON and Summary CSVs ────────────────────────────────────────────
out_json_path = os.path.join(out_dir, "standard_wyckoff_complete_weighted_with_bec.json")
print(f"\n[1/3] Saving complete JSON with Mixed + Pure Parents BEC -> {out_json_path} ...")
with open(out_json_path, 'w', encoding='utf-8') as f:
    json.dump(compounds_data, f, indent=2)

# Copy to root and artifacts
import shutil
root_json = "ammq3_predictions_with_bec_and_oxidation_states.json"
shutil.copy2(out_json_path, root_json)
print(f"  -> Also copied to root: {root_json}")

out_mixed_csv = os.path.join(out_dir, "ammq3_mixed_child_compounds_summary.csv")
pd.DataFrame(mixed_summaries).to_csv(out_mixed_csv, index=False)

out_parent_csv = os.path.join(out_dir, "ammq3_pure_parent_compounds_summary.csv")
pd.DataFrame(parent_summaries).to_csv(out_parent_csv, index=False)

combined_all_csv = os.path.join(out_dir, "ammq3_all_compounds_mixed_and_pure_summary.csv")
all_df = pd.concat([pd.DataFrame(mixed_summaries), pd.DataFrame(parent_summaries)], ignore_index=True)
all_df.to_csv(combined_all_csv, index=False)
print(f"[2/3] Saved combined crystal summaries -> {combined_all_csv}")

print(f"[3/3] Unique pure parents processed: {len(parent_cache)}")
print("\n" + "=" * 75)
print("  ALL MIXED AND PURE PARENT BEC PREDICTIONS COMPLETED SUCCESSFULLY!")
print("=" * 75)
