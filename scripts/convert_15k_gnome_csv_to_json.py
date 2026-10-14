import os
import json
import pandas as pd
from tqdm import tqdm

gnome_csv = "/home/vibhan23/gnome/gnome_semiconductors_full_3x3_predictions_full_3x3_15k.csv"
out_json = "/home/vibhan23/gnome/gnome_15k_full_3x3_bec_predictions.json"

print(f"Loading {gnome_csv}...")
df = pd.read_csv(gnome_csv)
print(f"Loaded {len(df):,} atom rows across {df['material_id'].nunique():,} materials.")

materials_dict = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping atoms by material"):
    mat_id = str(row['material_id'])
    if mat_id not in materials_dict:
        materials_dict[mat_id] = {
            "material_id": mat_id,
            "formula": str(row['formula']),
            "bandgap_eV": float(row['bandgap_eV']),
            "space_group": str(row['space_group']),
            "crystal_system": str(row['crystal_system']),
            "num_atoms": 0,
            "max_diagonal_anomaly": 0.0,
            "max_trace_anomaly": 0.0,
            "has_giant_anomalous_bec": False,
            "has_moderate_anomalous_bec": False,
            "predicted_atoms_bec": []
        }
    
    atom_rec = {
        "atom_index": int(row['atom_index']),
        "element": str(row['element']),
        "atomic_number": int(row['atomic_number']),
        "nominal_oxidation_state": float(row['nominal_oxidation_state']) if pd.notna(row['nominal_oxidation_state']) else None,
        "bec_tensor_3x3": [
            [round(float(row['pred_Zxx']), 4), round(float(row['pred_Zxy']), 4), round(float(row['pred_Zxz']), 4)],
            [round(float(row['pred_Zyx']), 4), round(float(row['pred_Zyy']), 4), round(float(row['pred_Zyz']), 4)],
            [round(float(row['pred_Zzx']), 4), round(float(row['pred_Zzy']), 4), round(float(row['pred_Zzz']), 4)]
        ],
        "Z_trace_mean": round(float(row['pred_Z_mean']), 4),
        "trace_anomaly": round(float(row['delta_trace_anomaly']), 4) if pd.notna(row['delta_trace_anomaly']) else None,
        "max_diagonal_anomaly": round(float(row['delta_max_component']), 4) if pd.notna(row['delta_max_component']) else None,
        "off_diagonal_frobenius_norm": round(float(row['offdiag_frobenius_norm']), 4) if pd.notna(row['offdiag_frobenius_norm']) else None,
        "tensor_asymmetry": round(float(row['tensor_max_asymmetry']), 4) if pd.notna(row['tensor_max_asymmetry']) else None,
        "is_giant_anomalous": bool(row['is_giant_anomalous']),
        "is_moderate_anomalous": bool(row['is_moderate_anomalous'])
    }
    
    mat = materials_dict[mat_id]
    mat["predicted_atoms_bec"].append(atom_rec)
    mat["num_atoms"] += 1
    if atom_rec["max_diagonal_anomaly"] and atom_rec["max_diagonal_anomaly"] > mat["max_diagonal_anomaly"]:
        mat["max_diagonal_anomaly"] = atom_rec["max_diagonal_anomaly"]
    if atom_rec["trace_anomaly"] and atom_rec["trace_anomaly"] > mat["max_trace_anomaly"]:
        mat["max_trace_anomaly"] = atom_rec["trace_anomaly"]
    if atom_rec["is_giant_anomalous"]:
        mat["has_giant_anomalous_bec"] = True
    if atom_rec["is_moderate_anomalous"]:
        mat["has_moderate_anomalous_bec"] = True

output_list = list(materials_dict.values())
print(f"Saving {len(output_list):,} structured crystal entries to {out_json}...")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(output_list, f, indent=2)

print("Finished successfully!")
