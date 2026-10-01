import os
import sys
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.analysis.bond_valence import BVAnalyzer

print("=" * 65)
print("  PARALLEL NOMINAL OXIDATION STATES & ANOMALOUS BEC SCREENING")
print("  Materials Project + Mendeley Oxides + JARVIS-DFT (48,186 crystals)")
print("=" * 65)

base_dir = "/home/vibhan23/combined_bec_dataset"
json_positions_path = os.path.join(base_dir, "combined_crystal_atomic_lattice_positions.json")
pred_csv_path = os.path.join(base_dir, "combined_bec_predictions_baseline_all_atoms.csv")

print(f"Loading predictions CSV: {pred_csv_path} ...")
pred_df = pd.read_csv(pred_csv_path)
print(f"Loaded {len(pred_df):,} atomic predictions.")

# Build fast dictionary for predictions
pred_lookup = {}
for row in pred_df.itertuples():
    key = (row.material_id, int(row.atom_index))
    pred_lookup[key] = (
        float(row.pred_Zxx),
        float(row.pred_Zyy),
        float(row.pred_Zzz),
        float(row.atom_trace_pred)
    )

print(f"Loading 3D crystal structures: {json_positions_path} ...")
with open(json_positions_path, 'r') as f:
    crystals = json.load(f)
print(f"Loaded {len(crystals):,} crystal structures.")

def process_single_crystal(c):
    mat_id = c["material_id"]
    source_ds = c.get("dataset", "mp" if mat_id.startswith("mp-") else ("jarvis" if mat_id.startswith("JVASP-") else "mendeley"))
    formula = c.get("formula", "")
    unit_formula = c.get("unit_cell_formula", formula)
    split = c.get("split_type", "train").lower()
    natoms = c.get("num_atoms", len(c.get("atoms", [])))
    
    lp = c.get("lattice_parameters", {})
    lattice = Lattice([lp["matrix_a"], lp["matrix_b"], lp["matrix_c"]])
    atoms = c.get("atoms", [])
    species = [a["element"] for a in atoms]
    coords = [a["fractional_coords"] for a in atoms]
    s = Structure(lattice, species, coords)
    
    bva = BVAnalyzer()
    oxis = None
    try:
        s_oxi = bva.get_oxi_state_decorated_structure(s)
        oxis = [float(site.specie.oxi_state) for site in s_oxi]
    except Exception:
        try:
            comp = Composition(formula)
            guesses = comp.oxi_state_guesses()
            if guesses and len(guesses) > 0:
                best_guess = guesses[0]
                oxis = [float(best_guess.get(el, 0.0)) for el in species]
            else:
                oxis = [0.0] * len(species)
        except Exception:
            oxis = [0.0] * len(species)
            
    atom_records = []
    has_giant = False
    has_mod = False
    
    for idx, a in enumerate(atoms):
        el = a["element"]
        z_num = a["atomic_number"]
        q_nom = oxis[idx] if idx < len(oxis) else 0.0
        
        bec_target = a.get("born_charge_tensor_3x3", [[0,0,0],[0,0,0],[0,0,0]])
        tgt_xx = float(bec_target[0][0])
        tgt_yy = float(bec_target[1][1])
        tgt_zz = float(bec_target[2][2])
        tgt_mean = (tgt_xx + tgt_yy + tgt_zz) / 3.0
        
        pred_tuple = pred_lookup.get((mat_id, idx), (tgt_xx, tgt_yy, tgt_zz, tgt_mean))
        pred_xx, pred_yy, pred_zz, pred_mean = pred_tuple
        
        delta_trace = abs(tgt_mean - q_nom)
        delta_xx = abs(tgt_xx - q_nom)
        delta_yy = abs(tgt_yy - q_nom)
        delta_zz = abs(tgt_zz - q_nom)
        delta_max = max(delta_xx, delta_yy, delta_zz)
        
        is_giant = (delta_trace >= 2.0)
        is_mod = (delta_trace >= 1.0)
        if is_giant:
            has_giant = True
        if is_mod:
            has_mod = True
            
        atom_records.append({
            'material_id': mat_id,
            'source_dataset': source_ds,
            'formula': formula,
            'unit_cell_formula': unit_formula,
            'split_type': split,
            'num_atoms': natoms,
            'atom_index': idx,
            'element': el,
            'atomic_number': z_num,
            'nominal_oxidation_state': round(q_nom, 2),
            'target_Zxx': round(tgt_xx, 4),
            'target_Zyy': round(tgt_yy, 4),
            'target_Zzz': round(tgt_zz, 4),
            'target_Z_mean': round(tgt_mean, 4),
            'pred_Zxx': round(pred_xx, 4),
            'pred_Zyy': round(pred_yy, 4),
            'pred_Zzz': round(pred_zz, 4),
            'pred_Z_mean': round(pred_mean, 4),
            'delta_trace_anomaly': round(delta_trace, 4),
            'delta_max_component': round(delta_max, 4),
            'is_giant_anomalous': bool(is_giant),
            'is_moderate_anomalous': bool(is_mod)
        })
        
    crystal_summary = {
        'material_id': mat_id,
        'source_dataset': source_ds,
        'formula': formula,
        'unit_cell_formula': unit_formula,
        'split_type': split,
        'num_atoms': natoms,
        'has_giant_anomalous_bec': has_giant,
        'has_moderate_anomalous_bec': has_mod
    }
    
    return atom_records, crystal_summary

N_WORKERS = min(12, cpu_count())
print(f"Launching parallel processing across {N_WORKERS} CPU worker processes...")

all_atom_rows = []
all_crystal_summaries = []

with Pool(N_WORKERS) as pool:
    results = list(tqdm(pool.imap(process_single_crystal, crystals, chunksize=50), total=len(crystals), desc="Processing in parallel"))
    
for atom_recs, cryst_sum in results:
    all_atom_rows.extend(atom_recs)
    all_crystal_summaries.append(cryst_sum)

out_csv = os.path.join(base_dir, "anomalous_bec_with_nominal_oxidation_states.csv")
print(f"\nExporting per-atom CSV ({len(all_atom_rows):,} rows) to: {out_csv} ...")
atom_df = pd.DataFrame(all_atom_rows)
atom_df.to_csv(out_csv, index=False)

cryst_csv = os.path.join(base_dir, "anomalous_bec_crystal_summaries.csv")
print(f"Exporting crystal summary CSV ({len(all_crystal_summaries):,} crystals) to: {cryst_csv} ...")
cryst_df = pd.DataFrame(all_crystal_summaries)
cryst_df.to_csv(cryst_csv, index=False)

# Summary statistics
giant_atoms = atom_df[atom_df['is_giant_anomalous'] == True]
mod_atoms = atom_df[atom_df['is_moderate_anomalous'] == True]
giant_crysts = cryst_df[cryst_df['has_giant_anomalous_bec'] == True]
mod_crysts = cryst_df[cryst_df['has_moderate_anomalous_bec'] == True]

print("\n" + "=" * 60)
print("  ANOMALOUS BEC SCREENING FINAL SUMMARY")
print("=" * 60)
print(f"Total Crystals Analyzed             : {len(cryst_df):,}")
print(f"Crystals with GIANT Anomaly (>=2.0e): {len(giant_crysts):,} ({len(giant_crysts)/len(cryst_df)*100:.2f}%)")
print(f"Crystals with Moderate Anom (>=1.0e): {len(mod_crysts):,} ({len(mod_crysts)/len(cryst_df)*100:.2f}%)")
print(f"Total Atoms Analyzed                : {len(atom_df):,}")
print(f"Atoms with GIANT Anomaly (>=2.0e)   : {len(giant_atoms):,} ({len(giant_atoms)/len(atom_df)*100:.2f}%)")
print(f"Atoms with Moderate Anom (>=1.0e)   : {len(mod_atoms):,} ({len(mod_atoms)/len(atom_df)*100:.2f}%)")

print("\nTop 10 Elements Exhibiting Giant Anomalous BECs:")
top_giant = giant_atoms['element'].value_counts().head(10)
for el, cnt in top_giant.items():
    print(f"  {el:4s}: {cnt:,} atoms")

summary_csv = os.path.join(base_dir, "anomalous_bec_summary_statistics.csv")
pd.DataFrame([
    {'Metric': 'Total Crystals', 'Value': len(cryst_df)},
    {'Metric': 'Crystals with Giant Anomalous BEC (|Δ|>=2e)', 'Value': len(giant_crysts)},
    {'Metric': 'Giant Anomalous Crystals Percentage', 'Value': f"{len(giant_crysts)/len(cryst_df)*100:.2f}%"},
    {'Metric': 'Crystals with Moderate Anomalous BEC (|Δ|>=1e)', 'Value': len(mod_crysts)},
    {'Metric': 'Moderate Anomalous Crystals Percentage', 'Value': f"{len(mod_crysts)/len(cryst_df)*100:.2f}%"},
    {'Metric': 'Total Atoms', 'Value': len(atom_df)},
    {'Metric': 'Giant Anomalous Atoms Count (|Δ|>=2e)', 'Value': len(giant_atoms)},
    {'Metric': 'Giant Anomalous Atoms Percentage', 'Value': f"{len(giant_atoms)/len(atom_df)*100:.2f}%"},
    {'Metric': 'Moderate Anomalous Atoms Count (|Δ|>=1e)', 'Value': len(mod_atoms)},
    {'Metric': 'Moderate Anomalous Atoms Percentage', 'Value': f"{len(mod_atoms)/len(atom_df)*100:.2f}%"}
]).to_csv(summary_csv, index=False)
print(f"\nSaved summary CSV to: {summary_csv}")
print("==================================================")
