import os
import sys
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

print("=" * 65)
print("  EXTRACTING CRYSTAL SYMMETRY METADATA & DIVERSITY QUANTIFICATION")
print("  Materials Project + Mendeley Oxides + JARVIS-DFT (48,186 crystals)")
print("=" * 65)

base_dir = "/home/vibhan23/combined_bec_dataset"
json_positions_path = os.path.join(base_dir, "combined_crystal_atomic_lattice_positions.json")

print(f"Loading 3D crystal structures: {json_positions_path} ...")
with open(json_positions_path, 'r') as f:
    crystals = json.load(f)
print(f"Loaded {len(crystals):,} crystal structures.")

symmetry_rows = []
dataset_elements = Counter()
dataset_spgs = Counter()
dataset_systems = Counter()

for c in tqdm(crystals, desc="Analyzing symmetry"):
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
    unique_elements = list(set(species))
    num_unique = len(unique_elements)
    
    for el in species:
        dataset_elements[el] += 1
        
    s = Structure(lattice, species, coords)
    
    try:
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        sg_symbol = sga.get_space_group_symbol()
        sg_num = int(sga.get_space_group_number())
        c_system = sga.get_crystal_system()
        pt_group = sga.get_point_group_symbol()
        
        # Wyckoff symbols
        symm_data = sga.get_symmetry_dataset()
        wyckoffs = list(symm_data['wyckoffs']) if (symm_data and 'wyckoffs' in symm_data) else []
        wyckoff_str = ";".join(wyckoffs)
    except Exception:
        sg_symbol = "P1"
        sg_num = 1
        c_system = "triclinic"
        pt_group = "1"
        wyckoff_str = "1a" * natoms
        
    dataset_spgs[sg_num] += 1
    dataset_systems[c_system] += 1
    
    # Prototype name estimation (e.g. ABX3 perovskite, AB rocksalt, A element)
    proto = "Unknown"
    if num_unique == 1:
        proto = "Elemental"
    elif num_unique == 2:
        proto = f"Binary ({formula})"
    elif num_unique == 3:
        if natoms == 5 and "O" in species:
            proto = "ABX3 Perovskite"
        else:
            proto = f"Ternary ({formula})"
    else:
        proto = f"Quaternary ({formula})"
        
    symmetry_rows.append({
        'material_id': mat_id,
        'source_dataset': source_ds,
        'formula': formula,
        'unit_cell_formula': unit_formula,
        'split_type': split,
        'num_atoms': natoms,
        'num_unique_elements': num_unique,
        'unique_elements_list': ",".join(sorted(unique_elements)),
        'space_group_symbol': sg_symbol,
        'space_group_number': sg_num,
        'crystal_system': c_system,
        'point_group': pt_group,
        'wyckoff_symbols': wyckoff_str,
        'prototype_name': proto,
        'volume_A3': round(lp.get('volume_A3', 0.0), 3)
    })

df_symm = pd.DataFrame(symmetry_rows)
out_csv = os.path.join(base_dir, "crystal_symmetry_and_diversity_metadata.csv")
print(f"Exporting symmetry metadata ({len(df_symm):,} rows) to: {out_csv} ...")
df_symm.to_csv(out_csv, index=False)

# Compute Shannon Entropies & Diversity Index
total_sites = sum(dataset_elements.values())
h_chem = -sum((cnt / total_sites) * math.log2(cnt / total_sites) for cnt in dataset_elements.values())
h_chem_norm = h_chem / math.log2(len(dataset_elements)) if len(dataset_elements) > 1 else 0.0

total_structs = len(df_symm)
h_spg = -sum((cnt / total_structs) * math.log2(cnt / total_structs) for cnt in dataset_spgs.values())
h_spg_norm = h_spg / math.log2(230)

h_sys = -sum((cnt / total_structs) * math.log2(cnt / total_structs) for cnt in dataset_systems.values())
h_sys_norm = h_sys / math.log2(7)

# Composite Dataset Diversity Index (CDI)
composite_diversity_index = 0.4 * h_chem_norm + 0.4 * h_spg_norm + 0.2 * h_sys_norm

print("\n" + "=" * 60)
print("  DATASET DIVERSITY QUANTIFICATION SUMMARY")
print("=" * 60)
print(f"Total Unique Elements Covered : {len(dataset_elements)} elements")
print(f"Chemical Shannon Entropy (H_chem): {h_chem:.4f} bits (Normalized: {h_chem_norm:.4f})")
print(f"Space Groups Represented      : {len(dataset_spgs)} / 230 space groups")
print(f"Space Group Entropy (H_spg)   : {h_spg:.4f} bits (Normalized: {h_spg_norm:.4f})")
print(f"Crystal Systems Represented   : {len(dataset_systems)} / 7 crystal systems")
print(f"Crystal System Entropy (H_sys): {h_sys:.4f} bits (Normalized: {h_sys_norm:.4f})")
print(f"Composite Diversity Index (CDI): {composite_diversity_index:.4f} / 1.0000")

# Save summary CSV
div_summary_csv = os.path.join(base_dir, "dataset_diversity_metrics_summary.csv")
pd.DataFrame([
    {'Metric': 'Total Structures', 'Value': total_structs},
    {'Metric': 'Total Unique Elements', 'Value': len(dataset_elements)},
    {'Metric': 'Chemical Shannon Entropy (bits)', 'Value': round(h_chem, 4)},
    {'Metric': 'Normalized Chemical Diversity', 'Value': round(h_chem_norm, 4)},
    {'Metric': 'Space Groups Represented', 'Value': f"{len(dataset_spgs)} / 230"},
    {'Metric': 'Space Group Shannon Entropy (bits)', 'Value': round(h_spg, 4)},
    {'Metric': 'Normalized Space Group Diversity', 'Value': round(h_spg_norm, 4)},
    {'Metric': 'Crystal Systems Represented', 'Value': f"{len(dataset_systems)} / 7"},
    {'Metric': 'Crystal System Shannon Entropy (bits)', 'Value': round(h_sys, 4)},
    {'Metric': 'Composite Dataset Diversity Index (CDI)', 'Value': round(composite_diversity_index, 4)}
]).to_csv(div_summary_csv, index=False)
print(f"Saved diversity summary to: {div_summary_csv}")
print("==================================================")
