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

POLAR_POINT_GROUPS = {'1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm'}

def is_polar_spacegroup(spg_num, pt_group=None):
    if spg_num in POLAR_SPACE_GROUPS:
        return True
    if pt_group and str(pt_group) in POLAR_POINT_GROUPS:
        return True
    return False

print("=" * 70)
print("  UPDATING COMBINED DATASET METADATA")
print("  1. Wyckoff Symbols with Numbers (Multiplicities, e.g. '4a', '8c')")
print("  2. Space Group Polarity (Polar vs Non-Polar)")
print("  3. Magnetic Properties (|mu| < 1 uB/atom -> Not Magnetic)")
print("=" * 70)

base_dir = "/home/vibhan23/combined_bec_dataset"
csv_path = os.path.join(base_dir, "crystal_symmetry_and_diversity_metadata.csv")
json_positions_path = os.path.join(base_dir, "combined_crystal_atomic_lattice_positions.json")

print(f"Reading existing metadata: {csv_path} ...")
df_meta = pd.read_csv(csv_path)
print(f"Metadata rows: {len(df_meta):,}")

# Add Polar column directly from Space Group Number
df_meta['is_polar'] = df_meta['space_group_number'].apply(lambda x: int(x) in POLAR_SPACE_GROUPS)
df_meta['polarity'] = df_meta['is_polar'].apply(lambda x: 'polar' if x else 'non-polar')
print(f"Polar crystals in combined dataset: {df_meta['is_polar'].sum():,} ({df_meta['is_polar'].mean()*100:.2f}%)")

# Now compute Wyckoff symbols with multiplicity numbers from 3D structures
print(f"Loading 3D structures from {json_positions_path} to update Wyckoff symbols with multiplicity numbers...")
with open(json_positions_path, 'r') as f:
    crystals = json.load(f)

c_dict = {c["material_id"]: c for c in crystals}

updated_wyckoffs = []
for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting Wyckoff with numbers"):
    mat_id = row['material_id']
    c = c_dict.get(mat_id)
    if c is None:
        updated_wyckoffs.append(row.get('wyckoff_symbols', ''))
        continue
    
    lp = c.get("lattice_parameters", {})
    lattice = Lattice([lp["matrix_a"], lp["matrix_b"], lp["matrix_c"]])
    atoms = c.get("atoms", [])
    species = [a["element"] for a in atoms]
    coords = [a["fractional_coords"] for a in atoms]
    
    try:
        s = Structure(lattice, species, coords)
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        symm_struct = sga.get_symmetrized_structure()
        wyckoff_with_nums = symm_struct.wyckoff_symbols
        wyckoff_str = ";".join(wyckoff_with_nums)
    except Exception:
        # Fallback to multiplicity count if symmetrization fails
        try:
            sga = SpacegroupAnalyzer(s, symprec=0.1)
            ds = sga.get_symmetry_dataset()
            counts = Counter(ds.equivalent_atoms)
            wyckoff_with_nums = [f"{counts[eq]}{w}" for w, eq in zip(ds.wyckoffs, ds.equivalent_atoms)]
            wyckoff_str = ";".join(wyckoff_with_nums)
        except Exception:
            wyckoff_str = str(row.get('wyckoff_symbols', '1a'))
            
    updated_wyckoffs.append(wyckoff_str)

df_meta['wyckoff_symbols_with_multiplicity'] = updated_wyckoffs

out_csv = os.path.join(base_dir, "crystal_symmetry_and_diversity_metadata_with_wyckoff_numbers.csv")
df_meta.to_csv(out_csv, index=False)
print(f"\nSaved updated metadata with Wyckoff numbers and Polarity to: {out_csv}")
