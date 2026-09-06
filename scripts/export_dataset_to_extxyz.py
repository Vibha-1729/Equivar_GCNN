import os
import json
import numpy as np
import pandas as pd
import ase
import ase.io
from pymatgen.core import Structure
from tqdm import tqdm

print("==================================================")
print("  EXPORTING DATASET TO EXTENDED XYZ (extxyz) FORMAT")
print("==================================================")

out_dir = "/home/vibhan23/data/processed"
os.makedirs(out_dir, exist_ok=True)

# 1. Materials Project JSON to Extended XYZ
mp_json = "/home/vibhan23/data/raw/materials_project/mp_bec_dataset.json"
out_xyz = os.path.join(out_dir, "BEC_MaterialsProject.xyz")

if os.path.exists(mp_json):
    print(f"Reading Materials Project dataset from {mp_json}...")
    with open(mp_json, "r") as f:
        mp_data = json.load(f)
        
    print(f"Converting {len(mp_data):,} structures into Extended XYZ (extxyz) format...")
    
    atoms_list = []
    for entry in tqdm(mp_data):
        try:
            struct = Structure.from_dict(entry['structure_dict'])
            born_charges = np.array(entry['born_charges'], dtype=np.float64)  # [N, 3, 3]
            
            # Convert pymatgen Structure to ASE Atoms
            symbols = [str(site.specie) for site in struct]
            positions = struct.cart_coords
            cell = struct.lattice.matrix
            pbc = (True, True, True)
            
            atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
            
            # Add metadata to info dict
            atoms.info['material_id'] = str(entry.get('material_id', ''))
            atoms.info['formula'] = str(entry.get('formula', ''))
            
            # Add per-atom 3x3 Born Effective Charge tensors (flattened to 9 scalars per atom: Z11 Z12 Z13 Z21 Z22 Z23 Z31 Z32 Z33)
            born_flat = born_charges.reshape(-1, 9)
            atoms.arrays['born_charges'] = born_flat
            
            atoms_list.append(atoms)
        except Exception:
            continue
            
    print(f"Writing {len(atoms_list):,} structures to {out_xyz}...")
    ase.io.write(out_xyz, atoms_list, format='extxyz')
    file_size_mb = os.path.getsize(out_xyz) / (1024 * 1024)
    print(f"Successfully generated Extended XYZ dataset file: {out_xyz} ({file_size_mb:.2f} MB)")

# 2. Inspection of generated Extended XYZ file
if os.path.exists(out_xyz):
    print("\n--- SAMPLE EXTENDED XYZ HEADER & FIRST ATOM RECORD ---")
    with open(out_xyz, "r") as f:
        sample_lines = [f.readline() for _ in range(12)]
    for line in sample_lines:
        print(line.rstrip())

print("==================================================")
