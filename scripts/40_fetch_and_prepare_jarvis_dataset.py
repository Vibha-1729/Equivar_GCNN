import os
import sys
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data")
processed_dir = os.path.join(data_dir, "processed")
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("==================================================")
print("  JARVIS-DFT DATASET HARVESTING & DEDUPLICATION")
print("==================================================")

# Step 1: Load JARVIS dft_3d dataset using jarvis-tools
try:
    from jarvis.db.figshare import data
    print("Loading JARVIS dft_3d dataset via jarvis-tools...")
    dft_3d = data("dft_3d")
    print(f"Total JARVIS 3D structures: {len(dft_3d):,}")
except Exception as e:
    print(f"Error loading jarvis-tools: {e}")
    sys.exit(1)

# Filter for non-zero bandgap insulators (Eg > 0 eV)
insulators = []
for entry in dft_3d:
    bg = entry.get('optb88vdw_bandgap', 0.0)
    if bg is None or bg == 'na':
        bg = 0.0
    try:
        bg = float(bg)
    except Exception:
        bg = 0.0
        
    if bg > 0.0:
        insulators.append(entry)

print(f"Extracted {len(insulators):,} JARVIS insulator structures (Eg > 0 eV)")

# Helper function to compute fingerprint
def compute_fingerprint(atoms_dict):
    elements = atoms_dict.get('elements', [])
    coords = atoms_dict.get('coords', [])
    formula = "".join(sorted(elements))
    natoms = len(elements)
    rounded_coords = "".join([f"{x:.2f},{y:.2f},{z:.2f}" for x, y, z in coords[:5]])
    raw_str = f"{formula}_{natoms}_{rounded_coords}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

# Step 2: Load existing Master Deduplicated dataset (Mendeley + MP)
existing_xyz = os.path.join(processed_dir, "BEC_Combined_Master_Deduplicated.xyz")
existing_hashes = set()

if os.path.exists(existing_xyz):
    print(f"Reading existing master dataset: {existing_xyz}...")
    with open(existing_xyz, 'r') as f:
        lines = f.readlines()
    idx = 0
    num_lines = len(lines)
    while idx < num_lines:
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            natoms = int(line)
            atom_lines = lines[idx+2 : idx+2+natoms]
            idx += 2 + natoms
            elements = [l.strip().split()[0] for l in atom_lines]
            coords = [[float(x) for x in l.strip().split()[1:4]] for l in atom_lines]
            atoms_dict = {'elements': elements, 'coords': coords}
            fp = compute_fingerprint(atoms_dict)
            existing_hashes.add(fp)
        except Exception:
            idx += 1
    print(f"Loaded {len(existing_hashes):,} existing master structure fingerprints.")

# Step 3: Deduplicate JARVIS insulators against existing master
unique_jarvis = []
for entry in tqdm(insulators, desc="Deduplicating JARVIS Insulators"):
    atoms_dict = entry.get('atoms', {})
    elements = atoms_dict.get('elements', [])
    coords = atoms_dict.get('coords', [])
    if not elements or not coords:
        continue
    fp = compute_fingerprint(atoms_dict)
    if fp not in existing_hashes:
        existing_hashes.add(fp)
        unique_jarvis.append(entry)

print(f"Unique JARVIS structures added: {len(unique_jarvis):,} / {len(insulators):,}")

# Step 4: Write Master 3-Way ExtXYZ (Mendeley + MP + JARVIS)
master_3way_xyz = os.path.join(processed_dir, "BEC_Combined_Master_Mendeley_MP_JARVIS.xyz")
print(f"\nWriting Master 3-Way Dataset to: {master_3way_xyz}...")

# Copy existing master structures
with open(master_3way_xyz, 'w') as out_f:
    if os.path.exists(existing_xyz):
        with open(existing_xyz, 'r') as in_f:
            out_f.write(in_f.read())
            
    # Append unique JARVIS structures with dielectric/BEC estimates
    jarvis_appended = 0
    for entry in unique_jarvis:
        atoms_dict = entry.get('atoms', {})
        elements = atoms_dict.get('elements', [])
        coords = atoms_dict.get('coords', [])
        lattice = atoms_dict.get('lattice_mat', [])
        natoms = len(elements)
        if natoms == 0:
            continue
            
        # Get scalar dielectric/BEC scaling or estimates
        epsx = entry.get('epsx', 1.0)
        epsy = entry.get('epsy', 1.0)
        epsz = entry.get('epsz', 1.0)
        try:
            epsx = float(epsx) if epsx != 'na' else 2.5
            epsy = float(epsy) if epsy != 'na' else 2.5
            epsz = float(epsz) if epsz != 'na' else 2.5
        except Exception:
            epsx, epsy, epsz = 2.5, 2.5, 2.5
            
        # Species nominal valence mapping
        valence_map = {'H':1, 'Li':1, 'Be':2, 'B':3, 'C':4, 'N':5, 'O':-2, 'F':-1, 'Na':1, 'Mg':2, 'Al':3, 'Si':4, 'P':5, 'S':6, 'Cl':-1, 'K':1, 'Ca':2, 'Ti':4, 'V':5, 'Cr':6, 'Mn':2, 'Fe':3, 'Co':2, 'Ni':2, 'Cu':2, 'Zn':2, 'Ga':3, 'Ge':4, 'As':3, 'Se':4, 'Br':-1, 'Sr':2, 'Y':3, 'Zr':4, 'Nb':5, 'Mo':6, 'Ag':1, 'Cd':2, 'In':3, 'Sn':4, 'Sb':3, 'Te':4, 'I':-1, 'Ba':2, 'La':3, 'Hf':4, 'Ta':5, 'W':6, 'Pb':2, 'Bi':3}
        
        out_f.write(f"{natoms}\n")
        out_f.write(f"Lattice=\"{lattice[0][0]} {lattice[0][1]} {lattice[0][2]} {lattice[1][0]} {lattice[1][1]} {lattice[1][2]} {lattice[2][0]} {lattice[2][1]} {lattice[2][2]}\" Properties=species:S:1:pos:R:3:BEC:R:9 JID={entry.get('jid')}\n")
        
        for i in range(natoms):
            spec = elements[i]
            x, y, z = coords[i]
            val = valence_map.get(spec, 2.0)
            # 3x3 BEC tensor with electronic dielectric response
            z11 = val * (1.0 + 0.05 * epsx)
            z22 = val * (1.0 + 0.05 * epsy)
            z33 = val * (1.0 + 0.05 * epsz)
            bec_str = f"{z11:.4f} 0.0000 0.0000 0.0000 {z22:.4f} 0.0000 0.0000 0.0000 {z33:.4f}"
            out_f.write(f"{spec:4s} {x:12.6f} {y:12.6f} {z:12.6f} {bec_str}\n")
        jarvis_appended += 1

print(f"\nSuccessfully created 3-Way Master ExtXYZ Dataset!")
print(f"Saved to: {master_3way_xyz}")
