import os
import sys
import json
import re
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm

print("==================================================")
print("  STEP 1: CLEAN DATASET DEDUPLICATION & MASTER COMBINED CREATION")
print("==================================================")

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Datasets Paths
mendeley_files = [
    os.path.join(data_dir, "raw", "mendeley_equivar", "BEC_perovsk.xyz"),
    os.path.join(data_dir, "raw", "mendeley_equivar", "BEC_ZrO2.xyz"),
    os.path.join(data_dir, "raw", "mendeley_equivar", "BEC_Li3PO4.xyz")
]
mp_file = os.path.join(data_dir, "processed", "BEC_MaterialsProject.xyz")

def compute_structure_fingerprint(natoms, species_list, positions_flat):
    sorted_species = "".join(sorted(species_list))
    pos_round = np.round(positions_flat, decimals=3)
    pos_str = ",".join(map(str, pos_round[:30]))
    key_str = f"{natoms}_{sorted_species}_{pos_str}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def parse_and_deduplicate_xyz(filepath_list, dataset_label):
    print(f"\nProcessing ExtXYZ files for dataset: {dataset_label}")
    parsed_structures = []
    seen_fingerprints = set()
    duplicate_count = 0
    asr_invalid_count = 0
    
    for filepath in filepath_list:
        if not os.path.exists(filepath):
            print(f"Warning: File not found ({filepath})")
            continue
            
        print(f"Reading ExtXYZ: {os.path.basename(filepath)}")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        idx = 0
        num_lines = len(lines)
        struct_id = 0
        
        pbar = tqdm(total=num_lines, desc=f"Parsing {os.path.basename(filepath)}")
        while idx < num_lines:
            try:
                line = lines[idx].strip()
                if not line:
                    idx += 1
                    pbar.update(1)
                    continue
                natoms = int(line)
                header = lines[idx+1].strip()
                atom_lines = lines[idx+2 : idx+2+natoms]
                idx += 2 + natoms
                pbar.update(2 + natoms)
                
                species_list = []
                positions_flat = []
                bec_tensors = []
                atom_tokens_list = []
                
                # Check for target="_JSON [[...]]" in header (Mendeley format)
                json_match = re.search(r'target="_JSON\s+([^"]+)"', header)
                header_bec_list = None
                if json_match:
                    try:
                        header_bec_list = json.loads(json_match.group(1))
                    except Exception:
                        header_bec_list = None

                for a_i, aline in enumerate(atom_lines):
                    tokens = aline.strip().split()
                    species = tokens[0]
                    species_list.append(species)
                    pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                    positions_flat.extend(pos)
                    
                    if header_bec_list is not None and a_i < len(header_bec_list):
                        bec_vals = header_bec_list[a_i]
                        bec_tensors.append(np.array(bec_vals).reshape(3, 3))
                        bec_tokens = [str(x) for x in bec_vals]
                        atom_tokens_list.append(tokens[:4] + bec_tokens)
                    elif len(tokens) >= 13:
                        bec_vals = [float(x) for x in tokens[4:13]]
                        bec_tensors.append(np.array(bec_vals).reshape(3, 3))
                        atom_tokens_list.append(tokens[:13])
                    elif len(tokens) >= 9:
                        bec_vals = [float(x) for x in tokens[-9:]]
                        bec_tensors.append(np.array(bec_vals).reshape(3, 3))
                        atom_tokens_list.append(tokens[:4] + [str(x) for x in bec_vals])
                
                # Check Acoustic Sum Rule (ASR) Charge Drift
                if len(bec_tensors) > 0:
                    unit_cell_sum = np.sum(bec_tensors, axis=0)
                    drift_norm = float(np.linalg.norm(unit_cell_sum))
                    
                    if drift_norm > 1.0:
                        asr_invalid_count += 1
                        continue
                        
                    # Structure Fingerprint for Deduplication
                    fp = compute_structure_fingerprint(natoms, species_list, positions_flat)
                    if fp in seen_fingerprints:
                        duplicate_count += 1
                        continue
                    seen_fingerprints.add(fp)
                    
                    parsed_structures.append({
                        'dataset': dataset_label,
                        'file': os.path.basename(filepath),
                        'struct_id': struct_id,
                        'natoms': natoms,
                        'header': header,
                        'species': species_list,
                        'atom_tokens': atom_tokens_list,
                        'asr_drift': drift_norm
                    })
                    struct_id += 1
            except Exception as e:
                idx += 1
                pbar.update(1)
        pbar.close()

    print(f"--- DEDUPLICATION & ASR SUMMARY: {dataset_label} ---")
    print(f"Total Unique Valid Structures Extracted: {len(parsed_structures):,}")
    print(f"Duplicate Structures Removed: {duplicate_count:,}")
    print(f"Unphysical ASR Drift (>1.0e) Structures Removed: {asr_invalid_count:,}")
    
    return parsed_structures

# Parse Mendeley and Materials Project datasets
mendeley_structs = parse_and_deduplicate_xyz(mendeley_files, "Mendeley")
mp_structs = parse_and_deduplicate_xyz([mp_file], "MaterialsProject")

# Combine datasets into Master Dataset
master_combined_structs = mendeley_structs + mp_structs
print(f"\n==================================================")
print(f"  MASTER COMBINED DATASET SUMMARY")
print(f"==================================================")
print(f"Mendeley Unique Structures:          {len(mendeley_structs):,}")
print(f"Materials Project Unique Structures: {len(mp_structs):,}")
print(f"TOTAL MASTER COMBINED STRUCTURES:    {len(master_combined_structs):,}")

# Export Master Combined Dataset to Extended XYZ format
master_xyz_out = os.path.join(data_dir, "processed", "BEC_Combined_Master_Deduplicated.xyz")
if not os.path.exists(os.path.dirname(master_xyz_out)):
    os.makedirs(os.path.dirname(master_xyz_out), exist_ok=True)

print(f"\nExporting Master Combined Dataset to ExtXYZ: {master_xyz_out}...")
with open(master_xyz_out, 'w') as f:
    for struct in master_combined_structs:
        f.write(f"{struct['natoms']}\n")
        f.write(f"{struct['header']}\n")
        for tokens in struct['atom_tokens']:
            f.write(f"{' '.join(tokens)}\n")

print(f"Successfully exported Master Combined Dataset ({len(master_combined_structs):,} structures) to: {master_xyz_out}")

# Save Summary Table CSV
summary_csv = os.path.join(output_dir, "master_dataset_deduplication_summary.csv")
pd.DataFrame([
    {'Dataset': 'Mendeley Dataset', 'Unique_Structures': len(mendeley_structs)},
    {'Dataset': 'Materials Project Dataset', 'Unique_Structures': len(mp_structs)},
    {'Dataset': 'Master Combined Dataset', 'Unique_Structures': len(master_combined_structs)}
]).to_csv(summary_csv, index=False)

print(f"Saved deduplication summary CSV to: {summary_csv}")
print("==================================================")
