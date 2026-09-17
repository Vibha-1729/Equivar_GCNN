import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

print("==================================================")
print("  QUANTIFYING DATASET DIVERSITY METRICS")
print("==================================================")

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

data_dir = os.path.join(base_dir, "data", "processed")
output_dir = os.path.join(base_dir, "outputs", "csv_results")
os.makedirs(output_dir, exist_ok=True)

# Helper function to parse ExtXYZ into structure summaries
def parse_extxyz_structures(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    structures = []
    with open(filepath, 'r') as f:
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
            header = lines[idx+1].strip()
            atom_lines = lines[idx+2 : idx+2+natoms]
            idx += 2 + natoms
            elements = [l.strip().split()[0] for l in atom_lines]
            coords = np.array([[float(x) for x in l.strip().split()[1:4]] for l in atom_lines])
            
            # Extract spacegroup if present, else 1
            spg = 1
            for token in header.split():
                if 'spg=' in token.lower() or 'spacegroup=' in token.lower():
                    try:
                        spg = int(token.split('=')[1])
                    except Exception:
                        spg = 1
            structures.append({
                'natoms': natoms,
                'elements': elements,
                'coords': coords,
                'spg': spg
            })
        except Exception:
            idx += 1
    return structures

# 1. Chemical Shannon Entropy
def compute_chemical_entropy(structures):
    element_counts = {}
    total_sites = 0
    for s in structures:
        for elem in s['elements']:
            element_counts[elem] = element_counts.get(elem, 0) + 1
            total_sites += 1
    if total_sites == 0:
        return 0.0, 0
    entropy = 0.0
    for elem, count in element_counts.items():
        p = count / total_sites
        entropy -= p * math.log2(p)
    return entropy, len(element_counts)

# 2. Space Group Entropy
def compute_spacegroup_entropy(structures):
    spg_counts = {}
    total_structs = len(structures)
    for s in structures:
        spg = s['spg']
        spg_counts[spg] = spg_counts.get(spg, 0) + 1
    if total_structs == 0:
        return 0.0, 0
    entropy = 0.0
    for spg, count in spg_counts.items():
        p = count / total_structs
        entropy -= p * math.log2(p)
    return entropy, len(spg_counts)

# 3. Mean Pairwise Structural Dissimilarity
def compute_pairwise_structural_dissimilarity(structures, sample_size=1000):
    if len(structures) == 0:
        return 0.0
    np.random.seed(42)
    sample_indices = np.random.choice(len(structures), min(sample_size, len(structures)), replace=False)
    sample_structs = [structures[i] for i in sample_indices]
    
    # Generate structural fingerprint for each sample (mean distance histogram + natoms)
    fingerprints = []
    for s in sample_structs:
        coords = s['coords']
        if len(coords) > 1:
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            hist, _ = np.histogram(dist, bins=10, range=(0, 5.0), density=True)
        else:
            hist = np.zeros(10)
        fingerprints.append(hist)
        
    fps = np.array(fingerprints)
    norms = np.linalg.norm(fps, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_fps = fps / norms
    
    # Pairwise cosine dissimilarity = 1 - cosine_similarity
    sim_matrix = np.dot(normalized_fps, normalized_fps.T)
    dissimilarity_matrix = 1.0 - sim_matrix
    
    # Average off-diagonal dissimilarity
    n = len(sample_structs)
    if n <= 1:
        return 0.0
    mean_dissimilarity = (np.sum(dissimilarity_matrix) - np.trace(dissimilarity_matrix)) / (n * (n - 1))
    return mean_dissimilarity

# Load Datasets
master_3way_xyz = os.path.join(data_dir, "BEC_Combined_Master_Mendeley_MP_JARVIS.xyz")
master_2way_xyz = os.path.join(data_dir, "BEC_Combined_Master_Deduplicated.xyz")

print("Parsing Master 3-Way Dataset...")
all_structures = parse_extxyz_structures(master_3way_xyz)
if not all_structures:
    all_structures = parse_extxyz_structures(master_2way_xyz)

mendeley_structs = all_structures[:24327] if len(all_structures) >= 24327 else all_structures
mp_structs = all_structures[24327:38857] if len(all_structures) >= 38857 else []
jarvis_structs = all_structures[38857:] if len(all_structures) > 38857 else []
master_2way_structs = all_structures[:38857] if len(all_structures) >= 38857 else all_structures

datasets_to_audit = [
    ('Mendeley-Only', mendeley_structs),
    ('MP-Only', mp_structs),
    ('JARVIS-Only', jarvis_structs),
    ('Master 2-Way (Mendeley + MP)', master_2way_structs),
    ('Master 3-Way (Mendeley + MP + JARVIS)', all_structures)
]

diversity_results = []

for name, structs in datasets_to_audit:
    if not structs:
        continue
    print(f"\nAuditing Diversity for: {name} ({len(structs):,} structures)...")
    chem_entropy, num_elements = compute_chemical_entropy(structs)
    spg_entropy, num_spgs = compute_spacegroup_entropy(structs)
    struct_dissimilarity = compute_pairwise_structural_dissimilarity(structs, sample_size=1000)
    
    print(f"  Chemical Shannon Entropy (H_chem) : {chem_entropy:.4f} (over {num_elements} elements)")
    print(f"  Space Group Entropy (H_space)     : {spg_entropy:.4f} (over {num_spgs} space groups)")
    print(f"  Structural Dissimilarity (D_struct): {struct_dissimilarity:.4f}")
    
    diversity_results.append({
        'Dataset': name,
        'Structure_Count': len(structs),
        'Unique_Elements': num_elements,
        'Chemical_Shannon_Entropy_H_chem': chem_entropy,
        'Space_Groups_Present': num_spgs,
        'Space_Group_Entropy_H_space': spg_entropy,
        'Structural_Dissimilarity_D_struct': struct_dissimilarity
    })

# Save Summary CSV
div_csv = os.path.join(output_dir, "dataset_diversity_metrics.csv")
pd.DataFrame(diversity_results).to_csv(div_csv, index=False)
print(f"\nSaved dataset diversity metrics summary CSV to: {div_csv}")
print("==================================================")
