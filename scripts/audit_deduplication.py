import json
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict

print('=' * 60)
print('  DEDUPLICATION AUDIT ACROSS UNIFIED 3-WAY DATASET')
print('=' * 60)

csv_path = '/home/vibhan23/combined_bec_dataset/combined_master_summary.csv'
df = pd.read_csv(csv_path)
print(f'Total records in master summary: {len(df):,}')
print('\nCounts by source dataset:')
print(df['source_dataset'].value_counts())

# 1. Check ID uniqueness
id_counts = df['material_id'].value_counts()
id_duplicates = id_counts[id_counts > 1]
print(f'\n[1] Duplicate Material IDs: {len(id_duplicates):,}')

# 2. Check Formula & Atom count duplicates across datasets
formula_groups = df.groupby(['formula', 'num_atoms'])
multi_dataset_formulas = []
for (formula, natoms), group in formula_groups:
    datasets = set(group['source_dataset'])
    if len(datasets) > 1:
        multi_dataset_formulas.append({
            'formula': formula,
            'num_atoms': natoms,
            'total_crystals': len(group),
            'datasets': list(datasets),
            'counts': group['source_dataset'].value_counts().to_dict()
        })

print(f'\n[2] Formulas appearing in MULTIPLE datasets (same formula + natoms): {len(multi_dataset_formulas):,}')

# 3. Geometric Fingerprint Deduplication using atomic lattice positions JSON
json_path = '/home/vibhan23/combined_bec_dataset/combined_crystal_atomic_lattice_positions.json'
print(f'\n[3] Loading {json_path} for 3D Geometric Fingerprint Analysis...')
with open(json_path, 'r') as f:
    crystals = json.load(f)

fingerprints = defaultdict(list)
for c in crystals:
    mat_id = c['material_id']
    formula = c.get('formula', '')
    natoms = c.get('num_atoms', 0)
    lp = c.get('lattice_parameters', {})
    vol = round(lp.get('volume_A3', 0.0), 1)
    
    atoms = c.get('atoms', [])
    species = sorted([a.get('element', '') for a in atoms])
    species_str = ''.join(species)
    
    coords = []
    for a in atoms[:10]:
        fc = a.get('fractional_coords', [0, 0, 0])
        coords.extend([round(x, 2) for x in fc])
    
    key = f'{formula}_{natoms}_{vol}_{species_str}_{coords}'
    fp = hashlib.md5(key.encode('utf-8')).hexdigest()
    
    ds = c.get('dataset', 'mp' if mat_id.startswith('mp-') else ('jarvis' if mat_id.startswith('JVASP-') else 'mendeley'))
    fingerprints[fp].append({
        'id': mat_id,
        'dataset': ds,
        'formula': formula,
        'vol': vol
    })

exact_geometric_dups = {k: v for k, v in fingerprints.items() if len(v) > 1}
cross_dataset_dups = {k: v for k, v in exact_geometric_dups.items() if len(set(x['dataset'] for x in v)) > 1}
intra_dataset_dups = {k: v for k, v in exact_geometric_dups.items() if len(set(x['dataset'] for x in v)) == 1}

print(f'Total Unique 3D Geometric Clusters: {len(fingerprints):,}')
print(f'Total Duplicate Clusters: {len(exact_geometric_dups):,}')
print(f'  - Intra-dataset duplicates (repeated configurations within same source): {len(intra_dataset_dups):,}')
print(f'  - Cross-dataset duplicates (exact same structure shared across MP, Mendeley, JARVIS): {len(cross_dataset_dups):,}')

if cross_dataset_dups:
    print('\nSample Cross-Dataset Duplicate Clusters:')
    for i, (fp, items) in enumerate(list(cross_dataset_dups.items())[:5]):
        form = items[0]['formula']
        v = items[0]['vol']
        print(f'  Cluster {i+1} ({form}, vol={v} A3):')
        for it in items:
            mid = it['id']
            dset = it['dataset']
            print(f'     - {mid} ({dset})')
