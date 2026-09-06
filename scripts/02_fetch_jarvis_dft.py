import os
import json
import pandas as pd
import numpy as np
from jarvis.db.figshare import data

print("==================================================")
print("  PHASE 1: JARVIS-DFT BEC & DIELECTRIC DATASET QUERY")
print("==================================================")

out_dir = '/home/vibhan23/data/raw/jarvis_dft'
os.makedirs(out_dir, exist_ok=True)

print("Loading JARVIS-DFT 3D dataset ('dft_3d' via jarvis-tools)...")
try:
    d3d = data('dft_3d')
    print(f"Total 3D structures in JARVIS-DFT: {len(d3d):,}")
    
    bec_records = []
    summaries = []
    
    # Sample keys
    if len(d3d) > 0:
        print(f"Available keys per record: {list(d3d[0].keys())[:25]}")
    
    for entry in d3d:
        jid = str(entry.get('jid', ''))
        formula = str(entry.get('formula', ''))
        
        # Check for BEC / dielectric / piezoelectric fields
        dielectric = entry.get('optb88vvd_dielectric', entry.get('epsx', None))
        piezo = entry.get('piezoelectric', entry.get('dfpt_piezoelectric', None))
        bec_data = entry.get('bec', entry.get('born', entry.get('born_effective_charges', None)))
        
        # Filter entries with dielectric/BEC/piezo properties
        if bec_data is not None or dielectric is not None or piezo is not None:
            atoms_info = entry.get('atoms', {})
            elements = entry.get('elements', [])
            if not elements and isinstance(atoms_info, dict):
                elements = atoms_info.get('elements', [])
            
            num_atoms = len(elements)
            
            record = {
                'jid': jid,
                'formula': formula,
                'num_atoms': num_atoms,
                'elements': elements,
                'has_bec': bec_data is not None,
                'bec': bec_data if bec_data is not None else None,
                'dielectric': dielectric,
                'piezoelectric': piezo,
                'atoms_dict': atoms_info
            }
            bec_records.append(record)
            summaries.append({
                'jid': jid,
                'formula': formula,
                'num_atoms': num_atoms,
                'has_bec': bec_data is not None,
                'has_dielectric': dielectric is not None,
                'elements': ', '.join(elements) if isinstance(elements, list) else str(elements)
            })
            
    print(f"\nTotal JARVIS-DFT Entries with Dielectric/BEC/Piezo Data: {len(bec_records):,}")
    has_bec_count = sum(1 for r in bec_records if r['has_bec'])
    print(f"  - Entries with explicit DFPT Born Effective Charges (BEC): {has_bec_count:,}")
    
    # Save full JSON
    json_path = os.path.join(out_dir, 'jarvis_bec_dataset.json')
    with open(json_path, 'w') as f:
        json.dump(bec_records, f)
    print(f"Saved JSON dataset to: {json_path}")
    
    # Save CSV Summary
    if summaries:
        df_sum = pd.DataFrame(summaries)
        csv_path = os.path.join(out_dir, 'jarvis_bec_summary.csv')
        df_sum.to_csv(csv_path, index=False)
        print(f"Saved CSV summary to: {csv_path}")

except Exception as e:
    print(f"Error querying JARVIS-DFT: {e}")

print("==================================================")
