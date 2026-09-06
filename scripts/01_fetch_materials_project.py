import os
import json
import pandas as pd
import numpy as np
import pyarrow.compute as pc
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()
api_key = os.environ.get('MP_API_KEY')

print("==================================================")
print("  PHASE 1: MATERIALS PROJECT BEC DATASET QUERY")
print("==================================================")

if not api_key:
    print("ERROR: MP_API_KEY environment variable not found!")
    exit(1)

out_dir = '/home/vibhan23/data/raw/materials_project'
os.makedirs(out_dir, exist_ok=True)

usable_records = []
summaries = []

with MPRester(api_key) as mpr:
    print("Querying phonon documents from Materials Project...")
    docs = mpr.materials.phonon.search()
    table = docs.pyarrow_dataset.to_table()
    
    # Filter where born is not null
    valid_mask = pc.invert(pc.is_null(table['born']))
    valid_table = table.filter(valid_mask)
    total_valid = valid_table.num_rows
    print(f"Total Materials Project entries with Born Effective Charges: {total_valid:,}")
    
    # Extract records cleanly
    print("Processing PyArrow table into dataset JSON & CSV...")
    pylist = valid_table.to_pylist()
    
    all_elements = set()
    
    for row in pylist:
        mat_id = str(row.get('identifier', ''))
        formula = str(row.get('formula_pretty', ''))
        num_atoms = row.get('nsites', 0)
        elements_raw = row.get('elements', [])
        
        if isinstance(elements_raw, list):
            elements = sorted(list(set(str(e) for e in elements_raw if e is not None)))
        else:
            elements = []
            
        all_elements.update(elements)
            
        born_charges = row.get('born', None)
        struct_dict = row.get('structure', None)
        eps_static = row.get('epsilon_static', None)
        eps_elec = row.get('epsilon_electronic', None)
        
        record = {
            'material_id': mat_id,
            'formula': formula,
            'num_atoms': num_atoms,
            'elements': elements,
            'epsilon_static': eps_static,
            'epsilon_electronic': eps_elec,
            'structure_dict': struct_dict,
            'born_charges': born_charges
        }
        usable_records.append(record)
        
        summaries.append({
            'material_id': mat_id,
            'formula': formula,
            'num_atoms': num_atoms,
            'num_elements': len(elements),
            'elements': ', '.join(elements)
        })

print(f"\nSuccessfully Processed {len(usable_records):,} MP Structures with BEC Data")

if len(usable_records) > 0:
    # Save full JSON
    json_path = os.path.join(out_dir, 'mp_bec_dataset.json')
    with open(json_path, 'w') as f:
        json.dump(usable_records, f)
    print(f"Saved JSON dataset to: {json_path}")
    
    # Save CSV Summary
    df_sum = pd.DataFrame(summaries)
    csv_path = os.path.join(out_dir, 'mp_bec_summary.csv')
    df_sum.to_csv(csv_path, index=False)
    print(f"Saved CSV summary to: {csv_path}")
    
    print(f"Total Element Coverage: {len(all_elements)} unique elements: {sorted(list(all_elements))}")

print("==================================================")
