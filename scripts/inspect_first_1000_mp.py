import json
import pandas as pd
import numpy as np

print("==================================================")
print("  FIRST 1,000 MATERIALS PROJECT LATTICE SYSTEMS")
print("==================================================")

json_path = "/home/vibhan23/data/raw/materials_project/mp_bec_dataset.json"
with open(json_path, "r") as f:
    data = json.load(f)

print(f"Total Materials Project Structures: {len(data):,}\n")

records = []
for i, entry in enumerate(data[:1000]):
    mat_id = entry.get('material_id', f'mp-{i}')
    formula = entry.get('formula', 'N/A')
    num_atoms = entry.get('num_atoms', 0)
    elements = entry.get('elements', [])
    
    born = np.array(entry.get('born_charges', []))
    if len(born) > 0:
        max_diag = np.max(np.abs(np.diagonal(born, axis1=1, axis2=2)))
    else:
        max_diag = 0.0
        
    records.append({
        'Index': i + 1,
        'Material_ID': mat_id,
        'Formula': formula,
        'Atoms': num_atoms,
        'Elements': ", ".join(elements[:3]),
        'Max_Diag_BEC_e': round(float(max_diag), 3)
    })

df = pd.DataFrame(records)

print("--- PREVIEW OF FIRST 30 LATTICE SYSTEMS ---")
print(df.head(30).to_string(index=False))

print("\n--- STATISTICAL SUMMARY OF FIRST 1,000 STRUCTURES ---")
print(f"Unique Formulas: {df['Formula'].nunique():,}")
print(f"Average Atoms per Unit Cell: {df['Atoms'].mean():.2f}")
print(f"Maximum Born Charge (|Z*|_max): {df['Max_Diag_BEC_e'].max():.3f} e")

out_csv = "/home/vibhan23/outputs/first_1000_mp_lattice_systems.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved CSV summary table of 1,000 entries to: {out_csv}")
print("==================================================")
