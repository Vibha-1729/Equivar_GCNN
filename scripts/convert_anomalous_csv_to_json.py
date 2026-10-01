import json
import pandas as pd
from tqdm import tqdm

csv_path = "/home/vibhan23/combined_bec_dataset/anomalous_bec_with_nominal_oxidation_states.csv"
json_path = "/home/vibhan23/combined_bec_dataset/anomalous_bec_with_nominal_oxidation_states.json"

print(f"Reading {csv_path} ...")
# Read chunk by chunk to be memory efficient and fast
materials_dict = {}

chunks = pd.read_csv(csv_path, chunksize=100000)
for chunk in tqdm(chunks, desc="Processing Chunks"):
    for _, row in chunk.iterrows():
        mid = str(row['material_id'])
        if mid not in materials_dict:
            materials_dict[mid] = {
                'material_id': mid,
                'source_dataset': str(row['source_dataset']),
                'formula': str(row['formula']),
                'split_type': str(row['split_type']),
                'num_atoms': int(row['num_atoms']),
                'elements': []
            }
        materials_dict[mid]['elements'].append({
            'atom_index': int(row['atom_index']),
            'element': str(row['element']),
            'atomic_number': int(row['atomic_number']),
            'nominal_oxidation_state': float(row['nominal_oxidation_state']),
            'bec_target': {
                'Zxx': float(row['target_Zxx']),
                'Zyy': float(row['target_Zyy']),
                'Zzz': float(row['target_Zzz']),
                'average': float(row['target_Z_mean'])
            },
            'bec_pred': {
                'Zxx': float(row['pred_Zxx']),
                'Zyy': float(row['pred_Zyy']),
                'Zzz': float(row['pred_Zzz']),
                'average': float(row['pred_Z_mean'])
            },
            'delta_trace_anomaly': float(row['delta_trace_anomaly']),
            'delta_max_component': float(row['delta_max_component']),
            'is_giant_anomalous': bool(row['is_giant_anomalous']),
            'is_moderate_anomalous': bool(row['is_moderate_anomalous'])
        })

print(f"Total compounds grouped: {len(materials_dict):,}")
print(f"Writing to {json_path} ...")
with open(json_path, 'w') as f:
    json.dump(materials_dict, f, indent=2)
print("Done!")
