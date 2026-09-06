import os
import json
import pandas as pd
import numpy as np

print("==================================================")
print("  COMPLETE MULTI-SOURCE BEC DATASET INSPECTION")
print("==================================================")

base_raw = "/home/vibhan23/data/raw"

# 1. Master Table
master_csv = os.path.join(base_raw, "phase1_master_dataset_summary.csv")
if os.path.exists(master_csv):
    df_master = pd.read_csv(master_csv)
    print("\n1. MASTER DATASET SUMMARY TABLE:")
    print(df_master.to_string(index=False))

# 2. Materials Project Sample
mp_json = os.path.join(base_raw, "materials_project/mp_bec_dataset.json")
if os.path.exists(mp_json):
    with open(mp_json, "r") as f:
        mp_data = json.load(f)
    print(f"\n2. MATERIALS PROJECT DATASET ({len(mp_data):,} Total Structures):")
    sample_mp = mp_data[0]
    print(f"   - Material ID: {sample_mp['material_id']}")
    print(f"   - Formula: {sample_mp['formula']}")
    print(f"   - Number of Atoms: {sample_mp['num_atoms']}")
    print(f"   - Elements Present: {sample_mp['elements']}")
    print("   - Sample 3x3 Born Effective Charge Tensor (Atom 0):")
    bec_sample = np.array(sample_mp['born_charges'][0])
    for row in bec_sample:
        print(f"       [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")

# 3. Mendeley Official Paper Datasets Sample
mendeley_summary = os.path.join(base_raw, "mendeley_equivar/mendeley_and_literature_summary.json")
if os.path.exists(mendeley_summary):
    with open(mendeley_summary, "r") as f:
        mend_data = json.load(f)
    print("\n3. OFFICIAL MENDELEY EQUIVAR DATASETS (29,318 Total Structures):")
    for fname, desc in mend_data['mendeley_equivar_paper_data']['datasets'].items():
        print(f"   - File: {fname} -> {desc}")

# 4. Processed PyG Graphs Sample
pyg_pt = "/home/vibhan23/data/processed/mp_pyg_dataset.pt"
if os.path.exists(pyg_pt):
    import torch
    graphs = torch.load(pyg_pt, weights_only=False)
    print(f"\n4. PROCESSED PYTORCH GEOMETRIC GRAPH DATASET ({len(graphs):,} PyG Graphs):")
    g0 = graphs[0]
    print(f"   - Sample Graph Material ID: {getattr(g0, 'material_id', 'N/A')}")
    print(f"   - Formula: {getattr(g0, 'formula', 'N/A')}")
    print(f"   - Node Feature Tensor (z - Atomic Numbers): {g0.z.tolist()}")
    print(f"   - Cartesian Positions Tensor (pos): Shape {g0.pos.shape}")
    print(f"   - Edge Index Tensor (edge_index): Shape {g0.edge_index.shape} ({g0.edge_index.shape[1]} 3D graph edges)")
    print(f"   - Target BEC Tensor (y): Shape {g0.y.shape} (3x3 BEC matrix per atom)")

print("==================================================")
