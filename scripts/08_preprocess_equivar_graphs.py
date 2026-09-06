import os
import json
import torch
import numpy as np
import pandas as pd
import ase.io
from pymatgen.core import Structure
from torch_geometric.data import Data
from tqdm import tqdm

print("==================================================")
print("  PHASE 3: PYTORCH GEOMETRIC GRAPH PREPROCESSING")
print("==================================================")

out_dir = "/home/vibhan23/data/processed"
os.makedirs(out_dir, exist_ok=True)

CUTOFF = 5.0  # Angstrom radial cutoff

# Periodic table element to Z mapping
ELEMENT_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93
}

def build_graph_from_pymatgen(struct, born_charges, mat_id="", formula=""):
    try:
        coords = np.array(struct.cart_coords, dtype=np.float32)
        atomic_numbers = [ELEMENT_TO_Z.get(str(site.specie), 0) for site in struct]
        
        if any(z == 0 for z in atomic_numbers):
            return None
            
        z_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
        pos_tensor = torch.tensor(coords, dtype=torch.float32)
        
        # Target BEC charges
        born_np = np.array(born_charges, dtype=np.float32)
        if born_np.shape != (len(struct), 3, 3):
            return None
        y_tensor = torch.tensor(born_np, dtype=torch.float32)
        
        # Get neighbor list with periodic boundary conditions
        all_neighbors = struct.get_all_neighbors(CUTOFF, include_index=True)
        edge_src, edge_dst, edge_vecs = [], [], []
        
        for i, neighbors in enumerate(all_neighbors):
            for neighbor in neighbors:
                j = neighbor.index
                vec = neighbor.coords - coords[i]
                edge_src.append(i)
                edge_dst.append(j)
                edge_vecs.append(vec)
                
        if not edge_src:
            return None
            
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_vec = torch.tensor(np.array(edge_vecs, dtype=np.float32), dtype=torch.float32)
        
        data = Data(
            z=z_tensor,
            pos=pos_tensor,
            edge_index=edge_index,
            edge_vec=edge_vec,
            y=y_tensor,
            num_nodes=len(struct),
            material_id=mat_id,
            formula=formula
        )
        return data
    except Exception:
        return None

# 1. Process Materials Project Dataset
mp_json = "/home/vibhan23/data/raw/materials_project/mp_bec_dataset.json"
mp_graphs = []

if os.path.exists(mp_json):
    print(f"Loading Materials Project JSON from {mp_json}...")
    with open(mp_json, "r") as f:
        mp_data = json.load(f)
        
    print(f"Converting {len(mp_data):,} MP structures into PyG graphs...")
    for entry in tqdm(mp_data[:5000]):  # Subset of 5,000 diverse structures for fast training
        try:
            struct = Structure.from_dict(entry['structure_dict'])
            born = entry['born_charges']
            mat_id = entry.get('material_id', '')
            formula = entry.get('formula', '')
            
            graph = build_graph_from_pymatgen(struct, born, mat_id, formula)
            if graph is not None:
                mp_graphs.append(graph)
        except Exception:
            continue
            
    print(f"Successfully constructed {len(mp_graphs):,} PyG graphs for Materials Project!")

# Save processed dataset
mp_out = os.path.join(out_dir, "mp_pyg_dataset.pt")
torch.save(mp_graphs, mp_out)
print(f"Saved processed MP PyG dataset to: {mp_out}")

print("==================================================")
