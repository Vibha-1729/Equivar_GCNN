import torch
import pandas as pd
import numpy as np
import equivar_eval.scripts.evaluate  # Registers custom TorchScript ops

# 1. Load the compiled TorchScript model
model_path = 'data/BM1.pt'
model = torch.jit.load(model_path, map_location='cpu')

print("==========================================")
print("EXPORTING NUMERICAL WEIGHT ROWS TO CSV FILES")
print("==========================================")

# Convert parameter generator to a dictionary
params = dict(model.named_parameters())

# Element symbols for Z=0 to Z=99
ELEMENT_SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es"
]

# --- 1. Atomic Embedding Matrix (Shape: [100, 32]) ---
emb = params['embedding.weight'].detach().numpy()
col_names = [f"dim_{i}" for i in range(32)]
emb_df = pd.DataFrame(emb, columns=col_names)
emb_df.insert(0, 'symbol', ELEMENT_SYMBOLS[:100])
emb_df.insert(0, 'atomic_number_Z', range(100))

csv_path_1 = 'data/BM1_atomic_embeddings.csv'
emb_df.to_csv(csv_path_1, index=False)
print(f"[1] Saved Atomic Embedding Matrix (Shape: {emb.shape}) -> '{csv_path_1}'")

# --- 2. Initial Linear Projection Weights (Shape: [2048]) ---
lin0 = params['lin0.weight'].detach().numpy()
lin0_df = pd.DataFrame({'weight_index': range(len(lin0)), 'weight_value': lin0})

csv_path_2 = 'data/BM1_initial_projection_lin0.csv'
lin0_df.to_csv(csv_path_2, index=False)
print(f"[2] Saved Initial Projection Layer (Shape: {lin0.shape}) -> '{csv_path_2}'")

# --- 3. First Convolution Layer Tensor Product Weights (Shape: [61440]) ---
conv0 = params['interactions.0.convolution.convolution.weight'].detach().numpy()
conv0_df = pd.DataFrame({'weight_index': range(len(conv0)), 'weight_value': conv0})

csv_path_3 = 'data/BM1_conv0_weights.csv'
conv0_df.to_csv(csv_path_3, index=False)
print(f"[3] Saved Interaction Block 0 Conv Weights (Shape: {conv0.shape}) -> '{csv_path_3}'")

# --- 4. Final Output Layer Weights (Shape: [192]) ---
lin1 = params['lin1.weight'].detach().numpy()
lin1_df = pd.DataFrame({'weight_index': range(len(lin1)), 'weight_value': lin1})

csv_path_4 = 'data/BM1_output_layer_lin1.csv'
lin1_df.to_csv(csv_path_4, index=False)
print(f"[4] Saved Output Layer Weights (Shape: {lin1.shape}) -> '{csv_path_4}'")

print("\n==========================================")
print("ALL CSV FILES EXPORTED SUCCESSFULLY!")
print("==========================================")