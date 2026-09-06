import os
import time
import json
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

import equivar_eval.scripts.evaluate  # Registers custom JIT ops for torch_scatter/torch_sparse
from equivar_eval.process import AtomsToGraphs, InMemoryDatasetUtil

print("==================================================")
print("  TRAINING BM2 MODEL FROM SCRATCH (RANDOM NOISE)")
print("==================================================")

data_dir = "data"
output_dir = "outputs"  # Output directory for saved models and plots
xyz_file = "data/BEC_perovsk.xyz"
bm2_model_path = "data/BM2.pt"

# Ensure outputs directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Extract Ground-Truth Targets from BEC_perovsk.xyz
print("\n[Step 1/5] Extracting Ground-Truth DFPT Targets...")
dfpt_targets = []
with open(xyz_file, 'r') as f:
    for line in f:
        if 'target="_JSON' in line:
            match = re.search(r'target="_JSON\s+(\[\[.*?\]\])"', line)
            if match:
                dfpt_targets.extend(json.loads(match.group(1)))

dfpt_matrix = torch.tensor(np.array(dfpt_targets), dtype=torch.float32)

# Step 2: Convert 3D XYZ Crystal Structures into PyG Graphs
print("[Step 2/5] Building Graph Representations from 3D Coordinates...")
a2g = AtomsToGraphs(
    path_in=data_dir,
    graph_max_radius=3.0,
    num_radial=32,
    edge_sh_lmax=2,
    radial_basis='Gaussian'
)

data, slices = a2g.convert()
dataset = InMemoryDatasetUtil(data, slices)

# Limit to 100 sample frames (2,000 atoms) for fast CPU demonstration
NUM_SAMPLE_FRAMES = 100
sample_dataset = dataset[:NUM_SAMPLE_FRAMES]
sample_targets = dfpt_matrix[:NUM_SAMPLE_FRAMES * 20]

print(f"-> Sample Dataset Ready: {NUM_SAMPLE_FRAMES} frames ({len(sample_targets)} total atoms)")

# Step 3: Load BM2 Architecture & Re-initialize Weights to Random Noise
print("\n[Step 3/5] Re-initializing BM2 Weights to Random Noise (Training From Scratch)...")
model = torch.jit.load(bm2_model_path, map_location='cpu')
model.train()  # Put model in training mode

for param in model.parameters():
    param.requires_grad = True
    # Re-initialize to random noise (Xavier / Uniform random)
    if param.dim() >= 2:
        nn.init.xavier_uniform_(param)
    elif param.dim() == 1:
        nn.init.uniform_(param, -0.1, 0.1)

total_params = sum(p.numel() for p in model.parameters())
print(f"-> All {total_params:,} parameters reset to Random Noise!")

# Change-of-Basis Transformation Matrix (9x9)
SQ2_1 = 1.0 / np.sqrt(2.0)
SQ3_1 = 1.0 / np.sqrt(3.0)
SQ23_1 = SQ2_1 * SQ3_1
cob = torch.tensor([
    [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,  -SQ2_1, 0.0,   0.0,   0.0,  ],
    [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,   SQ2_1,],
    [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,  -SQ2_1, 0.0,  ],
    [0.0,   0.0,   SQ2_1, 0.0,        0.0,   0.0,   0.0,   0.0,  -SQ2_1,],
    [SQ3_1, 0.0,   0.0,   2.0*SQ23_1, 0.0,   0.0,   0.0,   0.0,   0.0,  ],
    [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,   SQ2_1, 0.0,   0.0,  ],
    [0.0,   SQ2_1, 0.0,   0.0,        0.0,   0.0,   0.0,   SQ2_1, 0.0,  ],
    [0.0,   0.0,   0.0,   0.0,        SQ2_1, 0.0,  -SQ2_1, 0.0,   0.0,  ],
    [SQ3_1, 0.0,   0.0,  -SQ23_1,     0.0,   SQ2_1, 0.0,   0.0,   0.0,  ],
], dtype=torch.float32).T

# Step 4: Setup AdamW Optimizer & L1 Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.L1Loss()

loader = DataLoader(sample_dataset, batch_size=10, shuffle=False)

epochs = 15
loss_history = []

print("\n[Step 4/5] Starting BM2 Training From Scratch (15 Epochs)...")
print("-" * 50)
t0 = time.time()

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    atom_offset = 0
    
    for batch in loader:
        data_dict = batch.to_dict()
        if '_num_nodes' in data_dict:
            data_dict['_num_nodes'] = torch.tensor(data_dict['_num_nodes'], dtype=data_dict['_num_nodes'][0].dtype)
            
        optimizer.zero_grad()  # Reset gradients
        
        # Forward Pass through Equivar GCNN
        out = model(data_dict)
        out = out @ cob  # Change of basis to 3x3 Cartesian matrix
        
        # Match target for current batch
        batch_atoms = out.shape[0]
        batch_target = sample_targets[atom_offset : atom_offset + batch_atoms]
        atom_offset += batch_atoms
        
        # Compute L1 Loss & Backpropagation
        loss = criterion(out, batch_target)
        loss.backward()  # Compute partial derivatives ∂L/∂W
        optimizer.step() # Update weights W <- W - lr * grad
        
        total_loss += loss.item() * batch_atoms

    avg_loss = total_loss / len(sample_targets)
    loss_history.append(avg_loss)
    print(f"  Epoch {epoch:2d}/{epochs}  |  Training L1 Loss: {avg_loss:.4f} e")

t1 = time.time()
print("-" * 50)
print(f"-> BM2 Scratch Training Complete in {t1 - t0:.2f} seconds!")

# Step 5: Save Custom BM2 Weights & Plot Loss Curve to outputs/ folder
print("\n[Step 5/5] Saving Trained BM2 Model Weights & Generating Loss Plot to outputs/...")

save_path = os.path.join(output_dir, "BM2_trained_scratch.pt")
torch.jit.save(model, save_path)
print(f"  - Saved trained scratch model to: '{save_path}'")

# Plot Loss Curve
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.figure(figsize=(7, 4), dpi=150)
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='#2b5c8f', linewidth=2, label='BM2 Scratch Training Loss')
plt.title('BM2 Training From Scratch Loss Curve (Random Weight Start)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('L1 Loss Error (e)', fontsize=11)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, "bm2_scratch_training_loss.png")
plt.savefig(plot_path)
print(f"  - Saved scratch training loss plot image to: '{plot_path}'")

print("\n==================================================")
print("  ALL BM2 SCRATCH TRAINING STEPS FINISHED SUCCESSFULLY!")
print("==================================================")
