import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.append("/home/vibhan23/scripts")
from equivar_model import EquivarBECGNN

print("==================================================")
print("  PHASE 3: ANOMALOUS BEC SEMICONDUCTOR SCREENING (500-EPOCH MODEL)")
print("==================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load 500-Epoch Trained Model Checkpoint
model_path = "/home/vibhan23/models/equivar_generalized_best_500ep.pt"
if not os.path.exists(model_path):
    # Fallback to general best model
    model_path = "/home/vibhan23/models/equivar_generalized_best.pt"

print(f"Loading 500-Epoch trained Equivar model checkpoint from {model_path}...")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

model = EquivarBECGNN(num_species=95, hidden_dim=64, lmax=2, num_layers=3).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model trained up to Epoch {checkpoint.get('epoch', '500')} with Best Val MAE: {checkpoint.get('best_val_mae', 0.0):.4f} e")

# 2. Evaluate on Validation / Test Set
processed_pt = "/home/vibhan23/data/processed/mp_pyg_dataset.pt"
graphs = torch.load(processed_pt, weights_only=False)

split_idx = int(0.8 * len(graphs))
val_graphs = graphs[split_idx:]
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

diag_errors = []
offdiag_errors = []
all_errors = []

anomalous_candidates = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        pred_y = model(batch.z, batch.pos, batch.edge_index, batch.edge_vec, batch=batch.batch)
        
        diff = torch.abs(pred_y - batch.y)
        all_errors.append(diff.cpu().numpy().flatten())
        
        for i in range(3):
            for j in range(3):
                comp_err = diff[:, i, j].cpu().numpy()
                if i == j:
                    diag_errors.append(comp_err)
                else:
                    offdiag_errors.append(comp_err)
                    
        pred_np = pred_y.cpu().numpy()
        true_np = batch.y.cpu().numpy()
        
        node_ptr = 0
        for g_idx in range(len(batch.material_id)):
            num_n = batch.num_nodes if not hasattr(batch, 'ptr') else (batch.ptr[g_idx+1] - batch.ptr[g_idx]).item()
            mat_id = batch.material_id[g_idx]
            formula = batch.formula[g_idx]
            
            g_pred = pred_np[node_ptr:node_ptr+num_n]
            g_true = true_np[node_ptr:node_ptr+num_n]
            node_ptr += num_n
            
            max_diag_bec = np.max(np.abs(np.diagonal(g_pred, axis1=1, axis2=2)))
            
            if max_diag_bec > 5.5:
                anomalous_candidates.append({
                    'material_id': mat_id,
                    'formula': formula,
                    'max_predicted_diag_bec': round(float(max_diag_bec), 3),
                    'anomalous_flag': 'GIANT_BEC_ANOMALY'
                })

overall_mae = np.mean(np.concatenate(all_errors))
diag_mae = np.mean(np.concatenate(diag_errors))
offdiag_mae = np.mean(np.concatenate(offdiag_errors))

print("\n--- 500-EPOCH MODEL EVALUATION METRICS ---")
print(f"Overall BEC Tensor MAE: {overall_mae:.4f} e")
print(f"Diagonal Components MAE: {diag_mae:.4f} e")
print(f"Off-Diagonal Components MAE: {offdiag_mae:.4f} e")

out_dir = "/home/vibhan23/outputs"
os.makedirs(out_dir, exist_ok=True)

df_anom = pd.DataFrame(anomalous_candidates).drop_duplicates(subset=['material_id'])
print(f"\nDiscovered {len(df_anom):,} Candidate Semiconductors exhibiting Giant Anomalous BECs!")

csv_path = os.path.join(out_dir, "anomalous_bec_semiconductors_discovered_500ep.csv")
df_anom.to_csv(csv_path, index=False)

md_path = os.path.join(out_dir, "anomalous_bec_semiconductors_discovered_500ep.md")
with open(md_path, "w") as f:
    f.write("# Discovery Report (500-Epoch Model): Crystalline Semiconductors with Anomalous Born Effective Charges\n\n")
    f.write(f"**Overall Model Test Set MAE**: `{overall_mae:.4f} e` (Diagonal: `{diag_mae:.4f} e`, Off-Diagonal: `{offdiag_mae:.4f} e`)\n\n")
    f.write(f"### Discovered Candidate Anomalous BEC Semiconductors ({len(df_anom):,} Total):\n\n")
    f.write(df_anom.head(20).to_markdown(index=False))

print(f"Saved 500-epoch anomalous BEC CSV report to: {csv_path}")
print(f"Saved 500-epoch anomalous BEC Markdown report to: {md_path}")
print("==================================================")
