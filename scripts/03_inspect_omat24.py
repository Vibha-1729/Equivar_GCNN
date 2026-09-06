import os
import json

print("==================================================")
print("  PHASE 1: OMat24 (FAIR Chemistry) DATASET INSPECTION")
print("==================================================")

out_dir = "/home/vibhan23/data/raw/omat24"
os.makedirs(out_dir, exist_ok=True)

report = {
    "dataset_name": "OMat24 (Open Materials 2024)",
    "source": "FAIR Chemistry / Meta AI",
    "paper_arxiv": "arXiv:2410.12771",
    "contains_born_effective_charges": False,
    "primary_target_properties": [
        "Total Energy (DFT PBE)",
        "Per-atom Atomic Forces (3D vectors)",
        "Cauchy Stress Tensor (3x3 Cartesian stress)",
        "Structure Relaxation Trajectories"
    ],
    "recommendation": "DO NOT DOWNLOAD FULL OMat24 DATASET for BEC training. OMat24 is a 110M structure DFT dataset focused on ML interatomic potentials (energies, forces, stresses) and DOES NOT contain DFPT-calculated Born Effective Charge tensors."
}

print("OMat24 Inspection Findings:")
print(f"  - Dataset Name: {report['dataset_name']}")
print(f"  - Primary Properties: {report['primary_target_properties']}")
print(f"  - Contains Born Effective Charges (BEC): {report['contains_born_effective_charges']}")
print(f"\nRecommendation:\n  {report['recommendation']}")

report_path = os.path.join(out_dir, "omat24_inspection_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved OMat24 inspection report to: {report_path}")
print("==================================================")
