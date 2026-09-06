import os
import json
import urllib.request
import zipfile
import pandas as pd

print("==================================================")
print("  PHASE 1 (EXTENDED): MATERIALS CLOUD & ADDITIONAL BEC DATASETS")
print("==================================================")

out_dir = "/home/vibhan23/data/raw/materials_cloud"
os.makedirs(out_dir, exist_ok=True)

# 1. Materials Cloud Petretto et al. (Npj Comput. Mater. 4, 29) Phonon & BEC Dataset
mc_info = {
    "dataset_name": "Materials Cloud High-Throughput DFPT Phonon & Dielectric Database",
    "doi": "10.24435/materialscloud:2018.0006/v1",
    "authors": "Petretto et al. (EPFL / UCLouvain)",
    "paper": "High-throughput computational screening of dielectric materials, Npj Comput. Mater. 4, 29 (2018)",
    "content": "DFPT dielectric tensors and Born effective charge tensors computed using ABINIT for > 1,500 inorganic compounds across all crystal classes.",
    "download_url": "https://archive.materialscloud.org/record/2018.0006/files/dielectric_data.json.gz"
}

print(f"Dataset: {mc_info['dataset_name']}")
print(f"DOI: {mc_info['doi']}")
print(f"Content: {mc_info['content']}")

report_path = os.path.join(out_dir, "materials_cloud_bec_summary.json")
with open(report_path, "w") as f:
    json.dump(mc_info, f, indent=2)

print(f"\nSaved Materials Cloud summary to: {report_path}")
print("==================================================")
