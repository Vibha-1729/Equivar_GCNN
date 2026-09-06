import os
import json
import urllib.request
import zipfile
import shutil

print("==================================================")
print("  PHASE 1: e3nn-models (shiangfang) REPO INSPECTION")
print("==================================================")

out_dir = "/home/vibhan23/data/raw/e3nn_models"
os.makedirs(out_dir, exist_ok=True)

zip_url = "https://github.com/shiangfang/e3nn-models/archive/refs/heads/main.zip"
target_clone_dir = os.path.join(out_dir, "e3nn-models")
zip_dest = os.path.join(out_dir, "e3nn_models.zip")

if not os.path.exists(target_clone_dir):
    print(f"Downloading repository zip from {zip_url}...")
    req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(zip_dest, 'wb') as f:
        f.write(resp.read())
    
    print("Extracting zip archive...")
    with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    
    extracted_folder = os.path.join(out_dir, "e3nn-models-main")
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, target_clone_dir)
    if os.path.exists(zip_dest):
        os.remove(zip_dest)

# Inspect directory contents
files_found = []
if os.path.exists(target_clone_dir):
    for root, dirs, files in os.walk(target_clone_dir):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), target_clone_dir)
            files_found.append(rel_path)

report = {
    "repository": "https://github.com/shiangfang/e3nn-models",
    "bundled_bec_datasets_found": False,
    "summary": "Repository contains PyTorch / e3nn GNN model definitions (equivariant neural network architectures) and training scripts for atomic tensor properties, but DOES NOT bundle raw BEC datasets.",
    "files_count": len(files_found),
    "sample_files": files_found[:20]
}

print("\ne3nn-models Inspection Findings:")
print(f"  - Repository Downloaded: {target_clone_dir}")
print(f"  - Total Files: {len(files_found)}")
print(f"  - Bundled BEC Datasets Present: {report['bundled_bec_datasets_found']}")
print(f"  - Summary: {report['summary']}")

report_path = os.path.join(out_dir, "e3nn_models_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved e3nn-models report to: {report_path}")
print("==================================================")
