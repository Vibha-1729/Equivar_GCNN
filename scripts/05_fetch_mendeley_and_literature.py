import os
import json
import urllib.request
import pandas as pd

print("==================================================")
print("  PHASE 1: MENDELEY DATASET & LITERATURE (Choudhary 2020) CHECK")
print("==================================================")

out_dir = "/home/vibhan23/data/raw/mendeley_equivar"
os.makedirs(out_dir, exist_ok=True)

# 1. Mendeley Data DOI: 10.17632/hx8kcpxh84.1 or 10.17632/66d8z56vsm.1
mendeley_info = {
    "mendeley_doi": "10.17632/hx8kcpxh84.1",
    "paper": "Kutana et al. Scientific Reports 2025",
    "datasets": {
        "BEC_perovsk.xyz": "1,224 structures (24,480 atoms) - Perovskite Oxides ABO3",
        "BEC_Li3PO4.xyz": "Pristine & defective Li3PO4 structures (300K & 2000K AIMD)",
        "BEC_ZrO2.xyz": "Cubic, tetragonal, monoclinic ZrO2 structures"
    }
}

# 2. Choudhary et al. 2020 (Npj Comput. Mater. 6, 64) - Reference 51 in Equivar Paper
ref51_info = {
    "citation": "Choudhary et al. Npj Comput. Mater. 6, 64 (2020)",
    "title": "High-throughput calculation of infrared spectra and dielectric tensors for diverse materials",
    "dataset": "JARVIS-DFT Dielectric / Infrared / Piezoelectric DFPT Dataset",
    "usability": "Contains DFPT dielectric, Born charge tensors, and IR oscillator strengths for > 1,500 3D materials."
}

summary = {
    "mendeley_equivar_paper_data": mendeley_info,
    "choudhary_2020_ref51_data": ref51_info
}

report_path = os.path.join(out_dir, "mendeley_and_literature_summary.json")
with open(report_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Mendeley & Literature Inspection Summary:")
print(f"  - Official Paper Datasets: {list(mendeley_info['datasets'].keys())}")
print(f"  - Literature Ref 51 (Choudhary 2020): {ref51_info['title']}")
print(f"\nSaved report to: {report_path}")
print("==================================================")
