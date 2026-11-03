import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

crystal_systems = ['Cubic', 'Tetragonal', 'Orthorhombic', 'Hexagonal', 'Trigonal', 'Monoclinic', 'Triclinic']
palette = {
    'Cubic': '#1f77b4',
    'Tetragonal': '#ff7f0e',
    'Orthorhombic': '#2ca02c',
    'Hexagonal': '#d62728',
    'Trigonal': '#9467bd',
    'Monoclinic': '#8c564b',
    'Triclinic': '#e377c2'
}

def generate_clean_parity_plot(json_gz_path, out_png_path, dataset_name, exclude_zero_fallback=True):
    print("=" * 80)
    print(f"Generating Parity Plot for: {dataset_name}")
    print(f"Input: {json_gz_path}")
    print(f"Filter: exclude_zero_fallback = {exclude_zero_fallback}")
    print("=" * 80)

    nom_charges = []
    bec_traces = []
    c_systems = []
    
    total_materials = 0
    total_atoms_scanned = 0
    total_zero_fallback_atoms = 0

    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data):,} material entries from JSON.")

    for entry in tqdm(data, desc=f"Parsing {dataset_name} atoms"):
        total_materials += 1
        cs = str(entry.get("crystal_system", "Unknown")).capitalize()
        if cs not in crystal_systems:
            continue
            
        oxi_method = entry.get("nominal_oxidation_state_method", "")
        
        atoms = entry.get("predicted_atoms_bec", [])
        for atom in atoms:
            total_atoms_scanned += 1
            nom = atom.get("nominal_oxidation_state")
            if nom is None:
                continue
                
            nom = float(nom)
            if exclude_zero_fallback and (oxi_method == "Zero_Fallback" or nom == 0.0):
                total_zero_fallback_atoms += 1
                continue
                
            # BEC mean diagonal (trace / 3)
            bec_val = atom.get("Z_trace_mean")
            if bec_val is None:
                mat3x3 = atom.get("bec_tensor_3x3")
                if mat3x3:
                    bec_val = (mat3x3[0][0] + mat3x3[1][1] + mat3x3[2][2]) / 3.0
            if bec_val is None:
                continue

            nom_charges.append(nom)
            bec_traces.append(float(bec_val))
            c_systems.append(cs)

    df_plot = pd.DataFrame({
        "nominal_charge": nom_charges,
        "predicted_bec": bec_traces,
        "crystal_system": c_systems
    })

    print(f"Total atoms scanned: {total_atoms_scanned:,}")
    print(f"Excluded Zero_Fallback atoms: {total_zero_fallback_atoms:,}")
    print(f"Final atoms in parity plot: {len(df_plot):,} across {total_materials:,} materials.")

    # ── Create High-Density Parity Plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Parity reference line y = x
    min_val = min(df_plot["nominal_charge"].min(), df_plot["predicted_bec"].min()) - 0.5
    max_val = max(df_plot["nominal_charge"].max(), df_plot["predicted_bec"].max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.7, label='Ideal Parity (Z* = q_nom)')

    # Shaded anomalous regions |Z* - q| >= 2.0
    x_fill = np.linspace(min_val, max_val, 200)
    ax.fill_between(x_fill, x_fill + 2.0, max_val + 2, color='red', alpha=0.07, label='Giant Anomaly (|ΔZ*| ≥ 2.0e)')
    ax.fill_between(x_fill, min_val - 2, x_fill - 2.0, color='red', alpha=0.07)

    # Plot each crystal system with exact atom counts in legend
    for cs in crystal_systems:
        sub = df_plot[df_plot["crystal_system"] == cs]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["nominal_charge"],
            sub["predicted_bec"],
            color=palette[cs],
            alpha=0.25,
            s=8,
            rasterized=True,
            label=f"{cs} (N = {len(sub):,} atoms)"
        )

    ax.set_xlim(-6.5, 8.5)
    ax.set_ylim(-10, 14)
    ax.set_xlabel("Nominal Oxidation State $q_{\\mathrm{nominal}}$ ($e$)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Predicted Born Effective Charge Trace $Z^*_{\\mathrm{mean}}$ ($e$)", fontsize=13, fontweight='bold')
    
    title_str = f"Parity Plot: Nominal Charge vs Predicted BEC ({dataset_name})\n"
    title_str += f"All Physical Semiconductors (Total Atoms: {len(df_plot):,}, Zero_Fallback Excluded)"
    ax.set_title(title_str, fontsize=12, pad=12, fontweight='bold')

    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9.5)
    leg.set_title("Crystal Systems", prop={'weight':'bold', 'size': 10})
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    print(f"Saved plot successfully -> {out_png_path}\n")

if __name__ == "__main__":
    # 1. GNoME Full Parity Plot
    gnome_gz = "/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json.gz"
    gnome_out = "/home/vibhan23/gnome/plots/gnome_full_database_parity_plot_no_zero_fallback.png"
    if os.path.exists(gnome_gz):
        generate_clean_parity_plot(gnome_gz, gnome_out, "GNoME (385k Stable Semiconductors)", exclude_zero_fallback=True)

    # 2. Alexandria Full Parity Plot
    alex_gz = "/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json.gz"
    alex_out = "/home/vibhan23/alexandria/plots/alexandria_full_database_parity_plot_no_zero_fallback.png"
    if os.path.exists(alex_gz):
        generate_clean_parity_plot(alex_gz, alex_out, "Alexandria (230k Stable/Metastable Semiconductors)", exclude_zero_fallback=True)

    # 3. OMat24 Full Parity Plot
    omat_gz = "/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json.gz"
    omat_out = "/home/vibhan23/omat24/plots/omat24_full_database_parity_plot_no_zero_fallback.png"
    if os.path.exists(omat_gz):
        generate_clean_parity_plot(omat_gz, omat_out, "OMat24 (rattled-relax Ground States)", exclude_zero_fallback=True)
