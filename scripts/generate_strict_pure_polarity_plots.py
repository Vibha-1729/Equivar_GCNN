import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

POLAR_SPACE_GROUPS = {
    1,
    3, 4, 5, 6, 7, 8, 9,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    75, 76, 77, 78, 79, 80,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    143, 144, 145, 146,
    156, 157, 158, 159, 160, 161,
    168, 169, 170, 171, 172, 173,
    183, 184, 185, 186
}

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

def generate_strict_polarity_plot(
    materials_summary_csv,
    atoms_csv_gz,
    out_png_path,
    out_summary_csv,
    dataset_name
):
    print("=" * 80)
    print(f"GENERATING UPDATED POLAR VS NON-POLAR ANALYSIS: {dataset_name}")
    print(f"Materials: {materials_summary_csv}")
    print(f"Atoms: {atoms_csv_gz}")
    print("=" * 80)

    # 1. Load Materials
    df_mats = pd.read_csv(materials_summary_csv)
    print(f"Loaded {len(df_mats):,} strict pure materials.")

    # Determine polarity
    def is_polar_spg(val):
        try:
            return int(float(val)) in POLAR_SPACE_GROUPS
        except:
            return False

    df_mats['is_polar'] = df_mats['space_group'].apply(is_polar_spg)
    df_mats['polarity'] = df_mats['is_polar'].apply(lambda x: 'Polar' if x else 'Non-Polar')
    df_mats['crystal_system'] = df_mats['crystal_system'].astype(str).str.capitalize()
    df_mats = df_mats[df_mats['crystal_system'].isin(crystal_systems)].copy()

    # Crystal System Breakdown
    cryst_summary = []
    for cs in crystal_systems:
        sub = df_mats[df_mats['crystal_system'] == cs]
        n_tot = len(sub)
        if n_tot == 0:
            continue
        n_polar = (sub['is_polar'] == True).sum()
        n_nonpolar = (sub['is_polar'] == False).sum()
        n_giant = (sub['is_giant_anomalous_material'] == True).sum()
        
        giant_polar = ((sub['is_polar'] == True) & (sub['is_giant_anomalous_material'] == True)).sum()
        giant_nonpolar = ((sub['is_polar'] == False) & (sub['is_giant_anomalous_material'] == True)).sum()

        cryst_summary.append({
            'Crystal System': cs,
            'Total Materials': n_tot,
            'Polar Materials': n_polar,
            'Non-Polar Materials': n_nonpolar,
            'Polar %': round(n_polar / n_tot * 100, 1),
            'Giant Anomalous Materials': n_giant,
            'Giant %': round(n_giant / n_tot * 100, 2),
            'Giant in Polar': giant_polar,
            'Giant in Non-Polar': giant_nonpolar
        })

    df_cryst_summary = pd.DataFrame(cryst_summary)
    print("\n--- Crystal System Breakdown ---")
    print(df_cryst_summary.to_string(index=False))

    os.makedirs(os.path.dirname(out_summary_csv), exist_ok=True)
    df_cryst_summary.to_csv(out_summary_csv, index=False)
    print(f"Saved polarity summary CSV -> {out_summary_csv}")

    # 2. Load Atoms for KDE Distribution Plot
    print(f"\nLoading atom-level records for anomaly distribution KDE...")
    df_atoms = pd.read_csv(atoms_csv_gz, compression="gzip", usecols=['material_id', 'delta_z'])
    
    # Merge polarity from materials
    mat_pol_map = dict(zip(df_mats['material_id'].astype(str), df_mats['is_polar']))
    df_atoms['is_polar'] = df_atoms['material_id'].astype(str).map(mat_pol_map)
    df_atoms = df_atoms.dropna(subset=['is_polar']).copy()
    df_atoms['is_polar'] = df_atoms['is_polar'].astype(bool)

    polar_deltas = df_atoms[df_atoms['is_polar'] == True]['delta_z']
    nonpolar_deltas = df_atoms[df_atoms['is_polar'] == False]['delta_z']
    print(f"Polar atoms: {len(polar_deltas):,} | Non-Polar atoms: {len(nonpolar_deltas):,}")

    # Subsample if extremely large for fast, smooth KDE
    max_kde_samples = 150000
    if len(polar_deltas) > max_kde_samples:
        polar_sample = polar_deltas.sample(max_kde_samples, random_state=42)
    else:
        polar_sample = polar_deltas

    if len(nonpolar_deltas) > max_kde_samples:
        nonpolar_sample = nonpolar_deltas.sample(max_kde_samples, random_state=42)
    else:
        nonpolar_sample = nonpolar_deltas

    # 3. Create High-Resolution 2-Panel Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: KDE distribution of Delta Z
    sns.kdeplot(polar_sample, ax=ax1, color='#d62728', lw=2.5, fill=True, alpha=0.3, label=f'Polar Space Groups (N={len(polar_deltas):,})')
    sns.kdeplot(nonpolar_sample, ax=ax1, color='#1f77b4', lw=2.5, fill=True, alpha=0.2, label=f'Non-Polar Space Groups (N={len(nonpolar_deltas):,})')

    ax1.axvline(1.0, color='gray', ls=':', lw=1.5, label='Moderate Anomaly (1.0 e)')
    ax1.axvline(2.0, color='red', ls='-.', lw=1.5, label='Giant Anomaly (2.0 e)')
    ax1.set_xlim(0, 5.0)
    ax1.set_xlabel('Born Effective Charge Anomaly, $|Z^*_{\\mathrm{mean}} - q_{\\mathrm{nominal}}|$ ($e$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Anomaly Magnitude Distribution in {dataset_name}\n(Polar vs. Non-Polar Space Groups)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Right: Bar plot of Giant Anomaly % by Crystal System
    sns.barplot(
        data=df_cryst_summary,
        x='Crystal System',
        y='Giant %',
        ax=ax2,
        palette=palette
    )
    ax2.set_ylabel('Giant Anomalous Materials (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Crystal System', fontsize=12, fontweight='bold')
    ax2.set_title(f'Frequency of Giant Anomalies ($|\\Delta Z^*| \\geq 2.0\\,e$)\nAcross 7 Crystal Systems ({len(df_mats):,} Pure Materials)', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)

    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h:.1f}%", 
                     (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', xytext=(0, 2),
                     textcoords='offset points')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    print(f"Saved updated polarity plot -> {out_png_path}\n")

if __name__ == "__main__":
    # 1. GNoME
    generate_strict_polarity_plot(
        materials_summary_csv="/home/vibhan23/gnome/gnome_strict_pure_integer_compounds_summary.csv",
        atoms_csv_gz="/home/vibhan23/gnome/gnome_strict_pure_integer_atoms_data.csv.gz",
        out_png_path="/home/vibhan23/gnome/plots/gnome_polar_vs_nonpolar_anomaly_comparison.png",
        out_summary_csv="/home/vibhan23/gnome/gnome_crystal_system_polarity_anomaly_summary.csv",
        dataset_name="GNoME Stable Semiconductors"
    )

    # 2. Alexandria
    generate_strict_polarity_plot(
        materials_summary_csv="/home/vibhan23/alexandria/alexandria_strict_pure_integer_compounds_summary.csv",
        atoms_csv_gz="/home/vibhan23/alexandria/alexandria_strict_pure_integer_atoms_data.csv.gz",
        out_png_path="/home/vibhan23/alexandria/plots/alexandria_polar_vs_nonpolar_anomaly_comparison.png",
        out_summary_csv="/home/vibhan23/alexandria/alexandria_crystal_system_polarity_anomaly_summary.csv",
        dataset_name="Alexandria Semiconductors"
    )

    # 3. OMat24
    generate_strict_polarity_plot(
        materials_summary_csv="/home/vibhan23/omat24/omat24_strict_pure_integer_compounds_summary.csv",
        atoms_csv_gz="/home/vibhan23/omat24/omat24_strict_pure_integer_atoms_data.csv.gz",
        out_png_path="/home/vibhan23/omat24/plots/omat24_polar_vs_nonpolar_anomaly_comparison.png",
        out_summary_csv="/home/vibhan23/omat24/omat24_crystal_system_polarity_anomaly_summary.csv",
        dataset_name="OMat24 Verified Semiconductors"
    )
