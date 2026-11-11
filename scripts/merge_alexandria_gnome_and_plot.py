#!/usr/bin/env python3
"""
Merge the strict pure-integer Alexandria (206,398) and GNoME (94,524) datasets
and generate:
  1. Combined compounds summary CSV & combined atoms CSV.GZ
  2. Combined Strict Pure-Integer Parity Plot (Nominal Charge vs Predicted BEC)
  3. Combined Polar vs. Non-Polar Anomaly Comparison Plot + Crystal System Summary CSV
"""

import os
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

CRYSTAL_SYSTEMS = ['Cubic', 'Tetragonal', 'Orthorhombic', 'Hexagonal', 'Trigonal', 'Monoclinic', 'Triclinic']
PALETTE = {
    'Cubic': '#1f77b4',
    'Tetragonal': '#ff7f0e',
    'Orthorhombic': '#2ca02c',
    'Hexagonal': '#d62728',
    'Trigonal': '#9467bd',
    'Monoclinic': '#8c564b',
    'Triclinic': '#e377c2'
}


def is_polar_spg(val):
    try:
        return int(float(val)) in POLAR_SPACE_GROUPS
    except Exception:
        return False


def main():
    out_dir = "/home/vibhan23/alexandria_gnome_merged"
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load & Merge Compound Summaries
    # -------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("MERGING STRICT PURE-INTEGER DATASETS: ALEXANDRIA + GNOME", flush=True)
    print("=" * 80, flush=True)

    df_alex_mats = pd.read_csv("/home/vibhan23/alexandria/alexandria_strict_pure_integer_compounds_summary.csv")
    df_alex_mats["source_dataset"] = "Alexandria"

    df_gnome_mats = pd.read_csv("/home/vibhan23/gnome/gnome_strict_pure_integer_compounds_summary.csv")
    df_gnome_mats["source_dataset"] = "GNoME"

    print(f"Alexandria compounds: {len(df_alex_mats):,}", flush=True)
    print(f"GNoME compounds:      {len(df_gnome_mats):,}", flush=True)

    # Overlap check by (formula, space_group, num_atoms)
    keys_a = set(zip(df_alex_mats["formula"], df_alex_mats["space_group"], df_alex_mats["num_atoms"]))
    keys_g = set(zip(df_gnome_mats["formula"], df_gnome_mats["space_group"], df_gnome_mats["num_atoms"]))
    print(f"Shared (formula, space_group, num_atoms) tuples between Alexandria & GNoME: {len(keys_a & keys_g):,}", flush=True)

    df_mats = pd.concat([df_alex_mats, df_gnome_mats], ignore_index=True)
    df_mats["crystal_system"] = df_mats["crystal_system"].astype(str).str.capitalize()
    df_mats["is_polar"] = df_mats["space_group"].apply(is_polar_spg)
    df_mats["polarity"] = df_mats["is_polar"].apply(lambda x: "Polar" if x else "Non-Polar")

    merged_mats_csv = os.path.join(out_dir, "alexandria_gnome_strict_pure_integer_compounds_summary.csv")
    df_mats.to_csv(merged_mats_csv, index=False)
    print(f"Saved merged compounds summary ({len(df_mats):,} materials) -> {merged_mats_csv}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Load & Merge Atom-Level Records
    # -------------------------------------------------------------------------
    print("\nLoading atom-level CSV.GZ files...", flush=True)
    df_alex_atoms = pd.read_csv("/home/vibhan23/alexandria/alexandria_strict_pure_integer_atoms_data.csv.gz", compression="gzip")
    df_alex_atoms["source_dataset"] = "Alexandria"

    df_gnome_atoms = pd.read_csv("/home/vibhan23/gnome/gnome_strict_pure_integer_atoms_data.csv.gz", compression="gzip")
    df_gnome_atoms["source_dataset"] = "GNoME"

    print(f"Alexandria atoms: {len(df_alex_atoms):,}", flush=True)
    print(f"GNoME atoms:      {len(df_gnome_atoms):,}", flush=True)

    df_atoms = pd.concat([df_alex_atoms, df_gnome_atoms], ignore_index=True)
    df_atoms["crystal_system"] = df_atoms["crystal_system"].astype(str).str.capitalize()
    del df_alex_atoms, df_gnome_atoms

    merged_atoms_csv_gz = os.path.join(out_dir, "alexandria_gnome_strict_pure_integer_atoms_data.csv.gz")
    df_atoms.to_csv(merged_atoms_csv_gz, index=False, compression="gzip")
    print(f"Saved merged atoms data ({len(df_atoms):,} atoms) -> {merged_atoms_csv_gz}", flush=True)

    # -------------------------------------------------------------------------
    # 3. Generate Combined Strict Integer Parity Plot
    # -------------------------------------------------------------------------
    parity_png = os.path.join(plots_dir, "alexandria_gnome_strict_pure_compound_parity_plot.png")
    print(f"\nGenerating Combined Strict Parity Plot -> {parity_png}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))

    min_x, max_x = df_atoms["nominal_charge"].min() - 0.5, df_atoms["nominal_charge"].max() + 0.5
    min_y, max_y = df_atoms["predicted_bec"].min() - 0.5, df_atoms["predicted_bec"].max() + 0.5
    plot_min = min(min_x, min_y)
    plot_max = max(max_x, max_y)

    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, alpha=0.7, label='Ideal Parity ($Z^* = q_{\\mathrm{nominal}}$)')

    x_fill = np.linspace(plot_min, plot_max, 200)
    ax.fill_between(x_fill, x_fill + 2.0, plot_max + 2, color='red', alpha=0.07, label='Giant Anomaly ($|\\Delta Z^*| \\geq 2.0e$)')
    ax.fill_between(x_fill, plot_min - 2, x_fill - 2.0, color='red', alpha=0.07)

    for cs in CRYSTAL_SYSTEMS:
        sub = df_atoms[df_atoms["crystal_system"] == cs]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["nominal_charge"],
            sub["predicted_bec"],
            color=PALETTE[cs],
            alpha=0.25,
            s=8,
            rasterized=True,
            label=f"{cs} (N = {len(sub):,} atoms)"
        )

    int_ticks = sorted(df_atoms["nominal_charge"].unique())
    ax.set_xticks(int_ticks)
    ax.set_xlim(min(int_ticks) - 0.6, max(int_ticks) + 0.6)
    ax.set_ylim(-10, 14)

    ax.set_xlabel("Nominal Oxidation State $q_{\\mathrm{nominal}}$ ($e$, 100% Integer Compounds)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Predicted Born Effective Charge Trace $Z^*_{\\mathrm{mean}}$ ($e$)", fontsize=13, fontweight='bold')

    clean_title = "Strict Integer Parity Plot: Nominal Charge vs Predicted BEC (Merged Alexandria + GNoME)\n"
    clean_title += f"All Compounds with Any Fractional/Zero Charges Excluded | {len(df_atoms):,} Atoms across {len(df_mats):,} Materials"
    ax.set_title(clean_title, fontsize=11.5, pad=12, fontweight='bold')

    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9.5)
    leg.set_title("Crystal Systems", prop={'weight': 'bold', 'size': 10})
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(parity_png, dpi=300)
    plt.close()
    print(f"Saved combined parity plot -> {parity_png}", flush=True)

    # -------------------------------------------------------------------------
    # 4. Generate Combined Polar vs Non-Polar Plot & Crystal System Summary CSV
    # -------------------------------------------------------------------------
    df_mats_cs = df_mats[df_mats["crystal_system"].isin(CRYSTAL_SYSTEMS)].copy()

    cryst_summary = []
    for cs in CRYSTAL_SYSTEMS:
        sub = df_mats_cs[df_mats_cs["crystal_system"] == cs]
        n_tot = len(sub)
        if n_tot == 0:
            continue
        n_polar = int((sub["is_polar"] == True).sum())
        n_nonpolar = int((sub["is_polar"] == False).sum())
        n_giant = int((sub["is_giant_anomalous_material"] == True).sum())
        giant_polar = int(((sub["is_polar"] == True) & (sub["is_giant_anomalous_material"] == True)).sum())
        giant_nonpolar = int(((sub["is_polar"] == False) & (sub["is_giant_anomalous_material"] == True)).sum())

        cryst_summary.append({
            "Crystal System": cs,
            "Total Materials": n_tot,
            "Polar Materials": n_polar,
            "Non-Polar Materials": n_nonpolar,
            "Polar %": round(n_polar / n_tot * 100, 1),
            "Giant Anomalous Materials": n_giant,
            "Giant %": round(n_giant / n_tot * 100, 2),
            "Giant in Polar": giant_polar,
            "Giant in Non-Polar": giant_nonpolar
        })

    df_cryst_summary = pd.DataFrame(cryst_summary)
    print("\n--- Merged Crystal System Breakdown (Alexandria + GNoME) ---", flush=True)
    print(df_cryst_summary.to_string(index=False), flush=True)

    cryst_summary_csv = os.path.join(out_dir, "alexandria_gnome_crystal_system_polarity_anomaly_summary.csv")
    df_cryst_summary.to_csv(cryst_summary_csv, index=False)
    print(f"Saved merged polarity summary CSV -> {cryst_summary_csv}", flush=True)

    # Map polarity onto atoms for KDE
    mat_pol_map = dict(zip(df_mats_cs["material_id"].astype(str), df_mats_cs["is_polar"]))
    df_atoms_pol = df_atoms[["material_id", "delta_z"]].copy()
    df_atoms_pol["is_polar"] = df_atoms_pol["material_id"].astype(str).map(mat_pol_map)
    df_atoms_pol = df_atoms_pol.dropna(subset=["is_polar"])
    df_atoms_pol["is_polar"] = df_atoms_pol["is_polar"].astype(bool)

    polar_deltas = df_atoms_pol[df_atoms_pol["is_polar"] == True]["delta_z"]
    nonpolar_deltas = df_atoms_pol[df_atoms_pol["is_polar"] == False]["delta_z"]
    print(f"Polar atoms: {len(polar_deltas):,} | Non-Polar atoms: {len(nonpolar_deltas):,}", flush=True)

    max_kde_samples = 200000
    polar_sample = polar_deltas.sample(max_kde_samples, random_state=42) if len(polar_deltas) > max_kde_samples else polar_deltas
    nonpolar_sample = nonpolar_deltas.sample(max_kde_samples, random_state=42) if len(nonpolar_deltas) > max_kde_samples else nonpolar_deltas

    polarity_png = os.path.join(plots_dir, "alexandria_gnome_polar_vs_nonpolar_anomaly_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.kdeplot(polar_sample, ax=ax1, color='#d62728', lw=2.5, fill=True, alpha=0.3, label=f'Polar Space Groups (N={len(polar_deltas):,})')
    sns.kdeplot(nonpolar_sample, ax=ax1, color='#1f77b4', lw=2.5, fill=True, alpha=0.2, label=f'Non-Polar Space Groups (N={len(nonpolar_deltas):,})')

    ax1.axvline(1.0, color='gray', ls=':', lw=1.5, label='Moderate Anomaly (1.0 e)')
    ax1.axvline(2.0, color='red', ls='-.', lw=1.5, label='Giant Anomaly (2.0 e)')
    ax1.set_xlim(0, 5.0)
    ax1.set_xlabel('Born Effective Charge Anomaly, $|Z^*_{\\mathrm{mean}} - q_{\\mathrm{nominal}}|$ ($e$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Anomaly Magnitude Distribution (Alexandria + GNoME)\n(Polar vs. Non-Polar Space Groups)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)

    sns.barplot(
        data=df_cryst_summary,
        x='Crystal System',
        y='Giant %',
        ax=ax2,
        palette=PALETTE
    )
    ax2.set_ylabel('Giant Anomalous Materials (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Crystal System', fontsize=12, fontweight='bold')
    ax2.set_title(f'Frequency of Giant Anomalies ($|\\Delta Z^*| \\geq 2.0\\,e$)\nAcross 7 Crystal Systems ({len(df_mats_cs):,} Pure Materials)', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)

    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            (p.get_x() + p.get_width() / 2., h),
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', xytext=(0, 2),
            textcoords='offset points'
        )

    plt.tight_layout()
    plt.savefig(polarity_png, dpi=300)
    plt.close()
    print(f"Saved merged polarity plot -> {polarity_png}", flush=True)


if __name__ == "__main__":
    main()
