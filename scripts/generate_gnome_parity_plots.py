import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

gnome_dir = "/home/vibhan23/gnome"
plot_dir = os.path.join(gnome_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

print("=" * 75)
print("  GNoME PARITY PLOTS & POLARITY/CRYSTAL SYSTEM ANALYSIS")
print("=" * 75)

# Check for prediction summary file or atom-level prediction file
summary_csv = os.path.join(gnome_dir, "gnome_stable_semiconductors_bec_predictions_summary.csv")
per_atom_csv = os.path.join(gnome_dir, "gnome_semiconductors_full_3x3_predictions_full_3x3_15k.csv")

if not os.path.exists(summary_csv) and not os.path.exists(per_atom_csv):
    print(f"Error: Neither {summary_csv} nor {per_atom_csv} found yet.")
    sys.exit(1)

# Colors for crystal systems
palette = {
    'Cubic': '#1f77b4',        # Blue
    'Tetragonal': '#ff7f0e',   # Orange
    'Orthorhombic': '#2ca02c', # Green
    'Hexagonal': '#d62728',    # Red
    'Trigonal': '#9467bd',     # Purple
    'Monoclinic': '#8c564b',   # Brown
    'Triclinic': '#e377c2'     # Pink
}
crystal_systems = ['Cubic', 'Tetragonal', 'Orthorhombic', 'Hexagonal', 'Trigonal', 'Monoclinic', 'Triclinic']

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARITY PLOT: Nominal Oxidation State vs Predicted BEC
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(per_atom_csv):
    print(f"Loading per-atom predictions from {per_atom_csv} for Parity Plot...")
    df_atom = pd.read_csv(per_atom_csv)
    df_atom['crystal_system'] = df_atom['crystal_system'].str.capitalize()
    df_atom = df_atom[df_atom['crystal_system'].isin(crystal_systems)].copy()
    
    print(f"Total atom predictions: {len(df_atom):,}")
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Subsample if large for clean rasterization, preserving anomalies
    if len(df_atom) > 40000:
        df_plot = df_atom.sample(40000, random_state=42)
    else:
        df_plot = df_atom
        
    for cs in crystal_systems:
        sub = df_plot[df_plot['crystal_system'] == cs]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub['nominal_oxidation_state'],
            sub['pred_Z_mean'],
            c=palette[cs],
            label=f"{cs} (N={len(sub):,})",
            alpha=0.45,
            s=16,
            edgecolors='none'
        )
        
    # Parity line & anomaly bounds
    q_min, q_max = -3.5, 7.5
    ax.plot([q_min, q_max], [q_min, q_max], 'k--', lw=2.0, label='Rigid-Ion Parity ($Z^* = q_{\\mathrm{nominal}}$)')
    ax.plot([q_min, q_max], [q_min + 1.0, q_max + 1.0], 'gray', ls=':', lw=1.2, label='Moderate Anomaly ($|\\Delta| = 1.0\\,e$)')
    ax.plot([q_min, q_max], [q_min - 1.0, q_max - 1.0], 'gray', ls=':', lw=1.2)
    ax.plot([q_min, q_max], [q_min + 2.0, q_max + 2.0], 'red', ls='-.', lw=1.5, label='Giant Anomaly ($|\\Delta| = 2.0\\,e$)')
    ax.plot([q_min, q_max], [q_min - 2.0, q_max - 2.0], 'red', ls='-.', lw=1.5)
    
    ax.set_xlim(q_min, q_max)
    ax.set_ylim(-5.0, 11.0)
    ax.set_xlabel('Nominal Formal Oxidation State, $q_{\\mathrm{nominal}}$ ($e$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Predicted Born Effective Charge Trace, $\\bar{Z}^*_{\\mathrm{pred}}$ ($e$)', fontsize=13, fontweight='bold')
    ax.set_title('GNoME Parity Plot: Nominal Oxidation State vs. Predicted Born Charge\n(Colored by Crystal System across Google DeepMind GNoME)', fontsize=13, fontweight='bold', pad=12)
    
    ax.legend(loc='upper left', frameon=True, fontsize=9.5, markerscale=2.0, framealpha=0.92)
    plt.tight_layout()
    
    p1_path = os.path.join(plot_dir, "gnome_parity_plot_nominal_charge_vs_bec_by_crystal_system.png")
    plt.savefig(p1_path, dpi=300)
    plt.close()
    print(f"Saved GNoME Parity Plot -> {p1_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. POLAR VS NON-POLAR BREAKDOWN ON GNOME STABLE SEMICONDUCTORS
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(summary_csv):
    print(f"\nLoading GNoME crystal predictions from {summary_csv} ...")
    df_cryst = pd.read_csv(summary_csv)
    df_cryst['crystal_system'] = df_cryst['crystal_system'].str.capitalize()
    df_cryst = df_cryst[df_cryst['crystal_system'].isin(crystal_systems)].copy()
    
    print(f"Loaded {len(df_cryst):,} GNoME crystals.")
    
    cryst_summary = []
    for cs in crystal_systems:
        sub = df_cryst[df_cryst['crystal_system'] == cs]
        n_tot = len(sub)
        if n_tot == 0:
            continue
        n_polar = (sub['is_polar'] == True).sum()
        n_nonpolar = (sub['is_polar'] == False).sum()
        n_giant = sub['has_giant_anomalous_bec'].sum()
        n_mod = sub['has_moderate_anomalous_bec'].sum()
        
        giant_polar = (sub['is_polar'] & sub['has_giant_anomalous_bec']).sum()
        giant_nonpolar = (~sub['is_polar'] & sub['has_giant_anomalous_bec']).sum()
        
        cryst_summary.append({
            'Crystal System': cs,
            'Total Crystals': n_tot,
            'Polar Crystals': n_polar,
            'Non-Polar Crystals': n_nonpolar,
            'Polar %': round(n_polar / n_tot * 100, 1) if n_tot else 0,
            'Giant Anomalous (|Δ| >= 2.0e)': n_giant,
            'Giant %': round(n_giant / n_tot * 100, 2) if n_tot else 0,
            'Giant in Polar': giant_polar,
            'Giant in Non-Polar': giant_nonpolar,
            'Moderate Anomalous (|Δ| >= 1.0e)': n_mod,
            'Moderate %': round(n_mod / n_tot * 100, 1) if n_tot else 0
        })
        
    df_gnome_summary = pd.DataFrame(cryst_summary)
    print("\n" + "=" * 60)
    print("  GNoME POLARITY & CRYSTAL SYSTEM ANOMALY SUMMARY")
    print("=" * 60)
    print(df_gnome_summary.to_string(index=False))
    
    summary_out_csv = os.path.join(gnome_dir, "gnome_crystal_system_polarity_anomaly_summary.csv")
    df_gnome_summary.to_csv(summary_out_csv, index=False)
    print(f"Saved GNoME Polarity Summary Table -> {summary_out_csv}")
    
    # Plot Polar vs Non-Polar Anomaly Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    polar_anom = df_cryst[df_cryst['is_polar']]['max_diagonal_anomaly']
    nonpolar_anom = df_cryst[~df_cryst['is_polar']]['max_diagonal_anomaly']
    
    sns.kdeplot(polar_anom, ax=ax1, color='#d62728', lw=2.5, fill=True, alpha=0.3, label=f'Polar Space Groups (N={len(polar_anom):,})')
    sns.kdeplot(nonpolar_anom, ax=ax1, color='#1f77b4', lw=2.5, fill=True, alpha=0.2, label=f'Non-Polar Space Groups (N={len(nonpolar_anom):,})')
    ax1.axvline(1.0, color='gray', ls=':', lw=1.5, label='Moderate Anomaly (1.0 e)')
    ax1.axvline(2.0, color='red', ls='-.', lw=1.5, label='Giant Anomaly (2.0 e)')
    ax1.set_xlim(0, 6.0)
    ax1.set_xlabel('Max Diagonal Anomaly, $|Z_{ii} - q_{\\mathrm{nom}}|$ ($e$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('GNoME Anomaly Magnitude Distribution:\nPolar vs. Non-Polar Space Groups', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    
    # Subplot 2: Giant Anomaly Rate by Crystal System
    sns.barplot(
        data=df_gnome_summary,
        x='Crystal System',
        y='Giant %',
        hue='Crystal System',
        ax=ax2,
        palette=palette,
        legend=False
    )
    ax2.set_ylabel('Giant Anomalous Materials (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Crystal System', fontsize=12, fontweight='bold')
    ax2.set_title('Frequency of Giant Anomalies Across Crystal Systems\n(Google DeepMind GNoME Stable Semiconductors)', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", 
                         (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold', xytext=(0, 2),
                         textcoords='offset points')
                         
    plt.tight_layout()
    p2_path = os.path.join(plot_dir, "gnome_polar_vs_nonpolar_anomaly_comparison.png")
    plt.savefig(p2_path, dpi=300)
    plt.close()
    print(f"Saved GNoME Polarity Comparison Plot -> {p2_path}")

print("=" * 75)
print("  GNoME PLOTS COMPLETED SUCCESSFULLY!")
print("=" * 75)
