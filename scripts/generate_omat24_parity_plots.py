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

omat_dir = "/home/vibhan23/omat24"
plot_dir = os.path.join(omat_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

print("=" * 75)
print("  OMAT24 (META FAIRCHEM) POLARITY & CRYSTAL SYSTEM ANALYSIS")
print("=" * 75)

summary_csv = os.path.join(omat_dir, "omat24_rattled_relax_bec_predictions_summary.csv")
if not os.path.exists(summary_csv):
    print(f"Error: {summary_csv} not found.")
    sys.exit(1)

print(f"Loading OMat24 predictions from {summary_csv} ...")
df = pd.read_csv(summary_csv)
print(f"Loaded {len(df):,} materials.")

crystal_systems = ['Cubic', 'Tetragonal', 'Orthorhombic', 'Hexagonal', 'Trigonal', 'Monoclinic', 'Triclinic']
df['crystal_system'] = df['crystal_system'].str.capitalize()
df = df[df['crystal_system'].isin(crystal_systems)].copy()

palette = {
    'Cubic': '#1f77b4',
    'Tetragonal': '#ff7f0e',
    'Orthorhombic': '#2ca02c',
    'Hexagonal': '#d62728',
    'Trigonal': '#9467bd',
    'Monoclinic': '#8c564b',
    'Triclinic': '#e377c2'
}

# 1. Summary Table
cryst_summary = []
for cs in crystal_systems:
    sub = df[df['crystal_system'] == cs]
    n_tot = len(sub)
    if n_tot == 0:
        continue
    n_polar = (sub['is_polar'] == True).sum()
    n_nonpolar = (sub['is_polar'] == False).sum()
    n_giant = sub['has_giant_anomalous_bec'].sum()
    
    giant_polar = (sub['is_polar'] & sub['has_giant_anomalous_bec']).sum()
    giant_nonpolar = (~sub['is_polar'] & sub['has_giant_anomalous_bec']).sum()
    
    cryst_summary.append({
        'Crystal System': cs,
        'Total Crystals': n_tot,
        'Polar Crystals': n_polar,
        'Non-Polar Crystals': n_nonpolar,
        'Polar %': round(n_polar / n_tot * 100, 1),
        'Giant Anomalous (|Δ| >= 2.0e)': n_giant,
        'Giant %': round(n_giant / n_tot * 100, 2),
        'Giant in Polar': giant_polar,
        'Giant in Non-Polar': giant_nonpolar
    })

df_summary = pd.DataFrame(cryst_summary)
print("\n" + "=" * 60)
print("  OMAT24 POLARITY & CRYSTAL SYSTEM ANOMALY SUMMARY")
print("=" * 60)
print(df_summary.to_string(index=False))

out_csv = os.path.join(omat_dir, "omat24_crystal_system_polarity_anomaly_summary.csv")
df_summary.to_csv(out_csv, index=False)
print(f"Saved summary table -> {out_csv}")

# 2. Polar vs Non-Polar Anomaly Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

polar_anom = df[df['is_polar']]['max_diagonal_anomaly']
nonpolar_anom = df[~df['is_polar']]['max_diagonal_anomaly']

sns.kdeplot(polar_anom, ax=ax1, color='#d62728', lw=2.5, fill=True, alpha=0.3, label=f'Polar Space Groups (N={len(polar_anom):,})')
sns.kdeplot(nonpolar_anom, ax=ax1, color='#1f77b4', lw=2.5, fill=True, alpha=0.2, label=f'Non-Polar Space Groups (N={len(nonpolar_anom):,})')
ax1.axvline(1.0, color='gray', ls=':', lw=1.5, label='Moderate Anomaly (1.0 e)')
ax1.axvline(2.0, color='red', ls='-.', lw=1.5, label='Giant Anomaly (2.0 e)')
ax1.set_xlim(0, 6.0)
ax1.set_xlabel('Max Diagonal Anomaly, $|Z_{ii} - q_{\\mathrm{nom}}|$ ($e$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax1.set_title('OMat24 Anomaly Magnitude Distribution:\nPolar vs. Non-Polar Space Groups', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=10)

# Subplot 2: Giant Anomaly Rate by Crystal System
sns.barplot(
    data=df_summary,
    x='Crystal System',
    y='Giant %',
    hue='Crystal System',
    ax=ax2,
    palette=palette,
    legend=False
)
ax2.set_ylabel('Giant Anomalous Materials (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Crystal System', fontsize=12, fontweight='bold')
ax2.set_title('Frequency of Giant Anomalies Across Crystal Systems\n(OMat24 Meta FAIRchem Relaxed Structures, 100k Materials)', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=30)

for p in ax2.patches:
    h = p.get_height()
    if h > 0:
        ax2.annotate(f"{h:.1f}%", 
                     (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', xytext=(0, 2),
                     textcoords='offset points')

plt.tight_layout()
p_path = os.path.join(plot_dir, "omat24_polar_vs_nonpolar_anomaly_comparison.png")
plt.savefig(p_path, dpi=300)
plt.close()
print(f"Saved OMat24 Polarity Comparison Plot -> {p_path}")

print("=" * 75)
print("  OMAT24 ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 75)
