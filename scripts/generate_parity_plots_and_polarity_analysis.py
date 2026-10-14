import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

POLAR_SPACE_GROUPS = {
    1,
    3, 4, 5,
    6, 7, 8, 9,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    75, 76, 77, 78, 79, 80,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    143, 144, 145, 146,
    156, 157, 158, 159, 160, 161,
    168, 169, 170, 171, 172, 173,
    183, 184, 185, 186
}

base_dir = "/home/vibhan23/combined_bec_dataset"
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

print("=" * 75)
print("  PARITY PLOTS & POLARITY/CRYSTAL SYSTEM ANALYSIS")
print("  (Materials Project + JARVIS-DFT | Mendeley Omitted)")
print("=" * 75)

# 1. Load Data
bec_csv = os.path.join(base_dir, "anomalous_bec_with_nominal_oxidation_states.csv")
meta_csv = os.path.join(base_dir, "crystal_symmetry_and_diversity_metadata.csv")

print(f"Loading BEC predictions: {bec_csv} ...")
df_bec = pd.read_csv(bec_csv)
print(f"Total rows before filter: {len(df_bec):,}")

# Filter: OMIT MENDELEY
df_bec = df_bec[df_bec['source_dataset'] != 'mendeley'].copy()
print(f"Rows after omitting Mendeley (MP + JARVIS only): {len(df_bec):,}")

print(f"Loading crystal metadata: {meta_csv} ...")
df_meta = pd.read_csv(meta_csv)
df_meta = df_meta[df_meta['source_dataset'] != 'mendeley'].copy()

# Add Polarity to metadata
df_meta['is_polar'] = df_meta['space_group_number'].apply(lambda x: int(x) in POLAR_SPACE_GROUPS)
df_meta['polarity'] = df_meta['is_polar'].apply(lambda x: 'Polar' if x else 'Non-Polar')

# Merge
df = df_bec.merge(
    df_meta[['material_id', 'crystal_system', 'space_group_number', 'space_group_symbol', 'is_polar', 'polarity']],
    on='material_id',
    how='inner'
)
print(f"Merged dataset entries: {len(df):,}")

# Standardize crystal system names
df['crystal_system'] = df['crystal_system'].str.capitalize()
crystal_systems = ['Cubic', 'Tetragonal', 'Orthorhombic', 'Hexagonal', 'Trigonal', 'Monoclinic', 'Triclinic']
df = df[df['crystal_system'].isin(crystal_systems)].copy()

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

# ─────────────────────────────────────────────────────────────────────────────
# 2. Polarity & Crystal System Statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  POLARITY & CRYSTAL SYSTEM ANOMALY BREAKDOWN")
print("=" * 60)

# Crystal-level aggregation
cryst_grp = df.groupby(['material_id', 'crystal_system', 'is_polar', 'polarity']).agg({
    'delta_trace_anomaly': 'max',
    'is_giant_anomalous': 'any',
    'is_moderate_anomalous': 'any'
}).reset_index()

cryst_summary = []
for cs in crystal_systems:
    sub = cryst_grp[cryst_grp['crystal_system'] == cs]
    n_tot = len(sub)
    if n_tot == 0:
        continue
    n_polar = (sub['is_polar'] == True).sum()
    n_nonpolar = (sub['is_polar'] == False).sum()
    n_giant = sub['is_giant_anomalous'].sum()
    n_mod = sub['is_moderate_anomalous'].sum()
    
    giant_polar = (sub['is_polar'] & sub['is_giant_anomalous']).sum()
    giant_nonpolar = (~sub['is_polar'] & sub['is_giant_anomalous']).sum()
    
    cryst_summary.append({
        'Crystal System': cs,
        'Total Crystals': n_tot,
        'Polar Crystals': n_polar,
        'Non-Polar Crystals': n_nonpolar,
        'Polar %': round(n_polar / n_tot * 100, 1),
        'Giant Anomalous (|Δ| >= 2.0e)': n_giant,
        'Giant %': round(n_giant / n_tot * 100, 2),
        'Giant in Polar': giant_polar,
        'Giant in Non-Polar': giant_nonpolar,
        'Moderate Anomalous (|Δ| >= 1.0e)': n_mod,
        'Moderate %': round(n_mod / n_tot * 100, 1)
    })

df_cryst_summary = pd.DataFrame(cryst_summary)
print(df_cryst_summary.to_string(index=False))

summary_csv_path = os.path.join(base_dir, "crystal_system_polarity_anomalous_bec_summary.csv")
df_cryst_summary.to_csv(summary_csv_path, index=False)
print(f"Saved polarity & crystal system summary to: {summary_csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOT 1: PARITY PLOT - Nominal Charge vs Born Effective Charge
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Plot 1/3] Generating Parity Plot: Nominal Oxidation State vs BEC Trace...")

fig, ax = plt.subplots(figsize=(9, 8))

# Subsample if large for clean rasterization, but preserve anomalies
if len(df) > 30000:
    df_plot = df.sample(30000, random_state=42)
else:
    df_plot = df

# Plot each crystal system
for cs in crystal_systems:
    sub = df_plot[df_plot['crystal_system'] == cs]
    ax.scatter(
        sub['nominal_oxidation_state'],
        sub['pred_Z_mean'],
        c=palette[cs],
        label=f"{cs} (N={len(sub):,})",
        alpha=0.45,
        s=16,
        edgecolors='none'
    )

# Parity diagonal line (Rigid Ion Limit: Z* = q_nom)
q_min, q_max = -3.5, 7.5
ax.plot([q_min, q_max], [q_min, q_max], 'k--', lw=2.0, label='Rigid-Ion Parity ($Z^* = q_{\\mathrm{nominal}}$)')

# Anomaly threshold bands (|Δ| = 1.0 e and 2.0 e)
ax.plot([q_min, q_max], [q_min + 1.0, q_max + 1.0], 'gray', ls=':', lw=1.2, label='Moderate Anomaly ($|\\Delta| = 1.0\\,e$)')
ax.plot([q_min, q_max], [q_min - 1.0, q_max - 1.0], 'gray', ls=':', lw=1.2)
ax.plot([q_min, q_max], [q_min + 2.0, q_max + 2.0], 'red', ls='-.', lw=1.5, label='Giant Anomaly ($|\\Delta| = 2.0\\,e$)')
ax.plot([q_min, q_max], [q_min - 2.0, q_max - 2.0], 'red', ls='-.', lw=1.5)

ax.set_xlim(q_min, q_max)
ax.set_ylim(-5.0, 11.0)
ax.set_xlabel('Nominal Formal Oxidation State, $q_{\\mathrm{nominal}}$ ($e$)', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Born Effective Charge Trace, $\\bar{Z}^*_{\\mathrm{pred}}$ ($e$)', fontsize=13, fontweight='bold')
ax.set_title('Parity Plot: Nominal Oxidation State vs. Born Effective Charge\n(Colored by Crystal System | MP + JARVIS, Mendeley Omitted)', fontsize=14, fontweight='bold', pad=12)

ax.legend(loc='upper left', frameon=True, fontsize=9.5, markerscale=2.0, framealpha=0.92)
plt.tight_layout()

p1_path = os.path.join(plot_dir, "parity_plot_nominal_charge_vs_bec_by_crystal_system.png")
plt.savefig(p1_path, dpi=300)
plt.close()
print(f"Saved Plot 1 -> {p1_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT 2: PARITY PLOT - Predicted BEC vs DFT Ground Truth BEC
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Plot 2/3] Generating Parity Plot: Predicted BEC vs DFT Target...")

fig, ax = plt.subplots(figsize=(9, 8))

for cs in crystal_systems:
    sub = df_plot[df_plot['crystal_system'] == cs]
    ax.scatter(
        sub['target_Z_mean'],
        sub['pred_Z_mean'],
        c=palette[cs],
        label=cs,
        alpha=0.45,
        s=16,
        edgecolors='none'
    )

z_min, z_max = -5.0, 11.0
ax.plot([z_min, z_max], [z_min, z_max], 'k--', lw=2.0, label='Exact Parity (Predicted = DFT)')

# Compute MAE & R2
mae_overall = np.abs(df['pred_Z_mean'] - df['target_Z_mean']).mean()
r2_overall = np.corrcoef(df['pred_Z_mean'], df['target_Z_mean'])[0, 1] ** 2

ax.text(0.05, 0.90, f"Overall MAE = {mae_overall:.3f} e\n$R^2$ = {r2_overall:.4f}\nTotal Atoms = {len(df):,}", 
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

ax.set_xlim(z_min, z_max)
ax.set_ylim(z_min, z_max)
ax.set_xlabel('DFT Ground Truth Born Charge Trace, $\\bar{Z}^*_{\\mathrm{DFT}}$ ($e$)', fontsize=13, fontweight='bold')
ax.set_ylabel('Model Predicted Born Charge Trace, $\\bar{Z}^*_{\\mathrm{pred}}$ ($e$)', fontsize=13, fontweight='bold')
ax.set_title('Equivariant GNN Parity: Predicted BEC vs. DFT Ground Truth\n(Colored by Crystal System | MP + JARVIS)', fontsize=14, fontweight='bold', pad=12)

ax.legend(loc='lower right', frameon=True, fontsize=10, markerscale=2.0, framealpha=0.92)
plt.tight_layout()

p2_path = os.path.join(plot_dir, "parity_plot_predicted_vs_dft_bec_by_crystal_system.png")
plt.savefig(p2_path, dpi=300)
plt.close()
print(f"Saved Plot 2 -> {p2_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT 3: POLAR VS NON-POLAR ANOMALOUS BEC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Plot 3/3] Generating Polar vs Non-Polar Anomaly Distribution Plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Distribution of Anomalies in Polar vs Non-Polar
polar_anom = df[df['is_polar']]['delta_trace_anomaly']
nonpolar_anom = df[~df['is_polar']]['delta_trace_anomaly']

sns.kdeplot(polar_anom, ax=ax1, color='#d62728', lw=2.5, fill=True, alpha=0.3, label=f'Polar Space Groups (N={len(polar_anom):,})')
sns.kdeplot(nonpolar_anom, ax=ax1, color='#1f77b4', lw=2.5, fill=True, alpha=0.2, label=f'Non-Polar Space Groups (N={len(nonpolar_anom):,})')

ax1.axvline(1.0, color='gray', ls=':', lw=1.5, label='Moderate Anomaly (1.0 e)')
ax1.axvline(2.0, color='red', ls='-.', lw=1.5, label='Giant Anomaly (2.0 e)')
ax1.set_xlim(0, 5.0)
ax1.set_xlabel('Born Charge Anomaly, $|\\bar{Z}^* - q_{\\mathrm{nom}}|$ ($e$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax1.set_title('Anomaly Magnitude Distribution:\nPolar vs. Non-Polar Space Groups', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=10)

# Subplot 2: Crystal System breakdown of Giant Anomalies
sns.barplot(
    data=df_cryst_summary,
    x='Crystal System',
    y='Giant %',
    ax=ax2,
    palette=palette
)
ax2.set_ylabel('Giant Anomalous Materials (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Crystal System', fontsize=12, fontweight='bold')
ax2.set_title('Frequency of Giant Anomalies ($|\\Delta| \\geq 2.0\\,e$)\nAcross the 7 Crystal Systems', fontsize=13, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=10)

for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}%", 
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', xytext=(0, 2),
                 textcoords='offset points')

plt.tight_layout()
p3_path = os.path.join(plot_dir, "polar_vs_nonpolar_anomaly_comparison.png")
plt.savefig(p3_path, dpi=300)
plt.close()
print(f"Saved Plot 3 -> {p3_path}")

print("\n" + "=" * 75)
print("  ALL PARITY PLOTS & POLARITY ANALYSES COMPLETED SUCCESSFULLY!")
print("=" * 75)
