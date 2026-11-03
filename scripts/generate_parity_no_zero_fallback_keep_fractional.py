import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def generate_parity_plot_keep_fractional(
    json_gz_path,
    out_png_path,
    dataset_name,
    parent_bandgaps_json=None,
    uid_to_agm_json=None,
    filter_parent_eg_positive=False
):
    print("=" * 80)
    print(f"GENERATING PARITY PLOT (EXCLUDE ZERO_FALLBACK ONLY, KEEP FRACTIONAL): {dataset_name}")
    print(f"Input: {json_gz_path}")
    print("Filters:")
    print("  - Excluded: Zero_Fallback & q_nominal == 0")
    print("  - RETAINED: Fractional oxidation states (Composition Guesses included)")
    if filter_parent_eg_positive:
        print("  - Filter: Alexandria Parent Eg > 0 (Semiconductors only)")
    print("=" * 80)

    parent_bgs = {}
    if parent_bandgaps_json and os.path.exists(parent_bandgaps_json):
        with open(parent_bandgaps_json, "r", encoding="utf-8") as f:
            parent_bgs = json.load(f)

    uid_map = {}
    if uid_to_agm_json and os.path.exists(uid_to_agm_json):
        with open(uid_to_agm_json, "r", encoding="utf-8") as f:
            uid_map = json.load(f)

    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} material records.")

    nom_charges = []
    bec_traces = []
    c_systems = []

    total_materials = 0
    retained_materials = 0
    total_atoms_scanned = 0
    total_zero_fallback_atoms = 0
    fractional_atoms_retained = 0

    for entry in tqdm(data, desc=f"Processing {dataset_name}"):
        total_materials += 1
        mat_id = str(entry.get("material_id", entry.get("id", "")))
        cs = str(entry.get("crystal_system", "Unknown")).capitalize()
        if cs not in crystal_systems:
            continue

        if filter_parent_eg_positive:
            agm_id = uid_map.get(mat_id)
            if not agm_id:
                continue
            eg = parent_bgs.get(agm_id, 0.0)
            if eg <= 0.0:
                continue

        oxi_method = entry.get("nominal_oxidation_state_method", "")
        if oxi_method == "Zero_Fallback":
            total_zero_fallback_atoms += len(entry.get("predicted_atoms_bec", []))
            continue

        atoms = entry.get("predicted_atoms_bec", [])
        mat_has_atom = False

        for atom in atoms:
            total_atoms_scanned += 1
            nom = atom.get("nominal_oxidation_state")
            if nom is None:
                continue
            nom = float(nom)

            # Exclude ONLY zero fallback / 0.0
            if nom == 0.0:
                total_zero_fallback_atoms += 1
                continue

            if abs(round(nom) - nom) > 1e-4:
                fractional_atoms_retained += 1

            # BEC trace mean
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
            mat_has_atom = True

        if mat_has_atom:
            retained_materials += 1

    df_plot = pd.DataFrame({
        "nominal_charge": nom_charges,
        "predicted_bec": bec_traces,
        "crystal_system": c_systems
    })

    print(f"\n--- Statistics for {dataset_name} ---")
    print(f"Total Materials: {total_materials:,} | Retained: {retained_materials:,}")
    print(f"Total Atoms Scanned: {total_atoms_scanned:,}")
    print(f"Zero_Fallback Atoms Excluded: {total_zero_fallback_atoms:,}")
    print(f"Fractional Atoms Retained: {fractional_atoms_retained:,}")
    print(f"Final Atoms Plotted: {len(df_plot):,}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8.5))

    min_x, max_x = df_plot["nominal_charge"].min() - 0.5, df_plot["nominal_charge"].max() + 0.5
    min_y, max_y = df_plot["predicted_bec"].min() - 0.5, df_plot["predicted_bec"].max() + 0.5
    plot_min = min(min_x, min_y)
    plot_max = max(max_x, max_y)

    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, alpha=0.7, label='Ideal Parity ($Z^* = q_{\\mathrm{nominal}}$)')

    x_fill = np.linspace(plot_min, plot_max, 200)
    ax.fill_between(x_fill, x_fill + 2.0, plot_max + 2, color='red', alpha=0.07, label='Giant Anomaly ($|\\Delta Z^*| \\geq 2.0e$)')
    ax.fill_between(x_fill, plot_min - 2, x_fill - 2.0, color='red', alpha=0.07)

    for cs in crystal_systems:
        sub = df_plot[df_plot["crystal_system"] == cs]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["nominal_charge"],
            sub["predicted_bec"],
            color=palette[cs],
            alpha=0.30,
            s=8,
            rasterized=True,
            label=f"{cs} (N = {len(sub):,} atoms)"
        )

    ax.set_xlim(-6.5, 8.5)
    ax.set_ylim(-10, 14)
    ax.set_xlabel("Nominal Oxidation State $q_{\\mathrm{nominal}}$ ($e$, Includes Fractional Valences)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Predicted Born Effective Charge Trace $Z^*_{\\mathrm{mean}}$ ($e$)", fontsize=13, fontweight='bold')

    title_str = f"Parity Plot: Nominal Charge vs Predicted BEC ({dataset_name})\n"
    title_str += f"Zero_Fallback Excluded | Fractional Charges Retained | {len(df_plot):,} Atoms across {retained_materials:,} Materials"
    ax.set_title(title_str, fontsize=12, pad=12, fontweight='bold')

    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9.5)
    leg.set_title("Crystal Systems", prop={'weight':'bold', 'size': 10})
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    print(f"Saved parity plot successfully -> {out_png_path}\n")

if __name__ == "__main__":
    # 1. GNoME
    generate_parity_plot_keep_fractional(
        json_gz_path="/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json.gz",
        out_png_path="/home/vibhan23/gnome/plots/gnome_parity_plot_no_zero_fallback_with_fractional.png",
        dataset_name="GNoME Stable Semiconductors"
    )

    # 2. Alexandria
    generate_parity_plot_keep_fractional(
        json_gz_path="/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json.gz",
        out_png_path="/home/vibhan23/alexandria/plots/alexandria_parity_plot_no_zero_fallback_with_fractional.png",
        dataset_name="Alexandria Semiconductors"
    )

    # 3. OMat24
    generate_parity_plot_keep_fractional(
        json_gz_path="/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json.gz",
        out_png_path="/home/vibhan23/omat24/plots/omat24_parity_plot_no_zero_fallback_with_fractional.png",
        dataset_name="OMat24 Verified Semiconductors (Eg > 0)",
        parent_bandgaps_json="/home/vibhan23/omat24/omat24_parent_bandgaps.json",
        uid_to_agm_json="/home/vibhan23/omat24/omat24_uid_to_agm_parent.json",
        filter_parent_eg_positive=True
    )
