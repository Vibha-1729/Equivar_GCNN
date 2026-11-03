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

def process_and_plot_clean_dataset(
    json_gz_path,
    out_csv_gz_path,
    out_summary_csv_path,
    out_png_path,
    dataset_name,
    parent_bandgaps_json=None,
    uid_to_agm_json=None,
    filter_parent_eg_positive=False,
    skip_if_exists=True
):
    if skip_if_exists and os.path.exists(out_png_path) and os.path.exists(out_csv_gz_path):
        print(f"Skipping {dataset_name} because output already exists: {out_png_path}")
        return

    print("=" * 80)
    print(f"PROCESSING CLEAN DATASET & PARITY PLOT: {dataset_name}")
    print(f"Input: {json_gz_path}")
    print("Filters applied:")
    print("  1. Exclude Zero_Fallback & q_nominal == 0")
    print("  2. Exclude fractional nominal charges (strictly integer valences only)")
    if filter_parent_eg_positive:
        print("  3. Filter parent bandgap Eg > 0 (Verified Semiconductors only via Alexandria Mapping)")
    print("=" * 80)

    parent_bgs = {}
    if parent_bandgaps_json and os.path.exists(parent_bandgaps_json):
        with open(parent_bandgaps_json, "r", encoding="utf-8") as f:
            parent_bgs = json.load(f)
        print(f"Loaded {len(parent_bgs):,} parent bandgap records.")

    uid_map = {}
    if uid_to_agm_json and os.path.exists(uid_to_agm_json):
        with open(uid_to_agm_json, "r", encoding="utf-8") as f:
            uid_map = json.load(f)
        print(f"Loaded {len(uid_map):,} UID-to-AGM records.")

    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} material records from {os.path.basename(json_gz_path)}.")

    atom_records = []
    material_summaries = []

    total_materials_scanned = 0
    total_materials_retained = 0
    total_atoms_scanned = 0
    total_zero_fallback_atoms = 0
    total_fractional_atoms = 0
    total_metallic_omats_skipped = 0

    for entry in tqdm(data, desc=f"Processing {dataset_name}"):
        total_materials_scanned += 1
        mat_id = str(entry.get("material_id", entry.get("id", f"mat_{total_materials_scanned}")))
        formula = entry.get("formula", "")
        cs = str(entry.get("crystal_system", "Unknown")).capitalize()
        spg = entry.get("space_group_number", entry.get("space_group", None))
        
        # Bandgap determination
        eg = entry.get("band_gap", entry.get("bandgap", entry.get("band_gap_ind", None)))
        if filter_parent_eg_positive:
            agm_id = uid_map.get(mat_id)
            if not agm_id:
                total_metallic_omats_skipped += 1
                continue
            eg = parent_bgs.get(agm_id, 0.0)
            if eg <= 0.0:
                total_metallic_omats_skipped += 1
                continue

        oxi_method = entry.get("nominal_oxidation_state_method", "")
        if oxi_method == "Zero_Fallback":
            total_zero_fallback_atoms += len(entry.get("predicted_atoms_bec", []))
            continue

        atoms = entry.get("predicted_atoms_bec", [])
        clean_material_atoms = []
        num_anomalous_in_mat = 0
        max_delta_z = 0.0

        for atom in atoms:
            total_atoms_scanned += 1
            nom = atom.get("nominal_oxidation_state")
            if nom is None:
                continue
            nom = float(nom)

            if nom == 0.0:
                total_zero_fallback_atoms += 1
                continue

            # Check integer strictly:
            if abs(round(nom) - nom) > 1e-4:
                total_fractional_atoms += 1
                continue

            nom_int = int(round(nom))

            # BEC trace mean
            bec_val = atom.get("Z_trace_mean")
            if bec_val is None:
                mat3x3 = atom.get("bec_tensor_3x3")
                if mat3x3:
                    bec_val = (mat3x3[0][0] + mat3x3[1][1] + mat3x3[2][2]) / 3.0
            if bec_val is None:
                continue
            bec_val = float(bec_val)

            delta_z = abs(bec_val - nom_int)
            is_anomaly = delta_z >= 2.0
            if is_anomaly:
                num_anomalous_in_mat += 1
            if delta_z > max_delta_z:
                max_delta_z = delta_z

            sym = atom.get("element", atom.get("symbol", ""))

            rec = {
                "material_id": mat_id,
                "formula": formula,
                "crystal_system": cs,
                "element": sym,
                "nominal_charge": nom_int,
                "predicted_bec": round(bec_val, 4),
                "delta_z": round(delta_z, 4),
                "is_giant_anomaly": is_anomaly
            }
            clean_material_atoms.append(rec)
            atom_records.append(rec)

        if clean_material_atoms:
            total_materials_retained += 1
            material_summaries.append({
                "material_id": mat_id,
                "formula": formula,
                "crystal_system": cs,
                "space_group": spg,
                "band_gap_eV": eg,
                "nominal_method": oxi_method,
                "clean_atoms_count": len(clean_material_atoms),
                "num_giant_anomalous_atoms": num_anomalous_in_mat,
                "max_delta_z": round(max_delta_z, 4),
                "is_giant_anomalous_material": num_anomalous_in_mat > 0
            })

    print(f"\n--- Statistics for {dataset_name} ---")
    print(f"Total Materials Scanned: {total_materials_scanned:,}")
    if filter_parent_eg_positive:
        print(f"Metallic Materials Filtered (Eg <= 0): {total_metallic_omats_skipped:,}")
    print(f"Materials Retained in Clean Subset: {total_materials_retained:,}")
    print(f"Total Atoms Scanned: {total_atoms_scanned:,}")
    print(f"Zero_Fallback Atoms Excluded: {total_zero_fallback_atoms:,}")
    print(f"Fractional Nominal Charge Atoms Excluded: {total_fractional_atoms:,}")
    print(f"Clean Integer-Valence Atoms Retained: {len(atom_records):,}")

    if len(atom_records) == 0:
        print(f"Warning: No clean atoms retained for {dataset_name}!")
        return

    # 1. Save Clean Atoms CSV.GZ
    df_atoms = pd.DataFrame(atom_records)
    os.makedirs(os.path.dirname(out_csv_gz_path), exist_ok=True)
    df_atoms.to_csv(out_csv_gz_path, index=False, compression="gzip")
    print(f"Saved clean atom-level dataset -> {out_csv_gz_path}")

    # 2. Save Clean Materials Summary CSV
    df_materials = pd.DataFrame(material_summaries)
    os.makedirs(os.path.dirname(out_summary_csv_path), exist_ok=True)
    df_materials.to_csv(out_summary_csv_path, index=False)
    print(f"Saved clean materials summary -> {out_summary_csv_path}")

    # 3. Create Parity Plot
    print(f"Generating Parity Plot -> {out_png_path}")
    fig, ax = plt.subplots(figsize=(10, 8.5))

    min_x, max_x = df_atoms["nominal_charge"].min() - 0.5, df_atoms["nominal_charge"].max() + 0.5
    min_y, max_y = df_atoms["predicted_bec"].min() - 0.5, df_atoms["predicted_bec"].max() + 0.5
    plot_min = min(min_x, min_y)
    plot_max = max(max_x, max_y)

    # Parity line y = x
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, alpha=0.7, label='Ideal Parity ($Z^* = q_{\\mathrm{nominal}}$)')

    # Shaded anomalous region
    x_fill = np.linspace(plot_min, plot_max, 200)
    ax.fill_between(x_fill, x_fill + 2.0, plot_max + 2, color='red', alpha=0.07, label='Giant Anomaly ($|\\Delta Z^*| \\geq 2.0e$)')
    ax.fill_between(x_fill, plot_min - 2, x_fill - 2.0, color='red', alpha=0.07)

    # Plot crystal systems
    for cs in crystal_systems:
        sub = df_atoms[df_atoms["crystal_system"] == cs]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["nominal_charge"],
            sub["predicted_bec"],
            color=palette[cs],
            alpha=0.35,
            s=10,
            rasterized=True,
            label=f"{cs} (N = {len(sub):,} atoms)"
        )

    # Discrete integer ticks on x-axis
    int_ticks = sorted(df_atoms["nominal_charge"].unique())
    ax.set_xticks(int_ticks)
    ax.set_xlim(min(int_ticks) - 0.6, max(int_ticks) + 0.6)
    ax.set_ylim(-10, 14)

    ax.set_xlabel("Nominal Oxidation State $q_{\\mathrm{nominal}}$ ($e$, Integers Only)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Predicted Born Effective Charge Trace $Z^*_{\\mathrm{mean}}$ ($e$)", fontsize=13, fontweight='bold')

    clean_title = f"Clean Parity Plot: Nominal Charge vs Predicted BEC ({dataset_name})\n"
    clean_title += f"Zero_Fallback & Fractional Valences Excluded | {len(df_atoms):,} Atoms across {total_materials_retained:,} Materials"
    ax.set_title(clean_title, fontsize=12, pad=12, fontweight='bold')

    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9.5)
    leg.set_title("Crystal Systems", prop={'weight':'bold', 'size': 10})
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    print(f"Saved clean parity plot successfully -> {out_png_path}\n")

if __name__ == "__main__":
    # 1. GNoME
    gnome_gz = "/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json.gz"
    if os.path.exists(gnome_gz):
        process_and_plot_clean_dataset(
            json_gz_path=gnome_gz,
            out_csv_gz_path="/home/vibhan23/gnome/gnome_clean_integer_parity_data.csv.gz",
            out_summary_csv_path="/home/vibhan23/gnome/gnome_clean_materials_summary.csv",
            out_png_path="/home/vibhan23/gnome/plots/gnome_full_database_clean_parity_plot.png",
            dataset_name="GNoME Stable Semiconductors",
            skip_if_exists=True
        )

    # 2. Alexandria
    alex_gz = "/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json.gz"
    if os.path.exists(alex_gz):
        process_and_plot_clean_dataset(
            json_gz_path=alex_gz,
            out_csv_gz_path="/home/vibhan23/alexandria/alexandria_clean_integer_parity_data.csv.gz",
            out_summary_csv_path="/home/vibhan23/alexandria/alexandria_clean_materials_summary.csv",
            out_png_path="/home/vibhan23/alexandria/plots/alexandria_full_database_clean_parity_plot.png",
            dataset_name="Alexandria Semiconductors",
            skip_if_exists=True
        )

    # 3. OMat24 (Filtered for Parent Eg > 0 verified semiconductors)
    omat_gz = "/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json.gz"
    omat_parents = "/home/vibhan23/omat24/omat24_parent_bandgaps.json"
    omat_uid_map = "/home/vibhan23/omat24/omat24_uid_to_agm_parent.json"
    if os.path.exists(omat_gz):
        process_and_plot_clean_dataset(
            json_gz_path=omat_gz,
            out_csv_gz_path="/home/vibhan23/omat24/omat24_clean_integer_parity_data.csv.gz",
            out_summary_csv_path="/home/vibhan23/omat24/omat24_clean_materials_summary.csv",
            out_png_path="/home/vibhan23/omat24/plots/omat24_full_database_clean_parity_plot.png",
            dataset_name="OMat24 Verified Semiconductors (Eg > 0)",
            parent_bandgaps_json=omat_parents,
            uid_to_agm_json=omat_uid_map,
            filter_parent_eg_positive=True,
            skip_if_exists=False
        )
