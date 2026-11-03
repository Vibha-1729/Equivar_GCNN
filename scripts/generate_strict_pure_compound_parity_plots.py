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

def generate_strict_pure_compound_dataset_and_plot(
    json_gz_path,
    out_csv_gz_path,
    out_summary_csv_path,
    out_png_path,
    dataset_name,
    parent_bandgaps_json=None,
    uid_to_agm_json=None,
    filter_parent_eg_positive=False
):
    print("=" * 80)
    print(f"STRICT PURE-INTEGER COMPOUND FILTERING: {dataset_name}")
    print(f"Input: {json_gz_path}")
    print("Rule: Discard the ENTIRE compound if ANY atom has a fractional or zero nominal charge!")
    if filter_parent_eg_positive:
        print("Filter: Only verified semiconductors (Alexandria Parent Eg > 0)")
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

    atom_records = []
    material_summaries = []

    total_scanned = 0
    metallic_skipped = 0
    zero_fallback_rejected = 0
    fractional_rejected_compounds = 0
    retained_compounds = 0

    for m in tqdm(data, desc=f"Processing {dataset_name}"):
        total_scanned += 1
        mat_id = str(m.get("material_id", m.get("id", f"mat_{total_scanned}")))
        formula = m.get("formula", "")
        cs = str(m.get("crystal_system", "Unknown")).capitalize()
        spg = m.get("space_group_number", m.get("space_group", None))

        eg = m.get("band_gap", m.get("bandgap", m.get("band_gap_ind", None)))
        if filter_parent_eg_positive:
            agm_id = uid_map.get(mat_id)
            if not agm_id:
                metallic_skipped += 1
                continue
            eg = parent_bgs.get(agm_id, 0.0)
            if eg <= 0.0:
                metallic_skipped += 1
                continue

        oxi_method = m.get("nominal_oxidation_state_method", "")
        if oxi_method == "Zero_Fallback":
            zero_fallback_rejected += 1
            continue

        atoms = m.get("predicted_atoms_bec", [])
        if not atoms:
            continue

        # Strict test: does ANY atom have 0 or fractional?
        compound_is_pure = True
        temp_atoms = []
        num_anomalous = 0
        max_delta_z = 0.0

        for a in atoms:
            nom = a.get("nominal_oxidation_state")
            if nom is None:
                compound_is_pure = False
                break
            nom = float(nom)
            if nom == 0.0 or abs(round(nom) - nom) > 1e-4:
                compound_is_pure = False
                break

            nom_int = int(round(nom))

            bec_val = a.get("Z_trace_mean")
            if bec_val is None:
                mat3x3 = a.get("bec_tensor_3x3")
                if mat3x3:
                    bec_val = (mat3x3[0][0] + mat3x3[1][1] + mat3x3[2][2]) / 3.0
            if bec_val is None:
                compound_is_pure = False
                break
            bec_val = float(bec_val)

            delta_z = abs(bec_val - nom_int)
            is_anomaly = delta_z >= 2.0
            if is_anomaly:
                num_anomalous += 1
            if delta_z > max_delta_z:
                max_delta_z = delta_z

            sym = a.get("element", a.get("symbol", ""))

            temp_atoms.append({
                "material_id": mat_id,
                "formula": formula,
                "crystal_system": cs,
                "element": sym,
                "nominal_charge": nom_int,
                "predicted_bec": round(bec_val, 4),
                "delta_z": round(delta_z, 4),
                "is_giant_anomaly": is_anomaly
            })

        if not compound_is_pure:
            fractional_rejected_compounds += 1
            continue

        retained_compounds += 1
        atom_records.extend(temp_atoms)

        material_summaries.append({
            "material_id": mat_id,
            "formula": formula,
            "crystal_system": cs,
            "space_group": spg,
            "band_gap_eV": eg,
            "nominal_method": oxi_method,
            "num_atoms": len(temp_atoms),
            "num_giant_anomalous_atoms": num_anomalous,
            "max_delta_z": round(max_delta_z, 4),
            "is_giant_anomalous_material": num_anomalous > 0
        })

    print(f"\n--- Results for {dataset_name} ---")
    print(f"Total Scanned: {total_scanned:,}")
    if filter_parent_eg_positive:
        print(f"Metallic Filtered (Eg <= 0): {metallic_skipped:,}")
    print(f"Zero_Fallback Rejected Compounds: {zero_fallback_rejected:,}")
    print(f"Fractional/Zero Rejected Compounds: {fractional_rejected_compounds:,}")
    print(f"Strict Pure-Integer Compounds Retained: {retained_compounds:,}")
    print(f"Total Retained Atoms: {len(atom_records):,}")

    # 1. Save Clean Atoms CSV.GZ
    df_atoms = pd.DataFrame(atom_records)
    os.makedirs(os.path.dirname(out_csv_gz_path), exist_ok=True)
    df_atoms.to_csv(out_csv_gz_path, index=False, compression="gzip")
    print(f"Saved strict atom data -> {out_csv_gz_path}")

    # 2. Save Clean Materials Summary CSV
    df_materials = pd.DataFrame(material_summaries)
    os.makedirs(os.path.dirname(out_summary_csv_path), exist_ok=True)
    df_materials.to_csv(out_summary_csv_path, index=False)
    print(f"Saved strict materials summary -> {out_summary_csv_path}")

    # 3. Create Parity Plot
    print(f"Generating Strict Parity Plot -> {out_png_path}")
    fig, ax = plt.subplots(figsize=(10, 8.5))

    min_x, max_x = df_atoms["nominal_charge"].min() - 0.5, df_atoms["nominal_charge"].max() + 0.5
    min_y, max_y = df_atoms["predicted_bec"].min() - 0.5, df_atoms["predicted_bec"].max() + 0.5
    plot_min = min(min_x, min_y)
    plot_max = max(max_x, max_y)

    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, alpha=0.7, label='Ideal Parity ($Z^* = q_{\\mathrm{nominal}}$)')

    x_fill = np.linspace(plot_min, plot_max, 200)
    ax.fill_between(x_fill, x_fill + 2.0, plot_max + 2, color='red', alpha=0.07, label='Giant Anomaly ($|\\Delta Z^*| \\geq 2.0e$)')
    ax.fill_between(x_fill, plot_min - 2, x_fill - 2.0, color='red', alpha=0.07)

    for cs in crystal_systems:
        sub = df_atoms[df_atoms["crystal_system"] == cs]
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

    int_ticks = sorted(df_atoms["nominal_charge"].unique())
    ax.set_xticks(int_ticks)
    ax.set_xlim(min(int_ticks) - 0.6, max(int_ticks) + 0.6)
    ax.set_ylim(-10, 14)

    ax.set_xlabel("Nominal Oxidation State $q_{\\mathrm{nominal}}$ ($e$, 100% Integer Compounds)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Predicted Born Effective Charge Trace $Z^*_{\\mathrm{mean}}$ ($e$)", fontsize=13, fontweight='bold')

    clean_title = f"Strict Integer Parity Plot: Nominal Charge vs Predicted BEC ({dataset_name})\n"
    clean_title += f"All Compounds with Any Fractional/Zero Charges Excluded | {len(df_atoms):,} Atoms across {retained_compounds:,} Materials"
    ax.set_title(clean_title, fontsize=12, pad=12, fontweight='bold')

    leg = ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9.5)
    leg.set_title("Crystal Systems", prop={'weight':'bold', 'size': 10})
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    print(f"Saved strict pure-compound plot successfully -> {out_png_path}\n")

if __name__ == "__main__":
    # 1. GNoME
    generate_strict_pure_compound_dataset_and_plot(
        json_gz_path="/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json.gz",
        out_csv_gz_path="/home/vibhan23/gnome/gnome_strict_pure_integer_atoms_data.csv.gz",
        out_summary_csv_path="/home/vibhan23/gnome/gnome_strict_pure_integer_compounds_summary.csv",
        out_png_path="/home/vibhan23/gnome/plots/gnome_strict_pure_compound_parity_plot.png",
        dataset_name="GNoME Stable Semiconductors"
    )

    # 2. Alexandria
    generate_strict_pure_compound_dataset_and_plot(
        json_gz_path="/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json.gz",
        out_csv_gz_path="/home/vibhan23/alexandria/alexandria_strict_pure_integer_atoms_data.csv.gz",
        out_summary_csv_path="/home/vibhan23/alexandria/alexandria_strict_pure_integer_compounds_summary.csv",
        out_png_path="/home/vibhan23/alexandria/plots/alexandria_strict_pure_compound_parity_plot.png",
        dataset_name="Alexandria Semiconductors"
    )

    # 3. OMat24
    generate_strict_pure_compound_dataset_and_plot(
        json_gz_path="/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json.gz",
        out_csv_gz_path="/home/vibhan23/omat24/omat24_strict_pure_integer_atoms_data.csv.gz",
        out_summary_csv_path="/home/vibhan23/omat24/omat24_strict_pure_integer_compounds_summary.csv",
        out_png_path="/home/vibhan23/omat24/plots/omat24_strict_pure_compound_parity_plot.png",
        dataset_name="OMat24 Verified Semiconductors (Eg > 0)",
        parent_bandgaps_json="/home/vibhan23/omat24/omat24_parent_bandgaps.json",
        uid_to_agm_json="/home/vibhan23/omat24/omat24_uid_to_agm_parent.json",
        filter_parent_eg_positive=True
    )
