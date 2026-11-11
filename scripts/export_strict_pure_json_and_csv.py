#!/usr/bin/env python3
"""
Export strict pure-integer compound BEC prediction JSON files (both .json and .json.gz)
and add an explicit `atom_nominal_oxidation_states` column to the summary CSVs for
GNoME, Alexandria, and OMat24 (verified semiconductors Eg > 0).

Filtering rules (matching strict pure compound parity plots):
1. For OMat24, require Alexandria parent bandgap Eg > 0 (and inject parent_alexandria_id + bandgap_eV).
2. Exclude Zero_Fallback materials.
3. Exclude the entire compound if ANY atom has a zero (0.0) or fractional nominal oxidation state.
4. Store integer nominal_oxidation_state for every atom in `predicted_atoms_bec` and add
   `atom_nominal_oxidation_states` to both the JSON records and the summary CSVs.
"""

import gzip
import json
import os
import time
import pandas as pd


def stream_export_strict_pure(
    in_json_path,
    out_json_path,
    out_json_gz_path,
    summary_csv_path,
    dataset_name,
    parent_bandgaps_json=None,
    uid_to_agm_json=None,
    filter_parent_eg_positive=False,
):
    print("=" * 80, flush=True)
    print(f"Exporting strict pure-integer JSON & CSV column for: {dataset_name}", flush=True)
    print(f"Input JSON: {in_json_path}", flush=True)
    t0 = time.time()

    parent_bgs = {}
    uid_map = {}
    if filter_parent_eg_positive:
        with open(parent_bandgaps_json, "r", encoding="utf-8") as f:
            parent_bgs = json.load(f)
        with open(uid_to_agm_json, "r", encoding="utf-8") as f:
            uid_map = json.load(f)

    total_scanned = 0
    retained = 0
    mat_to_atom_oxis = {}

    with open(in_json_path, "r", encoding="utf-8") as fin, \
         open(out_json_path, "w", encoding="utf-8") as fout_json, \
         gzip.open(out_json_gz_path, "wt", encoding="utf-8") as fout_gz:

        fout_json.write("[\n")
        fout_gz.write("[\n")
        first = True

        buf = []
        in_obj = False
        for line in fin:
            if line == "{\n":
                in_obj = True
                buf = [line]
            elif in_obj:
                buf.append(line)
                if line.startswith("}"):
                    total_scanned += 1
                    obj_str = "".join(buf).rstrip().rstrip(",")
                    m = json.loads(obj_str)
                    in_obj = False
                    buf = []

                    mat_id = str(m.get("material_id", m.get("id", "")))

                    if filter_parent_eg_positive:
                        agm_id = uid_map.get(mat_id)
                        if not agm_id:
                            continue
                        eg = parent_bgs.get(agm_id, 0.0)
                        if eg <= 0.0:
                            continue
                        m["parent_alexandria_id"] = agm_id
                        m["bandgap_eV"] = round(float(eg), 4)

                    if m.get("nominal_oxidation_state_method", "") == "Zero_Fallback":
                        continue

                    atoms = m.get("predicted_atoms_bec", [])
                    if not atoms:
                        continue

                    compound_is_pure = True
                    atom_oxi_labels = []
                    for a in atoms:
                        nom = a.get("nominal_oxidation_state")
                        if nom is None:
                            compound_is_pure = False
                            break
                        nom_f = float(nom)
                        if nom_f == 0.0 or abs(round(nom_f) - nom_f) > 1e-4:
                            compound_is_pure = False
                            break

                        bec_val = a.get("Z_trace_mean")
                        if bec_val is None:
                            mat3x3 = a.get("bec_tensor_3x3")
                            if mat3x3:
                                bec_val = (mat3x3[0][0] + mat3x3[1][1] + mat3x3[2][2]) / 3.0
                        if bec_val is None:
                            compound_is_pure = False
                            break

                        nom_int = int(round(nom_f))
                        a["nominal_oxidation_state"] = nom_int
                        sym = a.get("element", a.get("symbol", ""))
                        atom_oxi_labels.append(f"{sym}:{nom_int:+d}")

                    if not compound_is_pure:
                        continue

                    oxi_summary_str = "; ".join(atom_oxi_labels)
                    m["atom_nominal_oxidation_states"] = oxi_summary_str
                    mat_to_atom_oxis[mat_id] = oxi_summary_str

                    dumped = json.dumps(m, indent=2)
                    if not first:
                        fout_json.write(",\n")
                        fout_gz.write(",\n")
                    fout_json.write(dumped)
                    fout_gz.write(dumped)
                    first = False
                    retained += 1

                    if retained % 25000 == 0:
                        print(f"  ...retained {retained:,} strict pure compounds (scanned {total_scanned:,})", flush=True)

        fout_json.write("\n]\n")
        fout_gz.write("\n]\n")

    print(f"Finished JSON export for {dataset_name} in {time.time() - t0:.1f}s:", flush=True)
    print(f"  Scanned: {total_scanned:,} | Retained strict pure: {retained:,}", flush=True)
    print(f"  JSON:    {out_json_path}", flush=True)
    print(f"  JSON.GZ: {out_json_gz_path}", flush=True)

    # Update summary CSV with atom_nominal_oxidation_states column
    if os.path.exists(summary_csv_path):
        df = pd.read_csv(summary_csv_path)
        df["atom_nominal_oxidation_states"] = df["material_id"].astype(str).map(mat_to_atom_oxis)
        matched = df["atom_nominal_oxidation_states"].notna().sum()
        # Put atom_nominal_oxidation_states right after nominal_method if present
        cols = list(df.columns)
        if "nominal_method" in cols and "atom_nominal_oxidation_states" in cols:
            cols.remove("atom_nominal_oxidation_states")
            idx = cols.index("nominal_method") + 1
            cols.insert(idx, "atom_nominal_oxidation_states")
            df = df[cols]
        df.to_csv(summary_csv_path, index=False)
        print(f"  Updated CSV ({matched:,}/{len(df):,} matched): {summary_csv_path}", flush=True)
        print(df[["material_id", "formula", "band_gap_eV", "nominal_method", "atom_nominal_oxidation_states"]].head(3), flush=True)


if __name__ == "__main__":
    # 1. OMat24 (fastest first for quick verification)
    stream_export_strict_pure(
        in_json_path="/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json",
        out_json_path="/home/vibhan23/omat24/omat24_strict_pure_integer_full_3x3_bec.json",
        out_json_gz_path="/home/vibhan23/omat24/omat24_strict_pure_integer_full_3x3_bec.json.gz",
        summary_csv_path="/home/vibhan23/omat24/omat24_strict_pure_integer_compounds_summary.csv",
        dataset_name="OMat24 Verified Semiconductors (Eg > 0)",
        parent_bandgaps_json="/home/vibhan23/omat24/omat24_parent_bandgaps.json",
        uid_to_agm_json="/home/vibhan23/omat24/omat24_uid_to_agm_parent.json",
        filter_parent_eg_positive=True,
    )

    # 2. Alexandria
    stream_export_strict_pure(
        in_json_path="/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json",
        out_json_path="/home/vibhan23/alexandria/alexandria_strict_pure_integer_full_3x3_bec.json",
        out_json_gz_path="/home/vibhan23/alexandria/alexandria_strict_pure_integer_full_3x3_bec.json.gz",
        summary_csv_path="/home/vibhan23/alexandria/alexandria_strict_pure_integer_compounds_summary.csv",
        dataset_name="Alexandria Semiconductors",
    )

    # 3. GNoME
    stream_export_strict_pure(
        in_json_path="/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json",
        out_json_path="/home/vibhan23/gnome/gnome_strict_pure_integer_full_3x3_bec.json",
        out_json_gz_path="/home/vibhan23/gnome/gnome_strict_pure_integer_full_3x3_bec.json.gz",
        summary_csv_path="/home/vibhan23/gnome/gnome_strict_pure_integer_compounds_summary.csv",
        dataset_name="GNoME Stable Semiconductors",
    )

    print("\nALL STRICT PURE JSON + CSV EXPORTS COMPLETED!", flush=True)
