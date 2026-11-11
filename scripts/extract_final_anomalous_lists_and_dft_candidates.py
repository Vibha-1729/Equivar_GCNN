#!/usr/bin/env python3
"""
1. Extract final strict pure-integer giant anomalous compound lists for GNoME and Alexandria.
2. Select 10 ideal fast-DFT validation candidates (small unit cell, wide bandgap Eg >= 1.5 eV,
   no f-elements, BVAnalyzer verified, giant predicted BEC anomaly).
"""

import re
import pandas as pd

# Elements to exclude for clean standard DFPT (noble gases, lanthanides 57-71, actinides >=89, Tc/Pm)
EXCLUDE_ELEMENTS = {
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Tc", "Po", "At", "Fr", "Ra"
}


def has_excluded_element(atom_oxi_str):
    if not isinstance(atom_oxi_str, str):
        return True
    elems = re.findall(r"([A-Z][a-z]?):", atom_oxi_str)
    return any(el in EXCLUDE_ELEMENTS for el in elems) or len(elems) == 0


def process_dataset(in_csv, out_csv, ds_name):
    df = pd.read_csv(in_csv)
    df_anom = df[df["is_giant_anomalous_material"] == True].copy()
    df_anom = df_anom.sort_values(by=["max_delta_z", "num_giant_anomalous_atoms"], ascending=[False, False]).reset_index(drop=True)
    df_anom.to_csv(out_csv, index=False)
    print(f"[{ds_name}] Total strict pure: {len(df):,} | Giant anomalous (|Delta Z*| >= 2.0e): {len(df_anom):,} ({len(df_anom)/len(df)*100:.2f}%) -> {out_csv}")
    return df_anom


def main():
    df_gnome_anom = process_dataset(
        "gnome_discovery/gnome_strict_pure_integer_compounds_summary.csv",
        "gnome_discovery/gnome_strict_pure_giant_anomalous_compounds.csv",
        "GNoME"
    )
    df_alex_anom = process_dataset(
        "alexandria/alexandria_strict_pure_integer_compounds_summary.csv",
        "alexandria/alexandria_strict_pure_giant_anomalous_compounds.csv",
        "Alexandria"
    )

    # Pick 5 clean candidates from Alexandria + 5 from GNoME (diverse, small cell, wide bandgap)
    df_alex_anom["source_dataset"] = "Alexandria"
    df_gnome_anom["source_dataset"] = "GNoME"

    candidates = []
    for ds_name, df_src in [("Alexandria", df_alex_anom), ("GNoME", df_gnome_anom)]:
        mask = (
            (df_src["nominal_method"] == "BVAnalyzer") &
            (df_src["num_atoms"] <= 8) &
            (df_src["band_gap_eV"] >= 1.5) &
            (df_src["max_delta_z"] >= 2.2) &
            (df_src["max_delta_z"] <= 8.0) &
            (~df_src["atom_nominal_oxidation_states"].apply(has_excluded_element))
        )
        pool = df_src[mask].copy()
        # Sort by smallest num_atoms first, then highest max_delta_z, pick unique formulas
        pool = pool.sort_values(by=["num_atoms", "max_delta_z"], ascending=[True, False]).drop_duplicates(subset=["formula"])
        print(f"\n[{ds_name}] Clean fast-DFT candidate pool (natoms<=8, Eg>=1.5eV, no f-block): {len(pool):,} materials")
        picked = pool.head(5)
        candidates.append(picked)

    df_10 = pd.concat(candidates, ignore_index=True)
    df_10.to_csv("alexandria_gnome_merged/dft_validation_10_candidates.csv", index=False)
    print("\n=== SELECTED 10 IDEAL DFT VALIDATION CANDIDATES ===")
    cols = ["source_dataset", "material_id", "formula", "crystal_system", "space_group", "band_gap_eV", "num_atoms", "max_delta_z", "atom_nominal_oxidation_states"]
    print(df_10[cols].to_string(index=False))


if __name__ == "__main__":
    main()
