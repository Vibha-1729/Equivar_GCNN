#!/usr/bin/env python3
"""
Populate missing band_gap_eV column in Alexandria and GNoME strict pure compound summary CSVs
by extracting bandgap_eV directly from the master JSON.GZ archives.
"""

import gzip
import os
import time
import pandas as pd


def extract_bandgaps_from_json_gz(json_gz_path):
    print(f"Streaming bandgaps from {json_gz_path}...")
    t0 = time.time()
    id_to_bg = {}
    current_id = None

    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith('"material_id":'):
                current_id = line_s.split(":")[1].strip().strip(",").strip('"')
            elif line_s.startswith('"bandgap_eV":') and current_id:
                try:
                    val = float(line_s.split(":")[1].strip().strip(","))
                    id_to_bg[current_id] = round(val, 4)
                except ValueError:
                    pass
                current_id = None

    elapsed = time.time() - t0
    print(f"Extracted {len(id_to_bg):,} material bandgaps in {elapsed:.2f}s.")
    return id_to_bg


def update_summary_csv(csv_path, id_to_bg, dataset_name):
    print(f"\nUpdating {dataset_name} summary CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist!")
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows from {csv_path}.")

    # Map material_id to bandgap
    df['band_gap_eV'] = df['material_id'].astype(str).map(id_to_bg)

    matched = df['band_gap_eV'].notna().sum()
    print(f"Matched {matched:,} / {total_rows:,} ({matched/total_rows*100:.2f}%) with valid DFT bandgaps.")

    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved updated CSV to {csv_path}!")

    # Show first 5 rows
    print("\nPreview:")
    print(df[['material_id', 'formula', 'crystal_system', 'space_group', 'band_gap_eV', 'is_giant_anomalous_material']].head())


if __name__ == "__main__":
    # 1. Alexandria
    alex_json = "/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json.gz"
    alex_csv = "/home/vibhan23/alexandria/alexandria_strict_pure_integer_compounds_summary.csv"
    alex_bgs = extract_bandgaps_from_json_gz(alex_json)
    update_summary_csv(alex_csv, alex_bgs, "Alexandria")

    # 2. GNoME
    gnome_json = "/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json.gz"
    gnome_csv = "/home/vibhan23/gnome/gnome_strict_pure_integer_compounds_summary.csv"
    gnome_bgs = extract_bandgaps_from_json_gz(gnome_json)
    update_summary_csv(gnome_csv, gnome_bgs, "GNoME")

    print("\nAll datasets updated successfully!")
