import os
import glob
import json
import bz2
import zlib
import lmdb
import pandas as pd
import numpy as np

print("=" * 80)
print("1. GNOME DATASET STATISTICAL AUDIT")
print("=" * 80)
gnome_csv = "/home/vibhan23/gnome/stable_materials_summary.csv"
if os.path.exists(gnome_csv):
    df_gnome = pd.read_csv(gnome_csv)
    print(f"Total raw materials in GNoME stable_materials_summary.csv: {len(df_gnome):,}")
    
    decomp = df_gnome["Decomposition Energy Per Atom"]
    print(f"Decomposition Energy range: [{decomp.min():.4f}, {decomp.max():.4f}]")
    print(f"Decomposition Energy <= 0.0: {(decomp <= 0.0).sum():,} ({(decomp <= 0.0).mean()*100:.2f}%)")
    
    bandgap = df_gnome["Bandgap"]
    print(f"Bandgap > 0.0: {(bandgap > 0.0).sum():,} ({(bandgap > 0.0).mean()*100:.2f}%)")
    
    both = (decomp <= 0.0) & (bandgap > 0.0)
    print(f"Decomposition <= 0.0 AND Bandgap > 0.0: {both.sum():,} ({both.mean()*100:.2f}%)")

gnome_screened = "/home/vibhan23/gnome/gnome_stable_semiconductors_bec_predictions_summary.csv"
if os.path.exists(gnome_screened):
    df_screened = pd.read_csv(gnome_screened)
    print(f"Screened GNoME summary CSV rows: {len(df_screened):,}")

print("\n" + "=" * 80)
print("2. ALEXANDRIA DATASET STATISTICAL AUDIT")
print("=" * 80)
alex_dir = "/home/vibhan23/alexandria"
alex_files = sorted(glob.glob(os.path.join(alex_dir, "alexandria_*.json.bz2")))
print(f"Total Alexandria chunk files found: {len(alex_files)}")

# Let's inspect the first 2 chunk files to see exact fields and counts
total_alex_entries_all_chunks = 0
total_hull_le_0 = 0
total_hull_le_005 = 0
total_gap_gt_0 = 0
total_both_0 = 0
total_both_005 = 0

sample_fields = None
for i, fpath in enumerate(alex_files[:3]):
    with bz2.open(fpath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", data) if isinstance(data, dict) else data
    total_alex_entries_all_chunks += len(entries)
    if sample_fields is None and len(entries) > 0:
        sample_fields = {
            "entry_keys": list(entries[0].keys()),
            "data_keys": list(entries[0].get("data", {}).keys()) if isinstance(entries[0].get("data"), dict) else []
        }
    for e in entries:
        edata = e.get("data", {}) if isinstance(e, dict) else {}
        ehull = float(edata.get("e_above_hull", 999.0))
        gap = float(edata.get("band_gap_ind", edata.get("band_gap", 0.0)))
        if ehull <= 0.0:
            total_hull_le_0 += 1
        if ehull <= 0.05:
            total_hull_le_005 += 1
        if gap > 0.0:
            total_gap_gt_0 += 1
        if ehull <= 0.0 and gap > 0.0:
            total_both_0 += 1
        if ehull <= 0.05 and gap > 0.0:
            total_both_005 += 1

print(f"Sample fields from chunk 0:")
print(f"  Entry keys: {sample_fields['entry_keys']}")
print(f"  Data keys: {sample_fields['data_keys']}")
print(f"Across first 3 chunks ({total_alex_entries_all_chunks:,} entries):")
print(f"  e_above_hull <= 0.0: {total_hull_le_0:,}")
print(f"  e_above_hull <= 0.05: {total_hull_le_005:,}")
print(f"  band_gap > 0.0: {total_gap_gt_0:,}")
print(f"  e_above_hull <= 0.0 AND band_gap > 0.0: {total_both_0:,}")
print(f"  e_above_hull <= 0.05 AND band_gap > 0.0: {total_both_005:,}")

alex_summary = "/home/vibhan23/alexandria/alexandria_stable_semiconductors_summary.csv"
if os.path.exists(alex_summary):
    df_alex = pd.read_csv(alex_summary)
    print(f"\nExisting alexandria_stable_semiconductors_summary.csv rows: {len(df_alex):,}")
    print(f"  e_above_hull <= 0.0: {(df_alex['e_above_hull'] <= 0.0).sum():,}")
    print(f"  e_above_hull <= 0.05: {(df_alex['e_above_hull'] <= 0.05).sum():,}")
    print(f"  e_above_hull == 0.0: {(df_alex['e_above_hull'] == 0.0).sum():,}")
    print(f"  bandgap > 0.0: {(df_alex['bandgap_eV'] > 0.0).sum():,}")

print("\n" + "=" * 80)
print("3. OMAT24 DATASET STATISTICAL AUDIT")
print("=" * 80)
omat_lmdb = "/home/vibhan23/omat24/extracted_1M/train/rattled-relax/train.aselmdb"
if os.path.exists(omat_lmdb):
    env = lmdb.open(omat_lmdb, readonly=True, lock=False, subdir=False)
    stat = env.stat()
    print(f"Total entries in OMat24 rattled-relax LMDB: {stat['entries']:,}")
    with env.begin() as txn:
        cursor = txn.cursor()
        sample_keys = []
        for idx, (k, v) in enumerate(cursor):
            if k == b"__len__": continue
            decomp = zlib.decompress(v)
            obj = json.loads(decomp.decode("utf-8"))
            if idx == 0:
                print("OMat24 record keys:", list(obj.keys()))
                print("OMat24 data field keys:", list(obj.get("data", {}).keys()))
                print("Sample record data:", obj.get("data"))
            break

omat_summary = "/home/vibhan23/omat24/omat24_rattled_relax_bec_predictions_summary.csv"
if os.path.exists(omat_summary):
    df_omat = pd.read_csv(omat_summary)
    print(f"\nExisting omat24_rattled_relax_bec_predictions_summary.csv rows: {len(df_omat):,}")
    print(f"Columns: {df_omat.columns.tolist()}")

print("\n" + "=" * 80)
print("4. PARITY PLOT FRACTIONAL NOMINAL CHARGES AUDIT")
print("=" * 80)
pilot_15k = "/home/vibhan23/gnome/gnome_semiconductors_full_3x3_predictions_full_3x3_15k.csv"
if os.path.exists(pilot_15k):
    df_15k = pd.read_csv(pilot_15k)
    print(f"Total atom rows in 15k pilot: {len(df_15k):,}")
    nom = df_15k["nominal_oxidation_state"]
    is_int = nom.apply(lambda x: x == int(x) if pd.notna(x) else False)
    print(f"Exact integer nominal charges: {is_int.sum():,} ({is_int.mean()*100:.2f}%)")
    print(f"Non-integer (fractional) nominal charges: {(~is_int).sum():,} ({(~is_int).mean()*100:.2f}%)")
    print("Sample fractional nominal charges:", nom[~is_int].head(15).tolist())
    print("Sample formulas with fractional nominal charges:", df_15k[~is_int][["formula", "element", "nominal_oxidation_state"]].head(10).to_dict("records"))
