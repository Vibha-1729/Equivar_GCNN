import os
import json
import bz2
import glob
from tqdm import tqdm

# 1. Load the required 78,615 unique agm IDs
with open("/home/vibhan23/omat24/omat24_uid_to_agm_parent.json") as f:
    uid_to_agm = json.load(f)

needed_agms = set(uid_to_agm.values())
print(f"Target unique Alexandria parent IDs to find: {len(needed_agms):,}")

# 2. Check how many are already in our 230k screened Alexandria summary
alex_csv = "/home/vibhan23/alexandria/alexandria_stable_semiconductors_summary.csv"
import pandas as pd
df_alex = pd.read_csv(alex_csv)
alex_gaps = dict(zip(df_alex["material_id"], df_alex["bandgap_eV"]))

found_gaps = {}
for agm in needed_agms:
    if agm in alex_gaps:
        found_gaps[agm] = float(alex_gaps[agm])

print(f"Already found in 230k stable semiconductors: {len(found_gaps):,} / {len(needed_agms):,} ({len(found_gaps)/len(needed_agms)*100:.1f}%)")

missing_agms = needed_agms - set(found_gaps.keys())
print(f"Remaining to search in raw 48 chunks: {len(missing_agms):,}")

# 3. Fast scan over 48 chunks only for missing agms
alex_dir = "/home/vibhan23/alexandria"
files = sorted(glob.glob(os.path.join(alex_dir, "alexandria_*.json.bz2")))

for fpath in tqdm(files, desc="Scanning Alexandria for OMat24 parents"):
    if not missing_agms:
        break
    try:
        with bz2.open(fpath, "rt", encoding="utf-8") as f:
            chunk_data = json.load(f)
        entries = chunk_data.get("entries", chunk_data) if isinstance(chunk_data, dict) else chunk_data
        for e in entries:
            edata = e.get("data", {}) if isinstance(e, dict) else {}
            mat_id = str(edata.get("mat_id", e.get("entry_id", "")))
            if mat_id in missing_agms:
                gap = float(edata.get("band_gap_ind", edata.get("band_gap", 0.0)))
                found_gaps[mat_id] = gap
                missing_agms.remove(mat_id)
                if not missing_agms:
                    break
    except Exception as err:
        print(f"Error in {fpath}: {err}")

print(f"\nFinal tally: Found bandgap for {len(found_gaps):,} / {len(needed_agms):,} ({len(found_gaps)/len(needed_agms)*100:.1f}%)")

# Save complete parent bandgaps
out_path = "/home/vibhan23/omat24/omat24_parent_bandgaps.json"
with open(out_path, "w") as f:
    json.dump(found_gaps, f)
print(f"Saved {out_path}")
