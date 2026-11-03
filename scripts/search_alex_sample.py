import os
import json
import bz2
import glob

sample_ids = ["agm004242068", "agm001812462"]
alex_dir = "/home/vibhan23/alexandria"
files = sorted(glob.glob(os.path.join(alex_dir, "alexandria_*.json.bz2")))

print(f"Scanning first 5 chunks for sample parent IDs: {sample_ids}...")
found = {}
for fpath in files[:5]:
    with bz2.open(fpath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", data) if isinstance(data, dict) else data
    for e in entries:
        edata = e.get("data", {}) if isinstance(e, dict) else {}
        mat_id = str(edata.get("mat_id", e.get("entry_id", "")))
        if mat_id in sample_ids:
            found[mat_id] = {
                "band_gap_ind": edata.get("band_gap_ind"),
                "band_gap_dir": edata.get("band_gap_dir"),
                "e_above_hull": edata.get("e_above_hull"),
                "formula": edata.get("formula")
            }
            print(f"Found {mat_id}: {found[mat_id]}")
    if len(found) == len(sample_ids):
        break
