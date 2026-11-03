import os
import json
import zlib
import lmdb
import re
import pandas as pd

# 1. Load Alexandria stable semiconductors summary to check known semiconductors
alex_csv = "/home/vibhan23/alexandria/alexandria_stable_semiconductors_summary.csv"
print(f"Loading {alex_csv}...")
df_alex = pd.read_csv(alex_csv)
alex_semi_gaps = dict(zip(df_alex["material_id"], df_alex["bandgap_eV"]))
print(f"Loaded {len(alex_semi_gaps):,} Alexandria stable semiconductors.")

# 2. Inspect OMat24 parent IDs from the summary and from LMDB
omat_summary_csv = "/home/vibhan23/omat24/omat24_rattled_relax_bec_predictions_summary.csv"
df_omat = pd.read_csv(omat_summary_csv)
print(f"Total OMat24 summary entries: {len(df_omat):,}")

# Read parent_id from LMDB for OMat24 entries
lmdb_path = "/home/vibhan23/omat24/extracted_1M/train/rattled-relax/train.aselmdb"
env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)

sample_matches = 0
sample_total = 0
sample_agm_ids = []

with env.begin() as txn:
    cursor = txn.cursor()
    for k, v in cursor:
        if k == b"__len__": continue
        sample_total += 1
        obj = json.loads(zlib.decompress(v).decode("utf-8"))
        raw_pid = obj.get("data", {}).get("parent_id", "")
        # Extract agm prefix
        match = re.search(r"(agm\d+)", raw_pid)
        if match:
            agm_id = match.group(1)
            sample_agm_ids.append(agm_id)
            if agm_id in alex_semi_gaps:
                sample_matches += 1
        if sample_total >= 5000:
            break

print(f"In a sample of 5,000 OMat24 entries:")
print(f"  Parent matches in Alexandria STABLE SEMICONDUCTOR dataset (Eg > 0): {sample_matches} ({sample_matches/sample_total*100:.1f}%)")
print(f"  Sample raw parent_id: {raw_pid}")
print(f"  Sample extracted agm_id: {agm_id}")
