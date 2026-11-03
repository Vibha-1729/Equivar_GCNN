import lmdb
import json
import zlib
import re

lmdb_path = "/home/vibhan23/omat24/extracted_1M/train/rattled-relax/train.aselmdb"
env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)

unique_pids = set()
unique_agms = set()
entry_to_agm = {}

with env.begin() as txn:
    cursor = txn.cursor()
    for k, v in cursor:
        if k in (b"__len__", b"__keys__"): continue
        try:
            decomp = zlib.decompress(v)
            obj = json.loads(decomp.decode("utf-8"))
            if not isinstance(obj, dict):
                continue
            uid = str(obj.get("unique_id", k.decode("utf-8", errors="ignore")))
            raw_pid = obj.get("data", {}).get("parent_id", "")
            unique_pids.add(raw_pid)
            m = re.search(r"(agm\d+)", raw_pid)
            if m:
                agm = m.group(1)
                unique_agms.add(agm)
                entry_to_agm[uid] = agm
        except Exception:
            continue

print(f"Total entries in OMat24: {len(entry_to_agm):,}")
print(f"Total unique raw parent_ids: {len(unique_pids):,}")
print(f"Total unique agm IDs: {len(unique_agms):,}")

# Save the entry -> agm mapping for fast lookup
with open("/home/vibhan23/omat24/omat24_uid_to_agm_parent.json", "w") as f:
    json.dump(entry_to_agm, f)
print("Saved /home/vibhan23/omat24/omat24_uid_to_agm_parent.json")
