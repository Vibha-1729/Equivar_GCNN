import os
import json
import zlib
import lmdb

lmdb_path = "/home/vibhan23/omat24/extracted_1M/train/rattled-relax/train.aselmdb"
env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        if key == b'__len__': continue
        decomp = zlib.decompress(value)
        obj = json.loads(decomp.decode('utf-8'))
        print("data field:", obj.get('data'))
        break
