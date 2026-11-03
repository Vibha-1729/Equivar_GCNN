import os
import gzip
import json
import pandas as pd
from tqdm import tqdm

def annotate_omat24():
    print("=" * 80)
    print("ANNOTATING OMAT24 WITH ALEXANDRIA PARENT IDS AND DFT BANDGAPS")
    print("=" * 80)

    uid_map_path = "/home/vibhan23/omat24/omat24_uid_to_agm_parent.json"
    bg_map_path = "/home/vibhan23/omat24/omat24_parent_bandgaps.json"
    csv_summary_path = "/home/vibhan23/omat24/omat24_rattled_relax_bec_predictions_summary.csv"
    annotated_csv_path = "/home/vibhan23/omat24/omat24_rattled_relax_with_alexandria_bandgaps.csv"
    semi_csv_path = "/home/vibhan23/omat24/omat24_verified_semiconductors_summary.csv"
    json_gz_path = "/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json.gz"

    # 1. Load mappings
    with open(uid_map_path, "r", encoding="utf-8") as f:
        uid_map = json.load(f)
    print(f"Loaded {len(uid_map):,} UID-to-AGM mappings.")

    with open(bg_map_path, "r", encoding="utf-8") as f:
        bg_map = json.load(f)
    print(f"Loaded {len(bg_map):,} Alexandria parent bandgap records.")

    # 2. Annotate CSV Summary
    if os.path.exists(csv_summary_path):
        print(f"\nAnnotating CSV summary: {csv_summary_path}")
        df = pd.read_csv(csv_summary_path)
        
        parent_ids = []
        band_gaps = []
        is_semis = []

        for mid in df["material_id"]:
            mid_str = str(mid)
            agm = uid_map.get(mid_str)
            if agm:
                parent_ids.append(agm)
                bg = float(bg_map.get(agm, 0.0))
                band_gaps.append(round(bg, 4))
                is_semis.append(bg > 0.0)
            else:
                parent_ids.append(None)
                band_gaps.append(0.0)
                is_semis.append(False)

        df["parent_alexandria_id"] = parent_ids
        df["band_gap_eV"] = band_gaps
        df["is_semiconductor"] = is_semis

        # Overwrite original and save annotated copy
        df.to_csv(csv_summary_path, index=False)
        df.to_csv(annotated_csv_path, index=False)
        print(f"Saved annotated CSV summary -> {csv_summary_path}")
        print(f"Saved annotated CSV copy -> {annotated_csv_path}")

        # Save verified semiconductors subset
        df_semi = df[df["is_semiconductor"] == True]
        df_semi.to_csv(semi_csv_path, index=False)
        print(f"Saved verified semiconductor subset ({len(df_semi):,} rows) -> {semi_csv_path}")

    # 3. Annotate JSON.GZ full tensor dataset
    if os.path.exists(json_gz_path):
        print(f"\nAnnotating full 3x3 BEC JSON.GZ: {json_gz_path}")
        with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        matched_count = 0
        semi_count = 0
        metal_count = 0

        for entry in tqdm(data, desc="Injecting parent IDs and bandgaps"):
            mid = str(entry.get("material_id", ""))
            agm = uid_map.get(mid)
            if agm:
                matched_count += 1
                bg = float(bg_map.get(agm, 0.0))
                entry["parent_id"] = agm
                entry["parent_alexandria_id"] = agm
                entry["band_gap"] = round(bg, 4)
                entry["band_gap_eV"] = round(bg, 4)
                entry["is_semiconductor"] = (bg > 0.0)
                if bg > 0.0:
                    semi_count += 1
                else:
                    metal_count += 1
            else:
                entry["parent_id"] = None
                entry["parent_alexandria_id"] = None
                entry["band_gap"] = 0.0
                entry["band_gap_eV"] = 0.0
                entry["is_semiconductor"] = False
                metal_count += 1

        print(f"Writing updated annotated JSON.GZ -> {json_gz_path}")
        with gzip.open(json_gz_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Successfully updated {json_gz_path}")

        print("\n" + "=" * 80)
        print("FINAL OMAT24 ANNOTATION AUDIT")
        print("=" * 80)
        print(f"Total OMat24 materials: {len(data):,}")
        print(f"Materials matched to Alexandria parents: {matched_count:,} ({matched_count / len(data) * 100:.2f}%)")
        print(f"Verified Semiconductors (Eg > 0 eV): {semi_count:,} ({semi_count / len(data) * 100:.2f}%)")
        print(f"Metals (Eg == 0.0 eV): {metal_count:,} ({metal_count / len(data) * 100:.2f}%)")
        print("=" * 80)

if __name__ == "__main__":
    annotate_omat24()
