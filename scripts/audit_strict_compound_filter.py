import gzip
import json

def audit_strict_compound_purity(json_gz_path, name, parent_bg_path=None, uid_map_path=None, filter_eg=False):
    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    parent_bgs = {}
    if parent_bg_path:
        with open(parent_bg_path) as f:
            parent_bgs = json.load(f)

    uid_map = {}
    if uid_map_path:
        with open(uid_map_path) as f:
            uid_map = json.load(f)

    total_mats = len(data)
    retained_mats = 0
    retained_atoms = 0
    zero_fallback_mats = 0
    fractional_rejected_mats = 0
    metallic_omats = 0

    for m in data:
        mat_id = str(m.get("material_id", m.get("id", "")))
        if filter_eg:
            agm = uid_map.get(mat_id)
            if not agm or parent_bgs.get(agm, 0.0) <= 0.0:
                metallic_omats += 1
                continue

        oxi_method = m.get("nominal_oxidation_state_method", "")
        if oxi_method == "Zero_Fallback":
            zero_fallback_mats += 1
            continue

        atoms = m.get("predicted_atoms_bec", [])
        if not atoms:
            continue

        all_atoms_valid = True
        for a in atoms:
            nom = a.get("nominal_oxidation_state")
            if nom is None:
                all_atoms_valid = False
                break
            nom = float(nom)
            if nom == 0.0 or abs(round(nom) - nom) > 1e-4:
                all_atoms_valid = False
                break

        if not all_atoms_valid:
            fractional_rejected_mats += 1
            continue

        retained_mats += 1
        retained_atoms += len(atoms)

    print(f"=== {name} ===")
    print(f"Total Materials: {total_mats:,}")
    if filter_eg:
        print(f"Metallic Filtered (Eg <= 0): {metallic_omats:,}")
    print(f"Zero_Fallback Rejected Materials: {zero_fallback_mats:,}")
    print(f"Fractional/Zero Atom Rejected Materials: {fractional_rejected_mats:,}")
    print(f"Strict Pure-Integer Materials Retained: {retained_mats:,}")
    print(f"Total Atoms in Retained Materials: {retained_atoms:,}\n")

if __name__ == "__main__":
    audit_strict_compound_purity(
        "/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json.gz",
        "GNoME"
    )
    audit_strict_compound_purity(
        "/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json.gz",
        "Alexandria"
    )
    audit_strict_compound_purity(
        "/home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json.gz",
        "OMat24 (Eg > 0)",
        parent_bg_path="/home/vibhan23/omat24/omat24_parent_bandgaps.json",
        uid_map_path="/home/vibhan23/omat24/omat24_uid_to_agm_parent.json",
        filter_eg=True
    )
