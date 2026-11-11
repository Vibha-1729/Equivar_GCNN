#!/usr/bin/env python3
"""
1. Inspect Fe3O4 in Alexandria (agm003236556, agm003215414) to see its exact Wyckoff sites,
   BVAnalyzer integer oxidation states, and predicted BECs.
2. Test Professor's Wyckoff-Site Mixed-Valence Hypothesis on GNoME fractional-charge compounds
   using TRUE crystallographic Wyckoff orbits (from SpacegroupAnalyzer.get_symmetrized_structure().equivalent_indices),
   so distinct orbits sharing the same Wyckoff letter (e.g., two different 2i sites) are kept separate!
"""

import json
import zipfile
import itertools
from collections import defaultdict, Counter
import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def inspect_alexandria_ids(json_path, target_ids):
    print(f"=== Inspecting exact Fe3O4 entries in {json_path} ===", flush=True)
    found = []
    with open(json_path, "r", encoding="utf-8") as f:
        buf = []
        in_obj = False
        keep = False
        for line in f:
            if line == "{\n":
                in_obj = True
                keep = False
                buf = [line]
            elif in_obj:
                buf.append(line)
                if '"material_id":' in line:
                    for tid in target_ids:
                        if f'"{tid}"' in line:
                            keep = True
                            break
                if line.startswith("}"):
                    if keep:
                        obj_str = "".join(buf).rstrip().rstrip(",")
                        found.append(json.loads(obj_str))
                        if len(found) == len(target_ids):
                            break
                    in_obj = False
                    buf = []
    for m in found:
        print(f"\n[{m['material_id']}] {m['formula']} | {m['space_group_symbol']} (#{m['space_group_number']}) | {m['crystal_system']}")
        print(f"  Method: {m['nominal_oxidation_state_method']} | Bandgap: {m['bandgap_eV']} eV | Wyckoff: {m.get('wyckoff_symbols_with_multiplicity')}")
        by_el_nom = defaultdict(list)
        for a in m["predicted_atoms_bec"]:
            by_el_nom[(a["element"], a["nominal_oxidation_state"])].append(a["Z_trace_mean"])
        for (el, nom), zts in sorted(by_el_nom.items()):
            print(f"    {el:2s} (q_nom={nom:+.1f}) [n={len(zts):2d}] -> Z*_pred mean={np.mean(zts):+.3f}, std={np.std(zts):.3f}, range=[{min(zts):+.3f}, {max(zts):+.3f}]")


def iter_gnome_fractional_compounds(json_path, target_count=200, max_scan=40000):
    found = []
    scanned = 0
    with open(json_path, "r", encoding="utf-8") as f:
        buf = []
        in_obj = False
        for line in f:
            if line == "{\n":
                in_obj = True
                buf = [line]
            elif in_obj:
                buf.append(line)
                if line.startswith("}"):
                    scanned += 1
                    obj_str = "".join(buf).rstrip().rstrip(",")
                    m = json.loads(obj_str)
                    method = m.get("nominal_oxidation_state_method", "")
                    if method != "Zero_Fallback":
                        atoms = m.get("predicted_atoms_bec", [])
                        has_frac = any(
                            a.get("nominal_oxidation_state") is not None
                            and abs(round(float(a["nominal_oxidation_state"])) - float(a["nominal_oxidation_state"])) > 1e-4
                            for a in atoms
                        )
                        if has_frac:
                            found.append(m)
                            if len(found) >= target_count:
                                break
                    in_obj = False
                    buf = []
                    if scanned >= max_scan:
                        break
    return found, scanned


def can_solve_integer_wyckoff_charges(group_counts, elements_in_groups):
    groups = list(group_counts.keys())
    if len(groups) > 12:
        return None

    candidate_states = []
    for g in groups:
        el = elements_in_groups[g]
        elem = Element(el)
        # Prefer common oxidation states first, then ICSD states
        common = [int(s) for s in elem.common_oxidation_states if s != 0 and float(s).is_integer()]
        icsd = [int(s) for s in elem.icsd_oxidation_states if s != 0 and float(s).is_integer()]
        # Preserve priority: common first, then remaining icsd
        states = list(dict.fromkeys(common + icsd))
        if not states:
            return None
        # Limit branching per group if huge
        candidate_states.append(states[:5])

    counts = [group_counts[g] for g in groups]
    for combo in itertools.product(*candidate_states):
        if sum(c * q for c, q in zip(counts, combo)) == 0:
            return dict(zip(groups, combo))
    return None


def main():
    # Part 1: Inspect Alexandria Fe3O4
    inspect_alexandria_ids(
        "/home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json",
        {"agm003236556", "agm003215414"}
    )

    # Part 2: Inspect GNoME fractional compounds with true distinct Wyckoff orbits
    json_path = "/home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json"
    zip_path = "/home/vibhan23/gnome/by_id.zip"

    print("\n=== Testing Wyckoff-Orbit Hypothesis on 200 GNoME Fractional Compounds ===", flush=True)
    frac_compounds, scanned = iter_gnome_fractional_compounds(json_path, target_count=200, max_scan=40000)
    print(f"Scanned {scanned:,} GNoME records -> found {len(frac_compounds)} fractional-charge compounds.\n", flush=True)

    stats = Counter()
    multi_orbit_examples = []
    single_orbit_examples = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for m in frac_compounds:
            mat_id = m["material_id"]
            formula = m["formula"]
            method = m.get("nominal_oxidation_state_method")
            atoms = m["predicted_atoms_bec"]

            try:
                cif_str = z.read(f"by_id/{mat_id}.CIF").decode("utf-8")
                struct = Structure.from_str(cif_str, fmt="cif")
                sga = SpacegroupAnalyzer(struct)
                sym_struct = sga.get_symmetrized_structure()
            except Exception:
                stats["cif_or_sym_error"] += 1
                continue

            # Map atom index -> UNIQUE Wyckoff orbit label e.g. "2i_#1", "2i_#2"
            idx_to_orbit = {}
            orbit_counter_per_el_letter = defaultdict(int)
            orbits_per_el = defaultdict(list)

            for idx_group, wyck in zip(sym_struct.equivalent_indices, sym_struct.wyckoff_symbols):
                el_of_orbit = struct[idx_group[0]].specie.symbol
                orbit_counter_per_el_letter[(el_of_orbit, wyck)] += 1
                k = orbit_counter_per_el_letter[(el_of_orbit, wyck)]
                orbit_label = f"{wyck}#{k}"
                orbits_per_el[el_of_orbit].append(orbit_label)
                for idx in idx_group:
                    idx_to_orbit[idx] = orbit_label

            frac_elements = set()
            bec_by_el_orbit = defaultdict(list)
            nom_by_el = {}
            group_counts = Counter()
            elements_in_groups = {}

            for a in atoms:
                idx = a["atom_index"]
                el = a["element"]
                nom = float(a["nominal_oxidation_state"])
                zt = float(a["Z_trace_mean"])
                orbit = idx_to_orbit.get(idx, "?")

                bec_by_el_orbit[(el, orbit)].append(zt)
                nom_by_el[el] = nom
                group_counts[(el, orbit)] += 1
                elements_in_groups[(el, orbit)] = el

                if abs(round(nom) - nom) > 1e-4:
                    frac_elements.add(el)

            stats["analyzed"] += 1
            multi_orbit_frac = any(len(orbits_per_el[el]) >= 2 for el in frac_elements)
            if multi_orbit_frac:
                stats["frac_element_has_multiple_distinct_wyckoff_orbits"] += 1
            else:
                stats["frac_element_has_STRICTLY_ONE_wyckoff_orbit"] += 1

            solved = None
            if multi_orbit_frac:
                solved = can_solve_integer_wyckoff_charges(group_counts, elements_in_groups)
                if solved is not None:
                    stats["resolved_to_clean_integers_via_wyckoff_orbit_split"] += 1
                    diff_charges = any(
                        len({solved[(el, o)] for o in orbits_per_el[el]}) >= 2
                        for el in frac_elements
                    )
                    if diff_charges:
                        stats["resolved_with_distinct_mixed_valences"] += 1
                else:
                    stats["multi_orbit_but_no_int_solution_found"] += 1

            info = {
                "mat_id": mat_id,
                "formula": formula,
                "spg": f"{sga.get_space_group_symbol()} (#{sga.get_space_group_number()})",
                "frac_elements": sorted(frac_elements),
                "nom_by_el": nom_by_el,
                "orbits_per_el": dict(orbits_per_el),
                "bec_stats": {
                    f"{el} ({orb}) [n={len(vals)}]": (round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3))
                    for (el, orb), vals in sorted(bec_by_el_orbit.items())
                },
                "wyckoff_int_solution": {f"{el} ({orb})": q for (el, orb), q in sorted(solved.items())} if solved else None,
            }

            if multi_orbit_frac and solved and len(multi_orbit_examples) < 6 and len(atoms) <= 24:
                multi_orbit_examples.append(info)
            elif not multi_orbit_frac and len(single_orbit_examples) < 5:
                single_orbit_examples.append(info)

    print("=== TRUE WYCKOFF-ORBIT AGGREGATE STATS (200 GNoME Fractional Compounds) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== EXAMPLES: MULTIPLE WYCKOFF ORBITS -> CLEAN INTEGER MIXED VALENCE ===")
    for ex in multi_orbit_examples:
        print(f"\n[{ex['mat_id']}] {ex['formula']} | {ex['spg']}")
        print(f"  Fractional element(s): {ex['frac_elements']} | Assigned q_nom: { {e: round(ex['nom_by_el'][e], 4) for e in ex['frac_elements']} }")
        print(f"  Distinct Wyckoff orbits: {ex['orbits_per_el']}")
        print(f"  Predicted BEC Z*_trace per Wyckoff orbit:")
        for lbl, (mz, sz) in ex["bec_stats"].items():
            print(f"    {lbl:20s} -> Z*_pred = {mz:+.3f} (std={sz:.3f})")
        print(f"  Wyckoff-split integer valence solution: {ex['wyckoff_int_solution']}")

    print("\n=== TRUE COUNTER-EXAMPLES: STRICTLY ONE WYCKOFF ORBIT (ALL ATOMS EQUIVALENT) ===")
    for ex in single_orbit_examples:
        print(f"\n[{ex['mat_id']}] {ex['formula']} | {ex['spg']}")
        print(f"  Fractional element(s): {ex['frac_elements']} | Assigned q_nom: { {e: round(ex['nom_by_el'][e], 4) for e in ex['frac_elements']} }")
        print(f"  Distinct Wyckoff orbits: {ex['orbits_per_el']}")
        for lbl, (mz, sz) in ex["bec_stats"].items():
            print(f"    {lbl:20s} -> Z*_pred = {mz:+.3f} (std={sz:.3f})")


if __name__ == "__main__":
    main()
