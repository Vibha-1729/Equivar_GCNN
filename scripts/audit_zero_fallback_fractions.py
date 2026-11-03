import pandas as pd

print("=" * 80)
print("ZERO_FALLBACK FRACTION AUDIT ACROSS ALL THREE DATASETS")
print("=" * 80)

# 1. GNoME
gnome_csv = "/home/vibhan23/gnome/gnome_stable_semiconductors_bec_predictions_summary.csv"
df_gnome = pd.read_csv(gnome_csv)
tot_gnome = len(df_gnome)
zf_gnome = (df_gnome["nominal_oxidation_state_method"] == "Zero_Fallback").sum()
bv_gnome = (df_gnome["nominal_oxidation_state_method"] == "BVAnalyzer").sum()
cg_gnome = (df_gnome["nominal_oxidation_state_method"] == "Composition_Guesses").sum()

print("\n1. GNoME (385,490 Stable Semiconductors):")
print(f"  Total Materials: {tot_gnome:,}")
print(f"  BVAnalyzer:          {bv_gnome:,} ({bv_gnome/tot_gnome*100:.2f}%)")
print(f"  Composition_Guesses: {cg_gnome:,} ({cg_gnome/tot_gnome*100:.2f}%)")
print(f"  Zero_Fallback:       {zf_gnome:,} ({zf_gnome/tot_gnome*100:.2f}%)")

# 2. Alexandria
alex_csv = "/home/vibhan23/alexandria/alexandria_stable_semiconductors_summary.csv"
df_alex = pd.read_csv(alex_csv)
tot_alex = len(df_alex)
zf_alex = (df_alex["nominal_oxidation_state_method"] == "Zero_Fallback").sum()
bv_alex = (df_alex["nominal_oxidation_state_method"] == "BVAnalyzer").sum()
cg_alex = (df_alex["nominal_oxidation_state_method"] == "Composition_Guesses").sum()

print("\n2. Alexandria (230,204 Stable/Metastable Semiconductors):")
print(f"  Total Materials: {tot_alex:,}")
print(f"  BVAnalyzer:          {bv_alex:,} ({bv_alex/tot_alex*100:.2f}%)")
print(f"  Composition_Guesses: {cg_alex:,} ({cg_alex/tot_alex*100:.2f}%)")
print(f"  Zero_Fallback:       {zf_alex:,} ({zf_alex/tot_alex*100:.2f}%)")

# 3. OMat24
omat_csv = "/home/vibhan23/omat24/omat24_rattled_relax_bec_predictions_summary.csv"
df_omat = pd.read_csv(omat_csv)
tot_omat = len(df_omat)
zf_omat = (df_omat["nominal_oxidation_state_method"] == "Zero_Fallback").sum()
bv_omat = (df_omat["nominal_oxidation_state_method"] == "BVAnalyzer").sum()
cg_omat = (df_omat["nominal_oxidation_state_method"] == "Composition_Guesses").sum()

print("\n3. OMat24 (99,976 rattled-relax Frames):")
print(f"  Total Materials: {tot_omat:,}")
print(f"  BVAnalyzer:          {bv_omat:,} ({bv_omat/tot_omat*100:.2f}%)")
print(f"  Composition_Guesses: {cg_omat:,} ({cg_omat/tot_omat*100:.2f}%)")
print(f"  Zero_Fallback:       {zf_omat:,} ({zf_omat/tot_omat*100:.2f}%)")

# 4. Check on Giant Candidates subset
print("\n" + "=" * 80)
print("ZERO_FALLBACK FRACTION AMONG GIANT ANOMALOUS CANDIDATES (|ΔZ*| >= 2.0e)")
print("=" * 80)
gnome_giant = pd.read_csv("/home/vibhan23/gnome/gnome_stable_semiconductors_giant_anomalous_candidates.csv")
alex_giant = pd.read_csv("/home/vibhan23/alexandria/alexandria_giant_anomalous_candidates.csv")
omat_giant = pd.read_csv("/home/vibhan23/omat24/omat24_giant_anomalous_candidates.csv")

print(f"GNoME Giant Candidates (319,250 total):")
print(f"  Zero_Fallback: {(gnome_giant['nominal_oxidation_state_method'] == 'Zero_Fallback').sum():,} ({(gnome_giant['nominal_oxidation_state_method'] == 'Zero_Fallback').mean()*100:.2f}%)")
print(f"  BVAnalyzer + Composition_Guesses: {(gnome_giant['nominal_oxidation_state_method'] != 'Zero_Fallback').sum():,} ({(gnome_giant['nominal_oxidation_state_method'] != 'Zero_Fallback').mean()*100:.2f}%)")

print(f"\nAlexandria Giant Candidates (77,843 total):")
print(f"  Zero_Fallback: {(alex_giant['nominal_oxidation_state_method'] == 'Zero_Fallback').sum():,} ({(alex_giant['nominal_oxidation_state_method'] == 'Zero_Fallback').mean()*100:.2f}%)")
print(f"  BVAnalyzer + Composition_Guesses: {(alex_giant['nominal_oxidation_state_method'] != 'Zero_Fallback').sum():,} ({(alex_giant['nominal_oxidation_state_method'] != 'Zero_Fallback').mean()*100:.2f}%)")

print(f"\nOMat24 Giant Candidates (55,488 total):")
print(f"  Zero_Fallback: {(omat_giant['nominal_oxidation_state_method'] == 'Zero_Fallback').sum():,} ({(omat_giant['nominal_oxidation_state_method'] == 'Zero_Fallback').mean()*100:.2f}%)")
print(f"  BVAnalyzer + Composition_Guesses: {(omat_giant['nominal_oxidation_state_method'] != 'Zero_Fallback').sum():,} ({(omat_giant['nominal_oxidation_state_method'] != 'Zero_Fallback').mean()*100:.2f}%)")
