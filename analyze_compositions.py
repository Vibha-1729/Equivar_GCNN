import ase.io
from collections import Counter
import pandas as pd

print("==================================================")
print("  ANALYZING ALL CHEMICAL COMPOSTIONS IN PEROVSKITE FILE")
print("==================================================")

xyz_file = 'data/BEC_ZrO2.xyz'

# 1. Read all crystal snapshot frames using ASE
print(f"Reading '{xyz_file}'... (This takes a few seconds)")
frames = ase.io.read(xyz_file, index=':', format='extxyz')

print(f"-> Total Crystal Snapshot Frames Loaded: {len(frames):,}")
total_atoms = sum(len(f) for f in frames)
print(f"-> Total Atoms across all frames: {total_atoms:,}")

# 2. Extract elements and formulas
frame_formulas = []
all_elements = set()

for f in frames:
    symbols = set(f.get_chemical_symbols())
    all_elements.update(symbols)
    formula = f.get_chemical_formula()
    frame_formulas.append(formula)

# 3. Print Summary Statistics
print("\n[1] All Unique Chemical Elements in Dataset:")
print("   ", sorted(list(all_elements)))

counts = Counter(frame_formulas)
print(f"\n[2] Total Distinct Chemical Compositions Found: {len(counts)}")

# 4. Display all compositions as a formatted table
comp_df = pd.DataFrame(counts.most_common(), columns=['Chemical_Formula', 'Snapshot_Frame_Count'])

print("\n[3] Top 20 Most Frequent Compositions:")
print("-" * 55)
print(f"{'No.':<4} | {'Chemical Formula':<32} | {'Frame Count':<10}")
print("-" * 55)
for idx, (formula, count) in enumerate(counts.most_common(20), 1):
    print(f"{idx:<4} | {formula:<32} | {count:<10}")

# 5. Export complete breakdown to outputs/
output_csv = 'outputs/all_ZrO2_compositions.csv'
comp_df.to_csv(output_csv, index=False)
print("-" * 55)
print(f"\n-> Full breakdown of all {len(counts)} compositions saved to: '{output_csv}'")
print("==================================================")
