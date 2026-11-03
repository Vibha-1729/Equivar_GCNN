import pandas as pd
import numpy as np

# Check Alexandria
df_alex = pd.read_csv("/home/vibhan23/alexandria/alexandria_stable_semiconductors_summary.csv")
print("=== ALEXANDRIA e_above_hull ===")
print("Min e_above_hull:", df_alex["e_above_hull"].min())
print("Max e_above_hull:", df_alex["e_above_hull"].max())
print("e_above_hull < 0:", (df_alex["e_above_hull"] < 0).sum())
print("e_above_hull == 0.0:", (df_alex["e_above_hull"] == 0.0).sum())
print("e_above_hull <= 0.0:", (df_alex["e_above_hull"] <= 0.0).sum())
print("0 < e_above_hull <= 0.05:", ((df_alex["e_above_hull"] > 0) & (df_alex["e_above_hull"] <= 0.05)).sum())

# Check GNoME
df_gnome = pd.read_csv("/home/vibhan23/gnome/stable_materials_summary.csv")
print("\n=== GNOME Decomposition Energy Per Atom ===")
print("Min Decomposition Energy:", df_gnome["Decomposition Energy Per Atom"].min())
print("Max Decomposition Energy:", df_gnome["Decomposition Energy Per Atom"].max())
print("Decomposition Energy < 0:", (df_gnome["Decomposition Energy Per Atom"] < 0).sum())
print("Decomposition Energy == 0:", (df_gnome["Decomposition Energy Per Atom"] == 0).sum())
print("Decomposition Energy <= 0:", (df_gnome["Decomposition Energy Per Atom"] <= 0).sum())
print("Decomposition Energy > 0:", (df_gnome["Decomposition Energy Per Atom"] > 0).sum())
