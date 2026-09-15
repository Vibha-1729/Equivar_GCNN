import os
import sys
import urllib.request
import pandas as pd
from tqdm import tqdm

base_dir = "/home/vibhan23"
if not os.path.exists(base_dir):
    base_dir = r"c:\Users\Vibha Narayan\OneDrive\Desktop\Coding\UGP"

gnome_dir = os.path.join(base_dir, "gnome")
data_dir = os.path.join(base_dir, "data", "processed")
output_dir = os.path.join(base_dir, "outputs", "csv_results")

os.makedirs(gnome_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("==================================================")
print("  GOOGLE GNoME DATASET DOWNLOAD & INSULATOR FILTERING")
print("==================================================")

# Download GNoME summary CSV from Google Cloud Bucket
summary_url = "https://storage.googleapis.com/gdm_materials_discovery/gnome_data/stable_materials_summary.csv"
dest_csv = os.path.join(gnome_dir, "stable_materials_summary.csv")

if not os.path.exists(dest_csv) or os.path.getsize(dest_csv) < 1000:
    print(f"Downloading GNoME stable_materials_summary.csv from {summary_url}...")
    try:
        urllib.request.urlretrieve(summary_url, dest_csv)
        print(f"Successfully downloaded GNoME summary CSV ({os.path.getsize(dest_csv):,} bytes)")
    except Exception as e:
        print(f"Error downloading via urllib: {e}")
        # Try curl/wget via subprocess if needed
        os.system(f"wget --no-check-certificate -O {dest_csv} {summary_url}")

# Read and audit GNoME materials
print(f"\nReading GNoME dataset summary: {dest_csv}...")
df = pd.read_csv(dest_csv)
total_materials = len(df)
print(f"Total GNoME Materials: {total_materials:,}")
print("Columns in GNoME summary:", list(df.columns))

# Audit Bandgaps (Eg > 0 eV)
# Look for bandgap column
bg_col = None
for col in df.columns:
    if 'gap' in col.lower() or 'band' in col.lower():
        bg_col = col
        break

if bg_col:
    print(f"\nFound Bandgap Column: '{bg_col}'")
    df[bg_col] = pd.to_numeric(df[bg_col], errors='coerce').fillna(0.0)
    
    insulators_df = df[df[bg_col] > 0.0].copy()
    metals_df = df[df[bg_col] == 0.0].copy()
    
    insulator_count = len(insulators_df)
    metal_count = len(metals_df)
    
    insulator_pct = (insulator_count / total_materials) * 100.0
    metal_pct = (metal_count / total_materials) * 100.0
    
    print(f"\n--- GNoME BANDGAP AUDIT SUMMARY ---")
    print(f"Total Materials          : {total_materials:,}")
    print(f"Insulators (Eg > 0 eV)  : {insulator_count:,} ({insulator_pct:.2f}%)")
    print(f"Metals (Eg = 0 eV)       : {metal_count:,} ({metal_pct:.2f}%)")
    
    # Save Filtered Insulators
    filtered_out = os.path.join(data_dir, "GNoME_insulators_filtered.csv")
    insulators_df.to_csv(filtered_out, index=False)
    print(f"\nSaved filtered GNoME insulators to: {filtered_out}")
    
    # Save Audit CSV
    audit_out = os.path.join(output_dir, "gnome_bandgap_audit_summary.csv")
    pd.DataFrame([{
        'Dataset': 'Google DeepMind GNoME',
        'Total_Materials': total_materials,
        'Insulators_Eg_gt_0': insulator_count,
        'Insulators_Percent': insulator_pct,
        'Metals_Eg_eq_0': metal_count,
        'Metals_Percent': metal_pct,
        'Filtered_Insulators_File': 'data/processed/GNoME_insulators_filtered.csv'
    }]).to_csv(audit_out, index=False)
    print(f"Saved bandgap audit summary CSV to: {audit_out}")
else:
    print("\nBandgap column not found explicitly in summary CSV. Inspecting sample rows:")
    print(df.head())

print("\n==================================================")
print("  GNoME INSULATOR FILTERING COMPLETE (STOPPED AT THIS STAGE)")
print("==================================================")
