#!/bin/bash
set -e

echo "================================================================================"
echo "  FULL DATABASE 3x3 BEC JSON EXPORT FOR ALL CRYSTAL STRUCTURES"
echo "  OMat24 (100k) -> Alexandria (230k) -> GNoME (385k)"
echo "  Compute Device: RTX 5060 GPU"
echo "================================================================================"

source /home/vibhan23/venv/bin/activate
cd /home/vibhan23

# 1. OMat24 Export (All 100k entries)
echo -e "\n[1/3] Starting OMat24 Full Database 3x3 BEC Export..."
python /home/vibhan23/scripts/export_all_structures_3x3_json.py --dataset omat24
echo "Compressing OMat24 Full JSON..."
gzip -k -f /home/vibhan23/omat24/omat24_all_structures_full_3x3_bec.json

# 2. Alexandria Export (All 230k entries)
echo -e "\n[2/3] Starting Alexandria Full Database 3x3 BEC Export..."
python /home/vibhan23/scripts/export_all_structures_3x3_json.py --dataset alexandria
echo "Compressing Alexandria Full JSON..."
gzip -k -f /home/vibhan23/alexandria/alexandria_all_structures_full_3x3_bec.json

# 3. GNoME Export (All 385k entries)
echo -e "\n[3/3] Starting GNoME Full Database 3x3 BEC Export..."
python /home/vibhan23/scripts/export_all_structures_3x3_json.py --dataset gnome
echo "Compressing GNoME Full JSON..."
gzip -k -f /home/vibhan23/gnome/gnome_all_structures_full_3x3_bec.json

echo -e "\n================================================================================"
echo "  ALL DATASETS FULL 3x3 BEC JSON EXPORT COMPLETED SUCCESSFULLY!"
echo "================================================================================"
