#!/bin/bash
set -e

echo "================================================================================"
echo "  STARTING COMPLETE 3x3 BEC JSON EXPORT FOR GIANT ANOMALOUS CANDIDATES"
echo "  Datasets: OMat24 (55k) -> Alexandria (78k) -> GNoME (319k)"
echo "================================================================================"

source /home/vibhan23/venv/bin/activate
cd /home/vibhan23

# 1. OMat24 Export
echo -e "\n[1/3] Exporting OMat24 Giant Anomalous 3x3 BEC JSON..."
python /home/vibhan23/scripts/export_giant_anomalous_3x3_json.py --dataset omat24
echo "Compressing OMat24 JSON..."
gzip -k -f /home/vibhan23/omat24/omat24_giant_anomalous_3x3_bec.json

# 2. Alexandria Export
echo -e "\n[2/3] Exporting Alexandria Giant Anomalous 3x3 BEC JSON..."
python /home/vibhan23/scripts/export_giant_anomalous_3x3_json.py --dataset alexandria
echo "Compressing Alexandria JSON..."
gzip -k -f /home/vibhan23/alexandria/alexandria_giant_anomalous_3x3_bec.json

# 3. GNoME Export
echo -e "\n[3/3] Exporting GNoME Giant Anomalous 3x3 BEC JSON..."
python /home/vibhan23/scripts/export_giant_anomalous_3x3_json.py --dataset gnome
echo "Compressing GNoME JSON..."
gzip -k -f /home/vibhan23/gnome/gnome_giant_anomalous_3x3_bec.json

echo -e "\n================================================================================"
echo "  ALL 3 DATASETS 3x3 BEC JSON EXPORTS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
