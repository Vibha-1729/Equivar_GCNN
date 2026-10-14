#!/bin/bash
set -e

echo "========================================================================"
echo "  AUTOMATED HIGH-THROUGHPUT PIPELINE ORCHESTRATOR"
echo "  GNoME -> Final Parity Plots -> Alexandria -> OMat24"
echo "========================================================================"

GNOME_PID=$(pgrep -f "run_gnome_filtered_production.py" || true)

if [ -n "$GNOME_PID" ]; then
    echo "[1/4] GNoME inference is currently running with PID: $GNOME_PID"
    echo "Waiting for GNoME inference to finish..."
    while kill -0 "$GNOME_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[1/4] GNoME inference completed!"
else
    echo "[1/4] GNoME process not found (already finished or not started)."
fi

# 2. Run GNoME Final Parity Plots
echo "[2/4] Generating final GNoME Parity Plots and Polarity Summaries..."
/home/vibhan23/venv/bin/python /home/vibhan23/generate_gnome_parity_plots.py

# 3. Launch Alexandria Inference
echo "[3/4] Starting Alexandria High-Throughput Inference..."
/home/vibhan23/venv/bin/python /home/vibhan23/run_alexandria_filtered_production.py > /home/vibhan23/alexandria_production_run.log 2>&1

echo "[4/4] Pipeline finished successfully!"
