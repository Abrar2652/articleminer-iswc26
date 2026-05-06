#!/usr/bin/env bash
# v17: Geochem few-shot baselines on the 28 GT papers.
# Closes the §5.2 baseline gap (currently shows --- for all baseline rows).
# Runs after v15+v16 finish.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
LOG=../results/_logs_v17_geochem_baselines
mkdir -p "$LOG"

echo "[$(date '+%H:%M:%S')] v17 waiting for v15+v16..."
while pgrep -af "_v15_v16_wrapper|_v15_preparsed_coverage|_v16_geochem_ablations" > /dev/null; do
    sleep 180
done
echo "[$(date '+%H:%M:%S')] v15+v16 done — starting v17 geochem baselines"

ISWC_DISABLE_MARKER=1 "$PYBIN" -u _v17_geochem_baselines.py \
    > "$LOG/v17.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] v17 ALL DONE"
