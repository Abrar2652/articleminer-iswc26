#!/usr/bin/env bash
# v15 + v16 wrapper: runs after v9-v14 finish.
# Closes most reviewer-flagged gaps in pre-parsed LLM coverage and geochem ablations.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
LOG=../results/_logs_v15_v16
mkdir -p "$LOG"

echo "[$(date '+%H:%M:%S')] v15+v16 waiting for v9..v14..."
while pgrep -af "_v9_optimal_gpu_wrapper|_v11_opus_rerun|_v12_llm_select_backends|_v13_full_backends_dedup|_v14_oss_llms" > /dev/null; do
    sleep 180
done
echo "[$(date '+%H:%M:%S')] prereqs done — starting v15"

ISWC_DISABLE_MARKER=1 "$PYBIN" -u _v15_preparsed_coverage.py \
    > "$LOG/v15_preparsed.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] v15 done — starting v16 geochem ablations"

"$PYBIN" -u _v16_geochem_ablations.py \
    > "$LOG/v16_geochem.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] v15+v16 ALL DONE"
