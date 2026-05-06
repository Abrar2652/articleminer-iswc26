#!/usr/bin/env bash
# v7: ChemTables + MLTables pipelines with --skip-intelligence (per v5b ablation).
# Runs after v6 DiSCoMaT to avoid CUDA contention.
# Marker disabled (CUDA driver too old on ckg08 → CPU OCR is too slow).
set -e
cd "$(dirname "$0")"
PYBIN=${HOME}/.venvs/iswc2026_ckg08/bin/python
LOG=../results/_logs_v7
mkdir -p "$LOG"

V6LOG=../results/_logs_v6_discomat
echo "[$(date '+%H:%M:%S')] v7 waiting for v6 discomat pipelines..."
while true; do
    s=0; g=0
    grep -q "^TOTAL:" "$V6LOG/pipe_discomat_sonnet.log" 2>/dev/null && s=1
    grep -q "^TOTAL:" "$V6LOG/pipe_discomat_gpt4o.log"  2>/dev/null && g=1
    [ $s -eq 1 ] && [ $g -eq 1 ] && break
    sleep 60
done
echo "[$(date '+%H:%M:%S')] v6 done — starting v7"

run_one () {
    local dom=$1 mdl=$2 tag=$3
    echo "=== [v7] $dom $mdl (no-intelligence) start $(date '+%H:%M:%S') ==="
    ISWC_DISABLE_MARKER=1 "$PYBIN" -u pipeline_adapter.py \
        --dataset "$dom" --model "$mdl" \
        --backends docling pdfplumber \
        --suffix v7_noint --skip-intelligence \
        > "$LOG/pipe_${dom}_${tag}_noint.log" 2>&1 || true
    echo "=== [v7] $dom $mdl done $(date '+%H:%M:%S') ==="
}

run_one chemtables claude-sonnet-4-6 sonnet
run_one chemtables gpt-4o            gpt4o
run_one mltables   claude-sonnet-4-6 sonnet
run_one mltables   gpt-4o            gpt4o

echo "[$(date '+%H:%M:%S')] v7 ALL DONE"
