#!/usr/bin/env bash
# May 2026: T2 — DiSCoMaT component ablation (Sonnet 4.6, matches existing
# ChemTables/MLTables rows in tab:ablation). 5 variants × 111 papers.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}:${REPO_ROOT}/..
LOG=../results/_logs_t2_disco_abl
mkdir -p "$LOG"

MODEL=claude-haiku-4-5-20251001
DATASET=discomat

BACKENDS="docling marker mineru pdfplumber camelot"

for ABL in no_ontology no_self_correct no_validation no_intelligence no_vision; do
    echo "[$(date '+%H:%M:%S')] [t2-${DATASET}] $ABL start"
    "$PYBIN" -u pipeline_ablations.py \
        --dataset "$DATASET" --model "$MODEL" --ablations "$ABL" \
        --backends $BACKENDS \
        > "$LOG/abl_${ABL}.log" 2>&1 || echo "[$(date '+%H:%M:%S')] [t2-${DATASET}] $ABL CRASHED"
    echo "[$(date '+%H:%M:%S')] [t2-${DATASET}] $ABL done"
done

echo "[$(date '+%H:%M:%S')] [t2-${DATASET}] ALL DONE"
