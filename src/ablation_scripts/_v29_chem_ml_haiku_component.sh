#!/usr/bin/env bash
# May 2026: T2 — re-run ChemTables and MLTables component ablations on
# Haiku 4.5 so the entire T2 row uses one model (matches DiSCoMaT/GeoScholar).
set -e
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}:${REPO_ROOT}/..
LOG=${REPO_ROOT}/results/_logs_t2_chem_ml_haiku
mkdir -p "$LOG"

MODEL=claude-haiku-4-5-20251001
BACKENDS="docling marker mineru pdfplumber camelot"
SCRIPTDIR=${REPO_ROOT}/scripts

for DATASET in chemtables mltables; do
    for ABL in no_ontology no_self_correct no_validation no_intelligence no_vision; do
        echo "[$(date '+%H:%M:%S')] [t2-${DATASET}-haiku] $ABL start"
        "$PYBIN" -u "$SCRIPTDIR/pipeline_ablations.py" \
            --dataset "$DATASET" --model "$MODEL" --ablations "$ABL" \
            --backends $BACKENDS \
            > "$LOG/${DATASET}_${ABL}.log" 2>&1 || echo "[$(date '+%H:%M:%S')] [t2-${DATASET}-haiku] $ABL CRASHED"
        echo "[$(date '+%H:%M:%S')] [t2-${DATASET}-haiku] $ABL done"
    done
done

echo "[$(date '+%H:%M:%S')] [t2-chem-ml-haiku] ALL DONE"
