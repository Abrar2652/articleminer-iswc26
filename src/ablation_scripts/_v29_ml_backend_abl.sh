#!/usr/bin/env bash
# May 2026: T3 — MLTables PDF-backend ablation (Haiku 4.5, matches existing
# ChemTables/GeoScholar columns). 5 single + 5 LOO + 1 full = 11 variants.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}:${REPO_ROOT}/..
LOG=../results/_logs_t3_ml_backend
mkdir -p "$LOG"

MODEL=claude-haiku-4-5-20251001
DATASET=mltables
SUFFIX_BASE=v29_t3

run_one () {
    local tag=$1; shift
    local backends="$@"
    echo "[$(date '+%H:%M:%S')] [t3-${DATASET}] backends=$tag start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset "$DATASET" --model "$MODEL" \
        --backends $backends \
        --suffix "${SUFFIX_BASE}_${tag}" \
        > "$LOG/pipe_${DATASET}_${tag}.log" 2>&1 || echo "[$(date '+%H:%M:%S')] [t3-${DATASET}] $tag CRASHED"
    echo "[$(date '+%H:%M:%S')] [t3-${DATASET}] backends=$tag done"
}

run_one single_docling     docling
run_one single_marker      marker
run_one single_mineru      mineru
run_one single_pdfplumber  pdfplumber
run_one single_camelot     camelot

run_one loo_docling     marker mineru pdfplumber camelot
run_one loo_marker      docling mineru pdfplumber camelot
run_one loo_mineru      docling marker pdfplumber camelot
run_one loo_pdfplumber  docling marker mineru camelot
run_one loo_camelot     docling marker mineru pdfplumber

run_one full            docling marker mineru pdfplumber camelot

echo "[$(date '+%H:%M:%S')] [t3-${DATASET}] ALL DONE"
