#!/usr/bin/env bash
# Apr 2026: Table 3 redesign — 12-variant PDF-backend ablation on
# ChemTables, using Haiku 4.5 throughout (matches GeoScholar column).
#
# 5 single-backend + 5 leave-one-out + full + full+dedup
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
LOG=../results/_logs_table3_ablation
mkdir -p "$LOG"

MODEL=claude-haiku-4-5-20251001
DATASET=chemtables
SUFFIX_BASE=v28_t3

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

# Single-backend (5)
run_one single_docling     docling
run_one single_marker      marker
run_one single_mineru      mineru
run_one single_pdfplumber  pdfplumber
run_one single_camelot     camelot

# Leave-one-out from full (5)
run_one loo_docling     marker mineru pdfplumber camelot
run_one loo_marker      docling mineru pdfplumber camelot
run_one loo_mineru      docling marker pdfplumber camelot
run_one loo_pdfplumber  docling marker mineru camelot
run_one loo_camelot     docling marker mineru pdfplumber

# Full (5 backends) — reference
run_one full            docling marker mineru pdfplumber camelot

echo "[$(date '+%H:%M:%S')] [t3-${DATASET}] ALL DONE"
