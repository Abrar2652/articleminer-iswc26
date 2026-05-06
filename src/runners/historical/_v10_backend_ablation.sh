#!/usr/bin/env bash
# v10: backend ablation — does multi-extractor consensus actually help?
#
# We test:
#   single-backend: docling | marker | mineru | pdfplumber | camelot   (5 runs)
#   pair:           docling+pdfplumber                                  (1 run; the v4 default)
#   triple:         docling+marker+pdfplumber                           (1 run)
#   full:           docling+marker+mineru+pdfplumber                    (1 run; same as v9)
#
# Run on ChemTables (9 PDFs — fastest domain) with --skip-intelligence
# (ChemTables' best config from v5b ablation). Single LLM (Sonnet) for clean
# isolation of the backend variable.
#
# All runs sequential to avoid GPU contention with v9.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
LOG=../results/_logs_v10_backends
mkdir -p "$LOG"

# Wait for v9 ChemTables runs to finish so we don't compete on GPU 2
echo "[$(date '+%H:%M:%S')] v10 waiting for v9 chemtables pipelines to finish..."
V9LOG=../results/_logs_v9
while true; do
    s_done=0; g_done=0
    grep -q "ALL DONE\|ALL_DONE" "$V9LOG/wrapper.log" 2>/dev/null && break
    grep -q "discomat gpt-4o.*done"   "$V9LOG/wrapper.log" 2>/dev/null && break  # last run
    sleep 120
done
echo "[$(date '+%H:%M:%S')] v9 done (or main matrix done) — starting v10 backend ablation"

run_one () {
    local tag=$1; shift
    local backends="$@"
    echo "[$(date '+%H:%M:%S')] [v10] backends=$tag start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset chemtables --model claude-sonnet-4-6 \
        --backends $backends \
        --suffix "v10_${tag}" --skip-intelligence \
        > "$LOG/pipe_chemtables_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v10] backends=$tag done"
}

# Single-backend
run_one only_docling     docling
run_one only_marker      marker
run_one only_mineru      mineru
run_one only_pdfplumber  pdfplumber
run_one only_camelot     camelot

# Multi-backend
run_one dual_doc_pdf      docling pdfplumber
run_one tri_doc_marker_pdf docling marker pdfplumber
run_one full_4backends    docling marker mineru pdfplumber

echo "[$(date '+%H:%M:%S')] v10 ALL DONE"
