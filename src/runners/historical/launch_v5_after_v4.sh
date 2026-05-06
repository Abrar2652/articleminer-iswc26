#!/usr/bin/env bash
# Wait for v4 wave to fully finish, then launch v5a (GPT-4o pipeline)
# and v5b (ChemTables component ablations) in sequence.
#
# Safe to start this script immediately — it blocks until v4 is done.
#
set -euo pipefail
cd "$(dirname "$0")"
LOG=../results/_logs_v5
mkdir -p "$LOG"

echo "[$(date '+%H:%M:%S')] v5 launcher armed. Will wait for v4 to finish."

# -- Wait for all v4 processes to exit --
while pgrep -f "(pipeline_adapter|run_fewshot_pdf)\.py.*v4" > /dev/null; do
    sleep 120
done
echo "[$(date '+%H:%M:%S')] v4 wave finished. Starting v5a."

# -- v5a: GPT-4o pipeline on all 3 domains (sequential, same reason as v4) --
python3 -u pipeline_adapter.py --dataset chemtables --model gpt-4o \
    --backends docling pdfplumber --suffix v5a \
    > "$LOG/pipe_gpt4o_chemtables.log" 2>&1 || true
python3 -u pipeline_adapter.py --dataset mltables --model gpt-4o \
    --backends docling pdfplumber --suffix v5a \
    > "$LOG/pipe_gpt4o_mltables.log" 2>&1 || true
python3 -u pipeline_adapter.py --dataset discomat --model gpt-4o \
    --backends docling pdfplumber --suffix v5a \
    > "$LOG/pipe_gpt4o_discomat.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] v5a done. Starting v5b ablations on chemtables."

# -- v5b: Component ablations on ChemTables (smallest domain — 9 PDFs) --
# Skip "full" since it equals the v4 pipe_chemtables run.
python3 -u pipeline_ablations.py --dataset chemtables --model claude-sonnet-4-6 \
    --backends docling pdfplumber \
    --ablations no_ontology no_self_correct no_vision no_validation no_intelligence \
    > "$LOG/ablations_chemtables.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] v5b done."

echo "=== ALL WAVES COMPLETE $(date '+%H:%M:%S') ==="
