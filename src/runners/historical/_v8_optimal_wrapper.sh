#!/usr/bin/env bash
# v8: BEST configs per domain to maximize F1 vs baselines.
#
# Per-domain choices (driven by v5b ablation findings):
#   chemtables : --skip-intelligence (boost +22.8 F1 vs full)
#   mltables   : full pipeline (intelligence helps for ML papers)
#   discomat   : full pipeline (default; first valid run on real PDFs)
#
# For each domain we run BOTH Sonnet and GPT-4o so we report the best LLM.
# Few-shot PDF baselines also re-run for fair same-cluster comparison.
#
# Marker disabled (other users have GPU). Docling auto-falls-back to CPU if needed.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export ISWC_DISABLE_MARKER=1
LOG=../results/_logs_v8
mkdir -p "$LOG"

# Few-shot baselines — all 3 in parallel (no GPU)
echo "[$(date '+%H:%M:%S')] launching 3 fs baselines (parallel)"
nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain chemtables --model claude-sonnet-4-6 --shots 3 --suffix v8 > "$LOG/fs_chemtables_sonnet.log" 2>&1 < /dev/null &
disown
nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain mltables   --model claude-sonnet-4-6 --shots 3 --suffix v8 > "$LOG/fs_mltables_sonnet.log"   2>&1 < /dev/null &
disown
nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain discomat   --model claude-sonnet-4-6 --shots 3 --suffix v8 > "$LOG/fs_discomat_sonnet.log"   2>&1 < /dev/null &
disown

# Pipeline runs — sequential to avoid GPU contention
run_pipe () {
    local dom=$1 mdl=$2 tag=$3; shift 3
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v8] $dom $mdl ($tag) start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset "$dom" --model "$mdl" \
        --backends docling pdfplumber \
        --suffix "v8_${tag}" $extra \
        > "$LOG/pipe_${dom}_$(echo $mdl | sed s/claude-//)_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v8] $dom $mdl ($tag) done"
}

# ChemTables — best with --skip-intelligence
run_pipe chemtables claude-sonnet-4-6 noint --skip-intelligence
run_pipe chemtables gpt-4o            noint --skip-intelligence

# MLTables — full pipeline
run_pipe mltables   claude-sonnet-4-6 full
run_pipe mltables   gpt-4o            full

# DiSCoMaT — full pipeline (first run on valid 111 PDFs)
run_pipe discomat   claude-sonnet-4-6 full
run_pipe discomat   gpt-4o            full

echo "[$(date '+%H:%M:%S')] v8 ALL DONE"
