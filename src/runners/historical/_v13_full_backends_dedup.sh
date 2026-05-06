#!/usr/bin/env bash
# v13: Tests the user's hypothesis — "with proper dedup, more backends
# should always win over fewer because more backends = more unique info."
#
# Configuration:
#   - All 5 backends (docling + marker + mineru + pdfplumber + camelot)
#   - --llm-dedup: LLM-judge merges cross-backend duplicates
#   - Per-domain best config (chem --skip-intelligence, ml/disco full)
#   - Both Sonnet and GPT-4o per domain
#
# Compares against v9 (4-backend, no LLM-dedup) and v10 (single-backend ablations)
# to isolate the effect of dedup quality vs backend count.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
unset ISWC_DISABLE_MARKER
LOG=../results/_logs_v13_full_dedup
mkdir -p "$LOG"

# Wait for v9, v10, v11, v12 to finish
echo "[$(date '+%H:%M:%S')] v13 waiting for prior waves..."
while pgrep -af "_v9_optimal_gpu_wrapper|_v10_backend_ablation|_v11_opus_rerun|_v12_llm_select_backends" > /dev/null; do
    sleep 180
done
echo "[$(date '+%H:%M:%S')] prereqs done — starting v13 (all 5 backends + LLM dedup)"

ALL5="docling marker mineru pdfplumber camelot"

run_pipe () {
    local dom=$1 mdl=$2 tag=$3; shift 3
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v13] $dom $mdl ($tag) start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset "$dom" --model "$mdl" \
        --backends $ALL5 --llm-dedup \
        --suffix "v13_5b_dedup_${tag}" $extra \
        > "$LOG/pipe_${dom}_$(echo $mdl | sed s/claude-//)_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v13] $dom $mdl ($tag) done"
}

run_pipe chemtables claude-sonnet-4-6 noint --skip-intelligence
run_pipe chemtables gpt-4o            noint --skip-intelligence
run_pipe mltables   claude-sonnet-4-6 full
run_pipe mltables   gpt-4o            full
run_pipe discomat   claude-sonnet-4-6 full
run_pipe discomat   gpt-4o            full

echo "[$(date '+%H:%M:%S')] v13 ALL DONE"
