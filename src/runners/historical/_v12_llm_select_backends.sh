#!/usr/bin/env bash
# v12: LLM-driven per-paper backend selection.
#
# For every PDF, the LLM peeks at the first 1500 chars and picks the
# OPTIMAL subset of backends (1-5) for that paper. Tests the user's
# hypothesis: "fixed 5-backend isn't always best — LLM should choose."
#
# Per-domain config:
#   chemtables: --skip-intelligence (per v5b ablation), Sonnet+GPT-4o
#   mltables:   full pipeline,                          Sonnet+GPT-4o
#   discomat:   full pipeline,                          Sonnet+GPT-4o
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
unset ISWC_DISABLE_MARKER
LOG=../results/_logs_v12_llmselect
mkdir -p "$LOG"

# Wait for v9, v10, v11 to finish (they share GPU 2)
echo "[$(date '+%H:%M:%S')] v12 waiting for v9, v10, v11..."
while pgrep -af "_v9_optimal_gpu_wrapper|_v10_backend_ablation|_v11_opus_rerun" > /dev/null; do
    sleep 180
done
echo "[$(date '+%H:%M:%S')] prereqs done — starting v12 LLM-select-backends"

# We pass --backends with all 5 to set the AVAILABLE pool;
# LLM then picks subset per paper.
ALL5="docling marker mineru pdfplumber camelot"

run_pipe () {
    local dom=$1 mdl=$2 tag=$3; shift 3
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v12] $dom $mdl ($tag) start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset "$dom" --model "$mdl" \
        --backends $ALL5 --llm-select-backends \
        --suffix "v12_llmsel_${tag}" $extra \
        > "$LOG/pipe_${dom}_$(echo $mdl | sed s/claude-//)_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v12] $dom $mdl ($tag) done"
}

run_pipe chemtables claude-sonnet-4-6 noint --skip-intelligence
run_pipe chemtables gpt-4o            noint --skip-intelligence
run_pipe mltables   claude-sonnet-4-6 full
run_pipe mltables   gpt-4o            full
run_pipe discomat   claude-sonnet-4-6 full
run_pipe discomat   gpt-4o            full

echo "[$(date '+%H:%M:%S')] v12 ALL DONE"
