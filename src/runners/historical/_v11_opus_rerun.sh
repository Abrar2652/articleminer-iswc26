#!/usr/bin/env bash
# v11: Re-run Opus configs that were broken in earlier runs.
# - chemtables fewshot (was 0/0/0 — silent parse failure)
# - chemtables pipeline (was 12.6 F1 — only 1/9 papers had preds, vision missing)
# - mltables/discomat Opus (for full per-domain best-LLM coverage)
#
# Marker enabled (GPU available on ckg12).
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
unset ISWC_DISABLE_MARKER
LOG=../results/_logs_v11_opus
mkdir -p "$LOG"

# Wait for v9 to finish first (sequential GPU access)
echo "[$(date '+%H:%M:%S')] v11 waiting for v9 wrapper to finish..."
V9LOG=../results/_logs_v9
while pgrep -af "_v9_optimal_gpu_wrapper" > /dev/null; do sleep 120; done
echo "[$(date '+%H:%M:%S')] v9 done — starting v11 Opus reruns"

BACKENDS="docling marker mineru pdfplumber"

# Opus fewshot baselines — parallel, no GPU
echo "[$(date '+%H:%M:%S')] launching 3 Opus fs baselines (parallel)"
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain chemtables --model claude-opus-4-6 --shots 3 --suffix v11 > "$LOG/fs_chemtables_opus.log" 2>&1 < /dev/null & disown
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain mltables   --model claude-opus-4-6 --shots 3 --suffix v11 > "$LOG/fs_mltables_opus.log"   2>&1 < /dev/null & disown
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain discomat   --model claude-opus-4-6 --shots 3 --suffix v11 > "$LOG/fs_discomat_opus.log"   2>&1 < /dev/null & disown

# Opus pipeline runs — sequential
run_pipe () {
    local dom=$1 tag=$2; shift 2
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v11] $dom Opus ($tag) start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset "$dom" --model claude-opus-4-6 \
        --backends $BACKENDS \
        --suffix "v11_${tag}" $extra \
        > "$LOG/pipe_${dom}_opus_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v11] $dom Opus ($tag) done"
}
run_pipe chemtables noint --skip-intelligence
run_pipe mltables   full
run_pipe discomat   full

echo "[$(date '+%H:%M:%S')] v11 ALL DONE"
