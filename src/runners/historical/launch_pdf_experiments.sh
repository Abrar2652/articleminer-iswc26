#!/usr/bin/env bash
# Launch the PDF-path experiments.
#
# Strategy:
#   - Few-shot PDF baselines (no GPU) run all 3 in parallel.
#   - Pipeline runs (use GPU via docling/marker) run SEQUENTIALLY to
#     avoid CUDA race-init OOM errors seen with parallel docling.
#
# Evaluated by scripts/eval_pdf_path.py after all runs finish.
#
# Usage:
#   bash launch_pdf_experiments.sh [model] [suffix]
#
set -euo pipefail

MODEL="${1:-claude-sonnet-4-6}"
SUFFIX="${2:-v4}"
BACKENDS="docling pdfplumber"

cd "$(dirname "$0")"
LOG_DIR="../results/_logs_${SUFFIX}"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] Launching experiments MODEL=$MODEL SUFFIX=$SUFFIX"
echo "  Logs: $LOG_DIR"

# -----------------------------------------------------------------------
# Few-shot PDF-text baselines — all 3 in parallel (no GPU competition)
# -----------------------------------------------------------------------
nohup python3 -u run_fewshot_pdf.py --domain chemtables --model "$MODEL" \
    --shots 3 --suffix "$SUFFIX" \
    > "$LOG_DIR/fs_chemtables.log" 2>&1 &
echo "  fs_chemtables  PID=$!"

nohup python3 -u run_fewshot_pdf.py --domain mltables --model "$MODEL" \
    --shots 3 --suffix "$SUFFIX" \
    > "$LOG_DIR/fs_mltables.log" 2>&1 &
echo "  fs_mltables    PID=$!"

nohup python3 -u run_fewshot_pdf.py --domain discomat --model "$MODEL" \
    --shots 3 --suffix "$SUFFIX" \
    > "$LOG_DIR/fs_discomat.log" 2>&1 &
echo "  fs_discomat    PID=$!"

# -----------------------------------------------------------------------
# Pipeline runs — sequential, chained in a single background process so
# docling/marker don't race each other on CUDA init.
# -----------------------------------------------------------------------
nohup bash -c "
  set -e
  echo '=== [pipeline] chemtables start $(date +%H:%M:%S) ==='
  python3 -u pipeline_adapter.py --dataset chemtables --model '$MODEL' \
      --backends $BACKENDS --suffix '$SUFFIX' \
      > '$LOG_DIR/pipe_chemtables.log' 2>&1
  echo '=== [pipeline] mltables start $(date +%H:%M:%S) ==='
  python3 -u pipeline_adapter.py --dataset mltables --model '$MODEL' \
      --backends $BACKENDS --suffix '$SUFFIX' \
      > '$LOG_DIR/pipe_mltables.log' 2>&1
  echo '=== [pipeline] discomat start $(date +%H:%M:%S) ==='
  python3 -u pipeline_adapter.py --dataset discomat --model '$MODEL' \
      --backends $BACKENDS --suffix '$SUFFIX' \
      > '$LOG_DIR/pipe_discomat.log' 2>&1
  echo '=== [pipeline] ALL DONE $(date +%H:%M:%S) ==='
" > "$LOG_DIR/pipe_chain.log" 2>&1 &
echo "  pipe_chain (seq chemtables → mltables → discomat) PID=$!"

echo "[$(date '+%H:%M:%S')] Launched. Monitor with:"
echo "    tail -f $LOG_DIR/pipe_chain.log"
echo "    tail -f $LOG_DIR/pipe_chemtables.log"
