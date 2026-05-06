#!/usr/bin/env bash
# Re-launch DiSCoMaT experiments after PDF corpus fix.
# Uses the ckg08-specific venv at ${HOME}/.venvs/iswc2026_ckg08.
#
# Cluster-aware: this venv is per-cluster. If you switch clusters again,
# create a new venv and update PYBIN below.
set -euo pipefail

PYBIN=${HOME}/.venvs/iswc2026_ckg08/bin/python
[ -x "$PYBIN" ] || { echo "ERROR: $PYBIN not found. Activate or recreate the venv."; exit 1; }

cd "$(dirname "$0")"
LOG=../results/_logs_v6_discomat
mkdir -p "$LOG"

echo "[$(date '+%H:%M:%S')] launching v6 DiSCoMaT runs (PYBIN=$PYBIN)"

# 1. Few-shot PDF baseline (no GPU) — start immediately
nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain discomat --model claude-sonnet-4-6 \
    --shots 3 --suffix v6 \
    > "$LOG/fs_discomat.log" 2>&1 < /dev/null &
disown
FS_PID=$!
echo "  fs_discomat PID=$FS_PID"

# 2. Pipeline chain: Sonnet → GPT-4o (sequential to avoid CUDA contention)
nohup setsid bash -c "
  set -e
  cd ${REPO_ROOT}/scripts
  echo '=== [v6] discomat pipeline Sonnet start \$(date +%H:%M:%S) ==='
  $PYBIN -u pipeline_adapter.py --dataset discomat --model claude-sonnet-4-6 \
      --backends docling pdfplumber --suffix v6 \
      > $LOG/pipe_discomat_sonnet.log 2>&1
  echo '=== [v6] discomat pipeline GPT-4o start \$(date +%H:%M:%S) ==='
  $PYBIN -u pipeline_adapter.py --dataset discomat --model gpt-4o \
      --backends docling pdfplumber --suffix v6 \
      > $LOG/pipe_discomat_gpt4o.log 2>&1
  echo '=== [v6] discomat pipelines done \$(date +%H:%M:%S) ==='
" > "$LOG/pipe_chain.log" 2>&1 < /dev/null &
disown
CHAIN_PID=$!
echo "  pipe_chain PID=$CHAIN_PID"

sleep 3
echo
echo "Process tree:"
pgrep -af "discomat.*v6|pipe_chain" | head -5
echo
echo "Both detached (PPID=1). Safe to close shell."
