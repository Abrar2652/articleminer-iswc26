#!/usr/bin/env bash
# v6: DiSCoMaT pipeline + fewshot-PDF runs after corpus fix.
# Marker disabled (CUDA too old on ckg08).
set -e
cd "$(dirname "$0")"
PYBIN=${HOME}/.venvs/iswc2026_ckg08/bin/python
LOG=../results/_logs_v6_discomat
mkdir -p "$LOG"

# Few-shot PDF baseline (no GPU needed, no marker either)
echo "[$(date '+%H:%M:%S')] launching fs_discomat (background)"
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py \
    --domain discomat --model claude-sonnet-4-6 \
    --shots 3 --suffix v6 \
    > "$LOG/fs_discomat.log" 2>&1 < /dev/null &
disown
FS=$!

# Pipeline chain — Sonnet then GPT-4o, sequential to avoid GPU contention
echo "[$(date '+%H:%M:%S')] [v6] discomat pipeline Sonnet start"
ISWC_DISABLE_MARKER=1 "$PYBIN" -u pipeline_adapter.py \
    --dataset discomat --model claude-sonnet-4-6 \
    --backends docling pdfplumber --suffix v6 \
    > "$LOG/pipe_discomat_sonnet.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] [v6] discomat pipeline GPT-4o start"
ISWC_DISABLE_MARKER=1 "$PYBIN" -u pipeline_adapter.py \
    --dataset discomat --model gpt-4o \
    --backends docling pdfplumber --suffix v6 \
    > "$LOG/pipe_discomat_gpt4o.log" 2>&1 || true
echo "[$(date '+%H:%M:%S')] [v6] discomat pipelines done"

# Wait for fs to finish too
wait $FS 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] v6 ALL DONE"
