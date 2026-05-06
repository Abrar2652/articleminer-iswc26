#!/usr/bin/env bash
# v9: BEST POSSIBLE pipeline runs on ckg12 (full GPU resources).
#
# Setup:
#   - All 5 PDF table backends (docling, marker, mineru, camelot, pdfplumber)
#   - Marker enabled (full text + table extraction quality)
#   - GPU 2 pinned (48 GB free at launch)
#   - Per-domain best configuration from v5b ablation:
#       chemtables : --skip-intelligence (+22.8 F1)
#       mltables   : full pipeline (also runs ablation in parallel)
#       discomat   : full pipeline
#   - Both Sonnet and GPT-4o per domain (best LLM choice)
#   - Few-shot PDF baselines (same LLM, no pipeline)
#   - MLTables 5-way ablation (find best per-domain config)
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2          # 48 GB free at launch
unset ISWC_DISABLE_MARKER              # full marker text extraction enabled
LOG=../results/_logs_v9
mkdir -p "$LOG"

BACKENDS="docling marker mineru pdfplumber"   # camelot is fragile; skip

# ---------- Few-shot PDF baselines (no GPU) — parallel ----------
echo "[$(date '+%H:%M:%S')] launching 3 fs baselines (parallel, marker disabled for speed)"
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain chemtables --model claude-sonnet-4-6 --shots 3 --suffix v9 > "$LOG/fs_chemtables_sonnet.log" 2>&1 < /dev/null & disown
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain mltables   --model claude-sonnet-4-6 --shots 3 --suffix v9 > "$LOG/fs_mltables_sonnet.log"   2>&1 < /dev/null & disown
ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain discomat   --model claude-sonnet-4-6 --shots 3 --suffix v9 > "$LOG/fs_discomat_sonnet.log"   2>&1 < /dev/null & disown

# ---------- Pipeline runs (GPU-bound) — sequential ----------
run_pipe () {
    local dom=$1 mdl=$2 tag=$3; shift 3
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v9] $dom $mdl ($tag) start"
    "$PYBIN" -u pipeline_adapter.py \
        --dataset "$dom" --model "$mdl" \
        --backends $BACKENDS \
        --suffix "v9_${tag}" $extra \
        > "$LOG/pipe_${dom}_$(echo $mdl | sed s/claude-//)_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v9] $dom $mdl ($tag) done"
}

# Best per-domain config × best LLM per domain
run_pipe chemtables claude-sonnet-4-6 noint --skip-intelligence
run_pipe chemtables gpt-4o            noint --skip-intelligence
run_pipe mltables   claude-sonnet-4-6 full
run_pipe mltables   gpt-4o            full
run_pipe discomat   claude-sonnet-4-6 full
run_pipe discomat   gpt-4o            full

echo "[$(date '+%H:%M:%S')] v9 main matrix done; starting MLTables 5-way ablation"

# ---------- MLTables ablations (find best config for ML) ----------
"$PYBIN" -u pipeline_ablations.py \
    --dataset mltables --model claude-sonnet-4-6 \
    --backends $BACKENDS \
    --ablations no_ontology no_self_correct no_vision no_validation no_intelligence \
    > "$LOG/ablations_mltables.log" 2>&1 || true

echo "[$(date '+%H:%M:%S')] v9 ALL DONE"
