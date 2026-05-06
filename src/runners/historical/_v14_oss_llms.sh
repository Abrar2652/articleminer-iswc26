#!/usr/bin/env bash
# v14: Comprehensive open-source LLM coverage on PDF-path.
#
# Adds open-source models (Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B-v0.3) to
# the pipeline + few-shot baselines on all 3 cross-domains, plus fills
# missing closed-LLM gaps (GPT-4o fewshot, Haiku fewshot+pipeline).
#
# Models tested in this wave:
#   open-source:
#     llama3-8b      (Meta-Llama-3.1-8B-Instruct)
#     qwen25-7b      (Qwen2.5-7B-Instruct)
#     mistral-7b     (Mistral-7B-Instruct-v0.3)
#   closed (filling gaps):
#     gpt-4o         (fewshot-PDF on all 3 — missing in earlier waves)
#     claude-haiku-4-5 (fewshot+pipeline on chemtables)
#
# Per-domain config: chem --skip-intelligence, ml/disco full
# Marker enabled (GPU available); mineru optional.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export PYTHONPATH=${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
unset ISWC_DISABLE_MARKER
LOG=../results/_logs_v14_oss
mkdir -p "$LOG"

# Wait for v9, v11, v12, v13 wrappers to finish
echo "[$(date '+%H:%M:%S')] v14 waiting for prior waves..."
while pgrep -af "_v9_optimal_gpu_wrapper|_v11_opus_rerun|_v12_llm_select_backends|_v13_full_backends_dedup" > /dev/null; do
    sleep 180
done
echo "[$(date '+%H:%M:%S')] prereqs done — starting v14"

BACKENDS="docling marker mineru pdfplumber"

# Few-shot baselines — closed LLMs first (parallel, no GPU contention with each other)
echo "[$(date '+%H:%M:%S')] launching CLOSED-LLM fewshot baselines (parallel)"
for dom in chemtables mltables discomat; do
  ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain $dom --model gpt-4o --shots 3 --suffix v14 > "$LOG/fs_${dom}_gpt4o.log" 2>&1 < /dev/null & disown
  ISWC_DISABLE_MARKER=1 nohup setsid "$PYBIN" -u run_fewshot_pdf.py --domain $dom --model claude-haiku-4-5 --shots 3 --suffix v14 > "$LOG/fs_${dom}_haiku.log" 2>&1 < /dev/null & disown
done

# Open-source LLM fewshot — sequential (each loads ~16GB GPU model)
run_oss_fewshot () {
    local dom=$1 mdl=$2 tag=$3
    echo "[$(date '+%H:%M:%S')] [v14] $dom fewshot $mdl ($tag) start"
    "$PYBIN" -u run_fewshot_pdf.py --domain "$dom" --model "$mdl" --shots 3 --suffix v14 \
        > "$LOG/fs_${dom}_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v14] $dom fewshot $mdl ($tag) done"
}

# Open-source LLM pipelines — sequential (GPU-bound and model-load-heavy)
run_oss_pipe () {
    local dom=$1 mdl=$2 tag=$3; shift 3
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v14] $dom pipeline $mdl ($tag) start"
    "$PYBIN" -u pipeline_adapter.py --dataset "$dom" --model "$mdl" \
        --backends $BACKENDS --suffix "v14_${tag}" $extra \
        > "$LOG/pipe_${dom}_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v14] $dom pipeline $mdl ($tag) done"
}

# Llama-3.1-8B
for dom in chemtables mltables discomat; do
  run_oss_fewshot $dom llama3-8b llama3
done
run_oss_pipe chemtables llama3-8b llama3 --skip-intelligence
run_oss_pipe mltables   llama3-8b llama3
run_oss_pipe discomat   llama3-8b llama3

# Qwen2.5-7B
for dom in chemtables mltables discomat; do
  run_oss_fewshot $dom qwen25-7b qwen
done
run_oss_pipe chemtables qwen25-7b qwen --skip-intelligence
run_oss_pipe mltables   qwen25-7b qwen
run_oss_pipe discomat   qwen25-7b qwen

# Mistral-7B-v0.3
for dom in chemtables mltables discomat; do
  run_oss_fewshot $dom mistral-7b mistral
done
run_oss_pipe chemtables mistral-7b mistral --skip-intelligence
run_oss_pipe mltables   mistral-7b mistral
run_oss_pipe discomat   mistral-7b mistral

# Closed-LLM pipelines — Haiku on chemtables
run_oss_pipe chemtables claude-haiku-4-5 haiku --skip-intelligence

echo "[$(date '+%H:%M:%S')] v14 ALL DONE"
