#!/usr/bin/env bash
# v18: Re-run OSS-LLM coverage with proper env (HF_HOME on <repo-cluster-path> which
# has 14T free; jinja2>=3.1 from <repo-cluster-path>).
# Open-weight LLMs only (Qwen-2.5-7B, Mistral-7B-v0.2, Phi-3-medium).
# v14 OSS jobs failed: jinja2 too old and <home-path> cache OOM.
set -e
cd "$(dirname "$0")"
PYBIN=/usr/bin/python3.10
export HF_HOME=${REPO_ROOT}/../.hf_cache
export HF_HUB_CACHE=${REPO_ROOT}/../.hf_cache
export PYTHONPATH=${REPO_ROOT}/../.local_pylibs:${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}
export CUDA_VISIBLE_DEVICES=2
export ISWC_DISABLE_MARKER=1   # save VRAM for OSS LLM weights
LOG=../results/_logs_v18_oss_rerun
mkdir -p "$LOG"

echo "[$(date '+%H:%M:%S')] v18 waiting for v15..v17 to finish..."
while pgrep -af "_v15_v16_wrapper|_v17_wrapper|_v15_preparsed|_v16_geochem|_v17_geochem" > /dev/null; do
    sleep 180
done
echo "[$(date '+%H:%M:%S')] prereqs done — starting v18 OSS-LLM PDF-path"

run_fewshot () {
    local dom=$1 mdl=$2 tag=$3
    echo "[$(date '+%H:%M:%S')] [v18] $dom fewshot $mdl ($tag) start"
    "$PYBIN" -u run_fewshot_pdf.py --domain "$dom" --model "$mdl" --shots 3 --suffix v18 \
        > "$LOG/fs_${dom}_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v18] $dom fewshot $mdl ($tag) done"
}
run_pipe () {
    local dom=$1 mdl=$2 tag=$3; shift 3
    local extra="$@"
    echo "[$(date '+%H:%M:%S')] [v18] $dom pipeline $mdl ($tag) start"
    "$PYBIN" -u pipeline_adapter.py --dataset "$dom" --model "$mdl" \
        --backends docling pdfplumber --suffix "v18_${tag}" $extra \
        > "$LOG/pipe_${dom}_${tag}.log" 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [v18] $dom pipeline $mdl ($tag) done"
}

# Qwen-2.5-7B
for dom in chemtables mltables discomat; do
  run_fewshot $dom qwen25-7b qwen
done
run_pipe chemtables qwen25-7b qwen --skip-intelligence
run_pipe mltables   qwen25-7b qwen
run_pipe discomat   qwen25-7b qwen

# Mistral-7B-Instruct-v0.2  (open weights)
for dom in chemtables mltables discomat; do
  run_fewshot $dom mistral-7b-v02 mistral
done
run_pipe chemtables mistral-7b-v02 mistral --skip-intelligence
run_pipe mltables   mistral-7b-v02 mistral
run_pipe discomat   mistral-7b-v02 mistral

# Phi-3-medium
for dom in chemtables mltables discomat; do
  run_fewshot $dom phi-3-medium phi3
done
run_pipe chemtables phi-3-medium phi3 --skip-intelligence
run_pipe mltables   phi-3-medium phi3
run_pipe discomat   phi-3-medium phi3

echo "[$(date '+%H:%M:%S')] v18 ALL DONE"
