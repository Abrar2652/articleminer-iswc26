#!/usr/bin/env python3
"""
v15: Pre-parsed table coverage for missing LLM × domain cells.

Closes these gaps from exp_design:
  E1b: Haiku 4.5 ours pre-parsed on chem/disco/ml
  E1c: GPT-4o ours pre-parsed on chem/disco
  E1d: Gemini 2.5 Flash ours pre-parsed on chem/disco/ml
  G1:  Open-source LLM (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B) on disco/ml
       (we already have Llama on ChemTables-fewshot)

Also fills missing fewshot baselines for these LLMs to enable apples-to-apples
comparison (3-shot for each).

Reuses `run_comprehensive.run_extraction` so output dirs match the existing
naming convention.

Usage:
    python _v15_preparsed_coverage.py
"""
from __future__ import annotations
import os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}")

# Load API keys from .env
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v.strip().strip('"').strip("'")

from run_comprehensive import run_extraction, load_dataset


# =============================================================================
# Coverage matrix to fill (model, prompt_type, shots, domain)
# =============================================================================
def collect_jobs():
    jobs = []

    # Closed-LLM ours-pre-parsed (E1b/c/d)
    for model in ["haiku", "gpt-4o", "gemini"]:
        for dom in ["chemtables", "discomat", "mltables"]:
            jobs.append((model, "ours", 0, dom))

    # Closed-LLM fewshot 3-shot (gaps in E2)
    # We already have Sonnet/Opus 3-shot on all 3, GPT-4o 3-shot on all 3,
    # Gemini 3-shot only on chem, Haiku 3-shot only on chem.
    for dom in ["discomat", "mltables"]:
        jobs.append(("haiku",  "fewshot", 3, dom))
        jobs.append(("gemini", "fewshot", 3, dom))

    # Open-source LLM ours-pre-parsed (G1)
    for model in ["llama3-8b", "qwen25-7b", "mistral-7b"]:
        for dom in ["chemtables", "discomat", "mltables"]:
            jobs.append((model, "ours", 0, dom))
            jobs.append((model, "fewshot", 3, dom))

    # Skip jobs whose output dir already exists with full files
    return jobs


def main():
    jobs = collect_jobs()
    print(f"v15: planning {len(jobs)} pre-parsed runs")
    t0 = time.time()
    for i, (model, kind, shots, dom) in enumerate(jobs, 1):
        suffix = f"_{shots}shot" if kind == "fewshot" else ""
        out_dir_name = f"{dom}_{kind}_{model}{suffix}"
        out_dir = ROOT.parent / "results" / out_dir_name
        if out_dir.exists():
            n = len([f for f in out_dir.iterdir()
                     if f.suffix == ".json"
                     and f.name not in ("summary.json", "evidence.json")])
            data = load_dataset(dom)
            target = len(data)
            if n >= int(0.9 * target):
                print(f"[{i}/{len(jobs)}] SKIP {out_dir_name} (exists, {n}/{target})")
                continue
        print(f"\n[{i}/{len(jobs)}] {model}/{kind}/{shots}/{dom} → {out_dir_name}")
        try:
            data = load_dataset(dom)
            run_extraction(dom, data, model, kind, shots=shots)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    elapsed = time.time() - t0
    print(f"\nv15 ALL DONE in {elapsed/3600:.1f} hr")


if __name__ == "__main__":
    main()
