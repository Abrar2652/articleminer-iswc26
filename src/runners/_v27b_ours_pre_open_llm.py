#!/usr/bin/env python3
"""Quarry-style "Ours" pre-parsed extraction for one open-source LLM.

Closes the missing 'Ours Pre' cells in Table 1 for open LLMs. Uses the
post-parser-fix `run_extraction(kind="ours")` which feeds pre-parsed
table_text through the Quarry structured-extraction prompt (no examples).

One invocation = ONE model on ALL three datasets sequentially, so each
parallel GPU only loads one HF checkpoint.

Usage::
    python scripts/_v27b_ours_pre_open_llm.py --model qwen25-7b
    python scripts/_v27b_ours_pre_open_llm.py --model mistral-7b-v02
    python scripts/_v27b_ours_pre_open_llm.py --model llama31-8b
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_comprehensive import run_extraction, load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["qwen25-7b", "mistral-7b-v02", "llama31-8b"])
    p.add_argument("--out-suffix", default="v27pre",
                   help="full dir = <dataset>_ours_<model>_<suffix>")
    args = p.parse_args()

    t_total = time.time()
    for domain in ["chemtables", "discomat", "mltables"]:
        out_dir = ROOT / "results" / f"{domain}_ours_{args.model}_{args.out_suffix}"
        if (out_dir / "evidence.json").exists():
            print(f"[SKIP] {out_dir.name} already done")
            continue
        print(f"\n[{domain}] Quarry 'Ours' pre-parsed × {args.model}")
        data = load_dataset(domain)
        run_extraction(domain, data, args.model, "ours", shots=0,
                       out_dir=out_dir)
    print(f"\nALL DONE in {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
