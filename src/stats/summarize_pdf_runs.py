#!/usr/bin/env python3
"""Produce a combined summary table of PDF-path experiments.

Reads results from `{domain}_{kind}_{model}_{suffix}/` directories and
emits a side-by-side comparison of our pipeline vs the PDF-text few-shot
baseline, per domain.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_pdf_path import eval_pdf_path

RES = Path(__file__).parent.parent / "results"


def summarise(model: str = "claude-sonnet-4-6", suffix: str = "v4"):
    domains = ["chemtables", "discomat", "mltables"]
    rows = []
    for d in domains:
        pipe_dir = RES / f"{d}_pipeline_{model}_{suffix}"
        fs_dir   = RES / f"{d}_fewshot_pdf_{model}_{suffix}"
        row = {"domain": d}
        if pipe_dir.exists() and any(pipe_dir.glob("*.json")):
            r = eval_pdf_path(d, pipe_dir)
            row["ours_F1"] = r["f1"]; row["ours_P"] = r["precision"]
            row["ours_R"] = r["recall"]; row["ours_n"] = r["n_pdfs"]
        else:
            row["ours_F1"] = row["ours_P"] = row["ours_R"] = row["ours_n"] = "—"
        if fs_dir.exists() and any(fs_dir.glob("*.json")):
            r = eval_pdf_path(d, fs_dir)
            row["fs_F1"] = r["f1"]; row["fs_P"] = r["precision"]
            row["fs_R"] = r["recall"]
        else:
            row["fs_F1"] = row["fs_P"] = row["fs_R"] = "—"
        rows.append(row)

    # Print ASCII table
    print(f"\n{'='*78}")
    print(f"PDF-path experiments  (model={model}, suffix={suffix})")
    print(f"{'='*78}")
    print(f"{'Domain':<12} {'n':>5} | {'Ours P':>7} {'Ours R':>7} {'Ours F1':>8} "
          f"| {'FS P':>7} {'FS R':>7} {'FS F1':>7} | {'Δ F1':>7}")
    print("-" * 78)
    for r in rows:
        try:
            delta = f"+{r['ours_F1'] - r['fs_F1']:.1f}" if \
                    all(isinstance(r[k], (int, float)) for k in ('ours_F1','fs_F1')) else "—"
        except TypeError:
            delta = "—"
        print(f"{r['domain']:<12} {str(r.get('ours_n','—')):>5} | "
              f"{str(r['ours_P']):>7} {str(r['ours_R']):>7} {str(r['ours_F1']):>8} | "
              f"{str(r['fs_P']):>7} {str(r['fs_R']):>7} {str(r['fs_F1']):>7} | "
              f"{str(delta):>7}")
    print("-" * 78)
    print("Ours = full pipeline (PDF → multi-extractor → ontology → self-correct → vision-fallback → validation)")
    print("FS   = few-shot LLM on raw PDF text (same model, no pipeline)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--suffix", default="v4")
    args = ap.parse_args()
    summarise(args.model, args.suffix)
