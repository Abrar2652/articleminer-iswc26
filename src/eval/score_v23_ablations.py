#!/usr/bin/env python3
"""Score v23 ablation extractions with the 4-tier Evaluator.

Reads extraction_<paper>.xlsx from each variant dir, matches to
ground_truth_corrected/<paper>.xlsx, and writes batch_metrics.json
per variant + a combined summary CSV.
"""
from __future__ import annotations
import json, sys, traceback, statistics
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))                  # geochem_benchmark package
import pandas as pd
from geochem_benchmark.evaluator import Evaluator

GT_DIR  = ROOT / "geochem_benchmark/ground_truth_corrected"
RES_DIR = ROOT / "iswc2026/results"

VARIANTS = ["no_ontology", "no_self_correct", "no_vision",
            "single_docling", "single_marker", "single_mineru",
            "single_pdfplumber", "single_camelot", "llm_only_numeric"]


def score_variant(variant: str) -> dict:
    vd = RES_DIR / f"geochem_pipeline_abl_{variant}_haiku28"
    if not vd.exists():
        return {"variant": variant, "error": f"no dir: {vd}"}

    per_paper = {}
    for xlsx in sorted(vd.glob("extraction_*.xlsx")):
        paper_id = xlsx.stem.replace("extraction_", "")
        gt_path = GT_DIR / f"{paper_id}.xlsx"
        if not gt_path.exists():
            continue
        try:
            pred_df = pd.read_excel(xlsx, sheet_name=0)
            ev = Evaluator(ground_truth_path=gt_path)
            rep = ev.evaluate_dataframe(pred_df,
                                        model="claude-haiku-4-5-20251001",
                                        provider="claude")
            per_paper[paper_id] = {
                "T1_metadata_%":   round(rep.t1_metadata_score   * 100, 2),
                "T2_numerical_%":  round(rep.t2_numerical_score  * 100, 2),
                "T3_structural_%": round(rep.t3_structural_score * 100, 2),
                "T4_null_%":       round(rep.t4_null_score       * 100, 2),
                "overall_%":       round(rep.overall_score       * 100, 2),
                "precision_%":     round(rep.sample_precision    * 100, 2),
                "recall_%":        round(rep.sample_recall       * 100, 2),
                "f1_%":            round(rep.sample_f1           * 100, 2),
                "predicted_n":     rep.predicted_n_samples,
                "gt_n":            rep.ground_truth_n_samples,
                "matched_n":       rep.matched_samples,
            }
        except Exception as e:
            per_paper[paper_id] = {"error": f"{type(e).__name__}: {e}"}

    # Aggregate
    valid = {k: v for k, v in per_paper.items() if "error" not in v}
    agg = {}
    if valid:
        for m in ["T1_metadata_%", "T2_numerical_%", "T3_structural_%",
                  "T4_null_%", "overall_%",
                  "precision_%", "recall_%", "f1_%"]:
            agg[f"mean_{m}"] = round(statistics.mean([v[m] for v in valid.values()]), 2)
        agg["n_papers"] = len(valid)

    out = {"variant": variant, "model": "claude-haiku-4-5-20251001",
           "aggregate": agg, "per_paper": per_paper}
    (vd / "batch_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main():
    print(f"Scoring {len(VARIANTS)} variants...\n")
    summary = []
    for v in VARIANTS:
        print(f"  {v}...", flush=True, end="")
        try:
            r = score_variant(v)
            a = r.get("aggregate", {})
            if a:
                summary.append({
                    "variant": v,
                    "n": a["n_papers"],
                    "T1": a["mean_T1_metadata_%"],
                    "T2": a["mean_T2_numerical_%"],
                    "T3": a["mean_T3_structural_%"],
                    "T4": a["mean_T4_null_%"],
                    "overall": a["mean_overall_%"],
                    "P": a["mean_precision_%"],
                    "R": a["mean_recall_%"],
                    "F1": a["mean_f1_%"],
                })
                print(f" n={a['n_papers']} overall={a['mean_overall_%']}")
            else:
                print(f" ERROR: no aggregate")
        except Exception as e:
            print(f" FAILED: {e}")
            traceback.print_exc()

    # Print summary table
    print("\n=== Ablation summary (Haiku, n=26) ===")
    print(f"{'variant':<22} {'n':>3} {'T1':>6} {'T2':>6} {'T3':>6} {'T4':>6} {'Ov':>6} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 82)
    for s in summary:
        print(f"{s['variant']:<22} {s['n']:>3} {s['T1']:>6} {s['T2']:>6} {s['T3']:>6} {s['T4']:>6} {s['overall']:>6} {s['P']:>6} {s['R']:>6} {s['F1']:>6}")

    # Also include "full" from gt_eval_v9_haiku
    full = json.load(open(ROOT / "geochem_benchmark/gt_eval_v9_haiku/batch_metrics.json"))
    fa = full["aggregate"]
    print(f"\n{'full (v9_haiku)':<22} {fa['n_papers']:>3} "
          f"{fa['mean_T1_metadata_%']:>6.2f} {fa['mean_T2_numerical_%']:>6.2f} "
          f"{fa['mean_T3_structural_%']:>6.2f} {fa['mean_T4_null_%']:>6.2f} "
          f"{fa['mean_overall_%']:>6.2f}  (full baseline)")

    # Save summary CSV
    csv_path = RES_DIR / "geochem_abl_haiku28_summary.csv"
    import csv
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","n","T1","T2","T3","T4","overall","P","R","F1"])
        w.writeheader()
        for s in summary: w.writerow(s)
    print(f"\nWrote summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
