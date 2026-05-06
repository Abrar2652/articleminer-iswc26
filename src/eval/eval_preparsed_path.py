#!/usr/bin/env python3
"""
Pre-parsed-path evaluator.

Mirror of eval_pdf_path.py for the pre-parsed-input experiments. The
preparsed dirs emit one prediction file per gold table (filename:
``{pdf_id}_{tbl_id}_{idx}.json``), so we aggregate predictions across all
tables of a PDF before scoring against the same per-PDF gold bag used by
eval_pdf_path. This guarantees identical scoring semantics across the two
input modalities so PDF/Pre cells in tab:main_pdf are directly
comparable.
"""
from __future__ import annotations
import json, sys, argparse, re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from eval_pdf_path import load_gold, SCORER

DATA_DIR = Path(__file__).parent.parent / "datasets"
RESULTS = Path(__file__).parent.parent / "results"


def _file_to_pdf_id(fname: str, domain: str) -> str | None:
    """Map a per-table prediction filename to its PDF id."""
    if not fname.endswith(".json"):
        return None
    if fname in {"summary.json", "evidence.json"}:
        return None
    stem = fname[:-5]  # strip .json
    if domain == "chemtables" or domain == "discomat":
        # e.g. PMC3241339_tbl1_0.json -> PMC3241339, S001234567_tbl0_0.json -> S001234567
        return stem.split("_")[0]
    if domain == "mltables":
        # e.g. 2210.00044v1_table0_0.json -> 2210.00044v1
        m = re.split(r"_table\d+", stem, maxsplit=1)
        return m[0] if m else None
    return stem


def load_preparsed_preds(pred_dir: Path, domain: str,
                         pdf_ids: set[str]) -> dict[str, list]:
    """Aggregate per-table predictions into per-PDF bags."""
    bag: dict[str, list] = defaultdict(list)
    for f in sorted(pred_dir.iterdir()):
        if not f.is_file():
            continue
        pid = _file_to_pdf_id(f.name, domain)
        if pid is None or pid not in pdf_ids:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if isinstance(data, list):
            preds = data
        elif isinstance(data, dict):
            preds = (data.get("predictions") or data.get("extracted")
                     or data.get("tuples") or [])
        else:
            preds = []
        bag[pid].extend(preds)
    # Ensure all expected PDFs appear (even if empty)
    for pid in pdf_ids:
        bag.setdefault(pid, [])
    return dict(bag)


def eval_preparsed_path(domain: str, pred_dir: Path) -> dict:
    gold = load_gold(domain)
    pdf_ids = set(gold.keys())
    preds = load_preparsed_preds(pred_dir, domain, pdf_ids)

    scorer = SCORER[domain]
    total_tp = total_fp = total_fn = 0
    per_pdf = {}
    n = 0
    for pid in sorted(pdf_ids):
        g = gold[pid]
        p = preds.get(pid, [])
        tp, fp, fn = scorer(p, g)
        total_tp += tp; total_fp += fp; total_fn += fn
        per_pdf[pid] = {"tp": tp, "fp": fp, "fn": fn,
                        "n_pred": len(p), "n_gold": len(g)}
        n += 1
    P = 100.0 * total_tp / max(1, total_tp + total_fp)
    R = 100.0 * total_tp / max(1, total_tp + total_fn)
    F1 = 2 * P * R / max(1e-9, P + R)
    return {"domain": domain, "pred_dir": str(pred_dir),
            "n_pdfs": n,
            "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
            "precision": round(P, 1), "recall": round(R, 1), "f1": round(F1, 1),
            "per_pdf": per_pdf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["chemtables", "mltables", "discomat"])
    ap.add_argument("pred_dir", type=Path)
    args = ap.parse_args()
    r = eval_preparsed_path(args.domain, args.pred_dir)
    print(f"{args.pred_dir.name}: P={r['precision']} R={r['recall']} F1={r['f1']}  ({r['n_pdfs']} PDFs)")


if __name__ == "__main__":
    main()
