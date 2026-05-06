#!/usr/bin/env python3
"""
PDF-path evaluator.

Evaluates PDF-level prediction files (one JSON per PDF, each with a
``predictions`` list) against per-table gold, by aggregating all tables
of a PDF into a single gold bag and computing micro-F1 over the bag.

This is the fair evaluation for end-to-end PDF→output experiments:
- Pipeline runs (`*_pipeline_*`) emit PDF-level predictions.
- Raw-PDF-text few-shot runs (`*_fewshot_pdf_*`) also emit PDF-level.
Both sides are evaluated identically here.
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from extract_and_eval import _extract_value_core, _normalize_str, _attr_match

DATA_DIR = Path(__file__).parent.parent / "datasets"
RESULTS = Path(__file__).parent.parent / "results"


def _pdf_key_for(tid: str, domain: str) -> str:
    if domain == "chemtables":     # tid = "PMC1234567::table_0"
        return tid.split("::")[0]
    if domain == "discomat":       # tid = "S012345::table_0"
        return tid.split("::")[0]
    if domain == "mltables":       # tid = "2210.00044v1_table0"
        return tid.rsplit("_table", 1)[0]
    return tid


def load_gold(domain: str) -> dict[str, list]:
    """Return {pdf_id: [gold cells aggregated over all its tables]}."""
    path = DATA_DIR / f"schema_driven_ie/data/{domain}/test.json"
    with open(path) as f:
        data = json.load(f)
    bag: dict[str, list] = defaultdict(list)
    for tid, entry in data.items():
        golds = entry.get("cell_list_gold", [])
        if not golds:
            continue
        pdf_id = _pdf_key_for(tid, domain)
        for g in golds:
            bag[pdf_id].append(g)
    return dict(bag)


def load_preds(pred_dir: Path, domain: str, pdf_ids: set[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for pdf in pdf_ids:
        f = pred_dir / f"{pdf}.json"
        if not f.exists():
            out[pdf] = []
            continue
        try:
            data = json.load(open(f))
        except Exception:
            out[pdf] = []
            continue
        if isinstance(data, list):
            preds = data
        elif isinstance(data, dict):
            preds = (data.get("predictions") or data.get("extracted")
                     or data.get("tuples") or [])
        else:
            preds = []
        out[pdf] = preds
    return out


# =============================================================================
# Per-domain scoring  (identical semantics to table-level evaluators)
# =============================================================================
def score_chemtables(pred: list[dict], gold: list[dict]) -> tuple[int, int, int]:
    """Bag-level value-then-attribute match (mirrors eval_chemtables)."""
    gm = [False] * len(gold); pm = [False] * len(pred); pairs = []
    # Pass 1: value + type
    for pi, p in enumerate(pred):
        pv = _extract_value_core(str(p.get("value", "")))
        pt = _normalize_str(str(p.get("type", "")))
        for gi, g in enumerate(gold):
            if gm[gi]: continue
            gv = _extract_value_core(str(g.get("value", "")))
            gt = _normalize_str(str(g.get("type", "")))
            if pv == gv and pt == gt:
                gm[gi] = pm[pi] = True; pairs.append((pi, gi)); break
    # Pass 2: value alone
    for pi, p in enumerate(pred):
        if pm[pi]: continue
        pv = _extract_value_core(str(p.get("value", "")))
        for gi, g in enumerate(gold):
            if gm[gi]: continue
            if pv == _extract_value_core(str(g.get("value", ""))):
                gm[gi] = pm[pi] = True; pairs.append((pi, gi)); break

    tp = fp = fn = 0
    for pi, gi in pairs:
        attrs_match = True
        for k in ("type", "target", "treatment", "unit"):
            if not _attr_match(str(pred[pi].get(k, "xx")), str(gold[gi].get(k, "xx"))):
                attrs_match = False; break
        if attrs_match:
            tp += 1
        else:
            fp += 1; fn += 1
    fp += sum(1 for i in range(len(pred)) if not pm[i])
    fn += sum(1 for i in range(len(gold)) if not gm[i])
    return tp, fp, fn


def score_discomat(pred: list[dict], gold: list) -> tuple[int, int, int]:
    """Match by (component, value, unit) — ignore sample_id (matches eval_discomat)."""
    def norm_g(g):
        # gold is [sample_id, component, value, unit]
        if isinstance(g, list) and len(g) >= 4:
            return (_normalize_str(str(g[1])), _extract_value_core(str(g[2])),
                    _normalize_str(str(g[3])))
        if isinstance(g, dict):
            return (_normalize_str(str(g.get("component", ""))),
                    _extract_value_core(str(g.get("value", ""))),
                    _normalize_str(str(g.get("unit", ""))))
        return None

    def norm_p(p):
        return (_normalize_str(str(p.get("component", ""))),
                _extract_value_core(str(p.get("value", ""))),
                _normalize_str(str(p.get("unit", ""))))

    gold_keys = [norm_g(g) for g in gold]
    pred_keys = [norm_p(p) for p in pred]
    gm = [False] * len(gold_keys); pm = [False] * len(pred_keys)
    # Exact 3-tuple
    for pi, pk in enumerate(pred_keys):
        for gi, gk in enumerate(gold_keys):
            if gm[gi] or pk is None or gk is None: continue
            if pk == gk:
                gm[gi] = pm[pi] = True; break
    # Fallback: component + value (ignore unit)
    for pi, pk in enumerate(pred_keys):
        if pm[pi] or pk is None: continue
        for gi, gk in enumerate(gold_keys):
            if gm[gi] or gk is None: continue
            if pk[0] == gk[0] and pk[1] == gk[1]:
                gm[gi] = pm[pi] = True; break
    tp = sum(1 for m in pm if m)
    fp = sum(1 for m in pm if not m)
    fn = sum(1 for m in gm if not m)
    return tp, fp, fn


def score_mltables(pred: list[dict], gold: list[dict]) -> tuple[int, int, int]:
    """MLTables bag match mirroring eval logic in run_comprehensive.py."""
    other_vals = set(
        _extract_value_core(str(g.get("value", "")))
        for g in gold if g.get("type") == "Other"
    )
    gold = [g for g in gold if g.get("type") != "Other"]
    pred = [p for p in pred
            if p.get("type") != "Other"
            and _extract_value_core(str(p.get("value", ""))) not in other_vals]

    gm = [False] * len(gold); pm = [False] * len(pred); pairs = []
    for pi, p in enumerate(pred):
        pv = _extract_value_core(str(p.get("value", "")))
        pt = _normalize_str(str(p.get("type", "")))
        for gi, g in enumerate(gold):
            if gm[gi]: continue
            if (pv == _extract_value_core(str(g.get("value", ""))) and
                pt == _normalize_str(str(g.get("type", "")))):
                gm[gi] = pm[pi] = True; pairs.append((pi, gi)); break
    for pi, p in enumerate(pred):
        if pm[pi]: continue
        pv = _extract_value_core(str(p.get("value", "")))
        for gi, g in enumerate(gold):
            if gm[gi]: continue
            if pv == _extract_value_core(str(g.get("value", ""))):
                gm[gi] = pm[pi] = True; pairs.append((pi, gi)); break

    tp = fp = fn = 0
    for pi, gi in pairs:
        if _attr_match(str(pred[pi].get("type", "xx")),
                       str(gold[gi].get("type", "xx"))):
            tp += 1
        else:
            fp += 1; fn += 1
    fp += sum(1 for i in range(len(pred)) if not pm[i])
    fn += sum(1 for i in range(len(gold)) if not gm[i])
    return tp, fp, fn


SCORER = {
    "chemtables": score_chemtables,
    "discomat":   score_discomat,
    "mltables":   score_mltables,
}


def eval_pdf_path(domain: str, pred_dir: Path) -> dict:
    gold_bag = load_gold(domain)
    preds = load_preds(pred_dir, domain, set(gold_bag))

    total_tp = total_fp = total_fn = 0
    per_pdf = {}
    for pdf_id, gold in gold_bag.items():
        pred = preds.get(pdf_id, [])
        tp, fp, fn = SCORER[domain](pred, gold)
        total_tp += tp; total_fp += fp; total_fn += fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_pdf[pdf_id] = {"tp": tp, "fp": fp, "fn": fn,
                           "P": round(p * 100, 1), "R": round(r * 100, 1),
                           "F1": round(f1 * 100, 1),
                           "n_pred": len(pred), "n_gold": len(gold)}

    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "domain": domain,
        "pred_dir": str(pred_dir),
        "n_pdfs": len(gold_bag),
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "precision": round(p * 100, 1),
        "recall":    round(r * 100, 1),
        "f1":        round(f1 * 100, 1),
        "per_pdf":   per_pdf,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    choices=["chemtables", "discomat", "mltables"])
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    r = eval_pdf_path(args.domain, args.pred_dir)
    print(f"\n[{args.domain}] {args.pred_dir.name}")
    print(f"  n_pdfs={r['n_pdfs']}  TP={r['total_tp']} FP={r['total_fp']} FN={r['total_fn']}")
    print(f"  P={r['precision']}  R={r['recall']}  F1={r['f1']}")
    if args.verbose:
        for pdf, m in r["per_pdf"].items():
            print(f"    {pdf}: P={m['P']} R={m['R']} F1={m['F1']} "
                  f"({m['n_pred']}p vs {m['n_gold']}g)")
