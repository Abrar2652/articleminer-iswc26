#!/usr/bin/env python3
"""
Error-taxonomy analysis.

Classifies failure modes of a prediction run against gold into a
6-category taxonomy so the paper can report a reviewer-friendly
error analysis beyond aggregate P/R/F1:

    VAL_MISMATCH  – predicted value does not match any gold value
    VAL_MATCH_ATTR_WRONG – value found but type/unit/target wrong
    OVER_EXTRACT  – predicted value has no gold counterpart
    MISSED        – gold value absent from predictions
    NEAR_MISS     – value within 5% but formatting differs
    SCHEMA_ERR    – predicted JSON missing required fields

Reports per-domain × per-category counts plus a few illustrative
examples per category.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from extract_and_eval import _extract_value_core, _normalize_str, _attr_match

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/schema_driven_ie/data"


def _pdf_key(tid: str, domain: str) -> str:
    if domain == "mltables":
        return tid.rsplit("_table", 1)[0]
    return tid.split("::")[0]


def _gold_bag(domain: str) -> dict[str, list]:
    bag: dict[str, list] = defaultdict(list)
    with open(DATA / f"{domain}/test.json") as f:
        data = json.load(f)
    for tid, entry in data.items():
        for g in entry.get("cell_list_gold", []):
            bag[_pdf_key(tid, domain)].append(g)
    return dict(bag)


def _load_preds(pred_dir: Path) -> dict[str, list]:
    out = {}
    for f in pred_dir.glob("*.json"):
        if f.name in ("summary.json", "evidence.json"):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        p = (j.get("predictions") or j.get("extracted") or j.get("tuples") or [])
        if isinstance(p, list):
            out[f.stem] = p
    return out


def _gold_vals(g, domain):
    if domain == "discomat" and isinstance(g, list) and len(g) >= 4:
        return _extract_value_core(str(g[2])), _normalize_str(str(g[1])), _normalize_str(str(g[3]))
    if isinstance(g, dict):
        return (_extract_value_core(str(g.get("value", ""))),
                _normalize_str(str(g.get("type", ""))),
                _normalize_str(str(g.get("unit", ""))))
    return None, None, None


def _pred_vals(p, domain):
    if domain == "discomat":
        return (_extract_value_core(str(p.get("value", ""))),
                _normalize_str(str(p.get("component", ""))),
                _normalize_str(str(p.get("unit", ""))))
    return (_extract_value_core(str(p.get("value", ""))),
            _normalize_str(str(p.get("type", ""))),
            _normalize_str(str(p.get("unit", ""))))


def classify(domain: str, pred_dir: Path, max_examples: int = 2):
    gold_bag = _gold_bag(domain)
    preds    = _load_preds(pred_dir)

    counts = defaultdict(int)
    examples = defaultdict(list)

    for pdf, gold in gold_bag.items():
        pred = preds.get(pdf, [])

        gold_keys = [_gold_vals(g, domain) for g in gold]
        pred_keys = [_pred_vals(p, domain) for p in pred]

        # Required-field check (schema)
        required = {
            "chemtables": ("value", "type"),
            "discomat":   ("value", "component"),
            "mltables":   ("value", "type"),
        }[domain]
        for i, p in enumerate(pred):
            if not all(p.get(k) for k in required):
                counts["SCHEMA_ERR"] += 1
                if len(examples["SCHEMA_ERR"]) < max_examples:
                    examples["SCHEMA_ERR"].append({"pdf": pdf, "pred": p})

        # Value-level matching
        gmatched = [False] * len(gold_keys)
        pmatched = [False] * len(pred_keys)
        for pi, pk in enumerate(pred_keys):
            if pmatched[pi] or pk[0] is None: continue
            for gi, gk in enumerate(gold_keys):
                if gmatched[gi] or gk[0] is None: continue
                if pk[0] == gk[0]:
                    gmatched[gi] = pmatched[pi] = True
                    # Value match — now check attributes
                    ok = pk[1] == gk[1] and pk[2] == gk[2]
                    if ok:
                        pass  # correct; no error
                    else:
                        counts["VAL_MATCH_ATTR_WRONG"] += 1
                        if len(examples["VAL_MATCH_ATTR_WRONG"]) < max_examples:
                            examples["VAL_MATCH_ATTR_WRONG"].append({
                                "pdf": pdf, "pred_key": pk, "gold_key": gk})
                    break

        # Near-miss check (numeric within 5%)
        import re
        def _as_float(s):
            m = re.search(r"[-+]?\d*\.?\d+", str(s))
            return float(m.group()) if m else None
        for pi, pk in enumerate(pred_keys):
            if pmatched[pi]: continue
            pv = _as_float(pk[0])
            if pv is None: continue
            for gi, gk in enumerate(gold_keys):
                if gmatched[gi]: continue
                gv = _as_float(gk[0])
                if gv is None: continue
                if abs(pv - gv) / max(abs(gv), 1e-9) < 0.05:
                    counts["NEAR_MISS"] += 1
                    gmatched[gi] = pmatched[pi] = True
                    if len(examples["NEAR_MISS"]) < max_examples:
                        examples["NEAR_MISS"].append({
                            "pdf": pdf, "pred": pk[0], "gold": gk[0]})
                    break

        # Unmatched preds (over-extractions)
        for pi, pk in enumerate(pred_keys):
            if pmatched[pi]: continue
            counts["OVER_EXTRACT"] += 1
            if len(examples["OVER_EXTRACT"]) < max_examples:
                examples["OVER_EXTRACT"].append({"pdf": pdf, "pred_key": pk})
        # Unmatched golds (missed)
        for gi, gk in enumerate(gold_keys):
            if gmatched[gi]: continue
            counts["MISSED"] += 1
            if len(examples["MISSED"]) < max_examples:
                examples["MISSED"].append({"pdf": pdf, "gold_key": gk})

    return dict(counts), dict(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    choices=["chemtables", "discomat", "mltables"])
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--max-examples", type=int, default=2)
    args = ap.parse_args()

    counts, examples = classify(args.domain, args.pred_dir,
                                max_examples=args.max_examples)
    total = sum(counts.values()) or 1

    print(f"\n[{args.domain}] error taxonomy: {args.pred_dir.name}")
    print(f"  total error events: {total}")
    for cat in ["OVER_EXTRACT", "MISSED", "VAL_MATCH_ATTR_WRONG",
                "NEAR_MISS", "SCHEMA_ERR"]:
        c = counts.get(cat, 0)
        pct = round(100*c/total, 1) if total else 0
        print(f"  {cat:<22} {c:>6}  ({pct}%)")
    print("\n  examples:")
    for cat, ex_list in examples.items():
        for ex in ex_list:
            print(f"    [{cat}] {json.dumps(ex)[:180]}")


if __name__ == "__main__":
    main()
