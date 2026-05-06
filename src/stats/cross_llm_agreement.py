#!/usr/bin/env python3
"""
Cross-LLM agreement on the same papers.

For each pair (Sonnet, Opus, GPT-4o) on each domain, compute the
Jaccard agreement between their predicted tuple sets per paper.
Reports both:
  - Pairwise agreement (mean Jaccard, mean overlap_count_diff)
  - Triple intersection (fraction of tuples found by all 3)

Source dirs:
  chemtables_pipeline_<llm>_v9_noint  (Sonnet) | _v5a (GPT-4o) | _v11_noint (Opus)
  mltables_pipeline_<llm>_v9_full     (Sonnet) | _v5a (GPT-4o) | _v11_full (Opus)
  discomat_pipeline_<llm>_v9_full     (Sonnet) | _v9_full (GPT-4o) | _v11_full (Opus)
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_pdf_path import _pdf_key_for, load_gold
from extract_and_eval import _extract_value_core, _normalize_str

RES = Path(__file__).parent.parent / "results"

DIRS = {
    'chemtables': {
        'Sonnet': 'chemtables_pipeline_claude-sonnet-4-6_v9_noint',
        'Opus':   'chemtables_pipeline_claude-opus-4-6_v11_noint',
        'GPT-4o': 'chemtables_pipeline_gpt-4o_v5a',
    },
    'mltables': {
        'Sonnet': 'mltables_pipeline_claude-sonnet-4-6_v9_full',
        'Opus':   'mltables_pipeline_claude-opus-4-6_v11_full',
        'GPT-4o': 'mltables_pipeline_gpt-4o_v9_full',
    },
    'discomat': {
        'Sonnet': 'discomat_pipeline_claude-sonnet-4-6_v9_full',
        'Opus':   'discomat_pipeline_claude-opus-4-6_v11_full',
        'GPT-4o': 'discomat_pipeline_gpt-4o_v9_full',
    },
}


def tuple_key(p, domain):
    """Canonical key for a single prediction tuple."""
    if domain == "discomat":
        return (_normalize_str(str(p.get("component", ""))),
                _extract_value_core(str(p.get("value", ""))),
                _normalize_str(str(p.get("unit", ""))))
    return (_extract_value_core(str(p.get("value", ""))),
            _normalize_str(str(p.get("type", ""))))


def load_per_paper(dir_path: Path, domain: str) -> dict:
    """Returns {pdf_id: set(tuple_keys)}."""
    out = {}
    if not dir_path.exists():
        return out
    for f in dir_path.glob("*.json"):
        if f.name in ("summary.json", "evidence.json"):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        preds = (j.get("predictions") or j.get("extracted") or
                 j.get("tuples") or [])
        if not isinstance(preds, list):
            continue
        out[f.stem] = set(tuple_key(p, domain) for p in preds)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main():
    print(f"\n{'Domain':<11} {'Pair':<22} {'Mean Jaccard':>13} "
          f"{'Mean |a∩b|/|a|':>16} {'n_papers':>10}")
    print("-" * 75)
    for dom, dirs in DIRS.items():
        per_llm = {llm: load_per_paper(RES / dirn, dom)
                   for llm, dirn in dirs.items()}
        common_pdfs = sorted(set.intersection(
            *(set(p.keys()) for p in per_llm.values()),
        ))
        if not common_pdfs:
            print(f"{dom:<11} no common papers")
            continue

        llms = list(per_llm.keys())
        for i, a in enumerate(llms):
            for b in llms[i+1:]:
                jac_vals = []
                rec_vals = []
                for pdf in common_pdfs:
                    sa = per_llm[a].get(pdf, set())
                    sb = per_llm[b].get(pdf, set())
                    jac_vals.append(jaccard(sa, sb))
                    if sa:
                        rec_vals.append(len(sa & sb) / len(sa))
                mj = sum(jac_vals) / len(jac_vals) if jac_vals else 0
                mr = sum(rec_vals) / len(rec_vals) if rec_vals else 0
                print(f"{dom:<11} {a:<8} vs {b:<10} "
                      f"{mj*100:>11.1f}% {mr*100:>14.1f}% {len(common_pdfs):>10}")

        # 3-way intersection
        if len(llms) == 3:
            inter_frac = []
            for pdf in common_pdfs:
                sets = [per_llm[l].get(pdf, set()) for l in llms]
                u = set.union(*sets) if any(sets) else set()
                if u:
                    i = set.intersection(*sets)
                    inter_frac.append(len(i) / len(u))
            mean_inter = sum(inter_frac) / len(inter_frac) if inter_frac else 0
            print(f"{dom:<11} {'3-way intersect':<22} "
                  f"{mean_inter*100:>11.1f}% {'—':>14} {len(common_pdfs):>10}")
        print()

    print("Interpretation:")
    print("  Higher Jaccard ⇒ LLMs agree more on extracted tuples.")
    print("  Lower 3-way intersect ⇒ LLM choice matters; ensemble would add.")


if __name__ == "__main__":
    main()
