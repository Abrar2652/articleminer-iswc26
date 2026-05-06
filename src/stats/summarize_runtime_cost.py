#!/usr/bin/env python3
"""
Runtime & cost summary.

Walks result directories and aggregates:
  - wall-clock seconds per paper
  - extrapolated LLM token cost per paper (coarse — based on Anthropic
    pricing for Claude Sonnet 4.6 / Opus 4.6 / Haiku 4.5)
  - number of predictions

Reports a per-domain × per-system table that answers reviewer questions
about efficiency and reproducibility cost.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

RES = Path(__file__).parent.parent / "results"

# Coarse USD-per-1K-token pricing (update if Anthropic changes schedule).
PRICING = {
    "claude-sonnet-4-6":     {"in": 3.0,   "out": 15.0},
    "claude-opus-4-6":       {"in": 15.0,  "out": 75.0},
    "claude-haiku-4-5":      {"in": 0.80,  "out": 4.0},
    "gpt-4o":                {"in": 2.50,  "out": 10.0},
}

# Rough call-count heuristics per paper (for cost extrapolation).
# Our pipeline: 1 intel + N table-filter + N extract [+ ~0.3*N self-correct]
#               + 1 validate + (0.2 probability of vision call).
#    assume ~6 calls per paper at avg 2500 input / 800 output tokens.
# Few-shot PDF: 1 call per paper at ~8000 input / 2000 output tokens.
PIPELINE_TOKENS = {"in": 2500 * 6, "out": 800 * 6}
FEWSHOT_TOKENS  = {"in": 8000,     "out": 2000}


def cost_for(model: str, tokens: dict) -> float:
    """Anthropic / OpenAI prices are quoted per 1M tokens."""
    p = PRICING.get(model, PRICING["claude-sonnet-4-6"])
    return (tokens["in"]  / 1_000_000) * p["in"] + (tokens["out"] / 1_000_000) * p["out"]


def walk_dir(d: Path):
    """Yield (paper_id, elapsed_seconds, n_preds) for every paper json."""
    if not d.exists():
        return
    for f in d.glob("*.json"):
        if f.name in ("summary.json", "evidence.json", "ontology_coverage.json"):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        preds = (j.get("predictions") or j.get("extracted") or
                 j.get("tuples") or [])
        t = j.get("elapsed_seconds")
        if t is None or not isinstance(t, (int, float)):
            continue
        yield f.stem, float(t), len(preds) if isinstance(preds, list) else 0


def summarise(model: str, suffix: str):
    domains = ["chemtables", "discomat", "mltables"]
    rows = []
    for d in domains:
        for kind, tokens_per_paper in (
                ("pipeline", PIPELINE_TOKENS),
                ("fewshot_pdf", FEWSHOT_TOKENS)):
            dir_path = RES / f"{d}_{kind}_{model}_{suffix}"
            data = list(walk_dir(dir_path))
            if not data:
                continue
            times = [t for _, t, _ in data]
            preds = [p for _, _, p in data]
            cost_per_paper = cost_for(model, tokens_per_paper)
            total_cost = cost_per_paper * len(data)
            rows.append({
                "domain": d, "system": kind,
                "n_papers": len(data),
                "mean_s":  round(statistics.mean(times), 1),
                "median_s": round(statistics.median(times), 1),
                "total_min": round(sum(times) / 60, 1),
                "mean_preds": round(statistics.mean(preds), 1),
                "$ per paper": f"${cost_per_paper:.3f}",
                "$ total":     f"${total_cost:.2f}",
            })

    if not rows:
        print(f"No result dirs found for model={model} suffix={suffix}")
        return

    print(f"\n{'='*92}")
    print(f"Runtime & cost summary  (model={model}, suffix={suffix})")
    print(f"{'='*92}")
    hdr = f"{'domain':<12} {'system':<12} {'n':>4} {'mean_s':>7} {'med_s':>6} " \
          f"{'total_min':>10} {'avg_preds':>10} {'$ paper':>10} {'$ total':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['domain']:<12} {r['system']:<12} {r['n_papers']:>4} "
              f"{r['mean_s']:>7} {r['median_s']:>6} "
              f"{r['total_min']:>10} {r['mean_preds']:>10} "
              f"{r['$ per paper']:>10} {r['$ total']:>10}")
    print("-" * len(hdr))
    total_runtime_min = sum(r["total_min"] for r in rows)
    total_cost = sum(float(r["$ total"].lstrip("$")) for r in rows)
    print(f"{'TOTAL':<12} {'':<12} {'':>4} {'':>7} {'':>6} "
          f"{total_runtime_min:>10.1f} {'':>10} {'':>10} ${total_cost:.2f}")
    print()
    print("Cost estimates use coarse token-budget assumptions:")
    print(f"  pipeline/paper: ~{PIPELINE_TOKENS['in']}in + {PIPELINE_TOKENS['out']}out tokens")
    print(f"  fewshot /paper: ~{FEWSHOT_TOKENS['in']}in + {FEWSHOT_TOKENS['out']}out tokens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--suffix", default="v4")
    args = ap.parse_args()
    summarise(args.model, args.suffix)
