#!/usr/bin/env python3
"""
Compute bootstrap 95% CI and paired-bootstrap test for
ours vs best baseline on each PDF-path domain.

Run after v9 results are in.
"""
from __future__ import annotations
import json, random, argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_pdf_path import (
    load_gold, load_preds, score_chemtables, score_discomat, score_mltables,
    _pdf_key_for,
)

SCORER = {"chemtables": score_chemtables,
          "discomat":   score_discomat,
          "mltables":   score_mltables}
RES = Path(__file__).parent.parent / "results"


def per_pdf_f1(domain: str, pred_dir: Path) -> dict[str, float]:
    gold_bag = load_gold(domain)
    preds = load_preds(pred_dir, domain, set(gold_bag))
    f1 = {}
    for pdf_id, gold in gold_bag.items():
        tp, fp, fn = SCORER[domain](preds.get(pdf_id, []), gold)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1[pdf_id] = 2*p*r/(p+r) if p+r else 0
    return f1


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95):
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [values[random.randint(0, n-1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1-ci)/2 * n_boot)]
    hi = means[int((1+ci)/2 * n_boot)]
    return (sum(values)/n, lo, hi)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000):
    """Returns p-value for H0: mean(a) == mean(b)."""
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b)]
    obs_mean = sum(diffs) / len(diffs)
    count_null = 0
    for _ in range(n_boot):
        n = len(diffs)
        sample = [diffs[random.randint(0, n-1)] for _ in range(n)]
        resampled_mean = sum(sample) / n
        if obs_mean >= 0:
            if resampled_mean <= 0:
                count_null += 1
        else:
            if resampled_mean >= 0:
                count_null += 1
    p = count_null / n_boot
    return obs_mean, p


def main():
    random.seed(42)
    print(f"\n{'domain':<11} {'system':<40} {'mean':>6} {'95%CI_lo':>10} {'95%CI_hi':>10}")
    print("-" * 85)
    pairs = [
        ('chemtables',
         'chemtables_pipeline_claude-sonnet-4-6_v9_noint', 'Ours Sonnet',
         'chemtables_fewshot_pdf_claude-sonnet-4-6_v9',   'Fewshot Sonnet 3-shot'),
        ('mltables',
         'mltables_pipeline_claude-sonnet-4-6_v9_full',   'Ours Sonnet',
         'mltables_fewshot_pdf_claude-sonnet-4-6_v9',     'Fewshot Sonnet 3-shot'),
        ('discomat',
         'discomat_pipeline_claude-sonnet-4-6_v9_full',   'Ours Sonnet',
         'discomat_fewshot_pdf_claude-sonnet-4-6_v9',     'Fewshot Sonnet 3-shot'),
    ]

    for dom, ours_dir, ours_name, base_dir, base_name in pairs:
        ours_path = RES / ours_dir
        base_path = RES / base_dir
        if not ours_path.exists() or not base_path.exists():
            print(f"{dom:<11} MISSING dir")
            continue
        a = per_pdf_f1(dom, ours_path)
        b = per_pdf_f1(dom, base_path)
        common = sorted(set(a) & set(b))
        a_vals = [a[k] for k in common]
        b_vals = [b[k] for k in common]
        for vals, name in [(a_vals, ours_name), (b_vals, base_name)]:
            m, lo, hi = bootstrap_ci(vals)
            print(f"{dom:<11} {name:<40} {m*100:>6.1f} {lo*100:>10.1f} {hi*100:>10.1f}")
        diff, pval = paired_bootstrap(a_vals, b_vals)
        marker = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else " "))
        print(f"{'':<11} {'  paired diff':<40} {diff*100:>+6.1f} F1   p={pval:.3f}{marker}")
        print()

    print("Significance legend: *** p<0.01  ** p<0.05  * p<0.1")


if __name__ == "__main__":
    main()
