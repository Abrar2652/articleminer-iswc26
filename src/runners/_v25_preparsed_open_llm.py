#!/usr/bin/env python3
"""Pre-parsed (table-text) few-shot extraction for a single open-source LLM.

Replaces the buggy code path in `run_comprehensive.py` (where Qwen/Mistral
fell through to the API branch and silently returned empty strings, leading
to all-zero pre-parsed F1 in every result file).

Each invocation runs ONE model on ONE benchmark, so we can fan out across
GPUs in parallel. Outputs match the existing `_fewshot_<model>_<k>shot`
directory layout consumed by the figure-regen scripts.

Usage::
    python scripts/_v25_preparsed_open_llm.py --model mistral-7b-v03 --dataset chemtables --shots 3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_adapter import LocalHFClient, _robust_parse_objects

ISWC_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ISWC_ROOT / "datasets" / "schema_driven_ie" / "data"
RESULTS_DIR = ISWC_ROOT / "results"

PROMPTS = {
    "chemtables": (
        "Extract ALL bioactivity measurements (IC50, EC50, GI50, MIC) from this table.\n"
        'For each, output one JSON per line:\n'
        '{"value": "<number>", "type": "<IC50|EC50|GI50|MIC>", '
        '"target": "<protein/cell line>", "treatment": "<compound>", '
        '"unit": "<µM|nM|µg/mL>"}\n'
        "Output JSON only."
    ),
    "discomat": (
        "Extract material compositions from this table.\n"
        'For each composition entry, output one JSON per line:\n'
        '{"sample_id": "<id>", "component": "<oxide formula, e.g. SiO2>", '
        '"value": <number>, "unit": "<mol|wt>"}\n'
        "Output JSON only."
    ),
    "mltables": (
        "Extract quantitative entries from this table.\n"
        'For each, output one JSON per line:\n'
        '{"value": "<number>", "type": "<Result|Data Stat.|Hyper-parameter/Architecture|Other>", '
        '... include attributes like model, dataset, metric, task as appropriate}\n'
        "Output JSON only."
    ),
}

REQUIRED = {
    "chemtables": ("value", "type"),
    "discomat":   ("component", "value"),
    "mltables":   ("value", "type"),
}


def load_dataset(domain: str) -> dict:
    p = DATA_DIR / domain / "test.json"
    raw = json.load(open(p))
    if domain == "discomat":
        raw = {k: v for k, v in raw.items() if len(v.get("cell_list_gold", [])) > 0}
    return raw


def get_table_text(entry: dict, domain: str) -> str:
    if domain == "mltables":
        return entry.get("table_source", entry.get("table_code", "")) or ""
    return entry.get("table_processed", "") or ""


def build_few_shot_examples(data: dict, domain: str, shots: int) -> str:
    """Pull `shots` (table, gold) pairs to prepend as in-context examples."""
    if shots <= 0:
        return ""
    examples = []
    for tid, entry in list(data.items())[:shots]:
        table = get_table_text(entry, domain)
        gold = entry.get("cell_list_gold", [])
        if not table or not gold:
            continue
        gold_lines = "\n".join(json.dumps(g) for g in gold[:5])
        examples.append(f"Table:\n{table[:1000]}\n\nExtraction:\n{gold_lines}")
        if len(examples) >= shots:
            break
    if not examples:
        return ""
    return "\n\nExamples:\n" + "\n---\n".join(examples) + "\n\nNow extract:\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["qwen25-7b", "qwen25-32b",
                            "mistral-7b-v02", "mistral-7b-v03",
                            "llama31-8b"])
    p.add_argument("--dataset", required=True,
                   choices=["chemtables", "discomat", "mltables"])
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--max-tables", type=int, default=None,
                   help="cap tables processed (debug)")
    p.add_argument("--out-suffix", default="v25_open",
                   help="full dir = <dataset>_fewshot_<model>_<shots>shot_<suffix>")
    args = p.parse_args()

    data = load_dataset(args.dataset)
    if args.max_tables:
        data = dict(list(data.items())[:args.max_tables])

    out_dir = RESULTS_DIR / (
        f"{args.dataset}_fewshot_{args.model}_{args.shots}shot_{args.out_suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.model} on {args.dataset} pre-parsed "
          f"({len(data)} tables, {args.shots}-shot) → {out_dir.name} ===")

    print(f"    Loading {args.model} ...")
    t_load = time.time()
    client = LocalHFClient(model=args.model)
    print(f"    Loaded in {time.time() - t_load:.1f}s")

    system = PROMPTS[args.dataset]
    fewshot_block = build_few_shot_examples(data, args.dataset, args.shots)
    if fewshot_block:
        system = system + fewshot_block

    required = REQUIRED[args.dataset]
    total_pred = 0
    n_zero = 0
    t0 = time.time()
    for i, (tid, entry) in enumerate(data.items()):
        if i % 20 == 0 and i > 0:
            print(f"      {i}/{len(data)} tables, {total_pred} preds so far")
        table = get_table_text(entry, args.dataset)
        if not table:
            n_zero += 1
            continue
        # Drop first `shots` tables from the eval set (they were used as examples).
        if i < args.shots:
            continue

        user = f"Table:\n{table}\n\nExtract:"
        try:
            text = client.complete(system, user, max_tokens=4096)
        except Exception as e:
            print(f"      LLM error on {tid}: {e}")
            text = ""

        preds = _robust_parse_objects(text, required=required)
        if args.dataset == "discomat":
            tmp = []
            for o in preds:
                try:
                    o["value"] = float(o["value"])
                    tmp.append(o)
                except (ValueError, TypeError):
                    pass
            preds = tmp

        total_pred += len(preds)
        if not preds:
            n_zero += 1

        safe_tid = tid.replace("::", "_").replace("/", "_")
        json.dump({
            "predictions": preds,
            "raw_response": text[:1000],
        }, open(out_dir / f"{safe_tid}.json", "w"), indent=2)

    elapsed = time.time() - t0
    json.dump({
        "experiment": f"{args.dataset}/fewshot/{args.model}",
        "model": args.model,
        "dataset": args.dataset,
        "shots": args.shots,
        "tables": len(data),
        "total_predictions": total_pred,
        "n_zero": n_zero,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }, open(out_dir / "evidence.json", "w"), indent=2)
    print(f"\n=== TOTAL: {total_pred} predictions across {len(data)} tables "
          f"({n_zero} zero-output) in {elapsed:.0f}s ===")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
