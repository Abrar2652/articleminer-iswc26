#!/usr/bin/env python3
"""
Cross-Domain Extraction and Evaluation Runner
ISWC 2026 — Ontology-Grounded Consensus Extraction

Runs our ontology-grounded LLM extraction on ChemTables and DiSCoMaT benchmarks,
then evaluates against ground truth using each benchmark's native metrics.

Usage:
  python extract_and_eval.py --dataset chemtables --model claude-sonnet --mode ours
  python extract_and_eval.py --dataset discomat --model claude-sonnet --mode ours
  python extract_and_eval.py --dataset chemtables --model claude-sonnet --mode fewshot --shots 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ISWC_ROOT = SCRIPT_DIR.parent
DATA_DIR = ISWC_ROOT / "datasets"
RESULTS_DIR = ISWC_ROOT / "results"

CHEMTABLES_TEST = DATA_DIR / "schema_driven_ie" / "data" / "chemtables" / "test.json"
DISCOMAT_TEST = DATA_DIR / "schema_driven_ie" / "data" / "discomat" / "test.json"


# ---------------------------------------------------------------------------
# LLM Client (unified interface — same pattern as geochem pipeline)
# ---------------------------------------------------------------------------
def call_llm(system: str, user: str, model: str, max_tokens: int = 4096,
             retries: int = 3) -> str:
    """Call an LLM and return the response text. Retries on transient failures."""
    for attempt in range(retries):
        try:
            return _call_llm_inner(system, user, model, max_tokens)
        except Exception as e:
            print(f"    LLM call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    All retries exhausted, returning empty response")
                return ""


def _call_llm_inner(system: str, user: str, model: str, max_tokens: int) -> str:
    """Inner LLM call without retry logic."""
    if model.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        model_id = {
            "claude-sonnet": "claude-sonnet-4-6",
            "claude-opus": "claude-opus-4-6",
            "claude-haiku": "claude-haiku-4-5-20251001",
        }.get(model, model)
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if not response.content:
            return ""
        return response.content[0].text

    elif model.startswith("gpt"):
        import openai
        client = openai.OpenAI()
        model_id = {
            "gpt-4o": "gpt-4o",
            "gpt-5": "gpt-5.2",
        }.get(model, model)
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    elif model.startswith("gemini"):
        import google.generativeai as genai
        model_id = {
            "gemini-flash": "gemini-2.5-flash",
        }.get(model, model)
        gmodel = genai.GenerativeModel(model_id,
            system_instruction=system)
        response = gmodel.generate_content(user)
        return response.text

    else:
        raise ValueError(f"Unknown model: {model}")


# ---------------------------------------------------------------------------
# ChemTables Extraction + Evaluation
# ---------------------------------------------------------------------------
def load_chemtables():
    """Load ChemTables test set."""
    with open(CHEMTABLES_TEST) as f:
        return json.load(f)


def extract_chemtables_ours(data: dict, model: str, output_dir: Path):
    """Run ontology-grounded extraction on ChemTables."""
    from prompts_chemtables import build_prompt, parse_response
    from ontology_chemtables import standardize_unit, standardize_target, classify_assay_type

    all_predictions = {}
    for tid, entry in data.items():
        print(f"  Extracting {tid}...")
        table_html = entry["table_processed"]

        # Extract caption from the HTML
        caption = ""
        if "<caption>" in table_html:
            start = table_html.index("<caption>") + len("<caption>")
            end = table_html.index("</caption>") if "</caption>" in table_html else start + 200
            caption = table_html[start:end]

        system, user = build_prompt(table_html, caption)
        response = call_llm(system, user, model, max_tokens=8192)
        parsed = parse_response(response)

        # Post-process with ontology
        for p in parsed:
            if "unit" in p:
                p["unit"] = standardize_unit(p["unit"])
            if "target" in p:
                p["target"] = standardize_target(p["target"])

        all_predictions[tid] = parsed

        # Save per-table results
        table_out = output_dir / f"{tid.replace('::', '_')}.json"
        with open(table_out, "w") as f:
            json.dump({"predictions": parsed, "raw_response": response}, f, indent=2)

        time.sleep(0.5)  # rate limiting

    return all_predictions


def extract_chemtables_fewshot(data: dict, model: str, shots: int, output_dir: Path):
    """Run few-shot ICL baseline on ChemTables."""
    # Build few-shot examples from dev set
    dev_path = DATA_DIR / "schema_driven_ie" / "data" / "chemtables" / "dev.json"
    with open(dev_path) as f:
        dev_data = json.load(f)

    # Take first `shots` tables as examples, keeping prompts manageable
    examples = []
    for i, (tid, entry) in enumerate(dev_data.items()):
        if i >= shots:
            break
        table = entry["table_processed"][:400]  # truncate for context
        golds = entry["cell_list_gold"][:3]  # show a few gold annotations
        examples.append(f"TABLE:\n{table}\n\nOUTPUT:\n" +
                       "\n".join(json.dumps(g) for g in golds))

    example_text = "\n\n---\n\n".join(examples)

    system = f"""You are an expert at extracting bioactivity data from chemistry tables.
Given a table, extract all bioactivity measurements as JSON objects.
Each object should have: value, type (IC50/EC50/GI50/MIC), target, treatment, unit.

Here are {shots} examples:

{example_text}

Now extract from the given table. Output one JSON per line."""

    all_predictions = {}
    for tid, entry in data.items():
        print(f"  [few-shot] Extracting {tid}...")
        table_text = entry['table_processed']
        # Truncate very large tables to avoid context overflow
        if len(table_text) > 6000:
            table_text = table_text[:6000] + "\n[... table truncated ...]"
        user = f"TABLE:\n{table_text}\n\nOUTPUT:"
        response = call_llm(system, user, model, max_tokens=8192)

        # Parse response
        parsed = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if "value" in obj and "type" in obj:
                        parsed.append(obj)
                except json.JSONDecodeError:
                    continue

        all_predictions[tid] = parsed
        table_out = output_dir / f"{tid.replace('::', '_')}.json"
        with open(table_out, "w") as f:
            json.dump({"predictions": parsed, "raw_response": response}, f, indent=2)

        time.sleep(0.5)

    return all_predictions


def _normalize_str(s: str) -> str:
    """Normalize a string for comparison — handles Unicode variants, whitespace."""
    import unicodedata
    s = str(s).strip().lower()
    # Normalize Unicode (NFKC maps µ U+00B5 and μ U+03BC to same form,
    # and − U+2212 to - U+002D)
    s = unicodedata.normalize("NFKC", s)
    # Normalize all dash/minus variants to ASCII hyphen
    s = s.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')
    s = s.replace('\u00ad', '-').replace('\uff0d', '-')
    # Strip common punctuation artifacts
    import re
    s = re.sub(r'[{}[\],()@$+%&#_^~|<>\\]', ' ', s)
    s = s.replace('\xa0', ' ')
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _word_f1(gold: str, pred: str) -> float:
    """Word-level F1 between two strings (per Bai et al. 2024 evaluation)."""
    from collections import Counter
    g_words = _normalize_str(gold).split()
    p_words = _normalize_str(pred).split()
    tp = len(list((Counter(g_words) & Counter(p_words)).elements()))
    fp = len(p_words) - tp
    fn = len(g_words) - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0


def _attr_match(gv: str, pv: str, threshold: float = 0.25) -> bool:
    """Check if predicted attribute matches gold, using word-F1 with threshold.
    This matches the evaluation methodology of Bai et al. (EMNLP 2024)."""
    gn = _normalize_str(gv)
    pn = _normalize_str(pv)
    # Gold "xx" means unknown/not-annotated — any prediction is acceptable
    if gn == "xx":
        return True
    # Exact match after normalization
    if gn == pn:
        return True
    # Hyphen-insensitive match (HL60 = HL-60, CCRF-CEM = CCRFCEM)
    if gn.replace("-", "").replace(" ", "") == pn.replace("-", "").replace(" ", ""):
        return True
    # Word-F1 soft match
    return _word_f1(gv, pv) >= threshold


def _extract_value_core(v: str) -> str:
    """Extract core numeric value from a cell value.

    Strips ± uncertainty, parenthesised ranges, NBSP, and a trailing
    unit token (e.g., "127 nM" -> "127"). Comparators ">" / "<" are
    preserved so we can still distinguish ">100" (inactive) from "100".
    """
    import re
    v = str(v).strip()
    v = re.split(r'[±\u00b1]', v)[0].strip()
    v = v.replace('\xa0', '').replace('\u2009', '')
    v = re.split(r'\s*\(', v)[0].strip()
    # If the string is "<number><whitespace><unit-like token>", keep just the number.
    m = re.match(
        r'^\s*([><]?\s*[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*([a-zA-Zμµ%/·\s]{1,15})\s*$',
        v,
    )
    if m:
        return m.group(1).replace(' ', '').strip()
    return v


def eval_chemtables(data: dict, predictions: dict) -> dict:
    """Evaluate ChemTables predictions using attribute-level P/R/F1.
    Uses value-based matching (not cell_index) for robustness against
    different row/column numbering conventions."""
    total_tp, total_fp, total_fn = 0, 0, 0
    per_table = {}

    for tid, entry in data.items():
        gold_list = entry["cell_list_gold"]
        pred_list = predictions.get(tid, [])

        # Multi-pass matching: value + treatment (most unique combo)
        gold_matched = [False] * len(gold_list)
        pred_matched = [False] * len(pred_list)
        match_pairs = []  # (pred_idx, gold_idx)

        def _match_key(item, attrs):
            parts = [_extract_value_core(item.get("value", ""))]
            for a in attrs:
                parts.append(_normalize_str(item.get(a, "xx")))
            return tuple(parts)

        # Pass 1: match by value + treatment (highest specificity)
        for pi, p in enumerate(pred_list):
            pk = _match_key(p, ["treatment"])
            for gi, g in enumerate(gold_list):
                if gold_matched[gi]:
                    continue
                gk = _match_key(g, ["treatment"])
                if pk == gk:
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    match_pairs.append((pi, gi))
                    break

        # Pass 2: match remaining by value + type
        for pi, p in enumerate(pred_list):
            if pred_matched[pi]:
                continue
            pk = _match_key(p, ["type"])
            for gi, g in enumerate(gold_list):
                if gold_matched[gi]:
                    continue
                gk = _match_key(g, ["type"])
                if pk == gk:
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    match_pairs.append((pi, gi))
                    break

        # Pass 3: match remaining by value alone (fallback)
        for pi, p in enumerate(pred_list):
            if pred_matched[pi]:
                continue
            pval = _extract_value_core(p.get("value", ""))
            for gi, g in enumerate(gold_list):
                if gold_matched[gi]:
                    continue
                gval = _extract_value_core(g.get("value", ""))
                if pval == gval:
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    match_pairs.append((pi, gi))
                    break

        tp, fp, fn = 0, 0, 0

        # Evaluate attribute accuracy on matched pairs
        for pi, gi in match_pairs:
            p = pred_list[pi]
            g = gold_list[gi]
            for attr in ["type", "target", "treatment", "unit"]:
                gv = g.get(attr, "xx")
                pv = p.get(attr, "xx")
                if gv == "xx" and pv == "xx":
                    continue
                elif gv == "xx":
                    fp += 1
                elif pv == "xx":
                    fn += 1
                elif _attr_match(gv, pv):
                    tp += 1
                else:
                    fp += 1
                    fn += 1

        # Unmatched predictions (false positives)
        for pi, p in enumerate(pred_list):
            if not pred_matched[pi]:
                fp += sum(1 for attr in ["type", "target", "treatment", "unit"]
                         if p.get(attr, "xx") != "xx")

        # Unmatched gold (false negatives)
        for gi, g in enumerate(gold_list):
            if not gold_matched[gi]:
                fn += sum(1 for attr in ["type", "target", "treatment", "unit"]
                         if g.get(attr, "xx") != "xx")

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_table[tid] = {"precision": round(prec * 100, 1),
                          "recall": round(rec * 100, 1),
                          "f1": round(f1 * 100, 1),
                          "gold_cells": len(gold_list),
                          "pred_cells": len(pred_list)}

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "aggregate": {
            "precision": round(prec * 100, 1),
            "recall": round(rec * 100, 1),
            "f1": round(f1 * 100, 1),
            "total_gold": sum(len(data[t]["cell_list_gold"]) for t in data),
            "total_pred": sum(len(predictions.get(t, [])) for t in data),
        },
        "per_table": per_table,
    }


# ---------------------------------------------------------------------------
# DiSCoMaT Extraction + Evaluation
# ---------------------------------------------------------------------------
def load_discomat():
    """Load DiSCoMaT test set (only tables with gold annotations)."""
    with open(DISCOMAT_TEST) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if len(v.get("cell_list_gold", [])) > 0}


def extract_discomat_ours(data: dict, model: str, output_dir: Path):
    """Run ontology-grounded extraction on DiSCoMaT."""
    from prompts_discomat import build_prompt, parse_response
    from ontology_discomat import standardize_component, infer_unit_from_caption

    all_predictions = {}
    for i, (tid, entry) in enumerate(data.items()):
        if i % 10 == 0:
            print(f"  Extracting {i+1}/{len(data)}: {tid}...")

        table_md = entry["table_processed"]

        # Extract caption
        caption = ""
        table_data = entry.get("table_data_org", {})
        if isinstance(table_data, dict):
            caption = table_data.get("caption", "")
        elif "Caption:" in table_md:
            caption = table_md.split("Caption:")[-1].strip()

        system, user = build_prompt(table_md, caption)
        response = call_llm(system, user, model, max_tokens=8192)
        parsed = parse_response(response)

        # Post-process with ontology
        inferred_unit = infer_unit_from_caption(caption) if caption else None
        for p in parsed:
            # Standardize component
            canonical = standardize_component(p.get("component", ""))
            if canonical:
                p["component"] = canonical
            # Apply inferred unit if missing
            if inferred_unit and (not p.get("unit") or p["unit"] not in ("mol", "wt")):
                p["unit"] = inferred_unit

        all_predictions[tid] = parsed

        table_out = output_dir / f"{tid.replace('::', '_')}.json"
        with open(table_out, "w") as f:
            json.dump({"predictions": parsed, "raw_response": response}, f, indent=2)

        time.sleep(0.3)

    return all_predictions


def extract_discomat_fewshot(data: dict, model: str, shots: int, output_dir: Path):
    """Run few-shot ICL baseline on DiSCoMaT."""
    dev_path = DATA_DIR / "schema_driven_ie" / "data" / "discomat" / "dev.json"
    with open(dev_path) as f:
        dev_data = json.load(f)

    examples = []
    for i, (tid, entry) in enumerate(dev_data.items()):
        if len(entry.get("cell_list_gold", [])) == 0:
            continue
        if len(examples) >= shots:
            break
        table = entry["table_processed"][:400]
        golds = entry["cell_list_gold"][:5]
        gold_strs = [json.dumps({"sample_id": g[0], "component": g[1],
                                  "value": g[2], "unit": g[3]}) for g in golds]
        examples.append(f"TABLE:\n{table}\n\nOUTPUT:\n" + "\n".join(gold_strs))

    example_text = "\n\n---\n\n".join(examples)

    system = f"""You are an expert at extracting material compositions from materials science tables.
Given a composition table, extract each component percentage as a JSON object.
Each object should have: sample_id, component (chemical formula), value (number), unit (mol or wt).

Here are {shots} examples:

{example_text}

Now extract from the given table. Output one JSON per line."""

    all_predictions = {}
    for i, (tid, entry) in enumerate(data.items()):
        if i % 10 == 0:
            print(f"  [few-shot] Extracting {i+1}/{len(data)}: {tid}...")
        user = f"TABLE:\n{entry['table_processed']}\n\nOUTPUT:"
        response = call_llm(system, user, model, max_tokens=8192)

        parsed = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if "component" in obj and "value" in obj:
                        try:
                            obj["value"] = float(obj["value"])
                            parsed.append(obj)
                        except (ValueError, TypeError):
                            continue
                except json.JSONDecodeError:
                    continue

        all_predictions[tid] = parsed
        table_out = output_dir / f"{tid.replace('::', '_')}.json"
        with open(table_out, "w") as f:
            json.dump({"predictions": parsed, "raw_response": response}, f, indent=2)

        time.sleep(0.3)

    return all_predictions


def eval_discomat(data: dict, predictions: dict) -> dict:
    """
    Evaluate DiSCoMaT predictions using component-value-unit matching.
    Gold tuples: [sample_id, component, value, unit]

    Sample IDs are paper-specific encodings that differ between gold and predictions.
    We match by (component, value, unit) within each table, which is the semantically
    meaningful comparison — did we extract the right composition data?
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    per_table = {}

    for tid, entry in data.items():
        # Build gold tuples: (component, value, unit) — ignoring sample_id for matching
        gold_tuples = []
        for g in entry["cell_list_gold"]:
            gold_tuples.append((_normalize_str(str(g[1])), float(g[2]), str(g[3])))

        pred_tuples = []
        for p in predictions.get(tid, []):
            comp = _normalize_str(str(p.get("component", "")))
            val = p.get("value", 0)
            unit = str(p.get("unit", "mol"))
            if comp and val:
                try:
                    pred_tuples.append((comp, float(val), unit))
                except (ValueError, TypeError):
                    continue

        # Greedy matching: match each pred to a gold tuple
        gold_matched = [False] * len(gold_tuples)
        pred_matched = [False] * len(pred_tuples)

        for pi, pt in enumerate(pred_tuples):
            for gi, gt in enumerate(gold_tuples):
                if gold_matched[gi]:
                    continue
                # Match component (normalized), value (exact), unit
                if pt[0] == gt[0] and abs(pt[1] - gt[1]) < 0.01 and pt[2] == gt[2]:
                    gold_matched[gi] = True
                    pred_matched[pi] = True
                    break

        tp = sum(pred_matched)
        fp = len(pred_tuples) - tp
        fn = len(gold_tuples) - sum(gold_matched)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_table[tid] = {"precision": round(prec * 100, 1),
                          "recall": round(rec * 100, 1),
                          "f1": round(f1 * 100, 1),
                          "gold_tuples": len(gold_tuples),
                          "pred_tuples": len(pred_tuples)}

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "aggregate": {
            "precision": round(prec * 100, 1),
            "recall": round(rec * 100, 1),
            "f1": round(f1 * 100, 1),
            "total_gold": sum(len(data[t]["cell_list_gold"]) for t in data),
            "total_pred": sum(len(predictions.get(t, [])) for t in data),
        },
        "per_table": per_table,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Extraction & Evaluation")
    parser.add_argument("--dataset", required=True, choices=["chemtables", "discomat"],
                        help="Which benchmark to run on")
    parser.add_argument("--model", default="claude-sonnet",
                        help="LLM model (claude-sonnet, claude-opus, gpt-4o, gemini-flash)")
    parser.add_argument("--mode", default="ours", choices=["ours", "fewshot"],
                        help="Extraction mode: 'ours' (ontology-grounded) or 'fewshot' (ICL baseline)")
    parser.add_argument("--shots", type=int, default=3,
                        help="Number of few-shot examples (for fewshot mode)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and print stats without running LLM")
    args = parser.parse_args()

    # Setup output directory
    run_name = f"{args.dataset}_{args.mode}_{args.model}"
    if args.mode == "fewshot":
        run_name += f"_{args.shots}shot"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ISWC 2026 Cross-Domain Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    if args.dataset == "chemtables":
        data = load_chemtables()
        print(f"Loaded {len(data)} ChemTables test tables")
        total_gold = sum(len(v["cell_list_gold"]) for v in data.values())
        print(f"Total gold annotations: {total_gold}")
    else:
        data = load_discomat()
        print(f"Loaded {len(data)} DiSCoMaT test tables (with gold)")
        total_gold = sum(len(v["cell_list_gold"]) for v in data.values())
        print(f"Total gold tuples: {total_gold}")

    if args.dry_run:
        print("\n[dry-run] Exiting without LLM calls.")
        return

    # Run extraction
    print(f"\n--- Running extraction ---")
    start_time = time.time()

    if args.dataset == "chemtables":
        if args.mode == "ours":
            predictions = extract_chemtables_ours(data, args.model, output_dir)
        else:
            predictions = extract_chemtables_fewshot(data, args.model, args.shots, output_dir)
        results = eval_chemtables(data, predictions)
    else:
        if args.mode == "ours":
            predictions = extract_discomat_ours(data, args.model, output_dir)
        else:
            predictions = extract_discomat_fewshot(data, args.model, args.shots, output_dir)
        results = eval_discomat(data, predictions)

    elapsed = time.time() - start_time

    # Print results
    print(f"\n=== RESULTS ({args.dataset}, {args.mode}, {args.model}) ===")
    agg = results["aggregate"]
    print(f"Precision: {agg['precision']}%")
    print(f"Recall:    {agg['recall']}%")
    print(f"F1:        {agg['f1']}%")
    print(f"Gold: {agg['total_gold']} | Pred: {agg['total_pred']}")
    print(f"Time: {elapsed:.1f}s")

    # Save results
    results["metadata"] = {
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "shots": args.shots if args.mode == "fewshot" else None,
        "elapsed_seconds": round(elapsed, 1),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
