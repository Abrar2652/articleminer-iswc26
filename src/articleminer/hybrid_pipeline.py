#!/usr/bin/env python3
"""
Hybrid Extraction Pipeline for Cross-Domain Benchmarks
=======================================================
Mirrors the geochem pipeline architecture:
  Stage 1: Deterministic table parsing (table_reader.py equivalent)
  Stage 2: LLM-based header interpretation (prompts.py equivalent)
  Stage 3: Ontology-grounded validation (knowledge_base.py equivalent)

Key principle: LLM NEVER touches numerical values.
  - Numbers come from deterministic parser (perfect recall)
  - LLM interprets column headers (what does each column measure?)
  - Ontology validates and normalizes (is this a valid type/component?)
"""

import json
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from deterministic_parser import (
    parse_html_table, parse_markdown_table, parse_table,
    extract_all_numeric_cells, _is_numeric, extract_raw_value,
)


# =============================================================================
# Stage 2: LLM Header Interpretation (metadata only — never touches values)
# =============================================================================
HEADER_INTERPRETATION_PROMPT = """You are a scientific data interpretation expert. Given column headers from a scientific table, classify each header.

For EACH header, output a JSON object on one line:
{{"col_idx": <int>, "header": "<original header text>", "is_data": true/false, "type": "<category>", "target": "<what is measured against>", "unit": "<measurement unit>"}}

Rules:
- is_data=true for columns containing measurement data (concentrations, activities, compositions)
- is_data=false for columns containing labels, identifiers, structural descriptions, ratios, or non-measurement data
- type: the measurement category (e.g., "IC50", "EC50", "GI50", "MIC", "composition", "physical_property")
- target: what the measurement is against (protein name, cell line, organism, or "xx" if unknown)
- unit: measurement unit from the header (µM, nM, µg/mL, mol%, wt%, or "xx" if unknown)

Output one JSON per header, nothing else."""


def interpret_headers_llm(headers: list[str], caption: str, model: str = "claude-sonnet") -> list[dict]:
    """Use LLM to interpret column headers — what does each column measure?"""
    if not headers:
        return []

    header_list = "\n".join(f"  Column {i}: \"{h}\"" for i, h in enumerate(headers))
    user_msg = f"Caption: {caption}\n\nColumn headers:\n{header_list}\n\nClassify each:"

    import anthropic
    client = anthropic.Anthropic()
    model_id = {"claude-sonnet": "claude-sonnet-4-6", "claude-haiku": "claude-haiku-4-5-20251001"}.get(model, model)

    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=2048,
            system=HEADER_INTERPRETATION_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text if response.content else ""
    except Exception as e:
        print(f"    Header LLM call failed: {e}")
        return []

    # Parse response
    interpretations = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                interpretations.append(obj)
            except json.JSONDecodeError:
                continue

    return interpretations


def interpret_headers_rules(headers: list[str], caption: str, domain: str) -> list[dict]:
    """Rule-based header interpretation using ontology (no LLM).
    Falls back to this when LLM is unavailable or for speed."""

    if domain == "chemtables":
        from ontology_chemtables import classify_assay_type, UNIT_STANDARDIZATION
        results = []
        for i, h in enumerate(headers):
            h_lower = h.lower().strip()
            assay_type = classify_assay_type(h)
            # Detect unit
            unit = "xx"
            for raw, canonical in UNIT_STANDARDIZATION.items():
                if raw in h_lower:
                    unit = canonical
                    break
            # Detect target (everything before the assay type keyword)
            target = "xx"
            if assay_type:
                # Try to extract target from header like "CHK1 IC50 (µM)"
                for kw in ["ic50", "ec50", "gi50", "mic", "ic 50"]:
                    if kw in h_lower:
                        idx = h_lower.index(kw)
                        candidate = h[:idx].strip().rstrip("-").strip()
                        if candidate and len(candidate) > 1:
                            target = candidate
                        break

            is_data = assay_type is not None
            # Also check for % inhibition
            if not is_data and ("inhibit" in h_lower or "% inhib" in h_lower):
                is_data = True
                assay_type = "MIC"
                # Try unit from header
                if "µg/ml" in h_lower or "ug/ml" in h_lower:
                    unit = "µg/mL"
                elif "µm" in h_lower or "um" in h_lower:
                    unit = "µM"

            results.append({
                "col_idx": i,
                "header": h,
                "is_data": is_data,
                "type": assay_type or "xx",
                "target": target,
                "unit": unit,
            })
        return results

    elif domain == "discomat":
        from ontology_discomat import standardize_component, infer_unit_from_header, infer_unit_from_caption
        caption_unit = infer_unit_from_caption(caption) if caption else None

        results = []
        for i, h in enumerate(headers):
            component = standardize_component(h)
            is_data = component is not None

            # Infer unit from header or caption
            unit = infer_unit_from_header(h)
            if not unit:
                unit = caption_unit
            if not unit:
                unit = "mol"  # default

            results.append({
                "col_idx": i,
                "header": h,
                "is_data": is_data,
                "type": "composition" if is_data else "xx",
                "component": component or h.strip(),
                "unit": unit,
            })
        return results

    return []


# =============================================================================
# Stage 3: Ontology Validation & Assembly
# =============================================================================
def assemble_chemtables_tuples(numeric_cells: list[dict],
                                header_info: list[dict],
                                parsed_table: dict) -> list[dict]:
    """Assemble ChemTables extraction tuples from deterministic cells + header info.

    For each numeric cell in a data column:
      - value: from deterministic parser (NEVER hallucinated)
      - type: from header interpretation (IC50/EC50/GI50/MIC)
      - target: from header interpretation
      - treatment: from the row label (compound identifier in first column)
      - unit: from header interpretation
    """
    from ontology_chemtables import standardize_unit, standardize_target

    # Build column classification map
    data_cols = {}
    for hi in header_info:
        if hi.get("is_data", False):
            data_cols[hi["col_idx"]] = hi

    results = []
    for cell in numeric_cells:
        ci = cell["col_idx"]
        if ci not in data_cols:
            continue

        col_info = data_cols[ci]
        assay_type = col_info.get("type", "xx")
        if assay_type == "xx" or assay_type == "Other":
            continue

        raw_val = cell["raw_value"]
        treatment = cell["row_label"]
        target = col_info.get("target", "xx")
        unit = col_info.get("unit", "xx")

        # Ontology normalization
        if unit != "xx":
            unit = standardize_unit(unit)
        if target != "xx":
            target = standardize_target(target)

        results.append({
            "cell_index": cell["cell_index"],
            "value": raw_val,
            "type": assay_type,
            "target": target,
            "treatment": treatment,
            "unit": unit,
        })

    return results


def assemble_discomat_tuples(numeric_cells: list[dict],
                              header_info: list[dict],
                              parsed_table: dict,
                              paper_id: str, table_idx: int) -> list[dict]:
    """Assemble DiSCoMaT composition tuples from deterministic cells + header info."""
    from ontology_discomat import standardize_component, is_null_value

    # Build column classification map
    data_cols = {}
    for hi in header_info:
        if hi.get("is_data", False):
            data_cols[hi["col_idx"]] = hi

    results = []
    headers = parsed_table["headers"]
    data_rows = parsed_table["data_rows"]

    # Detect if table is transposed (components in rows, samples in columns)
    # Heuristic: if headers mostly contain sample-like labels (numbers), it's normal
    # If headers contain component names (SiO2, PbO), it might be transposed
    component_in_headers = sum(1 for hi in header_info if hi.get("is_data", False))
    is_transposed = False
    if component_in_headers > len(headers) * 0.5:
        is_transposed = False  # Normal: components as columns

    for cell in numeric_cells:
        ci = cell["col_idx"]
        ri = cell["row_idx"]

        if ci not in data_cols:
            continue

        col_info = data_cols[ci]
        component = col_info.get("component", "")
        unit = col_info.get("unit", "mol")

        # Skip null/zero values
        if is_null_value(cell["raw_value"]):
            continue

        # Sample ID: use row label or construct from paper_id + table_idx + row
        sample_label = cell["row_label"]
        if not sample_label or not sample_label.strip():
            sample_label = str(ri + 1)

        # Build DiSCoMaT-format sample_id
        sample_id = f"{paper_id}_{table_idx}_2_{sample_label}"

        val = cell["numeric_value"]
        if val is None:
            continue

        results.append({
            "sample_id": sample_id,
            "component": component,
            "value": val,
            "unit": unit,
        })

    return results


# =============================================================================
# Full Pipeline Runner
# =============================================================================
def run_chemtables_hybrid(data: dict, model: str, output_dir: Path,
                           use_llm_headers: bool = True) -> dict:
    """Run hybrid pipeline on ChemTables benchmark."""
    all_predictions = {}

    for tid, entry in data.items():
        print(f"  {tid}...")
        html = entry["table_processed"]

        # Stage 1: Deterministic parsing
        parsed = parse_html_table(html)

        # Stage 2: Header interpretation
        if use_llm_headers:
            header_info = interpret_headers_llm(parsed["headers"], parsed["caption"], model)
            if not header_info:
                # Fallback to rule-based
                header_info = interpret_headers_rules(parsed["headers"], parsed["caption"], "chemtables")
        else:
            header_info = interpret_headers_rules(parsed["headers"], parsed["caption"], "chemtables")

        # Stage 1 continued: Extract all numeric cells
        numeric_cells = extract_all_numeric_cells(parsed)

        # Stage 3: Assembly with ontology validation
        tuples = assemble_chemtables_tuples(numeric_cells, header_info, parsed)

        all_predictions[tid] = tuples

        # Save
        out_path = output_dir / f"{tid.replace('::', '_')}.json"
        with open(out_path, "w") as f:
            json.dump({
                "predictions": tuples,
                "header_info": header_info,
                "numeric_cells_count": len(numeric_cells),
                "filtered_count": len(tuples),
            }, f, indent=2)

        if use_llm_headers:
            time.sleep(0.3)

    return all_predictions


def run_discomat_hybrid(data: dict, model: str, output_dir: Path,
                         use_llm_headers: bool = True) -> dict:
    """Run hybrid pipeline on DiSCoMaT benchmark."""
    all_predictions = {}

    for i, (tid, entry) in enumerate(data.items()):
        if i % 20 == 0:
            print(f"  {i+1}/{len(data)}: {tid}...")

        md = entry["table_processed"]
        pii = entry.get("pii", tid.split("::")[0])
        t_idx = entry.get("t_idx", 0)

        # Stage 1: Deterministic parsing
        parsed = parse_markdown_table(md)

        # Stage 2: Header interpretation
        if use_llm_headers and i < 5:  # LLM for first few, then rule-based (cost control)
            header_info = interpret_headers_llm(parsed["headers"], parsed["caption"], model)
            if not header_info:
                header_info = interpret_headers_rules(parsed["headers"], parsed["caption"], "discomat")
        else:
            header_info = interpret_headers_rules(parsed["headers"], parsed["caption"], "discomat")

        # Stage 1 continued: Extract all numeric cells
        numeric_cells = extract_all_numeric_cells(parsed)

        # Stage 3: Assembly with ontology validation
        tuples = assemble_discomat_tuples(numeric_cells, header_info, parsed, pii, t_idx)

        all_predictions[tid] = tuples

        out_path = output_dir / f"{tid.replace('::', '_')}.json"
        with open(out_path, "w") as f:
            json.dump({
                "predictions": tuples,
                "header_info": header_info,
                "numeric_cells_count": len(numeric_cells),
                "filtered_count": len(tuples),
            }, f, indent=2)

        if use_llm_headers and i < 5:
            time.sleep(0.3)

    return all_predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Pipeline Runner")
    parser.add_argument("--dataset", required=True, choices=["chemtables", "discomat"])
    parser.add_argument("--model", default="claude-sonnet")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based headers only (no LLM)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ISWC_ROOT = Path(__file__).parent.parent
    RESULTS_DIR = ISWC_ROOT / "results"
    DATA_DIR = ISWC_ROOT / "datasets"

    suffix = "hybrid_rules" if args.no_llm else f"hybrid_{args.model}"
    output_dir = RESULTS_DIR / f"{args.dataset}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Hybrid Pipeline: {args.dataset} ===")
    print(f"Mode: {'rule-based' if args.no_llm else f'LLM ({args.model})'}")
    print(f"Output: {output_dir}")

    if args.dataset == "chemtables":
        with open(DATA_DIR / "schema_driven_ie" / "data" / "chemtables" / "test.json") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} tables")

        if args.dry_run:
            # Just test parsing
            for tid, entry in list(data.items())[:2]:
                parsed = parse_html_table(entry["table_processed"])
                cells = extract_all_numeric_cells(parsed)
                hinfo = interpret_headers_rules(parsed["headers"], parsed["caption"], "chemtables")
                tuples = assemble_chemtables_tuples(cells, hinfo, parsed)
                print(f"  {tid}: {len(cells)} cells -> {len(tuples)} tuples (gold: {len(entry['cell_list_gold'])})")
                data_cols = [h for h in hinfo if h["is_data"]]
                print(f"    Data columns: {[(h['col_idx'], h['type'], h['target'][:15]) for h in data_cols]}")
            sys.exit(0)

        predictions = run_chemtables_hybrid(data, args.model, output_dir,
                                             use_llm_headers=not args.no_llm)

        # Evaluate
        from extract_and_eval import eval_chemtables
        results = eval_chemtables(data, predictions)
        agg = results["aggregate"]
        print(f"\n=== RESULTS ===")
        print(f"P={agg['precision']}% R={agg['recall']}% F1={agg['f1']}%")

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.dataset == "discomat":
        with open(DATA_DIR / "schema_driven_ie" / "data" / "discomat" / "test.json") as f:
            raw = json.load(f)
        data = {k: v for k, v in raw.items() if len(v.get("cell_list_gold", [])) > 0}
        print(f"Loaded {len(data)} tables with gold")

        if args.dry_run:
            for tid, entry in list(data.items())[:2]:
                parsed = parse_markdown_table(entry["table_processed"])
                cells = extract_all_numeric_cells(parsed)
                hinfo = interpret_headers_rules(parsed["headers"], parsed["caption"], "discomat")
                tuples = assemble_discomat_tuples(cells, hinfo, parsed,
                                                   entry.get("pii", ""), entry.get("t_idx", 0))
                print(f"  {tid}: {len(cells)} cells -> {len(tuples)} tuples (gold: {len(entry['cell_list_gold'])})")
            sys.exit(0)

        predictions = run_discomat_hybrid(data, args.model, output_dir,
                                           use_llm_headers=not args.no_llm)

        from extract_and_eval import eval_discomat
        results = eval_discomat(data, predictions)
        agg = results["aggregate"]
        print(f"\n=== RESULTS ===")
        print(f"P={agg['precision']}% R={agg['recall']}% F1={agg['f1']}%")

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
