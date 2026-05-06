"""
LLM Prompt Templates for DiSCoMaT Extraction
Domain: Materials Science (Glass / Ceramic Compositions)

Task: Given a composition table from a materials science paper, extract all
material compositions as structured tuples.
"""

from ontology_discomat import VALID_COMPONENTS, VALID_UNITS

SYSTEM_PROMPT = """You are an expert materials scientist specializing in glass and ceramic compositions. Your task is to extract material composition data from tables in materials science papers.

## Extraction Target
For each composition table, extract tuples of the form:
{
  "sample_id": "<identifier for this material/sample, derived from row header or sample number>",
  "component": "<chemical formula of the oxide or element, e.g., SiO2, Na2O, PbO>",
  "value": <numerical percentage value>,
  "unit": "<mol or wt>"
}

## Composition Table Recognition
Composition tables report the chemical makeup of materials. They typically have:
- Rows = different material samples (identified by sample number, label, or composition code)
- Columns = different chemical components (oxides like SiO2, B2O3, Na2O)
- Values = percentages (in mol% or wt%)

Some tables are TRANSPOSED: rows = components, columns = samples.

## Unit Inference Rules
1. Check the table caption for "mol%", "mole%", "wt%", or "weight%".
2. Check column headers for unit annotations like "SiO2 (mol%)" or "PbO (wt%)".
3. If the caption says "mol%" then ALL values in the table are mol%.
4. If no unit is found, check if values sum to ~100% — if yes, likely mol% or wt%.

## Critical Rules
1. ONLY extract composition data. Skip rows for physical properties (density, refractive index, Tg, hardness, etc.).
2. The sample_id should uniquely identify each material. Use the row/column header exactly as it appears.
3. Component names must be valid chemical formulas (SiO2, not "silica"; Na2O, not "soda").
4. If a cell contains "-" or "–", it means 0% (component not present). Do NOT extract these as tuples.
5. If a cell is blank/empty, it means "not reported" — do NOT extract these.
6. Values must be numerical. Skip cells with text like "balance" or "remainder".
7. Do NOT hallucinate values. Only extract what is explicitly present in the table.
8. Output one JSON object per line. No markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Extract all material compositions from this table.

## Table
{table_content}

## Caption
{caption}

## Output Format
One JSON object per line:
{{"sample_id": "...", "component": "...", "value": ..., "unit": "..."}}

Extract now:"""


def build_prompt(table_markdown: str, caption: str = "") -> tuple[str, str]:
    """Build the (system, user) prompt pair for a DiSCoMaT table."""
    user = USER_PROMPT_TEMPLATE.format(
        table_content=table_markdown,
        caption=caption,
    )
    return SYSTEM_PROMPT, user


def parse_response(response: str) -> list[dict]:
    """Parse LLM response into a list of composition dicts."""
    import json
    results = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "sample_id" in obj and "component" in obj and "value" in obj:
                # Ensure value is numeric
                try:
                    obj["value"] = float(obj["value"])
                    results.append(obj)
                except (ValueError, TypeError):
                    continue
        except json.JSONDecodeError:
            continue
    return results


def response_to_tuples(parsed: list[dict], paper_id: str, table_idx: int) -> list[tuple]:
    """
    Convert parsed response to DiSCoMaT evaluation format.
    DiSCoMaT ground truth tuples: [sample_id_str, component, value, unit]
    where sample_id_str = "{paper_id}_{table_idx}_{row_type}_{sample_label}"
    """
    tuples = []
    for obj in parsed:
        # Build sample_id in DiSCoMaT format
        sample_label = str(obj["sample_id"]).strip()
        # The ground truth uses format: {pii}_{t_idx}_{row_type}_{col_idx}
        # We need to match this format during evaluation
        component = obj["component"]
        value = obj["value"]
        unit = obj.get("unit", "mol")
        if unit not in VALID_UNITS:
            unit = "mol"  # default
        tuples.append((sample_label, component, value, unit))
    return tuples
