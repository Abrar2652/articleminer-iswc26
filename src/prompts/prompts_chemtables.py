"""
LLM Prompt Templates for ChemTables Extraction
Domain: Drug Discovery / Medicinal Chemistry

Task: Given a table from a medicinal chemistry paper, extract all bioactivity
measurements as structured tuples.
"""

from ontology_chemtables import VALID_ASSAY_TYPES

SYSTEM_PROMPT = """You are an expert medicinal chemist specializing in drug discovery data extraction. Your task is to extract bioactivity measurements from tables in medicinal chemistry papers.

## Extraction Target
For each numerical cell in the table that represents a bioactivity measurement, produce a JSON object:
{
  "cell_index": "CA(row,col)",
  "value": "<exact numerical value from the cell>",
  "type": "<one of: IC50, EC50, GI50, MIC>",
  "target": "<protein target, cell line, or organism being assayed>",
  "treatment": "<compound identifier (number, name, or code) being tested>",
  "unit": "<measurement unit, e.g., µM, nM, µg/mL>"
}

## Assay Type Definitions
- IC50: Half-maximal inhibitory concentration (enzyme/protein inhibition AND cell viability/proliferation assays)
- EC50: Half-maximal effective concentration (cell-based functional assays)
- GI50: Concentration causing 50% growth inhibition (ONLY when paper explicitly uses the term GI50)
- MIC: Minimum inhibitory concentration (antimicrobial assays)

IMPORTANT: If the table caption or column header explicitly states "IC50", classify as IC50 even if the assay measures cell proliferation. The paper's own terminology takes precedence over domain conventions. Only use GI50 if the paper explicitly labels values as GI50.

## Critical Rules
1. Extract ALL cells containing numerical bioactivity/potency values. This includes IC50, EC50, GI50, MIC, and related measurements like % inhibition (which should be classified as MIC if units are µg/mL or µM) or cytotoxicity values.
2. The "treatment" is the compound being tested — usually identified by a number in the first column or a compound name.
3. The "target" is what the compound is tested against — a protein (CHK1, EGFR), cell line (A549, HeLa), or organism (S. aureus, E. coli). If the target is not clear, use "xx".
4. Infer type, target, and unit from column headers. Column headers often follow patterns like "CHK1 IC50 (µM)" or "MIC (µg/mL) S. aureus".
5. If a column header says "% inhibition" or "inhibition (%)" with units like µg/mL, classify as MIC.
6. If a cell contains a range like "1.0 (0.86, 1.2)", extract the primary value "1.0" AND include the full string with ± if present (e.g., "265 ± 17" should have value "265 ± 17").
7. If a cell contains ">100" or ">50", this means the compound is inactive at the tested range. Still extract it with value ">100" or ">50".
8. Do NOT hallucinate values. Only extract what is explicitly present in the table.
9. Skip truly non-bioactivity data like selectivity ratios, ligand efficiency, or physical properties (LogP, MW).
10. Output one JSON object per line. No markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Extract all bioactivity measurements from this table.

## Table
{table_content}

## Caption
{caption}

## Output Format
One JSON object per line for each bioactivity measurement cell:
{{"cell_index": "CA(row,col)", "value": "...", "type": "...", "target": "...", "treatment": "...", "unit": "..."}}

Extract now:"""


def build_prompt(table_html: str, caption: str = "") -> tuple[str, str]:
    """Build the (system, user) prompt pair for a ChemTables table."""
    user = USER_PROMPT_TEMPLATE.format(
        table_content=table_html,
        caption=caption,
    )
    return SYSTEM_PROMPT, user


def parse_response(response: str) -> list[dict]:
    """Parse LLM response into a list of annotation dicts."""
    import json
    results = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "value" in obj and "type" in obj:
                # Validate type
                if obj["type"] in VALID_ASSAY_TYPES:
                    results.append(obj)
        except json.JSONDecodeError:
            continue
    return results
