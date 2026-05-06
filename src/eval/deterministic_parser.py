"""
Deterministic Table Parser
Mirrors the geochem pipeline's table_reader.py approach:
  - Parse table structure (HTML or markdown) into rows/columns
  - Identify header rows vs data rows
  - Extract ALL numerical cells with their position context
  - NO LLM calls — purely rule-based

This is the key to the hybrid extraction principle:
  Deterministic parser → perfect numerical recall
  LLM → header interpretation (metadata only)
  Ontology → validation and normalization
"""

import re
from html.parser import HTMLParser
from typing import Optional


# =============================================================================
# HTML Table Parser
# =============================================================================
class _HTMLTableParser(HTMLParser):
    """Parse an HTML table into a list of rows, each a list of cell strings."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: str = ""
        self.in_cell = False
        self.in_thead = False
        self.header_row_count = 0
        self.caption = ""
        self.in_caption = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = ""
        elif tag == "tr":
            self.current_row = []
        elif tag == "thead":
            self.in_thead = True
        elif tag in ("caption", "title"):
            self.in_caption = True

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
                if self.in_thead:
                    self.header_row_count += 1
        elif tag == "thead":
            self.in_thead = False
        elif tag in ("caption", "title"):
            self.in_caption = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data
        elif self.in_caption:
            self.caption += data.strip() + " "


def parse_html_table(html: str) -> dict:
    """Parse HTML table into structured format.

    Returns:
        {
            "headers": list of header cell strings (from thead or first row),
            "data_rows": list of lists (each row is a list of cell strings),
            "caption": str,
            "num_cols": int,
            "num_data_rows": int,
        }
    """
    parser = _HTMLTableParser()
    parser.feed(html)

    if not parser.rows:
        return {"headers": [], "data_rows": [], "caption": parser.caption.strip(),
                "num_cols": 0, "num_data_rows": 0}

    # Separate headers from data
    if parser.header_row_count > 0:
        header_rows = parser.rows[:parser.header_row_count]
        data_rows = parser.rows[parser.header_row_count:]
    else:
        # Heuristic: first row is header if it has fewer numeric cells
        first_row = parser.rows[0]
        num_numeric = sum(1 for c in first_row if _is_numeric(c))
        if num_numeric < len(first_row) / 2:
            header_rows = [first_row]
            data_rows = parser.rows[1:]
        else:
            header_rows = []
            data_rows = parser.rows

    # Flatten multi-row headers into single header
    if header_rows:
        num_cols = max(len(r) for r in header_rows)
        merged_headers = [""] * num_cols
        for row in header_rows:
            for i, cell in enumerate(row):
                if i < num_cols and cell.strip():
                    if merged_headers[i]:
                        merged_headers[i] += " " + cell.strip()
                    else:
                        merged_headers[i] = cell.strip()
        headers = merged_headers
    else:
        headers = []

    return {
        "headers": headers,
        "data_rows": data_rows,
        "caption": parser.caption.strip(),
        "num_cols": len(headers) if headers else (max(len(r) for r in data_rows) if data_rows else 0),
        "num_data_rows": len(data_rows),
    }


# =============================================================================
# Markdown Table Parser
# =============================================================================
def parse_markdown_table(md: str) -> dict:
    """Parse a pipe-delimited markdown table into structured format."""
    lines = md.strip().split("\n")

    # Find table lines (contain |)
    table_lines = [l for l in lines if "|" in l]
    if not table_lines:
        return {"headers": [], "data_rows": [], "caption": "",
                "num_cols": 0, "num_data_rows": 0}

    # Extract caption (lines before or after the table that don't have |)
    caption_lines = []
    for l in lines:
        if "|" not in l and l.strip():
            if l.strip().lower().startswith("caption"):
                caption_lines.append(l.strip().replace("Caption:", "").replace("caption:", "").strip())
            elif not l.strip().startswith(("-", "=")):
                caption_lines.append(l.strip())
    caption = " ".join(caption_lines)

    # Parse rows
    def split_row(line):
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last from leading/trailing |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        return cells

    rows = []
    for line in table_lines:
        # Skip separator lines (---|---|---)
        if re.match(r"^\s*\|?\s*[-:]+(\s*\|\s*[-:]+)*\s*\|?\s*$", line):
            continue
        cells = split_row(line)
        if cells:
            rows.append(cells)

    if not rows:
        return {"headers": [], "data_rows": [], "caption": caption,
                "num_cols": 0, "num_data_rows": 0}

    # Detect header: first row(s) with mostly non-numeric cells
    # For multi-row headers, keep merging until we hit a mostly-numeric row
    header_end = 0
    for i, row in enumerate(rows):
        num_numeric = sum(1 for c in row if _is_numeric(c))
        if num_numeric > len(row) / 2:
            break
        header_end = i + 1

    if header_end == 0:
        headers = []
        data_rows = rows
    else:
        # Merge multi-row headers
        num_cols = max(len(r) for r in rows[:header_end])
        merged = [""] * num_cols
        for row in rows[:header_end]:
            for j, cell in enumerate(row):
                if j < num_cols and cell.strip():
                    if merged[j]:
                        merged[j] += " " + cell.strip()
                    else:
                        merged[j] = cell.strip()
        headers = merged
        data_rows = rows[header_end:]

    return {
        "headers": headers,
        "data_rows": data_rows,
        "caption": caption,
        "num_cols": len(headers) if headers else (max(len(r) for r in data_rows) if data_rows else 0),
        "num_data_rows": len(data_rows),
    }


# =============================================================================
# Numerical Cell Detection
# =============================================================================
def _is_numeric(s: str) -> bool:
    """Check if a string represents a numeric value (including >, <, ±)."""
    s = s.strip()
    if not s or s == "-" or s == "–" or s == "—":
        return False
    # Strip prefixes
    s = s.lstrip(">< ")
    # Strip ± and everything after
    s = re.split(r"[±\u00b1]", s)[0].strip()
    # Strip parenthesized ranges
    s = re.split(r"\s*\(", s)[0].strip()
    # Remove non-breaking spaces
    s = s.replace("\xa0", "").replace("\u2009", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_numeric_value(s: str) -> Optional[float]:
    """Extract the primary numeric value from a cell string."""
    s = s.strip()
    if not s:
        return None
    # Preserve > and < prefixes (they have semantic meaning)
    prefix = ""
    if s.startswith(">") or s.startswith("<"):
        prefix = s[0]
        s = s[1:].strip()
    # Strip ± uncertainty
    s = re.split(r"[±\u00b1]", s)[0].strip()
    # Strip parenthesized ranges
    s = re.split(r"\s*\(", s)[0].strip()
    # Remove non-breaking spaces
    s = s.replace("\xa0", "").replace("\u2009", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_raw_value(s: str) -> str:
    """Extract the raw value string preserving ± for gold matching."""
    s = s.strip()
    # Remove non-breaking spaces but keep ±
    s = s.replace("\xa0", " ").replace("\u2009", " ")
    # Collapse multiple spaces
    s = " ".join(s.split())
    return s


# =============================================================================
# Complete Table Extraction (Deterministic)
# =============================================================================
def extract_all_numeric_cells(parsed_table: dict) -> list[dict]:
    """Extract all numeric cells from a parsed table with their context.

    Returns a list of dicts:
        {
            "row_idx": int (0-based, in data_rows),
            "col_idx": int (0-based),
            "raw_value": str (original cell content),
            "numeric_value": float or None,
            "row_label": str (first cell in the row, typically compound/sample ID),
            "col_header": str (header for this column),
            "cell_index": str (CA(row, col) format for evaluation matching),
        }
    """
    headers = parsed_table["headers"]
    data_rows = parsed_table["data_rows"]
    results = []

    for ri, row in enumerate(data_rows):
        # Row label is typically the first cell
        row_label = row[0].strip() if row else ""

        for ci, cell in enumerate(row):
            if not _is_numeric(cell):
                continue

            col_header = headers[ci] if ci < len(headers) else ""
            raw = extract_raw_value(cell)
            numeric = extract_numeric_value(cell)

            # Build cell_index: CA(row, col) — 1-based row (header is row 0)
            # This matches the gold standard convention
            cell_index = f"CA({ri + 1},{ci})"

            results.append({
                "row_idx": ri,
                "col_idx": ci,
                "raw_value": raw,
                "numeric_value": numeric,
                "row_label": row_label,
                "col_header": col_header,
                "cell_index": cell_index,
            })

    return results


# =============================================================================
# Auto-detect format and parse
# =============================================================================
def parse_table(content: str) -> dict:
    """Auto-detect HTML vs markdown and parse."""
    if "<table" in content.lower() or "<tr" in content.lower():
        return parse_html_table(content)
    elif "|" in content:
        return parse_markdown_table(content)
    else:
        return {"headers": [], "data_rows": [], "caption": "",
                "num_cols": 0, "num_data_rows": 0}


if __name__ == "__main__":
    # Test with a ChemTables HTML table
    import json
    with open("../datasets/schema_driven_ie/data/chemtables/test.json") as f:
        data = json.load(f)

    tid = list(data.keys())[0]
    entry = data[tid]
    parsed = parse_html_table(entry["table_processed"])

    print(f"=== {tid} ===")
    print(f"Caption: {parsed['caption'][:80]}")
    print(f"Headers ({len(parsed['headers'])}): {parsed['headers']}")
    print(f"Data rows: {parsed['num_data_rows']}")
    print()

    cells = extract_all_numeric_cells(parsed)
    print(f"Numeric cells extracted: {len(cells)}")
    print(f"Gold annotations: {len(entry['cell_list_gold'])}")
    print()
    for c in cells[:5]:
        print(f"  {c['cell_index']}: value={c['raw_value']} col={c['col_header'][:30]} row_label={c['row_label']}")

    # Test with DiSCoMaT markdown table
    print("\n" + "=" * 50)
    with open("../datasets/schema_driven_ie/data/discomat/test.json") as f:
        ddata = json.load(f)

    for tid2, entry2 in ddata.items():
        if len(entry2.get("cell_list_gold", [])) > 0:
            parsed2 = parse_markdown_table(entry2["table_processed"])
            print(f"\n=== {tid2} ===")
            print(f"Caption: {parsed2['caption'][:80]}")
            print(f"Headers ({len(parsed2['headers'])}): {parsed2['headers']}")
            print(f"Data rows: {parsed2['num_data_rows']}")
            cells2 = extract_all_numeric_cells(parsed2)
            print(f"Numeric cells: {len(cells2)}")
            print(f"Gold tuples: {len(entry2['cell_list_gold'])}")
            for c in cells2[:3]:
                print(f"  {c['cell_index']}: value={c['raw_value']} col={c['col_header'][:20]} row_label={c['row_label']}")
            break
