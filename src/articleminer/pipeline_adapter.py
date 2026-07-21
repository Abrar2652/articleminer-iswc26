#!/usr/bin/env python3
"""
Pipeline Adapter — Run the geochem extraction pipeline on cross-domain PDFs.

This is the core of the ISWC paper's claim: the SAME architecture (multi-extractor
PDF parsing + deterministic table reader + LLM metadata + ontology validation)
works across domains by swapping ONLY the ontology module and prompts.

Architecture (identical to geochem):
  Stage 1: PDF text extraction (pdf_reader.py — unchanged)
  Stage 2: Multi-extractor table extraction (tabledetector.py — unchanged)
           Runs Docling, Marker, MinerU, Camelot, pdfplumber in parallel
  Stage 3: LLM-based metadata + header interpretation (domain-specific prompts)
  Stage 4: Ontology-grounded validation (domain-specific knowledge_base)
  Stage 5: Agentic self-correction if quality check fails (unchanged logic)

What changes per domain:
  - Ontology module (ontology_chemtables.py / ontology_discomat.py)
  - LLM prompts (what to extract from headers/text)
  - Output schema (what fields the tuples have)
  - Evaluation metrics (ChemTables: attr-level F1, DiSCoMaT: tuple-level F1)
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure src/ is importable so `articleminer` resolves when run standalone.
sys.path.insert(0, str(Path(__file__).parents[1]))  # src/

from articleminer.pdf_reader import extract_pdf, get_paper_text_for_llm
from articleminer.llm_clients import ClaudeClient, OpenAIClient, GeminiClient


# Local HuggingFace models — short-name → HF repo.
# OPEN-WEIGHT ONLY (no HF auth required). Gated repos (meta-llama/*,
# mistralai/Mistral-7B-Instruct-v0.3) are intentionally excluded so the
# paper's open-source-LLM coverage claim is reproducible without auth.
LOCAL_MODEL_MAP = {
    "qwen25-7b":      "Qwen/Qwen2.5-7B-Instruct",
    "qwen25-32b":     "Qwen/Qwen2.5-32B-Instruct",
    "mistral-7b-v02": "mistralai/Mistral-7B-Instruct-v0.2",  # v0.2 open
    "phi-3-medium":   "microsoft/Phi-3-medium-4k-instruct",
    "falcon-7b":      "tiiuae/falcon-7b-instruct",
    "deepseek-llm-7b":"deepseek-ai/deepseek-llm-7b-chat",
}


class LocalHFClient:
    """Minimal LLM client for local HuggingFace models. Mirrors the geochem
    LLMClient interface (`complete`, optionally `complete_json`). Used for
    open-source LLM coverage in cross-domain experiments."""

    _instances: dict = {}

    def __init__(self, model: str):
        self.model_name = model
        repo = LOCAL_MODEL_MAP.get(model, model)
        if repo in LocalHFClient._instances:
            self.tokenizer, self.model = LocalHFClient._instances[repo]
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            logger.info(f"Loading local HF model {repo} (one-time)...")
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True,
            )
            LocalHFClient._instances[repo] = (tok, mdl)
            self.tokenizer, self.model = tok, mdl

    def complete(self, system: str, user: str, max_tokens: int = 4096,
                 images=None) -> str:
        if images:
            logger.warning("LocalHFClient: images param ignored (no vision)")
        import torch
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=8192,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=min(max_tokens, 4096),
                do_sample=False, temperature=None, top_p=None,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )


def make_client(model: str):
    """Return the right LLM client for a given model name."""
    m = model.lower()
    if m in LOCAL_MODEL_MAP or "/" in model:
        return LocalHFClient(model=model)
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return OpenAIClient(model=model)
    if m.startswith(("gemini",)):
        return GeminiClient(model=model)
    return ClaudeClient(model=model)
from articleminer import tabledetector as _td
from articleminer.tabledetector import (
    extract_tables_from_pdf, TableDetectorBackend, _parse_markdown_tables,
)

logger = logging.getLogger(__name__)

# Marker requires GPU. On clusters with old CUDA driver it falls back to CPU
# OCR and is ~20× slower. Set ISWC_DISABLE_MARKER=1 to skip marker entirely
# and rely on pdfplumber + docling.
if os.environ.get("ISWC_DISABLE_MARKER", "").lower() in ("1", "true", "yes"):
    _td.marker_pdf_to_markdown = lambda *a, **k: ""
    logger.info("ISWC_DISABLE_MARKER=1 — marker text/markdown disabled")

ISWC_ROOT = Path(__file__).parent.parent
CHEMTABLES_PDFS = ISWC_ROOT / "datasets" / "chemtables_pdfs"
DISCOMAT_PDFS = ISWC_ROOT / "datasets" / "discomat_pdfs"
MLTABLES_PDFS = ISWC_ROOT / "datasets" / "mltables_pdfs"
RESULTS_DIR = ISWC_ROOT / "results"


# =============================================================================
# Domain-Specific Prompt for ChemTables Table Extraction
# =============================================================================
CHEMTABLES_TABLE_EXTRACTION_PROMPT = """You are an expert medicinal chemist. Given a table extracted from a drug discovery paper, extract ALL bioactivity measurements.

For each measurement, output a JSON object on one line:
{{"value": "<number only — NO unit>", "type": "<IC50|EC50|GI50|MIC>", "target": "<protein/cell line/organism>", "treatment": "<compound ID>", "unit": "<µM|nM|µg/mL>"}}

RULES:
1. `value` must be the EXACT numerical value ONLY — put the unit in the `unit` field, not in the value. E.g. "127" + unit="nM", NOT "127 nM".
2. Determine type from column headers: IC50, EC50, GI50, MIC. If header says "% inhibition" with concentration units, classify as MIC.
3. Treatment = compound identifier (usually first column: a number like "2", "31a", or a name).
4. Target = what is being measured against (protein: CHK1, EGFR; cell line: A549, HeLa; organism: S. aureus).
5. If target cannot be determined, use "xx".
6. If a cell has ">" prefix (e.g., ">100"), keep it as ">100".
7. Skip non-bioactivity columns (selectivity ratios, LE, logP, MW, etc.).
8. Do NOT hallucinate. Only extract what is explicitly in the table.
9. Output ONLY JSON lines, no explanation."""


# =============================================================================
# LLM-driven per-paper backend selection
# (User insight: fixed 5-backend isn't always optimal — let an LLM choose
# per paper based on a quick text-glance.)
# =============================================================================
BACKEND_SELECT_SYSTEM = """\
You are a PDF-table-extraction expert. Given a short text snippet from a
scientific paper, select the OPTIMAL subset of PDF backends for table
extraction. Return ONLY a JSON list of backend names from this set:
  docling     -- best for typeset complex tables, multi-line headers
  marker      -- best for image-rich PDFs, OCR fallback, surya-based
  mineru      -- best for scientific layouts (multi-column, equations)
  pdfplumber  -- best for simple text-grid tables (fast)
  camelot     -- best for bordered tables (PDF lattice)

Rules:
- Pick anywhere from 1 to ALL 5 backends — whatever maximises extraction
  quality for THIS paper. Heterogeneous papers (mixed simple + complex
  tables, mixed text + image) often benefit from full consensus (all 5).
- Homogeneous, simple papers (one or two clean text tables) need only 1-2
  backends.
- Suggested heuristics (not hard rules):
    * prose-heavy with simple tables → ["docling","pdfplumber"]
    * image-heavy / scanned          → ["marker","mineru"]
    * materials/glass compositions   → ["docling","mineru"]
    * ML numeric result tables       → ["docling","pdfplumber"]
    * drug-discovery bioactivity     → ["docling","marker"]
    * complex/heterogeneous paper    → ALL 5
- Prefer fewer backends only if compute matters; otherwise err on more.

Return JSON only, e.g. ["docling","marker","mineru","pdfplumber","camelot"]"""


def select_backends_for_paper(paper_text: str, llm_client,
                               available: list[str] | None = None) -> list[str]:
    """Ask the LLM which backends to use for this paper. Returns a subset of
    `available` (default: all 5). Falls back to ["docling", "pdfplumber"] on error.
    """
    if available is None:
        available = ["docling", "marker", "mineru", "pdfplumber", "camelot"]
    snippet = paper_text[:1500]
    user = (
        "## PAPER TEXT (first 1500 chars)\n"
        f"{snippet}\n\n"
        f"Available backends: {available}\n"
        "Which backends should we use? Return JSON list."
    )
    try:
        if hasattr(llm_client, "complete_json"):
            choice = llm_client.complete_json(
                system=BACKEND_SELECT_SYSTEM, user=user, max_tokens=256,
            )
        else:
            raw = llm_client.complete(BACKEND_SELECT_SYSTEM, user, max_tokens=256)
            import json as _json
            choice = _json.loads(raw.strip()) if raw.strip().startswith("[") else None
    except Exception as e:
        logger.warning(f"backend selection failed: {e}; defaulting")
        return ["docling", "pdfplumber"]

    if not isinstance(choice, list) or not choice:
        return ["docling", "pdfplumber"]
    selected = [b for b in choice if b in available]
    if not selected:
        return ["docling", "pdfplumber"]
    logger.info(f"LLM-selected backends: {selected}")
    return selected


# =============================================================================
# Optional LLM-judge for ambiguous duplicates
# Catches the residual cases that escape canonical-fingerprint dedup.
# =============================================================================
LLM_DEDUP_SYSTEM = """\
You are deduplicating tables extracted from a single PDF by multiple backends.
You will see N candidate table-texts. Group them: tables in the same group
represent the SAME PHYSICAL TABLE in the PDF (just transcribed slightly
differently by different backends).

Rules:
- Two tables are duplicates if their values, row labels, and column
  headers correspond — even if formatting differs (subscript characters,
  spacing, border style, slight OCR noise).
- Tables with different VALUES or different ROWS are NOT duplicates.

Return ONLY JSON: {"groups": [[0,2,4],[1],[3]]}  meaning indexes 0,2,4
are the same physical table; index 1 stands alone; index 3 stands alone."""


def llm_dedup_tables(tables: list[str], llm_client, paper_id: str = "") -> list[str]:
    """For each LLM-identified group, keep the longest variant."""
    if len(tables) <= 1:
        return tables
    # Build compact preview for LLM
    preview = []
    for i, t in enumerate(tables[:30]):  # cap at 30 to fit context
        head = t[:600].replace("\n", " ")
        preview.append(f"### Table {i}\n{head}")
    user = ("## CANDIDATE TABLES (one block per index)\n"
            + "\n\n".join(preview)
            + f"\n\n## TASK\nReturn JSON: {{\"groups\": [[index,...],...]}}")
    try:
        if hasattr(llm_client, "complete_json"):
            res = llm_client.complete_json(
                system=LLM_DEDUP_SYSTEM, user=user, max_tokens=512,
            )
        else:
            raw = llm_client.complete(LLM_DEDUP_SYSTEM, user, max_tokens=512)
            import json as _json
            res = _json.loads(raw) if raw.strip().startswith("{") else {}
    except Exception as e:
        logger.warning(f"[{paper_id}] llm_dedup failed: {e}")
        return tables
    groups = res.get("groups") if isinstance(res, dict) else None
    if not isinstance(groups, list):
        return tables
    kept: list[str] = []
    used = set()
    for grp in groups:
        if not isinstance(grp, list) or not grp:
            continue
        # Pick the longest table in the group
        idxs = [i for i in grp if isinstance(i, int) and 0 <= i < len(tables)]
        if not idxs:
            continue
        best = max(idxs, key=lambda i: len(tables[i]))
        kept.append(tables[best])
        used.update(idxs)
    # Tables not mentioned by LLM are kept as-is (fail-safe)
    for i, t in enumerate(tables):
        if i not in used:
            kept.append(t)
    if len(kept) < len(tables):
        logger.info(f"[{paper_id}] LLM-judge dedup: {len(tables)} → {len(kept)}")
    return kept


# =============================================================================
# Stage 1+2: PDF → Multi-Extractor Table Extraction
# =============================================================================
def extract_tables_from_paper(pdf_path: Path, backends: list[str] = None) -> dict:
    """Run multi-extractor PDF table extraction (same as geochem pipeline).

    Returns:
        {
            "pdf_text": str,
            "tables_by_backend": {backend_name: [table_text, ...]},
            "all_tables": [table_text, ...],  # deduplicated
        }
    """
    if backends is None:
        backends = ["docling", "marker", "pdfplumber"]  # most reliable 3

    pdf_str = str(pdf_path)

    # Stage 1: Extract PDF text
    logger.info(f"Extracting PDF text from {pdf_path.name}...")
    pdf_content = extract_pdf(pdf_str)
    full_text = get_paper_text_for_llm(pdf_content)

    # Stage 2: Multi-extractor table extraction
    tables_by_backend = {}
    all_tables_raw = []

    for backend_name in backends:
        try:
            backend = TableDetectorBackend(backend_name)
            logger.info(f"  Running {backend_name}...")
            result = extract_tables_from_pdf(pdf_str, backend=backend)
            # Handle tuple return (tables, metrics) or just list
            if isinstance(result, tuple):
                tables, metrics = result
            else:
                tables = result
            # Convert extracted tables to strings
            table_strings = []
            for t in tables:
                if isinstance(t, str):
                    table_strings.append(t)
                elif hasattr(t, 'df'):
                    # ExtractedTable object from articleminer.tabledetector
                    try:
                        md = t.df.to_markdown(index=False)
                        if md and len(md.strip()) > 20:
                            table_strings.append(md)
                    except Exception:
                        # Fallback to to_text() if available
                        if hasattr(t, 'to_text'):
                            txt = t.to_text()
                            if txt and len(txt.strip()) > 20:
                                table_strings.append(txt)
                elif hasattr(t, 'to_text'):
                    txt = t.to_text()
                    if txt and len(txt.strip()) > 20:
                        table_strings.append(txt)
                elif isinstance(t, list):
                    # Convert list-of-lists table to markdown
                    lines = []
                    for row in t:
                        if isinstance(row, (list, tuple)):
                            lines.append("| " + " | ".join(str(c) for c in row) + " |")
                    if lines:
                        table_strings.append("\n".join(lines))
            tables_by_backend[backend_name] = table_strings
            all_tables_raw.extend(table_strings)
            logger.info(f"  {backend_name}: {len(table_strings)} tables found")
        except Exception as e:
            logger.warning(f"  {backend_name} failed: {e}")
            tables_by_backend[backend_name] = []

    # Cross-backend deduplication.
    # Different backends transcribe the SAME physical table with slightly
    # different formatting (subscript chars, spacing, pipe delimiters,
    # newlines). Naive prefix-dedup misses these and the LLM ends up
    # extracting the same gold value 3-5× → over-extraction.
    #
    # Two-pass dedup:
    #   pass 1 — canonical-fingerprint hash (cheap; catches 80%+ of dupes)
    #   pass 2 — group near-identical tables and pick the most complete
    #            version (longest table-text wins, since shorter usually
    #            means truncated by a backend)
    valid_tables = [t for t in all_tables_raw
                    if isinstance(t, str) and len(t.strip()) >= 20]

    def _canon_fingerprint(t: str, n: int = 400) -> str:
        """Normalise a table-text into a comparable fingerprint."""
        import re, unicodedata
        s = unicodedata.normalize("NFKC", t[:n * 3])
        s = s.lower()
        s = re.sub(r"\s+", "", s)              # collapse all whitespace
        s = re.sub(r"[|+\-_=*]", "", s)        # strip table-borders
        s = re.sub(r"[^\w\d.,]", "", s)        # keep only alphanumerics + . ,
        return s[:n]

    # Pass 1 — group by canonical fingerprint
    groups: dict[str, list[str]] = {}
    for t in valid_tables:
        fp = _canon_fingerprint(t)
        groups.setdefault(fp, []).append(t)

    # From each group keep the longest version (most complete transcription)
    unique_tables = [max(grp, key=len) for grp in groups.values()]

    logger.info(
        f"Cross-backend dedup: {len(all_tables_raw)} raw → "
        f"{len(valid_tables)} valid → {len(unique_tables)} unique "
        f"({len(unique_tables)} groups, avg {sum(len(g) for g in groups.values())/max(len(groups),1):.1f} variants/table)"
    )

    # Vision fallback handles the case where text extractors found nothing —
    # we render pages as images and attach them to the returned dict so the
    # domain pipelines can feed them to the vision LLM.
    # (Equivalent to geochem pipeline.py Step 3b-3.)
    page_images: list[dict] = []
    if not unique_tables:
        try:
            from articleminer.pdf_vision import render_data_pages
            logger.info("No tables from text extractors — rendering pages for vision...")
            page_images = render_data_pages(
                pdf_path=pdf_str, content=pdf_content,
                max_pages=4, dpi=150,
            )
            if page_images:
                logger.info(f"Vision: rendered {len(page_images)} data pages")
                tables_by_backend["vision"] = [
                    f"[Vision: {len(page_images)} page images rendered — sent to vision LLM]"
                ]
        except Exception as e:
            logger.warning(f"Vision fallback failed: {e}")

    return {
        "pdf_text": full_text,
        "tables_by_backend": tables_by_backend,
        "all_tables": unique_tables,
        "pdf_content": pdf_content,    # for downstream use
        "pdf_path": pdf_str,
        "page_images": page_images,    # consumed by vision LLM when text fails
    }


# =============================================================================
# Vision LLM extraction helper
# When text-extractors produce nothing, feed rendered pages to the vision LLM
# and parse its output exactly as a table text.
# =============================================================================
VISION_TABLE_PROMPT = """You are viewing {n_pages} rendered page images from a scientific \
paper. Identify every table on these pages and transcribe each table as Markdown \
(| col1 | col2 | ... |). Keep rows and cells exactly as they appear. \
If there is no readable table, output nothing. Output ONLY the Markdown tables, \
separated by blank lines — no prose commentary."""


def extract_tables_via_vision(page_images: list[dict], vision_client, paper_id: str = "") -> list[str]:
    """Ask the vision LLM to transcribe tables from rendered PDF pages.

    Returns a list of markdown-table strings (each ready to feed the
    downstream extraction LLM). Safe to call with empty ``page_images``.
    """
    if not page_images:
        return []

    system = "You transcribe tables from scientific paper pages with high fidelity."
    user = VISION_TABLE_PROMPT.format(n_pages=len(page_images))
    try:
        # geochem ClaudeClient.complete accepts images= kwarg
        raw = vision_client.complete(system, user, max_tokens=8192, images=page_images)
    except Exception as e:
        logger.warning(f"[{paper_id}] vision LLM call failed: {e}")
        return []

    if not raw or not isinstance(raw, str):
        return []

    # Split the response into table blocks. A Markdown-pipe table is a run of
    # consecutive lines each containing at least two '|' characters.
    tables: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if "|" in stripped and stripped.count("|") >= 2:
            current.append(line)
        else:
            if len(current) >= 2:   # need header + at least one data row
                block = "\n".join(current).strip()
                if len(block) > 30:
                    tables.append(block)
            current = []
    # Flush final block
    if len(current) >= 2:
        block = "\n".join(current).strip()
        if len(block) > 30:
            tables.append(block)

    # If the LLM returned a single giant table with no blank lines, the
    # above still works — it just yields one long block.
    logger.info(f"[{paper_id}] vision transcribed {len(tables)} tables")
    return tables


# =============================================================================
# Agentic Self-Correction + Post-Extraction Validation
# (Generic equivalent of geochem's agentic_corrector.pdf_correction_loop and
# extraction_validator.validate_extraction — same architecture, domain-agnostic.)
# =============================================================================
AGENTIC_DIAGNOSIS_SYSTEM = """\
You are an extraction-quality diagnostician. You are given a scientific
table (as text), the predictions an extractor produced from it, and an
estimate of how many entries the paper says should be present.
Your job is to diagnose WHY the extraction count is wrong and propose a
retry strategy. Return ONLY a JSON object:
{
  "likely_cause": "<one short phrase>",
  "retry_hint":   "<one sentence describing what the extractor should do differently>",
  "expected_count_delta": "<'too few' | 'too many' | 'ok'>"
}"""


def agentic_self_correct(
    table_text: str,
    initial_preds: list[dict],
    extract_fn,
    llm_client,
    expected_range: tuple[int, int] | None = None,
    intelligence: dict | None = None,
    paper_id: str = "",
    max_attempts: int = 1,
) -> list[dict]:
    """Generic self-correction loop.

    If ``len(initial_preds)`` is far outside ``expected_range`` (or obviously
    zero), ask the LLM to diagnose the table and retry via ``extract_fn``
    with an appended hint.

    Args:
        table_text : raw table text fed to the extractor.
        initial_preds : predictions from the first pass.
        extract_fn : callable(table_text, hint_text) -> list[dict]. Receives
            an extra sentence of guidance appended to its prompt.
        llm_client : LLM client (needs a .complete_json or .complete method).
        expected_range : (lo, hi) plausible count inferred from intelligence.
        intelligence : paper-intelligence dict (for context in the prompt).
        paper_id : for logging.
        max_attempts : max diagnosis + retry rounds (default 1 — conservative).

    Returns:
        The best prediction list seen across attempts (max-cardinality wins,
        matching geochem's behaviour in ``pdf_correction_loop``).
    """
    # Decide whether correction is warranted.
    #
    # NB: we only fire self-correction on UNDER-extraction (0 or << expected).
    # Firing on over-extraction was a prior bug: paper-intelligence gave
    # a rough paper-level count, divided across tables it over-constrained
    # per-table expectations, and the diagnosis LLM then told the extractor
    # to aggregate legitimate multi-row data into single entries, collapsing
    # real gold tuples. Over-extraction is instead handled by dedup and the
    # post-extraction validation pass.
    n = len(initial_preds)
    needs_fix = (n == 0) or (
        expected_range is not None and n < expected_range[0]
    )
    if not needs_fix:
        return initial_preds

    best = initial_preds
    for attempt in range(1, max_attempts + 1):
        diag_user = (
            f"## TABLE\n{table_text[:2500]}\n\n"
            f"## INITIAL PREDICTIONS ({n} entries)\n"
            f"{json.dumps(initial_preds[:10], indent=2)[:1500]}\n\n"
            f"## EXPECTED RANGE\n"
            f"{expected_range if expected_range else 'unknown'}\n\n"
            f"## INTELLIGENCE\n{json.dumps(intelligence or {}, default=str)[:800]}\n\n"
            f"Diagnose and return the JSON hint object."
        )
        try:
            # Prefer complete_json if the client exposes it (geochem ClaudeClient does)
            if hasattr(llm_client, "complete_json"):
                diag = llm_client.complete_json(
                    system=AGENTIC_DIAGNOSIS_SYSTEM, user=diag_user, max_tokens=512,
                )
            else:
                raw = llm_client.complete(AGENTIC_DIAGNOSIS_SYSTEM, diag_user, max_tokens=512)
                diag = json.loads(raw) if raw and raw.strip().startswith("{") else {}
        except Exception as e:
            logger.warning(f"[{paper_id}] self-correct diagnosis failed: {e}")
            return best

        if not isinstance(diag, dict):
            return best

        hint = (
            f"RETRY HINT (self-correction attempt {attempt}): "
            f"{diag.get('retry_hint', '')} "
            f"(Diagnosed cause: {diag.get('likely_cause', 'unknown')}.)"
        )
        logger.info(f"[{paper_id}] self-correction attempt {attempt}: {hint}")

        try:
            retry_preds = extract_fn(table_text, hint)
        except Exception as e:
            logger.warning(f"[{paper_id}] self-correct retry failed: {e}")
            return best

        # Keep the better of (initial, retry) by "closer to expected, then count".
        if expected_range is not None:
            lo, hi = expected_range
            def _score(preds):
                c = len(preds)
                if lo <= c <= hi:
                    return (2, c)
                # outside range — prefer closer edge
                return (1, -abs(c - (lo + hi) / 2))
            if _score(retry_preds) > _score(best):
                best = retry_preds
        else:
            if len(retry_preds) > len(best):
                best = retry_preds

    return best


VALIDATION_SYSTEM = """\
You are a post-extraction quality controller. Given (a) a short paper
description and (b) a list of extracted entries, your job is to identify
entries that are CLEARLY wrong — e.g. values that do not appear anywhere
in the paper, units impossible for the reported quantity, entries that
were extracted from a reference table not this paper's data.

Be CONSERVATIVE: only flag entries you are >90% confident are invalid.
Return ONLY a JSON object:
{
  "invalid_indices": [ <0-indexed positions in the input list> ],
  "reason": "<one short sentence>"
}
If everything looks fine return {"invalid_indices": [], "reason": "all valid"}."""


def validate_predictions(
    preds: list[dict],
    paper_text: str,
    intelligence: dict | None,
    llm_client,
    domain: str,
    paper_id: str = "",
) -> list[dict]:
    """Generic post-extraction validation (equivalent to geochem's
    ``extraction_validator.validate_extraction``).

    Asks the LLM to review predictions in context, and drops only those
    flagged with high confidence. Hard-coded safety cap: never remove
    more than 20% of predictions (same cap as geochem).
    """
    if not preds:
        return preds

    # Limit context to stay under token budget.
    preds_preview = [
        {i: {k: v for k, v in p.items() if k in ("value", "type", "unit",
            "target", "treatment", "component", "sample_id",
            "metric", "model", "dataset", "task", "parameter_name",
            "attribute_name")}}
        for i, p in enumerate(preds[:50])
    ]

    user = (
        f"## DOMAIN\n{domain}\n\n"
        f"## PAPER TEXT (first 3000 chars)\n{paper_text[:3000]}\n\n"
        f"## EXTRACTED ENTRIES (0-indexed, first 50)\n"
        f"{json.dumps(preds_preview, indent=2)[:4500]}\n\n"
        f"## INTELLIGENCE\n{json.dumps(intelligence or {}, default=str)[:800]}\n\n"
        f"Flag indices that are clearly invalid."
    )

    try:
        if hasattr(llm_client, "complete_json"):
            res = llm_client.complete_json(
                system=VALIDATION_SYSTEM, user=user, max_tokens=512,
            )
        else:
            raw = llm_client.complete(VALIDATION_SYSTEM, user, max_tokens=512)
            res = json.loads(raw) if raw and raw.strip().startswith("{") else {}
    except Exception as e:
        logger.warning(f"[{paper_id}] validation call failed: {e}")
        return preds

    if not isinstance(res, dict):
        return preds

    bad = set(i for i in (res.get("invalid_indices") or []) if isinstance(i, int))
    if not bad:
        return preds

    # Safety cap — match geochem's 20% rule.
    cap = max(1, int(len(preds) * 0.20))
    if len(bad) > cap:
        logger.info(
            f"[{paper_id}] validation wanted to drop {len(bad)} preds but "
            f"cap is {cap} (20%). Keeping first {cap} flagged.")
        bad = set(sorted(bad)[:cap])

    kept = [p for i, p in enumerate(preds) if i not in bad]
    logger.info(
        f"[{paper_id}] validation: {len(preds)}→{len(kept)} "
        f"(dropped {len(preds) - len(kept)}; reason: {res.get('reason', '')[:80]})"
    )
    return kept


# =============================================================================
# Stage 0: Paper Intelligence (equivalent to geochem _extract_paper_intelligence)
# Reads the paper text FIRST, determines what data to expect, then uses that
# to classify each extracted table as relevant or not.
# This is the key to achieving BOTH high precision AND high recall.
# =============================================================================
PAPER_INTELLIGENCE_PROMPT = {
    "chemtables": """You are a medicinal chemistry expert. Read this paper text and answer:
1. What bioassay types are reported? (IC50, EC50, GI50, MIC, or other)
2. How many compound/table entries approximately?
3. What biological targets are tested? (protein names, cell lines, organisms)
4. What units are used? (µM, nM, µg/mL)

Output as JSON:
{"assay_types": [...], "targets": [...], "units": [...], "approx_entries": N}""",

    "discomat": """You are a materials science expert. Read this paper text and answer:
1. What type of materials are studied? (glass, ceramic, etc.)
2. What oxide components are reported? (SiO2, Na2O, PbO, etc.)
3. Are compositions given in mol% or wt%?
4. How many samples/compositions approximately?

Output as JSON:
{"material_type": "...", "components": [...], "unit": "mol or wt", "approx_samples": N}""",

    "mltables": """You are an ML researcher. Read this paper text and answer:
1. What ML tasks are addressed? (classification, generation, etc.)
2. What metrics are reported? (accuracy, F1, BLEU, etc.)
3. What datasets are used?
4. How many models/methods are compared?

Output as JSON:
{"tasks": [...], "metrics": [...], "datasets": [...], "approx_models": N}""",
}

TABLE_CLASSIFY_PROMPT = """Does this table contain {target_type}? Answer ONLY "yes" or "no".

Context from paper: {intelligence_summary}

Table (first 400 chars):
{table_preview}

Answer:"""


def extract_paper_intelligence(paper_text: str, domain: str, llm_client) -> dict:
    """Extract domain-specific intelligence from paper text.
    Equivalent to geochem's _extract_paper_intelligence."""
    system = PAPER_INTELLIGENCE_PROMPT.get(domain, "")
    if not system:
        return {}

    # Truncate paper text
    text = paper_text[:8000] if len(paper_text) > 8000 else paper_text

    try:
        response = llm_client.complete(system, text, max_tokens=512)
        # Parse JSON from response
        import re
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning(f"Paper intelligence failed: {e}")

    return {}


def classify_table_relevance(table_text: str, domain: str, intelligence: dict,
                              llm_client) -> bool:
    """Classify whether a table is relevant using LLM + paper intelligence.
    One lightweight LLM call per table (max 10 tokens output)."""
    if domain == "chemtables":
        target = "bioactivity data (IC50, EC50, GI50, MIC values for compounds)"
    elif domain == "discomat":
        target = "material composition data (oxide percentages)"
    elif domain == "mltables":
        target = "quantitative ML results, dataset statistics, or hyperparameters"
    else:
        return True

    intel_summary = json.dumps(intelligence)[:200] if intelligence else "unknown"
    preview = table_text[:400]

    prompt = TABLE_CLASSIFY_PROMPT.format(
        target_type=target,
        intelligence_summary=intel_summary,
        table_preview=preview,
    )

    try:
        response = llm_client.complete(
            "Answer only yes or no.", prompt, max_tokens=5
        )
        return response.strip().lower().startswith("yes")
    except Exception:
        return True  # on failure, keep the table


def filter_relevant_tables(tables: list[str], domain: str, llm_client,
                            paper_text: str = "") -> tuple[list[str], dict]:
    """Filter extracted tables using paper intelligence + per-table classification.

    Returns (filtered_tables, intelligence_dict).
    The intelligence dict is passed to extraction functions as scope constraint
    (same pattern as geochem's elements_measured).
    """
    if not tables:
        return tables, {}

    # Step 1: Paper intelligence (one call per paper)
    intelligence = extract_paper_intelligence(paper_text, domain, llm_client)
    logger.info(f"Paper intelligence: {json.dumps(intelligence)[:100]}")

    # Step 2: Classify each table
    relevant = []
    for table_text in tables:
        # Quick pre-filter: skip very short non-data content
        if len(table_text.strip()) < 30:
            continue
        if not any(c.isdigit() for c in table_text[:500]):
            continue

        # LLM classification
        is_relevant = classify_table_relevance(
            table_text, domain, intelligence, llm_client
        )
        if is_relevant:
            relevant.append(table_text)

    logger.info(f"Table filter: {len(tables)} -> {len(relevant)} relevant tables")
    return relevant, intelligence


# =============================================================================
# Stage 3: LLM-Based Table Interpretation (domain-specific)
# =============================================================================
def extract_bioactivity_from_table(table_text: str, llm_client, paper_context: str = "",
                                    intelligence: dict = None) -> list[dict]:
    """Use LLM to extract bioactivity tuples from a single table.
    The LLM sees the table text extracted by deterministic backends.
    Intelligence dict constrains extraction scope (like geochem's elements_measured).
    """
    system = CHEMTABLES_TABLE_EXTRACTION_PROMPT
    user = f"Paper context (first 500 chars):\n{paper_context[:500]}\n\n---\n\nTable:\n{table_text}\n\nExtract all bioactivity measurements:"

    # SCOPE CONSTRAINT from paper intelligence (key to precision)
    # Same pattern as geochem: "Focus extraction on ONLY these elements"
    if intelligence:
        scope_parts = []
        if intelligence.get("assay_types"):
            scope_parts.append(f"Assay types: {', '.join(intelligence['assay_types'])}")
        if intelligence.get("targets"):
            scope_parts.append(f"Targets: {', '.join(intelligence['targets'][:10])}")
        if intelligence.get("units"):
            scope_parts.append(f"Units: {', '.join(intelligence['units'])}")
        if scope_parts:
            user += (
                f"\n\n## EXTRACTION SCOPE (from paper analysis)\n"
                f"This paper reports: {'; '.join(scope_parts)}\n"
                f"Extract ONLY bioactivity measurements matching these types/targets. "
                f"Skip physical properties, selectivity ratios, and non-bioactivity values.\n"
            )

    try:
        response = llm_client.complete(system, user, max_tokens=8192)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []

    # Parse response
    results = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "value" in obj and "type" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return results


# =============================================================================
# Stage 4: Ontology Validation
# =============================================================================
def validate_chemtables_extraction(tuples: list[dict]) -> list[dict]:
    """Apply ontology-grounded validation to extracted tuples."""
    from ontology_chemtables import (
        standardize_unit, standardize_target, VALID_ASSAY_TYPES, is_null_value
    )

    validated = []
    for t in tuples:
        # Validate type
        assay_type = t.get("type", "")
        if assay_type not in VALID_ASSAY_TYPES:
            continue  # Skip non-bioactivity extractions

        # Normalize unit
        unit = t.get("unit", "xx")
        if unit != "xx":
            t["unit"] = standardize_unit(unit)

        # Normalize target
        target = t.get("target", "xx")
        if target != "xx":
            t["target"] = standardize_target(target)

        # Check null semantics
        value = t.get("value", "")
        null_type = is_null_value(str(value))
        if null_type == "not_tested":
            continue  # Skip non-measurements

        validated.append(t)

    return validated


# =============================================================================
# Full Pipeline: PDF → Structured Tuples
# =============================================================================
def run_chemtables_pipeline(pdf_path: Path, llm_client,
                             backends: list[str] = None) -> list[dict]:
    """Run the full extraction pipeline on a single ChemTables PDF.

    This is architecturally identical to the geochem pipeline:
      PDF → multi-extractor → LLM interpretation → ontology validation
    """
    # Stages 1+2: PDF text + multi-extractor tables
    extraction = extract_tables_from_paper(pdf_path, backends)

    # Stage 2b: Vision fallback — if text backends produced nothing, ask the
    # vision LLM to transcribe tables from rendered pages.
    if not extraction["all_tables"] and extraction.get("page_images"):
        logger.info(f"[{pdf_path.stem}] text backends empty — using vision LLM")
        vision_tables = extract_tables_via_vision(
            extraction["page_images"], llm_client, paper_id=pdf_path.stem,
        )
        if vision_tables:
            extraction["all_tables"] = vision_tables

    # Stage 0: Paper intelligence + table relevance filter
    relevant_tables, intelligence = filter_relevant_tables(
        extraction["all_tables"], "chemtables", llm_client, extraction["pdf_text"]
    )

    # Stage 3: LLM interprets each relevant table WITH scope constraints
    all_tuples = []
    # Approx per-table count from intelligence (used for self-correction triggers)
    approx_entries = (intelligence or {}).get("approx_entries")
    expected_range = None
    if isinstance(approx_entries, int) and approx_entries > 0 and relevant_tables:
        per_tbl = max(1, approx_entries // max(1, len(relevant_tables)))
        expected_range = (max(1, per_tbl // 3), per_tbl * 3)

    for i, table_text in enumerate(relevant_tables):
        if len(table_text.strip()) < 50:
            continue

        def _extract(tt, extra_hint=""):
            return extract_bioactivity_from_table(
                tt + ("\n\n" + extra_hint if extra_hint else ""),
                llm_client, extraction["pdf_text"], intelligence=intelligence,
            )

        tuples = _extract(table_text)

        # Stage 3b: Agentic self-correction (only fires if first pass looks off)
        tuples = agentic_self_correct(
            table_text, tuples, _extract, llm_client,
            expected_range=expected_range, intelligence=intelligence,
            paper_id=f"{pdf_path.stem}:tbl{i}", max_attempts=1,
        )
        all_tuples.extend(tuples)

    # Stage 4: Ontology validation
    validated = validate_chemtables_extraction(all_tuples)

    # Stage 4b: LLM post-extraction validation (conservative — ≤20% drop)
    validated = validate_predictions(
        validated, extraction["pdf_text"], intelligence, llm_client,
        domain="chemtables", paper_id=pdf_path.stem,
    )

    # Deduplicate by (value, treatment, target, type)
    seen = set()
    deduped = []
    for t in validated:
        key = (str(t.get("value", "")), str(t.get("treatment", "")),
               str(t.get("target", "")), str(t.get("type", "")))
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    logger.info(f"Pipeline: {len(extraction['all_tables'])} tables → "
                f"{len(all_tuples)} raw → {len(validated)} validated → "
                f"{len(deduped)} deduped")

    return deduped


# =============================================================================
# Main runner
# =============================================================================
# =============================================================================
# DiSCoMaT Domain: Materials Science Composition Extraction
# =============================================================================
DISCOMAT_TABLE_EXTRACTION_PROMPT = """You are an expert materials scientist. Given a table extracted from a glass/ceramic science paper, extract ALL material compositions.

For each composition entry, output a JSON object on one line:
{"sample_id": "<sample identifier>", "component": "<oxide formula>", "value": <number>, "unit": "<mol or wt>"}

RULES:
1. Extract the EXACT numerical value from the table cell.
2. component must be a chemical formula (SiO2, Na2O, PbO, B2O3, etc.) — NOT a property name.
3. sample_id = the row identifier (sample number, glass code, or composition label).
4. unit = "mol" for mol%, "wt" for wt%. Infer from table caption or column headers.
5. If a cell contains "-" or "–", it means 0% (component absent). Do NOT extract these.
6. Skip rows for physical properties (density, Tg, refractive index, viscosity, etc.).
7. Skip rows that are clearly NOT composition data (e.g., "Total", "Balance").
8. Do NOT hallucinate. Only extract what is explicitly in the table.
9. Output ONLY JSON lines, no explanation."""


def extract_composition_from_table(table_text: str, llm_client, paper_context: str = "",
                                    intelligence: dict = None) -> list[dict]:
    """Use LLM to extract composition tuples from a single table."""
    system = DISCOMAT_TABLE_EXTRACTION_PROMPT
    user = f"Paper context (first 500 chars):\n{paper_context[:500]}\n\n---\n\nTable:\n{table_text}\n\nExtract all compositions:"

    if intelligence:
        scope_parts = []
        if intelligence.get("components"):
            scope_parts.append(f"Components: {', '.join(intelligence['components'][:15])}")
        if intelligence.get("unit"):
            scope_parts.append(f"Unit: {intelligence['unit']}%")
        if intelligence.get("material_type"):
            scope_parts.append(f"Material: {intelligence['material_type']}")
        if scope_parts:
            user += (
                f"\n\n## EXTRACTION SCOPE (from paper analysis)\n"
                f"This paper reports: {'; '.join(scope_parts)}\n"
                f"Extract ONLY composition data with these components. "
                f"Skip physical properties (density, Tg, refractive index, etc.).\n"
            )

    try:
        response = llm_client.complete(system, user, max_tokens=8192)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []

    results = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "component" in obj and "value" in obj:
                try:
                    obj["value"] = float(obj["value"])
                    results.append(obj)
                except (ValueError, TypeError):
                    continue
        except json.JSONDecodeError:
            continue

    return results


def validate_discomat_extraction(tuples: list[dict]) -> list[dict]:
    """Apply ontology-grounded validation to extracted composition tuples."""
    from ontology_discomat import standardize_component, is_null_value, VALID_UNITS

    validated = []
    for t in tuples:
        component = t.get("component", "")
        canonical = standardize_component(component)
        if canonical:
            t["component"] = canonical

        unit = t.get("unit", "mol")
        if unit not in VALID_UNITS:
            t["unit"] = "mol"

        value_str = str(t.get("value", ""))
        if is_null_value(value_str):
            continue

        validated.append(t)

    return validated


def run_discomat_pipeline(pdf_path: Path, llm_client,
                           backends: list[str] = None) -> list[dict]:
    """Run full extraction pipeline on a single DiSCoMaT PDF."""
    extraction = extract_tables_from_paper(pdf_path, backends)

    if not extraction["all_tables"] and extraction.get("page_images"):
        logger.info(f"[{pdf_path.stem}] text backends empty — using vision LLM")
        vision_tables = extract_tables_via_vision(
            extraction["page_images"], llm_client, paper_id=pdf_path.stem,
        )
        if vision_tables:
            extraction["all_tables"] = vision_tables

    # Stage 0: Paper intelligence + table relevance filter
    relevant_tables, intelligence = filter_relevant_tables(
        extraction["all_tables"], "discomat", llm_client, extraction["pdf_text"]
    )

    all_tuples = []
    approx_samples = (intelligence or {}).get("approx_samples")
    expected_range = None
    if isinstance(approx_samples, int) and approx_samples > 0 and relevant_tables:
        # DiSCoMaT rows are (sample × component); ~5-8 components per sample typical.
        per_tbl = max(1, (approx_samples * 6) // max(1, len(relevant_tables)))
        expected_range = (max(1, per_tbl // 3), per_tbl * 3)

    for i, table_text in enumerate(relevant_tables):
        if len(table_text.strip()) < 50:
            continue

        def _extract(tt, extra_hint=""):
            return extract_composition_from_table(
                tt + ("\n\n" + extra_hint if extra_hint else ""),
                llm_client, extraction["pdf_text"], intelligence=intelligence,
            )

        tuples = _extract(table_text)
        tuples = agentic_self_correct(
            table_text, tuples, _extract, llm_client,
            expected_range=expected_range, intelligence=intelligence,
            paper_id=f"{pdf_path.stem}:tbl{i}", max_attempts=1,
        )
        all_tuples.extend(tuples)

    validated = validate_discomat_extraction(all_tuples)

    # LLM post-extraction validation (conservative — ≤20% drop)
    validated = validate_predictions(
        validated, extraction["pdf_text"], intelligence, llm_client,
        domain="discomat", paper_id=pdf_path.stem,
    )

    # Deduplicate by (sample_id, component, value)
    seen = set()
    deduped = []
    for t in validated:
        key = (str(t.get("sample_id", "")), str(t.get("component", "")),
               float(t.get("value", 0)))
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    logger.info(f"Pipeline: {len(extraction['all_tables'])} tables → "
                f"{len(all_tuples)} raw → {len(validated)} validated → "
                f"{len(deduped)} deduped")

    return deduped


# =============================================================================
# MLTables Domain: ML Paper Results/Hyperparameters Extraction
# =============================================================================
MLTABLES_TABLE_EXTRACTION_PROMPT = """You are an expert ML researcher. Given a table extracted from a machine learning paper, extract ALL quantitative entries (results, dataset statistics, hyperparameters).

For each entry, output a JSON object on one line:
{"value": "<exact number>", "type": "<Result|Data Stat.|Hyper-parameter/Architecture|Other>", ...attributes...}

For type "Result": include "model", "dataset", "metric", "task" if identifiable.
For type "Data Stat.": include "dataset", "attribute name" (e.g., "number of samples", "number of classes").
For type "Hyper-parameter/Architecture": include "parameter/architecture name", "model".

RULES:
1. Extract the EXACT numerical value from the table cell.
2. Classify each value into one of the four types based on context.
3. For Results: the model is usually in the row, the metric in the column header, the dataset from caption or headers.
4. Skip non-quantitative cells (model names, dataset names used as labels).
5. Do NOT hallucinate. Only extract what is explicitly in the table.
6. Output ONLY JSON lines, no explanation."""


def extract_ml_data_from_table(table_text: str, llm_client, paper_context: str = "",
                                intelligence: dict = None) -> list[dict]:
    """Use LLM to extract ML results/stats from a single table."""
    system = MLTABLES_TABLE_EXTRACTION_PROMPT
    user = f"Paper context (first 500 chars):\n{paper_context[:500]}\n\n---\n\nTable:\n{table_text}\n\nExtract all quantitative entries:"

    if intelligence:
        scope_parts = []
        if intelligence.get("metrics"):
            scope_parts.append(f"Metrics: {', '.join(intelligence['metrics'][:10])}")
        if intelligence.get("tasks"):
            scope_parts.append(f"Tasks: {', '.join(intelligence['tasks'][:5])}")
        if intelligence.get("datasets"):
            scope_parts.append(f"Datasets: {', '.join(intelligence['datasets'][:5])}")
        if scope_parts:
            user += (
                f"\n\n## EXTRACTION SCOPE (from paper analysis)\n"
                f"This paper reports: {'; '.join(scope_parts)}\n"
                f"Extract ONLY quantitative entries related to these metrics/tasks/datasets. "
                f"Skip table indices, row numbers, and non-result values.\n"
            )

    try:
        response = llm_client.complete(system, user, max_tokens=8192)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []

    results = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "value" in obj and "type" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return results


def run_mltables_pipeline(pdf_path: Path, llm_client,
                           backends: list[str] = None) -> list[dict]:
    """Run full extraction pipeline on a single MLTables PDF."""
    extraction = extract_tables_from_paper(pdf_path, backends)

    if not extraction["all_tables"] and extraction.get("page_images"):
        logger.info(f"[{pdf_path.stem}] text backends empty — using vision LLM")
        vision_tables = extract_tables_via_vision(
            extraction["page_images"], llm_client, paper_id=pdf_path.stem,
        )
        if vision_tables:
            extraction["all_tables"] = vision_tables

    # Stage 0: Paper intelligence + table relevance filter
    relevant_tables, intelligence = filter_relevant_tables(
        extraction["all_tables"], "mltables", llm_client, extraction["pdf_text"]
    )

    all_tuples = []
    approx_models = (intelligence or {}).get("approx_models")
    expected_range = None
    if isinstance(approx_models, int) and approx_models > 0 and relevant_tables:
        # Rough: ~5-15 quantitative entries per table in ML papers.
        per_tbl = 8
        expected_range = (max(1, per_tbl // 3), per_tbl * 3)

    for i, table_text in enumerate(relevant_tables):
        if len(table_text.strip()) < 50:
            continue

        def _extract(tt, extra_hint=""):
            return extract_ml_data_from_table(
                tt + ("\n\n" + extra_hint if extra_hint else ""),
                llm_client, extraction["pdf_text"], intelligence=intelligence,
            )

        tuples = _extract(table_text)
        tuples = agentic_self_correct(
            table_text, tuples, _extract, llm_client,
            expected_range=expected_range, intelligence=intelligence,
            paper_id=f"{pdf_path.stem}:tbl{i}", max_attempts=1,
        )
        all_tuples.extend(tuples)

    # Ontology validation (type filter) + LLM post-extraction validation
    try:
        from ontology_mltables import VALID_CELL_TYPES, standardize_metric, standardize_task
        typed = []
        for t in all_tuples:
            ttype = t.get("type", "")
            if ttype not in VALID_CELL_TYPES:
                continue
            if "metric" in t and isinstance(t["metric"], str):
                t["metric"] = standardize_metric(t["metric"])
            if "task" in t and isinstance(t["task"], str):
                t["task"] = standardize_task(t["task"])
            typed.append(t)
        all_tuples = typed
    except Exception as e:
        logger.warning(f"mltables ontology validation skipped: {e}")

    all_tuples = validate_predictions(
        all_tuples, extraction["pdf_text"], intelligence, llm_client,
        domain="mltables", paper_id=pdf_path.stem,
    )

    # Deduplicate by (value, type, model/dataset)
    seen = set()
    deduped = []
    for t in all_tuples:
        key = (str(t.get("value", "")), str(t.get("type", "")),
               str(t.get("model", "")), str(t.get("dataset", "")))
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    logger.info(f"Pipeline: {len(extraction['all_tables'])} tables → "
                f"{len(all_tuples)} raw → {len(deduped)} deduped")

    return deduped


# =============================================================================
# Main runner
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-Domain Pipeline Adapter")
    parser.add_argument("--dataset", required=True, choices=["chemtables", "discomat", "mltables"])
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--backends", nargs="+",
                        default=["docling", "marker", "pdfplumber"],
                        help="PDF extraction backends to use")
    parser.add_argument("--paper", help="Run on a single paper (PMC ID)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output directory (to avoid clobbering prior runs)")
    # Per-domain configurability — let us drop components shown to hurt
    # in the v5b ablations (e.g. paper-intelligence over-filters ChemTables).
    parser.add_argument("--skip-intelligence", action="store_true",
                        help="Skip paper-intelligence + per-table relevance filter.")
    parser.add_argument("--skip-self-correct", action="store_true",
                        help="Skip agentic self-correction retry loop.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip post-extraction LLM validation step.")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision-LLM fallback when text backends find no tables.")
    parser.add_argument("--llm-select-backends", action="store_true",
                        help="Let LLM pick the best backend(s) per paper instead of using --backends.")
    parser.add_argument("--llm-dedup", action="store_true",
                        help="After cross-backend dedup, run an LLM-judge pass to merge near-duplicate table variants (catches residual cross-backend dupes).")
    args = parser.parse_args()

    # Apply skip flags by monkey-patching this module's functions.
    if args.skip_intelligence:
        def _no_filter(tables, domain, llm_client, paper_text=""):
            return tables, {}
        globals()["filter_relevant_tables"] = _no_filter
        logger.info("CONFIG: skip_intelligence — paper-intelligence disabled")
    if args.skip_self_correct:
        def _identity(table_text, initial_preds, extract_fn, llm_client,
                      expected_range=None, intelligence=None,
                      paper_id="", max_attempts=1):
            return initial_preds
        globals()["agentic_self_correct"] = _identity
        logger.info("CONFIG: skip_self_correct — agentic retry disabled")
    if args.skip_validation:
        def _identity(preds, paper_text, intelligence, llm_client,
                      domain, paper_id=""):
            return preds
        globals()["validate_predictions"] = _identity
        logger.info("CONFIG: skip_validation — post-extraction validation disabled")
    if args.skip_vision:
        def _no_vision(page_images, vision_client, paper_id=""):
            return []
        globals()["extract_tables_via_vision"] = _no_vision
        logger.info("CONFIG: skip_vision — vision-LLM fallback disabled")

    # LLM-judge dedup wrapper: after extract_tables_from_paper finishes,
    # run a 2nd-pass LLM-based dedup on the unique_tables list.
    if args.llm_dedup:
        _orig_extract_dd = extract_tables_from_paper
        def _extract_with_llm_dedup(pdf_path, backends=None):
            result = _orig_extract_dd(pdf_path, backends)
            client = globals().get("_LLM_SELECT_CLIENT")
            if client is not None and result.get("all_tables"):
                pid = Path(pdf_path).stem
                deduped = llm_dedup_tables(result["all_tables"], client, paper_id=pid)
                result["all_tables"] = deduped
            return result
        globals()["extract_tables_from_paper"] = _extract_with_llm_dedup
        logger.info("CONFIG: llm_dedup — LLM-judge will merge cross-backend duplicates")

    # LLM-driven per-paper backend selection: wrap extract_tables_from_paper
    # to first ask the LLM which backends to use for each PDF, then call the
    # original function with that subset.
    if args.llm_select_backends:
        _orig_extract = extract_tables_from_paper
        # We need a client visible at call time — capture from main's local later.
        # Use a module-level placeholder set just before run_*_pipeline call.
        globals()["_LLM_SELECT_CLIENT"] = None
        available_pool = list(args.backends) if args.backends else \
            ["docling", "marker", "mineru", "pdfplumber", "camelot"]

        def _llm_pick_extract(pdf_path, backends=None):
            client = globals().get("_LLM_SELECT_CLIENT")
            if client is None:
                return _orig_extract(pdf_path, backends)
            # Quick text peek
            try:
                content = extract_pdf(str(pdf_path))
                snippet = get_paper_text_for_llm(content)[:1500]
            except Exception:
                snippet = ""
            chosen = select_backends_for_paper(snippet, client, available=available_pool)
            logger.info(f"[{Path(pdf_path).stem}] LLM picked backends: {chosen}")
            return _orig_extract(pdf_path, chosen)
        globals()["extract_tables_from_paper"] = _llm_pick_extract
        logger.info(f"CONFIG: llm_select_backends — LLM picks per-paper from {available_pool}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load API key
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val

    # Setup output
    run_name = f"{args.dataset}_pipeline_{args.model.replace('/', '_')}"
    if args.suffix:
        run_name = f"{run_name}_{args.suffix}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client (same as geochem)
    llm_client = make_client(args.model)
    # Bind to module global so the llm_select_backends wrapper can use it.
    globals()["_LLM_SELECT_CLIENT"] = llm_client

    if args.dataset == "chemtables":
        # Load gold data for evaluation
        gold_path = ISWC_ROOT / "datasets" / "schema_driven_ie" / "data" / "chemtables" / "test.json"
        with open(gold_path) as f:
            gold_data = json.load(f)

        # Map PMC IDs to their tables
        pmc_to_tables = {}
        for tid, entry in gold_data.items():
            pmc = tid.split("::")[0]
            if pmc not in pmc_to_tables:
                pmc_to_tables[pmc] = {}
            pmc_to_tables[pmc][tid] = entry

        # Find available PDFs
        pdf_dir = CHEMTABLES_PDFS
        available_pdfs = sorted(pdf_dir.glob("PMC*.pdf"))
        print(f"Found {len(available_pdfs)} ChemTables PDFs")

        if args.paper:
            available_pdfs = [p for p in available_pdfs if args.paper in p.name]

        if args.dry_run:
            for pdf in available_pdfs:
                pmc = pdf.stem
                n_tables = len(pmc_to_tables.get(pmc, {}))
                n_gold = sum(len(e["cell_list_gold"]) for e in pmc_to_tables.get(pmc, {}).values())
                print(f"  {pmc}: {n_tables} tables, {n_gold} gold annotations")
            return

        # Run pipeline on each paper
        all_results = {}
        total_time = 0
        for pdf in available_pdfs:
            pmc = pdf.stem
            print(f"\n{'='*60}")
            print(f"Processing {pmc}...")
            print(f"{'='*60}")

            # Skip empty/corrupt PDFs
            pdf_size = pdf.stat().st_size
            if pdf_size < 1000:
                print(f"  SKIPPED: {pmc}.pdf is empty or corrupt ({pdf_size} bytes)")
                all_results[pmc] = []
                continue

            t0 = time.time()
            try:
                tuples = run_chemtables_pipeline(pdf, llm_client, args.backends)
            except Exception as e:
                logger.error(f"  Pipeline failed for {pmc}: {e}")
                tuples = []
            elapsed = time.time() - t0
            total_time += elapsed

            print(f"Extracted {len(tuples)} tuples in {elapsed:.1f}s")

            # Save per-paper results
            paper_result = {
                "pmc_id": pmc,
                "predictions": tuples,
                "num_tables_in_pdf": len(pmc_to_tables.get(pmc, {})),
                "num_gold_annotations": sum(
                    len(e["cell_list_gold"]) for e in pmc_to_tables.get(pmc, {}).values()
                ),
                "elapsed_seconds": round(elapsed, 1),
                "backends": args.backends,
            }
            with open(output_dir / f"{pmc}.json", "w") as f:
                json.dump(paper_result, f, indent=2)

            all_results[pmc] = tuples

        print(f"\n{'='*60}")
        print(f"TOTAL: {sum(len(v) for v in all_results.values())} tuples from "
              f"{len(all_results)} papers in {total_time:.1f}s")
        print(f"Results saved to {output_dir}")

        # Save summary
        summary = {
            "dataset": args.dataset,
            "model": args.model,
            "backends": args.backends,
            "total_papers": len(all_results),
            "total_tuples": sum(len(v) for v in all_results.values()),
            "total_time": round(total_time, 1),
            "per_paper": {
                pmc: {"tuples": len(tuples), "gold": sum(
                    len(e["cell_list_gold"]) for e in pmc_to_tables.get(pmc, {}).values()
                )}
                for pmc, tuples in all_results.items()
            },
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    elif args.dataset == "discomat":
        # Load gold data
        gold_path = ISWC_ROOT / "datasets" / "schema_driven_ie" / "data" / "discomat" / "test.json"
        with open(gold_path) as f:
            gold_data = json.load(f)

        # Map PIIs to their gold tuples
        pii_gold = {}
        for tid, entry in gold_data.items():
            pii = tid.split("::")[0]
            n = len(entry.get("cell_list_gold", []))
            if n > 0:
                if pii not in pii_gold:
                    pii_gold[pii] = 0
                pii_gold[pii] += n

        # Find available PDFs
        pdf_dir = DISCOMAT_PDFS
        available_pdfs = sorted(pdf_dir.glob("S*.pdf"))
        print(f"Found {len(available_pdfs)} DiSCoMaT PDFs")

        if args.paper:
            available_pdfs = [p for p in available_pdfs if args.paper in p.name]

        if args.dry_run:
            for pdf in available_pdfs[:10]:
                pii = pdf.stem
                n_gold = pii_gold.get(pii, 0)
                print(f"  {pii}: {n_gold} gold tuples")
            print(f"  ... ({len(available_pdfs)} total)")
            return

        # Run pipeline on each paper
        all_results = {}
        total_time = 0
        for i, pdf in enumerate(available_pdfs):
            pii = pdf.stem
            if pii not in pii_gold:
                continue  # Skip PDFs without gold annotations

            if i % 10 == 0:
                print(f"\n[{i+1}/{len(available_pdfs)}] Processing {pii}...")
            else:
                print(f"  {pii}...", end=" ", flush=True)

            pdf_size = pdf.stat().st_size
            if pdf_size < 1000:
                print(f"SKIPPED (empty)")
                all_results[pii] = []
                continue

            t0 = time.time()
            try:
                tuples = run_discomat_pipeline(pdf, llm_client, args.backends)
            except Exception as e:
                logger.error(f"Pipeline failed for {pii}: {e}")
                tuples = []
            elapsed = time.time() - t0
            total_time += elapsed

            print(f"{len(tuples)} tuples ({elapsed:.0f}s)")

            paper_result = {
                "pii": pii,
                "predictions": tuples,
                "num_gold_tuples": pii_gold.get(pii, 0),
                "elapsed_seconds": round(elapsed, 1),
                "backends": args.backends,
            }
            with open(output_dir / f"{pii}.json", "w") as f:
                json.dump(paper_result, f, indent=2)

            all_results[pii] = tuples

        print(f"\n{'='*60}")
        print(f"TOTAL: {sum(len(v) for v in all_results.values())} tuples from "
              f"{len(all_results)} papers in {total_time:.1f}s")
        print(f"Results saved to {output_dir}")

        summary = {
            "dataset": "discomat",
            "model": args.model,
            "backends": args.backends,
            "total_papers": len(all_results),
            "total_tuples": sum(len(v) for v in all_results.values()),
            "total_time": round(total_time, 1),
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    elif args.dataset == "mltables":
        # Load gold data
        gold_path = ISWC_ROOT / "datasets" / "schema_driven_ie" / "data" / "mltables" / "test.json"
        with open(gold_path) as f:
            gold_data = json.load(f)

        # Map arXiv IDs to their gold
        arxiv_gold = {}
        for tid, entry in gold_data.items():
            arxiv_id = tid.rsplit("_table", 1)[0]
            if arxiv_id not in arxiv_gold:
                arxiv_gold[arxiv_id] = 0
            arxiv_gold[arxiv_id] += len(entry["cell_list_gold"])

        pdf_dir = MLTABLES_PDFS
        available_pdfs = sorted(pdf_dir.glob("*.pdf"))
        print(f"Found {len(available_pdfs)} MLTables PDFs")

        if args.paper:
            available_pdfs = [p for p in available_pdfs if args.paper in p.name]

        if args.dry_run:
            for pdf in available_pdfs:
                aid = pdf.stem
                n_gold = arxiv_gold.get(aid, 0)
                print(f"  {aid}: {n_gold} gold annotations")
            return

        all_results = {}
        total_time = 0
        for i, pdf in enumerate(available_pdfs):
            aid = pdf.stem
            if aid not in arxiv_gold:
                continue

            print(f"[{i+1}/{len(available_pdfs)}] {aid}...", end=" ", flush=True)

            pdf_size = pdf.stat().st_size
            if pdf_size < 1000:
                print("SKIPPED (empty)")
                all_results[aid] = []
                continue

            t0 = time.time()
            try:
                tuples = run_mltables_pipeline(pdf, llm_client, args.backends)
            except Exception as e:
                logger.error(f"Pipeline failed for {aid}: {e}")
                tuples = []
            elapsed = time.time() - t0
            total_time += elapsed

            print(f"{len(tuples)} tuples ({elapsed:.0f}s)")

            paper_result = {
                "arxiv_id": aid,
                "predictions": tuples,
                "num_gold_annotations": arxiv_gold.get(aid, 0),
                "elapsed_seconds": round(elapsed, 1),
                "backends": args.backends,
            }
            with open(output_dir / f"{aid}.json", "w") as f:
                json.dump(paper_result, f, indent=2)

            all_results[aid] = tuples

        print(f"\n{'='*60}")
        print(f"TOTAL: {sum(len(v) for v in all_results.values())} tuples from "
              f"{len(all_results)} papers in {total_time:.1f}s")
        print(f"Results saved to {output_dir}")

        summary = {
            "dataset": "mltables",
            "model": args.model,
            "backends": args.backends,
            "total_papers": len(all_results),
            "total_tuples": sum(len(v) for v in all_results.values()),
            "total_time": round(total_time, 1),
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
