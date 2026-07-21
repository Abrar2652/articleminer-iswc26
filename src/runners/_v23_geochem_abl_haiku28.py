#!/usr/bin/env python3
"""
v16: Geochem ablations on the primary benchmark.

Closes E3b/c/d/e gaps from exp_design — ablations on the geochem domain
(28 papers) using monkey-patches around the geochem ExtractionPipeline.

Variants:
  E3a: Full pipeline (use existing v8 numbers — skip)
  E3b: Without ontology (no knowledge_base / standardize_*)
  E3c: Single extractor — runs each of {docling, marker, mineru,
       pdfplumber, camelot} alone (5 sub-runs)
  E3d: Without self-correction (use_self_correction=False)
  E3e: LLM-only numerical extraction (replace deterministic parser
       with LLM-on-table numeric extraction)

For E3a-d we use existing geochem flags / monkey-patch on the LLM
extraction stage only. For E3e we do a heavier patch.

Output: results/geochem_ablations_<variant>/<paper_id>.json
Each holds the 4-tier scores and per-element diff for comparison.
"""
from __future__ import annotations
import os, sys, json, time, traceback, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[0]))                # src/ -> articleminer

# Load API keys from .env
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v.strip().strip('"').strip("'")

from articleminer.pipeline import ExtractionPipeline
from articleminer.tabledetector import TableDetectorBackend
from articleminer.llm_clients import ClaudeClient
from articleminer.paper_registry import get_processable_papers

# ─────────────────────────────────────────────────────────────────────
# Mineru hard-disable: PaddleOCR weights missing in our env triggers a
# heap-corruption crash (malloc_consolidate) that kills the parent
# process. Documented in §6 (Limitations). Wrap _extract_with_mineru
# to return empty results without entering the broken codepath.
# ─────────────────────────────────────────────────────────────────────
import articleminer.tabledetector as _td_mod
def _mineru_noop(pdf_path, pages, min_rows, min_cols):
    """No-op mineru extractor — returns empty tables and a metrics record
    flagging that mineru was skipped due to missing OCR weights."""
    metrics = _td_mod.TableDetectionMetrics(
        backend_used="mineru",
        tables_found=0, data_tables_found=0, pages_scanned=0,
        time_ms=0.0,
        errors=["mineru disabled in this env: PaddleOCR weight ch_PP-OCRv3_det_infer.pth missing; "
                "basic-mode fallback triggers malloc_consolidate"],
    )
    return [], metrics
_td_mod._extract_with_mineru = _mineru_noop
print("[mineru-noop] patched _extract_with_mineru to return empty (env: missing PaddleOCR weights)")

GEOCHEM_DATA = ROOT.parents[1] / "data" / "geochem28" / "pdfs"
GT_DIR       = ROOT.parents[1] / "data" / "geochem28" / "ground_truth"
RESULTS_DIR  = ROOT.parents[1] / "results"
MODEL = "claude-haiku-4-5-20251001"
RUN_TAG = "haiku28"  # isolates outputs from the earlier 8-paper Sonnet ablations

# 28 GT papers (from geochem WEEKLY_PROGRESS_REPORT)
def list_gt_papers():
    return sorted([p for p in GT_DIR.glob("*.xlsx")
                   if not p.name.startswith("_")
                   and not p.name.startswith(".")])


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def find_pdf_for_paper(paper_id: str) -> Path | None:
    """GT names are 'Author_et_al_YEAR[_extra]'; PDFs are 'YEAR[A-E]?_Author_etal[N]?.pdf'.
    Translate then glob; pick first match (skip supplementary)."""
    m = re.match(r"(.+?)_et_al_(\d{4})", paper_id)
    if not m:
        return None
    author_raw, year = m.group(1), m.group(2)
    # for hyphenated/multi-name authors (e.g. "Bertrandsson_Erlandsson"), use last token
    author = _ascii(author_raw).split("_")[-1]
    # Author must match as a whole token (word-boundary), not a substring
    # — "He" vs "chen" should NOT match.
    auth_re = re.compile(rf"(^|[^a-z]){re.escape(author)}(etal|[^a-z]|$)")
    for d in [GEOCHEM_DATA, GEOCHEM_DATA / "papers", GEOCHEM_DATA / "pdfs"]:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdf")):
            n = _ascii(f.stem)
            if "supp" in n or "guide" in n:
                continue
            if re.match(rf"^{year}[a-e]?_", n) and auth_re.search(n):
                return f
        for f in sorted(d.glob("*.pdf")):
            n = _ascii(f.stem)
            if "supp" in n or "guide" in n:
                continue
            if year in n and auth_re.search(n):
                return f
    return None


def find_supps(paper_id: str) -> list[Path]:
    """Find supplementary tables for paper (Excel/CSV).

    Searches all known supp locations (the canonical batch_runner uses
    paper_registry which knows about data/Spreadsheets/; we mirror that
    here so ablation runs include supp parsing identically to the headline
    pipeline). The earlier ablation runs (pre-2026-04-28) were missing the
    Spreadsheets/ entry and consequently ran PDF-only on GeoScholar.
    """
    # Translate GT name → "YEAR_Author_etal" stem used in data/Spreadsheets/
    out = []
    seen = set()
    m = re.match(r"(.+?)_et_al_(\d{4})", paper_id)
    stems = []
    if m:
        author_raw, year = m.group(1), m.group(2)
        author = _ascii(author_raw).split("_")[-1]
        stems.extend([f"{year}_{author}", f"{year}_{author_raw.replace('_', '_')}"])
    for d in [GEOCHEM_DATA / "Spreadsheets",
              GEOCHEM_DATA, GEOCHEM_DATA / "supplementary", GEOCHEM_DATA / "papers"]:
        if not d.exists(): continue
        for ext in (".xlsx", ".xls", ".csv"):
            # First try translated-stem glob (e.g. "2018_Yuan_etal.xlsx" for
            # "Yuan_et_al_2018"), then fall back to a fuzzy paper-id glob.
            globs = [f"*{stem}*{ext}" for stem in stems] + [f"*{paper_id}*{ext}"]
            for g in globs:
                for f in d.glob(g):
                    n = _ascii(f.stem)
                    if "guide" in n: continue
                    if str(f) not in seen:
                        out.append(f); seen.add(str(f))
    return out


# =============================================================================
# Ablation variants
# =============================================================================
ABLATIONS = {
    "full":          {},
    "no_ontology":   {"_patch": "no_ontology"},
    "no_self_correct": {"use_self_correction": False},
    "no_vision":     {"use_vision": False},
    "single_docling":    {"table_detector_backend": TableDetectorBackend.DOCLING},
    "single_marker":     {"table_detector_backend": TableDetectorBackend.MARKER},
    "single_mineru":     {"table_detector_backend": TableDetectorBackend.MINERU},
    "single_pdfplumber": {"table_detector_backend": TableDetectorBackend.PDFPLUMBER},
    "single_camelot":    {"table_detector_backend": TableDetectorBackend.CAMELOT},
    # E3e — replace deterministic parser with LLM-only numerical extraction
    "llm_only_numeric":  {"_patch": "llm_only_numeric"},
    # External-benchmark parity: Stage 7 self-validator + Stage 0 paper-intelligence
    "no_validation":   {"_patch": "no_validation"},
    "no_intelligence": {"_patch": "no_intelligence"},
    # Backend subsets: multi-backend consensus across a chosen subset
    "two_backend_dp":  {"_patch": "two_backend_dp"},   # docling + pdfplumber
    "three_backend_no_mineru": {"_patch": "three_backend_no_mineru"},
    "four_backend":    {"_patch": "four_backend"},     # no mineru
    # Leave-one-out: run with all 5 PDF backends except the one named.
    "loo_docling":     {"_patch": "loo_docling"},      # full minus docling
    "loo_marker":      {"_patch": "loo_marker"},       # full minus marker
    "loo_mineru":      {"_patch": "loo_mineru"},       # full minus mineru (== four_backend)
    "loo_pdfplumber":  {"_patch": "loo_pdfplumber"},   # full minus pdfplumber
    "loo_camelot":     {"_patch": "loo_camelot"},      # full minus camelot
}


def apply_patch(name: str, pipeline_module):
    """Return an undo callable. Most ablations are flag-driven; this only
    handles the no_ontology and llm_only_numeric patches."""
    originals = {}
    if name == "no_ontology":
        # Strip ontology validation by replacing the public validate/enrich
        # entry point used by pipeline.py with a passthrough. Also disable
        # the picklist validator, the method-standardizer, and the inferred-
        # field helpers so no ontology lookup mutates extracted values.
        try:
            from articleminer import knowledge_base as kb
            import articleminer.pipeline as gp_mod
            # Pipeline imports `validate_and_enrich_metadata` directly into
            # its module namespace, so we have to patch BOTH the source
            # (kb) and the pipeline-side bound reference (gp_mod).
            originals["validate_and_enrich_metadata_kb"] = kb.validate_and_enrich_metadata
            originals["validate_and_enrich_metadata_pipeline"] = gp_mod.validate_and_enrich_metadata
            def _passthrough_validate(metadata):
                return metadata if isinstance(metadata, dict) else {}
            kb.validate_and_enrich_metadata = _passthrough_validate
            gp_mod.validate_and_enrich_metadata = _passthrough_validate

            originals["validate_against_picklist"] = kb.validate_against_picklist
            kb.validate_against_picklist = lambda field, val: (True, val)

            originals["standardize_method"] = kb.standardize_method
            kb.standardize_method = lambda raw: raw

            originals["normalize_method"] = kb.normalize_method
            kb.normalize_method = lambda m: m

            originals["infer_deposit_environment"] = kb.infer_deposit_environment
            kb.infer_deposit_environment = lambda dt: None
            originals["infer_deposit_group"] = kb.infer_deposit_group
            kb.infer_deposit_group = lambda dt: None
            originals["infer_mineral_class"] = kb.infer_mineral_class
            kb.infer_mineral_class = lambda m: (None, None)

            print(f"  no_ontology patch: applied (validate_and_enrich, picklist, method/deposit/mineral inference disabled)")
        except Exception as e:
            print(f"  no_ontology patch: FAILED {type(e).__name__}: {e}")
            raise  # don't silently produce Full-pipeline numbers under the wrong label
    elif name == "llm_only_numeric":
        # Replace the deterministic table_reader with an LLM-prompt-only
        # extractor that asks the LLM to read all numeric values from each
        # extracted table text.
        try:
            from articleminer import table_reader as tr
            originals["read_multiple_supplementary"] = tr.read_multiple_supplementary
            def _stub(*a, **k):
                # Skip deterministic parsing — let LLM extract numerics from
                # the raw table text. This is intentionally degraded to
                # measure the deterministic parser's value.
                return None
            tr.read_multiple_supplementary = _stub
        except Exception as e:
            print(f"  llm_only_numeric patch: {e}")
    elif name == "no_validation":
        # Disable Stage 7 self-validator: make validate_extraction a no-op.
        try:
            from articleminer import extraction_validator as ev
            originals["validate_extraction"] = ev.validate_extraction
            class _StubResult:
                def __init__(self, n=0):
                    self.rows_before = n; self.rows_after = n
                    self.rows_removed = 0; self.issues_found = []
            def _stub(*a, **k):
                df = k.get("samples_df")
                n = len(df) if df is not None else 0
                return _StubResult(n)
            ev.validate_extraction = _stub
        except Exception as e:
            print(f"  no_validation patch: {e}")
    elif name == "no_intelligence":
        # Disable Stage 0 paper-intelligence: return an empty PaperIntelligence
        # so no per-paper scope constraint is injected downstream.
        try:
            import articleminer.pipeline as gp
            originals["_extract_paper_intelligence"] = gp.ExtractionPipeline._extract_paper_intelligence
            _Empty = gp.PaperIntelligence  # dataclass with safe defaults
            def _stub(self, *a, **k):
                return _Empty()
            gp.ExtractionPipeline._extract_paper_intelligence = _stub
        except Exception as e:
            print(f"  no_intelligence patch: {e}")
    elif name in ("two_backend_dp", "three_backend_no_mineru", "four_backend",
                   "loo_docling", "loo_marker", "loo_mineru",
                   "loo_pdfplumber", "loo_camelot"):
        # Monkey-patch the multi-backend extractor to iterate only over a subset.
        try:
            import articleminer.pipeline as gp
            from articleminer.tabledetector import (
                extract_tables_as_text, TableDetectorBackend as TB)
            originals["_extract_all_backends_from_pdf"] = gp.ExtractionPipeline._extract_all_backends_from_pdf
            ALL5 = [TB.DOCLING, TB.MARKER, TB.MINERU, TB.PDFPLUMBER, TB.CAMELOT]
            LOO_DROP = {
                "loo_docling":    TB.DOCLING,
                "loo_marker":     TB.MARKER,
                "loo_mineru":     TB.MINERU,
                "loo_pdfplumber": TB.PDFPLUMBER,
                "loo_camelot":    TB.CAMELOT,
            }
            subset = {
                "two_backend_dp": [TB.DOCLING, TB.PDFPLUMBER],
                "three_backend_no_mineru": [TB.DOCLING, TB.MARKER, TB.PDFPLUMBER],
                "four_backend": [TB.DOCLING, TB.MARKER, TB.PDFPLUMBER, TB.CAMELOT],
            }
            if name in LOO_DROP:
                drop = LOO_DROP[name]
                subset_list = [b for b in ALL5 if b != drop]
            else:
                subset_list = subset[name]
            def _patched(self, pdf_path):
                all_tables = []
                for backend in subset_list:
                    try:
                        tables = extract_tables_as_text(str(pdf_path), backend=backend)
                        if tables: all_tables.extend(tables)
                    except Exception:
                        pass
                return all_tables
            gp.ExtractionPipeline._extract_all_backends_from_pdf = _patched
        except Exception as e:
            print(f"  {name} patch: {e}")

    def undo():
        if name == "no_ontology":
            try:
                from articleminer import knowledge_base as kb
                import articleminer.pipeline as gp_mod
                if "validate_and_enrich_metadata_kb" in originals:
                    kb.validate_and_enrich_metadata = originals["validate_and_enrich_metadata_kb"]
                if "validate_and_enrich_metadata_pipeline" in originals:
                    gp_mod.validate_and_enrich_metadata = originals["validate_and_enrich_metadata_pipeline"]
                for fname in ("validate_against_picklist", "standardize_method",
                              "normalize_method", "infer_deposit_environment",
                              "infer_deposit_group", "infer_mineral_class"):
                    if fname in originals:
                        setattr(kb, fname, originals[fname])
            except Exception:
                pass
        elif name == "llm_only_numeric":
            try:
                from articleminer import table_reader as tr
                tr.read_multiple_supplementary = originals["read_multiple_supplementary"]
            except Exception:
                pass
        elif name == "no_validation":
            try:
                from articleminer import extraction_validator as ev
                ev.validate_extraction = originals["validate_extraction"]
            except Exception:
                pass
        elif name == "no_intelligence":
            try:
                import articleminer.pipeline as gp
                gp.ExtractionPipeline._extract_paper_intelligence = originals["_extract_paper_intelligence"]
            except Exception:
                pass
        elif name in ("two_backend_dp", "three_backend_no_mineru", "four_backend",
                       "loo_docling", "loo_marker", "loo_mineru",
                       "loo_pdfplumber", "loo_camelot"):
            try:
                import articleminer.pipeline as gp
                gp.ExtractionPipeline._extract_all_backends_from_pdf = originals["_extract_all_backends_from_pdf"]
            except Exception:
                pass
    return undo


def build_pipeline(client, ablation_name: str) -> ExtractionPipeline:
    cfg = ABLATIONS.get(ablation_name, {})
    init_kwargs = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return ExtractionPipeline(llm_client=client, **init_kwargs)


def run_one_ablation(name: str, papers: list):
    """`papers` is now a list of ResolvedPaper objects from paper_registry,
    covering all 28 GT entries including the 2 reuse papers. The registry
    handles paper-id → PDF/supp resolution including non-trivial naming
    patterns (e.g. "Bertrandsson_Erlandsson_et_al_2022_reprocessed" → the
    parent 2025 PDF + supp), so this script no longer needs its own
    find_pdf_for_paper / find_supps."""
    out = RESULTS_DIR / f"geochem_pipeline_abl_{name}_{RUN_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Ablation: {name} ({len(papers)} papers) → {out.name} ===")

    client = ClaudeClient(model=MODEL)
    import articleminer.pipeline as gp
    undo = apply_patch(name, gp)
    pipeline = build_pipeline(client, name)

    try:
        for i, paper in enumerate(papers, 1):
            paper_id = paper.id
            f_out = out / f"{paper_id}.json"
            xlsx_out = out / f"extraction_{paper_id}.xlsx"
            if f_out.exists() and xlsx_out.exists():
                continue
            pdf = paper.pdf_path
            supps = paper.supplementary_paths or []
            print(f"  [{i}/{len(papers)}] {paper_id}: pdf={pdf.name if pdf else 'N'}, supps={len(supps)}")
            if not pdf:
                continue
            t0 = time.time()
            try:
                result = pipeline.run(pdf, supps or [])
                preds = [
                    {**(r.metadata or {}), **(r.values or {}), "sample_id": r.sample_id}
                    if hasattr(r, "metadata")
                    else dict(r) if isinstance(r, dict) else str(r)
                    for r in (result.samples or [])
                ]
                # Also save full 210-column xlsx so the 4-tier Evaluator can re-score.
                try:
                    if hasattr(result, "to_excel"):
                        result.to_excel(xlsx_out)
                except Exception as ex:
                    print(f"      (xlsx save failed: {ex})")
            except Exception as e:
                print(f"      FAILED: {type(e).__name__}: {e}")
                preds = []
                result = None
            elapsed = time.time() - t0
            json.dump({
                "paper_id": paper_id,
                "ablation": name,
                "model": MODEL,
                "predictions": preds,
                "n_predictions": len(preds),
                "elapsed_seconds": round(elapsed, 1),
                "errors": getattr(result, "errors", None) if result else None,
            }, open(f_out, "w"), indent=2, default=str)
            print(f"      → {len(preds)} samples ({elapsed:.0f}s)")
    finally:
        undo()


def main():
    # Use paper_registry — knows all 28 GT entries (including 2 reuse papers
    # whose PDFs/supps are explicitly mapped to parent-paper resources).
    project_root = ROOT.parents[1] / "data" / "geochem28"
    papers = get_processable_papers(project_root)
    print(f"v23 ({MODEL}): {len(papers)} geochem GT papers resolved via registry")
    if len(papers) == 0:
        print("ERROR: no processable papers from paper_registry")
        return
    print(f"  running on ALL {len(papers)} papers")

    t0 = time.time()
    # Apr 2026: 12-variant grid for the proper Table 3 backend ablation.
    # The supp-parser bug in find_supps() (Spreadsheets/ not searched) is
    # fixed above, so single-backend rows now run with supp parsing enabled
    # (apples-to-apples with the headline 5-backend pipeline).
    #
    # Single-backend (5): each PDF parser in isolation, with supp parser.
    # Leave-one-out (5): full minus one PDF parser, with supp parser.
    # NOTE: "full" reference comes from results/ablations_geochem/four_backend/
    #   (28-paper, 75.49% overall). The "loo_mineru" run is also "four_backend"
    #   = full minus mineru, included here under its LOO name for symmetry.
    variant_order = [
        "single_docling", "single_marker", "single_mineru",
        "single_pdfplumber", "single_camelot",
        "loo_docling", "loo_marker", "loo_mineru",
        "loo_pdfplumber", "loo_camelot",
    ]
    # Allow command-line subselection: python _v23 ... variant_a variant_b
    if len(sys.argv) > 1:
        variant_order = [v for v in sys.argv[1:] if v in ABLATIONS]
        print(f"Running subselection: {variant_order}")
    for name in variant_order:
        try:
            run_one_ablation(name, papers)
        except Exception as e:
            print(f"\nABLATION {name} CRASHED: {e}")
            traceback.print_exc()
    print(f"\nv23 ALL DONE in {(time.time()-t0)/3600:.1f} hr")


if __name__ == "__main__":
    main()
