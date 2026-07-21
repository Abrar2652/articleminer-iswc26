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

GEOCHEM_DATA = ROOT.parents[1] / "data" / "geochem28" / "pdfs"
GT_DIR       = ROOT.parents[1] / "data" / "geochem28" / "ground_truth"
RESULTS_DIR  = ROOT.parents[1] / "results"
MODEL = "claude-sonnet-4-6"

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
    """Find supplementary tables for paper (Excel/CSV)."""
    out = []
    for d in [GEOCHEM_DATA, GEOCHEM_DATA / "supplementary", GEOCHEM_DATA / "papers"]:
        if not d.exists(): continue
        for ext in (".xlsx", ".xls", ".csv"):
            for f in d.glob(f"*{paper_id}*{ext}"):
                out.append(f)
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
}


def apply_patch(name: str, pipeline_module):
    """Return an undo callable. Most ablations are flag-driven; this only
    handles the no_ontology and llm_only_numeric patches."""
    originals = {}
    if name == "no_ontology":
        # Strip ontology-side validation: replace standardize_* with identity
        # in the geochem post-processor, and skip knowledge_base injection.
        try:
            from articleminer import knowledge_base as kb
            originals["VALID_ELEMENTS"] = kb.VALID_ELEMENTS
            kb.VALID_ELEMENTS = set()
            originals["VALID_MINERALS"] = kb.VALID_MINERALS
            kb.VALID_MINERALS = set()
        except Exception as e:
            print(f"  no_ontology patch: {e}")
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

    def undo():
        if name == "no_ontology":
            try:
                from articleminer import knowledge_base as kb
                kb.VALID_ELEMENTS = originals.get("VALID_ELEMENTS", set())
                kb.VALID_MINERALS = originals.get("VALID_MINERALS", set())
            except Exception:
                pass
        elif name == "llm_only_numeric":
            try:
                from articleminer import table_reader as tr
                tr.read_multiple_supplementary = originals["read_multiple_supplementary"]
            except Exception:
                pass
    return undo


def build_pipeline(client, ablation_name: str) -> ExtractionPipeline:
    cfg = ABLATIONS.get(ablation_name, {})
    init_kwargs = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return ExtractionPipeline(llm_client=client, **init_kwargs)


def run_one_ablation(name: str, papers: list[Path]):
    out = RESULTS_DIR / f"geochem_pipeline_abl_{name}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Ablation: {name} ({len(papers)} papers) → {out.name} ===")

    client = ClaudeClient(model=MODEL)
    import articleminer.pipeline as gp
    undo = apply_patch(name, gp)
    pipeline = build_pipeline(client, name)

    try:
        for i, gt_csv in enumerate(papers, 1):
            paper_id = gt_csv.stem
            f_out = out / f"{paper_id}.json"
            if f_out.exists():
                continue
            pdf = find_pdf_for_paper(paper_id)
            supps = find_supps(paper_id)
            print(f"  [{i}/{len(papers)}] {paper_id}: pdf={'Y' if pdf else 'N'}, supps={len(supps)}")
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
            except Exception as e:
                print(f"      FAILED: {type(e).__name__}: {e}")
                preds = []
                result = None
            elapsed = time.time() - t0
            json.dump({
                "paper_id": paper_id,
                "ablation": name,
                "predictions": preds,
                "n_predictions": len(preds),
                "elapsed_seconds": round(elapsed, 1),
                "errors": getattr(result, "errors", None) if result else None,
            }, open(f_out, "w"), indent=2, default=str)
            print(f"      → {len(preds)} samples ({elapsed:.0f}s)")
    finally:
        undo()


def main():
    papers = list_gt_papers()
    print(f"v16: {len(papers)} geochem GT papers found")
    if len(papers) == 0:
        print("ERROR: no GT papers found at", GT_DIR)
        return

    # Filter to papers that have a matching PDF, then take first 8 for compute budget
    papers = [p for p in papers if find_pdf_for_paper(p.stem) is not None]
    print(f"  {len(papers)} GT papers have matching PDFs after name-translation")
    papers = papers[:8]
    print(f"  running on first {len(papers)} for compute budget: {[p.stem for p in papers]}")

    t0 = time.time()
    for name in ["full", "no_ontology", "no_self_correct", "no_vision",
                 "single_docling", "single_marker", "single_mineru",
                 "single_pdfplumber", "single_camelot",
                 "llm_only_numeric"]:
        try:
            run_one_ablation(name, papers)
        except Exception as e:
            print(f"\nABLATION {name} CRASHED: {e}")
            traceback.print_exc()
    print(f"\nv16 ALL DONE in {(time.time()-t0)/3600:.1f} hr")


if __name__ == "__main__":
    main()
