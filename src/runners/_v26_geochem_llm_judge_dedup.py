#!/usr/bin/env python3
"""Geochem pipeline with explicit LLM-judge cross-backend dedup.

Fills the previously-n/a row "5-backend + LLM-judge dedup" in
Table~\\ref{tab:backend_ablation}. The geochem pipeline already keeps
multiple backends' textual transcriptions of the same physical table in
``backend_table_texts``; this wrapper adds the same two-pass dedup that
``pipeline_adapter.py`` uses for the cross-domain pipelines:

    Pass 1: canonical NFKC fingerprint group-by, keep longest per group.
    Pass 2: LLM-judge merges near-duplicate transcriptions that differ in
            border characters, OCR noise, etc.

Default ablation budget: 26 papers, Haiku 4.5 (matches the existing
26-paper ablation row so the +LLM-judge column is directly comparable).

Usage::
    python scripts/_v26_geochem_llm_judge_dedup.py
"""
from __future__ import annotations
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_adapter import llm_dedup_tables  # reuse cross-domain helper
from geochem_benchmark.llm_clients import ClaudeClient
from geochem_benchmark.pipeline import ExtractionPipeline

ISWC_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ISWC_ROOT / "results"
GT_DIR = ROOT / "geochem_benchmark" / "ground_truth_corrected"
GEOCHEM_DATA = ROOT / "geochem_benchmark" / "data"

MODEL = "claude-haiku-4-5-20251001"
RUN_TAG = "haiku26_llmjudge"


def _canon_fingerprint(t: str, n: int = 400) -> str:
    """NFKC-normalised, border-stripped fingerprint of a table-text snippet."""
    s = unicodedata.normalize("NFKC", t[:n * 3])
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[|+\-_=*]", "", s)
    s = re.sub(r"[^\w\d.,]", "", s)
    return s[:n]


def install_llm_judge_dedup_patch(client) -> callable:
    """Monkey-patch ExtractionPipeline._direct_pdf_table_extraction so that
    its returned ``backend_table_texts`` is run through canonical-fingerprint
    + LLM-judge dedup. Returns an undo() callable.
    """
    import geochem_benchmark.pipeline as gp
    original = gp.ExtractionPipeline._direct_pdf_table_extraction

    def _patched(self, *args, **kwargs):
        samples, backend_table_texts, extracted_pages = original(
            self, *args, **kwargs)
        if not backend_table_texts:
            return samples, backend_table_texts, extracted_pages

        # Pass 1: canonical fingerprint group-by (keeps longest per group).
        groups: dict[str, list[str]] = {}
        for t in backend_table_texts:
            if not isinstance(t, str) or len(t.strip()) < 20:
                continue
            fp = _canon_fingerprint(t)
            groups.setdefault(fp, []).append(t)
        pass1 = [max(grp, key=len) for grp in groups.values()]
        n_in, n_pass1 = len(backend_table_texts), len(pass1)

        # Pass 2: LLM-judge merges remaining near-duplicates.
        n_pass2 = n_pass1
        try:
            if len(pass1) > 1:
                pass2 = llm_dedup_tables(pass1, client, paper_id="geochem")
                n_pass2 = len(pass2)
            else:
                pass2 = pass1
        except Exception as e:
            print(f"  (LLM-judge dedup error, falling back to pass-1: {e})")
            pass2 = pass1
        print(f"  cross-backend dedup: {n_in} → {n_pass1} (canonical)"
              f" → {n_pass2} (LLM-judge)")
        return samples, pass2, extracted_pages

    gp.ExtractionPipeline._direct_pdf_table_extraction = _patched

    def undo():
        gp.ExtractionPipeline._direct_pdf_table_extraction = original
    return undo


def list_gt_papers() -> list[Path]:
    return sorted([p for p in GT_DIR.glob("*.xlsx")
                   if not p.name.startswith("_") and not p.name.startswith(".")])


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def find_pdf_for_paper(paper_id: str) -> Path | None:
    m = re.match(r"(.+?)_et_al_(\d{4})", paper_id)
    if not m:
        return None
    author_raw, year = m.group(1), m.group(2)
    author = _ascii(author_raw).split("_")[-1]
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
    m = re.match(r"(.+?)_et_al_(\d{4})", paper_id)
    if not m:
        return []
    author_raw, year = m.group(1), m.group(2)
    author = _ascii(author_raw).split("_")[-1]
    auth_re = re.compile(rf"(^|[^a-z]){re.escape(author)}([^a-z]|$)")
    supp_dir = GEOCHEM_DATA / "Spreadsheets"
    out = []
    if supp_dir.exists():
        for f in sorted(list(supp_dir.glob("*.xlsx")) + list(supp_dir.glob("*.csv"))):
            n = _ascii(f.stem)
            if year in n and auth_re.search(n):
                out.append(f)
    return out


def main() -> None:
    out_dir = RESULTS_DIR / f"geochem_pipeline_abl_5backend_llmjudge_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = list_gt_papers()
    papers = [p for p in papers if find_pdf_for_paper(p.stem) is not None]
    # Match the 26-paper ablation set used by the rest of Table 3.
    if len(papers) > 26:
        papers = papers[:26]
    print(f"=== 5-backend + LLM-judge dedup on geochem ({len(papers)} papers, {MODEL}) ===")
    print(f"    output dir: {out_dir.name}")

    client = ClaudeClient(model=MODEL)
    pipeline = ExtractionPipeline(
        llm_client=client,
        vision_client=client,
        use_vision=True,
        use_self_correction=True,
        use_llm_table_filter=False,
    )
    undo = install_llm_judge_dedup_patch(client)

    t_total = time.time()
    summary = {
        "model": MODEL, "n_papers": len(papers),
        "n_zero": 0, "n_failed": 0,
        "started": datetime.now().isoformat(),
        "per_paper": {},
    }
    try:
        for i, gt_csv in enumerate(papers, 1):
            paper_id = gt_csv.stem
            f_out = out_dir / f"{paper_id}.json"
            xlsx_out = out_dir / f"extraction_{paper_id}.xlsx"
            if f_out.exists() and xlsx_out.exists():
                print(f"  [{i}/{len(papers)}] {paper_id}: cached")
                continue
            pdf = find_pdf_for_paper(paper_id)
            supps = find_supps(paper_id)
            print(f"  [{i}/{len(papers)}] {paper_id}: pdf={pdf.name if pdf else 'NONE'} supps={len(supps)}")
            t0 = time.time()
            n_preds = 0
            errors_str = None
            try:
                result = pipeline.run(pdf, supps or [])
                preds = []
                for r in (result.samples or []):
                    if hasattr(r, "model_dump"):
                        preds.append(r.model_dump())
                    elif isinstance(r, dict):
                        preds.append(r)
                    else:
                        preds.append(str(r))
                n_preds = len(preds)
                try:
                    if hasattr(result, "to_excel"):
                        result.to_excel(xlsx_out)
                except Exception as ex:
                    print(f"      (xlsx save failed: {ex})")
                errors_str = result.errors if hasattr(result, "errors") else None
            except Exception as e:
                print(f"      FAILED: {type(e).__name__}: {e}")
                preds = []
                errors_str = [f"{type(e).__name__}: {e}"]
                summary["n_failed"] += 1

            if n_preds == 0:
                summary["n_zero"] += 1

            elapsed = time.time() - t0
            json.dump({
                "paper_id": paper_id, "model": MODEL,
                "predictions": preds, "n_predictions": n_preds,
                "elapsed_seconds": round(elapsed, 1),
                "errors": errors_str,
            }, open(f_out, "w"), indent=2, default=str)
            summary["per_paper"][paper_id] = {
                "n": n_preds, "elapsed": round(elapsed, 1),
            }
            print(f"      → {n_preds} samples ({elapsed:.0f}s)")
    finally:
        undo()

    summary["finished"] = datetime.now().isoformat()
    summary["total_seconds"] = round(time.time() - t_total, 1)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nTOTAL: {sum(p['n'] for p in summary['per_paper'].values())} samples"
          f" across {len(summary['per_paper'])} papers"
          f" ({summary['n_zero']} zero, {summary['n_failed']} crashed)"
          f" in {summary['total_seconds']:.0f}s")
    print(f"Results: {out_dir}")
    print(f"\nNext: score with the 4-tier Evaluator on this directory's xlsx files")
    print(f"      to populate Table~\\ref{{tab:backend_ablation}} P/R/F1/Overall.")


if __name__ == "__main__":
    main()
