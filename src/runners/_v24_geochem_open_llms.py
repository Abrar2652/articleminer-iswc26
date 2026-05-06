#!/usr/bin/env python3
"""Run the geochem ExtractionPipeline with an open-source local-HF LLM.

Why this script exists:
    The paper claims open 7-8B LLMs ``cannot follow GeoScholar-28's
    210-column schema''. That hand-wavy claim is replaced here with a
    measured number: each open backbone runs the full geochem pipeline on
    all 28 GT papers and is scored with the standard 4-tier evaluator.

Design notes:
    * `use_vision=False`: open 7-8B have no vision capability; the
      `LocalHFClient` returns "" when images are passed, but disabling
      vision outright avoids spinning the renderer and saves ~140s/paper.
    * Deposit classifier is wrapped in try/except inside `pipeline.run()`
      already, so a model that emits unparseable JSON for the 189-type
      taxonomy does not crash the run; the row simply gets `deposit_type=null`.
    * Per-paper failures are caught locally so one bad paper does not
      kill the model's entire batch.

Usage::
    python scripts/_v24_geochem_open_llms.py --model qwen25-7b
    python scripts/_v24_geochem_open_llms.py --model mistral-7b-v03
    python scripts/_v24_geochem_open_llms.py --model llama31-8b
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                # ${REPO_ROOT}/..
sys.path.insert(0, str(ROOT))                              # find geochem_benchmark
from geochem_benchmark.llm_clients import LocalHFClient
from geochem_benchmark.pipeline import ExtractionPipeline

ISWC_ROOT = Path(__file__).resolve().parents[1]            # iswc2026
RESULTS_DIR = ISWC_ROOT / "results"
GT_DIR = ROOT / "geochem_benchmark" / "ground_truth_corrected"
GEOCHEM_DATA = ROOT / "geochem_benchmark" / "data"


def list_gt_papers() -> list[Path]:
    return sorted([p for p in GT_DIR.glob("*.xlsx")
                   if not p.name.startswith("_") and not p.name.startswith(".")])


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def find_pdf_for_paper(paper_id: str) -> Path | None:
    """GT names are 'Author_et_al_YEAR[_extra]'; PDFs are 'YEAR[A-E]?_Author_etal[N]?.pdf'.
    (logic mirrored from _v23_geochem_abl_haiku28.py)"""
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
    """Locate any supplementary spreadsheets for a paper (xlsx/csv)."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["qwen25-7b", "qwen25-32b",
                            "mistral-7b-v02", "mistral-7b-v03",
                            "llama31-8b"])
    p.add_argument("--out-suffix", default="v24_open",
                   help="Output dir suffix; full dir = "
                        "geochem_pipeline_<model>_<suffix>")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N papers (for smoke testing)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Cap LLM max_new_tokens (e.g. 512) to short-circuit "
                        "hallucination spirals on small open models")
    args = p.parse_args()

    out_dir = RESULTS_DIR / f"geochem_pipeline_{args.model}_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = list_gt_papers()
    papers = [p for p in papers if find_pdf_for_paper(p.stem) is not None]
    if args.limit:
        papers = papers[:args.limit]
    print(f"=== {args.model} on geochem ({len(papers)} papers) → {out_dir.name} ===")
    print(f"    use_vision=False; deposit-classifier failures swallowed by pipeline.run()")

    print(f"    Loading {args.model} ...")
    t_load = time.time()
    client = LocalHFClient(model=args.model)
    print(f"    Loaded in {time.time() - t_load:.1f}s")

    # Optional: clamp max_new_tokens to reduce hallucination cost on small models.
    if args.max_tokens:
        _orig_complete = client.complete
        cap = args.max_tokens
        def _capped_complete(system, user, max_tokens=4096, images=None):
            return _orig_complete(system, user,
                                  max_tokens=min(max_tokens, cap),
                                  images=images)
        client.complete = _capped_complete  # type: ignore
        print(f"    max_new_tokens capped at {cap} for this run")

    pipeline = ExtractionPipeline(
        llm_client=client,
        vision_client=client,        # ignored when use_vision=False
        use_vision=False,
        use_self_correction=True,
        use_llm_table_filter=False,
    )

    t_total = time.time()
    summary = {
        "model": args.model,
        "n_papers": len(papers),
        "n_zero": 0,
        "n_failed": 0,
        "started": datetime.now().isoformat(),
        "per_paper": {},
    }

    for i, gt_csv in enumerate(papers, 1):
        paper_id = gt_csv.stem
        f_out = out_dir / f"{paper_id}.json"
        xlsx_out = out_dir / f"extraction_{paper_id}.xlsx"
        if f_out.exists() and xlsx_out.exists():
            print(f"  [{i}/{len(papers)}] {paper_id}: already done — skipping")
            continue

        pdf = find_pdf_for_paper(paper_id)
        supps = find_supps(paper_id)
        print(f"  [{i}/{len(papers)}] {paper_id}: pdf={pdf.name if pdf else 'NONE'} supps={len(supps)}")
        t_paper = time.time()
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

        elapsed = time.time() - t_paper
        json.dump({
            "paper_id": paper_id,
            "model": args.model,
            "predictions": preds,
            "n_predictions": n_preds,
            "elapsed_seconds": round(elapsed, 1),
            "errors": errors_str,
        }, open(f_out, "w"), indent=2, default=str)
        summary["per_paper"][paper_id] = {
            "n": n_preds, "elapsed": round(elapsed, 1),
        }
        print(f"      → {n_preds} samples ({elapsed:.0f}s)")

    summary["finished"] = datetime.now().isoformat()
    summary["total_seconds"] = round(time.time() - t_total, 1)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nTOTAL: {sum(p['n'] for p in summary['per_paper'].values())} samples"
          f" across {len(summary['per_paper'])} papers"
          f" ({summary['n_zero']} zero-paper, {summary['n_failed']} crashed)"
          f" in {summary['total_seconds']:.0f}s")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
