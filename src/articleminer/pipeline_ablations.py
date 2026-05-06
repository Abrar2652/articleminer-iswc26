#!/usr/bin/env python3
"""
Component-ablation harness.

Runs a domain pipeline with individual components disabled via monkey-
patches of `pipeline_adapter`, so each ablation isolates one component's
contribution. Every variant writes to its own output dir.

Usage:
    python3 pipeline_ablations.py --dataset chemtables --model claude-sonnet-4-6
"""
from __future__ import annotations
import argparse, json, os, sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ABLATIONS = {
    # name:            (human-readable description, patch-fn)
    "no_ontology":      "Skip ontology validation (VALID_* checks, standardize_*)",
    "no_self_correct":  "Skip agentic_self_correct retry loop",
    "no_vision":        "Skip vision LLM fallback when text backends empty",
    "no_validation":    "Skip post-extraction LLM validation",
    "no_intelligence":  "Skip paper-intelligence scope constraint",
    "full":             "No ablation — same as main pipeline (sanity check)",
}


def patch_for(ablation: str, pa_mod):
    """Return a closure that (a) monkey-patches pa_mod and (b) returns an
    undo callable so we can restore between ablations."""
    originals = {}

    if ablation == "no_ontology":
        def _noop_chem(tuples): return tuples
        def _noop_disco(tuples): return tuples
        originals["validate_chemtables_extraction"] = pa_mod.validate_chemtables_extraction
        originals["validate_discomat_extraction"] = pa_mod.validate_discomat_extraction
        pa_mod.validate_chemtables_extraction = _noop_chem
        pa_mod.validate_discomat_extraction = _noop_disco

    elif ablation == "no_self_correct":
        def _passthrough(table_text, initial_preds, extract_fn, llm_client,
                         expected_range=None, intelligence=None, paper_id="",
                         max_attempts=1):
            return initial_preds
        originals["agentic_self_correct"] = pa_mod.agentic_self_correct
        pa_mod.agentic_self_correct = _passthrough

    elif ablation == "no_vision":
        def _noop_vision(page_images, vision_client, paper_id=""): return []
        originals["extract_tables_via_vision"] = pa_mod.extract_tables_via_vision
        pa_mod.extract_tables_via_vision = _noop_vision

    elif ablation == "no_validation":
        def _passthrough(preds, paper_text, intelligence, llm_client,
                         domain, paper_id=""):
            return preds
        originals["validate_predictions"] = pa_mod.validate_predictions
        pa_mod.validate_predictions = _passthrough

    elif ablation == "no_intelligence":
        def _empty_intel(paper_text, domain, llm_client):
            return {}
        originals["extract_paper_intelligence"] = pa_mod.extract_paper_intelligence
        pa_mod.extract_paper_intelligence = _empty_intel

    elif ablation == "full":
        pass

    else:
        raise ValueError(f"unknown ablation: {ablation}")

    def undo():
        for name, fn in originals.items():
            setattr(pa_mod, name, fn)
    return undo


def run_one_ablation(dataset: str, model: str, ablation: str, backends: list[str]):
    """Run the full dataset under one ablation. Writes to
    `{dataset}_pipeline_{model}_abl_{ablation}/`."""
    import pipeline_adapter as pa
    from pipeline_adapter import (
        run_chemtables_pipeline, run_discomat_pipeline, run_mltables_pipeline,
        ISWC_ROOT, CHEMTABLES_PDFS, DISCOMAT_PDFS, MLTABLES_PDFS, RESULTS_DIR,
    )
    from geochem_benchmark.llm_clients import ClaudeClient

    run_fn = {
        "chemtables": run_chemtables_pipeline,
        "discomat":   run_discomat_pipeline,
        "mltables":   run_mltables_pipeline,
    }[dataset]
    pdf_dir = {
        "chemtables": CHEMTABLES_PDFS,
        "discomat":   DISCOMAT_PDFS,
        "mltables":   MLTABLES_PDFS,
    }[dataset]
    pdf_glob = {"chemtables": "PMC*.pdf", "discomat": "S*.pdf",
                "mltables": "*.pdf"}[dataset]

    out = RESULTS_DIR / f"{dataset}_pipeline_{model.replace('/', '_')}_abl_{ablation}"
    out.mkdir(parents=True, exist_ok=True)

    # Filter to PDFs that have gold annotations (avoid wasted work)
    gold_path = ISWC_ROOT / "datasets/schema_driven_ie/data" / dataset / "test.json"
    with open(gold_path) as f:
        gold_data = json.load(f)
    wanted = set()
    for tid, entry in gold_data.items():
        if not entry.get("cell_list_gold"):
            continue
        if dataset == "mltables":
            wanted.add(tid.rsplit("_table", 1)[0])
        else:
            wanted.add(tid.split("::")[0])
    pdfs = sorted(p for p in pdf_dir.glob(pdf_glob) if p.stem in wanted)
    print(f"[{ablation}] {dataset}: {len(pdfs)} PDFs, out={out}")

    undo = patch_for(ablation, pa)
    client = ClaudeClient(model=model)

    t_start = time.time()
    try:
        for i, pdf in enumerate(pdfs, 1):
            pid = pdf.stem
            f = out / f"{pid}.json"
            if f.exists():
                continue
            t0 = time.time()
            try:
                preds = run_fn(pdf, client, backends)
            except Exception as e:
                print(f"  [{i}/{len(pdfs)}] {pid}: FAILED — {e}")
                preds = []
            elapsed = time.time() - t0
            print(f"  [{i}/{len(pdfs)}] {pid}: {len(preds)} preds ({elapsed:.0f}s)")
            json.dump({"pdf_id": pid, "predictions": preds,
                       "num_gold_annotations": sum(
                           len(gold_data[tid]["cell_list_gold"])
                           for tid in gold_data
                           if (tid.rsplit("_table", 1)[0] if dataset == "mltables"
                               else tid.split("::")[0]) == pid
                       ),
                       "ablation": ablation,
                       "elapsed_seconds": round(elapsed, 1)},
                      open(f, "w"), indent=2)
    finally:
        undo()

    summary = {
        "ablation": ablation, "dataset": dataset, "model": model,
        "n_pdfs": len(pdfs), "elapsed_seconds": round(time.time() - t_start, 1),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"[{ablation}] done in {time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["chemtables", "discomat", "mltables"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--backends", nargs="+",
                    default=["docling", "pdfplumber"])
    ap.add_argument("--ablations", nargs="+",
                    default=list(ABLATIONS.keys()),
                    help="Ablations to run (default: all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load API keys from scripts/.env
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v.strip().strip('"').strip("'")

    print(f"Running {len(args.ablations)} ablation(s) on {args.dataset}")
    for abl in args.ablations:
        if abl not in ABLATIONS:
            print(f"  skip unknown: {abl}")
            continue
        print(f"\n{'='*60}")
        print(f"ABLATION: {abl} — {ABLATIONS[abl]}")
        print(f"{'='*60}")
        run_one_ablation(args.dataset, args.model, abl, args.backends)


if __name__ == "__main__":
    main()
