#!/usr/bin/env python3
"""
PDF-text few-shot baseline runner.

For each PDF in a domain's test set:
  1. Read full PDF text (no table extraction, no ontology, no pipeline).
  2. Send text + a minimal few-shot prompt to an LLM.
  3. Save predictions in {domain}_fewshot_pdf_{model}{suffix}/{pdf_id}.json

This is the honest baseline for the "raw PDF → structured output" comparison.
Same LLM, same PDF, no pipeline — isolates the value of our full pipeline.

Usage:
    python3 run_fewshot_pdf.py --domain chemtables --model claude-sonnet-4-6 [--shots 3] [--suffix v2]
"""
from __future__ import annotations
import argparse, json, os, sys, time, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # src/ -> articleminer
sys.path.insert(0, str(Path(__file__).parent))

from articleminer.pdf_reader import extract_pdf, get_paper_text_for_llm
from articleminer.llm_clients import ClaudeClient

# Disable marker text extraction on clusters with old CUDA (set via env).
if os.environ.get("ISWC_DISABLE_MARKER", "").lower() in ("1", "true", "yes"):
    from articleminer import tabledetector as _td
    _td.marker_pdf_to_markdown = lambda *a, **k: ""

ISWC = ROOT
RES = ISWC / "results"
DATA = ISWC / "datasets"

PDF_DIRS = {
    "chemtables": ISWC / "datasets" / "chemtables_pdfs",
    "discomat":   ISWC / "datasets" / "discomat_pdfs",
    "mltables":   ISWC / "datasets" / "mltables_pdfs",
}
GOLD = {
    "chemtables": DATA / "schema_driven_ie/data/chemtables/test.json",
    "discomat":   DATA / "schema_driven_ie/data/discomat/test.json",
    "mltables":   DATA / "schema_driven_ie/data/mltables/test.json",
}
PDF_GLOB = {
    "chemtables": "PMC*.pdf",
    "discomat":   "S*.pdf",
    "mltables":   "*.pdf",
}

SYSTEM_PROMPTS = {
    "chemtables": """Extract ALL bioactivity measurements (IC50, EC50, GI50, MIC) from this paper.
For each measurement, output one JSON per line:
{"value": "<number>", "type": "<IC50|EC50|GI50|MIC>", "target": "<protein/cell/organism>", "treatment": "<compound id>", "unit": "<µM|nM|µg/mL>"}
Output ONLY JSON lines — no prose.""",

    "discomat": """Extract ALL material-composition entries from this glass/ceramic paper.
For each entry, output one JSON per line:
{"sample_id": "<sample id>", "component": "<oxide formula>", "value": <number>, "unit": "<mol or wt>"}
Skip "-" values. Output ONLY JSON lines.""",

    "mltables": """Extract ALL quantitative entries from this ML paper's tables. For each, output one JSON per line:
{"value": "<number>", "type": "<Result|Data Stat.|Hyper-parameter/Architecture|Other>", "model": "<if applicable>", "dataset": "<if applicable>", "metric": "<if applicable>"}
Output ONLY JSON lines.""",
}


def load_fewshot_examples(domain: str, n: int) -> str:
    """Build example blocks from the dev split (small tables).

    Returns "" when n=0. Kept short so the prompt stays under token limits.
    """
    if n <= 0:
        return ""
    dev_path = DATA / f"schema_driven_ie/data/{domain}/dev.json"
    if not dev_path.exists():
        return ""
    with open(dev_path) as f:
        dev = json.load(f)
    blocks = []
    for tid, entry in dev.items():
        golds = entry.get("cell_list_gold") or []
        if not golds:
            continue
        table = entry.get("table_processed", "")[:500] or \
                entry.get("table_source", "")[:500] or \
                entry.get("table_code", "")[:500]
        lines = []
        for g in golds[:3]:
            if domain == "discomat" and isinstance(g, list):
                lines.append(json.dumps({
                    "sample_id": g[0], "component": g[1],
                    "value": g[2], "unit": g[3],
                }))
            elif isinstance(g, dict):
                lines.append(json.dumps({
                    k: v for k, v in g.items()
                    if k not in ("char_index", "cell_index")
                }))
        blocks.append(f"TABLE:\n{table}\nOUTPUT:\n" + "\n".join(lines))
        if len(blocks) >= n:
            break
    if not blocks:
        return ""
    return "\n\nExamples:\n" + "\n---\n".join(blocks) + "\n\nNow extract from the paper text:"


def pdf_ids_with_gold(domain: str) -> set[str]:
    with open(GOLD[domain]) as f:
        data = json.load(f)
    out = set()
    for tid, entry in data.items():
        if not entry.get("cell_list_gold"):
            continue
        if domain == "mltables":
            out.add(tid.rsplit("_table", 1)[0])
        else:
            out.add(tid.split("::")[0])
    return out


def parse_preds(raw: str, domain: str) -> list[dict]:
    preds = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "value" not in obj:
            continue
        if domain == "discomat":
            try:
                obj["value"] = float(obj["value"])
            except Exception:
                continue
        preds.append(obj)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    choices=["chemtables", "discomat", "mltables"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--suffix", default="",
                    help="Output dir suffix (to avoid clobbering).")
    ap.add_argument("--max-chars", type=int, default=40000,
                    help="Truncate PDF text to this many chars.")
    ap.add_argument("--paper", help="Run only the matching PDF (ID substring).")
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

    out_dir = RES / f"{args.domain}_fewshot_pdf_{args.model.replace('/', '_')}"
    if args.suffix:
        out_dir = RES / f"{args.domain}_fewshot_pdf_{args.model.replace('/', '_')}_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the same multi-provider dispatcher as pipeline_adapter
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline_adapter import make_client
    client = make_client(args.model)
    system = SYSTEM_PROMPTS[args.domain]
    examples = load_fewshot_examples(args.domain, args.shots)

    pdf_dir = PDF_DIRS[args.domain]
    need = pdf_ids_with_gold(args.domain)
    pdfs = sorted(p for p in pdf_dir.glob(PDF_GLOB[args.domain])
                  if p.stem in need)
    if args.paper:
        pdfs = [p for p in pdfs if args.paper in p.name]
    print(f"[{args.domain}] {len(pdfs)} PDFs to run, model={args.model}, "
          f"shots={args.shots}, out={out_dir}")

    total_preds = 0; t_start = time.time()
    for i, pdf in enumerate(pdfs, 1):
        pid = pdf.stem
        out_file = out_dir / f"{pid}.json"
        if out_file.exists():
            print(f"  [{i}/{len(pdfs)}] {pid}: already done, skipping")
            continue

        print(f"  [{i}/{len(pdfs)}] {pid}...", end=" ", flush=True)
        t0 = time.time()
        try:
            pdf_content = extract_pdf(str(pdf))
            text = get_paper_text_for_llm(pdf_content)[:args.max_chars]
        except Exception as e:
            print(f"PDF read FAILED: {e}")
            json.dump({"predictions": [], "error": str(e)},
                      open(out_file, "w"), indent=2)
            continue

        user = f"PAPER TEXT:\n{text}\n\n{examples or 'Extract all relevant entries:'}"
        try:
            raw = client.complete(system, user, max_tokens=8192)
        except Exception as e:
            print(f"LLM FAILED: {e}")
            json.dump({"predictions": [], "error": str(e)},
                      open(out_file, "w"), indent=2)
            continue

        preds = parse_preds(raw, args.domain)
        total_preds += len(preds)
        elapsed = time.time() - t0
        print(f"{len(preds)} preds ({elapsed:.0f}s)")

        json.dump({"pdf_id": pid, "predictions": preds,
                   "elapsed_seconds": round(elapsed, 1),
                   "raw_preview": raw[:500]},
                  open(out_file, "w"), indent=2)

    total_elapsed = time.time() - t_start
    summary = {
        "domain": args.domain, "model": args.model,
        "shots": args.shots, "n_pdfs": len(pdfs),
        "total_preds": total_preds,
        "elapsed_seconds": round(total_elapsed, 1),
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"\nDone: {total_preds} preds from {len(pdfs)} PDFs "
          f"in {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
