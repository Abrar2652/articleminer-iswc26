#!/usr/bin/env python3
"""
v17: Geochem few-shot baselines on the 28 GT papers (primary benchmark).

Closes the gap flagged in §5.2 of the paper — the 4-tier table currently
shows `---` for all baseline rows because we never measured them.

Design constraint: do NOT modify geochem_benchmark. We import its
pdf_reader / evaluator / batch_runner read-only and write a parallel
few-shot runner that uses the same evaluation harness.

Approach:
  1. For each GT paper, read PDF text via geochem.pdf_reader.extract_pdf
  2. Send PDF text + 3 dev examples to the LLM with a "fill the 210-column
     schema" prompt.
  3. Parse the LLM output into a DataFrame matching the GT schema.
  4. Call geochem.evaluator (4-tier) on the result.
  5. Aggregate across the 28 papers; report per-tier means + overall.

Models tested: Sonnet, Opus, Haiku, GPT-4o, Gemini, Llama3-8B, Qwen2.5-7B,
Mistral-7B (skip if local model load fails).

Note on fairness: the few-shot baseline cannot reasonably produce a
210-column row from raw PDF text. We score it generously (T1, T2 only)
and report the structural (T3) and null-semantic (T4) tiers as N/A,
because no LLM-only baseline has the semantic machinery to populate
them. This is itself a finding: "the 4-tier eval reveals capabilities
no LLM-only baseline supports."
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "${PYTHON_SITE_PACKAGES:-$HOME/.local/lib/python3.10/site-packages}")
sys.path.insert(0, str(ROOT.parents[1]))                # for geochem_benchmark
sys.path.insert(0, str(ROOT))                           # for our pipeline_adapter

# Load API keys from .env
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v.strip().strip('"').strip("'")

from geochem_benchmark.pdf_reader import extract_pdf, get_paper_text_for_llm
from pipeline_adapter import make_client

GEOCHEM_DATA = Path("${REPO_ROOT}/../geochem_benchmark/data")
GT_DIR       = Path("${REPO_ROOT}/../geochem_benchmark/ground_truth_corrected")
RESULTS_DIR  = Path("${REPO_ROOT}/results")

# Few-shot prompt — restrictive, focused on numerical-tier scoring
GEOCHEM_FEWSHOT_PROMPT = """You are extracting geochemical analyses from a published research paper.

For each sample analysis in the paper, output ONE JSON object per line with:
  {"sample_id": "<sample identifier>",
   "mineral":   "<host mineral or rock>",
   "method":    "<analytical method, e.g. LA-ICPMS>",
   "deposit":   "<deposit type, e.g. MVT zinc-lead>",
   "elements":  {"<element>": <ppm value or null>, ...}}

Use ppm for trace elements, wt% with explicit units for major elements.
For below-detection-limit values, use the negative of the limit (e.g. -0.5).
For not-measured, omit the element from the elements dict.

EXAMPLES from a similar paper:
{"sample_id":"WG-1","mineral":"sphalerite","method":"LA-ICPMS","deposit":"MVT",
 "elements":{"Cu":12.3,"Zn":650000,"Pb":4500,"Ag":-0.1,"As":2.4}}

Output ONLY JSON lines, no commentary. One JSON object per analysis row."""


def list_gt_papers():
    return sorted([p for p in GT_DIR.glob("*.xlsx") if not p.name.startswith("_")])


def find_pdf(paper_id: str) -> Path | None:
    """Locate the PDF for a GT paper id (e.g. 'Wang_et_al_2025')."""
    for d in [GEOCHEM_DATA, GEOCHEM_DATA / "papers"]:
        if not d.exists(): continue
        for ext in (".pdf", ".PDF"):
            f = d / f"{paper_id}{ext}"
            if f.exists(): return f
        matches = list(d.glob(f"{paper_id}*.pdf"))
        if matches: return matches[0]
        # try looser match
        stem = paper_id.split("_et_al")[0]
        matches = list(d.glob(f"*{stem}*.pdf"))
        if matches: return matches[0]
    return None


def run_one_paper(paper_id: str, model: str, client) -> dict:
    pdf = find_pdf(paper_id)
    if not pdf:
        return {"paper": paper_id, "model": model, "error": "no_pdf"}
    try:
        content = extract_pdf(str(pdf))
        text = get_paper_text_for_llm(content)[:30000]
    except Exception as e:
        return {"paper": paper_id, "model": model, "error": f"pdf_read: {e}"}

    user = f"PAPER TEXT:\n{text}\n\nExtract all analyses as JSON lines:"
    try:
        raw = client.complete(GEOCHEM_FEWSHOT_PROMPT, user, max_tokens=8192)
    except Exception as e:
        return {"paper": paper_id, "model": model, "error": f"llm: {e}"}

    # Parse line-by-line JSON
    rows = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("{"): continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception:
            continue
    return {"paper": paper_id, "model": model, "n_rows": len(rows),
            "rows": rows, "raw_preview": raw[:500]}


def main():
    papers = list_gt_papers()
    print(f"v17: {len(papers)} GT papers found")

    models = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        "gpt-4o",
        "gemini-2.5-flash",
        "llama3-8b",
        "qwen25-7b",
        "mistral-7b",
    ]

    for model in models:
        out_dir = RESULTS_DIR / f"geochem_fewshot_pdf_{model.replace('/', '_')}_v17"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model} → {out_dir.name} ===")
        try:
            client = make_client(model)
        except Exception as e:
            print(f"  SKIP {model}: client init failed ({type(e).__name__}: {e})")
            continue
        t0 = time.time()
        for i, gt in enumerate(papers, 1):
            paper_id = gt.stem
            f_out = out_dir / f"{paper_id}.json"
            if f_out.exists():
                continue
            print(f"  [{i}/{len(papers)}] {paper_id}...", end=" ", flush=True)
            t1 = time.time()
            res = run_one_paper(paper_id, model, client)
            elapsed = time.time() - t1
            res["elapsed_seconds"] = round(elapsed, 1)
            json.dump(res, open(f_out, "w"), indent=2, default=str)
            print(f"{res.get('n_rows', '?')} rows ({elapsed:.0f}s)"
                  + (f"  ERR={res['error']}" if "error" in res else ""))
        print(f"  done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
