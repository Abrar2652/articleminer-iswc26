#!/usr/bin/env python3
"""v20: Chunked few-shot geochem extraction.

Fix for v17's tiny outputs (avg 12-25 rows/paper while gold has 100s):
walk the PDF text in 6k-char chunks, call the LLM per chunk with a
3-shot prompt + 16k max_tokens, accumulate rows. Each LLM ends up
producing ~50-200 rows per paper, which is what the 4-tier evaluator
needs to score meaningfully.

Output: results/geochem_fewshot_pdf_<model>_v20/<paper>.json (same
schema as v17 — {paper, model, n_rows, rows, raw_preview}).
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[0]))  # src/ -> articleminer

env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v.strip().strip('"').strip("'")

from articleminer.pdf_reader import extract_pdf, get_paper_text_for_llm
from pipeline_adapter import make_client

GT_DIR      = ROOT.parents[1] / "data" / "geochem28" / "ground_truth"
RESULTS_DIR = ROOT.parents[1] / "results"
GEOCHEM_DATA = ROOT.parents[1] / "data" / "geochem28" / "pdfs"

PROMPT = """You are extracting geochemical analyses from a chunk of a published research paper.

For EACH sample analysis present in this chunk, output ONE JSON object per line:
  {"sample_id": "<exact sample identifier as in paper>",
   "mineral":   "<host mineral or rock>",
   "method":    "<analytical method, e.g. LA-ICPMS>",
   "elements":  {"<element>": <ppm value>, ...}}

Rules:
- Use ppm for trace elements; convert wt% → ppm (×10000).
- Use the SAMPLE ID exactly as written in the paper (do not invent or shorten).
- For below-detection-limit values, use the negative of the limit (e.g. -0.5).
- Skip sample identifiers; if the chunk has no analytical data, output nothing.
- Output ONLY JSON lines.

EXAMPLES:
{"sample_id":"WG-1","mineral":"sphalerite","method":"LA-ICPMS","elements":{"Cu":12.3,"Zn":650000,"Pb":4500,"Ag":-0.1,"As":2.4}}
{"sample_id":"WG-2","mineral":"sphalerite","method":"LA-ICPMS","elements":{"Cu":8.2,"Zn":640000,"Pb":3900,"Ag":0.8}}
{"sample_id":"DL-3","mineral":"galena","method":"EPMA","elements":{"Pb":850000,"S":135000,"Ag":120}}"""


def list_gt_papers():
    return sorted([p for p in GT_DIR.glob("*.xlsx") if not p.name.startswith("_")])


def find_pdf(paper_id: str):
    for d in [GEOCHEM_DATA, GEOCHEM_DATA / "papers"]:
        if not d.exists(): continue
        for ext in (".pdf", ".PDF"):
            f = d / f"{paper_id}{ext}"
            if f.exists(): return f
        matches = list(d.glob(f"{paper_id}*.pdf"))
        if matches: return matches[0]
        stem = paper_id.split("_et_al")[0]
        matches = list(d.glob(f"*{stem}*.pdf"))
        if matches: return matches[0]
    return None


def chunked(text: str, size: int = 6000, overlap: int = 400):
    out = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i+size])
        i += size - overlap
    return out


def parse_jsonl(raw: str):
    rows = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"): continue
        try:
            obj = json.loads(line)
            if "sample_id" in obj:
                rows.append(obj)
        except Exception:
            continue
    return rows


def run_one_paper(paper_id: str, model: str, client) -> dict:
    pdf = find_pdf(paper_id)
    if not pdf:
        return {"paper": paper_id, "model": model, "error": "no_pdf"}
    try:
        content = extract_pdf(str(pdf))
        text = get_paper_text_for_llm(content)
    except Exception as e:
        return {"paper": paper_id, "model": model, "error": f"pdf_read: {e}"}

    rows = []
    seen = set()
    raw_previews = []
    chunks = chunked(text, size=6000, overlap=400)
    for ci, chunk in enumerate(chunks[:12]):  # cap at 12 chunks (~70k chars)
        user = f"PAPER CHUNK {ci+1}/{min(len(chunks),12)}:\n{chunk}\n\nExtract analyses from THIS chunk only as JSON lines:"
        try:
            raw = client.complete(PROMPT, user, max_tokens=16384)
        except Exception as e:
            raw_previews.append(f"[chunk {ci+1} err: {e}]")
            continue
        if ci == 0:
            raw_previews.append(raw[:300])
        new_rows = parse_jsonl(raw)
        for r in new_rows:
            key = (str(r.get("sample_id","")), str(r.get("mineral","")))
            if key in seen: continue
            seen.add(key)
            rows.append(r)

    return {"paper": paper_id, "model": model, "n_rows": len(rows),
            "rows": rows, "n_chunks": min(len(chunks), 12),
            "raw_preview": " | ".join(raw_previews)[:600]}


def main(model_filter=None):
    papers = list_gt_papers()
    models = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
        "gpt-4o",
        "gemini-2.5-flash",
        "qwen25-7b",
        "mistral-7b-v02",
        "llama31-8b",
    ]
    if model_filter:
        models = [m for m in models if model_filter in m]
    for model in models:
        out_dir = RESULTS_DIR / f"geochem_fewshot_pdf_{model.replace('/', '_')}_v20"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model} → {out_dir.name} ===", flush=True)
        try:
            client = make_client(model)
        except Exception as e:
            print(f"  SKIP {model}: client init failed ({e})")
            continue
        t0 = time.time()
        for i, gt in enumerate(papers, 1):
            paper_id = gt.stem
            f_out = out_dir / f"{paper_id}.json"
            if f_out.exists():
                continue
            t1 = time.time()
            res = run_one_paper(paper_id, model, client)
            res["elapsed_seconds"] = round(time.time() - t1, 1)
            json.dump(res, open(f_out, "w"), indent=2, default=str)
            print(f"  [{i}/{len(papers)}] {paper_id}: {res.get('n_rows','?')} rows "
                  f"({res['elapsed_seconds']:.0f}s)"
                  + (f"  ERR={res['error']}" if "error" in res else ""), flush=True)
        print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else None
    main(flt)
