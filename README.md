# ArticleMiner — ISWC 2026 Submission Repository (anonymized)

This repository accompanies the ISWC 2026 Research Track submission
**"ArticleMiner: Ontology-Guided Knowledge Graph Construction from
Scientific Publications."**
It contains the system source, the new **GeoChem-28** benchmark
(PDFs + curated 209-column ground truth), and all per-paper
predictions backing the tables and figures in the paper.

```
articleminer-iswc26/
├── src/            # pipeline, ontology modules, prompts, evaluator, stats, figure regen
├── data/           # GeoChem-28: PDFs + 209-column ground-truth xlsx + annotation protocol
├── results/        # per-paper predictions for 5 closed + 3 open LLMs (ArticleMiner + few-shot baselines)
└── figures/
```

## 1. What's here

| Path | Contents |
|---|---|
| `paper/figures/` | All figures (cross_domain, cross_llm_invariance, ontology_size_vs_f1, precision_recall, shot_scaling, multi_llm) plus the drawio source for the architecture diagram. |
| `src/articleminer/` | Pipeline: 5-backend extraction, consensus, vision fallback, self-correction, deposit classifier, validator, KG builder. |
| `src/ontology/` | 4 domain modules: `ontology_geochemistry.py` (220 entries: 90 deposit types, 80 minerals, 50 method synonyms), `ontology_chemtables.py` (80), `ontology_discomat.py` (167), `ontology_mltables.py` (60). |
| `src/prompts/` | Per-domain prompt templates (paper-intelligence, table mapping, self-correction). |
| `src/eval/` | 4-tier evaluator, deterministic supplementary parser, tuple-level F1, error taxonomy. |
| `src/stats/` | Bootstrap CIs, paired significance (seed 42, 2k resamples), cross-LLM Jaccard agreement, table generation. |
| `src/figures/` | Figure regeneration scripts. |
| `data/geochem28/pdfs/` | 26 paper PDFs (two reuse entries derive their gold from a companion paper in the same corpus; see `annotation_protocol.md`). |
| `data/geochem28/ground_truth/` | 28 curated 209-column `{paper_id}.xlsx` files. |
| `data/geochem28/annotation_protocol.md` | USGS CMiO-MIN protocol, BDL convention (-99999 sentinel + negative LOD), per-row procedure, license. |
| `results/geochem28/{sonnet,haiku,opus,gpt-4o,gemini}/` | 5 closed-LLM × 28 = 140 per-paper extractions + 4-tier reports. |
| `results/geochem28/{llama31-8b,mistral-7b,qwen25-7b}/` | 3 open-LLM (7--8B) per-paper extractions + per-run summary. |
| `results/{chemtables,mltables,discomat}/articleminer_{sonnet,opus,haiku,gpt-4o,gemini}/` | ArticleMiner per-paper predictions, 5 closed LLMs. |
| `results/{chemtables,mltables,discomat}/articleminer_{llama31-8b,mistral-7b,qwen25-7b}/` | ArticleMiner per-paper predictions, 3 open LLMs. |
| `results/{chemtables,mltables,discomat}/fewshot_baseline_*/` | Matched same-LLM PDF few-shot baselines (Sonnet 4.6 + open backbones). |
| `results/*/v29_t3_*/` | Backend ablation runs (single-backend + leave-one-out × 5 backends + full), Haiku 4.5. |
| `results/*/abl_no_*/` | Component ablation runs (no_ontology, no_self_correct, no_validation, no_intelligence, no_vision), Haiku 4.5. |

## 2. Anonymization

All file paths, usernames, email addresses and API-key defaults have been
stripped. No authorship metadata remains. The only remaining secret-shaped
strings are `api_key=` parameter names in `llm_clients.py`; values default
to reading from environment variables (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`).

## 3. Setup

**Python 3.10+.** Install the dependencies used by the pipeline:

```bash
pip install anthropic openai google-genai \
            docling marker-pdf mineru pdfplumber camelot-py[cv] \
            openpyxl pandas numpy scipy matplotlib seaborn \
            tqdm pyyaml python-dotenv rapidfuzz vllm
```

Drop an `.env` at the repo root (or export directly) with:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AIza...
```

**Hardware tested:** Linux + NVIDIA H100 80GB for open-weight LLM inference (vLLM 0.6.x);
API-only runs need no GPU.

**Pinned model IDs:** `claude-sonnet-4-6`, `claude-opus-4-6`,
`claude-haiku-4-5-20251001`, `gpt-4o-2024-08-06`, `gemini-2.5-flash`,
`Qwen/Qwen-2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`,
`meta-llama/Llama-3.1-8B-Instruct`.

## 4. Reproducing each paper table / figure

Every reproducible number comes from files already in `results/`.
No rerun of extraction is required to reproduce the tables; the
scripts read the predictions in place.

| Paper element | Command |
|---|---|
| **Table 1** (cross-domain raw-PDF P/R/F1, 4 domains × 8 LLMs × ArticleMiner vs. few-shot) | `python src/stats/gen_paper_tables.py --table main_pdf` |
| **Table 1b** (cross-domain pre-parsed P/R/F1, 3 domains) | `python src/stats/gen_paper_tables.py --table main_preparsed` |
| **Table 2** (component ablation on Haiku 4.5, 4 domains) | `python src/stats/gen_paper_tables.py --table ablation` |
| **Table 3** (PDF-backend ablation on Haiku 4.5, 4 domains, 11 variants each) | `python src/stats/gen_paper_tables.py --table backend` |
| **Table 4** (significance: paired ΔF1, bootstrap CIs) | `python src/stats/compute_significance.py --all` |
| **Table 5** (cross-LLM Jaccard agreement, 4 domains) | `python src/stats/cross_llm_agreement.py` |
| **Table 6** (runtime / cost per paper, 2 backbone tiers) | `python src/stats/summarize_runtime_cost.py` |
| **Table 7** (benchmark statistics) | `python src/stats/gen_paper_tables.py --table datasets` |
| **Table 8** (error taxonomy, 3 domains × 3 systems) | `python src/stats/error_taxonomy.py --all` |
| **GeoChem detail table** (4-tier across iterations) | `python src/stats/gen_paper_tables.py --table geochem_main` |
| **App. Table A1** (Sonnet sensitivity ablation, ChemTables/MLTables) | `python src/stats/gen_paper_tables.py --table sonnet_ablation` |
| **Fig. 1** (architecture) | drawio source: `paper/figures/ARTICLEMINER.xml` |
| **Fig. 2** (cross-domain PDF-path bar) | `python src/figures/regen_pdf_path_figures.py` |
| **Fig. 3** (ontology size vs. F1) | `python src/figures/regen_nips_figures.py --figure ontology_size` |
| **Fig. 4** (multi-LLM PDF-path) | `python src/figures/regen_nips_figures.py --figure multi_llm` |
| **Fig. 5** (cross-LLM invariance on GeoChem) | `python src/figures/regen_nips_figures.py --figure cross_llm_invariance` |

## 5. Re-running extraction end-to-end

To regenerate predictions from scratch (requires PDFs + API keys + ~4h
wall time on Sonnet 4.6 for the full cross-domain sweep, or ~1.5h on Haiku 4.5):

```bash
# GeoChem-28 (one LLM)
python -m articleminer.main \
    --domain geochemistry \
    --pdfs data/geochem28/pdfs \
    --out results/geochem28/haiku \
    --llm claude-haiku-4-5-20251001

# ChemTables / MLTables / DiSCoMaT
python src/eval/eval_pdf_path.py \
    --benchmark chemtables \
    --llm claude-haiku-4-5-20251001 \
    --mode pipeline
```

Deterministic runs: `temperature=0`, seed 42, 4096 input / 8192 output
token budget by default (caps documented per stage in
`src/articleminer/config.yaml`).

## 6. Reproduced Tables

All tables from the paper, in markdown form, for quick reference.

### Table 1 — Cross-domain raw-PDF input (P / R / F₁ in %)

ArticleMiner vs. same-LLM PDF few-shot baseline. **Bold = per-domain best (Ours rows).**

| LLM | System | Chem P | Chem R | Chem F₁ | ML P | ML R | ML F₁ | DiSC P | DiSC R | DiSC F₁ | Geo P | Geo R | Geo F₁ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sonnet 4.6 | Ours | 30.4 | 34.0 | **32.1** | 52.6 | 51.7 | 52.2 | 98.2 | 66.4 | 79.3 | 58.0 | 82.9 | 60.8 |
| Sonnet 4.6 | Few-shot | 41.7 | 9.7 | 15.8 | 54.8 | 35.6 | 43.2 | 82.7 | 52.9 | 64.5 | 3.0 | 0.5 | 0.8 |
| Opus 4.6 | Ours | 30.6 | 33.1 | 31.8 | 57.0 | 54.9 | **55.9** | 97.0 | 69.5 | **81.0** | 58.1 | 83.5 | 60.6 |
| Opus 4.6 | Few-shot | 43.7 | 9.7 | 15.9 | 54.1 | 33.8 | 41.6 | 80.4 | 47.2 | 59.5 | 8.0 | 2.1 | 3.0 |
| Haiku 4.5 | Ours | 19.3 | 39.4 | 25.9 | 45.7 | 59.4 | 51.6 | 97.0 | 57.8 | 72.4 | 50.5 | 83.2 | 54.3 |
| Haiku 4.5 | Few-shot | 42.1 | 9.7 | 15.8 | 52.3 | 29.9 | 38.0 | 73.7 | 44.9 | 55.8 | 14.0 | 3.1 | 4.1 |
| GPT-4o | Ours | 28.2 | 32.5 | 30.2 | 55.0 | 38.9 | 45.6 | 87.3 | 47.9 | 61.9 | 62.2 | 82.9 | **63.1** |
| GPT-4o | Few-shot | 62.0 | 9.5 | 16.5 | 41.8 | 27.0 | 32.8 | 63.0 | 44.1 | 51.9 | 7.7 | 2.8 | 3.6 |
| Gemini 2.5 Flash | Ours | 16.5 | 35.9 | 22.6 | 44.7 | 40.8 | 42.7 | 92.9 | 64.1 | 75.9 | 54.9 | 83.1 | 57.6 |
| Gemini 2.5 Flash | Few-shot | 24.0 | 10.2 | 14.3 | 44.5 | 40.5 | 42.4 | 62.5 | 44.7 | 52.1 | 3.2 | 0.2 | 0.3 |
| Qwen-2.5-7B (open) | Ours | 11.9 | 13.2 | 12.5 | 49.2 | 51.3 | 50.2 | 88.3 | 43.0 | 57.8 | 55.0 | 74.8 | 56.2 |
| Qwen-2.5-7B (open) | Few-shot | 37.7 | 5.0 | 8.8 | 38.7 | 13.5 | 20.0 | 33.8 | 22.5 | 27.0 | 3.1 | 0.2 | 0.3 |
| Mistral-7B (open) | Ours | 22.3 | 17.7 | 19.8 | 42.8 | 21.5 | 28.6 | 81.1 | 38.2 | 51.9 | 55.0 | 74.8 | 56.2 |
| Mistral-7B (open) | Few-shot | 6.7 | 6.5 | 6.6 | 36.9 | 12.7 | 18.9 | 11.5 | 2.8 | 4.5 | 0.0 | 0.0 | 0.0 |
| Llama-3.1-8B (open) | Ours | 18.6 | 14.9 | 16.6 | 51.9 | 50.4 | 51.1 | 79.0 | 10.3 | 18.2 | 55.0 | 74.8 | 56.2 |
| Llama-3.1-8B (open) | Few-shot | 15.4 | 8.7 | 11.1 | 31.6 | 18.3 | 23.2 | 15.4 | 26.0 | 19.4 | 4.6 | 1.9 | 2.6 |

### Table 2 — Cross-domain pre-parsed input (P / R / F₁ in %)

Pre-parsed table text from each benchmark's release. GeoChem is omitted (per-paper sample-row aggregation has no per-table pre-parsed analog). **Bold = per-domain best.**

| LLM | System | Chem P | Chem R | Chem F₁ | ML P | ML R | ML F₁ | DiSC P | DiSC R | DiSC F₁ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sonnet 4.6 | Ours | 57.8 | 60.2 | 59.0 | 70.9 | 94.9 | 81.2 | 91.5 | 91.1 | 91.3 |
| Sonnet 4.6 | Few-shot | 48.3 | 39.2 | 43.2 | 68.8 | 92.1 | 78.8 | 92.1 | 94.7 | 93.4 |
| Opus 4.6 | Ours | 59.9 | 61.5 | 60.7 | 72.1 | 95.4 | 82.1 | 95.1 | 94.2 | **94.7** |
| Opus 4.6 | Few-shot | 50.9 | 50.9 | 50.9 | 75.6 | 75.9 | 75.8 | 91.2 | 85.5 | 88.3 |
| Haiku 4.5 | Ours | 55.0 | 57.1 | 56.1 | 70.8 | 92.8 | 80.3 | 92.8 | 91.2 | 92.0 |
| Haiku 4.5 | Few-shot | 48.4 | 45.9 | 47.1 | 79.9 | 75.4 | 77.6 | 92.4 | 90.1 | 91.2 |
| GPT-4o | Ours | 55.0 | 52.4 | 53.7 | 78.2 | 92.4 | **84.7** | 92.4 | 88.6 | 90.4 |
| GPT-4o | Few-shot | 45.4 | 43.1 | 44.2 | 80.9 | 91.5 | 85.9 | 88.2 | 90.8 | 89.5 |
| Gemini 2.5 Flash | Ours | 61.3 | 61.3 | 61.3 | 63.9 | 92.2 | 75.4 | 85.4 | 88.7 | 87.0 |
| Gemini 2.5 Flash | Few-shot | 47.6 | 44.8 | 46.2 | 75.5 | 99.0 | 85.7 | 89.3 | 91.8 | 90.5 |
| Qwen-2.5-7B (open) | Ours | 79.0 | 53.9 | **64.1** | 75.3 | 79.3 | 77.3 | 80.8 | 78.5 | 79.6 |
| Qwen-2.5-7B (open) | Few-shot | 44.7 | 28.6 | 34.9 | 76.4 | 59.7 | 67.0 | 73.4 | 13.6 | 23.0 |
| Mistral-7B (open) | Ours | 16.9 | 19.0 | 17.9 | 58.5 | 63.7 | 61.0 | 60.2 | 54.1 | 57.0 |
| Mistral-7B (open) | Few-shot | 36.0 | 20.8 | 26.3 | 53.6 | 37.3 | 44.0 | 0.0 | 0.0 | 0.0 |
| Llama-3.1-8B (open) | Ours | 49.9 | 54.3 | 52.0 | 59.9 | 80.2 | 68.6 | 69.3 | 73.8 | 71.5 |
| Llama-3.1-8B (open) | Few-shot | 45.3 | 30.5 | 36.5 | 69.5 | 59.9 | 64.3 | 60.0 | 26.0 | 36.3 |
| **Published baseline** | DiSCoMaT GNN | — | — | — | — | — | — | — | — | **70.04** |

### Table 3 — Component ablation on Haiku 4.5 (P / R / F₁ in %)

Per-domain effect of disabling each pipeline stage. **Bold = largest F₁ drop per domain; ↑ = ablation improves over the full pipeline.**

| Variant | Chem P | Chem R | Chem F₁ | ML P | ML R | ML F₁ | DiSC P | DiSC R | DiSC F₁ | Geo P | Geo R | Geo F₁ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full pipeline** | 19.3 | 39.4 | 25.9 | 45.7 | 59.4 | 51.6 | 97.0 | 57.8 | 72.4 | 50.5 | 83.2 | 54.3 |
| w/o ontology | 18.0 | 34.6 | 23.7 | 42.2 | 45.8 | 43.9 | 88.3 | 41.1 | **56.1** | 49.2 | 83.3 | 53.6 |
| w/o self-correction | 16.0 | 41.8 | 23.1 | 39.5 | 67.2 | 49.8 | 91.0 | 53.2 | 67.1 | 49.2 | 83.3 | 53.6 |
| w/o validation | 15.8 | 34.4 | **21.7** | 36.5 | 47.6 | **41.3** | 90.8 | 58.5 | 71.2 | 49.2 | 83.3 | 53.6 |
| w/o intelligence | 21.8 | 25.1 | 23.4 | 48.4 | 44.3 | 46.3 | 94.8 | 64.1 | 76.5 ↑ | 49.2 | 83.3 | 53.6 |
| w/o vision | 17.1 | 32.3 | 22.3 | 39.5 | 62.6 | 48.4 | 90.1 | 59.4 | 71.6 | 51.6 | 83.3 | 55.4 ↑ |

Sonnet 4.6 sensitivity for the small ChemTables / MLTables benchmarks is in the supplementary appendix.

### Table 4 — PDF backend ablation on Haiku 4.5 (P / R / F₁ in %)

Single-backend (one parser only), leave-one-out (full minus one), and full 5-backend pipeline. **Bold = per-domain best F₁.**

| Configuration | Chem P | Chem R | Chem F₁ | ML P | ML R | ML F₁ | DiSC P | DiSC R | DiSC F₁ | Geo P | Geo R | Geo F₁ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| docling | 23.4 | 29.9 | **26.2** | 56.4 | 56.2 | **56.3** | 96.3 | 57.1 | **71.7** | 43.9 | 69.3 | 47.9 |
| marker | 22.7 | 23.8 | 23.2 | 48.4 | 54.9 | 51.4 | 95.0 | 50.8 | 66.2 | 44.0 | 69.4 | **48.0** |
| mineru | 22.7 | 23.8 | 23.2 | 48.6 | 54.9 | 51.6 | 95.2 | 51.1 | 66.5 | 41.7 | 69.7 | 46.3 |
| pdfplumber | 22.2 | 24.7 | 23.4 | 52.2 | 49.8 | 51.0 | 95.2 | 50.9 | 66.3 | 41.6 | 69.6 | 46.3 |
| camelot | 20.5 | 33.8 | 25.5 | 42.6 | 43.8 | 43.2 | 92.8 | 28.6 | 43.7 | 44.1 | 69.4 | 48.0 |
| w/o docling | 18.5 | 34.8 | 24.2 | 42.5 | 41.3 | 41.9 | 92.4 | 28.5 | 43.5 | 41.3 | 69.4 | 46.0 |
| w/o marker | 13.8 | 36.8 | 20.1 | 40.9 | 47.7 | 44.1 | 90.0 | 59.9 | 72.0 | 43.1 | 69.4 | 47.3 |
| w/o mineru | 14.2 | 36.8 | 20.5 | 45.0 | 45.5 | 45.3 | 90.6 | 59.1 | 71.6 | 44.0 | 69.4 | 48.0 |
| w/o pdfplumber | 15.3 | 36.8 | 21.6 | 35.2 | 53.4 | 42.4 | 91.1 | 60.5 | 72.7 | 41.4 | 69.4 | 46.0 |
| w/o camelot | 18.9 | 36.1 | 24.8 | 46.8 | 67.6 | 55.3 | 97.4 | 57.7 | **72.5** | 43.9 | 69.3 | 47.9 |
| **Full** (5 backends) | 14.5 | 36.8 | 20.8 | 35.6 | 71.8 | 47.6 | 90.2 | 58.6 | 71.0 | 50.5 | 83.2 | **54.3** |

### Table 5 — Statistical significance: ArticleMiner vs. same-LLM PDF few-shot

Sonnet 4.6, paired bootstrap on per-paper F₁ (seed 42, 2k resamples). Bold = winning median.

| Domain (n) | Ours mean [95% CI] | Baseline mean [95% CI] | Ours median | Base. median | Δ F₁ | p |
|---|---|---|---:|---:|---:|---:|
| ChemTables (n = 9) | 36.4 [10.8, 66.6] | 27.6 [5.9, 54.8] | 6.3 | 2.5 | +8.8 | 0.342 |
| MLTables (n = 15) | 52.2 [36.2, 68.1] | 37.8 [25.5, 51.4] | 55.8 | 35.6 | +14.4 | 0.074 ⁎ |
| DiSCoMaT (n = 111) | 66.5 [58.1, 73.8] | 51.1 [43.3, 59.2] | **96.7** | 60.0 | +15.5 | <0.001 ⁂ |
| GeoChem (n = 25) | 63.2 [49.9, 76.1] | 0.8 [0.3, 1.6] | 65.7 | 0.0 | +62.3 | <0.001 ⁂ |

### Table 6 — Cross-LLM agreement (Jaccard %)

Pairwise Jaccard of predicted tuples between backbones, plus the three-way intersection (Sonnet ∩ Opus ∩ GPT-4o).

| LLM pair | ChemTables | MLTables | DiSCoMaT | GeoChem |
|---|---:|---:|---:|---:|
| Sonnet vs. Opus | 99.4 | 84.9 | 79.2 | 89.6 |
| Sonnet vs. GPT-4o | 71.7 | 57.2 | 74.1 | 79.0 |
| Opus vs. GPT-4o | 71.9 | 60.1 | 63.8 | 81.5 |
| **3-way intersection** | 57.5 | 52.8 | 51.6 | 75.3 |

### Table 7 — Per-benchmark statistics

| Dataset | PDFs | Tables | Gold tuples |
|---|---:|---:|---:|
| GeoChem (ours) | 28 | ~80 | 5,307 |
| ChemTables | 9 | 14 | 462 |
| MLTables | 15 | 68 | 2,060 |
| DiSCoMaT | 111 | 175 | 4,755 |

MLTables excludes "Other"-typed cells per the published convention.

### Table 8 — Cost / runtime per paper (mean)

Same ArticleMiner pipeline at two backbone tiers (Sonnet 4.6 — \$3/M input, \$15/M output; Haiku 4.5 — \$1/M input, \$5/M output) plus the same-LLM PDF few-shot baseline (Sonnet 4.6).

| Domain | n | AM-Sonnet (s/paper) | AM-Sonnet (\$/paper) | AM-Haiku (s/paper) | AM-Haiku (\$/paper) | FS-Sonnet (s/paper) | FS-Sonnet (\$/paper) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ChemTables | 9 | 60 | \$0.10 | 86 | \$0.03 | 17 | \$0.05 |
| MLTables | 15 | 120 | \$0.10 | 118 | \$0.03 | 62 | \$0.05 |
| DiSCoMaT | 111 | 140 | \$0.10 | 40 | \$0.03 | 25 | \$0.05 |
| GeoChem | 28 | 180 | \$0.30 | 120 | \$0.10 | 21 | \$0.05 |

### Table 9 — Error distribution (% of total error events)

Error categories: **over-extraction** (predicted tuple absent from gold, FP), **omission** (gold tuple missed, FN), **attribute mismatch** (value matches but type/unit/target wrong), **near miss** (numeric value within 5% of gold but formatting differs). N = absolute total error count for that row. GeoChem errors are characterized by the four-tier breakdown ($T_1$/$T_2$/$T_3$/$T_4$) instead, since the sample-row schema does not map to cell-tuple categories.

| Domain | System | Over-extraction | Omission | Attr. mismatch | Near miss | N |
|---|---|---:|---:|---:|---:|---:|
| ChemTables | Ours-Sonnet | 57.0 | 16.1 | 26.6 | 0.4 | 704 |
| ChemTables | Ours-GPT4o | 5.2 | 62.7 | 31.2 | 0.9 | 346 |
| ChemTables | Few-shot-Sonnet | 10.5 | 70.9 | 18.0 | 0.7 | 440 |
| MLTables | Ours-Sonnet | 59.9 | 30.7 | 3.9 | 5.5 | 2,163 |
| MLTables | Ours-GPT4o | 27.1 | 59.3 | 0.1 | 13.5 | 2,020 |
| MLTables | Few-shot-Sonnet | 27.6 | 49.8 | 5.8 | 16.7 | 1,679 |
| DiSCoMaT | Ours-Sonnet | 2.7 | 92.1 | 5.2 | 0.0 | 1,721 |
| DiSCoMaT | Ours-GPT4o | 5.8 | 82.7 | 10.0 | 1.4 | 2,788 |
| DiSCoMaT | Few-shot-Sonnet | 12.1 | 75.5 | 6.5 | 2.5 | 2,697 |


## 7. Licences

- **Code** (`src/`): MIT — see [`LICENSE-CODE`](LICENSE-CODE).
- **Data** (`data/`, `results/`): CC-BY-4.0 — see [`LICENSE-DATA`](LICENSE-DATA).
