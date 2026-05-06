# Historical wrapper scripts

These shell scripts are pre-v23 experiment-orchestration wrappers, kept for the
audit trail. They have been **superseded** by:

- `src/ablation_scripts/_v28_chemtables_haiku_table3.sh` — Table 4 ChemTables backend ablation
- `src/ablation_scripts/_v29_*.sh` — Tables 3 and 4 component / backend ablations on Haiku 4.5
- `src/runners/_v23_geochem_abl_haiku28.py` — GeoChem n=28 ablation runner

The current paper's tables are produced by the v23/v28/v29 scripts. The v5--v18
wrappers represent earlier iterations (oss-LLM debugging, dedup variants,
different backend mixes) that informed pipeline-design choices but are not
needed to reproduce the published numbers.
