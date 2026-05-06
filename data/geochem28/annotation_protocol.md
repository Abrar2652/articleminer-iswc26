# GeoScholar-28 Annotation Protocol

## Scope

28 peer-reviewed papers reporting LA-ICP-MS trace-element analyses of sulphide
and sulphosalt minerals in ore deposits, published 2004--2025. Each paper is
annotated against the 210-column **CMiO-MIN** schema (an extension of CMiO/
Hofstra 2021 for individual-grain mineral geochemistry).

## Annotators

Multiple geochemistry annotators participated, coordinated with the U.S.
Geological Survey (USGS). Annotations were independently spot-checked by a
second annotator and any disagreements reconciled against the source PDF +
supplementary tables.

## Per-paper procedure

1. **Deposit metadata (T1).** Extract the 15 deposit/analytical-method fields
   (deposit_name, deposit_type, deposit_environment, deposit_group,
   all_commodities, mineral, analytical_method, instrument_type_model,
   laboratory_location, operating_conditions, standards_used, country,
   age, host_rock, sample_reference). Values are taken verbatim from the
   paper text when possible, normalised to the controlled vocabulary in
   the ontology module otherwise.

2. **Per-sample rows.** Each row corresponds to one (sample_id, mineral)
   analysis (three-tier sample ID: top-level sample, sub-sample/thin section,
   spot). For each element column (element_ppm / element_wtpct), record:
   - A numeric value if reported.
   - `-99999` (five nines) if explicitly reported as below detection limit.
   - Blank if not measured / not reported.
3. **Mineral assignment.** One mineral per row. Mineral identity is taken
   from the per-analysis annotation in the paper's supp tables
   (`analysis_id` prefix, data sheet name, or explicit column), never
   inferred from abundance patterns.
4. **Units.** Preserved as reported; no conversion. Columns carry the unit
   in their name (e.g. `cu_ppm`, `s_wtpct`).

## Conventions

- **Below detection limit:** `-99999` (five nines, negative). Distinguishes
  BDL from "not measured" (blank).
- **Not applicable / not measured:** blank cell.
- **Reference materials (MASS-1, NIST 610, etc.):** included as sample rows
  tagged with the reference-material flag; do not filter at annotation time.
- **Data-reuse papers** (suffix `_as_reported_in_X_et_al_YYYY`): gold tuples
  are taken from paper X's supplementary tables. The original paper PDF is
  not required because all numbers reappear in the companion paper.

## Licence

Ground-truth annotations are released under CC-BY-4.0. Source-paper PDFs
remain under the publishers' original licences and are included here under
fair-use for academic reproducibility of the benchmark.
