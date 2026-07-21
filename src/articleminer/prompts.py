"""
prompts.py - LLM prompt templates for multi-stage geochemical data extraction.

Design principle: two-stage prompting
  Stage 1 → extract paper-level metadata from PDF text (applies to all rows)
  Stage 2 → verify / supplement table row data with LLM assistance

This module also builds picklist-constraint clauses from `picklists.yaml`
(loaded once at import time) so the LLM is forced to pick controlled-vocabulary
values from the USGS curation app rather than invent labels.
"""

from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

from .schema import PaperMetadata
from .knowledge_base import get_knowledge_base_prompt

# ──────────────────────────────────────────────────────────────────────────────
# Picklist constraints (Layer 1 — built from Picklist.xlsx via
# scripts_build_picklists.py).
# ──────────────────────────────────────────────────────────────────────────────
# picklists.yaml lives at the package root, alongside this module.
_PICKLIST_PATH = Path(__file__).resolve().parent / "picklists.yaml"
try:
    import yaml as _yaml
    PICKLISTS: dict[str, list[str]] = _yaml.safe_load(_PICKLIST_PATH.read_text())
except Exception:
    PICKLISTS = {}


def picklist_clause(field: str, *, max_inline: int = 60) -> str:
    """Return a 'MUST be one of …' clause for `field`, ready to append to a
    field-description string in the JSON-schema example block.

    For picklists with ≤ max_inline values, all values are listed inline.
    For larger picklists (mineral=138, deposit_type=189, etc.) the full list
    is provided in a separate `PICKLIST VALUES` section appended below the
    JSON example block; this clause then just points at the field name.
    """
    values = PICKLISTS.get(field) or []
    if not values:
        return ""
    if len(values) <= max_inline:
        v = "  |  ".join(f"'{x}'" for x in values)
        return f" PICKLIST: must be EXACTLY one of [{v}]. Return null if none fit. Do NOT invent a label."
    return (f" PICKLIST: must be EXACTLY one value listed under '{field}' in the "
            f"PICKLIST VALUES section below. Return null if none fit. Do NOT invent.")


def large_picklist_section(fields: list[str], *, threshold: int = 60) -> str:
    """Return a `## PICKLIST VALUES` markdown block listing every picklist
    that exceeds `threshold` values (mineral, country, etc.)."""
    blocks: list[str] = []
    for f in fields:
        vals = PICKLISTS.get(f) or []
        if len(vals) <= threshold:
            continue
        lines = [f"### {f} ({len(vals)} allowed values)"]
        # Format compactly: 4 per line for readability
        for i in range(0, len(vals), 4):
            row = vals[i:i+4]
            lines.append(", ".join(f"'{v}'" for v in row))
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return "## PICKLIST VALUES (use EXACTLY these strings)\n\n" + "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 0: Paper Intelligence Blueprint (pre-extraction analysis)
# ──────────────────────────────────────────────────────────────────────────────

PAPER_INTELLIGENCE_SYSTEM_PROMPT = dedent("""\
You are an expert analytical geochemist. Your task is to carefully read a research paper
and extract specific analytical methodology details. Focus on the Methods, Analytical,
and Results sections.

Be precise — copy instrument descriptions, lab names, standards, and conditions VERBATIM
from the paper. Do not paraphrase or fabricate any details.
""")

PAPER_INTELLIGENCE_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT
{paper_text}

## YOUR TASK
Analyze this geochemistry paper and extract analytical intelligence.
Return a JSON object with EXACTLY these fields:

{{
  "elements_measured": ["fe", "cu", "zn"],
  "expected_sample_count": 50,
  "minerals_analyzed": ["pyrite", "sphalerite"],
  "analytical_methods": ["LA-ICPMS"],
  "instrument": "exact instrument description from paper or null",
  "laboratory": "exact lab name and location from paper or null",
  "standards_used": "all standards mentioned, verbatim, or null",
  "operating_conditions": "all conditions mentioned, verbatim, or null"
}}

## FIELD INSTRUCTIONS
1. **elements_measured**: List ALL element symbols (lowercase) that have actual analytical
   data in the paper's tables. Check table headers and results text carefully.
   Valid symbols: fe, cu, zn, pb, ag, au, as, sb, bi, co, ni, mn, cd, in, ga, ge, se, te,
   tl, sn, mo, w, v, cr, ti, sc, ba, sr, rb, cs, li, be, b, p, s, cl, f, br, hg, re,
   si, al, ca, mg, na, k, la, ce, pr, nd, sm, eu, gd, tb, dy, ho, er, tm, yb, lu, y,
   zr, hf, nb, ta, th, u

2. **expected_sample_count**: Approximate number of individual spot analyses/measurements
   in the paper's data tables. Count analytical spots, not hand samples.

3. **minerals_analyzed**: Mineral names (lowercase) that were analyzed (e.g., pyrite,
   sphalerite, chalcopyrite). Use null if whole-rock analysis.

4. **analytical_methods**: Standardized method names used. Common: LA-ICPMS, EPMA, ICP-MS,
   XRF, SEM-EDS, SIMS.

5. **instrument**: Copy the EXACT instrument description from the paper. Include manufacturer,
   model, and attached components (laser system, etc.). Use null if not stated.

6. **laboratory**: Copy the EXACT laboratory name and location. Use null if not stated.

7. **standards_used**: Copy ALL reference/calibration standards mentioned — internal standards,
   external standards, quality control materials. Verbatim. Use null if not stated.

8. **operating_conditions**: Copy ALL analytical conditions — spot size, beam diameter,
   accelerating voltage, beam current, laser frequency, energy density, carrier gas, dwell
   time, repetition rate, etc. Verbatim. Use null if not stated.

## CRITICAL
- For fields 5-8, copy text VERBATIM — do not summarize or paraphrase.
- If multiple methods were used, separate details with " | " (pipe character).
- For elements_measured, check ACTUAL data table headers, not just the methods description.

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Paper-level metadata extraction
# ──────────────────────────────────────────────────────────────────────────────

METADATA_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator with deep knowledge of economic geology,
analytical geochemistry, and mineralogy. Your task is to extract structured metadata from
research papers for a standardized geochemical database.

Extract information EXACTLY as stated in the paper — do not paraphrase or invent values.
Return ONLY valid JSON. If a field cannot be determined from the paper, use null.

{knowledge_base}
""")

METADATA_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT
{paper_text}

{paper_map_block}

## SUPPLEMENTARY TABLE PREVIEW
The supplementary table (first few rows) looks like:
{table_preview}

## YOUR TASK
Extract paper-level metadata from this research paper. This metadata generally applies to
all samples, but note that some fields (mineral, analytical_method, deposit_name) MAY vary
per sample — if so, extract the MOST COMMON or PRIMARY value here; per-row overrides will
be handled separately from the supplementary table columns.

### START WITH THE TITLE
Most geochem papers state the studied deposit, the commodities, and a country/region in
the title itself. Examples:
  - "...Ayawilca Zn-Pb-Ag-In-Sn-Cu deposit, Pasco, Peru"  → deposit "Ayawilca", commodities Zn,Pb,Ag,In,Sn,Cu, location "Pasco, Peru", country PER
  - "Hilton Zn-Pb (Ag) Deposit, Australia"                → deposit "Hilton", commodities Zn,Pb,Ag, country AUS
  - "Bahçecik Au±Ag mineralization in the Eastern Pontides, Gümüşhane-NE Türkiye"
                                                          → deposit "Bahçecik", commodities Au,Ag, location "Eastern Pontides, Gümüşhane-NE Türkiye", country TUR
Parse the title FIRST for deposit + commodities + location, then refine from the abstract,
intro, and Fig. 1 caption. Bind deposit ↔ commodities ↔ location together: this triple is
used to query the Mindat database for coordinates downstream, and the same name (e.g.
"Victoria") points to many different mines unless commodity + country disambiguates it.

### STUDIED HERE vs CITED ONLY
Many papers reference OTHER deposits for comparison without analyzing samples from them.
ONLY extract deposits where the paper actually performed analyses on samples. If the paper
just compares its results to "the Hilton Zn-Pb deposit (Cave B, Lilly R, Hong W 2020)",
the Hilton deposit is CITED, not studied — do NOT put it as the paper's deposit_name.

### NEVER MISTAKE A CITATION FOR A DEPOSIT NAME
A deposit name is a place. A bibliographic citation is "Author Year". Strings like
"Cave B, Lilly R, Hong W (2020) The effect of co-crystallising sulphides..." are NOT
deposit names — they are references.

Return a JSON object with EXACTLY these fields (use null if not found in the paper):

```json
{{
  "deposit_name": "CONCISE deposit name only — no commodity metals, no 'deposit/mine' suffix. E.g. 'Bainiuchang' NOT 'Bainiuchang Zn-Sn polymetallic deposit'. If multiple, comma-separated.",
  "deposit_local_id": "Local deposit identifier code, if any",
  "deposit_environment": "MUST be one of: 'Basin hydrothermal', 'Seafloor hydrothermal', 'Magmatic hydrothermal', 'Magmatic', 'Epithermal', 'Metamorphic', 'Sediment-hosted', 'Supergene', 'Surficial', 'unknown'. Infer from deposit type.",
  "deposit_group": "Broad deposit class. MUST use standard form: 'Mississippi Valley-type (MVT)', 'Volcanic-hosted massive sulfide (VMS)', 'Sedimentary Exhalative (SEDEX)', 'Skarn', 'Porphyry', 'Epithermal', 'Orogenic gold', 'Iron oxide copper-gold (IOCG)', 'Carlin-type', 'Stratiform sediment-hosted Cu', 'Magmatic Ni-Cu-PGE', 'Greisen', 'Pegmatite', 'unknown'",
  "deposit_type": "Specific deposit type per Hofstra et al. 2021. E.g. 'MVT zinc-lead', 'Cu-Au porphyry', 'orogenic gold', 'stratiform sediment-hosted Cu-Co'. Use 'unknown' if not determinable.",
  "deposit_classification_source": "Classification scheme used. Always set to 'Hofstra et al. 2021'. If the paper uses a different scheme, note it here: 'Hofstra et al. 2021 (paper uses: <their scheme>)'.",
  "deposit_type_original": "Deposit type EXACTLY as stated by the paper's authors, verbatim. May differ from Hofstra 2021 (e.g., paper says 'SEDEX-type Zn-Pb' but Hofstra says 'Sedimentary Exhalative'). null if same as deposit_type.",
  "primary_commodities": "Commodities the deposit ACTUALLY PRODUCES (mined for sale). Usually 1-3 metals — typically the metals named in the paper title or the abstract's first sentence. Comma-separated. DO NOT list every trace element measured in the analyses; those go nowhere in this field. Example for a 'Maweishan Pb-Zn deposit' paper: 'Pb, Zn' (NOT 'Pb, Zn, Cu, Ag, Cd, Ga, Ge, In, Sb, Sn'). CRITICAL — classify the MINERALISATION BEING ANALYSED, not a larger host/overprinted system: when the title names both (e.g. 'Zoned Polymetallic (Pb-Zn-Cu-Ag-Au) Veins from the Bingham Canyon Porphyry Cu-Au-Mo Deposit'), the commodities are the analysed mineralisation's metals — the polymetallic-vein metals in the parenthetical (Ag, Au, Cu, Pb, Zn) — NOT the host porphyry's Mo. {primary_commodities_picklist},
  "secondary_commodities": "By-product commodities the deposit also produces (e.g. Ag as a by-product of Pb-Zn mining). Comma-separated, may be empty. Again — only actually-produced metals, NOT trace-enriched elements. {secondary_commodities_picklist},
  "all_commodities": "STRICT format: '<primary> (<secondary>)' with parentheses. E.g. 'Pb, Zn (Ag)' for a Pb-Zn deposit with Ag by-product. If no secondary commodities: just '<primary>' with NO parentheses, e.g. 'Pb, Zn'. NEVER list trace-enriched elements here that are not actual primary or secondary commodities. The string MUST mirror primary_commodities and secondary_commodities exactly (any element not in those two fields must NOT appear here).",
  "deposit_source": "Short author + year citation. E.g. 'Yuan et al., 2018'",

  "feature_type": "Sampling-feature category — what kind of place / structure the sample was taken from. Match the ACTUAL structure named in the paper, do not default: a surface mine working (open pit, pit wall, bench, quarry face) is 'mine'; an underground working (adit, drift, shaft, stope, tunnel) is 'underground mine'; a natural rock exposure is 'outcrop'; a trench is 'trench'; drill material is 'borehole'. Key distinction: a sample from an open pit or pit wall is a SURFACE working, so it is 'mine', never 'underground mine'. If the paper does not say which, leave it unknown rather than guessing underground. {feature_type_picklist},
  "feature_name": "Free text — the MOST SPECIFIC name or ID of the sampling feature. INTERPRET the paper: include every qualifier it attaches (orebody number, level/elevation, depth range, drill-hole ID, vein name), even when those qualifiers are split across sentences. Paraphrase as needed for clarity. Example correct: 'orebody no. 1 level 1884 to 2064 m', 'main vein, level 1944 m', 'DDH 879 at 1207 m'. Example WRONG (too vague): 'main orebody', 'No. I orebody at different depths' — those drop the depth/level qualifier that makes the feature uniquely identifiable. Look in the geological-setting / sample-description sections of the paper body, NOT the supp table. Evidence-anchored: provide a supporting paper snippet (paraphrase or close-quote, see instructions below).",
  "sample_deposit_relation": "What is the sample's relation to the ore body. Choose by what the SAMPLE physically IS, not by what mineral was analysed within it: an ore-bearing rock or vein specimen is 'ore material' even when individual mineral grains inside it were the analysis target; a physically separated single ore-mineral grain is 'ore mineral'; a concentrate is 'ore mineral concentrate'; barren wall/host rock is 'host rock'; altered rock is 'alteration material'; gangue is 'gangue'. Do not default to 'ore material' when the sample is actually host, altered, or gangue material. {sample_deposit_relation_picklist},
  "sample_type": "The physical form of the sample as collected. {sample_type_picklist},
  "sampling_method": "How the sample was physically obtained. {sampling_method_picklist},
  "material_class": "Broad material category. HARD RULES (these decide the answer, do not deliberate):\\n  (a) 'mineral separate' is a SAMPLE_TYPE value, NEVER a material_class. Do not write it in this field even if the paper uses the phrase.\\n  (b) When the analysed mineral grain SITS WITHIN a rock matrix (any in-situ analysis: LA-ICPMS spots on a polished thin section, EMPA spots on a mineral in a rock, SIMS spots on accessory phases in a host rock) → ALWAYS 'mineralised rock'. This is the default for nearly all mineral-chemistry papers in this corpus.\\n  (c) Use 'mineral' only when the paper analysed a HAND-PICKED, fully separated, monomineralic specimen in isolation from its host rock.\\n  (d) Use 'rock' for bulk-rock analyses (whole-rock chemistry).\\nIn short: in-situ on a thin section or mounted grain → 'mineralised rock'. Hand-picked monomineralic separate → 'mineral'. Bulk rock powder → 'rock'. {material_class_picklist},
  "earth_material_group": "Dominant lithological grouping. {earth_material_group_picklist},
  "earth_material": "Specific lithology name. {earth_material_picklist},
  "strat_unit_name": "Free text — stratigraphic unit hosting the deposit. E.g. 'Dengying Formation', 'Bonneterre Dolomite'. Look in the geological-setting section of the paper body, NOT the supp table. Evidence-anchored (paraphrase OK; supporting snippet must mention the unit).",
  "alteration": "Alteration styles spatially or temporally associated with the MINERALIZATION the paper studies, comma-separated. USE THE PAPER'S OWN WORDING (e.g. write '-ization' style names verbatim, do not condense 'carbonatization' to 'carbonate').\\nProcedure:\\n  Step 1: scan the paper for sentences that link alteration to mineralization with phrases like: 'placed closely to', 'spatially associated with', 'closely related to', 'contemporaneous with', 'coeval with', 'accompanying', 'related to ore formation', 'developed along the mineralization'.\\n  Step 2: extract EVERY alteration name in that sentence. When the sentence uses 'both X and Y', 'X and Y are', or 'X, Y, and Z are', the output MUST include EVERY name conjoined that way. Listing only the first is wrong.\\n  Step 3: when no linking sentence exists, fall back to the general list of wall-rock alterations the paper describes (still comma-separated).\\nGeneric worked example (template, not paper-specific):\\n  Paper text says: 'Wall-rock alterations include [A], [B], [C], [D], and [E], of which both [A] and [B] are placed closely to the mineralization.'\\n  Correct output: '[A], [B]' (the subset the paper links to mineralization, both names because BOTH/AND is used).\\n  Wrong output: '[A]' alone (dropped the second linked name) or '[A], [B], [C], [D], [E]' (returned the full general list instead of the linked subset).\\nReturn at least two styles whenever the paper uses BOTH/AND or X-and-Y phrasing. Map to USGS picklist where possible, keeping the paper's wording verbatim. Evidence-anchored. {alteration_picklist},
  "mineral": "Target mineral analyzed (IMA-canonical name). {mineral_picklist},
  "associated_minerals": "The full paragenetic-stage assemblage that contains the TARGET MINERAL, as the paper writes it, comma-separated. USGS curator convention: list EVERY mineral the paper names in that stage, including the target itself and ALL gangue (quartz, carbonate, dolomite, calcite, barite, fluorite, etc.).\\nProcedure:\\n  Step 1: identify how the paper describes paragenesis. Many ore-mineral papers split it into named stages (Stage 1 / Stage 2 / hypogene / supergene / hydrothermal / oxidation, etc.) and list members with hyphens or commas: 'Stage 2: A-B-T-C-D-E' where T is the target.\\n  Step 2: find the stage that contains the TARGET mineral and copy every named member of that stage into the answer, in paper order.\\n  Step 3: if the same paragenesis is also described in a paragenetic-sequence figure or a textural-relationships paragraph, cross-check the two and include any member named in either.\\nStrict rules:\\n  (a) INCLUDE the target mineral itself in the list. INCLUDE every gangue mineral in the stage. Never silently drop a gangue (quartz, carbonate, etc.) just because it is not an ore mineral.\\n  (b) DO NOT include minerals from EARLIER (pre-target) or LATER (post-target / supergene / oxidation overprint) stages. Typical supergene/oxidation minerals to exclude: limonite, cerussite, hemimorphite, smithsonite, anglesite, malachite, azurite, goethite, jarosite — unless the paper explicitly places them in the same hydrothermal stage as the target.\\n  (c) DO NOT include micro-inclusions TRAPPED INSIDE the target mineral grain (e.g. small sulfosalt blebs inside a sphalerite host, fluid inclusions, exsolution lamellae). These are textural inclusions, not stage minerals.\\n  (d) Resolve a generic 'carbonate' to the specific host carbonate: 'dolomite' when the host rock is dolomitic, 'calcite' when calcitic, otherwise keep 'carbonate'.\\n  (e) Match the USGS-curated picklist where possible; when a specific paper-mentioned mineral is not in the picklist, keep the paper's name. {associated_minerals_picklist},

  "analytical_method": "Standardized method abbreviation. Use the ANALYTICAL METHOD STANDARDIZATION reference above. Common: 'LA-ICPMS', 'EMPA', 'XRF', 'ICP-MS'. If multiple methods were used, list comma-separated. {analytical_method_picklist},
  "instrument_type_model": "Instrument manufacturer and model ONLY. E.g. 'Agilent 7900 ICP-MS coupled with RESOlution 193nm ArF excimer laser'. Copy from paper, no extra description.",
  "laboratory_location": "Lab name and city/institution ONLY. E.g. 'State Key Laboratory of Geological Processes and Mineral Resources, China University of Geosciences, Wuhan'",
  "operating_conditions": "Key analytical parameters: spot size, beam energy, repetition rate, carrier gas, dwell time. E.g. '44 µm spot, 80 mJ, 6 Hz, He carrier gas'",
  "standards_used": "All calibration/reference standards with their purpose, as written in the paper. E.g. 'STDGL3 for chalcophile elements, GSD-1G for lithophile elements, MASS-1 as quality check'",
  "sample_preparation": "How samples were prepared (polished thin sections, mounted in epoxy, pressed pellets, etc.)",
  "aggregation_method": "If the paper explicitly states how reported values are aggregated (e.g. 'reported values are means of n=3 spots', 'median of duplicates', 'single spot analyses'), copy that statement. null if not stated.",
  "data_quality": "Author-stated overall data-quality / QA notes that apply paper-wide (e.g. '2σ uncertainty', 'spots with totals 98-102 wt% accepted', 'analyses with mixed phases discarded'). null if not stated.",
  "sample_date": "If the paper or supplementary states a sample collection date / field campaign date, provide ISO YYYY-MM-DD (or YYYY) form. null if not stated.",
  "analysis_datetime": "If the methods section states when the analyses were performed (e.g. 'samples analysed at LSU in March 2024' or 'analytical campaign Aug-Sep 2022'), provide ISO YYYY-MM-DD or YYYY-MM form. Distinct from sample_date (when the rock was collected). null if not stated.",

  "publication_date": "Year of publication as integer",
  "sample_source": "Full bibliographic citation FOR THIS PAPER: '<authors from page 1>. (<year>). <title from page 1>. <journal>, <volume>, <pages>'. CRITICAL: authors and title MUST come from the page-1 author byline (immediately under the title, before affiliations) — NOT from a recommended-citation box, NOT from the references section, NOT from a self-citation later in the paper. A LLM-extracted self-citation from references commonly differs from the actual page-1 byline (different author order, different paper title) — when in doubt, prefer the page-1 byline.",
  "paper_title": "Full title of the paper, exactly as printed at the TOP of page 1 (between the journal header and the author byline). Do NOT copy from any 'Recommended citation:' box or from a citation in the references — those refer to DIFFERENT papers. The page-1 title block is authoritative.",
  "paper_doi": "DOI of the paper. Find it in the page-1 header/footer margin (publisher-stamped: 'https://doi.org/10.xxx/...' or 'Available online at...' or 'doi.org/10.xxx/...'). Do NOT take a DOI from the references section — those are citations of OTHER papers. If the references section contains DOIs, ignore them. The publisher's own stamp on page 1 is the only valid source.",
  "paper_journal": "Journal name where this paper is published (e.g. 'Ore Geology Reviews', 'Economic Geology', 'Lithos'). null if not stated.",
  "paper_url": "Canonical URL (DOI URL or publisher landing page). Build as 'https://doi.org/<DOI>' if a DOI is found. null otherwise.",
  "country": "ISO 3-letter country code where the deposit is located. ALWAYS FILL THIS when the paper mentions ANY country, province, region, or place name in the country (e.g. 'Sichuan' → CHN, 'Eastern Pontides' → TUR, 'Atacama' → CHL, 'Witwatersrand' → ZAF, 'Pasco' → PER, 'Nevada' → USA). Infer from the place name when the country isn't named verbatim. Use the ISO COUNTRY CODES reference above. {country_picklist},
  "state": "State / province / region as written in the paper (e.g. 'Sichuan Province', 'Nevada', 'Western Cape', 'Atacama Region'). When the paper names a specific city, region, or geological province within a country, capture that here. {state_picklist},

  "deposit_longitude_wgs84": "DEPOSIT longitude in decimal degrees WGS84 if explicitly given in the paper (title, abstract, intro, Fig. 1 caption, methods). Convert DMS (e.g. '109°35'12\"E') to decimal. Positive=East, negative=West. null if not stated (do NOT guess from country).",
  "deposit_latitude_wgs84": "DEPOSIT latitude in decimal degrees WGS84 if explicitly given. Convert DMS (e.g. '24°35'12\"S') to decimal. Positive=North, negative=South. null if not stated.",
  "deposit_location_description": "Free-text geographic location of the deposit as written in the paper. Usually appears in the TITLE or abstract. Examples: 'Pasco, Peru', 'Eastern Pontides, NE Türkiye', 'Western Serbia', 'South China, Sichuan Province'. ALWAYS FILL THIS when the paper mentions ANY geographic location (a country, province, region, city). Combine the most specific named places into a single comma-separated string. Only return null if the paper genuinely contains no geographic identifier at all (very rare).",
  "deposit_name_clean": "Mindat-friendly deposit name with no suffix: strip 'deposit', 'mine', 'district', 'prospect'. E.g. 'Pale Bidau deposit' -> 'Pale Bidau'. Keep multi-word names ('Pale Bidau' stays two words). null if no deposit_name.",
  "deposits_studied_json": "JSON array of deposits the paper ACTUALLY analyzed samples from (NOT cited-only references). Required if >1 deposit was studied. Each entry: {{\"name\":\"…\", \"name_clean\":\"…\", \"commodities\":\"Zn,Pb,Ag\", \"location\":\"Pasco, Peru\", \"country\":\"PER\", \"longitude\":-77.12, \"latitude\":-10.35}}. Coordinates null if not stated; commodities tied to THAT deposit specifically. null if only one deposit. Example for the Hwanggangri paper studying 7 deposits: [{{\"name\":\"Geumseong\", \"commodities\":\"Zn\", \"country\":\"KOR\"}}, {{\"name\":\"Dangdu\", \"commodities\":\"Zn\", \"country\":\"KOR\"}}, …]."
}}
```

## CRITICAL GUIDELINES (USGS/CMMI MANDATORY STANDARDS)
1. **deposit_environment, deposit_group, deposit_type MUST follow the Hofstra et al. 2021 CMMI
   classification scheme** (see DEPOSIT CLASSIFICATION REFERENCE above). Do NOT use any alternative
   classification system, even if the original authors use different terminology. Map the author's
   terminology to the closest Hofstra 2021 category.
2. Use the ANALYTICAL METHOD STANDARDIZATION reference for `analytical_method`.
3. `country` must be an ISO 3-letter code (CHN, AUS, USA, CAN, ZAF, SWE, etc.)
4. Do NOT fabricate coordinates. Only fill `deposit_longitude_wgs84` / `deposit_latitude_wgs84`
   when EXPLICITLY stated in the paper text (typical places: title/abstract, "Geological
   setting" intro paragraph, Fig. 1 caption, sometimes a "Coordinates" line in methods).
   - Coordinates may be given in decimal degrees ("24.345°N, 109.876°E"),
     degrees-decimal-minutes ("24°35.7'N"), or degrees-minutes-seconds ("24°35'42\"N").
   - Convert ALL to decimal degrees. Apply hemisphere signs: S/W are NEGATIVE.
   - Example conversion: 24°35'42"S → -(24 + 35/60 + 42/3600) = -24.5950
   - If only a textual location is given (e.g. "Pasco, Peru" — no numeric coords) put it
     in `deposit_location_description` and leave `*_longitude_wgs84` / `*_latitude_wgs84` null.
5. For `sample_source`, construct the full bibliographic citation from the paper header/footer.
6. Copy instrument descriptions, lab locations, and analytical conditions VERBATIM from the paper.
7. If the paper studies multiple deposits, list comma-separated. But for `mineral`, each row
   can only have ONE mineral — if multiple minerals were analyzed, the per-row assignment
   will be handled downstream. List the primary mineral here.
8. Do NOT hallucinate or assume field values without strong confidence.
9. `deposit_name_clean` should be the same as `deposit_name` with these suffixes stripped:
   "deposit", "Deposit", "mine", "Mine", "district", "District", "prospect", "ore field".
   Keep diacritics and internal spacing. Use for downstream Mindat search.
10. For every PICKLIST-tagged field above, you MUST emit one of the listed values
    exactly. If the paper does not state a value that matches the picklist, return
    null. Inventing a label is strictly forbidden and will be rejected downstream.

## EVIDENCE-ANCHORED FIELDS

For EVERY paper-level metadata field whose value comes from the paper body
(not the manifest, not derived from a picklist alone), emit the field as a
{{value, evidence_quote}} OBJECT instead of a bare string. The downstream
pipeline locates the quote in the PDF and attaches a (page, bbox) to the
field so the value is fully traceable. The fields that MUST come back as
objects are:

  feature_name, strat_unit_name, alteration, deposit_location_description,
  deposit_name, deposit_name_clean, deposit_type, deposit_type_original,
  deposit_group, deposit_environment, primary_commodities,
  secondary_commodities, associated_minerals, mineral, material_class,
  country, state, analytical_method, instrument_type_model,
  laboratory_location, operating_conditions, standards_used,
  paper_title, paper_doi, paper_journal, sample_source.

Use the form:

```json
"feature_name": {{
  "value": "orebody no. 1 level 1884 to 2064 m",
  "evidence_quote": "Samples La-1 to La-5 are from orebody no. 1, levels 1884 m to 2064 m"
}}
```

INTERPRET the paper — do not parrot it. The `value` is YOUR best, most useful
summary of what the paper says about this field. PARAPHRASING IS WELCOME and often
the right move:
  - Combine multiple paper sentences into one concise value when that captures the
    full picture (e.g. paper says "carbonate alteration is widespread" in section 3
    and "silicification occurs along faults" in section 4 → value: "carbonatization,
    silicification").
  - Re-order or re-word geographic descriptions for clarity (e.g. paper says
    "located in the southwestern Sichuan Basin of South China" → value:
    "South China, Sichuan Province").
  - For feature_name, include every QUANTITATIVE qualifier (depth, level, orebody
    number, drill-hole ID) even when those qualifiers are split across sentences.

The `evidence_quote` is a SUPPORTING SNIPPET — roughly 8 to 30 words copied
from the paper that show where the value came from. It need NOT be a verbatim
substring; the downstream verifier checks key-phrase overlap, not exact match.
Choose a snippet where the relevant proper nouns, technical terms, or named
entities appear so the downstream PDF-bbox locator can find the source span.

PICKLIST FIELDS STILL NEED EVIDENCE. When the `value` is a picklist code
(material_class, country, mineral, deposit_type, sample_type, etc.), the
quote must point at the paper text that JUSTIFIES the picklist selection.
Examples of what to quote:
  - For material_class = 'mineralised rock': quote the paper passage that
    says samples are in-situ analyses on the host rock (e.g. 'thin sections
    of dolomitic ore from the orebody were prepared for LA-ICPMS spot
    analysis of sphalerite').
  - For country = 'CHN': quote the sentence naming the country
    (e.g. 'the Daliangzi Pb-Zn deposit in South China').
  - For mineral = 'sphalerite': quote the analytical-method or
    sample-description sentence naming the target mineral.
NEVER emit a picklist value with a null or empty quote. Every value must
trace to a span of paper text.

If the paper genuinely does not discuss the field at all, return:

```json
"feature_name": {{ "value": null, "evidence_quote": null }}
```

For fields that are derived programmatically (DOI numbers, dates as integers,
WGS84 coordinates as floats), emit a bare value as before — those do not
need a quote because they are exact strings/numbers parsed from a specific
place by the manifest scanner.

{picklist_values_block}

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: LLM-assisted table verification / ambiguity resolution
# ──────────────────────────────────────────────────────────────────────────────

TABLE_FILTER_SYSTEM_PROMPT = dedent("""\
You are a geochemistry database curator. You will be given a raw supplementary table
from a research paper. Your job is to identify which rows are genuine sample analyses
from THIS paper (not comparison data from other references or summary statistics).
""")

TABLE_FILTER_USER_PROMPT_TEMPLATE = dedent("""\
## SUPPLEMENTARY TABLE
{table_text}

## TASK
1. Identify rows that are GENUINE SAMPLE ANALYSES from this paper.
   - INCLUDE rows where reference/source = "this paper", "this study", or is blank/missing.
   - EXCLUDE rows cited from other papers (e.g., "Ye et al., 2011").
   - EXCLUDE statistical summary rows (MEAN, STD, MINIMA, MAXIMA, AVERAGE, etc.).
   - EXCLUDE rows with no sample identifier.

2. For each valid sample row, extract the sample name and ALL element concentrations.

## ELEMENT COLUMN MAPPING
{column_mapping}

## OUTPUT FORMAT
Return a JSON array. Each element is one valid sample:
[
  {{
    "sample_name": "LA1-1_1",
    "deposit_name": "Daliangzi",
    "ag_ppm": 43.7,
    "as_ppm": 13.94,
    "bi_ppm": 0.0196,
    "cd_ppm": 1479.04,
    "co_ppm": 2.22,
    "fe_ppm": 32998.28,
    ... (all measured elements)
  }},
  ...
]

Rules:
- CRITICAL DISTINCTION between below-detection and not-measured (USGS protocol):
  * If analysis was performed but the value is below detection limit (BDL, b.d.l., n.d., <DL, "-", "--"):
    - If NO specific detection limit is given → use -99999
    - If a specific detection limit IS given (e.g., "<0.5 ppm") → use the NEGATIVE of that limit (e.g., -0.5)
    - Detection limits may be listed in the table or buried in the paper text
  * If "N/A", "not analyzed", "not measured", or "not reported" → use null (blank = not attempted)
  * -99999 means "measured but too low to quantify, LOD unknown". null means "not measured at all".
  * Leaving a below-detection field as null LOSES valuable information.
- Each data row must be assigned to exactly ONE mineral. Never group minerals (e.g., NOT "chalcopyrite, sphalerite").
  If a table mixes minerals, use the mineral column to assign one mineral per row.
- USGS THREE-TIER SAMPLE ID RULE (mandatory, per USGS CMiO-MIN protocol):
  * "sample_name":     CORE numerical identifier of the PHYSICAL sample
                       (e.g., "2002063521" — the core, NOT the full string).
                       Multiple analysis rows on the SAME physical sample
                       MUST share this value. It is NOT unique per row.
  * "sample_local_id": IDENTICAL to "sample_name" (same core ID).
                       The slides are explicit: BOTH are "isolated as the
                       core numerical identifier". Always emit the same
                       value as sample_name. NOT unique per row.
  * "analysis_id":     FULL analysis string per row (e.g.,
                       "5-2002063521cpy1-1.d"). Strip a trailing "-1"
                       version suffix if present. UNIQUE per row.
  Example: raw supp ID "5-2002063521cpy1-1.d" with 12 rows on this sample
    →  sample_name="2002063521", sample_local_id="2002063521",
       analysis_id="5-2002063521cpy1" (each row gets its own analysis_id).
  Common LLM error: making sample_name UNIQUE per row by including the
  spot/run number. WRONG. Strip the trailing spot number from sample_name.
- UNIT CONVERSION: All values MUST be in ppm. Convert wt% × 10000, ppb ÷ 1000.
- Output samples in the EXACT order they appear in the table — no sorting/reordering.
- Round values to 4 significant figures maximum.
- Do NOT add elements not present in the supplementary table.
- Do NOT hallucinate or assume values — only extract what is explicitly reported.

Return ONLY the JSON array. No explanation.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Combined one-shot prompt (for very capable models / small tables)
# ──────────────────────────────────────────────────────────────────────────────

FULL_EXTRACTION_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator. Extract structured data from a research
paper (PDF text + supplementary table) and output it in the standardized 210-column schema.
Be precise: extract values EXACTLY as reported, using null for any field not in the paper.
""")

FULL_EXTRACTION_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT (key sections)
{paper_text}

## SUPPLEMENTARY TABLE
{table_text}

## YOUR TASK
Extract ALL sample analytical data into the standardized schema. Output a JSON object with:
  1. "metadata": paper-level fields that apply to every row
  2. "samples": array of per-sample rows with element concentrations

## METADATA SCHEMA (same for all rows)
{metadata_schema}

## SAMPLE SCHEMA (one per analytical spot)
Each sample object must have:
- "sample_name": sample identifier from the table
- "deposit_name": deposit name (may differ per row if multiple deposits)
- Element columns: "{element}_ppm" for each measured element (null if not measured)

## FILTERING RULES
- Include ONLY rows from THIS paper (reference = "this paper" / "this study" / blank)
- EXCLUDE statistical summary rows: MEAN, STD, MINIMA, MAXIMA
- EXCLUDE rows cited from other papers

## OUTPUT FORMAT
{{
  "metadata": {{
    "deposit_name": "...",
    "deposit_environment": "...",
    ... (all PaperMetadata fields)
  }},
  "samples": [
    {{
      "sample_name": "...",
      "ag_ppm": null,
      "fe_ppm": 32998.28,
      ... (only include elements present in the table)
    }},
    ...
  ]
}}

Return ONLY valid JSON. No markdown fences, no explanation.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: PDF-only table extraction (no supplementary file)
# ──────────────────────────────────────────────────────────────────────────────

PDF_TABLE_EXTRACTION_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator. You are given the text of a research paper
(including any tables embedded in the PDF). There is NO supplementary spreadsheet available.
Your task is to extract individual sample analytical data (element concentrations) directly
from tables in the PDF text.

Extract information EXACTLY as reported — do not paraphrase, estimate, or fabricate values.

{knowledge_base}
""")

PDF_TABLE_EXTRACTION_USER_PROMPT_TEMPLATE = dedent("""\
## PAPER TEXT (relevant sections)
{paper_text}

## TABLES FOUND IN PDF
{pdf_tables}

## YOUR TASK
Extract ALL individual sample analyses from the tables above. Each row should be one
analytical spot/measurement with its element concentrations.

Return a JSON object with:
1. "samples": array of per-sample objects
2. "extraction_notes": brief description of which table(s) you extracted from

## SAMPLE FORMAT
Each sample object must include:
- "sample_name": the sample identifier from the table
- Element concentrations as "{{element_symbol}}_ppm" (e.g., "fe_ppm", "cu_ppm", "zn_ppm")
  * Keep values in their ORIGINAL units exactly as printed — do NOT convert between units
  * The "_ppm" suffix is just the column name convention — it does NOT mean you should convert to ppm
  * If the table says Fe = 2.19 wt%, record "fe_ppm": 2.19
  * If the table says Cu = 123.5 ppm, record "cu_ppm": 123.5
  * If the table says Au = 50 ppb, record "au_ppm": 50
  * USGS BDL protocol:
    - If below detection limit and NO specific LOD given (BDL, b.d.l., n.d., dash) → use -99999
    - If below detection limit and a specific LOD IS given (e.g., "<0.5") → use NEGATIVE of that limit (e.g., -0.5)
    - If "N/A", "not analyzed", "not measured" → use null (blank = not attempted at all)
    - -99999 = "measured, below detection, LOD unknown". null = "not measured".
  * Use null ONLY for elements that were NOT measured / NOT reported at all
- Optional per-sample metadata fields (include only if they vary per sample):
  * "deposit_name": deposit name if it differs per sample
  * "mineral": EXACTLY ONE mineral name per row — never group (e.g., NOT "pyrite, chalcopyrite")
  * "analytical_method": method if it differs per sample
  * "sample_name":     PHYSICAL sample CORE id (e.g. "2002063521"); NOT unique per row — repeats across all analytical spots on the same sample
  * "sample_local_id": IDENTICAL to sample_name (USGS rule); always equal to it
  * "analysis_id":     full analysis-run identifier (e.g. "5-2002063521cpy1"); UNIQUE per row. Strip trailing "-1" version suffix if present
  * "texture": texture description if available

## FILTERING RULES
- Include ONLY rows that are actual sample measurements (individual spots/analyses)
- EXCLUDE statistical summary rows: MEAN, AVERAGE, STD, MIN, MAX, MEDIAN
- EXCLUDE rows cited from other studies/references
- EXCLUDE detection limit rows
- EXCLUDE blank or header rows

## UNIT CONVERSION (MANDATORY)
- All element values in the output MUST be in ppm.
- If the table reports values in wt%, convert to ppm: multiply by 10,000
  (e.g., Fe = 2.19 wt% → "fe_ppm": 21900)
- If the table reports values in ppb, convert to ppm: divide by 1,000
  (e.g., Au = 50 ppb → "au_ppm": 0.05)
- If the table reports values in ppm, µg/g, or mg/kg — no conversion needed
- The "_ppm" column suffix means the value MUST be in ppm units

## SAMPLE ORDER (CRITICAL)
- Output samples in EXACTLY the same order they appear in the paper tables
- Do NOT sort, group, or reorder samples by name, mineral, or any other field
- The human evaluator will compare row-by-row with the source table

## ELEMENT SYMBOL REFERENCE
Common elements and their symbols (use lowercase):
  Fe=fe, Cu=cu, Zn=zn, Pb=pb, Ag=ag, Au=au, As=as, Sb=sb, Bi=bi, Co=co,
  Ni=ni, Mn=mn, Cd=cd, In=in, Ga=ga, Ge=ge, Se=se, Te=te, Tl=tl, Sn=sn,
  Mo=mo, W=w, V=v, Cr=cr, Ti=ti, Sc=sc, Ba=ba, Sr=sr, Rb=rb, Cs=cs,
  Li=li, Be=be, B=b, P=p, S=s, Cl=cl, F=f, Br=br, Hg=hg, Re=re,
  Si=si, Al=al, Ca=ca, Mg=mg, Na=na, K=k,
  La=la, Ce=ce, Pr=pr, Nd=nd, Sm=sm, Eu=eu, Gd=gd, Tb=tb, Dy=dy,
  Ho=ho, Er=er, Tm=tm, Yb=yb, Lu=lu, Y=y, Zr=zr, Hf=hf, Nb=nb, Ta=ta,
  Th=th, U=u

## OUTPUT FORMAT
{{
  "samples": [
    {{
      "sample_name": "PY-1-1",
      "fe_ppm": 460000,
      "cu_ppm": 123.5,
      "zn_ppm": 45.2,
      "as_ppm": null,
      ...
    }},
    ...
  ],
  "extraction_notes": "Extracted 45 analyses from Table 2 (LA-ICP-MS trace elements in pyrite)"
}}

If the paper has NO extractable sample-level analytical data in the PDF tables
(e.g., only summary statistics, or tables are about something else entirely),
return: {{"samples": [], "extraction_notes": "No individual sample analyses found in PDF tables"}}

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: Vision-based PDF table extraction (page images)
# ──────────────────────────────────────────────────────────────────────────────

# Shared sample format instructions (used by both text and vision prompts)
_SAMPLE_FORMAT_INSTRUCTIONS = dedent("""\
## SAMPLE FORMAT
Each sample object must include:
- "sample_name": the sample identifier from the table
- Element concentrations as "{{element_symbol}}_ppm" (e.g., "fe_ppm", "cu_ppm", "zn_ppm")
  * Keep values in their ORIGINAL units exactly as printed in the table — do NOT convert between units
  * If the table says Fe = 2.19 wt%, record "fe_ppm": 2.19
  * If the table says Cu = 123.5 ppm, record "cu_ppm": 123.5
  * If the table says Au = 50 ppb, record "au_ppm": 50
  * The "_ppm" suffix is just the column name — it does NOT mean you should convert to ppm
  * USGS BDL protocol:
    - If below detection limit and NO specific LOD given (BDL, b.d.l., n.d., dash) → use -99999
    - If below detection limit and a specific LOD IS given (e.g., "<0.5") → use NEGATIVE of that limit (e.g., -0.5)
    - If "N/A", "not analyzed", "not measured" → use null (blank = not attempted at all)
    - -99999 = "measured, below detection, LOD unknown". null = "not measured".
  * Use null ONLY for elements that were NOT measured / NOT reported at all
- Optional per-sample metadata fields (include only if they vary per sample):
  * "deposit_name": deposit name if it differs per sample
  * "mineral": EXACTLY ONE mineral name per row — never group (e.g., NOT "pyrite, chalcopyrite")
  * "analytical_method": method if it differs per sample
  * "sample_name":     PHYSICAL sample CORE id (e.g. "2002063521"); NOT unique per row — repeats across all analytical spots on the same sample
  * "sample_local_id": IDENTICAL to sample_name (USGS rule); always equal to it
  * "analysis_id":     full analysis-run identifier (e.g. "5-2002063521cpy1"); UNIQUE per row. Strip trailing "-1" version suffix if present
  * "texture": texture description if available

## FILTERING RULES
- Include ONLY rows that are actual sample measurements (individual spots/analyses)
- EXCLUDE statistical summary rows: MEAN, AVERAGE, STD, MIN, MAX, MEDIAN
- EXCLUDE rows cited from other studies/references
- EXCLUDE detection limit rows
- EXCLUDE blank or header rows

## UNIT CONVERSION (MANDATORY)
- All element values in the output MUST be in ppm.
- If the table reports values in wt%, convert to ppm: multiply by 10,000
  (e.g., Fe = 2.19 wt% → "fe_ppm": 21900)
- If the table reports values in ppb, convert to ppm: divide by 1,000
  (e.g., Au = 50 ppb → "au_ppm": 0.05)
- If already in ppm, µg/g, or mg/kg — no conversion needed

## SAMPLE ORDER (CRITICAL)
- Output samples in EXACTLY the same order they appear in the paper tables
- Do NOT sort, group, or reorder samples

## ELEMENT SYMBOL REFERENCE
Common elements and their symbols (use lowercase):
  Fe=fe, Cu=cu, Zn=zn, Pb=pb, Ag=ag, Au=au, As=as, Sb=sb, Bi=bi, Co=co,
  Ni=ni, Mn=mn, Cd=cd, In=in, Ga=ga, Ge=ge, Se=se, Te=te, Tl=tl, Sn=sn,
  Mo=mo, W=w, V=v, Cr=cr, Ti=ti, Sc=sc, Ba=ba, Sr=sr, Rb=rb, Cs=cs,
  Li=li, Be=be, B=b, P=p, S=s, Cl=cl, F=f, Br=br, Hg=hg, Re=re,
  Si=si, Al=al, Ca=ca, Mg=mg, Na=na, K=k,
  La=la, Ce=ce, Pr=pr, Nd=nd, Sm=sm, Eu=eu, Gd=gd, Tb=tb, Dy=dy,
  Ho=ho, Er=er, Tm=tm, Yb=yb, Lu=lu, Y=y, Zr=zr, Hf=hf, Nb=nb, Ta=ta,
  Th=th, U=u

## OUTPUT FORMAT
{{
  "samples": [
    {{
      "sample_name": "PY-1-1",
      "fe_ppm": 460000,
      "cu_ppm": 123.5,
      "zn_ppm": 45.2,
      "as_ppm": null,
      ...
    }},
    ...
  ],
  "extraction_notes": "Extracted 45 analyses from Table 2 (LA-ICP-MS trace elements in pyrite)"
}}

If there are NO extractable sample-level analytical data,
return: {{"samples": [], "extraction_notes": "No individual sample analyses found"}}

Return ONLY the JSON object. No explanation, no markdown code blocks.
""")


VISION_TABLE_EXTRACTION_SYSTEM_PROMPT = dedent("""\
You are an expert geochemistry database curator. You are given IMAGES of pages from a
research paper PDF that contain geochemical data tables. There is NO supplementary
spreadsheet available. Your task is to visually read the tables and extract individual
sample analytical data (element concentrations).

Read the tables EXACTLY as printed — do not estimate, interpolate, or fabricate values.

## TABLE RECOGNITION RULES

**Orientation & Layout:**
- LANDSCAPE (rotated) pages are COMMON in geochem papers — wide tables with many element
  columns are often printed sideways. Read the table in its natural reading direction
  regardless of page rotation. If a page appears rotated 90°, mentally rotate it and
  read normally.
- Tables may span the full page width OR be split into two side-by-side sub-tables.
- Some tables are TRANSPOSED: elements as rows, samples as columns. Detect this and
  extract correctly — each column becomes one sample, each row is an element.

**Headers & Structure:**
- Column headers may contain element symbols (Fe, Cu, Zn), isotope prefixes (55Mn, 59Co),
  oxide notation (SiO2, Al2O3, FeO), or unit suffixes (ppm, wt%, ppb).
- MULTI-ROW headers are common: Row 1 = element symbols, Row 2 = units, Row 3 = detection limits.
  Combine all header rows to identify what each column represents.
- Merged cells: sample name or mineral name may span multiple rows. Apply the merged value
  to ALL rows it spans.

**Continuation tables:**
- "Table X (continued)" or "Table X (cont.)" means this is a continuation from a previous
  page — the column headers may be repeated but the data rows are NEW samples.
- Some continuation tables continue the SAME samples with DIFFERENT elements (horizontal
  continuation). In this case, merge the element values into the same sample row.

**Data values (USGS BDL protocol):**
- Below detection limit with NO specific LOD: bdl, b.d.l., n.d., dash (—, –, -) → use -99999
- Below detection limit WITH a specific LOD: "<0.5" → use -0.5 (NEGATIVE of the reported LOD)
- "N/A", "not analyzed", "not measured" → use null (blank = not attempted at all)
- -99999 means "measured but too low, LOD unknown". Negative values (e.g., -0.5) mean "below LOD of 0.5".
  null/blank means "not measured at all".
- Commas in numbers are THOUSANDS separators (50,525 = 50525), NOT decimal points.
- Keep values in their ORIGINAL units as printed. Do NOT convert wt% to ppm or vice versa.
- Do NOT hallucinate or assume values. Only extract what is explicitly in the table.

**What to EXCLUDE:**
- Summary/statistics rows: Mean, Median, Average, Std Dev, Min, Max, Range, n= → SKIP these.
- Reference/comparison data from OTHER papers cited in the table → only extract data from THIS paper.
- Table captions, footnotes, and annotations → do not extract as data rows.

{knowledge_base}
""")


VISION_TABLE_EXTRACTION_USER_PROMPT = dedent("""\
## IMAGES
The attached {n_pages} image(s) show pages from the PDF that likely contain
geochemical data tables. Read them carefully — these are the actual page renders.
{orientation_hint}

## PAPER CONTEXT (text from the paper for additional context)
{paper_context}

## YOUR TASK
Extract ALL individual sample analyses visible in the table images. Each row should be
one analytical spot/measurement with its element concentrations.

If a page shows NO data table (only text, figures, or maps), return an empty samples
array for that page — do not fabricate data.

If a table is in LANDSCAPE orientation (rotated 90°), mentally rotate the page and read
the table in its natural left-to-right, top-to-bottom direction.

Return a JSON object with:
1. "samples": array of per-sample objects
2. "extraction_notes": brief description of which table(s) you extracted from,
   including table number/caption if visible

{sample_format_instructions}
""")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for building prompts
# ──────────────────────────────────────────────────────────────────────────────

def build_paper_intelligence_prompt(
    paper_text: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage 0 paper intelligence extraction.

    This focused prompt extracts analytical methodology details (instrument,
    lab, standards, conditions) and the list of elements measured — information
    that guides downstream metadata and table extraction.
    """
    user = PAPER_INTELLIGENCE_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:20000],
    )
    return PAPER_INTELLIGENCE_SYSTEM_PROMPT, user


def build_metadata_prompt(
    paper_text: str,
    table_preview: str,
    paper_map_block: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage 1 metadata extraction.

    `paper_map_block` is the rendered output of
    ``paper_map.format_for_extraction_prompt(pmap)`` — the per-paper map
    that tells the LLM where to look for each field. Pass an empty
    string to skip injection (Layer 5 will be active when this is
    populated upstream)."""
    system = METADATA_SYSTEM_PROMPT.format(
        knowledge_base=get_knowledge_base_prompt(),
    )
    # Large picklists (>60 values) are listed under a separate
    # PICKLIST VALUES section so the per-field clause stays short.
    picklist_values_block = large_picklist_section([
        "mineral", "associated_minerals", "earth_material",
        "earth_material_group", "deposit_type", "deposit_group",
        "country",
    ])
    # Per-field picklist clauses to inline into the JSON-schema example.
    picklist_kwargs = {
        f"{field}_picklist": picklist_clause(field) for field in (
            "primary_commodities", "secondary_commodities",
            "feature_type", "sample_deposit_relation", "sample_type",
            "sampling_method", "material_class", "earth_material_group",
            "earth_material", "alteration", "mineral",
            "associated_minerals", "analytical_method",
            "country", "state",
        )
    }
    user = METADATA_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:30000],
        table_preview=table_preview[:3000],
        picklist_values_block=picklist_values_block,
        paper_map_block=paper_map_block,
        **picklist_kwargs,
    )
    return system, user


def build_table_filter_prompt(
    table_text: str,
    column_mapping: dict[str, str],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage 2 table filtering."""
    mapping_lines = "\n".join(
        f'  Column "{raw}" → schema field "{sym}_ppm"'
        for raw, sym in column_mapping.items()
    )
    user = TABLE_FILTER_USER_PROMPT_TEMPLATE.format(
        table_text=table_text[:8000],
        column_mapping=mapping_lines or "  (auto-detected from column names)",
    )
    return TABLE_FILTER_SYSTEM_PROMPT, user


def build_pdf_table_extraction_prompt(
    paper_text: str,
    pdf_tables: list[str],
    data_pages_text: str = "",
    elements_measured: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for PDF-only table extraction.

    Used when no supplementary file is available — extracts sample data
    directly from tables embedded in the PDF.

    Args:
        paper_text: Prioritised paper text (abstract, methods, results).
        pdf_tables: Structured tables extracted by pdfplumber.
        data_pages_text: Raw text from pages with high numeric density
            (fallback when pdfplumber can't extract structured tables).
        elements_measured: Optional list of element symbols (lowercase)
            from Paper Intelligence — constrains extraction scope.
    """
    system = PDF_TABLE_EXTRACTION_SYSTEM_PROMPT.format(
        knowledge_base=get_knowledge_base_prompt(),
    )

    # Format PDF tables
    if pdf_tables:
        tables_text = "\n\n".join(
            f"--- Table {i+1} ---\n{tbl}"
            for i, tbl in enumerate(pdf_tables)
            if tbl.strip()
        )
        if not tables_text.strip():
            tables_text = ""
    else:
        tables_text = ""

    # When no structured tables, include raw data-dense pages
    if not tables_text and data_pages_text:
        tables_text = (
            "(No structured tables could be extracted from the PDF. "
            "Below are raw text pages that may contain tabular data. "
            "Look for data patterns — numbers aligned in columns, "
            "sample IDs followed by element concentrations, etc.)\n\n"
            + data_pages_text[:10000]
        )
    elif not tables_text:
        tables_text = (
            "(No structured tables could be extracted from the PDF. "
            "Look carefully in the PAPER TEXT above for any inline "
            "sample data — tables rendered as text, data in the results "
            "section, appendices, or any section with element concentrations.)"
        )

    user = PDF_TABLE_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:20000],
        pdf_tables=tables_text[:12000],
    )

    # Append element scope constraint from Paper Intelligence
    if elements_measured:
        scope = ", ".join(sorted(elements_measured))
        user += (
            f"\n\n## ELEMENT SCOPE (from paper analysis)\n"
            f"This paper measures these specific elements: {scope}\n"
            f"Focus extraction on ONLY these elements. Use null for all others.\n"
        )

    return system, user


def build_vision_table_extraction_prompt(
    n_pages: int,
    paper_context: str = "",
    elements_measured: list[str] | None = None,
    has_landscape: bool = False,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for vision-based table extraction.

    Used when pdfplumber cannot extract structured tables — the LLM reads
    rendered page images directly to extract tabular data.

    Args:
        n_pages: Number of page images being sent.
        paper_context: Truncated paper text for context (abstract, methods).
        elements_measured: Optional list of element symbols (lowercase)
            from Paper Intelligence — constrains extraction scope.
        has_landscape: Whether any of the pages are landscape-oriented.
    """
    system = VISION_TABLE_EXTRACTION_SYSTEM_PROMPT.format(
        knowledge_base=get_knowledge_base_prompt(),
    )

    orientation_hint = ""
    if has_landscape:
        orientation_hint = (
            "\nIMPORTANT: Some pages are LANDSCAPE-ORIENTED (rotated 90°). "
            "These contain wide data tables with many element columns. "
            "Rotate mentally and read carefully — landscape tables often "
            "contain the bulk of the paper's geochemical data."
        )

    user = VISION_TABLE_EXTRACTION_USER_PROMPT.format(
        n_pages=n_pages,
        paper_context=paper_context[:6000] if paper_context else "(no text context available)",
        sample_format_instructions=_SAMPLE_FORMAT_INSTRUCTIONS,
        orientation_hint=orientation_hint,
    )

    # Append element scope constraint from Paper Intelligence
    if elements_measured:
        scope = ", ".join(sorted(elements_measured))
        user += (
            f"\n\n## ELEMENT SCOPE (from paper analysis)\n"
            f"This paper measures these specific elements: {scope}\n"
            f"Focus extraction on ONLY these elements. Use null for all others.\n"
        )

    return system, user


def build_full_extraction_prompt(
    paper_text: str,
    table_text: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one-shot full extraction."""
    metadata_fields = json.dumps(
        {k: None for k in PaperMetadata.model_fields.keys()},
        indent=2
    )
    user = FULL_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_text[:10000],
        table_text=table_text[:8000],
        metadata_schema=metadata_fields,
    )
    return FULL_EXTRACTION_SYSTEM_PROMPT, user


def build_metadata_tool_schema() -> dict:
    """Return a valid Anthropic tool schema (JSON Schema draft 2020-12) for metadata extraction.

    All fields are Optional so every property is typed as ["<type>", "null"].
    Python type names are mapped to their JSON Schema equivalents:
        str   → "string"
        int   → "integer"
        float → "number"
    """
    _py_to_json = {"str": "string", "int": "integer", "float": "number"}

    # Sample-level coordinates are skipped: those are per-row from supp tables.
    # Deposit-level coords ARE requested (often stated in title/intro/Fig. 1
    # caption) — they disambiguate same-name deposits for downstream geocoding.
    skip = {"sample_longitude_wgs84", "sample_latitude_wgs84"}

    properties = {}
    for field_name, field_info in PaperMetadata.model_fields.items():
        if field_name in skip:
            continue
        # Resolve the inner type of Optional[X] → get X
        annotation = field_info.annotation
        if hasattr(annotation, "__args__"):
            # Optional[X] is Union[X, None]; take the first non-None arg
            inner = next(
                (a for a in annotation.__args__ if a is not type(None)),
                str,
            )
        else:
            inner = annotation
        json_type = _py_to_json.get(getattr(inner, "__name__", "str"), "string")
        properties[field_name] = {
            "type": [json_type, "null"],
            "description": field_info.description or field_name,
        }

    return {
        "name": "extract_paper_metadata",
        "description": (
            "Extract paper-level geochemical metadata that applies to all samples. "
            "Use null for any field not explicitly stated in the paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [],
        },
    }
