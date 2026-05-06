"""
ChemTables Ontology Module
Domain: Drug Discovery / Medicinal Chemistry
Task: Extract bioactivity tuples from compound-assay tables

Ground truth format: {value, type, target, treatment, unit}
where type in {IC50, EC50, GI50, MIC}
"""

# =============================================================================
# Bioassay Type Taxonomy
# Maps raw table header/caption strings to canonical assay types
# =============================================================================
ASSAY_TAXONOMY = {
    # IC50 variants
    "ic50": "IC50",
    "ic 50": "IC50",
    "ic₅₀": "IC50",
    "inhibition ic50": "IC50",
    "inhibitory concentration": "IC50",
    "half maximal inhibitory": "IC50",
    "half-maximal inhibitory": "IC50",

    # EC50 variants
    "ec50": "EC50",
    "ec 50": "EC50",
    "ec₅₀": "EC50",
    "effective concentration": "EC50",
    "half maximal effective": "EC50",

    # GI50 variants
    "gi50": "GI50",
    "gi 50": "GI50",
    "gi₅₀": "GI50",
    "growth inhibition": "GI50",
    "growth inhibitory": "GI50",

    # MIC variants
    "mic": "MIC",
    "minimum inhibitory concentration": "MIC",
    "minimal inhibitory concentration": "MIC",
    "minimum inhibitory conc": "MIC",

    # CC50 (sometimes appears)
    "cc50": "CC50",
    "cytotoxic concentration": "CC50",
}

VALID_ASSAY_TYPES = {"IC50", "EC50", "GI50", "MIC", "CC50"}

# =============================================================================
# Unit Standardization
# Maps raw unit strings to canonical forms
# =============================================================================
UNIT_STANDARDIZATION = {
    # Micromolar
    "µm": "µM", "μm": "µM", "um": "µM", "microm": "µM",
    "µmol/l": "µM", "μmol/l": "µM", "micromolar": "µM",
    # Nanomolar
    "nm": "nM", "nanomolar": "nM", "nmol/l": "nM",
    # Microgram/mL
    "µg/ml": "µg/mL", "μg/ml": "µg/mL", "ug/ml": "µg/mL",
    "mg/l": "µg/mL",  # 1 mg/L = 1 µg/mL
    # Millimolar
    "mm": "mM", "mmol/l": "mM", "millimolar": "mM",
    # Picomolar
    "pm": "pM", "picomolar": "pM",
}

# =============================================================================
# Target Protein / Organism Taxonomy
# Common bioassay targets for normalization (not exhaustive — expands with data)
# =============================================================================
TARGET_TAXONOMY = {
    # Kinases
    "chk1": "CHK1", "chk-1": "CHK1", "checkpoint kinase 1": "CHK1",
    "chk2": "CHK2", "chk-2": "CHK2", "checkpoint kinase 2": "CHK2",
    "egfr": "EGFR", "her1": "EGFR",
    "vegfr": "VEGFR", "vegfr-2": "VEGFR-2", "kdr": "VEGFR-2",
    "abl": "ABL", "bcr-abl": "BCR-ABL",

    # Cell lines (GI50 targets)
    "a549": "A549", "hela": "HeLa", "mcf-7": "MCF-7", "mcf7": "MCF-7",
    "du145": "DU145", "du-145": "DU145",
    "hl-60": "HL-60", "hl60": "HL-60",
    "k562": "K562",

    # Microbial (MIC targets)
    "s. aureus": "S. aureus", "staphylococcus aureus": "S. aureus",
    "e. coli": "E. coli", "escherichia coli": "E. coli",
    "mrsa": "MRSA",
    "m. tuberculosis": "MTB", "mtb": "MTB",
    "mycobacterium tuberculosis": "MTB",
    "c. albicans": "C. albicans", "candida albicans": "C. albicans",

    # Cell line name normalization
    "hl-60": "HL60", "hl60": "HL60",
    "ccrf-cem": "CCRF-CEM", "ccrfcem": "CCRF-CEM",
    "molt-3": "MOLT-3", "molt3": "MOLT-3",
    "hd2 (cho-d2)": "D2", "hd2": "D2", "cho-d2": "D2",
    "hd3 (cho-d3)": "D3", "hd3": "D3", "cho-d3": "D3",
}

# =============================================================================
# Semantic Constraints
# =============================================================================
CONSTRAINTS = {
    # Null semantics for drug discovery
    # ">" prefix means above assay range (inactive at max tested conc)
    # "N/T" or "NT" = not tested
    # blank = not tested (same as N/T)
    "null_conventions": {
        "not_tested": ["N/T", "NT", "n.t.", "not tested", ""],
        "inactive": [">"],  # prefix: >100 µM means inactive
        "below_detection": ["<"],  # prefix: <0.01 µM means very active
    },

    # Value plausibility: IC50/EC50 typically 0.001 nM to 1000 µM
    "plausibility": {
        "IC50": {"min_nM": 0.001, "max_uM": 1000},
        "EC50": {"min_nM": 0.001, "max_uM": 1000},
        "GI50": {"min_nM": 0.01, "max_uM": 500},
        "MIC":  {"min_ug_ml": 0.001, "max_ug_ml": 1000},
    },

    # Type determines what "target" means
    "type_target_semantics": {
        "IC50": "protein/enzyme target",
        "EC50": "cell/receptor target",
        "GI50": "cell line",
        "MIC":  "organism/strain",
    },
}


def standardize_unit(raw_unit: str) -> str:
    """Normalize a unit string to canonical form."""
    key = raw_unit.strip().lower().replace(" ", "")
    return UNIT_STANDARDIZATION.get(key, raw_unit.strip())


def classify_assay_type(header_text: str) -> str | None:
    """Attempt to classify assay type from a column header or caption string."""
    text = header_text.lower().strip()
    for pattern, canonical in ASSAY_TAXONOMY.items():
        if pattern in text:
            return canonical
    return None


def standardize_target(raw_target: str) -> str:
    """Normalize a target name."""
    key = raw_target.strip().lower()
    return TARGET_TAXONOMY.get(key, raw_target.strip())


def is_null_value(raw_value: str) -> str | None:
    """
    Check if a raw cell value represents a null/non-measurement.
    Returns: 'not_tested', 'inactive', 'below_detection', or None (real value).
    """
    v = raw_value.strip()
    if v in CONSTRAINTS["null_conventions"]["not_tested"]:
        return "not_tested"
    if v.startswith(">"):
        return "inactive"
    if v.startswith("<"):
        return "below_detection"
    return None


# =============================================================================
# Ontology metadata
# =============================================================================
ONTOLOGY_INFO = {
    "name": "ChemTables Drug Discovery Ontology",
    "domain": "Medicinal Chemistry / Drug Discovery",
    "version": "1.0",
    "total_entries": len(ASSAY_TAXONOMY) + len(UNIT_STANDARDIZATION) + len(TARGET_TAXONOMY),
    "source_benchmark": "Schema-Driven IE (Bai et al., EMNLP 2024 Findings)",
    "ground_truth_format": "{value, type, target, treatment, unit}",
    "valid_types": sorted(VALID_ASSAY_TYPES),
}

if __name__ == "__main__":
    print(f"ChemTables Ontology Module")
    print(f"  Total entries: {ONTOLOGY_INFO['total_entries']}")
    print(f"  Assay types: {len(ASSAY_TAXONOMY)} mappings -> {sorted(VALID_ASSAY_TYPES)}")
    print(f"  Units: {len(UNIT_STANDARDIZATION)} mappings")
    print(f"  Targets: {len(TARGET_TAXONOMY)} mappings")
    print(f"  Null conventions: {CONSTRAINTS['null_conventions']}")
