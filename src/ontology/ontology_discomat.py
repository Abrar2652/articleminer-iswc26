"""
DiSCoMaT Ontology Module
Domain: Materials Science (Glass / Ceramic Compositions)
Task: Extract composition tuples from materials science tables

Ground truth format: [sample_id, component, percentage, unit]
where unit in {mol, wt} (mol% or wt%)
"""

# =============================================================================
# Oxide Component Taxonomy
# Maps raw component strings (from table headers/cells) to canonical forms.
# Covers ~130 components observed in the DiSCoMaT dataset.
# =============================================================================
COMPONENT_TAXONOMY = {
    # Major glass-forming oxides
    "sio2": "SiO2", "silicon dioxide": "SiO2", "silica": "SiO2",
    "b2o3": "B2O3", "boron oxide": "B2O3", "boric oxide": "B2O3",
    "p2o5": "P2O5", "phosphorus pentoxide": "P2O5",
    "geo2": "GeO2", "germanium dioxide": "GeO2",

    # Alkali oxides
    "na2o": "Na2O", "sodium oxide": "Na2O", "soda": "Na2O",
    "k2o": "K2O", "potassium oxide": "K2O", "potash": "K2O",
    "li2o": "Li2O", "lithium oxide": "Li2O",
    "rb2o": "Rb2O", "cs2o": "Cs2O",

    # Alkaline earth oxides
    "cao": "CaO", "calcium oxide": "CaO", "lime": "CaO",
    "mgo": "MgO", "magnesium oxide": "MgO", "magnesia": "MgO",
    "bao": "BaO", "barium oxide": "BaO",
    "sro": "SrO", "strontium oxide": "SrO",
    "zno": "ZnO", "zinc oxide": "ZnO",

    # Transition metal oxides
    "al2o3": "Al2O3", "aluminium oxide": "Al2O3", "alumina": "Al2O3",
    "aluminum oxide": "Al2O3",
    "fe2o3": "Fe2O3", "iron(iii) oxide": "Fe2O3", "ferric oxide": "Fe2O3",
    "feo": "FeO", "iron(ii) oxide": "FeO", "ferrous oxide": "FeO",
    "tio2": "TiO2", "titanium dioxide": "TiO2", "titania": "TiO2",
    "zro2": "ZrO2", "zirconia": "ZrO2", "zirconium dioxide": "ZrO2",
    "mno": "MnO", "manganese oxide": "MnO",
    "mno2": "MnO2", "manganese dioxide": "MnO2",
    "cuo": "CuO", "copper oxide": "CuO",
    "cr2o3": "Cr2O3", "chromium oxide": "Cr2O3",
    "nio": "NiO", "nickel oxide": "NiO",
    "coo": "CoO", "cobalt oxide": "CoO",
    "v2o5": "V2O5", "vanadium pentoxide": "V2O5",
    "wo3": "WO3", "tungsten trioxide": "WO3",
    "moo3": "MoO3", "molybdenum trioxide": "MoO3",
    "nb2o5": "Nb2O5", "niobium pentoxide": "Nb2O5",
    "ta2o5": "Ta2O5", "tantalum pentoxide": "Ta2O5",
    "y2o3": "Y2O3", "yttrium oxide": "Y2O3", "yttria": "Y2O3",
    "la2o3": "La2O3", "lanthanum oxide": "La2O3",
    "ceo2": "CeO2", "cerium dioxide": "CeO2", "ceria": "CeO2",
    "nd2o3": "Nd2O3", "neodymium oxide": "Nd2O3",
    "gd2o3": "Gd2O3", "gadolinium oxide": "Gd2O3",
    "er2o3": "Er2O3", "erbium oxide": "Er2O3",
    "eu2o3": "Eu2O3", "europium oxide": "Eu2O3",
    "sm2o3": "Sm2O3", "samarium oxide": "Sm2O3",
    "dy2o3": "Dy2O3", "dysprosium oxide": "Dy2O3",
    "ho2o3": "Ho2O3", "holmium oxide": "Ho2O3",
    "pr6o11": "Pr6O11", "praseodymium oxide": "Pr6O11",
    "tb4o7": "Tb4O7", "terbium oxide": "Tb4O7",
    "yb2o3": "Yb2O3", "ytterbium oxide": "Yb2O3",

    # Lead oxides
    "pbo": "PbO", "lead oxide": "PbO", "lead(ii) oxide": "PbO",
    "pbo2": "PbO2", "lead dioxide": "PbO2",
    "pb3o4": "Pb3O4",

    # Bismuth / Antimony / Tellurium
    "bi2o3": "Bi2O3", "bismuth oxide": "Bi2O3",
    "sb2o3": "Sb2O3", "antimony oxide": "Sb2O3",
    "teo2": "TeO2", "tellurium dioxide": "TeO2",
    "as2o3": "As2O3", "arsenic trioxide": "As2O3",

    # Fluorides (also appear in glass compositions)
    "naf": "NaF", "kf": "KF", "lif": "LiF",
    "caf2": "CaF2", "baf2": "BaF2", "mgf2": "MgF2",
    "pbf2": "PbF2", "alf3": "AlF3", "zrf4": "ZrF4",

    # Sulfides / Selenides / other chalcogenides
    "as2s3": "As2S3", "as2se3": "As2Se3",
    "ge": "Ge", "se": "Se", "te": "Te", "sb": "Sb",
    "ag": "Ag", "au": "Au", "cu": "Cu",

    # Special entries
    "sno2": "SnO2", "tin dioxide": "SnO2",
    "in2o3": "In2O3", "indium oxide": "In2O3",
    "ga2o3": "Ga2O3", "gallium oxide": "Ga2O3",

    # Half-notation (AlO1.5, etc.)
    "alo1.5": "AlO1.5", "bo1.5": "BO1.5",

    # Pure elements (appear as components in some compositions)
    "o": "O", "p": "P", "si": "Si", "la": "La", "cl": "Cl", "f": "F",
    "b": "B", "ni": "Ni", "ti": "Ti", "al": "Al", "ca": "Ca", "zr": "Zr",
    "ba": "Ba", "hf": "Hf", "fe": "Fe", "sr": "Sr", "n": "N", "pd": "Pd",
    "na": "Na", "zn": "Zn", "y": "Y", "co": "Co", "s": "S", "nd": "Nd",
    "li": "Li", "v": "V", "ga": "Ga", "eu": "Eu", "cd": "Cd", "nb": "Nb",
    "u": "U", "cs": "Cs", "cr": "Cr", "c": "C", "h": "H", "pt": "Pt",
    "er": "Er", "k": "K", "pb": "Pb", "br": "Br",

    # Halide compounds
    "pbbr2": "PbBr2", "pbcl2": "PbCl2", "srf2": "SrF2", "znf2": "ZnF2",
    "yf3": "YF3", "cef3": "CeF3", "cdf2": "CdF2", "erf3": "ErF3",

    # Additional oxides and compounds from gold annotations
    "ag2o": "Ag2O", "cdo": "CdO", "sno": "SnO", "so3": "SO3",
    "mn2o3": "Mn2O3", "nao2": "NaO2", "ce2o3": "Ce2O3", "tm2o3": "Tm2O3",
    "nbo2.5": "NbO2.5", "si3n4": "Si3N4",

    # Complex compounds (phosphates, carbonates, etc.)
    "cahpo4": "CaHPO4", "nah2po4": "NaH2PO4", "napo3": "NaPO3",
    "sr(po3)2": "Sr(PO3)2", "srco3": "SrCO3", "li2co3": "Li2CO3",
    "linbo3": "LiNbO3", "h3bo3": "H3BO3",
    "fepo4 *2h2o": "FePO4 *2H2O", "mghpo4 *3h2o": "MgHPO4 *3H2O",

    # Spinel/complex phases
    "znal2o4": "ZnAl2O4", "amorphous": "Amorphous",
    "oh": "OH",
}

# Canonical component list (unique values)
VALID_COMPONENTS = sorted(set(COMPONENT_TAXONOMY.values()))

# =============================================================================
# Unit Handling
# DiSCoMaT uses only two units: mol% and wt%
# =============================================================================
UNIT_STANDARDIZATION = {
    "mol%": "mol", "mol.%": "mol", "mol %": "mol", "mole%": "mol",
    "mole %": "mol", "molar%": "mol", "molar %": "mol", "mol": "mol",
    "wt%": "wt", "wt.%": "wt", "wt %": "wt", "weight%": "wt",
    "weight %": "wt", "mass%": "wt", "mass %": "wt", "wt": "wt",
    "%": None,  # ambiguous — need context (caption) to resolve
}

VALID_UNITS = {"mol", "wt"}

# =============================================================================
# Material Classification
# High-level material categories for context understanding
# =============================================================================
MATERIAL_TAXONOMY = {
    "glass": "glass",
    "ceramic": "ceramic",
    "glass-ceramic": "glass-ceramic",
    "enamel": "enamel",
    "glaze": "glaze",
    "frit": "frit",
    "slag": "slag",
    "optical fiber": "optical fiber",
    "thin film": "thin film",
    "phosphor": "phosphor",
    "bioactive glass": "bioactive glass",
    "bioglass": "bioactive glass",
}

# =============================================================================
# Semantic Constraints
# =============================================================================
CONSTRAINTS = {
    # Null semantics for materials science
    # "-" typically means "not present in this composition" (0 mol%)
    # blank = "not reported for this sample"
    "null_conventions": {
        "zero_component": ["-", "–", "—", "0", "0.0", "nil"],
        "not_reported": ["", "n.r.", "n/a"],
    },

    # Composition sum validation
    # For a well-defined composition, components should sum to ~100%
    # Tolerance accounts for rounding and trace impurities
    "composition_sum": {
        "target": 100.0,
        "tolerance": 5.0,  # allow 95-105%
    },

    # Value plausibility per component type
    # Major components: 0-100%, minor: 0-30%, trace: 0-5%
    "plausibility": {
        "major_oxides": {"max_pct": 100.0},  # SiO2, B2O3, P2O5
        "modifier_oxides": {"max_pct": 70.0},  # Na2O, CaO, etc.
        "dopants": {"max_pct": 15.0},  # rare earths, transition metals
    },

    # The "-" value in DiSCoMaT tables means 0% (component absent)
    # This is different from geochem where "-" means BDL
    "dash_means_zero": True,
}


def standardize_component(raw: str) -> str | None:
    """Normalize a component string to canonical chemical formula."""
    key = raw.strip().lower().replace(" ", "")
    # Try direct lookup
    if key in COMPONENT_TAXONOMY:
        return COMPONENT_TAXONOMY[key]
    # Try removing trailing (mol%) or (wt%) annotation
    import re
    cleaned = re.sub(r'\(.*?\)', '', key).strip()
    if cleaned in COMPONENT_TAXONOMY:
        return COMPONENT_TAXONOMY[cleaned]
    return None


def infer_unit_from_caption(caption: str) -> str | None:
    """Infer unit (mol or wt) from a table caption string."""
    caption_lower = caption.lower()
    if "mol%" in caption_lower or "mol %" in caption_lower or "mole" in caption_lower:
        return "mol"
    if "wt%" in caption_lower or "wt %" in caption_lower or "weight" in caption_lower or "mass%" in caption_lower:
        return "wt"
    return None


def infer_unit_from_header(header: str) -> str | None:
    """Infer unit from a column header like 'PbO (mol%)'."""
    header_lower = header.lower()
    if "mol" in header_lower:
        return "mol"
    if "wt" in header_lower or "weight" in header_lower:
        return "wt"
    return None


def is_null_value(raw_value: str) -> str | None:
    """
    Check if a raw cell value represents a null.
    Returns: 'zero_component', 'not_reported', or None (real value).
    """
    v = raw_value.strip()
    if v in CONSTRAINTS["null_conventions"]["zero_component"]:
        return "zero_component"
    if v in CONSTRAINTS["null_conventions"]["not_reported"]:
        return "not_reported"
    return None


def validate_composition_sum(components: list[tuple[str, float]],
                              tolerance: float = 5.0) -> bool:
    """Check if component percentages sum to approximately 100%."""
    total = sum(val for _, val in components)
    target = CONSTRAINTS["composition_sum"]["target"]
    return abs(total - target) <= tolerance


# =============================================================================
# Ontology metadata
# =============================================================================
ONTOLOGY_INFO = {
    "name": "DiSCoMaT Materials Science Ontology",
    "domain": "Materials Science (Glass/Ceramic Compositions)",
    "version": "1.0",
    "total_entries": len(COMPONENT_TAXONOMY) + len(UNIT_STANDARDIZATION) + len(MATERIAL_TAXONOMY),
    "source_benchmark": "DiSCoMaT (Gupta et al., ACL 2023)",
    "ground_truth_format": "[sample_id, component, percentage, unit]",
    "valid_units": sorted(VALID_UNITS),
    "num_components": len(VALID_COMPONENTS),
}

if __name__ == "__main__":
    print(f"DiSCoMaT Ontology Module")
    print(f"  Total entries: {ONTOLOGY_INFO['total_entries']}")
    print(f"  Components: {len(COMPONENT_TAXONOMY)} mappings -> {len(VALID_COMPONENTS)} unique")
    print(f"  Units: {sorted(VALID_UNITS)}")
    print(f"  Materials: {len(MATERIAL_TAXONOMY)} types")
    print(f"  Sample components: {VALID_COMPONENTS[:15]}...")
