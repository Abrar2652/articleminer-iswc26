#!/usr/bin/env python3
"""
Audit ontology coverage of gold annotations.

For each domain, compute what fraction of gold values fall into the
ontology's canonical taxonomy — answers the reviewer concern that
"the ontology is cherry-picked to gold."

Metric for each field:
    coverage = (# gold values mapped to a canonical term) /
               (# gold values with that field populated)

Reported per domain × per field, plus an overall coverage number.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/schema_driven_ie/data"


def _norm(s): return str(s or "").strip().lower().replace("  ", " ")


def audit_chemtables():
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from ontology_chemtables import (
        ASSAY_TAXONOMY, UNIT_STANDARDIZATION, TARGET_TAXONOMY, VALID_ASSAY_TYPES,
    )
    with open(DATA / "chemtables/test.json") as f:
        data = json.load(f)

    type_tot = unit_tot = tgt_tot = 0
    type_hit = unit_hit = tgt_hit = 0
    for _, entry in data.items():
        for g in entry.get("cell_list_gold", []):
            t = g.get("type")
            if t:
                type_tot += 1
                if t in VALID_ASSAY_TYPES or _norm(t) in ASSAY_TAXONOMY:
                    type_hit += 1
            u = g.get("unit")
            if u:
                unit_tot += 1
                if u in UNIT_STANDARDIZATION.values() or _norm(u) in UNIT_STANDARDIZATION:
                    unit_hit += 1
            tg = g.get("target")
            if tg and tg != "xx":
                tgt_tot += 1
                if tg in TARGET_TAXONOMY.values() or _norm(tg) in TARGET_TAXONOMY:
                    tgt_hit += 1

    return {
        "domain": "chemtables",
        "type":   (type_hit, type_tot, round(100*type_hit/max(type_tot,1), 1)),
        "unit":   (unit_hit, unit_tot, round(100*unit_hit/max(unit_tot,1), 1)),
        "target": (tgt_hit, tgt_tot, round(100*tgt_hit/max(tgt_tot,1), 1)),
    }


def audit_discomat():
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from ontology_discomat import COMPONENT_TAXONOMY, VALID_UNITS
    with open(DATA / "discomat/test.json") as f:
        data = json.load(f)

    comp_tot = unit_tot = 0
    comp_hit = unit_hit = 0
    for _, entry in data.items():
        for g in entry.get("cell_list_gold", []):
            # gold is a list [sample_id, component, value, unit]
            if isinstance(g, list) and len(g) >= 4:
                comp = str(g[1]); unit = str(g[3])
                if comp:
                    comp_tot += 1
                    if comp in COMPONENT_TAXONOMY.values() or _norm(comp) in COMPONENT_TAXONOMY:
                        comp_hit += 1
                if unit:
                    unit_tot += 1
                    if unit in VALID_UNITS:
                        unit_hit += 1
    return {
        "domain": "discomat",
        "component": (comp_hit, comp_tot, round(100*comp_hit/max(comp_tot,1), 1)),
        "unit":      (unit_hit, unit_tot, round(100*unit_hit/max(unit_tot,1), 1)),
    }


def audit_mltables():
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from ontology_mltables import (
        VALID_CELL_TYPES, METRIC_TAXONOMY, TASK_TAXONOMY,
        ATTR_TAXONOMY, PARAM_TAXONOMY,
    )
    with open(DATA / "mltables/test.json") as f:
        data = json.load(f)

    type_tot = metr_tot = task_tot = attr_tot = param_tot = 0
    type_hit = metr_hit = task_hit = attr_hit = param_hit = 0
    for _, entry in data.items():
        for g in entry.get("cell_list_gold", []):
            if not isinstance(g, dict): continue
            t = g.get("type")
            if t:
                type_tot += 1
                if t in VALID_CELL_TYPES:
                    type_hit += 1
            m = g.get("metric")
            if m:
                m_str = m if isinstance(m, str) else (m[0] if isinstance(m, list) and m else "")
                if m_str:
                    metr_tot += 1
                    if m_str in METRIC_TAXONOMY.values() or _norm(m_str) in METRIC_TAXONOMY:
                        metr_hit += 1
            tk = g.get("task")
            if tk:
                tk_str = tk if isinstance(tk, str) else (tk[0] if isinstance(tk, list) and tk else "")
                if tk_str:
                    task_tot += 1
                    if tk_str in TASK_TAXONOMY.values() or _norm(tk_str) in TASK_TAXONOMY:
                        task_hit += 1
            an = g.get("attribute name")
            if an:
                attr_tot += 1
                if an in ATTR_TAXONOMY.values() or _norm(an) in ATTR_TAXONOMY:
                    attr_hit += 1
            pn = g.get("parameter/architecture name")
            if pn:
                param_tot += 1
                if pn in PARAM_TAXONOMY.values() or _norm(pn) in PARAM_TAXONOMY:
                    param_hit += 1

    return {
        "domain": "mltables",
        "type":       (type_hit, type_tot, round(100*type_hit/max(type_tot,1), 1)),
        "metric":     (metr_hit, metr_tot, round(100*metr_hit/max(metr_tot,1), 1)),
        "task":       (task_hit, task_tot, round(100*task_hit/max(task_tot,1), 1)),
        "attribute":  (attr_hit, attr_tot, round(100*attr_hit/max(attr_tot,1), 1)),
        "parameter":  (param_hit, param_tot, round(100*param_hit/max(param_tot,1), 1)),
    }


def main():
    results = [audit_chemtables(), audit_discomat(), audit_mltables()]
    print(f"\n{'='*76}")
    print(f"{'Ontology coverage of gold annotations':^76}")
    print(f"{'='*76}")
    for r in results:
        print(f"\n[{r['domain']}]")
        for k, v in r.items():
            if k == "domain": continue
            hit, tot, pct = v
            print(f"  {k:<12}: {hit:>6} / {tot:<6}  = {pct:>5.1f}%")
    print(f"\nInterpretation:")
    print("  High pct ⇒ ontology canonical terms cover most gold values "
          "(expected for well-defined domains).")
    print("  Low pct in free-text fields (e.g., model names, dataset names)")
    print("  is expected — those fall back to surface-string matching.")
    out = Path(__file__).parent.parent / "results" / "ontology_coverage.json"
    json.dump(results, open(out, "w"), indent=2, default=list)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
