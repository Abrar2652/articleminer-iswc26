#!/usr/bin/env python3
"""v23b: Run a single ablation variant (parallelizes the v23 sequential loop).

Usage: python3 _v23b_run_one_variant.py <variant_name>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import the v23 module by filename (leading underscore is not importable normally)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v23", ROOT / "_v23_geochem_abl_haiku28.py"
)
v23 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v23)


def main():
    if len(sys.argv) < 2:
        print("usage: _v23b_run_one_variant.py <variant_name>")
        print(f"valid variants: {list(v23.ABLATIONS)}")
        sys.exit(1)

    variant = sys.argv[1]
    if variant not in v23.ABLATIONS:
        print(f"unknown variant {variant!r}; valid: {list(v23.ABLATIONS)}")
        sys.exit(1)

    papers = v23.list_gt_papers()
    papers = [p for p in papers if v23.find_pdf_for_paper(p.stem) is not None]
    print(f"variant={variant}  n_papers={len(papers)}")
    v23.run_one_ablation(variant, papers)


if __name__ == "__main__":
    main()
