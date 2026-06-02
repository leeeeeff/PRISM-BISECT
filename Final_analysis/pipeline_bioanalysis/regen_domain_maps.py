#!/usr/bin/env python3
"""Regenerate domain_map.png for all 84 BISECT PASS cases.

Run from repo root:
  conda run -n isoform_env python Final_analysis/pipeline_bioanalysis/regen_domain_maps.py
"""
import json, os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Final_analysis.pipeline_bioanalysis.modules.m14_report import plot_domain_map

CASES_JSON = 'prism_app/data/demo/bisect_cases.json'
OUTPUTS_BASE = 'Final_analysis/pipeline_bioanalysis/outputs'

DEFAULT_CONFIG = {
    'domain_colors': {},
    'repeat_colors': {},
    'width_per_100aa': 3.0,
    'min_width': 8,
    'max_width': 20,
}

def main():
    with open(CASES_JSON) as f:
        cases = json.load(f)

    ok, fail = 0, 0
    for case in cases:
        gene = case['gene']
        ct   = case['cell_type']
        d    = f"{gene}_{ct}"
        out_dir = os.path.join(OUTPUTS_BASE, d)
        ana_fp  = os.path.join(out_dir, 'analysis.json')

        if not os.path.exists(ana_fp):
            print(f"[SKIP]  {d} — analysis.json not found")
            fail += 1
            continue

        with open(ana_fp) as f:
            case_result = json.load(f)

        try:
            png_path = plot_domain_map(case_result, out_dir, DEFAULT_CONFIG)
            print(f"[OK]    {d} → {png_path}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {d}: {e}")
            traceback.print_exc()
            fail += 1

    print(f"\nDone: {ok} OK, {fail} failed")

if __name__ == '__main__':
    main()
