"""
Retroactively apply M14 (Exon Comparator), M15 (NMD v2), M16 (Mechanism Classifier)
to all existing analysis.json files in the outputs directory.
Updates bisect_cases.json with m14_exon_comparator, m15_nmd_v2, m16_mechanism fields.

Usage:
    python scripts/apply_m14_m15_m16_retroactive.py \
        --outputs Final_analysis/pipeline_bioanalysis/outputs \
        --bisect-json prism_app/data/demo/bisect_cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'Final_analysis/pipeline_bioanalysis'))

from modules.m14_exon_comparator import run_m14
from modules.m15_nmd_v2 import run_m15
from modules.m16_mechanism_classifier import run_m16


def process_analysis_json(path: Path, config: dict) -> dict | None:
    try:
        with open(path) as f:
            case = json.load(f)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None

    # Run M14
    m14_out = run_m14(case, config)
    case.update(m14_out)

    # Run M15
    m15_out = run_m15(case, config)
    case.update(m15_out)

    # Run M16 (needs M14 + M15 results in case)
    m16_out = run_m16(case, config)
    case.update(m16_out)

    # Write back to analysis.json
    with open(path, 'w') as f:
        json.dump(case, f, indent=2, default=str)

    return {
        'gene':        case.get('gene_name') or case.get('gene') or '',
        'cell_type':   case.get('cell_type') or '',
        'event_type':  m14_out.get('m14_exon_comparator', {}).get('event_type', '?'),
        'nmd_switch':  m15_out.get('m15_nmd_v2', {}).get('nmd_switch', None),
        'mechanism':   m16_out.get('m16_mechanism_classifier', {}).get('primary_mechanism', '?'),
    }


def update_bisect_cases(bisect_path: Path, updates: dict[tuple, dict]) -> None:
    """Add m14_event_type and m16_mechanism to matching bisect_cases rows."""
    with open(bisect_path) as f:
        cases = json.load(f)

    updated = 0
    for c in cases:
        gene = str(c.get('gene') or c.get('gene_name') or '').strip()
        ct   = str(c.get('cell_type') or '').strip()
        key  = (gene, ct)
        if key in updates:
            u = updates[key]
            c['m14_event_type']  = u.get('event_type')
            c['m15_nmd_switch']  = u.get('nmd_switch')
            c['m16_mechanism']   = u.get('mechanism')
            updated += 1

    with open(bisect_path, 'w') as f:
        json.dump(cases, f, indent=2, default=str)

    print(f"  Updated {updated}/{len(cases)} bisect_cases rows with M14/M15/M16 results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs',     default='Final_analysis/pipeline_bioanalysis/outputs')
    parser.add_argument('--bisect-json', default='prism_app/data/demo/bisect_cases.json')
    args = parser.parse_args()

    outputs_dir  = ROOT / args.outputs
    bisect_path  = ROOT / args.bisect_json
    config       = {}

    analysis_files = list(outputs_dir.rglob('analysis.json'))
    print(f"Found {len(analysis_files)} analysis.json files in {outputs_dir}")

    updates: dict[tuple, dict] = {}
    success, fail = 0, 0

    for apath in sorted(analysis_files):
        result = process_analysis_json(apath, config)
        if result:
            gene = result['gene']
            ct   = result['cell_type']
            key  = (gene, ct)
            updates[key] = result
            print(f"  OK: {gene} / {ct} → event={result['event_type']}, "
                  f"nmd_switch={result['nmd_switch']}, mechanism={result['mechanism']}")
            success += 1
        else:
            fail += 1

    print(f"\nProcessed {success} OK, {fail} failed")

    if bisect_path.exists() and updates:
        update_bisect_cases(bisect_path, updates)

    print("Done.")


if __name__ == '__main__':
    main()
