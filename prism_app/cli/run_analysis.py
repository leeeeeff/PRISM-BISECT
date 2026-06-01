"""CLI: prism-app analyze — run PRISM analysis on a score matrix.

Usage:
    prism-app analyze \\
        --scores scores.npy \\
        --ids isoform_ids.npy \\
        --tissue muscle \\
        --output results/

    prism-app analyze \\
        --scores scores.npy --ids ids.npy --genes gene_ids.npy \\
        --types isoform_types.npy \\
        --dtu dtu_results.tsv \\
        --tissue brain --threshold 0.5 --output results/
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='prism-app analyze',
        description='Run PRISM isoform function analysis pipeline.',
    )
    p.add_argument('--scores',    required=True, help='Score matrix NPY (n_isoforms × n_GO)')
    p.add_argument('--ids',       required=True, help='Isoform ID list (NPY or TXT)')
    p.add_argument('--genes',     default=None,  help='Gene ID list (NPY or TXT, optional)')
    p.add_argument('--types',     default=None,  help='Isoform type labels (NPY or TXT, optional)')
    p.add_argument('--dtu',       default=None,  help='DTU results TSV/CSV (optional)')
    p.add_argument('--annotations', default=None,
                   help='GO annotation file for validation (optional)')
    p.add_argument('--tissue',    default='muscle',
                   choices=['muscle', 'brain', 'brain_extended', 'muscle_only'],
                   help='Tissue GO preset (default: muscle)')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='PRISM score threshold (default: 0.5)')
    p.add_argument('--output',    default='prism_results',
                   help='Output directory (default: prism_results/)')
    p.add_argument('--no-validation', action='store_true',
                   help='Skip AUPRC validation (faster)')
    return p.parse_args(argv)


def _load_array(path: str):
    import numpy as np
    p = Path(path)
    if p.suffix == '.npy':
        arr = np.load(p, allow_pickle=True)
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    lines = p.read_text().strip().splitlines()
    return np.array(lines)


def main(argv=None):
    import numpy as np
    import pandas as pd

    args = parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parents[2]))

    from prism_app.core.go_utils import TISSUE_PRESETS
    from prism_app.core.classifier import classify_isoforms, scenario_summary
    from prism_app.reports.coverage import generate_coverage_report
    from prism_app.reports.novel_summary import generate_novel_summary

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PRISM] Loading score matrix: {args.scores}")
    sm    = np.load(args.scores, allow_pickle=True).astype(np.float32)
    ids   = _load_array(args.ids)
    genes = _load_array(args.genes) if args.genes else None
    types = _load_array(args.types) if args.types else None

    go_preset = TISSUE_PRESETS[args.tissue]
    go_terms  = list(go_preset.keys())
    go_names  = go_preset

    print(f"[PRISM] Score matrix: {sm.shape}  GO terms: {len(go_terms)}  "
          f"Threshold: {args.threshold}")

    # DTU
    dtu_df = None
    if args.dtu:
        sep = '\t' if args.dtu.endswith('.tsv') else ','
        dtu_df = pd.read_csv(args.dtu, sep=sep)
        print(f"[PRISM] DTU results: {len(dtu_df)} records")

    # 1. Classification
    print("[PRISM] Running 4-Scenario classification…")
    classified = classify_isoforms(sm, ids, genes, go_terms,
                                    dtu_df=dtu_df, score_threshold=args.threshold)
    scenarios_path = out_dir / 'scenarios.tsv'
    classified.to_csv(scenarios_path, sep='\t', index=False)
    print(f"[PRISM]  → {scenarios_path}")

    summ = scenario_summary(classified)
    for _, row in summ.iterrows():
        print(f"        Scenario {row['scenario']}: {row['count']:,} isoforms ({row['pct']}%)")

    # 2. Coverage report
    print("[PRISM] Generating coverage report…")
    rep = generate_coverage_report(sm, ids, types, go_terms, go_names, args.threshold)
    coverage_data = {
        'total_isoforms': rep.total_isoforms,
        'n_known': rep.n_known, 'n_nic': rep.n_nic, 'n_nnic': rep.n_nnic,
        'n_with_any_high': rep.n_with_any_high,
        'pct_with_any_high': rep.pct_with_any_high,
        'score_mean': rep.score_mean, 'score_median': rep.score_median,
        'per_go': rep.per_go[:10],
    }
    coverage_path = out_dir / 'coverage.json'
    coverage_path.write_text(json.dumps(coverage_data, indent=2))
    print(f"[PRISM]  → {coverage_path}")

    # 3. Novel summary (if types available)
    if types is not None:
        print("[PRISM] Generating novel isoform summary…")
        novel_rep = generate_novel_summary(sm, ids, types, go_terms, go_names, args.threshold)
        novel_path = out_dir / 'novel_summary.tsv'
        novel_rep.to_dataframe().to_csv(novel_path, sep='\t', index=False)
        print(f"[PRISM]  → {novel_path}  "
              f"({novel_rep.total_novel} novel, "
              f"{novel_rep.n_novel_with_any_high} with predictions)")

    # 4. Validation (optional)
    if not args.no_validation and genes is not None:
        print("[PRISM] Running AUPRC validation…")
        from prism_app.reports.validation import generate_validation_report
        val_rep = generate_validation_report(
            sm, ids, go_terms, go_names,
            annotation_path=args.annotations,
            gene_ids=genes, n_bootstrap=200,
        )
        if val_rep:
            val_data = {
                'macro_auprc': val_rep.macro_auprc,
                'ci_95': list(val_rep.macro_auprc_ci),
                'n_annotated': val_rep.n_isoforms_with_annotation,
                'per_go': val_rep.per_go,
            }
            val_path = out_dir / 'validation.json'
            val_path.write_text(json.dumps(val_data, indent=2))
            print(f"[PRISM]  → {val_path}  "
                  f"Macro AUPRC: {val_rep.macro_auprc:.4f} "
                  f"CI: {val_rep.macro_auprc_ci}")
        else:
            print("[PRISM]  Validation skipped (no annotation overlap found)")

    print(f"[PRISM] Done. Results in: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
