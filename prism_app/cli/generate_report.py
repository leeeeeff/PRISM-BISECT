"""CLI: prism-app report — generate a case report for a specific isoform.

Usage:
    prism-app report \\
        --scores scores.npy --ids ids.npy \\
        --isoform NDUFS4-201 \\
        --tissue muscle

    prism-app report \\
        --scores scores.npy --ids ids.npy --genes gene_ids.npy \\
        --gene KIF21B --tissue brain --format html
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='prism-app report',
        description='Generate a PRISM case report for one isoform or gene.',
    )
    p.add_argument('--scores',   required=True, help='Score matrix NPY')
    p.add_argument('--ids',      required=True, help='Isoform IDs (NPY/TXT)')
    p.add_argument('--genes',    default=None,  help='Gene IDs (NPY/TXT, optional)')
    p.add_argument('--types',    default=None,  help='Isoform type labels (optional)')
    p.add_argument('--dtu',      default=None,  help='DTU results TSV/CSV (optional)')
    p.add_argument('--isoform',  default=None,  help='Specific isoform ID to report')
    p.add_argument('--gene',     default=None,  help='Gene symbol (report all isoforms)')
    p.add_argument('--tissue',   default='muscle',
                   choices=['muscle', 'brain', 'brain_extended', 'muscle_only'])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--format',   default='markdown', choices=['markdown', 'tsv'],
                   help='Output format (default: markdown)')
    p.add_argument('--output',   default=None,
                   help='Output file path (default: stdout or auto-named)')
    return p.parse_args(argv)


def _load_array(path: str):
    import numpy as np
    p = Path(path)
    if p.suffix == '.npy':
        arr = np.load(p, allow_pickle=True)
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return np.array(p.read_text().strip().splitlines())


def main(argv=None):
    import numpy as np
    import pandas as pd

    args = parse_args(argv)
    if not args.isoform and not args.gene:
        print("Error: specify --isoform or --gene", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parents[2]))

    from prism_app.core.go_utils import TISSUE_PRESETS
    from prism_app.core.classifier import classify_isoforms

    sm    = np.load(args.scores, allow_pickle=True).astype(np.float32)
    ids   = _load_array(args.ids)
    genes = _load_array(args.genes) if args.genes else None

    go_preset = TISSUE_PRESETS[args.tissue]
    go_terms  = list(go_preset.keys())
    go_names  = go_preset

    dtu_df = None
    if args.dtu:
        sep = '\t' if args.dtu.endswith('.tsv') else ','
        dtu_df = pd.read_csv(args.dtu, sep=sep)

    classified = classify_isoforms(sm, ids, genes, go_terms,
                                    dtu_df=dtu_df, score_threshold=args.threshold)

    # Filter to target isoform(s)
    ids_arr = np.asarray(ids, dtype=str)
    if args.isoform:
        hits = classified[classified['isoform_id'].str.contains(
            args.isoform, case=False, na=False)]
    else:
        if genes is not None:
            genes_arr = np.asarray(genes, dtype=str)
            hits = classified[
                classified['isoform_id'].str.contains(args.gene, case=False, na=False)
                | classified['gene_id'].str.contains(args.gene, case=False, na=False)
            ]
        else:
            hits = classified[classified['isoform_id'].str.contains(
                args.gene, case=False, na=False)]

    if hits.empty:
        query = args.isoform or args.gene
        print(f"No isoforms matching '{query}' found.", file=sys.stderr)
        sys.exit(1)

    print(f"[PRISM] Found {len(hits)} matching isoform(s)")

    if args.format == 'tsv':
        out = hits.to_csv(sep='\t', index=False)
    else:
        out = _build_markdown_report(hits, sm, ids_arr, go_terms, go_names, args.threshold)

    if args.output:
        Path(args.output).write_text(out)
        print(f"[PRISM] Report written to: {args.output}")
    else:
        print(out)


def _build_markdown_report(hits, sm, ids_arr, go_terms, go_names, thr) -> str:
    import numpy as np
    lines = ["# PRISM Analysis Report\n"]
    for _, row in hits.iterrows():
        idx_match = np.where(ids_arr == row['isoform_id'])[0]
        idx = idx_match[0] if len(idx_match) else None

        lines.append(f"## {row['isoform_id']}")
        lines.append(f"- **Gene**: {row.get('gene_id', 'N/A')}")
        lines.append(f"- **Scenario**: {row['scenario']} — {row['scenario_label']}")
        lines.append(f"- **Max score**: {row['max_score']:.4f} on {row['max_go']} "
                     f"({go_names.get(row['max_go'], '')})")
        lines.append(f"- **DTU flag**: {row['dtu_flag']}")
        if row.get('dtu_pvalue') is not None:
            lines.append(f"- **DTU p-value**: {row['dtu_pvalue']:.2e}")
        lines.append("")

        if idx is not None:
            scores = sm[idx]
            high = [(go_terms[i], go_names.get(go_terms[i], go_terms[i]), float(scores[i]))
                    for i in range(len(go_terms)) if scores[i] > thr]
            if high:
                lines.append(f"### High-confidence GO predictions (score > {thr})")
                lines.append("| GO ID | Function | Score |")
                lines.append("|-------|----------|-------|")
                for go_id, go_name, score in sorted(high, key=lambda x: -x[2]):
                    lines.append(f"| {go_id} | {go_name[:50]} | {score:.4f} |")
                lines.append("")
            else:
                lines.append(f"*No GO predictions above threshold ({thr})*\n")

        lines.append("---\n")

    lines.append("*Generated by PRISM v0.1.0 · Lee et al. (2026)*")
    return "\n".join(lines)


if __name__ == '__main__':
    main()
