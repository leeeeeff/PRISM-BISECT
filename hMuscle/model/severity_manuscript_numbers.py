#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
severity_manuscript_numbers.py
================================
Emits every number cited by natcomm_v0.md sections 165/332 for the
canonical-anchored severity regression, computed from the committed
reconstruction (build_severity_pairs.py + build_severity_score.py), so the
manuscript can be updated to the reproducible source of truth (user decision
2026-07-20). Reuses severity_regression.py's fit machinery.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from severity_regression import (load, ols_fit, cluster_robust_se, design_matrix,
                                  permutation_null_r2)

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')


def main():
    df_m = load('muscle')
    df_b = load('brain')
    df = pd.concat([df_m, df_b], ignore_index=True)
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()

    print("=== PAIR / GENE COUNTS ===")
    for name, d in [('muscle', df_m), ('brain', df_b), ('pooled', df)]:
        print(f"  {name}: pairs={len(d)}  genes(nunique)={d['gene'].nunique()}")
    # two-isoform instrument reference (unchanged, from old analysis): 1683 muscle / 2179 brain
    print(f"  multiplier vs 2-iso instrument: muscle {len(df_m)/1683:.1f}x  brain {len(df_b)/2179:.1f}x")

    print("\n=== POOLED FIT (with tissue_brain) ===")
    X, cols = design_matrix(df, with_tissue=True)
    y = df['severity_score'].to_numpy(float)
    beta, resid, r2 = ols_fit(X, y)
    print(f"  N={len(df)} genes={df['gene'].nunique()} R2={r2:.4f}")
    for c, b in zip(cols, beta):
        print(f"    {c:<22} {b:+.3f}")
    null = permutation_null_r2(X, y, n_perm=1000)
    print(f"  permutation null R2 mean={null.mean():.5f} max={null.max():.5f} P(null>=obs)={(null>=r2).mean():.4f}")

    print("\n=== PER-TISSUE ===")
    for name, d in [('muscle', df_m), ('brain', df_b)]:
        Xt, colst = design_matrix(d, with_tissue=False)
        yt = d['severity_score'].to_numpy(float)
        bt, rt, r2t = ols_fit(Xt, yt)
        print(f"  {name}: N={len(d)} genes={d['gene'].nunique()} R2={r2t:.4f}")
        for c, b in zip(colst, bt):
            print(f"    {c:<22} {b:+.3f}")

    print("\n=== WITHIN-GENE FIXED-EFFECTS REFIT (demean by gene) ===")
    covs = ['size_z', 'domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']
    # restrict to genes with >=2 pairs
    counts = df.groupby('gene').size()
    multi = counts[counts >= 2].index
    dm = df[df['gene'].isin(multi)].copy()
    print(f"  genes with >=2 pairs: {len(multi)}  pairs used: {len(dm)}")
    # pooled fit on identical subsample
    Xs, _ = design_matrix(dm, with_tissue=True)
    ys = dm['severity_score'].to_numpy(float)
    _, _, r2_sub = ols_fit(Xs, ys)
    print(f"  pooled R2 on same subsample: {r2_sub:.4f}")
    # within-gene: demean y and covariates by gene
    dmw = dm.copy()
    for col in covs + ['severity_score']:
        dmw[col + '_w'] = dmw[col] - dmw.groupby('gene')[col].transform('mean')
    Xw = np.column_stack([dmw[c + '_w'].to_numpy() for c in covs])  # no intercept (demeaned)
    yw = dmw['severity_score_w'].to_numpy()
    bw, _, r2w = ols_fit(np.column_stack([np.ones(len(Xw)), Xw]), yw)
    print(f"  within-gene R2: {r2w:.4f}")
    for c, b in zip(['intercept'] + covs, bw):
        print(f"    {c:<22} {b:+.3f}")

    print("\n=== canonical_is_lo STRATIFICATION (domain_binary caveat) ===")
    # canonical_is_lo=1 means canonical is the SHORTER; manuscript 'canonical longer' = is_lo==0
    df['canon_longer'] = (df['canonical_is_lo'] == 0).astype(int)
    n_longer = df['canon_longer'].sum()
    print(f"  canonical is longer: {n_longer}/{len(df)} = {100*n_longer/len(df):.1f}%")
    for lab, sub in [('canon_longer (majority)', df[df['canon_longer'] == 1]),
                     ('canon_shorter (reverse)', df[df['canon_longer'] == 0])]:
        Xz, colz = design_matrix(sub, with_tissue=True)
        yz = sub['severity_score'].to_numpy(float)
        bz, rz, r2z = ols_fit(Xz, yz)
        db = dict(zip(colz, bz))['domain_binary']
        se = cluster_robust_se(Xz, rz, sub['gene'].to_numpy())[colz.index('domain_binary')]
        print(f"  {lab}: n={len(sub)} domain_binary beta={db:+.3f} (SE={se:.3f}, t={db/se:+.1f})")


if __name__ == '__main__':
    main()
