#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
severity_numbers_faithful.py
==============================
Re-runs the severity regression with the manuscript-faithful definitions
(domain_binary_faithful from build_severity_domain_faithful.py; size RAW z-scored
per §338(i)) to test whether the manuscript's numbers are reproducible under the
original definitions -- and to supply the numbers for the §165/§332 update.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from severity_regression import ols_fit, cluster_robust_se, permutation_null_r2

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
COVS = ['size_z', 'domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']


def load(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    df = df[df['domain_binary_faithful'] >= 0].copy()   # drop unmapped (brain)
    df['domain_binary'] = df['domain_binary_faithful']   # faithful definition
    df['size_z'] = (df['size'] - df['size'].mean()) / df['size'].std()   # RAW z-score per manuscript
    return df


def design(df, with_tissue=False):
    cols = list(COVS)
    X = df[cols].to_numpy(float)
    if with_tissue:
        X = np.column_stack([X, (df['tissue'] == 'brain').astype(float).to_numpy()])
        cols = cols + ['tissue_brain']
    X = np.column_stack([np.ones(len(X)), X])
    return X, ['intercept'] + cols


def fit_report(label, df, with_tissue=False, perm=False):
    X, cols = design(df, with_tissue)
    y = df['severity_score'].to_numpy(float)
    beta, resid, r2 = ols_fit(X, y)
    se = cluster_robust_se(X, resid, df['gene'].to_numpy())
    print(f"\n=== {label}  N={len(df)} genes={df['gene'].nunique()} R2={r2:.4f} ===")
    for c, b, s in zip(cols, beta, se):
        print(f"    {c:<22} {b:+.3f}  (t={b/s:+.1f})")
    if perm:
        null = permutation_null_r2(X, y, n_perm=1000)
        print(f"    permutation null R2 mean={null.mean():.5f} P(>=obs)={(null>=r2).mean():.4f}")
    return dict(zip(cols, beta)), r2


def main():
    df_m, df_b = load('muscle'), load('brain')
    df = pd.concat([df_m, df_b], ignore_index=True)
    df['size_z'] = (df['size'] - df['size'].mean()) / df['size'].std()

    print("PAIR COUNTS:", f"muscle={len(df_m)} brain={len(df_b)} pooled={len(df)}")
    fit_report('POOLED (raw size, faithful domain)', df, with_tissue=True, perm=True)
    fit_report('MUSCLE', df_m)
    fit_report('BRAIN', df_b)

    # within-gene
    counts = df.groupby('gene').size()
    multi = counts[counts >= 2].index
    dm = df[df['gene'].isin(multi)].copy()
    Xs, _ = design(dm, with_tissue=True)
    _, _, r2sub = ols_fit(Xs, dm['severity_score'].to_numpy(float))
    for col in COVS + ['severity_score']:
        dm[col + '_w'] = dm[col] - dm.groupby('gene')[col].transform('mean')
    Xw = np.column_stack([np.ones(len(dm))] + [dm[c + '_w'].to_numpy() for c in COVS])
    _, _, r2w = ols_fit(Xw, dm['severity_score_w'].to_numpy())
    print(f"\n=== WITHIN-GENE: genes>=2pairs={len(multi)} pairs={len(dm)} "
          f"pooled-subsample R2={r2sub:.4f} within-gene R2={r2w:.4f} ===")

    # canonical_is_lo stratification (domain_binary caveat)
    df['canon_longer'] = (df['canonical_is_lo'] == 0).astype(int)
    print(f"\n=== canonical longer: {df['canon_longer'].sum()}/{len(df)} = "
          f"{100*df['canon_longer'].mean():.1f}% ===")
    for lab, sub in [('majority(canon longer)', df[df['canon_longer'] == 1]),
                     ('reverse(canon shorter)', df[df['canon_longer'] == 0])]:
        X, cols = design(sub, with_tissue=True)
        b, r, _ = ols_fit(X, sub['severity_score'].to_numpy(float))
        se = cluster_robust_se(X, r, sub['gene'].to_numpy())[cols.index('domain_binary')]
        db = dict(zip(cols, b))['domain_binary']
        print(f"  {lab}: n={len(sub)} domain_binary={db:+.3f} (t={db/se:+.1f})")


if __name__ == '__main__':
    main()
