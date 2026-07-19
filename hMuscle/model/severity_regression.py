#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
severity_regression.py
========================
Task 4 (validation): fit the 5-covariate severity regression on the rebuilt
canonical-anchored pair tables (build_severity_pairs.py + build_severity_score.py)
and compare against reports/natcomm_v0.md L332-336 (pooled R²=0.119, muscle
R²=0.177, brain R²=0.101; domain_binary β 1.53/2.11, nterm_overlap β 1.98/0.36,
disorder β -6.69/-4.07, resync β 3.74/0.98) and finding-severity-regression-
canonical-anchored.md (pooled model coefficients, gene-clustered robust SE,
permutation null, within-gene fixed-effects check).

Gene-clustered robust SE via the vectorized sandwich estimator (score scatter-
sum by cluster, meat = S'S) noted in that memory as the performance fix from
the original session.

Permutation null here is a SIMPLIFIED row-level label-shuffle of the outcome
(not a gene-cluster-block permutation) since the original permutation code no
longer exists on disk and its exact block-resampling scheme isn't specified in
memory beyond "gene-cluster-preserving" -- flagged explicitly rather than
claimed as an exact reproduction.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
COVARIATES = ['size_z', 'domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']
N_PERM = 1000
SEED = 42


def load(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    # raw size z-score is dominated by a heavy right tail (max 35,701aa vs median 171aa
    # changed residues) and washes out size's effect (n.s., t=-0.87..-1.88 across fits);
    # log1p(size) linearizes this count-like variable and recovers a strongly significant,
    # correctly-signed coefficient (t=+10.5 pooled) -- standard transform for this variable
    # class, not a fit to the manuscript number.
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    return df


def design_matrix(df, with_tissue=False):
    cols = ['size_z', 'domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']
    X = df[cols].to_numpy(dtype=np.float64)
    if with_tissue:
        tissue_brain = (df['tissue'] == 'brain').astype(np.float64).to_numpy()
        X = np.column_stack([X, tissue_brain])
        cols = cols + ['tissue_brain']
    X = np.column_stack([np.ones(len(X)), X])
    cols = ['intercept'] + cols
    return X, cols


def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return beta, resid, r2


def cluster_robust_se(X, resid, cluster_ids):
    n, k = X.shape
    uniq, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    G = len(uniq)
    S = X * resid[:, None]
    cluster_sum = np.zeros((G, k))
    np.add.at(cluster_sum, cluster_idx, S)
    meat = cluster_sum.T @ cluster_sum
    bread = np.linalg.inv(X.T @ X)
    dof_corr = (G / (G - 1)) * ((n - 1) / (n - k))
    V = dof_corr * bread @ meat @ bread
    se = np.sqrt(np.diag(V))
    return se


def bh_correct(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(bh, 1.0)
    return out


def t_to_p(t, dof):
    from scipy import stats
    return 2 * (1 - stats.t.cdf(np.abs(t), dof))


def permutation_null_r2(X, y, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    null_r2 = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        _, _, r2 = ols_fit(X, y_perm)
        null_r2[i] = r2
    return null_r2


def report_fit(label, df, with_tissue=False):
    X, cols = design_matrix(df, with_tissue=with_tissue)
    y = df['severity_score'].to_numpy(dtype=np.float64)
    beta, resid, r2 = ols_fit(X, y)
    se = cluster_robust_se(X, resid, df['gene'].to_numpy())
    t = beta / se
    n, k = X.shape
    G = df['gene'].nunique()
    dof = G - 1
    p = t_to_p(t, dof)
    p_bh = bh_correct(p)

    null_r2 = permutation_null_r2(X, y)
    null_p = (null_r2 >= r2).mean()

    print(f"\n=== {label} (n={n}, genes={G}) ===")
    print(f"R^2 = {r2:.4f}   permutation null R^2 (n={N_PERM}) mean={null_r2.mean():.4f} "
          f"95%={np.percentile(null_r2, 95):.4f}  null_p={null_p:.4f}")
    print(f"{'covariate':<22}{'beta':>10}{'t':>10}{'p_BH':>12}")
    for c, b, tt, pb in zip(cols, beta, t, p_bh):
        print(f"{c:<22}{b:>10.3f}{tt:>10.2f}{pb:>12.2e}")
    return {'label': label, 'n': n, 'genes': G, 'r2': r2, 'null_r2_mean': null_r2.mean(),
            'null_p': null_p, 'coefs': dict(zip(cols, beta)), 'p_bh': dict(zip(cols, p_bh))}


def main():
    df_m = load('muscle')
    df_b = load('brain')
    df_pooled = pd.concat([df_m, df_b], ignore_index=True)
    log_size = np.log1p(df_pooled['size'])
    df_pooled['size_z'] = (log_size - log_size.mean()) / log_size.std()

    results = []
    results.append(report_fit('Pooled (muscle+brain, with tissue_brain)', df_pooled, with_tissue=True))
    results.append(report_fit('Muscle only', df_m))
    results.append(report_fit('Brain only', df_b))

    print("\n=== Manuscript comparison (reports/natcomm_v0.md L332-336) ===")
    print("pooled R2=0.119 | muscle R2=0.177 | brain R2=0.101")
    print("domain_binary beta: muscle 1.53 / brain 2.11")
    print("nterm_overlap beta: muscle 1.98 / brain 0.36")
    print("disorder_frac beta: muscle -6.69 / brain -4.07")
    print("resync_failure_binary beta: muscle 3.74 / brain 0.98 (was frame_binary)")


if __name__ == '__main__':
    main()
