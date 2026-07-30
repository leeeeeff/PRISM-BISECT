#!/usr/bin/env python3
"""
disorder_severity_metric.py
=============================
Elevate disorder_frac (continuous, VIF=1.00, essentially orthogonal to the
other 4 covariates) to a standalone citable metric, matching the evidentiary
grammar used for the 3 binary covariates (score + gene-cluster bootstrap CI +
permutation null), via two complementary framings:

  (1) Partial correlation: disorder_frac vs severity_score, with size_z,
      domain_binary, nterm_overlap, resync_failure_binary regressed out of
      BOTH variables first (residual-on-residual correlation). Null =
      permutation of disorder_frac across pairs (breaks pair-level
      correspondence, preserves marginal distribution), 1000 resamples.
  (2) Quartile-AUC: top-quartile vs bottom-quartile disorder_frac pairs,
      AUC of severity_score discrimination (for direct comparability to the
      domain/nterm/resync AUC numbers already computed).

Pre-registered prediction (S2, before running): since disorder_frac is
already near-orthogonal to every other covariate (|rho|<0.05, VIF=1.00), its
raw and "matched" estimates should be nearly IDENTICAL -- this covariate
needed no confound control to begin with, unlike resync/domain. If partial
and raw diverge substantially, that itself would be a surprise worth
investigating (possible nonlinear confound VIF misses).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_BOOT = 1000
SEED = 42
OTHER_COVS = ['size_z', 'domain_binary', 'nterm_overlap', 'resync_failure_binary']


def residualize(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def partial_corr(df):
    X = df[OTHER_COVS].to_numpy(dtype=float)
    d_res = residualize(df['disorder_frac'].to_numpy(dtype=float), X)
    s_res = residualize(df['severity_score'].to_numpy(dtype=float), X)
    return np.corrcoef(d_res, s_res)[0, 1], d_res, s_res


def gene_cluster_bootstrap_corr(df, d_res, s_res, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    genes = df['gene'].to_numpy()
    uniq = np.unique(genes)
    gene_rows = {g: np.where(genes == g)[0] for g in uniq}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        samp_genes = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([gene_rows[g] for g in samp_genes])
        boot[b] = np.corrcoef(d_res[rows], s_res[rows])[0, 1]
    return np.nanpercentile(boot, [2.5, 97.5])


def permutation_null_corr(d_res, s_res, n_perm=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = np.corrcoef(d_res, s_res)[0, 1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(d_res)
        null[i] = np.corrcoef(perm, s_res)[0, 1]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, null, p


def quartile_auc(df):
    q1, q3 = df['disorder_frac'].quantile([0.25, 0.75])
    lo = df[df['disorder_frac'] <= q1]
    hi = df[df['disorder_frac'] >= q3]
    scores = np.concatenate([hi['severity_score'], lo['severity_score']])
    labels = np.concatenate([np.ones(len(hi)), np.zeros(len(lo))])
    ranks = pd.Series(scores).rank().to_numpy()
    n1, n2 = labels.sum(), len(labels) - labels.sum()
    r_pos = ranks[labels == 1].sum()
    auc = (r_pos - n1 * (n1 + 1) / 2) / (n1 * n2)
    return auc, int(n1), int(n2)


def main():
    for tissue in ['muscle', 'brain']:
        df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
        log_size = np.log1p(df['size'])
        df['size_z'] = (log_size - log_size.mean()) / log_size.std()

        raw_r = np.corrcoef(df['disorder_frac'], df['severity_score'])[0, 1]
        part_r, d_res, s_res = partial_corr(df)
        ci = gene_cluster_bootstrap_corr(df, d_res, s_res)
        obs, null, p = permutation_null_corr(d_res, s_res)
        auc, n_hi, n_lo = quartile_auc(df)

        print(f"\n=== {tissue} (n={len(df)}) ===")
        print(f"raw Pearson r (disorder_frac, severity_score):      {raw_r:+.4f}")
        print(f"partial r (size+domain+nterm+resync controlled):    {part_r:+.4f}  "
              f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  perm_p={p:.4f}")
        print(f"quartile AUC (top-25% vs bottom-25% disorder_frac): {auc:.3f}  "
              f"(n_hi={n_hi}, n_lo={n_lo})")


if __name__ == '__main__':
    main()
