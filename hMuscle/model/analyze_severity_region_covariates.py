#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_severity_region_covariates.py

Muscle-only pilot analysis (Option C follow-up): does region-pool improve
covariate discrimination relative to mean-pool, for ALL 5 severity covariates
(not just domain_binary, which was already tested and found flat)?

Consumes reports/severity_pairs/muscle_severity_pairs_region.tsv, produced by
build_severity_region_embeddings_muscle.py -- has severity_score (mean-pool,
original), severity_score_region, severity_score_scram (C1 null), all scored
with the SAME fixed mean-pool-trained coherence direction (C2 guard).

Two independent tests, reusing established machinery without modification of
its statistical logic:
  (A) 5-covariate OLS (severity_regression.py's design_matrix / ols_fit /
      cluster_robust_se / permutation_null_r2), refit three times (y = mean /
      region / scram) on the IDENTICAL row set, so any beta or R^2 shift is
      attributable to pooling alone, not to sample composition.
  (B) domain_binary size-quintile-matched direction-discrimination
      (domain_direction_size_matched.py's decoupled-subset + bin-weighted-
      reweighting logic), refit three times on the score sign.

PRE-REGISTERED (S2, stated before running on the real data):
  - domain_binary / resync_failure_binary / size: H_reproduce (region ~= mean
    ~= scram) predicted -- these already conflate in the mean-pool axis3, and
    domain_binary's region-pool flatness was already shown at the 2-iso-subset
    trajectory-PCA level (finding-region-pool-trajectory-pca).
  - nterm_overlap: H_falsify (region > mean AND region > scram) is the live,
    motivated hypothesis -- mean-pool axis-covariate-decodability found NO
    interpretable axis carries it despite high ceiling (0.84), and region-pool
    directly targets edit location, which is mechanistically what nterm_overlap
    measures.
  - disorder_frac: genuinely open (axis0 is a whole-protein composition axis,
    not obviously a localized-pooling beneficiary; but disorder_frac itself is
    already computed FROM the local window, so a positive finding here would
    not be shocking either).
A "real" region-pool gain requires beta_region notably > beta_mean AND
beta_region > beta_scram (rules out generic local-window artifacts, C1).
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
TISSUE = os.environ.get('ANALYZE_TISSUE', 'muscle')
IN_TSV = ROOT / f'reports/severity_pairs/{TISSUE}_severity_pairs_region.tsv'
DOMAIN_MAT = {'muscle': 'domain_matrix_proper_test.npy', 'brain': 'domain_matrix_brain_full.npy'}[TISSUE]
N_BOOT = 1000
SEED = 42
N_BINS = 5

spec = importlib.util.spec_from_file_location('sr', MODEL / 'severity_regression.py')
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def load():
    df = pd.read_csv(IN_TSV, sep='\t')
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    print(f"[load] {len(df)} {TISSUE} region-pool-valid pairs, {df['gene'].nunique()} genes", flush=True)
    return df


def fit_all_three(df):
    """Part A: 5-covariate OLS on the SAME row set for mean/region/scram."""
    X, cols = sr.design_matrix(df, with_tissue=False)
    genes = df['gene'].to_numpy()
    results = {}
    for score_col in ['severity_score', 'severity_score_region', 'severity_score_scram']:
        y = df[score_col].to_numpy(dtype=np.float64)
        beta, resid, r2 = sr.ols_fit(X, y)
        se = sr.cluster_robust_se(X, resid, genes)
        t = beta / se
        G = df['gene'].nunique()
        dof = G - 1
        p = sr.t_to_p(t, dof)
        p_bh = sr.bh_correct(p)
        null_r2 = sr.permutation_null_r2(X, y, n_perm=N_BOOT, seed=SEED)
        null_p = (null_r2 >= r2).mean()
        results[score_col] = {'beta': dict(zip(cols, beta)), 'se': dict(zip(cols, se)),
                               't': dict(zip(cols, t)), 'p_bh': dict(zip(cols, p_bh)),
                               'r2': r2, 'null_r2_mean': null_r2.mean(), 'null_p': null_p}
    return results, cols


def bootstrap_beta_diff(df, cols, score_a, score_b, n_boot=N_BOOT, seed=SEED):
    """Gene-cluster bootstrap of beta[score_a] - beta[score_b] per covariate."""
    rng = np.random.default_rng(seed)
    genes = df['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}
    diffs = {c: np.empty(n_boot) for c in cols}
    for b in range(n_boot):
        sampled = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled])
        sub = df.iloc[rows]
        X, _ = sr.design_matrix(sub, with_tissue=False)
        ya = sub[score_a].to_numpy(dtype=np.float64)
        yb = sub[score_b].to_numpy(dtype=np.float64)
        beta_a, *_ = sr.ols_fit(X, ya)
        beta_b, *_ = sr.ols_fit(X, yb)
        for i, c in enumerate(cols):
            diffs[c][b] = beta_a[i] - beta_b[i]
    return diffs


def print_ols_comparison(results, cols):
    print("\n=== Part A: 5-covariate OLS, mean-pool vs region-pool vs scrambled-null ===")
    print(f"R^2:  mean={results['severity_score']['r2']:.4f}  "
          f"region={results['severity_score_region']['r2']:.4f}  "
          f"scram={results['severity_score_scram']['r2']:.4f}")
    print(f"null_p:  mean={results['severity_score']['null_p']:.4f}  "
          f"region={results['severity_score_region']['null_p']:.4f}  "
          f"scram={results['severity_score_scram']['null_p']:.4f}")
    print(f"\n{'covariate':<22}{'beta_mean':>11}{'beta_region':>13}{'beta_scram':>12}"
          f"{'p_BH_region':>13}")
    for c in cols:
        if c == 'intercept':
            continue
        bm = results['severity_score']['beta'][c]
        br = results['severity_score_region']['beta'][c]
        bs = results['severity_score_scram']['beta'][c]
        pbh = results['severity_score_region']['p_bh'][c]
        print(f"{c:<22}{bm:>11.3f}{br:>13.3f}{bs:>12.3f}{pbh:>13.2e}")


def domain_size_matched(df, score_col):
    """Part B, generalized from domain_direction_size_matched.py to accept any
    score column -- IDENTICAL bootstrap/reweighting logic, only the sign-test
    column changes."""
    dom = np.load(ROOT / 'hMuscle/results_isoform/features' / DOMAIN_MAT)
    dom_count = dom.sum(axis=1).astype(np.int32)
    sub = df[df['domain_binary'] == 1].copy()
    sub['domain_diff'] = dom_count[sub['long_idx'].to_numpy()] - dom_count[sub['short_idx'].to_numpy()]
    pos = sub[sub['domain_diff'] > 0].copy()
    neg = sub[sub['domain_diff'] < 0].copy()
    if len(neg) < 10:
        print(f"  [{score_col}] decoupled subset too small (n={len(neg)}), skipping quintile match")
        return None

    quantiles = np.linspace(0, 1, N_BINS + 1)
    bin_edges = np.quantile(neg['size'], quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    neg['bin'] = pd.cut(neg['size'], bin_edges, labels=False, include_lowest=True)
    pos['bin'] = pd.cut(pos['size'], bin_edges, labels=False, include_lowest=True)
    bin_weights = neg['bin'].value_counts(normalize=True).sort_index()
    bin_weights = bin_weights.reindex(range(N_BINS), fill_value=0.0).to_numpy()

    pos_valid = pos.dropna(subset=['bin'])
    pos_valid = pos_valid[pos_valid['bin'].isin(range(N_BINS))]
    raw_rate = float(np.mean(pos[score_col] > 0))
    decoupled_rate = float(np.mean(neg[score_col] > 0))
    size_matched_rate, total_w = 0.0, 0.0
    for k in range(N_BINS):
        pos_k = pos_valid[pos_valid['bin'] == k]
        if len(pos_k) == 0 or bin_weights[k] == 0:
            continue
        size_matched_rate += bin_weights[k] * np.mean(pos_k[score_col] > 0)
        total_w += bin_weights[k]
    size_matched_rate = size_matched_rate / total_w if total_w > 0 else float('nan')

    rng = np.random.default_rng(SEED)
    genes = pos_valid['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}
    aligned = (pos_valid[score_col].to_numpy() > 0).astype(float)
    bin_id = pos_valid['bin'].to_numpy()
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sampled = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled])
        r_aligned = aligned[rows]; r_bin = bin_id[rows]
        w, tw = 0.0, 0.0
        for k in range(N_BINS):
            mask = r_bin == k
            if mask.sum() == 0:
                continue
            w += bin_weights[k] * r_aligned[mask].mean(); tw += bin_weights[k]
        boot[b] = w / tw if tw > 0 else np.nan
    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])
    print(f"  [{score_col}] n_aligned={len(pos)} n_decoupled={len(neg)}  "
          f"raw={raw_rate:.3f}  size-matched={size_matched_rate:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"decoupled={decoupled_rate:.3f}")
    return size_matched_rate, (ci_lo, ci_hi), decoupled_rate


def main():
    df = load()
    results, cols = fit_all_three(df)
    print_ols_comparison(results, cols)

    print("\n[bootstrap] region - mean beta difference per covariate (gene-cluster resample, "
          f"n={N_BOOT}):")
    diffs_rm = bootstrap_beta_diff(df, cols, 'severity_score_region', 'severity_score')
    diffs_rs = bootstrap_beta_diff(df, cols, 'severity_score_region', 'severity_score_scram')
    print(f"{'covariate':<22}{'region-mean CI':>28}{'region-scram CI':>28}")
    for c in cols:
        if c == 'intercept':
            continue
        lo_m, hi_m = np.percentile(diffs_rm[c], [2.5, 97.5])
        lo_s, hi_s = np.percentile(diffs_rs[c], [2.5, 97.5])
        print(f"{c:<22}[{lo_m:>8.3f},{hi_m:>8.3f}]{'':>6}[{lo_s:>8.3f},{hi_s:>8.3f}]")

    print("\n=== Part B: domain_binary size-quintile-matched discrimination "
          "(mean-pool result already known: raw 0.668 -> size-matched 0.559 ~= decoupled 0.530) ===")
    for score_col in ['severity_score', 'severity_score_region', 'severity_score_scram']:
        domain_size_matched(df, score_col)


if __name__ == '__main__':
    main()
