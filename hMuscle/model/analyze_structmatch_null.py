#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_structmatch_null.py

M3 resolution: compare severity_score_structmatch (position-distribution-
matched null) against severity_score_region and the original uniform
severity_score_scram, for the two survivor findings from the C1/C2/C3
devils-advocate verification:
  - nterm_overlap collapse under region-pooling (survived C1 norm-check and
    C3 fallback-stratification -- does it survive a HARDER, position-matched
    null, or does the collapse partly evaporate because uniform scram was
    an easy/biased null?)
  - domain_binary size-matched rescue (Part B paired bootstrap already beat
    uniform scram -- does it still beat the harder structure-matched null?)
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
N_BOOT = 1000
SEED = 42
N_BINS = 5

DOMAIN_MAT = {'muscle': 'domain_matrix_proper_test.npy', 'brain': 'domain_matrix_brain_full.npy'}

spec = importlib.util.spec_from_file_location('sr', MODEL / 'severity_regression.py')
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def load(tissue):
    df_region = pd.read_csv(SEV / f'{tissue}_severity_pairs_region.tsv', sep='\t')
    df_sm = pd.read_csv(SEV / f'{tissue}_severity_pairs_structmatch.tsv', sep='\t')
    key = ['gene', 'canonical_idx', 'other_idx', 'long_idx', 'short_idx']
    df = df_region.merge(df_sm[key + ['severity_score_structmatch']], on=key, how='inner')
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    print(f"[load] {tissue}: {len(df)} pairs with region+structmatch both available "
          f"(region table had {len(df_region)}, structmatch had {len(df_sm)})", flush=True)
    return df


def design_matrix(df):
    X, cols = sr.design_matrix(df, with_tissue=False)
    return X, cols


def nterm_ols_comparison(tissue, df):
    print(f"\n=== [{tissue}] nterm_overlap coefficient: mean / region / scram(uniform) / structmatch ===")
    X, cols = design_matrix(df)
    idx = cols.index('nterm_overlap')
    genes = df['gene'].to_numpy()
    for col in ['severity_score', 'severity_score_region', 'severity_score_scram', 'severity_score_structmatch']:
        y = df[col].to_numpy(dtype=np.float64)
        beta, resid, r2 = sr.ols_fit(X, y)
        se = sr.cluster_robust_se(X, resid, genes)
        t = beta[idx] / se[idx]
        print(f"  {col:<32} beta_nterm={beta[idx]:>8.3f}  t={t:>7.2f}")


def domain_size_matched(df, score_col):
    dom = np.load(ROOT / 'hMuscle/results_isoform/features' / DOMAIN_MAT[TISSUE_CTX[0]])
    dom_count = dom.sum(axis=1).astype(np.int32)
    sub = df[df['domain_binary'] == 1].copy()
    sub['domain_diff'] = dom_count[sub['long_idx'].to_numpy()] - dom_count[sub['short_idx'].to_numpy()]
    pos = sub[sub['domain_diff'] > 0].copy()
    neg = sub[sub['domain_diff'] < 0].copy()
    if len(neg) < 10:
        return None
    bin_edges = np.quantile(neg['size'], np.linspace(0, 1, N_BINS + 1))
    bin_edges[0] = -np.inf; bin_edges[-1] = np.inf
    neg['bin'] = pd.cut(neg['size'], bin_edges, labels=False, include_lowest=True)
    pos['bin'] = pd.cut(pos['size'], bin_edges, labels=False, include_lowest=True)
    bin_weights = neg['bin'].value_counts(normalize=True).sort_index()
    bin_weights = bin_weights.reindex(range(N_BINS), fill_value=0.0).to_numpy()
    pos_valid = pos.dropna(subset=['bin'])
    pos_valid = pos_valid[pos_valid['bin'].isin(range(N_BINS))]
    rate, tw = 0.0, 0.0
    for k in range(N_BINS):
        pos_k = pos_valid[pos_valid['bin'] == k]
        if len(pos_k) == 0 or bin_weights[k] == 0:
            continue
        rate += bin_weights[k] * np.mean(pos_k[score_col] > 0); tw += bin_weights[k]
    return (rate / tw if tw > 0 else np.nan), pos_valid, bin_weights


def domain_paired_bootstrap(tissue, df):
    print(f"\n=== [{tissue}] domain_binary size-matched: region vs structmatch (paired bootstrap) ===")
    global TISSUE_CTX
    TISSUE_CTX = [tissue]
    r_region, pos_valid, bin_weights = domain_size_matched(df, 'severity_score_region')
    r_sm, _, _ = domain_size_matched(df, 'severity_score_structmatch')
    r_scram, _, _ = domain_size_matched(df, 'severity_score_scram')
    r_mean, _, _ = domain_size_matched(df, 'severity_score')
    print(f"  size-matched rate: mean={r_mean:.3f}  region={r_region:.3f}  "
          f"scram(uniform)={r_scram:.3f}  structmatch={r_sm:.3f}")

    aligned_region = (pos_valid['severity_score_region'].to_numpy() > 0).astype(float)
    aligned_sm = (pos_valid['severity_score_structmatch'].to_numpy() > 0).astype(float)
    bin_id = pos_valid['bin'].to_numpy()
    genes = pos_valid['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}

    def weighted_rate(aligned_arr, rows):
        r_bin = bin_id[rows]; r_al = aligned_arr[rows]
        w, tw = 0.0, 0.0
        for k in range(N_BINS):
            mask = r_bin == k
            if mask.sum() == 0:
                continue
            w += bin_weights[k] * r_al[mask].mean(); tw += bin_weights[k]
        return w / tw if tw > 0 else np.nan

    rng = np.random.default_rng(SEED)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sampled = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled])
        diffs[b] = weighted_rate(aligned_region, rows) - weighted_rate(aligned_sm, rows)
    ci = np.nanpercentile(diffs, [2.5, 97.5])
    point = r_region - r_sm
    print(f"  PAIRED (region - structmatch): {point:+.3f}  CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  "
          f"{'EXCLUDES 0 -> region beats the HARDER null too' if ci[0] * ci[1] > 0 else 'INCLUDES 0 -> does NOT beat the harder null'}")


TISSUE_CTX = ['muscle']


def main():
    for tissue in ['muscle', 'brain']:
        df = load(tissue)
        nterm_ols_comparison(tissue, df)
        domain_paired_bootstrap(tissue, df)


if __name__ == '__main__':
    main()
