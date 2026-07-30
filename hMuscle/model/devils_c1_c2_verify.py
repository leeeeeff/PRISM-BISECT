#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c1_c2_verify.py

Decisive tests requested by the devils-advocate review of the region-pool
covariate pilot (2026-07-21):

C1 -- is the domain_binary/size_z "gain" under region-pool a real encoding
      effect, or an L2-norm scaling artifact of projecting a small-window
      difference vector onto a direction trained on whole-protein-scale
      mean-pool vectors? Test: report |D_mean|/|D_region|/|D_scram| norm
      distributions by size quintile, then re-run the Part-A OLS on
      UNIT-NORMALIZED D-vectors (score = (D/|D|)@direction instead of D@direction).
      If domain_binary/size_z gains collapse after normalization -> norm artifact.

C2 -- is the domain_binary "region rescue" (Part B) actually distinguishable
      from the scrambled-window null, or does it ride entirely on region's
      window length being tautologically closer to the true edit size (which
      conflates with domain_binary via known encoding conflation, see
      approach-covariate-functional-localization-8axis)? Test: PAIRED
      gene-cluster bootstrap of (region_size_matched_rate - scram_size_matched_rate),
      resampling genes ONCE per draw and computing both rates on the same
      resample (not two separate bootstraps compared post-hoc).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
SEV = ROOT / 'reports/severity_pairs'
N_BOOT = 1000
SEED = 42
N_BINS = 5
N_FOLDS = 5

DOMAIN_MAT = {'muscle': 'domain_matrix_proper_test.npy', 'brain': 'domain_matrix_brain_full.npy'}
MEANPOOL_L15 = {'muscle': 'esm2_layer_15_t30_150M.npy',
                'brain': 'brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy'}
MEANPOOL_L30 = {'muscle': 'esm2_layer_30_t30_150M.npy',
                'brain': 'brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy'}


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=42):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def load(tissue):
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_region.tsv', sep='\t')
    cache = np.load(SEV / f'{tissue}_region_embeddings.npz')
    D_region, D_scram = cache['D_region'], cache['D_scram']
    L15 = np.load(ROOT / 'hMuscle/data' / MEANPOOL_L15[tissue]).astype(np.float32)
    L30 = np.load(ROOT / 'hMuscle/data' / MEANPOOL_L30[tissue]).astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    D_mean = emb[df['long_idx'].to_numpy()] - emb[df['short_idx'].to_numpy()]
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()

    # C2-safety: fold assignment must come from the FULL original scored population
    # (same gene set/shuffle order as build_severity_score.py), not the region-pool-
    # valid subset -- gene_disjoint_folds' shuffle order depends on WHICH genes are
    # passed in, so training on a subset silently gives a DIFFERENT fold/direction.
    df_full = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df_full = df_full[df_full['tissue'] == tissue].reset_index(drop=True)
    fold_full = gene_disjoint_folds(df_full['gene'].to_numpy())
    gene_to_fold = dict(zip(df_full['gene'].to_numpy(), fold_full))
    fold = df['gene'].map(gene_to_fold).to_numpy()

    D_mean_full = emb[df_full['long_idx'].to_numpy()] - emb[df_full['short_idx'].to_numpy()]
    directions = {}
    for k in range(N_FOLDS):
        train_mask = fold_full != k
        direction = D_mean_full[train_mask].mean(axis=0)
        norm = np.linalg.norm(direction)
        directions[k] = direction / norm if norm > 0 else direction
    check = np.zeros(len(df_full))
    for k, direction in directions.items():
        mask = fold_full == k
        check[mask] = D_mean_full[mask] @ direction
    assert np.allclose(check, df_full['severity_score'].to_numpy(), atol=1e-3), \
        f"[{tissue}] fold/direction mismatch vs stored severity_score (full population)"

    return df, D_mean, D_region, D_scram, fold, directions


def c1_norm_check(tissue, df, D_mean, D_region, D_scram):
    print(f"\n=== C1 [{tissue}]: |D| norm distributions by size quintile ===")
    nm = np.linalg.norm(D_mean, axis=1)
    nr = np.linalg.norm(D_region, axis=1)
    ns = np.linalg.norm(D_scram, axis=1)
    print(f"  overall median |D|: mean-pool={np.median(nm):.2f}  region={np.median(nr):.2f}  "
          f"scram={np.median(ns):.2f}")
    bin_edges = np.quantile(df['size'], np.linspace(0, 1, N_BINS + 1))
    bin_edges[0] = -np.inf; bin_edges[-1] = np.inf
    bins = pd.cut(df['size'], bin_edges, labels=False, include_lowest=True)
    print(f"  {'bin':<5}{'n':<8}{'|D_mean|':<12}{'|D_region|':<12}{'|D_scram|':<12}")
    for k in range(N_BINS):
        mask = (bins == k).to_numpy()
        print(f"  {k:<5}{mask.sum():<8}{np.median(nm[mask]):<12.2f}"
              f"{np.median(nr[mask]):<12.2f}{np.median(ns[mask]):<12.2f}")
    return nm, nr, ns


def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def design_matrix(df):
    cols = ['size_z', 'domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(X)), X])
    return X, ['intercept'] + cols


def c1_normalized_ols(tissue, df, D_mean, D_region, D_scram, fold, directions):
    print(f"\n=== C1 [{tissue}]: 5-covariate OLS on UNIT-NORMALIZED D-vectors ===")
    X, cols = design_matrix(df)

    def score_normalized(D):
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        out = np.zeros(len(D))
        for k, direction in directions.items():
            mask = fold == k
            out[mask] = Dn[mask] @ direction
        return out

    y_mean = score_normalized(D_mean)
    y_region = score_normalized(D_region)
    y_scram = score_normalized(D_scram)
    betas = {}
    for name, y in [('mean', y_mean), ('region', y_region), ('scram', y_scram)]:
        betas[name] = dict(zip(cols, ols_fit(X, y)))
    print(f"  {'covariate':<22}{'beta_mean':>11}{'beta_region':>13}{'beta_scram':>12}")
    for c in cols:
        if c == 'intercept':
            continue
        print(f"  {c:<22}{betas['mean'][c]:>11.4f}{betas['region'][c]:>13.4f}{betas['scram'][c]:>12.4f}")
    return betas


def c2_paired_bootstrap(tissue, df):
    print(f"\n=== C2 [{tissue}]: PAIRED bootstrap, region_size_matched - scram_size_matched "
          f"(domain_binary discrimination) ===")
    dom = np.load(ROOT / 'hMuscle/results_isoform/features' / DOMAIN_MAT[tissue])
    dom_count = dom.sum(axis=1).astype(np.int32)
    sub = df[df['domain_binary'] == 1].copy()
    sub['domain_diff'] = dom_count[sub['long_idx'].to_numpy()] - dom_count[sub['short_idx'].to_numpy()]
    pos = sub[sub['domain_diff'] > 0].copy()
    neg = sub[sub['domain_diff'] < 0].copy()
    if len(neg) < 10:
        print(f"  decoupled subset too small (n={len(neg)}), skipping")
        return

    bin_edges = np.quantile(neg['size'], np.linspace(0, 1, N_BINS + 1))
    bin_edges[0] = -np.inf; bin_edges[-1] = np.inf
    neg['bin'] = pd.cut(neg['size'], bin_edges, labels=False, include_lowest=True)
    pos['bin'] = pd.cut(pos['size'], bin_edges, labels=False, include_lowest=True)
    bin_weights = neg['bin'].value_counts(normalize=True).sort_index()
    bin_weights = bin_weights.reindex(range(N_BINS), fill_value=0.0).to_numpy()

    pos_valid = pos.dropna(subset=['bin'])
    pos_valid = pos_valid[pos_valid['bin'].isin(range(N_BINS))].reset_index(drop=True)

    def weighted_rate(score_col, aligned_arr, bin_id, rows):
        r_aligned = aligned_arr[rows]; r_bin = bin_id[rows]
        w, tw = 0.0, 0.0
        for k in range(N_BINS):
            mask = r_bin == k
            if mask.sum() == 0:
                continue
            w += bin_weights[k] * r_aligned[mask].mean(); tw += bin_weights[k]
        return w / tw if tw > 0 else np.nan

    aligned_region = (pos_valid['severity_score_region'].to_numpy() > 0).astype(float)
    aligned_scram = (pos_valid['severity_score_scram'].to_numpy() > 0).astype(float)
    aligned_mean = (pos_valid['severity_score'].to_numpy() > 0).astype(float)
    bin_id = pos_valid['bin'].to_numpy()
    genes = pos_valid['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}

    rng = np.random.default_rng(SEED)
    diff_region_scram = np.empty(N_BOOT)
    diff_region_mean = np.empty(N_BOOT)
    point_region = weighted_rate(None, aligned_region, bin_id, np.arange(len(pos_valid)))
    point_scram = weighted_rate(None, aligned_scram, bin_id, np.arange(len(pos_valid)))
    point_mean = weighted_rate(None, aligned_mean, bin_id, np.arange(len(pos_valid)))
    for b in range(N_BOOT):
        sampled = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled])
        r_reg = weighted_rate(None, aligned_region, bin_id, rows)
        r_scr = weighted_rate(None, aligned_scram, bin_id, rows)
        r_mean = weighted_rate(None, aligned_mean, bin_id, rows)
        diff_region_scram[b] = r_reg - r_scr
        diff_region_mean[b] = r_reg - r_mean

    ci_rs = np.nanpercentile(diff_region_scram, [2.5, 97.5])
    ci_rm = np.nanpercentile(diff_region_mean, [2.5, 97.5])
    print(f"  point estimates: mean={point_mean:.3f}  region={point_region:.3f}  scram={point_scram:.3f}")
    print(f"  PAIRED (region - scram): {point_region - point_scram:+.3f}  "
          f"CI=[{ci_rs[0]:+.3f},{ci_rs[1]:+.3f}]  "
          f"{'EXCLUDES 0 -> region genuinely beats scram' if ci_rs[0] * ci_rs[1] > 0 else 'INCLUDES 0 -> NOT distinguishable from scram'}")
    print(f"  PAIRED (region - mean):  {point_region - point_mean:+.3f}  "
          f"CI=[{ci_rm[0]:+.3f},{ci_rm[1]:+.3f}]  "
          f"{'EXCLUDES 0' if ci_rm[0] * ci_rm[1] > 0 else 'INCLUDES 0'}")


def main():
    for tissue in ['muscle', 'brain']:
        df, D_mean, D_region, D_scram, fold, directions = load(tissue)
        c1_norm_check(tissue, df, D_mean, D_region, D_scram)

        # sanity: fixed-direction score on this subset's own D_mean should reproduce
        # the region TSV's stored severity_score column (subset of the full check)
        check = np.zeros(len(df))
        for k, direction in directions.items():
            mask = fold == k
            check[mask] = D_mean[mask] @ direction
        assert np.allclose(check, df['severity_score'].to_numpy(), atol=1e-3), \
            f"[{tissue}] fold/direction mismatch vs stored severity_score (subset check)"

        c1_normalized_ols(tissue, df, D_mean, D_region, D_scram, fold, directions)
        c2_paired_bootstrap(tissue, df)


if __name__ == '__main__':
    main()
