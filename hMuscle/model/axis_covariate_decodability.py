#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis_covariate_decodability.py
================================
Stable functional-LOCALISATION deliverable, after both occlusion designs
degenerated (axis_functional_matrix.py retrain -> universal redundancy ~0;
axis_functional_matrix_reliance.py test-time unreg-lstsq -> universal probe
brittleness ~100%, real indistinguishable from random). Root cause: raw
edit-covariates have no external target-trained model, so occlusion cannot
cleanly isolate an "encoding != usage" usage axis for them.

Well-posed question instead: how much of each covariate is functionally
DECODABLE from (a) the full 1280-d Dphi (ceiling), (b) the 8-axis interpretable
subspace, (c) each single interpretable axis (localisation). This generalises
"axis3 carries the domain-ranking signal" to all 5 isoform-local-change
dimensions, is numerically stable (low-dim probes), and recovers the positive
controls by construction.

  features: Dphi = phi(long)-phi(short), L15||L30 (1280-d) projected to the
            8-axis space via W (per-layer z-score then W, layer-averaged), i.e.
            Daxis (8-d) -- same construction as reference-esm2-pca-axes-final.md.
  probe: gene-disjoint 5-fold, out-of-fold. binary C -> AUROC; continuous
         (size=log1p, disorder) -> Spearman.
  (a) full-1280 ceiling: lstsq on Dphi (standardised) -- decodability ceiling.
  (b) 8-axis subspace: lstsq on the 8-d Daxis -- how much lives in interpretable axes.
  (c) single axis: Daxis_k alone as the score (sign-oriented) -> metric.

PRE-REGISTERED (S2): size -> axis5 top; domain_binary -> axis3 top (positive
controls). disorder_frac -> axis0 top (this session). nterm/resync: open.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
N_AXES = 8
SEED = 42
BINARY_COVS = ['domain_binary', 'nterm_overlap', 'resync_failure_binary']
AXIS_LABEL = {0: 'axis0(betaSheet/TM)', 1: 'axis1(LRR/Ig)', 2: 'axis2(Pro-turn)',
              3: 'axis3(domain)', 4: 'axis4(helix-charge)', 5: 'axis5(LENGTH)',
              6: 'axis6(KRAB-ZNF)', 7: 'axis7(acidic-hel)'}


def load_axis_scalars(tissue):
    """Daxis per isoform: 8-d, = 30-layer-averaged joint-PCA projection (same as
    reports/v20b_pca_interp/Z_{tissue}_Nx30x8.npy convention)."""
    Z = np.load(ROOT / f'reports/v20b_pca_interp/Z_{tissue}_Nx30x8.npy')
    return Z.mean(axis=1)  # (N,8)


def load_full_dphi(tissue, long_idx, short_idx):
    if tissue == 'muscle':
        L15 = np.load(ROOT / 'hMuscle/data/esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/esm2_layer_30_t30_150M.npy').astype(np.float32)
    else:
        L15 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    return np.concatenate([L15[long_idx] - L15[short_idx], L30[long_idx] - L30[short_idx]], axis=1)


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fmap = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fmap[g] for g in genes])


def oof_probe(X, y, fold):
    if X.ndim == 1:
        X = X[:, None]
    pred = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        Xte = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd])
        beta, *_ = np.linalg.lstsq(Xtr, y[tr] - y[tr].mean(), rcond=None)
        pred[te] = Xte @ beta
    return pred


def metric(y, pred, ctype):
    return roc_auc_score(y, pred) if ctype == 'binary' else stats.spearmanr(y, pred).correlation


def run_tissue(tissue):
    print(f"\n{'='*72}\n{tissue}\n{'='*72}", flush=True)
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    long_idx, short_idx = df['long_idx'].to_numpy(), df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())
    axis_iso = load_axis_scalars(tissue)
    Daxis = axis_iso[long_idx] - axis_iso[short_idx]          # (n,8)
    Dphi = load_full_dphi(tissue, long_idx, short_idx)        # (n,1280)

    targets = {c: ('binary', df[c].to_numpy(float)) for c in BINARY_COVS}
    targets['size'] = ('cont', np.log1p(df['size'].to_numpy(float)))
    targets['disorder_frac'] = ('cont', df['disorder_frac'].to_numpy(float))

    rows = []
    for cname, (ctype, y) in targets.items():
        ceil = metric(y, oof_probe(Dphi, y, fold), ctype)
        sub8 = metric(y, oof_probe(Daxis, y, fold), ctype)
        per_axis = {}
        for k in range(N_AXES):
            m = metric(y, Daxis[:, k], ctype)  # single axis as score (sign-aware)
            # orient so higher = better decodability magnitude for binary
            per_axis[k] = m
        # report
        print(f"  [{cname}] full-1280 ceiling={ceil:+.4f} | 8-axis subspace={sub8:+.4f}", flush=True)
        if ctype == 'binary':
            ordered = sorted(range(N_AXES), key=lambda k: abs(per_axis[k] - 0.5), reverse=True)
        else:
            ordered = sorted(range(N_AXES), key=lambda k: abs(per_axis[k]), reverse=True)
        for k in ordered:
            mag = abs(per_axis[k] - 0.5) if ctype == 'binary' else abs(per_axis[k])
            print(f"      {AXIS_LABEL[k]:<20} single-axis={per_axis[k]:+.4f} (|signal|={mag:.4f})", flush=True)
        for k in range(N_AXES):
            rows.append({'tissue': tissue, 'covariate': cname, 'ctype': ctype,
                         'axis': k, 'axis_label': AXIS_LABEL[k],
                         'single_axis_metric': per_axis[k], 'ceiling_1280': ceil,
                         'subspace_8axis': sub8})
    return rows


def main():
    rows = []
    for tissue in ['muscle', 'brain']:
        rows += run_tissue(tissue)
    out = pd.DataFrame(rows)
    out_path = ROOT / 'reports/severity_pairs/axis_covariate_decodability.tsv'
    out.to_csv(out_path, sep='\t', index=False)
    print(f"\n[Save] {out_path}")

    print("\n=== per-covariate top interpretable axis (max |single-axis signal|) ===")
    for (tis, cov), g in out.groupby(['tissue', 'covariate']):
        ctype = g['ctype'].iloc[0]
        if ctype == 'binary':
            top = g.loc[(g['single_axis_metric'] - 0.5).abs().idxmax()]
            sig = abs(top['single_axis_metric'] - 0.5)
        else:
            top = g.loc[g['single_axis_metric'].abs().idxmax()]
            sig = abs(top['single_axis_metric'])
        print(f"  {tis:<6} {cov:<22} -> {top['axis_label']:<20} "
              f"single={top['single_axis_metric']:+.3f} |sig|={sig:.3f} "
              f"(ceiling={top['ceiling_1280']:+.3f}, 8axis={top['subspace_8axis']:+.3f})")


if __name__ == '__main__':
    main()
