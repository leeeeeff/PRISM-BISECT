#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""covariate_direction_axis_overlap.py

Extends the established 8-axis PCA "axis dossier" (reference-esm2-pca-axes-final)
to the 5 severity covariates (size, domain_binary, nterm_overlap, disorder_frac,
resync_failure_binary) -- Phase 1 of the user's request: "PCA 8축 프레임 안에서
설명 안 되는(특히 nterm_overlap) covariate들을, 별도 프레임이 아니라 하나의
통합된 인코딩 맵으로 정량화."

For domain_binary, this EXACTLY reproduces (as a sanity check) the established
result in ceiling_640dim_domain.py: 8-axis AUROC ~0.71-0.72 vs full-640dim
AUROC ~0.84, i.e. the 8-axis PCA basis is a lossy compression (interpretability/
discriminability tradeoff), not an information ceiling. This script additionally
does what that earlier work did NOT: explicitly extracts a single, gene-disjoint-
CV-trained SUPERVISED direction per covariate (not just an AUROC number), so it
can be directly compared (cosine similarity) against the 8 existing PCA axes --
quantifying, per covariate, "how much of this signal's direction is already
inside the 8-axis subspace vs orthogonal to it."

Feature convention: z-scored 640-dim layer-mean (IDENTICAL basis to the existing
axis dossier: layer_stats_mu.npy/sd.npy + W_axes_8x640.npy), NOT the 1280-dim
L15+L30-concat basis used elsewhere this session for severity_score -- chosen
for direct comparability with the established axis-biology reference.

Population: severity_pairs_scored.tsv (already has domain_binary, nterm_overlap,
disorder_frac, resync_failure_binary, size per pair) -- reused as-is rather than
rebuilding the all-combinations within-gene-pair set ceiling_640dim_domain.py
used, since it already has clean covariate labels and gene-disjoint CV machinery
is already established on it this session.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
SEV = ROOT / 'reports/severity_pairs'
PCA_DIR = ROOT / 'reports/v20b_pca_interp'
N_FOLDS = 5
SEED = 42

BINARY_COVS = ['domain_binary', 'nterm_overlap', 'resync_failure_binary']
CONT_COVS = ['size', 'disorder_frac']


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def build_640_layermean(tissue):
    """z-scored 640-dim layer-mean, IDENTICAL basis to layer_stats_mu/sd.npy,
    accumulated one layer at a time (memory: don't load all 30 at once)."""
    mu = np.load(PCA_DIR / 'layer_stats_mu.npy').astype(np.float64)
    sd = np.load(PCA_DIR / 'layer_stats_sd.npy').astype(np.float64)
    if tissue == 'muscle':
        N = np.load(ROOT / 'hMuscle/data/esm2_layer_01_t30_150M.npy', mmap_mode='r').shape[0]
        path_fmt = str(ROOT / 'hMuscle/data/esm2_layer_{:02d}_t30_150M.npy')
    else:
        N = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer01_t30_150M.npy',
                     mmap_mode='r').shape[0]
        path_fmt = str(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer{:02d}_t30_150M.npy')
    X640 = np.zeros((N, 640), dtype=np.float64)
    for L in range(30):
        arr = np.load(path_fmt.format(L + 1), mmap_mode='r')
        X640 += (arr.astype(np.float64) - mu[L]) / sd[L]
        del arr
    X640 /= 30.0
    return X640


def load_pairs(tissue):
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == tissue].reset_index(drop=True)
    return df


def cv_direction_and_auroc_binary(absD, y, genes):
    fold = gene_disjoint_folds(genes)
    oof_pred = np.zeros(len(y))
    directions = []
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if y[tr].sum() < 5 or (len(y[tr]) - y[tr].sum()) < 5:
            continue
        mu_, sd_ = absD[tr].mean(0), absD[tr].std(0) + 1e-8
        Xtr = (absD[tr] - mu_) / sd_
        Xte = (absD[te] - mu_) / sd_
        clf = LogisticRegression(max_iter=2000, C=0.05).fit(Xtr, y[tr])
        oof_pred[te] = clf.predict_proba(Xte)[:, 1]
        directions.append(clf.coef_[0] / sd_)  # back into raw-feature direction
    auroc = roc_auc_score(y, oof_pred)
    direction = np.mean(directions, axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return auroc, direction


def cv_direction_and_corr_continuous(absD, y, genes):
    fold = gene_disjoint_folds(genes)
    oof_pred = np.zeros(len(y))
    directions = []
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        mu_, sd_ = absD[tr].mean(0), absD[tr].std(0) + 1e-8
        Xtr = (absD[tr] - mu_) / sd_
        Xte = (absD[te] - mu_) / sd_
        reg = Ridge(alpha=10.0).fit(Xtr, y[tr])
        oof_pred[te] = reg.predict(Xte)
        directions.append(reg.coef_ / sd_)
    rho, _ = spearmanr(y, oof_pred)
    direction = np.mean(directions, axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return rho, direction


def analyze(tissue):
    print(f"\n{'='*70}\n[{tissue}] covariate direction extraction + 8-axis overlap\n{'='*70}")
    df = load_pairs(tissue)
    X640 = build_640_layermean(tissue)
    absD = np.abs(X640[df['long_idx'].to_numpy()] - X640[df['short_idx'].to_numpy()])
    genes = df['gene'].to_numpy()

    W = np.load(PCA_DIR / 'W_axes_8x640.npy')  # (8,640)
    W_unit = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    directions = {}
    for cov in BINARY_COVS:
        y = df[cov].to_numpy()
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            print(f"  [{cov}] insufficient class balance, skipped")
            continue
        auroc, direction = cv_direction_and_auroc_binary(absD, y, genes)
        directions[cov] = direction
        cos_to_axes = W_unit @ direction
        top_axis = np.argmax(np.abs(cos_to_axes))
        print(f"\n  [{cov}] out-of-fold AUROC={auroc:.3f} (n={len(y)}, pos={int(y.sum())})")
        print(f"    cosine to 8 axes: " + " ".join(f"ax{k}={c:+.3f}" for k, c in enumerate(cos_to_axes)))
        print(f"    top-|cos| axis: axis{top_axis} ({cos_to_axes[top_axis]:+.3f}); "
              f"sum|cos|^2 across 8 axes = {np.sum(cos_to_axes**2):.3f} "
              f"(1.0 = fully inside 8-axis subspace, 0 = fully orthogonal)")

    for cov in CONT_COVS:
        y = df[cov].to_numpy(dtype=np.float64)
        rho, direction = cv_direction_and_corr_continuous(absD, y, genes)
        directions[cov] = direction
        cos_to_axes = W_unit @ direction
        top_axis = np.argmax(np.abs(cos_to_axes))
        print(f"\n  [{cov}] out-of-fold Spearman rho={rho:.3f} (n={len(y)})")
        print(f"    cosine to 8 axes: " + " ".join(f"ax{k}={c:+.3f}" for k, c in enumerate(cos_to_axes)))
        print(f"    top-|cos| axis: axis{top_axis} ({cos_to_axes[top_axis]:+.3f}); "
              f"sum|cos|^2 across 8 axes = {np.sum(cos_to_axes**2):.3f}")

    print(f"\n  --- pairwise cosine similarity AMONG covariate directions ({tissue}) ---")
    covs = list(directions.keys())
    header = " " * 24 + "".join(f"{c[:10]:>12}" for c in covs)
    print(header)
    for c1 in covs:
        row = f"  {c1:<22}"
        for c2 in covs:
            row += f"{np.dot(directions[c1], directions[c2]):>12.3f}"
        print(row)
    return directions


def main():
    all_dirs = {}
    for tissue in ['muscle', 'brain']:
        all_dirs[tissue] = analyze(tissue)


if __name__ == '__main__':
    main()
