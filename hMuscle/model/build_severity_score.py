#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_severity_score.py
=========================
Adds the gene-disjoint out-of-fold coherence-projection severity SCORE to the
pair tables produced by build_severity_pairs.py. Construction matches the
2026-07-20 session's DV (memory finding-severity-regression-canonical-anchored.md):
per-isoform mean-pooled ESM-2 representations at layers 15 and 30, concatenated
(1280-dim); each pair's difference vector D = emb[long] - emb[short]; 5-fold
gene-disjoint CV, each fold scored by its dot product with the L2-normalised
mean D of the OTHER folds (so no pair contributes to its own scoring direction).

출력: reports/severity_pairs/{tissue}_severity_pairs_scored.tsv (adds `severity_score` column)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
SEED = 42


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def score_tissue(tissue, emb):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs.tsv', sep='\t')
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    D = emb[long_idx] - emb[short_idx]  # (n_pairs, 1280)

    fold = gene_disjoint_folds(df['gene'].to_numpy())
    scores = np.zeros(len(df), dtype=np.float64)

    for k in range(N_FOLDS):
        train_mask = fold != k
        test_mask = fold == k
        direction = D[train_mask].mean(axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else direction
        scores[test_mask] = D[test_mask] @ direction

    df['severity_score'] = scores
    out_path = ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv'
    df.to_csv(out_path, sep='\t', index=False)
    print(f"[{tissue}] scored {len(df)} pairs -> {out_path}")
    print(f"  severity_score: mean={scores.mean():.3f} std={scores.std():.3f} "
          f"min={scores.min():.3f} max={scores.max():.3f}")
    return df


def load_meanpool_L15_L30(tissue):
    if tissue == 'muscle':
        L15 = np.load(ROOT / 'hMuscle/data/esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/esm2_layer_30_t30_150M.npy').astype(np.float32)
    else:
        L15 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    return np.concatenate([L15, L30], axis=1)


def main():
    for tissue in ['muscle', 'brain']:
        emb = load_meanpool_L15_L30(tissue)
        print(f"[{tissue}] embedding matrix {emb.shape}")
        score_tissue(tissue, emb)


if __name__ == '__main__':
    main()
