#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis0_disorder_layer_profile.py
==================================
Plan C, narrowed (axis6-resync failed the usage gate in
axis_occlusion_usage.py -- only axis0-disorder cleared it, so only that pair
is worth layer-decomposing; occam gate from the Plan B/A discussion).

For each of the 30 ESM-2 layers INDEPENDENTLY (not the L15+L30 concat used by
the main severity pipeline), computes a per-layer coherence-projection score
and its correlation with disorder_frac (size_z partialled out), with and
without axis0 occluded from that layer. The layer where axis0-occlusion does
the most damage to the correlation is where axis0's causal disorder-tracking
role is concentrated -- analogous to the project's established mid-network
domain-signal peak (L17, axis3) but for a different axis/covariate pair.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
SEED = 42
AXIS = 0
COVARIATE = 'disorder_frac'


def load_layer_stats():
    W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')
    mu = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy')
    sd = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')
    return W, mu, sd


def load_raw_layer(tissue, layer):
    if tissue == 'muscle':
        return np.load(ROOT / f'hMuscle/data/esm2_layer_{layer:02d}_t30_150M.npy').astype(np.float32)
    return np.load(ROOT / f'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer{layer:02d}_t30_150M.npy').astype(np.float32)


def occlude(raw, mu_l, sd_l, direction):
    z = (raw - mu_l) / sd_l
    z_occ = z - np.outer(z @ direction, direction)
    return z_occ * sd_l + mu_l


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def coherence_score(emb, long_idx, short_idx, fold):
    D = emb[long_idx] - emb[short_idx]
    scores = np.zeros(len(D))
    for k in range(N_FOLDS):
        train_mask = fold != k
        test_mask = fold == k
        direction = D[train_mask].mean(axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else direction
        scores[test_mask] = D[test_mask] @ direction
    return scores


def residualize(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr(score, covariate, size_z):
    return np.corrcoef(residualize(score, size_z), residualize(covariate, size_z))[0, 1]


def run_tissue(tissue):
    W, mu, sd = load_layer_stats()
    w0 = W[AXIS] / np.linalg.norm(W[AXIS])

    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    log_size = np.log1p(df['size'])
    size_z = ((log_size - log_size.mean()) / log_size.std()).to_numpy()
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())
    cov = df[COVARIATE].to_numpy(dtype=np.float64)

    rows = []
    for layer in range(1, 31):
        raw = load_raw_layer(tissue, layer)
        score_orig = coherence_score(raw, long_idx, short_idx, fold)
        corr_orig = partial_corr(score_orig, cov, size_z)

        occ = occlude(raw, mu[layer - 1], sd[layer - 1], w0)
        score_occ = coherence_score(occ, long_idx, short_idx, fold)
        corr_occ = partial_corr(score_occ, cov, size_z)

        delta = corr_occ - corr_orig
        rel = delta / corr_orig if abs(corr_orig) > 1e-6 else np.nan
        rows.append({'tissue': tissue, 'layer': layer, 'corr_orig': corr_orig,
                      'corr_occ': corr_occ, 'delta': delta, 'rel_effect': rel})
        print(f"  [{tissue}] layer {layer:02d}: corr_orig={corr_orig:+.4f} "
              f"corr_occ={corr_occ:+.4f} delta={delta:+.4f} rel={rel:+.1%}" if not np.isnan(rel)
              else f"  [{tissue}] layer {layer:02d}: corr_orig={corr_orig:+.4f} corr_occ={corr_occ:+.4f}",
              flush=True)
    return pd.DataFrame(rows)


def main():
    all_res = []
    for tissue in ['muscle', 'brain']:
        print(f"\n=== {tissue}: axis0 occlusion effect on disorder_frac tracking, per layer ===")
        res = run_tissue(tissue)
        all_res.append(res)
        peak = res.loc[res['rel_effect'].abs().idxmax()]
        print(f"\n  PEAK relative effect: layer {int(peak['layer'])} "
              f"(rel_effect={peak['rel_effect']:+.1%}, corr_orig={peak['corr_orig']:+.4f})")

    out = pd.concat(all_res, ignore_index=True)
    out_path = ROOT / 'reports/severity_pairs/axis0_disorder_layer_profile.tsv'
    out.to_csv(out_path, sep='\t', index=False)
    print(f"\n[Save] {out_path}")


if __name__ == '__main__':
    main()
