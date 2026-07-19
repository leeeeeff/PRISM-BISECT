#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis_occlusion_usage.py
=========================
Follow-up to axis_covariate_partial_corr.py (Plan B). That script only measured
ENCODING: does axis0's/axis6's PCA coordinate covary with disorder_frac/
resync_failure_binary. It says nothing about USAGE: is that axis direction
causally load-bearing for the coherence-projection severity_score's own
ability to track those covariates, or would severity_score track them equally
well without it (i.e. the correlation lives elsewhere in the 1280-dim space
the 8-axis PCA doesn't capture)?

Occlusion protocol (matches reference-esm2-pca-axes-final.md's
occlusion_retrain.py convention: remove the axis direction from the
representation BEFORE re-deriving the downstream quantity, not just at
test-time):
  1. Per layer (L15, L30), z-score the mean-pooled embedding, project out the
     axis direction (w_k, L2-normalised), reconstruct back to raw scale.
  2. Re-run the FULL gene-disjoint 5-fold coherence-direction fit on the
     occluded embeddings (the direction itself is re-derived from occluded
     data, not just evaluated on it -- "retrain", not "test-time only").
  3. Compare corr(severity_score, covariate | size_z) BEFORE vs AFTER
     occlusion of the specific axis under test.
  4. Firming null: repeat with 20 random unit directions (same 640-dim space,
     applied identically to L15/L30) to see whether the real axis's effect on
     the correlation exceeds what removing ANY generic direction does.

PRE-REGISTERED PREDICTION (S2): if axis0/axis6 are causally load-bearing
("encoded+used", like axis3 for domain-ranking), occluding them should shrink
|corr(severity_score, covariate)| by MORE than the random-direction null
range. If severity_score's covariate-tracking is robust to their removal
(within the random-null band), the Plan B correlation was real encoding but
NOT specifically routed through axis0/axis6 -- the coherence axis gets the
same information from elsewhere in the 1280-dim space ("encoded-only" at the
8-axis level, i.e. redundantly encoded).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_FOLDS = 5
N_RANDOM = 20
SEED = 42

TESTS = [
    {'axis': 0, 'covariate': 'disorder_frac'},
    {'axis': 6, 'covariate': 'resync_failure_binary'},
    # follow-up: three cross-tissue-replicated (|partial_rho|>=0.10 both tissues)
    # encoding-level pairs from axis_covariate_partial_corr.py that were never
    # occlusion-tested for causal usage.
    {'axis': 1, 'covariate': 'disorder_frac'},
    {'axis': 3, 'covariate': 'resync_failure_binary'},
    {'axis': 4, 'covariate': 'resync_failure_binary'},
]


def load_layer_stats():
    W = np.load(ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy')  # (8, 640)
    mu = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_mu.npy')  # (30, 640)
    sd = np.load(ROOT / 'reports/v20b_pca_interp/layer_stats_sd.npy')  # (30, 640)
    return W, mu, sd


def load_raw_layer(tissue, layer):
    if tissue == 'muscle':
        return np.load(ROOT / f'hMuscle/data/esm2_layer_{layer:02d}_t30_150M.npy').astype(np.float32)
    else:
        return np.load(ROOT / f'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer{layer:02d}_t30_150M.npy').astype(np.float32)


def occlude(raw, mu_l, sd_l, direction):
    """Project `direction` (unit, 640-dim) out of the per-layer z-scored embedding,
    reconstruct back to raw scale."""
    z = (raw - mu_l) / sd_l
    z_occ = z - np.outer(z @ direction, direction)
    return z_occ * sd_l + mu_l


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def coherence_severity_score(emb, long_idx, short_idx, fold):
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
    r_s = residualize(score, size_z)
    r_c = residualize(covariate, size_z)
    return np.corrcoef(r_s, r_c)[0, 1]


def run_tissue(tissue):
    print(f"\n{'='*70}\n{tissue}\n{'='*70}")
    W, mu, sd = load_layer_stats()
    L15_raw = load_raw_layer(tissue, 15)
    L30_raw = load_raw_layer(tissue, 30)

    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    log_size = np.log1p(df['size'])
    size_z = ((log_size - log_size.mean()) / log_size.std()).to_numpy()
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    fold = gene_disjoint_folds(df['gene'].to_numpy())

    # baseline (unoccluded)
    emb_orig = np.concatenate([L15_raw, L30_raw], axis=1)
    score_orig = coherence_severity_score(emb_orig, long_idx, short_idx, fold)

    rng = np.random.default_rng(SEED)
    random_dirs = rng.normal(size=(N_RANDOM, 640))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

    for test in TESTS:
        axis_k = test['axis']
        cov_name = test['covariate']
        cov = df[cov_name].to_numpy(dtype=np.float64)

        corr_orig = partial_corr(score_orig, cov, size_z)

        w_k = W[axis_k] / np.linalg.norm(W[axis_k])
        L15_occ = occlude(L15_raw, mu[14], sd[14], w_k)
        L30_occ = occlude(L30_raw, mu[29], sd[29], w_k)
        emb_occ = np.concatenate([L15_occ, L30_occ], axis=1)
        score_occ = coherence_severity_score(emb_occ, long_idx, short_idx, fold)
        corr_occ_real = partial_corr(score_occ, cov, size_z)
        delta_real = corr_occ_real - corr_orig

        null_deltas = np.empty(N_RANDOM)
        for i in range(N_RANDOM):
            d = random_dirs[i]
            L15_r = occlude(L15_raw, mu[14], sd[14], d)
            L30_r = occlude(L30_raw, mu[29], sd[29], d)
            emb_r = np.concatenate([L15_r, L30_r], axis=1)
            score_r = coherence_severity_score(emb_r, long_idx, short_idx, fold)
            corr_r = partial_corr(score_r, cov, size_z)
            null_deltas[i] = corr_r - corr_orig

        null_lo, null_hi = np.percentile(null_deltas, [2.5, 97.5])
        z = (delta_real - null_deltas.mean()) / null_deltas.std() if null_deltas.std() > 0 else np.nan
        rel_effect = delta_real / abs(corr_orig) if abs(corr_orig) > 1e-6 else np.nan

        print(f"\n  axis{axis_k} occlusion -> corr(severity_score, {cov_name} | size_z)")
        print(f"    original corr           = {corr_orig:+.4f}")
        print(f"    occluded (real axis{axis_k}) = {corr_occ_real:+.4f}  (delta={delta_real:+.4f}, "
              f"rel={rel_effect:+.1%})")
        print(f"    occluded (20 random dir): delta mean={null_deltas.mean():+.4f} "
              f"CI=[{null_lo:+.4f},{null_hi:+.4f}]  z={z:+.2f}")
        # statistical AND practical significance -- neither alone is sufficient
        # (cf. approach-axis-encoding-vs-usage-occlusion.md self-correction)
        outside_null = not (null_lo <= delta_real <= null_hi)
        practically_meaningful = abs(rel_effect) >= 0.10 if not np.isnan(rel_effect) else False
        if outside_null and practically_meaningful:
            verdict = f"EXCEEDS random-null AND rel_effect={rel_effect:+.1%} -> causally load-bearing (encoded+used)"
        elif outside_null and not practically_meaningful:
            verdict = f"statistically outside null but rel_effect only {rel_effect:+.1%} -> encoded-only (trivial effect size)"
        else:
            verdict = "WITHIN random-null -> redundantly encoded elsewhere (encoded-only at 8-axis level)"
        print(f"    verdict: {verdict}")


def main():
    for tissue in ['muscle', 'brain']:
        run_tissue(tissue)


if __name__ == '__main__':
    main()
