#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c3_severity_score_check.py

THE decisive question: does the manuscript's ORIGINAL severity_score
(coherence-projection: direction = mean(train-fold D), D = long-short mean-
pool embedding diff, gene-disjoint 5-fold CV) suffer from the SAME flaw just
found in today's internal-edit/N-terminal self-consistency tests -- i.e., is
the trained "coherence direction" itself just reflecting a gene-independent
population bias in D's marginal distribution, rather than genuine gene-
specific structure?

DECISIVE TEST: rebuild severity_score TWICE on the full muscle canonical-
anchored population --
  (a) REAL: gene-disjoint 5-fold CV using the ACTUAL gene assignment (must
      reproduce the exact stored severity_score column, sanity-checked).
  (b) PERMUTED: gene-disjoint 5-fold CV using a RANDOMLY PERMUTED gene
      assignment for fold-splitting purposes only (destroys which pairs
      share a gene when deciding folds and training each fold's direction),
      but the D vectors and covariates stay attached to their real pairs.
Then refit the SAME 5-covariate OLS (severity_score ~ size_z + domain_binary
+ nterm_overlap + disorder_frac + resync_failure_binary) on BOTH severity
scores and compare R^2 and domain_binary's coefficient (cluster-robust SE,
using the REAL gene clusters for the SE regardless of which severity_score
is used, since SE-clustering and direction-training are separate questions).
  H_bias: permuted-gene severity_score gives similar R^2/domain_binary beta
    to the real one -> the manuscript's coherence-projection framework does
    NOT actually depend on genuine gene-specific structure, a MAJOR problem.
  H_real: permuted-gene version's R^2/beta collapse toward null/noise level
    -> severity_score genuinely needs real gene identity, framework is sound.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
N_FOLDS = 5
SEED = 42
N_REPS = 30  # permuted-gene reps (each is a full 5-fold refit + OLS)
rng = np.random.default_rng(777)

spec = importlib.util.spec_from_file_location('sr', MODEL / 'severity_regression.py')
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def score_with_fold_assignment(D, fold):
    scores = np.zeros(len(D))
    for k in range(N_FOLDS):
        train_mask = fold != k
        direction = D[train_mask].mean(axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else direction
        scores[fold == k] = D[fold == k] @ direction
    return scores


def main():
    df = pd.read_csv(SEV / 'muscle_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'muscle'].reset_index(drop=True)
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    D = emb[df['long_idx'].to_numpy()] - emb[df['short_idx'].to_numpy()]
    real_genes = df['gene'].to_numpy()

    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    X, cols = sr.design_matrix(df, with_tissue=False)
    idx_domain = cols.index('domain_binary')

    # --- (a) REAL: reproduce stored severity_score exactly ---
    fold_real = gene_disjoint_folds(real_genes)
    score_real = score_with_fold_assignment(D, fold_real)
    assert np.allclose(score_real, df['severity_score'].to_numpy(), atol=1e-3), \
        "failed to reproduce stored severity_score -- fold/direction mismatch"
    print("[check] reproduced stored severity_score exactly.\n")

    beta_r, resid_r, r2_r = sr.ols_fit(X, score_real)
    se_r = sr.cluster_robust_se(X, resid_r, real_genes)
    print(f"[REAL gene assignment] R^2={r2_r:.4f}  domain_binary beta={beta_r[idx_domain]:.3f} "
          f"t={beta_r[idx_domain]/se_r[idx_domain]:.2f}")

    # --- (b) PERMUTED: fold assignment uses SCRAMBLED gene labels ---
    print(f"\n[PERMUTED gene assignment for fold-splitting, {N_REPS} reps]")
    r2_perm = np.empty(N_REPS)
    beta_perm = np.empty(N_REPS)
    for i in range(N_REPS):
        perm_genes = rng.permutation(real_genes)
        fold_perm = gene_disjoint_folds(perm_genes, seed=1000 + i)
        # NOTE: fold assignment now keyed to SHUFFLED gene labels, but applied
        # to the SAME (real) rows -- i.e., which fold a ROW lands in no longer
        # respects its TRUE gene's cluster membership.
        score_perm = score_with_fold_assignment(D, fold_perm)
        beta_p, resid_p, r2_p = sr.ols_fit(X, score_perm)
        # cluster-robust SE still uses REAL gene id (only the training/fold
        # split for direction is corrupted, not the covariate/outcome pairing)
        r2_perm[i] = r2_p
        beta_perm[i] = beta_p[idx_domain]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_REPS} reps done", flush=True)

    print(f"\n  permuted R^2: mean={r2_perm.mean():.4f} CI=[{np.percentile(r2_perm,2.5):.4f},"
          f"{np.percentile(r2_perm,97.5):.4f}]   (real R^2={r2_r:.4f})")
    print(f"  permuted domain_binary beta: mean={beta_perm.mean():.3f} "
          f"CI=[{np.percentile(beta_perm,2.5):.3f},{np.percentile(beta_perm,97.5):.3f}]   "
          f"(real beta={beta_r[idx_domain]:.3f})")

    r2_lo, r2_hi = np.percentile(r2_perm, [2.5, 97.5])
    beta_lo, beta_hi = np.percentile(beta_perm, [2.5, 97.5])
    verdict_r2 = 'REAL beats permuted-gene null' if r2_r > r2_hi else 'INDISTINGUISHABLE from permuted-gene null'
    verdict_beta = 'REAL beats permuted-gene null' if beta_r[idx_domain] > beta_hi or beta_r[idx_domain] < beta_lo \
        else 'INDISTINGUISHABLE from permuted-gene null'
    print(f"\n  => R^2 verdict: {verdict_r2}")
    print(f"  => domain_binary beta verdict: {verdict_beta}")


if __name__ == '__main__':
    main()
