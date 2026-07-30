#!/usr/bin/env python3
"""
size_severity_tissue_gap.py
=============================
Validate the new finding that size_z (edit magnitude) has a stronger
relationship to severity_score in muscle than brain (raw r 0.351 vs 0.277).
Two things needed before this can be cited as an independent result:
  1. Gene-cluster bootstrap CI per tissue (is each tissue's r/AUC itself
     robust, not driven by a few large-gene clusters).
  2. A null for the TISSUE GAP specifically: pool all pairs from both
     tissues, randomly relabel tissue (preserving the 15,885/33,802 group
     sizes), recompute the gap under this "no true tissue difference" null,
     1000 times -- gives an empirical p-value for the observed gap.
  3. Paired bootstrap of the gap itself (resample genes within each tissue
     independently, recompute muscle_r - brain_r each time) -- CI on the gap.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_BOOT = 1000
SEED = 42


def load(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    df['tissue'] = tissue
    return df


def gene_cluster_bootstrap_r(df, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    genes = df['gene'].to_numpy()
    uniq = np.unique(genes)
    gene_rows = {g: np.where(genes == g)[0] for g in uniq}
    x = df['size_z'].to_numpy()
    y = df['severity_score'].to_numpy()
    boot = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([gene_rows[g] for g in samp])
        boot[b] = np.corrcoef(x[rows], y[rows])[0, 1]
    return boot


def main():
    dfm, dfb = load('muscle'), load('brain')

    r_m = np.corrcoef(dfm['size_z'], dfm['severity_score'])[0, 1]
    r_b = np.corrcoef(dfb['size_z'], dfb['severity_score'])[0, 1]
    print(f"Point estimate: muscle r={r_m:.4f}  brain r={r_b:.4f}  gap={r_m - r_b:+.4f}")

    boot_m = gene_cluster_bootstrap_r(dfm, seed=1)
    boot_b = gene_cluster_bootstrap_r(dfb, seed=2)
    ci_m = np.percentile(boot_m, [2.5, 97.5])
    ci_b = np.percentile(boot_b, [2.5, 97.5])
    print(f"\nPer-tissue gene-cluster bootstrap CI:")
    print(f"  muscle r CI = [{ci_m[0]:.4f}, {ci_m[1]:.4f}]")
    print(f"  brain  r CI = [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")

    # paired gap bootstrap: independently resample genes within each tissue, same iteration
    rng_m = np.random.default_rng(11)
    rng_b = np.random.default_rng(12)
    genes_m = dfm['gene'].to_numpy(); uniq_m = np.unique(genes_m)
    gene_rows_m = {g: np.where(genes_m == g)[0] for g in uniq_m}
    genes_b = dfb['gene'].to_numpy(); uniq_b = np.unique(genes_b)
    gene_rows_b = {g: np.where(genes_b == g)[0] for g in uniq_b}
    xm, ym = dfm['size_z'].to_numpy(), dfm['severity_score'].to_numpy()
    xb, yb = dfb['size_z'].to_numpy(), dfb['severity_score'].to_numpy()
    gap_boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sm = rng_m.choice(uniq_m, size=len(uniq_m), replace=True)
        rm = np.concatenate([gene_rows_m[g] for g in sm])
        sb = rng_b.choice(uniq_b, size=len(uniq_b), replace=True)
        rb = np.concatenate([gene_rows_b[g] for g in sb])
        gap_boot[b] = np.corrcoef(xm[rm], ym[rm])[0, 1] - np.corrcoef(xb[rb], yb[rb])[0, 1]
    gap_ci = np.percentile(gap_boot, [2.5, 97.5])
    print(f"\nGap (muscle_r - brain_r) gene-cluster bootstrap CI = [{gap_ci[0]:+.4f}, {gap_ci[1]:+.4f}]")
    print(f"  fraction of bootstrap gap <= 0: {(gap_boot <= 0).mean():.4f}")

    # permutation null: pool pairs, randomly relabel tissue preserving group sizes
    pooled_x = np.concatenate([xm, xb])
    pooled_y = np.concatenate([ym, yb])
    n_m, n_b = len(xm), len(xb)
    rng_perm = np.random.default_rng(SEED)
    null_gap = np.empty(N_BOOT)
    idx_all = np.arange(n_m + n_b)
    for i in range(N_BOOT):
        perm = rng_perm.permutation(idx_all)
        idx_m_perm = perm[:n_m]
        idx_b_perm = perm[n_m:]
        r_m_null = np.corrcoef(pooled_x[idx_m_perm], pooled_y[idx_m_perm])[0, 1]
        r_b_null = np.corrcoef(pooled_x[idx_b_perm], pooled_y[idx_b_perm])[0, 1]
        null_gap[i] = r_m_null - r_b_null
    obs_gap = r_m - r_b
    p_perm = (np.sum(np.abs(null_gap) >= abs(obs_gap)) + 1) / (N_BOOT + 1)
    print(f"\nPermutation null (tissue-label shuffle, pooled pairs): "
          f"null gap mean={null_gap.mean():+.4f}, sd={null_gap.std():.4f}")
    print(f"  observed gap={obs_gap:+.4f}  permutation p={p_perm:.4f}")


if __name__ == '__main__':
    main()
