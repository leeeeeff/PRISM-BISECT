#!/usr/bin/env python3
"""
dispersion_rigor_and_concentration.py
========================================
Option A: gene-cluster bootstrap CI + permutation null for the n_intervals
mediation finding (89.2% shrinkage of the size_z x is_muscle interaction).
Option C: concentration check -- is the muscle n_intervals excess (at large
size) broad-based across many genes, or driven by a handful of outlier
mega-genes (e.g. TTN/titin, a well-known outlier in muscle isoform data)?
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_BOOT = 1000
SEED = 42


def load():
    dfs = {}
    for t in ['muscle', 'brain']:
        df = pd.read_csv(ROOT / f'reports/severity_pairs/{t}_severity_pairs_with_dispersion.tsv', sep='\t')
        df['tissue'] = t
        df['gene_tq'] = t + '::' + df['gene'].astype(str)
        dfs[t] = df
    pooled = pd.concat(dfs.values(), ignore_index=True)
    pooled['is_muscle'] = (pooled['tissue'] == 'muscle').astype(float)
    # z-score n_intervals on the POOLED distribution (matches original dispersion_mechanism_test.py Model A/B fit)
    log_ni_p = np.log1p(pooled['n_intervals'])
    pooled['n_intervals_z'] = (log_ni_p - log_ni_p.mean()) / log_ni_p.std()
    return dfs, pooled


def fit_models(df):
    y = df['severity_score'].to_numpy()
    size_z, is_m, ni_z = df['size_z'].to_numpy(), df['is_muscle'].to_numpy(), df['n_intervals_z'].to_numpy()

    Xa = np.column_stack([np.ones(len(df)), size_z, is_m, size_z * is_m])
    beta_a, *_ = np.linalg.lstsq(Xa, y, rcond=None)

    Xb = np.column_stack([np.ones(len(df)), size_z, is_m, size_z * is_m, ni_z, ni_z * is_m])
    beta_b, *_ = np.linalg.lstsq(Xb, y, rcond=None)

    return beta_a[3], beta_b[3]  # interaction coefficients


def main():
    dfs, pooled = load()

    # ---- OPTION A: gene-cluster bootstrap CI ----
    print("=== OPTION A: gene-cluster bootstrap (n=1000) ===")
    obs_a, obs_b = fit_models(pooled)
    obs_shrink = 1 - obs_b / obs_a
    print(f"Observed: interaction_A={obs_a:+.4f}  interaction_B={obs_b:+.4f}  shrinkage={obs_shrink*100:.1f}%")

    rng = np.random.default_rng(SEED)
    uniq_clusters = pooled['gene_tq'].unique()
    cluster_rows = {g: np.where(pooled['gene_tq'].to_numpy() == g)[0] for g in uniq_clusters}

    boot_a, boot_b, boot_shrink = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        samp = rng.choice(uniq_clusters, size=len(uniq_clusters), replace=True)
        rows = np.concatenate([cluster_rows[g] for g in samp])
        sub = pooled.iloc[rows]
        a, b = fit_models(sub)
        boot_a[i], boot_b[i] = a, b
        boot_shrink[i] = 1 - b / a if abs(a) > 1e-6 else np.nan

    ci_a = np.percentile(boot_a, [2.5, 97.5])
    ci_b = np.percentile(boot_b, [2.5, 97.5])
    ci_shrink = np.nanpercentile(boot_shrink, [2.5, 97.5])
    print(f"interaction_A CI = [{ci_a[0]:+.4f}, {ci_a[1]:+.4f}]  (excludes 0: {ci_a[0] > 0 or ci_a[1] < 0})")
    print(f"interaction_B CI = [{ci_b[0]:+.4f}, {ci_b[1]:+.4f}]  (excludes 0: {ci_b[0] > 0 or ci_b[1] < 0})")
    print(f"shrinkage %  CI = [{ci_shrink[0]*100:.1f}%, {ci_shrink[1]*100:.1f}%]")

    # ---- permutation null: shuffle n_intervals_z across pairs, refit, see if shrinkage survives ----
    print("\n=== Permutation null: shuffle n_intervals_z (break pair-level link), refit shrinkage ===")
    rng2 = np.random.default_rng(SEED + 1)
    null_shrink = np.empty(N_BOOT)
    y = pooled['severity_score'].to_numpy()
    size_z, is_m = pooled['size_z'].to_numpy(), pooled['is_muscle'].to_numpy()
    ni_z_real = pooled['n_intervals_z'].to_numpy()
    Xa_fixed = np.column_stack([np.ones(len(pooled)), size_z, is_m, size_z * is_m])
    beta_a_fixed, *_ = np.linalg.lstsq(Xa_fixed, y, rcond=None)
    a_fixed = beta_a_fixed[3]
    for i in range(N_BOOT):
        ni_perm = rng2.permutation(ni_z_real)
        Xb = np.column_stack([np.ones(len(pooled)), size_z, is_m, size_z * is_m, ni_perm, ni_perm * is_m])
        beta_b, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        null_shrink[i] = 1 - beta_b[3] / a_fixed
    p_perm = (np.sum(null_shrink >= obs_shrink) + 1) / (N_BOOT + 1)
    print(f"null shrinkage: mean={null_shrink.mean()*100:.1f}%  sd={null_shrink.std()*100:.1f}%  "
          f"95th pct={np.percentile(null_shrink, 95)*100:.1f}%")
    print(f"observed shrinkage={obs_shrink*100:.1f}%  permutation p={p_perm:.4f}")

    # ---- OPTION C: concentration check (is muscle's large-size fragmentation gene-broad or outlier-driven?) ----
    print("\n=== OPTION C: concentration of muscle large-size n_intervals excess ===")
    dfm = dfs['muscle']
    large = dfm[dfm['size'] >= dfm['size'].quantile(0.7)].copy()  # top-30% size (~deciles 7-9)
    print(f"muscle large-size subset (top 30% by size): n={len(large)}, "
          f"n_genes={large['gene'].nunique()}, mean n_intervals={large['n_intervals'].mean():.2f}")

    gene_stats = large.groupby('gene').agg(n_pairs=('n_intervals', 'size'),
                                            sum_intervals=('n_intervals', 'sum'),
                                            mean_intervals=('n_intervals', 'mean'),
                                            max_size=('size', 'max')).sort_values('sum_intervals', ascending=False)
    total_intervals = gene_stats['sum_intervals'].sum()
    top10_share = gene_stats['sum_intervals'].head(10).sum() / total_intervals
    top1pct_n = max(1, int(0.01 * len(gene_stats)))
    top1pct_share = gene_stats['sum_intervals'].head(top1pct_n).sum() / total_intervals
    print(f"top-10 genes' share of total n_intervals (large-size subset): {top10_share*100:.1f}%")
    print(f"top-1% genes ({top1pct_n} genes) share: {top1pct_share*100:.1f}%")
    print(f"\nTop 10 genes by summed n_intervals in muscle large-size subset:")
    print(gene_stats.head(10).round(2).to_string())

    # exclude top-10 genes, recompute the muscle vs brain decile-8/9 gap to see if it survives
    top10_genes = set(gene_stats.head(10).index)
    dfm_excl = dfm[~dfm['gene'].isin(top10_genes)]
    dfb = dfs['brain']
    for lo_q, hi_q, label in [(0.7, 1.0, 'top-30%-size')]:
        m_sub = dfm_excl[dfm_excl['size'] >= dfm_excl['size'].quantile(lo_q)]
        b_sub = dfb[dfb['size'] >= dfb['size'].quantile(lo_q)]
        print(f"\nAfter excluding top-10 genes -- {label}: "
              f"muscle mean n_intervals={m_sub['n_intervals'].mean():.2f} (n={len(m_sub)}) "
              f"vs brain={b_sub['n_intervals'].mean():.2f} (n={len(b_sub)})")


if __name__ == '__main__':
    main()
