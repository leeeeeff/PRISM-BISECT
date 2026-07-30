#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""investigate_disorder_region_tissue_asymmetry.py

Follow-up (Option B, 2026-07-21): disorder_frac showed a real region>scram
region-pool gain in muscle (beta diff -3.06, CI excludes 0) but NOT in brain
(beta diff -0.58, CI [-1.98,0.86] includes 0), while size_z/domain_binary/
nterm_overlap all replicated cleanly across tissue. Why does disorder_frac
alone fail to generalize?

WORKING HYPOTHESIS (S1, one of several -- tested here, not assumed):
region-pooling averages ONLY over the changed interval. For a SMALL interval,
that average is dominated by the amino-acid composition of a few residues --
if those residues happen to be an intrinsically-disordered linker/loop (common
in alternatively-spliced regions), the pooled vector's disorder signature is
concentrated, not diluted by surrounding structured content the way a large
interval's average would be. Muscle's edits are known to be more fragmented
(more, smaller intervals; n_intervals explained 89.2% of the size x tissue
mediation, [[approach-severity-covariate-surrogate-chain]]). PREDICTION (S2,
stated before running): if this mechanism is right, the region-scram
disorder_frac gap should be concentrated in the SMALLEST size quintile and
should be proportionally larger in muscle there specifically -- not a uniform
tissue-wide shift. A null/excluded alternative: disorder_frac's raw
distribution or its correlation with size simply differs by tissue for
unrelated compositional reasons (checked first, S0, before the interaction
hypothesis).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
SEV = ROOT / 'reports/severity_pairs'
N_BINS = 5


def load(tissue):
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_region.tsv', sep='\t')
    df['log_size'] = np.log1p(df['size'])
    df['gap_region_scram'] = df['severity_score_region'] - df['severity_score_scram']
    return df


def describe_disorder(df, tissue):
    print(f"\n--- {tissue}: disorder_frac / size descriptives (n={len(df)}) ---")
    print(f"  disorder_frac: mean={df['disorder_frac'].mean():.4f} std={df['disorder_frac'].std():.4f} "
          f"median={df['disorder_frac'].median():.4f}")
    print(f"  size: median={df['size'].median():.0f} mean={df['size'].mean():.1f} "
          f"(size_long_region median={df['size_long_region'].median():.0f})")
    rho, p = stats.spearmanr(df['disorder_frac'], df['log_size'])
    print(f"  Spearman(disorder_frac, log_size) = {rho:.3f} (p={p:.2e})")


def quintile_gap(df, tissue):
    """For each size quintile, OLS coefficient of disorder_frac on
    (severity_score_region - severity_score_scram), controlling nothing else
    (single-covariate slope within-bin, cluster-robust-free quick diagnostic --
    this is exploratory triage, NOT the confirmatory Part-A model)."""
    print(f"\n--- {tissue}: disorder_frac's region-scram gap by size quintile ---")
    bin_edges = np.quantile(df['size'], np.linspace(0, 1, N_BINS + 1))
    bin_edges[0] = -np.inf; bin_edges[-1] = np.inf
    df = df.copy()
    df['bin'] = pd.cut(df['size'], bin_edges, labels=False, include_lowest=True)
    print(f"{'bin':<5}{'n':<8}{'size range':<22}{'disorder~gap slope':<20}{'p':<12}{'gap mean':<12}")
    for k in range(N_BINS):
        sub = df[df['bin'] == k]
        if len(sub) < 20:
            continue
        slope, intercept, r, p, se = stats.linregress(sub['disorder_frac'], sub['gap_region_scram'])
        lo, hi = bin_edges[k], bin_edges[k + 1]
        rng_str = f"[{lo:.0f},{hi:.0f}]" if np.isfinite(lo) and np.isfinite(hi) else (
            f"[..,{hi:.0f}]" if not np.isfinite(lo) else f"[{lo:.0f},..]")
        print(f"{k:<5}{len(sub):<8}{rng_str:<22}{slope:<20.3f}{p:<12.2e}{sub['gap_region_scram'].mean():<12.3f}")


def main():
    dfs = {}
    for tissue in ['muscle', 'brain']:
        df = load(tissue)
        dfs[tissue] = df
        describe_disorder(df, tissue)
        quintile_gap(df, tissue)

    print("\n--- cross-tissue: fraction of pairs in the smallest-size quintile "
          "(pooled edges from each tissue's own distribution) ---")
    for tissue, df in dfs.items():
        small_frac = (df['size'] <= df['size'].quantile(0.2)).mean()
        print(f"  {tissue}: median size={df['size'].median():.0f}, "
              f"n_intervals proxy not available here -- see size_long_region/size_short_region "
              f"asymmetry as fallback: median size_short_region={df['size_short_region'].median():.0f}")


if __name__ == '__main__':
    main()
