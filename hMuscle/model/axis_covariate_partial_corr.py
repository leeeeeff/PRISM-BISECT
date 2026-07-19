#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis_covariate_partial_corr.py
=================================
Plan B: does the introduction of the 5 severity covariates (size, domain_binary,
nterm_overlap, disorder_frac, resync_failure_binary) let us assign a biological
identity to any of the joint-PCA axes that reference-esm2-pca-axes-final.md
left weakly characterised (axis1 LRR/Ig, axis2 Pro-turn order, axis4
helix-charge, axis6 KRAB-ZNF/inverse-domain, axis7 acidic-helical) -- beyond
the two axes already firmly identified (axis3=domain, axis5=length)?

PRE-REGISTERED PREDICTION (S2, before running): axis3 and axis5 should show
the largest |partial correlation| with domain_binary and size respectively --
this is the expected baseline, not a new finding. What would be NEW: any of
axis1/2/4/6/7 showing a partial correlation with disorder_frac, nterm_overlap,
or resync_failure_binary that survives (a) partialling out size_z (the
standing length confound this whole project keeps re-discovering, cf.
finding-pooling-ablation-length-confound.md) and (b) a within-size-bin
permutation null (destroys the specific axis-covariate pairing while
preserving each pair's size-conditional Δaxis distribution).

REFUTATION: no axis outside {3,5} survives BH-correction against the null --
the "dark variance" trap from finding-axis3-extravar-undescribable.md would
then generalise (weakly-characterised axes stay uncharacterised even against
these richer covariates).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
COVARIATES = ['domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']
N_AXES = 8
N_PERM = 1000
N_SIZE_BINS = 10
SEED = 42

AXIS_LABEL = {
    0: 'axis0 (beta-sheet/TM, gene-level)',
    1: 'axis1 (LRR/Ig, weak)',
    2: 'axis2 (Pro-turn order, weak)',
    3: 'axis3 (domain, KNOWN)',
    4: 'axis4 (helix-charge, weak)',
    5: 'axis5 (length, KNOWN)',
    6: 'axis6 (KRAB-ZNF/inverse-domain, weak)',
    7: 'axis7 (acidic-helical, weak)',
}


def load_isoform_axes(tissue):
    Z = np.load(ROOT / f'reports/v20b_pca_interp/Z_{tissue}_Nx30x8.npy')  # (N, 30, 8)
    return Z.mean(axis=1)  # (N, 8), "axis scalar = 30-layer average" convention


def load_pairs(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    return df


def residualize(y, x):
    """OLS-residualize y on [1, x]."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def bh_correct(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(bh, 1.0)
    return out


def size_bin_permutation_null(delta_axis, r_cov, size_z, bin_idx, n_perm=N_PERM, seed=SEED):
    """Permute delta_axis WITHIN size-decile bins (preserves the size-conditional
    marginal of delta_axis; destroys the specific pairing with the covariate),
    then recompute partial correlation each time. `r_cov` (covariate residualized
    on size_z) and `bin_idx` (precomputed bin membership) are passed in since
    they don't change across permutations."""
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_perm)
    r_cov_std = (r_cov - r_cov.mean())
    r_cov_norm = np.linalg.norm(r_cov_std)
    for p in range(n_perm):
        perm_axis = delta_axis.copy()
        for idx in bin_idx:
            perm_axis[idx] = rng.permutation(delta_axis[idx])
        r_axis = residualize(perm_axis, size_z)
        r_axis_std = r_axis - r_axis.mean()
        null_rhos[p] = (r_axis_std @ r_cov_std) / (np.linalg.norm(r_axis_std) * r_cov_norm)
    return null_rhos


def analyze_tissue(tissue):
    axes = load_isoform_axes(tissue)  # (N, 8)
    df = load_pairs(tissue)
    long_idx = df['long_idx'].to_numpy()
    short_idx = df['short_idx'].to_numpy()
    size_z = df['size_z'].to_numpy()

    delta_axes = axes[long_idx] - axes[short_idx]  # (n_pairs, 8)

    bins = pd.qcut(size_z, N_SIZE_BINS, labels=False, duplicates='drop')
    bin_idx = [np.where(bins == b)[0] for b in np.unique(bins)]

    rows = []
    for cov_name in COVARIATES:
        cov = df[cov_name].to_numpy(dtype=np.float64)
        r_c = residualize(cov, size_z)
        for k in range(N_AXES):
            d = delta_axes[:, k]
            rho_size = np.corrcoef(d, size_z)[0, 1]
            r_d = residualize(d, size_z)
            partial_rho = np.corrcoef(r_d, r_c)[0, 1]

            null = size_bin_permutation_null(d, r_c, size_z, bin_idx)
            null_lo, null_hi = np.percentile(null, [2.5, 97.5])
            null_p = (np.abs(null) >= np.abs(partial_rho)).mean()

            rows.append({
                'tissue': tissue, 'axis': k, 'axis_label': AXIS_LABEL[k],
                'covariate': cov_name, 'partial_rho': partial_rho,
                'rho_with_size': rho_size, 'null_lo': null_lo, 'null_hi': null_hi,
                'null_p': null_p,
            })

    res = pd.DataFrame(rows)
    res['p_bh'] = bh_correct(res['null_p'].to_numpy())
    res['survives'] = (res['p_bh'] < 0.05) & (res['null_p'] < (1.0 / N_PERM) * 5)  # strict: essentially 0/1000
    return res


def main():
    all_res = []
    for tissue in ['muscle', 'brain']:
        print(f"\n=== {tissue} ===")
        res = analyze_tissue(tissue)
        all_res.append(res)
        for _, r in res.sort_values('partial_rho', key=np.abs, ascending=False).iterrows():
            flag = '  <-- SURVIVES' if r['survives'] else ''
            print(f"  {r['axis_label']:<32} x {r['covariate']:<22} "
                  f"partial_rho={r['partial_rho']:+.3f}  null=[{r['null_lo']:+.3f},{r['null_hi']:+.3f}]  "
                  f"null_p={r['null_p']:.4f}  p_BH={r['p_bh']:.4f}{flag}")

    out = pd.concat(all_res, ignore_index=True)
    out_path = ROOT / 'reports/severity_pairs/axis_covariate_partial_corr.tsv'
    out.to_csv(out_path, sep='\t', index=False)
    print(f"\n[Save] {out_path}")

    print("\n=== Surviving axis x covariate pairs (p_BH<0.05 AND null_p<0.005) ===")
    surv = out[out['survives']]
    if len(surv) == 0:
        print("  NONE -- refutation: no axis outside {3,5} newly characterised by these covariates.")
    else:
        print(surv[['tissue', 'axis_label', 'covariate', 'partial_rho', 'null_p', 'p_bh']].to_string(index=False))


if __name__ == '__main__':
    main()
