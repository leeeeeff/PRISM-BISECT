#!/usr/bin/env python3
"""
covariate_severity_auc.py
==========================
S2 discriminating test for "elevate severity-regression covariates to standalone
DR-AUC-style metrics": for each binary covariate, compute a SIZE-QUINTILE-MATCHED
AUC of severity_score discriminating covariate=1 vs covariate=0 pairs, mirroring
the exact evidentiary grammar of Domain-Ranking AUC (score + gene-cluster
bootstrap CI + null=0.5, decoupled from the size confound already found via VIF).

Pre-registered predictions (stated before running):
  - domain_binary: brain size-matched AUC should stay clearly >0.5 (replicates the
    already-published Domain-Direction Size-Matched finding: brain robust, muscle
    weak/collapsing). This is the sanity-check covariate.
  - resync_failure_binary (rho=0.56-0.58 with size -- the strongest confound found):
    if the manuscript's "resync is a real reading-frame-disruption axis" claim is
    genuine and not just "resync=long edit", muscle size-matched AUC should stay
    meaningfully >0.5. If it collapses to ~0.5 like domain did for muscle, the
    resync covariate's severity contribution is largely a length proxy.
  - nterm_overlap (rho=0.23-0.29 with size -- modest confound): expect a smaller
    but non-trivial size-matched attenuation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_BINS = 5
N_BOOT = 1000
SEED = 42


def auc_mw(scores, labels):
    """Mann-Whitney rank-biserial AUC: P(score_pos > score_neg)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, len(pos), len(neg)
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    r_pos = ranks[:len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    auc = (r_pos - n1 * (n1 + 1) / 2) / (n1 * n2)
    return auc, n1, n2


def size_matched_auc(df, covariate, n_bins=N_BINS, seed=SEED, n_boot=N_BOOT):
    edges = np.quantile(df['size'], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    df = df.copy()
    df['bin'] = pd.cut(df['size'], edges, labels=False, include_lowest=True)

    raw_auc, n_pos, n_neg = auc_mw(df['severity_score'].to_numpy(), df[covariate].to_numpy())

    bin_aucs, bin_weights = [], []
    for k in range(n_bins):
        sub = df[df['bin'] == k]
        a, np_, nn_ = auc_mw(sub['severity_score'].to_numpy(), sub[covariate].to_numpy())
        if not np.isnan(a) and np_ > 5 and nn_ > 5:
            bin_aucs.append(a)
            bin_weights.append(len(sub))
    bin_weights = np.array(bin_weights) / sum(bin_weights)
    matched_auc = float(np.dot(bin_aucs, bin_weights))

    # gene-cluster bootstrap on the size-matched estimator
    rng = np.random.default_rng(seed)
    genes = df['gene'].to_numpy()
    uniq = np.unique(genes)
    gene_rows = {g: np.where(genes == g)[0] for g in uniq}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        samp_genes = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([gene_rows[g] for g in samp_genes])
        sub = df.iloc[rows]
        vals, wts = [], []
        for k in range(n_bins):
            s2 = sub[sub['bin'] == k]
            a, np_, nn_ = auc_mw(s2['severity_score'].to_numpy(), s2[covariate].to_numpy())
            if not np.isnan(a) and np_ > 5 and nn_ > 5:
                vals.append(a); wts.append(len(s2))
        if not vals:
            boot[b] = np.nan
            continue
        wts = np.array(wts) / sum(wts)
        boot[b] = np.dot(vals, wts)
    ci = np.nanpercentile(boot, [2.5, 97.5])

    return dict(covariate=covariate, n_pos=n_pos, n_neg=n_neg,
                raw_auc=raw_auc, size_matched_auc=matched_auc,
                ci_lo=ci[0], ci_hi=ci[1])


def main():
    for tissue in ['muscle', 'brain']:
        df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
        print(f"\n=== {tissue} (n={len(df)}) ===")
        for cov in ['domain_binary', 'nterm_overlap', 'resync_failure_binary']:
            r = size_matched_auc(df, cov)
            print(f"{cov:<24} raw_AUC={r['raw_auc']:.3f}  size-matched_AUC={r['size_matched_auc']:.3f} "
                  f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  (n_pos={r['n_pos']}, n_neg={r['n_neg']})")


if __name__ == '__main__':
    main()
