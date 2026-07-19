#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_direction_size_matched.py
===================================
Follow-up to domain_direction_alignment.py: muscle's domain_diff<0 (decoupled)
subset showed a chance-level alignment rate (0.530, n.s.) while brain's fully
reversed (0.356, p=4.6e-8). Two explanations are indistinguishable from that
result alone:
  H_attenuation: the muscle decoupled subset is dominated by SMALL edits
                  (median size 120aa vs 691aa in the aligned subset), and small
                  edits generically produce a weak/noisy severity_score
                  regardless of domain direction -- the chance-level rate is a
                  power artefact, not evidence about domain-direction tracking.
  H_domain_weak: muscle genuinely has weaker (but real) domain-completeness
                  tracking than brain; the length-headwind in the decoupled
                  subset roughly cancels a real-but-modest domain signal.

PRE-REGISTERED DISCRIMINATING TEST (S2): post-stratify the ALIGNED
(domain_diff>0) subset onto the SAME size-quintile bins as the decoupled
subset (so both subsets have an identical size distribution by construction),
then recompute the aligned rate as the bin-count-weighted average.
  H_attenuation predicts: size-matched aligned rate collapses toward chance
      too (small edits are uninformative for EITHER direction) -- the
      decoupled subset's chance result would then carry no discriminating
      information about domain-tracking.
  H_domain_weak predicts: size-matched aligned rate stays clearly elevated
      (small aligned edits still mostly point the right way) -- so the
      decoupled subset's drop to chance is specifically about the
      length/domain conflict, not generic small-edit noise.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_BOOT = 1000
SEED = 42
N_BINS = 5


def load_domain_counts(tissue):
    path = 'domain_matrix_proper_test.npy' if tissue == 'muscle' else 'domain_matrix_brain_full.npy'
    dom = np.load(ROOT / 'hMuscle/results_isoform/features' / path)
    return dom.sum(axis=1).astype(np.int32)


def load_domain_binary_pairs(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    dom_count = load_domain_counts(tissue)
    df = df[df['domain_binary'] == 1].copy()
    df['domain_diff'] = dom_count[df['long_idx'].to_numpy()] - dom_count[df['short_idx'].to_numpy()]
    return df


def gene_cluster_bootstrap_weighted_rate(pos_bins, bin_edges, bin_weights, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the size-matched (bin-count-weighted) aligned rate by resampling
    genes within the aligned subset (weights = decoupled subset's bin proportions,
    held fixed across resamples -- only the aligned subset's composition is resampled)."""
    rng = np.random.default_rng(seed)
    genes = pos_bins['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}
    aligned = (pos_bins['severity_score'].to_numpy() > 0).astype(float)
    bin_id = pos_bins['bin'].to_numpy()

    rates = np.empty(n_boot)
    for b in range(n_boot):
        sampled_genes = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled_genes])
        r_aligned = aligned[rows]
        r_bin = bin_id[rows]
        weighted = 0.0
        total_w = 0.0
        for k in range(len(bin_weights)):
            mask = r_bin == k
            if mask.sum() == 0:
                continue
            weighted += bin_weights[k] * r_aligned[mask].mean()
            total_w += bin_weights[k]
        rates[b] = weighted / total_w if total_w > 0 else np.nan
    return rates


def analyze(tissue):
    df = load_domain_binary_pairs(tissue)
    pos = df[df['domain_diff'] > 0].copy()
    neg = df[df['domain_diff'] < 0].copy()

    print(f"\n=== {tissue} ===")
    print(f"aligned (domain_diff>0): n={len(pos)}, size median={pos['size'].median():.0f}, "
          f"raw rate={np.mean(pos['severity_score']>0):.3f}")
    print(f"decoupled (domain_diff<0): n={len(neg)}, size median={neg['size'].median():.0f}, "
          f"raw rate={np.mean(neg['severity_score']>0):.3f}")

    # quintile bin edges from the DECOUPLED subset's size distribution
    quantiles = np.linspace(0, 1, N_BINS + 1)
    bin_edges = np.quantile(neg['size'], quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    neg['bin'] = pd.cut(neg['size'], bin_edges, labels=False, include_lowest=True)
    pos['bin'] = pd.cut(pos['size'], bin_edges, labels=False, include_lowest=True)

    bin_weights = neg['bin'].value_counts(normalize=True).sort_index()
    bin_weights = bin_weights.reindex(range(N_BINS), fill_value=0.0).to_numpy()

    print(f"\nSize-quintile bins (edges from decoupled subset): {np.round(bin_edges[1:-1], 0)}")
    print(f"{'bin':<6}{'decoupled n':<14}{'decoupled rate':<18}{'aligned n':<12}{'aligned rate':<14}")
    for k in range(N_BINS):
        neg_k = neg[neg['bin'] == k]
        pos_k = pos[pos['bin'] == k]
        neg_rate = np.mean(neg_k['severity_score'] > 0) if len(neg_k) else float('nan')
        pos_rate = np.mean(pos_k['severity_score'] > 0) if len(pos_k) else float('nan')
        print(f"{k:<6}{len(neg_k):<14}{neg_rate:<18.3f}{len(pos_k):<12}{pos_rate:<14.3f}")

    pos_valid = pos.dropna(subset=['bin'])
    pos_valid = pos_valid[pos_valid['bin'].isin(range(N_BINS))]
    size_matched_rate = 0.0
    total_w = 0.0
    for k in range(N_BINS):
        pos_k = pos_valid[pos_valid['bin'] == k]
        if len(pos_k) == 0 or bin_weights[k] == 0:
            continue
        size_matched_rate += bin_weights[k] * np.mean(pos_k['severity_score'] > 0)
        total_w += bin_weights[k]
    size_matched_rate /= total_w

    boot = gene_cluster_bootstrap_weighted_rate(pos_valid, bin_edges, bin_weights)
    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])

    print(f"\nSize-MATCHED aligned rate (weighted by decoupled's size distribution): "
          f"{size_matched_rate:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]")
    print(f"  vs raw aligned rate (unmatched): {np.mean(pos['severity_score']>0):.3f}")
    print(f"  vs decoupled rate: {np.mean(neg['severity_score']>0):.3f}")

    n_success = int((pos_valid['severity_score'] > 0).sum())  # unweighted binomial, conservative
    return size_matched_rate, (ci_lo, ci_hi)


def main():
    for tissue in ['muscle', 'brain']:
        analyze(tissue)


if __name__ == '__main__':
    main()
