#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_direction_alignment.py
===============================
Plan A (2026-07-20 session follow-up): does the mean-pool coherence-projection
severity_score's SIGN track domain-COMPLETENESS direction, or merely the
long-minus-short LENGTH direction it was trained on?

domain_binary=1 pairs (Pfam presence differs between long/short) are not, by
themselves, a clean test: canonical (usually the longer, more complete
isoform) drives both "long" and "more domains" together most of the time, so
naive severity_score>0 agreement would be confounded with the length-artefact
mechanism already confirmed elsewhere in this project
(finding-pooling-ablation-length-confound.md).

Natural decoupling subset: among domain_binary=1 pairs, split by
domain_diff = domain_count(long) - domain_count(short):
  domain_diff > 0  (majority) : long is BOTH longer AND more domain-complete
                                  -- length and domain-direction are confounded here
  domain_diff < 0  (minority) : long is longer but LESS domain-complete
                                  -- length and domain-direction are DECOUPLED here

PRE-REGISTERED PREDICTION (S2, before running):
  H_domain (coherence axis tracks domain-completeness, as DR-AUC/severity
  regression's domain_binary coefficient implies): rate(severity_score>0) in
  the domain_diff<0 subset should be LOWER than in domain_diff>0 (ideally
  <50%, i.e. sign flips to track domain loss, not length).
  H_length (coherence axis is actually just a length axis, as the
  pooling-ablation length-confound chain found for max-pooling): rate stays
  high (~same as domain_diff>0 subset) regardless of domain_diff sign --
  REFUTES the "shared direction organizer" claim from the muscle/brain
  discussion this analysis follows up on.

Gene-cluster bootstrap CI (resample genes with replacement, keep all their
rows) + binomial test vs 0.5 chance, per tissue and pooled.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
N_BOOT = 1000
SEED = 42


def load_domain_counts(tissue):
    if tissue == 'muscle':
        dom = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy')
    else:
        dom = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_brain_full.npy')
    return dom.sum(axis=1).astype(np.int32)


def load_pairs(tissue):
    df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
    dom_count = load_domain_counts(tissue)
    df = df[df['domain_binary'] == 1].copy()
    df['domain_diff'] = dom_count[df['long_idx'].to_numpy()] - dom_count[df['short_idx'].to_numpy()]
    return df


def gene_cluster_bootstrap_rate(df, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    genes = df['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}
    aligned = (df['severity_score'].to_numpy() > 0).astype(float)

    rates = np.empty(n_boot)
    for b in range(n_boot):
        sampled_genes = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled_genes])
        rates[b] = aligned[rows].mean()
    return rates


def summarize(label, df):
    n = len(df)
    if n == 0:
        print(f"{label}: n=0, skipped")
        return None
    aligned = (df['severity_score'] > 0)
    rate = aligned.mean()
    boot = gene_cluster_bootstrap_rate(df)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    n_success = int(aligned.sum())
    binom_p = stats.binomtest(n_success, n, 0.5).pvalue
    print(f"{label:<40} n={n:>6}  rate={rate:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"binom_p={binom_p:.2e}  n_genes={df['gene'].nunique()}")
    return {'label': label, 'n': n, 'rate': rate, 'ci': (ci_lo, ci_hi), 'binom_p': binom_p}


def main():
    for tissue in ['muscle', 'brain']:
        df = load_pairs(tissue)
        print(f"\n=== {tissue} (domain_binary=1 pairs, n={len(df)}) ===")
        pos = df[df['domain_diff'] > 0]
        neg = df[df['domain_diff'] < 0]
        tie = df[df['domain_diff'] == 0]
        print(f"  domain_diff>0 (long more complete, length&domain aligned): {len(pos)} "
              f"({100*len(pos)/len(df):.1f}%)")
        print(f"  domain_diff<0 (long longer but LESS complete, decoupled):   {len(neg)} "
              f"({100*len(neg)/len(df):.1f}%)")
        print(f"  domain_diff==0 (presence swap, tied count):                {len(tie)} "
              f"({100*len(tie)/len(df):.1f}%)")
        r_pos = summarize(f"  {tissue} domain_diff>0 (aligned subset)", pos)
        r_neg = summarize(f"  {tissue} domain_diff<0 (decoupled subset)", neg)

    print("\n=== Pooled ===")
    df_all = pd.concat([load_pairs('muscle'), load_pairs('brain')], ignore_index=True)
    pos = df_all[df_all['domain_diff'] > 0]
    neg = df_all[df_all['domain_diff'] < 0]
    r_pos = summarize("  pooled domain_diff>0 (aligned subset)", pos)
    r_neg = summarize("  pooled domain_diff<0 (decoupled subset)", neg)

    if r_pos and r_neg:
        diff = r_pos['rate'] - r_neg['rate']
        print(f"\n  rate(domain_diff>0) - rate(domain_diff<0) = {diff:+.3f}")
        print("  H_domain predicts this gap is large and positive (decoupled-subset rate "
              "drops toward/below 0.5); H_length predicts this gap is ~0 (both subsets high).")


if __name__ == '__main__':
    main()
