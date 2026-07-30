#!/usr/bin/env python3
"""
dispersion_mechanism_test.py
===============================
S1/S2 test of the "exon-unit vs dispersed-edit" mechanism hypothesis (H1) for
why size_z's relationship to severity_score is stronger in muscle (r=0.351)
than brain (r=0.277) -- an already-validated finding (gap CI [0.047,0.100]
excl. 0, permutation p=0.001).

H1 (dispersion): muscle's large edits tend to be single contiguous blocks
(clean exon-unit swaps); brain's large edits of the same total size tend to
be split across MORE separate non-contiguous change intervals (n_intervals).
A single big contiguous block should perturb the ESM-2 embedding more
predictably/severely per residue-changed than the same total residue count
spread across many small scattered edits (each of which the mean-pooled/
layer-contrast representation may partially "average out"). This would make
size a better severity predictor in muscle than brain.

Pre-registered predictions (S2, before running):
  P1: at matched size decile, muscle n_intervals < brain n_intervals (muscle
      edits are more contiguous).
  P2: n_intervals (or its inverse, mean block length) has an independent
      partial association with severity_score beyond size_z alone.
  P3: adding n_intervals_z to a pooled regression with a tissue x size_z
      interaction term shrinks the interaction coefficient (i.e. dispersion
      explains part of, not all of, the tissue gap in size's severity
      coupling).
  H0 (null / H1 refuted): n_intervals does not differ by tissue at matched
      size, OR does not predict severity_score independent of size, OR does
      not attenuate the tissue x size interaction -- in which case some
      other mechanism (not edit contiguity) drives the gap.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
sys.path.insert(0, str(ROOT / 'hMuscle/model'))
import build_severity_pairs as bsp  # reuse exact sequence-loading + interval logic


def load_seqs_and_ids(tissue):
    if tissue == 'muscle':
        iso = np.load(ROOT / 'hMuscle/model/my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    else:
        iso = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    return iso, seqs


def compute_n_intervals(df, iso_ids, seqs):
    n_ivs = np.full(len(df), -1, dtype=int)
    missing = 0
    for row_i, (li, si) in enumerate(zip(df['long_idx'].to_numpy(), df['short_idx'].to_numpy())):
        long_id, short_id = iso_ids[li], iso_ids[si]
        if long_id not in seqs or short_id not in seqs:
            missing += 1
            continue
        long_s, short_s = seqs[long_id], seqs[short_id]
        ivs, changed, ops = bsp.changed_intervals(long_s, short_s)
        n_ivs[row_i] = len(ivs)
    print(f"  missing sequences: {missing}/{len(df)}")
    return n_ivs


def main():
    dfs = {}
    for tissue in ['muscle', 'brain']:
        print(f"\n[{tissue}] loading sequences...")
        iso_ids, seqs = load_seqs_and_ids(tissue)
        df = pd.read_csv(ROOT / f'reports/severity_pairs/{tissue}_severity_pairs_scored.tsv', sep='\t')
        print(f"  computing n_intervals for {len(df)} pairs...")
        df['n_intervals'] = compute_n_intervals(df, iso_ids, seqs)
        df = df[df['n_intervals'] > 0].copy()
        log_size = np.log1p(df['size'])
        df['size_z'] = (log_size - log_size.mean()) / log_size.std()
        df['block_len'] = df['size'] / df['n_intervals']
        df['tissue'] = tissue
        dfs[tissue] = df
        print(f"  n_intervals: mean={df['n_intervals'].mean():.2f}, median={df['n_intervals'].median():.0f}")

    out_path = ROOT / 'reports/severity_pairs'
    for t, df in dfs.items():
        df.to_csv(out_path / f'{t}_severity_pairs_with_dispersion.tsv', sep='\t', index=False)
        print(f"[saved] {out_path / f'{t}_severity_pairs_with_dispersion.tsv'} ({len(df)} rows)")

    # ---- P1: matched-size decile comparison of n_intervals ----
    print("\n=== P1: n_intervals by matched size decile (muscle vs brain) ===")
    pooled_size = pd.concat([dfs['muscle']['size'], dfs['brain']['size']])
    edges = np.quantile(pooled_size, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for tissue, df in dfs.items():
        d = df.copy()
        d['decile'] = pd.cut(d['size'], edges, labels=False, include_lowest=True)
        g = d.groupby('decile').agg(n=('n_intervals', 'size'),
                                     size_median=('size', 'median'),
                                     n_intervals_mean=('n_intervals', 'mean'),
                                     block_len_mean=('block_len', 'mean'))
        g['tissue'] = tissue
        rows.append(g)
    tab = pd.concat(rows).reset_index()
    piv = tab.pivot(index='decile', columns='tissue', values=['size_median', 'n_intervals_mean', 'block_len_mean'])
    print(piv.round(2).to_string())

    # ---- P2: does n_intervals predict severity beyond size? (partial corr) ----
    print("\n=== P2: partial correlation of n_intervals_z with severity_score, controlling size_z ===")
    for tissue, df in dfs.items():
        log_ni = np.log1p(df['n_intervals'])
        ni_z = (log_ni - log_ni.mean()) / log_ni.std()
        X = df[['size_z']].to_numpy()
        X1 = np.column_stack([np.ones(len(X)), X])
        beta_ni, *_ = np.linalg.lstsq(X1, ni_z, rcond=None)
        ni_res = ni_z - X1 @ beta_ni
        beta_sv, *_ = np.linalg.lstsq(X1, df['severity_score'].to_numpy(), rcond=None)
        sv_res = df['severity_score'].to_numpy() - X1 @ beta_sv
        r = np.corrcoef(ni_res, sv_res)[0, 1]
        raw_r = np.corrcoef(ni_z, df['severity_score'])[0, 1]
        print(f"{tissue}: raw r(n_intervals_z, severity)={raw_r:+.3f}  "
              f"partial r (size controlled)={r:+.3f}")

    # ---- P3: pooled regression with tissue x size interaction, with/without n_intervals ----
    print("\n=== P3: does n_intervals_z attenuate the tissue x size_z interaction? ===")
    pooled = pd.concat([dfs['muscle'], dfs['brain']], ignore_index=True)
    pooled['is_muscle'] = (pooled['tissue'] == 'muscle').astype(float)
    log_ni_p = np.log1p(pooled['n_intervals'])
    pooled['n_intervals_z'] = (log_ni_p - log_ni_p.mean()) / log_ni_p.std()
    y = pooled['severity_score'].to_numpy()

    # Model A: severity ~ size_z + is_muscle + size_z:is_muscle
    Xa = np.column_stack([np.ones(len(pooled)), pooled['size_z'], pooled['is_muscle'],
                           pooled['size_z'] * pooled['is_muscle']])
    beta_a, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid_a = y - Xa @ beta_a
    ss_res_a = (resid_a ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    print(f"Model A (no n_intervals): size_z:is_muscle interaction beta = {beta_a[3]:+.4f}  "
          f"R2={1 - ss_res_a / ss_tot:.4f}")

    # Model B: + n_intervals_z + n_intervals_z:is_muscle
    Xb = np.column_stack([np.ones(len(pooled)), pooled['size_z'], pooled['is_muscle'],
                           pooled['size_z'] * pooled['is_muscle'],
                           pooled['n_intervals_z'], pooled['n_intervals_z'] * pooled['is_muscle']])
    beta_b, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid_b = y - Xb @ beta_b
    ss_res_b = (resid_b ** 2).sum()
    print(f"Model B (+ n_intervals_z, +interaction): size_z:is_muscle interaction beta = {beta_b[3]:+.4f}  "
          f"n_intervals_z beta={beta_b[4]:+.4f}  n_intervals_z:is_muscle beta={beta_b[5]:+.4f}  "
          f"R2={1 - ss_res_b / ss_tot:.4f}")
    shrink = 1 - beta_b[3] / beta_a[3]
    print(f"\nInteraction coefficient shrinkage from adding n_intervals: {shrink*100:.1f}%")


if __name__ == '__main__':
    main()
