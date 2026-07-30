#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c3_verify.py

C3 (devils-advocate CRITICAL attack): is the nterm_overlap collapse under
region-pooling a real biological effect, or a tautological artifact of the
NTERM_FALLBACK=60 constant coinciding EXACTLY with nterm_overlap's own
definition threshold (edit touches first 60 residues)?

Mechanism under attack: for pure-deletion pairs (short-side interval empty),
region-pooling falls back to pooling the short isoform's first 60 residues
regardless of WHERE on the long isoform the deletion actually occurred. If
nterm_overlap=1 pairs are disproportionately fallback-triggered (plausible:
an N-terminal deletion is more likely to leave nothing on the short side to
align near that same region), the "region-pool destroys nterm signal" result
could be entirely a recipe artifact concentrated in fallback pairs, not a
general finding about localized pooling.

This script recomputes per-pair fallback flags (CPU-only, reuses the exact
opcode_intervals logic from the embedding-building scripts, no GPU/ESM-2
needed) and checks:
  1. Is P(fallback | nterm_overlap=1) > P(fallback | nterm_overlap=0)?
  2. Does the nterm_overlap OLS coefficient collapse ONLY in the
     fallback-triggered subset, while non-fallback pairs show mean-pool-like
     behavior under region-pool?
"""
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
SEV = ROOT / 'reports/severity_pairs'
MODEL = ROOT / 'hMuscle/model'
MAXLEN = 1022
NTERM_FALLBACK = 60


def opcode_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    livs, sivs = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if i2 > i1:
            livs.append((i1, i2))
        if j2 > j1:
            sivs.append((j1, j2))
    return livs, sivs


def recompute_fallback_flags(tissue, df):
    import importlib.util
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    if tissue == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    else:
        iso = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')

    fb_long, fb_short = [], []
    for _, r in df.iterrows():
        long_s = seqs[iso[int(r['long_idx'])]][:MAXLEN]
        short_s = seqs[iso[int(r['short_idx'])]][:MAXLEN]
        livs, sivs = opcode_intervals(long_s, short_s)
        fb_long.append(not livs)
        fb_short.append(not sivs)
    df['fallback_long'] = fb_long
    df['fallback_short'] = fb_short
    df['fallback_any'] = df['fallback_long'] | df['fallback_short']
    return df


def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def design_matrix(df):
    cols = ['size_z', 'domain_binary', 'nterm_overlap', 'disorder_frac', 'resync_failure_binary']
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(X)), X])
    return X, ['intercept'] + cols


def analyze(tissue):
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_region.tsv', sep='\t')
    log_size = np.log1p(df['size'])
    df['size_z'] = (log_size - log_size.mean()) / log_size.std()
    df = recompute_fallback_flags(tissue, df)

    print(f"\n=== C3 [{tissue}]: fallback rate by nterm_overlap ===")
    ct = pd.crosstab(df['nterm_overlap'], df['fallback_short'], normalize='index')
    print(ct)
    n1 = df[df['nterm_overlap'] == 1]
    n0 = df[df['nterm_overlap'] == 0]
    print(f"  P(fallback_short | nterm_overlap=1) = {n1['fallback_short'].mean():.3f} (n={len(n1)})")
    print(f"  P(fallback_short | nterm_overlap=0) = {n0['fallback_short'].mean():.3f} (n={len(n0)})")

    print(f"\n=== C3 [{tissue}]: nterm_overlap OLS coefficient, fallback-stratified ===")
    for label, sub in [('ALL', df), ('fallback_short=True', df[df['fallback_short']]),
                        ('fallback_short=False', df[~df['fallback_short']])]:
        if sub['nterm_overlap'].nunique() < 2 or len(sub) < 50:
            print(f"  [{label}] n={len(sub)} -- insufficient variation, skipped")
            continue
        X, cols = design_matrix(sub)
        idx = cols.index('nterm_overlap')
        b_mean = ols_fit(X, sub['severity_score'].to_numpy(dtype=np.float64))[idx]
        b_region = ols_fit(X, sub['severity_score_region'].to_numpy(dtype=np.float64))[idx]
        b_scram = ols_fit(X, sub['severity_score_scram'].to_numpy(dtype=np.float64))[idx]
        print(f"  [{label:<22}] n={len(sub):<7} beta_mean={b_mean:>8.3f}  "
              f"beta_region={b_region:>8.3f}  beta_scram={b_scram:>8.3f}")


def main():
    for tissue in ['muscle', 'brain']:
        analyze(tissue)


if __name__ == '__main__':
    main()
