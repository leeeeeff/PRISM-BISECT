#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_m3_position_check.py

M3 (devils-advocate MODERATE attack), diagnostic-first step: before building a
structure-matched null, check whether the premise even holds -- are REAL
changed-residue intervals positioned non-uniformly along the protein (e.g.
enriched near the N-terminus), while the existing scrambled control draws
a UNIFORM random start? If real intervals are actually close to uniformly
distributed, M3's concern is moot and no new null is needed (S0: verify the
problem exists before building the fix).

This reuses the already-recomputed fallback flags' underlying interval data
(CPU-only, no GPU) and additionally records each REAL (non-fallback) interval's
relative position (start / sequence_length) on whichever side (long/short) it
was computed for, separately by tissue.
"""
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
SEV = ROOT / 'reports/severity_pairs'
MODEL = ROOT / 'hMuscle/model'
MAXLEN = 1022


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


def analyze(tissue):
    import importlib.util
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_region.tsv', sep='\t')

    if tissue == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    else:
        iso = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')

    rel_pos_long, rel_pos_short = [], []
    nterm_flag_long, nterm_flag_short = [], []
    for _, r in df.iterrows():
        long_s = seqs[iso[int(r['long_idx'])]][:MAXLEN]
        short_s = seqs[iso[int(r['short_idx'])]][:MAXLEN]
        livs, sivs = opcode_intervals(long_s, short_s)
        if livs:
            start = min(u for u, v in livs)
            rel_pos_long.append(start / len(long_s))
            nterm_flag_long.append(start < 60)
        if sivs:
            start = min(u for u, v in sivs)
            rel_pos_short.append(start / len(short_s))
            nterm_flag_short.append(start < 60)

    rel_pos_long = np.array(rel_pos_long)
    rel_pos_short = np.array(rel_pos_short)
    print(f"\n=== M3 [{tissue}]: relative start-position of REAL (non-empty) intervals ===")
    print(f"  long-side (n={len(rel_pos_long)}): "
          f"mean={rel_pos_long.mean():.3f} median={np.median(rel_pos_long):.3f} "
          f"frac_in_first_10%={np.mean(rel_pos_long < 0.1):.3f} "
          f"frac_in_last_10%={np.mean(rel_pos_long > 0.9):.3f}")
    print(f"  short-side (n={len(rel_pos_short)}): "
          f"mean={rel_pos_short.mean():.3f} median={np.median(rel_pos_short):.3f} "
          f"frac_in_first_10%={np.mean(rel_pos_short < 0.1):.3f} "
          f"frac_in_last_10%={np.mean(rel_pos_short > 0.9):.3f}")
    print(f"  UNIFORM reference: frac_in_first_10%=0.100  frac_in_last_10%=0.100  mean=0.500")

    # decile histogram
    for side, arr in [('long', rel_pos_long), ('short', rel_pos_short)]:
        hist, edges = np.histogram(arr, bins=10, range=(0, 1))
        frac = hist / hist.sum()
        print(f"  {side}-side decile histogram (0=N-term .. 1=C-term): " +
              " ".join(f"{f:.3f}" for f in frac))


def main():
    for tissue in ['muscle', 'brain']:
        analyze(tissue)


if __name__ == '__main__':
    main()
