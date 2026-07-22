#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nondomain_mechanism_fractions.py

Measure (not schematise) the sub-mechanism composition of NON-DOMAIN within-gene edits, so the
cascade Panel A widths become quantitative. Fast: needs only the severity-pair covariates
(domain/nterm/disorder) + protein sequences (for a specific-SLiM scan) — NO embeddings.

Non-domain = domain_binary==0. We report:
  - a mutually-EXCLUSIVE 3-way partition for the flow widths (priority: N-terminal targeting >
    disorder-dominant > structured-internal), each mapped to the bottleneck where its signal is lost;
  - the pairwise OVERLAPS (the C6 caveat: the underlying categories are non-exclusive);
  - a specific-SLiM-bearing fraction (subset of structured-internal) as a lower bound.
Brain primary; muscle reported too.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
import re
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022

# specific short-linear-motif regexes (moderate specificity; a lower bound on motif-bearing edits)
SLIMS = {
    'NLS_mono': re.compile(r'K[KR][KR]'),
    'NES_LeuRich': re.compile(r'[LMVIF].{2,3}[LMVIF].{2,3}[LMVIF].[LMVIF]'),
    'RGD': re.compile(r'RGD'),
    'KFERQ_CMA': re.compile(r'[KR][FLIVQN][KR][QN]|[QN][KR][FLIV][KR]'),
    'SH3_PxxP': re.compile(r'[RKY]..P..P|P..P.[RK]'),
}


def changed_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal' and i2 > i1:
            ivs.append((i1, i2))
    return ivs


def load(tissue):
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df = df[(df['tissue'] == tissue) & (df['domain_binary'] == 0)].reset_index(drop=True)
    if tissue == 'brain':
        iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    else:
        iso = [s.decode() if isinstance(s, bytes) else str(s)
               for s in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    return df, iso, seqs


def slim_in_changed(ls, ivs):
    for (u, v) in ivs:
        seg = ls[u:min(v, len(ls))]
        for rx in SLIMS.values():
            if rx.search(seg):
                return True
    return False


def analyze(tissue):
    df, iso, seqs = load(tissue)
    nterm = (df['nterm_overlap'] == 1).to_numpy()
    disorder = (df['disorder_frac'] > 0.5).to_numpy()
    n = len(df)
    slim = np.zeros(n, dtype=bool)
    for k, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs:
            continue
        ls, ss = seqs[lid][:MAXLEN], seqs[sid][:MAXLEN]
        if ls == ss:
            continue
        slim[k] = slim_in_changed(ls, changed_intervals(ls, ss))
    # mutually-exclusive partition for flow widths (priority order)
    p_nterm = nterm
    p_dis = (~nterm) & disorder
    p_struct = (~nterm) & (~disorder)
    print(f"\n[{tissue.upper()}] non-domain within-gene pairs: n={n}")
    print(f"  EXCLUSIVE partition (flow widths):")
    print(f"    N-terminal targeting (-> partial B4)        : {p_nterm.mean():.1%}")
    print(f"    disorder-dominant   (-> dies B3, un-anchored): {p_dis.mean():.1%}")
    print(f"    structured-internal (-> SLiM B2 / comp B3)  : {p_struct.mean():.1%}")
    print(f"  OVERLAPS (C6 — categories are non-exclusive):")
    print(f"    N-terminal (any)        : {nterm.mean():.1%}")
    print(f"    disorder-dominant (any) : {disorder.mean():.1%}")
    print(f"    N-term ∩ disorder       : {(nterm & disorder).mean():.1%}")
    print(f"    specific-SLiM in changed region (lower bound): {slim.mean():.1%} "
          f"(of structured-internal: {slim[p_struct].mean() if p_struct.sum() else 0:.1%})")
    return dict(n=n, nterm=p_nterm.mean(), dis=p_dis.mean(), struct=p_struct.mean(),
                slim=slim.mean(), slim_struct=(slim[p_struct].mean() if p_struct.sum() else 0.0))


def main():
    print("="*74)
    print("NON-DOMAIN MECHANISM FRACTIONS (measured, for cascade Panel A)")
    print("="*74)
    res = {}
    for t in ['brain', 'muscle']:
        try:
            res[t] = analyze(t)
        except Exception as e:
            print(f"[{t}] skipped: {e}")
    print("\n" + "="*74)
    print("For Panel A (brain, of the 30.2% non-domain slice): scale each stream by the exclusive")
    print("partition above; annotate overlaps as the C6 caveat.")
    if 'brain' in res:
        b = res['brain']
        print(f"  brain exclusive: N-term {b['nterm']:.0%} / disorder {b['dis']:.0%} / "
              f"structured {b['struct']:.0%}  (SLiM lower-bound {b['slim']:.0%})")


if __name__ == '__main__':
    main()
