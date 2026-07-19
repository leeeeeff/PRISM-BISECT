#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_severity_domain_faithful.py
===================================
Faithful reconstruction of the manuscript's domain-architecture-change covariate
(natcomm_v0.md §338(ii)), replacing build_severity_pairs.py's uniform Hamming>0
definition with the manuscript's tissue-specific definitions, so the severity
regression stays CONSISTENT with the §6 domain-envelope rate analysis:
  - muscle: change in TOTAL Pfam domain count (domain_matrix_proper_test row sums)
  - brain : hmmscan ENVELOPE OVERLAP -- a changed interval overlaps a Pfam domain
            envelope (E-value <= 0.01) in either isoform, exactly reusing
            exp_brain_labelgap_rate.py's logic (§6 method).

Adds `domain_binary_faithful` to the scored pair tables. Size stays raw (the
regression variant uses raw z-scored size, per manuscript §338(i)).
"""
import re, glob
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
FAA_BRAIN = ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa'
DOMTBL = ROOT / 'reports/truebrain_rerun_20260714/data/hmmscan_out'
IEVAL_MAX = 0.01
_AA = set('ACDEFGHIKLMNPQRSTVWY')


def sani(s):
    s = s.replace('*', '')
    return ''.join(c if c in _AA else 'X' for c in s)


def strip_orf(name):
    return re.sub(r'\.p\d+$', '', name)


def parse_faa_brain():
    """base -> (fullname, sani'd seq), longest ORF per base (matches §6)."""
    best = {}
    cur_full, buf = None, []

    def flush():
        if cur_full is None:
            return
        seq = sani(''.join(buf))
        base = strip_orf(cur_full)
        if base not in best or len(seq) > len(best[base][1]):
            best[base] = (cur_full, seq)
    for line in open(FAA_BRAIN):
        if line.startswith('>'):
            flush(); cur_full = line[1:].split()[0]; buf = []
        else:
            buf.append(line.strip())
    flush()
    return best


def parse_domtbl():
    """fullname -> list of (env_from, env_to) confident domain envelopes."""
    dom = defaultdict(list)
    for f in sorted(glob.glob(str(DOMTBL / '*.domtbl'))):
        for line in open(f):
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 23:
                continue
            try:
                ievalue = float(p[12]); q = p[3]
                ef, et = int(p[19]), int(p[20])
            except (ValueError, IndexError):
                continue
            if ievalue <= IEVAL_MAX and et > ef:
                dom[q].append((ef, et))
    return dom


def overlaps(iv, doms):
    for (df, dt) in doms:
        if iv[0] < dt and df < iv[1]:
            return True
    return False


def changed_intervals(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    iva, ivb = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if i2 > i1:
            iva.append((i1, i2))
        if j2 > j1:
            ivb.append((j1, j2))
    return iva, ivb


def faithful_muscle():
    df = pd.read_csv(ROOT / 'reports/severity_pairs/muscle_severity_pairs_scored.tsv', sep='\t')
    dom_count = np.load(ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy').sum(1)
    long_c = dom_count[df['long_idx'].to_numpy()]
    short_c = dom_count[df['short_idx'].to_numpy()]
    df['domain_binary_faithful'] = (long_c != short_c).astype(int)
    print(f"[muscle] count-change domain_binary rate {df['domain_binary_faithful'].mean():.3f} "
          f"(vs uniform-Hamming {df['domain_binary'].mean():.3f})")
    df.to_csv(ROOT / 'reports/severity_pairs/muscle_severity_pairs_scored.tsv', sep='\t', index=False)


def faithful_brain():
    df = pd.read_csv(ROOT / 'reports/severity_pairs/brain_severity_pairs_scored.tsv', sep='\t')
    faa = parse_faa_brain()
    dom = parse_domtbl()
    print(f"[brain] faa bases {len(faa)}  domtbl ORFs {len(dom)}", flush=True)

    ids = [str(x) for x in np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)]
    # brain idx -> (seq, domains)
    rec = {}
    for k, bid in enumerate(ids):
        base = strip_orf(str(bid).replace("b'", "").replace("'", ""))
        if base in faa:
            full, seq = faa[base]
            rec[k] = (seq, dom.get(full, []))

    vals = np.zeros(len(df), int)
    missing = 0
    for r, (lo, sh) in enumerate(zip(df['long_idx'].to_numpy(), df['short_idx'].to_numpy())):
        if lo not in rec or sh not in rec:
            vals[r] = -1; missing += 1; continue
        (sl, dl), (ss, ds) = rec[lo], rec[sh]
        iva, ivb = changed_intervals(sl, ss)
        hit = any(overlaps(iv, dl) for iv in iva) or any(overlaps(iv, ds) for iv in ivb)
        vals[r] = int(hit)
    df['domain_binary_faithful'] = vals
    ok = df['domain_binary_faithful'] >= 0
    print(f"[brain] envelope-overlap domain_binary rate {df.loc[ok,'domain_binary_faithful'].mean():.3f} "
          f"(vs uniform-Hamming {df['domain_binary'].mean():.3f}); missing-seq pairs {missing}")
    df.to_csv(ROOT / 'reports/severity_pairs/brain_severity_pairs_scored.tsv', sep='\t', index=False)


def main():
    faithful_muscle()
    faithful_brain()


if __name__ == '__main__':
    main()
