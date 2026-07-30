#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prep_muscle_2iso_fasta.py — extract longest-ORF proteins for muscle 2-iso-gene isoforms.
Output: reports/muscle_labelgap/muscle_2iso.fa (header = BambuTx.pN base).
Mirrors the brain label-gap instrument on the muscle held-out fold for an apples-to-apples
non-domain RATE comparison (Option A)."""
import os, re
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
ISO = ROOT / 'hMuscle/model/my_isoform_list_fixed.npy'
GEN = ROOT / 'hMuscle/model/my_gene_list_fixed.npy'
PEP = ROOT / 'hMuscle/data/transcripts.fasta.transdecoder.pep'
OUT = ROOT / 'reports/muscle_labelgap'
_AA = set('ACDEFGHIKLMNPQRSTVWY')


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', '')


def sani(s):
    s = s.replace('*', '')
    return ''.join(c if c in _AA else 'X' for c in s)


def strip_orf(name):
    return re.sub(r'\.p\d+$', '', name)


def main():
    iso = np.array([clean(x) for x in np.load(ISO, allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(GEN, allow_pickle=True)])
    gl, gi = np.unique(gen, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))
    two = set(np.where(cnt == 2)[0])
    need = set()
    for k, g in enumerate(gi):
        if g in two:
            need.add(iso[k])   # BambuTx base id
    print(f"2-iso genes {len(two)}  needed isoform bases {len(need)}", flush=True)

    # parse .pep, keep longest ORF per needed base
    best = {}   # base -> (fullname, seq)
    cur_full, buf = None, []

    def flush():
        if cur_full is None:
            return
        base = strip_orf(cur_full)
        if base not in need:
            return
        seq = sani(''.join(buf))
        if base not in best or len(seq) > len(best[base][1]):
            best[base] = (cur_full, seq)
    for line in open(PEP):
        if line.startswith('>'):
            flush()
            cur_full = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    flush()

    OUT.mkdir(parents=True, exist_ok=True)
    fa = OUT / 'muscle_2iso.fa'
    n = 0
    with open(fa, 'w') as fh:
        for base, (full, seq) in best.items():
            if 20 <= len(seq) <= 5000:
                fh.write(f">{full}\n{seq}\n")
                n += 1
    print(f"wrote {n} proteins -> {fa}  (bases with ORF: {len(best)}/{len(need)})", flush=True)


if __name__ == '__main__':
    main()
