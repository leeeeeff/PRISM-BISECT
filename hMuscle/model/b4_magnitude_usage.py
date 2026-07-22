#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_magnitude_usage.py  (devils-advocate ATTACK 2/5: is 'not used' too strict?)

The cv_dir_acc B4 test says compositional/N-terminal directions are not used at output *as a
gene-reproducible direction*. But PRISM output reacts strongly (|diff| ~16.7, hundreds of GO terms
move). A reviewer will ask: how is that 'not used'? This test separates the two senses of usage:

  DIRECTIONAL usage  (already tested, negative): does the compositional ORIENTATION reach output?
  MAGNITUDE usage    (this test): does PRISM's output-shift SIZE depend on the edit's composition,
                     beyond edit size? Gene-disjoint ridge predicting |D_prism|_L1 from:
                       (a) size alone
                       (b) size + 4 compositional (helix/sheet/hydro/charge)
                       (c) 4 compositional alone
                     Held-out R^2. If (b) ~= (a) >> (c): output magnitude is driven by SIZE, and
                     composition matters only via its size-correlation -> 'not used' is precise
                     (no compositional signal beyond size). If (b) >> (a): composition IS used in
                     magnitude -> soften the claim.

Brain primary. No embeddings needed (PRISM output + covariates only). Read-only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}


def changed(ls, ss):
    sm = SequenceMatcher(None, ls, ss, autojunk=False)
    return [i for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal' and i2 > i1
            for i in range(i1, min(i2, len(ls)))]


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[(df['tissue'] == 'brain') & (df['domain_binary'] == 0)].reset_index(drop=True)
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)

    ymag, size, comp, gene = [], [], [], []
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs:
            continue
        ls, ss = seqs[lid][:MAXLEN], seqs[sid][:MAXLEN]
        if ls == ss:
            continue
        cri = changed(ls, ss)
        if not cri:
            continue
        res = [ls[i] for i in cri]
        ymag.append(float(np.abs(prism[li] - prism[si]).sum()))
        size.append(float(r['size']))
        comp.append([np.mean([HELIX.get(a,1.0) for a in res]), np.mean([SHEET.get(a,1.0) for a in res]),
                     np.mean([HYDRO.get(a,1.0) for a in res]), sum(CHARGE.get(a,0.0) for a in res)/len(res)])
        gene.append(str(r['gene']))
    y = np.array(ymag); size = np.array(size)[:, None]; comp = np.array(comp); grp = np.array(gene)
    print("=" * 80)
    print("B4 MAGNITUDE USAGE — does PRISM output-shift size depend on composition beyond size?")
    print("brain non-domain, n=%d, genes=%d" % (len(y), len(np.unique(grp))))
    print("=" * 80)
    r2_size = cvr2(size, y, grp)
    r2_comp = cvr2(comp, y, grp)
    r2_both = cvr2(np.hstack([size, comp]), y, grp)
    print(f"  (a) size alone            : held-out R^2 = {r2_size:+.3f}")
    print(f"  (c) 4 compositional alone : held-out R^2 = {r2_comp:+.3f}")
    print(f"  (b) size + compositional  : held-out R^2 = {r2_both:+.3f}")
    print(f"  incremental R^2 of composition beyond size = {r2_both - r2_size:+.3f}")
    print("-" * 80)
    if r2_both - r2_size < 0.01:
        print("  -> composition adds ~nothing beyond size: PRISM output magnitude is a SIZE reaction.")
        print("     'not used' is precise (no compositional signal, directional OR magnitude, beyond size).")
    else:
        print("  -> composition DOES add magnitude signal beyond size: soften 'not used' to")
        print("     'not used DIRECTIONALLY; a weak compositional magnitude dependence exists'.")


if __name__ == '__main__':
    main()
