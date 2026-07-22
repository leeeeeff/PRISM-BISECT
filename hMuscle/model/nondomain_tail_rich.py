#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nondomain_tail_rich.py  (Option B refinement — how much of the tail is beyond COMPOSITION?)

The earlier tail (nondomain_tail_and_floor.py) removed 7 hand-crafted covariates and still left a
large reproducible residual (+0.427). Caveat: "un-named" only meant "outside those 7". Here we test
the richest purely-COMPOSITIONAL descriptor — the full 20-dim amino-acid composition of the changed
residues (helix/sheet/hydro/charge are linear functions of this, so it strictly dominates them). If a
large reproducible residual survives removal of the full 20-dim composition subspace, the non-domain
tail is genuinely BEYOND amino-acid composition (structural / positional / contextual), not merely
un-named by a coarse descriptor.

Metric parallels the prior script: held-out variance captured by the (orthonormalized median-split)
descriptor subspace, and the reproducible structure of the descriptor-orthogonal residual (K=50 PCA
excess over a random-K null). Gene-disjoint; orientations learned on train only. Brain non-domain.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022
AAS = list('ACDEFGHIKLMNPQRSTVWY')

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}


def changed_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs, changed = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            ivs.append((i1, i2))
    return ivs, changed


def load_brain():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'brain'].reset_index(drop=True)
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    L15 = np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
    L30 = np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    return df, iso, seqs, np.concatenate([L15, L30], axis=1)


def build(df, iso, seqs, emb):
    sub = df[(df['domain_binary'] == 0) & (df['nterm_overlap'] == 0)]
    D, gene, comp4, aa20 = [], [], [], []
    for _, r in sub.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs:
            continue
        ls, ss = seqs[lid][:MAXLEN], seqs[sid][:MAXLEN]
        if ls == ss:
            continue
        ivs, ch = changed_intervals(ls, ss)
        cri = [i for (u, v) in ivs for i in range(u, v) if i < len(ls)]
        if not cri:
            continue
        res = [ls[i] for i in cri]
        D.append(emb[li] - emb[si]); gene.append(str(r['gene']))
        comp4.append([np.mean([HELIX.get(a,1.0) for a in res]),
                      np.mean([SHEET.get(a,1.0) for a in res]),
                      np.mean([HYDRO.get(a,1.0) for a in res]),
                      sum(CHARGE.get(a,0.0) for a in res)/len(res)])
        aa20.append([res.count(a)/len(res) for a in AAS])
    return np.array(D), np.array(gene), np.array(comp4), np.array(aa20)


def gene_folds(gene, k=5, seed=42):
    ug = np.unique(gene); r = np.random.default_rng(seed); ugc = ug.copy(); r.shuffle(ugc)
    fmap = {g: i % k for i, g in enumerate(ugc)}
    return np.array([fmap[g] for g in gene])


def cap(Xte, V):
    proj = Xte @ V
    return float((proj ** 2).sum() / ((Xte ** 2).sum() + 1e-12))


def rand_cap(Xte, d, K, n=5, seed=0):
    return np.mean([cap(Xte, np.linalg.qr(np.random.default_rng(seed+j).standard_normal((d, K)))[0][:, :K])
                    for j in range(n)])


def descr_dirs(Dtr, C, tr):
    dirs = []
    for j in range(C.shape[1]):
        c = C[tr, j]; orient = np.where(c > np.mean(c), 1.0, -1.0)
        dirs.append((Dtr * orient[:, None]).mean(0))
    Q, _ = np.linalg.qr(np.array(dirs).T)
    return Q[:, :C.shape[1]]


def evaluate(tag, D, gene, C):
    Dc = D - D.mean(0); fid = gene_folds(gene); d = Dc.shape[1]; K = 50
    capd, capdr, repr_, reprr = [], [], [], []
    for kf in range(5):
        te = fid == kf; tr = ~te
        if tr.sum() < K + 5 or te.sum() < 5:
            continue
        Q = descr_dirs(Dc[tr], C, tr)
        capd.append(cap(Dc[te], Q)); capdr.append(rand_cap(Dc[te], d, Q.shape[1], seed=700))
        Rtr = Dc[tr] - (Dc[tr] @ Q) @ Q.T
        Rte = Dc[te] - (Dc[te] @ Q) @ Q.T
        V = TruncatedSVD(n_components=K, random_state=0).fit(Rtr).components_.T
        repr_.append(cap(Rte, V)); reprr.append(rand_cap(Rte, d, K))
    ce, re = np.mean(capd)-np.mean(capdr), np.mean(repr_)-np.mean(reprr)
    print(f"  [{tag:16}] dims={C.shape[1]:2d}  descriptor-captured EXCESS={ce:+.3f}  "
          f"|  residual reproducible (K=50) EXCESS={re:+.3f}")
    return ce, re


def main():
    print("="*82)
    print("OPTION B REFINEMENT — is the non-domain tail beyond amino-acid COMPOSITION?")
    print("brain non-domain; reference: total reproducible ~0.547, 7-covariate named 0.252")
    print("="*82)
    df, iso, seqs, emb = load_brain()
    D, gene, comp4, aa20 = build(df, iso, seqs, emb)
    print(f"non-domain pairs: n={len(D)}  genes={len(np.unique(gene))}\n")
    evaluate("compositional-4", D, gene, comp4)
    ce, re = evaluate("AA-composition-20", D, gene, aa20)
    print(f"\n  -> full 20-dim amino-acid composition names {ce:+.3f} of centered variance;")
    print(f"     reproducible structure surviving it = {re:+.3f} (of the ~0.547 total) is BEYOND")
    print(f"     amino-acid composition = structural / positional / contextual tail (true future work).")


if __name__ == '__main__':
    main()
