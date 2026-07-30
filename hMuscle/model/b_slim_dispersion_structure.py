#!/usr/bin/env python
"""
Option A (SLiM-dispersion): is the pooling-DISCARDED (non-DC) SLiM energy STRUCTURED (a real
per-isoform direction, just multi-directional across isoforms) or per-pair NOISE? Label-free.

Reconciles the biology (SLiM = specific per-isoform identity switch, multi-directional by nature)
with the B-causal result (discarded energy uninformative for GO/disorder labels): both hold if the
discarded direction is INTERNALLY reproducible per pair but ISOTROPIC across pairs.

Per SLiM/domain pair, equal-aligned delta D (n_p x 640), residual R = D - mean(D) (the non-DC part
mean-pool discards). At layer L9 (also L15/L30):
  T1a  per-pair INTERNAL reproducibility: random-split residues into halves A,B; top-PC direction of
       R on each; |cos(v_A,v_B)|. NULL = per-dimension column permutation (destroys the joint
       per-residue vector, keeps marginals) -> baseline stable-direction level. real >> null => each
       pair has a genuine discarded direction (structured), not noise.
  T1b  cross-pair STRUCTURE: stack per-pair top discarded modes d_i = s1*v1 (sign-fixed); gene-disjoint
       top-K subspace reproducibility (held-out variance captured) vs random-K null; + participation
       ratio (effective dimensionality) of the d_i covariance. Low reproducibility / high PR = the
       multi-directional / isotropic signature.
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='4'
import numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
PERRES = OUT / 'b_perres'
LAYERS = [9, 15, 30]
MINRES = 24
rng = np.random.default_rng(0)

seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t')

def load(idx, l):
    p = PERRES / f'{idx}.npz'
    return np.load(p)[f'L{l}'].astype(np.float32) if p.exists() else None

def residual(r, l):
    ls, ss = seq.get(str(r.long_idx)), seq.get(str(r.short_idx))
    HL, HS = load(r.long_idx, l), load(r.short_idx, l)
    if ls is None or ss is None or HL is None or HS is None: return None
    ops = SequenceMatcher(None, ls, ss, autojunk=False).get_opcodes()
    D = [HL[i1+o]-HS[j1+o] for tag,i1,i2,j1,j2 in ops if tag=='equal'
         for o in range(i2-i1) if i1+o<HL.shape[0] and j1+o<HS.shape[0]]
    if len(D) < MINRES: return None
    D = np.stack(D); return D - D.mean(0)      # non-DC residual (what mean-pool discards)

def top_dir(M):
    _,_,Vt = np.linalg.svd(M - M.mean(0), full_matrices=False); return Vt[0]

def splithalf_cos(R):
    n = R.shape[0]; idx = rng.permutation(n); h = n//2
    vA, vB = top_dir(R[idx[:h]]), top_dir(R[idx[h:]])
    return abs(float(vA @ vB))

def perm_null_R(R):
    # independently permute each column (destroys joint per-residue vector, keeps marginals) — vectorized
    order = np.argsort(rng.random(R.shape), axis=0)
    return np.take_along_axis(R, order, axis=0)

def part_ratio(vecs):
    C = np.cov(vecs.T); ev = np.linalg.eigvalsh(C); ev = ev[ev > 0]
    return float(ev.sum()**2 / (ev**2).sum())

def subspace_repro(d, genes, K=50, nrand=20):
    # gene-disjoint: fit PCA(K) on train genes, held-out captured var vs random-K null
    g = np.array(genes); ug = np.unique(g); rng.shuffle(ug)
    tr = np.isin(g, ug[:len(ug)//2]); te = ~tr
    Xtr, Xte = d[tr], d[te]
    mu = Xtr.mean(0); Xtr = Xtr-mu; Xte = Xte-mu
    _,_,Vt = np.linalg.svd(Xtr, full_matrices=False); P = Vt[:K]
    cap = np.sum((Xte @ P.T)**2) / np.sum(Xte**2)
    nulls = []
    for _ in range(nrand):
        Q = np.linalg.qr(rng.standard_normal((d.shape[1], K)))[0].T
        nulls.append(np.sum((Xte @ Q.T)**2) / np.sum(Xte**2))
    return cap, float(np.mean(nulls)), cap - float(np.mean(nulls))

for l in LAYERS:
    print(f'\n===== layer L{l} =====')
    for cls in ['slim', 'domain']:
        sub = man[man.cls == cls]
        real_cos, null_cos, modes, genes = [], [], [], []
        for _, r in sub.iterrows():
            R = residual(r, l)
            if R is None: continue
            real_cos.append(splithalf_cos(R))
            null_cos.append(splithalf_cos(perm_null_R(R)))
            _, S, Vt = np.linalg.svd(R, full_matrices=False)
            v = Vt[0] * S[0]; v = v * np.sign(v[np.argmax(np.abs(v))])   # sign-fix
            modes.append(v); genes.append(r.gene)
        modes = np.stack(modes)
        rc, nc = np.median(real_cos), np.median(null_cos)
        cap, ncap, exc = subspace_repro(modes, genes)
        pr = part_ratio(modes)
        print(f'  {cls:7s} n={len(real_cos):4d}  T1a split-half |cos|: real={rc:.3f} null={nc:.3f} '
              f'(Δ{rc-nc:+.3f})   T1b subspace cap={cap:.3f} rand={ncap:.3f} (excess{exc:+.3f}) '
              f'PR(effdim)={pr:.1f}/640')
print('\n[done]')
