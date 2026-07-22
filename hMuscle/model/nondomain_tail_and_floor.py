#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nondomain_tail_and_floor.py

Follow-ups to nondomain_residual_decomposition.py, brain non-domain internal-edit pairs.

OPTION A -- narrow the R2/R3 tail: remove the FULL covariate subspace (not just the 4 compositional
  but also size / disorder / resync — every covariate that VARIES inside the non-domain subset;
  domain & nterm are constant there and cannot be used) and measure the reproducible structure that
  survives = the genuinely-un-named ("novel") tail. Gene-disjoint; orientations learned on train only.

OPTION B -- probe the 45.3% noise floor without per-residue embeddings. If the floor is mean-pooling
  DILUTION of a fixed local signal, the reproducible fraction should RISE with edit changed-fraction
  (a larger changed fraction is diluted less by 1/L pooling); if the floor is genuine per-pair noise,
  the reproducible fraction is flat across edit size. We stratify pairs by changed-fraction quintile
  and measure the reproducible-structure excess per stratum + the trend. This is an architectural
  discriminator (pooling-dilution vs true floor), not a per-residue recovery.

Read-only; efficient (precomputed stds; randomized SVD). Brain = primary tissue.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD

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
    D, gene = [], []
    cov = {k: [] for k in ['size', 'disorder', 'resync', 'helix', 'sheet', 'hydro', 'charge']}
    chfrac = []
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
        D.append(emb[li] - emb[si]); gene.append(str(r['gene']))
        cov['size'].append(float(r['size']))
        cov['disorder'].append(float(r['disorder_frac']))
        cov['resync'].append(float(r['resync_failure_binary']))
        cov['helix'].append(np.mean([HELIX.get(ls[i], 1.0) for i in cri]))
        cov['sheet'].append(np.mean([SHEET.get(ls[i], 1.0) for i in cri]))
        cov['hydro'].append(np.mean([HYDRO.get(ls[i], 1.0) for i in cri]))
        cov['charge'].append(sum(CHARGE.get(ls[i], 0.0) for i in cri) / len(cri))
        chfrac.append(ch / max(len(ls), 1))
    return (np.array(D), np.array(gene), {k: np.array(v) for k, v in cov.items()},
            np.array(chfrac))


def gene_folds(gene, k=5, seed=42):
    ug = np.unique(gene); r = np.random.default_rng(seed); ugc = ug.copy(); r.shuffle(ugc)
    fmap = {g: i % k for i, g in enumerate(ugc)}
    return np.array([fmap[g] for g in gene])


def cap(Xte, V):
    proj = Xte @ V
    return float((proj ** 2).sum() / ((Xte ** 2).sum() + 1e-12))


def rand_cap(Xte, d, K, n=5, seed=0):
    out = []
    for j in range(n):
        G = np.random.default_rng(seed + j).standard_normal((d, K))
        Q, _ = np.linalg.qr(G)
        out.append(cap(Xte, Q[:, :K]))
    return np.mean(out)


def reproducible_excess(Dc, gene, K, seed=42):
    fid = gene_folds(gene, seed=seed); cr, cn = [], []; d = Dc.shape[1]
    for kf in range(5):
        te = fid == kf; tr = ~te
        if tr.sum() < K + 5 or te.sum() < 5:
            continue
        V = TruncatedSVD(n_components=K, random_state=0).fit(Dc[tr]).components_.T
        cr.append(cap(Dc[te], V)); cn.append(rand_cap(Dc[te], d, K))
    return np.mean(cr), np.mean(cn)


def cov_dirs(Dtr, cov, tr, names):
    dirs = []
    for name in names:
        c = cov[name][tr]
        orient = np.where(c > np.median(c), 1.0, -1.0)
        dirs.append((Dtr * orient[:, None]).mean(0))
    Q, _ = np.linalg.qr(np.array(dirs).T)
    return Q[:, :len(names)]


def option_A(D, gene, cov):
    print(f"\n{'='*82}\nOPTION A — narrow the tail: reproducible structure surviving FULL covariate removal")
    print('='*82)
    names = ['size', 'disorder', 'resync', 'helix', 'sheet', 'hydro', 'charge']
    Dc = D - D.mean(0)
    fid = gene_folds(gene)
    # covariate subspace captured (held-out), and reproducible structure of the covariate-orthogonal residual
    cap_cov, cap_cov_rand = [], []
    rep_resid, rep_resid_rand = [], []
    K = 50; d = Dc.shape[1]
    for kf in range(5):
        te = fid == kf; tr = ~te
        if tr.sum() < K + 5 or te.sum() < 5:
            continue
        C = cov_dirs(Dc[tr], cov, tr, names)                 # (d, 7) orthonormal
        cap_cov.append(cap(Dc[te], C)); cap_cov_rand.append(rand_cap(Dc[te], d, C.shape[1], seed=500))
        # residualize both train & test against covariate subspace
        Rtr = Dc[tr] - (Dc[tr] @ C) @ C.T
        Rte = Dc[te] - (Dc[te] @ C) @ C.T
        V = TruncatedSVD(n_components=K, random_state=0).fit(Rtr).components_.T
        rep_resid.append(cap(Rte, V)); rep_resid_rand.append(rand_cap(Rte, d, K))
    cc, ccr = np.mean(cap_cov), np.mean(cap_cov_rand)
    rr, rrr = np.mean(rep_resid), np.mean(rep_resid_rand)
    print(f"  full 7-covariate subspace captured (held-out): {cc:.3f} (random-7 null {ccr:.3f}) "
          f"EXCESS={cc-ccr:+.3f}")
    print(f"  reproducible structure of covariate-ORTHOGONAL residual (K=50): {rr:.3f} "
          f"(random null {rrr:.3f}) EXCESS={rr-rrr:+.3f}")
    print(f"  -> prior total reproducible=0.547; compositional-only named 0.184; full 7-cov named {cc-ccr:+.3f};")
    print(f"     genuinely-UN-NAMED reproducible tail (survives all covariates) = {rr-rrr:+.3f} of centered var.")
    return cc - ccr, rr - rrr


def option_B(D, gene, chfrac):
    print(f"\n{'='*82}\nOPTION B — is the 45% floor pooling-dilution or true noise? (edit changed-fraction strata)")
    print('='*82)
    Dc = D - D.mean(0)
    q = np.quantile(chfrac, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(chfrac, q)                          # 0..4
    print(f"  changed-fraction quintile cutoffs: {np.round(q,3)}")
    reps, mids = [], []
    for s in range(5):
        m = strata == s
        if m.sum() < 60:
            continue
        cr, cn = reproducible_excess(Dc[m], gene[m], K=20)
        reps.append(cr - cn); mids.append(np.median(chfrac[m]))
        print(f"  Q{s+1}: n={m.sum():5d}  median changed-frac={np.median(chfrac[m]):.3f}  "
              f"reproducible-excess(K=20)={cr-cn:+.3f}  (floor~={1-(cr-cn):.0%})")
    if len(reps) >= 3:
        rho, p = spearmanr(mids, reps)
        print(f"  trend: Spearman(changed-fraction, reproducible-excess) = {rho:+.3f} (p={p:.3f})")
        if rho > 0.5:
            verdict = ("RISES with edit size = consistent with pooling-DILUTION of a local signal "
                       "(part of the floor is recoverable in principle by per-residue architectures)")
        else:
            verdict = ("flat/weak = the floor is closer to genuine per-pair noise, not simple "
                       "pooling dilution")
        print(f"  -> {verdict}")


def main():
    print("="*82)
    print("NON-DOMAIN TAIL (A) + FLOOR (B) — brain primary")
    print("="*82)
    df, iso, seqs, emb = load_brain()
    D, gene, cov, chfrac = build(df, iso, seqs, emb)
    print(f"non-domain pairs: n={len(D)}  genes={len(np.unique(gene))}")
    option_A(D, gene, cov)
    option_B(D, gene, chfrac)


if __name__ == '__main__':
    main()
