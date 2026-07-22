#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nondomain_residual_decomposition.py   (Option A — the open number in FRAMEWORK.md §5)

Question: of the NON-DOMAIN within-gene embedding-difference signal that PRISM encodes, what
fraction is future-recoverable structure vs an irreducible noise floor, and how much of the
recoverable part is already named by the compositional descriptors (helix/sheet/hydro/charge)?

We decompose the non-domain embedding-difference D (1280-dim = concat[L15,L30], long-short) of the
non-domain internal-edit subset into:

  (0) dataset-wide MEAN orientation   -- reproducibility of mean(D) across gene-disjoint folds
                                          (the "weakly surfaced dataset-wide direction" already known)
  (A) reproducible STRUCTURED variance -- fraction of centered-D variance captured, on HELD-OUT genes,
                                          by the train-derived top-K PC subspace, IN EXCESS of a random
                                          K-dim subspace null. This is the gene-generalizing (=labelable
                                          in principle, future-recoverable) structure.
  (B) compositional-explained slice    -- held-out variance captured by the 4 median-split compositional
                                          direction vectors (orthonormalized), excess over random-4 null.
  (C) non-compositional reproducible   -- (A at K=4) - (B): reproducible structure beyond compositional.
  (D) noise floor                      -- 1 - (total reproducible) : per-pair idiosyncratic, not gene-
                                          generalizing = the true describability floor.

DOMAIN pairs are run as a positive reference (expected: much larger reproducible structure).
Brain = primary/clean tissue (C3 gate). Read-only. Efficient (precomputed stds; randomized SVD).
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
rng = np.random.default_rng(42)

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
    emb = np.concatenate([L15, L30], axis=1)
    return df, iso, seqs, emb


def build(df, iso, seqs, emb, domain_val):
    """domain_val=0 -> non-domain internal-edit subset; =1 -> domain-change reference."""
    if domain_val == 0:
        sub = df[(df['domain_binary'] == 0) & (df['nterm_overlap'] == 0)]
    else:
        sub = df[df['domain_binary'] == 1]
    D, gene, cov = [], [], {'helix': [], 'sheet': [], 'hydro': [], 'charge': []}
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
        cov['helix'].append(np.mean([HELIX.get(ls[i], 1.0) for i in cri]))
        cov['sheet'].append(np.mean([SHEET.get(ls[i], 1.0) for i in cri]))
        cov['hydro'].append(np.mean([HYDRO.get(ls[i], 1.0) for i in cri]))
        cov['charge'].append(sum(CHARGE.get(ls[i], 0.0) for i in cri) / len(cri))
    return np.array(D), np.array(gene), {k: np.array(v) for k, v in cov.items()}


def gene_folds(gene, k=5, seed=42):
    ug = np.unique(gene); r = np.random.default_rng(seed); ugc = ug.copy(); r.shuffle(ugc)
    fmap = {g: i % k for i, g in enumerate(ugc)}
    return np.array([fmap[g] for g in gene])


def captured_by_subspace(Xte, V):
    """fraction of held-out Frobenius energy of Xte captured by orthonormal-column subspace V."""
    proj = Xte @ V                     # (n, K)
    return float((proj ** 2).sum() / ((Xte ** 2).sum() + 1e-12))


def reproducible_excess(Dc, gene, K, n_rand=5):
    """held-out variance captured by train top-K PCs, minus random-K-subspace null. Gene-disjoint 5-fold."""
    fid = gene_folds(gene)
    cap_real, cap_rand = [], []
    d = Dc.shape[1]
    for kf in range(5):
        te = fid == kf; tr = ~te
        if tr.sum() < K + 5 or te.sum() < 5:
            continue
        svd = TruncatedSVD(n_components=K, random_state=0).fit(Dc[tr])
        V = svd.components_.T                       # (d, K) orthonormal-ish (right singular vecs)
        cap_real.append(captured_by_subspace(Dc[te], V))
        rr = []
        for j in range(n_rand):
            G = np.random.default_rng(1000 + j).standard_normal((d, K))
            Q, _ = np.linalg.qr(G)
            rr.append(captured_by_subspace(Dc[te], Q))
        cap_rand.append(np.mean(rr))
    return np.mean(cap_real), np.mean(cap_rand)


def compositional_captured(Dc, gene, cov, n_rand=5):
    """held-out variance captured by the 4 orthonormalized median-split compositional directions."""
    fid = gene_folds(gene)
    cap_real, cap_rand = [], []
    d = Dc.shape[1]
    for kf in range(5):
        te = fid == kf; tr = ~te
        if tr.sum() < 10 or te.sum() < 5:
            continue
        dirs = []
        for name in ['helix', 'sheet', 'hydro', 'charge']:
            orient = np.where(cov[name][tr] > np.median(cov[name][tr]), 1.0, -1.0)
            v = (Dc[tr] * orient[:, None]).mean(0)
            dirs.append(v)
        Vc, _ = np.linalg.qr(np.array(dirs).T)      # (d,4) orthonormal
        cap_real.append(captured_by_subspace(Dc[te], Vc[:, :4]))
        rr = []
        for j in range(n_rand):
            G = np.random.default_rng(2000 + j).standard_normal((d, 4))
            Q, _ = np.linalg.qr(G)
            rr.append(captured_by_subspace(Dc[te], Q))
        cap_rand.append(np.mean(rr))
    return np.mean(cap_real), np.mean(cap_rand)


def mean_orientation_cvacc(D, gene):
    """gene-disjoint reproducibility of the dataset-wide mean(long-short) direction."""
    fid = gene_folds(gene); n = len(D); correct = 0
    for kf in range(5):
        te = fid == kf; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        m = D[tr].mean(0); m /= (np.linalg.norm(m) + 1e-9)
        correct += int((D[te] @ m > 0).sum())
    return correct / n


def analyze(tag, D, gene, cov):
    print(f"\n{'='*82}\n[{tag}]  n_pairs={len(D)}  n_genes={len(np.unique(gene))}  dim={D.shape[1]}")
    print('-'*82)
    Dc = D - D.mean(0)
    tot = float((Dc ** 2).sum())
    # (0) dataset-wide mean orientation
    macc = mean_orientation_cvacc(D, gene)
    mean_energy_frac = float((D.mean(0) ** 2).sum() * len(D) / ((D ** 2).sum() + 1e-12))
    print(f"  (0) dataset-wide mean orientation: cv-dir-acc={macc:.3f}  "
          f"(mean-direction energy = {mean_energy_frac:.1%} of raw |D|^2)")
    # (A) reproducible structured variance, K-sweep on centered D
    print("  (A) reproducible structured variance (held-out captured MINUS random-subspace null):")
    excess = {}
    for K in [4, 10, 20, 50]:
        cr, cn = reproducible_excess(Dc, gene, K)
        excess[K] = cr - cn
        print(f"       K={K:3d}: real={cr:.3f}  random-null={cn:.3f}  EXCESS={cr-cn:+.3f}")
    # (B) compositional-captured
    ccr, ccn = compositional_captured(Dc, gene, cov)
    comp_excess = ccr - ccn
    print(f"  (B) compositional 4-dir captured: real={ccr:.3f}  random-4-null={ccn:.3f}  "
          f"EXCESS={comp_excess:+.3f}")
    # (C) non-compositional reproducible at matched K=4
    nc4 = excess[4] - comp_excess
    print(f"  (C) non-compositional reproducible (K=4 excess - compositional excess) = {nc4:+.3f}")
    print(f"  (D) interpretation: reproducible-structure(K=50 excess)={excess[50]:+.3f} of centered var; "
          f"compositional explains {comp_excess:+.3f}; "
          f"floor (non-reproducible) ~= {1-excess[50]:.1%} of centered var")
    return dict(mean_acc=macc, excess=excess, comp_excess=comp_excess, nc4=nc4)


def main():
    print("="*82)
    print("NON-DOMAIN RESIDUAL DECOMPOSITION (Option A): future-recoverable vs true floor")
    print("brain (primary/clean tissue)")
    print("="*82)
    df, iso, seqs, emb = load_brain()
    Dn, gn, cn = build(df, iso, seqs, emb, domain_val=0)
    rn = analyze("NON-DOMAIN internal-edit", Dn, gn, cn)
    Dd, gd, cd = build(df, iso, seqs, emb, domain_val=1)
    rd = analyze("DOMAIN-change (positive reference)", Dd, gd, cd)

    print(f"\n{'='*82}\nVERDICT\n{'='*82}")
    print(f"  reproducible structured variance (K=50 excess over random): "
          f"non-domain {rn['excess'][50]:+.3f} vs domain {rd['excess'][50]:+.3f}")
    print(f"  compositional-explained (excess):   non-domain {rn['comp_excess']:+.3f} "
          f"vs domain {rd['comp_excess']:+.3f}")
    if rn['excess'][50] > 0.01:
        share = rn['comp_excess'] / rn['excess'][50]
        print(f"  -> of the non-domain reproducible structure, compositional names ~{share:.0%}; "
              f"the rest is decodable-but-non-compositional (R2/R3 tail, future work).")
    print(f"  -> the non-reproducible remainder (~{1-rn['excess'][50]:.0%} of centered variance) is the "
          f"per-pair noise floor for non-domain edits.")


if __name__ == '__main__':
    main()
