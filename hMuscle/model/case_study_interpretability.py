#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""case_study_interpretability.py

Two deliverables for the PRISM interpretability MAP (FRAMEWORK.md §6, §3/C4).

PART 1 (Option A) -- per-isoform case study on the 8-axis x 7-covariate grid.
  For each within-gene pair of a diagnostic gene, decompose the pair simultaneously onto:
    - 8 joint-PCA encoding axes  (unsupervised geometry): dA_k = axis_k(long) - axis_k(short),
      reported in population-SD units (|dA| large = the pair separates along that encoding axis).
    - 7 edit covariates (supervised descriptors): size, domain_binary, disorder_frac, and the
      four compositional propensities helix/sheet/hydro/charge of the changed residues.
    - PRISM output shift: |score(long) - score(short)| L1 and #terms moved (does the model react?).
  Diagnostic triple (FRAMEWORK.md §6):
    Type A NDUFS4  (MTS-exon / domain loss) -> expect axis3 loaded, PRISM reacts   ("everything works")
    Type B MAPT    (N-term insert, no domain)-> expect axes flat, PRISM near-silent ("encoded-but-lost")
    Type C LRPPRC  (same-domain)             -> negative control, flat everywhere.

  NOTE on axis construction: joint-PCA basis W (8x640) is applied per layer to the z-scored
  embedding (Methods (2)); axis scalar = mean over layers. Here we use the two loaded layers
  {L15, L30} as a 2-layer trajectory-mean proxy (the full 30-layer mean is not needed for a
  legibility instrument). z-score stats are computed over the tissue's own population.

PART 2 (Option B / devils-advocate C4) -- replace the indefensible "64%" AUROC-difference figure.
  AUROC differences are not additive information quantities. We refit a COMMON logistic decoder
  for domain-architecture change (domain_binary) on (i) the 8 per-axis |dA| and (ii) the full
  640-dim |dEmb|, and report McFadden pseudo-R^2 (a deviance / log-loss ratio, which IS additive),
  with the 8-axis / 640-linear ratio as the honest "fraction of the linearly-decodable domain
  signal that the 8 interpretable axes capture." AUROC is reported alongside for continuity only.

Brain is the primary/clean tissue (C3 gate, FRAMEWORK.md §8d). Read-only; no data mutated.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
W_AXES = ROOT / 'reports/v20b_pca_interp/W_axes_8x640.npy'
MAXLEN = 1022

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}

CASES = [('NDUFS4', 'A', 'MTS-exon / domain loss'),
         ('MAPT',   'B', 'N-terminal insert, no domain change'),
         ('LRPPRC', 'C', 'same-domain (negative control)')]


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
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)
    return df, iso, seqs, L15, L30, prism


def axis_scores(L15, L30, W):
    """Per-isoform 8-axis score = mean over {L15,L30} of (population-z-scored layer) . W^T."""
    def proj(L):
        z = (L - L.mean(0)) / (L.std(0) + 1e-6)
        return z @ W.T           # (N, 8)
    A = 0.5 * (proj(L15) + proj(L30))
    A_sd = A.std(0) + 1e-9       # per-axis population SD (units for dA)
    return A, A_sd


def comp_covariates(ls, cri):
    return (np.mean([HELIX.get(ls[i], 1.0) for i in cri]),
            np.mean([SHEET.get(ls[i], 1.0) for i in cri]),
            np.mean([HYDRO.get(ls[i], 1.0) for i in cri]),
            sum(CHARGE.get(ls[i], 0.0) for i in cri) / len(cri))


def main():
    print("="*90)
    print("PRISM INTERPRETABILITY MAP -- case study (Part 1) + C4 deviance recompute (Part 2)")
    print("brain (primary/clean tissue; C3 gate FRAMEWORK.md §8d)")
    print("="*90)
    W = np.load(W_AXES)                                   # (8, 640)
    df, iso, seqs, L15, L30, prism = load_brain()
    A, A_sd = axis_scores(L15, L30, W)                    # (N,8), (8,)
    id2row = {s: k for k, s in enumerate(iso)}

    # ---------- PART 1: case study ----------
    for gene, typ, desc in CASES:
        sub = df[df['gene'] == gene]
        print(f"\n{'-'*90}\n[Type {typ}] {gene}  ({desc})   -- {len(sub)} within-gene pair(s)")
        if len(sub) == 0:
            print("  (no pairs)"); continue
        # rank pairs by total axis displacement so the most informative pair shows first
        rows = []
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
            dA = (A[li] - A[si]) / A_sd                    # 8-vector, SD units
            hx, sh, hy, cg = comp_covariates(ls, cri)
            dp = prism[li] - prism[si]
            rows.append(dict(li=li, si=si, ch=ch, dA=dA,
                             dom=int(r['domain_binary']), dis=float(r['disorder_frac']),
                             nterm=int(r['nterm_overlap']), resync=int(r['resync_failure_binary']),
                             hx=hx, sh=sh, hy=hy, cg=cg,
                             pnorm=float(np.abs(dp).sum()), pmoved=int((np.abs(dp) > 0.05).sum()),
                             ptop=int(np.argmax(np.abs(dp))), ptopv=float(dp[np.argmax(np.abs(dp))])))
        rows.sort(key=lambda d: -np.abs(d['dA']).sum())
        for d in rows[:3]:
            top = np.argsort(-np.abs(d['dA']))[:3]
            axstr = "  ".join(f"ax{k}={d['dA'][k]:+.2f}" for k in top)
            print(f"  pair(edit {d['ch']:4d} aa)  domain={d['dom']} nterm={d['nterm']} "
                  f"resync={d['resync']} disorder={d['dis']:.2f}  |  "
                  f"comp[hlx {d['hx']:.2f} sht {d['sh']:.2f} hyd {d['hy']:+.2f} chg {d['cg']:+.2f}]")
            print(f"     top axes: {axstr}   (|dA|_total={np.abs(d['dA']).sum():.2f} SD)")
            print(f"     PRISM out: |dScore|_L1={d['pnorm']:.2f}  #terms>0.05={d['pmoved']}/672  "
                  f"topGO#{d['ptop']} d={d['ptopv']:+.2f}")

    # ---------- PART 2: C4 deviance-based 8-axis vs 640-linear domain decodability ----------
    print(f"\n{'='*90}\nPART 2 -- C4 fix: deviance-based 8-axis vs 640-linear domain-change decodability")
    print("(replaces the AUROC-difference '64%'; McFadden pseudo-R^2 is additive, AUROC is not)")
    print("="*90)
    sd15 = L15.std(0) + 1e-6                 # precompute once (was recomputed per-row = O(N*40M) bug)
    sd30 = L30.std(0) + 1e-6
    li_all = df['long_idx'].astype(int).to_numpy()
    si_all = df['short_idx'].astype(int).to_numpy()
    dA_all = np.abs((A[li_all] - A[si_all]) / A_sd)                       # (P,8)
    dE_all = np.abs(0.5 * ((L15[li_all] - L15[si_all]) / sd15
                           + (L30[li_all] - L30[si_all]) / sd30))         # (P,640)
    X8 = dA_all; X640 = dE_all
    y = df['domain_binary'].astype(int).to_numpy()
    grp = df['gene'].astype(str).to_numpy()
    print(f"  brain pairs: n={len(y)}  domain-change prevalence={y.mean():.3f}")

    def cv_probe(X):
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(y))
        for tr, te in gkf.split(X, y, grp):
            sc = StandardScaler().fit(X[tr])          # scaling -> lbfgs converges in <100 iters
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        auc = roc_auc_score(y, oof)
        ll = log_loss(y, np.clip(oof, 1e-6, 1-1e-6))
        p = y.mean(); ll0 = log_loss(y, np.full(len(y), p))     # intercept-only deviance
        mcf = 1.0 - ll / ll0                                    # McFadden pseudo-R^2
        return auc, mcf

    auc8, mcf8 = cv_probe(X8)
    auc640, mcf640 = cv_probe(X640)
    ratio = mcf8 / mcf640 if mcf640 > 0 else np.nan
    print(f"  8-axis  |dA| : AUROC={auc8:.3f}   McFadden R^2={mcf8:.4f}")
    print(f"  640-dim |dE| : AUROC={auc640:.3f}   McFadden R^2={mcf640:.4f}")
    print(f"  -> 8-axis captures {ratio:.1%} of the linearly-decodable domain deviance "
          f"(additive measure; NOT the old AUROC-difference 64%)")


if __name__ == '__main__':
    main()
