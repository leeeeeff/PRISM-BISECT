#!/usr/bin/env python
"""
Close Option A (cached, before per-residue re-extraction). Three questions A left open:
  Q1  Is the L9-peak + late-layer erosion of pooled DOMAIN decodability TISSUE-GENERAL?
      -> replicate the 30-layer domain depth curve in MUSCLE.
  Q2  Is the erosion DOMAIN-SPECIFIC, or a general property of pooled-signal decodability?
      -> 30-layer depth curve for a within-NON-DOMAIN target (disorder-binary), both tissues.
  Q3  Does anchoring the depth-contrast at the L9 peak beat PRISM's L30-L15?
      -> compare phi9/phi15/phi30 alone, (phi30-phi9) vs (phi30-phi15), concat[phi9,phi30].

Canonical-anchored dphi^(l) = phi^(l)[other_idx] - phi^(l)[canonical_idx]. Gene-disjoint 5-fold
logistic AUROC. Read-only. Resource-limited.
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(v,'4')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
NL = 30

TISSUE = {
  'brain': dict(
     lp=lambda l: ROOT/f'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer{l:02d}_t30_150M.npy',
     pairs=ROOT/'reports/severity_pairs/brain_severity_pairs_scored.tsv'),
  'muscle': dict(
     lp=lambda l: ROOT/f'hMuscle/data/esm2_layer_{l:02d}_t30_150M.npy',
     pairs=ROOT/'reports/severity_pairs/muscle_severity_pairs_scored.tsv'),
}

def cv_auroc(X, target, folds):
    oof = np.zeros(len(target))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=300, C=0.5, tol=1e-3)
        clf.fit(sc.transform(X[tr]), target[tr]); oof[te] = clf.predict_proba(sc.transform(X[te]))[:,1]
    return roc_auc_score(target, oof)

summary = []
for tis, cfg in TISSUE.items():
    df = pd.read_csv(cfg['pairs'], sep='\t')
    ca, ot = df['canonical_idx'].to_numpy(), df['other_idx'].to_numpy()
    y = df['domain_binary'].astype(int).to_numpy(); genes = df['gene'].to_numpy()
    nd = y == 0
    dis = df['disorder_frac'].to_numpy(); dis_bin = (dis > np.median(dis[nd])).astype(int)
    n = len(df); emax = int(max(ca.max(), ot.max()))
    need = np.unique(np.concatenate([ca, ot])); remap = {r:i for i,r in enumerate(need)}
    cal = np.array([remap[r] for r in ca]); otl = np.array([remap[r] for r in ot])
    folds = list(GroupKFold(5).split(np.zeros(n), y, groups=genes))
    folds_nd = list(GroupKFold(5).split(np.zeros(nd.sum()), y[nd], groups=genes[nd]))
    print(f'\n===== {tis}: n={n} genes={df.gene.nunique()} dom_prev={y.mean():.3f} '
          f'nd={int(nd.sum())} emax={emax} =====', flush=True)
    keep = {}
    for l in range(1, NL+1):
        emb = np.load(cfg['lp'](l), mmap_mode='r')[need].astype(np.float32)
        d = emb[otl] - emb[cal]
        if l in (9,15,30): keep[l] = d.copy()
        au_dom = cv_auroc(d, y, folds)
        au_dis = cv_auroc(d[nd], dis_bin[nd], folds_nd)
        summary.append(dict(tissue=tis, layer=l, auroc_domain=au_dom, auroc_disorder_nd=au_dis))
        print(f'  L{l:02d}  domain={au_dom:.4f}   disorder|nd={au_dis:.4f}', flush=True)
        del emb, d
    # Q3 re-anchor (domain target)
    p9,p15,p30 = keep[9],keep[15],keep[30]
    combos = {'phi9':p9,'phi15':p15,'phi30':p30,'phi30-phi9':p30-p9,'phi30-phi15':p30-p15,
              'concat[phi9,phi30]':np.concatenate([p9,p30],1)}
    print(f'  -- {tis} re-anchor (domain) --', flush=True)
    for name,X in combos.items():
        au = cv_auroc(X, y, folds); print(f'     {name:20s} AUROC={au:.4f}', flush=True)
        summary.append(dict(tissue=tis, layer=f'reanchor:{name}', auroc_domain=au, auroc_disorder_nd=np.nan))

pd.DataFrame(summary).to_csv(OUT/'layer_depth_close_A.tsv', sep='\t', index=False)
print('\n[done] wrote layer_depth_close_A.tsv', flush=True)
