#!/usr/bin/env python
"""
Option B causal test — is the pooling-DISCARDED (incoherent, non-DC) energy INFORMATIVE, or noise?

If mean-pool being a coherence filter is the OPERATIVE bottleneck (a recoverable architecture limit),
then the discarded non-DC component of the per-residue delta carries real label signal, and a
coherence-restoring pool would recover decodability mean-pool loses. If the discarded energy is noise,
adding it does not beat a magnitude-matched RANDOM direction -> SLiM is genuinely lost, not just
mis-pooled.

Predict-before-you-look (from region-pool raising coherence but LOWERING DR-AUC, FRAMEWORK §4b):
the discarded principal mode is expected to be ~= its magnitude-matched random null (NOT informative).

Per pair, on equal-aligned per-residue delta D (n_p x 640) at a layer:
  DC   = mean_p D                              (survives mean-pool)
  mode = S1 * Vt1 of centered D                (dominant DISCARDED direction x magnitude)
  rand = |S1| * random_unit_vector            (magnitude-matched null direction)
Decode within-SLiM targets (disorder-bin, nterm) gene-disjoint; compare AUROC(DC) vs AUROC(DC+mode)
vs AUROC(DC+rand). Delta(mode) - Delta(rand) = informativeness of the discarded mode beyond capacity.
"""
import numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
PERRES = OUT / 'b_perres'
LAYER = 9
rng = np.random.default_rng(0)

seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t')
sc = pd.read_csv(ROOT / 'reports/severity_pairs/brain_severity_pairs_scored.tsv', sep='\t')
man = man.merge(sc[['canonical_idx','other_idx','disorder_frac','nterm_overlap']],
                on=['canonical_idx','other_idx'], how='left')

def load(idx, l):
    p = PERRES / f'{idx}.npz'
    return np.load(p)[f'L{l}'].astype(np.float32) if p.exists() else None

def pair_features(r, l):
    ls, ss = seq.get(str(r.long_idx)), seq.get(str(r.short_idx))
    HL, HS = load(r.long_idx, l), load(r.short_idx, l)
    if ls is None or ss is None or HL is None or HS is None: return None
    ops = SequenceMatcher(None, ls, ss, autojunk=False).get_opcodes()
    D = [HL[i1+o] - HS[j1+o] for tag,i1,i2,j1,j2 in ops if tag=='equal'
         for o in range(i2-i1) if i1+o < HL.shape[0] and j1+o < HS.shape[0]]
    if len(D) < 3: return None
    D = np.stack(D)
    DC = D.mean(0)
    Dc = D - DC
    U,S,Vt = np.linalg.svd(Dc, full_matrices=False)
    mode = S[0]*Vt[0]
    rand = np.linalg.norm(mode) * (lambda v: v/np.linalg.norm(v))(rng.standard_normal(D.shape[1]))
    return DC, mode, rand

def cv(X, y, g):
    oof = np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=300,C=0.5,tol=1e-3)
        c.fit(s.transform(X[tr]),y[tr]); oof[te]=c.predict_proba(s.transform(X[te]))[:,1]
    return roc_auc_score(y,oof)

print(f'layer L{LAYER}')
for cls in ['slim','domain']:
    sub = man[man.cls==cls].reset_index(drop=True)
    feats, keep = [], []
    for i,r in sub.iterrows():
        f = pair_features(r, LAYER)
        if f is not None: feats.append(f); keep.append(i)
    sub = sub.iloc[keep].reset_index(drop=True)
    DC = np.stack([f[0] for f in feats]); MODE = np.stack([f[1] for f in feats]); RAND = np.stack([f[2] for f in feats])
    g = sub.gene.to_numpy()
    print(f'\n===== {cls}  n={len(sub)} =====')
    for tgt in ['disorder','nterm']:
        if tgt=='disorder':
            y = (sub.disorder_frac > sub.disorder_frac.median()).astype(int).to_numpy()
        else:
            nt = sub.nterm_overlap.to_numpy(); y = (nt > np.median(nt)).astype(int)
        if len(np.unique(y))<2 or min(np.bincount(y))<20:
            print(f'  {tgt:9s} skipped (degenerate)'); continue
        a_dc  = cv(DC, y, g)
        a_mode= cv(np.concatenate([DC,MODE],1), y, g)
        a_rand= cv(np.concatenate([DC,RAND],1), y, g)
        print(f'  {tgt:9s} DC={a_dc:.4f}  DC+mode={a_mode:.4f} (Δ{a_mode-a_dc:+.4f})  '
              f'DC+rand={a_rand:.4f} (Δ{a_rand-a_dc:+.4f})  '
              f'mode-beyond-null={a_mode-a_rand:+.4f}')
print('\n[done]')
