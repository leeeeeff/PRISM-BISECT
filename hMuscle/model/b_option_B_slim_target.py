#!/usr/bin/env python
"""
Option B (decisive) — is the pooling-DISCARDED (non-DC) structure SLiM-FUNCTIONAL, or generic?
Target = SLiM-class change in the edit region (project's own regex set from exp_true_motif_level.py),
NOT GO/disorder. Tests the user's thesis: SLiM multi-directional dispersion carries isoform-identity.

Per brain SLiM pair (L9): decode "does this edit change SLiM class c" (hits_long != hits_short) from
  DC   (survives mean-pool; edit-core diluted at ~3.8% weight lives here)
  DC+mode  (mode = top discarded non-DC direction of equal-aligned residual = the discarded context ripple)
  DC+rand  (magnitude-matched null)
gene-disjoint. mode-beyond-null > 0 => discarded structure carries SLiM info beyond edit-core & capacity
=> user's thesis holds (mean-pool discards recoverable SLiM signal). ~0 => SLiM identity is in the
surviving edit-core (DC) or not linearly in the discarded context.
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[v]='4'
import re, numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); OUT = ROOT/'reports/model_interpretability_map'
PERRES = OUT/'b_perres'; L = 9; rng = np.random.default_rng(0)
SLIMS = {  # verbatim from exp_true_motif_level.py A3
 'NLS_basic_cluster': r'[KR]{3,}|[KR].{1,2}[KR]{2,}', 'NES_leucine_rich': r'L.{2,3}[LIVMF].{2,3}L.{2,3}L',
 'PXXP_SH3_binding': r'P.{2}P', 'RGD_integrin': r'RGD', 'KFERQ_autophagy': r'[KQRE].{1,3}[KQRE].*F',
 'phospho_CK2': r'[ST].{2}[ED]', 'phospho_PKA': r'[RK].{2}[ST]', 'phospho_CDK': r'[ST]P[KR]',
 'CAAX_prenyl': r'C[AC].{1}[LIVMF]$'}
PATS = {k: re.compile(v) for k,v in SLIMS.items()}
def nhits(s, p): return len(p.findall(s))

seq = {}
with open(OUT/'b_extract_sequences.fasta') as f:
    cur=None
    for line in f:
        line=line.rstrip('\n')
        if line.startswith('>'): cur=line[1:]
        elif line: seq[cur]=line
man = pd.read_csv(OUT/'b_manifest_pairs.tsv', sep='\t'); man=man[man.cls=='slim'].reset_index(drop=True)

def feats(r):
    ls,ss=seq.get(str(r.long_idx)),seq.get(str(r.short_idx))
    pl=PERRES/f'{r.long_idx}.npz'; ps=PERRES/f'{r.short_idx}.npz'
    if ls is None or ss is None or not pl.exists() or not ps.exists(): return None
    HL=np.load(pl)[f'L{L}'].astype(np.float32); HS=np.load(ps)[f'L{L}'].astype(np.float32)
    ops=SequenceMatcher(None,ls,ss,autojunk=False).get_opcodes()
    D=[HL[i1+o]-HS[j1+o] for tag,i1,i2,j1,j2 in ops if tag=='equal' for o in range(i2-i1)
       if i1+o<HL.shape[0] and j1+o<HS.shape[0]]
    if len(D)<24: return None
    D=np.stack(D); DC=D.mean(0); _,S,Vt=np.linalg.svd(D-DC,full_matrices=False)
    mode=S[0]*Vt[0]; rand=np.linalg.norm(mode)*(lambda v:v/np.linalg.norm(v))(rng.standard_normal(D.shape[1]))
    chg={k:int(nhits(ls,p)!=nhits(ss,p)) for k,p in PATS.items()}
    return DC,mode,rand,chg

rows=[]; DCl=[];MODEl=[];RANDl=[];CHG=[];G=[]
for _,r in man.iterrows():
    f=feats(r)
    if f is None: continue
    DCl.append(f[0]);MODEl.append(f[1]);RANDl.append(f[2]);CHG.append(f[3]);G.append(r.gene)
DC=np.stack(DCl);MODE=np.stack(MODEl);RAND=np.stack(RANDl);G=np.array(G)
chgdf=pd.DataFrame(CHG); print(f'n pairs={len(DC)}  genes={len(set(G))}')
print('SLiM-class change prevalence:'); print(chgdf.mean().round(3).to_string())

def cv(X,y,g):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        s=StandardScaler().fit(X[tr]);c=LogisticRegression(max_iter=300,C=0.5,tol=1e-3)
        c.fit(s.transform(X[tr]),y[tr]);oof[te]=c.predict_proba(s.transform(X[te]))[:,1]
    return roc_auc_score(y,oof)

print(f'\n{"SLiM class":20s}{"n_chg":>6s}{"DC":>8s}{"DC+mode":>9s}{"DC+rand":>9s}{"mode-null":>10s}')
for k in SLIMS:
    y=chgdf[k].to_numpy()
    if y.sum()<40 or (len(y)-y.sum())<40:
        print(f'{k:20s}{int(y.sum()):>6d}   (skip: too imbalanced)'); continue
    a_dc=cv(DC,y,G); a_m=cv(np.concatenate([DC,MODE],1),y,G); a_r=cv(np.concatenate([DC,RAND],1),y,G)
    print(f'{k:20s}{int(y.sum()):>6d}{a_dc:>8.3f}{a_m:>9.3f}{a_r:>9.3f}{a_m-a_r:>+10.3f}')
    rows.append(dict(slim=k,n_chg=int(y.sum()),dc=a_dc,dc_mode=a_m,dc_rand=a_r,mode_beyond_null=a_m-a_r))
pd.DataFrame(rows).to_csv(OUT/'b_optionB_slim_target.tsv',sep='\t',index=False)
print('\n[done]')
