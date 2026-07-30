#!/usr/bin/env python
"""
Option A (edit-core): does SLiM identity live in the (pooling-diluted but surviving) EDIT-CORE
representation, or in the discarded context? Compares, for predicting SLiM-class change:
  pooled_dphi = mean-pool(HL) - mean-pool(HS)     -- what PRISM actually ingests (edit-core at ~3.8%)
  editcore    = mean over edit residues of HL      -- the SLiM residues' contextual ESM encoding
  editcore_rel= editcore - mean-pool(HL)           -- SLiM relative to its own protein context
  edit_aacomp = 20-dim AA composition of edit region -- sequence-only reference ceiling
Target = SLiM-class change (project regex). Gene-disjoint logistic AUROC. L9.
editcore >> pooled => identity survives in the core, pooling dilutes it (core-focused pooling would
rescue). editcore ~ pooled ~ chance => ESM does not linearly encode SLiM class even at the motif.
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

ROOT=Path('/home/welcome1/sw1686/DIFFUSE'); OUT=ROOT/'reports/model_interpretability_map'; PERRES=OUT/'b_perres'; L=9
SLIMS={'NLS_basic_cluster':r'[KR]{3,}|[KR].{1,2}[KR]{2,}','NES_leucine_rich':r'L.{2,3}[LIVMF].{2,3}L.{2,3}L',
 'PXXP_SH3_binding':r'P.{2}P','phospho_CK2':r'[ST].{2}[ED]','phospho_PKA':r'[RK].{2}[ST]'}
PATS={k:re.compile(v) for k,v in SLIMS.items()}
AA='ACDEFGHIKLMNPQRSTVWY'; aaidx={c:i for i,c in enumerate(AA)}

seq={}
with open(OUT/'b_extract_sequences.fasta') as f:
    cur=None
    for line in f:
        line=line.rstrip('\n')
        if line.startswith('>'): cur=line[1:]
        elif line: seq[cur]=line
man=pd.read_csv(OUT/'b_manifest_pairs.tsv',sep='\t'); man=man[man.cls=='slim'].reset_index(drop=True)

def comp(s):
    v=np.zeros(20)
    for c in s:
        if c in aaidx: v[aaidx[c]]+=1
    return v/max(len(s),1)

def feats(r):
    ls,ss=seq.get(str(r.long_idx)),seq.get(str(r.short_idx))
    pl,ps=PERRES/f'{r.long_idx}.npz',PERRES/f'{r.short_idx}.npz'
    if ls is None or ss is None or not pl.exists() or not ps.exists(): return None
    HL=np.load(pl)[f'L{L}'].astype(np.float32); HS=np.load(ps)[f'L{L}'].astype(np.float32)
    ops=SequenceMatcher(None,ls,ss,autojunk=False).get_opcodes()
    edits=[(i1,i2) for tag,i1,i2,j1,j2 in ops if tag!='equal']
    epos=[p for i1,i2 in edits for p in range(i1,min(i2,HL.shape[0]))]
    editseq=''.join(ls[i1:i2] for i1,i2 in edits)
    if len(epos)<1 or HS.shape[0]<1: return None
    dphi=HL.mean(0)-HS.mean(0); ec=HL[epos].mean(0); ecrel=ec-HL.mean(0)
    chg={k:int(len(p.findall(ls))!=len(p.findall(ss))) for k,p in PATS.items()}
    return dphi,ec,ecrel,comp(editseq),chg

D=[]
for _,r in man.iterrows():
    f=feats(r)
    if f is None: continue
    D.append((f[0],f[1],f[2],f[3],f[4],r.gene))
DPHI=np.stack([d[0] for d in D]); EC=np.stack([d[1] for d in D]); ECR=np.stack([d[2] for d in D])
AAC=np.stack([d[3] for d in D]); G=np.array([d[5] for d in D]); chgdf=pd.DataFrame([d[4] for d in D])
print(f'n={len(D)} genes={len(set(G))}')

def cv(X,y,g):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        s=StandardScaler().fit(X[tr]);c=LogisticRegression(max_iter=300,C=0.5,tol=1e-3)
        c.fit(s.transform(X[tr]),y[tr]);oof[te]=c.predict_proba(s.transform(X[te]))[:,1]
    return roc_auc_score(y,oof)

feat_sets={'pooled_dphi':DPHI,'editcore':EC,'editcore_rel':ECR,'edit_aacomp':AAC}
print(f'\n{"SLiM class":18s}{"n":>5s}  '+'  '.join(f'{n:>12s}' for n in feat_sets))
rows=[]
for k in SLIMS:
    y=chgdf[k].to_numpy()
    if y.sum()<40 or len(y)-y.sum()<40: continue
    aus={n:cv(X,y,G) for n,X in feat_sets.items()}
    print(f'{k:18s}{int(y.sum()):>5d}  '+'  '.join(f'{aus[n]:>12.3f}' for n in feat_sets))
    rows.append(dict(slim=k,n_chg=int(y.sum()),**aus))
pd.DataFrame(rows).to_csv(OUT/'b_editcore_slim.tsv',sep='\t',index=False)
print('\n[done]')
