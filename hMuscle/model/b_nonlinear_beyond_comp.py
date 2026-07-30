#!/usr/bin/env python
"""
Option B (non-linear, composition-controlled) — the sharpest remaining SLiM test. edit-core (§A)
showed SLiM-class is largely COMPOSITION (edit_aacomp >= ESM editcore, linear). So the decisive
question is: does ANY ESM representation predict SLiM-class BEYOND edit AA-composition, and does
NON-LINEARITY rescue signal the linear probes missed?

For each SLiM class (gene-disjoint AUROC, GradientBoosting = non-linear):
  comp          : 20-dim edit AA composition (the ceiling from §A)
  editcore(nl)  : ESM edit-core, non-linear
  comp+editcore : does editcore ADD beyond composition, non-linearly?
  discard(nl)   : the pooling-discarded top-mode, non-linear (does non-linearity rescue §14e's null?)
If comp+editcore ~ comp and discard(nl) ~ chance => ESM adds ~nothing beyond composition for
regex-SLiM; the biological thesis needs higher-specificity labels, not a richer probe.
"""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[v]='4'
import re, numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT=Path('/home/welcome1/sw1686/DIFFUSE'); OUT=ROOT/'reports/model_interpretability_map'; PERRES=OUT/'b_perres'; L=9
SLIMS={'NLS_basic_cluster':r'[KR]{3,}|[KR].{1,2}[KR]{2,}','PXXP_SH3_binding':r'P.{2}P',
 'phospho_CK2':r'[ST].{2}[ED]','phospho_PKA':r'[RK].{2}[ST]'}
PATS={k:re.compile(v) for k,v in SLIMS.items()}; AA='ACDEFGHIKLMNPQRSTVWY'; aaidx={c:i for i,c in enumerate(AA)}
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
    if len(epos)<1: return None
    Deq=[HL[i1+o]-HS[j1+o] for tag,i1,i2,j1,j2 in ops if tag=='equal' for o in range(i2-i1)
         if i1+o<HL.shape[0] and j1+o<HS.shape[0]]
    if len(Deq)<24: return None
    Deq=np.stack(Deq); _,S,Vt=np.linalg.svd(Deq-Deq.mean(0),full_matrices=False); mode=S[0]*Vt[0]
    ec=HL[epos].mean(0); editseq=''.join(ls[i1:i2] for i1,i2 in edits)
    chg={k:int(len(p.findall(ls))!=len(p.findall(ss))) for k,p in PATS.items()}
    return comp(editseq),ec,mode,chg
D=[]
for _,r in man.iterrows():
    f=feats(r)
    if f: D.append((f[0],f[1],f[2],f[3],r.gene))
COMP=np.stack([d[0] for d in D]); EC=np.stack([d[1] for d in D]); MODE=np.stack([d[2] for d in D])
G=np.array([d[4] for d in D]); chgdf=pd.DataFrame([d[3] for d in D])
print(f'n={len(D)} genes={len(set(G))}')
def cv(X,y,g):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        c=HistGradientBoostingClassifier(max_iter=200,max_depth=3,learning_rate=0.06,l2_regularization=1.0)
        c.fit(X[tr],y[tr]); oof[te]=c.predict_proba(X[te])[:,1]
    return roc_auc_score(y,oof)
sets={'comp':COMP,'editcore':EC,'comp+editcore':np.concatenate([COMP,EC],1),'discard_mode':MODE}
print(f'\n{"SLiM class":18s}{"n":>5s}  '+'  '.join(f'{n:>14s}' for n in sets))
for k in SLIMS:
    y=chgdf[k].to_numpy()
    if y.sum()<40 or len(y)-y.sum()<40: continue
    aus={n:cv(X,y,G) for n,X in sets.items()}
    print(f'{k:18s}{int(y.sum()):>5d}  '+'  '.join(f'{aus[n]:>14.3f}' for n in sets)
          +f'   |  editcore-beyond-comp={aus["comp+editcore"]-aus["comp"]:+.3f}')
print('\n[done]')
