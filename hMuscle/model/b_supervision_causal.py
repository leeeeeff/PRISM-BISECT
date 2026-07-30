#!/usr/bin/env python
"""
Option B (supervision-causal, no retraining) — tests the §14e claim "SLiM bottleneck = supervision".
The positional-SLiM encoding (+0.24 beyond comp) was measured on the EDIT-CORE. But PRISM ingests the
MEAN-POOLED delta. So: is the N-end-degron signal accessible from PRISM's ACTUAL INPUT (pooled delta),
or only from the edit-core it never sees?

For positional ELM classes (encoded) + composition-ELM controls, non-linear gene-disjoint AUROC:
  comp            : 20-dim edit AA composition (ceiling)
  comp+pooled     : + mean-pool(HL)-mean-pool(HS)  = PRISM's actual input  -> pooled-beyond-comp
  comp+editcore   : + edit-core mean                = a core-focused model  -> editcore-beyond-comp
pooled-beyond-comp ~ editcore-beyond-comp => pure SUPERVISION bottleneck (PRISM input carries it).
pooled-beyond-comp << editcore-beyond-comp => pooling already destroyed accessibility =>
                                              SUPERVISION + ARCHITECTURE joint bottleneck.
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
SCRATCH=Path('/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/76e93aef-8837-4578-8767-18a31bfa00ce/scratchpad')
AA='ACDEFGHIKLMNPQRSTVWY'; aaidx={c:i for i,c in enumerate(AA)}
POSITIONAL=['DEG_Nend_UBRbox_1','DEG_Nend_Nbox_1','DEG_Nend_UBRbox_2','LIG_BIR_II_1']  # encoded (+beyond comp)
CONTROL   =['LIG_SH2_STAT5','TRG_ENDOCYTIC_2','LIG_SH2_SFK_2']                          # composition-driven (~0)
WANT=POSITIONAL+CONTROL

elm=pd.read_csv(SCRATCH/'elms_index.tsv',sep='\t',comment='#'); elm.columns=[c.strip('"') for c in elm.columns]
pats={r['ELMIdentifier']:re.compile(r['Regex']) for _,r in elm.iterrows() if r['ELMIdentifier'] in WANT}
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
    ls,ss=seq.get(str(r.long_idx)),seq.get(str(r.short_idx)); pl,ps=PERRES/f'{r.long_idx}.npz',PERRES/f'{r.short_idx}.npz'
    if ls is None or ss is None or not pl.exists() or not ps.exists(): return None
    HL=np.load(pl)[f'L{L}'].astype(np.float32); HS=np.load(ps)[f'L{L}'].astype(np.float32)
    ops=SequenceMatcher(None,ls,ss,autojunk=False).get_opcodes()
    epos=[p for tag,i1,i2,j1,j2 in ops if tag!='equal' for p in range(i1,min(i2,HL.shape[0]))]
    if not epos: return None
    editseq=''.join(ls[i1:i2] for tag,i1,i2,j1,j2 in ops if tag!='equal')
    pooled=HL.mean(0)-HS.mean(0); ec=HL[epos].mean(0)
    chg={k:int(len(p.findall(ls))!=len(p.findall(ss))) for k,p in pats.items()}
    return comp(editseq),pooled,ec,chg
COMP=[];POOL=[];EC=[];CHG=[];G=[]
for _,r in man.iterrows():
    f=feats(r)
    if f: COMP.append(f[0]);POOL.append(f[1]);EC.append(f[2]);CHG.append(f[3]);G.append(r.gene)
COMP=np.stack(COMP);POOL=np.stack(POOL);EC=np.stack(EC);G=np.array(G);chgdf=pd.DataFrame(CHG)
print(f'n={len(COMP)} genes={len(set(G))}')
def cv(X,y,g):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        c=HistGradientBoostingClassifier(max_iter=150,max_depth=3,learning_rate=0.06,l2_regularization=1.0)
        c.fit(X[tr],y[tr]); oof[te]=c.predict_proba(X[te])[:,1]
    return roc_auc_score(y,oof)
print(f'\n{"ELM class":22s}{"type":>6s}{"n":>5s}{"comp":>7s}{"+pooled":>9s}{"+editcore":>10s}  {"pool-Δ":>7s}{"ec-Δ":>7s}')
for k in WANT:
    if k not in chgdf.columns: continue
    y=chgdf[k].to_numpy()
    if y.sum()<40 or len(y)-y.sum()<40: print(f'{k:22s} (skip)'); continue
    a_c=cv(COMP,y,G); a_p=cv(np.concatenate([COMP,POOL],1),y,G); a_e=cv(np.concatenate([COMP,EC],1),y,G)
    typ='POS' if k in POSITIONAL else 'ctrl'
    print(f'{k:22s}{typ:>6s}{int(y.sum()):>5d}{a_c:>7.3f}{a_p:>9.3f}{a_e:>10.3f}  {a_p-a_c:>+7.3f}{a_e-a_c:>+7.3f}')
print('\n[done]')
