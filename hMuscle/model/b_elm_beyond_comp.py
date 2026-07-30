#!/usr/bin/env python
"""
Option A (ELM route) — the decisive composition-controlled SLiM test with HIGH-SPECIFICITY labels.
Uses the 353 ELM class regexes (elms_index.tsv) instead of the 10 broad ones. For each ELM class with
enough changed pairs, ask: does the ESM edit-core predict the motif change BEYOND edit AA-composition
(non-linear)? If even specific ELM classes give editcore-beyond-comp ~ 0, the "label-limited" verdict
strengthens (label-independent); systematic >0 for specific motifs = ESM captures SLiM beyond composition.
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

# ---- load ELM regexes ----
elm=pd.read_csv(SCRATCH/'elms_index.tsv', sep='\t', comment='#')
elm.columns=[c.strip('"') for c in elm.columns]
pats={}
for _,r in elm.iterrows():
    try: pats[r['ELMIdentifier']]=(re.compile(r['Regex']), float(r['Probability']))
    except re.error: pass
print(f'ELM classes compiled: {len(pats)}/{len(elm)}')

seq={}
with open(OUT/'b_extract_sequences.fasta') as f:
    cur=None
    for line in f:
        line=line.rstrip('\n')
        if line.startswith('>'): cur=line[1:]
        elif line: seq[cur]=line
man=pd.read_csv(OUT/'b_manifest_pairs.tsv',sep='\t'); man=man[man.cls=='slim'].reset_index(drop=True)

# precompute hit-count vector per unique isoform
need=set(man.long_idx.astype(str))|set(man.short_idx.astype(str))
hitcache={}
names=list(pats)
for iid in need:
    s=seq.get(iid)
    hitcache[iid]=np.array([len(pats[n][0].findall(s)) for n in names]) if s else None

def comp(s):
    v=np.zeros(20)
    for c in s:
        if c in aaidx: v[aaidx[c]]+=1
    return v/max(len(s),1)

def editcore(r):
    ls,ss=seq.get(str(r.long_idx)),seq.get(str(r.short_idx))
    pl=PERRES/f'{r.long_idx}.npz'
    if ls is None or ss is None or not pl.exists(): return None,None
    HL=np.load(pl)[f'L{L}'].astype(np.float32)
    ops=SequenceMatcher(None,ls,ss,autojunk=False).get_opcodes()
    epos=[p for tag,i1,i2,j1,j2 in ops if tag!='equal' for p in range(i1,min(i2,HL.shape[0]))]
    if not epos: return None,None
    editseq=''.join(ls[i1:i2] for tag,i1,i2,j1,j2 in ops if tag!='equal')
    return HL[epos].mean(0), comp(editseq)

EC=[];COMP=[];CHG=[];G=[]
for _,r in man.iterrows():
    hl,hs=hitcache.get(str(r.long_idx)),hitcache.get(str(r.short_idx))
    ec,cp=editcore(r)
    if hl is None or hs is None or ec is None: continue
    EC.append(ec);COMP.append(cp);CHG.append((hl!=hs).astype(int));G.append(r.gene)
EC=np.stack(EC);COMP=np.stack(COMP);CHG=np.stack(CHG);G=np.array(G)
chgdf=pd.DataFrame(CHG,columns=names)
print(f'n pairs={len(EC)} genes={len(set(G))}')

def cv(X,y,g):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        c=HistGradientBoostingClassifier(max_iter=150,max_depth=3,learning_rate=0.06,l2_regularization=1.0)
        c.fit(X[tr],y[tr]); oof[te]=c.predict_proba(X[te])[:,1]
    return roc_auc_score(y,oof)

# select specific (low background prob) + enough-changed classes
cand=[]
for n in names:
    y=chgdf[n].to_numpy()
    if y.sum()>=40 and (len(y)-y.sum())>=40:
        cand.append((n, int(y.sum()), pats[n][1]))
cand.sort(key=lambda x:x[2])   # most specific first
cand=cand[:25]
print(f'\ntestable ELM classes (n_chg>=40): {len(cand)} (showing most-specific 25)')
print(f'{"ELM class":26s}{"prob":>10s}{"n":>5s}{"comp":>7s}{"c+ec":>7s}{"ec":>7s}{"ec-beyond":>10s}')
rows=[]
for n,ns,pr in cand:
    y=chgdf[n].to_numpy()
    a_c=cv(COMP,y,G); a_ce=cv(np.concatenate([COMP,EC],1),y,G); a_e=cv(EC,y,G)
    print(f'{n:26s}{pr:>10.2e}{ns:>5d}{a_c:>7.3f}{a_ce:>7.3f}{a_e:>7.3f}{a_ce-a_c:>+10.3f}')
    rows.append(dict(elm=n,prob=pr,n_chg=ns,comp=a_c,comp_ec=a_ce,ec=a_e,ec_beyond_comp=a_ce-a_c))
res=pd.DataFrame(rows); res.to_csv(OUT/'b_elm_beyond_comp.tsv',sep='\t',index=False)
print(f'\n=== SUMMARY over {len(res)} specific ELM classes ===')
print(f'  median editcore-beyond-comp = {res.ec_beyond_comp.median():+.4f}')
print(f'  classes with beyond-comp > +0.02: {(res.ec_beyond_comp>0.02).sum()}/{len(res)}')
print(f'  median comp AUROC = {res.comp.median():.3f}  median editcore-alone = {res.ec.median():.3f}')
print('\n[done]')
