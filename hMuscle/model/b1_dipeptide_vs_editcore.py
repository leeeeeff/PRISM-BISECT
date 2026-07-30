#!/usr/bin/env python
"""
b1_dipeptide_vs_editcore.py

B1 (R2 -> R3 tail), corrected target: b_elm_beyond_comp.py already found the classes where ESM
editcore genuinely beats mono-AA composition -- almost entirely N-terminal degron-type ELM classes
(DEG_Nend_Nbox_1 +0.239, DEG_Nend_UBRbox_2 +0.236, LIG_BIR_II_1 +0.197, DOC_MAPK_gen_1 +0.106,
DEG_Nend_UBRbox_1 +0.093 -- from reports/model_interpretability_map/b_elm_beyond_comp.tsv). Occam
gate (S3): does a MUCH cheaper local-order descriptor (dipeptide-class composition, 36-dim
physicochemical-class bigrams -- NOT full 400-dim AA dipeptides, too sparse for ~20aa edit regions)
already explain most of that gain, without any ESM embedding at all?

Same exact population/labels/editcore construction as b_elm_beyond_comp.py (elms_index.tsv regexes,
b_manifest_pairs.tsv slim population, layer-9 edit-core mean-pool). Adds DIPEP (36-dim) alongside
COMP (20-dim mono) and EC (640-dim ESM editcore, reused as the established upper bound).

PREDICT-BEFORE-LOOK: given these are N-terminal DEGRON classes (N-end rule: identity of position-1/2
residues after any processing, e.g. bulky/hydrophobic N-terminal residues -> UBR-box recognition),
the signal is fundamentally about WHICH specific residue sits at specific positions, not just local
dipeptide-class pattern -- predict dipep-beyond-comp will be small (~0.02-0.05), well short of ESM's
0.09-0.24, i.e. Occam does NOT fully collapse this signal (some genuine positional/identity
information survives that a coarse 6-class bigram alphabet can't capture). A large dipep gain would
be a surprise parsimony win; a null result still narrows "what R2/R3 descriptor could name this" by
ruling out simple local-order.
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[v] = '4'
import re
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
PERRES = OUT / 'b_perres'
L = 9
SCRATCH = Path('/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/76e93aef-8837-4578-8767-18a31bfa00ce/scratchpad')
AA = 'ACDEFGHIKLMNPQRSTVWY'
aaidx = {c: i for i, c in enumerate(AA)}
CLASS = {}
for a in 'KR': CLASS[a] = 'basic'
for a in 'DE': CLASS[a] = 'acidic'
for a in 'STNQ': CLASS[a] = 'polar'
for a in 'AVLIMC': CLASS[a] = 'hydrophobic'
for a in 'FYW': CLASS[a] = 'aromatic'
for a in 'GPH': CLASS[a] = 'special'
CLASSES = ['basic', 'acidic', 'polar', 'hydrophobic', 'aromatic', 'special']
classidx = {c: i for i, c in enumerate(CLASSES)}

TOP_CLASSES = ['DEG_Nend_Nbox_1', 'DEG_Nend_UBRbox_2', 'LIG_BIR_II_1', 'DOC_MAPK_gen_1', 'DEG_Nend_UBRbox_1']

elm = pd.read_csv(SCRATCH / 'elms_index.tsv', sep='\t', comment='#')
elm.columns = [c.strip('"') for c in elm.columns]
pats = {}
for _, r in elm.iterrows():
    try: pats[r['ELMIdentifier']] = re.compile(r['Regex'])
    except re.error: pass

seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t'); man = man[man.cls == 'slim'].reset_index(drop=True)

need = set(man.long_idx.astype(str)) | set(man.short_idx.astype(str))
hitcache = {}
for iid in need:
    s = seq.get(iid)
    hitcache[iid] = {n: len(p.findall(s)) for n, p in pats.items() if n in TOP_CLASSES} if s else None


def comp(s):
    v = np.zeros(20)
    for c in s:
        if c in aaidx: v[aaidx[c]] += 1
    return v / max(len(s), 1)


def dipep_class(s):
    v = np.zeros(36); n = 0
    for i in range(len(s) - 1):
        c1, c2 = CLASS.get(s[i]), CLASS.get(s[i + 1])
        if c1 is None or c2 is None: continue
        v[classidx[c1] * 6 + classidx[c2]] += 1; n += 1
    return v / max(n, 1)


def feats(r):
    ls, ss = seq.get(str(r.long_idx)), seq.get(str(r.short_idx))
    pl = PERRES / f'{r.long_idx}.npz'
    if ls is None or ss is None or not pl.exists(): return None
    HL = np.load(pl)[f'L{L}'].astype(np.float32)
    ops = SequenceMatcher(None, ls, ss, autojunk=False).get_opcodes()
    epos = [p for tag, i1, i2, j1, j2 in ops if tag != 'equal' for p in range(i1, min(i2, HL.shape[0]))]
    if not epos: return None
    editseq = ''.join(ls[i1:i2] for tag, i1, i2, j1, j2 in ops if tag != 'equal')
    return HL[epos].mean(0), comp(editseq), dipep_class(editseq)


EC, COMP, DIPEP, CHG, G = [], [], [], [], []
for _, r in man.iterrows():
    hl, hs = hitcache.get(str(r.long_idx)), hitcache.get(str(r.short_idx))
    f = feats(r)
    if hl is None or hs is None or f is None: continue
    ec, cp, dp = f
    EC.append(ec); COMP.append(cp); DIPEP.append(dp)
    CHG.append({n: int(hl[n] != hs[n]) for n in TOP_CLASSES}); G.append(r.gene)
EC, COMP, DIPEP = np.stack(EC), np.stack(COMP), np.stack(DIPEP)
chgdf = pd.DataFrame(CHG); G = np.array(G)
print(f'n pairs={len(EC)} genes={len(set(G))}')


def cv(X, y, g):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        c = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.06, l2_regularization=1.0)
        c.fit(X[tr], y[tr]); oof[te] = c.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


print(f'\n{"ELM class":20s}{"n_chg":>6s}{"comp":>7s}{"c+dipep":>8s}{"dipep-beyond":>13s}{"c+ec(prior)":>12s}{"ec-beyond(prior)":>17s}')
rows = []
prior = {'DEG_Nend_Nbox_1': 0.239, 'DEG_Nend_UBRbox_2': 0.236, 'LIG_BIR_II_1': 0.197,
         'DOC_MAPK_gen_1': 0.106, 'DEG_Nend_UBRbox_1': 0.093}
for n in TOP_CLASSES:
    y = chgdf[n].to_numpy()
    if y.sum() < 20 or (len(y) - y.sum()) < 20:
        print(f'{n:20s}{int(y.sum()):>6d}   (skip: too imbalanced, n={y.sum()})'); continue
    a_c = cv(COMP, y, G); a_cd = cv(np.concatenate([COMP, DIPEP], 1), y, G)
    print(f'{n:20s}{int(y.sum()):>6d}{a_c:>7.3f}{a_cd:>8.3f}{a_cd-a_c:>+13.3f}{a_c+prior[n]:>12.3f}{prior[n]:>+17.3f}')
    rows.append(dict(elm=n, n_chg=int(y.sum()), comp=a_c, comp_dipep=a_cd,
                      dipep_beyond_comp=a_cd - a_c, ec_beyond_comp_prior=prior[n]))
pd.DataFrame(rows).to_csv(OUT / 'b1_dipeptide_vs_editcore.tsv', sep='\t', index=False)
print('\n[done]')
