#!/usr/bin/env python
"""
b1_dipeptide_beyond_comp.py

B1 (R2 -> R3 tail): "name" the beyond-composition reproducible structure this session's tracks kept
finding but never fully characterized (editcore beyond mono-AA-comp for N-terminal degron classes,
+0.09-0.24; the ~48-55D discarded manifold; axis0-alignment). Occam gate (S3) BEFORE crediting ESM's
640-dim editcore embedding with "beyond composition": does a much cheaper LOCAL-ORDER descriptor --
dipeptide-class composition (physicochemical-class bigrams, not full 400-dim AA dipeptides, which
would be too sparse over ~20-residue edit regions) -- already explain the beyond-mono-comp gain,
without any learned representation at all?

Same population/labels as b_option_B_slim_target.py (9 project SLiM regex classes, change-label per
manifest slim pair). Adds ONE new feature block: dipeptide-CLASS composition of the edited residues
(6 physicochemical classes -> 36 bigram bins), compared against mono-AA composition (20-dim) alone.

PREDICT-BEFORE-LOOK: if dipeptide-class composition explains most of what ESM editcore explained
beyond mono-comp for the N-terminal degron classes (DEG_Nend_Nbox etc. in the ELM-25 set -- not
directly available here, so using this project's 9-class SLIMS proxy instead, expect the same
qualitative pattern: position-defined classes gain more from local-order features), that would be a
parsimony win (Occam: order captures it, no need for a 640-dim transformer). If dipeptide-class adds
~0 beyond mono-comp too, the "beyond composition" signal genuinely requires something dipeptide-class
composition can't see (still open, doesn't collapse to a cheap descriptor).
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '4'
import re
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
rng = np.random.default_rng(0)

SLIMS = {
 'NLS_basic_cluster': r'[KR]{3,}|[KR].{1,2}[KR]{2,}', 'NES_leucine_rich': r'L.{2,3}[LIVMF].{2,3}L.{2,3}L',
 'PXXP_SH3_binding': r'P.{2}P', 'RGD_integrin': r'RGD', 'KFERQ_autophagy': r'[KQRE].{1,3}[KQRE].*F',
 'phospho_CK2': r'[ST].{2}[ED]', 'phospho_PKA': r'[RK].{2}[ST]', 'phospho_CDK': r'[ST]P[KR]',
 'CAAX_prenyl': r'C[AC].{1}[LIVMF]$'}
PATS = {k: re.compile(v) for k, v in SLIMS.items()}
def nhits(s, p): return len(p.findall(s))

AA = 'ACDEFGHIKLMNPQRSTVWY'
aaidx = {c: i for i, c in enumerate(AA)}
CLASS = {}  # 6 physicochemical classes -> reduced-alphabet dipeptide, robust for short edit regions
for a in 'KR': CLASS[a] = 'basic'
for a in 'DE': CLASS[a] = 'acidic'
for a in 'STNQ': CLASS[a] = 'polar'
for a in 'AVLIMC': CLASS[a] = 'hydrophobic'
for a in 'FYW': CLASS[a] = 'aromatic'
for a in 'GPH': CLASS[a] = 'special'
CLASSES = ['basic', 'acidic', 'polar', 'hydrophobic', 'aromatic', 'special']
classidx = {c: i for i, c in enumerate(CLASSES)}


def mono_comp(s):
    v = np.zeros(20)
    for c in s:
        if c in aaidx: v[aaidx[c]] += 1
    return v / max(len(s), 1)


def dipeptide_class_comp(s):
    v = np.zeros(36)
    n = 0
    for i in range(len(s) - 1):
        c1, c2 = CLASS.get(s[i]), CLASS.get(s[i + 1])
        if c1 is None or c2 is None: continue
        v[classidx[c1] * 6 + classidx[c2]] += 1
        n += 1
    return v / max(n, 1)


seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t'); man = man[man.cls == 'slim'].reset_index(drop=True)

MONO, DIPEP, CHG, G = [], [], [], []
for _, r in man.iterrows():
    ls, ss = seq.get(str(r.long_idx)), seq.get(str(r.short_idx))
    if ls is None or ss is None: continue
    ops = SequenceMatcher(None, ls, ss, autojunk=False).get_opcodes()
    editseq = ''.join(ls[i1:i2] for tag, i1, i2, j1, j2 in ops if tag != 'equal')
    if len(editseq) < 4: continue
    chg = {k: int(nhits(ls, p) != nhits(ss, p)) for k, p in PATS.items()}
    MONO.append(mono_comp(editseq)); DIPEP.append(dipeptide_class_comp(editseq))
    CHG.append(chg); G.append(r.gene)

MONO, DIPEP = np.stack(MONO), np.stack(DIPEP)
chgdf = pd.DataFrame(CHG); G = np.array(G)
print(f'n pairs={len(MONO)}  genes={len(set(G))}')
print('SLiM-class change prevalence:'); print(chgdf.mean().round(3).to_string())


def cv(X, y, g):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=300, C=0.5, tol=1e-3)
        c.fit(s.transform(X[tr]), y[tr]); oof[te] = c.predict_proba(s.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


print(f'\n{"SLiM class":20s}{"n_chg":>6s}{"mono":>8s}{"mono+dipep":>12s}{"dipep-beyond-mono":>18s}')
rows = []
for k in SLIMS:
    y = chgdf[k].to_numpy()
    if y.sum() < 40 or (len(y) - y.sum()) < 40:
        print(f'{k:20s}{int(y.sum()):>6d}   (skip: too imbalanced)'); continue
    a_mono = cv(MONO, y, G)
    a_both = cv(np.concatenate([MONO, DIPEP], 1), y, G)
    print(f'{k:20s}{int(y.sum()):>6d}{a_mono:>8.3f}{a_both:>12.3f}{a_both-a_mono:>+18.3f}')
    rows.append(dict(slim=k, n_chg=int(y.sum()), mono=a_mono, mono_dipep=a_both, dipep_beyond_mono=a_both - a_mono))
pd.DataFrame(rows).to_csv(OUT / 'b1_dipeptide_beyond_comp.tsv', sep='\t', index=False)
print('\n[done]')
