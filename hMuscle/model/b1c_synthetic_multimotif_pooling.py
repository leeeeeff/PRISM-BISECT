#!/usr/bin/env python
"""
b1c_synthetic_multimotif_pooling.py

Extends b1b_synthetic_motif_insertion_pilot.py (single motif SV40 NLS, editcore only, AUC=0.759
beyond mono+dipeptide-class) two ways per user request:
  (C) robustness/generalization -- 3 validated, textbook motifs instead of 1 (SV40 NLS "PKKKRKV",
      c-Myc NLS "PAAKRVKLD", Nucleoplasmin bipartite NLS "KRPAATKKAGQAKKKK"), more backbones.
  (A) pooling survival -- does the SAME within-class-identity-order signal editcore captures also
      show up in a PRODUCTION-style POOLED feature (mean over the WHOLE sequence, exactly how PRISM's
      real input is built), or does it die at pooling the way A2's real signal did
      (finding-a2-nterm60-dead-4tier)?

Same within-class-permutation control design as b1b (fixes which POSITIONS hold which CLASS of
residue, only permutes identity WITHIN class -- keeps mono-AA and dipeptide-class composition of the
insert identical between conditions by construction, isolating within-class residue-identity-order).

PREDICT-BEFORE-LOOK (per approach-proxy-metric-vs-deployed-task-ablation, this session's 3x-confirmed
pattern): expect editcore to replicate/strengthen the 0.759 signal across motifs (real, local,
order-sensitive encoding). Expect POOLED to lose most/all of this signal once the insert is diluted
into a mean over the whole backbone (typically 150-1000+ residues vs a 7-16 residue insert) --
predict pooled AUC close to 0.5, i.e. a 4th independent confirmation of the encoding-without-usage
pattern. A surprise (pooled retains real signal) would be the first genuine counter-example to that
pattern and would deserve serious follow-up (a real, tractable path to recovering positional signal
without an architecture change).
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '4'
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
MOTIFS = {
    'SV40_NLS': 'PKKKRKV',
    'cMyc_NLS': 'PAAKRVKLD',
    'Nucleoplasmin_NLS': 'KRPAATKKAGQAKKKK',
}
N_BACKBONES = 200
LAYER = 9
rng = np.random.default_rng(1)

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


def mono_comp(s):
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


def within_class_scramble(motif):
    """Fixes which position holds which CLASS; shuffles identity only among same-class positions.
    Keeps class-level sequence (and hence dipeptide-class composition) exactly identical."""
    chars = list(motif)
    by_class = {}
    for i, c in enumerate(chars):
        by_class.setdefault(CLASS.get(c, c), []).append(i)
    out = chars[:]
    for cls, idxs in by_class.items():
        if len(idxs) < 2: continue
        vals = [chars[i] for i in idxs]
        rng.shuffle(vals)
        for i, v in zip(idxs, vals): out[i] = v
    return ''.join(out)


# ---- backbone sequences ----
seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
long_seqs = list(set(s for s in seq.values() if len(s) >= 150))
rng.shuffle(long_seqs)
backbones = long_seqs[:N_BACKBONES]
print(f'backbones: {len(backbones)}  motifs: {list(MOTIFS)}')

records = []
for bi, bseq in enumerate(backbones):
    pos = rng.integers(int(0.1 * len(bseq)), int(0.9 * len(bseq)))
    for mname, motif in MOTIFS.items():
        motif_seq = bseq[:pos] + motif + bseq[pos:]
        scramble = within_class_scramble(motif)
        control_seq = bseq[:pos] + scramble + bseq[pos:]
        records.append(dict(backbone=bi, motif=mname, condition=1, seq=motif_seq,
                             insert_start=pos, insert=motif, full_len=len(motif_seq)))
        records.append(dict(backbone=bi, motif=mname, condition=0, seq=control_seq,
                             insert_start=pos, insert=scramble, full_len=len(control_seq)))
df = pd.DataFrame(records)
print(f'total sequences: {len(df)}')

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
import esm
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

EC, POOL, MONO, DIPEP = [], [], [], []
BATCH = 16
with torch.no_grad():
    for bstart in range(0, len(df), BATCH):
        batch = df.iloc[bstart:bstart + BATCH]
        seqs_batch = [(str(i), row['seq'][:1022]) for i, row in batch.iterrows()]
        _, _, tokens = batch_converter(seqs_batch)
        tokens = tokens.to(device)
        results = model(tokens, repr_layers=[LAYER], return_contacts=False)
        reps = results['representations'][LAYER]
        for k, (_, row) in enumerate(batch.iterrows()):
            start, ins = row['insert_start'], row['insert']
            L = min(row['full_len'], 1022)
            ec = reps[k, 1 + start:1 + start + len(ins), :].mean(0).cpu().float().numpy()
            pool = reps[k, 1:1 + L, :].mean(0).cpu().float().numpy()
            EC.append(ec); POOL.append(pool); MONO.append(mono_comp(ins)); DIPEP.append(dipep_class(ins))
        if (bstart // BATCH) % 10 == 0:
            print(f'  {bstart+len(batch)}/{len(df)}', flush=True)

EC, POOL, MONO, DIPEP = np.stack(EC), np.stack(POOL), np.stack(MONO), np.stack(DIPEP)
Y = df['condition'].to_numpy(); G = df['backbone'].to_numpy(); M = df['motif'].to_numpy()
keep = ~(np.isnan(EC).any(1) | np.isnan(POOL).any(1))
if (~keep).sum():
    print(f'  [drop] {(~keep).sum()} NaN (truncated insert)')
EC, POOL, MONO, DIPEP, Y, G, M = EC[keep], POOL[keep], MONO[keep], DIPEP[keep], Y[keep], G[keep], M[keep]
print(f'\nn={len(Y)}  backbones={len(set(G))}  label balance={Y.sum()}/{len(Y)-Y.sum()}')


def cv(X, y, g):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=500, C=1.0)
        c.fit(s.transform(X[tr]), y[tr]); oof[te] = c.predict_proba(s.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


print('\n=== per-motif (editcore vs pooled), backbone-disjoint CV ===')
rows = []
for mname in MOTIFS:
    msk = M == mname
    a_mono = cv(MONO[msk], Y[msk], G[msk])
    a_dipep = cv(np.concatenate([MONO[msk], DIPEP[msk]], 1), Y[msk], G[msk])
    a_ec = cv(EC[msk], Y[msk], G[msk])
    a_pool = cv(POOL[msk], Y[msk], G[msk])
    print(f'  {mname:20s} n={msk.sum():4d}  mono={a_mono:.3f}  mono+dipep={a_dipep:.3f}  '
          f'editcore={a_ec:.3f}  pooled={a_pool:.3f}  (pool-ec gap={a_pool-a_ec:+.3f})')
    rows.append(dict(motif=mname, n=int(msk.sum()), mono=a_mono, mono_dipep=a_dipep, editcore=a_ec, pooled=a_pool))

print('\n=== combined (all 3 motifs pooled together) ===')
a_mono = cv(MONO, Y, G); a_dipep = cv(np.concatenate([MONO, DIPEP], 1), Y, G)
a_ec = cv(EC, Y, G); a_pool = cv(POOL, Y, G)
print(f'  n={len(Y)}  mono={a_mono:.3f}  mono+dipep={a_dipep:.3f}  editcore={a_ec:.3f}  pooled={a_pool:.3f}')
print(f'  editcore-beyond-comp = {a_ec-a_mono:+.3f}   pooled-beyond-comp = {a_pool-a_mono:+.3f}   '
      f'pooling-survival-gap (editcore-pooled) = {a_ec-a_pool:+.3f}')

pd.DataFrame(rows).to_csv(OUT / 'b1c_synthetic_multimotif_results.tsv', sep='\t', index=False)
with open(OUT / 'assets' / 'b1c_synthetic_multimotif_combined.txt', 'w') as f:
    f.write(f'n={len(Y)} mono={a_mono:.4f} mono_dipep={a_dipep:.4f} editcore={a_ec:.4f} pooled={a_pool:.4f}\n')
print('\n[done]')
