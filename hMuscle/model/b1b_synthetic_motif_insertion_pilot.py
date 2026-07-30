#!/usr/bin/env python
"""
b1b_synthetic_motif_insertion_pilot.py

Corrected synthetic B5 pilot (see finding-nend-rule-b5-candidate-power-dead-refined-synthetic-plan):
insert a well-established, non-regex-guessed validated SLiM (SV40 large-T-antigen NLS, "PKKKRKV" --
textbook-validated, used in essentially every NLS-tagging construct in molecular biology, chosen
specifically to avoid the regex-guessed-instance risk flagged in finding-elm-instance-power-degenerate)
at a random internal position in real backbone sequences, vs a composition-matched random permutation
of the SAME 7 residues at the SAME position. Because both conditions share the identical multiset of
residues, mono-AA composition of the insert is IDENTICAL by construction (sanity check: comp-only AUC
must be ~0.5). This directly extends this session's B1 finding (finding-b1-dipeptide-occam-fails-
positional-identity: ESM editcore beyond-comp signal for N-terminal degrons resisted collapse to
dipeptide-class composition, pointing at genuine positional/order information) -- here testing whether
ESM's local (editcore-style) representation can detect real-motif-vs-scrambled ORDER at all, in a
fully synthetic, scarcity-free (as many examples as needed), non-circular (order defined independently
of any labeling scheme this project uses) setting.

PREDICT-BEFORE-LOOK: expect editcore-based classification to clear both comp-only (~0.5 by
construction) and dipeptide-class-only (~0.5 expected, all 7 residues collapse to 2 classes: K,R->
basic (5/7), P->special (1/7), V->hydrophobic (1/7), so class-bigram patterns barely differ between
real and scrambled) by a wide margin -- ESM's attention easily distinguishes "PKKKRKV" from a
scrambled anagram since it's WELL beyond superficial statistics (a genuine order-sensitive language
model task). This is a sanity check that local order-sensitivity EXISTS in principle -- a necessary
but not sufficient condition for building real isoform-supervision on top of it (A2's decisive fail
means the SAME order info likely won't survive PRODUCTION pooling even if editcore sees it clearly
here; that gap is the next open question, not resolved by this pilot).
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
NLS = 'PKKKRKV'  # SV40 large-T-antigen NLS, canonical validated monopartite NLS
N_BACKBONES = 150
LAYER = 9
rng = np.random.default_rng(0)

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


# ---- load backbone sequences (reuse manifest long-sequences, length>=150 for room to insert) ----
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
print(f'backbones selected: {len(backbones)}')

records = []  # (backbone_id, condition, full_sequence, insert_start)
for bi, bseq in enumerate(backbones):
    pos = rng.integers(int(0.1 * len(bseq)), int(0.9 * len(bseq)))
    motif_seq = bseq[:pos] + NLS + bseq[pos:]
    # within-class permutation control: fixes P (pos0, class 'special') and V (pos6, class
    # 'hydrophobic') in place, only shuffles the 5 basic-class residues (K,K,K,R,K) among
    # themselves -- keeps the CLASS-level sequence (and hence dipeptide-class composition)
    # IDENTICAL between conditions, isolating within-class identity-order (K-vs-R placement)
    # as the only remaining difference. A full random permutation (tried first) let P/V land
    # at non-extremal positions, which a coarse class-bigram already resolves almost perfectly
    # (0.994 AUC) -- a confound this control removes.
    basics = [c for c in NLS if CLASS[c] == 'basic']
    rng.shuffle(basics)
    scramble = NLS[0] + ''.join(basics) + NLS[-1]
    control_seq = bseq[:pos] + scramble + bseq[pos:]
    records.append(dict(backbone=bi, condition=1, seq=motif_seq, insert_start=pos, insert=NLS))
    records.append(dict(backbone=bi, condition=0, seq=control_seq, insert_start=pos, insert=scramble))
df = pd.DataFrame(records)
print(f'total sequences to embed: {len(df)}  (motif={sum(df.condition==1)}, control={sum(df.condition==0)})')

# ---- ESM-2 layer-9 forward pass ----
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
import esm
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

EC, MONO, DIPEP = [], [], []
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
            ec = reps[k, 1 + start:1 + start + len(ins), :].mean(0).cpu().float().numpy()
            EC.append(ec); MONO.append(mono_comp(ins)); DIPEP.append(dipep_class(ins))
        if (bstart // BATCH) % 5 == 0:
            print(f'  {bstart+len(batch)}/{len(df)}', flush=True)

EC, MONO, DIPEP = np.stack(EC), np.stack(MONO), np.stack(DIPEP)
Y = df['condition'].to_numpy(); G = df['backbone'].to_numpy()
keep = ~np.isnan(EC).any(1)
if (~keep).sum():
    print(f'  [drop] {(~keep).sum()} sequences with NaN editcore (insert truncated by MAXLEN)')
EC, MONO, DIPEP, Y, G = EC[keep], MONO[keep], DIPEP[keep], Y[keep], G[keep]
print(f'\nn={len(Y)}  backbones(groups)={len(set(G))}  label balance: {Y.sum()}/{len(Y)-Y.sum()}')


def cv(X, y, g):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=500, C=1.0)
        c.fit(s.transform(X[tr]), y[tr]); oof[te] = c.predict_proba(s.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


a_mono = cv(MONO, Y, G)
a_dipep = cv(np.concatenate([MONO, DIPEP], 1), Y, G)
a_ec = cv(EC, Y, G)
a_mono_ec = cv(np.concatenate([MONO, EC], 1), Y, G)
print(f'\n=== real-NLS vs scrambled-control classification (backbone-disjoint CV) ===')
print(f'  mono-comp only        AUC={a_mono:.3f}   (sanity check: should be ~0.5, identical multiset)')
print(f'  mono+dipeptide-class  AUC={a_dipep:.3f}   (expect ~0.5, 5/7 residues collapse to "basic")')
print(f'  ESM editcore only     AUC={a_ec:.3f}')
print(f'  mono+ESM editcore     AUC={a_mono_ec:.3f}   (editcore-beyond-comp = {a_mono_ec-a_mono:+.3f})')

out = OUT / 'assets' / 'b1b_synthetic_motif_pilot_results.txt'
out.parent.mkdir(exist_ok=True)
with open(out, 'w') as f:
    f.write(f'n={len(Y)} backbones={len(set(G))}\n')
    f.write(f'mono={a_mono:.4f} mono_dipep={a_dipep:.4f} ec={a_ec:.4f} mono_ec={a_mono_ec:.4f}\n')
print(f'\nsaved -> {out}')
print('[done]')
