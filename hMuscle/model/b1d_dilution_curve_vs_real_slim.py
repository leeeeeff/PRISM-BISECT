#!/usr/bin/env python
"""
b1d_dilution_curve_vs_real_slim.py

Option A follow-up: quantify the pooling-dilution curve from b1c's 3-motif synthetic data as a
CONTINUOUS function of insert_fraction (= insert_length / full_sequence_length), not just 3 discrete
motif-length points, then compare directly against this project's REAL SLiM manifest insert_fraction
distribution (b_manifest_pairs.tsv, cls=slim: size/len_long) to see where real edits actually sit on
this curve -- i.e., does the geometry predict real SLiM edits should retain much/little signal after
production mean-pooling?

Per-(backbone,motif) pair: ec_delta = EC(motif) - EC(control), pool_delta = POOL(motif) - POOL(control).
Simple dilution model (per finding-pooling-coherence-filter): if the K-residue insert region is the
only place the two conditions differ, pool_delta approx (K/L) * ec_delta (to first order, ignoring
attention-mediated flanking-context shifts) -- so ||pool_delta||/||ec_delta|| should scale with
insert_fraction=K/L. Test this directly and locate the real SLiM population (median insert_fraction
0.041, IQR [0.020, 0.073], from b_manifest_pairs.tsv) on the empirical curve.
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '4'
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
MOTIFS = {'SV40_NLS': 'PKKKRKV', 'cMyc_NLS': 'PAAKRVKLD', 'Nucleoplasmin_NLS': 'KRPAATKKAGQAKKKK'}
N_BACKBONES = 200
LAYER = 9
rng = np.random.default_rng(1)  # same seed as b1c -> same backbones/scrambles

CLASS = {}
for a in 'KR': CLASS[a] = 'basic'
for a in 'DE': CLASS[a] = 'acidic'
for a in 'STNQ': CLASS[a] = 'polar'
for a in 'AVLIMC': CLASS[a] = 'hydrophobic'
for a in 'FYW': CLASS[a] = 'aromatic'
for a in 'GPH': CLASS[a] = 'special'


def within_class_scramble(motif):
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
import esm
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

EC, POOL = [], []
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
            EC.append(ec); POOL.append(pool)
        if (bstart // BATCH) % 10 == 0: print(f'  {bstart+len(batch)}/{len(df)}', flush=True)

df['EC'] = list(np.stack(EC)); df['POOL'] = list(np.stack(POOL))

rows = []
for (bi, mname), g in df.groupby(['backbone', 'motif']):
    g1 = g[g.condition == 1]; g0 = g[g.condition == 0]
    if len(g1) != 1 or len(g0) != 1: continue
    r1, r0 = g1.iloc[0], g0.iloc[0]
    if np.isnan(r1.EC).any() or np.isnan(r0.EC).any(): continue
    ec_delta = r1.EC - r0.EC; pool_delta = r1.POOL - r0.POOL
    ec_norm, pool_norm = np.linalg.norm(ec_delta), np.linalg.norm(pool_delta)
    if ec_norm < 1e-8: continue
    ratio = pool_norm / ec_norm
    insert_frac = len(r1.insert) / r1.full_len
    rows.append(dict(backbone=bi, motif=mname, insert_frac=insert_frac, ratio=ratio,
                      insert_len=len(r1.insert), full_len=r1.full_len))
res = pd.DataFrame(rows)
print(f'\nn pairs analyzed: {len(res)}')

r, p = stats.pearsonr(res.insert_frac, res.ratio)
print(f'\nPearson r(insert_fraction, ||pool_delta||/||ec_delta||) = {r:.3f}  p={p:.2e}')
print('\nBy motif:')
for mname, g in res.groupby('motif'):
    print(f'  {mname:20s} insert_frac median={g.insert_frac.median():.4f}  ratio median={g.ratio.median():.4f}')

# real SLiM population insert-fraction
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t')
slim = man[man.cls == 'slim']
real_frac = (slim['size'] / slim['len_long'])
print(f'\nReal SLiM manifest insert_fraction: median={real_frac.median():.4f}  '
      f'IQR=[{real_frac.quantile(.25):.4f},{real_frac.quantile(.75):.4f}]')

# simple log-log fit (ratio ~ insert_frac^b) for extrapolation
mask = (res.ratio > 0) & (res.insert_frac > 0)
slope, intercept, rr, pp, se = stats.linregress(np.log(res.insert_frac[mask]), np.log(res.ratio[mask]))
pred_ratio_at_real_median = np.exp(intercept + slope * np.log(real_frac.median()))
print(f'\nlog-log fit: ratio = exp({intercept:.3f}) * insert_frac^{slope:.3f}  (R²={rr**2:.3f})')
print(f'Predicted pool/editcore norm-ratio at REAL SLiM median insert_fraction ({real_frac.median():.4f}):'
      f' {pred_ratio_at_real_median:.4f}')
print(f'(for reference, observed ratios at synthetic insert_fractions {res.insert_frac.min():.4f}-{res.insert_frac.max():.4f}: '
      f'{res.ratio.min():.4f}-{res.ratio.max():.4f})')

res.to_csv(OUT / 'b1d_dilution_curve.tsv', sep='\t', index=False)
print('\n[done]')
