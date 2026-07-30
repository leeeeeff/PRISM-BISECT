#!/usr/bin/env python
"""
b1e_dilution_law_real_slim_validation.py

Cheap validation of b1d's synthetic dilution law (||pool_delta||/||ec_delta|| ~ insert_fraction^0.48,
r=0.482, p=2e-31) using REAL SLiM manifest pairs -- no new GPU extraction, reuses cached per-residue
embeddings (b_perres/) already computed earlier this session. Real pairs differ from the synthetic
setup (long vs short, not motif vs scrambled-control), so the exact quantities differ slightly:
  editcore  = mean ESM embedding over the LONG isoform's edited (non-aligned) residues
  pooled    = HL.mean(0) - HS.mean(0)  (production-style whole-sequence pooled difference)
Tests whether ||pooled|| / ||editcore|| still correlates with insert_fraction (size/len_long) in real
data the way the synthetic law predicts -- if yes, the dilution law generalizes beyond the synthetic
construction and can be used to sanity-check window-size choices for any future A2-style redesign
BEFORE committing to a new full-corpus extraction.
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '4'
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from scipy import stats

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
PERRES = OUT / 'b_perres'
L = 9

seq = {}
with open(OUT / 'b_extract_sequences.fasta') as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(OUT / 'b_manifest_pairs.tsv', sep='\t')
slim = man[man.cls == 'slim'].reset_index(drop=True)

rows = []
for _, r in slim.iterrows():
    ls, ss = seq.get(str(r.long_idx)), seq.get(str(r.short_idx))
    pl, ps = PERRES / f'{r.long_idx}.npz', PERRES / f'{r.short_idx}.npz'
    if ls is None or ss is None or not pl.exists() or not ps.exists(): continue
    HL = np.load(pl)[f'L{L}'].astype(np.float32)
    HS = np.load(ps)[f'L{L}'].astype(np.float32)
    ops = SequenceMatcher(None, ls, ss, autojunk=False).get_opcodes()
    epos_long = [p for tag, i1, i2, j1, j2 in ops if tag != 'equal' for p in range(i1, min(i2, HL.shape[0]))]
    if not epos_long: continue
    editcore = HL[epos_long].mean(0)
    pooled = HL.mean(0) - HS.mean(0)
    ec_norm, pool_norm = np.linalg.norm(editcore), np.linalg.norm(pooled)
    if ec_norm < 1e-8: continue
    insert_frac = r['size'] / r['len_long']
    rows.append(dict(gene=r.gene, insert_frac=insert_frac, ratio=pool_norm / ec_norm,
                      ec_norm=ec_norm, pool_norm=pool_norm, size=r['size'], len_long=r['len_long']))

res = pd.DataFrame(rows)
print(f'n real SLiM pairs analyzed: {len(res)}')
r, p = stats.pearsonr(res.insert_frac, res.ratio)
print(f'Pearson r(insert_fraction, ||pooled||/||editcore||) = {r:.3f}  p={p:.2e}')

mask = (res.ratio > 0) & (res.insert_frac > 0)
slope, intercept, rr, pp, se = stats.linregress(np.log(res.insert_frac[mask]), np.log(res.ratio[mask]))
print(f'log-log fit (real data): ratio = exp({intercept:.3f}) * insert_frac^{slope:.3f}  (R²={rr**2:.3f})')
print(f'(synthetic pilot fit was: ratio = exp(-0.292) * insert_frac^0.480, R²=0.263)')

print(f'\nreal insert_fraction: median={res.insert_frac.median():.4f}  '
      f'IQR=[{res.insert_frac.quantile(.25):.4f},{res.insert_frac.quantile(.75):.4f}]')
print(f'real ratio: median={res.ratio.median():.4f}  IQR=[{res.ratio.quantile(.25):.4f},{res.ratio.quantile(.75):.4f}]')

res.to_csv(OUT / 'b1e_real_slim_dilution.tsv', sep='\t', index=False)
print('\n[done]')
