#!/usr/bin/env python
"""
Option B prep (no GPU): select the per-residue re-extraction subset and de-risk the
transcript->ORF_seq join + SequenceMatcher alignment. Writes a manifest for the extractor.

Subset (brain, canonical-anchored pairs):
  - SLiM-candidate : domain_binary==0 (non-domain) AND 3 <= size <= 40   (localized edits)
  - domain control : domain_binary==1 AND size >= 80                     (large contiguous)
Cap each class; collect unique isoforms (long+short) that have an ORF_seq.
"""
import numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
CSV = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/SQANTI3_output/'
           'isoforms_classification_with_tx_name_and_gene_name.csv')
OUT = ROOT / 'reports/model_interpretability_map'
BR = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
CAP_SLIM, CAP_DOM = 800, 400
SEED = 0

ids = np.load(BR / 'brain_full_ids.npy', allow_pickle=True).astype(str)
df = pd.read_csv(ROOT / 'reports/severity_pairs/brain_severity_pairs_scored.tsv', sep='\t')

slim = df[(df.domain_binary == 0) & (df['size'] >= 3) & (df['size'] <= 40)].copy()
dom  = df[(df.domain_binary == 1) & (df['size'] >= 80)].copy()
rng = np.random.default_rng(SEED)
slim = slim.sample(min(CAP_SLIM, len(slim)), random_state=SEED)
dom  = dom.sample(min(CAP_DOM, len(dom)), random_state=SEED)
print(f'candidate pools: SLiM(nd,3-40aa)={((df.domain_binary==0)&(df["size"]>=3)&(df["size"]<=40)).sum()} '
      f'dom(>=80aa)={((df.domain_binary==1)&(df["size"]>=80)).sum()}')
print(f'sampled: SLiM={len(slim)} dom={len(dom)}')

sub = pd.concat([slim.assign(cls='slim'), dom.assign(cls='domain')], ignore_index=True)

# join transcript_name -> ORF_seq
seqmap = {}
for chunk in pd.read_csv(CSV, usecols=['transcript_name', 'ORF_seq'], chunksize=200000,
                         dtype=str, low_memory=False):
    for tn, s in zip(chunk['transcript_name'], chunk['ORF_seq']):
        if isinstance(s, str) and s and s != 'nan':
            seqmap[tn] = s.rstrip('*')
print(f'ORF_seq map entries: {len(seqmap)}')

def seq_of(idx):
    return seqmap.get(ids[int(idx)])

# coverage + alignment sanity
rows, ncov, nalign = [], 0, 0
for _, r in sub.iterrows():
    ls, ss = seq_of(r.long_idx), seq_of(r.short_idx)
    if ls is None or ss is None:
        continue
    ncov += 1
    sm = SequenceMatcher(None, ls, ss, autojunk=False)
    ivs = [(i1, i2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal']
    edit_span = sum(i2 - i1 for i1, i2 in ivs)
    if ivs:
        nalign += 1
    rows.append(dict(cls=r.cls, gene=r.gene, long_idx=int(r.long_idx), short_idx=int(r.short_idx),
                     canonical_idx=int(r.canonical_idx), other_idx=int(r.other_idx),
                     size=int(r['size']), len_long=len(ls), len_short=len(ss),
                     edit_span_long=edit_span, n_intervals=len(ivs),
                     first_edit_start=(ivs[0][0] if ivs else -1)))
man = pd.DataFrame(rows)
print(f'coverage: {ncov}/{len(sub)} pairs have both seqs; {nalign} have >=1 edit interval')
print(man.groupby('cls').agg(n=('gene','size'), med_len_long=('len_long','median'),
                             med_edit=('edit_span_long','median')).round(1))

# unique isoforms to extract
uniq = pd.unique(pd.concat([man.long_idx, man.short_idx]))
uniq = [int(i) for i in uniq if seq_of(i) is not None]
np.save(OUT / 'b_extract_isoform_idx.npy', np.array(uniq, dtype=np.int64))
with open(OUT / 'b_extract_sequences.fasta', 'w') as f:
    for i in uniq:
        f.write(f'>{i}\n{seq_of(i)}\n')
man.to_csv(OUT / 'b_manifest_pairs.tsv', sep='\t', index=False)
print(f'unique isoforms to extract: {len(uniq)}  -> b_extract_sequences.fasta / b_extract_isoform_idx.npy')
print(f'seq length stats: min={min(len(seq_of(i)) for i in uniq)} '
      f'max={max(len(seq_of(i)) for i in uniq)} '
      f'median={int(np.median([len(seq_of(i)) for i in uniq]))}')
print('[done] manifest written')
