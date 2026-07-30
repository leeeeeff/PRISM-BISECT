#!/usr/bin/env python
"""Option B muscle prep: mirror b_prep_subset.py for muscle (cross-tissue replication of the
pooling-kernel coherence result). Sequences from top30k_isoforms.pep via my_isoform_list_fixed order."""
import numpy as np, pandas as pd, importlib.util
from pathlib import Path
from difflib import SequenceMatcher

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
ISO = ROOT / 'hMuscle/model/my_isoform_list_fixed.npy'
PEP = ROOT / 'hMuscle/data/top30k_isoforms.pep'
CAP_SLIM, CAP_DOM = 800, 400

# use the SAME parser build_severity_pairs.py used (parse_pep_sequences), not compute_esm2's
spec = importlib.util.spec_from_file_location('bsp', ROOT/'hMuscle/model/build_severity_pairs.py')
bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
recs = bsp.parse_pep_sequences(PEP)                        # id -> seq (str)
iso = np.load(ISO, allow_pickle=True).astype(str)
idx2seq = {i: recs[iso[i]][:1022] for i in range(len(iso)) if iso[i] in recs}
print(f'iso list={len(iso)}  pep records={len(recs)}  idx->seq coverage={len(idx2seq)}')

df = pd.read_csv(ROOT/'reports/severity_pairs/muscle_severity_pairs_scored.tsv', sep='\t')
slim = df[(df.domain_binary==0)&(df['size']>=3)&(df['size']<=40)]
dom  = df[(df.domain_binary==1)&(df['size']>=80)]
print(f'pools: slim={len(slim)} dom={len(dom)}')
slim = slim.sample(min(CAP_SLIM,len(slim)), random_state=0)
dom  = dom.sample(min(CAP_DOM,len(dom)), random_state=0)
sub = pd.concat([slim.assign(cls='slim'), dom.assign(cls='domain')], ignore_index=True)

rows, ncov = [], 0
for _,r in sub.iterrows():
    ls, ss = idx2seq.get(int(r.long_idx)), idx2seq.get(int(r.short_idx))
    if ls is None or ss is None: continue
    ncov += 1
    ivs=[(i1,i2) for tag,i1,i2,j1,j2 in SequenceMatcher(None,ls,ss,autojunk=False).get_opcodes() if tag!='equal']
    rows.append(dict(cls=r.cls, gene=r.gene, long_idx=int(r.long_idx), short_idx=int(r.short_idx),
                     canonical_idx=int(r.canonical_idx), other_idx=int(r.other_idx), size=int(r['size']),
                     len_long=len(ls), n_intervals=len(ivs)))
man = pd.DataFrame(rows)
print(f'coverage {ncov}/{len(sub)}')
print(man.groupby('cls').size())
uniq = [int(i) for i in pd.unique(pd.concat([man.long_idx, man.short_idx])) if int(i) in idx2seq]
with open(OUT/'b_extract_sequences_muscle.fasta','w') as f:
    for i in uniq: f.write(f'>{i}\n{idx2seq[i]}\n')
man.to_csv(OUT/'b_manifest_pairs_muscle.tsv', sep='\t', index=False)
print(f'unique isoforms={len(uniq)} -> b_extract_sequences_muscle.fasta')
