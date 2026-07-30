#!/usr/bin/env python
"""
Option B analysis — adjudicate devils-advocate Attack 2 (mean-pool = low-pass, O(k/L) dilution)
and the clean side of Attack 1 (is the localized edit encoded per-residue BEFORE pooling).

Per pair, align long/short (SequenceMatcher). At 'equal'-aligned positions the residue is identical,
so dh_p = h_long[p] - h_short[aligned] is a PURE CONTEXT effect. Two decisive readouts, per class
(slim = non-domain 3-40aa localized edit; domain = >=80aa contiguous) and per layer (L9/L15/L30):

  T1  Contextual-spread curve: mean ||dh_p|| vs distance of p from the nearest edit interval,
      normalized per pair to the edit-core magnitude. Local decay (-> 0 within ~10 aa) supports the
      O(k/L) 'averaging kills SLiMs' mechanism; a persistent far-field supports the attention-spread
      counter (O(1), death not by averaging).
  T2  Pooling survival (DC coherence): frac_kept = ||mean_p dh_p||^2 / mean_p ||dh_p||^2 over equal
      positions. High = pooling preserves the shift (coherent); low = pooling discards it (incoherent).
      Also the edit-core survival relative to full-length pooling: edit contributes ~ (edit_len/L).
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
TAG = sys.argv[1] if len(sys.argv) > 1 else ''          # '' = brain, '_muscle' = muscle
PERRES = OUT / f'b_perres{TAG}'
FASTA_P = OUT / f'b_extract_sequences{TAG}.fasta'
MAN_P = OUT / f'b_manifest_pairs{TAG}.tsv'
RES_P = OUT / f'b_pooling_kernel_results{TAG}.tsv'
LAYERS = [9, 15, 30]
BINS = [(0,0),(1,2),(3,5),(6,10),(11,20),(21,50),(51,100),(101,10**6)]
BINLAB = ['edit','1-2','3-5','6-10','11-20','21-50','51-100','>100']

# sequences (idx -> truncated seq, matches extracted reps)
seq = {}
with open(FASTA_P) as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'): cur = line[1:]
        elif line: seq[cur] = line
man = pd.read_csv(MAN_P, sep='\t')
print(f'pairs in manifest: {len(man)}  ({(man.cls=="slim").sum()} slim / {(man.cls=="domain").sum()} domain)')

def load(idx):
    p = PERRES / f'{idx}.npz'
    if not p.exists(): return None
    z = np.load(p); return {l: z[f'L{l}'].astype(np.float32) for l in LAYERS}

# accumulators: per class/layer/bin -> list of normalized ||dh||; and coherence
spread = defaultdict(list)     # (cls,layer,binlab) -> normalized ||dh||
kept = defaultdict(list)       # (cls,layer) -> frac_kept over equal positions
editfrac = defaultdict(list)   # (cls) -> edit_len / L_long  (naive pooling weight of the edit)
nfail = 0

for _, r in man.iterrows():
    li, si, cls = str(r.long_idx), str(r.short_idx), r.cls
    ls, ss = seq.get(li), seq.get(si)
    HL, HS = load(li), load(si)
    if ls is None or ss is None or HL is None or HS is None: nfail += 1; continue
    sm = SequenceMatcher(None, ls, ss, autojunk=False)
    ops = sm.get_opcodes()
    edits = [(i1,i2) for tag,i1,i2,j1,j2 in ops if tag != 'equal']
    if not edits: continue
    edit_starts = np.array([i1 for i1,i2 in edits]); edit_ends = np.array([i2 for i1,i2 in edits])
    editfrac[cls].append(sum(i2-i1 for i1,i2 in edits) / max(len(ls),1))
    for l in LAYERS:
        Ll = HL[l].shape[0]; Ls = HS[l].shape[0]
        # equal-aligned per-residue context differences
        dhs, dists = [], []
        for tag,i1,i2,j1,j2 in ops:
            if tag != 'equal': continue
            for off in range(i2-i1):
                p, q = i1+off, j1+off
                if p >= Ll or q >= Ls: continue
                d = HL[l][p] - HS[l][q]
                # distance to nearest edit interval on long
                dist = int(np.min(np.minimum(np.abs(p-edit_starts), np.abs(p-edit_ends+1))))
                if p < edit_starts.min(): pass
                dhs.append(np.linalg.norm(d)); dists.append(dist)
        # edit-core magnitude: norm of long reps within edit intervals (no short counterpart)
        core = []
        for i1,i2 in edits:
            for p in range(i1, min(i2, Ll)):
                core.append(np.linalg.norm(HL[l][p]))
        core_mag = np.median(core) if core else np.nan
        if not dhs or not np.isfinite(core_mag) or core_mag == 0: continue
        dhs = np.array(dhs); dists = np.array(dists)
        for (lo,hi),lab in zip(BINS,BINLAB):
            if lab == 'edit': continue
            m = (dists>=lo)&(dists<=hi)
            if m.any(): spread[(cls,l,lab)].append(float(dhs[m].mean())/core_mag)
        # DC coherence (survival) over equal positions
        D = np.stack([HL[l][i1+off]-HS[l][j1+off]
                      for tag,i1,i2,j1,j2 in ops if tag=='equal'
                      for off in range(i2-i1) if i1+off<Ll and j1+off<Ls])
        num = np.sum(np.mean(D,axis=0)**2); den = np.mean(np.sum(D**2,axis=1))
        if den>0: kept[(cls,l)].append(num/den)

print(f'\nfailed/skipped pairs: {nfail}')
print(f'\n=== T2 pooling survival (DC coherence, frac_kept over equal positions) ===')
print(f'{"cls":8s}{"layer":>6s}{"frac_kept":>12s}{"n":>7s}')
rows=[]
for cls in ['slim','domain']:
    for l in LAYERS:
        v=np.array(kept[(cls,l)]);
        if len(v): print(f'{cls:8s}{l:>6d}{np.median(v):>12.4f}{len(v):>7d}'); rows.append(dict(test='survival',cls=cls,layer=l,val=float(np.median(v)),n=len(v)))
print(f'\nedit pooling weight (edit_len/L_long): slim med={np.median(editfrac["slim"]):.4f}  domain med={np.median(editfrac["domain"]):.4f}')

print(f'\n=== T1 contextual-spread: ||dh|| at equal positions / edit-core magnitude, by distance ===')
print(f'{"cls":7s}{"L":>4s}  ' + '  '.join(f'{b:>7s}' for b in BINLAB[1:]))
for cls in ['slim','domain']:
    for l in LAYERS:
        vals=[np.median(spread[(cls,l,lab)]) if spread[(cls,l,lab)] else np.nan for lab in BINLAB[1:]]
        print(f'{cls:7s}{l:>4d}  ' + '  '.join(f'{v:>7.3f}' if np.isfinite(v) else f'{"--":>7s}' for v in vals))
        for lab,v in zip(BINLAB[1:],vals):
            if np.isfinite(v): rows.append(dict(test='spread',cls=cls,layer=l,bin=lab,val=float(v)))
pd.DataFrame(rows).to_csv(RES_P,sep='\t',index=False)
print('\n[done] wrote b_pooling_kernel_results.tsv')
