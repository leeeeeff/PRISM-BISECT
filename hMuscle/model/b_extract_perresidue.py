#!/usr/bin/env python
"""
Option B extractor (GPU): re-run ESM-2 t30_150M forward on the B subset, KEEPING per-residue
representations at layers 9, 15, 30 (the A-identified depth landmarks). Saves one .npz per
isoform: keys 'L9','L15','L30' each (Li, 640) float16. Truncation to 1022 (matches the cached
pipeline). Read-only inputs; writes only under reports/model_interpretability_map/b_perres/.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4'); os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np, torch
from pathlib import Path

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAX_LEN = 1022
LAYERS = [9, 15, 30]
ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
OUT = ROOT / 'reports/model_interpretability_map'
FASTA = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / 'b_extract_sequences.fasta'
PERRES = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT / 'b_perres'; PERRES.mkdir(exist_ok=True)

device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)

# read fasta (idx -> seq)
items = []
with open(FASTA) as f:
    cur = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'):
            cur = line[1:]
        elif line:
            items.append((cur, line[:MAX_LEN]))
print(f'[{time.strftime("%H:%M:%S")}] {len(items)} sequences, device={device}', flush=True)

import esm
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.eval().to(device)
bc = alphabet.get_batch_converter()
print(f'  loaded esm2_t30_150M num_layers={model.num_layers} embed_dim={model.embed_dim}', flush=True)

# length-sorted batching to minimise padding
items.sort(key=lambda x: len(x[1]))
done = {p.stem for p in PERRES.glob('*.npz')}
todo = [(i, s) for i, s in items if i not in done]
print(f'  already done={len(done)}  todo={len(todo)}', flush=True)

def batches(lst):
    b, tok = [], 0
    for it in lst:
        L = len(it[1]) + 2
        if b and (tok + L > 12000 or len(b) >= 32):   # token budget
            yield b; b, tok = [], 0
        b.append(it); tok += L
    if b: yield b

t0 = time.time(); n = 0
with torch.no_grad():
    for batch in batches(todo):
        labels = [x[0] for x in batch]
        _, _, toks = bc([(lab, seq) for lab, seq in batch])
        toks = toks.to(device)
        out = model(toks, repr_layers=LAYERS, return_contacts=False)['representations']
        for i, (lab, seq) in enumerate(batch):
            L = len(seq)
            d = {f'L{l}': out[l][i, 1:L+1, :].cpu().numpy().astype(np.float16) for l in LAYERS}
            np.savez(PERRES / f'{lab}.npz', **d)
        n += len(batch)
        if n % 256 < len(batch):
            el = time.time() - t0
            print(f'  {n}/{len(todo)}  {el:.0f}s  ({n/max(el,1):.1f} seq/s)', flush=True)
print(f'[done] extracted {n} isoforms in {time.time()-t0:.0f}s -> {PERRES}', flush=True)
