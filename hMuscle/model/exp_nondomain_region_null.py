#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_nondomain_region_null.py  (devils-advocate Attack 1: is changed-region pool INFORMATIVE
or does ANY local pool beat global mean?)

Control for exp_nondomain_region_pool. For each isoform, in addition to the changed-region pool,
compute a LENGTH-MATCHED SCRAMBLED-region pool: pool over the same NUMBER of residues but drawn
(i) as a random contiguous window at a random position, and (ii) as a random residue subset.
If changed-region CV-dir-acc >> scrambled CV-dir-acc, the CHANGED region carries specific signal
(not an artifact of local pooling). If scrambled ~= changed, the effect is generic locality.

Reports the CLEAN internal subset (both sides' changed region non-empty) since exp_region_pool
showed fallback pairs inflate the number.
"""
import os, re, time
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
import json, torch
import importlib.util

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
OUT = ROOT / 'reports/muscle_labelgap/nondomain_region_null.json'
spec = importlib.util.spec_from_file_location('rp', MODEL / 'exp_nondomain_region_pool.py')
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
NTERM_WIN = rp.NTERM_WIN
rng = np.random.default_rng(123)


def pool_idx(res, idx):
    idx = [i for i in idx if 0 <= i < res.shape[0]]
    return res[idx].mean(0) if idx else None


def compute(pairs, gpu):
    import esm
    seqs = {}
    ivs_map = {}
    for p in pairs:
        seqs[p['lo_id']] = p['lo_seq']; seqs[p['sh_id']] = p['sh_seq']
        ivs_map[p['lo_id']] = p['lo_ivs']; ivs_map[p['sh_id']] = p['sh_ivs']
    ids = sorted(seqs.keys(), key=lambda i: len(seqs[i]))
    dev = f'cuda:{gpu}'
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    bc = alphabet.get_batch_converter(); model.eval().to(dev)
    pools = {}
    t0 = time.time(); B = 16
    with torch.no_grad():
        for s in range(0, len(ids), B):
            chunk = ids[s:s + B]
            _, _, toks = bc([(i, seqs[i]) for i in chunk]); toks = toks.to(dev)
            out = model(toks, repr_layers=[15, 30])
            r15 = out['representations'][15].cpu().numpy(); r30 = out['representations'][30].cpu().numpy()
            for bi, i in enumerate(chunk):
                Ln = len(seqs[i]); a15 = r15[bi, 1:Ln + 1]; a30 = r30[bi, 1:Ln + 1]
                ivs = ivs_map[i]
                cidx = [k for a, b in ivs for k in range(a, b) if k < Ln]
                ksz = len(cidx)
                empty = ksz == 0
                if empty:
                    # match region_pool fallback: use nterm window; flag it
                    cidx = list(range(min(NTERM_WIN, Ln))); ksz = len(cidx)
                d = {'m15': a15.mean(0), 'm30': a30.mean(0),
                     'c15': a15[cidx].mean(0), 'c30': a30[cidx].mean(0),
                     'empty': empty}
                # length-matched scrambled: random contiguous window of size ksz
                if Ln > ksz:
                    st = int(rng.integers(0, Ln - ksz + 1)); win = list(range(st, st + ksz))
                else:
                    win = list(range(Ln))
                d['w15'] = a15[win].mean(0); d['w30'] = a30[win].mean(0)
                # random subset of size ksz
                sub = rng.choice(Ln, size=min(ksz, Ln), replace=False)
                d['s15'] = a15[sub].mean(0); d['s30'] = a30[sub].mean(0)
                pools[i] = d
            if s % (B * 30) == 0:
                print(f"  {s+len(chunk)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    return pools


def cv(D, gid, seed):
    r = np.random.default_rng(seed)
    ug = np.unique(gid); r.shuffle(ug); f = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([f[g] for g in gid]); c = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = D[tr].mean(0); a /= np.linalg.norm(a) + 1e-9; c += int((D[te] @ a > 0).sum())
    return c / len(D)


def stab(D, gid):
    if len(D) < 10:
        return float('nan'), float('nan')
    a = [cv(D, gid, s) for s in range(30)]; return float(np.mean(a)), float(np.std(a))


def main():
    pairs = rp.build_pairs()
    pools = compute(pairs, int(os.environ.get('EMB_GPU', '0')))
    nt = np.array([p['first'] < NTERM_WIN for p in pairs])
    # clean = neither side fell back
    clean = np.array([(not pools[p['lo_id']]['empty']) and (not pools[p['sh_id']]['empty']) for p in pairs])

    def delta(key, mask):
        D = [], []; DD = []; gid = []
        for p, keep in zip(pairs, mask):
            if not keep:
                continue
            lo, sh = pools[p['lo_id']], pools[p['sh_id']]
            DD.append(np.concatenate([lo[key + '15'] - sh[key + '15'], lo[key + '30'] - sh[key + '30']]))
            gid.append(p['g'])
        return np.array(DD), np.array(gid)

    res = {'n_pairs': len(pairs)}
    print("\n=== ATTACK 1 null: changed vs length-matched scrambled region pool (CV-dir-acc, 30-seed) ===")
    print(f"{'class':10}{'subset':16}{'pool':22}{'n':>5}  acc±sd")
    for cname, cm in [('N-term', nt), ('internal', ~nt)]:
        for sname, sm in [('ALL', cm), ('clean', cm & clean)]:
            row = {}
            for key, kn in [('c', 'changed_region'), ('w', 'scrambled_contig'), ('s', 'scrambled_subset')]:
                D, gid = delta(key, sm)
                mu, sd = stab(D, gid)
                row[kn] = [mu, sd]
                print(f"{cname:10}{sname:16}{kn:22}{int(sm.sum()):>5}  {mu:.3f}±{sd:.3f}")
            res[f'{cname}_{sname}'] = row
            print()
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
