#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_domdiff_null.py  (S0 gate for the H_break surprise: is domain-differing region-pool 0.844
REAL localization or a tautology/fallback artifact?)

domain-differing changed_region pool hit 0.844 (vs mean 0.708). But changed region = the gained/
lost DOMAIN, so "pool over the domain -> see the domain" may be near-tautological, and 27-30%
fallback (domain-less side empty -> nterm window) may inflate. Apply the SAME controls used for the
non-domain scrambled null:
  - scrambled_contig / scrambled_subset: length-matched RANDOM region. If they also ~0.84 => generic
    locality artifact. If << 0.844 => changed region carries specific (domain) signal.
  - clean (both sides non-empty) vs fallback split.
Orientation = toward domain-complete (curated label).
"""
import os, time
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from pathlib import Path
import json, torch, importlib.util

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); MODEL = ROOT / 'hMuscle/model'
OUT = ROOT / 'reports/muscle_labelgap/domdiff_null.json'
spec = importlib.util.spec_from_file_location('dd', MODEL / 'exp_domdiff_region_pool.py')
dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
NTERM_WIN = dd.NTERM_WIN
rng = np.random.default_rng(303)


def compute(pairs, gpu):
    import esm
    seqs = {}; ivs_map = {}
    for p in pairs:
        seqs[p['hi_id']] = p['hi_seq']; seqs[p['lo_id']] = p['lo_seq']
        ivs_map[p['hi_id']] = p['hi_ivs']; ivs_map[p['lo_id']] = p['lo_ivs']
    ids = sorted(seqs.keys(), key=lambda i: len(seqs[i]))
    dev = f'cuda:{gpu}'
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    bc = alphabet.get_batch_converter(); model.eval().to(dev)
    pools = {}; B = 16
    with torch.no_grad():
        for s in range(0, len(ids), B):
            chunk = ids[s:s + B]
            _, _, toks = bc([(i, seqs[i]) for i in chunk]); toks = toks.to(dev)
            out = model(toks, repr_layers=[15, 30])
            r15 = out['representations'][15].cpu().numpy(); r30 = out['representations'][30].cpu().numpy()
            for bi, i in enumerate(chunk):
                Ln = len(seqs[i]); a15 = r15[bi, 1:Ln + 1]; a30 = r30[bi, 1:Ln + 1]
                cidx = [k for a, b in ivs_map[i] for k in range(a, b) if k < Ln]
                empty = len(cidx) == 0
                if empty: cidx = list(range(min(NTERM_WIN, Ln)))
                ksz = len(cidx)
                d = {'m15': a15.mean(0), 'm30': a30.mean(0),
                     'c15': a15[cidx].mean(0), 'c30': a30[cidx].mean(0), 'empty': empty}
                if Ln > ksz:
                    st = int(rng.integers(0, Ln - ksz + 1)); win = list(range(st, st + ksz))
                else:
                    win = list(range(Ln))
                d['w15'] = a15[win].mean(0); d['w30'] = a30[win].mean(0)
                sub = rng.choice(Ln, size=min(ksz, Ln), replace=False)
                d['s15'] = a15[sub].mean(0); d['s30'] = a30[sub].mean(0)
                pools[i] = d
    return pools


def cv(D, gid, seed):
    r = np.random.default_rng(seed)
    ug = np.unique(gid); r.shuffle(ug); f = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([f[g] for g in gid]); c = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0: continue
        a = D[tr].mean(0); a /= np.linalg.norm(a) + 1e-9; c += int((D[te] @ a > 0).sum())
    return c / len(D)


def stab(D, gid):
    if len(D) < 10: return float('nan'), float('nan')
    a = [cv(D, gid, s) for s in range(30)]; return float(np.mean(a)), float(np.std(a))


def main():
    pairs = dd.build_domdiff_pairs()
    print(f"domain-differing pairs: {len(pairs)}", flush=True)
    pools = compute(pairs, int(os.environ.get('EMB_GPU', '0')))
    gid = np.array([p['g'] for p in pairs])
    clean = np.array([(not pools[p['hi_id']]['empty']) and (not pools[p['lo_id']]['empty']) for p in pairs])
    print(f"clean (both non-empty): {clean.sum()}  fallback: {(~clean).sum()}", flush=True)

    def delta(key, mask):
        return np.array([np.concatenate([pools[p['hi_id']][key + '15'] - pools[p['lo_id']][key + '15'],
                                         pools[p['hi_id']][key + '30'] - pools[p['lo_id']][key + '30']])
                         for p, k in zip(pairs, mask) if k])

    res = {'n_pairs': len(pairs), 'n_clean': int(clean.sum())}
    print(f"\n{'subset':16}{'pool':22}{'n':>5}  acc±sd")
    allmask = np.ones(len(pairs), bool)
    for sname, sm in [('ALL', allmask), ('clean', clean)]:
        row = {}
        for key, kn in [('m', 'meanpool'), ('c', 'changed_region'),
                        ('w', 'scrambled_contig'), ('s', 'scrambled_subset')]:
            D = delta(key, sm); g = gid[sm]
            mu, sd = stab(D, g); row[kn] = [mu, sd]
            print(f"{sname:16}{kn:22}{int(sm.sum()):>5}  {mu:.3f}±{sd:.3f}")
        res[sname] = row
        print()
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
