#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_nondomain_region_pool.py  (Option B: does REGION-POOL rescue diluted encoded signal?)

exp_nondomain_anchor_decomp.py used MEAN-POOL (whole-sequence) L15+L30 deltas and found:
  N-terminal edits  CV-dir-acc 0.805  (strong anchor survives mean-pool)
  internal edits    CV-dir-acc 0.488  (CHANCE -- no anchor at mean-pool)
The pooling-not-encoding result ([[finding-pooling-not-encoding-boundary]]) says splice signal is
PRESENT per-residue but mean-pool DILUTES it. So the internal-edit chance result may be dilution,
not absence. This experiment recomputes per-residue ESM-2 (L15,L30) for the SAME 1505 pairs and
REGION-POOLS over the changed interval (and an N-terminal window), then re-measures coherence and
gene-disjoint CV-dir-acc.

PRE-REGISTERED (S2):
  H1 (dilution only): region-pool >> mean-pool for INTERNAL edits (internal rises above chance),
     confirming internal signal is encoded-but-diluted (recoverable headroom).
  H2 (genuine absence): region-pool ~= mean-pool for internal (stays at chance) => the internal
     non-domain class truly lacks a common anchor (manuscript's §6 claim holds for internal).
  N-terminal is expected to stay high or sharpen (>=0.805) either way (positive control on pooling).

Independence of machinery from anchor_decomp: identical pair set + identical CV folding; only the
POOLING changes. Per-residue embeddings recomputed from muscle_2iso.fa (same sequences).
"""
import os, re, time
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import json
import torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'
MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/nondomain_region_pool.json'
CACHE = ROOT / 'reports/muscle_labelgap/region_pool_cache.npz'
NTERM_WIN = 60
MAXLEN = 1022
rng = np.random.default_rng(42)


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', "")


def strip_orf(n):
    return re.sub(r'\.p\d+$', '', n)


def parse_faa():
    best, cur, buf = {}, None, []

    def flush():
        if cur is None:
            return
        s = ''.join(buf); b = strip_orf(cur)
        if b not in best or len(s) > len(best[b]):
            best[b] = s
    for line in open(FAA):
        if line.startswith('>'):
            flush(); cur = line[1:].split()[0]; buf = []
        else:
            buf.append(line.strip())
    flush()
    return best


def opcode_intervals(long_s, short_s):
    """returns (long_ivs, short_ivs, first_change_on_long, total_changed)."""
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    livs, sivs, changed, first = [], [], 0, None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            livs.append((i1, i2))
            if first is None:
                first = i1
        if j2 > j1:
            sivs.append((j1, j2))
    return livs, sivs, (first if first is not None else 0), changed


def build_pairs():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    faa = parse_faa()
    gl, gi = np.unique(gen, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))
    pairs = []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if dom[a] != dom[b]:
            continue
        if iso[a] not in faa or iso[b] not in faa:
            continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb:
            continue
        lo, sh = (a, b) if len(sa) >= len(sb) else (b, a)
        ls, ss = faa[iso[lo]], faa[iso[sh]]
        livs, sivs, first, changed = opcode_intervals(ls, ss)
        if changed == 0 or not livs:
            continue
        pairs.append({'g': int(g),
                      'lo_id': iso[lo], 'sh_id': iso[sh],
                      'lo_seq': ls[:MAXLEN], 'sh_seq': ss[:MAXLEN],
                      'lo_ivs': livs, 'sh_ivs': sivs,
                      'first': int(first), 'size': int(changed)})
    return pairs


def pool_ranges(res, ivs):
    """mean over residues in the given intervals; res is [L,640]; ivs list of (a,b) 0-indexed."""
    idx = []
    for a, b in ivs:
        idx.extend(range(min(a, res.shape[0]), min(b, res.shape[0])))
    idx = [i for i in idx if 0 <= i < res.shape[0]]
    if not idx:
        return None
    return res[idx].mean(0)


def compute_embeddings(pairs, gpu=1):
    import esm
    seqs = {}
    for p in pairs:
        seqs[p['lo_id']] = p['lo_seq']
        seqs[p['sh_id']] = p['sh_seq']
    ids = list(seqs.keys())
    print(f"[emb] unique isoforms to embed: {len(ids)}", flush=True)
    dev = f'cuda:{gpu}'
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval().to(dev)
    # per-isoform stored pools: mean, nterm(first 60), changed-region
    pools = {}  # id -> dict with m15,m30,n15,n30 (,c15,c30 if changed defined for that iso in its pair)
    # build changed intervals per isoform id from pairs
    iso_ivs = {}
    for p in pairs:
        iso_ivs[p['lo_id']] = p['lo_ivs']
        iso_ivs[p['sh_id']] = p['sh_ivs']
    order = sorted(ids, key=lambda i: len(seqs[i]))
    B = 16
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), B):
            chunk = order[s:s + B]
            data = [(i, seqs[i]) for i in chunk]
            _, _, toks = bc(data)
            toks = toks.to(dev)
            out = model(toks, repr_layers=[15, 30])
            r15 = out['representations'][15].cpu().numpy()
            r30 = out['representations'][30].cpu().numpy()
            for bi, i in enumerate(chunk):
                L = len(seqs[i])
                a15 = r15[bi, 1:L + 1]   # strip BOS
                a30 = r30[bi, 1:L + 1]
                d = {'m15': a15.mean(0), 'm30': a30.mean(0),
                     'n15': a15[:NTERM_WIN].mean(0), 'n30': a30[:NTERM_WIN].mean(0)}
                c15 = pool_ranges(a15, iso_ivs[i]); c30 = pool_ranges(a30, iso_ivs[i])
                d['c15'] = c15 if c15 is not None else d['n15']
                d['c30'] = c30 if c30 is not None else d['n30']
                pools[i] = d
            if s % (B * 20) == 0:
                print(f"  {s+len(chunk)}/{len(order)}  {time.time()-t0:.0f}s", flush=True)
    return pools


def coherence(D):
    U = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    R = float(np.linalg.norm(U.mean(0)))
    nulls = []
    for _ in range(1000):
        s = rng.choice([-1.0, 1.0], size=len(U))[:, None]
        nulls.append(np.linalg.norm((U * s).mean(0)))
    nulls = np.array(nulls)
    return R, float(nulls.mean()), float(np.percentile(nulls, 97.5)), float((nulls >= R).mean())


def cv_dir_acc(D, gene_id):
    n = len(D)
    ug = np.unique(gene_id); rng.shuffle(ug)
    folds = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([folds[g] for g in gene_id])
    correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = D[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        correct += int((D[te] @ a > 0).sum())
    acc = correct / n
    se = np.sqrt(acc * (1 - acc) / n)
    return acc, acc - 1.96 * se, acc + 1.96 * se


def make_delta(pairs, pools, key15, key30):
    D, gid, nterm, size = [], [], [], []
    for p in pairs:
        lo, sh = pools[p['lo_id']], pools[p['sh_id']]
        d = np.concatenate([lo[key15] - sh[key15], lo[key30] - sh[key30]])
        D.append(d); gid.append(p['g'])
        nterm.append(p['first'] < NTERM_WIN); size.append(p['size'])
    return np.array(D), np.array(gid), np.array(nterm), np.array(size)


def evaluate(name, pairs, pools, key15, key30, res):
    D, gid, nterm, size = make_delta(pairs, pools, key15, key30)
    block = {}
    print(f"\n=== {name} ===", flush=True)
    for lab, m in [('N-terminal', nterm), ('internal', ~nterm)]:
        R, nmean, nhi, pR = coherence(D[m])
        acc, lo, hi = cv_dir_acc(D[m], gid[m])
        block[lab] = {'n': int(m.sum()), 'R': R, 'R_null_p975': nhi, 'p_R': pR,
                      'cv_dir_acc': acc, 'cv_ci': [lo, hi]}
        print(f"  {lab:11} n={m.sum():4} R={R:.3f}(null {nmean:.3f}) CV-dir-acc={acc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    res[name] = block


def main():
    pairs = build_pairs()
    print(f"pairs={len(pairs)}", flush=True)
    if CACHE.exists():
        print("[cache] loading pooled embeddings", flush=True)
        z = np.load(CACHE, allow_pickle=True)
        pools = z['pools'].item()
    else:
        pools = compute_embeddings(pairs, gpu=int(os.environ.get('EMB_GPU', '1')))
        np.savez(CACHE, pools=np.array(pools, dtype=object))
        print(f"[cache] saved {CACHE}", flush=True)

    res = {'n_pairs': len(pairs), 'NTERM_WIN': NTERM_WIN,
           'prediction': 'H1: region-pool rescues internal above chance (encoded-but-diluted)'}
    evaluate('meanpool_baseline', pairs, pools, 'm15', 'm30', res)   # sanity vs 0.805/0.488
    evaluate('nterm_window_pool', pairs, pools, 'n15', 'n30', res)
    evaluate('changed_region_pool', pairs, pools, 'c15', 'c30', res)

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
