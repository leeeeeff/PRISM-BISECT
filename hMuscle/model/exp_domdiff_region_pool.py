#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_domdiff_region_pool.py  (Option B: SEAL Attack 4 via double dissociation)

Attack 4: region-pool non-domain coherence 0.739 > DR ceiling 0.63 -> "then it's not a ceiling".
Rebuttal to test empirically: region-pool recovers label-FREE direction COHERENCE, not
label-ALIGNED accuracy. Discriminator = DOUBLE DISSOCIATION between two populations under the
SAME pooling manipulation:
  - domain-DIFFERING pairs: orientation = domain-count label (the DR curated axis). mean-pool
    anchor already 0.708 (positive control in anchor_decomp). Does region-pool RAISE it?
  - non-domain pairs: label-free self-orientation. region-pool raised internal 0.481->0.739.
PRE-REGISTERED (S2):
  H_seal: region-pool helps the label-aligned domain axis MUCH LESS than the label-free axis
    (domain-differing region-pool - mean-pool << non-domain internal gain). => region-pool
    recovers coherence, not curated-label alignment; the DR ceiling framing holds.
  H_break: region-pool raises domain-differing direction acc by >0.02 (gene-cluster bootstrap CI
    excludes 0). => pooling lifts the curated (domain-completeness) ranking; ceiling is liftable.
gene-cluster bootstrap CI on the mean-pool vs region-pool DELTA (paired by gene fold).
"""
import os, re, time
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import json, torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE'); DATA = ROOT / 'hMuscle/data'; MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/domdiff_region_pool.json'
NTERM_WIN = 60; MAXLEN = 1022
rng = np.random.default_rng(202)


def clean(g): return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', "")
def strip_orf(n): return re.sub(r'\.p\d+$', '', n)


def parse_faa():
    best, cur, buf = {}, None, []
    def flush():
        if cur is None: return
        s = ''.join(buf); b = strip_orf(cur)
        if b not in best or len(s) > len(best[b]): best[b] = s
    for line in open(FAA):
        if line.startswith('>'): flush(); cur = line[1:].split()[0]; buf = []
        else: buf.append(line.strip())
    flush(); return best


def opcode_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    livs, sivs = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        if i2 > i1: livs.append((i1, i2))
        if j2 > j1: sivs.append((j1, j2))
    return livs, sivs


def build_domdiff_pairs():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    faa = parse_faa()
    gl, gi = np.unique(gen, return_inverse=True); cnt = np.bincount(gi, minlength=len(gl))
    pairs = []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if dom[a] == dom[b]: continue                     # want domain-DIFFERING
        if iso[a] not in faa or iso[b] not in faa: continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb: continue
        # orient toward domain-complete (higher domain count)
        hi, lo = (a, b) if dom[a] > dom[b] else (b, a)
        shi, slo = faa[iso[hi]][:MAXLEN], faa[iso[lo]][:MAXLEN]
        hivs, livs = opcode_intervals(shi, slo)
        pairs.append({'g': int(g), 'hi_id': iso[hi], 'lo_id': iso[lo],
                      'hi_seq': shi, 'lo_seq': slo, 'hi_ivs': hivs, 'lo_ivs': livs})
    return pairs


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
    pools = {}; t0 = time.time(); B = 16
    with torch.no_grad():
        for s in range(0, len(ids), B):
            chunk = ids[s:s + B]
            _, _, toks = bc([(i, seqs[i]) for i in chunk]); toks = toks.to(dev)
            out = model(toks, repr_layers=[15, 30])
            r15 = out['representations'][15].cpu().numpy(); r30 = out['representations'][30].cpu().numpy()
            for bi, i in enumerate(chunk):
                Ln = len(seqs[i]); a15 = r15[bi, 1:Ln + 1]; a30 = r30[bi, 1:Ln + 1]
                cidx = [k for a, b in ivs_map[i] for k in range(a, b) if k < Ln]
                if not cidx: cidx = list(range(min(NTERM_WIN, Ln)))
                pools[i] = {'m15': a15.mean(0), 'm30': a30.mean(0),
                            'n15': a15[:NTERM_WIN].mean(0), 'n30': a30[:NTERM_WIN].mean(0),
                            'c15': a15[cidx].mean(0), 'c30': a30[cidx].mean(0)}
            if s % (B * 30) == 0: print(f"  {s+len(chunk)}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    return pools


def cv_perfold_correct(D, gid, seed):
    """returns per-gene-fold correct fraction so mean/region can be paired for bootstrap."""
    r = np.random.default_rng(seed)
    ug = np.unique(gid); r.shuffle(ug); f = {g: i % 5 for i, g in enumerate(ug)}
    fid = np.array([f[g] for g in gid]); preds = np.zeros(len(D), bool)
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0: continue
        a = D[tr].mean(0); a /= np.linalg.norm(a) + 1e-9; preds[te] = (D[te] @ a) > 0
    return preds, fid


def acc_stab(D, gid):
    accs = []
    for s in range(30):
        preds, _ = cv_perfold_correct(D, gid, s); accs.append(preds.mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    pairs = build_domdiff_pairs()
    print(f"domain-differing pairs: {len(pairs)}", flush=True)
    pools = compute(pairs, int(os.environ.get('EMB_GPU', '0')))
    gid = np.array([p['g'] for p in pairs])

    def delta(key):
        return np.array([np.concatenate([pools[p['hi_id']][key + '15'] - pools[p['lo_id']][key + '15'],
                                         pools[p['hi_id']][key + '30'] - pools[p['lo_id']][key + '30']])
                         for p in pairs])
    res = {'n_pairs': len(pairs), 'orientation': 'toward domain-complete (curated structural label)',
           'prediction': 'H_seal: region-pool raises domain-aligned axis << non-domain label-free axis'}
    print("\n=== domain-differing (label=domain count) direction accuracy, 30-seed ===")
    for key, name in [('m', 'meanpool'), ('n', 'nterm_window'), ('c', 'changed_region')]:
        D = delta(key); mu, sd = acc_stab(D, gid)
        res[name] = {'acc': mu, 'sd': sd}
        print(f"  {name:16} acc={mu:.3f} ± {sd:.3f}", flush=True)

    # gene-cluster paired bootstrap on region - mean delta (per-gene aggregated)
    print("\n=== gene-cluster paired bootstrap: changed_region - meanpool (domain-differing) ===")
    Dm, Dc = delta('m'), delta('c')
    ug = np.unique(gid)
    # per-gene correctness under a fixed fold assignment (seed 0)
    pm, _ = cv_perfold_correct(Dm, gid, 0); pc, _ = cv_perfold_correct(Dc, gid, 0)
    gene_dm = np.array([pm[gid == g].mean() for g in ug])
    gene_dc = np.array([pc[gid == g].mean() for g in ug])
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, len(ug), len(ug))
        diffs.append((gene_dc[idx] - gene_dm[idx]).mean())
    diffs = np.array(diffs)
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    res['bootstrap_region_minus_mean'] = {'delta': float(gene_dc.mean() - gene_dm.mean()), 'ci95': ci}
    print(f"  Δ(region-mean) = {gene_dc.mean()-gene_dm.mean():+.3f}  95%CI [{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)
    print(f"  -> {'H_break (CI excludes 0, region helps domain axis)' if ci[0]>0 else 'H_seal (CI includes 0 or negative)'}")

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == '__main__':
    main()
