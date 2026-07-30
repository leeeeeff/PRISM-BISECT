#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_saturation_confound_check.py  (Option B / question ③)
Is the saturation dichotomy (domain-completeness SATURATES vs non-domain disorder does NOT) driven
by SPLICE SIZE (large vs small) or by SIGNAL TYPE (domain context vs non-domain)? Disentangle with a
TARGET-MATCHED, SIZE-STRATIFIED test: run the SAME disorder-decoding saturation probe on BOTH
domain-differing (dom count differs) and non-domain (dom count equal) 2-iso pairs, in matched size
bins, and compare the incremental value of adding φ_L30 to φ_L15.

If size confound: L30-increment tracks size regardless of domain context (both types behave alike in
a shared size window). If signal-type: at matched size, non-domain still benefits from L30 while
domain-context does not (or differs)."""
import os, re
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'; MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/saturation_confound_check.json'
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318, 'E': 0.736,
          'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586, 'M': -0.397,
          'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121}
rng = np.random.default_rng(42)


def clean(g): return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', '')
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


def changed_of_longer(ls, ss):
    sm = SequenceMatcher(None, ls, ss, autojunk=False); ivs, ch = [], 0
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == 'equal': continue
        ch += max(i2 - i1, j2 - j1)
        if i2 > i1: ivs.append((i1, i2))
    return ivs, ch


def collect(iso, gen, dom, faa, want_domain_differ):
    gl, gi = np.unique(gen, return_inverse=True); cnt = np.bincount(gi, minlength=len(gl))
    lo_i, sh_i, dis, sz = [], [], [], []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        differ = dom[a] != dom[b]
        if differ != want_domain_differ: continue
        if iso[a] not in faa or iso[b] not in faa: continue
        sa, sb = faa[iso[a]], faa[iso[b]]
        if sa == sb: continue
        lo, sh = (a, b) if len(sa) >= len(sb) else (b, a)
        ls, ss = faa[iso[lo]], faa[iso[sh]]
        ivs, ch = changed_of_longer(ls, ss)
        if ch == 0 or not ivs: continue
        res = [i for (x, y) in ivs for i in range(x, y)]
        lo_i.append(lo); sh_i.append(sh)
        dis.append(float(np.mean([TOPIDP.get(ls[i], 0.0) for i in res]))); sz.append(ch)
    return np.array(lo_i), np.array(sh_i), np.array(dis), np.array(sz)


def cv_incr(L15, L30, lo, sh, y, mask):
    """5-fold gene-disjoint AUROC for L15 vs L15+L30 on masked pairs; bootstrap incr CI."""
    idx = np.where(mask)[0]
    if len(idx) < 40 or y[mask].sum() < 5 or y[mask].sum() > len(idx) - 5:
        return None
    dL15 = (L15[lo] - L15[sh])[idx]; dL30 = (L30[lo] - L30[sh])[idx]
    yy = y[idx]
    Xs = {'L15': dL15, 'L15+L30': np.concatenate([dL15, dL30], 1)}
    order = np.arange(len(idx)); rng.shuffle(order); folds = np.array_split(order, 5)
    oof = {}
    for name, X in Xs.items():
        o = np.zeros(len(idx))
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), yy[tr])
            o[te] = clf.decision_function(sc.transform(X[te]))
        oof[name] = o
    a15 = roc_auc_score(yy, oof['L15']); aboth = roc_auc_score(yy, oof['L15+L30'])
    diffs = []
    for _ in range(1000):
        s = rng.integers(0, len(idx), len(idx))
        if yy[s].sum() == 0 or yy[s].sum() == len(s): continue
        diffs.append(roc_auc_score(yy[s], oof['L15+L30'][s]) - roc_auc_score(yy[s], oof['L15'][s]))
    return {'n': len(idx), 'AUROC_L15': float(a15), 'AUROC_L15_L30': float(aboth),
            'incr': [float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def main():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    faa = parse_faa()

    res = {'note': 'disorder-decoding saturation, target-matched across domain-differing vs non-domain, size-matched'}
    # common size window where both types have data
    WIN = (40, 175)
    print(f"common size window: {WIN[0]}-{WIN[1]} aa  (target=disorder for BOTH)\n")
    for label, want in [('domain-differing', True), ('non-domain(equal)', False)]:
        lo, sh, dis, sz = collect(iso, gen, dom, faa, want)
        # global-median disorder split (per group) as target
        y = (dis > np.median(dis)).astype(int)
        allm = np.ones(len(lo), bool)
        winm = (sz >= WIN[0]) & (sz <= WIN[1])
        full = cv_incr(L15, L30, lo, sh, y, allm)
        win = cv_incr(L15, L30, lo, sh, y, winm)
        res[label] = {'n_total': int(len(lo)), 'size_median': float(np.median(sz)),
                      'full_range': full, 'common_window_%d_%d' % WIN: win}
        print(f"[{label}]  n={len(lo)}  size median {np.median(sz):.0f}aa")
        if full: print(f"   full : n={full['n']} L15 {full['AUROC_L15']:.3f} L15+L30 {full['AUROC_L15_L30']:.3f} "
                       f"incr {full['incr'][0]:+.3f}[{full['incr'][1]:+.3f},{full['incr'][2]:+.3f}]")
        if win: print(f"   {WIN[0]}-{WIN[1]}aa: n={win['n']} L15 {win['AUROC_L15']:.3f} L15+L30 {win['AUROC_L15_L30']:.3f} "
                      f"incr {win['incr'][0]:+.3f}[{win['incr'][1]:+.3f},{win['incr'][2]:+.3f}]")
        else: print(f"   {WIN[0]}-{WIN[1]}aa: insufficient n")
        print()
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
