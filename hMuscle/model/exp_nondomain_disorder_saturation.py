#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_nondomain_disorder_saturation.py  (Option B / design B extended to non-domain regime)
Completes the saturation picture across the FULL splice-size range. The domain-completeness test
(exp_saturation_by_splicesize.py) only covers domain-differing (large) splices. Here we test the
COMPLEMENTARY non-domain regime (equal-domain-count 2-iso pairs, the smaller splices) using the
structural signature §6 attributes to non-domain edits: DISORDER (TOP-IDP).

Two questions:
 (Q1, tests §6 'encoded by ESM-2') Does the inter-isoform embedding difference encode whether a
     non-domain splice edits a DISORDERED region?  AUROC(δ_R -> disorder-high) > 0.5 ?
 (Q2, saturation) Does adding φ_L30 to φ_L15 improve this (refute saturation) or not (confirm)?
     Compare 5-fold gene-disjoint CV AUROC for R=L15 vs R=[L15,L30]; stratify by splice size.

PREDICTION (S2): (Q1) encoded, AUROC>0.5; (Q2) uniform saturation — [L15,L30] ≈ L15 in every bin.
REFUTATION: some size bin where [L15,L30] >> L15 (non-domain regime leaves layer headroom).
"""
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
DATA = ROOT / 'hMuscle/data'
MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/nondomain_disorder_saturation.json'
TOPIDP = {'A': 0.06, 'R': 0.180, 'N': 0.007, 'D': 0.192, 'C': 0.02, 'Q': 0.318, 'E': 0.736,
          'G': 0.166, 'H': 0.303, 'I': -0.486, 'L': -0.326, 'K': 0.586, 'M': -0.397,
          'F': -0.697, 'P': 0.987, 'S': 0.341, 'T': 0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.121}
rng = np.random.default_rng(42)


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(' ', '')


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


def changed_of_longer(long_s, short_s):
    """changed intervals on the longer sequence + total changed aa."""
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs, changed = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            ivs.append((i1, i2))
    return ivs, changed


def main():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    faa = parse_faa()
    gl, gi = np.unique(gen, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))

    # equal-domain (non-domain-completeness) 2-iso pairs
    long_idx, short_idx, disorder, size, gene_id = [], [], [], [], []
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
        ivs, changed = changed_of_longer(ls, ss)
        if changed == 0 or not ivs:
            continue
        res = [i for (x, y) in ivs for i in range(x, y)]
        dis = float(np.mean([TOPIDP.get(ls[i], 0.0) for i in res]))
        long_idx.append(lo); short_idx.append(sh); disorder.append(dis); size.append(changed); gene_id.append(g)
    long_idx = np.array(long_idx); short_idx = np.array(short_idx)
    disorder = np.array(disorder); size = np.array(size); gene_id = np.array(gene_id)
    n = len(long_idx)
    print(f"non-domain (equal-domain) pairs: {n}  splice size median {np.median(size):.0f}aa  "
          f"disorder median {np.median(disorder):.3f}", flush=True)

    # target: disordered splice (above-median disorder)
    y = (disorder > np.median(disorder)).astype(int)

    # delta features per representation
    dL15 = L15[long_idx] - L15[short_idx]
    dL30 = L30[long_idx] - L30[short_idx]
    reps = {'L15': dL15, 'L15+L30': np.concatenate([dL15, dL30], 1), 'delta(L30-L15)': dL30 - dL15}

    # gene-disjoint 5-fold (each pair already 1 gene, so just index folds)
    order = np.arange(n); rng.shuffle(order)
    folds = np.array_split(order, 5)

    def cv_auroc(X):
        oof = np.zeros(n)
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(X[te]))
        return oof

    oof = {r: cv_auroc(X) for r, X in reps.items()}
    auroc = {r: float(roc_auc_score(y, oof[r])) for r in reps}

    # size tertiles (this regime = smaller splices; completes the range vs domain-differing)
    q1, q2 = np.percentile(size, [33.33, 66.67])
    bin_of = np.where(size <= q1, 0, np.where(size <= q2, 1, 2))
    bin_names = [f"small(<= {q1:.0f}aa)", f"mid({q1:.0f}-{q2:.0f}aa)", f"large(> {q2:.0f}aa)"]

    def boot_auroc_diff(mask, rB, rA, nb=1000):
        yy = y[mask]; a = oof[rA][mask]; b = oof[rB][mask]
        idx = np.where(mask)[0]; diffs = []
        for _ in range(nb):
            s = rng.integers(0, len(idx), len(idx))
            if yy[s].sum() == 0 or yy[s].sum() == len(s):
                continue
            diffs.append(roc_auc_score(yy[s], b[s]) - roc_auc_score(yy[s], a[s]))
        return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    res = {'n_pairs': int(n), 'splice_size_median': float(np.median(size)),
           'disorder_median': float(np.median(disorder)),
           'overall_AUROC': auroc, 'size_bins': {},
           'prediction': 'Q1 encoded (AUROC>0.5); Q2 uniform saturation (L15+L30 ~= L15)'}
    print("\n=== Q1: is non-domain splice disorder ENCODED by ESM-2 (delta embedding)? ===")
    for r in reps:
        print(f"  AUROC({r:>13} -> disorder) = {auroc[r]:.3f}")
    print("\n=== Q2: saturation by splice size (incremental of adding L30 to L15) ===")
    print(f"{'bin':<20} {'n':>5} {'AUROC_L15':>10} {'AUROC_L15+L30':>13}  {'incr [CI]'}")
    for bi in range(3):
        mask = bin_of == bi
        aL15 = roc_auc_score(y[mask], oof['L15'][mask])
        aBoth = roc_auc_score(y[mask], oof['L15+L30'][mask])
        d, lo, hi = boot_auroc_diff(mask, 'L15+L30', 'L15')
        res['size_bins'][bin_names[bi]] = {'n': int(mask.sum()), 'AUROC_L15': float(aL15),
                                           'AUROC_L15_L30': float(aBoth), 'incremental': [d, lo, hi]}
        print(f"{bin_names[bi]:<20} {int(mask.sum()):>5} {aL15:>10.3f} {aBoth:>13.3f}  {d:>+.3f}[{lo:+.3f},{hi:+.3f}]")

    d, lo, hi = boot_auroc_diff(np.ones(n, bool), 'L15+L30', 'L15')
    res['overall_incremental_L30'] = [d, lo, hi]
    print(f"\noverall incremental (L15+L30 − L15): {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    encoded = auroc['L15'] > 0.55 or auroc['L15+L30'] > 0.55
    uniform_sat = all(res['size_bins'][b]['incremental'][1] <= 0 for b in res['size_bins'])
    res['Q1_encoded'] = bool(encoded)
    res['Q2_uniform_saturation'] = bool(uniform_sat)
    print(f"\nQ1 non-domain disorder ENCODED: {encoded}   Q2 UNIFORM SATURATION: {uniform_sat}")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
