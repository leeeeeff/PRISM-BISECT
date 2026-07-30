#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_saturation_by_splicesize.py  (Option B / design B)
Tests whether the single-layer SATURATION of the within-gene domain-completeness signal is UNIFORM
across splice size, or whether some size window leaves headroom that the L30/δ_layer contrast can
exploit. Instrument: domain-completeness DIRECTION (within-pair anchor, cf. manuscript §794),
computed per representation R∈{φ_L15, φ_L30, δ_layer=L30−L15}, evaluated on gene-disjoint 2-isoform
pairs, stratified by difflib splice size (changed aa).

PREDICTION (pre-registered, S2): saturation is uniform — in EVERY size bin, neither φ_L30 nor
δ_layer exceeds φ_L15 (Δacc ≤ 0 within gene-bootstrap CI). Since δ_layer≡L30 given L15, this is
the direct 'does the second layer add within-gene domain info beyond L15' test.
REFUTATION: some size bin where L30 or δ beats L15 (partial headroom) → δ_layer has a size regime.
"""
import os, re
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
import json

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
DATA = ROOT / 'hMuscle/data'
MODEL = ROOT / 'hMuscle/model'
FAA = ROOT / 'reports/muscle_labelgap/muscle_2iso.fa'          # 2-iso proteins (BambuTx base)
DOMAIN_MAT = ROOT / 'hMuscle/results_isoform/features/domain_matrix_proper_test.npy'
OUT = ROOT / 'reports/muscle_labelgap/saturation_by_splicesize.json'
_AA = set('ACDEFGHIKLMNPQRSTVWY')
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
        seq = ''.join(buf)
        base = strip_orf(cur)
        if base not in best or len(seq) > len(best[base]):
            best[base] = seq
    for line in open(FAA):
        if line.startswith('>'):
            flush(); cur = line[1:].split()[0]; buf = []
        else:
            buf.append(line.strip())
    flush()
    return best


def changed_aa(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal')


def main():
    iso = np.array([clean(x) for x in np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)])
    gen = np.array([clean(x) for x in np.load(MODEL / 'my_gene_list_fixed.npy', allow_pickle=True)])
    dom = np.load(DOMAIN_MAT).sum(1).astype(int)               # per-isoform Pfam domain count
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    dlt = L30 - L15
    faa = parse_faa()
    print(f"iso {len(iso)}  faa {len(faa)}  L15 {L15.shape}", flush=True)

    gl, gi = np.unique(gen, return_inverse=True)
    cnt = np.bincount(gi, minlength=len(gl))

    # collect 2-iso genes with domain difference + both proteins → (complete_idx, trunc_idx, changed_aa)
    pairs = []
    for g in np.where(cnt == 2)[0]:
        a, b = np.where(gi == g)[0]
        if dom[a] == dom[b]:
            continue
        ba, bb = iso[a], iso[b]
        if ba not in faa or bb not in faa:
            continue
        sa, sb = faa[ba], faa[bb]
        if sa == sb:
            continue
        c, t = (a, b) if dom[a] > dom[b] else (b, a)           # complete has more domains
        pairs.append((c, t, changed_aa(sa, sb)))
    pairs = np.array(pairs, dtype=object)
    sizes = np.array([p[2] for p in pairs])
    print(f"usable 2-iso domain-diff pairs: {len(pairs)}  splice size median {np.median(sizes):.0f}aa", flush=True)

    reps = {'phi_L15': L15, 'phi_L30': L30, 'delta_layer': dlt}

    def acc_for(rep_mat, anchor_pairs, eval_pairs):
        # anchor direction = mean(complete - trunc) over anchor pairs, normalized
        v = np.zeros(rep_mat.shape[1], np.float32)
        for c, t, _ in anchor_pairs:
            v += rep_mat[c] - rep_mat[t]
        v /= (np.linalg.norm(v) + 1e-8)
        correct = np.array([1.0 if float(v @ (rep_mat[c] - rep_mat[t])) > 0 else 0.0
                            for c, t, _ in eval_pairs])
        return correct

    # 2-fold gene-disjoint (pairs already gene-disjoint: 1 pair per gene), swap anchor/eval
    idx = np.arange(len(pairs)); rng.shuffle(idx)
    half = len(idx) // 2
    folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]

    # per pair: pooled correctness for each rep (from the fold where it is in EVAL)
    per_pair_correct = {r: np.full(len(pairs), np.nan) for r in reps}
    for anc_i, ev_i in folds:
        anc = [tuple(pairs[k]) for k in anc_i]
        ev = [tuple(pairs[k]) for k in ev_i]
        for r, mat in reps.items():
            corr = acc_for(mat, anc, ev)
            for k, cval in zip(ev_i, corr):
                per_pair_correct[r][k] = cval

    # size tertiles
    q1, q2 = np.percentile(sizes, [33.33, 66.67])
    bin_of = np.where(sizes <= q1, 0, np.where(sizes <= q2, 1, 2))
    bin_names = [f"small(<= {q1:.0f}aa)", f"mid({q1:.0f}-{q2:.0f}aa)", f"large(> {q2:.0f}aa)"]

    def boot_diff(mask, rB, rA, nboot=1000):
        a = per_pair_correct[rA][mask]; b = per_pair_correct[rB][mask]
        n = len(a); diffs = []
        for _ in range(nboot):
            s = rng.integers(0, n, n)
            diffs.append(b[s].mean() - a[s].mean())
        return float(np.mean(b - a)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    res = {'n_pairs': len(pairs), 'splice_size_median': float(np.median(sizes)),
           'size_bins': {}, 'prediction': 'uniform saturation: delta & L30 <= L15 in every bin'}
    print("\n=== within-gene domain-completeness accuracy by splice-size bin ===")
    print(f"{'bin':<20} {'n':>5} {'L15':>7} {'L30':>7} {'delta':>7}  {'d(delta-L15) [CI]':>26}  {'d(L30-L15) [CI]'}")
    for bi in range(3):
        mask = bin_of == bi
        n = int(mask.sum())
        aL15 = np.nanmean(per_pair_correct['phi_L15'][mask])
        aL30 = np.nanmean(per_pair_correct['phi_L30'][mask])
        aDlt = np.nanmean(per_pair_correct['delta_layer'][mask])
        dd, dlo, dhi = boot_diff(mask, 'delta_layer', 'phi_L15')
        ld, llo, lhi = boot_diff(mask, 'phi_L30', 'phi_L15')
        res['size_bins'][bin_names[bi]] = {
            'n': n, 'acc_L15': aL15, 'acc_L30': aL30, 'acc_delta': aDlt,
            'delta_minus_L15': [dd, dlo, dhi], 'L30_minus_L15': [ld, llo, lhi]}
        print(f"{bin_names[bi]:<20} {n:>5} {aL15:>7.3f} {aL30:>7.3f} {aDlt:>7.3f}  "
              f"{dd:>+7.3f}[{dlo:+.3f},{dhi:+.3f}]  {ld:>+7.3f}[{llo:+.3f},{lhi:+.3f}]")

    # overall
    allmask = np.ones(len(pairs), bool)
    for r in reps:
        res.setdefault('overall_acc', {})[r] = float(np.nanmean(per_pair_correct[r]))
    dd, dlo, dhi = boot_diff(allmask, 'delta_layer', 'phi_L15')
    res['overall_delta_minus_L15'] = [dd, dlo, dhi]
    print(f"\noverall: L15 {res['overall_acc']['phi_L15']:.3f}  L30 {res['overall_acc']['phi_L30']:.3f}  "
          f"delta {res['overall_acc']['delta_layer']:.3f}  | delta-L15 {dd:+.3f}[{dlo:+.3f},{dhi:+.3f}]")
    verdict = all(res['size_bins'][b]['delta_minus_L15'][2] <= 0.005 and
                  res['size_bins'][b]['L30_minus_L15'][2] <= 0.02 for b in res['size_bins'])
    res['uniform_saturation_confirmed'] = bool(verdict)
    print(f"\nUNIFORM SATURATION {'CONFIRMED' if verdict else 'REFUTED (some bin shows headroom)'}")
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[saved] {OUT}")


if __name__ == '__main__':
    main()
