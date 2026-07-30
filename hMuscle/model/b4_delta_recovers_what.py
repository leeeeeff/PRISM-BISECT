#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_delta_recovers_what.py — Option A: what signal does δ=L30−L15 surface (given it is NOT domain)?

Paired brain outputs exist on the SAME 63994 isoforms × 41 BP GO terms:
  reports/expanded41_truebrain/v15d_score_matrix.npy   (L30-only model)
  reports/expanded41_truebrain/v17fstar_score_matrix.npy (δ=L30−L15 model)
(δ is net-harmful on BP but injects the SAME representational signal it uses to HELP MF.)

The δ-injected output change per within-gene pair: Δδout = |Δv17f*|_1 − |Δv15d|_1 (L1 over 41 terms).
We ask which signal explains Δδout:
  (1) per-axis incremental R^2 of {axis_k@L30, axis_k@δ} beyond size, for predicting Δδout — ranks the
      8 interpretable axes (axis0 disorder, axis3 domain[ruled out], axis5 length, ...).
  (2) correlation of Δδout with interpretable per-pair covariates (size, disorder_frac, nterm, resync,
      domain_faithful).
Gene-disjoint ridge CV(5). Brain severity pairs. Read-only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
BRAIN = ROOT / 'hMuscle/data/brain_isoquant_esm2/full'
INTERP = ROOT / 'reports/v20b_pca_interp'
EXP41 = ROOT / 'reports/expanded41_truebrain'
MAXLEN = 1022
L15_IDX, L30_IDX = 14, 29
AXLAB = {0: 'axis0(disorder)', 1: 'axis1', 2: 'axis2', 3: 'axis3(DOMAIN-ruledout)',
         4: 'axis4', 5: 'axis5(length)', 6: 'axis6', 7: 'axis7'}


def cvr2(X, y, grp):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    return r2_score(y, oof)


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bsp)
    df = pd.read_csv(SEV / 'brain_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'brain'].reset_index(drop=True)
    biso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    Z = np.load(INTERP / 'Z_brain_Nx30x8.npy')
    # paired model outputs — matrices are row-aligned to brain_full order (verified: expanded41 stores
    # gene symbols per row; row i == brain_full isoform i). Use identity indexing.
    V15 = np.load(EXP41 / 'v15d_score_matrix.npy').astype(np.float32)
    V17 = np.load(EXP41 / 'v17fstar_score_matrix.npy').astype(np.float32)
    assert V15.shape[0] == len(biso), f"row mismatch {V15.shape[0]} vs {len(biso)}"

    rows = {'d': [], 'size': [], 'axL30': [], 'axD': [], 'g': [],
            'disorder': [], 'nterm': [], 'resync': [], 'faith': []}
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = biso[li], biso[si]
        if lid not in seqs or sid not in seqs or seqs[lid][:MAXLEN] == seqs[sid][:MAXLEN]:
            continue
        d15 = float(np.abs(V15[li] - V15[si]).sum())
        d17 = float(np.abs(V17[li] - V17[si]).sum())
        rows['d'].append(d17 - d15)                                  # δ-injected output change
        rows['size'].append(float(r['size']))
        rows['axL30'].append(np.abs(Z[li, L30_IDX, :] - Z[si, L30_IDX, :]))
        dl = (Z[li, L30_IDX, :] - Z[li, L15_IDX, :]); ds = (Z[si, L30_IDX, :] - Z[si, L15_IDX, :])
        rows['axD'].append(np.abs(dl - ds))
        rows['g'].append(str(r['gene']))
        rows['disorder'].append(float(r['disorder_frac'])); rows['nterm'].append(int(r['nterm_overlap']))
        rows['resync'].append(int(r['resync_failure_binary'])); rows['faith'].append(int(r['domain_binary_faithful']))
    y = np.array(rows['d']); size = np.array(rows['size'])[:, None]
    axL30 = np.stack(rows['axL30']); axD = np.stack(rows['axD']); grp = np.array(rows['g'])
    print(f"n pairs={len(y)}, genes={len(np.unique(grp))}, mean Δδout={y.mean():+.4f} (sd {y.std():.3f})")

    print("\n" + "=" * 84)
    print("(1) Which AXIS carries δ's output-contribution? incremental R^2({axis_k@L30,axis_k@δ} | size)")
    print("=" * 84)
    r2_size = cvr2(size, y, grp)
    print(f"  size-alone R^2 = {r2_size:+.4f}")
    res = []
    for k in range(8):
        Xk = np.hstack([size, axL30[:, [k]], axD[:, [k]]])
        res.append((k, cvr2(Xk, y, grp) - r2_size))
    for k, inc in sorted(res, key=lambda t: -t[1]):
        bar = '#' * max(0, int(inc / 0.002))
        print(f"  {AXLAB[k]:<26} incremental R^2 = {inc:+.4f} {bar}")
    r2_all = cvr2(np.hstack([size, axL30, axD]), y, grp)
    print(f"  [all 8 axes @L30+δ | size] R^2 = {r2_all:+.4f}  (incremental {r2_all - r2_size:+.4f})")

    print("\n" + "=" * 84)
    print("(2) Correlation of δ-injected output change with interpretable covariates")
    print("=" * 84)
    for name, v in [('size', size[:, 0]), ('disorder_frac', np.array(rows['disorder'])),
                    ('nterm_overlap', np.array(rows['nterm'])), ('resync_fail', np.array(rows['resync'])),
                    ('domain_faithful', np.array(rows['faith']))]:
        rho = spearmanr(v, y).correlation
        print(f"  Spearman(Δδout, {name:<16}) = {rho:+.4f}")


if __name__ == '__main__':
    main()
