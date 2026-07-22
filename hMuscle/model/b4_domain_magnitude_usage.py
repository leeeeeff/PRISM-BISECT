#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""b4_domain_magnitude_usage.py  (Option B / [LEARN] q(a): why is domain size->output rho only 0.41?)

S_map Panel D showed Spearman(edit size, |dPRISM|_1) = 0.68 non-domain but only 0.41 domain. Hypothesis:
for DOMAIN pairs, output magnitude is NOT just a size reaction — it also depends on WHICH domain-
structural content is removed, carried by the output-USED domain axis (axis3). If so, adding axis3
trajectory displacement should raise held-out R^2 BEYOND size for domain pairs (and NOT for non-domain).
This is the magnitude-side mirror of the ridge-reliance B4 result: domain is the used class.

Predict-before-you-look:
  domain:      incremental R^2(axis3 | size) > 0.01  (meaningfully above non-domain composition +0.003)
  non-domain:  incremental R^2(axis3 | size) ~ 0     (axis3 is the domain channel, inactive here)
  null:        both ~ 0  -> the 0.41 gap is noise/nonlinearity, not domain-structural usage.

Features per pair: edit size; per-axis trajectory displacement ||Z[li,:,k]-Z[si,:,k]||_2 over 30 layers
(axis3 = domain axis; all-8 = full structural). Gene-disjoint ridge, held-out R^2. Brain. Read-only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
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
MAXLEN = 1022


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
    iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    prism = np.load(ROOT / 'reports/brain_full_672_scores.npy').astype(np.float32)
    Z = np.load(INTERP / 'Z_brain_Nx30x8.npy')                       # (63994, 30, 8)

    rows = {'domain': {'y': [], 'size': [], 'ax': [], 'g': []},
            'nondomain': {'y': [], 'size': [], 'ax': [], 'g': []}}
    for _, r in df.iterrows():
        li, si = int(r['long_idx']), int(r['short_idx'])
        lid, sid = iso[li], iso[si]
        if lid not in seqs or sid not in seqs or seqs[lid][:MAXLEN] == seqs[sid][:MAXLEN]:
            continue
        disp = np.linalg.norm(Z[li] - Z[si], axis=0)                 # (8,) trajectory displacement per axis
        key = 'domain' if int(r['domain_binary']) == 1 else 'nondomain'
        rows[key]['y'].append(float(np.abs(prism[li] - prism[si]).sum()))
        rows[key]['size'].append(float(r['size']))
        rows[key]['ax'].append(disp)
        rows[key]['g'].append(str(r['gene']))

    print("=" * 90)
    print("DOMAIN-CLASS MAGNITUDE USAGE — does the domain axis (axis3) add output signal beyond size?")
    print("=" * 90)
    for key in ['domain', 'nondomain']:
        y = np.array(rows[key]['y']); size = np.array(rows[key]['size'])[:, None]
        ax = np.array(rows[key]['ax']); grp = np.array(rows[key]['g'])
        ax3 = ax[:, [3]]
        ax_no3 = ax[:, [0, 1, 2, 4, 5, 6, 7]]                        # other 7 axes (total-perturbation proxy)
        rho = spearmanr(size[:, 0], y).correlation
        r2_size = cvr2(size, y, grp)
        r2_ax3 = cvr2(np.hstack([size, ax3]), y, grp)
        r2_all = cvr2(np.hstack([size, ax]), y, grp)
        r2_ax3only = cvr2(ax3, y, grp)
        r2_no3 = cvr2(np.hstack([size, ax_no3]), y, grp)            # size + other 7 axes (baseline for specificity)
        print(f"\n[{key.upper()}]  n={len(y)}, genes={len(np.unique(grp))}, "
              f"Spearman(size,|dPRISM|)={rho:.3f}")
        print(f"  (a) size alone                 : R^2 = {r2_size:+.3f}")
        print(f"  (b) size + axis3 displacement  : R^2 = {r2_ax3:+.3f}   "
              f"(incremental axis3 | size = {r2_ax3 - r2_size:+.3f})")
        print(f"  (c) size + all 8 axes          : R^2 = {r2_all:+.3f}   "
              f"(incremental 8-axis | size = {r2_all - r2_size:+.3f})")
        print(f"  (d) axis3 displacement alone   : R^2 = {r2_ax3only:+.3f}")
        print(f"  (e) size + OTHER 7 axes        : R^2 = {r2_no3:+.3f}")
        print(f"  >> DISCRIMINATING: incremental axis3 BEYOND size+other-7 = {r2_all - r2_no3:+.3f}  "
              f"(if ~0 => generic magnitude confound, NOT axis3-specific)")
    print("\n" + "-" * 90)
    print("Read: if domain incremental(axis3|size) >> non-domain incremental, output magnitude for domain")
    print("      pairs uses domain-structural content (axis3), not just size -> explains the 0.41 vs 0.68 gap")
    print("      and re-confirms axis3 as the output-used domain channel from the MAGNITUDE side.")


if __name__ == '__main__':
    main()
