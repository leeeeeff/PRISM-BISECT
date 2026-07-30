#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c3_candidates_check.py

Parallel check for candidates 1-4 (helix/sheet/hydro/charge median-split
orientation), which use a PER-PAIR orient (not the constant orient=+1 that
was just shown to be contaminated by a gene-independent population bias).
Structurally more like domain_binary's valid positive control (external,
data-independent covariate defines orientation) -- but still needs its own
null: is the SPECIFIC covariate-based split doing better than an ARBITRARY
balanced binary split of the same population?

DECISIVE TEST: permute the orient LABELS themselves (reshuffle which pairs
get +1 vs -1, preserving the exact same class balance as the real median
split) while keeping D vectors and gene_id fixed. If real orient's CV-dir-acc
sits outside this "random balanced split" null band, the covariate carries
real information beyond an arbitrary partition. If not, whatever "signal"
appeared is just an artifact of imposing ANY balanced binary split on this
population (same root cause as the orient=+1 problem, manifesting differently).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
from difflib import SequenceMatcher

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
MAXLEN = 1022
N_PERM = 300
rng = np.random.default_rng(123)

HELIX = {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,'H':1.00,
         'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,'T':0.83,'W':1.08,
         'Y':0.69,'V':1.06}
SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,
         'Y':1.47,'V':1.70}
HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
         'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
         'Y':-1.3,'V':4.2}
CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.1}


def changed_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    ivs, changed = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        changed += max(i2 - i1, j2 - j1)
        if i2 > i1:
            ivs.append((i1, i2))
    return ivs, changed


def cv_dir_acc(D, gene_id, orient, seed=None):
    """Gene-disjoint fold split is FIXED (not permuted here) -- only orient
    assignment is tested for informativeness, gene structure held constant."""
    r = np.random.default_rng(seed) if seed is not None else rng
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)
    ug_perm = ug.copy(); r.shuffle(ug_perm)
    folds = {g: i % 5 for i, g in enumerate(ug_perm)}
    fid = np.array([folds[g] for g in gene_id])
    correct = 0
    for k in range(5):
        te = fid == k; tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = Do[tr].mean(0); a /= (np.linalg.norm(a) + 1e-9)
        pred = np.dot(Do[te], a) > 0
        correct += int(pred.sum())
    return correct / n


def report(name, D, gene_id, cov):
    real_orient = np.where(cov > np.median(cov), 1.0, -1.0)
    n_pos = int((real_orient > 0).sum())
    real_acc = cv_dir_acc(D, gene_id, real_orient, seed=999999)

    null = np.empty(N_PERM)
    for p in range(N_PERM):
        perm_orient = real_orient.copy()
        rng.shuffle(perm_orient)  # same class balance, random pair assignment
        null[p] = cv_dir_acc(D, gene_id, perm_orient, seed=p)
    lo, hi = np.percentile(null, [2.5, 97.5])
    verdict = ('REAL signal beyond arbitrary-split artifact' if real_acc > hi else
               'INDISTINGUISHABLE from arbitrary-split artifact' if lo <= real_acc <= hi else
               'BELOW null(?)')
    print(f"\n[{name}] n={len(D)}, n_pos={n_pos}/{len(D)}")
    print(f"  real (covariate-based orient) CV-dir-acc = {real_acc:.4f}")
    print(f"  random-balanced-split null: mean={null.mean():.4f} CI=[{lo:.4f},{hi:.4f}]")
    print(f"  => {verdict}  (real - null_mean = {real_acc - null.mean():+.4f})")


def main():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    df = pd.read_csv(SEV / 'muscle_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == 'muscle'].reset_index(drop=True)
    iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
    iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
    seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)

    sub = df[(df['domain_binary'] == 0) & (df['nterm_overlap'] == 0)]
    D, gene_id, helix_d, sheet_d, hydro_d, charge_d = [], [], [], [], [], []
    for _, r in sub.iterrows():
        long_id, short_id = iso[int(r['long_idx'])], iso[int(r['short_idx'])]
        if long_id not in seqs or short_id not in seqs:
            continue
        ls, ss = seqs[long_id][:MAXLEN], seqs[short_id][:MAXLEN]
        if ls == ss:
            continue
        ivs, changed = changed_intervals(ls, ss)
        if changed == 0 or not ivs:
            continue
        changed_res_idx = [i for (u, v) in ivs for i in range(u, v) if i < len(ls)]
        if not changed_res_idx:
            continue
        D.append(emb[int(r['long_idx'])] - emb[int(r['short_idx'])])
        gene_id.append(r['gene'])
        helix_d.append(np.mean([HELIX.get(ls[i], 1.0) for i in changed_res_idx]))
        sheet_d.append(np.mean([SHEET.get(ls[i], 1.0) for i in changed_res_idx]))
        hydro_d.append(np.mean([HYDRO.get(ls[i], 0.0) for i in changed_res_idx]))
        charge_d.append(np.mean([CHARGE.get(ls[i], 0.0) for i in changed_res_idx]))
    D = np.array(D); gene_id = np.array(gene_id)
    print(f"[build] internal-edit pairs n={len(D)}")

    report('helix_delta', D, gene_id, np.array(helix_d))
    report('sheet_delta', D, gene_id, np.array(sheet_d))
    report('hydrophobicity_delta', D, gene_id, np.array(hydro_d))
    report('charge_delta', D, gene_id, np.array(charge_d))


if __name__ == '__main__':
    main()
