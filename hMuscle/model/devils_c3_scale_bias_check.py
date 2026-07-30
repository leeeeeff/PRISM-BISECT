#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c3_scale_bias_check.py

Devils-advocate CRITICAL C3: does CV-dir-acc (gene-disjoint 5-fold, orient=+1
self-consistency test) rise with n EVEN UNDER THE NULL (no real shared
direction), purely as a statistical artifact of larger training folds giving
a more stable mean-direction estimator? If so, the internal-edit "rise from
chance" (0.488 at n=627 -> 0.602/0.662 at n=4185/11666) and the hydrophilic-
cluster result could be partly/fully this artifact rather than real signal.

DECISIVE TEST (pre-registered): take the REAL internal-edit D vectors (muscle,
n=4185) but PERMUTE the gene_id labels (shuffle which gene each D-vector is
attached to) -- this destroys any genuine gene-family/mechanism structure
while preserving the exact marginal distribution of D vectors. Under this
permutation, CV-dir-acc SHOULD be ~0.5 regardless of n if the statistic is
unbiased. Repeat at matched small-n (subsample to 627, matching the original
2-iso pilot) and full n (4185), many replicates, compare the null
distributions.
  H_bias (devils-advocate's claim): null CV-dir-acc at n=4185 is
    systematically higher than at n=627 (CI's don't overlap / clear upward trend).
  H_unbiased: both null distributions center near 0.5 with no systematic
    n-dependent shift (the observed real-data rise is then genuine signal,
    not a statistic artifact).
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
SMALL_N = 627
rng = np.random.default_rng(123)


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


def cv_dir_acc(D, gene_id, orient, seed):
    r = np.random.default_rng(seed)
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)
    ug_perm = ug.copy()
    r.shuffle(ug_perm)
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


def build_internal_muscle():
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    df = pd.read_csv(SEV / 'muscle_severity_pairs_scored.tsv', sep='\t')
    df = df[(df['tissue'] == 'muscle') & (df['domain_binary'] == 0) & (df['nterm_overlap'] == 0)]
    iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
    iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
    seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)

    D, gene_id = [], []
    for _, r in df.iterrows():
        long_id, short_id = iso[int(r['long_idx'])], iso[int(r['short_idx'])]
        if long_id not in seqs or short_id not in seqs:
            continue
        ls, ss = seqs[long_id][:MAXLEN], seqs[short_id][:MAXLEN]
        if ls == ss:
            continue
        ivs, changed = changed_intervals(ls, ss)
        if changed == 0 or not ivs:
            continue
        D.append(emb[int(r['long_idx'])] - emb[int(r['short_idx'])])
        gene_id.append(r['gene'])
    return np.array(D), np.array(gene_id)


def main():
    print("[load] building internal-edit (domain_binary==0 & nterm_overlap==0) muscle population...")
    D, gene_id = build_internal_muscle()
    n_full = len(D)
    print(f"[load] n={n_full} pairs, {len(np.unique(gene_id))} genes")

    null_small = np.empty(N_PERM)
    null_full = np.empty(N_PERM)
    for p in range(N_PERM):
        perm_gene = rng.permutation(gene_id)  # break gene-family structure, keep D fixed
        # full-n null
        null_full[p] = cv_dir_acc(D, perm_gene, np.ones(n_full), seed=p)
        # matched small-n null: subsample SMALL_N rows (with their permuted gene labels)
        sel = rng.choice(n_full, SMALL_N, replace=False)
        null_small[p] = cv_dir_acc(D[sel], perm_gene[sel], np.ones(SMALL_N), seed=p + 10000)
        if (p + 1) % 50 == 0:
            print(f"  {p+1}/{N_PERM} permutation replicates done", flush=True)

    print(f"\n=== NULL distribution (gene-id permuted, real D vectors, orient=+1) ===")
    print(f"  small-n (n={SMALL_N}): mean={null_small.mean():.4f} std={null_small.std():.4f} "
          f"[{np.percentile(null_small,2.5):.4f},{np.percentile(null_small,97.5):.4f}]")
    print(f"  full-n  (n={n_full}): mean={null_full.mean():.4f} std={null_full.std():.4f} "
          f"[{np.percentile(null_full,2.5):.4f},{np.percentile(null_full,97.5):.4f}]")
    diff = null_full - null_small
    print(f"  full-n minus small-n null (paired by permutation draw): mean={diff.mean():+.4f} "
          f"CI=[{np.percentile(diff,2.5):+.4f},{np.percentile(diff,97.5):+.4f}]")
    if np.percentile(diff, 2.5) > 0:
        print("  => H_bias SUPPORTED: null CV-dir-acc is systematically higher at larger n")
    else:
        print("  => H_bias NOT supported at this sample-size gap: null stays ~flat with n")

    print(f"\n  reference: REAL (unpermuted) CV-dir-acc was 0.488 (2-iso small-n) and "
          f"0.602 (this full muscle population) -- compare against the null bands above.")


if __name__ == '__main__':
    main()
