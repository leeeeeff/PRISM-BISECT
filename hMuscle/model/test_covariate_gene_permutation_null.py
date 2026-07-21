#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_covariate_gene_permutation_null.py

Gene-permutation null check for 4 internal-edit covariates (helix/sheet/hydro/charge).
Reuses the framework from explore_internal_edit_full_population.py but adds
gene-permutation null verification to check if orient signals are cross-gene
mechanisms or gene-identity-independent artifacts.

Key method:
- REAL: cv_dir_acc(D, gene_id, orient)
- NULL: cv_dir_acc(D, gene_id_PERMUTED, orient) — same orient, shuffled gene IDs
- If REAL ≈ NULL, then orient signal is gene-independent artifact (fails cross-gene test)
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022
rng = np.random.default_rng(42)

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


def cv_dir_acc(D, gene_id, orient):
    """Cross-validated direction accuracy with fixed random seed."""
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)

    # Use fixed seed for reproducible fold assignment
    local_rng = np.random.default_rng(42)
    ug_copy = ug.copy()
    local_rng.shuffle(ug_copy)
    folds = {g: i % 5 for i, g in enumerate(ug_copy)}
    fid = np.array([folds[g] for g in gene_id])

    correct = 0
    for k in range(5):
        te = fid == k
        tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        a = Do[tr].mean(0)
        a /= (np.linalg.norm(a) + 1e-9)
        pred = np.dot(Do[te], a) > 0
        correct += int(pred.sum())

    acc = correct / n
    se = np.sqrt(acc * (1 - acc) / n)
    return acc, acc - 1.96 * se, acc + 1.96 * se


def cv_dir_acc_permuted(D, gene_id, orient, n_perm=10):
    """Compute cv_dir_acc with permuted gene IDs (null hypothesis)."""
    accs = []
    for perm_idx in range(n_perm):
        perm_rng = np.random.default_rng(100 + perm_idx)
        gene_permuted = gene_id.copy()
        perm_rng.shuffle(gene_permuted)
        acc_perm, _, _ = cv_dir_acc(D, gene_permuted, orient)
        accs.append(acc_perm)

    accs = np.array(accs)
    return accs.mean(), accs.std(), np.percentile(accs, [2.5, 97.5])


def load_pairs_and_seqs(tissue):
    spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
    bsp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsp)

    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df = df[df['tissue'] == tissue].reset_index(drop=True)

    if tissue == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
        L15 = np.load(DATA / 'esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(DATA / 'esm2_layer_30_t30_150M.npy').astype(np.float32)
    else:
        iso = [str(x) for x in np.load(BRAIN / 'brain_full_ids.npy', allow_pickle=True)]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
        L15 = np.load(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(BRAIN / 'brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)

    emb = np.concatenate([L15, L30], axis=1)
    return df, iso, seqs, emb


def build_group(df, iso, seqs, emb, domain0, nterm_val):
    sub = df[(df['domain_binary'] == domain0) & (df['nterm_overlap'] == nterm_val)]
    D, gene_id, helix_d, sheet_d, hydro_d, charge_d = [], [], [], [], [], []
    n_missing = 0

    for _, r in sub.iterrows():
        try:
            long_id, short_id = iso[int(r['long_idx'])], iso[int(r['short_idx'])]
            if long_id not in seqs or short_id not in seqs:
                n_missing += 1
                continue

            long_s, short_s = seqs[long_id][:MAXLEN], seqs[short_id][:MAXLEN]
            if long_s == short_s:
                n_missing += 1
                continue

            ivs, changed = changed_intervals(long_s, short_s)
            if changed == 0 or not ivs:
                n_missing += 1
                continue

            changed_res_idx = [i for (u, v) in ivs for i in range(u, v) if i < len(long_s)]
            if not changed_res_idx:
                n_missing += 1
                continue

            D.append(emb[int(r['long_idx'])] - emb[int(r['short_idx'])])
            gene_id.append(r['gene'])
            helix_d.append(np.mean([HELIX.get(long_s[i], 1.0) for i in changed_res_idx]))
            sheet_d.append(np.mean([SHEET.get(long_s[i], 1.0) for i in changed_res_idx]))
            hydro_d.append(np.mean([HYDRO.get(long_s[i], 1.0) for i in changed_res_idx]))
            charge_d.append(sum([CHARGE.get(long_s[i], 0.0) for i in changed_res_idx]) / len(changed_res_idx))
        except Exception as e:
            n_missing += 1
            continue

    return (np.array(D), np.array(gene_id), np.array(helix_d),
            np.array(sheet_d), np.array(hydro_d), np.array(charge_d), n_missing)


def test_covariate_null(name, D, gene_id, covariate):
    """Test single covariate with gene-permutation null."""
    if len(D) < 5:
        print(f"  [{name}] n={len(D)} (too small, skip)")
        return

    orient = np.where(covariate > np.median(covariate), 1.0, -1.0)

    # Real cv_dir_acc
    real_acc, real_lo, real_hi = cv_dir_acc(D, gene_id, orient)

    # Gene-permutation null (10 permutations)
    null_mean, null_std, (null_lo, null_hi) = cv_dir_acc_permuted(D, gene_id, orient, n_perm=10)

    # Verdict
    if abs(real_acc - null_mean) < null_std:
        verdict = "❌ GENE-INDEPENDENT (artifact)"
    elif real_lo > 0.5:
        verdict = "✅ CROSS-GENE SIGNAL (explained)"
    else:
        verdict = "⚠️ MARGINAL (unclear)"

    print(f"  [{name:15}] n={len(D):4d}")
    print(f"    REAL:     {real_acc:.4f} [{real_lo:.4f}, {real_hi:.4f}]")
    print(f"    NULL:     {null_mean:.4f} ± {null_std:.4f} [{null_lo:.4f}, {null_hi:.4f}]")
    print(f"    Δ (real - null): {real_acc - null_mean:+.4f}")
    print(f"    {verdict}\n")

    return real_acc, real_lo, real_hi, null_mean, null_std


# Main
print("=" * 80)
print("GENE-PERMUTATION NULL CHECK: 4 Internal-Edit Covariates")
print("=" * 80)

for tissue in ['muscle', 'brain']:
    print(f"\n[{tissue.upper()}]")
    print("-" * 80)

    try:
        print(f"Loading {tissue} data...")
        df, iso, seqs, emb = load_pairs_and_seqs(tissue)
        print(f"  Loaded {len(df)} pairs, {len(iso)} isoforms, {len(seqs)} sequences")

        # Internal edits only (nterm_overlap==0, domain_binary==0 for non-domain)
        print(f"\nAnalyzing internal edits (nterm_overlap==0, domain_binary==0)...")
        D, gene_id, helix_d, sheet_d, hydro_d, charge_d, n_miss = build_group(
            df, iso, seqs, emb, domain0=0, nterm_val=0
        )

        if len(D) == 0:
            print("  No valid data for this group.")
            continue

        print(f"  Valid pairs: {len(D)} (missing: {n_miss})\n")

        # Test each covariate
        results = {}
        for name, cov in [('helix_delta', helix_d), ('sheet_delta', sheet_d),
                          ('hydro_delta', hydro_d), ('charge_delta', charge_d)]:
            result = test_covariate_null(name, D, gene_id, cov)
            if result:
                results[name] = result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
