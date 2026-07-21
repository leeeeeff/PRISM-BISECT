#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sheet_delta_label_permutation.py

Final validation: Label-Permutation Null for sheet_delta

Distinguishes between:
  A) Artifact (sheet_delta orthogonal to embedding direction):
     label-perm CV-acc ≈ real CV-acc
  B) Real signal (sheet_delta captures real structure):
     label-perm CV-acc << real CV-acc (loses sensitivity when labels scrambled)

If brain sheet_delta is truly an artifact of fold-universal direction,
it will be INVARIANT to both gene permutation AND label permutation.

If brain sheet_delta has real biological content, label permutation
will break it even if gene permutation doesn't.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
MAXLEN = 1022

SHEET = {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,'H':0.87,
         'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,'T':1.19,'W':1.37,
         'Y':1.47,'V':1.70}


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
    """Cross-validated direction accuracy."""
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)

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
    D, gene_id, sheet_d = [], [], []

    for _, r in sub.iterrows():
        try:
            long_id, short_id = iso[int(r['long_idx'])], iso[int(r['short_idx'])]
            if long_id not in seqs or short_id not in seqs:
                continue

            long_s, short_s = seqs[long_id][:MAXLEN], seqs[short_id][:MAXLEN]
            if long_s == short_s:
                continue

            ivs, changed = changed_intervals(long_s, short_s)
            if changed == 0 or not ivs:
                continue

            changed_res_idx = [i for (u, v) in ivs for i in range(u, v) if i < len(long_s)]
            if not changed_res_idx:
                continue

            D.append(emb[int(r['long_idx'])] - emb[int(r['short_idx'])])
            gene_id.append(str(r['gene']))
            sheet_d.append(np.mean([SHEET.get(long_s[i], 1.0) for i in changed_res_idx]))
        except Exception:
            continue

    return np.array(D), np.array(gene_id), np.array(sheet_d)


print("=" * 80)
print("LABEL-PERMUTATION NULL: Sheet_Delta Validation")
print("=" * 80)

for tissue in ['muscle', 'brain']:
    print(f"\n[{tissue.upper()}]")
    print("-" * 80)

    try:
        print(f"Loading {tissue} data...")
        df, iso, seqs, emb = load_pairs_and_seqs(tissue)

        print(f"Building internal-edit (non-domain) group...")
        D, gene_id, sheet_d = build_group(df, iso, seqs, emb, domain0=0, nterm_val=0)

        if len(D) == 0:
            print("No data for this tissue.")
            continue

        print(f"✓ Valid pairs: {len(D)}\n")

        # Real sheet_delta test
        orient_real = np.where(sheet_d > np.median(sheet_d), 1.0, -1.0)
        acc_real, lo_real, hi_real = cv_dir_acc(D, gene_id, orient_real)

        print(f"REAL sheet_delta:")
        print(f"  CV-dir-acc: {acc_real:.4f} [{lo_real:.4f}, {hi_real:.4f}]")

        # Label-permutation null: scramble long/short labels
        print(f"\nLABEL-PERMUTATION NULL (n=10):")
        accs_label_perm = []
        label_rng = np.random.default_rng(200)

        for perm_i in range(10):
            # Flip orientation: treat short as long (negate D)
            D_perm = D * np.where(label_rng.random(len(D)) < 0.5, 1.0, -1.0)[:, None]
            # Keep real orient (based on real sheet_delta)
            acc_perm, _, _ = cv_dir_acc(D_perm, gene_id, orient_real)
            accs_label_perm.append(acc_perm)
            if perm_i < 3:
                print(f"  Perm {perm_i}: {acc_perm:.4f}")

        accs_label_perm = np.array(accs_label_perm)
        acc_perm_mean = accs_label_perm.mean()
        acc_perm_std = accs_label_perm.std()
        acc_perm_ci = np.percentile(accs_label_perm, [2.5, 97.5])

        print(f"  Label-perm mean: {acc_perm_mean:.4f} ± {acc_perm_std:.4f} [{acc_perm_ci[0]:.4f}, {acc_perm_ci[1]:.4f}]")

        # Interpretation
        delta_label = acc_real - acc_perm_mean
        print(f"\n  Δ (real - label_perm): {delta_label:+.4f}")

        if lo_real > acc_perm_ci[1]:
            verdict = "✅ REAL SIGNAL: Real >> Label-Perm (label-permutation breaks it)"
        elif hi_real < acc_perm_ci[0]:
            verdict = "❌ ARTIFACT: Real ≈ Label-Perm (label-permutation doesn't break it)"
        else:
            if abs(delta_label) < 0.01:
                verdict = "❌ LIKELY ARTIFACT: Δ ≈ 0 even after label permutation"
            else:
                verdict = "⚠️ MARGINAL: Unclear, effect exists but small"

        print(f"\n  Verdict: {verdict}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY & INTERPRETATION")
print("=" * 80)
print("""
If real CV-acc >> label-perm CV-acc:
  → Sheet_delta captures real biological structure
  → Include in model with confidence

If real CV-acc ≈ label-perm CV-acc:
  → Sheet_delta is artifact of embedding space geometry
  → Exclude or mark as tissue-specific anomaly

Why label permutation?:
  - Gene-permutation tests: "is signal independent of gene identity?"
  - Label-permutation tests: "is signal independent of isoform pair orientation?"

  Brain shows gene-permutation invariance (artifact of large n)
  But should show label-permutation SENSITIVITY (if real signal)
  This test distinguishes between these two.
""")
print("=" * 80)
