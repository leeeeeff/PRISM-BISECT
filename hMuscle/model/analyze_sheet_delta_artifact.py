#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_sheet_delta_artifact.py

Deep mechanistic analysis of sheet_delta artifact in brain but not muscle:
1. Sheet_delta distribution and median values
2. Gene-identity correlations with sheet_delta
3. Permutation mechanics: why brain ~0.653 unchanged but muscle varies
4. Hypothesis testing: sample size vs. biological difference
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from difflib import SequenceMatcher
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, pointbiserialr

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


def cv_dir_acc_full(D, gene_id, orient):
    """CV-dir-acc with fold details for debugging."""
    n = len(D)
    Do = D * orient[:, None]
    ug = np.unique(gene_id)

    local_rng = np.random.default_rng(42)
    ug_copy = ug.copy()
    local_rng.shuffle(ug_copy)
    folds = {g: i % 5 for i, g in enumerate(ug_copy)}
    fid = np.array([folds[g] for g in gene_id])

    correct_by_fold = []
    for k in range(5):
        te = fid == k
        tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            correct_by_fold.append((0, 0))
            continue
        a = Do[tr].mean(0)
        a /= (np.linalg.norm(a) + 1e-9)
        pred = np.dot(Do[te], a) > 0
        correct_by_fold.append((int(pred.sum()), int(te.sum())))

    total_correct = sum(c for c, _ in correct_by_fold)
    total_test = sum(t for _, t in correct_by_fold)
    acc = total_correct / n if n > 0 else 0.0

    return acc, correct_by_fold, fid, Do


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


def build_group_detailed(df, iso, seqs, emb, domain0, nterm_val):
    """Build group with detailed sheet_delta metadata."""
    sub = df[(df['domain_binary'] == domain0) & (df['nterm_overlap'] == nterm_val)]
    D, gene_id, sheet_d, edit_sizes = [], [], [], []

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
            edit_sizes.append(changed)
        except Exception:
            continue

    return np.array(D), np.array(gene_id), np.array(sheet_d), np.array(edit_sizes)


print("=" * 80)
print("SHEET_DELTA ARTIFACT MECHANISTIC ANALYSIS")
print("=" * 80)

for tissue in ['muscle', 'brain']:
    print(f"\n[{tissue.upper()}]")
    print("-" * 80)

    try:
        print(f"Loading {tissue} data...")
        df, iso, seqs, emb = load_pairs_and_seqs(tissue)

        print(f"Building internal-edit (non-domain) group...")
        D, gene_id, sheet_d, edit_sizes = build_group_detailed(
            df, iso, seqs, emb, domain0=0, nterm_val=0
        )

        if len(D) == 0:
            print("No data for this tissue.")
            continue

        print(f"✓ Valid pairs: {len(D)}")

        # 1. Sheet_delta distribution
        print(f"\n[1] Sheet_delta Distribution:")
        print(f"    Mean: {np.mean(sheet_d):.4f}, Std: {np.std(sheet_d):.4f}")
        print(f"    Min:  {np.min(sheet_d):.4f}, Max: {np.max(sheet_d):.4f}")
        print(f"    Median: {np.median(sheet_d):.4f}")
        print(f"    Q1/Q3: {np.percentile(sheet_d, 25):.4f} / {np.percentile(sheet_d, 75):.4f}")

        # 2. Median split effect
        median_val = np.median(sheet_d)
        orient = np.where(sheet_d > median_val, 1.0, -1.0)
        n_above = np.sum(orient > 0)
        n_below = np.sum(orient < 0)
        print(f"\n[2] Median-split (threshold={median_val:.4f}):")
        print(f"    Above median: {n_above} ({100*n_above/len(sheet_d):.1f}%)")
        print(f"    Below median: {n_below} ({100*n_below/len(sheet_d):.1f}%)")

        # 3. Gene-identity correlation with sheet_delta
        unique_genes = np.unique(gene_id)
        gene_idx = np.array([np.where(unique_genes == g)[0][0] for g in gene_id])
        corr_sheet_gene, p_corr = spearmanr(sheet_d, gene_idx)
        print(f"\n[3] Gene-identity Correlation:")
        print(f"    Spearman(sheet_delta, gene_rank): {corr_sheet_gene:.4f}, p={p_corr:.6f}")
        print(f"    (Higher correlation = sheet_delta biased toward specific gene clusters)")

        # 4. Gene distribution in orient=+1 class
        print(f"\n[4] Gene bias in median-split:")
        gene_dist_above = pd.Series(gene_id[orient > 0]).value_counts()
        gene_dist_below = pd.Series(gene_id[orient < 0]).value_counts()
        top_above = gene_dist_above.head(5)
        top_below = gene_dist_below.head(5)
        print(f"    Top 5 genes in orient=+1 (above median):")
        for g, c in top_above.items():
            print(f"      {g}: {c} pairs ({100*c/n_above:.1f}%)")
        print(f"    Top 5 genes in orient=-1 (below median):")
        for g, c in top_below.items():
            print(f"      {g}: {c} pairs ({100*c/n_below:.1f}%)")

        # 5. CV-dir-acc with detailed fold breakdown
        print(f"\n[5] CV-dir-acc Fold Breakdown:")
        acc_real, fold_details, fid, Do = cv_dir_acc_full(D, gene_id, orient)
        print(f"    Overall: {acc_real:.4f}")
        for fold_idx, (correct, total) in enumerate(fold_details):
            if total > 0:
                print(f"      Fold {fold_idx}: {correct}/{total} = {correct/total:.3f}")

        # 6. Edit size correlation with sheet_delta
        corr_sheet_size, p_size = spearmanr(sheet_d, edit_sizes)
        print(f"\n[6] Edit Size Correlation:")
        print(f"    Spearman(sheet_delta, edit_size): {corr_sheet_size:.4f}, p={p_size:.6f}")
        print(f"    (sheet_delta may be confounded with edit size)")

        # 7. Permutation detail: what happens to distribution
        print(f"\n[7] Permutation Mechanics:")
        print(f"    Real orient range: {orient.min():.1f} to {orient.max():.1f}")
        print(f"    Real direction strength (mean abs(Do)): {np.abs(Do).mean(axis=0).mean():.4f}")

        # Sample a few random permutations to see what changes
        local_rng = np.random.default_rng(99)
        for perm_i in range(3):
            gene_perm = gene_id.copy()
            local_rng.shuffle(gene_perm)
            acc_perm, fold_perm, _, _ = cv_dir_acc_full(D, gene_perm, orient)
            print(f"    Permutation {perm_i}: CV-acc = {acc_perm:.4f}")

        print(f"\n    → If permutation accuracy doesn't change, it means:")
        print(f"      'the orient signal is independent of which genes are in which fold'")
        print(f"      (i.e., fold assignment doesn't matter — same direction learned in all folds)")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("HYPOTHESIS GENERATION")
print("=" * 80)
print("""
If brain sheet_delta artifact (Δ=0.0000) but muscle signal (Δ=-0.0029):

Possible mechanisms:
  A) Larger brain sample (11.6k vs 4.1k) causes orient to become random-like
     → In small samples, random orient can still have fold-wise consistency
     → In large samples, any orient becomes fold-invariant (permutation-proof)

  B) Brain genes have intrinsically different sheet_delta distribution
     → Brain sheet_delta = random noise (high variance, low signal)
     → Muscle sheet_delta = biological signal (low variance, high signal)

  C) Median-split in brain happens to select genes orthogonal to embedding direction
     → Pure coincidence, requires checking gene overlap between tissues

  D) Brain has more uniform sheet composition across genes
     → If all brain genes similar in sheet_delta, median-split is meaningless
     → Permutation wouldn't matter (all pairs equally "sheet-like")

Next step: Check which hypothesis fits the data.
""")

print("=" * 80)
