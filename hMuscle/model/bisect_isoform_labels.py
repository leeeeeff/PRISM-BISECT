#!/usr/bin/env python3
"""
bisect_isoform_labels.py
=========================
BISECT 구조 증거 기반 이소폼 수준 GO 정답지 생성 및 PRISM 검증.

3가지 레이블 체계:
  [L1] LOC-직접 매핑: mts_score → GO:0007005 (Mito org) 이소폼별 신뢰도
  [L2] Reference-based: sd_norm=0(canonical)=1, 나머지=soft label
  [L3] 역할 전환 탐지: 같은 유전자 내 mts/nls 역전 케이스

목적:
  - PRISM이 gene-level label을 뛰어넘는 이소폼 특이적 예측을 하는가?
  - BISECT 구조 증거와 PRISM 예측의 일치도 측정
  - Mode A/B 분류의 검증
"""

import json, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_DIR = os.path.join(BASE, 'data/raw_data/data/id_lists')
ANNOT  = os.path.join(BASE, 'data/raw_data/data/annotations')
FEAT   = os.path.join(BASE, 'results_isoform/features')
MODEL  = os.path.join(BASE, 'model')
REPORT = os.path.join(BASE, '../reports/bisect_isoform_labels')
os.makedirs(REPORT, exist_ok=True)

GO_ORDER = ['GO:0007204','GO:0045214','GO:0006941','GO:0006914','GO:0043161',
            'GO:0007519','GO:0042692','GO:0055074','GO:0007005','GO:0007517',
            'GO:0032006','GO:0030048','GO:0006096','GO:0007268','GO:0007018',
            'GO:0031175','GO:0030182','GO:0000226']
GO_NAMES = ['Ca2+','Sarcomere','Muscle cont','Autophagy','Proteasome',
            'Skel dev','Muscle diff','Ca2+ homeo','Mito org','Muscle dev',
            'TOR','Actin move','Glycolysis','Synaptic','MT move',
            'Neuron proj','Neuron diff','MT cyto']
MITO_IDX = 8   # GO:0007005 Mitochondrion org


def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x,bytes) else str(x) for x in arr]


print("[1] Loading data...")
te_iso  = load_ids(os.path.join(MODEL, 'my_isoform_list_fixed.npy'))
te_gene = load_ids(os.path.join(MODEL, 'my_gene_list_fixed.npy'))
S_v15d  = np.load(os.path.join(BASE, '../reports/v15_bp_clean/score_matrix_18go_20260519_1914.npy')).astype(np.float32)
X_loc   = np.load(os.path.join(FEAT, 'loc/loc_features_test.npy')).astype(np.float32)
X_sd    = np.load(os.path.join(FEAT, 'splicing/splicing_delta_v2.npy')).astype(np.float32)
X_esm   = np.load(os.path.join(BASE, 'data/esm2_embeddings_t30_150M.npy')).astype(np.float32)
N = len(te_iso)
N_GO = len(GO_ORDER)

ENSG2SYM = {}
with open(os.path.join(ID_DIR, 'ensembl_to_symbol.txt')) as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]

gene_to_idx = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_to_idx[sym].append(i)

# Gene-level GO annotations
print("[2] Loading gene-level GO annotations...")
pos_gene_sets = {}
for go in GO_ORDER:
    pos = set()
    with open(os.path.join(ANNOT, 'human_annotations_unified_bp.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    pos_gene_sets[go] = pos

# Gene-level AUPRC baseline
print("[3] Gene-level AUPRC baseline (v15d)...")
Y_gene = np.zeros((N, N_GO), dtype=np.float32)
for j, go in enumerate(GO_ORDER):
    pos = pos_gene_sets[go]
    Y_gene[:, j] = np.array([1.0 if te_sym[i] in pos else 0.0 for i in range(N)])

def macro_auprc(Y, S):
    auprcs = []
    for j in range(Y.shape[1]):
        y_bin = (Y[:, j] > 0.5).astype(int)
        if y_bin.sum() < 1: continue
        auprcs.append(average_precision_score(y_bin, S[:, j]))
    return np.mean(auprcs), auprcs

macro_gene, per_go_gene = macro_auprc(Y_gene, S_v15d)
print(f"  Gene-level Macro AUPRC: {macro_gene:.4f}")

sd_norms = np.linalg.norm(X_sd, axis=1)
mts_scores = X_loc[:, 0]


# ── [L1] LOC 직접 매핑: mts_score → GO:0007005 ────────────────────────────────
print("\n[4] L1: mts_score-based isoform labels for Mito org...")

Y_mito_iso = np.zeros(N, dtype=np.float32)
n_genes_labeled = 0
mts_switch_genes = []

for sym, idxs in gene_to_idx.items():
    if sym not in pos_gene_sets['GO:0007005']:
        continue
    if len(idxs) < 2:
        # Single isoform → keep gene label
        Y_mito_iso[idxs[0]] = 1.0
        continue

    mts_vals = mts_scores[idxs]
    mts_max = mts_vals.max()
    mts_min = mts_vals.min()
    mts_range = mts_max - mts_min

    if mts_range > 0.10:  # meaningful variation
        # Soft label: proportional to mts_score within gene
        for idx in idxs:
            Y_mito_iso[idx] = (mts_scores[idx] - mts_min) / mts_range
        mts_switch_genes.append((sym, idxs, mts_vals))
        n_genes_labeled += 1
    else:
        # Uniform variation → all isoforms equally functional
        for idx in idxs:
            Y_mito_iso[idx] = 1.0

print(f"  Mito-annotated genes with mts variation >0.10: {n_genes_labeled}")
print(f"  Total mito-annotated isoforms with L1 label: {(Y_mito_iso > 0).sum()}")

# PRISM AUPRC on L1 labels (Mito org only)
mito_auprc_gene = per_go_gene[MITO_IDX]
mito_auprc_L1   = average_precision_score((Y_mito_iso > 0.5).astype(int), S_v15d[:, MITO_IDX])
print(f"\n  Mito org AUPRC:")
print(f"    Gene-level labels:     {mito_auprc_gene:.4f}")
print(f"    L1 (mts-based) labels: {mito_auprc_L1:.4f}  (Δ={mito_auprc_L1-mito_auprc_gene:+.4f})")

# Top switch genes
print(f"\n  Top MTS-switch genes (mts variation > 0.15):")
mts_switch_genes.sort(key=lambda x: -x[2].max() + x[2].min())
for sym, idxs, mts_v in mts_switch_genes[:10]:
    mts_str = '  '.join([f'{mts_scores[i]:.3f}(v15d={S_v15d[i,MITO_IDX]:.3f})' for i in idxs[:3]])
    print(f"    {sym:<12}: mts=[{mts_str}]")


# ── [L2] Reference-based 소프트 레이블 (전체 18 GO terms) ────────────────────────
print("\n[5] L2: Reference-isoform-based soft labels (all 18 GO terms)...")

Y_ref = Y_gene.copy()  # start from gene-level labels

# For genes with multiple isoforms: demote non-reference isoforms
for sym, idxs in gene_to_idx.items():
    if len(idxs) < 2:
        continue
    sd_vals = sd_norms[idxs]
    ref_idx = idxs[np.argmin(sd_vals)]  # reference = lowest sd_norm
    ref_sd  = sd_vals.min()
    max_sd  = sd_vals.max()
    if max_sd < 0.1:  # all isoforms similar → keep gene-level labels
        continue

    for j in range(N_GO):
        if te_sym[ref_idx] not in pos_gene_sets[GO_ORDER[j]]:
            continue
        # Graded label: reference=1.0, others proportionally reduced
        for idx in idxs:
            if idx == ref_idx:
                Y_ref[idx, j] = 1.0
            else:
                iso_sd = sd_norms[idx]
                # Soft demotion: more deviation → lower label
                decay = min(iso_sd / (max_sd + 1e-10), 1.0)
                Y_ref[idx, j] = max(0.0, 1.0 - 0.5 * decay)

macro_L2, per_go_L2 = macro_auprc(Y_ref, S_v15d)
print(f"  L2 Macro AUPRC: {macro_L2:.4f}  (vs gene-level: {macro_gene:.4f}  Δ={macro_L2-macro_gene:+.4f})")


# ── [L3] MTS/NLS 역전 케이스: BISECT 스타일 이진 레이블 ──────────────────────
print("\n[6] L3: MTS reversal-based binary isoform labels (BISECT-style)...")

Y_mts_binary = Y_gene.copy()
reversal_cases = []

for sym, idxs in gene_to_idx.items():
    if len(idxs) < 2:
        continue
    mts_v = mts_scores[idxs]
    if mts_v.max() - mts_v.min() < 0.15:
        continue

    # Check if gene is annotated for any mito-related GO term
    is_mito = (sym in pos_gene_sets['GO:0007005'])

    # Binary: rank isoforms by mts_score
    mts_rank = np.argsort(mts_v)[::-1]  # highest mts first
    ref_idx   = idxs[mts_rank[0]]   # highest mts = functional isoform
    other_idx = [idxs[k] for k in mts_rank[1:]]

    if is_mito:
        Y_mts_binary[ref_idx, MITO_IDX] = 1.0
        for oi in other_idx:
            decay = (mts_scores[ref_idx] - mts_scores[oi]) / mts_scores[ref_idx]
            if decay > 0.30:   # >30% MTS loss → label = 0 (lost mito function)
                Y_mts_binary[oi, MITO_IDX] = 0.0
                reversal_cases.append((sym, ref_idx, oi,
                                       float(mts_scores[ref_idx]),
                                       float(mts_scores[oi]),
                                       float(S_v15d[ref_idx, MITO_IDX]),
                                       float(S_v15d[oi, MITO_IDX])))

mito_auprc_L3 = average_precision_score(Y_mts_binary[:, MITO_IDX], S_v15d[:, MITO_IDX])
print(f"  Mito org AUPRC (L3 binary reversal): {mito_auprc_L3:.4f}")
print(f"  Number of reversal cases (>30% MTS loss): {len(reversal_cases)}")

print(f"\n  Top reversal cases:")
reversal_cases.sort(key=lambda x: -(x[3]-x[4]))
print(f"  {'Gene':<12} {'mts_ref':>8} {'mts_alt':>8} {'v15d_ref':>10} {'v15d_alt':>10} {'PRISM correct?':>15}")
for sym, ri, ai, mts_r, mts_a, v_r, v_a in reversal_cases[:15]:
    correct = '✓' if v_r > v_a else '✗'
    print(f"  {sym:<12} {mts_r:>8.3f} {mts_a:>8.3f} {v_r:>10.4f} {v_a:>10.4f} {correct:>15}")

# Accuracy of PRISM on reversal cases
n_correct = sum(1 for sym,ri,ai,mts_r,mts_a,v_r,v_a in reversal_cases if v_r > v_a)
print(f"\n  PRISM correctly ranks high-MTS isoform higher: {n_correct}/{len(reversal_cases)} ({n_correct/len(reversal_cases)*100:.1f}%)")


# ── 요약 ──────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  BISECT-DERIVED ISOFORM LABEL EVALUATION SUMMARY")
print("="*65)
print(f"\n  {'Label scheme':<35} {'Mito AUPRC':>12} {'Macro AUPRC':>13}")
print("  " + "-"*60)
print(f"  {'Gene-level (baseline)':<35} {mito_auprc_gene:>12.4f} {macro_gene:>13.4f}")
print(f"  {'L1: mts soft label (mito only)':<35} {mito_auprc_L1:>12.4f} {'—':>13}")
print(f"  {'L2: reference-based soft (all GO)':<35} {'—':>12} {macro_L2:>13.4f}")
print(f"  {'L3: mts binary reversal':<35} {mito_auprc_L3:>12.4f} {'—':>13}")
print()
print(f"  PRISM reversal accuracy (mts >0.15 diff, >30% loss):")
print(f"    {n_correct}/{len(reversal_cases)} = {n_correct/max(1,len(reversal_cases))*100:.1f}%  (expected random: 50%)")

# Save
out = {
    'gene_level_mito_auprc': float(mito_auprc_gene),
    'L1_mts_soft_mito_auprc': float(mito_auprc_L1),
    'L2_ref_macro_auprc': float(macro_L2),
    'L3_binary_mito_auprc': float(mito_auprc_L3),
    'n_reversal_cases': len(reversal_cases),
    'prism_reversal_accuracy': float(n_correct / max(1, len(reversal_cases))),
    'n_mts_switch_genes': n_genes_labeled,
}
with open(os.path.join(REPORT, 'bisect_label_eval.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: {REPORT}/bisect_label_eval.json")
print("="*65)
