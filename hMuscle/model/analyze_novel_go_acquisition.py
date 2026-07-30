#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_novel_go_acquisition.py
================================
PRISM이 gene-level GO annotation의 경계를 넘어서는 예측을 하는 두 케이스를 통계화한다.

Case A (완전 무주석 gene): gene이 18개 학습 GO term 전체에서 음성(annotation 0개)인데,
        그 gene의 isoform 중 하나가 특정 GO term에서 score > THRESHOLD를 획득.
Case B (부분 주석 gene, 타 term 획득): gene이 일부 GO term에서는 양성이지만,
        그 gene에 배정되지 않은(음성) 다른 GO term에서 isoform의 score가
        THRESHOLD를 넘고, 그 isoform 자신의 gene-annotated term 최고 score마저 넘어선 경우
        (= 그 isoform 입장에서 "내 gene이 원래 가진 기능"보다 "gene에 없는 기능"이 더 강하게 예측됨).

Data: TRUE brain (63,994 isoforms, brain_isoquant_esm2/full) — v15d_brain_eval.py의
      저장된 score matrix(원본 18-BP PRISM, brain zero-shot) 재사용. 이 트랙은
      muscle/brain mislabeling 버그와 무관(별도 확인됨).
"""

import os, json
import numpy as np
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BRAIN_DIR  = '../data/brain_isoquant_esm2/full'
ANNOT_DIR  = '../data/raw_data/data/annotations'
SCORE_MAT  = '../../reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy'
OUT_DIR    = '../../reports/novel_go_acquisition_20260714'
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO     = len(GO_KEYS)

def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

print("=" * 70)
print("  Novel GO acquisition beyond gene-level annotation (TRUE brain, 63,994)")
print("=" * 70)

te_isoid = load_ids(f'{BRAIN_DIR}/brain_full_ids.npy')
te_sym   = load_ids(f'{BRAIN_DIR}/brain_full_gene_names.npy')
N_TE     = len(te_isoid)
score_matrix = np.load(SCORE_MAT).astype(np.float32)
assert score_matrix.shape == (N_TE, N_GO), f"shape mismatch: {score_matrix.shape} vs ({N_TE},{N_GO})"

# ── gene-level 18-term label (positive set per gene symbol) ──────────────────
gene_pos = defaultdict(set)  # gene_sym -> set of GO indices positive
for k_idx, go_term in enumerate(GO_KEYS):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    for sym in pos:
        gene_pos[sym].add(k_idx)

print(f"  Loaded {N_TE} brain isoforms, {N_GO} GO terms")
print(f"  Genes with >=1 positive GO label (any of 18 terms): {len(gene_pos)}")

# ── group isoform indices by gene symbol ─────────────────────────────────────
gene_to_idx = defaultdict(list)
for i, sym in enumerate(te_sym):
    gene_to_idx[sym].append(i)

n_genes_total = len(gene_to_idx)
n_genes_fully_negative = sum(1 for g in gene_to_idx if len(gene_pos.get(g, set())) == 0)
n_genes_partial = n_genes_total - n_genes_fully_negative
print(f"  Unique genes in brain set: {n_genes_total}")
print(f"  Genes fully GO-negative (0/{N_GO} annotated): {n_genes_fully_negative}")
print(f"  Genes with >=1 annotated term: {n_genes_partial}")

# ── Case A: fully-negative gene, isoform crosses threshold on any term ───────
caseA_genes = set()
caseA_isoforms = []       # (gene, isoform_id, term_key, score)
caseA_term_counts = defaultdict(int)

# ── Case B: partially-annotated gene, isoform's foreign-term score
#            exceeds threshold AND exceeds its own best gene-annotated-term score
caseB_genes = set()
caseB_isoforms = []
caseB_term_counts = defaultdict(int)

for g, idxs in gene_to_idx.items():
    pos_k = gene_pos.get(g, set())
    if len(pos_k) == 0:
        # Case A
        for i in idxs:
            row = score_matrix[i]
            hit_ks = np.where(row > THRESHOLD)[0]
            if len(hit_ks) > 0:
                caseA_genes.add(g)
                for k in hit_ks:
                    caseA_isoforms.append((g, te_isoid[i], GO_KEYS[k], float(row[k])))
                    caseA_term_counts[GO_KEYS[k]] += 1
    else:
        neg_k = [k for k in range(N_GO) if k not in pos_k]
        if not neg_k:
            continue
        for i in idxs:
            row = score_matrix[i]
            own_best = row[list(pos_k)].max()
            for k in neg_k:
                if row[k] > THRESHOLD and row[k] > own_best:
                    caseB_genes.add(g)
                    caseB_isoforms.append((g, te_isoid[i], GO_KEYS[k], float(row[k]), float(own_best)))
                    caseB_term_counts[GO_KEYS[k]] += 1

print("\n" + "=" * 70)
print("  CASE A — fully unannotated gene, isoform newly crosses threshold")
print("=" * 70)
print(f"  Genes affected: {len(caseA_genes)} / {n_genes_fully_negative} fully-negative genes "
      f"({100*len(caseA_genes)/max(n_genes_fully_negative,1):.2f}%)")
print(f"  (isoform, term) hits: {len(caseA_isoforms)}")
print(f"  Distinct isoforms involved: {len(set(x[1] for x in caseA_isoforms))}")
print("  Top terms acquired:")
for k, c in sorted(caseA_term_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {k} ({GO_TERMS[k]}): {c}")

print("\n" + "=" * 70)
print("  CASE B — partially-annotated gene, isoform's foreign-term score")
print("  exceeds both threshold AND its own gene-annotated best score")
print("=" * 70)
print(f"  Genes affected: {len(caseB_genes)} / {n_genes_partial} partially-annotated genes "
      f"({100*len(caseB_genes)/max(n_genes_partial,1):.2f}%)")
print(f"  (isoform, term) hits: {len(caseB_isoforms)}")
print(f"  Distinct isoforms involved: {len(set(x[1] for x in caseB_isoforms))}")
print("  Top terms acquired:")
for k, c in sorted(caseB_term_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"    {k} ({GO_TERMS[k]}): {c}")

# ── example cases for spot-check (top by score / by gap) ────────────────────
caseA_sorted = sorted(caseA_isoforms, key=lambda x: -x[3])[:15]
caseB_sorted = sorted(caseB_isoforms, key=lambda x: -(x[3]-x[4]))[:15]

print("\n  Case A top-15 examples (gene, isoform, term, score):")
for row in caseA_sorted:
    print(f"    {row}")
print("\n  Case B top-15 examples by gap (gene, isoform, foreign_term, foreign_score, own_best):")
for row in caseB_sorted:
    print(f"    {row}")

result = {
    'threshold': THRESHOLD,
    'n_isoforms': N_TE,
    'n_go_terms': N_GO,
    'n_genes_total': n_genes_total,
    'n_genes_fully_negative': n_genes_fully_negative,
    'n_genes_partial': n_genes_partial,
    'caseA': {
        'genes_affected': len(caseA_genes),
        'genes_affected_pct_of_fully_negative': 100*len(caseA_genes)/max(n_genes_fully_negative,1),
        'isoform_term_hits': len(caseA_isoforms),
        'distinct_isoforms': len(set(x[1] for x in caseA_isoforms)),
        'term_counts': dict(caseA_term_counts),
        'top15_examples': caseA_sorted,
    },
    'caseB': {
        'genes_affected': len(caseB_genes),
        'genes_affected_pct_of_partial': 100*len(caseB_genes)/max(n_genes_partial,1),
        'isoform_term_hits': len(caseB_isoforms),
        'distinct_isoforms': len(set(x[1] for x in caseB_isoforms)),
        'term_counts': dict(caseB_term_counts),
        'top15_examples': caseB_sorted,
    },
}
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n  [Saved] {OUT_DIR}/results.json")
