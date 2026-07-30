#!/usr/bin/env python3
"""
reranking_v15d.py
==================
전략 C: v15d 예측값 + splice_delta 기반 post-hoc reranking.

방식: score_adj[i] = score_v15d[i] + λ * (SD_norm[i] - mean_SD_norm[gene(i)])
  - 유전자 내에서 splice_delta L2 norm이 평균보다 큰 이소폼을 상향 조정
  - λ=0.02: 이전 실험에서 안전성 검증됨 (Session 2026-06-07)

의미: splice_delta가 큰 이소폼 = 유전자 평균 splicing 패턴과 더 많이 다름
     → 기능 차이 가능성이 높음 → GO 예측값 상향 조정이 타당

λ sweep: 0.0, 0.005, 0.01, 0.02, 0.05, 0.10 테스트
"""

import json, os
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT    = os.path.join(BASE, 'results_isoform/features')
MODEL   = os.path.join(BASE, 'model')
ANNO    = os.path.join(BASE, 'data/raw_data/data')
REPORT  = os.path.join(BASE, '../reports/reranking')
os.makedirs(REPORT, exist_ok=True)

SCORE_PATH = os.path.join(BASE, '../reports/v15_bp_clean/score_matrix_18go_20260519_1914.npy')
SD_PATH    = os.path.join(FEAT, 'splicing/splicing_delta_v2.npy')
ANNOT_DIR  = os.path.join(BASE, 'data/raw_data/data/annotations')
ID_DIR     = os.path.join(BASE, 'data/raw_data/data/id_lists')

LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]

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

BASELINE_AUPRC = 0.7022


def load_ids(path):
    arr = np.load(path, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


print("[1] Loading data...")
S_base  = np.load(SCORE_PATH).astype(np.float32)
X_sd    = np.load(SD_PATH).astype(np.float32)
te_gene = load_ids(os.path.join(MODEL, 'my_gene_list_fixed.npy'))
te_iso  = load_ids(os.path.join(MODEL, 'my_isoform_list_fixed.npy'))
print(f"  Scores: {S_base.shape}, SD: {X_sd.shape}")

# Symbol mapping (ENSG → gene symbol)
print("[2] Loading gene symbols and GO labels...")
ENSG2SYM = {}
with open(os.path.join(ID_DIR, 'ensembl_to_symbol.txt')) as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]

# Load all GO labels at once
N = len(te_sym)
N_GO = len(GO_TERMS)
Y = np.zeros((N, N_GO), dtype=np.float32)
for gi, go_term in enumerate(GO_TERMS):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    Y[:, gi] = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
print(f"  Labels: {Y.shape}, pos/GO avg={Y.sum(0).mean():.1f}")

# Compute SD L2 norm per isoform
SD_norm = np.linalg.norm(X_sd, axis=1)  # (N,)
print(f"  SD norm: mean={SD_norm.mean():.4f}  std={SD_norm.std():.4f}")

# Gene-mean SD norm
gene_to_idx = defaultdict(list)
for i, g in enumerate(te_gene):
    gene_to_idx[g].append(i)

gene_mean_SD_norm = np.zeros(len(te_gene), dtype=np.float32)
for g, idxs in gene_to_idx.items():
    gm = SD_norm[idxs].mean()
    for idx in idxs:
        gene_mean_SD_norm[idx] = gm

SD_delta = SD_norm - gene_mean_SD_norm  # centered per gene


def evaluate(S, Y):
    auprcs = []
    for j in range(Y.shape[1]):
        y_true = Y[:, j]
        y_pred = S[:, j]
        if y_true.sum() < 1:
            continue
        auprcs.append(average_precision_score(y_true, y_pred))
    return np.mean(auprcs)


print("\n[3] Lambda sweep...")
print(f"{'λ':>8}  {'Macro AUPRC':>12}  {'Δ vs baseline':>14}  {'vs λ=0':>10}")
print("-" * 55)

results = {}
auprc_at_zero = None

for lam in LAMBDA_VALUES:
    S_adj = S_base + lam * SD_delta[:, None]
    macro = evaluate(S_adj, Y)
    delta_base = macro - BASELINE_AUPRC
    if lam == 0.0:
        auprc_at_zero = macro
    delta_zero = macro - auprc_at_zero
    marker = " *" if lam == 0.02 else ""
    print(f"  {lam:>6.3f}  {macro:>12.4f}  {delta_base:>+14.4f}  {delta_zero:>+10.4f}{marker}")
    results[lam] = float(macro)

# Per-GO-term breakdown at λ=0.02
print("\n[4] Per-term AUPRC at λ=0.02:")
S_adj = S_base + 0.02 * SD_delta[:, None]
go_names = list(GO_TERMS.values())
print(f"{'GO term':<35}  {'λ=0':>8}  {'λ=0.02':>8}  {'Δ':>8}")
for j, name in enumerate(go_names):
    y_true = Y[:, j]
    if y_true.sum() < 1:
        continue
    a0  = average_precision_score(y_true, S_base[:, j])
    a02 = average_precision_score(y_true, S_adj[:, j])
    print(f"  {name:<33}  {a0:>8.4f}  {a02:>8.4f}  {a02-a0:>+8.4f}")

# DIFF_SPLICE subset
print("\n[5] DIFF_SPLICE subset (isoforms with distinct splice_delta within gene):")
ds_idx = []
for g, idxs in gene_to_idx.items():
    if len(idxs) < 2:
        continue
    sd_rows = X_sd[idxs]
    if not np.allclose(sd_rows, sd_rows[0]):
        ds_idx.extend(idxs)

print(f"  DIFF_SPLICE isoforms: {len(ds_idx)}")
if ds_idx:
    Y_ds   = Y[ds_idx]
    S0_ds  = S_base[ds_idx]
    Sa_ds  = S_adj[ds_idx]
    m0  = evaluate(S0_ds,  Y_ds)
    m02 = evaluate(Sa_ds,  Y_ds)
    print(f"  λ=0:    {m0:.4f}")
    print(f"  λ=0.02: {m02:.4f}  (Δ={m02-m0:+.4f})")

# Save
out_path = os.path.join(REPORT, 'reranking_lambda_sweep.json')
with open(out_path, 'w') as f:
    json.dump({'lambda_sweep': results, 'baseline': BASELINE_AUPRC}, f, indent=2)
print(f"\nSaved: {out_path}")
