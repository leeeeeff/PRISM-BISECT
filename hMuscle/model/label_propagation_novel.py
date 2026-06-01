#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_propagation_novel.py
===========================
Novel isoform OOD 정량화 실험 — 두 분석 축:

[1] KNN Distance Analysis
    novel isoform이 ESM-2 공간에서 known isoform과 실제로 얼마나 먼가?
    → OOD 심각도를 코사인 거리로 정량화

[2] KNN Score Propagation
    novel isoform의 DIFFUSE 스코어를 이웃 known isoform 스코어로 대체 시
    AUPRC가 개선되는가?
    → KNN AUPRC vs DIFFUSE AUPRC per GO term

결과 해석:
  KNN_AUPRC > DIFFUSE_AUPRC (+0.03+) : KNN이 모델보다 우월 → OOD 해결 가능성
  KNN_AUPRC ≈ DIFFUSE_AUPRC          : KNN 불도움 → 순수 OOD, representation 문제
  DIFFUSE_AUPRC > KNN_AUPRC          : 모델이 implicit KNN 이미 수행 → 강건성 확인

conda run -n isoform_env python label_propagation_novel.py
"""

import os, json, time
import numpy as np
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BRAIN_DIR  = '../data/brain_isoquant_esm2/full'
ANNOT_DIR  = '../data/raw_data/data/annotations'
SCORE_FILE = '../../reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy'
OUT_DIR    = '../../reports/label_propagation_novel'
os.makedirs(OUT_DIR, exist_ok=True)

K_VALS = [5, 10, 20]  # KNN candidates

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere org',
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
    'GO:0031175': 'Neuron proj dev',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS  = list(GO_TERMS.keys())
GO_NAMES = list(GO_TERMS.values())
N_GO     = len(GO_KEYS)

print("=" * 70)
print("  Label Propagation Experiment — Novel Brain Isoforms OOD Analysis")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────────
def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

X_te     = np.load(f'{BRAIN_DIR}/brain_full_esm2_t30_150M.npy').astype(np.float32)
te_isoid = load_ids(f'{BRAIN_DIR}/brain_full_ids.npy')
te_sym   = load_ids(f'{BRAIN_DIR}/brain_full_gene_names.npy')

is_novel  = np.array(['transcript' in x for x in te_isoid], dtype=bool)
is_known  = ~is_novel

N_TE     = len(te_isoid)
N_NOVEL  = int(is_novel.sum())
N_KNOWN  = int(is_known.sum())
print(f"  Total: {N_TE} | Known: {N_KNOWN} | Novel: {N_NOVEL}")

# Pre-computed DIFFUSE scores (63994 × 18)
score_matrix = np.load(SCORE_FILE).astype(np.float32)
print(f"  Score matrix: {score_matrix.shape}")

# ── Load GO labels ──────────────────────────────────────────────────────────────
def load_labels_te(go_term, te_sym):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)

print("\n  Loading GO labels...")
Y_te = np.zeros((N_TE, N_GO), dtype=np.float32)
for gi, go_term in enumerate(GO_KEYS):
    Y_te[:, gi] = load_labels_te(go_term, te_sym)
print(f"  Labels loaded. Novel positives per term:")
for gi, name in enumerate(GO_NAMES):
    n_pos_all   = int(Y_te[:, gi].sum())
    n_pos_novel = int(Y_te[is_novel, gi].sum())
    if n_pos_novel > 0:
        print(f"    {name[:22]:22s}: total={n_pos_all:4d}, novel={n_pos_novel:3d}")

# ── [1] KNN Distance Analysis ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  [1] KNN Distance Analysis (novel → known, cosine)")
print("=" * 70)

X_novel = X_te[is_novel]
X_known = X_te[is_known]

# L2-normalise for cosine similarity
def l2_norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)

X_novel_n = l2_norm(X_novel)
X_known_n = l2_norm(X_known)

# compute top-K cosine distances in batches to manage memory
BATCH = 256
max_K = max(K_VALS)

print(f"  Computing top-{max_K} KNN for {N_NOVEL} novel isoforms (batched)...")
t0 = time.time()

# Store: top-K indices (into is_known subset) and distances
knn_idx_known   = np.zeros((N_NOVEL, max_K), dtype=np.int32)
knn_cos_sim     = np.zeros((N_NOVEL, max_K), dtype=np.float32)

for start in range(0, N_NOVEL, BATCH):
    end   = min(start + BATCH, N_NOVEL)
    batch = X_novel_n[start:end]                        # (batch, 640)
    sims  = batch @ X_known_n.T                         # (batch, N_KNOWN)
    topk  = np.argpartition(sims, -max_K, axis=1)[:, -max_K:]
    for bi in range(end - start):
        row = topk[bi]
        sorted_row = row[np.argsort(-sims[bi, row])]
        knn_idx_known[start + bi] = sorted_row
        knn_cos_sim[start + bi]   = sims[bi, sorted_row]

print(f"  KNN computation done: {time.time()-t0:.1f}s")

# Distance statistics
top1_cos_dist = 1.0 - knn_cos_sim[:, 0]
top5_cos_dist = 1.0 - knn_cos_sim[:, :5].mean(axis=1)

dist_stats = {
    'nearest_neighbor': {
        'mean_cosine_dist':   float(top1_cos_dist.mean()),
        'median_cosine_dist': float(np.median(top1_cos_dist)),
        'p25_cosine_dist':    float(np.percentile(top1_cos_dist, 25)),
        'p75_cosine_dist':    float(np.percentile(top1_cos_dist, 75)),
        'p90_cosine_dist':    float(np.percentile(top1_cos_dist, 90)),
        'pct_very_close':     float((top1_cos_dist < 0.05).mean()),
        'pct_close':          float((top1_cos_dist < 0.10).mean()),
        'pct_far':            float((top1_cos_dist > 0.30).mean()),
    },
    'top5_mean': {
        'mean_cosine_dist':   float(top5_cos_dist.mean()),
        'median_cosine_dist': float(np.median(top5_cos_dist)),
    }
}

print(f"\n  === Distance Statistics (novel → nearest known) ===")
print(f"  Mean cosine dist (top-1):   {dist_stats['nearest_neighbor']['mean_cosine_dist']:.4f}")
print(f"  Median cosine dist (top-1): {dist_stats['nearest_neighbor']['median_cosine_dist']:.4f}")
print(f"  % very close (dist<0.05):   {dist_stats['nearest_neighbor']['pct_very_close']*100:.1f}%")
print(f"  % close (dist<0.10):        {dist_stats['nearest_neighbor']['pct_close']*100:.1f}%")
print(f"  % far (dist>0.30):          {dist_stats['nearest_neighbor']['pct_far']*100:.1f}%")

# ── [2] KNN Score Propagation ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  [2] KNN Score Propagation — AUPRC comparison per GO term")
print("=" * 70)

# known isoform indices in full array
known_indices = np.where(is_known)[0]

# Precompute: per novel isoform, KNN indices in KNOWN → map back to global indices
# knn_idx_known[i, k] is the index in X_known → global: known_indices[knn_idx_known[i,k]]

results_per_go = {}
summary = {}

for gi, (go_term, go_name) in enumerate(GO_TERMS.items()):
    y_novel = Y_te[is_novel, gi]
    if y_novel.sum() == 0:
        continue

    # Baseline: DIFFUSE scores for novel isoforms
    diffuse_novel = score_matrix[is_novel, gi]
    auprc_diffuse = average_precision_score(y_novel, diffuse_novel)

    # KNN propagation for different K values
    knn_auprcs = {}
    for k in K_VALS:
        # Soft score = similarity-weighted average of known isoform DIFFUSE scores
        neighbor_scores  = score_matrix[known_indices[knn_idx_known[:, :k]], gi]  # (N_NOVEL, k)
        neighbor_weights = knn_cos_sim[:, :k]  # cosine similarity as weights
        neighbor_weights = np.clip(neighbor_weights, 0, None)  # clip negative sims
        weight_sum = neighbor_weights.sum(axis=1, keepdims=True)
        weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
        knn_scores = (neighbor_scores * neighbor_weights).sum(axis=1) / weight_sum.squeeze()

        auprc_knn = average_precision_score(y_novel, knn_scores)
        knn_auprcs[k] = float(auprc_knn)

    best_k   = max(K_VALS, key=lambda k: knn_auprcs[k])
    best_knn = knn_auprcs[best_k]
    delta    = best_knn - auprc_diffuse
    n_pos    = int(y_novel.sum())

    results_per_go[go_term] = {
        'go_name':      go_name,
        'n_pos_novel':  n_pos,
        'auprc_diffuse': float(auprc_diffuse),
        'auprc_knn':    {k: float(v) for k, v in knn_auprcs.items()},
        'best_k':       best_k,
        'best_knn':     float(best_knn),
        'delta_knn_minus_diffuse': float(delta),
    }

    symbol = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "≈")
    print(f"  {go_name[:22]:22s}  DIFFUSE={auprc_diffuse:.4f}  "
          f"KNN(k={best_k})={best_knn:.4f}  Δ={delta:+.4f} {symbol}  n_pos={n_pos}")

# Macro averages
valid = [v for v in results_per_go.values() if v['n_pos_novel'] > 0]
macro_diffuse = np.mean([v['auprc_diffuse'] for v in valid])
macro_knn     = np.mean([v['best_knn']      for v in valid])

print(f"\n  === Macro AUPRC (novel isoforms, {len(valid)} terms with n_pos>0) ===")
print(f"  DIFFUSE (baseline):    {macro_diffuse:.4f}")
print(f"  KNN best-K propagation:{macro_knn:.4f}")
print(f"  Δ (KNN - DIFFUSE):     {macro_knn - macro_diffuse:+.4f}")

# ── Interpretation ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Interpretation")
print("=" * 70)
delta_macro = macro_knn - macro_diffuse
if delta_macro > 0.05:
    interp = "KNN propagation significantly outperforms DIFFUSE for novel isoforms (+{:.3f}). OOD can be partially addressed by proximity-based score transfer.".format(delta_macro)
elif delta_macro > 0.02:
    interp = "KNN propagation provides modest improvement (+{:.3f}). Some OOD mitigation possible for 'mildly novel' isoforms.".format(delta_macro)
elif delta_macro > -0.02:
    interp = "KNN propagation provides no benefit (Δ≈0). Novel isoform OOD is NOT addressable by ESM-2 proximity — confirms fundamental sequence-level OOD.".format(delta_macro)
else:
    interp = "KNN propagation degrades performance ({:.3f}). Model already implicitly smooths better than explicit KNN.".format(delta_macro)
print(f"  {interp}")

# ── Save results ────────────────────────────────────────────────────────────────
import datetime
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
out = {
    'timestamp': ts,
    'n_novel': N_NOVEL,
    'n_known': N_KNOWN,
    'k_vals': K_VALS,
    'distance_analysis': dist_stats,
    'macro_auprc_diffuse': float(macro_diffuse),
    'macro_auprc_knn_best': float(macro_knn),
    'macro_delta': float(delta_macro),
    'interpretation': interp,
    'per_go_term': results_per_go,
}
out_path = f'{OUT_DIR}/knn_propagation_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  [Saved] {out_path}")
print("=" * 70)
