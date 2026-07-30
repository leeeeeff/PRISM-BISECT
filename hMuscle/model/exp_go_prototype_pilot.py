"""
exp_go_prototype_pilot.py
==========================
Phase 1: GO prototype centroid 파일럿

질문: GO-positive isoform들의 embedding centroid가
      gene-level annotation 없이 새 isoform의 GO membership을
      cosine distance만으로 예측할 수 있는가?

설계:
  - Gene-level 2-fold split (leakage 없음)
  - Fold A genes → GO centroid (prototype)
  - Fold B genes → AUPRC 평가
  - Feature: L30 / δ_layer(L30-L15) / concatenated

비교:
  - PRISM v17f*:   0.734
  - k-NN:          0.636
  - gene-mean:     0.803 (label ceiling)
  - domain-LR:     0.163
"""

import numpy as np
import json, os
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(BASE, '../reports/exp_go_prototype')
os.makedirs(REPORT, exist_ok=True)

# ── 1. Data loading ─────────────────────────────────────────────────────
print("[1] Loading data...")

X_l30   = np.load(os.path.join(BASE, 'data/esm2_layer_30_t30_150M.npy')).astype(np.float32)
X_l15   = np.load(os.path.join(BASE, 'data/esm2_layer_15_t30_150M.npy')).astype(np.float32)
X_delta = X_l30 - X_l15
X_cat   = np.concatenate([X_l30, X_delta], axis=1)

Y_all = np.load(os.path.join(BASE, '../reports/v17f_star_bootstrap/Y_te.npy')).astype(np.float32)
prism_preds = np.load(os.path.join(BASE, '../reports/v17f_star_bootstrap/v17f_star_preds.npy')).astype(np.float32)

valid_mask = Y_all.sum(0) >= 2
Y_v        = Y_all[:, valid_mask]
prism_v    = prism_preds[:, valid_mask]
n_valid    = valid_mask.sum()
print(f"  36748 isoforms, {n_valid} valid GO terms")
print(f"  L30: {X_l30.shape}, delta: {X_delta.shape}")

# ── 2. Gene-level split ─────────────────────────────────────────────────
print("\n[2] Gene-level 2-fold split...")

te_gene_raw = np.load(os.path.join(BASE, 'model/my_gene_list_fixed.npy'), allow_pickle=True)
te_gene = [g.decode() if isinstance(g, bytes) else str(g) for g in te_gene_raw]

gene2idxs = defaultdict(list)
for i, g in enumerate(te_gene):
    gene2idxs[g.split('.')[0]].append(i)

all_genes = list(gene2idxs.keys())
n_genes   = len(all_genes)
np.random.seed(42)
perm    = np.random.permutation(n_genes)
n_half  = n_genes // 2
genes_a = [all_genes[perm[i]] for i in range(n_half)]
genes_b = [all_genes[perm[i]] for i in range(n_half, n_genes)]

idx_a = np.array([i for g in genes_a for i in gene2idxs[g]])
idx_b = np.array([i for g in genes_b for i in gene2idxs[g]])
print(f"  Genes A: {len(genes_a)} ({len(idx_a)} isoforms)  |  B: {len(genes_b)} ({len(idx_b)} isoforms)")

# ── 3. GO centroid AUPRC ────────────────────────────────────────────────
def go_centroid_auprc(X_feat, Y_v, idx_proto, idx_eval):
    X_proto = X_feat[idx_proto]
    Y_proto = Y_v[idx_proto]
    X_eval  = X_feat[idx_eval]
    Y_eval  = Y_v[idx_eval]

    n_k = Y_v.shape[1]
    prototypes = np.zeros((n_k, X_feat.shape[1]), dtype=np.float32)
    n_pos_list = []
    for k in range(n_k):
        pos = Y_proto[:, k] == 1
        n_pos_list.append(pos.sum())
        if pos.sum() > 0:
            prototypes[k] = X_proto[pos].mean(0)

    sim = cosine_similarity(X_eval, prototypes)  # (n_eval, n_k)

    auprc_list = []
    for k in range(n_k):
        if Y_eval[:, k].sum() >= 2:
            auprc_list.append(average_precision_score(Y_eval[:, k], sim[:, k]))

    return np.mean(auprc_list), len(auprc_list), np.mean(n_pos_list)

print("\n[3] Computing GO centroid AUPRC (gene-level 2-fold)...")
print(f"{'Feature':<20} {'A→B':>8} {'B→A':>8} {'mean':>8}")
print("-" * 50)

results = {}
for feat_name, X_feat in [
    ('L30',          X_l30),
    ('delta(L30-15)', X_delta),
    ('L30+delta',    X_cat),
]:
    f1, n1, np1 = go_centroid_auprc(X_feat, Y_v, idx_a, idx_b)
    f2, n2, np2 = go_centroid_auprc(X_feat, Y_v, idx_b, idx_a)
    mean_auprc  = (f1 + f2) / 2
    print(f"{feat_name:<20} {f1:>8.4f} {f2:>8.4f} {mean_auprc:>8.4f}")
    results[feat_name] = {'fold_ab': float(f1), 'fold_ba': float(f2), 'mean': float(mean_auprc)}

# ── 4. PRISM v17f* baseline on same split (reference) ──────────────────
print("\n[4] PRISM v17f* reference on same gene-level splits...")
for fold_name, proto_idx, eval_idx in [('A→B', idx_a, idx_b), ('B→A', idx_b, idx_a)]:
    Y_eval = Y_v[eval_idx]
    P_eval = prism_v[eval_idx]
    auprc_list = [average_precision_score(Y_eval[:, k], P_eval[:, k])
                  for k in range(n_valid) if Y_eval[:, k].sum() >= 2]
    print(f"  PRISM {fold_name}: {np.mean(auprc_list):.4f}")

# ── 5. Summary ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SUMMARY")
print("="*55)
print(f"  PRISM v17f*  (full eval):  0.7340")
print(f"  k-NN         (full eval):  0.6360")
print(f"  gene-mean    (full eval):  0.8030  [label ceiling]")
print(f"  domain-LR    (full eval):  0.1625")
print()
for feat, r in results.items():
    print(f"  GO centroid ({feat}):  {r['mean']:.4f}")
print()
print("  Interpretation:")
print("  > k-NN 0.636  → prototype > retrieval, Phase 2 justified")
print("  ≈ k-NN 0.636  → centroid = k-NN variant, limited novelty")
print("  < k-NN 0.636  → embedding space not GO-aligned")
print("="*55)

out = {
    'go_centroid': results,
    'baselines': {
        'prism_v17f_star': 0.734,
        'knn':             0.636,
        'gene_mean':       0.803,
        'domain_lr':       0.163,
    },
    'note': 'Phase1: label-weighted centroid. Phase2: SwissProt canonical prototype (annotation-free).'
}
with open(os.path.join(REPORT, 'phase1_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {REPORT}/phase1_results.json")
