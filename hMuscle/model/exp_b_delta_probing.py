#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_b_delta_probing.py
----------------------
Mechanistic probing: what does δ_layer = φ_L30 - φ_L15 encode?

Hypothesis: δ_layer encodes domain-composition-level splice differences.
Splicing events that alter Pfam domain boundaries create larger δ_layer
signals in dimensions correlated with those domain features.

Analysis:
  1. Domain predictive power: LR on δ_layer vs φ_L30 vs φ_L15
     → Does δ_layer encode domain presence better than L30 alone?
  2. Within-gene isoform pair analysis
     → Δδ_layer ↔ Δdomain_vec correlation
  3. L2_Structural GO membership probing
     → Which δ_layer dims drive L2_Structural AUPRC?
  4. Per-dimension domain correlation map (top-20 dims)
"""

import os, json, gzip
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from scipy.stats import spearmanr, pearsonr
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_b_probing'
os.makedirs(OUT_DIR, exist_ok=True)

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  Experiment B: δ_layer Mechanistic Probing")
print("=" * 65)

# ── Load embeddings ──────────────────────────────────────────────
print("\n[1] Loading embeddings...")
X_l30 = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)
X_l15 = np.load(f'{DATA_DIR}/esm2_layer_15_t30_150M.npy').astype(np.float32)
X_l01 = np.load(f'{DATA_DIR}/esm2_layer_01_t30_150M.npy').astype(np.float32)
delta  = (X_l30 - X_l15).astype(np.float32)
N, D   = delta.shape
print(f"  δ_layer shape: {delta.shape}  norm mean: {np.linalg.norm(delta, axis=1).mean():.2f}")

# Scale
scaler = MaxAbsScaler()
delta_s = scaler.fit_transform(delta).astype(np.float32)

# ── Load domain matrix ───────────────────────────────────────────
print("\n[2] Loading domain matrix...")
domain_mat = np.load('../results_isoform/features/domain_matrix_proper_test.npy').astype(np.float32)
n_domains = domain_mat.shape[1]
n_with_domain = (domain_mat.sum(axis=1) > 0).sum()
print(f"  domain_matrix: {domain_mat.shape}  isoforms_with_domains: {n_with_domain}/{N} ({n_with_domain/N:.1%})")
print(f"  Pfam slots: {n_domains}  active domains: {domain_mat.sum(axis=0).astype(bool).sum()}")

# ── Load gene mappings ───────────────────────────────────────────
print("\n[3] Loading gene mappings...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
multi_gene_idxs = [idxs for g, idxs in gene2idxs.items() if len(idxs) > 1]
print(f"  Multi-isoform genes: {len(multi_gene_idxs)}")

# ── Load GO H2 layer classification ─────────────────────────────
print("\n[4] Loading H2 layer classification and GO labels...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]
go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])

H2_LAYER = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: H2_LAYER[p[0]] = p[11]

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2

l2_terms  = [go for go in mf_terms if H2_LAYER.get(go) == 'L2_Structural']
l4_terms  = [go for go in mf_terms if H2_LAYER.get(go) == 'L4_CellState']
l1h_terms = [go for go in mf_terms if H2_LAYER.get(go) == 'L1_Generic_high']
l2_idx    = [i for i, go in enumerate(mf_terms) if go in set(l2_terms) and valid_mask[i]]
l4_idx    = [i for i, go in enumerate(mf_terms) if go in set(l4_terms)  and valid_mask[i]]
print(f"  L2_Structural terms: {len(l2_idx)}  L4_CellState: {len(l4_idx)}")

# ── Analysis 1: Domain predictive power ─────────────────────────
print("\n[Analysis 1] Domain presence prediction: δ_layer vs φ_L30 vs φ_L15")
print("  Using logistic regression, 5-fold CV on test set...")

# Select domains with sufficient coverage (≥50 positives, ≤10% positive rate → 3678)
domain_counts = domain_mat.sum(axis=0)
active_domains = np.where((domain_counts >= 30) & (domain_counts <= N * 0.8))[0]
print(f"  Active domains (30≤n≤{int(N*0.8)}): {len(active_domains)}")

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

features = {
    'delta_layer': delta_s,
    'L30':         StandardScaler().fit_transform(X_l30),
    'L15':         StandardScaler().fit_transform(X_l15),
    'L30_minus_L15_raw': StandardScaler().fit_transform(delta),
}

domain_results = {fname: [] for fname in features}
n_eval_domains = min(100, len(active_domains))  # evaluate 100 domains for speed
eval_domains = active_domains[:n_eval_domains]

print(f"  Evaluating {n_eval_domains} domains...")
for d_idx in eval_domains:
    y_dom = domain_mat[:, d_idx]
    if y_dom.sum() < 10: continue
    for fname, X in features.items():
        aucs = []
        for tr_i, te_i in skf.split(X, y_dom.astype(int)):
            clf = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
            clf.fit(X[tr_i], y_dom[tr_i])
            prob = clf.predict_proba(X[te_i])[:, 1]
            if y_dom[te_i].sum() >= 2:
                aucs.append(roc_auc_score(y_dom[te_i], prob))
        if aucs: domain_results[fname].append(np.mean(aucs))

print(f"\n  Domain prediction AUROC (mean over {n_eval_domains} domains):")
for fname, aucs in domain_results.items():
    print(f"    {fname:<30} {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  (n={len(aucs)})")

domain_summary = {fname: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
                  for fname, v in domain_results.items()}

# ── Analysis 2: Within-gene pair analysis ────────────────────────
print("\n[Analysis 2] Within-gene isoform pair: Δδ_layer ↔ Δdomain correlation")

pair_delta_diffs = []
pair_domain_diffs = []
pair_l30_diffs = []
pair_l2_go_diffs = []  # mean L2_Struct GO score difference

rng = np.random.default_rng(42)
for gene_idxs in multi_gene_idxs:
    for i in range(len(gene_idxs)):
        for j in range(i+1, len(gene_idxs)):
            a, b = gene_idxs[i], gene_idxs[j]
            # L2 norm of δ_layer difference
            pair_delta_diffs.append(np.linalg.norm(delta_s[a] - delta_s[b]))
            # L1 norm of domain vector difference (Hamming distance)
            pair_domain_diffs.append(np.sum(np.abs(domain_mat[a] - domain_mat[b])))
            # L2 norm of L30 difference
            pair_l30_diffs.append(np.linalg.norm(X_l30[a] - X_l30[b]))

pair_delta_diffs  = np.array(pair_delta_diffs)
pair_domain_diffs = np.array(pair_domain_diffs)
pair_l30_diffs    = np.array(pair_l30_diffs)

rho_delta_domain, p_delta_domain = spearmanr(pair_delta_diffs, pair_domain_diffs)
rho_l30_domain,   p_l30_domain   = spearmanr(pair_l30_diffs,   pair_domain_diffs)
rho_delta_l30,    p_delta_l30    = spearmanr(pair_delta_diffs,  pair_l30_diffs)

print(f"  N pairs: {len(pair_delta_diffs)}")
print(f"  Spearman ρ(Δδ_layer, Δdomain):  {rho_delta_domain:.4f}  p={p_delta_domain:.2e}")
print(f"  Spearman ρ(ΔL30, Δdomain):      {rho_l30_domain:.4f}    p={p_l30_domain:.2e}")
print(f"  Spearman ρ(Δδ_layer, ΔL30):     {rho_delta_l30:.4f}     p={p_delta_l30:.2e}")

# Also compute for pairs WITH domain difference (structural splice events)
structural_pair_mask = pair_domain_diffs > 0
n_struct = structural_pair_mask.sum()
if n_struct > 100:
    rho_struct, p_struct = spearmanr(
        pair_delta_diffs[structural_pair_mask],
        pair_domain_diffs[structural_pair_mask]
    )
    print(f"  ρ(Δδ_layer, Δdomain) [domain-diff pairs only, n={n_struct}]: {rho_struct:.4f}  p={p_struct:.2e}")
else:
    rho_struct, p_struct = float('nan'), float('nan')

pair_analysis = {
    'n_pairs': int(len(pair_delta_diffs)),
    'n_domain_diff_pairs': int(n_struct),
    'rho_delta_domain': float(rho_delta_domain), 'p_delta_domain': float(p_delta_domain),
    'rho_l30_domain':   float(rho_l30_domain),   'p_l30_domain':   float(p_l30_domain),
    'rho_delta_l30':    float(rho_delta_l30),     'p_delta_l30':    float(p_delta_l30),
    'rho_struct_only':  float(rho_struct),
}

# ── Analysis 3: δ_layer dimension → L2_Struct GO membership ─────
print("\n[Analysis 3] Per-dimension correlation: δ_layer dims → L2_Struct GO membership")

# Aggregate L2_Structural GO membership as binary vector
y_l2_any = (Y_te[:, l2_idx].sum(axis=1) > 0).astype(float)
y_l4_any = (Y_te[:, l4_idx].sum(axis=1) > 0).astype(float) if l4_idx else np.zeros(N)
print(f"  L2_Struct positive isoforms: {y_l2_any.sum():.0f}/{N}")
print(f"  L4_CellState positive isoforms: {y_l4_any.sum():.0f}/{N}")

# Per-dimension correlation with L2_Struct membership
dim_corr_l2 = np.array([pearsonr(delta_s[:, k], y_l2_any)[0] for k in range(D)])
dim_corr_l4 = np.array([pearsonr(delta_s[:, k], y_l4_any)[0] for k in range(D)])
# Same for L30
dim_corr_l30_l2 = np.array([pearsonr(X_l30[:, k], y_l2_any)[0] for k in range(D)])

# Top-20 most predictive δ dims for L2_Struct
top20_l2_idx = np.argsort(np.abs(dim_corr_l2))[-20:][::-1]
top20_l4_idx = np.argsort(np.abs(dim_corr_l4))[-20:][::-1]

print(f"\n  Top-10 δ_layer dims for L2_Structural:")
for rank, k in enumerate(top20_l2_idx[:10]):
    print(f"    dim {k:3d}: r={dim_corr_l2[k]:+.4f}  L30 r={dim_corr_l30_l2[k]:+.4f}")

# Are L2 top dims different from L4 top dims? (overlap analysis)
overlap_l2_l4 = len(set(top20_l2_idx) & set(top20_l4_idx))
print(f"\n  Overlap top-20 L2 vs L4 dims: {overlap_l2_l4}/20")

# For top-20 L2 dims: what domain features do they correlate with?
top_dim_domain_corr = []
for k in top20_l2_idx[:5]:
    best_domains = []
    for d_idx in active_domains[:200]:
        r, _ = pearsonr(delta_s[:, k], domain_mat[:, d_idx])
        best_domains.append((d_idx, r))
    best_domains.sort(key=lambda x: abs(x[1]), reverse=True)
    top_dim_domain_corr.append({
        'dim': int(k),
        'l2_corr': float(dim_corr_l2[k]),
        'top_domains': [(int(d), float(r)) for d, r in best_domains[:5]],
    })
    print(f"  dim {k}: top domain corrs = {[(d, f'{r:.3f}') for d, r in best_domains[:3]]}")

# ── Analysis 4: δ norm vs protein length and domain count ────────
print("\n[Analysis 4] δ_layer norm vs structural properties")
delta_norms = np.linalg.norm(delta_s, axis=1)
n_domains_per_iso = domain_mat.sum(axis=1)
rho_norm_dom, p_norm_dom = spearmanr(delta_norms, n_domains_per_iso)
print(f"  Spearman ρ(δ_norm, n_domains): {rho_norm_dom:.4f}  p={p_norm_dom:.2e}")
print(f"  Mean δ_norm | no domain: {delta_norms[n_domains_per_iso==0].mean():.3f}")
print(f"  Mean δ_norm | ≥1 domain: {delta_norms[n_domains_per_iso>0].mean():.3f}")
print(f"  Mean δ_norm | ≥3 domains: {delta_norms[n_domains_per_iso>=3].mean():.3f}")

# ── Save results ─────────────────────────────────────────────────
results = {
    'domain_predictive_power': domain_summary,
    'within_gene_pair_analysis': pair_analysis,
    'l2_go_probing': {
        'n_l2_positive': int(y_l2_any.sum()),
        'top20_l2_dims': [int(k) for k in top20_l2_idx],
        'top20_l4_dims': [int(k) for k in top20_l4_idx],
        'overlap_l2_l4': int(overlap_l2_l4),
        'top_dim_domain_corr': top_dim_domain_corr,
        'dim_corr_summary': {
            'delta_l2_abs_mean_top20': float(np.abs(dim_corr_l2[top20_l2_idx]).mean()),
            'l30_l2_abs_mean_top20':  float(np.abs(dim_corr_l30_l2[top20_l2_idx]).mean()),
        }
    },
    'norm_vs_domain_analysis': {
        'rho_norm_ndomain': float(rho_norm_dom),
        'p_norm_ndomain':   float(p_norm_dom),
        'mean_norm_no_domain':  float(delta_norms[n_domains_per_iso==0].mean()),
        'mean_norm_ge1_domain': float(delta_norms[n_domains_per_iso>0].mean()),
        'mean_norm_ge3_domain': float(delta_norms[n_domains_per_iso>=3].mean()),
    }
}

# Save full dim correlation arrays
np.save(f'{OUT_DIR}/dim_corr_delta_l2.npy', dim_corr_l2.astype(np.float32))
np.save(f'{OUT_DIR}/dim_corr_delta_l4.npy', dim_corr_l4.astype(np.float32))
np.save(f'{OUT_DIR}/dim_corr_l30_l2.npy',  dim_corr_l30_l2.astype(np.float32))
json.dump(results, open(f'{OUT_DIR}/probing_results.json', 'w'), indent=2)

print(f"\n[Saved] {OUT_DIR}/probing_results.json")
print(f"[Saved] dim_corr arrays (640-dim)")

# ── Generate summary figure ──────────────────────────────────────
print("\n[5] Generating probing summary figure...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('δ_layer = φ_L30 − φ_L15: Mechanistic Probing', fontsize=14, fontweight='bold')

    # Panel A: Domain predictive power comparison
    ax = axes[0, 0]
    methods = list(domain_summary.keys())
    means = [domain_summary[m]['mean'] for m in methods]
    stds  = [domain_summary[m]['std']  for m in methods]
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color=colors, alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['δ_layer', 'L30', 'L15', 'δ_raw'], rotation=15)
    ax.set_ylabel('Domain Prediction AUROC')
    ax.set_title('(A) Pfam Domain Predictability\n(LR, 5-fold CV, 100 domains)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='random')
    ax.set_ylim(0.45, 1.0)
    ax.legend()

    # Panel B: Within-gene pair Δδ vs Δdomain
    ax = axes[0, 1]
    sample_size = min(5000, len(pair_delta_diffs))
    idx_sample = np.random.choice(len(pair_delta_diffs), sample_size, replace=False)
    sc = ax.scatter(pair_domain_diffs[idx_sample], pair_delta_diffs[idx_sample],
                    alpha=0.2, s=3, c='#2196F3')
    ax.set_xlabel('Δdomain (Hamming distance)')
    ax.set_ylabel('Δδ_layer (L2 norm)')
    ax.set_title(f'(B) Within-gene Pairs: Δδ_layer vs Δdomain\nρ={rho_delta_domain:.3f}, p={p_delta_domain:.1e}')

    # Panel C: Per-dimension correlation with L2_Struct vs L4_CellState
    ax = axes[1, 0]
    ax.scatter(dim_corr_l2, dim_corr_l4, alpha=0.3, s=5, c='#555555')
    ax.scatter(dim_corr_l2[top20_l2_idx], dim_corr_l4[top20_l2_idx],
               alpha=0.9, s=30, c='#FF5722', label='top-20 L2 dims', zorder=5)
    ax.scatter(dim_corr_l2[top20_l4_idx], dim_corr_l4[top20_l4_idx],
               alpha=0.9, s=30, c='#4CAF50', label='top-20 L4 dims', zorder=5)
    ax.axhline(0, color='gray', alpha=0.3); ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('r(δ_dim, L2_Structural GO)')
    ax.set_ylabel('r(δ_dim, L4_CellState GO)')
    ax.set_title(f'(C) δ_layer dims: L2 vs L4 correlation\noverlap top-20: {overlap_l2_l4}/20')
    ax.legend(fontsize=8)

    # Panel D: δ norm by domain count
    ax = axes[1, 1]
    categories = ['0', '1', '2', '≥3']
    masks = [n_domains_per_iso == 0, n_domains_per_iso == 1,
             n_domains_per_iso == 2, n_domains_per_iso >= 3]
    means_norm = [delta_norms[m].mean() for m in masks]
    stds_norm  = [delta_norms[m].std()  for m in masks]
    ns_norm    = [m.sum() for m in masks]
    colors_d = ['#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1']
    bars = ax.bar(categories, means_norm, yerr=stds_norm, capsize=4,
                  color=colors_d, alpha=0.8)
    for bar, n in zip(bars, ns_norm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds_norm)*0.05,
                f'n={n}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Number of Pfam domains')
    ax.set_ylabel('Mean δ_layer norm')
    ax.set_title(f'(D) δ_layer Norm by Domain Count\nρ={rho_norm_dom:.3f}, p={p_norm_dom:.1e}')

    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/probing_summary.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(f'{OUT_DIR}/probing_summary.png', bbox_inches='tight', dpi=150)
    print(f"[Saved] {OUT_DIR}/probing_summary.pdf")
except Exception as e:
    print(f"  Figure error: {e}")

print("\n" + "=" * 65)
print("  Experiment B: COMPLETE")
print("=" * 65)
