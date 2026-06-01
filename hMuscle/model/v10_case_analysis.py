"""
v10_case_analysis.py — 3-Case Quantitative Classification
==========================================================
목적: positive class의 구조적 이질성을 5개 지표로 정량화하여
      Case 1/2/3의 객관적 근거를 확립

새로 계산 (이 스크립트에서 처음 계산):
  1. sep_cosine       — inter/intra cosine ratio (5개 GO term 누락분 포함 전체 13개)
  2. intra_cos_mean   — positive class 내 평균 pairwise cosine 유사도
                        높을수록 단일 패밀리 (Case 1 예상 > 0.7)
                        낮을수록 구조적으로 파편화 (Case 3 예상 < 0.5)
  3. intra_cos_std    — pairwise cosine 분산 → sub-cluster 존재 지표
  4. n_clusters       — k-means silhouette 최적화 (k=1..6)
                        Case 1: 1개, Case 2: 2-3개, Case 3: 4개+
  5. pc1_var_ratio    — PCA PC1 설명 분산 비율
                        높을수록 positive class가 단일 방향 → 구조 응집

기존 데이터 (F45 결과):
  - LR_AUPRC, v10B_AUPRC, Δ AUPRC, TBS, TCS, SMSI

3-case 정의:
  Case 1: LR >= 0.60  (LR-sufficient)
  Case 2: 0.45 <= LR < 0.60  (LR-partial)
  Case 3: LR < 0.45   (LR-insufficient)

출력: reports/case_analysis/
"""

import os, sys, json, time
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = '../../reports/case_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Hardcoded F45 + TBS/TCS results ─────────────────────────────────────────
# go: (label, case, lr, v10b, delta, tbs, tcs, smsi, sp_dep, n_sp, n_hum)
GO_DATA = {
    'GO:0003774': ('Motor activity',      1, 0.825, 0.813, -0.013, 0.833, 0.853, 1.313, 0.796,  399, 102),
    'GO:0006096': ('Glycolysis',          1, 0.695, 0.671, -0.023, 0.667, 0.849, 2.509, 0.876,  227,  32),
    'GO:0032006': ('TOR signaling',       2, 0.510, 0.602,  0.092, 0.667, 0.687, 1.256, 0.786,  331,  90),
    'GO:0007519': ('Skeletal muscle dev', 2, 0.587, 0.725,  0.138, 0.333, 0.885, 6.327, 0.884,  328,  43),
    'GO:0030017': ('Sarcomere org',       2, 0.564, 0.743,  0.179, 0.333, 0.919, 7.649, 0.779,  454, 129),
    'GO:0006941': ('Muscle contraction',  3, 0.310, 0.597,  0.287, 0.333, 0.908, 6.050, 0.715,  203,  81),
    'GO:0007204': ('Ca2+ signaling',      3, 0.415, 0.765,  0.350, 0.667, 0.869, 1.245, 0.714,  541, 217),
    'GO:0006914': ('Autophagy',           3, 0.285, 0.640,  0.354, 0.667, 0.661, 1.351, 0.817,  744, 167),
    'GO:0043161': ('Proteasome-UPS',      3, 0.362, 0.717,  0.356, 0.667, 0.684, 1.196, 0.831, 1362, 277),
    'GO:0042692': ('Muscle cell diff',    3, 0.232, 0.653,  0.421, 0.333, 0.884, 5.493, 0.841,  657, 124),
    'GO:0007005': ('Mitochondrion org',   3, 0.238, 0.662,  0.424, 0.667, 0.718, 1.684, 0.839, 2010, 385),
    'GO:0007517': ('Muscle organ dev',    3, 0.237, 0.702,  0.465, 0.333, 0.873, 5.117, 0.808,  629, 149),
    'GO:0055074': ('Ca2+ homeostasis',    3, 0.251, 0.726,  0.475, 0.833, 0.855, 1.348, 0.738,  964, 343),
}

GO_TERMS = list(GO_DATA.keys())
CASE_COLORS = {1: '#2ca02c', 2: '#ff7f0e', 3: '#d62728'}
CASE_LABELS = {1: 'Case 1 (LR-sufficient)',
               2: 'Case 2 (LR-partial)',
               3: 'Case 3 (LR-insufficient)'}

N_SAMPLE_COS  = 400   # pairwise cosine sample size
N_SAMPLE_CLUS = 300   # k-means sample size
MAX_K = 6
SILHOUETTE_MIN = 0.15  # k>1 only accepted if silhouette > this

# ─── Load embeddings & annotations ───────────────────────────────────────────
print("=" * 65)
print("  Loading embeddings & annotations")
print("=" * 65)

X_te = np.load('../data/esm2_embeddings_t30_150M.npy').astype(np.float32)
te_gene_ids = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_gene_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in te_gene_ids]
print(f"  Test embeddings: {X_te.shape}")

ensg2sym = {}
with open('../data/raw_data/data/id_lists/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5:
            ensg2sym[p[0]] = p[4]
te_syms = [ensg2sym.get(g.split('.')[0], g.split('.')[0]) for g in te_gene_ids]

human_prot_go = {}
with open('../data/raw_data/data/annotations/human_annotations.txt') as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            human_prot_go[p[0]] = set(p[1:])

# L2-normalize embeddings (cosine operations)
X_norm = normalize(X_te, norm='l2')
print(f"  Annotations: {len(human_prot_go):,} proteins\n")


# ─── Core metric functions ────────────────────────────────────────────────────

def get_label_vec(go_id):
    return np.array([1 if (s in human_prot_go and go_id in human_prot_go[s])
                     else 0 for s in te_syms], dtype=np.int32)


def compute_sep_cosine(Xn, y, rs=42):
    """inter_dist / intra_diversity (same formula as v10_sarcopenia_eval)"""
    rng = np.random.RandomState(rs)
    pi = np.where(y == 1)[0]
    ni = np.where(y == 0)[0]
    if len(pi) < 5:
        return np.nan
    if len(ni) > 2000: ni = rng.choice(ni, 2000, replace=False)
    if len(pi) > 2000: pi = rng.choice(pi, 2000, replace=False)
    pc = Xn[pi].mean(0); nc = Xn[ni].mean(0)
    pc /= (np.linalg.norm(pc) + 1e-10)
    nc /= (np.linalg.norm(nc) + 1e-10)
    inter = float(1 - np.dot(pc, nc))
    Xp = Xn[pi[:500]]
    m = Xp @ Xp.T
    up = np.triu(np.ones(m.shape, bool), k=1)
    intra = float(max(1 - m[up].mean(), 1e-10))
    return inter / intra


def compute_intra_cosine(Xn, y, rs=42):
    """mean & std of pairwise cosine similarities within positive class"""
    rng = np.random.RandomState(rs)
    pi = np.where(y == 1)[0]
    if len(pi) < 5:
        return np.nan, np.nan
    if len(pi) > N_SAMPLE_COS:
        pi = rng.choice(pi, N_SAMPLE_COS, replace=False)
    Xp = Xn[pi]  # already L2-normalized → dot product = cosine similarity
    M = Xp @ Xp.T  # (n × n) cosine similarity matrix
    up = np.triu(np.ones(M.shape, bool), k=1)
    cos_vals = M[up]
    return float(cos_vals.mean()), float(cos_vals.std())


def compute_n_clusters(Xn, y, rs=42):
    """
    Optimal cluster count via silhouette score (k=1..MAX_K).
    Returns (n_clusters, best_silhouette, silhouettes_list)
    k=1 if no multi-cluster structure is significant.
    """
    rng = np.random.RandomState(rs)
    pi = np.where(y == 1)[0]
    if len(pi) < 10:
        return 1, 0.0, []
    if len(pi) > N_SAMPLE_CLUS:
        pi = rng.choice(pi, N_SAMPLE_CLUS, replace=False)
    Xp = Xn[pi].copy()

    # PCA to reduce dimensionality (stable k-means)
    n_comp = min(50, len(pi) - 1, Xp.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, random_state=rs)
        Xp_r = pca.fit_transform(Xp)
    else:
        Xp_r = Xp

    sils = []
    for k in range(2, MAX_K + 1):
        if k >= len(pi):
            break
        km = KMeans(n_clusters=k, random_state=rs, n_init=10)
        labels = km.fit_predict(Xp_r)
        if len(np.unique(labels)) < 2:
            sils.append(0.0)
            continue
        sil = silhouette_score(Xp_r, labels, metric='euclidean')
        sils.append(float(sil))

    if not sils or max(sils) < SILHOUETTE_MIN:
        return 1, 0.0, sils
    best_k = int(np.argmax(sils)) + 2
    return best_k, float(max(sils)), sils


def compute_pc1_var(Xn, y, rs=42):
    """PC1 explained variance ratio of positive class embeddings"""
    pi = np.where(y == 1)[0]
    if len(pi) < 5:
        return np.nan
    Xp = Xn[pi]
    pca = PCA(n_components=min(10, len(pi) - 1, Xp.shape[1]), random_state=rs)
    pca.fit(Xp)
    return float(pca.explained_variance_ratio_[0])


# ─── Compute all metrics ──────────────────────────────────────────────────────
print("=" * 65)
print("  Computing embedding-based metrics for 13 GO terms")
print("=" * 65)

results = []
for go in GO_TERMS:
    label, case, lr, v10b, delta, tbs, tcs, smsi, sp_dep, n_sp, n_hum = GO_DATA[go]
    y = get_label_vec(go)
    n_pos = int(y.sum())

    t0 = time.time()
    sep      = compute_sep_cosine(X_norm, y)
    ic_mean, ic_std = compute_intra_cosine(X_norm, y)
    n_clus, best_sil, sil_list = compute_n_clusters(X_norm, y)
    pc1_var  = compute_pc1_var(X_norm, y)

    print(f"  {go} ({label:<22}) Case{case} "
          f"n_pos={n_pos:4d} | "
          f"sep={sep:.4f}  intra_cos={ic_mean:.4f}±{ic_std:.4f}  "
          f"n_clus={n_clus}(sil={best_sil:.3f})  pc1={pc1_var:.4f} "
          f"({time.time()-t0:.1f}s)")

    results.append({
        'go': go, 'label': label, 'case': case,
        'n_pos': n_pos,
        # Performance
        'lr':    lr,   'v10b':  v10b,  'delta': delta,
        # Annotation quality
        'tbs':   tbs,  'tcs':   tcs,   'smsi':  smsi,
        'sp_dep': sp_dep, 'n_sp': n_sp, 'n_hum': n_hum,
        # NEW: embedding-based structural metrics
        'sep_cosine':    round(sep, 4) if not np.isnan(sep) else None,
        'intra_cos_mean': round(ic_mean, 4) if not np.isnan(ic_mean) else None,
        'intra_cos_std':  round(ic_std, 4) if not np.isnan(ic_std) else None,
        'n_clusters':    n_clus,
        'best_silhouette': round(best_sil, 4),
        'silhouette_by_k': sil_list,
        'pc1_var_ratio': round(pc1_var, 4) if not np.isnan(pc1_var) else None,
    })

# ─── Correlations ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  Correlation analysis")
print("=" * 65)

def corr(arr1, arr2, name1, name2):
    valid = [(a, b) for a, b in zip(arr1, arr2)
             if a is not None and b is not None and not np.isnan(a) and not np.isnan(b)]
    if len(valid) < 4:
        print(f"  r({name1}, {name2}) = N/A (n={len(valid)})")
        return
    a, b = zip(*valid)
    r, p = stats.pearsonr(a, b)
    print(f"  r({name1:<18}, {name2:<12}) = {r:+.3f}  p={p:.4f}  n={len(valid)}")

metrics = ['sep_cosine','intra_cos_mean','n_clusters','pc1_var_ratio','tbs','tcs','smsi']
for m in metrics:
    arr = [r.get(m) for r in results]
    corr(arr, [r['delta'] for r in results], m, 'delta')
print()
for m in metrics:
    arr = [r.get(m) for r in results]
    corr(arr, [r['lr'] for r in results], m, 'lr_auprc')

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  FULL SUMMARY TABLE")
print(f"{'='*90}")
print(f"{'Label':<22} C  n_pos  sep     intra_μ  intra_σ  k   sil   pc1    LR     v10B   Δ")
print("-" * 90)
for r in results:
    print(f"{r['label']:<22} {r['case']}  {r['n_pos']:4d}  "
          f"{(r['sep_cosine'] or 0):.4f}  "
          f"{(r['intra_cos_mean'] or 0):.4f}   {(r['intra_cos_std'] or 0):.4f}   "
          f"{r['n_clusters']}  {r['best_silhouette']:.3f}  "
          f"{(r['pc1_var_ratio'] or 0):.4f}  "
          f"{r['lr']:.3f}  {r['v10b']:.3f}  {r['delta']:+.3f}")

# Case-level summary
print(f"\n{'='*65}")
print("  Case-level means (new embedding metrics)")
print("=" * 65)
for c in [1, 2, 3]:
    sub = [r for r in results if r['case'] == c]
    def mean_safe(key): return np.mean([r[key] for r in sub if r[key] is not None])
    print(f"  Case {c} (n={len(sub)}):")
    print(f"    intra_cos_mean = {mean_safe('intra_cos_mean'):.4f}")
    print(f"    n_clusters     = {mean_safe('n_clusters'):.2f}")
    print(f"    pc1_var_ratio  = {mean_safe('pc1_var_ratio'):.4f}")
    print(f"    sep_cosine     = {mean_safe('sep_cosine'):.4f}")
    print(f"    LR_AUPRC       = {mean_safe('lr'):.4f}")
    print(f"    Δ AUPRC        = {mean_safe('delta'):+.4f}")
    print()

# ─── Figure ───────────────────────────────────────────────────────────────────
print("  Generating figure...")

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

def scatter_panel(ax, xkey, ykey, xlabel, ylabel, title):
    for r in results:
        x = r.get(xkey); y = r.get(ykey)
        if x is None or y is None:
            continue
        c = CASE_COLORS[r['case']]
        ax.scatter(x, y, s=120, c=c, alpha=0.85,
                   edgecolors='black', linewidth=0.7, zorder=5)
        ax.annotate(r['go'].split(':')[1], xy=(x, y),
                    xytext=(4, 3), textcoords='offset points', fontsize=7, color='#333')
    # regression line
    pairs = [(r.get(xkey), r.get(ykey)) for r in results
             if r.get(xkey) is not None and r.get(ykey) is not None]
    if len(pairs) >= 4:
        xs, ys = zip(*pairs)
        r_val, p_val = stats.pearsonr(xs, ys)
        m, b = np.polyfit(xs, ys, 1)
        xlim = np.array([min(xs) - 0.02 * (max(xs)-min(xs)),
                         max(xs) + 0.02 * (max(xs)-min(xs))])
        ax.plot(xlim, m*xlim + b, 'k--', alpha=0.45, lw=1.2)
        ax.set_title(f'{title}\nr={r_val:+.3f}  p={p_val:.3f}  n={len(pairs)}', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color='gray', lw=0.6, alpha=0.4)

# Panel A: intra_cos_mean → LR_AUPRC
ax = fig.add_subplot(gs[0, 0])
scatter_panel(ax, 'intra_cos_mean', 'lr',
              'Intra-class cosine similarity (mean)',
              'LR AUPRC',
              'Structural coherence → LR performance')
ax.axhline(0.60, color=CASE_COLORS[1], lw=1, ls=':', alpha=0.7)
ax.axhline(0.45, color=CASE_COLORS[2], lw=1, ls=':', alpha=0.7)

# Panel B: intra_cos_mean → Δ AUPRC
ax = fig.add_subplot(gs[0, 1])
scatter_panel(ax, 'intra_cos_mean', 'delta',
              'Intra-class cosine similarity (mean)',
              'Δ AUPRC (v10-B − LR)',
              'Structural coherence → performance gap')

# Panel C: sep_cosine → Δ AUPRC
ax = fig.add_subplot(gs[0, 2])
scatter_panel(ax, 'sep_cosine', 'delta',
              'sep_cosine (inter/intra ratio)',
              'Δ AUPRC (v10-B − LR)',
              'sep_cosine → performance gap')

# Panel D: n_clusters → Δ AUPRC
ax = fig.add_subplot(gs[1, 0])
scatter_panel(ax, 'n_clusters', 'delta',
              'Optimal cluster count (k-means silhouette)',
              'Δ AUPRC (v10-B − LR)',
              'Sub-cluster count → performance gap')

# Panel E: pc1_var_ratio → LR_AUPRC
ax = fig.add_subplot(gs[1, 1])
scatter_panel(ax, 'pc1_var_ratio', 'lr',
              'PC1 explained variance (positive class)',
              'LR AUPRC',
              'Positive class directionality → LR')

# Panel F: 2D key plot — intra_cos_mean × sep_cosine, color/size = delta
ax = fig.add_subplot(gs[1, 2])
valid_r = [r for r in results if r['sep_cosine'] is not None and r['intra_cos_mean'] is not None]
delta_vals = np.array([r['delta'] for r in valid_r])
vmax = max(abs(delta_vals.min()), abs(delta_vals.max()))
norm_c = plt.Normalize(vmin=-vmax, vmax=vmax)
cmap = plt.cm.RdBu_r
for r in valid_r:
    color = cmap(norm_c(r['delta']))
    sz = 80 + abs(r['delta']) * 500
    ax.scatter(r['intra_cos_mean'], r['sep_cosine'], s=sz, c=[color],
               alpha=0.85, edgecolors=CASE_COLORS[r['case']], linewidth=1.8, zorder=5)
    ax.annotate(r['go'].split(':')[1], xy=(r['intra_cos_mean'], r['sep_cosine']),
                xytext=(4, 3), textcoords='offset points', fontsize=7)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Δ AUPRC', shrink=0.8)
ax.set_xlabel('Intra-class cosine mean', fontsize=9)
ax.set_ylabel('sep_cosine', fontsize=9)
ax.set_title('2D: Coherence Space (edge color = Case)', fontsize=9)

# Panel G: Case-level grouped bar (v10-B vs LR)
ax = fig.add_subplot(gs[2, 0:2])
cases_n = [1, 2, 3]
x_pos = np.arange(len(cases_n))
width = 0.32
for offset, (model, style) in enumerate([('lr', '///'), ('v10b', '')]):
    vals = []
    errs = []
    for c in cases_n:
        sub = [r for r in results if r['case'] == c]
        v = [r[model] for r in sub]
        vals.append(np.mean(v))
        errs.append(np.std(v))
    bars = ax.bar(x_pos + (offset - 0.5) * width, vals, width,
                  label='LR' if model == 'lr' else 'v10-B',
                  alpha=0.8, hatch=style,
                  color=['#aec7e8' if model == 'lr' else '#1f77b4'] * 3,
                  yerr=errs, capsize=4)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'Case {c}\n{CASE_LABELS[c].split("(")[1].rstrip(")")}' for c in cases_n], fontsize=9)
ax.set_ylabel('Mean AUPRC', fontsize=9)
ax.set_title('Case-level Performance: v10-B vs LR (mean ± SD)', fontsize=9)
ax.legend(fontsize=9); ax.set_ylim(0, 1.0)
ax.axhline(0.5, color='gray', lw=0.8, alpha=0.4, ls='--')

# Panel H: Metric correlation heatmap
ax = fig.add_subplot(gs[2, 2])
metric_keys = ['intra_cos_mean', 'sep_cosine', 'n_clusters', 'pc1_var_ratio',
               'tbs', 'tcs', 'smsi', 'lr', 'delta']
metric_labels = ['intra_cos', 'sep_cos', 'n_clus', 'pc1_var',
                 'TBS', 'TCS', 'SMSI', 'LR', 'Δ']
n_m = len(metric_keys)
corr_matrix = np.full((n_m, n_m), np.nan)
for i, mi in enumerate(metric_keys):
    for j, mj in enumerate(metric_keys):
        pairs = [(r.get(mi), r.get(mj)) for r in results
                 if r.get(mi) is not None and r.get(mj) is not None
                 and not (np.isnan(r.get(mi, 0)) or np.isnan(r.get(mj, 0)))]
        if len(pairs) >= 4:
            a, b = zip(*pairs)
            corr_matrix[i, j], _ = stats.pearsonr(a, b)

im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(metric_labels, fontsize=7)
for i in range(n_m):
    for j in range(n_m):
        if not np.isnan(corr_matrix[i, j]):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color='black' if abs(corr_matrix[i, j]) < 0.7 else 'white')
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
ax.set_title('Metric Correlation Heatmap', fontsize=9)

# Legend
patches = [mpatches.Patch(color=c, label=CASE_LABELS[n]) for n, c in CASE_COLORS.items()]
fig.legend(handles=patches, loc='upper center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, 0.99),
           frameon=True, fancybox=True)

fig.suptitle('v10-B vs LR: Positive Class Structural Heterogeneity Drives Performance Gap\n'
             '(13 GO terms, Δ AUPRC = v10-B mean − LR mean, 5 seeds)',
             fontsize=11, fontweight='bold', y=1.02)

pdf_path = f'{OUT_DIR}/case_analysis.pdf'
png_path = f'{OUT_DIR}/case_analysis.png'
plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figure saved: {pdf_path}")

# ─── Save JSON ────────────────────────────────────────────────────────────────
ts = time.strftime('%Y%m%d_%H%M')
out = {
    'timestamp': ts,
    'n_go_terms': len(results),
    'case_thresholds': {'case1_lr_min': 0.60, 'case2_lr_min': 0.45},
    'correlations_with_delta': {
        m: dict(zip(['r', 'p', 'n'], [
            *stats.pearsonr(
                [r[m] for r in results if r.get(m) is not None],
                [r['delta'] for r in results if r.get(m) is not None]
            ),
            sum(1 for r in results if r.get(m) is not None)
        ]))
        for m in ['sep_cosine', 'intra_cos_mean', 'n_clusters', 'pc1_var_ratio',
                  'tbs', 'tcs', 'smsi', 'lr']
        if sum(1 for r in results if r.get(m) is not None) >= 4
    },
    'per_go': results,
}
json_path = f'{OUT_DIR}/case_analysis_{ts}.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"  JSON saved:   {json_path}")
print("\nDone.")
