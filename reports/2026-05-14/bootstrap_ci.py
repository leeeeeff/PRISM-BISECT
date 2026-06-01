"""
Bootstrap CI Computation — 2026-05-14
======================================
Gene-block bootstrap (n=1000) for:
  - v8b, P3-512, D256, Selective Ensemble (pipeline models)
  - ESM-2 LR baseline (human-only, 640d, logistic regression)

Method: resample at GENE level (all isoforms of a gene move together)
→ prevents within-gene data leakage.

Outputs:
  bootstrap_ci_results.json
  fig_bootstrap_ci.png
"""

import os, json, time, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = '/home/welcome1/sw1686/DIFFUSE'
RES_DIR  = f'{ROOT}/hMuscle/results_isoform'
DATA_DIR = f'{ROOT}/hMuscle/data'
MODEL_DIR = f'{ROOT}/hMuscle/model'
OUT_DIR  = f'{ROOT}/reports/2026-05-14'

TARGET_GOS = ['GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096']
GO_LABEL   = {
    'GO:0007204': 'Ca²⁺',
    'GO:0030017': 'Sarcomere',
    'GO:0006941': 'Contraction',
    'GO:0003774': 'Myosin',
    'GO:0006096': 'Glycolysis',
}

N_BOOT = 1000
SEED   = 42
rng    = np.random.default_rng(SEED)

# Point estimates (from previous experiments)
POINT = {
    'v8b':       {'GO:0007204':0.1462,'GO:0030017':0.1570,'GO:0006941':0.1177,'GO:0003774':0.5686,'GO:0006096':0.7945},
    'P3-512':    {'GO:0007204':0.4055,'GO:0030017':0.3553,'GO:0006941':0.1954,'GO:0003774':0.6192,'GO:0006096':0.4939},
    'D256':      {'GO:0007204':0.1947,'GO:0030017':0.1366,'GO:0006941':0.1567,'GO:0003774':0.5982,'GO:0006096':0.8331},
    'LR':        {'GO:0007204':0.4140,'GO:0030017':0.5610,'GO:0006941':0.3120,'GO:0003774':0.8250,'GO:0006096':0.6950},
    'Selective': {'GO:0007204':0.4055,'GO:0030017':0.3553,'GO:0006941':0.1954,'GO:0003774':0.6192,'GO:0006096':0.8331},
}

# Result directory map
RES_DIRS = {
    'v8b':    'v8b_integrated_20260430_{time}',
    'P3-512': 'v8b-P3-512_integrated_20260514_{time}',
    'D256':   'v8b-D256_integrated_20260514_{time}',
}
RES_DIR_MAP = {
    'GO:0007204': {
        'v8b':    f'{RES_DIR}/GO_0007204/v8b_integrated_20260430_1459',
        'P3-512': f'{RES_DIR}/GO_0007204/v8b-P3-512_integrated_20260514_1114',
        'D256':   f'{RES_DIR}/GO_0007204/v8b-D256_integrated_20260514_0315',
    },
    'GO:0030017': {
        'v8b':    f'{RES_DIR}/GO_0030017/v8b_integrated_20260430_1543',
        'P3-512': f'{RES_DIR}/GO_0030017/v8b-P3-512_integrated_20260514_1137',
        'D256':   f'{RES_DIR}/GO_0030017/v8b-D256_integrated_20260514_0355',
    },
    'GO:0006941': {
        'v8b':    f'{RES_DIR}/GO_0006941/v8b_integrated_20260430_1618',
        'P3-512': f'{RES_DIR}/GO_0006941/v8b-P3-512_integrated_20260514_1200',
        'D256':   f'{RES_DIR}/GO_0006941/v8b-D256_integrated_20260514_0431',
    },
    'GO:0003774': {
        'v8b':    f'{RES_DIR}/GO_0003774/v8b_integrated_20260430_1459',
        'P3-512': f'{RES_DIR}/GO_0003774/v8b-P3-512_integrated_20260514_1114',
        'D256':   f'{RES_DIR}/GO_0003774/v8b-D256_integrated_20260514_0315',
    },
    'GO:0006096': {
        'v8b':    f'{RES_DIR}/GO_0006096/v8b_integrated_20260430_1539',
        'P3-512': f'{RES_DIR}/GO_0006096/v8b-P3-512_integrated_20260514_1132',
        'D256':   f'{RES_DIR}/GO_0006096/v8b-D256_integrated_20260514_0353',
    },
}

# ─── Step 1: Build gene→isoform index ────────────────────────────────────────
print("=" * 60)
print(" Bootstrap CI — Gene-block resampling")
print(f" N_BOOT={N_BOOT}, SEED={SEED}")
print("=" * 60)

iso_list  = np.load(f'{MODEL_DIR}/my_isoform_list_fixed.npy', allow_pickle=True)
gene_list = np.load(f'{MODEL_DIR}/my_gene_list_fixed.npy',    allow_pickle=True)
iso_list  = [x.decode() if isinstance(x, bytes) else x for x in iso_list]
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_list]
N = len(iso_list)

gene_to_idx = collections.defaultdict(list)
for idx, g in enumerate(gene_list):
    gene_to_idx[g].append(idx)
unique_genes = list(gene_to_idx.keys())
n_genes = len(unique_genes)
print(f"\n[Index] {N} isoforms, {n_genes} unique genes")

# ─── Step 2: Load pipeline scores per GO term ─────────────────────────────────
def load_scores(go, model_key):
    """Load Final_scores.txt and Final_labels.npy, return (scores, labels) arrays."""
    d = RES_DIR_MAP[go][model_key]
    go_tag  = go.replace(':', '_')
    prefix  = {'v8b': 'v8b_integrated', 'P3-512': 'v8b-P3-512_integrated',
                'D256': 'v8b-D256_integrated'}[model_key]
    score_file = f'{d}/{prefix}_{go_tag}_Final_scores.txt'
    label_file = f'{d}/{prefix}_{go_tag}_Final_labels.npy'
    if not os.path.exists(score_file):
        # try alternate naming
        score_file = os.path.join(d, [f for f in os.listdir(d) if 'Final_scores' in f][0])
        label_file = os.path.join(d, [f for f in os.listdir(d) if 'Final_labels' in f][0])
    df = pd.read_csv(score_file, sep='\t', header=None, names=['GeneID','IsoformID','Score'])
    labels = np.load(label_file)
    assert len(df) == N == len(labels), f"Size mismatch: {len(df)} {N} {len(labels)}"
    return df['Score'].values.astype(float), labels.astype(float)

print("\n[1] Loading pipeline scores ...")
all_scores = {}   # all_scores[go][model] = (scores, labels)
for go in TARGET_GOS:
    all_scores[go] = {}
    for model in ['v8b', 'P3-512', 'D256']:
        scores, labels = load_scores(go, model)
        all_scores[go][model] = (scores, labels)
        auprc = average_precision_score(labels, scores)
        print(f"  {go} {model}: AUPRC={auprc:.4f}  pos={int(labels.sum())}")

# ─── Step 3: Train LR baseline and get scores ─────────────────────────────────
print("\n[2] Training LR baseline (ESM-2 640d, human-only) ...")

X_train = np.load(f'{DATA_DIR}/esm2_train_human.npy').astype(np.float32)   # (31668, 640)
X_test  = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)  # (36748, 640)

# Load training human annotations
go_genes_train = collections.defaultdict(set)
anno_path = f'{DATA_DIR}/raw_data/data/annotations/human_annotations.txt'
train_iso_arr  = np.load(f'{DATA_DIR}/raw_data/data/id_lists/train_isoform_list.npy', allow_pickle=True)
train_gene_arr = np.load(f'{DATA_DIR}/raw_data/data/id_lists/train_gene_list.npy',    allow_pickle=True)
train_iso  = [x.decode() if isinstance(x,bytes) else x for x in train_iso_arr]
train_gene = [x.decode() if isinstance(x,bytes) else x for x in train_gene_arr]

with open(anno_path) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        gene = parts[0]
        for go in parts[1:]:
            go = go.strip()
            if go in TARGET_GOS:
                go_genes_train[go].add(gene)

for go in TARGET_GOS:
    t0 = time.time()
    y_train = np.array([1.0 if g in go_genes_train[go] else 0.0 for g in train_gene])
    y_test  = all_scores[go]['v8b'][1]   # same labels regardless of model

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                             solver='lbfgs', C=1.0, n_jobs=-1)
    clf.fit(X_tr, y_train)
    proba = clf.predict_proba(X_te)[:, 1]

    all_scores[go]['LR'] = (proba, y_test)
    auprc = average_precision_score(y_test, proba)
    print(f"  {go} LR: AUPRC={auprc:.4f}  ({time.time()-t0:.1f}s)")

# Add Selective ensemble scores
for go in TARGET_GOS:
    sel_model = 'D256' if go == 'GO:0006096' else 'P3-512'
    all_scores[go]['Selective'] = all_scores[go][sel_model]

# ─── Step 4: Gene-block bootstrap ─────────────────────────────────────────────
print(f"\n[3] Gene-block bootstrap (N={N_BOOT}) ...")

def bootstrap_auprc(scores, labels, gene_to_idx, unique_genes, n_boot=1000, rng=None):
    """
    Gene-block bootstrap: resample genes with replacement.
    Returns array of n_boot AUPRC values.
    """
    if rng is None:
        rng = np.random.default_rng()
    boot_auprcs = []
    for _ in range(n_boot):
        sampled_genes = rng.choice(len(unique_genes), size=len(unique_genes), replace=True)
        idx_all = []
        for gi in sampled_genes:
            idx_all.extend(gene_to_idx[unique_genes[gi]])
        idx_all = np.array(idx_all)
        y_b = labels[idx_all]
        s_b = scores[idx_all]
        if y_b.sum() == 0:
            boot_auprcs.append(0.0)
            continue
        boot_auprcs.append(average_precision_score(y_b, s_b))
    return np.array(boot_auprcs)

results = {}   # results[go][model] = {point, ci_lo, ci_hi, boot_dist}
MODELS  = ['v8b', 'P3-512', 'D256', 'LR', 'Selective']

for go in TARGET_GOS:
    results[go] = {}
    for model in MODELS:
        scores, labels = all_scores[go][model]
        t0 = time.time()
        boot = bootstrap_auprc(scores, labels, gene_to_idx, unique_genes, N_BOOT, rng)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        point = average_precision_score(labels, scores)
        results[go][model] = {
            'point': float(point),
            'ci_lo': float(ci_lo),
            'ci_hi': float(ci_hi),
            'boot_mean': float(boot.mean()),
            'boot_std':  float(boot.std()),
        }
        print(f"  {go} {model:10s}: {point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  ({time.time()-t0:.1f}s)")

# ─── Step 5: Paired bootstrap significance tests ───────────────────────────────
print(f"\n[4] Paired bootstrap significance tests ...")

sig_tests = {}
KEY_PAIRS = [
    # (go, model_A, model_B, description)
    ('GO:0006096', 'D256',  'LR',       'D256 vs LR (GO:0006096)'),
    ('GO:0007204', 'P3-512','LR',       'P3-512 vs LR (GO:0007204)'),
    ('GO:0007204', 'P3-512','v8b',      'P3-512 vs v8b (GO:0007204)'),
    ('GO:0030017', 'P3-512','v8b',      'P3-512 vs v8b (GO:0030017)'),
    ('GO:0006096', 'D256',  'v8b',      'D256 vs v8b (GO:0006096)'),
    ('GO:0003774', 'P3-512','v8b',      'P3-512 vs v8b (GO:0003774)'),
]

def paired_boot_pval(scores_A, scores_B, labels, gene_to_idx, unique_genes, n_boot=1000, rng=None):
    """
    H0: AUPRC(A) <= AUPRC(B)  (A is NOT better than B)
    p-value = fraction of bootstrap samples where A <= B
    (one-sided, A > B direction)
    """
    if rng is None:
        rng = np.random.default_rng()
    diffs = []
    for _ in range(n_boot):
        sampled_genes = rng.choice(len(unique_genes), size=len(unique_genes), replace=True)
        idx_all = []
        for gi in sampled_genes:
            idx_all.extend(gene_to_idx[unique_genes[gi]])
        idx_all = np.array(idx_all)
        y_b = labels[idx_all]
        if y_b.sum() == 0:
            diffs.append(0.0)
            continue
        auprc_a = average_precision_score(y_b, scores_A[idx_all])
        auprc_b = average_precision_score(y_b, scores_B[idx_all])
        diffs.append(auprc_a - auprc_b)
    diffs = np.array(diffs)
    # p-value: fraction of bootstrap where A <= B (i.e. diff <= 0)
    p = (diffs <= 0).mean()
    return float(p), diffs

for go, mA, mB, desc in KEY_PAIRS:
    scores_A, labels = all_scores[go][mA]
    scores_B, _      = all_scores[go][mB]
    pval, diffs = paired_boot_pval(scores_A, scores_B, labels,
                                   gene_to_idx, unique_genes, N_BOOT, rng)
    delta = results[go][mA]['point'] - results[go][mB]['point']
    sig_tests[desc] = {
        'go': go, 'model_A': mA, 'model_B': mB,
        'delta': float(delta),
        'p_value': pval,
        'significant_p05': bool(pval < 0.05),
        'significant_p01': bool(pval < 0.01),
        'diff_ci': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }
    sig = '**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns')
    print(f"  {desc}: Δ={delta:+.4f}  p={pval:.4f}  {sig}")

# ─── Step 6: Save results ─────────────────────────────────────────────────────
out = {
    'method': f'Gene-block bootstrap, N={N_BOOT}, seed={SEED}',
    'n_genes': n_genes,
    'n_isoforms': N,
    'results': results,
    'significance_tests': sig_tests,
}

json_path = f'{OUT_DIR}/bootstrap_ci_results.json'
with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n[5] Saved: {json_path}")

# ─── Step 7: Summary table ────────────────────────────────────────────────────
print()
print("=" * 90)
print(" Bootstrap CI Summary  [95% CI, gene-block bootstrap]")
print("=" * 90)
print(f"{'GO term':<14} {'Model':<12} {'AUPRC':>7} {'CI_lo':>7} {'CI_hi':>7} {'CI_width':>9}")
print("-" * 90)

for go in TARGET_GOS:
    for model in MODELS:
        r = results[go][model]
        print(f"{go:<14} {model:<12} {r['point']:>7.4f} {r['ci_lo']:>7.4f} {r['ci_hi']:>7.4f} {r['ci_hi']-r['ci_lo']:>9.4f}")
    print()

print("=" * 90)
print("\n Significance tests (one-sided: A > B):")
for desc, st in sig_tests.items():
    sig = '**' if st['p_value'] < 0.01 else ('*' if st['p_value'] < 0.05 else 'ns')
    print(f"  {desc:<40} Δ={st['delta']:+.4f}  p={st['p_value']:.4f}  [{st['diff_ci'][0]:+.4f}, {st['diff_ci'][1]:+.4f}]  {sig}")
print()
print("  * p<0.05  ** p<0.01  ns not significant")

# ─── Step 8: Visualization ───────────────────────────────────────────────────
print("\n[6] Generating figure ...")

MODEL_COLORS = {
    'v8b':       '#aec7e8',
    'P3-512':    '#98df8a',
    'D256':      '#ffbb78',
    'LR':        '#ff9896',
    'Selective': '#2ca02c',
}
MODEL_LABELS = {
    'v8b': 'v8b\n(SP✓,64d)',
    'P3-512': 'P3-512\n(SP✗,512d)',
    'D256': 'D256\n(SP✓,256d)',
    'LR': 'LR\n(640d)',
    'Selective': 'Selective\n(Best)',
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

for ax_idx, go in enumerate(TARGET_GOS):
    ax = axes_flat[ax_idx]

    xs     = np.arange(len(MODELS))
    points = [results[go][m]['point'] for m in MODELS]
    ci_los = [results[go][m]['ci_lo'] for m in MODELS]
    ci_his = [results[go][m]['ci_hi'] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    bars = ax.bar(xs, points, color=colors, edgecolor='black', linewidth=0.7, width=0.6)

    # CI error bars
    yerr_lo = np.array(points) - np.array(ci_los)
    yerr_hi = np.array(ci_his) - np.array(points)
    ax.errorbar(xs, points, yerr=[yerr_lo, yerr_hi],
                fmt='none', color='black', capsize=4, capthick=1.2, linewidth=1.2, zorder=5)

    # Value + CI labels
    for i, m in enumerate(MODELS):
        r = results[go][m]
        ax.text(i, r['ci_hi'] + 0.005,
                f"{r['point']:.3f}\n[{r['ci_lo']:.3f},\n {r['ci_hi']:.3f}]",
                ha='center', va='bottom', fontsize=6.5, linespacing=1.2)

    # Positives count
    n_pos = int(all_scores[go]['v8b'][1].sum())
    ax.set_title(f"{GO_LABEL[go]} ({go})\nn_pos={n_pos}", fontsize=10, fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=7.5)
    ax.set_ylabel('AUPRC', fontsize=9)
    ax.set_ylim(0, min(1.15, max(ci_his) + 0.18))
    ax.grid(axis='y', alpha=0.3)

    # Mark significance
    sig_pairs_for_go = [(d, st) for d, st in sig_tests.items() if st['go'] == go and st['significant_p05']]
    y_sig = max(ci_his) + 0.09
    for desc, st in sig_pairs_for_go:
        iA = MODELS.index(st['model_A'])
        iB = MODELS.index(st['model_B'])
        sig_str = '**' if st['p_value'] < 0.01 else '*'
        ax.annotate('', xy=(iA, y_sig), xytext=(iB, y_sig),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
        ax.text((iA + iB) / 2, y_sig + 0.01, sig_str,
                ha='center', fontsize=9, fontweight='bold')
        y_sig += 0.04

# Last panel: Macro-AUPRC with CI
ax = axes_flat[5]
macro_points = {}
macro_ci_lo  = {}
macro_ci_hi  = {}
for m in MODELS:
    macro_points[m] = np.mean([results[go][m]['point'] for go in TARGET_GOS])
    macro_ci_lo[m]  = np.mean([results[go][m]['ci_lo']  for go in TARGET_GOS])
    macro_ci_hi[m]  = np.mean([results[go][m]['ci_hi']  for go in TARGET_GOS])

xs = np.arange(len(MODELS))
bars = ax.bar(xs, [macro_points[m] for m in MODELS],
              color=[MODEL_COLORS[m] for m in MODELS],
              edgecolor='black', linewidth=0.7, width=0.6)

yerr_lo = np.array([macro_points[m] - macro_ci_lo[m] for m in MODELS])
yerr_hi = np.array([macro_ci_hi[m] - macro_points[m] for m in MODELS])
ax.errorbar(xs, [macro_points[m] for m in MODELS],
            yerr=[yerr_lo, yerr_hi],
            fmt='none', color='black', capsize=4, capthick=1.2, linewidth=1.2, zorder=5)

for i, m in enumerate(MODELS):
    ax.text(i, macro_ci_hi[m] + 0.004,
            f"{macro_points[m]:.4f}\n[{macro_ci_lo[m]:.4f},\n {macro_ci_hi[m]:.4f}]",
            ha='center', va='bottom', fontsize=6.5, linespacing=1.2)

ax.set_xticks(xs)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=7.5)
ax.set_title('Macro-AUPRC (mean across 5 GO terms)\n95% CI (per-GO mean)', fontsize=10, fontweight='bold')
ax.set_ylabel('Macro-AUPRC', fontsize=9)
ax.set_ylim(0, max(macro_ci_hi.values()) + 0.12)
ax.grid(axis='y', alpha=0.3)

legend_patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m].replace('\n', ' '))
                  for m in MODELS]
ax.legend(handles=legend_patches, fontsize=7.5, loc='upper left')

plt.suptitle(f'Bootstrap CI — Gene-block, N={N_BOOT}, Seed={SEED}\n'
             '* p<0.05  ** p<0.01  (one-sided paired bootstrap, A > B)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()

out_fig = f'{OUT_DIR}/fig_bootstrap_ci.png'
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_fig}")

print("\nDone.")
