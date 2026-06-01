"""
TCS (Tissue Context Specificity) Quantification
================================================
Tau index + SMSI (Skeletal Muscle Specificity Index)
for 5 GO terms in hMuscle project.

Metric definitions:
  Tau(gene) = sum_i(1 - x_hat_i) / (N - 1)
              where x_hat_i = x_i / max_tissue(x_i), N = number of tissues
  TCS(GO)   = mean(Tau over positive genes for that GO term)

  SMSI(gene) = nTPM_skeletal_muscle / mean(nTPM_all_tissues)
  SMSI(GO)   = mean(SMSI over positive genes for that GO term)

Data sources:
  - HPA rna_tissue_consensus.tsv.zip  (51 tissues)
  - human_annotations.txt             (gene → GO terms)

Output:
  tcs_results.json
  tbs_tcs_scatter.png   (2D: TBS x-axis, TCS y-axis, color = FTI)
  smsi_barplot.png
"""

import os
import json
import zipfile
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Paths ───────────────────────────────────────────────────────────────────
HUMAN_ANNO   = '/home/welcome1/sw1686/DIFFUSE/hMuscle/data/raw_data/data/annotations/human_annotations.txt'
HPA_ZIP      = '/tmp/hpa_tissue2.zip'
TBS_JSON     = '/home/welcome1/sw1686/DIFFUSE/reports/2026-05-14/tbs_results.json'
OUT_DIR      = '/home/welcome1/sw1686/DIFFUSE/reports/2026-05-14'

TARGET_GOS   = ['GO:0007204', 'GO:0030017', 'GO:0006941', 'GO:0003774', 'GO:0006096']
GO_LABELS    = {
    'GO:0007204': 'Ca²⁺ signaling\n(GO:0007204)',
    'GO:0030017': 'Sarcomere\n(GO:0030017)',
    'GO:0006941': 'Striated contraction\n(GO:0006941)',
    'GO:0003774': 'Motor activity\n(GO:0003774)',
    'GO:0006096': 'Glycolysis\n(GO:0006096)',
}
GO_SHORT = {
    'GO:0007204': 'Ca²⁺',
    'GO:0030017': 'Sarcomere',
    'GO:0006941': 'Contraction',
    'GO:0003774': 'Myosin',
    'GO:0006096': 'Glycolysis',
}

FTI_256 = {
    'GO:0007204': 0.626,
    'GO:0030017': 0.482,
    'GO:0006941': 1.018,
    'GO:0003774': 1.026,
    'GO:0006096': 1.874,
}

TIER_COLOR = {
    'Tier 1 (SP Required)':  '#d62728',
    'Tier 2 (SP Neutral)':   '#ff7f0e',
    'Tier 3 (SP Harmful)':   '#1f77b4',
}

# ─── Step 1: Parse human annotations ─────────────────────────────────────────
print("=" * 60)
print(" TCS Quantification")
print("=" * 60)
print("\n[1] Parsing human annotations ...")

go_genes = collections.defaultdict(set)
with open(HUMAN_ANNO) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        gene = parts[0].strip()
        gos  = {g.strip() for g in parts[1:]}
        for go in gos:
            if go in TARGET_GOS:
                go_genes[go].add(gene)

for go in TARGET_GOS:
    print(f"  {go}: {len(go_genes[go])} positive genes")

all_positive_genes = set()
for go in TARGET_GOS:
    all_positive_genes |= go_genes[go]
print(f"  Total unique positive genes: {len(all_positive_genes)}")

# ─── Step 2: Load HPA tissue expression ──────────────────────────────────────
print("\n[2] Loading HPA tissue expression ...")

# gene_name → {tissue: nTPM}
gene_expr = collections.defaultdict(dict)
tissues_found = set()

with zipfile.ZipFile(HPA_ZIP) as z:
    with z.open('rna_tissue_consensus.tsv') as f:
        header = f.readline()
        for line in f:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) < 4:
                continue
            _, gene_name, tissue, ntpm_str = parts[0], parts[1], parts[2], parts[3]
            try:
                ntpm = float(ntpm_str)
            except ValueError:
                continue
            gene_expr[gene_name][tissue] = ntpm
            tissues_found.add(tissue)

tissues_list = sorted(tissues_found)
N_tissues    = len(tissues_list)
print(f"  Tissues: {N_tissues}")
print(f"  Genes with expression data: {len(gene_expr)}")

matched = all_positive_genes & set(gene_expr.keys())
unmatched = all_positive_genes - matched
print(f"  Positive genes matched in HPA: {len(matched)} / {len(all_positive_genes)}")
if unmatched:
    print(f"  Unmatched genes: {sorted(unmatched)[:10]} ...")

# ─── Step 3: Compute Tau index per gene ──────────────────────────────────────
print("\n[3] Computing Tau index per gene ...")

def tau_index(expr_vec):
    """
    Tau = sum_i(1 - x_hat_i) / (N - 1)
    x_hat_i = x_i / max(x)
    Returns tau in [0, 1]. Higher = more tissue-specific.
    Returns NaN if max expression = 0 (unexpressed everywhere).
    """
    x = np.array(expr_vec, dtype=float)
    max_x = x.max()
    if max_x == 0:
        return np.nan
    x_hat = x / max_x
    return np.sum(1.0 - x_hat) / (N_tissues - 1)

def smsi(expr_dict):
    """SMSI = nTPM_skeletal_muscle / mean(nTPM_all_tissues)"""
    skel = expr_dict.get('skeletal muscle', 0.0)
    vals = list(expr_dict.values())
    mean_all = np.mean(vals) if vals else 0.0
    if mean_all == 0:
        return np.nan
    return skel / mean_all

gene_tau  = {}
gene_smsi = {}

for gene in matched:
    expr_dict = gene_expr[gene]
    expr_vec  = [expr_dict.get(t, 0.0) for t in tissues_list]
    gene_tau[gene]  = tau_index(expr_vec)
    gene_smsi[gene] = smsi(expr_dict)

# ─── Step 4: Compute TCS per GO term ─────────────────────────────────────────
print("\n[4] Computing TCS per GO term ...")

results = {}
for go in TARGET_GOS:
    pos_genes = go_genes[go] & matched
    taus  = [gene_tau[g]  for g in pos_genes if not np.isnan(gene_tau[g])]
    smsis = [gene_smsi[g] for g in pos_genes if not np.isnan(gene_smsi[g])]

    tcs     = float(np.mean(taus))  if taus  else float('nan')
    smsi_go = float(np.mean(smsis)) if smsis else float('nan')

    results[go] = {
        'tcs':          tcs,
        'smsi':         smsi_go,
        'n_genes':      len(pos_genes),
        'n_valid_tau':  len(taus),
        'tau_per_gene': {g: float(gene_tau[g]) for g in pos_genes if not np.isnan(gene_tau[g])},
        'smsi_per_gene': {g: float(gene_smsi[g]) for g in pos_genes if not np.isnan(gene_smsi[g])},
    }

    # Print per-gene tau for top genes
    top_tau  = sorted(results[go]['tau_per_gene'].items(), key=lambda x: -x[1])[:5]
    top_smsi = sorted(results[go]['smsi_per_gene'].items(), key=lambda x: -x[1])[:5]
    print(f"\n  {go}:")
    print(f"    TCS (mean tau) = {tcs:.3f}   SMSI = {smsi_go:.3f}")
    print(f"    n_genes = {len(pos_genes)}, n_valid = {len(taus)}")
    print(f"    Top tau genes:  {top_tau}")
    print(f"    Top SMSI genes: {top_smsi}")

# ─── Step 5: Load TBS results ─────────────────────────────────────────────────
print("\n[5] Loading TBS results ...")
with open(TBS_JSON) as f:
    tbs_data = json.load(f)

tbs_vals = {go: tbs_data[go]['tbs'] for go in TARGET_GOS}
fti_vals = FTI_256

# ─── Step 6: Save combined results ───────────────────────────────────────────
combined = {}
for go in TARGET_GOS:
    combined[go] = {
        'tbs':     tbs_vals[go],
        'tcs':     results[go]['tcs'],
        'smsi':    results[go]['smsi'],
        'fti_256': fti_vals[go],
        'n_genes': results[go]['n_genes'],
    }

out_json = os.path.join(OUT_DIR, 'tcs_results.json')
with open(out_json, 'w') as f:
    json.dump({'per_go': combined, 'per_gene': {go: results[go]['tau_per_gene'] for go in TARGET_GOS}}, f, indent=2)
print(f"\n[6] Saved: {out_json}")

# ─── Step 7: 2D TBS × TCS scatter (main Figure) ──────────────────────────────
print("\n[7] Generating 2D TBS × TCS scatter ...")

fig, ax = plt.subplots(figsize=(7, 6))

tbs_arr = np.array([tbs_vals[go] for go in TARGET_GOS])
tcs_arr = np.array([results[go]['tcs'] for go in TARGET_GOS])
fti_arr = np.array([fti_vals[go] for go in TARGET_GOS])

# Color = FTI_256 (red = high, blue = low)
norm   = plt.Normalize(vmin=0.4, vmax=2.0)
cmap   = cm.RdYlGn
colors = [cmap(norm(v)) for v in fti_arr]

scatter = ax.scatter(tbs_arr, tcs_arr, c=fti_arr, cmap='RdYlGn',
                     norm=norm, s=300, zorder=5, edgecolors='black', linewidths=0.8)

# Annotations
offsets = {
    'GO:0007204': (0.03,  0.01),
    'GO:0030017': (-0.18, -0.03),
    'GO:0006941': (0.03,  -0.035),
    'GO:0003774': (0.03,  0.01),
    'GO:0006096': (0.03,  -0.035),
}
for go in TARGET_GOS:
    dx, dy = offsets.get(go, (0.02, 0.01))
    ax.annotate(
        f"{GO_SHORT[go]}\nFTI={fti_vals[go]:.2f}",
        xy=(tbs_vals[go], results[go]['tcs']),
        xytext=(tbs_vals[go] + dx, results[go]['tcs'] + dy),
        fontsize=9, ha='left', va='center',
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
    )

# Quadrant lines (median)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.4, lw=0.8)
ax.axhline(np.median(tcs_arr), color='gray', linestyle='--', alpha=0.4, lw=0.8)

# Quadrant labels
tcs_med = np.median(tcs_arr)
ax.text(0.15, tcs_med + 0.015, 'Low TBS\nHigh TCS\n(Tier 3)',
        fontsize=7.5, color='#1f77b4', ha='center', alpha=0.8)
ax.text(0.73, tcs_med + 0.015, 'High TBS\nHigh TCS\n(→ Tier 3)',
        fontsize=7.5, color='#1f77b4', ha='center', alpha=0.8)
ax.text(0.15, tcs_med - 0.025, 'Low TBS\nLow TCS\n(Tier 2)',
        fontsize=7.5, color='#ff7f0e', ha='center', alpha=0.8)
ax.text(0.73, tcs_med - 0.025, 'High TBS\nLow TCS\n(→ Tier 1)',
        fontsize=7.5, color='#d62728', ha='center', alpha=0.8)

cb = plt.colorbar(scatter, ax=ax, shrink=0.85)
cb.set_label('FTI₂₅₆ = AUPRC(SP✓) / AUPRC(SP✗)', fontsize=10)
cb.ax.axhline(1.0, color='black', linestyle='--', lw=1.0)
cb.ax.text(2.2, 1.0, 'FTI=1\n(neutral)', fontsize=7, va='center')

ax.set_xlabel('TBS (Taxonomic Breadth Score)', fontsize=12)
ax.set_ylabel('TCS (Tissue Context Specificity, τ)', fontsize=12)
ax.set_title('2D SP Dependency Framework\nTBS × TCS → FTI Prediction', fontsize=12, fontweight='bold')
ax.set_xlim(0.05, 1.0)
ax.set_ylim(tcs_arr.min() - 0.06, tcs_arr.max() + 0.08)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out_2d = os.path.join(OUT_DIR, 'tbs_tcs_scatter.png')
plt.savefig(out_2d, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_2d}")

# ─── Step 8: SMSI bar plot ────────────────────────────────────────────────────
print("\n[8] Generating SMSI bar plot ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

go_order = sorted(TARGET_GOS, key=lambda g: fti_vals[g])
labels   = [GO_SHORT[g] for g in go_order]
tcs_plot = [results[g]['tcs'] for g in go_order]
smsi_plot = [results[g]['smsi'] for g in go_order]
fti_plot  = [fti_vals[g] for g in go_order]
colors_plot = [cmap(norm(v)) for v in fti_plot]

bars1 = ax1.bar(labels, tcs_plot, color=colors_plot, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('TCS (mean τ per GO term)', fontsize=11)
ax1.set_title('TCS by GO term\n(sorted by FTI)', fontsize=11)
ax1.set_ylim(0, max(tcs_plot) * 1.3)
for bar, val, fti in zip(bars1, tcs_plot, fti_plot):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'τ={val:.3f}\nFTI={fti:.2f}', ha='center', va='bottom', fontsize=8)

bars2 = ax2.bar(labels, smsi_plot, color=colors_plot, edgecolor='black', linewidth=0.8)
ax2.axhline(1.0, color='gray', linestyle='--', lw=1, alpha=0.7, label='SMSI=1 (baseline)')
ax2.set_ylabel('SMSI (nTPM skeletal muscle / mean all)', fontsize=11)
ax2.set_title('SMSI by GO term\n(sorted by FTI)', fontsize=11)
ax2.set_ylim(0, max(smsi_plot) * 1.35)
for bar, val in zip(bars2, smsi_plot):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
ax2.legend(fontsize=9)

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
sm.set_array([])
cb2 = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02)
cb2.set_label('FTI₂₅₆', fontsize=10)

plt.suptitle('Tissue Context Metrics by GO Term', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out_smsi = os.path.join(OUT_DIR, 'tcs_smsi_barplot.png')
plt.savefig(out_smsi, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_smsi}")

# ─── Step 9: Summary table ────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(" Combined TBS × TCS × FTI Summary")
print("=" * 68)
print(f"{'GO term':<14} {'TBS':>5} {'TCS(τ)':>8} {'SMSI':>7} {'FTI_256':>8}  Tier")
print("-" * 68)

tier_map = {
    'GO:0006096': 'Tier 1 (SP Required)',
    'GO:0003774': 'Tier 2 (SP Neutral+)',
    'GO:0006941': 'Tier 2 (SP Neutral)',
    'GO:0007204': 'Tier 3 (SP Harmful)',
    'GO:0030017': 'Tier 3 (SP Harmful)',
}

for go in TARGET_GOS:
    c = combined[go]
    print(f"{go:<14} {c['tbs']:>5.3f} {c['tcs']:>8.3f} {c['smsi']:>7.2f} {c['fti_256']:>8.3f}  {tier_map[go]}")

print()

# Pearson r (TCS vs FTI)
from scipy.stats import pearsonr
tcs_v = np.array([combined[g]['tcs'] for g in TARGET_GOS])
fti_v = np.array([combined[g]['fti_256'] for g in TARGET_GOS])
tbs_v = np.array([combined[g]['tbs'] for g in TARGET_GOS])

r_tcs, p_tcs = pearsonr(tcs_v, fti_v)
r_tbs, p_tbs = pearsonr(tbs_v, fti_v)
print(f"  TBS vs FTI_256: r = {r_tbs:.3f}  (p={p_tbs:.3f})")
print(f"  TCS vs FTI_256: r = {r_tcs:.3f}  (p={p_tcs:.3f})")

# Linear fit: FTI ~ TBS + TCS + TBS*TCS
from numpy.linalg import lstsq
X = np.column_stack([np.ones(5), tbs_v, tcs_v, tbs_v * tcs_v])
coef, residuals, rank, sv = lstsq(X, fti_v, rcond=None)
fti_pred = X @ coef
ss_res = np.sum((fti_v - fti_pred)**2)
ss_tot = np.sum((fti_v - fti_v.mean())**2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
print(f"\n  2D regression: FTI = {coef[0]:.3f} + {coef[1]:.3f}*TBS + {coef[2]:.3f}*TCS + {coef[3]:.3f}*(TBS*TCS)")
print(f"  R² = {r2:.3f}  (n=5, interpretive only)")
print()
print("  → GO:0007204 is the key counter-example:")
print(f"    High TBS ({tbs_vals['GO:0007204']:.3f}) but TCS={results['GO:0007204']['tcs']:.3f} → FTI={FTI_256['GO:0007204']:.3f}")
print("    This requires 2D framework (TBS alone r=0.359 insufficient)")
