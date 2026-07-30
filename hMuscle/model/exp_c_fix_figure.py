#!/usr/bin/env python3
"""Fix and regenerate decomposition figure from saved data."""
import json, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = '../../reports/exp_c_decomposition'

# ── Hardcoded values from all experiments ───────────────────────
# PRISM: from mf_domain_vs_prism.tsv (brain zero-shot, 82 MF terms)
# random-δ: from v17f_layer_breakdown/layer_breakdown.tsv
# v17e_layer: from exp_c run (concat[δ_layer, φ_L30] no T_ψ)
# v17f: from v17f_layer_breakdown.tsv

ref_data = {}
with open('../../reports/v17f_layer_breakdown/layer_breakdown.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ref_data[p[0]] = {
                'prism':      float(p[2]),
                'rand_delta': float(p[4]),
                'v17f':       float(p[3]),
            }

# v17e_layer from exp_c (measured)
v17e_layer = {
    'L2_Structural':  0.5527,
    'L4_CellState':   0.5316,
}

# For groups without direct v17e_layer measurement, use None (not available)
# We'll show only the groups we have data for in the full decomposition

# Build final decomposition table
groups = ['L2_Structural', 'L2_Structural*', 'L4_CellState', 'L1_Generic_mid', 'L1_Generic_high']
decomp = {}
for g in groups:
    if g not in ref_data:
        continue
    r = ref_data[g]
    e = v17e_layer.get(g, None)
    decomp[g] = {
        'prism':       r['prism'],
        'v17e_layer':  e,
        'rand_delta':  r['rand_delta'],
        'v17f':        r['v17f'],
    }
    decomp[g]['v17f_total_gain'] = r['v17f'] - r['prism']
    decomp[g]['raw_tpsi_cap']    = r['rand_delta'] - r['prism']
    if e is not None:
        decomp[g]['delta_signal'] = e - r['prism']
        decomp[g]['tpsi_add']     = r['v17f'] - e

# Compute All MF
all_mf = {
    'prism':      0.5962,
    'v17e_layer': 0.6817,
    'rand_delta': 0.6416,
    'v17f':       0.7173,
}
all_mf['delta_signal']    = all_mf['v17e_layer'] - all_mf['prism']   # +0.086
all_mf['tpsi_add']        = all_mf['v17f'] - all_mf['v17e_layer']    # +0.036
all_mf['raw_tpsi_cap']    = all_mf['rand_delta'] - all_mf['prism']   # +0.045
all_mf['v17f_total_gain'] = all_mf['v17f'] - all_mf['prism']         # +0.121
decomp['All MF'] = all_mf

json.dump(decomp, open(f'{OUT_DIR}/decomposition_fixed.json', 'w'), indent=2)
print("Decomposition table:")
print(f"{'Group':<20} {'PRISM':>7} {'v17e_L':>7} {'rand_δ':>7} {'v17f':>7} | {'δ_sig':>7} {'T_ψadd':>7} {'total':>7}")
for g, d in decomp.items():
    e = f"{d['v17e_layer']:.4f}" if d['v17e_layer'] is not None else "  N/A "
    ds = f"{d.get('delta_signal',float('nan')):+.4f}" if d.get('delta_signal') is not None else "  N/A "
    ta = f"{d.get('tpsi_add',float('nan')):+.4f}" if d.get('tpsi_add') is not None else "  N/A "
    ta  = f"{d['tpsi_add']:>+7.4f}"      if 'tpsi_add'     in d else "   N/A "
    ds  = f"{d['delta_signal']:>+7.4f}" if 'delta_signal' in d else "   N/A "
    print(f"{g:<20} {d['prism']:>7.4f} {e:>7} {d['rand_delta']:>7.4f} {d['v17f']:>7.4f} | {ds:>7} {ta:>7} {d['v17f_total_gain']:>+7.4f}")

# ── Figure ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('δ_layer + T_ψ: Three-Component Gain Decomposition\n'
             'Brain zero-shot evaluation (82 Molecular Function GO terms)',
             fontsize=13, fontweight='bold')

# Panel A: Absolute AUPRC comparison
ax = axes[0]
groups_plot = ['L2_Structural', 'L2_Structural*', 'L4_CellState', 'L1_Generic_mid', 'L1_Generic_high']
x = np.arange(len(groups_plot))
w = 0.2
model_colors = {
    'PRISM (baseline)':             ('#9E9E9E', 'prism'),
    'v17e_layer (δ_layer, no T_ψ)': ('#42A5F5', 'v17e_layer'),
    'random-δ (T_ψ capacity)':      ('#FFA726', 'rand_delta'),
    'v17f (δ_layer + T_ψ)':         ('#EF5350', 'v17f'),
}
for j, (label, (col, key)) in enumerate(model_colors.items()):
    vals = []
    for g in groups_plot:
        v = decomp.get(g, {}).get(key)
        vals.append(v if v is not None else 0.0)
    ax.bar(x + (j-1.5)*w, vals, w, label=label, color=col, alpha=0.85,
           edgecolor='white', linewidth=0.5)

short = {'L2_Structural': 'L2_Struct', 'L2_Structural*': 'L2_Struct*',
         'L4_CellState': 'L4_Cell', 'L1_Generic_mid': 'L1_mid', 'L1_Generic_high': 'L1_high'}
ax.set_xticks(x)
ax.set_xticklabels([short.get(g, g) for g in groups_plot], rotation=20, ha='right')
ax.set_ylabel('Macro AUPRC (brain zero-shot)')
ax.set_title('(A) Absolute AUPRC per GO Difficulty Layer')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=0.8)
ax.set_xlabel('GO Term Predictability Layer (H2 taxonomy)')

# Panel B: Stacked gain decomposition (for measured groups only)
ax = axes[1]
groups_with_v17e = ['All MF', 'L2_Structural', 'L4_CellState']
x2 = np.arange(len(groups_with_v17e))
w2 = 0.5

for j, g in enumerate(groups_with_v17e):
    d = decomp[g]
    p   = d['prism']
    sig = d.get('delta_signal', 0)
    tpa = d.get('tpsi_add', 0)
    cap = d.get('raw_tpsi_cap', 0)

    # PRISM base (gray)
    ax.bar(j, p, w2, color='#9E9E9E', alpha=0.4, label='PRISM baseline' if j==0 else '')
    # δ_layer signal (blue)
    ax.bar(j, sig, w2, bottom=p, color='#42A5F5', alpha=0.85,
           label='δ_layer signal (v17e_layer − PRISM)' if j==0 else '')
    # T_ψ additional gain on top of δ_layer (red)
    ax.bar(j, tpa, w2, bottom=p + sig, color='#EF5350', alpha=0.85,
           label='T_ψ gain (v17f − v17e_layer)' if j==0 else '')
    # v17f marker
    ax.plot([j-0.28, j+0.28], [d['v17f'], d['v17f']], 'k-', lw=2,
            label='v17f total' if j==0 else '', zorder=5)
    # Annotate fractions
    total_gain = d['v17f_total_gain']
    if total_gain > 0:
        frac_sig = sig / total_gain
        frac_tpsi = tpa / total_gain
        ax.text(j, p + sig/2, f'{frac_sig:.0%}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(j, p + sig + tpa/2, f'{frac_tpsi:.0%}', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

# Also show rand_delta reference
for j, g in enumerate(groups_with_v17e):
    d = decomp[g]
    ax.plot([j-0.28, j+0.28], [d['rand_delta'], d['rand_delta']],
            color='#FFA726', ls='--', lw=1.5,
            label='random-δ (T_ψ capacity ref)' if j==0 else '')

ax.set_xticks(x2)
ax.set_xticklabels(['All MF\n(n=82)', 'L2_Structural\n(n=33)', 'L4_CellState\n(n=2)'],
                   fontsize=10)
ax.set_ylabel('AUPRC')
ax.set_title('(B) Stacked Gain Decomposition\n'
             'δ_layer signal (blue) + T_ψ organization (red) = v17f')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color='gray', ls='--', alpha=0.4, lw=0.8)
ax.text(0.02, 0.97, 'Percentages = fraction of total v17f gain\nover PRISM baseline',
        transform=ax.transAxes, fontsize=8, va='top', alpha=0.7)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/decomposition_figure.pdf', bbox_inches='tight', dpi=150)
fig.savefig(f'{OUT_DIR}/decomposition_figure.png', bbox_inches='tight', dpi=150)
print(f"\n[Saved] {OUT_DIR}/decomposition_figure.pdf")
print(f"[Saved] {OUT_DIR}/decomposition_figure.png")

# Print Nature Methods summary
print("\n" + "="*65)
print("  KEY FINDING for Nature Methods:")
print("="*65)
print(f"\n  δ_layer signal alone (v17e_layer):  All MF +{all_mf['delta_signal']:+.3f} "
      f"({all_mf['delta_signal']/all_mf['v17f_total_gain']:.0%} of total gain)")
print(f"  T_ψ organization adds on top:       All MF +{all_mf['tpsi_add']:+.3f} "
      f"({all_mf['tpsi_add']/all_mf['v17f_total_gain']:.0%} of total gain)")
print(f"  Combined v17f gain over PRISM:      All MF +{all_mf['v17f_total_gain']:+.3f} (100%)")
print(f"\n  L2_Structural (hardest for mean-pooling):")
l2 = decomp['L2_Structural']
print(f"    δ_layer: {l2['prism']:.3f} → {l2['v17e_layer']:.3f} (+{l2['delta_signal']:.3f}, "
      f"{l2['delta_signal']/l2['v17f_total_gain']:.0%} of gain)")
print(f"    +T_ψ:    {l2['v17e_layer']:.3f} → {l2['v17f']:.3f} (+{l2['tpsi_add']:.3f}, "
      f"{l2['tpsi_add']/l2['v17f_total_gain']:.0%} of gain)")
