"""
plot_window_ablation.py
========================
Fig 2 for natcomm_Flow.md — window ablation Pareto frontier.

Left  : macro AUPRC vs w (bar / line)
Right : AUPRC vs T3/T12_all scatter with w annotations
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family'    : ['DejaVu Sans', 'Arial'],
    'font.size'      : 11,
    'axes.titlesize' : 13,
    'axes.labelsize' : 11,
    'axes.linewidth' : 1.0,
})

ROOT = Path("/home/welcome1/sw1686/DIFFUSE/reports")
OUT  = ROOT / "curve_sweep"

paths = [
    (0, ROOT / "exp_B2_window_ablation/w0_results.json"),
    (3, ROOT / "exp_B2_window_ablation/w3_results.json"),
    (5, ROOT / "v20b/w5_results.json"),
    (7, ROOT / "v20b/w7_results.json"),
    (10, ROOT / "exp_B2_window_ablation/w10_results.json"),
]

ws, auprc, t3all, t3mid, dim = [], [], [], [], []
for w, p in paths:
    d = json.load(open(p))
    ws.append(w)
    auprc.append(d['macro_auprc'])
    t3all.append(d['t3_t12_ratio_all'])
    t3mid.append(d['t3_t12_ratio_mid'])
    dim.append(640 + d['n_layers_max']*8 if d['n_layers_max'] else 640)

ws     = np.array(ws)
auprc  = np.array(auprc)
t3all  = np.array(t3all)
t3mid  = np.array(t3mid)
dim    = np.array(dim)

# ── Figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Window-size ablation of the curve-informed layer selection (18 BP GO)",
             fontsize=14, fontweight='bold', y=1.01)

# Panel (a) — dual y-axis line plot
ax1 = axes[0]
c1, c2 = '#1E88E5', '#D81B60'

l1 = ax1.plot(ws, auprc, 'o-', color=c1, lw=2.4, ms=10,
              markeredgecolor='white', markeredgewidth=1.2,
              label='macro AUPRC')
ax1.set_xlabel('Window half-size w  (layers around Fisher peak)')
ax1.set_ylabel('macro AUPRC', color=c1)
ax1.tick_params(axis='y', labelcolor=c1)
ax1.grid(True, alpha=0.35, linestyle=':')

ax1b = ax1.twinx()
l2 = ax1b.plot(ws, t3all, 's--', color=c2, lw=2.4, ms=10,
               markeredgecolor='white', markeredgewidth=1.2,
               label='T3/T12 (all)')
l3 = ax1b.plot(ws, t3mid, 'D:', color='#8E24AA', lw=2.0, ms=8,
               markeredgecolor='white', markeredgewidth=1.0,
               label='T3/T12 (mid)')
ax1b.set_ylabel('T3/T12 within-gene spread', color=c2)
ax1b.tick_params(axis='y', labelcolor=c2)

# annotate w=0, w=5
for w_v in [0, 5]:
    i = list(ws).index(w_v)
    ax1.annotate(f'w={w_v}', (ws[i], auprc[i]),
                 xytext=(0, 12), textcoords='offset points',
                 fontsize=10, fontweight='bold', ha='center', color='black')

# selected marker
i5 = list(ws).index(5)
ax1.annotate('selected\n(Flow)', (ws[i5], auprc[i5]),
             xytext=(20, -25), textcoords='offset points',
             fontsize=9, fontweight='bold',
             color='darkred',
             arrowprops=dict(arrowstyle='-|>', color='darkred', lw=1.4))

ax1.set_title('(a) Metric-vs-w profile')

# combine legends
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', fontsize=10, framealpha=0.9)

# Panel (b) — Pareto scatter
ax2 = axes[1]
sizes = 220 + (dim - dim.min()) * 1.0
sc = ax2.scatter(auprc, t3all, s=sizes, c=ws, cmap='viridis',
                 edgecolors='black', linewidths=1.4, zorder=5)
for i, w_v in enumerate(ws):
    ax2.annotate(f'w={w_v}', (auprc[i], t3all[i]),
                 xytext=(8, 8), textcoords='offset points',
                 fontsize=11, fontweight='bold')

# Pareto arrows
ax2.annotate('', xy=(auprc[list(ws).index(5)], t3all[list(ws).index(5)]),
             xytext=(auprc[list(ws).index(0)], t3all[list(ws).index(0)]),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.8, alpha=0.8))
ax2.text(0.688, 0.36, 'Pareto\ngain',
         fontsize=10, color='darkred', fontweight='bold', ha='center')

# baseline lines
v15d = 0.7022
ax2.axvline(v15d, color='#B0B0B0', lw=1.0, ls='--')
ax2.text(v15d-0.001, 0.34, 'v15d\nbaseline', ha='right', va='top',
         fontsize=9, color='#606060')

ax2.set_xlabel('macro AUPRC')
ax2.set_ylabel('T3/T12 within-gene spread (all)')
ax2.set_title('(b) AUPRC vs T3/T12 Pareto frontier')
ax2.grid(True, alpha=0.35, linestyle=':')

cb = plt.colorbar(sc, ax=ax2, shrink=0.75)
cb.set_label('window half-size w', fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "fig_window_ablation.png", dpi=140, bbox_inches='tight')
fig.savefig(OUT / "fig_window_ablation.pdf", bbox_inches='tight')
plt.close(fig)
print(f"[saved] {OUT}/fig_window_ablation.{{png,pdf}}")

# Print summary for manuscript
print("\nSummary for natcomm_Flow.md:")
print(f"{'w':>3s} {'dim':>5s} {'AUPRC':>7s} {'T3/T12_all':>11s} {'T3/T12_mid':>11s}  {'ΔAUPRC':>7s} {'ΔT3/T12':>8s}")
for i in range(len(ws)):
    dA = auprc[i]-auprc[0]
    dT = t3all[i]-t3all[0]
    print(f"{ws[i]:>3d} {dim[i]:>5d} {auprc[i]:>7.4f} {t3all[i]:>11.4f} {t3mid[i]:>11.4f}  "
          f"{dA:>+7.4f} {dT:>+8.4f}")
