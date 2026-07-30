"""F3 — Why delta_layer: discriminative signal is distributed across layers, not at L30.
(a) master-panels heatmap: per-GO Fisher chi2 across 30 ESM-2 layers, rows sorted by peak
    layer, category color strip + colorbar. Overlaid population-mean raw-Fisher curve.
(b) per-GO peak-layer buckets (Early/Mid/Late) x category (BP/MF/CC).
(c) per-isoform attribution 3-bin (observed vs layer-shuffle null).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fig_style as fs
fs.apply()
ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT = os.path.join(ROOT, 'reports/figures_v2')
os.makedirs(OUT, exist_ok=True)

J = json.load(open(os.path.join(ROOT, 'reports/exp_C1_layer_probe_279/layer_probe_279_fisher.json')))
pg = J['per_go']
CATCOL = {'BP': fs.OI['blue'], 'MF': fs.OI['orange'], 'CC': fs.OI['green']}
rows, cats, peaks = [], [], []
for go, e in pg.items():
    f = np.array(e['fisher_per_layer'], float)
    rows.append(f); cats.append(e['category']); peaks.append(int(e['peak_layer']))
F = np.array(rows)                    # (277, 30)
cats = np.array(cats); peaks = np.array(peaks)
Fn = (F - F.min(1, keepdims=True)) / (F.max(1, keepdims=True) - F.min(1, keepdims=True) + 1e-9)
order = np.argsort(peaks)             # sort rows by peak layer
Fn_s = Fn[order]; cats_s = cats[order]; peaks_s = peaks[order]

# per-isoform 3-bin
b = np.load(os.path.join(ROOT, 'reports/v20b_pca_interp/per_iso_layer_bin.npy'))
early = np.mean((b >= 1) & (b <= 10)); mid = np.mean((b >= 11) & (b <= 20)); final = np.mean((b >= 21) & (b <= 29))
# null = bin-width expectation over layers 1..29
null_e, null_m, null_f = 10/29, 10/29, 9/29

fig = plt.figure(figsize=(7.2, 7.0))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.45, 1.0], width_ratios=[1.0, 1.0],
              hspace=0.42, wspace=0.42)

# ===== (a) heatmap =====
axA = fig.add_subplot(gs[0, :])
fs.panel_label(axA, 'a', dx=-0.035, dy=1.0)
axA.set_title('Discriminative signal (Fisher $\\chi^2$) is distributed across ESM-2 layers',
              loc='left', x=0.0)
im = axA.imshow(Fn_s, aspect='auto', cmap='magma', interpolation='nearest',
                extent=[1, 30, len(Fn_s), 0])
# category color strip
axStrip = axA.inset_axes([-0.028, 0, 0.018, 1], transform=axA.transAxes)
strip = np.array([[{'BP': 0, 'MF': 1, 'CC': 2}[c]] for c in cats_s])
from matplotlib.colors import ListedColormap
axStrip.imshow(strip, aspect='auto', cmap=ListedColormap([CATCOL['BP'], CATCOL['MF'], CATCOL['CC']]),
               interpolation='nearest'); axStrip.axis('off')
axA.set_xlabel('ESM-2 layer'); axA.set_ylabel('GO term (n=277, sorted by peak layer)')
axA.set_yticks([])
axA.axvline(15, color='w', ls='--', lw=0.9); axA.axvline(30, color='w', ls=':', lw=0.9)
axA.annotate(r'$\phi_{L15}$', xy=(15, 0), xytext=(15, -0.06), textcoords=('data', 'axes fraction'),
             color='w', ha='center', fontsize=6.5, bbox=dict(fc=fs.OI['verm'], ec='none', pad=0.15))
axA.annotate(r'$\phi_{L30}$', xy=(30, 0), xytext=(30, -0.06), textcoords=('data', 'axes fraction'),
             color='w', ha='center', fontsize=6.5, bbox=dict(fc=fs.C_PRISM, ec='none', pad=0.15))
med = int(np.median(peaks)); frac30 = np.mean(peaks == 30)
q1, q3 = int(np.percentile(peaks, 25)), int(np.percentile(peaks, 75))
axA.text(0.99, 0.03, f'median peak L{med} (IQR L{q1}-L{q3})  |  only {frac30*100:.1f}% peak at final layer L30',
         transform=axA.transAxes, ha='right', color='w', fontsize=7,
         bbox=dict(fc='#000000AA', ec='none', pad=0.3))
cb = fig.colorbar(im, ax=axA, fraction=0.025, pad=0.01)
cb.set_label('per-GO normalized Fisher $\\chi^2$', fontsize=6.5); cb.ax.tick_params(labelsize=6)
# category legend
from matplotlib.patches import Patch
axA.legend(handles=[Patch(fc=CATCOL[c], label=c) for c in ['BP', 'MF', 'CC']],
           loc='upper left', ncol=3, fontsize=6.5, bbox_to_anchor=(0, 1.0))
# population-mean raw curve inset
axIn = axA.inset_axes([0.62, 0.60, 0.35, 0.34])
axIn.plot(np.arange(1, 31), F.mean(0), color='w', lw=1.3)
axIn.set_facecolor('#00000000'); axIn.tick_params(colors='w', labelsize=5)
for s in axIn.spines.values(): s.set_color('w')
axIn.set_title('mean raw $\\chi^2$', color='w', fontsize=6, pad=1)
axIn.set_xticks([1, 15, 30])

# ===== (b) peak-layer buckets x category =====
axB = fig.add_subplot(gs[1, 0])
fs.panel_label(axB, 'b', dx=-0.16, dy=1.03)
axB.set_title('Peak-layer bucket by GO category', loc='left', x=0.0)
bs = J['bucket_summary']; tot = J['cat_totals']
buckets = ['Early (L1-10)', 'Mid (L11-20)', 'Late (L21-30)']
cats3 = ['BP', 'MF', 'CC']
xb = np.arange(len(buckets)); w = 0.26
for i, c in enumerate(cats3):
    frac = [bs[bk][c] / tot[c] for bk in buckets]
    axB.bar(xb + (i - 1) * w, frac, w, color=CATCOL[c], label=c)
axB.set_xticks(xb); axB.set_xticklabels(['Early\nL1-10', 'Mid\nL11-20', 'Late\nL21-30'], fontsize=6.2)
axB.set_ylabel('fraction of GO terms'); axB.legend(fontsize=6.5, ncol=3, loc='upper right')
axB.set_ylim(0, 0.6)

# ===== (c) per-isoform 3-bin =====
axC = fig.add_subplot(gs[1, 1])
fs.panel_label(axC, 'c', dx=-0.16, dy=1.03)
axC.set_title('Per-isoform information peak (N=63,994)', loc='left', x=0.0)
xc = np.arange(3)
obs = [early, mid, final]; null = [null_e, null_m, null_f]
bincol = [fs.LAYER_BINS['Early'], fs.LAYER_BINS['Mid'], fs.LAYER_BINS['Final']]
axC.bar(xc - 0.2, obs, 0.4, color=bincol, label='observed')
axC.bar(xc + 0.2, null, 0.4, color=fs.C_NULL, alpha=0.6, label='shuffle null')
for xi, (o, n) in enumerate(zip(obs, null)):
    axC.text(xi - 0.2, o + 0.008, f'{o*100:.1f}%', ha='center', fontsize=5.8)
axC.set_xticks(xc); axC.set_xticklabels(['Early\n[1-10]', 'Mid\n[11-20]', 'Final\n[21-29]'], fontsize=6.2)
axC.set_ylabel('fraction of isoforms'); axC.set_ylim(0, 0.5)
axC.legend(fontsize=6.3, loc='upper right')
axC.text(0.5, 0.90, 'Final layer depleted\n(21.6% vs null 31.0%)', transform=axC.transAxes,
         ha='center', fontsize=6, color=fs.OI['verm'])

fig.suptitle('Figure 3  |  Layer-contrast rationale: functional signal peaks in early-to-mid ESM-2 layers, not at the final layer',
             x=0.02, ha='left', fontsize=9.5, fontweight='bold', y=1.0)
p = os.path.join(OUT, 'F3_layer')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png', '| median peak L%d frac30 %.3f | obs' % (med, frac30), [round(v,3) for v in obs])
