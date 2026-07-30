"""F1 — PRISM+BISECT architecture & pipeline (schematic, ML-style)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import fig_style as fs
fs.apply()
OUT = os.path.join(os.path.dirname(__file__), '../../reports/figures_v2')
os.makedirs(OUT, exist_ok=True)

def box(ax, x, y, w, h, text, fc, ec='none', fs_=7, tc='black', lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs_, color=tc, zorder=3)

def arrow(ax, x1, y1, x2, y2, c='#333', lw=1.3, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 color=c, lw=lw, zorder=1, shrinkA=1, shrinkB=1))

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.4, 1.0], hspace=0.35)

# ---- (a) Biological problem ----
axA = fig.add_subplot(gs[0]); axA.set_xlim(0,10); axA.set_ylim(0,3); axA.axis('off')
fs.panel_label(axA, 'a', dx=0.0, dy=0.98)
axA.set_title('The isoform annotation gap', loc='left', x=0.02)
# gene with isoforms
box(axA, 0.3, 1.2, 1.5, 0.6, 'Gene X\n(long-read\nisoforms)', '#E8E8E8', fs_=6.5)
for i,yy in enumerate([2.2, 1.55, 0.9, 0.25]):
    box(axA, 2.4, yy, 2.0, 0.42, f'isoform {i+1}', '#D7E9F7', fs_=6)
# existing tools -> same GO
box(axA, 5.2, 1.0, 1.9, 1.0, 'domain tools\n(pfam2go,\nInterProScan)', '#F5D9C4', fs_=6.5)
axA.text(8.4, 2.35, 'ALL isoforms →\nidentical gene-level GO', ha='center', fontsize=6.5, color=fs.OI['verm'])
axA.text(8.4, 0.7, 'PRISM →\nisoform-specific GO', ha='center', fontsize=6.5, color=fs.OI['blue'], fontweight='bold')
for yy in [2.2,1.55,0.9,0.25]:
    arrow(axA, 4.4, yy+0.21, 5.2, 1.5, c='#BBB', lw=0.7)
arrow(axA, 7.1, 1.7, 8.0, 2.2, c=fs.OI['verm'])
arrow(axA, 4.45, 0.46, 8.0, 0.7, c=fs.OI['blue'], lw=1.4)

# ---- (b) PRISM architecture (ML tensor-flow) ----
axB = fig.add_subplot(gs[1]); axB.set_xlim(0,10); axB.set_ylim(0,3); axB.axis('off')
fs.panel_label(axB, 'b', dx=0.0, dy=0.98)
axB.set_title('PRISM: layer-contrast architecture', loc='left', x=0.02)
box(axB, 0.2, 1.2, 1.15, 0.7, 'isoform\nAA seq', '#E8E8E8', fs_=6.5)
box(axB, 1.7, 0.9, 1.5, 1.3, 'ESM-2\n(frozen)\n30 layers\n640-d', '#CFE8DE', ec=fs.OI['green'], fs_=6.5)
arrow(axB, 1.35, 1.55, 1.7, 1.55)
# extract two layers
box(axB, 3.55, 1.75, 1.15, 0.5, r'$\phi_{L30}$ 640-d', '#D7E9F7', fs_=6.5)
box(axB, 3.55, 0.85, 1.15, 0.5, r'$\phi_{L15}$ 640-d', '#D7E9F7', fs_=6.5)
arrow(axB, 3.2, 1.7, 3.55, 2.0); arrow(axB, 3.2, 1.4, 3.55, 1.1)
# delta
box(axB, 5.0, 0.85, 1.35, 0.5, r'$\delta = \phi_{L30}-\phi_{L15}$', '#F7D7DE', ec=fs.OI['verm'], fs_=6.5, lw=1.1)
arrow(axB, 4.7, 1.95, 5.0, 1.2, c=fs.OI['verm']); arrow(axB, 4.7, 1.1, 5.0, 1.1, c=fs.OI['verm'])
# concat
box(axB, 6.6, 1.25, 1.2, 0.65, r'concat'+'\n'+r'$[\phi_{L30}\Vert\delta]$'+'\n1280-d', '#EAD7F2', fs_=6)
arrow(axB, 4.7, 2.0, 6.6, 1.75); arrow(axB, 6.35, 1.1, 6.6, 1.45, c=fs.OI['verm'])
# MLP
box(axB, 8.05, 1.15, 1.15, 0.85, 'MLP\n256→128→64\nfocal γ=2', '#CFE8DE', fs_=6)
arrow(axB, 7.8, 1.57, 8.05, 1.57)
axB.text(9.55, 1.57, 'GO\nprob.', ha='center', va='center', fontsize=6.5, fontweight='bold')
arrow(axB, 9.2, 1.57, 9.42, 1.57)
axB.text(5.05, 0.35, r'$\delta_{layer}$ recovers mid-layer signal that L30 mean-pooling suppresses',
         fontsize=6, color=fs.OI['verm'], style='italic')

# ---- (c) BISECT pipeline + 3-tier taxonomy ----
axC = fig.add_subplot(gs[2]); axC.set_xlim(0,10); axC.set_ylim(0,2.2); axC.axis('off')
fs.panel_label(axC, 'c', dx=0.0, dy=1.02)
axC.set_title('BISECT multi-evidence pipeline  &  3-tier encoding taxonomy', loc='left', x=0.02)
box(axC, 0.2, 0.8, 2.6, 0.9, 'BISECT: 15 modules\n(domain·PPI·NMD·AlphaFold·\nlocalization·convergence·…)', '#F5D9C4', fs_=6)
arrow(axC, 2.8, 1.25, 3.3, 1.25)
box(axC, 3.3, 0.8, 2.2, 0.9, '6-tier evidence\nhierarchy\n(A-DR … D)', '#F5E6C4', fs_=6.5)
# taxonomy
tx = [('(i) encoded &\nsurfaced @L30', '#CFE8DE', 'domain-level'),
      ('(ii) encoded but\nsuppressed', '#F7E7C4', 'δ_layer recovers'),
      ('(iii) not\nencoded', '#F2D0D0', 'SLiM / motif')]
for i,(t,c,sub) in enumerate(tx):
    box(axC, 6.05+i*1.32, 0.95, 1.2, 0.75, t, c, fs_=5.8)
    axC.text(6.65+i*1.32, 0.72, sub, ha='center', fontsize=5, color='#555')
axC.text(6.05, 1.9, 'ESM-2 encoding dissected (§4c):', fontsize=6, fontweight='bold')

fig.suptitle('Figure 1  |  PRISM + BISECT: isoform-resolution functional prediction framework',
             x=0.02, ha='left', fontsize=10, fontweight='bold', y=0.995)
p = os.path.join(OUT, 'F1_architecture')
fig.savefig(p+'.png'); fig.savefig(p+'.pdf')
print('saved', p+'.png')
