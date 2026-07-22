#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_interpretability_figure.py

Publication figure for the PRISM/ESM-2 interpretability MAP (FRAMEWORK.md).
Four panels, one coherent visual language (colour = stage status; width/length = quantity):

  A  Information-flow cascade (B0->B5): domain vs non-domain as Sankey-style rivers narrowing
     and branching to labelled "sinks" at the stage each signal is lost. (schematic widths)
  B  The 8 joint-PCA axes as a compass + covariate PROBES (arrows) coloured by usage.
  C  Non-domain embedding-difference variance decomposition (sunburst): floor / reproducible
     (compositional-named / tail), with the amino-acid-composition refinement annotated.
  D  Description resolution dial (8-axis -> 640-linear -> non-linear) + the 3 diagnostic
     case-study isoforms placed by total axis displacement vs PRISM output reaction.

All numbers are from committed analyses (see annotations). Output: PNG + PDF at 300 dpi.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge, Rectangle, FancyBboxPatch
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from pathlib import Path

OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/model_interpretability_map/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette ----
C_DOM = '#2f6db0'      # domain (blue)
C_NDOM = '#e08a2b'     # non-domain (orange)
C_USED = '#2e8b57'     # used / passes (green)
C_PART = '#d6a800'     # partial (amber)
C_LOST = '#b0b0b0'     # lost / floor (grey)
C_FLOOR = '#9aa0a6'
C_TAIL = '#c98a3a'
C_COMP = '#e6c07a'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8})

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.20,
                      left=0.055, right=0.965, top=0.925, bottom=0.055)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

fig.suptitle("A cascade map of what ESM-2 / PRISM encodes, surfaces, uses, and cannot describe "
             "at isoform resolution", fontsize=14, fontweight='bold', y=0.975)


# ============================ Panel A : cascade Sankey ============================
def ribbon(ax, x0, x1, y_top, y_bot, w_top, w_bot, color, alpha=0.85, z=2):
    """filled ribbon between two stage levels; widths are half-widths in x."""
    verts = [(x0 - w_top, y_top), (x0 + w_top, y_top),
             (x1 + w_bot, y_bot), (x1 - w_bot, y_bot)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO]
    ax.add_patch(PathPatch(MPath(verts + [verts[0]], codes + [MPath.CLOSEPOLY]),
                           fc=color, ec='none', alpha=alpha, zorder=z))

stages = ['B0  Physical\n(splice change)', 'B1  Encoded\n(ESM-2 per-residue)',
          'B2  Pooled\n(mean-pool survives)', 'B3  Anchored\n(common direction)',
          'B4  Used\n(readout relies)', 'B5  Labelled\n(isoform-level GO)']
ys = np.linspace(9.2, 1.4, 6)
axA.set_xlim(0, 10); axA.set_ylim(-1.1, 10); axA.axis('off')
for i, (s, y) in enumerate(zip(stages, ys)):
    axA.axhspan(y - 0.42, y + 0.42, color='#f2f3f5', zorder=0)
    axA.text(0.05, y, s, va='center', ha='left', fontsize=7.6, color='#333', zorder=5)

# DOMAIN river (left), width ~ 69.8% -> stays wide until B5 where it splits to label-noise
xd = 3.4
wd = 1.15  # half-width proportional to 0.698
for i in range(5):
    axA.plot([], [])
ribbon(axA, xd, xd, ys[0], ys[1], wd, wd, C_DOM)
ribbon(axA, xd, xd, ys[1], ys[2], wd, wd, C_DOM)
ribbon(axA, xd, xd, ys[2], ys[3], wd, wd, C_DOM)
ribbon(axA, xd, xd, ys[3], ys[4], wd, wd, C_USED, alpha=0.9)
# B4->B5 : split into small labelled-clean (green) + big label-noise (grey)
ribbon(axA, xd, xd - 0.75, ys[4], ys[5], wd, 0.16, C_USED, alpha=0.9)      # clean label ~10%
ribbon(axA, xd, xd + 0.55, ys[4], ys[5], wd, 0.95, C_LOST, alpha=0.7)      # label-noise ~90%
axA.text(xd - 1.05, ys[5] - 0.05, 'labelled ~10%', ha='center', va='center', fontsize=6.6, color=C_USED)
axA.text(xd + 1.55, ys[5] - 0.05, 'label-noise\n~90% (Type-1)', ha='center', va='center', fontsize=6.6, color='#777')
axA.text(xd, ys[0] + 0.62, 'DOMAIN  69.8%', ha='center', fontsize=9, fontweight='bold', color=C_DOM)

# NON-DOMAIN river (right), ~30.2%, splits at B1 into MEASURED sub-streams:
#   N-terminal targeting 51.5%  (-> partial B4)   ;  structured-internal 48.1% (-> SLiM B2 / comp B3)
xn = 7.3
wn = 0.62
ribbon(axA, xn, xn, ys[0], ys[1], wn, wn, C_NDOM)            # enters full
axA.text(xn, ys[0] + 0.62, 'NON-DOMAIN  30.2%', ha='center', fontsize=9, fontweight='bold', color=C_NDOM)
xL, xR = 6.75, 7.85        # N-terminal (left) ; structured-internal (right)
# B1 split
ribbon(axA, xn, xL, ys[1], ys[1] - 0.01, wn, 0.34, C_NDOM)
ribbon(axA, xn, xR, ys[1], ys[1] - 0.01, wn, 0.30, C_NDOM)
# N-terminal stream: encoded -> partial B4 (targeting weakly used) -> unlabelled
ribbon(axA, xL, xL, ys[1], ys[2], 0.34, 0.34, C_NDOM)
ribbon(axA, xL, xL, ys[2], ys[3], 0.34, 0.30, C_NDOM)
ribbon(axA, xL, xL, ys[3], ys[4], 0.30, 0.22, C_PART, alpha=0.95)   # weakly used (targeting)
ribbon(axA, xL, xL, ys[4], ys[5], 0.22, 0.10, C_LOST, alpha=0.5)
axA.text(xL, ys[2] + 0.30, 'N-terminal\ntargeting 52%', ha='center', va='center',
         fontsize=6.3, color='#a5641a')
# structured-internal stream: loses SLiM at B2 (pooling), compositional at B3 (gene-indep)
ribbon(axA, xR, xR, ys[1], ys[2], 0.30, 0.20, C_NDOM)
ribbon(axA, xR + 0.15, xR + 1.75, ys[1] + 0.05, ys[2], 0.16, 0.08, C_LOST, alpha=0.6)  # SLiM peel
axA.text(xR + 1.85, ys[2] + 0.10, 'SLiM (~55% of\nstructured) pooling-lost',
         ha='left', va='center', fontsize=6.0, color='#888')
ribbon(axA, xR, xR, ys[2], ys[3], 0.20, 0.10, C_NDOM)
ribbon(axA, xR + 0.1, xR + 1.75, ys[2] + 0.02, ys[3], 0.12, 0.06, C_LOST, alpha=0.6)   # comp peel
axA.text(xR + 1.85, ys[3] + 0.10, 'compositional\ngene-indep. (B3)',
         ha='left', va='center', fontsize=6.0, color='#888')
ribbon(axA, xR, xR, ys[3], ys[4], 0.10, 0.05, C_LOST, alpha=0.5)
ribbon(axA, xR, xR, ys[4], ys[5], 0.05, 0.03, C_LOST, alpha=0.4)
axA.text(xR, ys[2] + 0.30, 'structured-\ninternal 48%', ha='center', va='center',
         fontsize=6.3, color='#a5641a')
axA.text(xL - 0.55, ys[5] - 0.05, 'un-labelled', ha='center', va='center', fontsize=6.2, color='#999')

axA.text(0.05, 9.95, 'A', fontsize=15, fontweight='bold', ha='left')
axA.text(5.0, -0.85, 'Information decreases downward. Domain/non-domain (69.8/30.2%) and non-domain '
         'sub-stream widths (N-terminal 52% / structured 48%) are MEASURED (brain); disorder-dominant '
         '<1% (an overlapping\nproperty, not a stream). Green = used, amber = partial, grey = lost.',
         ha='center', fontsize=6.1, color='#666')


# ============================ Panel B : axis compass + probes ============================
axB.set_xlim(-1.7, 1.7); axB.set_ylim(-1.55, 1.9); axB.set_aspect('equal'); axB.axis('off')
axB.text(-1.65, 1.78, 'B', fontsize=15, fontweight='bold', ha='left')
axB.text(0, 1.78, '8 joint-PCA axes (rays, length ∝ explained var) + covariate PROBES',
         ha='center', fontsize=8.2)
# axes: (label, evr, angle_deg) ; evr approximate
axes_info = [
    ('ax0  β-sheet / TM', 0.156, 90),
    ('ax1  LRR / Ig', 0.05, 45),
    ('ax2', 0.035, 0),
    ('ax3  DOMAIN', 0.025, 315),
    ('ax4', 0.02, 270),
    ('ax5  LENGTH', 0.014, 225),
    ('ax6  KRAB-ZNF', 0.03, 180),
    ('ax7', 0.018, 135),
]
maxevr = 0.156
def rlen(evr):
    return 0.32 + 0.55 * (evr / maxevr) ** 0.5
for lbl, evr, ang in axes_info:
    a = np.deg2rad(ang); r = rlen(evr)
    x, y = r * np.cos(a), r * np.sin(a)
    axB.plot([0, x], [0, y], color='#c9ccd1', lw=1.6, zorder=1)
    axB.scatter([x], [y], s=26, color='#8a8f98', zorder=2)
    ha = 'left' if np.cos(a) > 0.1 else ('right' if np.cos(a) < -0.1 else 'center')
    axB.text(x * 1.12, y * 1.12, lbl, ha=ha, va='center', fontsize=7.2, color='#333')
# probes: arrow from outer ring to the axis node, coloured by usage; label at a given anchor
def probe(ang, color, label, used_txt, r_axis, lab_xy):
    a = np.deg2rad(ang)
    x, y = r_axis * np.cos(a), r_axis * np.sin(a)
    r_start = r_axis + 0.55
    x0, y0 = r_start * np.cos(a), r_start * np.sin(a)
    axB.add_patch(FancyArrowPatch((x0, y0), (x * 1.03, y * 1.03), arrowstyle='-|>',
                  mutation_scale=12, color=color, lw=2.2, zorder=3))
    axB.text(lab_xy[0], lab_xy[1], f'{label}\n{used_txt}', ha='center', va='center',
             fontsize=6.4, color=color,
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.8), zorder=5)
r_ax3 = rlen(0.025)
r_ax0 = rlen(0.156)
probe(315, C_USED, 'domain probe', 'USED (ridge +11..50%)', r_ax3, (1.2, -1.2))
probe(90, C_LOST, 'disorder probe', 'encoded, NOT used (−3%)', r_ax0, (-1.0, 1.5))
# compositional probe -> dataset-wide (center), not an axis; draw as dashed to center
axB.add_patch(FancyArrowPatch((-1.15, -1.05), (-0.15, -0.15), arrowstyle='-|>', mutation_scale=11,
              color=C_LOST, lw=1.8, ls='--', zorder=3))
axB.text(-1.15, -1.16, 'compositional probe\nreal but gene-indep.,\nNOT used at output',
         ha='center', va='top', fontsize=6.4, color='#8a6a2a',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_LOST, lw=0.8))
axB.scatter([0], [0], s=40, color='#4a4a4a', zorder=4)
axB.text(0.08, 0.02, 'dataset-wide\norientation', fontsize=6.0, color='#666', va='center')


# ============================ Panel C : non-domain variance sunburst ============================
axC.set_xlim(-1.5, 1.9); axC.set_ylim(-1.4, 1.4); axC.set_aspect('equal'); axC.axis('off')
axC.text(-1.5, 1.32, 'C', fontsize=15, fontweight='bold', ha='left')
axC.text(0.2, 1.30, 'Non-domain Δ-embedding variance (brain, n=11,666)',
         ha='center', fontsize=8.2)
# ring 1: floor vs reproducible ; ring 2: compositional vs tail (within reproducible)
floor, repro = 0.453, 0.547
comp_named, tail = 0.184, 0.363
def ring(frac_list, colors, labels, r_in, r_out, start=90):
    ang = start
    for f, c, l in zip(frac_list, colors, labels):
        ext = 360 * f
        axC.add_patch(Wedge((0, 0), r_out, ang, ang + ext, width=r_out - r_in,
                            fc=c, ec='white', lw=1.5))
        mid = np.deg2rad(ang + ext / 2); rr = (r_in + r_out) / 2
        axC.text(rr * np.cos(mid), rr * np.sin(mid), l, ha='center', va='center',
                 fontsize=7.0, color='white', fontweight='bold')
        ang += ext
ring([floor, repro], [C_FLOOR, '#e9d3ad'], [f'FLOOR\n{floor:.0%}', f'reproducible\n{repro:.0%}'],
     0.0, 0.62)
# outer ring: split the reproducible arc into compositional-named + tail
ang = 90 + 360 * floor
for f, c, l in [(comp_named, C_COMP, f'comp.\nnamed\n{comp_named:.0%}'),
                (tail, C_TAIL, f'reproducible tail\n{tail:.0%}')]:
    ext = 360 * f
    axC.add_patch(Wedge((0, 0), 1.05, ang, ang + ext, width=0.4, fc=c, ec='white', lw=1.5))
    mid = np.deg2rad(ang + ext / 2)
    axC.text(0.84 * np.cos(mid), 0.84 * np.sin(mid), l, ha='center', va='center',
             fontsize=6.6, color='#5a4420', fontweight='bold')
    ang += ext
axC.text(0.2, -1.28,
         'Reproducible (gene-generalizing) = future-recoverable; floor = per-pair, not a pooling\n'
         'artefact (reproducible fraction does not rise with edit size, ρ=−0.40 n.s.).\n'
         'Full 20-aa composition names ~61% of reproducible; ~0.39 survives = structural/positional tail.',
         ha='center', fontsize=6.2, color='#555')


# ============================ Panel D : resolution dial + case study ============================
axD.set_xlim(0, 10); axD.set_ylim(0, 10); axD.axis('off')
axD.text(0.1, 9.7, 'D', fontsize=15, fontweight='bold', ha='left')
axD.text(5, 9.75, 'Description resolution (domain) + diagnostic isoform case study',
         ha='center', fontsize=8.2)
# --- resolution horizontal nested bar (top) ---
x0, x1, yb = 1.2, 8.8, 8.4
axD.add_patch(Rectangle((x0, yb), x1 - x0, 0.55, fc='#eef1f4', ec='#bbb', lw=0.8))
# fractions of decodable signal (AUROC anchors + McFadden 65.5%)
segs = [('8-axis\nR² 0.101 (65.5%)', 0.655, '#3d6fa5'),
        ('+640-linear', 0.24, '#7aa3cf'),
        ('+non-linear', 0.105, '#b9d0e8')]
xx = x0
for lbl, f, c in segs:
    w = (x1 - x0) * f
    axD.add_patch(Rectangle((xx, yb), w, 0.55, fc=c, ec='white', lw=1.0))
    axD.text(xx + w / 2, yb + 0.27, lbl, ha='center', va='center', fontsize=6.4,
             color='white' if c == '#3d6fa5' else '#333')
    xx += w
axD.text(x0, yb - 0.3, 'AUROC 0.71 → 0.84 → 0.89 ; McFadden R² ratio = 65.5% narratable by 8 axes',
         ha='left', fontsize=6.4, color='#555')

# --- case study scatter: x = total |dA| (axis displacement), y = PRISM #terms moved ---
axins = axD.inset_axes([0.1, 0.06, 0.85, 0.60])
cases = {
    'NDUFS4 (Type A: MTS/targeting, domain=0)': ([12.32, 8.54, 6.67], [122, 435, 51], C_DOM),
    'MAPT (Type B: N-term insert)':            ([11.77, 7.59, 7.37], [330, 107, 83], C_NDOM),
    'LRPPRC (Type C: same-domain ctrl)':       ([5.65, 5.52, 4.90], [95, 152, 119], C_USED),
}
for lbl, (dx, dy, c) in cases.items():
    axins.scatter(dx, dy, s=70, color=c, edgecolor='white', lw=1.0, label=lbl, zorder=3)
axins.set_xlabel('total axis displacement  |ΔA|  (SD units)', fontsize=7)
axins.set_ylabel('PRISM output terms moved (>0.05, /672)', fontsize=7)
axins.tick_params(labelsize=6.5)
axins.legend(fontsize=5.7, loc='upper left', framealpha=0.9)
axins.grid(alpha=0.25, lw=0.5)
axins.set_title('per-pair: output reaction tracks edit SIZE, not domain status; '
                'LRPPRC lowest displacement', fontsize=6.2, color='#555')

for ax in (axA, axB, axC, axD):
    for sp in ax.spines.values():
        sp.set_visible(False)

fig.savefig(OUT / 'interpretability_map.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'interpretability_map.pdf', bbox_inches='tight')
print(f"saved: {OUT/'interpretability_map.png'}")
print(f"saved: {OUT/'interpretability_map.pdf'}")
