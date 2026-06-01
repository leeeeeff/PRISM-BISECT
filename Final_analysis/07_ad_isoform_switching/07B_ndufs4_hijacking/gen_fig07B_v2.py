"""
Fig 7B v2 — NDUFS4 Locus Hijacking (Nature Methods style)
Updated with automated pipeline data: L1PA3/L1PA11 RepeatMasker (hg38),
RVT_1 hmmscan (E=4.6e-48), MTS composite scoring.

Layout (183 × 150 mm):
  Panel A (top):    Genomic locus schematic — tr73243 NAT + L1PA elements
  Panel B (left):   Protein domain comparison (NDUFS4-201 vs tr73243)
  Panel C (right):  DTU stacked bar (CT vs AD pseudobulk %)
  Panel D (far-r):  MTS feature comparison
"""

import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.75,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
MM = 1 / 25.4

# ── Colors ────────────────────────────────────────────────────────────────────
C_CT      = '#2166AC'   # Control (CT)
C_AD      = '#D73027'   # Alzheimer's (AD)
C_NAT     = '#E6550D'   # tr73243 NAT
C_CANON   = '#2166AC'   # canonical NDUFS4
C_RVT     = '#9B59B6'   # RVT_1 domain
C_L1PA3   = '#C0392B'   # L1PA3 (young, − strand)
C_L1PA11  = '#E74C3C'   # L1PA11 (young, + strand)
C_L2B     = '#BDC3C7'   # L2b (old, grey)
C_LINE    = '#E74C3C'
C_EXON    = '#E6550D'
C_EXON_CT = '#2166AC'
C_MTS_OK  = '#2166AC'
C_MTS_BAD = '#D73027'
C_GRAY    = '#AAAAAA'

# ── Data from pipeline analysis.json ──────────────────────────────────────────
# Genomic coordinates (hg38, chr5, + strand)
LOCUS_START = 53_555_000
LOCUS_END   = 53_695_000

TR73243_EXONS = [
    (53_560_626, 53_560_760),   # E1
    (53_603_452, 53_603_530),   # E2
    (53_604_724, 53_604_895),   # E3  ← L2b nearby
    (53_646_233, 53_646_405),   # E4
    (53_658_551, 53_658_649),   # E5
    (53_685_990, 53_688_219),   # E6  ← L1PA3+L1PA11
]

# NDUFS4 canonical on − strand: approximate model (9 exons, TSS ~53,688,219)
# Coordinates approximate from NCBI RefSeq NM_002495.3 (chr5, − strand)
NDUFS4_EXONS_APPROX = [
    (53_686_200, 53_688_219),   # Exon 1 (3'-most on + strand = 5'-most on − strand)
    (53_666_800, 53_666_980),   # Exon 2
    (53_655_200, 53_655_380),   # Exon 3
    (53_640_100, 53_640_230),   # Exon 4
    (53_618_900, 53_619_040),   # Exon 5
    (53_604_600, 53_604_750),   # Exon 6
    (53_594_800, 53_594_920),   # Exon 7
    (53_575_400, 53_575_500),   # Exon 8
    (53_560_500, 53_561_200),   # Exon 9 (5'-most on − strand)
]

REPEAT_ELEMENTS = [
    # E3 region
    {'name': 'L2b',    'class': 'LINE/L2',  'start': 53_604_430, 'end': 53_604_915,
     'strand': '+', 'pct_div': 33.5, 'young': False, 'color': C_L2B},
    # E6 region
    {'name': 'L1PA3',  'class': 'LINE/L1',  'start': 53_685_456, 'end': 53_686_732,
     'strand': '-', 'pct_div': 4.5,  'young': True,  'color': C_L1PA3},
    {'name': 'L1PA11', 'class': 'LINE/L1',  'start': 53_686_734, 'end': 53_689_784,
     'strand': '+', 'pct_div': 9.4,  'young': True,  'color': C_L1PA11},
]

CDS_START = 53_686_672   # tr73243 CDS start (within L1PA3 region)

# DTU pseudobulk proportions (from results_draft)
DTU_CT = {'NDUFS4-201': 44.1, 'tr73243': 5.2, 'Other': 50.7}
DTU_AD = {'NDUFS4-201':  7.1, 'tr73243': 42.9, 'Other': 50.0}

# MTS features
MTS = {
    'net_charge_30aa': {'canonical': '+2', 'tr73243': '−1'},
    'de_count_40aa':   {'canonical': '2',  'tr73243': '8'},
    'mu_H':            {'canonical': '0.28','tr73243': '0.33'},
    'HHH_motif':       {'canonical': 'absent', 'tr73243': 'aa 7–9 (blocks import)'},
    'LYR_motif':       {'canonical': 'present','tr73243': 'absent'},
    'MTS_score':       {'canonical': '3/5', 'tr73243': '1/5'},
    'MTS_pred':        {'canonical': 'MTS+', 'tr73243': 'LOW (cytoplasmic)'},
}

# Protein domains
NDUFS4_LENGTH = 175
TR73243_LENGTH = 378   # from analysis.json

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CONSOL  = '/home/welcome1/sw1686/DIFFUSE/Final_analysis/figures_consolidated/generated'

# ── Helper: coordinate → plot x ───────────────────────────────────────────────
def gc(pos, xlim=(LOCUS_START, LOCUS_END)):
    """Map genomic coordinate to [0, 1] within locus display window."""
    return (pos - xlim[0]) / (xlim[1] - xlim[0])


def draw_exon_block(ax, start, end, y, height, color, alpha=1.0, lw=0.5):
    xs = gc(start)
    xe = gc(end)
    rect = mpatches.Rectangle((xs, y - height/2), xe - xs, height,
                               facecolor=color, edgecolor='white',
                               linewidth=lw, alpha=alpha, zorder=4)
    ax.add_patch(rect)


def draw_intron_line(ax, exons_gc, y, color, lw=0.7):
    """Draw thin horizontal line between exons."""
    for i in range(len(exons_gc) - 1):
        x1 = gc(exons_gc[i][1])
        x2 = gc(exons_gc[i + 1][0])
        ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=2)


# ── MAIN FIGURE ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(183*MM, 150*MM))
gs  = GridSpec(2, 3, figure=fig,
               height_ratios=[1.15, 0.85],
               width_ratios=[1.65, 1.0, 1.0],
               hspace=0.45, wspace=0.40,
               left=0.07, right=0.97, top=0.95, bottom=0.07)

ax_locus  = fig.add_subplot(gs[0, :])   # Panel A: full-width locus
ax_domain = fig.add_subplot(gs[1, 0])   # Panel B: protein domains
ax_dtu    = fig.add_subplot(gs[1, 1])   # Panel C: DTU bar
ax_mts    = fig.add_subplot(gs[1, 2])   # Panel D: MTS features


# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Genomic locus schematic
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_locus
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1.0)
ax.axis('off')
ax.set_title('A', loc='left', fontweight='bold', fontsize=8, pad=3)

Y_TR73 = 0.72    # tr73243 (+strand) level
Y_CANON = 0.30   # NDUFS4 canonical (−strand) level
Y_REPEAT = 0.52  # repeat element band
H_EXON = 0.10
H_EXON_CANON = 0.08
H_REPEAT = 0.06

# Background chromosome band
ax.fill_between([0, 1], [0.50]*2, [0.54]*2,
                color='#F5F5F5', zorder=0)

# ── tr73243 exons (+ strand, NAT) ─────────────────────────────────────────────
# Intron lines
for i in range(len(TR73243_EXONS) - 1):
    x1 = gc(TR73243_EXONS[i][1])
    x2 = gc(TR73243_EXONS[i + 1][0])
    ax.plot([x1, x2], [Y_TR73, Y_TR73], color=C_NAT, lw=0.7, zorder=2, alpha=0.7)

# Exon blocks
for j, (es, ee) in enumerate(TR73243_EXONS):
    is_e6 = (j == 5)
    col = C_L1PA11 if is_e6 else C_NAT
    alpha = 1.0
    draw_exon_block(ax, es, ee, Y_TR73, H_EXON * (1.2 if is_e6 else 1.0),
                    col, alpha=alpha)
    # Exon label
    xm = gc((es + ee) / 2)
    ax.text(xm, Y_TR73 + H_EXON * 0.75, f'E{j+1}',
            ha='center', va='bottom', fontsize=4.5, color='#444444')

# tr73243 direction arrow (rightward, + strand)
ax.annotate('', xy=(gc(53_689_000), Y_TR73),
            xytext=(gc(53_560_000), Y_TR73),
            arrowprops=dict(arrowstyle='->', color=C_NAT, lw=1.0),
            annotation_clip=False)
ax.text(gc(53_558_000), Y_TR73, 'tr73243\n(+ strand)', ha='right', va='center',
        fontsize=5.5, color=C_NAT, fontweight='bold')
ax.text(gc(53_689_500), Y_TR73, '378 aa\n(NAT)', ha='left', va='center',
        fontsize=5.0, color=C_NAT)

# ── Repeat elements ───────────────────────────────────────────────────────────
for rep in REPEAT_ELEMENTS:
    xs, xe = gc(rep['start']), gc(rep['end'])
    col = rep['color']
    alpha = 0.85 if rep['young'] else 0.45
    lw_r = 1.0 if rep['young'] else 0.5
    rect = mpatches.Rectangle((xs, Y_REPEAT - H_REPEAT/2), xe - xs, H_REPEAT,
                               facecolor=col, edgecolor=col, linewidth=lw_r,
                               alpha=alpha, zorder=3)
    ax.add_patch(rect)
    # Label
    xm = (xs + xe) / 2
    yoff = H_REPEAT * (0.9 if rep['name'] == 'L1PA11' else -1.2)
    ax.text(xm, Y_REPEAT + yoff, rep['name'],
            ha='center', va='center', fontsize=4.8,
            color=col if rep['young'] else '#888888',
            fontweight='bold' if rep['young'] else 'normal')
    if rep['young']:
        ax.text(xm, Y_REPEAT + yoff - 0.06,
                f"{rep['pct_div']}% div.",
                ha='center', va='center', fontsize=4.0, color='#555555')

# Arrow for L1PA3 − strand ASP (pointing left, drives + strand transcription)
xm_l1pa3 = gc((53_685_456 + 53_686_732) / 2)
ax.annotate('ASP→\n(LINE-1 antisense\npromoter)',
            xy=(xm_l1pa3, Y_TR73 - H_EXON * 0.6),
            xytext=(xm_l1pa3 - 0.05, Y_TR73 - 0.18),
            fontsize=4.5, ha='center', color=C_L1PA3,
            arrowprops=dict(arrowstyle='->', color=C_L1PA3, lw=0.7))

# CDS start marker
xcds = gc(CDS_START)
ax.axvline(xcds, ymin=0.55, ymax=0.85, color='#333333', lw=0.8, linestyle='--', zorder=5)
ax.text(xcds, Y_TR73 - H_EXON * 0.9, 'CDS\nstart',
        ha='center', va='top', fontsize=4.5, color='#333333')

# ── NDUFS4 canonical (− strand) ───────────────────────────────────────────────
# Intron lines
for i in range(len(NDUFS4_EXONS_APPROX) - 1):
    x1 = gc(NDUFS4_EXONS_APPROX[i][0])   # leftmost of this exon
    x2 = gc(NDUFS4_EXONS_APPROX[i + 1][1])  # rightmost of next exon
    # Connect right end of lower exon to left end of upper exon
    xe_curr = gc(NDUFS4_EXONS_APPROX[i][0])
    xs_next = gc(NDUFS4_EXONS_APPROX[i + 1][1])
    ax.plot([xe_curr, xs_next], [Y_CANON, Y_CANON],
            color=C_CANON, lw=0.6, zorder=2, alpha=0.6)

for j, (es, ee) in enumerate(NDUFS4_EXONS_APPROX):
    draw_exon_block(ax, es, ee, Y_CANON, H_EXON_CANON, C_CANON, alpha=0.75)

# Direction arrow (leftward, − strand)
ax.annotate('', xy=(gc(53_558_000), Y_CANON),
            xytext=(gc(53_690_000), Y_CANON),
            arrowprops=dict(arrowstyle='->', color=C_CANON, lw=1.0),
            annotation_clip=False)
ax.text(gc(53_558_000), Y_CANON, 'NDUFS4-201\n(− strand)', ha='right', va='center',
        fontsize=5.5, color=C_CANON, fontweight='bold')
ax.text(gc(53_690_500), Y_CANON, '175 aa\n(canonical)', ha='left', va='center',
        fontsize=5.0, color=C_CANON)

# ── Scale bar ─────────────────────────────────────────────────────────────────
scale_len = 10_000
x_scale_s = gc(53_560_000)
x_scale_e = gc(53_560_000 + scale_len)
ax.plot([x_scale_s, x_scale_e], [0.03, 0.03], color='#333333', lw=1.2)
ax.plot([x_scale_s, x_scale_s], [0.01, 0.05], color='#333333', lw=1.0)
ax.plot([x_scale_e, x_scale_e], [0.01, 0.05], color='#333333', lw=1.0)
ax.text((x_scale_s + x_scale_e) / 2, -0.02, '10 kb',
        ha='center', va='top', fontsize=5.5)

# ── Strand indicator ──────────────────────────────────────────────────────────
ax.text(0.98, 0.95, 'chr5 (hg38)',
        transform=ax.transAxes, ha='right', va='top', fontsize=5.5, color='#555555')

# Legend
legend_els = [
    mpatches.Patch(color=C_NAT,    label='tr73243 exon'),
    mpatches.Patch(color=C_CANON,  alpha=0.75, label='NDUFS4-201 exon'),
    mpatches.Patch(color=C_L1PA3,  label='L1PA3 (4.5% div., −strand)'),
    mpatches.Patch(color=C_L1PA11, label='L1PA11 (9.4% div., +strand)'),
    mpatches.Patch(color=C_L2B,    alpha=0.45, label='L2b (33.5% div.)'),
]
ax.legend(handles=legend_els, loc='upper left', fontsize=4.8,
          framealpha=0.85, edgecolor='#cccccc',
          bbox_to_anchor=(0.0, 0.98))

# DIFFUSE stats annotation
ax.text(0.98, 0.08,
        'DIFFUSE Δ = −0.563\nDTU p = 3.62×10⁻⁶\nExcitatory neurons only',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=5.2,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3F3',
                  edgecolor=C_AD, alpha=0.9))


# ─────────────────────────────────────────────────────────────────────────────
# Panel B: Protein domain comparison
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_domain
ax.set_xlim(0, 410)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('B', loc='left', fontweight='bold', fontsize=8, pad=3)

H_PROT = 0.32
Y_CANON_PROT = 1.6
Y_NAT_PROT   = 0.5

def prot_rect(ax, aa_start, aa_end, y_c, h, color, label='', lw=0.7, alpha=1.0,
              fontsize=5.0):
    rect = mpatches.Rectangle((aa_start, y_c - h/2), aa_end - aa_start, h,
                               facecolor=color, edgecolor='white',
                               linewidth=lw, alpha=alpha, zorder=4)
    ax.add_patch(rect)
    if label:
        xm = (aa_start + aa_end) / 2
        ax.text(xm, y_c, label, ha='center', va='center',
                fontsize=fontsize, color='white', fontweight='bold')

# NDUFS4-201 backbone
ax.plot([0, NDUFS4_LENGTH], [Y_CANON_PROT, Y_CANON_PROT],
        color=C_CANON, lw=1.0, alpha=0.4, zorder=2)
prot_rect(ax, 0, NDUFS4_LENGTH, Y_CANON_PROT, H_PROT, C_CANON, alpha=0.2)
# MTS (approximate aa 1-30)
prot_rect(ax, 1, 30, Y_CANON_PROT, H_PROT, C_MTS_OK, label='MTS', alpha=0.9, fontsize=4.5)
# Complex I subunit body
prot_rect(ax, 30, 175, Y_CANON_PROT, H_PROT, '#4393C3', label='Complex I\nsubunit', alpha=0.8)
# LYR motif marker
ax.plot([80, 80], [Y_CANON_PROT - H_PROT*0.5, Y_CANON_PROT + H_PROT*0.9],
        color='#2ECC71', lw=1.5, zorder=5)
ax.text(80, Y_CANON_PROT + H_PROT, 'LYR', ha='center', va='bottom',
        fontsize=4.0, color='#2ECC71', fontweight='bold')
ax.text(-5, Y_CANON_PROT, 'NDUFS4-201\n175 aa', ha='right', va='center',
        fontsize=5.0, color=C_CANON, fontweight='bold')
ax.text(NDUFS4_LENGTH + 3, Y_CANON_PROT, f'MTS: 3/5\n✓ import',
        ha='left', va='center', fontsize=4.5, color=C_MTS_OK)

# tr73243 backbone
ax.plot([0, TR73243_LENGTH], [Y_NAT_PROT, Y_NAT_PROT],
        color=C_NAT, lw=1.0, alpha=0.4, zorder=2)
prot_rect(ax, 0, TR73243_LENGTH, Y_NAT_PROT, H_PROT, C_NAT, alpha=0.12)
# No-MTS region (HHH at 7-9)
prot_rect(ax, 0, 40, Y_NAT_PROT, H_PROT, C_MTS_BAD, alpha=0.7)
ax.text(20, Y_NAT_PROT, 'HHH\n(aa7–9)', ha='center', va='center',
        fontsize=4.0, color='white', fontweight='bold')
# RVT_1 domain (aa 141-366, E=4.6e-48)
prot_rect(ax, 141, 366, Y_NAT_PROT, H_PROT, C_RVT,
          label='RVT_1\naa 141–366', alpha=0.9)
ax.text(-5, Y_NAT_PROT, 'tr73243\n378 aa', ha='right', va='center',
        fontsize=5.0, color=C_NAT, fontweight='bold')
ax.text(TR73243_LENGTH + 3, Y_NAT_PROT,
        'MTS: 1/5\n✗ cytoplasmic',
        ha='left', va='center', fontsize=4.5, color=C_MTS_BAD)

# E-value annotation for RVT_1
ax.text(253, Y_NAT_PROT + H_PROT * 0.65, 'E = 4.6×10⁻⁴⁸',
        ha='center', va='bottom', fontsize=4.2, color=C_RVT)
ax.text(253, Y_NAT_PROT - H_PROT * 0.65, '(score = 149.7)',
        ha='center', va='top', fontsize=4.2, color=C_RVT)

# x-axis ruler
ax.plot([0, 380], [-0.15, -0.15], color='#888888', lw=0.7)
for tick in [0, 100, 200, 300, 378]:
    ax.plot([tick, tick], [-0.18, -0.12], color='#888888', lw=0.7)
    ax.text(tick, -0.22, str(tick), ha='center', va='top', fontsize=4.5)
ax.text(190, -0.38, 'Amino acid position', ha='center', va='top', fontsize=5.5)

# Legend for domains
leg_el = [
    mpatches.Patch(color=C_MTS_OK,  label='MTS (functional)'),
    mpatches.Patch(color=C_MTS_BAD, label='HHH motif (blocks import)'),
    mpatches.Patch(color='#4393C3', label='Complex I subunit'),
    mpatches.Patch(color=C_RVT,     label='RVT_1 (LINE-1 RT)'),
]
ax.legend(handles=leg_el, loc='upper left', fontsize=4.5, framealpha=0.85,
          bbox_to_anchor=(0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Panel C: DTU bar chart
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_dtu
ax.set_title('C', loc='left', fontweight='bold', fontsize=8, pad=3)

conditions = ['CT\n(n=2,847)', 'AD\n(n=3,192)']
x = np.array([0, 1])
w = 0.55

# Stacked bars
ct_vals = [DTU_CT['NDUFS4-201'], DTU_CT['tr73243'], DTU_CT['Other']]
ad_vals = [DTU_AD['NDUFS4-201'], DTU_AD['tr73243'], DTU_AD['Other']]

colors_stack = [C_CANON, C_NAT, '#DDDDDD']
labels_stack = ['NDUFS4-201', 'tr73243', 'Other isoforms']

bottoms_ct = [0, ct_vals[0], ct_vals[0] + ct_vals[1]]
bottoms_ad = [0, ad_vals[0], ad_vals[0] + ad_vals[1]]

for i, (ctv, adv, col, lbl) in enumerate(zip(ct_vals, ad_vals, colors_stack, labels_stack)):
    ax.bar(0, ctv, w, bottom=bottoms_ct[i], color=col,
           edgecolor='white', linewidth=0.5, label=lbl if i < 2 else None, zorder=3)
    ax.bar(1, adv, w, bottom=bottoms_ad[i], color=col,
           edgecolor='white', linewidth=0.5, zorder=3)

# Value labels
for xi, (vals, bots) in enumerate([(ct_vals, bottoms_ct), (ad_vals, bottoms_ad)]):
    for v, b, col in zip(vals[:2], bots[:2], colors_stack[:2]):
        if v > 5:
            ax.text(xi, b + v/2, f'{v:.1f}%', ha='center', va='center',
                    fontsize=5.5, color='white', fontweight='bold')

# Annotations for key changes
ax.annotate('', xy=(1, DTU_AD['tr73243']/2),
            xytext=(0, DTU_CT['tr73243']/2 + DTU_CT['NDUFS4-201']),
            arrowprops=dict(arrowstyle='->', color=C_NAT, lw=0.8,
                           connectionstyle='arc3,rad=0.3'))
ax.text(0.5, 35, '×8.3\nactivation',
        ha='center', va='center', fontsize=5.0, color=C_NAT, fontweight='bold')

ax.annotate('', xy=(1, DTU_AD['NDUFS4-201']/2),
            xytext=(0, DTU_CT['NDUFS4-201']/2),
            arrowprops=dict(arrowstyle='->', color=C_CANON, lw=0.8,
                           connectionstyle='arc3,rad=-0.3'))
ax.text(0.5, 25, '×6.2\ncollapse',
        ha='center', va='center', fontsize=5.0, color=C_CANON, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(conditions, fontsize=6.0)
ax.set_ylabel('Pseudobulk transcript usage (%)')
ax.set_ylim(0, 102)
ax.set_xlim(-0.45, 1.45)
ax.legend(fontsize=4.8, loc='upper right', framealpha=0.85)

# DTU significance
ax.text(0.5, 96, f'p = 3.62×10⁻⁶\n(Dirichlet-multinomial)',
        ha='center', va='top', fontsize=4.8,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0',
                  edgecolor=C_AD, alpha=0.9))


# ─────────────────────────────────────────────────────────────────────────────
# Panel D: MTS feature comparison
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_mts
ax.set_title('D', loc='left', fontweight='bold', fontsize=8, pad=3)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.05)

# Table header
row_h = 0.115
headers = ['Feature', 'NDUFS4-201', 'tr73243']
x_cols  = [0.0, 0.45, 0.78]
col_ws  = [0.43, 0.32, 0.25]

ax.text(0.5, 1.0, 'MTS Feature Analysis', ha='center', va='top',
        fontsize=6.0, fontweight='bold', color='#333333')

# Header row
for hdr, xc, cw in zip(headers, x_cols, col_ws):
    ax.text(xc + cw/2, 0.90, hdr, ha='center', va='center',
            fontsize=5.5, fontweight='bold', color='#333333')

ax.axhline(0.86, color='#888888', lw=0.7, xmin=0.0, xmax=1.0)

rows = [
    ('Net charge\n(first 30 aa)', '+2', '−1', True),
    ('D+E count\n(first 40 aa)', '2', '8', True),
    ('Hydrophobic\nmoment μH', '0.28', '0.33', False),
    ('HHH motif', 'absent', 'aa 7–9\n(disruptive)', True),
    ('LYR motif', 'present', 'absent', True),
    ('MTS score', '3/5', '1/5', True),
    ('Prediction', 'IMPORT', 'CYTOPLASMIC', True),
]

for i, (feat, canon_val, nat_val, bad_for_nat) in enumerate(rows):
    y_row = 0.82 - i * row_h
    bg_col = '#F8F8F8' if i % 2 == 0 else 'white'
    ax.fill_between([0, 1], [y_row - row_h/2, y_row - row_h/2],
                    [y_row + row_h/2, y_row + row_h/2],
                    color=bg_col, alpha=0.7, zorder=0)
    ax.text(x_cols[0] + col_ws[0]/2, y_row, feat,
            ha='center', va='center', fontsize=4.5, color='#444444')
    ax.text(x_cols[1] + col_ws[1]/2, y_row, canon_val,
            ha='center', va='center', fontsize=4.8,
            color=C_MTS_OK, fontweight='bold')
    nat_color = C_MTS_BAD if bad_for_nat else '#555555'
    ax.text(x_cols[2] + col_ws[2]/2, y_row, nat_val,
            ha='center', va='center', fontsize=4.5,
            color=nat_color, fontweight='bold' if bad_for_nat else 'normal')

ax.axhline(0.0, color='#888888', lw=0.5)

# MTS criterion note
ax.text(0.5, -0.04,
        'MTS criteria: charge≥+2, D+E≤3,\nμH≥0.12, no HHH, LYR present',
        ha='center', va='top', fontsize=4.0, color='#666666',
        style='italic')


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    out_path = os.path.join(OUT_DIR, f'fig_07B_ndufs4_hijacking_v2.{ext}')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    consol_path = os.path.join(CONSOL, f'fig_07B_ndufs4_hijacking_v2.{ext}')
    if os.path.exists(consol_path):
        os.remove(consol_path)
    import shutil
    shutil.copy2(out_path, consol_path)

print('Fig 07B v2 saved.')
plt.close(fig)
