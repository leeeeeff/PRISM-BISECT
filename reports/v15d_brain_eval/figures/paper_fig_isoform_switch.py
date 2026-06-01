#!/usr/bin/env python3
"""
Paper Figure: Novel isoform switches in Alzheimer's Disease
Three-panel figure: KIF21B / NDUFS4 / DLG1
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, FancyBboxPatch
import matplotlib.gridspec as gridspec

# ── Color palette ─────────────────────────────────────────────────────────────
COL_KNOWN   = '#4878CF'   # blue — known isoforms (KIF21B-201/203/204, DLG1 known)
COL_KNOWN2  = '#6BAED6'   # light blue — secondary known
COL_KNOWN3  = '#9ECAE1'   # lighter blue
COL_KNOWN4  = '#C6DBEF'   # palest blue
COL_NOVEL_AD = '#E6550D'  # orange-red — AD-enriched novel
COL_NOVEL_CT = '#74C476'  # green — CT-enriched novel
COL_NEUTRAL  = '#BDBDBD'  # gray — neutral / no sig change
COL_CANON    = '#2171B5'  # dark blue — canonical isoform
COL_HIGHLIGHT = '#FDD0A2' # pale orange — highlight background

# ── Data ──────────────────────────────────────────────────────────────────────
# KIF21B Excitatory neuron
kif21b_isos = [
    ('KIF21B-203',         0.297297, 0.368421, 'Known (motor domain, full)',  COL_KNOWN2),
    ('KIF21B-201',         0.162162, 0.276316, 'Known (canonical)',           COL_CANON),
    ('KIF21B-204',         0.189189, 0.000000, 'Known (truncated)',           COL_KNOWN3),
    ('tr293004\n(CT-only)', 0.351351, 0.000000, 'Novel NIC — MT-based movement (p=9.3e-8)',  COL_NOVEL_CT),
    ('tr292978\n(AD-only)', 0.000000, 0.355263, 'Novel NNIC — Autophagy (p=3.8e-6)',        COL_NOVEL_AD),
]

# NDUFS4 Excitatory neuron
ndufs4_isos = [
    ('NDUFS4-201\n(canonical)', 0.441176, 0.071429, 'Complex I subunit (MTS+LYR, 176aa)', COL_CANON),
    ('tr73433\n(NNIC)',          0.294118, 0.214286, 'Novel — non-coding assoc.',          COL_NEUTRAL),
    ('tr73272\n(NNIC)',          0.147059, 0.095238, 'Novel — alt CDS (153aa)',             COL_KNOWN3),
    ('tr73267\n(ISM)',           0.088235, 0.071429, 'Novel ISM — no MTS (95aa)',           COL_KNOWN2),
    ('tr73419\n(ISM)',           0.029412, 0.000000, 'Known ISM',                           COL_KNOWN4),
    ('tr73420\n(NNIC)',          0.000000, 0.047619, 'Novel NNIC',                          COL_NEUTRAL),
    ('tr73323\n(NNIC)',          0.000000, 0.071429, 'Novel NNIC',                          COL_NEUTRAL),
    ('tr73243\n(AD-only)',       0.000000, 0.428571, 'Novel NNIC — same TSS+7bp,\n379aa alt protein (p=3.6e-6)', COL_NOVEL_AD),
]

# DLG1 OPC
dlg1_isos = [
    ('tr319500\n(CT-dom)',  0.809524, 0.119048, 'Novel NNIC — Autophagy, 186aa (p=9.0e-10)', COL_NOVEL_CT),
    ('tr319160\n(ISM)',     0.142857, 0.047619, 'Novel ISM',                                  COL_KNOWN3),
    ('tr319159\n(NNIC)',    0.023810, 0.309524, 'Novel NIC — AD-enriched',                    COL_NOVEL_AD),
    ('DLG1-223',            0.000000, 0.190476, 'Known',                                      COL_KNOWN2),
    ('DLG1-256',            0.023810, 0.071429, 'Known',                                      COL_KNOWN4),
    ('Others',              0.000000, 0.261904, 'Other known/novel (5 isoforms)',              COL_NEUTRAL),
]

def stacked_bar_panel(ax, isos, title, cell_type, gene, show_xlabel=True):
    """Horizontal stacked bar chart, CT (top) vs AD (bottom)."""
    bar_h = 0.65
    x_ct = np.array([r[1] for r in isos])
    x_ad = np.array([r[2] for r in isos])

    ct_cumsum = np.zeros(1)
    ad_cumsum = np.zeros(1)

    for i, (label, ct, ad, desc, col) in enumerate(isos):
        # CT bar (y=1)
        ax.barh(1, ct, bar_h, left=ct_cumsum, color=col,
                edgecolor='white', linewidth=0.5)
        # AD bar (y=0)
        ax.barh(0, ad, bar_h, left=ad_cumsum, color=col,
                edgecolor='white', linewidth=0.5)
        # Label inside bar if wide enough
        if ct > 0.07:
            ax.text(ct_cumsum + ct/2, 1, f'{ct*100:.0f}%',
                    ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
        if ad > 0.07:
            ax.text(ad_cumsum + ad/2, 0, f'{ad*100:.0f}%',
                    ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
        ct_cumsum = ct_cumsum + ct
        ad_cumsum = ad_cumsum + ad

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['AD', 'CT'], fontsize=9, fontweight='bold')
    ax.set_xlabel('Isoform usage fraction', fontsize=9)
    ax.set_title(f'{gene}\n{cell_type}', fontsize=10, fontweight='bold', pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=r[4], label=r[0].replace('\n', ' ')) for r in isos]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, -0.18),
              fontsize=6.5, ncol=1, frameon=False,
              handlelength=1.0, handleheight=0.8)


def draw_ndufs4_domain(ax):
    """Linear protein domain schematic for NDUFS4 isoforms."""
    ax.set_xlim(0, 430)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title('NDUFS4 — Protein Architecture', fontsize=10, fontweight='bold', pad=6)

    isoforms = [
        # (label, y, segments: [(start, end, color, label)], note)
        ('NDUFS4-201\n(canonical, 176aa)', 3.8,
         [(0,  42,  '#2171B5', 'MTS'),
          (42, 100, '#4DAF4A', 'LYR motif'),
          (100,176, '#377EB8', 'Complex I\nbinding')],
         'Complex I subunit [CT dominant]', '#E8F4FD'),

        ('tr73327 (120aa)\nNNIC', 2.7,
         [(0,  42,  '#2171B5', 'MTS'),
          (42, 120, '#4DAF4A', 'LYR')],
         'Truncated — dominant negative?', '#F0F9F0'),

        ('tr73267 (95aa)\nISM, no MTS', 1.6,
         [(0,  95,  '#FDAE6B', 'Post-MTS\nfragment')],
         'Cytoplasmic — proteasome target', '#FFF5EB'),

        ('tr73243 (379aa)\nNNIC, TSS+7bp', 0.5,
         [(0, 379, '#E6550D', 'Novel protein\n(no MTS, no LYR)')],
         'Completely different protein [AD dominant]', '#FEE0D2'),
    ]

    scale = 380 / 176  # scale to canonical

    for label, y, segs, note, bgcol in isoforms:
        # Background box
        max_aa = max(s[1] for s in segs)
        w = max_aa * scale / 380 * 380
        bg = FancyBboxPatch((0, y-0.28), w + 5, 0.56,
                            boxstyle='round,pad=0.01',
                            facecolor=bgcol, edgecolor='none', zorder=1)
        ax.add_patch(bg)

        # Protein backbone line
        ax.plot([0, w], [y, y], color='#999999', lw=1.5, zorder=2)

        for s_start, s_end, col, s_label in segs:
            x0 = s_start * scale / 380 * 380
            x1 = s_end   * scale / 380 * 380
            rect = FancyBboxPatch((x0, y-0.2), x1-x0, 0.4,
                                  boxstyle='round,pad=0.01',
                                  facecolor=col, edgecolor='white',
                                  linewidth=0.5, zorder=3)
            ax.add_patch(rect)
            if (x1-x0) > 20:
                ax.text((x0+x1)/2, y, s_label,
                        ha='center', va='center', fontsize=5.5,
                        color='white', fontweight='bold', zorder=4)

        # Isoform label
        ax.text(-5, y, label, ha='right', va='center', fontsize=7.5,
                color='#333333')
        # Note on right
        ax.text(390, y, note, ha='left', va='center', fontsize=7,
                color='#555555', style='italic')

    # Amino acid axis
    for aa in [0, 50, 100, 150, 200, 250, 300, 350]:
        x = aa * scale / 380 * 380
        ax.plot([x, x], [-0.35, -0.2], color='#AAAAAA', lw=0.7)
        ax.text(x, -0.45, str(aa), ha='center', va='top', fontsize=6, color='#777777')
    ax.text(190, -0.65, 'Amino acid position', ha='center', va='top',
            fontsize=8, color='#555555')

    # Genomic position annotation for tr73243
    ax.annotate('', xy=(0, 0.5), xytext=(0, 3.8),
                arrowprops=dict(arrowstyle='<->', color='#888888', lw=0.8))
    ax.text(-42, 2.15, 'Same\nTSS\n(+7bp)', ha='center', va='center',
            fontsize=6, color='#888888')


# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3,
                       width_ratios=[1, 1.1, 1.4],
                       height_ratios=[1, 1],
                       hspace=0.62, wspace=0.45,
                       left=0.06, right=0.97, top=0.93, bottom=0.08)

# Panel a: KIF21B (top-left)
ax_kif = fig.add_subplot(gs[0, 0])
stacked_bar_panel(ax_kif, kif21b_isos,
                  'Isoform usage fraction',
                  'Excitatory neuron', 'KIF21B')
ax_kif.text(-0.15, 1.05, 'a', transform=ax_kif.transAxes,
            fontsize=14, fontweight='bold', va='top')

# Panel b: NDUFS4 DTU (top-middle)
ax_ndu_dtu = fig.add_subplot(gs[0, 1])
stacked_bar_panel(ax_ndu_dtu, ndufs4_isos,
                  'Isoform usage fraction',
                  'Excitatory neuron', 'NDUFS4')
ax_ndu_dtu.text(-0.15, 1.05, 'b', transform=ax_ndu_dtu.transAxes,
                fontsize=14, fontweight='bold', va='top')

# Panel c: NDUFS4 domain diagram (top-right, spans 2 rows)
ax_domain = fig.add_subplot(gs[:, 2])
draw_ndufs4_domain(ax_domain)
ax_domain.text(-0.08, 1.02, 'c', transform=ax_domain.transAxes,
               fontsize=14, fontweight='bold', va='top')

# Panel d: DLG1 OPC (bottom-left)
ax_dlg1 = fig.add_subplot(gs[1, 0])
stacked_bar_panel(ax_dlg1, dlg1_isos,
                  'Isoform usage fraction',
                  'OPC', 'DLG1')
ax_dlg1.text(-0.15, 1.05, 'd', transform=ax_dlg1.transAxes,
             fontsize=14, fontweight='bold', va='top')

# Genomic note for tr73243 (below panel b)
ax_ndu_dtu.text(0.5, -0.58,
    'tr73243: TSS within 7bp of canonical NDUFS4\nCDS at chr5:53,686,672–53,687,808 (3\' locus, 6 novel exons)',
    ha='center', va='top', transform=ax_ndu_dtu.transAxes,
    fontsize=7, color='#E6550D', style='italic')

# ── Figure title & subtitle ────────────────────────────────────────────────────
fig.suptitle('Novel isoform switching in Alzheimer\'s disease — brain long-read scRNA-seq',
             fontsize=12, fontweight='bold', y=0.98)

# Bottom caption
fig.text(0.5, 0.01,
         'AD: Alzheimer\'s Disease  |  CT: Control  |  NNIC: Novel Not In Catalog  '
         '|  ISM: Incomplete Splice Match\n'
         'Novel isoforms identified by IsoQuant (Samsung AD cohort); '
         'DTU: Dirichlet-multinomial test (chi_padj shown)',
         ha='center', va='bottom', fontsize=7.5, color='#666666')

out = '/home/welcome1/sw1686/DIFFUSE/reports/v15d_brain_eval/figures/paper_fig_isoform_switch.pdf'
out_png = out.replace('.pdf', '.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
print(f'Saved: {out_png}')
plt.close()
