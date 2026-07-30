"""
fig_style.py — Unified figure style for PRISM+BISECT NatComm manuscript.
NatComm × genomics × ML/DL conventions: sans-serif, lowercase bold panel labels,
colorblind-safe (Okabe-Ito), null/oracle as dashed reference lines.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OI = {
    'black':  '#000000',
    'orange': '#E69F00',
    'skyblue':'#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'verm':   '#D55E00',
    'purple': '#CC79A7',
    'grey':   '#999999',
}
# semantic roles
C_PRISM   = OI['blue']      # our model
C_PRISM2  = OI['verm']      # our model, secondary (v17f*/delta)
C_BASE    = OI['grey']      # baselines
C_ORACLE  = OI['black']     # oracle / homology ceiling
C_AD      = OI['verm']      # disease (warm)
C_CT      = OI['blue']      # control (cool)
C_NULL    = OI['grey']      # null reference
LAYER_BINS = {'Early': OI['skyblue'], 'Mid': OI['green'], 'Final': OI['orange']}
# 8 PCA axes fixed colors
AXCOL = [OI['blue'], OI['orange'], OI['green'], OI['verm'],
         OI['skyblue'], OI['purple'], OI['yellow'], OI['grey']]
AXLAB = ['ax0 soluble-β/TM', 'ax1 LRR/Ig', 'ax2 pro-turn', 'ax3 multidomain',
         'ax4 helix/cys', 'ax5 linker', 'ax6 KRAB-ZNF', 'ax7 acidic-helix']

def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,   # editable text in Illustrator
        'ps.fonttype': 42,
    })

def panel_label(ax, s, dx=-0.08, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='bottom', ha='right')

def sig_stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'
