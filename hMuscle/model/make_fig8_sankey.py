"""F8 — Whole-transcriptome isoform-function taxonomy (3-scheme Sankey).
520 GO-active multi-isoform genes (73-term panel, tau=0.5) flow through:
  scheme 2 (cardinality: single/multi-GO) -> scheme 1 (acquisition: shared/additional/
  divergent) -> scheme 3 (allocation: concentrated/distributed/mixed).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import fig_style as fs
fs.apply()
ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT = os.path.join(ROOT, 'reports/figures_v2')
SCR = '/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/79479b99-e4cd-447b-b678-41709c5db7dd/scratchpad'
D = json.load(open(os.path.join(SCR, 'f8_sankey.json')))
N = D['N']

# node order per column (top->bottom)
COLS = [
    ('cardinality', ['singleGO', 'multiGO'], D['c2']),
    ('acquisition mode', ['shared', 'additional', 'divergent', 'single-carrier'], D['c1']),
    ('allocation', ['distributed', 'mixed', 'concentrated'], D['c3']),
]
NLAB = {'singleGO': 'single-GO', 'multiGO': 'multi-GO',
        'shared': 'shared\n(redistribution)', 'additional': 'additional\n(layered)',
        'divergent': 'divergent\n(novel GO)', 'single-carrier': 'single-carrier',
        'concentrated': 'concentrated', 'mixed': 'mixed', 'distributed': 'distributed'}
NCOL = {'singleGO': fs.C_BASE, 'multiGO': fs.OI['blue'],
        'shared': fs.OI['green'], 'additional': fs.OI['orange'], 'divergent': fs.OI['verm'],
        'single-carrier': fs.C_BASE,
        'concentrated': fs.C_BASE, 'mixed': fs.OI['orange'], 'distributed': fs.OI['green']}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(-0.05, 3.20); ax.set_ylim(-0.14, 1.12); ax.axis('off')
GAP = 0.035; XW = 0.14
xpos = {0: 0.15, 1: 1.55, 2: 2.85}
# compute node y-positions (stacked, normalized to N with gaps)
node_span = {}   # (col,name) -> (y0,y1)
for ci, (title, order, counts) in enumerate(COLS):
    tot = sum(counts.get(n, 0) for n in order)
    ngap = (len(order) - 1) * GAP
    y = 1.0
    for n in order:
        h = counts.get(n, 0) / tot * (1.0 - ngap)
        node_span[(ci, n)] = (y - h, y)
        y -= h + GAP
    ax.text(xpos[ci] + XW / 2, 1.07, title, ha='center', fontsize=8, fontweight='bold')

# draw node bars + labels
for (ci, n), (y0, y1) in node_span.items():
    x = xpos[ci]
    ax.add_patch(plt.Rectangle((x, y0), XW, y1 - y0, fc=NCOL[n], ec='none'))
    cnt = COLS[ci][2].get(n, 0)
    lx = x - 0.02 if ci == 0 else x + XW + 0.02
    ha = 'right' if ci == 0 else 'left'
    ax.text(lx, (y0 + y1) / 2, f'{NLAB[n]}\n{cnt} ({cnt/N*100:.0f}%)', va='center', ha=ha, fontsize=6.2)

def ribbon(x0, y0a, y0b, x1, y1a, y1b, color):
    # cursors: left node segment [y0a,y0b] -> right [y1a,y1b]
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec='none', alpha=0.32))

def draw_flows(ci_l, ci_r, flowkey):
    flows = {}
    for k, v in D[flowkey].items():
        a, b = k.split('|'); flows[(a, b)] = v
    # track running offset within each node
    left_off = {n: node_span[(ci_l, n)][1] for n in COLS[ci_l][1]}
    right_off = {n: node_span[(ci_r, n)][1] for n in COLS[ci_r][1]}
    left_tot = {n: COLS[ci_l][2].get(n, 0) for n in COLS[ci_l][1]}
    right_tot = {n: COLS[ci_r][2].get(n, 0) for n in COLS[ci_r][1]}
    # order flows by left then right node order
    for ln in COLS[ci_l][1]:
        lh_full = node_span[(ci_l, ln)][1] - node_span[(ci_l, ln)][0]
        for rn in COLS[ci_r][1]:
            v = flows.get((ln, rn), 0)
            if not v: continue
            lh = v / left_tot[ln] * lh_full
            rh_full = node_span[(ci_r, rn)][1] - node_span[(ci_r, rn)][0]
            rh = v / right_tot[rn] * rh_full
            y0b = left_off[ln]; y0a = y0b - lh; left_off[ln] = y0a
            y1b = right_off[rn]; y1a = y1b - rh; right_off[rn] = y1a
            ribbon(xpos[ci_l] + XW, y0a, y0b, xpos[ci_r], y1a, y1b, NCOL[rn])

draw_flows(0, 1, 'f21')
draw_flows(1, 2, 'f13')

ax.text(0.5, -0.06, f'N = {N} GO-active multi-isoform genes  ·  73-term MF panel  ·  threshold 0.5',
        transform=ax.transAxes, ha='center', fontsize=6.5, color='#555')
fig.suptitle('Figure 8  |  A taxonomy of isoform-resolved function: GO is distributed across isoforms in 71% of genes (shared 54% + layered 20%), concentrated in a single carrier in 24%',
             x=0.02, ha='left', fontsize=7.6, fontweight='bold', y=1.0)
p = os.path.join(OUT, 'F8_sankey')
fig.savefig(p + '.png'); fig.savefig(p + '.pdf')
print('saved', p + '.png')
