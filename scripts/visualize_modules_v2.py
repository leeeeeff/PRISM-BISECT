#!/usr/bin/env python3
"""
visualize_modules_v2.py
========================
개선된 모듈 시각화 3종:

[1] GO-GO 상관 block heatmap (672×672, 모듈 순서 정렬 + 경계선)
    - within-module r=0.818 vs cross-module r=0.277 블록 구조 직접 시각화

[2] Module summary bubble chart
    - x=mean brain AUPRC, y=NIC+NNIC enrichment vs background
    - size=n_isoforms, color=module category (brain/general)

[3] Module × type enrichment horizontal bar chart
    - 각 모듈의 Known/NIC/NNIC 비율 (background=9% 점선)
    - 크기순 정렬 (top 25 모듈)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

BASE    = Path('/home/welcome1/sw1686/DIFFUSE')
REPORTS = BASE / 'reports'

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
R   = np.load(REPORTS / 'brain_go_corr_672.npy').astype(np.float32)
df  = pd.read_csv(REPORTS / 'brain_isoform_modules.tsv', sep='\t')
S   = np.load(REPORTS / 'brain_full_672_scores.npy')

with open(REPORTS / 'brain_go_modules_672.json') as f:
    mod_data = json.load(f)
with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    go_meta = json.load(f)
with open(REPORTS / 'brain_full_672_meta.json') as f:
    meta672 = json.load(f)

go_ids     = go_meta['go_ids']
go_names_d = go_meta['go_names']
modules    = mod_data['modules']
go_mod     = mod_data['go_module_map']  # go_id → mod_id
per_go     = {m['go']: m for m in meta672['per_go'] if m['auprc_brain']}

# High-confidence mask
max_scores = S.max(axis=1)
df['max_score'] = max_scores
df_hi = df[df['max_score'] > 0.3].copy()
bg_novel = len(df_hi[df_hi['type'].isin(['nic','nnic'])]) / len(df_hi)
print(f"  High-conf isoforms: {len(df_hi)}, background NIC+NNIC: {100*bg_novel:.1f}%")

import plotly.graph_objects as go
import plotly.express as px

# ── [1] GO-GO Correlation Block Heatmap ───────────────────────────────────────
print("\n[1] GO-GO correlation block heatmap...")

# Sort GO terms by module ID, then by within-module Pearson r
mod_ids_sorted = sorted(modules.keys(), key=int)
sorted_go_indices = []
mod_boundaries = []   # x positions of module boundaries
current_pos = 0

for mid_str in mod_ids_sorted:
    go_list = modules[mid_str]['go_ids']
    indices = [go_ids.index(g) for g in go_list if g in go_ids]
    sorted_go_indices.extend(indices)
    current_pos += len(indices)
    mod_boundaries.append(current_pos)

# Reorder R by sorted indices
R_sorted = R[np.ix_(sorted_go_indices, sorted_go_indices)]
n = len(sorted_go_indices)
print(f"  R_sorted shape: {R_sorted.shape}")

# Downsampled for plotly (max 200×200 for web performance)
step = max(1, n // 200)
R_ds = R_sorted[::step, ::step]
boundaries_ds = [b // step for b in mod_boundaries if b // step < R_ds.shape[0]]

fig1 = go.Figure(go.Heatmap(
    z=R_ds,
    colorscale='RdBu_r',
    zmin=-0.5, zmax=1.0,
    colorbar=dict(title='Pearson r', len=0.7),
    showscale=True,
))

# Add module boundary lines
for b in boundaries_ds[:-1]:
    fig1.add_shape(type='line', x0=b, x1=b, y0=0, y1=R_ds.shape[0]-1,
                   line=dict(color='black', width=0.5))
    fig1.add_shape(type='line', y0=b, y1=b, x0=0, x1=R_ds.shape[1]-1,
                   line=dict(color='black', width=0.5))

# Module labels at block centers
prev = 0
for mid_str, b in zip(mod_ids_sorted, boundaries_ds[:-1]):
    center = (prev + b) // 2
    label  = modules[mid_str]['label'].split('/')[0].strip()[:20]
    fig1.add_annotation(x=center, y=-5, text=f"M{mid_str}", showarrow=False,
                        font=dict(size=6), textangle=-45)
    prev = b

fig1.update_layout(
    title=dict(text='GO-GO Score Correlation Matrix (672 terms, 44 modules)<br>'
                    '<sub>Within-module mean r = 0.818; Cross-module mean r = 0.277</sub>',
               font=dict(size=14)),
    width=900, height=850,
    xaxis=dict(showticklabels=False, title='GO terms (sorted by module)'),
    yaxis=dict(showticklabels=False, title='GO terms (sorted by module)',
               autorange='reversed'),
    margin=dict(l=60, r=40, t=80, b=80),
)
fig1.write_html(str(REPORTS / 'brain_goterm_corr_heatmap.html'))
print(f"  Saved: brain_goterm_corr_heatmap.html")

# ── [2] Module Summary Bubble Chart ───────────────────────────────────────────
print("\n[2] Module summary bubble chart...")

# Brain-relevant module categories
brain_mods  = {13, 14, 36, 37, 11, 35, 12}
neuro_mods  = {36, 37, 13, 14, 35, 12}
ion_mods    = {11, 12}
immune_mods = {23, 24, 25}
cell_cycle  = {4, 34}

def get_category(mid):
    m = int(mid)
    if m in {36, 37}: return 'Neuronal development'
    if m in {13, 14}: return 'Synaptic / GPCR'
    if m in {11, 12}: return 'Ion transport'
    if m in {35}:     return 'Cell adhesion'
    if m in {23, 24, 25}: return 'Immune'
    if m in {4, 34}:  return 'Cell cycle'
    if m in {1, 2}:   return 'Transcription / DNA'
    if m in {10, 8}:  return 'RNA processing'
    return 'General'

bubble_data = []
for mid_str in modules.keys():
    mid = int(mid_str)
    go_list  = modules[mid_str]['go_ids']
    auprc_vals = [per_go[g]['auprc_brain'] for g in go_list if g in per_go]
    if not auprc_vals:
        continue
    mean_auprc = np.mean(auprc_vals)

    sub = df_hi[df_hi['primary_module'] == mid]
    if len(sub) < 10:
        continue
    novel_frac = len(sub[sub['type'].isin(['nic','nnic'])]) / len(sub)
    enrichment = (novel_frac - bg_novel) / bg_novel  # relative to background

    label  = modules[mid_str]['label'].split('/')[0].strip()[:30]
    bubble_data.append({
        'module_id':   mid,
        'label':       f"M{mid}: {label}",
        'mean_auprc':  mean_auprc,
        'enrichment':  enrichment,
        'n_isoforms':  len(sub),
        'n_go_terms':  len(go_list),
        'category':    get_category(mid_str),
        'novel_pct':   100 * novel_frac,
    })

bdf = pd.DataFrame(bubble_data)
print(f"  Modules in bubble chart: {len(bdf)}")

cat_colors = {
    'Neuronal development': '#1565C0',
    'Synaptic / GPCR':      '#0097A7',
    'Ion transport':         '#00897B',
    'Cell adhesion':         '#43A047',
    'Immune':                '#E53935',
    'Cell cycle':            '#F57F17',
    'Transcription / DNA':   '#6A1B9A',
    'RNA processing':        '#AD1457',
    'General':               '#9E9E9E',
}

fig2 = px.scatter(
    bdf,
    x='mean_auprc', y='enrichment',
    size='n_isoforms', color='category',
    color_discrete_map=cat_colors,
    text='label',
    hover_data={'module_id': True, 'n_isoforms': True, 'n_go_terms': True,
                'novel_pct': ':.1f', 'mean_auprc': ':.3f',
                'enrichment': ':.3f', 'label': False},
    size_max=40,
    title='Functional Module Summary: Predictability × Novel Isoform Enrichment<br>'
          '<sub>Bubble size = n isoforms assigned | y=0 → background NIC+NNIC rate (9.0%)</sub>',
)
fig2.add_hline(y=0, line_dash='dash', line_color='gray', annotation_text='background')
fig2.add_vline(x=0.4, line_dash='dot', line_color='lightgray',
               annotation_text='AUPRC=0.4')
fig2.update_traces(textposition='top center', textfont=dict(size=8))
fig2.update_layout(
    width=1000, height=650,
    xaxis_title='Mean Brain AUPRC (zero-shot prediction quality)',
    yaxis_title='NIC+NNIC enrichment vs background (relative)',
    legend=dict(title='Module category', font=dict(size=10)),
)
fig2.write_html(str(REPORTS / 'brain_module_bubble.html'))
print(f"  Saved: brain_module_bubble.html")

# ── [3] Module × Type enrichment bar chart ────────────────────────────────────
print("\n[3] Module × type enrichment bar chart (top 25 by isoform count)...")

bar_data = []
for mid_str in modules.keys():
    mid = int(mid_str)
    sub = df_hi[df_hi['primary_module'] == mid]
    if len(sub) < 20:
        continue
    for t in ['known', 'nic', 'nnic']:
        n_t = len(sub[sub['type'] == t])
        bar_data.append({
            'module': mid,
            'mod_label': f"M{mid}: {modules[mid_str]['label'].split('/')[0][:22]}",
            'type': t.upper(),
            'pct': 100 * n_t / len(sub),
            'n':  n_t,
            'n_total': len(sub),
        })

bdf3 = pd.DataFrame(bar_data)
# Sort modules by total isoform count (descending)
mod_order = df_hi['primary_module'].value_counts().head(25).index.tolist()
bdf3 = bdf3[bdf3['module'].isin(mod_order)].copy()
mod_label_order = [
    f"M{m}: {modules[str(m)]['label'].split('/')[0][:22]}"
    for m in mod_order if str(m) in modules
]

fig3 = px.bar(
    bdf3,
    y='mod_label', x='pct', color='type', barmode='stack',
    color_discrete_map={'KNOWN': '#2196F3', 'NIC': '#FF9800', 'NNIC': '#E91E63'},
    category_orders={'mod_label': mod_label_order},
    text='n',
    title='Isoform Type Distribution per Module (top 25, high-confidence isoforms only)<br>'
          f'<sub>Background: Known={100*(1-bg_novel):.0f}%, NIC+NNIC={100*bg_novel:.1f}%</sub>',
    labels={'pct': '% of isoforms', 'mod_label': 'Module'},
)
fig3.update_traces(textposition='inside', textfont=dict(size=8))
# Background reference line
fig3.add_vline(x=100*bg_novel, line_dash='dot', line_color='red',
               annotation_text=f'BG NIC+NNIC<br>{100*bg_novel:.0f}%',
               annotation_position='top right')
fig3.update_layout(
    width=950, height=750,
    xaxis_title='% of isoforms in module',
    yaxis_title='',
    yaxis=dict(autorange='reversed'),
    legend=dict(title='Isoform type', orientation='h', y=1.02),
)
fig3.write_html(str(REPORTS / 'brain_module_type_bar.html'))
print(f"  Saved: brain_module_type_bar.html")

# ── Print summary stats for sanity check ──────────────────────────────────────
print("\n=== Visual QC Summary ===")
print(f"[1] Block heatmap: {R_ds.shape[0]}×{R_ds.shape[1]} (downsampled {step}×), "
      f"{len(boundaries_ds)-1} module boundaries")
print(f"[2] Bubble chart: {len(bdf)} modules, AUPRC range [{bdf.mean_auprc.min():.3f}, {bdf.mean_auprc.max():.3f}]")
print(f"[3] Type bar: {len(bdf3.module.unique())} modules, "
      f"max novel enrichment: M{bdf3.loc[bdf3[bdf3.type.isin([\"NIC\",\"NNIC\"])].groupby(\"module\")[\"pct\"].sum().idxmax()]} ")

print("\nAll done.")
