#!/usr/bin/env python3
"""
visualize_modules.py
=====================
672-term 기능 모듈 시각화 3종:

[1] UMAP: 672-dim score space, 44 모듈 컬러링 (고신뢰도 isoform만)
[2] Module coherence bar chart: within-module Jaccard per module
[3] Type × Module heatmap: Known/NIC/NNIC 비율 히트맵
"""
import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

BASE    = Path('/home/welcome1/sw1686/DIFFUSE')
REPORTS = BASE / 'reports'

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
S  = np.load(REPORTS / 'brain_full_672_scores.npy')   # (63994, 672)
df = pd.read_csv(REPORTS / 'brain_isoform_modules.tsv', sep='\t')

with open(REPORTS / 'brain_go_modules_672.json') as f:
    mod_data = json.load(f)
with open(REPORTS / 'module_validation_672.json') as f:
    val_data = json.load(f)
with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    go_meta = json.load(f)

modules = mod_data['modules']
mod_coh = val_data['module_coherence']
go_ids  = go_meta['go_ids']

# High-confidence mask
max_scores = S.max(axis=1)
df['max_score'] = max_scores
hi_mask = df['max_score'] > 0.3
df_hi = df[hi_mask].copy()
S_hi  = S[hi_mask.values]
print(f"  High-conf isoforms: {len(df_hi)}/{len(df)}")

# ── [1] UMAP on 672-dim score space ───────────────────────────────────────────
print("\n[1] Computing UMAP (672-dim → 2D)...")
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric='cosine', random_state=42, n_jobs=4,
                        low_memory=True)
    # Subsample for speed if needed
    n_sub = min(len(df_hi), 20000)
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(df_hi), size=n_sub, replace=False)
    S_sub = S_hi[idx_sub]
    df_sub = df_hi.iloc[idx_sub].copy()
    emb = reducer.fit_transform(S_sub)
    df_sub['umap_x'] = emb[:, 0]
    df_sub['umap_y'] = emb[:, 1]
    print(f"  UMAP done. Shape: {emb.shape}")
    umap_ok = True
except Exception as e:
    print(f"  UMAP failed: {e}. Trying TSNE...")
    umap_ok = False
    try:
        from sklearn.manifold import TSNE
        n_sub = min(len(df_hi), 10000)
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(len(df_hi), size=n_sub, replace=False)
        S_sub = S_hi[idx_sub]
        df_sub = df_hi.iloc[idx_sub].copy()
        emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(S_sub)
        df_sub['umap_x'] = emb[:, 0]
        df_sub['umap_y'] = emb[:, 1]
        print(f"  TSNE done.")
        umap_ok = True
    except Exception as e2:
        print(f"  TSNE also failed: {e2}")
        umap_ok = False

if umap_ok:
    # Module labels for colormap
    mod_labels = {int(k): v['label'][:25] for k, v in modules.items()}
    df_sub['mod_label'] = df_sub['primary_module'].map(mod_labels)

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Top 15 modules by size (rest grouped as 'Other')
        top_mods = df_sub['primary_module'].value_counts().head(15).index.tolist()
        df_sub['mod_plot'] = df_sub['primary_module'].apply(
            lambda x: f"M{x}: {mod_labels.get(x,'?')}" if x in top_mods else 'Other')

        fig = px.scatter(
            df_sub, x='umap_x', y='umap_y', color='mod_plot',
            hover_data={'isoform_id': True, 'gene': True, 'type': True,
                        'module_score': ':.3f', 'umap_x': False, 'umap_y': False},
            title=f'Brain Isoform Functional Module Landscape (n={len(df_sub):,}, 672 GO terms, 44 modules)',
            labels={'umap_x': 'UMAP-1', 'umap_y': 'UMAP-2', 'mod_plot': 'Module'},
            width=1100, height=700, opacity=0.5,
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(legend=dict(itemsizing='constant', font=dict(size=9)))
        umap_path = REPORTS / 'brain_module_umap_672.html'
        fig.write_html(str(umap_path))
        print(f"  Saved: {umap_path}")

        # Also colored by isoform type
        fig2 = px.scatter(
            df_sub, x='umap_x', y='umap_y', color='type',
            color_discrete_map={'known': '#2196F3', 'nic': '#FF9800', 'nnic': '#E91E63'},
            hover_data={'isoform_id': True, 'gene': True, 'primary_module': True,
                        'umap_x': False, 'umap_y': False},
            title=f'Brain Isoform Functional Landscape — colored by type (n={len(df_sub):,})',
            labels={'umap_x': 'UMAP-1', 'umap_y': 'UMAP-2'},
            width=1000, height=650, opacity=0.5,
        )
        fig2.update_traces(marker=dict(size=3))
        umap_type_path = REPORTS / 'brain_module_umap_bytype_672.html'
        fig2.write_html(str(umap_type_path))
        print(f"  Saved: {umap_type_path}")

    except ImportError:
        print("  plotly not available — saving coordinates as NPY")

    # Save UMAP coords for Streamlit
    np.save(REPORTS / 'brain_module_umap_coords.npy', emb)
    df_sub[['isoform_id','gene','type','primary_module','module_score','max_score']].to_csv(
        REPORTS / 'brain_module_umap_meta.tsv', sep='\t', index=False)
    print(f"  Saved UMAP coords + meta TSV")

# ── [2] Module coherence bar chart ────────────────────────────────────────────
print("\n[2] Module coherence bar chart...")
try:
    import plotly.graph_objects as go_plt

    mod_sizes = {k: v['size'] for k, v in modules.items()}
    data = sorted([(int(k), float(v), mod_sizes.get(k, 0),
                    modules.get(k, {}).get('label','?')[:35])
                   for k, v in mod_coh.items()], key=lambda x: -x[1])
    mids, cohs, sizes, labels = zip(*data)

    fig3 = go_plt.Figure(go_plt.Bar(
        x=[f"M{m}: {l}" for m, l in zip(mids, labels)],
        y=cohs,
        marker_color=[f'rgb({int(255*(1-c))},{int(200*c)},{int(100*c)})' for c in cohs],
        text=[f"n={s}" for s in sizes],
        textposition='outside',
    ))
    fig3.update_layout(
        title='GO-GO Module Semantic Coherence (annotation-based Jaccard)',
        xaxis_title='Module', yaxis_title='Mean within-module Jaccard',
        xaxis_tickangle=45, height=500, width=1200,
    )
    fig3.write_html(str(REPORTS / 'brain_module_coherence_bar.html'))
    print(f"  Saved: brain_module_coherence_bar.html")
except Exception as e:
    print(f"  Bar chart failed: {e}")

# ── [3] Type × Module fraction heatmap ────────────────────────────────────────
print("\n[3] Type × Module heatmap...")
try:
    import plotly.graph_objects as go_plt

    # Fraction of each type in each module (high-conf only)
    mod_ids_sorted = sorted(modules.keys(), key=lambda x: -modules[x]['size'])[:30]  # top 30 by size
    types = ['known', 'nic', 'nnic']
    bg = {t: len(df_hi[df_hi['type']==t])/len(df_hi) for t in types}

    # Relative enrichment: (obs - expected) / expected
    heatmap_z = []
    heatmap_text = []
    for t in types:
        row = []
        row_text = []
        bg_frac = bg[t]
        for mid_str in mod_ids_sorted:
            mid = int(mid_str)
            sub = df_hi[df_hi['primary_module'] == mid]
            n_t = len(sub[sub['type'] == t])
            n_total = len(sub)
            obs_frac = n_t / max(n_total, 1)
            rel = (obs_frac - bg_frac) / max(bg_frac, 1e-4)  # relative enrichment
            row.append(rel)
            row_text.append(f'{100*obs_frac:.1f}%<br>n={n_t}')
        heatmap_z.append(row)
        heatmap_text.append(row_text)

    mod_xlabels = [f"M{mid_str}: {modules[mid_str]['label'][:20]}" for mid_str in mod_ids_sorted]

    fig4 = go_plt.Figure(go_plt.Heatmap(
        z=heatmap_z, x=mod_xlabels, y=types,
        text=heatmap_text, texttemplate='%{text}',
        colorscale='RdBu', zmid=0, zmin=-0.5, zmax=0.5,
        colorbar=dict(title='Relative enrichment<br>(vs background)'),
    ))
    fig4.update_layout(
        title='Isoform Type × Module Enrichment (high-confidence, score>0.3)',
        xaxis_tickangle=45, height=350, width=1400,
        font=dict(size=10),
    )
    fig4.write_html(str(REPORTS / 'brain_module_type_heatmap.html'))
    print(f"  Saved: brain_module_type_heatmap.html")
except Exception as e:
    print(f"  Heatmap failed: {e}")

print("\nAll done.")
