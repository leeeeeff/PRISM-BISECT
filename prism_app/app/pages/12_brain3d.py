"""
뇌 세포 3D 지도 — Brain Cell Atlas (3D)
Views:
  A) 해부학적 뇌 지도 — 실제 MNI pial mesh (fsaverage5) + 소뇌/뇌간
  B) 세포 스캐터 (UMAP)
  C) 지식 그래프 (클러스터 노드)
Phase 3 오버레이: AD vs CT Δ%, 유전자 강조, 레이어 L1→WM 애니메이션
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy.spatial import distance as sdist

cfg = st.session_state.get('cfg', {})

# ── 경로 ──────────────────────────────────────────────────────────────────────
_BRAIN_DATA  = Path(__file__).parents[2] / 'data' / 'brain'
_CELL_DATA   = Path(__file__).parents[2] / 'data' / 'cell_atlas'
ANAT_CSV    = _BRAIN_DATA / 'brain_3d_anatomical_coords.csv'
UMAP_CSV    = _BRAIN_DATA / 'brain_3d_coords.csv'
SUBTYPE_TSV = _CELL_DATA  / 'final_subtype_classification.tsv'
MESH_DIR    = _BRAIN_DATA / 'meshes'

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
LAYER_COLORS = {
    'L1': '#ff4444', 'L1-2': '#ff7744', 'L1-3': '#ff9944',
    'L2/3': '#ffbb44', 'L2-3': '#ffcc66', 'L2-4': '#ffee88',
    'L4': '#eeee44', 'L3-5': '#88cc44', 'L5': '#44cc88',
    'L5/6': '#44bbcc', 'L6': '#4488ee', 'WM/L6': '#8844ee',
    'WM': '#cc44ee', 'Mixed': '#aaaaaa', 'Perivasc': '#ff88bb',
    'Meningeal': '#ff44ff', 'OPC': '#f39c12', 'Microglia': '#1abc9c',
}
CELL_TYPE_COLORS = {
    'Excitatory neuron': '#e74c3c', 'Excitatory': '#e74c3c',
    'Inhibitory neuron': '#3498db', 'Inhibitory': '#3498db',
    'Oligodendrocyte': '#2ecc71', 'OPC': '#f39c12',
    'Astrocyte': '#9b59b6', 'Microglia': '#1abc9c',
    'Vascular cell': '#e67e22', 'Lymphocyte': '#95a5a6',
}
CONDITION_COLORS = {'AD': '#e74c3c', 'Control': '#3498db', 'Active control': '#f39c12'}
SIG_MARKS = {'***': '★★★', '**': '★★', '*': '★', '†': '◆', 'ns': ''}

REGION_META = {
    'lh_cerebrum': {'color': '#5577cc', 'opacity': 0.10, 'label': '대뇌 좌반구'},
    'rh_cerebrum': {'color': '#3355aa', 'opacity': 0.10, 'label': '대뇌 우반구'},
    'cerebellum':  {'color': '#ee8833', 'opacity': 0.22, 'label': '소뇌'},
    'midbrain':    {'color': '#44cc88', 'opacity': 0.38, 'label': '중뇌'},
    'pons':        {'color': '#33bb77', 'opacity': 0.38, 'label': '교뇌'},
    'medulla':     {'color': '#22aa66', 'opacity': 0.38, 'label': '연수'},
}
LOBE_COLORS = {
    'Frontal': '#4488dd', 'Parietal': '#dd6633',
    'Temporal': '#33cc77', 'Occipital': '#cc44cc', 'Other': '#888888',
}
LOBE_INT   = {'Frontal': 0, 'Parietal': 1, 'Temporal': 2, 'Occipital': 3, 'Other': 4}
# stepped colorscale: 5 discrete bands (0→1 normalized)
LOBE_CS = [
    [0.00, '#4488dd'], [0.19, '#4488dd'],
    [0.20, '#dd6633'], [0.39, '#dd6633'],
    [0.40, '#33cc77'], [0.59, '#33cc77'],
    [0.60, '#cc44cc'], [0.79, '#cc44cc'],
    [0.80, '#888888'], [1.00, '#888888'],
]

DEPTH_ORDER = {
    'Meningeal': -1, 'L1': 0, 'L1-2': 1, 'L1-3': 2, 'L2/3': 3,
    'L2-3': 4, 'L2-4': 5, 'L4': 6, 'L3-5': 7, 'L5': 8,
    'L5/6': 9, 'L6': 10, 'Mixed': 11, 'Perivasc': 12, 'WM/L6': 13, 'WM': 14,
}

# ── 피질 층 경계 쉘 파라미터 (단면 뷰용) ──────────────────────────────────────
BRAIN_A, BRAIN_B, BRAIN_C = 65, 87, 60  # MNI 타원체 반축 (mm)
LAYER_SHELLS = [
    ('L1/L2', 0.20, '#ff9999'),
    ('L2/L3', 0.50, '#ffcc99'),
    ('L3/L4', 1.20, '#ffff99'),
    ('L4/L5', 1.80, '#99ff99'),
    ('L5/L6', 2.70, '#99ccff'),
    ('L6/WM', 3.50, '#cc99ff'),
]

# ── 영역 레이블 3D 위치 (MNI mm 근사) ─────────────────────────────────────────
REGION_LABELS = {
    'lh_cerebrum': (-38, 15, 52, '대뇌 좌반구'),
    'rh_cerebrum': (38, 15, 52, '대뇌 우반구'),
    'cerebellum':  (0, -95, -15, '소뇌'),
    'midbrain':    (12, -22, -8, '중뇌/교뇌/연수'),
}

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(ANAT_CSV if ANAT_CSV.exists() else UMAP_CSV)
    sub = pd.read_csv(SUBTYPE_TSV, sep='\t')
    sub['cluster'] = sub['cluster'].astype(str)
    df['leiden'] = df['leiden'].astype(str)
    # 'subtype' already in df; drop before merge to avoid _x/_y duplication
    df = df.drop(columns=['subtype'], errors='ignore')
    sub_meta = sub[['cluster', 'delta', 'p', 'sig', 'n', 'markers', 'subtype']].copy()
    df = df.merge(sub_meta, left_on='leiden', right_on='cluster', how='left')
    return df, sub


@st.cache_data(show_spinner=False)
def load_brain_meshes():
    if not MESH_DIR.exists():
        return None
    meshes = {}
    for name in ['lh_pial', 'rh_pial', 'cerebellum', 'midbrain', 'pons', 'medulla']:
        fp = MESH_DIR / f'{name}.npz'
        if fp.exists():
            d = np.load(fp)
            meshes[name] = {'v': d['vertices'], 'f': d['faces']}
    for name in ['lh_lobe_labels', 'rh_lobe_labels']:
        fp = MESH_DIR / f'{name}.npy'
        if fp.exists():
            meshes[name] = np.load(fp, allow_pickle=True)
    return meshes


# ── 피질 층 경계 쉘 트레이스 (단면 뷰용) ─────────────────────────────────────
_SHELL_U = np.linspace(0, 2 * np.pi, 42)
_SHELL_V = np.linspace(0, np.pi, 30)
_SHELL_UU, _SHELL_VV = np.meshgrid(_SHELL_U, _SHELL_V, indexing='ij')


def _build_layer_shell_traces(x_cut=None, direction='lh', opacity=0.07):
    """Return list of go.Surface traces for cortical layer shells (no fig mutation)."""
    UU, VV = _SHELL_UU, _SHELL_VV
    traces = []
    for name, depth, color in LAYER_SHELLS:
        denom = np.sqrt(
            (np.sin(VV) * np.cos(UU) / BRAIN_A) ** 2 +
            (np.sin(VV) * np.sin(UU) / BRAIN_B) ** 2 +
            (np.cos(VV) / BRAIN_C) ** 2
        )
        r = np.clip(1.0 / denom - depth, 0.1, None)
        xs = r * np.sin(VV) * np.cos(UU)
        ys = r * np.sin(VV) * np.sin(UU)
        zs = r * np.cos(VV)
        if x_cut is not None:
            if direction == 'lh':
                zs = np.where(xs >= x_cut, zs, np.nan)   # show medial interior (LH)
            else:
                zs = np.where(xs <= x_cut, zs, np.nan)   # show medial interior (RH)
        traces.append(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=opacity,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            hoverinfo='none',
            name=name,
            showlegend=True,
            legendgroup='layer_shells',
        ))
    return traces


def add_layer_shell_traces(fig, x_cut=None, direction='lh', opacity=0.07):
    """Ellipsoid-based cortical layer boundary shells."""
    for t in _build_layer_shell_traces(x_cut=x_cut, direction=direction, opacity=opacity):
        fig.add_trace(t)


# ── 엽 경계선 (lobe_mode 전용) ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_lobe_boundaries():
    """Vectorized boundary-edge detection between lobe regions."""
    meshes = load_brain_meshes()
    if not meshes:
        return {}
    result = {}
    for hemi in ['lh', 'rh']:
        mk, lk = f'{hemi}_pial', f'{hemi}_lobe_labels'
        if mk not in meshes or lk not in meshes:
            continue
        v = meshes[mk]['v']
        f = meshes[mk]['f']
        labels = meshes[lk]
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        es = np.sort(edges, axis=1)
        l0, l1 = labels[es[:, 0]], labels[es[:, 1]]
        bnd_edges = np.unique(es[l0 != l1], axis=0)
        v0 = v[bnd_edges[:, 0]]
        v1 = v[bnd_edges[:, 1]]
        nans = np.full((len(bnd_edges), 1, 3), np.nan)
        segs = np.concatenate([v0[:, None], v1[:, None], nans], axis=1).reshape(-1, 3)
        result[hemi] = {'x': segs[:, 0].tolist(), 'y': segs[:, 1].tolist(), 'z': segs[:, 2].tolist()}
    return result


def _build_cut_plane_cap_trace(meshes, x_cut, opacity=0.22):
    """Return go.Mesh3d cap trace at X=x_cut, or None if hull fails."""
    from scipy.spatial import ConvexHull
    v_all = np.vstack([meshes['lh_pial']['v'], meshes['rh_pial']['v']])
    near = v_all[np.abs(v_all[:, 0] - x_cut) < 5.0]
    if len(near) < 6:
        return None
    yz = near[:, 1:]
    try:
        hull = ConvexHull(yz)
        hv = yz[hull.vertices]
    except Exception:
        return None
    n = len(hv)
    yc, zc = hv[:, 0].mean(), hv[:, 1].mean()
    vx = np.full(n + 1, float(x_cut))
    vy = np.append(hv[:, 0], yc)
    vz = np.append(hv[:, 1], zc)
    faces = np.array([[i, (i + 1) % n, n] for i in range(n)])
    return go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='#6699bb', opacity=opacity,
        showscale=False, hoverinfo='none',
        name='단면 캡', showlegend=False, flatshading=True,
    )


def add_cut_plane_cap(fig, meshes, x_cut, opacity=0.22):
    """Convex-hull-shaped semi-transparent cap at the cut plane X=x_cut."""
    t = _build_cut_plane_cap_trace(meshes, x_cut, opacity=opacity)
    if t is not None:
        fig.add_trace(t)


# ── Phase A: PRISM score 로드/집계 ────────────────────────────────────────────
_REPORTS   = Path(__file__).parents[3] / 'reports'
ISO_TSV    = _REPORTS / 'brain_isoform_modules.tsv'
SCORES_NPY = _REPORTS / 'brain_full_672_scores.npy'   # 165MB — not in git; optional
PRISM_META = _REPORTS / 'brain_full_672_meta.json'


@st.cache_data(show_spinner=False)
def load_prism_data():
    if not all(p.exists() for p in [ISO_TSV, SCORES_NPY, PRISM_META]):
        return None, None, None
    iso_df = pd.read_csv(ISO_TSV, sep='\t')
    scores = np.load(SCORES_NPY, mmap_mode='r')
    with open(PRISM_META) as f:
        meta = json.load(f)
    return iso_df, scores, meta


@st.cache_data(show_spinner=False)
def build_cluster_prism_scores(go_id, _iso_df, _scores_col):
    """Per-cluster average PRISM score for one GO term column (1-D array)."""
    import re
    _iso_df = _iso_df.copy()
    _iso_df['_ps'] = _scores_col
    gene_scores = _iso_df.groupby('gene')['_ps'].mean().to_dict()
    sub_df = pd.read_csv(SUBTYPE_TSV, sep='\t')
    result = {}
    for _, row in sub_df.iterrows():
        genes = re.findall(r'([A-Z][A-Z0-9]+)=', str(row.get('markers', '')))
        vals = [gene_scores[g] for g in genes if g in gene_scores]
        result[str(row['cluster'])] = float(np.mean(vals)) if vals else np.nan
    return result


# ── 뇌 메시 트레이스 ──────────────────────────────────────────────────────────
def add_brain_traces(fig, meshes, show_regions, lobe_mode, opacity_scale=1.0,
                     x_cut=None, show_labels=False, show_lobe_boundaries=False):
    """
    add Mesh3d traces for each brain region.
    x_cut: sagittal cross-section cut position (mm).
      - LH: keep faces with centroid_x < x_cut  (lateral portion; window at x_cut~0)
      - RH: keep faces with centroid_x > x_cut  (lateral portion; window at 0~x_cut)
      - x_cut = None: no cut (full brain)
    """
    for region in ['lh_cerebrum', 'rh_cerebrum', 'cerebellum', 'midbrain', 'pons', 'medulla']:
        if region not in show_regions or region not in meshes:
            continue
        mesh_key = 'lh_pial' if region == 'lh_cerebrum' else ('rh_pial' if region == 'rh_cerebrum' else region)
        if mesh_key not in meshes:
            continue
        v = meshes[mesh_key]['v']
        f = meshes[mesh_key]['f']
        meta = REGION_META[region]
        opacity = meta['opacity'] * opacity_scale

        # ── 시상면 단면 페이스 필터 ────────────────────────────────────────────
        if x_cut is not None:
            fc_x = v[f, 0].mean(axis=1)  # face centroid X
            if region == 'lh_cerebrum':
                # LH: keep faces left of x_cut (lateral shell, window opens medially)
                face_mask = fc_x < x_cut
            elif region == 'rh_cerebrum':
                # RH: keep faces right of x_cut (lateral shell, window opens medially)
                face_mask = fc_x > x_cut
            else:
                # 소뇌/뇌간: 단면 위치 한쪽만 표시
                face_mask = fc_x > x_cut if x_cut < 0 else fc_x < x_cut
            f = f[face_mask]
            if len(f) == 0:
                continue

        kwargs = dict(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            opacity=opacity,
            showscale=False,
            hoverinfo='none',
            name=meta['label'],
            showlegend=True,
            legendgroup='brain_regions',
        )

        if lobe_mode and region in ('lh_cerebrum', 'rh_cerebrum'):
            lobe_key = 'lh_lobe_labels' if region == 'lh_cerebrum' else 'rh_lobe_labels'
            lobe_labels = meshes.get(lobe_key)
            if lobe_labels is not None:
                intensity = np.array([LOBE_INT.get(str(l), 4) for l in lobe_labels], dtype=float) / 4.0
                kwargs.update(intensity=intensity, intensitymode='vertex',
                              colorscale=LOBE_CS, showscale=False)
            else:
                kwargs['color'] = meta['color']
        else:
            kwargs['color'] = meta['color']

        fig.add_trace(go.Mesh3d(**kwargs))

    # ── 엽 경계선 ───────────────────────────────────────────────────────────────
    if lobe_mode and show_lobe_boundaries:
        bounds = compute_lobe_boundaries()
        for hemi, bdata in bounds.items():
            region_key = f'{hemi}_cerebrum'
            if region_key not in show_regions:
                continue
            fig.add_trace(go.Scatter3d(
                x=bdata['x'], y=bdata['y'], z=bdata['z'],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.65)', width=1),
                hoverinfo='none',
                showlegend=False,
                name=f'{hemi} 엽 경계',
            ))

    # ── 영역 레이블 ─────────────────────────────────────────────────────────────
    if show_labels:
        for region, (lx, ly, lz, label) in REGION_LABELS.items():
            if region in show_regions:
                fig.add_trace(go.Scatter3d(
                    x=[lx], y=[ly], z=[lz],
                    mode='text',
                    text=[label],
                    textfont=dict(size=13, color='white', family='Arial Black'),
                    hoverinfo='none',
                    showlegend=False,
                ))


# ── 세포 스캐터 트레이스 ──────────────────────────────────────────────────────
def _get_color_map(df_plot, color_by):
    """Return (df_plot with '_ck', color_map dict or None, use_continuous bool)."""
    use_cont = False
    color_map = None

    if color_by == '세포 타입':
        df_plot['_ck'] = df_plot['cell_type'].map(lambda x: x.split(' ')[0])
        color_map = CELL_TYPE_COLORS
    elif color_by == '피질 층':
        col = 'ana_layer' if 'ana_layer' in df_plot.columns else 'layer'
        df_plot['_ck'] = df_plot[col]
        color_map = LAYER_COLORS
    elif color_by == '조건 (AD/CT)':
        df_plot['_ck'] = df_plot['condition']
        color_map = CONDITION_COLORS
    elif color_by == '서브타입':
        df_plot['_ck'] = df_plot['subtype'].fillna('Unknown')
        uniq = sorted(df_plot['_ck'].unique())
        pal = px.colors.qualitative.Alphabet
        color_map = {v: pal[i % len(pal)] for i, v in enumerate(uniq)}
    elif color_by == 'AD vs CT Δ%':
        df_plot['_ck'] = df_plot['delta'].astype(float)
        use_cont = True
    elif color_by == 'PRISM Score':
        # Scores injected externally via df_plot['_prism_score'] column
        if '_prism_score' in df_plot.columns:
            df_plot['_ck'] = df_plot['_prism_score'].astype(float)
        else:
            df_plot['_ck'] = 0.0
        use_cont = True
    else:
        df_plot['_ck'] = df_plot['leiden']
        uniq = sorted(df_plot['_ck'].unique())
        pal = px.colors.qualitative.Alphabet
        color_map = {v: pal[i % len(pal)] for i, v in enumerate(uniq)}

    return df_plot, color_map, use_cont


def add_cell_scatter(fig, df_plot, color_by, pt_size, pt_opacity,
                     x_col='ax', y_col='ay', z_col='az',
                     gene_highlight=None):
    """Add cell scatter traces to fig."""
    df_plot = df_plot.copy()
    df_plot, color_map, use_cont = _get_color_map(df_plot, color_by)

    layer_col = 'ana_layer' if 'ana_layer' in df_plot.columns else 'layer'

    if use_cont:
        vals = df_plot['_ck'].values.astype(float)
        finite_mask = np.isfinite(vals)
        cmin = float(np.nanmin(vals))
        cmax = float(np.nanmax(vals))
        is_prism = color_by == 'PRISM Score'
        if is_prism:
            cs = 'Viridis'
            cb_title = 'PRISM Score'
            hover = (df_plot['subtype'].fillna('?') + '<br>'
                     + 'PRISM: ' + pd.Series(vals).round(4).astype(str).values + '<br>'
                     + 'Cond: ' + df_plot['condition'])
        else:
            cs = 'RdBu_r'
            cb_title = 'Δ% AD-CT'
            hover = (df_plot['subtype'].fillna('?') + '<br>'
                     + 'Δ%: ' + df_plot['delta'].round(2).astype(str) + '<br>'
                     + 'p=' + df_plot['p'].round(4).astype(str) + '<br>'
                     + 'Cond: ' + df_plot['condition'])
        df_plot_f = df_plot[finite_mask] if is_prism else df_plot
        vals_f = vals[finite_mask] if is_prism else vals
        fig.add_trace(go.Scatter3d(
            x=df_plot_f[x_col], y=df_plot_f[y_col], z=df_plot_f[z_col],
            mode='markers',
            marker=dict(size=pt_size, color=vals_f, colorscale=cs,
                        cmin=cmin, cmax=cmax, opacity=pt_opacity,
                        colorbar=dict(title=cb_title, thickness=12, len=0.5)),
            text=hover[finite_mask] if is_prism else hover,
            hovertemplate='%{text}<extra></extra>',
            name=cb_title, showlegend=False,
        ))
        return

    # Discrete categories
    for key in sorted(df_plot['_ck'].dropna().unique()):
        sub = df_plot[df_plot['_ck'] == key]
        base_color = color_map.get(str(key), '#aaaaaa') if color_map else '#aaaaaa'

        # Gene highlight: dim non-matching clusters
        if gene_highlight:
            gene = gene_highlight.strip().upper()
            markers_col = sub['markers'].fillna('') if 'markers' in sub.columns else pd.Series([''] * len(sub))
            has_gene = markers_col.str.upper().str.contains(gene, regex=False)
            opacity_arr = np.where(has_gene, pt_opacity, pt_opacity * 0.12)
            size_arr = np.where(has_gene, pt_size + 1, max(1, pt_size - 1))
        else:
            opacity_arr = pt_opacity
            size_arr = pt_size

        hover = (sub['subtype'].fillna('?') + '<br>'
                 + sub[layer_col].fillna('-') + '<br>'
                 + 'Cond: ' + sub['condition'] + '<br>'
                 + 'C' + sub['leiden'])
        fig.add_trace(go.Scatter3d(
            x=sub[x_col], y=sub[y_col], z=sub[z_col],
            mode='markers',
            marker=dict(size=size_arr, color=base_color, opacity=opacity_arr),
            name=str(key), text=hover,
            hovertemplate='%{text}<extra></extra>',
        ))


# ── 유의 클러스터 강조 마커 ────────────────────────────────────────────────────
def add_sig_markers(fig, df_full, sub_df, x_col='ax', y_col='ay', z_col='az'):
    sig_clusters = sub_df[sub_df['sig'].isin(['**', '*', '†'])]['cluster'].astype(str).tolist()
    for cl in sig_clusters:
        cl_data = df_full[df_full['leiden'] == cl]
        if len(cl_data) == 0:
            continue
        cx = cl_data[x_col].mean()
        cy = cl_data[y_col].mean()
        cz = cl_data[z_col].mean()
        row = sub_df[sub_df['cluster'].astype(str) == cl].iloc[0]
        label = f"C{cl} {row['subtype']} {SIG_MARKS.get(str(row['sig']), '')} Δ{float(row['delta']):.1f}%"
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode='markers+text',
            marker=dict(size=10, color='yellow', symbol='diamond',
                        line=dict(color='orange', width=2)),
            text=[label], textposition='top center',
            textfont=dict(size=10, color='yellow'),
            showlegend=False, hoverinfo='text',
        ))


# ── 레이어 애니메이션 빌드 ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_layer_animation(_df, n_sample, pt_size, x_col, y_col, z_col):
    """Build Plotly frames for L1→WM layer sweep animation."""
    layer_col = 'ana_layer' if 'ana_layer' in _df.columns else 'layer'
    layers_ordered = sorted(_df[layer_col].dropna().unique(),
                            key=lambda x: DEPTH_ORDER.get(x, 99))
    df_s = _df.sample(min(n_sample, len(_df)), random_state=42)

    frames = []
    for i, layer in enumerate(layers_ordered):
        mask = df_s[layer_col] == layer
        sub = df_s[mask]
        sub_other = df_s[~mask]
        trace_highlight = go.Scatter3d(
            x=sub[x_col], y=sub[y_col], z=sub[z_col],
            mode='markers',
            marker=dict(size=pt_size + 2,
                        color=LAYER_COLORS.get(layer, '#ffffff'),
                        opacity=0.9),
            name=layer,
        )
        trace_bg = go.Scatter3d(
            x=sub_other[x_col], y=sub_other[y_col], z=sub_other[z_col],
            mode='markers',
            marker=dict(size=max(1, pt_size - 1), color='#333333', opacity=0.08),
            showlegend=False,
        )
        frames.append(go.Frame(
            data=[trace_highlight, trace_bg],
            name=layer,
            layout=go.Layout(title_text=f'Layer: {layer}'),
        ))

    sliders = [dict(
        active=0,
        steps=[dict(args=[[f.name], dict(frame=dict(duration=600, redraw=True),
                                         mode='immediate')],
                    label=f.name, method='animate')
               for f in frames],
        currentvalue=dict(prefix='Layer: ', font=dict(size=14, color='white')),
        font=dict(color='white'),
    )]
    return frames, sliders


@st.cache_data(show_spinner=False)
def precompute_sweep_faces(direction: str):
    """단면 스윕 애니메이션용 frame별 face 인덱스 사전 계산."""
    meshes = load_brain_meshes()
    if not meshes:
        return None
    mk = 'lh_pial' if direction == 'lh' else 'rh_pial'
    if mk not in meshes:
        return None
    v = meshes[mk]['v']
    f = meshes[mk]['f']
    fc_x = v[f, 0].mean(axis=1)
    if direction == 'lh':
        x_values = list(range(-5, -70, -5))  # -5, -10, ..., -65
    else:
        x_values = list(range(5, 70, 5))     # 5, 10, ..., 65
    frames_faces = []
    for x in x_values:
        mask = fc_x < x if direction == 'lh' else fc_x > x
        frames_faces.append(f[mask])
    return x_values, v, frames_faces


# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
st.title("🧠 Brain Cell Atlas — 3D")

if not (ANAT_CSV.exists() or UMAP_CSV.exists()):
    st.error("3D 좌표 파일이 없습니다.")
    st.stop()

meshes_available = MESH_DIR.exists() and (MESH_DIR / 'lh_pial.npz').exists()

with st.spinner("데이터 로딩 중..."):
    df, sub_df = load_data()
    meshes = load_brain_meshes() if meshes_available else None

# ── 사이드바 ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🧠 Brain 3D 설정")

view_mode = st.sidebar.radio(
    "뷰 모드",
    ["🧠 해부학적 뇌 지도", "🔵 세포 스캐터 (UMAP)", "🌐 지식 그래프"],
    index=0,
)

color_by = st.sidebar.selectbox(
    "세포 색상 기준",
    ["세포 타입", "서브타입", "피질 층", "조건 (AD/CT)", "AD vs CT Δ%", "Leiden 클러스터"],
)

st.sidebar.subheader("Phase 3 오버레이")
overlay_mode = st.sidebar.radio(
    "오버레이",
    ["없음", "유전자 강조", "레이어 애니메이션 (L1→WM)", "PRISM Score (GO term)"],
    index=0,
)
gene_highlight = None
if overlay_mode == "유전자 강조":
    _marker_genes = ['LAMP5', 'KIT', 'SST', 'PRSS12', 'CUX1', 'RORB', 'FEZF2',
                     'BCL11B', 'GAD2', 'ADARB2', 'MBP', 'PDGFRA', 'AQP4', 'P2RY12',
                     'SPP1', 'C3', 'FLT1']
    gene_highlight = st.sidebar.selectbox(
        "유전자 (marker 기반 강조)",
        options=_marker_genes,
        index=0,
        help="해당 유전자가 marker인 클러스터의 세포를 강조. marker 외 유전자는 검색 안됨.",
    )

# PRISM Score 오버레이 설정
prism_go_id = None
prism_label = ''
if overlay_mode == "PRISM Score (GO term)":
    iso_df, prism_scores, prism_meta = load_prism_data()
    if prism_meta is not None:
        top_go = sorted(prism_meta['per_go'], key=lambda x: x['auprc_brain'], reverse=True)[:30]
        go_options = {f"{g['name']} [{g['go']}] AUPRC={g['auprc_brain']:.3f}": g['go']
                      for g in top_go}
        selected_label = st.sidebar.selectbox(
            "GO term (AUPRC 상위 30)",
            options=list(go_options.keys()),
            index=0,
        )
        prism_go_id = go_options[selected_label]
        prism_label = selected_label.split('[')[0].strip()
    else:
        st.sidebar.warning("PRISM 데이터 없음.")

st.sidebar.subheader("필터")
all_ct = sorted(df['cell_type'].unique())
sel_ct = st.sidebar.multiselect("세포 타입", all_ct, default=all_ct)
all_cond = sorted(df['condition'].unique())
sel_cond = st.sidebar.multiselect("조건", all_cond, default=all_cond)
n_sample = st.sidebar.slider("표시 세포 수", 5000, 40000, 15000, 5000)
pt_size = st.sidebar.slider("점 크기", 1, 6, 2)
pt_opacity = st.sidebar.slider("투명도", 0.1, 1.0, 0.5, 0.1)

# 해부학적 뷰 전용 설정
if view_mode == "🧠 해부학적 뇌 지도":
    st.sidebar.subheader("뇌 영역 표시")
    if not meshes_available:
        st.sidebar.warning("뇌 메시 없음. setup_brain_meshes.py 실행 필요.")

    lobe_mode = st.sidebar.checkbox("엽 구분 색상 (대뇌)", value=False,
                                     help="Frontal/Parietal/Temporal/Occipital 엽별 색상")
    show_lobe_boundaries = st.sidebar.checkbox("엽 경계선 표시", value=False,
                                                disabled=not lobe_mode,
                                                help="엽 구분 모드 활성화 후 사용 가능")
    show_regions = []
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.checkbox("좌반구", value=True):  show_regions.append('lh_cerebrum')
        if st.checkbox("우반구", value=True):  show_regions.append('rh_cerebrum')
        if st.checkbox("소뇌",   value=True):  show_regions.append('cerebellum')
    with col_b:
        if st.checkbox("중뇌",   value=True):  show_regions.append('midbrain')
        if st.checkbox("교뇌",   value=True):  show_regions.append('pons')
        if st.checkbox("연수",   value=True):  show_regions.append('medulla')

    brain_opacity = st.sidebar.slider("뇌 표면 투명도", 0.02, 0.5, 0.12, 0.02)
    show_sig_markers = st.sidebar.checkbox("유의 클러스터 강조 (★)", value=True)
    show_region_labels = st.sidebar.checkbox("영역 레이블 표시", value=True)

    # 단면 뷰
    st.sidebar.subheader("🔪 시상면 단면 뷰")
    cross_section = st.sidebar.radio(
        "단면 방향",
        ["없음", "좌반구 열기 (→)", "우반구 열기 (←)",
         "좌반구 스윕 애니메이션 ▶", "우반구 스윕 애니메이션 ▶"],
        index=0,
        help="선택 반구의 내부를 열어 세포와 층 경계를 노출. 스윕: 단면이 순차적으로 이동하는 애니메이션",
    )
    x_cut_val = None
    show_layer_shells = False
    show_layer_shell_opacity = 0.07
    _sweep_mode = cross_section in ("좌반구 스윕 애니메이션 ▶", "우반구 스윕 애니메이션 ▶")
    if cross_section != "없음" and not _sweep_mode:
        if cross_section == "좌반구 열기 (→)":
            x_cut_raw = st.sidebar.slider(
                "좌반구 노출 깊이 (mm)", -5, -60, -20, -5,
                help="절단 X 위치 (음수 = 좌). -5: 거의 없음, -60: 최대 노출",
            )
            x_cut_val = x_cut_raw  # LH: keep faces with centroid_x < x_cut_val
        else:
            x_cut_raw = st.sidebar.slider(
                "우반구 노출 깊이 (mm)", 5, 60, 20, 5,
                help="절단 X 위치 (양수 = 우). 5: 거의 없음, 60: 최대 노출",
            )
            x_cut_val = x_cut_raw  # RH: keep faces with centroid_x > x_cut_val
        show_layer_shells = st.sidebar.checkbox("피질 층 경계 쉘 표시", value=True)
        show_layer_shell_opacity = st.sidebar.slider("층 경계 투명도", 0.03, 0.20, 0.07, 0.01)  # noqa: F841
    elif _sweep_mode:
        show_layer_shells = st.sidebar.checkbox("피질 층 경계 쉘 애니메이션", value=True,
                                                help="단면이 이동할 때 층 경계 쉘도 함께 이동")
        show_layer_shell_opacity = st.sidebar.slider("층 경계 투명도", 0.03, 0.20, 0.07, 0.01)
        sweep_frame_ms = st.sidebar.slider(
            "애니메이션 속도 (ms/프레임)", 80, 1200, 350, 40,
            help="값이 낮을수록 빠름. 80ms=매우 빠름, 600ms=느림",
        )
    if not _sweep_mode:
        sweep_frame_ms = 350  # unused outside sweep, but must be defined

# 필터 적용
mask = df['cell_type'].isin(sel_ct) & df['condition'].isin(sel_cond)
df_filt = df[mask].copy()

has_anat = 'ax' in df.columns

# ════════════════════════════════════════════════════════════════════════════════
# View A: 해부학적 뇌 지도
# ════════════════════════════════════════════════════════════════════════════════
if view_mode == "🧠 해부학적 뇌 지도":
    if not has_anat:
        st.warning("해부학적 좌표(ax, ay, az)가 없습니다. compute_brain_anatomical_coords.py를 먼저 실행하세요.")
        st.stop()

    st.subheader("해부학적 뇌 지도 — MNI pial mesh + 세포")
    st.caption(
        "뇌 표면: FreeSurfer fsaverage5 pial (MNI305 공간). "
        "소뇌/뇌간: 해부학적 위치 근사 절차 메시. "
        "세포 위치: MNI 타원체 공간 (층 깊이 기반)."
    )

    # 레이어 애니메이션 모드
    if overlay_mode == "레이어 애니메이션 (L1→WM)":
        frames, sliders = build_layer_animation(
            df_filt, n_sample, pt_size, 'ax', 'ay', 'az'
        )
        layer_col = 'ana_layer' if 'ana_layer' in df_filt.columns else 'layer'
        df_s = df_filt.sample(min(n_sample, len(df_filt)), random_state=42)
        first_layer = sorted(df_s[layer_col].dropna().unique(),
                             key=lambda x: DEPTH_ORDER.get(x, 99))[0]
        sub_first = df_s[df_s[layer_col] == first_layer]
        sub_other = df_s[df_s[layer_col] != first_layer]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=sub_first['ax'], y=sub_first['ay'], z=sub_first['az'],
                    mode='markers',
                    marker=dict(size=pt_size + 2,
                                color=LAYER_COLORS.get(first_layer, '#ffffff'),
                                opacity=0.9),
                    name=first_layer,
                ),
                go.Scatter3d(
                    x=sub_other['ax'], y=sub_other['ay'], z=sub_other['az'],
                    mode='markers',
                    marker=dict(size=max(1, pt_size - 1), color='#333333', opacity=0.08),
                    showlegend=False,
                ),
            ],
            frames=frames,
        )
        if meshes:
            add_brain_traces(fig, meshes, show_regions, lobe_mode,
                             opacity_scale=brain_opacity / 0.12,
                             x_cut=x_cut_val,
                             show_labels=show_region_labels)

        fig.update_layout(
            height=700,
            updatemenus=[dict(
                type='buttons', showactive=False,
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=700, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode='immediate')]),
                ],
                x=0.1, y=0, xanchor='right', yanchor='top',
            )],
            sliders=sliders,
            scene=dict(
                xaxis_title='LR (mm)', yaxis_title='AP (mm)', zaxis_title='IS (mm)',
                bgcolor='#0e1117',
                xaxis=dict(showgrid=False, zeroline=False, range=[-100, 100]),
                yaxis=dict(showgrid=False, zeroline=False, range=[-130, 100]),
                zaxis=dict(showgrid=False, zeroline=False, range=[-80, 100]),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
            ),
            paper_bgcolor='#0e1117', font=dict(color='white'),
            legend=dict(itemsizing='constant', font=dict(size=10)),
            margin=dict(l=0, r=0, t=30, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif _sweep_mode:
        # ── 단면 스윕 애니메이션 ─────────────────────────────────────────────────
        direction = 'lh' if '좌반구' in cross_section else 'rh'
        sweep_data = precompute_sweep_faces(direction)
        if sweep_data is None:
            st.warning("뇌 메시 없음. setup_brain_meshes.py 실행 필요.")
            st.stop()

        x_values, v_surf, frames_faces = sweep_data
        lobe_key = 'lh_lobe_labels' if direction == 'lh' else 'rh_lobe_labels'
        lobe_labels = (meshes.get(lobe_key) if meshes else None) if meshes else None
        surf_color = '#5577cc' if direction == 'lh' else '#4466bb'

        # 엽 모드용 per-vertex intensity 사전 계산
        intensity_full = None
        if lobe_mode and lobe_labels is not None:
            intensity_full = np.array([LOBE_INT.get(lb, 4) / 4.0 for lb in lobe_labels],
                                      dtype=np.float32)

        def _make_mesh_trace(ff):
            if intensity_full is not None:
                return go.Mesh3d(
                    x=v_surf[:, 0], y=v_surf[:, 1], z=v_surf[:, 2],
                    i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
                    intensity=intensity_full,
                    colorscale=LOBE_CS, cmin=0, cmax=1,
                    opacity=brain_opacity, showscale=False,
                    name=f'대뇌 {"좌" if direction == "lh" else "우"}반구',
                )
            return go.Mesh3d(
                x=v_surf[:, 0], y=v_surf[:, 1], z=v_surf[:, 2],
                i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
                color=surf_color, opacity=brain_opacity,
                showscale=False,
                name=f'대뇌 {"좌" if direction == "lh" else "우"}반구',
            )

        # trace 0: animated hemisphere (initial = first frame)
        init_x = x_values[0]
        fig = go.Figure(data=[_make_mesh_trace(frames_faces[0])])

        # static traces: other hemisphere + subcortical
        static_regions = [r for r in show_regions
                          if r != ('lh_cerebrum' if direction == 'lh' else 'rh_cerebrum')]
        if meshes:
            add_brain_traces(fig, meshes, static_regions, lobe_mode,
                             opacity_scale=brain_opacity / 0.12,
                             x_cut=None,
                             show_labels=show_region_labels,
                             show_lobe_boundaries=False)

        # static cell scatter — with PRISM score injection if active
        df_plot = df_filt.sample(min(n_sample, len(df_filt)), random_state=42)
        if overlay_mode == "PRISM Score (GO term)" and prism_go_id is not None:
            iso_df, prism_scores, prism_meta = load_prism_data()
            go_idx = prism_meta['go_ids'].index(prism_go_id)
            scores_col = np.array(prism_scores[:, go_idx])
            cluster_ps = build_cluster_prism_scores(prism_go_id, iso_df, scores_col)
            df_plot = df_plot.copy()
            df_plot['_prism_score'] = df_plot['leiden'].map(cluster_ps)
            eff_color_by = 'PRISM Score'
            st.caption(f"🔬 PRISM Score: **{prism_label}** | GO: `{prism_go_id}`")
        else:
            eff_color_by = color_by
        add_cell_scatter(fig, df_plot, eff_color_by, pt_size, pt_opacity,
                         gene_highlight=gene_highlight if overlay_mode == "유전자 강조" else None)

        # layer shell traces (animated) — added AFTER static traces so we know their indices
        shell_trace_indices = []
        if show_layer_shells:
            shell_start = len(fig.data)
            for t in _build_layer_shell_traces(x_cut=init_x, direction=direction,
                                               opacity=show_layer_shell_opacity):
                fig.add_trace(t)
            shell_trace_indices = list(range(shell_start, len(fig.data)))

        # animated cut-plane cap — add initial trace after shells
        cap_trace_idx = None
        if meshes and 'lh_pial' in meshes and 'rh_pial' in meshes:
            init_cap = _build_cut_plane_cap_trace(meshes, init_x, opacity=0.22)
            if init_cap is not None:
                cap_trace_idx = len(fig.data)
                fig.add_trace(init_cap)

        # animated trace indices = hemisphere (0) + shells + cap
        anim_trace_ids = [0] + shell_trace_indices
        if cap_trace_idx is not None:
            anim_trace_ids.append(cap_trace_idx)

        # animation frames — updates hemisphere + shells + cap together
        anim_frames = []
        for x, ff in zip(x_values, frames_faces):
            if len(ff) == 0:
                continue
            frame_data = [_make_mesh_trace(ff)]
            if show_layer_shells:
                frame_data += _build_layer_shell_traces(x_cut=x, direction=direction,
                                                        opacity=show_layer_shell_opacity)
            if cap_trace_idx is not None:
                cap_t = _build_cut_plane_cap_trace(meshes, x, opacity=0.22)
                frame_data.append(cap_t if cap_t is not None else go.Mesh3d(
                    x=[], y=[], z=[], i=[], j=[], k=[],
                    showscale=False, hoverinfo='none', showlegend=False,
                ))
            anim_frames.append(go.Frame(
                data=frame_data,
                traces=anim_trace_ids,
                name=str(x),
                layout=go.Layout(title_text=f'단면 X = {x} mm'),
            ))

        fig.frames = anim_frames
        _ms = sweep_frame_ms
        sweep_sliders = [dict(
            active=0,
            steps=[dict(
                args=[[fr.name], dict(frame=dict(duration=_ms, redraw=True), mode='immediate')],
                label=f'{fr.name}mm',
                method='animate',
            ) for fr in anim_frames],
            currentvalue=dict(prefix='단면: ', font=dict(size=13, color='white')),
            font=dict(color='white'),
        )]

        cam_x = 2.5 if direction == 'lh' else -2.5
        fig.update_layout(
            height=720,
            updatemenus=[dict(
                type='buttons', showactive=False,
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=_ms, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode='immediate')]),
                ],
                x=0.1, y=0, xanchor='right', yanchor='top',
            )],
            sliders=sweep_sliders,
            scene=dict(
                xaxis_title='LR (mm)', yaxis_title='AP (mm)', zaxis_title='IS (mm)',
                bgcolor='#0e1117',
                xaxis=dict(showgrid=False, zeroline=False, range=[-100, 100]),
                yaxis=dict(showgrid=False, zeroline=False, range=[-130, 100]),
                zaxis=dict(showgrid=False, zeroline=False, range=[-80, 100]),
                camera=dict(eye=dict(x=cam_x, y=0.0, z=0.3), up=dict(x=0, y=0, z=1)),
            ),
            paper_bgcolor='#0e1117', font=dict(color='white'),
            legend=dict(itemsizing='constant', font=dict(size=10)),
            margin=dict(l=0, r=0, t=30, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # 일반 해부학적 뷰
        df_plot = df_filt.sample(min(n_sample, len(df_filt)), random_state=42)
        fig = go.Figure()

        # 1) 뇌 메시 (단면 뷰 + 엽 경계선 포함)
        if meshes:
            add_brain_traces(fig, meshes, show_regions, lobe_mode,
                             opacity_scale=brain_opacity / 0.12,
                             x_cut=x_cut_val,
                             show_labels=show_region_labels,
                             show_lobe_boundaries=show_lobe_boundaries)
            # 피질 층 경계 쉘 (단면 뷰 전용)
            if show_layer_shells:
                _dir = 'lh' if cross_section == "좌반구 열기 (→)" else 'rh'
                add_layer_shell_traces(fig, x_cut=x_cut_val, direction=_dir,
                                       opacity=show_layer_shell_opacity)
            # 단면 캡 디스크
            if x_cut_val is not None and 'lh_pial' in meshes and 'rh_pial' in meshes:
                add_cut_plane_cap(fig, meshes, x_cut_val, opacity=0.20)
        else:
            st.info(
                "실제 뇌 메시 없음. "
                "`conda run -n isoform_env python reports/setup_brain_meshes.py` 실행 후 새로고침.",
                icon="⚠️",
            )

        # 2) PRISM score 세포 색상 주입
        if overlay_mode == "PRISM Score (GO term)" and prism_go_id is not None:
            iso_df, prism_scores, prism_meta = load_prism_data()
            go_idx = prism_meta['go_ids'].index(prism_go_id)
            scores_col = np.array(prism_scores[:, go_idx])
            cluster_ps = build_cluster_prism_scores(
                prism_go_id, iso_df, scores_col
            )
            df_plot = df_plot.copy()
            df_plot['_prism_score'] = df_plot['leiden'].map(cluster_ps)
            eff_color_by = 'PRISM Score'
            st.caption(f"🔬 PRISM Score: **{prism_label}**  |  GO: `{prism_go_id}`")
        else:
            eff_color_by = color_by

        # 3) 세포 스캐터
        add_cell_scatter(fig, df_plot, eff_color_by, pt_size, pt_opacity,
                         gene_highlight=gene_highlight if overlay_mode == "유전자 강조" else None)

        # 4) 유의 클러스터 마커
        if show_sig_markers:
            add_sig_markers(fig, df_filt, sub_df)

        # 엽 구분 범례
        if lobe_mode:
            for lobe, color in LOBE_COLORS.items():
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'엽: {lobe}', showlegend=True,
                ))

        # 유전자 강조 안내
        if gene_highlight and overlay_mode == "유전자 강조":
            matched = sub_df[sub_df['markers'].fillna('').str.upper()
                             .str.contains(gene_highlight.upper(), regex=False)]
            if len(matched):
                st.success(
                    f"**{gene_highlight.upper()}** marker clusters: "
                    + ", ".join(f"C{r['cluster']} {r['subtype']}" for _, r in matched.iterrows())
                )
            else:
                st.warning(f"'{gene_highlight}'가 marker에 포함된 클러스터 없음.")

        # 단면 뷰일 때 카메라를 측면(시상면 방향)으로
        if x_cut_val is not None:
            if cross_section == "좌반구 열기 (→)":
                cam = dict(eye=dict(x=2.2, y=0.0, z=0.3), up=dict(x=0, y=0, z=1))
            else:
                cam = dict(eye=dict(x=-2.2, y=0.0, z=0.3), up=dict(x=0, y=0, z=1))
        else:
            cam = dict(eye=dict(x=1.5, y=1.5, z=0.5))

        fig.update_layout(
            height=720,
            scene=dict(
                xaxis_title='LR (mm)', yaxis_title='AP (mm)', zaxis_title='IS (mm)',
                bgcolor='#0e1117',
                xaxis=dict(showgrid=False, zeroline=False, range=[-100, 100]),
                yaxis=dict(showgrid=False, zeroline=False, range=[-130, 100]),
                zaxis=dict(showgrid=False, zeroline=False, range=[-80, 100]),
                camera=cam,
            ),
            paper_bgcolor='#0e1117', font=dict(color='white'),
            legend=dict(itemsizing='constant', font=dict(size=10)),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 하단: 층별 분포 + 유의 클러스터
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("피질 층별 세포 분포")
        layer_col = 'ana_layer' if 'ana_layer' in df_filt.columns else 'layer'
        layer_pivot = (df_filt.groupby([layer_col, 'cell_type'])
                       .size().reset_index(name='n')
                       .pivot(index=layer_col, columns='cell_type', values='n')
                       .fillna(0).astype(int))
        ord_vals = [DEPTH_ORDER.get(i, 99) for i in layer_pivot.index]
        layer_pivot = layer_pivot.iloc[np.argsort(ord_vals)]
        st.dataframe(layer_pivot, use_container_width=True, height=320)
    with col2:
        st.subheader("유의 클러스터")
        sig_df = sub_df[sub_df['sig'].isin(['**', '*', '†'])].sort_values('p')
        for _, r in sig_df.iterrows():
            direction = "↑" if float(r.get('delta', 0)) > 0 else "↓"
            st.markdown(
                f"**C{r['cluster']} {r['subtype']}**  \n"
                f"{direction} Δ{float(r.get('delta', 0)):.1f}%  \n"
                f"p={float(r.get('p', 1)):.4f} {SIG_MARKS.get(str(r['sig']), '')}"
            )
            st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# View B: UMAP
# ════════════════════════════════════════════════════════════════════════════════
elif view_mode == "🔵 세포 스캐터 (UMAP)":
    st.subheader(f"UMAP 세포 스캐터 — {len(df_filt):,} 세포")
    df_plot = df_filt.sample(min(n_sample, len(df_filt)), random_state=42)
    fig = go.Figure()
    add_cell_scatter(fig, df_plot, color_by, pt_size, pt_opacity,
                     x_col='x', y_col='y', z_col='z',
                     gene_highlight=gene_highlight if overlay_mode == "유전자 강조" else None)
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3',
            bgcolor='#0e1117',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        paper_bgcolor='#0e1117', font=dict(color='white'),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# View C: 지식 그래프
# ════════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("지식 그래프 — 클러스터 노드")
    st.caption("노드 크기 = 세포 수 | 엣지 = 같은 세포 타입 | ★ = AD/CT 유의 변화")

    coord_key = ('ax', 'ay', 'az') if has_anat else ('x', 'y', 'z')

    @st.cache_data(show_spinner=False)
    def compute_centroids(_df, ck):
        cx, cy, cz = ck
        grp = _df.groupby('leiden')
        cent = grp[[cx, cy, cz]].mean().rename(columns={cx: 'cx', cy: 'cy', cz: 'cz'})
        first_cols = ['cell_type', 'subtype', 'layer', 'delta', 'p', 'sig', 'n', 'markers']
        meta = grp.first()[[c for c in first_cols if c in _df.columns]]
        if 'ana_layer' in _df.columns:
            meta['ana_layer'] = grp.first()['ana_layer']
        cent = cent.join(meta)
        cent['n_cells'] = grp.size()
        return cent.reset_index()

    centroids = compute_centroids(df_filt, coord_key)
    df_plot_c, color_map_c, use_cont_c = _get_color_map(centroids.copy(), color_by)
    centroids['_ck'] = df_plot_c['_ck']

    fig = go.Figure()
    if meshes and has_anat:
        add_brain_traces(fig, meshes, ['lh_cerebrum', 'rh_cerebrum', 'cerebellum'],
                         False, opacity_scale=0.5)

    # 엣지
    edge_x, edge_y, edge_z = [], [], []
    cx_col, cy_col, cz_col = 'cx', 'cy', 'cz'
    for ct in centroids['cell_type'].unique():
        cl_sub = centroids[centroids['cell_type'] == ct]
        pts = cl_sub[[cx_col, cy_col, cz_col]].values
        if len(pts) < 2:
            continue
        dmat = sdist.cdist(pts, pts)
        np.fill_diagonal(dmat, np.inf)
        for i in range(len(pts)):
            j = np.argmin(dmat[i])
            edge_x += [pts[i, 0], pts[j, 0], None]
            edge_y += [pts[i, 1], pts[j, 1], None]
            edge_z += [pts[i, 2], pts[j, 2], None]
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(width=1, color='rgba(150,150,150,0.3)'),
        hoverinfo='none', showlegend=False, name='',
    ))

    layer_col_c = 'ana_layer' if 'ana_layer' in centroids.columns else 'layer'
    if not use_cont_c:
        for key in sorted(centroids['_ck'].dropna().unique()):
            sub = centroids[centroids['_ck'] == key]
            base_color = color_map_c.get(str(key), '#aaaaaa') if color_map_c else '#aaaaaa'
            hover = (
                'C' + sub['leiden'] + ' ' + sub['subtype'].fillna('?') + '<br>'
                + '세포 수: ' + sub['n_cells'].astype(str) + '<br>'
                + sub['cell_type'] + '<br>'
                + 'Layer: ' + sub[layer_col_c].fillna('-') + '<br>'
                + 'Δ%: ' + sub['delta'].round(2).astype(str) + '<br>'
                + 'p=' + sub['p'].round(4).astype(str)
            )
            node_size = np.clip(sub['n_cells'] / 200, 8, 40)
            fig.add_trace(go.Scatter3d(
                x=sub['cx'], y=sub['cy'], z=sub['cz'],
                mode='markers+text',
                marker=dict(size=node_size, color=base_color, opacity=0.9,
                            line=dict(
                                color=sub['sig'].map(
                                    lambda s: '#ffff00' if s in ['**', '*', '†'] else 'rgba(100,100,100,0.3)'
                                ),
                                width=2,
                            )),
                text='C' + sub['leiden'], textposition='top center',
                textfont=dict(size=8, color='white'),
                name=str(key), customdata=hover,
                hovertemplate='%{customdata}<extra></extra>',
            ))

    sig_nodes = centroids[centroids['sig'].isin(['**', '*', '†'])]
    if len(sig_nodes):
        fig.add_trace(go.Scatter3d(
            x=sig_nodes['cx'], y=sig_nodes['cy'], z=sig_nodes['cz'],
            mode='text',
            text='C' + sig_nodes['leiden'] + ' ' + sig_nodes['subtype'].fillna('?') + ' '
                 + sig_nodes['sig'].map(SIG_MARKS),
            textfont=dict(size=11, color='yellow'),
            hoverinfo='none', showlegend=False, name='',
        ))

    scene_kw = dict(
        bgcolor='#0e1117',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        zaxis=dict(showgrid=False, zeroline=False),
    )
    if has_anat:
        scene_kw.update(
            xaxis_title='LR (mm)', yaxis_title='AP (mm)', zaxis_title='IS (mm)',
            xaxis=dict(showgrid=False, zeroline=False, range=[-100, 100]),
            yaxis=dict(showgrid=False, zeroline=False, range=[-130, 100]),
            zaxis=dict(showgrid=False, zeroline=False, range=[-80, 100]),
        )
    else:
        scene_kw.update(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3')

    fig.update_layout(
        height=750, scene=scene_kw,
        paper_bgcolor='#0e1117', font=dict(color='white'),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── 참고 표 ───────────────────────────────────────────────────────────────────
with st.expander("층 깊이 참고 기준 (Allen Human Brain Atlas + FreeSurfer fsaverage5)"):
    st.markdown("""
    | 층 | 깊이 (pial=0) | 주요 세포 |
    |---|---|---|
    | L1 | 0–0.2 mm | LAMP5/NDNF, LAMP5/KIT ivy cells |
    | L2 | 0.2–0.5 mm | L2/3 IT Exc, VIP |
    | L3 | 0.5–1.2 mm | L2/3 IT Exc (large pyramid), PV, SST |
    | **L4** | **1.2–1.8 mm** | **C18 PRSS12⁺ AD-enriched ★★** |
    | **L5** | **1.8–2.7 mm** | **C19 L5 ET depleted ★** |
    | L6 | 2.7–3.5 mm | L6 CT Exc (C25) |
    | WM | > 3.5 mm | MOL Oligodendrocytes, OPC |

    **뇌 메시 출처**: FreeSurfer fsaverage5 pial surface (MNI305 공간, nilearn 제공)
    **소뇌/뇌간**: 해부학적 위치 기반 절차 메시 (MNI 근사)
    **주의**: 세포 XY 위치는 MNI 타원체 공간 (무작위 배치), Z = 층 깊이만 해부학적 의미
    """)

# ── 페이지 간 이동 ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 관련 페이지")
_b3d_n1, _b3d_n2, _b3d_n3 = st.columns(3)
_b3d_n1.page_link("pages/13_cell_composition.py", label="📊 세포 구성 AD/CT",  help="AD vs Control 세포 구성 비율 변화")
_b3d_n2.page_link("pages/14_bisect_context.py",   label="🔬 BISECT 세포 맥락", help="세포 타입별 아이소폼 스위치 분석")
_b3d_n3.page_link("pages/07_bisect.py",           label="🧫 BISECT Cases",     help="개별 케이스 상세 분석")
