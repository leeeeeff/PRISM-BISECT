"""GO Score Space UMAP visualization (Module B1).

Projects isoforms into 2D using their GO-score vectors, then renders
an interactive Plotly scatter with flexible coloring options.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd

# umap-learn's parametric_umap imports TensorFlow → protobuf C extension.
# Streamlit loads protobuf before user code runs, so the env-var workaround
# cannot be applied in time. We therefore catch both ImportError AND TypeError
# (the protobuf descriptor error) and fall back to sklearn TSNE gracefully.
_HAS_UMAP = False
_UMAP = None
try:
    from umap import UMAP as _UMAP   # type: ignore[assignment]
    _HAS_UMAP = True
except (ImportError, TypeError, Exception):
    pass

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def compute_umap_coords(
    score_matrix: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    """Compute 2D embedding from GO score vectors.

    Returns UMAP when available; falls back to sklearn TSNE when umap-learn
    cannot be imported (e.g. TF/protobuf conflict in the local conda env).

    Returns
    -------
    coords : ndarray (n_isoforms, 2)
    method : 'UMAP' | 't-SNE'
    """
    X = score_matrix.astype(np.float32)
    if _HAS_UMAP:
        reducer = _UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_components=2,
            low_memory=True,
        )
        return reducer.fit_transform(X).astype(np.float32), 'UMAP'

    # ── TSNE fallback ────────────────────────────────────────────────────────
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, X.shape[0] // 5))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        max_iter=1000,
    )
    return reducer.fit_transform(X).astype(np.float32), 't-SNE'


def _kmeans_cluster_labels(
    coords: np.ndarray,
    score_matrix: Optional[np.ndarray],
    go_terms: Optional[List[str]],
    go_names: Optional[Dict[str, str]],
    k: int = 6,
) -> Tuple[np.ndarray, List[dict]]:
    """KMeans cluster assignment + dominant-GO label per cluster.

    Returns
    -------
    labels   : (n,) int cluster assignment
    centroids: list of dicts with keys x, y, label, count
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(coords)

    centroids = []
    for c in range(k):
        mask = labels == c
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        count = int(mask.sum())

        # Dominant GO term in this cluster
        label = f"Cluster {c+1}"
        if score_matrix is not None and go_terms and mask.sum() > 0:
            mean_scores = score_matrix[mask].mean(axis=0)
            top_idx = int(mean_scores.argmax())
            top_go  = go_terms[top_idx]
            top_name = (go_names or {}).get(top_go, top_go)
            # Abbreviate long names
            if len(top_name) > 28:
                top_name = top_name[:26] + '…'
            label = f"#{c+1} {top_name}"

        centroids.append(dict(x=cx, y=cy, label=label, count=count))

    return labels, centroids


def build_umap_figure(
    coords: np.ndarray,
    isoform_ids: np.ndarray,
    color_by: str = 'isoform_type',
    metadata_df: Optional[pd.DataFrame] = None,
    go_terms: Optional[List[str]] = None,
    go_names: Optional[Dict[str, str]] = None,
    score_matrix: Optional[np.ndarray] = None,
    title: str = 'PRISM GO-Score Space UMAP',
    point_size: int = 4,
    opacity: float = 0.7,
    show_cluster_labels: bool = True,
    n_clusters: int = 6,
) -> 'go.Figure':
    """Build an interactive Plotly UMAP scatter figure.

    Parameters
    ----------
    coords       : (n_isoforms, 2) — precomputed UMAP coordinates
    isoform_ids  : (n_isoforms,) str
    color_by     : one of 'isoform_type' | 'scenario' | 'max_go' | 'max_score'
    metadata_df  : DataFrame with isoform_id as index or column.
                   Should contain: isoform_type, scenario, scenario_label,
                   max_go, max_score, gene_id (optional).
    go_terms     : GO IDs for score_matrix columns
    go_names     : {GO_ID: full name}
    score_matrix : (n_isoforms, n_go) — needed for hover info
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required: pip install plotly")

    n = len(isoform_ids)
    isoform_ids = np.asarray(isoform_ids, dtype=str)
    go_names    = go_names or {}

    # ── Build base DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame({
        'umap_x':     coords[:, 0],
        'umap_y':     coords[:, 1],
        'isoform_id': isoform_ids,
    })

    if metadata_df is not None:
        # Align by isoform_id
        id_col = 'isoform_id' if 'isoform_id' in metadata_df.columns else metadata_df.index.name
        if id_col and id_col != 'isoform_id':
            meta = metadata_df.reset_index().rename(columns={id_col: 'isoform_id'})
        else:
            meta = metadata_df.reset_index(drop=True) if id_col is None else metadata_df.copy()
        meta['isoform_id'] = meta['isoform_id'].astype(str)
        df = df.merge(meta, on='isoform_id', how='left')

    # ── Add max_score / max_go if score_matrix available ─────────────────
    if score_matrix is not None and 'max_score' not in df.columns:
        max_idx = score_matrix.argmax(axis=1)
        df['max_score'] = score_matrix.max(axis=1).round(3)
        if go_terms:
            df['max_go'] = [go_terms[i] for i in max_idx]
            df['max_go_name'] = df['max_go'].map(lambda x: go_names.get(x, x))

    # ── Color mapping ─────────────────────────────────────────────────────
    COLOR_MAPS = {
        'isoform_type': {
            'palette': {'known': '#4c72b0', 'nic': '#55a868', 'nnic': '#c44e52'},
            'col':     'isoform_type',
        },
        'scenario': {
            'col': 'scenario_label',
            'palette': {
                'Scenario 1: Functional Switch':     '#e63946',
                'Scenario 2: Expression Switch':     '#f4a261',
                'Scenario 3: Constitutive Novel Function': '#2a9d8f',
                'Scenario 4: Background/Unknown':    '#adb5bd',
            },
        },
        'max_go': {'col': 'max_go_name'},
        'max_score': {'col': 'max_score'},
    }

    cfg   = COLOR_MAPS.get(color_by, COLOR_MAPS['isoform_type'])
    c_col = cfg.get('col', color_by)

    # Fill missing colour column gracefully
    if c_col not in df.columns:
        df[c_col] = 'unknown'

    # ── Hover text ────────────────────────────────────────────────────────
    hover_parts = ['<b>%{customdata[0]}</b>']
    custom_cols = ['isoform_id']
    if 'gene_id' in df.columns:
        hover_parts.append('Gene: %{customdata[1]}')
        custom_cols.append('gene_id')
    if 'max_score' in df.columns:
        hover_parts.append('Max score: %{customdata[2]:.3f}')
        custom_cols.append('max_score')
    if 'max_go_name' in df.columns:
        hover_parts.append('Top GO: %{customdata[3]}')
        custom_cols.append('max_go_name')
    if 'isoform_type' in df.columns:
        hover_parts.append('Type: %{customdata[4]}')
        custom_cols.append('isoform_type')

    hover_template = '<br>'.join(hover_parts) + '<extra></extra>'

    custom_data = df[custom_cols].values

    # ── Build figure ──────────────────────────────────────────────────────
    if color_by == 'max_score' and 'max_score' in df.columns:
        fig = px.scatter(
            df, x='umap_x', y='umap_y',
            color='max_score',
            color_continuous_scale='Viridis',
            opacity=opacity,
            title=title,
            labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'},
        )
    else:
        palette = cfg.get('palette', None)
        fig = px.scatter(
            df, x='umap_x', y='umap_y',
            color=c_col,
            color_discrete_map=palette,
            opacity=opacity,
            title=title,
            labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'},
        )

    fig.update_traces(
        marker=dict(size=point_size),
        customdata=custom_data,
        hovertemplate=hover_template,
    )

    fig.update_layout(
        legend=dict(title=color_by.replace('_', ' ').title(),
                    itemsizing='constant'),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=60, r=20, t=50, b=60),
    )

    # ── Cluster label overlay ─────────────────────────────────────────────
    if show_cluster_labels and len(coords) >= max(10, n_clusters):
        try:
            _, centroids = _kmeans_cluster_labels(
                coords, score_matrix, go_terms, go_names, k=n_clusters
            )
            for c in centroids:
                fig.add_annotation(
                    x=c['x'], y=c['y'],
                    text=f"<b>{c['label']}</b><br><span style='font-size:9px'>n={c['count']:,}</span>",
                    showarrow=False,
                    font=dict(size=10, color='#1e293b'),
                    bgcolor='rgba(255,255,255,0.82)',
                    bordercolor='#94a3b8',
                    borderwidth=1,
                    borderpad=4,
                )
        except Exception:
            pass  # cluster annotation is optional; never break the main figure

    return fig


def load_precomputed_coords(coords_path: str) -> np.ndarray:
    """Load pre-computed UMAP coordinates from a .npy file."""
    return np.load(coords_path)
