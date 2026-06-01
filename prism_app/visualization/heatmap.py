"""Isoform Type × GO Score Heatmap (Module A4) and GO Co-occurrence Network (B2)."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def build_type_go_heatmap(
    score_matrix: np.ndarray,
    isoform_types: np.ndarray,
    go_terms: List[str],
    go_names: Optional[Dict[str, str]] = None,
    title: str = 'Mean PRISM Score by Isoform Type and GO Term',
) -> 'go.Figure':
    """Heatmap: rows = isoform type, columns = GO terms, cells = mean score.

    Parameters
    ----------
    score_matrix  : (n_isoforms, n_go)
    isoform_types : (n_isoforms,) — 'known' / 'nic' / 'nnic'
    go_terms      : list of GO IDs, length n_go
    go_names      : {GO_ID: display name}
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    go_names   = go_names or {}
    types_arr  = np.asarray([str(t).lower().strip() for t in isoform_types])
    type_order = ['known', 'nic', 'nnic']

    # Compute mean score per (type, GO term)
    matrix = np.zeros((len(type_order), len(go_terms)))
    for i, t in enumerate(type_order):
        mask = types_arr == t
        if mask.sum() > 0:
            matrix[i] = score_matrix[mask].mean(axis=0)

    # Display labels
    go_labels  = [go_names.get(g, g) for g in go_terms]
    # Truncate long names for display
    go_labels  = [n[:35] + '…' if len(n) > 35 else n for n in go_labels]
    type_labels = ['Known (Ensembl)', 'NIC', 'NNIC']

    # Annotation text
    annot = [[f'{matrix[i, j]:.2f}' for j in range(len(go_terms))]
             for i in range(len(type_order))]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=go_labels,
        y=type_labels,
        colorscale='RdYlGn',
        zmin=0, zmax=1,
        text=annot,
        texttemplate='%{text}',
        hovertemplate='%{y} × %{x}<br>Mean score: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Mean score', thickness=15),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=12)),
        height=300,
        margin=dict(l=140, r=60, t=60, b=160),
        font=dict(family='Arial'),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


def build_go_cooccurrence_network(
    score_matrix: np.ndarray,
    go_terms: List[str],
    go_names: Optional[Dict[str, str]] = None,
    min_corr: float = 0.3,
    title: str = 'GO Term Co-prediction Network',
) -> 'go.Figure':
    """Network graph: nodes = GO terms, edges = Pearson r of cross-isoform scores.

    Parameters
    ----------
    score_matrix : (n_isoforms, n_go)
    min_corr     : minimum absolute Pearson r to draw an edge
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    go_names = go_names or {}
    n_go = len(go_terms)

    # Correlation matrix
    corr = np.corrcoef(score_matrix.T)  # (n_go, n_go)

    # Simple circular layout
    angles = np.linspace(0, 2 * np.pi, n_go, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    # Build edges
    edge_x, edge_y, edge_w = [], [], []
    for i in range(n_go):
        for j in range(i + 1, n_go):
            r = corr[i, j]
            if abs(r) >= min_corr:
                edge_x += [xs[i], xs[j], None]
                edge_y += [ys[i], ys[j], None]
                edge_w.append(abs(r))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
    )

    labels  = [go_names.get(g, g)[:30] for g in go_terms]
    node_trace = go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        marker=dict(size=14, color='#4c72b0',
                    line=dict(width=1, color='white')),
        text=labels,
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><extra></extra>',
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
