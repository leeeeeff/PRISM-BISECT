"""Within-gene isoform spread visualization (Module B4)."""
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


def build_within_gene_chart(
    gene_name: str,
    isoform_ids: np.ndarray,
    score_matrix: np.ndarray,
    go_terms: List[str],
    go_names: Optional[Dict[str, str]] = None,
    gene_ids: Optional[np.ndarray] = None,
    chart_type: str = 'bar',
    title: Optional[str] = None,
) -> 'go.Figure':
    """Compare all isoforms of a gene across GO terms.

    Parameters
    ----------
    gene_name   : gene symbol to filter isoforms
    isoform_ids : (n_isoforms,)
    score_matrix: (n_isoforms, n_go)
    go_terms    : list of GO IDs
    gene_ids    : (n_isoforms,) — if None, filters by isoform_id prefix
    chart_type  : 'bar' (grouped) | 'heatmap' | 'parallel'
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    go_names   = go_names or {}
    isoform_ids = np.asarray(isoform_ids, dtype=str)

    # ── Filter to gene ────────────────────────────────────────────────────
    if gene_ids is not None:
        gene_ids_arr = np.asarray(gene_ids, dtype=str)
        mask = np.isin(gene_ids_arr, [gene_name, gene_name.upper()])
        if mask.sum() == 0:
            # Try partial match
            mask = np.array([gene_name.lower() in g.lower() for g in gene_ids_arr])
    else:
        # Fall back to isoform ID prefix
        mask = np.array([gene_name.lower() in iso.lower() for iso in isoform_ids])

    if mask.sum() == 0:
        fig = go.Figure()
        fig.add_annotation(text=f'No isoforms found for gene: {gene_name}',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14))
        return fig

    iso_ids = isoform_ids[mask]
    scores  = score_matrix[mask]              # (n_gene_iso, n_go)
    go_labels = [go_names.get(g, g)[:30] for g in go_terms]

    title = title or f'{gene_name}: Isoform × GO-score Comparison'

    if chart_type == 'heatmap':
        fig = go.Figure(data=go.Heatmap(
            z=scores,
            x=go_labels,
            y=[str(i) for i in iso_ids],
            colorscale='RdYlGn',
            zmin=0, zmax=1,
            hovertemplate='%{y}<br>%{x}<br>Score: %{z:.3f}<extra></extra>',
            colorbar=dict(title='Score', thickness=15),
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            height=max(300, 40 * len(iso_ids) + 120),
            margin=dict(l=200, r=40, t=60, b=160),
        )

    elif chart_type == 'parallel':
        df_par = pd.DataFrame(scores, columns=go_labels)
        df_par.insert(0, 'isoform', [str(i) for i in iso_ids])
        fig = px.parallel_coordinates(
            df_par,
            color=df_par.index,
            dimensions=go_labels,
            title=title,
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.update_layout(height=450)

    else:  # 'bar' — grouped bar chart
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for k, (iso_id, row_sc) in enumerate(zip(iso_ids, scores)):
            fig.add_trace(go.Bar(
                name=str(iso_id),
                x=go_labels,
                y=row_sc,
                marker_color=colors[k % len(colors)],
                opacity=0.85,
                hovertemplate=f'<b>{iso_id}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>',
            ))
        fig.update_layout(
            title=title,
            barmode='group',
            xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
            yaxis=dict(title='PRISM Score', range=[0, 1]),
            legend=dict(title='Isoform'),
            height=450,
            margin=dict(l=60, r=20, t=60, b=160),
        )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
    )
    return fig


def build_scenario_bar(classified_df: pd.DataFrame, title: str = 'Isoform Scenario Distribution') -> 'go.Figure':
    """Simple bar chart of scenario counts."""
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    from prism_app.core.classifier import _SCENARIO_COLORS, _SCENARIO_LABELS

    counts = classified_df.groupby(['scenario', 'scenario_label']).size().reset_index(name='count')
    counts = counts.sort_values('scenario')

    fig = go.Figure(go.Bar(
        x=counts['scenario_label'],
        y=counts['count'],
        marker_color=[_SCENARIO_COLORS.get(s, '#888') for s in counts['scenario']],
        text=counts['count'],
        textposition='outside',
        hovertemplate='%{x}<br>N = %{y:,}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-20),
        yaxis=dict(title='Number of isoforms'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=60, r=20, t=50, b=120),
        height=380,
    )
    return fig
