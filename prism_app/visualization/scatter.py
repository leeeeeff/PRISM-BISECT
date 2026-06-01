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


def detect_functional_divergence(scores: np.ndarray, iso_ids, go_labels):
    """Find the isoform pair with the largest score difference per GO term.

    Returns dict with keys:
      - max_delta: float (largest pairwise difference found)
      - go_term: str (GO label where max delta occurs)
      - iso_high: str (isoform ID with higher score)
      - iso_low: str (isoform ID with lower score)
      - score_high: float
      - score_low: float
      - per_go_max_delta: list of (go_label, max_delta) sorted descending

    If < 2 isoforms, return None.
    """
    n_iso, n_go = scores.shape
    if n_iso < 2:
        return None

    max_delta = 0.0
    best_go_idx = 0
    best_i, best_j = 0, 1

    for g in range(n_go):
        col = scores[:, g]
        i_max = int(col.argmax())
        i_min = int(col.argmin())
        delta = float(col[i_max] - col[i_min])
        if delta > max_delta:
            max_delta = delta
            best_go_idx = g
            best_i, best_j = i_max, i_min

    per_go_max_delta = []
    for g in range(n_go):
        col = scores[:, g]
        d = float(col.max() - col.min())
        per_go_max_delta.append((go_labels[g], round(d, 3)))
    per_go_max_delta.sort(key=lambda x: x[1], reverse=True)

    return dict(
        max_delta=round(max_delta, 3),
        go_term=go_labels[best_go_idx],
        iso_high=str(iso_ids[best_i]),
        iso_low=str(iso_ids[best_j]),
        score_high=round(float(scores[best_i, best_go_idx]), 3),
        score_low=round(float(scores[best_j, best_go_idx]), 3),
        per_go_max_delta=per_go_max_delta,
    )


def build_within_gene_chart(
    gene_name: str,
    isoform_ids: np.ndarray,
    score_matrix: np.ndarray,
    go_terms: List[str],
    go_names: Optional[Dict[str, str]] = None,
    gene_ids: Optional[np.ndarray] = None,
    chart_type: str = 'bar',
    title: Optional[str] = None,
) -> 'tuple[go.Figure, Optional[dict]]':
    """Compare all isoforms of a gene across GO terms.

    Parameters
    ----------
    gene_name   : gene symbol to filter isoforms
    isoform_ids : (n_isoforms,)
    score_matrix: (n_isoforms, n_go)
    go_terms    : list of GO IDs
    gene_ids    : (n_isoforms,) — if None, filters by isoform_id prefix
    chart_type  : 'bar' (grouped) | 'heatmap' | 'parallel'

    Returns
    -------
    (fig, divergence_info) tuple where divergence_info is from
    detect_functional_divergence() or None.
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
        return fig, None

    iso_ids = isoform_ids[mask]
    scores  = score_matrix[mask]              # (n_gene_iso, n_go)
    go_labels = [go_names.get(g, g)[:30] for g in go_terms]

    title = title or f'{gene_name}: Isoform × GO-score Comparison'

    div_info = detect_functional_divergence(scores, iso_ids, go_labels)

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

        # Add annotation highlighting the GO term with the largest divergence
        if div_info is not None:
            fig.add_annotation(
                x=div_info['go_term'],
                y=div_info['score_high'],
                text=f"⬆ 최대 분기 (Δ={div_info['max_delta']:.2f})",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#eab308',
                font=dict(color='#92400e', size=11, family='Arial'),
                bgcolor='#fef9c3',
                bordercolor='#eab308',
                borderwidth=1,
                ax=0,
                ay=-40,
            )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
    )
    return fig, div_info


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
