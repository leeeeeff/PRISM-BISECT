"""GO Term Enrichment of Switched Isoforms (Module C2).

Hypergeometric test to identify GO terms enriched among DTU-switched
isoforms vs the background set of all isoforms.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats as spstats


def test_go_enrichment(
    query_indices: np.ndarray,
    score_matrix: np.ndarray,
    go_terms: List[str],
    go_names: dict,
    score_threshold: float = 0.5,
    min_observed: int = 2,
) -> pd.DataFrame:
    """Hypergeometric enrichment test for GO terms among a query isoform set.

    Parameters
    ----------
    query_indices : indices into score_matrix for the query set (e.g. DTU isoforms)
    score_matrix  : (n_isoforms, n_go) PRISM score array
    go_terms      : list of GO term IDs matching score_matrix columns
    go_names      : GO ID → name mapping
    score_threshold : threshold for calling a prediction "high"
    min_observed  : minimum k to include a term in output

    Returns
    -------
    DataFrame sorted by pvalue with columns:
        GO_ID, GO_term, observed, expected, N_background,
        fold_enrichment, pvalue, FDR
    """
    N  = score_matrix.shape[0]          # total background
    n  = len(query_indices)             # query set size
    if n == 0 or N == 0:
        return pd.DataFrame()

    bg_high  = (score_matrix > score_threshold).sum(axis=0)   # M per GO
    qry_high = (score_matrix[query_indices] > score_threshold).sum(axis=0)  # k per GO

    rows = []
    for i, g in enumerate(go_terms):
        k = int(qry_high[i])
        M = int(bg_high[i])
        if k < min_observed or M == 0:
            continue
        # P(X >= k) under hypergeometric(N, M, n)
        pv = spstats.hypergeom.sf(k - 1, N, M, n)
        expected = round(n * M / N, 2)
        fe = round(k / max(expected, 0.01), 2)
        rows.append({
            'GO_ID':           g,
            'GO_term':         go_names.get(g, g)[:55],
            'observed':        k,
            'expected':        expected,
            'N_background':    M,
            'fold_enrichment': fe,
            'pvalue':          pv,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('pvalue').reset_index(drop=True)
    # Benjamini-Hochberg FDR
    m = len(df)
    ranks = np.arange(1, m + 1)
    df['FDR'] = np.minimum(df['pvalue'].values * m / ranks, 1.0)
    df['FDR'] = np.minimum.accumulate(df['FDR'].values[::-1])[::-1]
    return df.round({'pvalue': 6, 'FDR': 6})


def enrich_dtu_switched(
    dtu_df: pd.DataFrame,
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    go_terms: List[str],
    go_names: dict,
    score_threshold: float = 0.5,
    dtu_pval_threshold: float = 0.05,
    direction: str = 'both',
) -> pd.DataFrame:
    """Convenience wrapper: enrichment for DTU-significant isoforms.

    Parameters
    ----------
    direction : 'both' | 'up' | 'down'
        'up'   — isoforms with delta_IF > 0 (upregulated in condition)
        'down' — isoforms with delta_IF < 0 (downregulated)
        'both' — all significant DTU isoforms
    """
    from prism_app.pipeline.dtu_connector import parse_dtu_result

    try:
        dtu = parse_dtu_result(dtu_df.copy())
    except ValueError:
        return pd.DataFrame()

    if 'pvalue' in dtu.columns and dtu['pvalue'].notna().any():
        sig = dtu[dtu['pvalue'] <= dtu_pval_threshold]
    else:
        sig = dtu

    if direction == 'up' and 'delta_IF' in sig.columns:
        sig = sig[sig['delta_IF'] > 0]
    elif direction == 'down' and 'delta_IF' in sig.columns:
        sig = sig[sig['delta_IF'] < 0]

    ids_arr = np.asarray(isoform_ids, dtype=str)
    query_idx = []
    for iso in sig['isoform_id'].dropna():
        m = np.where(ids_arr == str(iso))[0]
        if len(m):
            query_idx.append(m[0])
    query_idx = np.array(query_idx)

    result = test_go_enrichment(query_idx, score_matrix, go_terms, go_names,
                                score_threshold=score_threshold)
    if not result.empty:
        result.insert(0, 'direction', direction)
    return result
