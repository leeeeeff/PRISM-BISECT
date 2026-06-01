"""Expression × Score Joint Filter (Module E4).

Filters isoform candidates by requiring both high PRISM score
AND sufficient expression (CPM threshold), reducing false positives
from lowly/un-expressed isoforms.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_cpm(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to CPM (counts per million).

    Parameters
    ----------
    count_matrix : DataFrame (rows=isoforms, cols=samples), index=isoform_id
    """
    lib_sizes = count_matrix.sum(axis=0)
    cpm = count_matrix.div(lib_sizes, axis=1) * 1e6
    return cpm


def load_count_matrix(filepath: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load count matrix from TSV/CSV. First column assumed to be isoform IDs."""
    if sep is None:
        sep = '\t' if filepath.endswith('.tsv') or filepath.endswith('.txt') else ','
    df = pd.read_csv(filepath, sep=sep, index_col=0)
    return df


def apply_expression_filter(
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    count_matrix: pd.DataFrame,
    cpm_threshold: float = 1.0,
    score_threshold: float = 0.5,
    aggregation: str = 'mean',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply joint expression + score filter.

    Parameters
    ----------
    aggregation : 'mean' | 'max' | 'median' — how to aggregate CPM across samples

    Returns
    -------
    (filtered_mask, cpm_values, high_score_mask)
        filtered_mask  : boolean array of len(isoform_ids), True = passes both filters
        cpm_values     : CPM value per isoform (NaN if not in count matrix)
        high_score_mask: boolean array, True = any GO score > score_threshold
    """
    cpm = compute_cpm(count_matrix)

    agg_fn = {'mean': cpm.mean, 'max': cpm.max, 'median': cpm.median}.get(
        aggregation, cpm.mean
    )
    mean_cpm = agg_fn(axis=1)

    ids_arr = np.asarray(isoform_ids, dtype=str)
    cpm_vals = np.array([float(mean_cpm.get(iso, np.nan)) for iso in ids_arr])

    expressed_mask   = np.where(np.isnan(cpm_vals), False, cpm_vals >= cpm_threshold)
    high_score_mask  = (score_matrix > score_threshold).any(axis=1)
    filtered_mask    = expressed_mask & high_score_mask

    return filtered_mask, cpm_vals, high_score_mask


def expression_filter_summary(
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    count_matrix: pd.DataFrame,
    cpm_threshold: float = 1.0,
    score_threshold: float = 0.5,
) -> dict:
    """Return a summary dict of expression filter statistics."""
    mask, cpm_vals, hs_mask = apply_expression_filter(
        score_matrix, isoform_ids, count_matrix,
        cpm_threshold=cpm_threshold, score_threshold=score_threshold,
    )
    n_total    = len(isoform_ids)
    n_in_count = int((~np.isnan(cpm_vals)).sum())
    return {
        'n_total':            n_total,
        'n_in_count_matrix':  n_in_count,
        'n_high_score':       int(hs_mask.sum()),
        'n_expressed':        int((~np.isnan(cpm_vals) & (cpm_vals >= cpm_threshold)).sum()),
        'n_joint':            int(mask.sum()),
        'pct_joint':          round(100 * mask.sum() / max(1, n_total), 2),
        'median_cpm_joint':   round(float(np.nanmedian(cpm_vals[mask])), 2) if mask.sum() else 0.0,
    }
