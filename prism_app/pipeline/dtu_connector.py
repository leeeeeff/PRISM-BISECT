"""DTU → Functional Consequence Matrix (Module C1).

Connects differential transcript usage (DTU) results with PRISM GO scores
to classify each gene × GO term pair as GAIN / LOSS / NEUTRAL.

Supported DTU input formats:
  - satuRn output TSV
  - DEXSeq output TSV
  - IsoformSwitchAnalyzeR output TSV
  - Generic: isoform_id, condition, delta_IF, pvalue columns
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ── DTU parsing ───────────────────────────────────────────────────────────────

_COL_ALIASES: Dict[str, List[str]] = {
    'isoform_id': ['isoform_id', 'transcriptID', 'transcript_id', 'tx_id',
                   'isoform', 'feature_id', 'TXNAME'],
    'gene_id':    ['gene_id', 'geneID', 'gene', 'GENEID', 'gene_name'],
    'condition':  ['condition', 'contrast', 'comparison', 'group'],
    'delta_IF':   ['delta_IF', 'dIF', 'deltaPSI', 'delta_psi', 'logFC', 'log2FC'],
    'pvalue':     ['pvalue', 'padj', 'FDR', 'adj.p.value', 'P.Value', 'pval'],
}


def parse_dtu_result(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a DTU result DataFrame to canonical column names.

    Returns DataFrame with columns:
        isoform_id, gene_id (optional), condition (optional),
        delta_IF (optional), pvalue
    """
    rename = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename[alias] = canonical
                break
    df = df.rename(columns=rename)

    # Ensure required column exists
    if 'isoform_id' not in df.columns:
        raise ValueError(
            "DTU file must have an isoform ID column. "
            f"Found columns: {list(df.columns)}"
        )

    # Fill optional columns
    for col in ['gene_id', 'condition', 'delta_IF']:
        if col not in df.columns:
            df[col] = None

    if 'pvalue' not in df.columns:
        df['pvalue'] = np.nan

    return df[['isoform_id', 'gene_id', 'condition', 'delta_IF', 'pvalue']]


# ── Functional consequence computation ───────────────────────────────────────

def compute_functional_consequence(
    dtu_df: pd.DataFrame,
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    go_terms: List[str],
    go_names: Dict[str, str],
    score_threshold: float = 0.5,
    dtu_pval_threshold: float = 0.05,
    delta_if_threshold: float = 0.1,
    score_delta_threshold: float = 0.15,
) -> pd.DataFrame:
    """Compute per-gene × per-GO functional consequence of DTU.

    For each significant DTU event, compare the PRISM score of the
    up-regulated isoform vs the down-regulated isoform for every GO term.

    Returns
    -------
    DataFrame with columns:
        gene_id, condition, go_term, go_name,
        up_isoform, down_isoform,
        up_score, down_score, score_delta,
        consequence  (GAIN | LOSS | NEUTRAL | UNKNOWN)
    """
    try:
        dtu = parse_dtu_result(dtu_df.copy())
    except ValueError as e:
        return pd.DataFrame(columns=['gene_id', 'condition', 'go_term', 'go_name',
                                      'up_isoform', 'down_isoform',
                                      'up_score', 'down_score', 'score_delta',
                                      'consequence'])

    ids_arr = np.asarray(isoform_ids, dtype=str)

    def _get_score_vec(iso_id: str) -> Optional[np.ndarray]:
        m = np.where(ids_arr == iso_id)[0]
        if len(m) == 0:
            return None
        return score_matrix[m[0]]

    # Filter to significant DTU events
    if 'pvalue' in dtu.columns and dtu['pvalue'].notna().any():
        sig = dtu[dtu['pvalue'] <= dtu_pval_threshold].copy()
    else:
        sig = dtu.copy()

    if sig.empty:
        return pd.DataFrame()

    # Group by gene + condition → identify up/down isoforms per pair
    rows = []

    group_cols = [c for c in ['gene_id', 'condition'] if sig[c].notna().any()]
    if not group_cols:
        # No grouping possible — treat every row individually
        for _, r in sig.iterrows():
            vec = _get_score_vec(str(r['isoform_id']))
            if vec is None:
                continue
            delta_if = r.get('delta_IF', 0) or 0
            for i, g in enumerate(go_terms):
                rows.append({
                    'gene_id': r.get('gene_id', ''),
                    'condition': r.get('condition', ''),
                    'go_term': g,
                    'go_name': go_names.get(g, g)[:50],
                    'up_isoform': r['isoform_id'] if delta_if >= 0 else '',
                    'down_isoform': r['isoform_id'] if delta_if < 0 else '',
                    'up_score': float(vec[i]) if delta_if >= 0 else np.nan,
                    'down_score': float(vec[i]) if delta_if < 0 else np.nan,
                    'score_delta': np.nan,
                    'consequence': 'UNKNOWN',
                })
    else:
        for grp_key, grp in sig.groupby(group_cols, dropna=False):
            gene_id  = grp_key[0] if len(group_cols) > 0 else ''
            cond     = grp_key[1] if len(group_cols) > 1 else ''

            # Separate up (delta_IF > 0) and down (delta_IF < 0) isoforms
            if 'delta_IF' in grp.columns and grp['delta_IF'].notna().any():
                up_isos   = grp[grp['delta_IF'] > delta_if_threshold]['isoform_id'].tolist()
                down_isos = grp[grp['delta_IF'] < -delta_if_threshold]['isoform_id'].tolist()
            else:
                up_isos   = grp['isoform_id'].tolist()
                down_isos = []

            # Mean score vectors for up/down groups
            up_vecs   = [_get_score_vec(str(x)) for x in up_isos]
            down_vecs = [_get_score_vec(str(x)) for x in down_isos]
            up_vecs   = [v for v in up_vecs   if v is not None]
            down_vecs = [v for v in down_vecs if v is not None]

            if not up_vecs and not down_vecs:
                continue

            up_mean   = np.mean(up_vecs,   axis=0) if up_vecs   else None
            down_mean = np.mean(down_vecs, axis=0) if down_vecs else None

            for i, g in enumerate(go_terms):
                up_s   = float(up_mean[i])   if up_mean   is not None else np.nan
                down_s = float(down_mean[i]) if down_mean is not None else np.nan
                delta  = up_s - down_s if not (np.isnan(up_s) or np.isnan(down_s)) else np.nan

                if np.isnan(delta):
                    consequence = 'UNKNOWN'
                elif delta > score_delta_threshold:
                    consequence = 'GAIN'
                elif delta < -score_delta_threshold:
                    consequence = 'LOSS'
                else:
                    consequence = 'NEUTRAL'

                rows.append({
                    'gene_id':     gene_id,
                    'condition':   cond,
                    'go_term':     g,
                    'go_name':     go_names.get(g, g)[:50],
                    'up_isoform':  '; '.join(up_isos[:3]),
                    'down_isoform': '; '.join(down_isos[:3]),
                    'up_score':    round(up_s, 4)   if not np.isnan(up_s)   else None,
                    'down_score':  round(down_s, 4) if not np.isnan(down_s) else None,
                    'score_delta': round(delta, 4)  if not np.isnan(delta)  else None,
                    'consequence': consequence,
                })

    return pd.DataFrame(rows)


def build_consequence_pivot(
    consequence_df: pd.DataFrame,
    value_col: str = 'consequence',
) -> pd.DataFrame:
    """Pivot consequence DataFrame to gene × GO matrix.

    Each cell contains GAIN / LOSS / NEUTRAL / UNKNOWN.
    """
    if consequence_df.empty:
        return pd.DataFrame()

    pivot = consequence_df.pivot_table(
        index='gene_id',
        columns='go_name',
        values='score_delta',
        aggfunc='mean',
    )
    return pivot.fillna(0.0)


def consequence_summary(consequence_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise GAIN/LOSS/NEUTRAL counts per GO term."""
    if consequence_df.empty:
        return pd.DataFrame()

    counts = (
        consequence_df
        .groupby(['go_name', 'consequence'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ['GAIN', 'LOSS', 'NEUTRAL', 'UNKNOWN']:
        if col not in counts.columns:
            counts[col] = 0
    counts['total'] = counts[['GAIN', 'LOSS', 'NEUTRAL']].sum(axis=1)
    counts['gain_pct'] = (counts['GAIN'] / counts['total'].clip(lower=1) * 100).round(1)
    counts['loss_pct'] = (counts['LOSS'] / counts['total'].clip(lower=1) * 100).round(1)
    return counts.sort_values('total', ascending=False)
