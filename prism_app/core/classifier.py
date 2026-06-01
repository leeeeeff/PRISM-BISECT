"""4-Scenario Classifier: assigns each isoform to one of four functional categories.

Scenario 1 — Functional Switch:     DTU(+) + Novel GO prediction(+)
Scenario 2 — Expression Switch:     DTU(+) + Novel GO prediction(-)
Scenario 3 — Constitutive Novel:    DTU(-) + Novel GO prediction(+)
Scenario 4 — Background/Unknown:    DTU(-) + Novel GO prediction(-)
"""
from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class IsoformScenario(IntEnum):
    FUNCTIONAL_SWITCH    = 1  # DTU+ / novel GO+
    EXPRESSION_SWITCH    = 2  # DTU+ / novel GO-
    CONSTITUTIVE_NOVEL   = 3  # DTU- / novel GO+
    BACKGROUND           = 4  # DTU- / novel GO-


_SCENARIO_LABELS = {
    IsoformScenario.FUNCTIONAL_SWITCH:  "Scenario 1: Functional Switch",
    IsoformScenario.EXPRESSION_SWITCH:  "Scenario 2: Expression Switch",
    IsoformScenario.CONSTITUTIVE_NOVEL: "Scenario 3: Constitutive Novel Function",
    IsoformScenario.BACKGROUND:         "Scenario 4: Background/Unknown",
}

_SCENARIO_COLORS = {
    IsoformScenario.FUNCTIONAL_SWITCH:  "#e63946",  # red — highest priority
    IsoformScenario.EXPRESSION_SWITCH:  "#f4a261",  # orange
    IsoformScenario.CONSTITUTIVE_NOVEL: "#2a9d8f",  # teal
    IsoformScenario.BACKGROUND:         "#adb5bd",  # grey
}


def _assign_scenario(has_dtu: bool, has_novel_go: bool) -> IsoformScenario:
    if has_dtu and has_novel_go:
        return IsoformScenario.FUNCTIONAL_SWITCH
    if has_dtu and not has_novel_go:
        return IsoformScenario.EXPRESSION_SWITCH
    if not has_dtu and has_novel_go:
        return IsoformScenario.CONSTITUTIVE_NOVEL
    return IsoformScenario.BACKGROUND


def classify_isoforms(
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    gene_ids: Optional[np.ndarray] = None,
    go_terms: Optional[List[str]] = None,
    existing_annotations: Optional[Dict[str, List[str]]] = None,
    dtu_df: Optional[pd.DataFrame] = None,
    score_threshold: float = 0.5,
    dtu_pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """Classify every isoform into one of four functional scenarios.

    Parameters
    ----------
    score_matrix : ndarray, shape (n_isoforms, n_go)
        PRISM predicted GO scores.
    isoform_ids : ndarray, shape (n_isoforms,)
        Transcript identifiers (e.g. ENST00000001234 or IsoQuant tr-style).
    gene_ids : ndarray, shape (n_isoforms,), optional
        Gene identifiers matching each isoform. Used for within-gene ranking.
    go_terms : list of str, optional
        Ordered GO term IDs corresponding to score_matrix columns.
        If None, columns are labelled GO_0, GO_1, ...
    existing_annotations : dict {gene_symbol: [GO:xxx, ...]}, optional
        Current database annotation. Used to distinguish *novel* GO predictions
        (GO terms NOT in existing annotation) from confirmed ones.
        If None, all high-scoring terms are treated as "novel GO+".
    dtu_df : DataFrame, optional
        DTU results. Expected columns: isoform_id (str), pvalue (float),
        delta_if (float, optional). If None, all isoforms are treated as DTU(-).
    score_threshold : float
        Minimum score to consider a GO term "high confidence" (default 0.5).
    dtu_pval_threshold : float
        Maximum adjusted p-value for DTU flag (default 0.05).

    Returns
    -------
    DataFrame with columns:
        isoform_id, gene_id, max_score, max_go, max_go_rank,
        n_high_go, novel_go_terms, dtu_pvalue, dtu_delta_if,
        dtu_flag, novel_go_flag, scenario, scenario_label, scenario_color
    """
    n_iso, n_go = score_matrix.shape
    isoform_ids = np.asarray(isoform_ids, dtype=str)

    if go_terms is None:
        go_terms = [f"GO_{i}" for i in range(n_go)]

    # ── DTU lookup ──────────────────────────────────────────────────────────
    dtu_pvals: Dict[str, float] = {}
    dtu_deltas: Dict[str, float] = {}
    if dtu_df is not None:
        id_col  = _find_col(dtu_df, ['isoform_id', 'transcript_id', 'isoform', 'id'])
        pv_col  = _find_col(dtu_df, ['pvalue', 'p_value', 'padj', 'adj_pvalue'])
        dif_col = _find_col(dtu_df, ['delta_if', 'deltaIF', 'dIF', 'delta'], required=False)
        for _, row in dtu_df.iterrows():
            tid = str(row[id_col])
            dtu_pvals[tid]  = float(row[pv_col])
            dtu_deltas[tid] = float(row[dif_col]) if dif_col else float('nan')

    # ── Gene annotation lookup (for novel GO detection) ────────────────────
    # Map isoform → gene symbol for annotation lookup
    gene_ids_arr = np.asarray(gene_ids, dtype=str) if gene_ids is not None else None

    rows = []
    for i, iso_id in enumerate(isoform_ids):
        scores = score_matrix[i]
        gene_id = gene_ids_arr[i] if gene_ids_arr is not None else ""

        # Max-scoring GO term
        max_idx  = int(np.argmax(scores))
        max_go   = go_terms[max_idx]
        max_score = float(scores[max_idx])

        # All high-scoring GO terms
        high_mask = scores > score_threshold
        high_go   = [go_terms[j] for j in range(n_go) if high_mask[j]]
        n_high    = int(high_mask.sum())

        # Novel GO: high-scoring terms not in existing annotation for this gene
        if existing_annotations is not None:
            known_terms = set(existing_annotations.get(gene_id, []))
            novel_go_terms = [t for t in high_go if t not in known_terms]
        else:
            novel_go_terms = list(high_go)  # treat all as novel if no annotation

        # DTU flag
        pval    = dtu_pvals.get(iso_id, float('nan'))
        delta   = dtu_deltas.get(iso_id, float('nan'))
        has_dtu = (not np.isnan(pval)) and (pval < dtu_pval_threshold)

        has_novel_go = len(novel_go_terms) > 0
        scenario = _assign_scenario(has_dtu, has_novel_go)

        rows.append({
            'isoform_id':     iso_id,
            'gene_id':        gene_id,
            'max_score':      round(max_score, 4),
            'max_go':         max_go,
            'max_go_rank':    int(np.argsort(scores)[::-1].tolist().index(max_idx)) + 1,
            'n_high_go':      n_high,
            'novel_go_terms': ';'.join(novel_go_terms),
            'dtu_pvalue':     round(pval, 6) if not np.isnan(pval) else None,
            'dtu_delta_if':   round(delta, 4) if not np.isnan(delta) else None,
            'dtu_flag':       has_dtu,
            'novel_go_flag':  has_novel_go,
            'scenario':       int(scenario),
            'scenario_label': _SCENARIO_LABELS[scenario],
            'scenario_color': _SCENARIO_COLORS[scenario],
        })

    df = pd.DataFrame(rows)
    return df


def scenario_summary(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-scenario counts and percentages."""
    total = len(classified_df)
    summary = (
        classified_df.groupby(['scenario', 'scenario_label'])
        .size()
        .reset_index(name='count')
    )
    summary['pct'] = (summary['count'] / total * 100).round(1)
    summary = summary.sort_values('scenario').reset_index(drop=True)
    return summary


def get_scenario_candidates(
    classified_df: pd.DataFrame,
    scenario: int,
    min_score: float = 0.5,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Filter to isoforms of a given scenario, sorted by max_score."""
    subset = classified_df[
        (classified_df['scenario'] == scenario) &
        (classified_df['max_score'] >= min_score)
    ].sort_values('max_score', ascending=False)
    if top_n:
        subset = subset.head(top_n)
    return subset.reset_index(drop=True)


# ── helpers ─────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in DataFrame columns: {list(df.columns)}")
    return None
