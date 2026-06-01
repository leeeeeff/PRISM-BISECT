"""Metrics wrapper — delegates to existing evaluation.py (no duplication)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── Import existing evaluation.py (local research env only) ──────────────────
_EVAL_PATH = Path(__file__).parents[2] / 'hMuscle' / 'results_isoform'
if str(_EVAL_PATH) not in sys.path:
    sys.path.insert(0, str(_EVAL_PATH))

try:
    from evaluation import calculate_metrics, compute_bias_score  # noqa: F401
except ImportError:
    # Not available on Streamlit Community Cloud — provide stubs
    def calculate_metrics(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("evaluation.py not available in this deployment")

    def compute_bias_score(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("evaluation.py not available in this deployment")


def bootstrap_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    gene_ids: np.ndarray,
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Gene-block bootstrap AUPRC with confidence interval.

    Resamples at the gene level (all isoforms of a gene move together),
    consistent with the within-gene evaluation framework.

    Returns
    -------
    dict: mean_auprc, ci_lower, ci_upper, n_bootstrap, std
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    genes = np.unique(gene_ids)
    scores_boot = []

    for _ in range(n_iter):
        sampled_genes = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([np.where(gene_ids == g)[0] for g in sampled_genes])
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores_boot.append(average_precision_score(yt, ys))

    scores_boot = np.array(scores_boot)
    alpha = 1 - ci
    return {
        'mean_auprc': float(np.mean(scores_boot)),
        'ci_lower':   float(np.percentile(scores_boot, 100 * alpha / 2)),
        'ci_upper':   float(np.percentile(scores_boot, 100 * (1 - alpha / 2))),
        'std':        float(np.std(scores_boot)),
        'n_bootstrap': len(scores_boot),
    }


def per_go_metrics(
    score_matrix: np.ndarray,
    label_matrix: np.ndarray,
    gene_ids: np.ndarray,
    go_terms: list,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute per-GO-term AUPRC (+ optional bootstrap CI).

    Parameters
    ----------
    score_matrix : (n_isoforms, n_go)
    label_matrix : (n_isoforms, n_go)  binary
    gene_ids     : (n_isoforms,)
    go_terms     : list of GO IDs, length n_go
    bootstrap    : compute 95% CI via gene-block bootstrap
    """
    from sklearn.metrics import average_precision_score

    rows = []
    for j, go_id in enumerate(go_terms):
        yt = label_matrix[:, j]
        ys = score_matrix[:, j]
        n_pos = int(yt.sum())
        if n_pos == 0:
            continue
        auprc = float(average_precision_score(yt, ys))
        row = {'go': go_id, 'n_pos': n_pos, 'auprc': round(auprc, 4)}
        if bootstrap:
            ci = bootstrap_auprc(yt, ys, gene_ids, n_iter=n_bootstrap)
            row.update({
                'auprc_ci_lower': round(ci['ci_lower'], 4),
                'auprc_ci_upper': round(ci['ci_upper'], 4),
                'auprc_std':      round(ci['std'], 4),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
