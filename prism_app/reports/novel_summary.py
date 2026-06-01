"""Novel Isoform Function Summary (Module A3).

Characterises novel (NIC/NNIC) isoforms that receive high-confidence
PRISM predictions, identifies top predicted GO functions, and surfaces
representative candidate isoforms per GO term.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd


@dataclass
class GoTermSummary:
    go:              str
    name:            str
    n_high:          int      # isoforms with score > threshold
    mean_score:      float
    in_prism18:      bool     # Was this a PRISM training term?
    top_isoforms:    List[Dict] = field(default_factory=list)  # [{id, score}, ...]


@dataclass
class NovelSummaryReport:
    total_novel:          int = 0
    n_novel_with_any_high: int = 0
    pct_novel_with_high:  float = 0.0
    score_threshold:      float = 0.5
    n_go_evaluated:       int = 0

    # Ranked GO terms by novel isoform count
    go_summary:      List[GoTermSummary] = field(default_factory=list)

    # Top-N novel isoform candidates (across all GO terms)
    top_candidates:  List[Dict] = field(default_factory=list)

    # Breakdown: in-training vs out-of-training GO terms
    n_prism18_terms_with_novel:   int = 0
    n_extended_terms_with_novel:  int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for g in self.go_summary:
            rows.append({
                'GO Term':    g.go,
                'Function':   g.name,
                'N novel (>thr)': g.n_high,
                'Mean score': g.mean_score,
                'In PRISM-18 training': 'Yes' if g.in_prism18 else 'No',
            })
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        lines = [
            f"Novel isoforms analysed: {self.total_novel:,}",
            f"With ≥1 GO score>{self.score_threshold}: {self.n_novel_with_any_high:,} ({self.pct_novel_with_high:.1f}%)",
            f"GO terms evaluated: {self.n_go_evaluated}",
            "",
            f"Top predicted functions:",
        ]
        for g in self.go_summary[:10]:
            marker = "★" if g.in_prism18 else " "
            lines.append(f"  {marker} {g.go}  n={g.n_high:>4}  mean={g.mean_score:.3f}  {g.name[:55]}")
        return "\n".join(lines)


# PRISM-18 muscle training terms (to flag in-training vs extended)
_PRISM_18: Set[str] = {
    'GO:0007204', 'GO:0045214', 'GO:0006941', 'GO:0006914', 'GO:0043161',
    'GO:0007519', 'GO:0042692', 'GO:0055074', 'GO:0007005', 'GO:0007517',
    'GO:0032006', 'GO:0030048', 'GO:0006096', 'GO:0007268', 'GO:0007018',
    'GO:0031175', 'GO:0030182', 'GO:0000226',
}


def generate_novel_summary(
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    isoform_types: Optional[np.ndarray] = None,
    go_terms: Optional[List[str]] = None,
    go_names: Optional[Dict[str, str]] = None,
    score_threshold: float = 0.5,
    top_n_isoforms_per_go: int = 5,
    top_n_candidates: int = 20,
) -> NovelSummaryReport:
    """Generate novel isoform functional prediction summary.

    Parameters
    ----------
    score_matrix  : (n_isoforms, n_go)
    isoform_ids   : (n_isoforms,)
    isoform_types : (n_isoforms,) — 'known' / 'nic' / 'nnic'.
                    If None, all are treated as novel (for standalone use).
    go_terms      : list of GO IDs (len n_go)
    go_names      : {GO_ID: full name}
    score_threshold : high-confidence cut-off
    top_n_isoforms_per_go : representative isoforms to surface per GO term
    top_n_candidates : global top candidates table
    """
    n_iso, n_go = score_matrix.shape
    isoform_ids = np.asarray(isoform_ids, dtype=str)

    if go_terms is None:
        go_terms = [f"GO_{i}" for i in range(n_go)]
    go_names = go_names or {}

    # ── Select novel isoforms ─────────────────────────────────────────────
    if isoform_types is not None:
        types_lower = np.array([str(t).lower().strip() for t in isoform_types])
        novel_mask = np.isin(types_lower, ['nic', 'nnic', 'novel',
                                            'novel_in_catalog',
                                            'novel_not_in_catalog'])
    else:
        novel_mask = np.ones(n_iso, dtype=bool)

    novel_scores = score_matrix[novel_mask]
    novel_ids    = isoform_ids[novel_mask]
    n_novel      = int(novel_mask.sum())

    if n_novel == 0:
        return NovelSummaryReport(total_novel=0, score_threshold=score_threshold)

    any_high = (novel_scores > score_threshold).any(axis=1)
    n_any    = int(any_high.sum())

    # ── Per-GO summary ────────────────────────────────────────────────────
    go_summaries: List[GoTermSummary] = []
    for j, go_id in enumerate(go_terms):
        col       = novel_scores[:, j]
        high_mask = col > score_threshold
        n_high    = int(high_mask.sum())
        if n_high == 0:
            continue
        mean_sc = round(float(col[high_mask].mean()), 3)

        # Top representative isoforms
        top_idx = np.argsort(col)[::-1][:top_n_isoforms_per_go]
        top_isos = [
            {'isoform_id': str(novel_ids[k]), 'score': round(float(col[k]), 3)}
            for k in top_idx if col[k] > score_threshold
        ]

        go_summaries.append(GoTermSummary(
            go=go_id,
            name=go_names.get(go_id, go_id),
            n_high=n_high,
            mean_score=mean_sc,
            in_prism18=(go_id in _PRISM_18),
            top_isoforms=top_isos,
        ))

    go_summaries.sort(key=lambda x: x.n_high, reverse=True)

    # ── Global top candidates ─────────────────────────────────────────────
    max_scores   = novel_scores.max(axis=1)
    max_go_idx   = novel_scores.argmax(axis=1)
    top_global   = np.argsort(max_scores)[::-1][:top_n_candidates]
    top_cands    = []
    for k in top_global:
        if max_scores[k] < score_threshold:
            break
        best_go = go_terms[int(max_go_idx[k])]
        top_cands.append({
            'isoform_id': str(novel_ids[k]),
            'max_score':  round(float(max_scores[k]), 3),
            'max_go':     best_go,
            'max_go_name': go_names.get(best_go, best_go),
            'in_prism18': best_go in _PRISM_18,
        })

    n_p18 = sum(1 for g in go_summaries if g.in_prism18 and g.n_high > 0)
    n_ext = sum(1 for g in go_summaries if not g.in_prism18 and g.n_high > 0)

    return NovelSummaryReport(
        total_novel=n_novel,
        n_novel_with_any_high=n_any,
        pct_novel_with_high=round(100 * n_any / n_novel, 1),
        score_threshold=score_threshold,
        n_go_evaluated=n_go,
        go_summary=go_summaries,
        top_candidates=top_cands,
        n_prism18_terms_with_novel=n_p18,
        n_extended_terms_with_novel=n_ext,
    )
