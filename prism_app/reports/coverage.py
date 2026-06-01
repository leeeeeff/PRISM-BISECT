"""Coverage Summary Report (Module A1).

Generates a statistical overview of PRISM predictions across all isoforms,
broken down by isoform structural type (known / NIC / NNIC) and GO term.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class CoverageReport:
    # Overall counts
    total_isoforms:   int = 0
    n_known:          int = 0   # Ensembl-annotated
    n_nic:            int = 0   # Novel In Catalog
    n_nnic:           int = 0   # Novel Not In Catalog
    n_with_any_high:  int = 0   # score > threshold for ≥1 GO term
    pct_with_any_high: float = 0.0

    # Per-type breakdown
    n_known_with_high: int = 0
    n_nic_with_high:   int = 0
    n_nnic_with_high:  int = 0
    pct_known_with_high: float = 0.0
    pct_nic_with_high:   float = 0.0
    pct_nnic_with_high:  float = 0.0

    # Per-GO-term stats (list of dicts)
    per_go: List[Dict] = field(default_factory=list)

    # Top predicted functions
    top_go_by_count: List[Dict] = field(default_factory=list)

    # Score distribution
    score_mean:   float = 0.0
    score_median: float = 0.0
    score_max:    float = 0.0

    # Settings used
    score_threshold: float = 0.5
    go_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            f"Total isoforms:        {self.total_isoforms:>8,}",
            f"  Known (Ensembl):     {self.n_known:>8,}  ({100*self.n_known/max(1,self.total_isoforms):.1f}%)",
            f"  NIC:                 {self.n_nic:>8,}  ({100*self.n_nic/max(1,self.total_isoforms):.1f}%)",
            f"  NNIC:                {self.n_nnic:>8,}  ({100*self.n_nnic/max(1,self.total_isoforms):.1f}%)",
            "",
            f"With ≥1 GO score>{self.score_threshold}:  {self.n_with_any_high:>8,}  ({self.pct_with_any_high:.1f}%)",
            f"  Known:               {self.n_known_with_high:>8,}  ({self.pct_known_with_high:.1f}%)",
            f"  NIC:                 {self.n_nic_with_high:>8,}  ({self.pct_nic_with_high:.1f}%)",
            f"  NNIC:                {self.n_nnic_with_high:>8,}  ({self.pct_nnic_with_high:.1f}%)",
            "",
            f"GO terms evaluated:    {len(self.go_terms)}",
            f"Score stats:  mean={self.score_mean:.3f}  median={self.score_median:.3f}  max={self.score_max:.3f}",
        ]
        return "\n".join(lines)


# Accepted isoform type labels (case-insensitive)
_KNOWN_LABELS = {'known', 'ensembl', 'full_splice_match', 'fsm', 'ism', 'reference'}
_NIC_LABELS   = {'nic', 'novel_in_catalog', 'novel in catalog'}
_NNIC_LABELS  = {'nnic', 'novel_not_in_catalog', 'novel not in catalog', 'novel'}


def _normalise_type(t: str) -> str:
    t = str(t).lower().strip()
    if t in _KNOWN_LABELS:
        return 'known'
    if t in _NIC_LABELS:
        return 'nic'
    if t in _NNIC_LABELS:
        return 'nnic'
    return 'known'  # default


def generate_coverage_report(
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    isoform_types: Optional[np.ndarray] = None,
    go_terms: Optional[List[str]] = None,
    go_names: Optional[Dict[str, str]] = None,
    score_threshold: float = 0.5,
) -> CoverageReport:
    """Generate a coverage summary report from PRISM score matrix.

    Parameters
    ----------
    score_matrix : ndarray, shape (n_isoforms, n_go)
    isoform_ids  : ndarray of str, shape (n_isoforms,)
    isoform_types : ndarray of str, shape (n_isoforms,), optional
        Structural category per isoform. Recognised values:
        'known'/'ensembl'/'FSM' → known; 'NIC' → NIC; 'NNIC'/'novel' → NNIC.
        If None, all isoforms are treated as 'known'.
    go_terms : list of str, length n_go, optional
        GO IDs in the order matching score_matrix columns.
    go_names : dict {GO_ID: name}, optional
    score_threshold : float
    """
    n_iso, n_go = score_matrix.shape
    isoform_ids = np.asarray(isoform_ids, dtype=str)

    if go_terms is None:
        go_terms = [f"GO_{i}" for i in range(n_go)]
    go_names = go_names or {}

    # ── Isoform type classification ─────────────────────────────────────────
    if isoform_types is None:
        types_norm = np.array(['known'] * n_iso)
    else:
        types_norm = np.array([_normalise_type(t) for t in isoform_types])

    mask_known = types_norm == 'known'
    mask_nic   = types_norm == 'nic'
    mask_nnic  = types_norm == 'nnic'

    # ── High-score mask ─────────────────────────────────────────────────────
    any_high = (score_matrix > score_threshold).any(axis=1)  # (n_iso,)

    # ── Counts ──────────────────────────────────────────────────────────────
    n_known = int(mask_known.sum())
    n_nic   = int(mask_nic.sum())
    n_nnic  = int(mask_nnic.sum())
    n_with  = int(any_high.sum())

    def safe_pct(num, denom):
        return round(100 * num / denom, 1) if denom > 0 else 0.0

    n_known_h = int((any_high & mask_known).sum())
    n_nic_h   = int((any_high & mask_nic).sum())
    n_nnic_h  = int((any_high & mask_nnic).sum())

    # ── Per-GO stats ────────────────────────────────────────────────────────
    per_go = []
    for j, go_id in enumerate(go_terms):
        col = score_matrix[:, j]
        high = col > score_threshold
        per_go.append({
            'go':         go_id,
            'name':       go_names.get(go_id, go_id),
            'n_high':     int(high.sum()),
            'pct_high':   safe_pct(high.sum(), n_iso),
            'mean_score': round(float(col.mean()), 4),
            'max_score':  round(float(col.max()), 4),
            'n_high_known': int((high & mask_known).sum()),
            'n_high_nic':   int((high & mask_nic).sum()),
            'n_high_nnic':  int((high & mask_nnic).sum()),
        })

    top_go = sorted(per_go, key=lambda x: x['n_high'], reverse=True)[:15]

    # ── Score distribution ───────────────────────────────────────────────────
    flat = score_matrix.flatten()

    report = CoverageReport(
        total_isoforms=n_iso,
        n_known=n_known, n_nic=n_nic, n_nnic=n_nnic,
        n_with_any_high=n_with,
        pct_with_any_high=safe_pct(n_with, n_iso),
        n_known_with_high=n_known_h,
        n_nic_with_high=n_nic_h,
        n_nnic_with_high=n_nnic_h,
        pct_known_with_high=safe_pct(n_known_h, n_known),
        pct_nic_with_high=safe_pct(n_nic_h, n_nic),
        pct_nnic_with_high=safe_pct(n_nnic_h, n_nnic),
        per_go=per_go,
        top_go_by_count=top_go,
        score_mean=round(float(flat.mean()), 4),
        score_median=round(float(np.median(flat)), 4),
        score_max=round(float(flat.max()), 4),
        score_threshold=score_threshold,
        go_terms=list(go_terms),
    )
    return report
