"""Known Annotation Validation Report (Module A2).

Wraps hMuscle/results_isoform/evaluation.py metrics for the web app,
adding bootstrap CI and per-GO breakdown without duplicating logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    n_isoforms_with_annotation: int
    n_go_terms: int
    macro_auprc: float
    macro_auprc_ci: Tuple[float, float]   # (lo, hi) 95% bootstrap CI
    per_go: List[Dict]                    # [{go, name, auprc, n_pos, n_neg}]
    notes: str = ''

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_go).sort_values('auprc', ascending=False)


def generate_validation_report(
    score_matrix: np.ndarray,
    isoform_ids: np.ndarray,
    go_terms: List[str],
    go_names: Dict[str, str],
    annotation_path: Optional[str] = None,
    gene_ids: Optional[np.ndarray] = None,
    score_threshold: float = 0.5,
    n_bootstrap: int = 200,
) -> Optional[ValidationReport]:
    """Compute per-GO AUPRC and bootstrap CI against known GO annotations.

    Parameters
    ----------
    annotation_path : path to gene-level GO annotation TSV
        Format: gene_symbol<TAB>GO:xxx<TAB>GO:yyy ...
        If None, tries the default muscle annotation file.
    n_bootstrap : number of bootstrap replicates for CI (lower = faster)

    Returns None if no annotation overlap found.
    """
    from sklearn.metrics import average_precision_score

    # Load annotations
    annot = _load_annotations(annotation_path)
    if not annot:
        return None

    ids_arr  = np.asarray(isoform_ids, dtype=str)
    gene_arr = np.asarray(gene_ids, dtype=str) if gene_ids is not None else None

    # Build label matrix
    y_true = _build_label_matrix(ids_arr, gene_arr, annot, go_terms)
    if y_true is None or y_true.sum() == 0:
        return None

    # Per-GO AUPRC
    per_go = []
    auprcs = []
    for i, g in enumerate(go_terms):
        yt = y_true[:, i]
        yp = score_matrix[:, i]
        n_pos = int(yt.sum())
        if n_pos < 2:
            continue
        try:
            auprc = average_precision_score(yt, yp)
        except Exception:
            continue
        auprcs.append(auprc)
        per_go.append({
            'go':    g,
            'name':  go_names.get(g, g)[:45],
            'auprc': round(auprc, 4),
            'n_pos': n_pos,
            'n_neg': int((yt == 0).sum()),
        })

    if not auprcs:
        return None

    macro = float(np.mean(auprcs))

    # Bootstrap CI on macro AUPRC
    ci_lo, ci_hi = _bootstrap_macro_auprc(
        score_matrix, y_true, go_terms, n_bootstrap
    )

    return ValidationReport(
        n_isoforms_with_annotation=int(y_true.any(axis=1).sum()),
        n_go_terms=len(per_go),
        macro_auprc=round(macro, 4),
        macro_auprc_ci=(round(ci_lo, 4), round(ci_hi, 4)),
        per_go=per_go,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_annotations(path: Optional[str]) -> Dict[str, List[str]]:
    """Load gene → GO term list mapping from TSV."""
    import os
    from pathlib import Path

    candidates = []
    if path:
        candidates.append(path)
    base = Path(__file__).parents[2]
    candidates += [
        # Bundled copy — works on Streamlit Community Cloud and local
        str(base / 'prism_app/data/annotations/human_annotations_unified_bp.txt'),
        # Local research environment fallback
        str(base / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'),
        str(base / 'hMuscle/data/raw_data/data/annotations/ensembl_go_annotations.txt'),
    ]

    for p in candidates:
        if p and os.path.exists(p):
            return _parse_annotation_file(p)
    return {}


def _parse_annotation_file(path: str) -> Dict[str, List[str]]:
    gene_go: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            gene = parts[0]
            terms = [p for p in parts[1:] if p.startswith('GO:')]
            if terms:
                gene_go[gene] = terms
    return gene_go


def _load_ensg2sym() -> Dict[str, str]:
    """Load ENSG → gene symbol mapping.

    Prefers the compact bundled JSON (320 KB, demo data only).
    Falls back to the full TSV (~38 MB) in the local research environment.
    """
    import json
    from pathlib import Path
    base = Path(__file__).parents[2]

    # Compact JSON — bundled, works on Community Cloud
    json_path = base / 'prism_app/data/annotations/ensg_to_symbol.json'
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    # Full TSV — local research environment only
    tsv_path = base / 'hMuscle/data/raw_data/data/id_lists/ensembl_to_symbol.txt'
    if not tsv_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with open(tsv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                mapping[parts[0]] = parts[4]   # ENSG base → symbol
    return mapping


def _build_label_matrix(
    ids_arr: np.ndarray,
    gene_arr: Optional[np.ndarray],
    annot: Dict[str, List[str]],
    go_terms: List[str],
) -> Optional[np.ndarray]:
    """Build binary (n_isoforms, n_go) label matrix from gene annotations."""
    go_idx   = {g: i for i, g in enumerate(go_terms)}
    ensg2sym = _load_ensg2sym()
    n = len(ids_arr)
    m = len(go_terms)
    y = np.zeros((n, m), dtype=np.float32)

    for i, iso_id in enumerate(ids_arr):
        candidates = []
        if gene_arr is not None:
            gs       = str(gene_arr[i])
            ensg_base = gs.split('.')[0]          # strip version
            sym       = ensg2sym.get(ensg_base, ensg_base)
            candidates += [sym, ensg_base, gs]
        candidates += [str(iso_id).split('.')[0], str(iso_id)]

        for key in candidates:
            if key and key in annot:
                for go in annot[key]:
                    if go in go_idx:
                        y[i, go_idx[go]] = 1.0
                break

    return y if y.sum() > 0 else None


def _bootstrap_macro_auprc(
    score_matrix: np.ndarray,
    y_true: np.ndarray,
    go_terms: List[str],
    n_iter: int,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    n = score_matrix.shape[0]
    boot_macros = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        vals = []
        for j in range(len(go_terms)):
            yt = y_true[idx, j]
            if yt.sum() < 2:
                continue
            try:
                vals.append(average_precision_score(yt, score_matrix[idx, j]))
            except Exception:
                pass
        if vals:
            boot_macros.append(np.mean(vals))
    if not boot_macros:
        return (0.0, 0.0)
    alpha = (1 - ci) / 2
    return (float(np.quantile(boot_macros, alpha)),
            float(np.quantile(boot_macros, 1 - alpha)))
