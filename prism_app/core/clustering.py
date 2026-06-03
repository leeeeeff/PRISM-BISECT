"""Functional module clustering for arbitrary GO score matrices.

Generates a brain_672-compatible module dict from any score matrix
using Ward hierarchical clustering on the GO term axis.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np


def generate_user_modules(
    score_matrix: np.ndarray,
    go_terms: List[str],
    go_names: Optional[Dict[str, str]] = None,
    k_range: Optional[Tuple[int, int]] = None,
    random_state: int = 42,
) -> dict:
    """Cluster GO terms into functional modules via Ward linkage.

    Parameters
    ----------
    score_matrix : ndarray (n_isoforms, n_go)
    go_terms     : list of GO IDs (length n_go)
    go_names     : optional dict {GO_ID: name}
    k_range      : (k_min, k_max) for silhouette search. Auto-derived if None.
    random_state : for reproducibility

    Returns
    -------
    dict compatible with brain_672 modules format:
        n_go, n_modules, best_silhouette, go_module_map, modules
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    n_iso, n_go = score_matrix.shape
    if go_names is None:
        go_names = {}

    if n_go < 4:
        return _trivial_modules(go_terms, go_names)

    # Cluster on GO term axis: each GO term is a vector of isoform scores
    # StandardScale across isoforms per GO term (zero-mean, unit-var)
    X = score_matrix.T  # shape (n_go, n_iso) — one row per GO term
    X_scaled = StandardScaler().fit_transform(X)  # (n_go, n_iso)

    # Ward linkage on cosine distance between GO term profiles
    try:
        dist = pdist(X_scaled, metric='cosine')
    except Exception:
        dist = pdist(X_scaled, metric='euclidean')
    Z = linkage(dist, method='ward')

    # Silhouette-based k selection
    if k_range is None:
        k_min = max(2, min(3,  n_go // 6))
        k_max = min(n_go - 1, max(4, n_go // 3))
    else:
        k_min, k_max = k_range

    k_max = min(k_max, n_go - 1)
    if k_min >= k_max:
        k_min = max(2, k_max - 1)

    best_k, best_sil, best_labels = k_min, -1.0, None
    sil_by_k: Dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) < 2:
            continue
        try:
            sil = float(silhouette_score(X_scaled, labels, metric='cosine'))
        except Exception:
            sil = float(silhouette_score(X_scaled, labels))
        sil_by_k[k] = round(sil, 4)
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels.copy()

    if best_labels is None:
        best_labels = fcluster(Z, k_min, criterion='maxclust')
        best_sil = 0.0

    # Build module dict (1-indexed, matching brain_672 format)
    go_module_map: Dict[str, int] = {}
    modules_by_id: Dict[int, List[str]] = {}
    for go_id, mod_id in zip(go_terms, best_labels):
        mid = int(mod_id)
        go_module_map[go_id] = mid
        modules_by_id.setdefault(mid, []).append(go_id)

    modules: Dict[str, dict] = {}
    for mid, go_ids in sorted(modules_by_id.items()):
        names = [go_names.get(g, g)[:40] for g in go_ids]
        label = ' / '.join(names[:3])
        modules[str(mid)] = {
            'module_id': str(mid),
            'size':      str(len(go_ids)),
            'go_ids':    go_ids,
            'label':     label[:80],
            'top3_names': names[:3],
        }

    return {
        'n_go':            n_go,
        'n_modules':       best_k,
        'best_silhouette': round(best_sil, 4),
        'silhouette_by_k': sil_by_k,
        'go_module_map':   go_module_map,
        'modules':         modules,
    }


def assign_isoforms_to_modules(
    score_matrix: np.ndarray,
    go_terms: List[str],
    module_dict: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each isoform to its primary module.

    Returns
    -------
    primary_module : int array (n_isoforms,)  — 1-indexed module ID
    module_score   : float array (n_isoforms,) — mean score for primary module
    """
    go_module_map = module_dict.get('go_module_map', {})
    go_idx = {g: i for i, g in enumerate(go_terms)}
    n_modules = module_dict['n_modules']

    # Build (n_isoforms, n_modules) matrix of mean scores per module
    mod_score_matrix = np.zeros((len(score_matrix), n_modules), dtype=np.float32)
    mod_count = np.zeros(n_modules, dtype=np.int32)

    for go_id, mid in go_module_map.items():
        gi = go_idx.get(go_id)
        if gi is None:
            continue
        mi = mid - 1  # 0-indexed
        if 0 <= mi < n_modules:
            mod_score_matrix[:, mi] += score_matrix[:, gi]
            mod_count[mi] += 1

    mod_count_safe = np.where(mod_count > 0, mod_count, 1)
    mod_score_matrix /= mod_count_safe[np.newaxis, :]

    primary_module = mod_score_matrix.argmax(axis=1) + 1  # 1-indexed
    module_score   = mod_score_matrix.max(axis=1)

    return primary_module, module_score


def _trivial_modules(go_terms: List[str], go_names: Dict[str, str]) -> dict:
    """Fallback for tiny GO panels (n_go < 4): put all in one module."""
    names = [go_names.get(g, g)[:40] for g in go_terms]
    return {
        'n_go':            len(go_terms),
        'n_modules':       1,
        'best_silhouette': 0.0,
        'silhouette_by_k': {1: 0.0},
        'go_module_map':   {g: 1 for g in go_terms},
        'modules': {
            '1': {
                'module_id': '1',
                'size':      str(len(go_terms)),
                'go_ids':    go_terms,
                'label':     ' / '.join(names[:3]),
                'top3_names': names[:3],
            }
        },
    }
