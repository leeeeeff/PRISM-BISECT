"""Streamlit sidebar: data upload, mode selection, and global settings."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from prism_app.core.go_utils import TISSUE_PRESETS, GO_FULL_NAMES

# Demo data directory (bundled with the package)
DEMO_DIR = Path(__file__).parents[2] / 'data' / 'demo'

_TISSUE_OPTIONS = {
    'Skeletal Muscle (18 GO terms)':            'muscle',
    'Brain — 18-term Panel (zero-shot)':        'brain',
    'Brain — Extended Novel (73 GO terms)':     'brain_extended',
    'Muscle Training Terms Only':               'muscle_only',
}


def render_sidebar() -> dict:
    """Render the sidebar and return the resolved session configuration.

    Returns
    -------
    dict with keys:
        mode            : 'demo' | 'upload'
        tissue          : str (tissue preset key)
        go_terms        : list of str (selected GO IDs)
        go_names        : dict {GO_ID: name}
        score_threshold : float
        score_matrix    : ndarray (n_isoforms, n_go)
        isoform_ids     : ndarray of str
        isoform_types   : ndarray of str | None
        gene_ids        : ndarray of str | None
        dtu_df          : DataFrame | None
    """
    st.sidebar.markdown(
        "**PRISM + BISECT** `v0.1.0`  \n"
        "*Isoform Function Analysis*"
    )
    st.sidebar.title("PRISM Analysis")

    # ── Mode selection ────────────────────────────────────────────────────
    mode = st.sidebar.radio(
        "Data source",
        options=['Demo (paper data)', 'Upload my data'],
        index=0,
        help="Demo: explore the published muscle/brain results. Upload: analyse your own long-read data.",
    )
    mode_key = 'demo' if mode.startswith('Demo') else 'upload'

    st.sidebar.divider()

    # ── Tissue / GO preset ────────────────────────────────────────────────
    tissue_label = st.sidebar.selectbox(
        "Tissue / GO panel",
        list(_TISSUE_OPTIONS.keys()),
        index=0,
    )
    tissue_key = _TISSUE_OPTIONS[tissue_label]
    go_preset  = TISSUE_PRESETS[tissue_key]
    go_terms   = list(go_preset.keys())
    go_names   = {**GO_FULL_NAMES, **go_preset}

    # Optional: let user de-select specific GO terms
    with st.sidebar.expander("Customise GO terms", expanded=False):
        selected = st.multiselect(
            "Active GO terms",
            options=go_terms,
            default=go_terms,
            format_func=lambda g: f"{g}: {go_preset.get(g, g)[:35]}",
        )
        if selected:
            go_terms = selected

    st.sidebar.divider()

    # ── Score threshold ───────────────────────────────────────────────────
    score_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Isoforms with score > threshold are counted as high-confidence predictions.",
    )

    cfg = dict(
        mode=mode_key,
        tissue=tissue_key,
        go_terms=go_terms,
        go_names=go_names,
        score_threshold=score_threshold,
        score_matrix=None,
        isoform_ids=None,
        isoform_types=None,
        gene_ids=None,
        dtu_df=None,
    )

    # ── Demo mode: load bundled data ──────────────────────────────────────
    if mode_key == 'demo':
        cfg.update(_load_demo_data(tissue_key, go_terms))

    # ── Upload mode ───────────────────────────────────────────────────────
    else:
        st.sidebar.subheader("Upload files")
        cfg.update(_upload_section(go_terms))

    return cfg


# ── Demo data loader ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading demo data…")
def _load_demo_data(tissue: str, go_terms: list) -> dict:
    """Load pre-computed demo data; cached per session."""
    files = {
        'muscle':         ('muscle_scores.npy',     'muscle_ids.npy',     'muscle_types.npy',     'muscle_gene_ids.npy'),
        'brain':          ('brain_full_scores.npy',  'brain_full_ids.npy', 'brain_full_types.npy', 'brain_full_gene_ids.npy'),
        'brain_extended': ('brain_novel_scores.npy', 'brain_novel_ids.npy', None, None),
        'muscle_only':    ('muscle_scores.npy',     'muscle_ids.npy',     'muscle_types.npy',     'muscle_gene_ids.npy'),
    }
    score_f, id_f, type_f, gene_f = files.get(tissue, files['muscle'])

    def _load_npy(fname):
        if fname is None:
            return None
        p = DEMO_DIR / fname
        if p.exists():
            return np.load(p, allow_pickle=True)
        return None

    score_matrix  = _load_npy(score_f)
    isoform_ids   = _load_npy(id_f)
    isoform_types = _load_npy(type_f)
    gene_ids      = _load_npy(gene_f)

    if score_matrix is None or isoform_ids is None:
        st.sidebar.warning(f"Demo data not found in {DEMO_DIR}. Run `prism-app prepare-demo` first.")
        return {}

    # Slice columns to match selected GO terms
    all_go = list(TISSUE_PRESETS[tissue].keys())
    col_idx = [all_go.index(g) for g in go_terms if g in all_go]
    if col_idx and score_matrix.shape[1] > len(col_idx):
        score_matrix = score_matrix[:, col_idx]

    return dict(
        score_matrix=score_matrix,
        isoform_ids=isoform_ids,
        isoform_types=isoform_types,
        gene_ids=gene_ids,
    )


# ── Upload section ────────────────────────────────────────────────────────────

def _upload_section(go_terms: list) -> dict:
    """File uploader widgets. Returns partial config dict."""
    score_file  = st.sidebar.file_uploader("Score matrix (.npy, n×GO)", type=['npy'])
    id_file     = st.sidebar.file_uploader("Isoform IDs (.npy or .txt)", type=['npy', 'txt'])
    type_file   = st.sidebar.file_uploader("Isoform types (.npy or .txt, optional)", type=['npy', 'txt'])
    gene_file   = st.sidebar.file_uploader("Gene IDs (.npy or .txt, optional)", type=['npy', 'txt'])
    dtu_file    = st.sidebar.file_uploader("DTU results (.tsv, optional)", type=['tsv', 'csv'])

    result = dict(score_matrix=None, isoform_ids=None, isoform_types=None, gene_ids=None, dtu_df=None)

    if score_file:
        result['score_matrix'] = np.load(score_file, allow_pickle=True)
    if id_file:
        result['isoform_ids'] = _load_id_file(id_file)
    if type_file:
        result['isoform_types'] = _load_id_file(type_file)
    if gene_file:
        result['gene_ids'] = _load_id_file(gene_file)
    if dtu_file:
        sep = '\t' if dtu_file.name.endswith('.tsv') else ','
        result['dtu_df'] = pd.read_csv(dtu_file, sep=sep)

    return result


def _load_id_file(f) -> np.ndarray:
    if f.name.endswith('.npy'):
        return np.load(f, allow_pickle=True)
    lines = f.read().decode('utf-8').strip().splitlines()
    return np.array(lines, dtype=str)
