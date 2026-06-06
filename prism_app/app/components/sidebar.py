"""Streamlit sidebar: data upload, mode selection, and global settings."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from prism_app.core.go_utils import TISSUE_PRESETS, GO_FULL_NAMES

# ── Session-state initialisation (call once per app boot) ─────────────────────

def _init_session_state() -> None:
    defaults = {
        'search_gene':    '',
        'basket_genes':   [],
        'active_module':  None,
        'analysis_step':  {},   # {'qc': True, 'landscape': True, ...}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Demo data directory (bundled with the package)
DEMO_DIR = Path(__file__).parents[2] / 'data' / 'demo'

_TISSUE_OPTIONS = {
    'Skeletal Muscle (18 GO terms)':                     'muscle',
    'Brain — 18-term Panel (zero-shot)':                 'brain',
    'Brain — 41-term Panel (AUPRC 0.672, zero-shot)':   'brain_41',
    'Brain — Extended Novel (73 GO terms)':              'brain_extended',
    'Brain — Full Module Landscape (672 GO terms)':      'brain_672',
    'Muscle Training Terms Only':                        'muscle_only',
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
        search_gene     : str  (persistent gene search query)
    """
    _init_session_state()

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
    tissue_key = _TISSUE_OPTIONS.get(tissue_label, 'muscle')
    if tissue_key not in TISSUE_PRESETS:
        st.sidebar.warning(f"GO panel '{tissue_key}' not available — defaulting to Muscle 18.")
        tissue_key = 'muscle'
    go_preset  = TISSUE_PRESETS[tissue_key]
    go_terms   = list(go_preset.keys())
    go_names   = {**GO_FULL_NAMES, **go_preset}

    # Optional: let user de-select specific GO terms (disabled for large panels)
    if len(go_terms) <= 100:
        with st.sidebar.expander("Customise GO terms", expanded=False):
            selected = st.multiselect(
                "Active GO terms",
                options=go_terms,
                default=go_terms,
                format_func=lambda g: f"{g}: {go_preset.get(g, g)[:35]}",
            )
            if selected:
                go_terms = selected
    else:
        st.sidebar.caption(f"📋 {len(go_terms)} GO terms loaded (large panel — customisation disabled)")

    st.sidebar.divider()

    # ── Score threshold ───────────────────────────────────────────────────
    score_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Isoforms with score > threshold are counted as high-confidence predictions. "
             "Default 0.4 reproduces the paper's S1×BISECT cross-link count (32 genes).",
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
        search_gene='',
    )

    # ── Demo mode: load bundled data ──────────────────────────────────────
    if mode_key == 'demo':
        cfg.update(_load_demo_data(tissue_key, go_terms))
        _render_demo_context(tissue_key)

    # ── Upload mode ───────────────────────────────────────────────────────
    else:
        st.sidebar.subheader("Upload files")
        cfg.update(_upload_section(go_terms))
        _render_upload_context(cfg)
        _render_auto_cluster_panel(cfg, go_terms, go_names)

    # ── Active features panel ─────────────────────────────────────────────
    _render_active_features(mode_key, tissue_key, cfg)

    # ── Persistent gene search (always visible) ───────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("**🔍 Gene Search**")
    gene_query = st.sidebar.text_input(
        "Gene / isoform ID",
        value=st.session_state.get('search_gene', ''),
        placeholder="e.g. KIF21B, NDUFS4",
        key='sidebar_gene_input',
        label_visibility='collapsed',
    )
    if gene_query != st.session_state.get('search_gene', ''):
        st.session_state['search_gene'] = gene_query
    if gene_query:
        if st.sidebar.button("➕ Add to basket", key='sidebar_add_basket'):
            basket = st.session_state.get('basket_genes', [])
            if gene_query not in basket:
                basket.append(gene_query)
                st.session_state['basket_genes'] = basket
        st.sidebar.caption("*Enter gene name → navigate to Targets page for detailed analysis*")
    cfg['search_gene'] = st.session_state.get('search_gene', '')

    # ── Analysis progress + basket ────────────────────────────────────────
    _render_progress_panel()

    return cfg


def _render_auto_cluster_panel(cfg: dict, go_terms: list, go_names: dict) -> None:
    """Upload mode only: offer automatic functional module generation."""
    sm = cfg.get('score_matrix')
    if sm is None or len(go_terms) < 4:
        return

    existing = st.session_state.get('user_modules')
    st.sidebar.divider()
    st.sidebar.markdown("**🧩 기능 모듈 자동 생성**")

    if existing is not None:
        n_mods = existing.get('n_modules', '?')
        sil    = existing.get('best_silhouette', 0)
        st.sidebar.markdown(
            f"<div style='background:#f0fdf4;border-radius:6px;padding:6px 10px;"
            f"font-size:0.8rem;color:#15803d'>"
            f"✅ {n_mods}개 모듈 생성 완료 (silhouette={sil:.3f})</div>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("♻️ 재생성", key='recluster'):
            del st.session_state['user_modules']
            st.rerun()
    else:
        n_go = len(go_terms)
        st.sidebar.caption(
            f"{n_go}개 GO term → Ward 계층 클러스터링으로 기능 모듈 자동 생성. "
            "Module Landscape 페이지가 활성화됩니다."
        )
        if st.sidebar.button("⚙️ 모듈 생성 (10-30초)", key='run_cluster'):
            with st.sidebar.spinner("클러스터링 중…"):
                try:
                    from prism_app.core.clustering import generate_user_modules
                    result = generate_user_modules(sm, go_terms, go_names)
                    st.session_state['user_modules'] = result
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"클러스터링 오류: {e}")


def _render_active_features(mode: str, tissue: str, cfg: dict) -> None:
    """Show which analysis features are available for the current data config."""
    has_sm = cfg.get('score_matrix') is not None
    has_dtu = cfg.get('dtu_df') is not None
    n_go = len(cfg.get('go_terms', []))
    has_modules = tissue == 'brain_672' or st.session_state.get('user_modules') is not None

    features = [
        ('QC & Overview',       True,        '항상'),
        ('Target Analysis',     has_sm,      '스코어 데이터 있을 때'),
        ('Functional Patterns', has_sm,      '스코어 데이터 있을 때'),
        ('Module Landscape',    has_modules, 'Brain-672 또는 모듈 생성 후'),
        ('Condition Analysis',  has_dtu,     'DTU 데이터 있을 때'),
        ('Advanced',            True,        '항상 (count matrix 업로드 필요)'),
    ]

    st.sidebar.divider()
    with st.sidebar.expander("🛠️ 현재 활성 기능", expanded=False):
        for name, active, condition in features:
            icon = "✅" if active else "⬜"
            color = "#15803d" if active else "#9ca3af"
            st.markdown(
                f"<span style='font-size:0.78rem;color:{color}'>{icon} **{name}**</span>  \n"
                f"<span style='font-size:0.72rem;color:#9ca3af;padding-left:18px'>{condition}</span>",
                unsafe_allow_html=True,
            )
        if not has_modules:
            st.caption("Module Landscape 활성화: 사이드바에서 'Brain — Full Module Landscape (672)' 선택")


def _render_progress_panel() -> None:
    """Show analysis progress steps and gene basket in sidebar."""
    steps = st.session_state.get('analysis_step', {})
    basket = st.session_state.get('basket_genes', [])
    active_mod = st.session_state.get('active_module', None)

    step_labels = [
        ('hub',       '🏠 분석 시작'),
        ('qc',        '📊 데이터 QC'),
        ('landscape', '🗺️ 모듈 지형도'),
        ('patterns',  '🔬 기능 패턴'),
        ('condition', '🔄 조건별 변화'),
        ('targets',   '🎯 후보 타겟'),
    ]

    completed = [k for k, _ in step_labels if steps.get(k)]
    if not completed and not basket and active_mod is None:
        return  # nothing to show yet

    st.sidebar.divider()

    if completed:
        st.sidebar.markdown("**분석 진행 상황**")
        for k, label in step_labels:
            icon = "✅" if steps.get(k) else "⬜"
            st.sidebar.markdown(f"<span style='font-size:0.8rem'>{icon} {label}</span>",
                                unsafe_allow_html=True)

    if active_mod is not None:
        st.sidebar.markdown(
            f"<div style='background:#eff6ff;border-radius:4px;padding:4px 8px;"
            f"font-size:0.8rem;color:#1d4ed8;margin-top:6px'>"
            f"🔖 활성 모듈: <b>M{active_mod}</b></div>",
            unsafe_allow_html=True,
        )

    if basket:
        st.sidebar.markdown(
            f"<div style='background:#f0fdf4;border-radius:4px;padding:4px 8px;"
            f"font-size:0.8rem;color:#15803d;margin-top:6px'>"
            f"🧬 후보 바스켓 ({len(basket)}개): "
            f"<b>{', '.join(basket[:5])}{'…' if len(basket)>5 else ''}</b></div>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🗑️ 바스켓 초기화", key='clear_basket'):
            st.session_state['basket_genes'] = []
            st.rerun()


def _render_demo_context(tissue: str) -> None:
    """Show demo data info in sidebar."""
    info = {
        'muscle':         ("근골격근", "36,748", "18", False),
        'brain':          ("뇌 (zero-shot)", "63,994", "18", True),
        'brain_41':       ("뇌 41-term (zero-shot, AUPRC 0.672)", "63,994", "41", True),
        'brain_extended': ("뇌 전체 확장", "63,994", "73", True),
        'brain_672':      ("뇌 전체 모듈 (672 BP GO)", "63,994", "672", True),
        'muscle_only':    ("근골격근", "36,748", "18", False),
    }.get(tissue, ("—", "—", "—", False))
    tissue_name, n_iso, n_go, has_dtu = info

    dtu_html = (
        "<span style='color:#15803d'>포함 (AD vs CT, 8 cell types)</span> → S1·S2 활성"
        if has_dtu else
        "<span style='color:#dc2626'>미포함</span> → S1·S2 비활성"
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""<div style='background:#f0f7ff;border-radius:6px;padding:8px 10px;
        font-size:0.8rem;color:#1e40af'>
        📂 <b>Demo 데이터</b><br>
        조직: {tissue_name}<br>
        아이소폼: {n_iso}개<br>
        GO 패널: {n_go}개 term<br>
        DTU: {dtu_html}<br>
        PRISM 스코어: 논문 사전 계산값
        </div>""",
        unsafe_allow_html=True,
    )


def _render_upload_context(cfg: dict) -> None:
    """Show upload status in sidebar."""
    sm = cfg.get('score_matrix')
    has_dtu = cfg.get('dtu_df') is not None

    if sm is None:
        return

    n_iso, n_go = sm.shape
    dtu_str = "✅ 로드됨 → S1·S2 활성" if has_dtu else "❌ 없음 → S1·S2 비활성"

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""<div style='background:#f0fdf4;border-radius:6px;padding:8px 10px;
        font-size:0.8rem;color:#14532d'>
        📂 <b>업로드 데이터</b><br>
        아이소폼: {n_iso:,}개 · GO: {n_go}개<br>
        DTU: {dtu_str}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Demo data loader ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading demo data…")
def _load_demo_data(tissue: str, go_terms: list) -> dict:
    """Load pre-computed demo data; cached per session."""
    files = {
        'muscle':         ('muscle_scores.npy',                      'muscle_ids.npy',                      'muscle_types.npy',                      'muscle_gene_ids.npy'),
        'brain':          ('brain_full_scores.npy',                  'brain_full_ids.npy',                  'brain_full_types.npy',                  'brain_full_gene_ids.npy'),
        'brain_41':       ('brain_full_expanded_41_scores.npy',      'brain_full_expanded_41_ids.npy',      'brain_full_expanded_41_types.npy',      'brain_full_expanded_41_gene_ids.npy'),
        'brain_extended': ('brain_full_extended_scores.npy',         'brain_full_extended_ids.npy',         'brain_full_extended_types.npy',         'brain_full_extended_gene_ids.npy'),
        'brain_672':      ('brain_full_672_scores.npy',              'brain_full_672_ids.npy',              'brain_full_672_types.npy',              'brain_full_672_gene_ids.npy'),
        'muscle_only':    ('muscle_scores.npy',                      'muscle_ids.npy',                      'muscle_types.npy',                      'muscle_gene_ids.npy'),
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
    if len(go_terms) < len(all_go):
        go_idx_map = {g: i for i, g in enumerate(all_go)}
        col_idx = [go_idx_map[g] for g in go_terms if g in go_idx_map]
        if col_idx:
            score_matrix = score_matrix[:, col_idx]

    # Load DTU if available for this tissue
    dtu_df = _load_demo_dtu(tissue)

    return dict(
        score_matrix=score_matrix,
        isoform_ids=isoform_ids,
        isoform_types=isoform_types,
        gene_ids=gene_ids,
        dtu_df=dtu_df,
    )


@st.cache_data(show_spinner=False)
def _load_demo_dtu(tissue: str) -> Optional[pd.DataFrame]:
    """Load bundled DTU results for demo tissues that have one."""
    _DTU_FILES = {
        'brain':          'brain_dtu.tsv',
        'brain_41':       'brain_dtu.tsv',
        'brain_extended': 'brain_dtu.tsv',
        'brain_672':      'brain_dtu.tsv',
    }
    fname = _DTU_FILES.get(tissue)
    if fname is None:
        return None
    p = DEMO_DIR / fname
    if not p.exists():
        return None
    return pd.read_csv(p, sep='\t')


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
