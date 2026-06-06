"""Analysis context panel — reproducibility metadata shown on every major page."""
from __future__ import annotations
from datetime import datetime
import streamlit as st

PRISM_VERSION   = "v15d_bp_clean"
PRISM_ENSEMBLE  = "5-seed ensemble (seeds 0–4)"
PRISM_TRAINING  = "Human skeletal muscle, 18 BP GO terms"
GO_ONTOLOGY_VER = "2024-09-01 release"
APP_VERSION     = "0.1.0"

_TISSUE_LABEL = {
    'muscle':         'Skeletal Muscle (18 BP GO)',
    'brain':          'Brain — 18-term zero-shot',
    'brain_extended': 'Brain — Extended 73 GO',
    'brain_672':      'Brain — Full Module 672 GO',
    'muscle_only':    'Muscle Training Only',
}


def render_analysis_context(cfg: dict, *, expanded: bool = False) -> None:
    """Collapsible panel showing analysis metadata for reproducibility.

    Call this near the top of every major analysis page so results
    are always traceable to model version, parameters, and data.
    """
    sm     = cfg.get('score_matrix')
    tissue = cfg.get('tissue', '—')
    mode   = cfg.get('mode', 'demo')
    thr    = cfg.get('score_threshold', 0.4)
    n_go   = len(cfg.get('go_terms', []))
    n_iso  = sm.shape[0] if sm is not None else 0
    now    = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    tissue_str = _TISSUE_LABEL.get(tissue, tissue)
    mode_str   = 'Demo (논문 사전 계산)' if mode == 'demo' else 'Upload (사용자 데이터)'

    with st.expander("📋 Analysis Context — 재현성 정보", expanded=expanded):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"**🔬 PRISM 모델**\n"
                f"- Version: `{PRISM_VERSION}`\n"
                f"- Ensemble: `{PRISM_ENSEMBLE}`\n"
                f"- Training: {PRISM_TRAINING}\n"
                f"- GO ontology: `{GO_ONTOLOGY_VER}`\n"
            )
        with col2:
            st.markdown(
                f"**📊 데이터 설정**\n"
                f"- 모드: `{mode_str}`\n"
                f"- Tissue: `{tissue_str}`\n"
                f"- Isoforms: `{n_iso:,}`\n"
                f"- GO terms: `{n_go}`\n"
            )
        with col3:
            st.markdown(
                f"**⚙️ 분석 파라미터**\n"
                f"- Score threshold: `{thr}`\n"
                f"- App version: `{APP_VERSION}`\n"
                f"- Generated: `{now}`\n"
            )

        _cite = (
            f"PRISM {PRISM_VERSION} ({PRISM_ENSEMBLE}), "
            f"GO {GO_ONTOLOGY_VER}, "
            f"tissue={tissue_str}, threshold={thr}"
        )
        st.caption(
            f"📎 **논문/보고서 인용용**: {_cite} — "
            "Lee et al. (2026), *Nature Machine Intelligence* (in review)"
        )

        # Download context as text
        ctx_text = (
            f"# PRISM Analysis Context\n\n"
            f"Generated: {now}\n\n"
            f"## Model\n"
            f"- PRISM version: {PRISM_VERSION}\n"
            f"- Ensemble: {PRISM_ENSEMBLE}\n"
            f"- Training data: {PRISM_TRAINING}\n"
            f"- GO ontology: {GO_ONTOLOGY_VER}\n\n"
            f"## Data\n"
            f"- Mode: {mode_str}\n"
            f"- Tissue panel: {tissue_str}\n"
            f"- Isoforms loaded: {n_iso:,}\n"
            f"- GO terms: {n_go}\n\n"
            f"## Parameters\n"
            f"- Score threshold: {thr}\n"
            f"- App version: {APP_VERSION}\n\n"
            f"## Citation\n"
            f"Lee et al. (2026). PRISM+BISECT: Protein-Isoform Resolution via Intrinsic Sequence Modeling "
            f"for Long-Read Single-Cell Data. Nature Machine Intelligence (in review).\n"
        )
        st.download_button(
            "📥 Context 메타데이터 다운로드",
            data=ctx_text.encode(),
            file_name="prism_analysis_context.md",
            mime="text/markdown",
            key=f"ctx_dl_{tissue}_{thr}",
        )


def context_for_report(cfg: dict) -> str:
    """Return a Markdown metadata block to embed in case reports."""
    tissue = cfg.get('tissue', '—')
    thr    = cfg.get('score_threshold', 0.4)
    mode   = cfg.get('mode', 'demo')
    n_go   = len(cfg.get('go_terms', []))
    now    = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    return (
        f"\n\n---\n"
        f"**Analysis Context**\n"
        f"- PRISM model: `{PRISM_VERSION}` ({PRISM_ENSEMBLE})\n"
        f"- Training: {PRISM_TRAINING}\n"
        f"- GO ontology: `{GO_ONTOLOGY_VER}`\n"
        f"- Tissue panel: `{_TISSUE_LABEL.get(tissue, tissue)}`\n"
        f"- Mode: `{'Demo' if mode == 'demo' else 'Upload'}`\n"
        f"- GO terms: `{n_go}`\n"
        f"- Score threshold: `{thr}`\n"
        f"- Generated: `{now}`\n"
        f"- App version: `{APP_VERSION}`\n"
        f"\n*Lee et al. (2026), Nature Machine Intelligence (in review)*\n"
    )
