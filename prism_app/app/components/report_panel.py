"""Unified report panel widget — renders interpreter.py dicts as Streamlit UI.

Usage:
    from prism_app.app.components.report_panel import render_report_panel

    report = interpret_qc(cfg, classified_df)
    render_report_panel(report, section_name="QC & Overview",
                        download_filename="qc_report.md")
"""
from __future__ import annotations
import streamlit as st
from typing import Optional, Union


# ── Styling constants ─────────────────────────────────────────────────────────

_HEADER_STYLE = (
    "background:linear-gradient(90deg,#0f2942,#1e3a5f);"
    "border-radius:8px 8px 0 0;padding:10px 16px;"
    "font-size:0.85rem;font-weight:700;color:#93c5fd;"
    "letter-spacing:0.04em"
)
_BODY_STYLE = (
    "background:#f8fafc;border:1px solid #e2e8f0;"
    "border-top:none;border-radius:0 0 8px 8px;"
    "padding:14px 18px;margin-bottom:12px"
)
_BULLET_STYLE = (
    "background:#eff6ff;border-left:3px solid #3b82f6;"
    "border-radius:4px;padding:6px 12px;"
    "font-size:0.84rem;color:#1e3a5f;margin:3px 0"
)
_INTERP_STYLE = (
    "background:#f0fdf4;border-left:4px solid #22c55e;"
    "border-radius:6px;padding:10px 14px;"
    "font-size:0.87rem;color:#14532d;margin:8px 0"
)
_CAVEAT_STYLE = (
    "background:#fffbeb;border-left:3px solid #f59e0b;"
    "border-radius:4px;padding:6px 12px;"
    "font-size:0.82rem;color:#78350f;margin:3px 0"
)
_STEP_STYLE = (
    "background:#faf5ff;border-left:3px solid #8b5cf6;"
    "border-radius:4px;padding:6px 12px;"
    "font-size:0.82rem;color:#4c1d95;margin:3px 0"
)


# ── Main widget ───────────────────────────────────────────────────────────────

def render_report_panel(
    report: Union[dict, list[dict]],
    section_name: str = "분석 리포트",
    download_filename: str = "prism_report.md",
    expanded: bool = False,
    key: str = "",
) -> None:
    """Render a structured report dict (or list of dicts) as an expandable panel.

    Parameters
    ----------
    report : dict or list[dict]
        Output of any interpret_* function. List renders multiple sections.
    section_name : str
        Label shown in expander header.
    download_filename : str
        Filename for the Markdown download button.
    expanded : bool
        Whether expander starts open.
    key : str
        Unique suffix for Streamlit widget keys (needed when panel appears multiple times).
    """
    if report is None:
        return

    reports = report if isinstance(report, list) else [report]

    panel_key = f"report_panel_{section_name}_{key}"

    with st.expander(f"📋 분석 리포트 — {section_name}", expanded=expanded):
        for i, rep in enumerate(reports):
            if not rep:
                continue
            _render_single(rep, key=f"{panel_key}_{i}")

        # ── Download ──
        combined_md = "\n\n---\n\n".join(r.get('markdown', '') for r in reports if r)
        if combined_md:
            st.download_button(
                label="⬇️ 마크다운 리포트 다운로드",
                data=combined_md.encode('utf-8'),
                file_name=download_filename,
                mime="text/markdown",
                key=f"{panel_key}_dl",
                use_container_width=True,
            )


def _render_single(rep: dict, key: str = "") -> None:
    """Render one report dict."""
    headline      = rep.get('headline', '')
    bullets       = rep.get('bullets', [])
    interpretation = rep.get('interpretation', '')
    caveats       = rep.get('caveats', [])
    next_steps    = rep.get('next_steps', [])

    # ── Headline ──
    if headline:
        st.markdown(
            f"<div style='font-size:0.95rem;font-weight:700;color:#0f2942;"
            f"padding:6px 0 10px 0'>{headline}</div>",
            unsafe_allow_html=True,
        )

    # ── Two-column layout: observations + next steps ──
    col_obs, col_next = st.columns([3, 2])

    with col_obs:
        if bullets:
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px'>"
                "주요 관찰</div>",
                unsafe_allow_html=True,
            )
            bullets_html = "".join(
                f"<div style='{_BULLET_STYLE}'>• {b}</div>" for b in bullets
            )
            st.markdown(bullets_html, unsafe_allow_html=True)

    with col_next:
        if next_steps:
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px'>"
                "권장 다음 단계</div>",
                unsafe_allow_html=True,
            )
            steps_html = "".join(
                f"<div style='{_STEP_STYLE}'>▶ {s}</div>" for s in next_steps
            )
            st.markdown(steps_html, unsafe_allow_html=True)

    # ── Interpretation ──
    if interpretation:
        st.markdown(
            f"<div style='font-size:0.78rem;font-weight:700;color:#64748b;"
            f"text-transform:uppercase;letter-spacing:0.06em;"
            f"margin:10px 0 4px 0'>생물학적 해석</div>"
            f"<div style='{_INTERP_STYLE}'>{interpretation}</div>",
            unsafe_allow_html=True,
        )

    # ── Caveats ──
    if caveats:
        with st.expander("⚠️ 해석 주의사항", expanded=False):
            caveats_html = "".join(
                f"<div style='{_CAVEAT_STYLE}'>⚠️ {c}</div>" for c in caveats
            )
            st.markdown(caveats_html, unsafe_allow_html=True)


# ── Inline mini-report (no expander — for Quick Cards) ───────────────────────

def render_inline_report(
    report: dict,
    max_bullets: int = 3,
    show_next_steps: bool = True,
) -> None:
    """Compact inline version without expander — for Quick Card footers."""
    if not report:
        return

    headline   = report.get('headline', '')
    bullets    = report.get('bullets', [])[:max_bullets]
    interp     = report.get('interpretation', '')
    next_steps = report.get('next_steps', [])[:2]

    if headline:
        st.markdown(
            f"<div style='font-size:0.87rem;font-weight:700;color:#0f2942;"
            f"padding:4px 0 6px 0'>{headline}</div>",
            unsafe_allow_html=True,
        )

    if bullets:
        b_html = "".join(f"<div style='{_BULLET_STYLE}'>• {b}</div>" for b in bullets)
        st.markdown(b_html, unsafe_allow_html=True)

    if interp:
        st.markdown(
            f"<div style='{_INTERP_STYLE};margin-top:6px'>{interp}</div>",
            unsafe_allow_html=True,
        )

    if show_next_steps and next_steps:
        s_html = "".join(f"<div style='{_STEP_STYLE}'>▶ {s}</div>" for s in next_steps)
        st.markdown(s_html, unsafe_allow_html=True)
