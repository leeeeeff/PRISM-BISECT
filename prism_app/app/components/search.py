"""search.py — Reusable search-with-autocomplete widget for gene/isoform IDs.

Usage (main area):
    query = gene_search_autocomplete('my_key', gene_pool)

Usage (sidebar):
    query = gene_search_autocomplete('my_key', gene_pool, container=st.sidebar)

Implementation note:
    Suggestion buttons use on_click callbacks, not post-render session_state writes,
    to comply with Streamlit's "cannot modify widget key after instantiation" constraint.
    The callback fires before the next script run, so the text_input picks up the value.
"""
from __future__ import annotations
import streamlit as st


def _rank_matches(query: str, pool: list, max_results: int = 10) -> list:
    q = query.strip().lower()
    if not q:
        return []
    seen: set = set()
    result: list = []
    for tier in [
        [x for x in pool if x.lower() == q],
        [x for x in pool if x.lower().startswith(q) and x.lower() != q],
        [x for x in pool if q in x.lower() and not x.lower().startswith(q)],
    ]:
        for x in tier:
            if x not in seen:
                seen.add(x)
                result.append(x)
                if len(result) >= max_results:
                    return result
    return result


def _unique_pool(pool) -> list:
    if pool is None or len(pool) == 0:
        return []
    return list(dict.fromkeys(str(g) for g in pool if g and str(g).strip()))


def _make_fill_callback(target_key: str, value: str):
    """Return an on_click callback that writes value into target_key before next run."""
    def _cb():
        st.session_state[target_key] = value
    return _cb


def _render_suggestions(key: str, query: str, pool: list,
                        max_suggestions: int, container) -> None:
    """Shared suggestion-button renderer used by both autocomplete functions."""
    _c = container if container is not None else st
    _pool = _unique_pool(pool)
    if not (query and len(query.strip()) >= 1 and _pool):
        return
    _matches = _rank_matches(query, _pool, max_results=max_suggestions)
    if not _matches:
        return
    _c.caption(f"🔍 연관 검색 {len(_matches)}건 — 클릭으로 자동완성:")
    _n = min(len(_matches), 4)
    _cols = _c.columns(_n)
    for _i, _m in enumerate(_matches):
        with _cols[_i % _n]:
            # on_click fires BEFORE the next script run, so the text_input
            # with `key` reads the updated value on re-render — no error.
            st.button(
                _m,
                key=f"{key}_sug_{_i}",
                use_container_width=True,
                help=_m,
                on_click=_make_fill_callback(key, _m),
            )


def gene_search_autocomplete(
    key: str,
    gene_pool,
    label: str = "유전자 검색",
    placeholder: str = "예: NDUFS4",
    max_suggestions: int = 8,
    label_visibility: str = 'collapsed',
    container=None,
    pre_fill: str = '',
) -> str:
    """Text input with ranked gene-name suggestion buttons.

    Returns the current text-input value (str).
    """
    _c = container if container is not None else st

    if pre_fill and key not in st.session_state:
        st.session_state[key] = pre_fill

    query = _c.text_input(
        label, key=key, placeholder=placeholder,
        label_visibility=label_visibility,
    )
    _render_suggestions(key, query, gene_pool, max_suggestions, _c)
    return st.session_state.get(key, query)


def isoform_search_autocomplete(
    key: str,
    isoform_pool,
    label: str = "아이소폼 검색",
    placeholder: str = "예: NDUFS4-201",
    max_suggestions: int = 8,
    label_visibility: str = 'collapsed',
    container=None,
    pre_fill: str = '',
) -> str:
    """Text input with ranked isoform-ID suggestion buttons."""
    _c = container if container is not None else st

    if pre_fill and key not in st.session_state:
        st.session_state[key] = pre_fill

    query = _c.text_input(
        label, key=key, placeholder=placeholder,
        label_visibility=label_visibility,
    )
    _render_suggestions(key, query, isoform_pool, max_suggestions, _c)
    return st.session_state.get(key, query)
