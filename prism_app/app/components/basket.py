"""basket.py — Centralised basket state management for PRISM+BISECT.

basket_genes    : List[dict]  — gene-level items
basket_isoforms : List[dict]  — isoform-level items

Both lists accept legacy str entries so an in-session migration is transparent.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


# ── Initialisation ────────────────────────────────────────────────────────────

def init_basket() -> None:
    """Ensure basket and analysis_cases keys exist in session_state."""
    if 'basket_genes' not in st.session_state:
        st.session_state['basket_genes'] = []
    if 'basket_isoforms' not in st.session_state:
        st.session_state['basket_isoforms'] = []
    if 'analysis_cases' not in st.session_state:
        st.session_state['analysis_cases'] = []
    if 'sb_pending_items' not in st.session_state:
        st.session_state['sb_pending_items'] = []


# ── Read helpers (backward-compatible) ───────────────────────────────────────

def basket_gene_ids() -> list[str]:
    """Return plain list of gene ID strings (handles legacy str items)."""
    return [
        item['id'] if isinstance(item, dict) else item
        for item in st.session_state.get('basket_genes', [])
    ]


def basket_isoform_ids() -> list[str]:
    """Return plain list of isoform ID strings (handles legacy str items)."""
    return [
        item['id'] if isinstance(item, dict) else item
        for item in st.session_state.get('basket_isoforms', [])
    ]


def get_basket_genes() -> list[dict]:
    """Return basket_genes as a list of dicts (normalises legacy str entries)."""
    out = []
    for item in st.session_state.get('basket_genes', []):
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append(_make_gene_entry(item, source_page='unknown'))
    return out


def get_basket_isoforms() -> list[dict]:
    """Return basket_isoforms as a list of dicts (normalises legacy str entries)."""
    out = []
    for item in st.session_state.get('basket_isoforms', []):
        if isinstance(item, dict):
            out.append(item)
        else:
            gene = item.rsplit('-', 1)[0] if '-' in item else item
            out.append(_make_isoform_entry(item, gene=gene, source_page='unknown'))
    return out


# ── Write helpers ─────────────────────────────────────────────────────────────

def add_to_gene_basket(
    gene_id: str,
    source_page: str = 'unknown',
    tag_scenario: object = None,
    tag_module: object = None,
    tag_go: object = None,
    tag_condition: object = None,
) -> bool:
    """Add a gene to basket_genes. Returns True if added, False if duplicate."""
    init_basket()
    existing = [x.upper() for x in basket_gene_ids()]
    if gene_id.strip().upper() in existing:
        return False
    entry = _make_gene_entry(
        gene_id.strip(), source_page,
        tag_scenario=tag_scenario,
        tag_module=tag_module,
        tag_go=tag_go,
        tag_condition=tag_condition,
    )
    basket = st.session_state['basket_genes']
    basket.append(entry)
    st.session_state['basket_genes'] = basket
    return True


def add_to_isoform_basket(
    isoform_id: str,
    gene: str = '',
    source_page: str = 'unknown',
    tag_scenario: object = None,
    tag_module: object = None,
    tag_go: object = None,
    tag_condition: object = None,
) -> bool:
    """Add an isoform to basket_isoforms. Returns True if added, False if duplicate."""
    init_basket()
    existing = [x.upper() for x in basket_isoform_ids()]
    if isoform_id.strip().upper() in existing:
        return False
    if not gene:
        gene = isoform_id.rsplit('-', 1)[0] if '-' in isoform_id else isoform_id
    entry = _make_isoform_entry(
        isoform_id.strip(), gene=gene, source_page=source_page,
        tag_scenario=tag_scenario,
        tag_module=tag_module,
        tag_go=tag_go,
        tag_condition=tag_condition,
    )
    basket = st.session_state['basket_isoforms']
    basket.append(entry)
    st.session_state['basket_isoforms'] = basket
    return True


def remove_from_basket(item_id: str, kind: str = 'gene') -> None:
    """Remove an item by ID. kind='gene' or 'isoform'."""
    key = 'basket_genes' if kind == 'gene' else 'basket_isoforms'
    basket = st.session_state.get(key, [])
    item_id_up = item_id.upper()
    new_basket = [
        x for x in basket
        if (x['id'] if isinstance(x, dict) else x).upper() != item_id_up
    ]
    st.session_state[key] = new_basket


def update_note(item_id: str, kind: str, note: str) -> None:
    """Update the note field of a basket item (Hub-only feature)."""
    key = 'basket_genes' if kind == 'gene' else 'basket_isoforms'
    basket = st.session_state.get(key, [])
    item_id_up = item_id.upper()
    for item in basket:
        if isinstance(item, dict):
            if item.get('id', '').upper() == item_id_up:
                item['note'] = note
        else:
            pass
    st.session_state[key] = basket


def clear_basket(kind: object = None) -> None:
    """Clear basket. kind=None clears both, 'gene' or 'isoform' clears one."""
    if kind is None or kind == 'gene':
        st.session_state['basket_genes'] = []
    if kind is None or kind == 'isoform':
        st.session_state['basket_isoforms'] = []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_gene_entry(
    gene_id: str,
    source_page: str,
    tag_scenario: object = None,
    tag_module: object = None,
    tag_go: object = None,
    tag_condition: object = None,
) -> dict[str, Any]:
    return {
        'id': gene_id,
        'source_page': source_page,
        'tag_scenario': tag_scenario,
        'tag_module': tag_module,
        'tag_go': tag_go,
        'tag_condition': tag_condition,
        'note': '',
    }


def _make_isoform_entry(
    isoform_id: str,
    gene: str,
    source_page: str,
    tag_scenario: object = None,
    tag_module: object = None,
    tag_go: object = None,
    tag_condition: object = None,
) -> dict[str, Any]:
    return {
        'id': isoform_id,
        'gene': gene,
        'source_page': source_page,
        'tag_scenario': tag_scenario,
        'tag_module': tag_module,
        'tag_go': tag_go,
        'tag_condition': tag_condition,
        'note': '',
    }


# ── Analysis cases ────────────────────────────────────────────────────────────

def get_analysis_cases() -> list:
    return list(st.session_state.get('analysis_cases', []))


def save_analysis_case(item_type: str, mode: str, axis: object,
                       items: list, note: str = '') -> None:
    from datetime import datetime
    _cases = st.session_state.setdefault('analysis_cases', [])
    _existing = [c.get('case_id', 0) for c in _cases]
    _new_id = (max(_existing) + 1) if _existing else 1
    _cases.append({
        'case_id': _new_id,
        'item_type': item_type,
        'mode': mode,
        'axis': axis,
        'items': list(items),
        'note': note,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    })
    st.session_state['sb_pending_items'] = []


def delete_analysis_case(case_id: int) -> None:
    _cases = st.session_state.get('analysis_cases', [])
    st.session_state['analysis_cases'] = [c for c in _cases if c.get('case_id') != case_id]


def update_case_note(case_id: int, note: str) -> None:
    for c in st.session_state.get('analysis_cases', []):
        if c.get('case_id') == case_id:
            c['note'] = note


# ── Tag badge renderer (shared UI helper) ────────────────────────────────────

_SCENARIO_COLORS = {1: '#ef4444', 2: '#f97316', 3: '#22c55e', 4: '#94a3b8'}
_SOURCE_LABELS = {
    'landscape': 'LandScp', 'target_hub': 'TgtHub', 'targets': 'Targets',
    'hub': 'Hub', 'isoform': 'IsoForm', 'bisect': 'BISECT', 'unknown': '?',
}

def tag_badges_html(item: dict) -> str:
    """Return HTML string of compact tag badges for an item dict."""
    if not isinstance(item, dict):
        return ''
    parts = []
    if item.get('tag_scenario') is not None:
        s = item['tag_scenario']
        c = _SCENARIO_COLORS.get(s, '#94a3b8')
        parts.append(f"<span style='background:{c};color:white;border-radius:3px;"
                     f"padding:1px 5px;font-size:0.7rem'>S{s}</span>")
    if item.get('tag_module') is not None:
        parts.append(f"<span style='background:#3b82f6;color:white;border-radius:3px;"
                     f"padding:1px 5px;font-size:0.7rem'>M{item['tag_module']}</span>")
    if item.get('tag_condition'):
        cond = item['tag_condition'][:6]
        parts.append(f"<span style='background:#8b5cf6;color:white;border-radius:3px;"
                     f"padding:1px 5px;font-size:0.7rem'>{cond}</span>")
    if item.get('source_page') and item['source_page'] != 'unknown':
        lbl = _SOURCE_LABELS.get(item['source_page'], item['source_page'][:7])
        parts.append(f"<span style='background:#64748b;color:white;border-radius:3px;"
                     f"padding:1px 5px;font-size:0.7rem'>{lbl}</span>")
    return ' '.join(parts)
