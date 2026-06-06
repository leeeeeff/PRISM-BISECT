"""PRISM+BISECT — application entry point.

Run with:
    ./run_app.sh
    streamlit run prism_app/app/main.py
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st

st.set_page_config(
    page_title="PRISM · Isoform Function Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': (
            '**PRISM + BISECT** — Protein-Isoform Resolution via Intrinsic Sequence Modeling\n\n'
            'Interactive isoform-level functional annotation for long-read single-cell data.\n\n'
            'Lee et al. (2026) — *Nature Machine Intelligence* (in review)'
        ),
    },
)


def _precompute(cfg: dict) -> None:
    """Run all analyses up-front after data is applied.

    Shows st.status() in the main area so the user sees real-time progress.
    Results are stored in session_state so every page can access them
    without re-computing per-page.
    """
    from prism_app.core.classifier import classify_isoforms

    sm = cfg.get('score_matrix')
    if sm is None:
        return

    n_iso, n_go = sm.shape
    tissue_label = cfg.get('tissue', '데이터').replace('_', ' ')

    with st.status(f"📊 **{tissue_label}** 분석 데이터 준비 중…", expanded=True) as _sts:

        # Step 1: isoform classification
        st.write(f"🧬 아이소폼 분류 계산 중… ({n_iso:,}개 × {n_go} GO term)")
        try:
            classified = classify_isoforms(
                sm,
                cfg['isoform_ids'],
                gene_ids=cfg.get('gene_ids'),
                go_terms=cfg['go_terms'],
                dtu_df=cfg.get('dtu_df'),
                score_threshold=cfg['score_threshold'],
            )
            st.session_state['classified_df'] = classified
            n_s1 = int((classified['scenario'] == 1).sum()) if 'scenario' in classified.columns else 0
            st.write(f"  ✅ {len(classified):,}개 분류 완료 (S1 고신뢰: {n_s1:,}개)")
        except Exception as _e:
            st.write(f"  ⚠️ 분류 오류: {_e}")

        # Step 2: gene-level score summary (fast lookup for 06_gene.py)
        gene_ids = cfg.get('gene_ids')
        if gene_ids is not None:
            st.write("📊 유전자 스코어 요약 계산 중…")
            try:
                _g2idx: dict = {}
                for _i, (_iso, _gene) in enumerate(zip(cfg['isoform_ids'], gene_ids)):
                    _g2idx.setdefault(str(_gene), []).append(_i)
                _cache = {}
                for _gene, _idxs in _g2idx.items():
                    _rows = sm[_idxs, :]
                    _cache[_gene] = {
                        'max_score': float(_rows.max()),
                        'mean_score': float(_rows.mean()),
                        'n_isoforms': len(_idxs),
                        'isoform_list': [str(cfg['isoform_ids'][_i]) for _i in _idxs],
                    }
                st.session_state['gene_score_cache'] = _cache
                st.write(f"  ✅ {len(_cache):,}개 유전자 요약 완료")
            except Exception as _e:
                st.write(f"  ⚠️ 유전자 요약 오류: {_e}")

        # Step 3: UMAP embedding
        st.write("🗺️ UMAP 임베딩 계산 중…")
        try:
            from prism_app.visualization.umap_plot import compute_umap_coords
            from pathlib import Path as _Path
            _DEMO = _Path(__file__).parents[1] / 'data' / 'demo'
            _MAX_UMAP = 15_000
            _coords_p  = _DEMO / 'umap_coords.npy'
            _samp_p    = _DEMO / 'umap_sample_idx.npy'
            _tissue    = cfg.get('tissue', '')
            if _coords_p.exists() and _samp_p.exists() and _tissue == 'brain_672':
                import numpy as _np
                _coords   = _np.load(_coords_p)
                _samp_idx = _np.load(_samp_p)
                _method   = 'UMAP'
                st.write(f"  ✅ 사전 계산 UMAP 로드 완료 ({len(_samp_idx):,}개 샘플)")
            else:
                import numpy as _np
                _n_total = sm.shape[0]
                _rng = _np.random.default_rng(42)
                if _n_total > _MAX_UMAP:
                    _samp_idx = _np.sort(_rng.choice(_n_total, _MAX_UMAP, replace=False))
                else:
                    _samp_idx = _np.arange(_n_total)
                _coords, _method = compute_umap_coords(
                    sm[_samp_idx].astype(_np.float32), random_state=42
                )
                st.write(f"  ✅ UMAP 계산 완료 ({len(_samp_idx):,}개, 방법: {_method})")
            st.session_state['_precomp_umap'] = {
                'coords': _coords, 'sample_idx': _samp_idx, 'method': _method,
            }
        except Exception as _e:
            st.write(f"  ⚠️ UMAP 오류: {_e}")

        # Step 4: DTU functional consequence (condition analysis)
        # Skipped for large GO panels (≥100 terms) — O(n_dtu × n_go) makes it
        # prohibitively slow at Apply time; @st.cache_data in 04_condition.py
        # handles it lazily on first page visit instead.
        dtu_df = cfg.get('dtu_df')
        _n_go  = len(cfg.get('go_terms') or [])
        if dtu_df is not None and _n_go < 100:
            st.write("🔄 조건 분석 계산 중 (DTU × PRISM)…")
            try:
                from prism_app.pipeline.dtu_connector import compute_functional_consequence
                import numpy as _np
                _pval_thr  = 0.05
                _delta_thr = 0.1
                _score_thr = cfg.get('score_threshold', 0.4)
                _conseq = compute_functional_consequence(
                    dtu_df,
                    sm.astype(_np.float32),
                    _np.asarray(cfg['isoform_ids'], dtype=str),
                    cfg['go_terms'],
                    cfg['go_names'],
                    score_threshold=_score_thr,
                    dtu_pval_threshold=_pval_thr,
                    delta_if_threshold=_delta_thr,
                )
                st.session_state['_precomp_conseq_df'] = {
                    'df': _conseq,
                    'pval_thr': _pval_thr,
                    'delta_thr': _delta_thr,
                    'score_thr': _score_thr,
                }
                st.write(f"  ✅ 조건 분석 완료 ({len(_conseq):,}개 이벤트)")
            except Exception as _e:
                st.write(f"  ⚠️ 조건 분석 오류: {_e}")
        elif dtu_df is not None:
            st.write(f"🔄 조건 분석: GO {_n_go}개 대규모 패널 — 첫 방문 시 자동 계산됩니다")

        # Step 5: fingerprint so threshold changes re-trigger classification
        st.session_state['_clf_fingerprint'] = (
            f"{cfg.get('tissue', '')}_{cfg.get('score_threshold', 0.4)}"
        )

        _sts.update(label="✅ 분석 준비 완료 — 페이지를 선택하세요",
                    state="complete", expanded=False)


from prism_app.app.components.sidebar import render_sidebar

if 'cfg' not in st.session_state:
    st.session_state.cfg = None

# ── Hero page autostart: ?mode=demo&autostart=1 ───────────────────────────────
# When the user clicks "Explore demo data" on the hero landing page (port 8500),
# they land here with these query params. We pre-load the default dataset so the
# Apply button is bypassed and pre-computation starts immediately.
_qp = st.query_params
if _qp.get('autostart') == '1' and not st.session_state.get('_autostart_done'):
    st.session_state['_autostart_done'] = True
    _at_mode = _qp.get('mode', 'demo')
    if _at_mode == 'demo' and not st.session_state.get('_applied_data'):
        from prism_app.app.components.sidebar import _load_demo_data
        from prism_app.core.go_utils import TISSUE_PRESETS
        _at_tissue = 'brain_41'
        _at_go     = list(TISSUE_PRESETS.get(_at_tissue, TISSUE_PRESETS['muscle']).keys())
        _at_raw    = _load_demo_data(_at_tissue, _at_go)
        if _at_raw.get('score_matrix') is not None:
            st.session_state['_applied_data'] = {
                **_at_raw, 'tissue': _at_tissue, 'mode': 'demo',
            }
            st.session_state['_data_just_loaded'] = True
            st.session_state.pop('classified_df', None)
            st.session_state.pop('_clf_fingerprint', None)
    st.query_params.clear()

cfg = render_sidebar()
st.session_state.cfg = cfg

# Auto-apply brain_41 when user enters from hero splash ("Explore demo data")
if (st.session_state.get('_hero_mode') == 'demo'
        and not st.session_state.get('_applied_data')):
    from prism_app.app.components.sidebar import _load_demo_data
    from prism_app.core.go_utils import TISSUE_PRESETS
    _h_tissue = 'brain_41'
    _h_go     = list(TISSUE_PRESETS.get(_h_tissue, TISSUE_PRESETS['muscle']).keys())
    _h_raw    = _load_demo_data(_h_tissue, _h_go)
    if _h_raw.get('score_matrix') is not None:
        st.session_state['_applied_data'] = {
            **_h_raw, 'tissue': _h_tissue, 'mode': 'demo',
        }
        st.session_state['_data_just_loaded'] = True
        st.session_state.pop('classified_df',    None)
        st.session_state.pop('_clf_fingerprint', None)
        cfg.update({k: v for k, v in _h_raw.items() if k in cfg})
        cfg['tissue'] = _h_tissue
        st.session_state.cfg = cfg
    st.session_state.pop('_hero_mode', None)

# When Apply button was just clicked (or autostart triggered):
# run pre-computations with progress display, then rerun normally.
if st.session_state.pop('_data_just_loaded', False):
    _precompute(cfg)
    st.rerun()

pg = st.navigation(
    {
        "": [
            st.Page("pages/main_home.py", title="PRISM+BISECT",  icon="🧬", default=True),
            st.Page("pages/00_hub.py",    title="Analysis Hub",  icon="🏠"),
        ],
        "데이터셋 분석": [
            st.Page("pages/01_qc.py",        title="QC & Overview",       icon="📊"),
            st.Page("pages/02_landscape.py",  title="Module Landscape",    icon="🗺️"),
            st.Page("pages/03_patterns.py",   title="Functional Patterns", icon="🔬"),
            st.Page("pages/04_condition.py",  title="Condition Analysis",  icon="🔄"),
            st.Page("pages/06_advanced.py",   title="Advanced",            icon="⚙️"),
        ],
        "타겟 분석": [
            st.Page("pages/05_target_hub.py", title="타겟 탐색",       icon="🎯"),
            st.Page("pages/05_targets.py",    title="시나리오 & 분석",  icon="📋"),
            st.Page("pages/07_bisect.py",     title="BISECT Cases",    icon="🧫"),
        ],
        "개별 분석": [
            st.Page("pages/06_gene.py",     title="유전자 분석",   icon="🧬"),
            st.Page("pages/06_isoform.py",  title="아이소폼 분석", icon="🔬"),
        ],
        "다중 분석": [
            st.Page("pages/08_multi_scenario.py", title="Scenario 비교",  icon="📊"),
            st.Page("pages/09_multi_module.py",   title="기능 모듈 비교", icon="🗂️"),
            st.Page("pages/10_multi_go.py",        title="GO term 비교",   icon="🧬"),
            st.Page("pages/11_multi_dtu.py",       title="DTU 조건 비교",  icon="🔄"),
        ],
    },
    position="sidebar",
)
pg.run()
