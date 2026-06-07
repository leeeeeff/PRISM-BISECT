"""Page 07 — BISECT Cases standalone browser.
Extracted from 05_targets.py (BISECT Cases tab). Works with or without session state.
"""
import sys
import json
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from prism_app.app.components.interpretation import render_data_context_banner
from prism_app.app.components.basket import add_to_gene_basket, basket_gene_ids

# ── Session state ─────────────────────────────────────────────────────────────
cfg      = st.session_state.get('cfg') or {}
sm       = cfg.get('score_matrix')
ids      = cfg.get('isoform_ids')
genes    = cfg.get('gene_ids')
go       = cfg.get('go_terms', [])
gnames   = cfg.get('go_names', {})
thr      = cfg.get('score_threshold', 0.4)
dtu_df   = cfg.get('dtu_df')
classified = st.session_state.get('classified_df')

def _go_profile_fig(
    go_ids: list, go_names: dict, scores, title: str, thr: float, top_n: int = 15
):
    """Return (fig, full_df) — bar limited to top_n GO terms by score.

    Always shows top_n highest-scoring terms; remaining are in full_df for
    the expander table. This prevents x-axis explosion when n_go >= 100.
    """
    import numpy as _np2
    _scores_arr = _np2.asarray(scores, dtype=float)
    full_df = pd.DataFrame({
        'GO': [go_names.get(g, g)[:40] for g in go_ids],
        'Score': _scores_arr,
        'GO_ID': go_ids,
    }).sort_values('Score', ascending=False).reset_index(drop=True)

    n_total = len(full_df)
    cap = min(top_n, n_total)
    plot_df = full_df.head(cap)

    fig = px.bar(
        plot_df, x='GO', y='Score',
        color='Score', color_continuous_scale='RdYlGn', range_color=[0, 1],
        title=f"{title}  (상위 {cap} / {n_total}개 GO term)",
        height=300,
    )
    fig.update_layout(
        xaxis_tickangle=-40, showlegend=False, plot_bgcolor='white',
        margin=dict(t=50, b=80, l=10, r=10),
        coloraxis_showscale=False,
    )
    fig.add_hline(y=float(thr), line_dash='dash', line_color='grey',
                  annotation_text=f"threshold ({thr})")
    return fig, full_df


_REGULATOR_KB: dict = {
    'STAT1':   ('TF',         True,  'AD 신경염증 핵심 전사인자; 미세아교·흥분성 뉴런에서 억제됨 (Baranzini 2020)'),
    'REST':    ('TF',         True,  '신경보호 전사억제인자; AD에서 발현 감소 → 시냅스 유전자 억제 해제 (Lu 2014 Cell)'),
    'CREB1':   ('TF',         True,  '신경 생존·LTP 전사인자; AD에서 인산화 감소 → 기억 형성 장애 (Saura 2004)'),
    'SP1':     ('TF',         True,  'Tau·APP 프로모터에 직접 결합; AD 취약성 인자 (Citron 2008)'),
    'SP3':     ('TF',         True,  'SP1 길항 전사인자; AD에서 SP1 대비 과발현 → 프로모터 경쟁 (Black 2001)'),
    'SRSF5':   ('ASF',        True,  'Serine/Arginine Splicing Factor 5; AD 관련 스플라이싱 재편 (Raj 2018)'),
    'SRSF7':   ('ASF',        True,  'tau exon 10 포함 조절; FTLD-Tau 관련 (Jiang 1998)'),
    'RBFOX1':  ('ASF',        True,  '뇌 특이적 ASF; 신경 발달·AD 취약 exon 조절 (Bhatt 2020)'),
    'HDAC2':   ('Epigenetic', True,  'AD에서 히스톤 H3K27 탈아세틸화 과활성 → 신경 유전자 억제 (Gräff 2012)'),
    'SIRT1':   ('Epigenetic', True,  'AD에서 NAD+-의존 탈아세틸화 감소 → p53·NF-κB 과활성 (Kim 2007)'),
    'KLF9':    ('TF',         False, '새로 발견; 억제성 전사인자 후보, 산화 스트레스 반응 조절'),
    'YBX1':    ('RBP',        False, 'Y-box RNA 결합 단백질; 스플라이싱·번역 조절, AD 역할 미확립'),
    'HNRNPK':  ('ASF',        False, 'hnRNP K; pre-mRNA 스플라이싱·수송 조절, AD 연관 신규 발견'),
    'E2F3':    ('TF',         False, '세포주기·아포프토시스 전사인자; AD 신경세포 재진입 관련 가능성'),
    'SETDB2':  ('Epigenetic', False, 'H3K9me3 methyltransferase; 이형성질 억제 → 비정상 유전자 발현'),
}

_MECHANISM_KO: dict = {
    'alternative_promoter':   ('대체 프로모터', '#7c3aed',
                                '다른 프로모터 활성화로 전사 시작 위치가 이동. '
                                'N-말단 구조가 달라져 신호 펩타이드·막 결합 도메인 변화 가능.'),
    'alternative_splicing':   ('선택적 스플라이싱', '#0ea5e9',
                                'exon inclusion/exclusion으로 도메인 구성이 직접 변화. '
                                'ASF(SRSF, RBFOX 등)의 결합 부위 변화가 주요 원인.'),
    'transcriptional':        ('전사 조절 변화', '#d97706',
                                '동일 프로모터에서 TF 결합 변화로 전사량이 조절됨. '
                                'TF 활성 변화가 아이소폼 비율 변화의 직접 원인.'),
    'epigenetic_derepression': ('후성유전학적 탈억제', '#dc2626',
                                'HDAC 과활성 또는 DNA 메틸화 변화로 억제되어 있던 엑손이 개방됨. '
                                '염색질 접근성 변화가 스플라이싱 패턴을 재편함.'),
    'intron_retention':       ('인트론 유지', '#059669',
                                '스플라이싱 효율 저하로 인트론이 성숙 mRNA에 잔존. '
                                'NMD 위험 증가; 단백질 번역 여부 검증 필요.'),
}


def _build_prism_tier_badge(brow: dict) -> str:
    """Return an HTML badge for the PRISM functional tier classification."""
    tier = str(brow.get('prism_tier') or '').strip()
    _TIER_STYLE = {
        'tier1_functional_switch': ('#4f46e5', '🔬 PRISM Tier 1: 기능 스위치'),
        'tier2_functional_loss':   ('#dc2626', '📉 PRISM Tier 2: 기능 소실'),
        'tier2_complex_loss':      ('#7f1d1d', '⚡ PRISM Tier 2: Complex I 붕괴'),
        'tier2_partial_change':    ('#d97706', '↔ PRISM Tier 2: 부분 변화'),
        'tier2_gain_no_direction': ('#d97706', '↑ PRISM Tier 2: 기능 획득'),
        'tier3_gene_median':       ('#94a3b8', '〜 Tier 3: 대표서열 추정'),
        'tier3_structural_only':   ('#94a3b8', '△ Tier 3: 구조 증거'),
        'tier3_no_match':          ('#94a3b8', '? Tier 3: 미매칭'),
    }
    color, label = _TIER_STYLE.get(tier, ('#94a3b8', f'Tier: {tier or "N/A"}'))
    return (f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:10px;font-size:0.75rem;font-weight:600;"
            f"margin-right:6px'>{label}</span>")


def _build_module_grid_html(brow: dict) -> str:
    """Build a 15-module evidence grid HTML from BISECT case fields."""
    delta    = brow.get('delta')
    dtu_p    = brow.get('dtu_p')
    dg       = str(brow.get('domains_gained')      or '').strip()
    dl       = str(brow.get('domains_lost')        or '').strip()
    af_d     = brow.get('af_delta_plddt')
    af_gain  = str(brow.get('af_gained_confident') or '').strip()
    af_lost  = str(brow.get('af_lost_confident')   or '').strip()
    ppi_v    = str(brow.get('ppi_verdict')         or '').strip()
    cons_c   = str(brow.get('cons_ad_class')       or '').strip()
    phylo    = brow.get('cons_ad_phylop')
    regs     = brow.get('top_regulators', '')
    tss      = str(brow.get('tss_class')           or '').strip()
    apa      = str(brow.get('apa_class')            or '').strip()
    nmd_rel  = brow.get('nmd_relevant')
    ad_nmd   = brow.get('ad_nmd')
    ct_nmd   = brow.get('ct_nmd')
    nat      = brow.get('nat')
    l1       = brow.get('young_l1_cds')
    seq_id   = brow.get('seq_val_identity')
    seq_con  = str(brow.get('seq_val_conclusion')  or '').strip()
    m_ct     = str(brow.get('prism_match_ct')      or '').strip()
    m_ad     = str(brow.get('prism_match_ad')      or '').strip()

    def _is_reg(raw):
        from ast import literal_eval
        if not raw or str(raw) in ('None', ''):
            return False
        for p in str(raw).split(';'):
            p = p.strip()
            if p:
                return True
        return False

    modules = [
        # (label, badge_text, color, detail)
        ("DTU (Stage 1)",
         "PASS" if (delta and abs(float(delta)) > 0.1 and dtu_p and float(dtu_p) < 0.05)
         else ("단일조건" if (dtu_p is None and brow.get('dtu_note') == 'single_condition_no_comparison') else ("Δ<0.1" if delta else "—")),
         "#15803d" if (delta and abs(float(delta)) > 0.1 and dtu_p and float(dtu_p) < 0.05)
         else ("#78716c" if (dtu_p is None and brow.get('dtu_note') == 'single_condition_no_comparison') else "#94a3b8"),
         f"Δ={float(delta):+.3f}, p={float(dtu_p):.1e}" if (delta and dtu_p) else
         ("단일조건 데이터 — AD/CT 비교군 없음" if brow.get('dtu_note') == 'single_condition_no_comparison' else "—")),

        ("M1 AlphaFold",
         "PASS" if (af_gain or af_lost or (af_d and abs(float(af_d)) > 5))
         else ("ΔpLDDT" if af_d else "—"),
         "#15803d" if (af_gain or af_lost or (af_d and abs(float(af_d)) > 5)) else "#94a3b8",
         f"ΔpLDDT={float(af_d):+.1f}" if af_d else ("conf.region changed" if (af_gain or af_lost) else "—")),

        ("M2 Domain",
         "PASS" if (dg or dl) else "—",
         "#15803d" if (dg or dl) else "#94a3b8",
         (f"+{dg}" if dg else "") + (" / " if dg and dl else "") + (f"-{dl}" if dl else "") or "—"),

        ("M3 PPI",
         ppi_v if ppi_v else "—",
         "#15803d" if ppi_v == 'SUPPORTED' else ("#b91c1c" if ppi_v == 'UNSUPPORTED' else "#94a3b8"),
         f"top: {brow.get('ppi_top_partner','?')} ({int(brow.get('ppi_top_score',0))})" if ppi_v == 'SUPPORTED' else "—"),

        ("M4 Conservation",
         cons_c.replace('_', ' ') if cons_c else "—",
         "#15803d" if ('conserved' in cons_c.lower()) else "#94a3b8",
         f"phyloP={float(phylo):.3f}" if phylo else "—"),

        ("M5 Regulators",
         "PASS" if _is_reg(regs) else "—",
         "#15803d" if _is_reg(regs) else "#94a3b8",
         ", ".join(p.strip().split("'gene': '")[1].split("'")[0]
                   for p in str(regs).split(';') if "'gene':" in p)[:40] if _is_reg(regs) else "—"),

        ("M6 TSS",
         tss.replace('_', ' ') if tss else "—",
         "#15803d" if (tss and tss not in ('same_promoter', '')) else "#94a3b8",
         (f"{int(float(brow.get('tss_diff_bp',0))):+d}bp" if brow.get('tss_diff_bp') else "") or "—"),

        ("M7 APA",
         apa.replace('_', ' ') if apa else "—",
         "#15803d" if (apa and apa not in ('same_apa', '')) else "#94a3b8",
         (f"{int(float(brow.get('tts_diff_bp',0))):+d}bp" if brow.get('tts_diff_bp') else "") or "—"),

        ("M8 NMD",
         "relevant" if nmd_rel else ("AD✓" if ad_nmd else "—"),
         "#d97706" if (nmd_rel or ad_nmd) else "#94a3b8",
         ("NMD target" if nmd_rel else "") + (" AD" if ad_nmd else "") + (" CT" if ct_nmd else "") or "—"),

        ("M9 NAT",
         "overlap" if nat else "—",
         "#d97706" if nat else "#94a3b8",
         "antisense overlap" if nat else "—"),

        ("M10 L1-CDS",
         "present" if l1 else "—",
         "#d97706" if l1 else "#94a3b8",
         "retrotransposon CDS" if l1 else "—"),

        ("M11 Seq Val",
         f"{float(seq_id):.0%}" if seq_id and float(seq_id) > 0 else "—",
         "#15803d" if (seq_id and float(seq_id) > 0.9) else ("#d97706" if seq_id else "#94a3b8"),
         seq_con[:30] if seq_con else "—"),

        ("M12 PRISM",
         f"CT:{m_ct or '?'} / AD:{m_ad or '?'}",
         "#15803d" if ('exact' in (m_ct + m_ad)) else "#f59e0b",
         "exact=이소폼 특이적 예측" if 'exact' in (m_ct + m_ad) else "gene_median=유전자 중앙값"),
    ]

    n_pass = sum(1 for _, badge, color, _ in modules if color == '#15803d')

    cells = ''
    for label, badge, color, detail in modules:
        cells += (
            f"<td style='padding:4px 6px;vertical-align:top;border-bottom:1px solid #f1f5f9'>"
            f"<div style='font-size:0.72rem;color:#64748b;margin-bottom:2px'>{label}</div>"
            f"<div style='display:inline-block;background:{color};color:white;"
            f"border-radius:3px;padding:1px 6px;font-size:0.73rem;font-weight:600'>{badge}</div>"
            f"<div style='font-size:0.68rem;color:#94a3b8;margin-top:2px;word-break:break-word'>{detail}</div>"
            f"</td>"
        )

    # 4 columns
    rows_html = ''
    mlist = list(enumerate(modules))
    for row_i in range(0, len(mlist), 4):
        row_cells = [cells[j] for j in range(
            sum(len(cells[k]) for k in range(row_i)),
            sum(len(cells[k]) for k in range(min(row_i+4, len(modules))))
        )]
    # Simpler: just use flat cells in a table
    table_cells = ''.join(
        f"<td style='padding:5px 8px;vertical-align:top;border-bottom:1px solid #f1f5f9;"
        f"width:25%;min-width:120px'>"
        f"<div style='font-size:0.73rem;color:#64748b;margin-bottom:2px'>{label}</div>"
        f"<div style='display:inline-block;background:{color};color:white;"
        f"border-radius:3px;padding:1px 7px;font-size:0.74rem;font-weight:600'>{badge}</div>"
        f"<div style='font-size:0.68rem;color:#94a3b8;margin-top:1px'>{detail}</div>"
        f"</td>"
        + ("<tr>" if (i + 1) % 4 == 0 and i + 1 < len(modules) else "")
        for i, (label, badge, color, detail) in enumerate(modules)
    )

    return (
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:12px 14px;margin:10px 0'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:#374151;margin-bottom:8px;"
        f"display:flex;justify-content:space-between;align-items:center'>"
        f"<span>🔬 BISECT 모듈 증거 요약</span>"
        f"<span style='background:#1e3a8a;color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.74rem'>{n_pass}/13 PASS</span></div>"
        f"<table style='width:100%;border-collapse:collapse'><tr>"
        + table_cells
        + f"</tr></table>"
        f"<div style='font-size:0.68rem;color:#9ca3af;margin-top:6px'>"
        f"🟢 PASS | 🟡 주의 요함 | ⚫ 해당 없음</div>"
        f"</div>"
    )


def _parse_regulators(raw: str) -> list:
    """Parse BISECT top_regulators string → list of dicts."""
    import ast
    result = []
    if not raw or str(raw) in ('None', ''):
        return result
    for p in str(raw).split(';'):
        p = p.strip()
        if not p:
            continue
        try:
            result.append(ast.literal_eval(p))
        except Exception:
            pass
    return result


@st.cache_data(show_spinner=False)
def _load_case_sig_regs(gene: str, cell_type: str) -> list:
    """Load significant_regulators from case analysis.json (up to 14 per case)."""
    import json as _json
    _base = Path(__file__).parents[3] / 'Final_analysis' / 'pipeline_bioanalysis' / 'outputs'
    _aj = _base / f"{gene}_{cell_type}" / 'analysis.json'
    if not _aj.exists():
        return []
    try:
        with open(_aj) as _f:
            _d = _json.load(_f)
        _m8 = _d.get('m8_regulatory_context', {}) or {}
        return _m8.get('significant_regulators', []) or []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _load_all_sig_regulators(bisect_path: str) -> list:
    """Load all significant_regulators from every BISECT case analysis.json."""
    import json as _json
    _base = Path(__file__).parents[3] / 'Final_analysis' / 'pipeline_bioanalysis' / 'outputs'
    try:
        with open(bisect_path) as _f:
            _cases = _json.load(_f)
    except Exception:
        return []
    _rows = []
    for _c in _cases:
        _g  = _c.get('gene', '')
        _ct = _c.get('cell_type', '')
        _aj = _base / f"{_g}_{_ct}" / 'analysis.json'
        if not _aj.exists():
            continue
        try:
            with open(_aj) as _f:
                _d = _json.load(_f)
            _m8 = _d.get('m8_regulatory_context', {}) or {}
            for _r in (_m8.get('significant_regulators', []) or []):
                _rows.append({
                    'Gene':         _r.get('gene', ''),
                    'logFC':        float(_r.get('logFC', 0)),
                    '-log10(padj)': float(_r.get('neg_log10_padj', 0)),
                    'Direction':    _r.get('direction', '').capitalize(),
                    'Case':         _g,
                    'CellType':     _ct,
                })
        except Exception:
            pass
    return _rows


@st.cache_data(show_spinner=False)
def _load_bisect_splice_diversity() -> dict:
    """Return {GENE_UPPER: float(max_pairwise_splice_dist)} from splicing_delta_v2."""
    _SD   = Path(__file__).parents[3] / 'hMuscle/results_isoform/features/splicing/splicing_delta_v2.npy'
    _GENE = Path(__file__).parents[3] / 'hMuscle/model/my_gene_list_fixed.npy'
    _SYM  = (Path(__file__).parents[3]
             / 'hMuscle/data/raw_data/data/id_lists/ensembl_to_symbol.txt')
    if not _SD.exists() or not _GENE.exists():
        return {}
    _sd    = np.load(_SD).astype(np.float32)
    _graw  = np.load(_GENE, allow_pickle=True)
    _genes = [x.decode() if isinstance(x, bytes) else str(x) for x in _graw]
    _smap: dict = {}
    if _SYM.exists():
        with open(_SYM) as _f:
            next(_f)
            for _ln in _f:
                _p = _ln.strip().split()
                if len(_p) >= 5:
                    _smap[_p[0]] = _p[4]
    _syms = [_smap.get(g.split('.')[0], g.split('.')[0]) for g in _genes]
    _g2i: dict = {}
    for _i, _g in enumerate(_syms):
        _g2i.setdefault(_g, []).append(_i)
    _out: dict = {}
    for _g, _idxs in _g2i.items():
        if len(_idxs) < 2:
            continue
        _d = _sd[_idxs]
        _mx = 0.0
        for _a in range(len(_idxs)):
            for _b in range(_a + 1, len(_idxs)):
                _dist = float(np.linalg.norm(_d[_a] - _d[_b]))
                if _dist > _mx:
                    _mx = _dist
        _out[_g.upper()] = round(_mx, 2)
    return _out


@st.cache_data(show_spinner=False)
def _load_bisect_celltypes() -> dict:
    """Return {GENE_UPPER: {'n_ct': int, 'n_ad': int, 'cell_types': list}} from 8 DIU CSVs."""
    _diu_dir = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU')
    _cell_names = [
        'Excitatory_neuron', 'Inhibitory_neuron', 'Astrocyte', 'Microglia',
        'Oligodendrocyte', 'OPC', 'Vascular_cell', 'Lymphocyte',
    ]
    _frames = []
    for _ct in _cell_names:
        _fp = _diu_dir / f'DIU_by_condition_{_ct}.csv'
        if _fp.exists():
            try:
                _df = pd.read_csv(_fp, usecols=['gene_name', 'chi_significant', 'usage_direction'])
                _df['cell_type'] = _ct.replace('_', ' ')
                _frames.append(_df)
            except Exception:
                pass
    if not _frames:
        return {}
    _all = pd.concat(_frames, ignore_index=True)
    _sig = _all[_all['chi_significant'] == True]
    _out: dict = {}
    for _gene, _grp in _sig.groupby('gene_name'):
        _n_ct  = _grp['cell_type'].nunique()
        _n_ad  = int(_grp['usage_direction'].str.contains('AD_enriched', na=False).sum())
        _cts   = sorted(_grp['cell_type'].unique().tolist())
        _out[str(_gene).upper()] = {'n_ct': _n_ct, 'n_ad': _n_ad, 'cell_types': _cts}
    return _out


@st.cache_data(show_spinner=False)
def _load_bisect_diu_full() -> 'pd.DataFrame | None':
    """Load 8 cell-type DIU CSVs with delta_usage for per-case heatmaps."""
    _diu_dir = Path('/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU')
    _cell_names = [
        'Excitatory_neuron', 'Inhibitory_neuron', 'Astrocyte', 'Microglia',
        'Oligodendrocyte', 'OPC', 'Vascular_cell', 'Lymphocyte',
    ]
    _frames = []
    for _ct in _cell_names:
        _fp = _diu_dir / f'DIU_by_condition_{_ct}.csv'
        if _fp.exists():
            try:
                _df = pd.read_csv(_fp)
                _df['cell_type'] = _ct.replace('_', ' ')
                _frames.append(_df)
            except Exception:
                pass
    return pd.concat(_frames, ignore_index=True) if _frames else None


_DOMAIN_FUNC_MAP = {
    'Kinesin':       'microtubule-based motor activity (ATP-dependent)',
    'WD40':          'β-propeller scaffold for protein–protein interactions',
    'PDZ':           'synaptic scaffolding, C-terminal peptide binding',
    'SAM':           'oligomerization / RNA-binding (context-dependent)',
    'SH3':           'proline-rich sequence binding, signaling assembly',
    'SH2':           'phosphotyrosine binding, downstream signaling',
    'RRM':           'RNA recognition motif, post-transcriptional regulation',
    'Microtub_bd':   'direct microtubule binding and stabilization',
    'NDUS4':         'NADH:ubiquinone oxidoreductase (Complex I) assembly',
    'RVT_1':         'reverse-transcriptase / RNA-dependent DNA polymerase',
    'DUF5082':       'domain of unknown function (DUF5082)',
    'ANAPC4_WD40':   'APC/C complex scaffold, cell-cycle regulation',
    'Nup160':        'nuclear pore complex, nucleocytoplasmic transport',
    'PH':            'phosphoinositide binding, membrane recruitment',
    'Guanylate_kin': 'guanylate kinase activity, scaffolding at PSD',
    'RhoGAP':        'Rho GTPase-activating protein, cytoskeleton regulation',
    'RhoGEF':        'Rho guanine-nucleotide exchange factor',
    'Pkinase':       'serine/threonine protein kinase, signal transduction',
    'CARD':          'caspase recruitment domain, apoptosis regulation',
    'FN3':           'fibronectin type-III fold, cell adhesion',
    'EGF':           'EGF receptor binding, proliferation signaling',
    'BEACH':         'lysosome/endosome biogenesis regulation',
    'GRAM':          'membrane association with PH domain',
}


def _build_bio_report_html(
    brow: dict,
    gene: str,
    ct_type: str,
    ct_tx: str,
    ad_tx: str,
    ct_scores,
    ad_scores,
    go_ids: list,
    go_names: dict,
    threshold: float,
) -> str:
    """Return styled HTML biological prediction report from BISECT evidence."""
    # ── Extract fields ────────────────────────────────────────────────────────
    delta   = brow.get('delta')
    dtu_p   = brow.get('dtu_p')
    dg      = str(brow.get('domains_gained') or '').strip()
    dl      = str(brow.get('domains_lost')   or '').strip()
    ppi_v   = str(brow.get('ppi_verdict')    or '').strip()
    ppi_p   = str(brow.get('ppi_top_partner')or '').strip()
    ppi_s   = brow.get('ppi_top_score')
    phylo   = brow.get('cons_ad_phylop')
    cons_c  = str(brow.get('cons_ad_class')  or '').strip()
    mech    = str(brow.get('mechanism_type') or '').strip()
    tss_cls = str(brow.get('tss_class')      or '').strip()
    apa_cls = str(brow.get('apa_class')      or '').strip()
    tss_bp  = brow.get('tss_diff_bp')
    apa_bp  = brow.get('tts_diff_bp')
    ad_nmd  = brow.get('ad_nmd')
    ct_nmd  = brow.get('ct_nmd')
    af_ad   = brow.get('af_ad_plddt_mean')
    af_ct   = brow.get('af_ct_plddt_mean')
    af_delta= brow.get('af_delta_plddt')

    # Parse regulators using shared helper
    all_regs  = _parse_regulators(str(brow.get('top_regulators') or ''))
    reg_name  = all_regs[0].get('gene', '') if all_regs else ''

    dg_list = [d for d in dg.split(';') if d]
    dl_list = [d for d in dl.split(';') if d]

    def _domain_func(d):
        for k, v in _DOMAIN_FUNC_MAP.items():
            if k.lower() in d.lower():
                return v
        return 'function uncharacterised'

    # ── PRISM match quality ───────────────────────────────────────────────────
    m_ct = str(brow.get('prism_match_ct') or '').strip()
    m_ad = str(brow.get('prism_match_ad') or '').strip()
    prism_exact = ('exact' in m_ct or 'exact' in m_ad or m_ad.startswith('proxy:'))

    # ── PRISM top GO terms — prefer sm scores, fall back to JSON pre-built ───
    def _top_go(scores, n=3):
        if scores is None:
            return []
        idxs = np.argsort(scores)[-n:][::-1]
        return [(go_ids[i], go_names.get(go_ids[i], go_ids[i]), float(scores[i]))
                for i in idxs if scores[i] > 0.15]

    def _top_go_from_json(lst, n=3):
        return [(d.get('go_id',''), d.get('go_name','')[:40], float(d.get('score',0)))
                for d in (lst or [])[:n] if float(d.get('score',0)) > 0.15]

    ct_top = _top_go(ct_scores) or _top_go_from_json(brow.get('prism_ct_top_go'))
    ad_top = _top_go(ad_scores) or _top_go_from_json(brow.get('prism_ad_top_go'))
    ct_go_ids_set = {g for g, _, _ in ct_top}
    ad_go_ids_set = {g for g, _, _ in ad_top}

    # For gene_median cases: both lists are identical → no meaningful GO shift
    _gene_median_both = (m_ct == 'gene_median' and m_ad == 'gene_median')

    # GAIN/LOSS: prefer JSON pre-built (already computed with delta); fallback to set-diff
    _gain_go_json = brow.get('prism_gain_go') or []
    _loss_go_json = brow.get('prism_loss_go') or []
    if _gain_go_json or _loss_go_json:
        gained_go = [(d.get('go_id',''), d.get('go_name','')[:40], float(d.get('ad_score',0)))
                     for d in _gain_go_json[:3]]
        lost_go   = [(d.get('go_id',''), d.get('go_name','')[:40], float(d.get('ct_score',0)))
                     for d in _loss_go_json[:3]]
    else:
        gained_go = [(g, n, s) for g, n, s in ad_top if g not in ct_go_ids_set] if not _gene_median_both else []
        lost_go   = [(g, n, s) for g, n, s in ct_top if g not in ad_go_ids_set] if not _gene_median_both else []

    # ── Confidence score (module-informed) ───────────────────────────────────
    _module_pass = sum([
        bool(delta and abs(float(delta)) > 0.1 and dtu_p and float(dtu_p) < 0.05),
        bool(str(brow.get('af_gained_confident') or '').strip() or
             str(brow.get('af_lost_confident') or '').strip() or
             (brow.get('af_delta_plddt') and abs(float(brow.get('af_delta_plddt'))) > 5)),
        bool(dg_list or dl_list),
        ppi_v == 'SUPPORTED',
        'conserved' in str(brow.get('cons_ad_class') or '').lower(),
        bool(all_regs),
        bool(mech and mech != 'transcriptional'),
        bool(str(brow.get('tss_class') or '') not in ('same_promoter', '', 'None')),
        bool(str(brow.get('apa_class') or '') not in ('same_apa', '', 'None')),
        bool(brow.get('nmd_relevant') or brow.get('ad_nmd')),
        bool(gained_go or lost_go) and prism_exact,
    ])
    ev_count = _module_pass
    conf_label = ['Low', 'Low', 'Moderate', 'Moderate', 'High', 'High', 'Very High',
                  'Very High', 'Very High', 'Very High', 'Very High'][min(ev_count, 10)]
    conf_color = {'Low': '#ef4444', 'Moderate': '#f59e0b',
                  'High': '#22c55e', 'Very High': '#15803d'}[conf_label]

    # ── Regulatory context ────────────────────────────────────────────────────
    known_regs  = [r for r in all_regs if _REGULATOR_KB.get(r['gene'], (None, None))[1] is True]
    novel_regs  = [r for r in all_regs if _REGULATOR_KB.get(r['gene'], (None, None))[1] is False]
    mech_info   = _MECHANISM_KO.get(mech, ('', '#64748b', ''))

    # ── Narrative sentences ───────────────────────────────────────────────────
    lines = []

    # 0. Causal origin (upstream mechanism)
    if mech and all_regs:
        mech_ko = mech_info[0] or mech
        top_reg = all_regs[0]
        top_reg_name = top_reg.get('gene', '')
        top_dir  = '활성 증가' if top_reg.get('direction') == 'up' else '억제'
        top_lfc  = top_reg.get('logFC', 0)
        kb_desc  = _REGULATOR_KB.get(top_reg_name, ('', '', ''))[2]
        lines.append(
            f"이 아이소폼 전환의 상류 원인으로 <b>{mech_ko}</b> 기전이 예측된다. "
            f"핵심 조절 인자 <b>{top_reg_name}</b> (logFC = {float(top_lfc):+.3f}, AD에서 {top_dir})"
            + (f" — {kb_desc}" if kb_desc else "") + "."
        )

    # 1. Isoform switch
    try:
        dv = float(delta)
    except Exception:
        dv = None
    if dv is not None:
        direction = '감소하며 대체됨' if dv < 0 else '증가함'
        lines.append(
            f"알츠하이머 조건 {ct_type} 세포에서 <b>{ct_tx or 'CT 이소폼'}</b>의 "
            f"사용 비율이 <b>Δ = {dv:+.3f}</b>로 {direction}하고 "
            f"<b>{ad_tx or 'AD 이소폼'}</b>으로 전환이 관측되었다"
            + (f" (DTU p = {float(dtu_p):.2e})" if dtu_p else
               " (단일조건 데이터 — AD/CT DTU 검정 미적용)" if brow.get('dtu_note') == 'single_condition_no_comparison'
               else "") + "."
        )

    # 2. Structural domain change
    if dg_list:
        gained_descs = '; '.join(f"<b>{d}</b> ({_domain_func(d)})" for d in dg_list)
        lines.append(f"AD 이소폼은 {gained_descs} 도메인을 새로 획득하여 기능적 다양성이 증가한다.")
    if dl_list:
        lost_descs = '; '.join(f"<b>{d}</b> ({_domain_func(d)})" for d in dl_list)
        lines.append(f"반면 {lost_descs} 도메인이 제거됨으로써 정상 이소폼의 주요 기능적 역량이 소실된다.")

    # 3. Structural stability (AlphaFold)
    if af_ad and af_ct:
        try:
            af_a = float(af_ad)
            af_c = float(af_ct)
            af_d = float(af_delta) if af_delta else af_a - af_c
            q_ad = "고신뢰 구조 (pLDDT ≥ 70)" if af_a >= 70 else "부분 무질서 구조 (pLDDT < 70)"
            q_ct = "고신뢰 구조" if af_c >= 70 else "무질서 포함"
            stab_interp = (
                "AD 이소폼이 CT 이소폼보다 더 안정된 구조를 형성한다" if af_d > 5
                else ("CT 이소폼이 구조적으로 더 안정적이며 AD 이소폼은 무질서 증가" if af_d < -5
                      else "두 이소폼의 구조적 안정성이 유사하다")
            )
            lines.append(
                f"AlphaFold 구조 예측: CT 이소폼 pLDDT = {af_c:.1f} ({q_ct}), "
                f"AD 이소폼 pLDDT = {af_a:.1f} ({q_ad}), ΔpLDDT = {af_d:+.1f}. "
                f"{stab_interp}."
            )
        except Exception:
            pass
    elif af_ad:
        try:
            af_val = float(af_ad)
            qual = "구조적으로 신뢰도 높은 (pLDDT ≥ 70)" if af_val >= 70 else "부분적으로 무질서한"
            lines.append(
                f"AlphaFold 구조 예측에서 AD 이소폼은 {qual} 단백질로 예측된다 (pLDDT = {af_val:.1f})."
            )
        except Exception:
            pass

    # 4. PRISM functional shift
    if gained_go:
        gfstr = ', '.join(f"{n[:35]} ({s:.3f})" for _, n, s in gained_go[:2])
        lines.append(
            f"PRISM GO 기능 예측에서 AD 이소폼은 정상 이소폼에는 없는 "
            f"<b>{gfstr}</b> 기능 공간을 새로 점유한다."
        )
    if lost_go:
        lfstr = ', '.join(f"{n[:35]} ({s:.3f})" for _, n, s in lost_go[:2])
        lines.append(
            f"정상 이소폼에서 높았던 <b>{lfstr}</b> 기능 점수가 AD 이소폼에서 유의미하게 낮아져, "
            f"질병 전환에 의한 기능 소실이 시사된다."
        )

    # 5. PPI
    if ppi_v == 'SUPPORTED' and ppi_p:
        ppi_score_str = f" (STRING score = {int(float(ppi_s))})" if ppi_s else ""
        lines.append(
            f"STRING PPI 분석에서 AD 이소폼은 <b>{ppi_p}</b>와의 상호작용이 예측되며"
            f"{ppi_score_str}, 이는 {ct_type} 내 새로운 단백질 복합체 형성 가능성을 시사한다."
        )

    # 6. Conservation
    if phylo:
        try:
            phv = float(phylo)
            cs = ("고보존 — 100-way vertebrate alignment에서 강한 purifying selection" if phv > 1.5
                  else ("중간 보존" if phv > 0.5 else "낮은 보존 — 최근 진화적 혁신 가능성"))
            lines.append(
                f"AD 특이적 엑손의 보존성 (phyloP100way = {phv:.3f}, {cs})은 "
                f"{'이 서열의 기능적 중요성을 강하게 지지한다' if phv > 1.5 else '추가적인 기능 검증이 필요함을 시사한다'}."
            )
        except Exception:
            pass

    # 7. Regulatory mechanism (upgraded with KB descriptions)
    if mech:
        mech_ko_n = mech_info[0] or mech
        mech_detail = mech_info[2]
        tss_note = f" TSS 차이: {int(float(tss_bp)):+d}bp" if tss_bp else ""
        apa_note = f" APA 차이: {int(float(apa_bp)):+d}bp" if apa_bp else ""
        reg_note = f" 핵심 조절 인자: <b>{reg_name}</b>" if reg_name else ""
        lines.append(
            f"전사체 생성 기전: <b>{mech_ko_n}</b>.{tss_note}{apa_note}{reg_note} "
            + (f"— {mech_detail}" if mech_detail else "")
        )

    # 8. TF/ASF regulatory interpretation
    if known_regs:
        k_str = '; '.join(
            f"<b>{r['gene']}</b> ({r['direction']}, logFC={float(r.get('logFC',0)):+.3f})"
            for r in known_regs[:3]
        )
        lines.append(
            f"기존 AD 연관 전사·스플라이싱 인자의 활성 변화: {k_str}. "
            "이 인자들의 발현 변화가 해당 유전자좌의 아이소폼 전환을 직접 유도했을 가능성이 높다."
        )
    if novel_regs:
        n_str = '; '.join(
            f"<b>{r['gene']}</b> ({r['direction']}, logFC={float(r.get('logFC',0)):+.3f})"
            for r in novel_regs
        )
        kb_descs = '; '.join(
            _REGULATOR_KB.get(r['gene'], ('', '', ''))[2]
            for r in novel_regs if _REGULATOR_KB.get(r['gene'], ('', '', ''))[2]
        )
        lines.append(
            f"새로 발견된 조절 인자 후보: {n_str}. "
            + (f"이 인자들의 AD 특이적 역할은 아직 확립되지 않았으나 ({kb_descs}), "
               "현 데이터에서 통계적으로 유의미한 발현 변화가 관측된다." if kb_descs else "")
        )

    # 9. NMD caveat
    if ad_nmd and str(ad_nmd).lower() not in ('false', ''):
        lines.append(
            "⚠️ AD 이소폼은 NMD (Nonsense-Mediated Decay) 감수성 구조를 포함하므로, "
            "단백질 번역 여부를 Ribo-seq 또는 질량분석으로 검증해야 한다."
        )

    # ── HTML assembly (inline styles only — no CSS classes) ──────────────────
    _TD_L = "style='padding:4px 10px;color:#6b7280;font-size:0.83rem;white-space:nowrap;vertical-align:top'"
    _TD_V = "style='padding:4px 10px;font-weight:700;font-size:0.83rem;vertical-align:top'"
    _TD_C = "style='padding:4px 10px;font-size:0.75rem;color:#9ca3af;vertical-align:top'"

    def _tag(text, bg, fg='#1e293b'):
        return (f"<code style='background:{bg};color:{fg};padding:2px 6px;"
                f"border-radius:3px;font-size:0.82rem'>{text}</code>")

    evid_rows_html = ''
    if delta:
        evid_rows_html += f"<tr><td {_TD_L}>Δ Usage (AD−CT)</td><td {_TD_V}>{float(delta):+.3f}</td><td {_TD_C}>DTU</td></tr>"
    if dtu_p:
        evid_rows_html += f"<tr><td {_TD_L}>DTU p-value</td><td {_TD_V}>{float(dtu_p):.2e}</td><td {_TD_C}>DTU</td></tr>"
    elif brow.get('dtu_note') == 'single_condition_no_comparison':
        evid_rows_html += f"<tr><td {_TD_L}>DTU p-value</td><td {_TD_V}><span style='color:#78716c'>N/A (단일조건)</span></td><td {_TD_C}>DTU</td></tr>"
    if dg_list:
        evid_rows_html += f"<tr><td {_TD_L}>도메인 획득</td><td {_TD_V}>{'&nbsp;·&nbsp;'.join(dg_list)}</td><td {_TD_C}>Structure</td></tr>"
    if dl_list:
        evid_rows_html += f"<tr><td {_TD_L}>도메인 손실</td><td {_TD_V}>{'&nbsp;·&nbsp;'.join(dl_list)}</td><td {_TD_C}>Structure</td></tr>"
    if ppi_v:
        _ppi_clr = '#15803d' if ppi_v == 'SUPPORTED' else '#b91c1c'
        evid_rows_html += f"<tr><td {_TD_L}>PPI support</td><td {_TD_V}><span style='color:{_ppi_clr}'>{ppi_v}</span></td><td {_TD_C}>Interaction</td></tr>"
    if phylo:
        evid_rows_html += f"<tr><td {_TD_L}>phyloP (AD exon)</td><td {_TD_V}>{float(phylo):.3f}&nbsp;<span style='color:#9ca3af;font-size:0.75rem'>({cons_c or '?'})</span></td><td {_TD_C}>Conservation</td></tr>"
    if mech:
        evid_rows_html += f"<tr><td {_TD_L}>기전</td><td {_TD_V}>{mech_info[0] or mech}</td><td {_TD_C}>Regulation</td></tr>"
    if all_regs:
        _reg_short = ', '.join(
            f"{r['gene']}({'↑' if r.get('direction')=='up' else '↓'})"
            for r in all_regs[:3]
        )
        evid_rows_html += f"<tr><td {_TD_L}>TF / ASF</td><td {_TD_V} style='font-size:0.78rem'>{_reg_short}</td><td {_TD_C}>Regulator</td></tr>"
    if not evid_rows_html:
        evid_rows_html = f"<tr><td {_TD_L} colspan='3'>증거 데이터 없음</td></tr>"

    def _go_badges(top_list, bg, border):
        if not top_list:
            return "<span style='color:#9ca3af;font-size:0.82rem'>데이터 없음</span>"
        return ''.join(
            f"<div style='background:{bg};border-left:3px solid {border};"
            f"border-radius:4px;padding:5px 8px;margin:3px 0;font-size:0.83rem'>"
            f"<b>{n[:36]}</b>&nbsp;&nbsp;"
            f"<span style='color:#64748b'>{s:.3f}</span></div>"
            for _, n, s in top_list[:3]
        )

    domain_gained_li = ''.join(
        f"<div style='margin:4px 0;font-size:0.83rem'>"
        f"{_tag(d, '#dcfce7', '#14532d')}"
        f"<span style='color:#374151;margin-left:6px'>{_domain_func(d)}</span></div>"
        for d in dg_list
    ) or "<div style='color:#9ca3af;font-size:0.83rem;padding:4px 0'>변화 없음</div>"

    domain_lost_li = ''.join(
        f"<div style='margin:4px 0;font-size:0.83rem'>"
        f"{_tag(d, '#fee2e2', '#7f1d1d')}"
        f"<span style='color:#374151;margin-left:6px'>{_domain_func(d)}</span></div>"
        for d in dl_list
    ) or "<div style='color:#9ca3af;font-size:0.83rem;padding:4px 0'>변화 없음</div>"

    interp_html = ''.join(
        f"<p style='margin:0 0 10px 0;font-size:0.86rem;line-height:1.7;color:#1e293b'>{l}</p>"
        for l in lines
    ) or "<p style='color:#9ca3af;font-size:0.86rem'>해석 데이터 불충분</p>"

    # ── Regulatory origin HTML block ──────────────────────────────────────────
    def _reg_badge(r):
        g = r.get('gene', '?')
        d = r.get('direction', '')
        lfc = float(r.get('logFC', 0))
        neg_p = float(r.get('neg_log10_padj', 0))
        kb = _REGULATOR_KB.get(g, ('TF', None, ''))
        cat   = kb[0] or 'TF'
        known = kb[1]
        bg    = '#fee2e2' if d == 'down' else '#dcfce7'
        border= '#ef4444' if d == 'down' else '#22c55e'
        arrow = '↓' if d == 'down' else '↑'
        star  = '' if known else ' 🟠'
        return (
            f"<div style='background:{bg};border-left:3px solid {border};"
            f"border-radius:4px;padding:5px 10px;margin:3px 0;font-size:0.82rem'>"
            f"<b>{g}</b>{star}&nbsp;"
            f"<span style='color:#64748b;font-size:0.75rem'>[{cat}]</span>&nbsp;"
            f"<span style='font-weight:700'>{arrow} {lfc:+.3f}</span>&nbsp;"
            f"<span style='color:#9ca3af;font-size:0.72rem'>-log10p={neg_p:.1f}</span>"
            f"</div>"
        )

    reg_badges_html = ''.join(_reg_badge(r) for r in all_regs[:5])
    if not reg_badges_html:
        reg_badges_html = "<div style='color:#9ca3af;font-size:0.82rem'>조절 인자 데이터 없음</div>"

    mech_ko_label = mech_info[0] or mech or '—'
    mech_clr      = mech_info[1]

    # Causal pathway arrow (upstream → downstream)
    _pathway_steps = []
    if mech:
        _pathway_steps.append(f"<b style='color:{mech_clr}'>{mech_ko_label}</b>")
    if all_regs:
        _regs_short = ', '.join(r['gene'] for r in all_regs[:3])
        _pathway_steps.append(f"TF/ASF 활성 변화 ({_regs_short})")
    if tss_cls and tss_cls not in ('same_promoter', ''):
        _pathway_steps.append(f"전사 시작 위치 이동 ({tss_cls})")
    if apa_cls and apa_cls not in ('same_apa', ''):
        _pathway_steps.append(f"3′ 처리 변화 ({apa_cls})")
    _pathway_steps.append("아이소폼 비율 전환 (DTU)")
    if dg_list or dl_list:
        _pathway_steps.append("도메인 구성 변화")
    if gained_go or lost_go:
        _pathway_steps.append("GO 기능 공간 재편")
    pathway_html = " &rarr; ".join(
        f"<span style='background:#f1f5f9;padding:2px 6px;border-radius:3px;"
        f"font-size:0.78rem'>{s}</span>"
        for s in _pathway_steps
    )

    reg_origin_html = (
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
        f"padding:14px 16px;margin-bottom:14px'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:10px'>🔭 아이소폼 전환 인과 경로</div>"
        # Pathway arrows
        f"<div style='margin-bottom:10px;line-height:2'>{pathway_html}</div>"
        # Two-column: regulators | mechanism details
        f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:10px'>"
        f"<div style='font-size:0.75rem;color:#374151;font-weight:600;margin-bottom:4px'>"
        f"TF / ASF 활성 변화 (AD vs CT)</div>"
        f"{reg_badges_html}"
        f"<div style='font-size:0.7rem;color:#9ca3af;margin-top:4px'>"
        f"🟠 = 새로 발견된 인자 · ↑/↓ = AD에서 증가/감소</div>"
        f"</td>"
        f"<td width='50%' style='vertical-align:top;padding-left:10px;"
        f"border-left:1px solid #d1fae5'>"
        f"<div style='font-size:0.75rem;color:#374151;font-weight:600;margin-bottom:4px'>"
        f"프로모터 · APA 컨텍스트</div>"
        + (
            f"<div style='font-size:0.82rem;margin:2px 0'>"
            f"TSS: <b>{tss_cls or '—'}</b>"
            + (f" ({int(float(tss_bp)):+d}bp)" if tss_bp else "") + "</div>"
            if tss_cls else ""
        )
        + (
            f"<div style='font-size:0.82rem;margin:2px 0'>"
            f"APA: <b>{apa_cls or '—'}</b>"
            + (f" ({int(float(apa_bp)):+d}bp)" if apa_bp else "") + "</div>"
            if apa_cls else ""
        )
        + (
            f"<div style='font-size:0.82rem;margin:6px 0 2px;color:#7c3aed'>"
            f"기전: <b>{mech_ko_label}</b></div>"
            f"<div style='font-size:0.75rem;color:#6b7280'>{mech_info[2]}</div>"
            if mech else ""
        )
        + f"</td></tr></table>"
        f"</div>"
    )

    return (
        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
        f"padding:20px 22px;margin:14px 0;font-family:Arial,sans-serif'>"

        # ── Header ──
        f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td style='vertical-align:middle'>"
        f"<span style='font-size:1.0rem;font-weight:700;color:#1e293b'>"
        f"📋 생물학적 기능 예측 리포트 — 통합 분석</span>"
        f"&nbsp;<span style='font-size:0.88rem;color:#0ea5e9;font-weight:700'>{gene}</span>"
        f"&nbsp;<span style='font-size:0.85rem;color:#64748b'>· {ct_type}</span>"
        f"</td>"
        f"<td style='text-align:right;vertical-align:middle;white-space:nowrap'>"
        + _build_prism_tier_badge(brow) +
        f"&nbsp;<span style='background:{conf_color};color:white;padding:4px 14px;"
        f"border-radius:12px;font-size:0.8rem;font-weight:700'>신뢰도: {conf_label}</span>"
        f"</td></tr></table>"

        # ── Regulatory origin (causal pathway) ──
        + reg_origin_html

        # ── Row 1: Evidence table | Domain changes ──
        + f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:12px'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:8px'>📊 증거 요약</div>"
        f"<table width='100%' cellspacing='0' style='border-collapse:collapse'>{evid_rows_html}</table>"
        f"</td>"
        f"<td width='50%' style='vertical-align:top;padding-left:12px;"
        f"border-left:1px solid #e2e8f0'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#374151;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:8px'>🔩 도메인·구조 기능 변화</div>"
        f"<div style='font-size:0.78rem;color:#15803d;font-weight:600;margin-bottom:4px'>▲ 획득 (AD 이소폼)</div>"
        f"{domain_gained_li}"
        f"<div style='font-size:0.78rem;color:#dc2626;font-weight:600;margin:10px 0 4px'>▼ 손실 (CT 이소폼)</div>"
        f"{domain_lost_li}"
        + (
            f"<div style='font-size:0.78rem;color:#7e22ce;margin-top:8px'>"
            f"ΔpLDDT = {float(af_delta):+.1f} "
            f"({'AD 더 안정' if float(af_delta)>0 else 'CT 더 안정'})</div>"
            if af_delta else ""
        )
        + f"</td></tr></table>"

        # ── Row 2: CT GO | AD GO ──
        + f"<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:14px'><tr>"
        f"<td width='50%' style='vertical-align:top;padding-right:8px'>"
        f"<div style='background:#eff6ff;border-radius:6px;padding:10px 12px'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:#1d4ed8;margin-bottom:6px'>"
        f"🔵 Control 이소폼 TOP GO"
        f"<span style='font-weight:400;color:#94a3b8;font-size:0.72rem;display:block'>{(ct_tx or '—')[:35]}</span>"
        f"</div>"
        f"{_go_badges(ct_top, '#dbeafe', '#3b82f6')}"
        f"</div></td>"
        f"<td width='50%' style='vertical-align:top;padding-left:8px'>"
        f"<div style='background:#fef2f2;border-radius:6px;padding:10px 12px'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:#dc2626;margin-bottom:6px'>"
        f"🔴 AD 이소폼 TOP GO"
        f"<span style='font-weight:400;color:#94a3b8;font-size:0.72rem;display:block'>{(ad_tx or '—')[:35]}</span>"
        f"</div>"
        f"{_go_badges(ad_top, '#fee2e2', '#ef4444')}"
        f"</div></td>"
        f"</tr></table>"

        # ── Narrative ──
        f"<div style='background:white;border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:16px 18px;margin-bottom:10px'>"
        f"<div style='font-size:0.85rem;font-weight:700;color:#1e293b;margin-bottom:12px;"
        f"padding-bottom:8px;border-bottom:2px solid #f1f5f9'>🧬 종합 해석 및 기능 예측</div>"
        f"{interp_html}"
        f"</div>"

        # ── Footer ──
        f"<div style='font-size:0.72rem;color:#9ca3af;text-align:right'>"
        f"PRISM+BISECT 자동 생성 · Lee et al. (2026) · 실험적 검증 필요</div>"
        f"</div>"
    )


st.title("🧫 BISECT Cases")
st.caption(
    "**BISECT** (Biological Isoform-Switch Evidence Characterization Tool) — "
    "15개 독립 모듈로 기능 스위치를 다층 검증한 84개 PASS 케이스. "
    "유전자명을 검색하면 Volcano · 도메인 구조 · GO 비교 · 종합 리포트가 펼쳐집니다."
)

import json
from pathlib import Path

_BISECT_PATH = Path(__file__).parents[3] / 'prism_app' / 'data' / 'demo' / 'bisect_cases.json'

st.subheader("🧫 BISECT PASS Cases — 84개 기능 스위치 검증 케이스")
st.markdown(
    """
    **BISECT** (Biological Isoform-Switch Evidence Characterization Tool)는 15개의 독립 분석 모듈을 통해
    각 유전자의 아이소폼 전환이 실제로 **생물학적 의미**를 가지는지 다층적으로 검증합니다.

    아래 표는 두 단계 검증(stage1: 통계 + stage2: 생물학 증거)을 모두 통과한 **84개 케이스**입니다.
    🟡 **노란색 행** = 도메인 구조 변화(단백질 기능 변화 직접 증거)가 확인된 고신뢰 케이스입니다.

    > 유전자 이름을 검색창에 입력하면 도메인 구조 그림, 규제 인자 분석, 생물학 리포트가 펼쳐집니다.
    """
)

if not _BISECT_PATH.exists():
    st.warning("bisect_cases.json not found in demo data directory.")
    st.stop()

with open(_BISECT_PATH) as _f:
    _bisect_raw = json.load(_f)

_bdf = pd.DataFrame(_bisect_raw)

# Runtime PRISM score enrichment from session state score matrix
if 'prism_ad_max_score' not in _bdf.columns and sm is not None and ids is not None and len(go) > 0:
    _ids_arr = np.asarray(ids, dtype=str)
    _id_to_idx = {iid: idx for idx, iid in enumerate(_ids_arr)}
    _gene_to_idxs = {}
    if genes is not None:
        for _idx, _g in enumerate(genes):
            _gene_to_idxs.setdefault(str(_g).upper(), []).append(_idx)
    _gene_median = {}
    for _g, _idxs in _gene_to_idxs.items():
        _gene_median[_g] = np.median(sm[_idxs], axis=0)

    _ct_max_scores = []
    _ad_max_scores = []
    _ct_max_gos = []
    _ad_max_gos = []
    _match_cts = []
    _match_ads = []

    for _, row in _bdf.iterrows():
        _gene = str(row.get('gene', '')).upper()
        _ct_tx = row.get('ct_transcript_id')
        _ad_tx = row.get('ad_transcript_id')

        # Control (CT)
        _ct_vec = None
        _ct_m = 'no_match'
        if _ct_tx in _id_to_idx:
            _ct_vec = sm[_id_to_idx[_ct_tx]]
            _ct_m = 'exact'
        elif _gene in _gene_median:
            _ct_vec = _gene_median[_gene]
            _ct_m = 'gene_median'

        # Disease (AD)
        _ad_vec = None
        _ad_m = 'no_match'
        if _ad_tx in _id_to_idx:
            _ad_vec = sm[_id_to_idx[_ad_tx]]
            _ad_m = 'exact'
        elif _gene in _gene_median:
            _ad_vec = _gene_median[_gene]
            _ad_m = 'gene_median'

        # Default fallback
        if _ct_vec is None:
            _ct_vec = np.zeros(len(go))
        if _ad_vec is None:
            _ad_vec = np.zeros(len(go))

        _ct_max_scores.append(float(_ct_vec.max()))
        _ad_max_scores.append(float(_ad_vec.max()))
        _ct_max_gos.append(gnames.get(go[int(_ct_vec.argmax())], go[int(_ct_vec.argmax())]))
        _ad_max_gos.append(gnames.get(go[int(_ad_vec.argmax())], go[int(_ad_vec.argmax())]))
        _match_cts.append(_ct_m)
        _match_ads.append(_ad_m)

    _bdf['prism_ct_max_score'] = _ct_max_scores
    _bdf['prism_ad_max_score'] = _ad_max_scores
    _bdf['prism_ct_max_go'] = _ct_max_gos
    _bdf['prism_ad_max_go'] = _ad_max_gos
    _bdf['prism_match_ct'] = _match_cts
    _bdf['prism_match_ad'] = _match_ads

# ── Runtime tier inference (prism_tier stored as null → infer from evidence) ─
_COMPLEX1_GENES = {'NDUFS4', 'NDUFS7', 'NDUFS8'}

def _infer_prism_tier(row: dict) -> str:
    """Assign prism_tier from structural evidence fields when JSON value is null."""
    gene  = str(row.get('gene', '') or '').upper()
    af_g  = str(row.get('af_gained_confident', '') or '').strip()
    af_l  = str(row.get('af_lost_confident',  '') or '').strip()
    dom_g = str(row.get('domains_gained',     '') or '').strip()
    dom_l = str(row.get('domains_lost',       '') or '').strip()
    if gene in _COMPLEX1_GENES:
        return 'tier2_complex_loss'
    if af_g:
        return 'tier1_functional_switch'
    if af_l:
        return 'tier2_functional_loss'
    if dom_g or dom_l:
        return 'tier2_partial_change'
    return 'tier3_structural_only'

if 'prism_tier' not in _bdf.columns or _bdf['prism_tier'].isna().all():
    _bdf['prism_tier'] = [_infer_prism_tier(r) for r in _bisect_raw]

# ── Cross-link: build S1 gene → isoform map ───────────────────────────────
_s1_genes = set()
_gene_to_rows = {}   # gene_id → list of classified rows (for PRISM chart)
if classified is not None:
    _s1 = classified[classified['scenario'] == 1]
    _s1_genes = set(_s1['gene_id'].dropna().tolist())
    for _g, _grp in _s1.groupby('gene_id'):
        _gene_to_rows[_g] = _grp

# ── Summary metrics ───────────────────────────────────────────────────────
_mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
_mc1.metric(
    "BISECT PASS 케이스",
    len(_bdf),
    help="15-모듈 파이프라인을 통과한 기능 스위치 후보 총수",
)
_mc2.metric(
    "세포 유형 수",
    _bdf['cell_type'].nunique(),
    help="분석된 고유 세포 유형 수 (Excitatory/Inhibitory 뉴런 등)",
)
_mc3.metric(
    "도메인 변화 케이스",
    int((_bdf['domains_gained'].fillna('') != '').sum()),
    help="Pfam + AlphaFold로 단백질 도메인 획득이 확인된 케이스 수 — 가장 직접적인 기능 변화 증거",
)
_mc4.metric(
    "NAT 중복 케이스",
    int(_bdf['nat'].fillna(False).sum()),
    help="Natural Antisense Transcript(NAT)와 게놈 위치가 겹치는 케이스 — 발현 조절 복잡성 추가 증거",
)
_mc5.metric(
    "S1 교차 유전자",
    len(_s1_genes & set(_bdf['gene'].dropna())),
    help="현재 업로드 데이터의 Scenario 1 목록과 BISECT PASS 케이스가 겹치는 유전자 수 (DTU 데이터 있을 때만 표시)",
)

st.divider()

# ── Global TF / ASF / Epigenetic violin plot ──────────────────────────────
with st.expander("📊 전체 케이스 — 조절 인자 활성 변화 분석 (Volcano + Violin)", expanded=False):
    st.markdown("""
**이 섹션은 BISECT 파이프라인이 84개 케이스에서 감지한 TF·ASF·후성유전 조절 인자들이
AD vs. CT 조건에서 어떻게 달라졌는지를 전체적으로 조망합니다.**

- **Volcano Plot** (위): X축 = 발현 변화 크기(logFC), Y축 = 통계적 유의성(-log₁₀ p-adj).
  오른쪽 위 = AD에서 유의미하게 증가한 인자 | 왼쪽 위 = 유의미하게 감소한 인자.
  점선 내부(|logFC| < 0.1 또는 p > 0.01) = 유의미하지 않은 변화.
- **Violin Plot** (아래): 같은 인자가 여러 케이스에서 어떤 logFC 분포를 보이는지 시각화.
  3개 이상의 케이스에서 반복 감지된 인자만 표시합니다.
- **● 원 = 기존 AD 문헌에 알려진 인자** | **◆ 다이아몬드 = 이 연구에서 새로 발견된 인자**
    """)

    # Build global regulator dataframe
    _glob_rows = []
    for _gc in _bisect_raw:
        _gregs = _parse_regulators(_gc.get('top_regulators', ''))
        for _r in _gregs:
            _gene_r = _r.get('gene', '')
            _kb = _REGULATOR_KB.get(_gene_r, (None, None, ''))
            _cat   = _kb[0] or 'TF'
            _known = _kb[1] if _kb[1] is not None else False
            _glob_rows.append({
                'Regulator':  _gene_r,
                'logFC':      float(_r.get('logFC', 0)),
                'Direction':  _r.get('direction', '').capitalize(),
                'Category':   _cat,
                'Knowledge':  '🔵 Known AD' if _known else '🟠 Novel',
                '-log10(padj)': float(_r.get('neg_log10_padj', 0)),
                'Case':       _gc.get('gene', ''),
                'CellType':   _gc.get('cell_type', ''),
            })
    if _glob_rows:
        _gdf = pd.DataFrame(_glob_rows)

        # ── Global Volcano (full sig_regulators from analysis.json) ──────────
        _all_sig_rows = _load_all_sig_regulators(str(_BISECT_PATH))
        _using_fallback_volcano = False
        if _all_sig_rows:
            _gvdf = pd.DataFrame(_all_sig_rows)
        else:
            # Fall back to using the top regulators from bisect_cases.json if outputs are missing
            _gvdf = _gdf.rename(columns={'Regulator': 'Gene'})
            _using_fallback_volcano = True

        if not _gvdf.empty:
            _gvdf = _gvdf.copy()
            _gvdf['Category'] = _gvdf['Gene'].map(
                lambda _g: _REGULATOR_KB.get(_g, ('TF', None, ''))[0] or 'TF'
            )
            _gvdf['Knowledge'] = _gvdf['Gene'].map(
                lambda _g: (
                    '🔵 Known AD' if _REGULATOR_KB.get(_g, (None, None))[1] is True
                    else ('🟠 Novel' if _REGULATOR_KB.get(_g, (None, None))[1] is False
                          else '⚪ Unknown')
                )
            )
            # Label known regulators with high significance
            _gvdf['Label'] = _gvdf.apply(
                lambda _row: (
                    _row['Gene']
                    if (_REGULATOR_KB.get(_row['Gene'], (None, None))[1] is True
                        and float(_row['-log10(padj)']) > 10)
                    else ''
                ), axis=1
            )
            _vol_title = 'Volcano Plot — TF/ASF Activity (AD vs CT) · 26 Cases · 전체 Significant Regulators'
            if _using_fallback_volcano:
                _vol_title = 'Volcano Plot — TF/ASF Activity (AD vs CT) · 84 Cases · Top Regulators (Fallback)'

            _fig_gvol = px.scatter(
                _gvdf,
                x='logFC', y='-log10(padj)',
                color='Direction',
                symbol='Knowledge',
                color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                symbol_map={
                    '🔵 Known AD': 'circle',
                    '🟠 Novel':    'diamond',
                    '⚪ Unknown':  'square',
                },
                text='Label',
                hover_data=['Gene', 'Category', 'Knowledge', 'Case', 'CellType'],
                title=_vol_title,
                labels={'logFC': 'logFC (AD vs CT)', '-log10(padj)': '-log₁₀(p-adj)'},
                height=430,
            )
            _fig_gvol.update_traces(
                textposition='top center',
                textfont=dict(size=9, color='#1e293b'),
                marker=dict(size=8, opacity=0.75),
            )
            _fig_gvol.add_vline(x=0.1,  line_dash='dash', line_color='#94a3b8', line_width=1)
            _fig_gvol.add_vline(x=-0.1, line_dash='dash', line_color='#94a3b8', line_width=1)
            _fig_gvol.add_hline(y=2.0,  line_dash='dash', line_color='#94a3b8', line_width=1)
            _fig_gvol.add_vline(x=0,    line_color='#374151', line_width=1.2)
            _fig_gvol.update_layout(
                plot_bgcolor='white',
                xaxis=dict(gridcolor='#f0f0f0'),
                yaxis=dict(gridcolor='#f0f0f0'),
                legend_title='',
                margin=dict(t=50, b=20, l=10, r=10),
                font=dict(size=11),
            )
            st.plotly_chart(_fig_gvol, use_container_width=True, key='glob_volcano')
            if _using_fallback_volcano:
                st.caption(
                    f"Volcano (Fallback): 84개 케이스의 top_regulators 데이터 기반. "
                    f"({len(_gvdf)}개 관측). X=logFC, Y=-log₁₀(p-adj). "
                    "점선: |logFC|=0.1, -log₁₀p=2. ● = 기존 AD 연관, ◆ = 새로 발견. "
                    "레이블 = -log₁₀p > 10인 Known 인자."
                )
            else:
                st.caption(
                    f"Volcano: 26개 케이스 analysis.json의 모든 significant_regulators "
                    f"({len(_all_sig_rows)}개 관측). X=logFC, Y=-log₁₀(p-adj). "
                    "점선: |logFC|=0.1, -log₁₀p=2. ● = 기존 AD 연관, ◆ = 새로 발견. "
                    "레이블 = -log₁₀p > 10인 Known 인자."
                )
            st.divider()

        # Show violin only for regulators appearing ≥3 times (else strip plot)
        _freq = _gdf['Regulator'].value_counts()
        _violin_regs = _freq[_freq >= 3].index.tolist()
        _strip_regs  = _freq[_freq < 3].index.tolist()

        if _violin_regs:
            _gdf_v = _gdf[_gdf['Regulator'].isin(_violin_regs)].copy()
            # Sort regulators: Known first, then by median logFC desc
            _reg_order = (
                _gdf_v.groupby('Regulator')['logFC'].median()
                .sort_values(ascending=False).index.tolist()
            )
            _fig_vio = px.violin(
                _gdf_v,
                x='Regulator', y='logFC',
                color='Direction',
                box=True, points='all',
                color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                category_orders={
                    'Regulator': _reg_order,
                    'Direction': ['Up', 'Down'],
                },
                hover_data=['Case', 'CellType', 'Category', 'Knowledge', '-log10(padj)'],
                title='TF / ASF / Epigenetic logFC Distribution (AD vs CT) — 84 BISECT Cases',
                labels={'logFC': 'logFC (AD vs CT)', 'Regulator': ''},
                height=380,
            )
            _fig_vio.add_hline(y=0, line_dash='dash', line_color='#374151', line_width=1.5)
            _fig_vio.update_layout(
                plot_bgcolor='white',
                yaxis=dict(gridcolor='#f0f0f0', zeroline=False),
                legend_title='Direction (AD vs CT)',
                margin=dict(t=45, b=60, l=10, r=10),
                font=dict(size=11),
            )
            st.plotly_chart(_fig_vio, use_container_width=True, key='glob_violin')
            st.caption(
                "바이올린 = logFC 분포 | 박스 = IQR | 점 = 개별 케이스. "
                "n≥3인 인자만 바이올린으로 표시. "
                "Red = AD에서 활성 증가, Blue = AD에서 억제."
            )

        # Known vs Novel summary bar
        _know_summ = _gdf.groupby(['Knowledge', 'Direction']).size().reset_index(name='N')
        if not _know_summ.empty:
            _fig_ks = px.bar(
                _know_summ,
                x='Knowledge', y='N', color='Direction',
                barmode='group',
                color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                title='Known vs Novel 조절인자 방향성',
                labels={'N': '케이스 수', 'Knowledge': ''},
                height=260,
            )
            _fig_ks.update_layout(
                plot_bgcolor='white', legend_title='',
                margin=dict(t=38, b=20, l=10, r=10),
            )
            st.plotly_chart(_fig_ks, use_container_width=True, key='glob_known_bar')

        # Regulator knowledge table
        _kb_df = (
            _gdf.groupby(['Regulator', 'Category', 'Knowledge'])
            .agg(N_cases=('Case', 'count'),
                 mean_logFC=('logFC', 'mean'),
                 max_neg_log10p=('-log10(padj)', 'max'))
            .reset_index()
            .sort_values('N_cases', ascending=False)
        )
        _kb_df['설명'] = _kb_df['Regulator'].map(
            lambda g: _REGULATOR_KB.get(g, ('', '', ''))[2]
        )
        st.markdown("**조절 인자 요약표**")
        st.dataframe(
            _kb_df.rename(columns={
                'Regulator': '인자', 'Category': '분류',
                'Knowledge': '기존 AD 연관',
                'N_cases': 'N케이스', 'mean_logFC': '평균 logFC',
                'max_neg_log10p': '최대 -log10p',
            }).round({'평균 logFC': 3, '최대 -log10p': 1}),
            use_container_width=True, hide_index=True,
        )

# ── PRISM Tier 요약 카운터 ────────────────────────────────────────────────
_tier_counts = _bdf['prism_tier'].value_counts() if 'prism_tier' in _bdf.columns else {}
_n_t1  = int(_tier_counts.get('tier1_functional_switch', 0))
_n_t2  = int(_tier_counts.get('tier2_functional_loss', 0) +
             _tier_counts.get('tier2_complex_loss', 0) +
             _tier_counts.get('tier2_partial_change', 0) +
             _tier_counts.get('tier2_gain_no_direction', 0))
_n_t3  = int(_tier_counts.get('tier3_gene_median', 0) +
             _tier_counts.get('tier3_structural_only', 0) +
             _tier_counts.get('tier3_no_match', 0))

_tc1, _tc2, _tc3, _tc4 = st.columns(4)
_tc1.metric("전체 PASS 케이스", len(_bdf))
_tc2.metric("🔬 Tier 1 — 기능 스위치", _n_t1,
            help="AD 아이소폼이 새 기능을 고신뢰도(≥0.40)로 획득, delta≥0.15")
_tc3.metric("📉 Tier 2 — 기능 소실/변화", _n_t2,
            help="CT 고신뢰 기능이 AD에서 감소, 또는 부분적 기능 변화")
_tc4.metric("△ Tier 3 — 구조 증거 기반", _n_t3,
            help="DTU 유의하나 PRISM 기능 예측 낮음, 또는 대표서열 추정")

# ── Filters ───────────────────────────────────────────────────────────────
_TIER_FILTER_OPTS = {
    '전체': None,
    '🔬 Tier 1 — 기능 스위치': ['tier1_functional_switch'],
    '📉 Tier 2 — 기능 소실':   ['tier2_functional_loss', 'tier2_complex_loss',
                                 'tier2_partial_change', 'tier2_gain_no_direction'],
    '△ Tier 3 — 구조 증거':   ['tier3_gene_median', 'tier3_structural_only', 'tier3_no_match'],
}
_ff1, _ff2, _ff3, _ff4 = st.columns([2, 2, 2, 1])
with _ff1:
    _tier_sel_label = st.selectbox(
        "PRISM Tier 필터", list(_TIER_FILTER_OPTS.keys()), key='bisect_tier_filter')
with _ff2:
    # Expand combined cell_type values (e.g. "A / B") into individual options
    _ct_all: set = set()
    for _ct_raw in _bdf['cell_type'].dropna().unique():
        for _part in str(_ct_raw).split('/'):
            _p = _part.strip()
            if _p:
                _ct_all.add(_p)
    _ct_opts = sorted(_ct_all)
    _ct_sel  = st.multiselect("Cell type 필터", _ct_opts, default=_ct_opts, key='bisect_ct')
with _ff3:
    _from_gene_page = st.session_state.pop('bisect_filter_gene', None)
    if _from_gene_page and not st.session_state.get('bisect_gene_q'):
        st.session_state['bisect_gene_q'] = _from_gene_page
    _bq = st.text_input("유전자 검색", placeholder="예: KIF21B, DLG1, NEK1", key='bisect_gene_q')
with _ff4:
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    _dtu_only = st.checkbox("DTU p 있는 케이스만", value=False, key='bisect_dtu_only',
                            help="단일조건(AD 전용) 케이스를 제외하고 AD/CT 비교군 DTU p-value가 있는 26개 뇌 케이스만 표시")

_ff5, _ff6 = st.columns([2, 8])
with _ff5:
    _MIN_CT_OPTS = {'전체': 0, '≥ 2CT': 2, '≥ 3CT': 3, '≥ 4CT': 4, '≥ 5CT': 5}
    _min_ct_label = st.select_slider(
        "DIU 재현 세포유형 수",
        options=list(_MIN_CT_OPTS.keys()),
        value='전체',
        key='bisect_min_ct',
        help="8개 뇌 세포유형 중 유의한 DTU(chi_significant=True)가 확인된 세포유형이 N개 이상인 유전자만 표시",
    )
_min_ct = _MIN_CT_OPTS[_min_ct_label]

_bdf_filt = _bdf.copy()
_tier_vals = _TIER_FILTER_OPTS[_tier_sel_label]
if _tier_vals and 'prism_tier' in _bdf_filt.columns:
    _bdf_filt = _bdf_filt[_bdf_filt['prism_tier'].isin(_tier_vals)]
if _ct_sel:
    # Match rows where any selected cell type appears in the (possibly combined) cell_type field
    _bdf_filt = _bdf_filt[
        _bdf_filt['cell_type'].apply(
            lambda _v: any(_s in str(_v) for _s in _ct_sel)
        )
    ]
if _bq:
    _bdf_filt = _bdf_filt[_bdf_filt['gene'].str.contains(_bq, case=False, na=False)]
if _dtu_only and 'dtu_p' in _bdf_filt.columns:
    _bdf_filt = _bdf_filt[_bdf_filt['dtu_p'].notna() & (_bdf_filt['dtu_note'] != 'single_condition_no_comparison')]
if _min_ct > 0 and _bisect_ct_lookup:
    _bdf_filt = _bdf_filt[_bdf_filt['gene'].apply(
        lambda _g: _bisect_ct_lookup.get(str(_g).upper(), {}).get('n_ct', 0) >= _min_ct
    )]

# ── Summary table ─────────────────────────────────────────────────────────
# Build display-friendly tier label column
_TIER_LABEL_MAP = {
    'tier1_functional_switch': '🔬 T1 스위치',
    'tier2_functional_loss':   '📉 T2 소실',
    'tier2_complex_loss':      '⚡ T2 ComplexI',
    'tier2_partial_change':    '↔ T2 변화',
    'tier2_gain_no_direction': '↑ T2 획득',
    'tier3_gene_median':       '〜 T3 추정',
    'tier3_structural_only':   '△ T3 구조',
    'tier3_no_match':          '? T3 미매칭',
}
if 'prism_tier' in _bdf_filt.columns:
    _bdf_filt = _bdf_filt.copy()
    _bdf_filt['_tier_label'] = _bdf_filt['prism_tier'].map(
        lambda x: _TIER_LABEL_MAP.get(str(x), str(x)))
    if 'prism_ad_max_score' in _bdf_filt.columns:
        _bdf_filt['_prism_score'] = _bdf_filt['prism_ad_max_score'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else '—')
    else:
        _bdf_filt['_prism_score'] = '—'

# ── Splice diversity column ───────────────────────────────────────────────
_bisect_sd_lookup = _load_bisect_splice_diversity()
if _bisect_sd_lookup:
    _bdf_filt = _bdf_filt.copy()
    _bdf_filt['_splice_div'] = _bdf_filt['gene'].apply(
        lambda _g: _bisect_sd_lookup.get(str(_g).upper())
    )

# ── Cell-type concordance column ──────────────────────────────────────────
_bisect_ct_lookup = _load_bisect_celltypes()
if _bisect_ct_lookup:
    _bdf_filt = _bdf_filt.copy()

    def _ct_conc_str(_g):
        _e = _bisect_ct_lookup.get(str(_g).upper())
        if not _e:
            return None
        _n = _e['n_ct']
        _a = _e['n_ad']
        return f"{_n}CT (AD:{_a})"

    _bdf_filt['_ct_conc'] = _bdf_filt['gene'].apply(_ct_conc_str)

_col_map = {
    'gene': 'Gene', 'cell_type': 'Cell Type',
    '_tier_label': 'PRISM Tier',
    '_prism_score': 'AD Score',
    'prism_ad_max_go': 'AD Top GO',
    'delta': 'Δ Usage', 'dtu_p': 'DTU p-val',
    '_splice_div': 'Splice Div',
    '_ct_conc': 'DIU 세포유형',
    'domains_gained': 'Domains Gained', 'domains_lost': 'Domains Lost',
    'ppi_verdict': 'PPI', 'af_ad_plddt_mean': 'pLDDT',
    'cons_ad_phylop': 'phyloP',
}
_show_cols_raw = ['gene', 'cell_type', '_tier_label', '_prism_score', 'prism_ad_max_go',
                  'delta', 'dtu_p', '_splice_div', '_ct_conc', 'domains_gained', 'domains_lost',
                  'ppi_verdict', 'af_ad_plddt_mean', 'cons_ad_phylop']
_show_cols = [c for c in _show_cols_raw if c in _bdf_filt.columns]
_bdf_show  = _bdf_filt[_show_cols].rename(columns=_col_map).copy()
# Truncate long GO names
if 'AD Top GO' in _bdf_show.columns:
    _bdf_show['AD Top GO'] = _bdf_show['AD Top GO'].apply(
        lambda x: str(x)[:28] if pd.notna(x) else '—')

def _highlight_bisect_row(row):
    tier = str(row.get('PRISM Tier', '') or '')
    if 'T1' in tier:
        return ['background-color: #ede9fe'] * len(row)   # purple tint
    if 'ComplexI' in tier:
        return ['background-color: #fce7f3'] * len(row)   # pink tint — mitochondrial complex loss
    if 'T2 소실' in tier or 'T2 변화' in tier:
        return ['background-color: #fee2e2'] * len(row)   # red tint
    _dg = str(row.get('Domains Gained', '') or '').strip()
    _dl = str(row.get('Domains Lost',   '') or '').strip()
    if _dg or _dl or row.get('Gene', '') in _s1_genes:
        return ['background-color: #fef9c3'] * len(row)   # yellow tint
    return [''] * len(row)

_caption_parts = [
    "🟣 보라 = Tier 1 고신뢰 기능 스위치",
    "🔴 빨강 = Tier 2 기능 소실",
    "🩷 분홍 = Tier 2 Complex I 삼각 수렴",
    "🟡 노랑 = 도메인 구조 변화 또는 Scenario 1",
]
st.caption(" | ".join(_caption_parts))
_fmt_dict = {
    'Δ Usage': '{:.3f}', 'pLDDT': '{:.1f}', 'phyloP': '{:.3f}',
}
if 'Splice Div' in _bdf_show.columns:
    _fmt_dict['Splice Div'] = lambda v: f'{v:.2f}' if v == v and v is not None else '—'
if 'DIU 세포유형' in _bdf_show.columns:
    _fmt_dict['DIU 세포유형'] = lambda v: v if (v and v == v) else '—'
st.dataframe(
    _bdf_show.style.apply(_highlight_bisect_row, axis=1).format(
        _fmt_dict, na_rep='단일조건',
    ).format(
        {'DTU p-val': lambda v: f'{v:.2e}' if v == v and v is not None else '단일조건'},
    ),
    use_container_width=True, hide_index=True,
)

with st.expander("📋 표 컬럼 설명 — 약어가 낯설다면 펼쳐보세요", expanded=False):
    st.markdown("""
| 컬럼 | 의미 | 해석 기준 |
|------|------|-----------|
| **PRISM Tier** | 41GO 확장 모델 기반 기능 변화 분류 | T1=고신뢰 기능 스위치, T2=기능 소실/변화/ComplexI, T3=구조 증거 |
| **AD Score** | AD 아이소폼의 최대 GO term PRISM 예측 점수 | 41GO 모델 (macro AUPRC 0.672), ≥0.40 = 고신뢰 |
| **AD Top GO** | AD 아이소폼에서 가장 높은 점수의 GO 생물 과정 | 해당 아이소폼의 주요 예측 기능 |
| **Δ Usage** | AD − CT 조건 간 아이소폼 사용 비율 차이 | ±0.1 이상이면 의미 있는 전환 |
| **DTU p-val** | 아이소폼 비율 차이 통계 검정 p-value | < 0.05 (또는 < 1e-5) 이면 유의미 |
| **Splice Div** | 유전자 내 아이소폼 간 splice_delta 최대 pairwise L2 거리 | >1.0 = 주요 exon 변이, 0.1–1.0 = 부분(A5SS/A3SS), <0.1 = 미세/UTR |
| **DIU 세포유형** | 8개 뇌 세포유형 중 유의한 DTU 이벤트가 확인된 세포유형 수 (AD-enriched 건수) | 예: `3CT (AD:2)` = 3개 세포유형 유의 · AD-enriched 2건. 세포유형 재현성 지표 |
| **Domains Gained** | AD 아이소폼에서 새로 생긴 Pfam 도메인 | 도메인 획득 = 기능 추가 직접 증거 |
| **Domains Lost** | CT 아이소폼에서 제거된 Pfam 도메인 | 도메인 손실 = 정상 기능 소실 |
| **PPI** | STRING PPI 데이터베이스 기반 상호작용 지원 여부 | SUPPORTED = 새로운 단백질 파트너 예측 |
| **pLDDT** | AlphaFold 구조 예측 신뢰도 점수 (0~100) | ≥70 = 신뢰 가능한 구조, <50 = 무질서 영역 |
| **phyloP** | 100종 척추동물 서열 보존도 (phyloP100way) | >1.5 = 강한 진화적 선택압 (기능적으로 중요) |
    """)

# ── Per-case expanders — only rendered when a gene search is active ──────
# Rendering all 84 expanders at once is expensive (domain maps, DTU charts,
# IGV iframes). Gate on search query so the table always stays fast.
if not _bdf_filt.empty:
    st.divider()
    if not _bq:
        st.info(
            f"위 표에서 **{len(_bdf_filt)}건**의 PASS 케이스를 확인할 수 있습니다. "
            "**유전자 이름을 위 검색창에 입력**하면 아래 섹션들이 펼쳐집니다:\n\n"
            "▸ Volcano Plot (어떤 TF·ASF가 얼마나 바뀌었는지)  "
            "▸ 도메인 구조 변화 그림  "
            "▸ DTU Δ Usage 막대 차트  "
            "▸ GO 기능 비교 (CT vs AD 이소폼)  "
            "▸ 종합 생물학 해석 리포트\n\n"
            "추천 검색어: `KIF21B` `DLG1` `NDUFS4` `DMD`",
            icon="🔍",
        )
    else:
        st.markdown(
            f"**케이스 상세 분석** — '{_bq}' 검색 결과 **{len(_bdf_filt)}건** "
            "| 아래 각 케이스를 클릭해 상세 분석을 확인하세요."
        )

# ── DTU lookup dict (cached) — built once, O(1) per gene lookup ──────────
@st.cache_data(show_spinner=False)
def _build_dtu_lookup(dtu_bytes: bytes) -> dict:
    import io
    _d = pd.read_csv(io.BytesIO(dtu_bytes), sep='\t')
    _lk: dict = {}
    # Normalise gene column name (gene_id / gene / geneID / gene_name)
    _gene_col = next(
        (c for c in _d.columns
         if c.lower() in ('gene_id', 'gene', 'geneid', 'gene_name', 'gene_symbol')),
        None,
    )
    if _gene_col is None:
        return _lk
    if _gene_col != 'gene_id':
        _d = _d.rename(columns={_gene_col: 'gene_id'})
    # Normalise delta_IF column name
    _dif_col = next(
        (c for c in _d.columns
         if c.lower() in ('delta_if', 'dif', 'deltif', 'delta_usage', 'delta')),
        None,
    )
    if _dif_col and _dif_col != 'delta_IF':
        _d = _d.rename(columns={_dif_col: 'delta_IF'})
    if 'condition' in _d.columns:
        for (_g, _c), _grp in _d.groupby(['gene_id', 'condition']):
            _lk[(_g, _c)] = _grp.reset_index(drop=True)
    else:
        for _g, _grp in _d.groupby('gene_id'):
            _lk[(_g, '')] = _grp.reset_index(drop=True)
    return _lk

_dtu_src = cfg.get('dtu_df')
_dtu_lookup: dict = {}
if _dtu_src is not None:
    _dtu_lookup = _build_dtu_lookup(_dtu_src.to_csv(index=False).encode())

if _bq and not _bdf_filt.empty:
    for _, _brow in _bdf_filt.iterrows():
        _gene = _brow.get('gene', '?')
        _ct   = _brow.get('cell_type', '?')
        _is_s1 = _gene in _s1_genes
        _title = f"{'🟡 ' if _is_s1 else '🧫 '}{_gene} — {_ct}"
        if _is_s1:
            _title += "  ·  🔴 Scenario 1 PASS"

        with st.expander(_title, expanded=bool(_bq)):
            # ── Quick actions ──────────────────────────────────────────────
            _ba_c1, _ba_c2, _ba_c3 = st.columns([2, 2, 4])
            with _ba_c1:
                _in_bk = _gene.upper() in [g.upper() for g in basket_gene_ids()]
                if _in_bk:
                    st.success("✅ 바스켓에 있음", icon=None)
                else:
                    if st.button("➕ 바스켓 추가", key=f'bisect_bk_{_gene}_{_ct}',
                                 use_container_width=True):
                        add_to_gene_basket(
                            _gene, source_page='bisect',
                            tag_scenario=1 if _is_s1 else None,
                        )
                        st.toast(f"✅ {_gene} 바스켓 추가")
                        st.rerun()
            with _ba_c2:
                if st.button("🧬 유전자 분석", key=f'bisect_gene_{_gene}_{_ct}',
                             use_container_width=True):
                    st.session_state['search_gene'] = _gene
                    st.session_state['gene_from_gene_page'] = False
                    st.switch_page("pages/06_gene.py")

            # ── Isoform pair + reading guide ──────────────────────────────
            _ct_tx = str(_brow.get('ct_transcript_id') or '').strip()
            _ad_tx = str(_brow.get('ad_transcript_id') or '').strip()
            _safe_ct_key = _ct.replace(' ', '_').replace('-', '_')

            # Case reading guide banner
            st.markdown(
                "<div style='background:#f0f9ff;border-left:4px solid #0ea5e9;"
                "padding:10px 14px;border-radius:6px;font-size:0.83rem;"
                "color:#0c4a6e;margin-bottom:10px'>"
                "<b>📖 이 케이스 읽는 법</b> &nbsp;—&nbsp; "
                "①&nbsp;<b>Volcano/Bar</b>: 어떤 TF·ASF가 얼마나 변했는지 "
                "②&nbsp;<b>Δ Usage 차트</b>: 아이소폼 비율이 얼마나 바뀌었는지 "
                "③&nbsp;<b>도메인 구조</b>: 어떤 단백질 기능이 추가/제거됐는지 "
                "④&nbsp;<b>GO 비교</b>: CT vs AD 이소폼의 기능 공간 차이 "
                "⑤&nbsp;<b>종합 리포트</b>: 인과 경로 전체 서사 "
                "⑥&nbsp;<b>세포유형 재현성</b>: 8개 뇌 세포유형 간 DTU 일관성"
                "</div>",
                unsafe_allow_html=True,
            )
            if _ct_tx or _ad_tx:
                st.markdown(
                    f"<div style='background:#f8fafc;border-radius:6px;"
                    f"padding:6px 12px;font-size:0.82rem;color:#475569;margin-bottom:8px'>"
                    f"분석 대상 이소폼 쌍 &nbsp;|&nbsp; "
                    f"🔵 <b>Control (CT)</b>: <code>{_ct_tx or '—'}</code> "
                    f"&nbsp;→&nbsp; "
                    f"🔴 <b>AD (Disease)</b>: <code>{_ad_tx or '—'}</code></div>",
                    unsafe_allow_html=True,
                )

            # ── DTU Δ Usage bar chart (O(1) lookup via cached dict) ───────
            _CT_COND_MAP = {'Excitatory': 'Excitatory neuron',
                            'Inhibitory': 'Inhibitory neuron'}
            _dtu_cond = _CT_COND_MAP.get(_ct, _ct)
            _g_dtu = _dtu_lookup.get((_gene, _dtu_cond),
                     _dtu_lookup.get((_gene, ''), pd.DataFrame()))
            if not _g_dtu.empty:
                _g_dtu = _g_dtu.sort_values('delta_IF').reset_index(drop=True)
                _g_dtu['role'] = _g_dtu['isoform_id'].map(
                    lambda iso: ('CT (Control)' if iso == _ct_tx
                                 else ('AD (Disease)' if iso == _ad_tx
                                       else 'Other isoform'))
                )
                _g_dtu['label'] = _g_dtu['delta_IF'].map(lambda v: f'{v:+.3f}')
                _fig_dtu = px.bar(
                    _g_dtu,
                    x='isoform_id', y='delta_IF',
                    color='role',
                    color_discrete_map={
                        'CT (Control)':   '#3b82f6',
                        'AD (Disease)':   '#ef4444',
                        'Other isoform':  '#94a3b8',
                    },
                    title=f"② 아이소폼 사용 비율 변화 (Δ Usage = AD − CT) — {_gene} · {_ct}",
                    labels={
                        'delta_IF':   'Δ Usage (AD − CT)  ·  양수 = AD에서 증가, 음수 = CT에서 우세',
                        'isoform_id': '아이소폼 ID',
                    },
                    text='label',
                    height=max(260, len(_g_dtu) * 32 + 90),
                )
                _fig_dtu.add_hline(y=0, line_color='#1e293b', line_width=1.2)
                _fig_dtu.update_traces(textposition='outside', textfont_size=9)
                _fig_dtu.update_layout(
                    xaxis_tickangle=-35,
                    legend_title='이소폼 역할',
                    plot_bgcolor='white',
                    yaxis=dict(gridcolor='#f0f0f0'),
                    margin=dict(t=45, b=80, l=10, r=10),
                )
                st.plotly_chart(_fig_dtu, use_container_width=True,
                                key=f"dtu_usage_{_gene}_{_safe_ct_key}")
                st.caption(
                    "🔵 CT (Control) 이소폼은 정상 조건에서 우세 (Δ Usage < 0). "
                    "🔴 AD (Disease) 이소폼은 알츠하이머 조건에서 비율 증가 (Δ Usage > 0). "
                    "|Δ Usage| ≥ 0.1이면 통계적으로 의미 있는 전환으로 판단합니다."
                )

            # ── Row 1: core metrics (6-col) ───────────────────────────────
            st.markdown(
                "<div style='font-size:0.78rem;color:#6b7280;margin:12px 0 4px'>"
                "📊 <b>핵심 정량 지표</b> — 각 지표 위에 마우스를 올리면 설명이 나옵니다</div>",
                unsafe_allow_html=True,
            )
            _r1c1, _r1c2, _r1c3, _r1c4, _r1c5, _r1c6 = st.columns(6)
            _delta = _brow.get('delta')
            _r1c1.metric(
                "Δ Usage (AD−CT)",
                f"{float(_delta):.3f}" if _delta is not None else "N/A",
                help="AD 조건에서 이 아이소폼 사용 비율 − CT 조건 비율. ±0.1 이상이면 유의미한 전환.",
            )
            _dtu_p = _brow.get('dtu_p')
            _r1c2.metric(
                "DTU p-value",
                f"{float(_dtu_p):.2e}" if _dtu_p else "N/A",
                help="아이소폼 비율 차이 통계 검정 p-value. 1e-5 미만이면 매우 유의미.",
            )
            _ct_plddt = _brow.get('af_ct_plddt_mean')
            _r1c3.metric(
                "CT pLDDT",
                f"{float(_ct_plddt):.1f}" if _ct_plddt else "N/A",
                help="AlphaFold2로 예측한 Control 이소폼 구조 신뢰도. 70↑ = 신뢰, 50↓ = 무질서",
            )
            _ad_plddt = _brow.get('af_ad_plddt_mean')
            _r1c4.metric(
                "AD pLDDT",
                f"{float(_ad_plddt):.1f}" if _ad_plddt else "N/A",
                delta="구조 신뢰" if _ad_plddt and float(_ad_plddt) >= 70 else None,
                help="AlphaFold2로 예측한 AD 이소폼 구조 신뢰도. 70 이상이면 실험 가능한 구조 가짐",
            )
            _dplddt = _brow.get('af_delta_plddt')
            _r1c5.metric(
                "ΔpLDDT (AD−CT)",
                f"{float(_dplddt):+.1f}" if _dplddt else "N/A",
                help="양수 = AD 이소폼이 CT보다 더 안정된 구조 | 음수 = AD 이소폼이 더 무질서함",
            )
            _phylo = _brow.get('cons_ad_phylop')
            _r1c6.metric(
                "phyloP (AD exon)",
                f"{float(_phylo):.3f}" if _phylo else "N/A",
                help="AD 특이적 엑손의 100종 척추동물 보존도. 1.5↑ = 강한 기능적 선택압 하에 있음",
            )

            # ── Row 2: domain changes with AlphaFold confidence ───────────
            st.markdown(
                "<div style='font-size:0.82rem;font-weight:600;color:#1e293b;"
                "margin:14px 0 4px'>③ 단백질 도메인 구조 변화 "
                "<span style='font-weight:400;color:#6b7280;font-size:0.75rem'>"
                "— Pfam 도메인 데이터베이스 + AlphaFold2 구조 확인</span></div>",
                unsafe_allow_html=True,
            )
            _dg     = str(_brow.get('domains_gained')       or '').strip()
            _dl     = str(_brow.get('domains_lost')         or '').strip()
            _af_gd  = str(_brow.get('af_gained_confident')  or '').strip()
            _af_ld  = str(_brow.get('af_lost_confident')    or '').strip()

            _dc1, _dc2 = st.columns(2)
            with _dc1:
                _dg_items = [d for d in _dg.split(';') if d]
                _dg_html  = ''.join(
                    f"<code style='background:#dcfce7;padding:1px 5px;"
                    f"border-radius:3px'>{d}</code> " for d in _dg_items
                ) if _dg_items else '<span style="color:#94a3b8">없음</span>'
                _af_gd_html = (
                    f"<br><span style='font-size:0.75rem;color:#15803d'>"
                    f"🏗 AlphaFold 확인: {_af_gd}</span>"
                ) if _af_gd else ''
                st.markdown(
                    f"<div style='background:#f0fdf4;border-left:3px solid #22c55e;"
                    f"padding:8px 12px;border-radius:4px;font-size:0.85rem'>"
                    f"<b>도메인 획득 (AD)</b><br>{_dg_html}{_af_gd_html}</div>",
                    unsafe_allow_html=True,
                )
            with _dc2:
                _dl_items = [d for d in _dl.split(';') if d]
                _dl_html  = ''.join(
                    f"<code style='background:#fee2e2;padding:1px 5px;"
                    f"border-radius:3px'>{d}</code> " for d in _dl_items
                ) if _dl_items else '<span style="color:#94a3b8">없음</span>'
                _af_ld_html = (
                    f"<br><span style='font-size:0.75rem;color:#b91c1c'>"
                    f"🏗 AlphaFold 확인: {_af_ld}</span>"
                ) if _af_ld else ''
                st.markdown(
                    f"<div style='background:#fef2f2;border-left:3px solid #ef4444;"
                    f"padding:8px 12px;border-radius:4px;font-size:0.85rem'>"
                    f"<b>도메인 손실 (CT)</b><br>{_dl_html}{_af_ld_html}</div>",
                    unsafe_allow_html=True,
                )

            # ── Row 3: PPI + conservation detail ─────────────────────────
            _ppi_v  = str(_brow.get('ppi_verdict')    or '').strip()
            _ppi_p  = str(_brow.get('ppi_top_partner')or '').strip()
            _ppi_s  = _brow.get('ppi_top_score')
            _ppi_n  = _brow.get('ppi_n_string_hits')
            _cons_c = str(_brow.get('cons_ad_class')  or '').strip()
            _cons_bg= _brow.get('cons_background_phylop')
            _top_reg= str(_brow.get('top_regulators') or '').strip()

            _det_items = []
            if _ppi_v:
                _ppi_clr = '#15803d' if _ppi_v == 'SUPPORTED' else '#b91c1c'
                _ppi_txt = f"PPI: <b style='color:{_ppi_clr}'>{_ppi_v}</b>"
                if _ppi_p:
                    _ppi_txt += f" (top: {_ppi_p}"
                    if _ppi_s:
                        _ppi_txt += f" score={int(_ppi_s)}"
                    _ppi_txt += f", n={int(_ppi_n) if _ppi_n else '?'})"
                _det_items.append(_ppi_txt)
            if _phylo:
                _cons_txt = f"Conservation: phyloP={float(_phylo):.3f}"
                if _cons_c:
                    _cons_txt += f" ({_cons_c})"
                if _cons_bg:
                    _cons_txt += f" | bg={float(_cons_bg):.3f}"
                _det_items.append(_cons_txt)
            if _top_reg:
                _det_items.append(f"Top regulators: {_top_reg}")

            if _det_items:
                st.markdown(
                    "<div style='background:#f8fafc;border-radius:6px;padding:8px 12px;"
                    "font-size:0.82rem;color:#374151;margin:8px 0;line-height:1.8'>"
                    + "<br>".join(_det_items) + "</div>",
                    unsafe_allow_html=True,
                )

            # ── Row 4: BISECT 모듈 전체 그리드 ───────────────────────────
            st.markdown(_build_module_grid_html(dict(_brow)), unsafe_allow_html=True)

            # ── Regulatory Origin Analysis ────────────────────────────────
            _case_regs = _parse_regulators(_brow.get('top_regulators', ''))
            _case_mech = str(_brow.get('mechanism_type') or '').strip()
            _case_tss  = str(_brow.get('tss_class')      or '').strip()
            _case_apa  = str(_brow.get('apa_class')       or '').strip()

            if _case_regs or _case_mech:
                st.divider()
                st.markdown(
                    "**① 조절 인자 분석 — 아이소폼 전환의 상류 원인 (Regulatory Origin)**"
                )
                st.caption(
                    "아이소폼 비율이 달라진 직접적 원인(TF 활성, 스플라이싱 인자, 후성유전 변화)을 역추적합니다. "
                    "Volcano에서 왼쪽 위 = AD에서 억제된 인자, 오른쪽 위 = AD에서 활성화된 인자. "
                    "🔵 원 = 기존 AD 문헌에 알려진 인자, 🟠 다이아몬드 = 이 연구에서 새로 발견된 인자."
                )

                _mech_info = _MECHANISM_KO.get(_case_mech, ('', '#64748b', ''))
                _mech_ko_label, _mech_color, _mech_desc = _mech_info

                # Mechanism banner
                if _case_mech:
                    _tss_note = ''
                    _apa_note = ''
                    _tss_bp_v = _brow.get('tss_diff_bp')
                    _apa_bp_v = _brow.get('tts_diff_bp')
                    if _case_tss and _case_tss not in ('same_promoter', ''):
                        _tss_note = f" · TSS: <b>{_case_tss}</b>"
                        if _tss_bp_v:
                            _tss_note += f" ({int(float(_tss_bp_v)):+d}bp)"
                    if _case_apa and _case_apa not in ('same_apa', ''):
                        _apa_note = f" · APA: <b>{_case_apa}</b>"
                        if _apa_bp_v:
                            _apa_note += f" ({int(float(_apa_bp_v)):+d}bp)"
                    st.markdown(
                        f"<div style='background:#faf5ff;border-left:4px solid {_mech_color};"
                        f"padding:10px 14px;border-radius:6px;margin-bottom:10px;"
                        f"font-size:0.86rem'>"
                        f"<b style='color:{_mech_color}'>⚙️ {_mech_ko_label}</b>"
                        f"{_tss_note}{_apa_note}<br>"
                        f"<span style='color:#374151'>{_mech_desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # TF/ASF bar chart (logFC, colored by direction, annotated)
                if _case_regs:
                    _rrr = []
                    for _r in _case_regs:
                        _rg = _r.get('gene', '?')
                        _kb = _REGULATOR_KB.get(_rg, ('TF', None, ''))
                        _rrr.append({
                            'Gene':       _rg,
                            'logFC':      float(_r.get('logFC', 0)),
                            'Direction':  _r.get('direction', '').capitalize(),
                            'Category':   _kb[0] or 'TF',
                            'Known':      '🔵 Known AD' if _kb[1] else '🟠 Novel',
                            '-log10p':    float(_r.get('neg_log10_padj', 0)),
                            'Label':      f"{_rg}\n({_kb[0] or 'TF'})",
                        })
                    _rdf = pd.DataFrame(_rrr).sort_values('logFC')

                    # ── Per-case Volcano (full sig_regs from analysis.json) ──
                    _sig_regs_full = _load_case_sig_regs(_gene, _ct)
                    _vol_src = _sig_regs_full if _sig_regs_full else _case_regs
                    _vrows = []
                    for _vr in _vol_src:
                        _vg   = _vr.get('gene', '?')
                        _vlfc = float(_vr.get('logFC', 0))
                        _vnlp = float(_vr.get('neg_log10_padj', 0))
                        _vdir = _vr.get('direction', '').capitalize()
                        _kb2  = _REGULATOR_KB.get(_vg, ('TF', None, ''))
                        _vknown = (
                            '🔵 Known AD' if _kb2[1] is True
                            else ('🟠 Novel' if _kb2[1] is False else '⚪ Unknown')
                        )
                        _vrows.append({
                            'Gene':         _vg,
                            'logFC':        _vlfc,
                            '-log10(padj)': _vnlp,
                            'Direction':    _vdir,
                            'Category':     _kb2[0] or 'TF',
                            'Knowledge':    _vknown,
                            'Label': _vg if (_kb2[1] is True or _vnlp > 20) else '',
                        })
                    if _vrows:
                        _vdf = pd.DataFrame(_vrows)
                        _fig_vol = px.scatter(
                            _vdf,
                            x='logFC', y='-log10(padj)',
                            color='Direction',
                            symbol='Knowledge',
                            color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                            symbol_map={
                                '🔵 Known AD': 'circle',
                                '🟠 Novel':    'diamond',
                                '⚪ Unknown':  'square',
                            },
                            text='Label',
                            hover_data=['Gene', 'Category', 'Knowledge'],
                            title=f'Volcano — TF/ASF Activity · {_gene} · {_ct}',
                            labels={
                                'logFC':        'logFC (AD vs CT)',
                                '-log10(padj)': '-log₁₀(p-adj)',
                            },
                            height=360,
                        )
                        _fig_vol.update_traces(
                            textposition='top center',
                            textfont=dict(size=9, color='#1e293b'),
                            marker=dict(size=10, opacity=0.85),
                        )
                        _fig_vol.add_vline(x=0.1,  line_dash='dash', line_color='#94a3b8', line_width=1)
                        _fig_vol.add_vline(x=-0.1, line_dash='dash', line_color='#94a3b8', line_width=1)
                        _fig_vol.add_hline(y=2.0,  line_dash='dash', line_color='#94a3b8', line_width=1)
                        _fig_vol.add_vline(x=0,    line_color='#374151', line_width=1.2)
                        _fig_vol.update_layout(
                            plot_bgcolor='white',
                            xaxis=dict(gridcolor='#f0f0f0'),
                            yaxis=dict(gridcolor='#f0f0f0'),
                            legend_title='',
                            margin=dict(t=42, b=20, l=10, r=10),
                            font=dict(size=11),
                        )
                        st.plotly_chart(
                            _fig_vol, use_container_width=True,
                            key=f"vol_{_gene}_{_safe_ct_key}",
                        )
                        st.caption(
                            f"Volcano: X=logFC (AD vs CT), Y=-log₁₀(p-adj). "
                            f"점선: |logFC|=0.1, -log₁₀p=2. 총 {len(_vrows)}개 인자"
                            + (" (analysis.json 전체)" if _sig_regs_full else " (top regulators)")
                            + ". ● = 기존 AD, ◆ = 신규 발견."
                        )

                    _r_color = {
                        row['Gene']: ('#ef4444' if row['Direction'] == 'Up' else '#3b82f6')
                        for _, row in _rdf.iterrows()
                    }
                    _fig_reg = px.bar(
                        _rdf,
                        x='logFC', y='Gene',
                        orientation='h',
                        color='Direction',
                        color_discrete_map={'Up': '#ef4444', 'Down': '#3b82f6'},
                        text=_rdf['Gene'].map(
                            lambda g: _REGULATOR_KB.get(g, ('', '', ''))[0]
                        ),
                        hover_data=['Category', 'Known', '-log10p'],
                        title=f'TF / ASF 활성 변화 — {_gene}  ·  {_ct} (AD vs CT logFC)',
                        labels={'logFC': 'logFC (AD vs CT)', 'Gene': ''},
                        height=max(200, len(_rrr) * 52 + 80),
                    )
                    _fig_reg.add_vline(x=0, line_color='#374151', line_width=1.5,
                                       line_dash='dash')
                    _fig_reg.update_traces(textposition='outside', textfont_size=9)
                    _fig_reg.update_layout(
                        plot_bgcolor='white',
                        xaxis=dict(gridcolor='#f0f0f0'),
                        legend_title='',
                        margin=dict(t=40, b=20, l=60, r=90),
                    )
                    st.plotly_chart(_fig_reg, use_container_width=True,
                                    key=f"reg_chart_{_gene}_{_safe_ct_key}")

                    # Known vs Novel annotation cards
                    _known_here   = [r for r in _case_regs
                                     if _REGULATOR_KB.get(r['gene'], (None, None))[1] is True]
                    _novel_here   = [r for r in _case_regs
                                     if _REGULATOR_KB.get(r['gene'], (None, None))[1] is False]
                    _unknown_here = [r for r in _case_regs
                                     if r['gene'] not in _REGULATOR_KB]

                    _ann_parts = []
                    if _known_here:
                        _kh_str = '; '.join(
                            f"<b>{r['gene']}</b> ({r['direction']}, logFC={r['logFC']:+.3f}) "
                            f"— {_REGULATOR_KB[r['gene']][2]}"
                            for r in _known_here
                        )
                        _ann_parts.append(
                            f"<div style='background:#eff6ff;border-left:3px solid #3b82f6;"
                            f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                            f"font-size:0.82rem'>"
                            f"<b style='color:#1d4ed8'>🔵 기존 AD 연관 인자</b><br>{_kh_str}</div>"
                        )
                    if _novel_here:
                        _nh_str = '; '.join(
                            f"<b>{r['gene']}</b> ({r['direction']}, logFC={r['logFC']:+.3f}) "
                            f"— {_REGULATOR_KB[r['gene']][2]}"
                            for r in _novel_here
                        )
                        _ann_parts.append(
                            f"<div style='background:#fff7ed;border-left:3px solid #f59e0b;"
                            f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                            f"font-size:0.82rem'>"
                            f"<b style='color:#b45309'>🟠 새로 발견된 조절 인자</b><br>{_nh_str}</div>"
                        )
                    if _unknown_here:
                        _uh_str = ', '.join(
                            f"{r['gene']} ({r['direction']}, logFC={r['logFC']:+.3f})"
                            for r in _unknown_here
                        )
                        _ann_parts.append(
                            f"<div style='background:#f8fafc;border-left:3px solid #94a3b8;"
                            f"padding:8px 12px;border-radius:4px;margin:4px 0;"
                            f"font-size:0.82rem'>"
                            f"<b style='color:#475569'>⚪ 기타 인자</b>: {_uh_str}</div>"
                        )
                    if _ann_parts:
                        st.markdown("".join(_ann_parts), unsafe_allow_html=True)

            # ── Proportion chart + GO comparison + Bio report ────────────
            _ids_arr_b = np.asarray(ids, dtype=str)
            _ct_idx_b  = np.where(_ids_arr_b == _ct_tx)[0]
            _ad_idx_b  = np.where(_ids_arr_b == _ad_tx)[0]
            _ct_go_scores = sm[_ct_idx_b[0]] if len(_ct_idx_b) > 0 else None
            _ad_go_scores = sm[_ad_idx_b[0]] if len(_ad_idx_b) > 0 else None

            st.divider()

            # 1. Proportion chart — estimated isoform usage ratio CT vs AD
            st.markdown("**② 아이소폼 사용 비율 추정 — Control(CT) vs Disease(AD)**")
            if not _g_dtu.empty and (_ct_tx or _ad_tx):
                _n_iso = len(_g_dtu)
                _prop = _g_dtu[['isoform_id', 'delta_IF']].copy()
                _prop['ct_frac'] = 1.0 / _n_iso
                _prop['ad_frac'] = (_prop['ct_frac'] + _prop['delta_IF']).clip(lower=0)
                _sum_ad = _prop['ad_frac'].sum()
                if _sum_ad > 0:
                    _prop['ad_frac'] /= _sum_ad

                # Collapse non-focal isoforms into 'Other'
                _focal_set = {_ct_tx, _ad_tx}
                _prop_rows = []
                _other_ct = _other_ad = 0.0
                for _, _pr in _prop.iterrows():
                    _iso = _pr['isoform_id']
                    if _iso in _focal_set:
                        _label = (f'🔵 CT: {_iso[:22]}' if _iso == _ct_tx
                                  else f'🔴 AD: {_iso[:22]}')
                        _prop_rows.append({'Condition': 'Control (CT)', 'Label': _label,
                                           'Fraction': _pr['ct_frac'], 'IsoType': _iso})
                        _prop_rows.append({'Condition': 'Disease (AD)', 'Label': _label,
                                           'Fraction': _pr['ad_frac'], 'IsoType': _iso})
                    else:
                        _other_ct += _pr['ct_frac']
                        _other_ad += _pr['ad_frac']
                _prop_rows.append({'Condition': 'Control (CT)', 'Label': '◻ Other isoforms',
                                   'Fraction': _other_ct, 'IsoType': 'other'})
                _prop_rows.append({'Condition': 'Disease (AD)', 'Label': '◻ Other isoforms',
                                   'Fraction': _other_ad, 'IsoType': 'other'})

                _prop_df2 = pd.DataFrame(_prop_rows)
                _ct_label = f'🔵 CT: {_ct_tx[:22]}'
                _ad_label = f'🔴 AD: {_ad_tx[:22]}'
                _color_map_prop = {
                    _ct_label:           '#3b82f6',
                    _ad_label:           '#ef4444',
                    '◻ Other isoforms':  '#cbd5e1',
                }
                _cat_order_prop = [_ct_label, _ad_label, '◻ Other isoforms']

                _fig_prop = px.bar(
                    _prop_df2,
                    x='Condition', y='Fraction', color='Label',
                    barmode='stack',
                    color_discrete_map=_color_map_prop,
                    category_orders={'Label': _cat_order_prop},
                    title=f'Transcript Usage — {_gene}  ·  {_ct} (추정)',
                    labels={'Fraction': '이소폼 사용 비율 (추정)', 'Condition': ''},
                    height=310,
                )
                _fig_prop.update_layout(
                    plot_bgcolor='white',
                    yaxis=dict(tickformat='.0%', range=[0, 1.05], gridcolor='#f0f0f0'),
                    legend_title='이소폼',
                    margin=dict(t=38, b=30, l=10, r=10),
                    bargap=0.35,
                )
                _fig_prop.update_yaxes(tickformat='.0%')
                st.caption(
                    "CT 조건에서 모든 이소폼이 균등하게 발현된다고 가정(1/n)한 뒤, "
                    "DTU Δ Usage를 더해 AD 조건의 비율을 추정합니다. "
                    "🔵 CT 이소폼이 정상에서 주로 쓰이다가, 🔴 AD 이소폼으로 전환되는 비율 변화를 시각화합니다. "
                    "핵심 이소폼 쌍만 강조하며 나머지는 '◻ Other'로 묶습니다."
                )
                st.plotly_chart(_fig_prop, use_container_width=True,
                                key=f"prop_{_gene}_{_safe_ct_key}")

            elif _ct_tx or _ad_tx:
                # Fallback: DTU 데이터 없을 때 BISECT delta로 방향성 표시
                _dv = float(_brow.get('delta') or 0)
                _fb_rows = []
                if _ct_tx:
                    _fb_rows.append({'이소폼': f'CT: {_ct_tx[:30]}',
                                     '역할': '🔵 CT (Control)',
                                     'Δ Usage': _dv})
                if _ad_tx:
                    _fb_rows.append({'이소폼': f'AD: {_ad_tx[:30]}',
                                     '역할': '🔴 AD (Disease)',
                                     'Δ Usage': -_dv})
                if _fb_rows:
                    _fb_df = pd.DataFrame(_fb_rows)
                    _fig_fb = px.bar(
                        _fb_df, x='이소폼', y='Δ Usage', color='역할',
                        color_discrete_map={
                            '🔵 CT (Control)': '#3b82f6',
                            '🔴 AD (Disease)': '#ef4444',
                        },
                        title=f'BISECT Δ Usage 방향 — {_gene} · {_ct}',
                        labels={'Δ Usage': 'Δ Usage (AD − CT)', '이소폼': ''},
                        height=260,
                    )
                    _fig_fb.add_hline(y=0, line_color='#1e293b', line_width=1.2)
                    _fig_fb.update_layout(
                        plot_bgcolor='white',
                        yaxis=dict(gridcolor='#f0f0f0'),
                        legend_title='',
                        margin=dict(t=38, b=40, l=10, r=10),
                    )
                    st.caption(
                        "DTU 상세 데이터 없음 — BISECT delta로 방향성만 표시. "
                        "Brain 조직 선택 시 전체 이소폼 비율 차트로 전환됩니다."
                    )
                    st.plotly_chart(_fig_fb, use_container_width=True,
                                    key=f"prop_fb_{_gene}_{_safe_ct_key}")

            # 2. GO function comparison chart — CT vs AD isoform
            _pmatch_ct = str(_brow.get('prism_match_ct') or '').strip()
            _pmatch_ad = str(_brow.get('prism_match_ad') or '').strip()
            _gene_median_both_flag = (_pmatch_ct == 'gene_median' and _pmatch_ad == 'gene_median')
            _proxy_ad_flag = _pmatch_ad.startswith('proxy:')

            if _ct_go_scores is not None or _ad_go_scores is not None:
                st.markdown("**④ GO 기능 예측 점수 비교 — CT 이소폼 vs AD 이소폼**")

                # PRISM match quality badges
                def _match_badge(m: str) -> str:
                    if 'exact' in m:
                        return (f"<span style='background:#dcfce7;color:#15803d;"
                                f"border-radius:4px;padding:2px 7px;font-size:0.78rem;"
                                f"font-weight:600'>● exact</span>")
                    elif m.startswith('proxy:'):
                        _proxy_id = m[len('proxy:'):]
                        return (f"<span style='background:#ede9fe;color:#5b21b6;"
                                f"border-radius:4px;padding:2px 7px;font-size:0.78rem;"
                                f"font-weight:600'>◆ proxy ({_proxy_id})</span>")
                    elif m == 'gene_median':
                        return (f"<span style='background:#fef3c7;color:#b45309;"
                                f"border-radius:4px;padding:2px 7px;font-size:0.78rem;"
                                f"font-weight:600'>▲ gene median</span>")
                    return (f"<span style='background:#f1f5f9;color:#64748b;"
                            f"border-radius:4px;padding:2px 7px;font-size:0.78rem'>— N/A</span>")

                st.markdown(
                    f"<div style='margin:4px 0 8px;font-size:0.82rem;color:#374151'>"
                    f"PRISM 매칭 품질 &nbsp;"
                    f"CT {_match_badge(_pmatch_ct)} &nbsp;"
                    f"AD {_match_badge(_pmatch_ad)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if _gene_median_both_flag:
                    # Both isoforms fall back to gene median — comparison not informative
                    st.warning(
                        "⚠️ CT·AD 이소폼 모두 PRISM 임베딩에서 **유전자 중앙값(gene median)**으로 대체되어 "
                        "두 아이소폼의 GO 점수가 동일합니다. 이소폼-특이적 비교가 불가능하므로 아래 차트는 "
                        "참고용입니다. 대신 사전 계산된 GAIN/LOSS 데이터를 생물학 해석 리포트(⑤)에서 확인하세요.",
                        icon=None,
                    )
                    _go_title = f'PRISM GO Score (gene median) · {_gene}'
                elif _proxy_ad_flag:
                    _proxy_label = _pmatch_ad[len('proxy:'):]
                    st.info(
                        f"ℹ️ AD 이소폼({_brow.get('ad_transcript_id','?')})이 brain long-read 데이터셋에 "
                        f"직접 관측되지 않아 **{_proxy_label}** (코사인 유사도 0.999)를 proxy로 사용합니다. "
                        f"동일 유전자 36개 아이소폼의 중앙값과 거의 동일 — GO 점수 방향성은 신뢰할 수 있습니다.",
                        icon=None,
                    )
                    _go_title = f'PRISM GO Score — CT vs AD proxy ({_proxy_label}) · {_gene}'
                else:
                    _go_title = f'PRISM GO Score — CT vs AD Isoform · {_gene}'

                _top_n_go = 6
                _union_idx: set = set()
                if _ct_go_scores is not None:
                    _union_idx |= set(np.argsort(_ct_go_scores)[-_top_n_go:].tolist())
                if _ad_go_scores is not None:
                    _union_idx |= set(np.argsort(_ad_go_scores)[-_top_n_go:].tolist())

                _cmp_rows = []
                for _gi in sorted(_union_idx):
                    _gn = gnames.get(go[_gi], go[_gi])[:38]
                    if _ct_go_scores is not None:
                        _cmp_rows.append({'GO term': _gn, 'Score': float(_ct_go_scores[_gi]),
                                          'Isoform': 'CT', 'IsoLabel': f'🔵 CT ({(_ct_tx or "—")[:18]})'})
                    if _ad_go_scores is not None:
                        _cmp_rows.append({'GO term': _gn, 'Score': float(_ad_go_scores[_gi]),
                                          'Isoform': 'AD', 'IsoLabel': f'🔴 AD ({(_ad_tx or "—")[:18]})'})

                if _cmp_rows:
                    _cmp_df = pd.DataFrame(_cmp_rows)
                    _go_order = (
                        _cmp_df.groupby('GO term')['Score'].max()
                        .sort_values(ascending=False).index.tolist()
                    )
                    _ct_iso_label = f'🔵 CT ({(_ct_tx or "—")[:18]})'
                    _ad_iso_label = f'🔴 AD ({(_ad_tx or "—")[:18]})'
                    _fig_cmp = px.bar(
                        _cmp_df, x='GO term', y='Score', color='IsoLabel',
                        barmode='group',
                        color_discrete_map={
                            _ct_iso_label: '#3b82f6',
                            _ad_iso_label: '#ef4444',
                        },
                        category_orders={
                            'GO term': _go_order,
                            'IsoLabel': [_ct_iso_label, _ad_iso_label],
                        },
                        title=_go_title,
                        labels={'Score': 'PRISM Score', 'GO term': '', 'IsoLabel': '이소폼'},
                        height=330,
                    )
                    _fig_cmp.add_hline(
                        y=float(thr), line_dash='dash', line_color='#94a3b8',
                        annotation_text=f'threshold ({thr})',
                    )
                    _fig_cmp.update_layout(
                        xaxis_tickangle=-38,
                        plot_bgcolor='white',
                        yaxis=dict(range=[0, 1.05], gridcolor='#f0f0f0'),
                        legend_title='',
                        margin=dict(t=38, b=80, l=10, r=10),
                    )
                    if _gene_median_both_flag:
                        st.caption(
                            "두 이소폼 모두 유전자 중앙값으로 PRISM 점수가 계산되어 CT와 AD 막대가 "
                            "동일합니다. 이소폼-특이적 비교 의미가 없으므로 절대 점수 참고에만 활용하세요."
                        )
                    else:
                        st.caption(
                            "PRISM이 예측한 GO term 기능 점수(0~1)를 두 이소폼 간 나란히 비교합니다. "
                            "🔵 CT 이소폼과 🔴 AD 이소폼에서 높이 차이가 큰 GO term이 "
                            "이소폼 전환으로 인한 **기능 변화 후보**입니다. "
                            f"점선(threshold = {thr})을 넘는 항목만 신뢰도 높은 예측으로 간주합니다."
                        )
                    st.plotly_chart(_fig_cmp, use_container_width=True,
                                    key=f"go_cmp_{_gene}_{_safe_ct_key}")

                # JSON pre-computed GAIN/LOSS (available regardless of match quality)
                _gain_json = _brow.get('prism_gain_go') or []
                _loss_json = _brow.get('prism_loss_go') or []
                if _gain_json or _loss_json:
                    st.markdown(
                        "<div style='font-size:0.83rem;font-weight:600;"
                        "color:#1e293b;margin:14px 0 6px'>"
                        "PRISM 기능 변화 — 사전 계산된 GAIN / LOSS GO terms"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    _gl_cols = st.columns(2)
                    with _gl_cols[0]:
                        if _gain_json:
                            st.markdown(
                                "<span style='color:#15803d;font-size:0.82rem;font-weight:600'>"
                                "▲ AD에서 기능 획득 (GAIN)</span>",
                                unsafe_allow_html=True,
                            )
                            for _d in _gain_json[:5]:
                                _gs = float(_d.get('ad_score', 0))
                                _cs = float(_d.get('ct_score', 0))
                                _delta = _gs - _cs
                                st.markdown(
                                    f"<div style='font-size:0.8rem;padding:3px 0;"
                                    f"border-bottom:1px solid #f0f0f0'>"
                                    f"<b>{_d.get('go_name','')[:36]}</b><br>"
                                    f"<span style='color:#6b7280'>{_d.get('go_id','')}</span>"
                                    f"&nbsp; AD {_gs:.3f} / CT {_cs:.3f}"
                                    f"&nbsp; <b style='color:#15803d'>Δ+{_delta:.3f}</b>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    with _gl_cols[1]:
                        if _loss_json:
                            st.markdown(
                                "<span style='color:#dc2626;font-size:0.82rem;font-weight:600'>"
                                "▼ AD에서 기능 소실 (LOSS)</span>",
                                unsafe_allow_html=True,
                            )
                            for _d in _loss_json[:5]:
                                _cs2 = float(_d.get('ct_score', 0))
                                _as2 = float(_d.get('ad_score', 0))
                                _delta2 = _as2 - _cs2
                                st.markdown(
                                    f"<div style='font-size:0.8rem;padding:3px 0;"
                                    f"border-bottom:1px solid #f0f0f0'>"
                                    f"<b>{_d.get('go_name','')[:36]}</b><br>"
                                    f"<span style='color:#6b7280'>{_d.get('go_id','')}</span>"
                                    f"&nbsp; CT {_cs2:.3f} / AD {_as2:.3f}"
                                    f"&nbsp; <b style='color:#dc2626'>Δ{_delta2:.3f}</b>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

            # 3. Biological prediction report (⑤)
            st.markdown(
                "**⑤ 종합 생물학 해석 리포트**&nbsp;"
                "<span style='font-size:0.8rem;color:#6b7280;font-weight:400'>"
                "— PRISM+BISECT 15개 모듈 분석을 인과 경로 형태로 통합 서술</span>",
                unsafe_allow_html=True,
            )
            _bio_html = _build_bio_report_html(
                brow=_brow, gene=_gene, ct_type=_ct,
                ct_tx=_ct_tx, ad_tx=_ad_tx,
                ct_scores=_ct_go_scores, ad_scores=_ad_go_scores,
                go_ids=go, go_names=gnames, threshold=thr,
            )
            st.markdown(_bio_html, unsafe_allow_html=True)

            # ── ⑥ Cell-type Concordance ───────────────────────────────────
            _diu_full = _load_bisect_diu_full()
            if _diu_full is not None:
                _diu_gene = _diu_full[
                    _diu_full['gene_name'].str.upper() == _gene.upper()
                ].copy()
                _sig_mask = _diu_gene['chi_significant'] == True
                _pivot_ct = (
                    _diu_gene[_sig_mask]
                    .pivot_table(
                        index='transcript_name', columns='cell_type',
                        values='delta_usage', aggfunc='first',
                    )
                    .fillna(0)
                )
                _cell_order_ct = [c for c in [
                    'Excitatory neuron', 'Inhibitory neuron', 'Astrocyte',
                    'Microglia', 'Oligodendrocyte', 'OPC',
                    'Vascular cell', 'Lymphocyte',
                ] if c in _pivot_ct.columns]
                _pivot_ct = _pivot_ct[[c for c in _cell_order_ct if c in _pivot_ct.columns]]

                if not _pivot_ct.empty:
                    st.divider()
                    _n_ct_gene = len(_pivot_ct.columns)
                    _dir_counts = _diu_gene[_sig_mask]['usage_direction'].value_counts()
                    _n_ad_gene  = int(_dir_counts.get('AD_enriched', 0))
                    _n_ctrl_gene = int(_dir_counts.get('CT_enriched', 0))
                    st.markdown(
                        f"**⑥ 세포유형 재현성**&nbsp;"
                        f"<span style='font-size:0.8rem;color:#6b7280;font-weight:400'>"
                        f"— {_n_ct_gene}개 세포유형 · "
                        f"유의 DTU {int(_sig_mask.sum())}건 · "
                        f"AD-enriched {_n_ad_gene}건 · CT-enriched {_n_ctrl_gene}건</span>",
                        unsafe_allow_html=True,
                    )

                    _fig_ct6 = px.imshow(
                        _pivot_ct,
                        color_continuous_scale='RdBu_r',
                        color_continuous_midpoint=0,
                        zmin=-0.5, zmax=0.5,
                        labels={'color': 'ΔIF'},
                        aspect='auto',
                        height=max(240, 38 * len(_pivot_ct) + 80),
                    )
                    _fig_ct6.update_layout(
                        xaxis_tickangle=-25,
                        margin=dict(t=20, b=40, l=10, r=10),
                        coloraxis_colorbar=dict(title='ΔIF', len=0.6, thickness=12),
                    )
                    st.plotly_chart(_fig_ct6, use_container_width=True,
                                    key=f'bisect_ct6_{_gene}_{_ct}')

                    # Concordance table (compact)
                    _conc6 = {}
                    for _iso6 in _pivot_ct.index:
                        _v6 = _pivot_ct.loc[_iso6]
                        _nz6 = _v6[_v6 != 0]
                        if len(_nz6) >= 2:
                            _p6, _n6 = int((_nz6 > 0).sum()), int((_nz6 < 0).sum())
                            _conc6[_iso6] = round(max(_p6, _n6) / len(_nz6), 2)
                        elif len(_nz6) == 1:
                            _conc6[_iso6] = 1.0
                        else:
                            _conc6[_iso6] = 0.0
                    _conc6_df = pd.DataFrame([
                        {'아이소폼': k,
                         'Concordance': v,
                         'N 세포유형': int((_pivot_ct.loc[k] != 0).sum()),
                         '방향': ('AD↑' if _pivot_ct.loc[k][_pivot_ct.loc[k] != 0].mean() > 0
                                  else 'CT↑')}
                        for k, v in sorted(_conc6.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(
                        _conc6_df.style.background_gradient(
                            subset=['Concordance'], cmap='Greens', vmin=0, vmax=1,
                        ),
                        use_container_width=True, hide_index=True,
                        height=min(220, 38 * len(_conc6_df) + 40),
                    )
                    st.caption(
                        "빨강 = AD에서 아이소폼 사용 증가 (AD-enriched) · "
                        "파랑 = CT에서 증가. "
                        "Concordance 1.0 = 모든 세포유형에서 같은 방향."
                    )

            # ── Domain structure + IGV genomic view ───────────────────────
            _BISECT_OUT = Path(__file__).parents[3] / \
                'Final_analysis' / 'pipeline_bioanalysis' / 'outputs'
            _dmap = _BISECT_OUT / f"{_gene}_{_ct}" / "domain_map.png"
            if _dmap.exists():
                st.divider()
                st.markdown(
                    "<div style='font-size:0.88rem;font-weight:600;"
                    "color:#1e293b;margin-bottom:4px'>"
                    "③ 단백질 도메인 구조 변화 지도 (CT → AD)</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"🔵 Control 이소폼: `{_ct_tx or '—'}` &nbsp;→&nbsp; "
                    f"🔴 AD 이소폼: `{_ad_tx or '—'}`. "
                    "각 막대가 하나의 이소폼을 나타내며, 색상 블록이 Pfam 도메인 위치입니다. "
                    "초록 = AD에서 새로 생긴 도메인(GAIN) | 빨강 = CT에서 사라진 도메인(LOSS). "
                    "AlphaFold pLDDT ≥ 70인 도메인만 구조적으로 신뢰 가능합니다."
                )
                st.image(str(_dmap), use_column_width=True)

            # ── IGV / UCSC quick links ────────────────────────────────────
            st.divider()
            st.markdown(
                "<div style='font-size:0.88rem;font-weight:600;"
                "color:#1e293b;margin-bottom:2px'>"
                "🧬 유전체 브라우저 — 엑손 구조 직접 확인 (외부 링크)</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "아래 버튼으로 외부 유전체 브라우저를 열어 실제 엑손 배치와 전사체 구조를 확인하세요. "
                "IGV / UCSC에서 아이소폼 ID로 검색하면 CT·AD 전사체의 엑손 차이를 시각적으로 비교할 수 있습니다."
            )
            _ql1, _ql2, _ql3 = st.columns(3)
            _igv_url  = f"https://igv.org/app/?genome=hg38&locus={_gene}"
            _ucsc_url = (f"https://genome.ucsc.edu/cgi-bin/hgTracks"
                         f"?db=hg38&position={_gene}&knownGene=pack"
                         f"&wgEncodeGencodeCompV45=pack")
            _ens_url  = (f"https://www.ensembl.org/Homo_sapiens/Gene/Summary"
                         f"?q={_gene};db=core")
            _ql1.link_button("🔬 IGV Web (hg38)", _igv_url)
            _ql2.link_button("🌐 UCSC Genome Browser", _ucsc_url)
            _ql3.link_button("🧫 Ensembl Gene View", _ens_url)
            st.caption(
                f"핵심 전사체: 🔵 CT `{_ct_tx or '—'}` / "
                f"🔴 AD `{_ad_tx or '—'}` — 각 브라우저에서 해당 전사체 ID로 검색하세요."
            )

            # ── Warnings ──────────────────────────────────────────────────
            if _brow.get('nat'):
                st.warning("⚠️ NAT (Natural Antisense Transcript) overlap 감지됨")
            if _brow.get('young_l1_cds'):
                st.warning("⚠️ Young L1 retrotransposon이 CDS 내 삽입됨")
            if _brow.get('nmd_relevant'):
                st.info("ℹ️ NMD (Nonsense-Mediated Decay) 관련 구조 감지됨")

            # ── Option C: PRISM GO score chart for S1-overlapping genes ──
            if _is_s1 and _gene in _gene_to_rows:
                st.divider()
                st.markdown(
                    "<div style='background:#fef9c3;border-left:3px solid #eab308;"
                    "padding:8px 12px;border-radius:4px;font-size:0.85rem;margin-bottom:8px'>"
                    "🔴 <b>Scenario 1 확인됨</b> — 현재 데이터셋에서 DTU + 신규 GO 예측 교차 검증"
                    "</div>",
                    unsafe_allow_html=True,
                )
                _gene_rows = _gene_to_rows[_gene]
                # Show top isoforms by max_score
                _top_isos = _gene_rows.nlargest(min(5, len(_gene_rows)), 'max_score')
                for _, _irow in _top_isos.iterrows():
                    _iso_idx_arr = np.where(np.asarray(ids, dtype=str) == _irow['isoform_id'])[0]
                    if len(_iso_idx_arr) == 0:
                        continue
                    _iso_idx = _iso_idx_arr[0]
                    _go_scores = sm[_iso_idx]
                    _safe_irow_key = _irow['isoform_id'].replace('/', '_').replace('.', '_')
                    _fig, _ = _go_profile_fig(
                        go, gnames, _go_scores,
                        f"PRISM: {_irow['isoform_id']} (max={_irow['max_score']:.3f})",
                        thr, top_n=15,
                    )
                    st.plotly_chart(_fig, use_container_width=True,
                                    key=f"bisect_s1_go_{_gene}_{_safe_ct_key}_{_safe_irow_key}")

st.divider()
st.download_button(
    "⬇️ BISECT PASS cases 다운로드 (CSV)",
    _bdf_filt.to_csv(index=False).encode('utf-8'),
    "bisect_pass_cases.csv", "text/csv",
    use_container_width=True,
)

st.divider()
st.markdown("""
<div style='background:linear-gradient(90deg,#f5f3ff,#ede9fe);border-radius:10px;
padding:16px 24px;border-left:4px solid #7c3aed;margin-top:16px'>
<b>관련 분석 페이지:</b><br>
<span style='color:#374151;font-size:0.9rem'>
🎯 <b>Target Analysis</b>: 유전자 Quick Card + DTU ΔIF 차트 + 바스켓 비교<br>
🔬 <b>시나리오 & 분석</b> → BISECT Cases 탭: S1 교차 분석, 전체 이소폼 DTU 차트 포함
</span>
</div>
""", unsafe_allow_html=True)
