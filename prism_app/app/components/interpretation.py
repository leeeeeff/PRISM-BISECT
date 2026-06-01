"""Dynamic interpretation panels — natural-language summaries of PRISM analysis results.

Each function returns a Markdown string (or calls st.* directly) that explains
what the visualisation means for a non-expert user.
"""
from __future__ import annotations
import numpy as np
import streamlit as st


# ── Shared styling helpers ────────────────────────────────────────────────────

def _info_box(title: str, body: str, icon: str = "💡") -> None:
    st.markdown(
        f"""<div style='background:#f0f7ff;border-left:4px solid #3b82f6;
        padding:14px 18px;border-radius:6px;margin:8px 0 16px 0'>
        <b style='font-size:0.97rem'>{icon} {title}</b><br>
        <span style='font-size:0.88rem;color:#1e293b;line-height:1.6'>{body}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def _tip_box(body: str) -> None:
    st.markdown(
        f"""<div style='background:#f0fdf4;border-left:4px solid #22c55e;
        padding:10px 16px;border-radius:6px;margin:4px 0 12px 0;
        font-size:0.86rem;color:#14532d;line-height:1.6'>{body}</div>""",
        unsafe_allow_html=True,
    )


def _warn_box(body: str) -> None:
    st.markdown(
        f"""<div style='background:#fffbeb;border-left:4px solid #f59e0b;
        padding:10px 16px;border-radius:6px;margin:4px 0 12px 0;
        font-size:0.86rem;color:#78350f;line-height:1.6'>{body}</div>""",
        unsafe_allow_html=True,
    )


# ── Data context banner ───────────────────────────────────────────────────────

def render_data_context_banner(cfg: dict) -> None:
    """Show what data is currently loaded — always visible at top of each page."""
    sm = cfg.get('score_matrix')
    if sm is None:
        return

    tissue_label = {
        'muscle':         '근골격근 (Skeletal Muscle)',
        'brain':          '뇌 — 18 GO 패널 (Brain, zero-shot)',
        'brain_extended': '뇌 — 확장 73 GO 패널 (Brain Novel)',
        'muscle_only':    '근골격근 학습 패널',
    }.get(cfg.get('tissue', ''), cfg.get('tissue', '알 수 없음'))

    n_iso  = sm.shape[0]
    n_go   = sm.shape[1]
    thr    = cfg.get('score_threshold', 0.5)
    n_high = int((sm > thr).any(axis=1).sum())
    mode   = cfg.get('mode', 'demo')
    has_dtu = cfg.get('dtu_df') is not None
    has_types = cfg.get('isoform_types') is not None

    dtu_status = (
        "✅ DTU 결과 로드됨 → Scenario 1·2 활성화"
        if has_dtu else
        "⚠️ DTU 없음 → Scenario 1·2 비활성 (S3·S4만 표시)"
    )

    st.markdown(
        f"""<div style='background:#f8fafc;border:1px solid #e2e8f0;
        border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:0.85rem;color:#475569'>
        <b>📂 현재 데이터</b> &nbsp;|&nbsp;
        모드: <b>{'Demo (논문 결과)' if mode == 'demo' else 'Upload'}</b> &nbsp;|&nbsp;
        조직 패널: <b>{tissue_label}</b> &nbsp;|&nbsp;
        아이소폼: <b>{n_iso:,}개</b> &nbsp;|&nbsp;
        GO 기능: <b>{n_go}개</b> &nbsp;|&nbsp;
        고신뢰 예측: <b>{n_high:,}개</b> ({100*n_high/max(1,n_iso):.1f}%) &nbsp;|&nbsp;
        {dtu_status}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Coverage section ──────────────────────────────────────────────────────────

def render_coverage_interpretation(rep, thr: float, has_types: bool) -> None:
    """Explain coverage report results in plain Korean."""
    pct = rep.pct_with_any_high
    nic_pct = 100 * rep.n_nic / max(1, rep.total_isoforms)
    nnic_pct = 100 * rep.n_nnic / max(1, rep.total_isoforms)
    novel_pct = nic_pct + nnic_pct

    if pct >= 50:
        coverage_comment = f"전체 아이소폼의 절반 이상({pct:.1f}%)에서 고신뢰 GO 예측이 확인되었습니다. <b>예측 커버리지가 우수합니다.</b>"
    elif pct >= 20:
        coverage_comment = f"{pct:.1f}%의 아이소폼에서 Score > {thr} 예측이 있습니다. 임계값을 낮추면 더 많은 아이소폼을 분석할 수 있습니다."
    else:
        coverage_comment = f"{pct:.1f}%만 고신뢰 예측 대상입니다. 임계값을 {thr - 0.1:.1f}로 낮춰보세요."

    novel_comment = ""
    if novel_pct > 5 and has_types:
        novel_comment = (
            f"<br>NIC/NNIC Novel 아이소폼이 <b>{novel_pct:.1f}%</b>를 차지합니다. "
            "이 아이소폼들은 기존 Ensembl 데이터베이스에 없는 전사체로, "
            "InterPro·Pfam·IsoformSwitchAnalyzeR 등 주석 기반 도구로는 기능 예측이 불가능합니다. "
            "<b>PRISM만이 이 아이소폼들을 분석할 수 있습니다.</b>"
        )

    with st.expander("📊 이 결과는 무엇을 의미하나요? (Coverage 해석)", expanded=False):
        _info_box(
            "Coverage Summary 해석",
            coverage_comment + novel_comment,
            icon="📊",
        )
        st.markdown("""
**각 지표 설명:**
- **Known**: Ensembl 데이터베이스에 이미 등록된 아이소폼 (기존 실험으로 확인됨)
- **NIC** (Novel In Catalog): 알려진 스플라이스 사이트 조합이지만 Ensembl에 미등록
- **NNIC** (Novel Not In Catalog): 새로운 스플라이스 사이트가 포함된 완전한 신규 전사체
- **Score > 임계값**: PRISM이 해당 GO 기능을 "고신뢰"로 예측한 아이소폼 수
""")


# ── Scenario classification ───────────────────────────────────────────────────

def render_scenario_interpretation(summ_df, has_dtu: bool) -> None:
    """Explain scenario distribution and DTU absence."""
    counts = dict(zip(summ_df['scenario'], summ_df['count']))
    total  = summ_df['count'].sum()
    s1, s2, s3, s4 = (counts.get(i, 0) for i in [1, 2, 3, 4])
    s3_pct = 100 * s3 / max(1, total)

    with st.expander("📊 시나리오 분류 결과 해석 + DTU 미포함 이유", expanded=True):
        if not has_dtu:
            _warn_box(
                "<b>왜 Scenario 1·2가 비어있나요?</b><br><br>"
                "현재 데이터에는 <b>DTU(Differential Transcript Usage) 결과가 포함되지 않았습니다.</b> "
                "DTU 분석은 두 조건(예: 질병 vs. 정상) 사이에서 각 아이소폼의 사용 비율이 "
                "통계적으로 달라졌는지를 검정합니다.<br><br>"
                "<b>Scenario 1 (기능 스위치):</b> DTU+ 아이소폼 중 PRISM이 신규 GO 기능을 예측한 경우 → "
                "조건에 따라 세포가 실제로 다른 기능을 수행하는 최우선 후보<br>"
                "<b>Scenario 2 (발현 스위치):</b> DTU+이지만 기능 차이 없음 → 발현량만 변화<br><br>"
                "✅ satuRn, DEXSeq, IsoformSwitchAnalyzeR 등 DTU 분석 결과를 "
                "<b>Upload 모드에서 업로드</b>하면 즉시 활성화됩니다."
            )

        body = (
            f"<b>S3 (신규 기능) {s3:,}개 ({s3_pct:.1f}%)</b> — "
            "DTU와 무관하게 PRISM이 기존 주석에 없는 GO 기능을 예측한 아이소폼입니다. "
            "같은 유전자의 다른 아이소폼과 기능이 다를 가능성이 높으며, "
            "논문에서 뇌 데이터로 541개가 이 범주에서 발견됐습니다.<br><br>"
        )
        if has_dtu and s1 > 0:
            s1_pct = 100 * s1 / max(1, total)
            body += (
                f"<b>S1 (기능 스위치) {s1:,}개 ({s1_pct:.1f}%)</b> — "
                "가장 주목해야 할 아이소폼입니다. 조건 변화에 따라 전사체 비율이 바뀌면서 "
                "동시에 기능도 달라진 것으로 예측됩니다. 실험 검증의 최우선 후보입니다."
            )

        _info_box("시나리오 분포 해석", body, icon="🔬")

        st.markdown("""
**4-시나리오 정의:**

| 시나리오 | DTU | PRISM 신규 GO | 의미 | 우선순위 |
|----------|:---:|:---:|------|:---:|
| S1 기능 스위치 | ✅ | ✅ | 조건에 따라 기능이 바뀌는 아이소폼 | ⭐⭐⭐ |
| S2 발현 스위치 | ✅ | ❌ | 발현량만 바뀜, 기능 차이 없음 | ⭐⭐ |
| S3 신규 기능 | ❌ | ✅ | 항상 발현, 기존 주석에 없는 기능 | ⭐⭐⭐ |
| S4 배경 | ❌ | ❌ | 분석 우선순위 낮음 | — |
""")


# ── Novel isoform section ─────────────────────────────────────────────────────

def render_novel_interpretation(novel_rep) -> None:
    total = novel_rep.total_novel
    with_high = novel_rep.n_novel_with_any_high
    pct = novel_rep.pct_novel_with_high

    body = (
        f"총 <b>{total:,}개</b>의 Novel 아이소폼(NIC+NNIC) 중 "
        f"<b>{with_high:,}개 ({pct:.1f}%)</b>에서 기존 Ensembl 주석에 없는 "
        "GO 기능이 PRISM에 의해 예측되었습니다.<br><br>"
        "이 아이소폼들은 데이터베이스에 등재되지 않은 전사체이기 때문에, "
        "종래의 도구(InterPro 도메인 기반, 서열 상동성 기반)로는 기능 예측이 불가능합니다. "
        "PRISM은 ESM-2 단백질 언어 모델의 임베딩을 사용해 서열 자체로부터 "
        "기능을 추론합니다. 아래 표는 GO 기능별로 Novel 아이소폼이 몇 개나 "
        "해당 기능을 수행할 것으로 예측됐는지 보여줍니다."
    )

    with st.expander("📊 Novel 아이소폼 예측 결과 해석", expanded=False):
        _info_box("Novel Isoform Function Predictions 해석", body, icon="🆕")


# ── AUPRC validation ──────────────────────────────────────────────────────────

def render_auprc_interpretation(val_rep) -> None:
    macro = val_rep.macro_auprc
    ci_lo, ci_hi = val_rep.macro_auprc_ci
    n_terms = val_rep.n_go_terms

    if macro >= 0.65:
        grade = "✅ 우수 (0.65 이상)"
        grade_color = "#14532d"
    elif macro >= 0.55:
        grade = "🔶 양호 (0.55~0.65)"
        grade_color = "#78350f"
    else:
        grade = "⚠️ 기준선 근접 (0.55 미만)"
        grade_color = "#7f1d1d"

    body = (
        f"<b>AUPRC란?</b> — Precision-Recall 곡선의 면적. "
        "GO 어노테이션처럼 양성 예시가 희소한 데이터에서 예측 성능을 평가하는 가장 적합한 지표입니다. "
        "AUROC는 불균형 데이터에서 과대평가되므로, 논문에서는 AUPRC를 주 지표로 사용합니다.<br><br>"
        f"<b>무작위 분류기 기준선</b>: 각 GO term의 양성 비율 (≈ 0.10–0.30). "
        f"일반적으로 <b>0.5를 넘으면 모델이 유용한 예측을 한다</b>고 봅니다.<br><br>"
        f"<b>현재 Macro AUPRC</b>: {macro:.4f} (95% CI: {ci_lo:.4f}–{ci_hi:.4f}) — "
        f"<span style='color:{grade_color}'>{grade}</span><br>"
        f"{n_terms}개 GO term에서 평가. "
        "근육 데이터 레퍼런스: Macro AUPRC 0.7022 (Lee et al. 2026, §3.3)."
    )

    with st.expander("📊 AUPRC 검증 결과 해석", expanded=False):
        _info_box("Known Annotation Validation — AUPRC 해석", body, icon="📏")
        _tip_box(
            "💡 <b>막대 그래프 읽는 법</b>: 각 막대는 GO term별 AUPRC를 나타냅니다. "
            "점선(random 0.5)을 넘는 GO term일수록 PRISM이 해당 기능을 "
            "신뢰성 있게 예측하고 있습니다. 근육 수축(GO:0006936)·미토콘드리아 전자 전달(GO:0022900) 등 "
            "조직 특이적 기능에서 높은 AUPRC가 나타납니다."
        )


# ── UMAP ─────────────────────────────────────────────────────────────────────

def render_umap_interpretation(embed_method: str, n_total: int, sampled: int,
                               color_by: str) -> None:
    body = (
        f"UMAP(Uniform Manifold Approximation and Projection)은 각 아이소폼의 "
        f"GO 스코어 벡터({sampled:,}차원)를 2D 공간으로 투영합니다. "
        "<b>기능이 유사한 아이소폼들이 가까이 모입니다.</b><br><br>"
    )

    if color_by == 'isoform_type':
        body += (
            "현재 <b>아이소폼 구조 타입</b>으로 색칠되어 있습니다. "
            "Known(파란색)과 NIC/NNIC(초록/빨간색)가 뚜렷이 구분된 클러스터를 형성한다면, "
            "Novel 아이소폼이 Known과는 다른 기능을 예측받고 있다는 의미입니다."
        )
    elif color_by == 'scenario':
        body += (
            "현재 <b>4-시나리오 분류</b>로 색칠되어 있습니다. "
            "S3(신규 기능, 초록)가 공간적으로 어디에 위치하는지 확인하세요. "
            "특정 클러스터에 몰려있다면 해당 GO 기능이 Novel 아이소폼에서 집중 발현되는 것입니다."
        )
    elif color_by == 'max_score':
        body += (
            "현재 <b>최고 GO 스코어</b>로 색칠되어 있습니다. "
            "밝은 색(높은 스코어) 영역의 클러스터가 PRISM이 가장 확신하는 기능 예측 군집입니다."
        )

    if n_total > sampled:
        body += (
            f"<br><br>⚠️ 전체 {n_total:,}개 중 속도를 위해 {sampled:,}개를 무작위 샘플링했습니다. "
            "전체 데이터를 보려면 포인트 수 제한을 높이세요."
        )

    if embed_method == 't-SNE':
        body += (
            "<br><br>현재 환경에서 umap-learn이 충돌해 <b>t-SNE</b>로 대체됐습니다. "
            "결과 품질은 유사하지만, <code>./run_app.sh</code>로 실행하면 UMAP이 정상 동작합니다."
        )

    with st.expander("📊 UMAP 투영 해석", expanded=False):
        _info_box("Functional Map — UMAP 읽는 법", body, icon="🗺️")


# ── Within-gene comparison ────────────────────────────────────────────────────

def render_within_gene_interpretation(gene_name: str, n_isoforms: int) -> None:
    body = (
        f"<b>{gene_name}</b> 유전자의 <b>{n_isoforms}개</b> 아이소폼 간 GO 스코어 차이를 보여줍니다. "
        "같은 유전자에서 유래했지만 <b>아이소폼마다 예측 기능 프로파일이 다를 수 있습니다.</b><br><br>"
        "막대/히트맵에서 특정 아이소폼이 다른 아이소폼과 뚜렷이 구별된다면, "
        "그 아이소폼이 유전자의 '기능 스위치' 후보일 가능성이 높습니다. "
        "DTU 결과와 함께 보면 Scenario 1 판단 근거가 됩니다."
    )
    _info_box(f"{gene_name} 아이소폼 비교 해석", body, icon="🔬")


# ── Condition analysis ────────────────────────────────────────────────────────

def render_condition_interpretation(gain: int, loss: int, neutral: int) -> None:
    total = gain + loss + neutral
    if total == 0:
        return

    gain_pct  = 100 * gain  / total
    loss_pct  = 100 * loss  / total

    body = (
        f"DTU 아이소폼 {total:,}개를 분석한 결과: "
        f"<b>GAIN {gain:,}개 ({gain_pct:.1f}%)</b> — 조건 B에서 상위 아이소폼이 하위 아이소폼보다 "
        "더 높은 PRISM 기능 스코어를 가짐(기능 획득).<br>"
        f"<b>LOSS {loss:,}개 ({loss_pct:.1f}%)</b> — 조건 B에서 기능 스코어 감소(기능 상실).<br>"
        f"<b>NEUTRAL {neutral:,}개</b> — 기능 스코어 차이 미미.<br><br>"
        "GAIN/LOSS 비율이 높을수록 두 조건 간 기능적 차이가 크다는 의미입니다. "
        "GO Enrichment 탭에서 어떤 생물학적 과정이 enriched됐는지 확인하세요."
    )

    _info_box("Functional Consequence Matrix 해석", body, icon="🔄")


# ── First-time user onboarding ────────────────────────────────────────────────

def render_onboarding_guide() -> None:
    """Render a comprehensive first-time user guide on the main page."""
    st.markdown("---")
    st.subheader("🚀 처음 사용하시나요? — 5분 시작 가이드")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
#### 지금 보고 있는 데이터는?

왼쪽 사이드바에서 **Demo** 모드가 선택되어 있다면,
논문(Lee et al. 2026)의 근골격근(Skeletal Muscle) 데이터가 자동 로드됩니다:

| 항목 | 값 |
|------|-----|
| 데이터 출처 | Long-read single-cell (근골격근) |
| 아이소폼 수 | 36,748개 |
| GO 기능 패널 | BP 18개 term |
| PRISM 스코어 | 사전 계산됨 (ESM-2 → MLP) |
| DTU 결과 | **미포함** (아래 참고) |
| AUPRC | 0.7022 (논문 검증 수치) |

뇌 데이터로 바꾸려면: 사이드바 → "Brain — 18-term Panel" 선택
""")

    with col2:
        st.markdown("""
#### 왜 Scenario 1·2가 비어있나요?

Demo 데이터에는 **DTU(Differential Transcript Usage) 결과가 포함되지 않습니다.**

DTU 분석은 두 조건(예: AD vs. 정상) 사이에서 각 아이소폼의
사용 비율이 바뀌었는지 검정합니다.

- satuRn, DEXSeq, IsoformSwitchAnalyzeR 등의 도구 필요
- Demo에는 단일 조건 데이터만 있으므로 DTU 불가능
- **Upload 모드**에서 DTU `.tsv` 파일을 올리면 즉시 활성화

> S3 (신규 기능)은 DTU 없이도 동작합니다. 논문의 뇌 541개 케이스가 S3입니다.
""")

    st.divider()

    st.markdown("#### 순서대로 분석하는 법")

    s1c, s2c, s3c, s4c, s5c = st.columns(5)
    steps = [
        ("①", "사이드바", "Demo 또는 Upload 선택\n\n조직 패널 선택\n(근육 18 / 뇌 73 GO)", "#e0f2fe"),
        ("②", "Overview", "A1: 커버리지 확인\nD1: 시나리오 분포\nA2: AUPRC 검증", "#f0fdf4"),
        ("③", "Functional Map", "UMAP으로 기능 공간 탐색\n비슷한 기능 아이소폼이\n같은 클러스터에 모임", "#fdf4ff"),
        ("④", "Individual", "유전자 이름 검색\nS3/S1 후보 케이스 리포트\nCSV·Markdown 다운로드", "#fff7ed"),
        ("⑤", "Condition", "DTU 파일 있을 때만\nGAIN/LOSS 분석\nGO Enrichment 확인", "#fef2f2"),
    ]
    for col, (num, title, desc, bg) in zip([s1c, s2c, s3c, s4c, s5c], steps):
        col.markdown(
            f"<div style='background:{bg};padding:12px;border-radius:8px;text-align:center;height:160px'>"
            f"<h2 style='margin:0;color:#374151'>{num}</h2>"
            f"<b style='font-size:0.9rem'>{title}</b><br>"
            f"<span style='font-size:0.78rem;color:#4b5563;white-space:pre-line'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔍 핵심 용어 사전", expanded=False):
        st.markdown("""
| 용어 | 설명 |
|------|------|
| **아이소폼 (Isoform)** | 하나의 유전자에서 다른 방식으로 스플라이싱된 전사체. 같은 유전자라도 단백질 기능이 다를 수 있음 |
| **PRISM 스코어** | 아이소폼이 특정 GO 기능을 수행할 확률 (0~1). ESM-2 임베딩 기반 MLP 예측 |
| **GO term (Gene Ontology)** | 생물학적 기능의 표준 분류 체계. "GO:0006936 muscle contraction" 등 |
| **AUPRC** | 예측 정확도 지표. 양성 예시 희소 데이터에 적합. 0.5 = 랜덤, 0.7 이상 = 우수 |
| **DTU** | 두 조건 간 아이소폼 사용 비율 변화. satuRn·DEXSeq 등으로 검정 |
| **NIC** | Novel In Catalog — 알려진 스플라이스 사이트 조합이지만 Ensembl 미등록 |
| **NNIC** | Novel Not In Catalog — 완전히 새로운 스플라이스 사이트 포함 |
| **Scenario 1 (S1)** | DTU+ & 신규 GO 예측 → 조건 의존적 기능 변화. 최우선 실험 후보 |
| **Scenario 3 (S3)** | DTU 없음 & 신규 GO 예측 → 구성적 신규 기능 아이소폼 |
| **Novel GO** | PRISM이 예측했지만 기존 UniProt 어노테이션에 없는 GO 기능 |
""")
