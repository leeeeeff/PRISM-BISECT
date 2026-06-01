"""PRISM Interactive Analysis Tool — Streamlit entry point.

Run with:
    ./run_app.sh          (로컬, UMAP 정상)
    streamlit run prism_app/app/main.py   (Community Cloud)
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

from prism_app.app.components.sidebar import render_sidebar
from prism_app.app.components.interpretation import render_onboarding_guide, render_data_context_banner

if 'cfg' not in st.session_state:
    st.session_state.cfg = None

cfg = render_sidebar()
st.session_state.cfg = cfg

# ── Hero 헤더 (항상 표시) ─────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🧬 PRISM <span style='color:#2a9d8f'>+</span> BISECT</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:1.15rem; color:#555; margin-top:4px;'>"
    "롱리드 싱글셀 아이소폼의 <b>GO 기능을 예측</b>하고, "
    "DTU와 결합해 <b>기능 스위치를 4가지 시나리오로 분류</b>하는 인터랙티브 분석 도구"
    "</p>",
    unsafe_allow_html=True,
)

# ── 핵심 성능 지표 (항상 표시) ────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Macro AUPRC (근육)", "0.7022",   "랜덤 기준(0.5) 대비 +40%")
m2.metric("Zero-shot 뇌 전이",  "0.5998",   "학습 없이 다른 조직 적용")
m3.metric("신규 기능 발견 (뇌)", "541개",    "기존 주석 없는 Novel 아이소폼")
m4.metric("BISECT PASS cases",  "84개",     "15개 모듈 × 증거 등급화")

st.divider()

# ── 분석 모듈 타일 (항상 표시) ────────────────────────────────────────────────
st.subheader("분석 모듈")

# Row 1
row1 = st.columns(3)
_TILES_R1 = [
    ("#f0f9ff", "#0ea5e9", "📊", "Overview",
     "기능 커버리지 · 시나리오 분류 · AUPRC 검증",
     "#e0f2fe", "#0369a1", "Demo 제공"),
    ("#f0fdf4", "#22c55e", "🗺️", "Functional Map",
     "GO 기능 공간 UMAP · 타입별 히트맵 · 유전자 내 비교",
     "#dcfce7", "#15803d", "Demo 제공"),
    ("#fff7ed", "#f59e0b", "🔄", "Condition Analysis",
     "DTU 연계 기능 GAIN/LOSS · GO Enrichment · Sankey",
     "#fef3c7", "#b45309", "⚠️ DTU 파일 필요"),
]
for col, (bg, bd, icon, name, desc, bbg, bc, badge) in zip(row1, _TILES_R1):
    col.markdown(
        f"<div style='background:{bg};border:2px solid {bd};border-radius:12px;"
        f"padding:18px 16px;text-align:center;height:190px'>"
        f"<div style='font-size:2rem'>{icon}</div>"
        f"<b style='font-size:1.05rem;color:#1e293b'>{name}</b><br>"
        f"<span style='font-size:0.78rem;color:#475569;line-height:1.5'>{desc}</span><br><br>"
        f"<span style='background:{bbg};color:{bc};font-size:0.72rem;"
        f"padding:2px 8px;border-radius:12px'>{badge}</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Row 2
row2 = st.columns(3)
_TILES_R2 = [
    ("#fdf4ff", "#a855f7", "🔬", "Individual Analysis",
     "시나리오별 후보 탐색 · 유전자 검색 · 케이스 리포트",
     "#f3e8ff", "#7e22ce", "Demo 제공"),
    ("#fef2f2", "#ef4444", "🔭", "Advanced",
     "조직 간 비교 · 발현 필터 · NMD 위험 스크리닝",
     "#fee2e2", "#b91c1c", "Demo 제공"),
    ("#f8fafc", "#94a3b8", "🚀", "바로 시작하기",
     "왼쪽 사이드바에서 Demo 또는 Upload를 선택하면 분석이 즉시 시작됩니다.",
     "#f1f5f9", "#64748b", "👈 사이드바 선택"),
]
for col, (bg, bd, icon, name, desc, bbg, bc, badge) in zip(row2, _TILES_R2):
    col.markdown(
        f"<div style='background:{bg};border:2px {'dashed' if name == '바로 시작하기' else 'solid'} {bd};"
        f"border-radius:12px;padding:18px 16px;text-align:center;height:190px'>"
        f"<div style='font-size:2rem'>{icon}</div>"
        f"<b style='font-size:1.05rem;color:#1e293b'>{name}</b><br>"
        f"<span style='font-size:0.78rem;color:#475569;line-height:1.5'>{desc}</span><br><br>"
        f"<span style='background:{bbg};color:{bc};font-size:0.72rem;"
        f"padding:2px 8px;border-radius:12px'>{badge}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── 데이터가 없을 때만: 가이드 + stop ─────────────────────────────────────────
if cfg.get('score_matrix') is None:

    # ── 3단계 시작 가이드 ──────────────────────────────────────────────────────
    with st.expander("📋 3단계 시작 가이드", expanded=True):
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("""
<div style='background:#e8f4f8;padding:20px;border-radius:10px;text-align:center'>
<h2 style='color:#2a9d8f;margin:0'>①</h2>
<h4>데이터 선택</h4>
<p style='font-size:0.9rem'>
왼쪽 사이드바에서<br>
<b>Demo</b> (논문 결과 탐색) 또는<br>
<b>Upload</b> (자체 NPY 파일) 선택
</p>
<p style='font-size:0.85rem;color:#666'>
조직 패널(근육 18 GO · 뇌 18 GO · 뇌 확장 73 GO)과<br>
Score 임계값(기본 0.5) 조정
</p>
</div>
            """, unsafe_allow_html=True)

        with s2:
            st.markdown("""
<div style='background:#e8f4f8;padding:20px;border-radius:10px;text-align:center'>
<h2 style='color:#2a9d8f;margin:0'>②</h2>
<h4>Overview 확인</h4>
<p style='font-size:0.9rem'>
📊 Overview 페이지에서<br>
<b>커버리지 · 시나리오 분포 · AUPRC</b>를<br>
한눈에 파악
</p>
<p style='font-size:0.85rem;color:#666'>
S3(신규 기능)와 S1(기능 스위치) 비율이<br>
분석의 핵심 지표입니다
</p>
</div>
            """, unsafe_allow_html=True)

        with s3:
            st.markdown("""
<div style='background:#e8f4f8;padding:20px;border-radius:10px;text-align:center'>
<h2 style='color:#2a9d8f;margin:0'>③</h2>
<h4>후보 아이소폼 탐색</h4>
<p style='font-size:0.9rem'>
🔬 Individual 페이지에서<br>
유전자 이름 검색 →<br>
<b>케이스 리포트 다운로드</b>
</p>
<p style='font-size:0.85rem;color:#666'>
DTU 파일 추가 시 🔄 Condition에서<br>
GAIN/LOSS 기능 변화 분석 가능
</p>
</div>
            """, unsafe_allow_html=True)

    # ── 4. 기존 도구 대비 차별점 ──────────────────────────────────────────────
    with st.expander("🔬 기존 도구 대비 차별점", expanded=False):
        with st.container():
            col_what, col_how = st.columns([1, 1], gap="large")

            with col_what:
                st.subheader("무엇을 해결하나요?")
                st.markdown("""
롱리드 싱글셀 시퀀싱은 수만 개의 **아이소폼(전사체 변이)**을 발견하지만,
대부분에 기능 주석이 없습니다.

기존 방법의 한계:
- **유전자 수준 주석**만 존재 → 아이소폼별 차이 구별 불가
- **Novel isoform**(NIC/NNIC)은 주석 자체가 없음
- **DTU 분석**은 "어떤 아이소폼이 바뀌었나"만 알려주고, "무슨 기능이 바뀌었나"는 말해주지 않음

**PRISM+BISECT**는 이 세 가지를 동시에 해결합니다.
                """)

            with col_how:
                st.subheader("어떻게 해결하나요?")
                st.markdown("""
**PRISM** — GO 기능 예측 모델
- ESM-2 단백질 언어 모델로 아이소폼 고유 서열 특징 추출
- 18~73개 GO BP 기능을 동시에 예측 (0~1 신뢰도 점수)
- 기존 주석이 없는 Novel 아이소폼도 예측 가능

**BISECT** — 기능 스위치 분류 파이프라인
- DTU + PRISM 스코어 결합 → **4-시나리오 분류**
- 15개 독립 증거 모듈로 케이스별 등급 산정
- S1(기능 스위치) · S2(발현 스위치) · S3(신규 기능) · S4(배경)
                """)

        st.markdown("---")
        feat_cols = st.columns(4)
        features = [
            ("Novel 아이소폼 예측", "🆕",
             "NIC·NNIC 등 **기존 Ensembl 주석이 없는** 아이소폼도 GO 기능 예측 가능. "
             "InterPro·Pfam 기반 도구는 알려진 도메인이 있어야 작동함."),
            ("Gene-level bias 극복", "🎯",
             "ESM-2 기반 **아이소폼 고유 서열 특징** 사용. "
             "DIFFUSE(Yao et al., 2022) 등 기존 모델은 유전자 평균 특징에 수렴하는 문제가 있음."),
            ("DTU → 기능 변화 통합", "🔄",
             "satuRn·DEXSeq 등 DTU 결과를 직접 연결해 **어떤 기능이 획득/손실됐는지** 정량화. "
             "IsoformSwitchAnalyzeR는 DTU 탐지만 하고 기능 예측은 별도 과정이 필요함."),
            ("Zero-shot 조직 전이", "🧠",
             "근육 데이터로 학습한 모델을 추가 학습 없이 **뇌에 바로 적용** (AUPRC 0.5998). "
             "조직별 재학습 없이 다양한 데이터셋에 활용 가능."),
        ]
        for col, (title, icon, desc) in zip(feat_cols, features):
            col.markdown(
                f"<div style='background:#f0faf9;border-left:4px solid #2a9d8f;"
                f"padding:14px;border-radius:6px;height:100%'>"
                f"<b style='font-size:1rem'>{icon} {title}</b><br><br>"
                f"<span style='font-size:0.88rem;color:#333'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    # ── 5. 도구 비교표 ─────────────────────────────────────────────────────────
    with st.expander("도구 비교표 (PRISM vs 기존 방법)", expanded=False):
        st.markdown("""
| 기능 | **PRISM+BISECT** | DIFFUSE (2022) | IsoformSwitchAnalyzeR | InterPro/Pfam |
|------|:---:|:---:|:---:|:---:|
| 아이소폼별 기능 예측 | ✅ | △ (유전자 기준) | ❌ | ❌ |
| Novel isoform 지원 | ✅ | ❌ | ❌ | ❌ |
| Gene-level bias 극복 | ✅ | ❌ | — | — |
| DTU 결과 통합 | ✅ | ❌ | ✅ (DTU만) | ❌ |
| 기능 변화 정량화 | ✅ GAIN/LOSS | ❌ | ❌ | ❌ |
| Zero-shot 조직 전이 | ✅ | ❌ | — | △ |
| 인터랙티브 웹 분석 | ✅ | ❌ | ❌ | ❌ |
| 케이스 증거 등급화 | ✅ (BISECT 15모듈) | ❌ | ❌ | ❌ |

> ✅ 완전 지원 · △ 부분 지원 · ❌ 미지원 · — 해당 없음
        """)

    # ── 6. 입력 파일 형식 ──────────────────────────────────────────────────────
    with st.expander("📁 Upload 모드 — 입력 파일 형식", expanded=False):
        st.markdown("""
| 파일 | 필수 | 형식 | 설명 |
|------|:----:|------|------|
| 스코어 매트릭스 | **필수** | `.npy` — shape `(N × GO)` float32 | PRISM 예측 결과 (0~1) |
| 아이소폼 ID | **필수** | `.npy` 또는 `.txt` — 1D 문자열 | 전사체 ID 목록 (행 순서 일치) |
| 유전자 ID | 권장 | `.npy` 또는 `.txt` — 1D 문자열 | 유전자 심볼 또는 ENSG ID |
| 아이소폼 타입 | 선택 | `.npy` 또는 `.txt` — `known`/`nic`/`nnic` | Novel 섹션(A3) 활성화 |
| DTU 결과 | 선택 | `.tsv` 또는 `.csv` | Condition 분석 활성화 |

**스코어 매트릭스 생성 방법** (로컬에서 PRISM 추론 후 업로드):
```bash
conda activate isoform_env
python hMuscle/model/v15d_bp_clean.py --predict \\
    --embeddings your_esm2.npy --output scores.npy
```

**DTU 파일 허용 컬럼 이름** (자동 인식):
- 아이소폼 ID: `isoform_id`, `transcriptID`, `featureID`
- 발현 변화: `delta_IF`, `dIF`, `deltaPSI`, `logFC`
- 유의도: `pvalue`, `padj`, `FDR`, `adj.p.value`
- 조건: `condition`, `comparison`, `contrast`
        """)

    st.info("👈 왼쪽 사이드바에서 **Demo** 또는 **Upload** 를 선택하면 분석이 시작됩니다.", icon="ℹ️")
    render_onboarding_guide()
    st.stop()


# ── 데이터 로드 후: 현재 데이터셋 요약 배너 ──────────────────────────────────
sm  = cfg['score_matrix']
ids = cfg['isoform_ids']
thr = cfg['score_threshold']

n_iso  = len(ids)
n_go   = sm.shape[1]
n_high = int((sm > thr).any(axis=1).sum())

st.divider()
render_data_context_banner(cfg)

col1, col2, col3, col4 = st.columns(4)
col1.metric("로드된 아이소폼",       f"{n_iso:,}")
col2.metric("GO 기능 패널",          f"{n_go}개")
col3.metric(f"Score > {thr} (신뢰)", f"{n_high:,}", f"{100*n_high/n_iso:.1f}%")
col4.metric("분석 모드",             cfg['mode'].capitalize())

st.divider()

st.markdown("""
### 다음 단계로 이동하세요

| 페이지 | 내용 |
|--------|------|
| 📊 **Overview** | 전체 커버리지 · 4-시나리오 분포 · AUPRC 검증 |
| 🗺️ **Functional Map** | GO 기능 공간 UMAP · 타입별 히트맵 · 유전자 내 비교 |
| 🔄 **Condition Analysis** | DTU 연계 GAIN/LOSS · GO enrichment *(DTU 파일 필요)* |
| 🔬 **Individual Analysis** | 시나리오별 후보 · 유전자 검색 · 케이스 리포트 다운로드 |
| 🔭 **Advanced** | 조직 간 비교 · 발현량 필터 · NMD 위험 스크리닝 |

> 사이드바의 **Score 임계값 슬라이더**를 조정하면 모든 페이지의 분류 기준이 동시에 바뀝니다.
""")
