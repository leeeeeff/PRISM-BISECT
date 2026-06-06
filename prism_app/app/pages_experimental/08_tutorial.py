"""Page 08 — Tutorial & Reference Guide.

Covers: PRISM score interpretation, BISECT tier system, DTU p-value context,
example case (KIF21B), Upload mode guide, and the 18 GO BP term glossary.
"""
import sys
from pathlib import Path
_root = str(Path(__file__).parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st

st.markdown(
    """
    <div style='background:linear-gradient(135deg,#0f2942 0%,#1a4a6e 60%,#0f4c75 100%);
    border-radius:14px;padding:36px 48px 28px;margin-bottom:24px'>
    <h1 style='color:white;margin:0;font-size:2rem'>&#128218; PRISM+BISECT — Tutorial &amp; Reference</h1>
    <p style='color:#93c5fd;margin:8px 0 0;font-size:1rem'>
    Score interpretation &middot; Tier system &middot; Upload guide &middot; GO term glossary
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "이 페이지는 PRISM 점수 해석, BISECT 티어 기준, 업로드 형식, "
    "18개 GO 용어 설명을 한 곳에 정리한 참고 자료입니다. "
    "분석 중 궁금한 항목을 펼쳐 확인하세요."
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRISM Score Interpretation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("1. PRISM 점수 해석")

with st.expander("PRISM 점수란 무엇인가요? (0–1 범위, GO term별 신뢰도)", expanded=True):
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(
            """
**PRISM (Protein-Isoform Resolution via Intrinsic Sequence Modeling)** 은
각 아이소폼이 특정 GO Biological Process(BP) 기능을 가질 확률을 **0.0–1.0** 스케일로 예측합니다.

#### 점수 구간 해석

| 범위 | 해석 | 권장 활용 |
|------|------|-----------|
| **≥ 0.70** | 고신뢰 기능 예측 | 실험 검증 우선 대상 |
| **0.50 – 0.69** | 중간 신뢰도 — 기능 가능성 있음 | 추가 증거 필요 |
| **0.40 – 0.49** | 저신뢰 경계 — BISECT Tier 1 최소 기준 | 맥락과 함께 해석 |
| **< 0.40** | 예측 불가 / 기능 없을 가능성 높음 | 단독으로 결론 내리지 않음 |

#### 핵심 원칙

- **점수는 GO term 별로 독립적입니다.**
  한 아이소폼이 `GO:0007018 Microtubule-based movement` 에서 0.97점이라 해도
  `GO:0006096 Glycolytic process` 에서는 0.01점일 수 있습니다.

- **0.5 = 경계선이 아닙니다.**
  PRISM은 sparse label 데이터로 훈련되어 캘리브레이션이 완전하지 않습니다.
  절대값보다는 **같은 유전자 내 아이소폼 간 상대적 차이(delta)** 를 우선 해석하세요.

- **AUPRC 기준 성능**: Skeletal muscle (훈련 조직) Macro AUPRC = **0.7022**,
  Brain zero-shot Macro AUPRC = **0.5998** (18-term) / **0.672** (41-term 확장 panel).
            """
        )

    with col_b:
        st.markdown(
            """
<div style='background:#f0f7ff;border-left:4px solid #3b82f6;
padding:14px 18px;border-radius:6px;margin-bottom:12px'>
<b style='font-size:0.95rem'>&#128161; delta 해석 예시</b><br><br>
<span style='font-size:0.87rem;color:#1e293b;line-height:1.8'>
<b>CT 아이소폼</b><br>
microtubule-based movement: <b>0.975</b><br><br>
<b>AD 아이소폼</b><br>
microtubule-based movement: <b>0.105</b><br><br>
<b>delta = &minus;0.870</b><br>
&#8594; AD에서 운동 기능 <b>완전 소실</b>
</span>
</div>

<div style='background:#f0fdf4;border-left:4px solid #22c55e;
padding:10px 16px;border-radius:6px;font-size:0.86rem;color:#14532d;line-height:1.6'>
<b>Tip</b>: BISECT Cases 페이지의
<code>prism_ct_max_score</code> /
<code>prism_ad_max_score</code> 컬럼에서
각 아이소폼의 최고 점수를 확인할 수 있습니다.
</div>
            """,
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BISECT Tier System
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("2. BISECT 티어 시스템")

with st.expander("Tier 1 / Tier 2 / Tier 3 기준 및 의미", expanded=True):
    st.markdown(
        """
**BISECT (Biological Isoform-Switch Evidence Characterization Tool)** 은
PRISM 점수 + DTU(Differential Transcript Usage) 통계 + 도메인 증거를 결합하여
각 케이스를 세 티어로 분류합니다.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;
padding:16px 18px;height:100%'>
<h4 style='color:#dc2626;margin:0 0 8px'>&#128308; Tier 1 &mdash; AD 기능 획득</h4>
<b style='font-size:0.85rem'>조건:</b>
<ul style='font-size:0.85rem;margin:6px 0;padding-left:18px;line-height:1.7'>
<li>AD 아이소폼 <code>max(PRISM)</code> &ge; <b>0.40</b></li>
<li><code>delta_max</code> &ge; <b>+0.15</b></li>
<li>(DTU 검증 있는 경우 우선순위 상승)</li>
</ul>
<b style='font-size:0.85rem'>의미:</b><br>
<span style='font-size:0.85rem;line-height:1.6'>
알츠하이머 조건에서 <i>새로운 기능을 획득</i>한 아이소폼.
CT에서는 낮은 점수였지만 AD에서 특정 GO 기능 점수가
유의미하게 상승합니다.
</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div style='background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
padding:16px 18px;height:100%'>
<h4 style='color:#d97706;margin:0 0 8px'>&#128993; Tier 2 &mdash; CT 기능 소실</h4>
<b style='font-size:0.85rem'>조건:</b>
<ul style='font-size:0.85rem;margin:6px 0;padding-left:18px;line-height:1.7'>
<li>CT 아이소폼 <code>max(PRISM)</code> &ge; <b>0.40</b></li>
<li><code>delta_min</code> &le; <b>&minus;0.15</b> (기능 소실)</li>
<li>AD 아이소폼이 해당 기능을 회복하지 못함</li>
</ul>
<b style='font-size:0.85rem'>의미:</b><br>
<span style='font-size:0.85rem;line-height:1.6'>
정상 조건에서는 존재하던 기능이 AD 아이소폼으로 교체되면서
<i>소실</i>된 케이스. 보호적 기능의 상실로 해석할 수 있습니다.
</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
<div style='background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;
padding:16px 18px;height:100%'>
<h4 style='color:#64748b;margin:0 0 8px'>&#9898; Tier 3 &mdash; 저신뢰 / 미결정</h4>
<b style='font-size:0.85rem'>조건:</b>
<ul style='font-size:0.85rem;margin:6px 0;padding-left:18px;line-height:1.7'>
<li>Tier 1/2 기준 미충족</li>
<li>PRISM 점수 모두 &lt; 0.40</li>
<li>또는 delta 절댓값 &lt; 0.15</li>
</ul>
<b style='font-size:0.85rem'>의미:</b><br>
<span style='font-size:0.85rem;line-height:1.6'>
예측 신뢰도가 낮아 결론을 내리기 어려운 케이스.
단독으로 기능적 해석 금지. 다른 증거(도메인, NMD, PPI)와
함께 검토 권장.
</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
<div style='background:#fffbeb;border-left:4px solid #f59e0b;
padding:10px 16px;border-radius:6px;font-size:0.86rem;color:#78350f;line-height:1.6'>
<b>주의:</b> Tier 분류는 PRISM 점수만으로 결정되지 않습니다.
<code>stage2_pass</code> 컬럼이 <code>True</code>인 케이스만 BISECT 2단계(도메인 + PPI + 보존도) 검증을 통과한 것입니다.
Tier 1/2이더라도 <code>stage2_pass=False</code>이면 예비 케이스로 처리하세요.
</div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DTU p-value
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("3. DTU p-value 해석 안내")

with st.expander("DTU p-value가 있는 케이스 vs. 단일조건 SRA 케이스"):
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            """
<div style='background:#f0f7ff;border-left:4px solid #3b82f6;
padding:14px 18px;border-radius:6px'>
<b>&#129504; 뇌 조직 26개 케이스 (DTU 검증 있음)</b><br><br>
<span style='font-size:0.87rem;color:#1e293b;line-height:1.8'>
&bull; ROSMAP AD 코호트 데이터 기반<br>
&bull; AD vs. CT (Cognitively Normal) 비교 설계<br>
&bull; <b>DTU p-value</b>: satuRn 알고리즘으로 계산된
  아이소폼 비율 변화의 통계적 유의성<br>
&bull; <code>dtu_note = &quot;dtu_tested&quot;</code> 로 표시됨<br>
&bull; 기준: FDR-adjusted p &lt; 0.05 권장<br><br>
이 케이스들은 PRISM 점수 + DTU 통계의
<b>이중 검증</b>을 거친 가장 신뢰도 높은 결과입니다.
</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            """
<div style='background:#f8fafc;border-left:4px solid #94a3b8;
padding:14px 18px;border-radius:6px'>
<b>&#128300; SRA 단일조건 57개 케이스</b><br><br>
<span style='font-size:0.87rem;color:#1e293b;line-height:1.8'>
&bull; NCBI SRA 공개 데이터 (다양한 뇌 조직/세포 유형)<br>
&bull; <b>단일조건 (AD/CT 비교 없음)</b><br>
&bull; DTU p-value 없음 &mdash; 앱에서 <b>&quot;단일조건&quot;</b> 표시<br>
&bull; <code>dtu_note = &quot;single_condition&quot;</code> 으로 표시됨<br><br>
이 케이스들은 PRISM 점수와 도메인/PPI 증거로만
기능적 함의를 평가합니다.<br>
AD vs. CT 비교 결론 도출 시 주의가 필요합니다.
</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<br>
<div style='background:#f0fdf4;border-left:4px solid #22c55e;
padding:10px 16px;border-radius:6px;font-size:0.86rem;color:#14532d;line-height:1.6'>
<b>Tip</b>: BISECT Cases 페이지 상단 필터에서 <b>"DTU 검증 포함만"</b> 옵션을 켜면
뇌 조직 26개 케이스만 표시됩니다. 논문 재현 결과를 확인할 때 권장합니다.
</div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Example Case: KIF21B
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("4. 예시 케이스 — KIF21B (Motor Polarity Reversal)")

with st.expander(
    "KIF21B: 흥분성 뉴런에서 미세소관 운동 기능 소실 (DTU p = 3.81 × 10⁻⁶)",
    expanded=True,
):
    st.markdown(
        """
**KIF21B** 는 Kinesin superfamily motor protein으로, 수상돌기 내 미세소관 극성 조절 및
시냅스 강도 조절에 관여하는 유전자입니다.
BISECT는 이 유전자의 아이소폼 전환을 흥분성 뉴런에서 검출하였습니다.
        """
    )

    col_case1, col_case2 = st.columns(2)

    with col_case1:
        st.markdown(
            """
#### CT 아이소폼 (정상 조건)
`transcript293004.chr1.nic` — NIC novel isoform

| GO term | PRISM 점수 |
|---------|-----------|
| microtubule-based movement | **0.975** |
| microtubule cytoskeleton organization | 0.391 |
| neuron projection development | 0.186 |
| neuron differentiation | 0.170 |

도메인: **Kinesin** + **Microtub_bd** + DUF5082
→ 정상 운동 단백질 기능 유지
            """
        )

    with col_case2:
        st.markdown(
            """
#### AD 아이소폼 (알츠하이머 조건)
`transcript292978.chr1.nnic` — NNIC novel isoform

| GO term | PRISM 점수 |
|---------|-----------|
| regulation of neuron differentiation | 0.191 |
| autophagy | 0.167 |
| chemical synaptic transmission | 0.158 |
| microtubule-based movement | **0.105** |

도메인 획득: **ANAPC4_WD40** + NBCH_WD40 + Nup160 + WD40
→ Kinesin 도메인 소실, WD40 repeat 도메인으로 교체
            """
        )

    st.markdown(
        """
<div style='background:#fef2f2;border-left:4px solid #dc2626;
padding:14px 18px;border-radius:6px;margin:12px 0'>
<b style='font-size:0.95rem'>핵심 발견</b><br>
<span style='font-size:0.88rem;color:#1e293b;line-height:1.7'>
&bull; <b>delta(microtubule-based movement) = &minus;0.870</b> &mdash; 운동 기능 거의 완전 소실<br>
&bull; DTU p-value = <b>3.81 &times; 10<sup>&minus;6</sup></b> (흥분성 뉴런, satuRn)<br>
&bull; APA class: <b>major_apa</b>, 3&prime; 말단 28,492 bp 연장<br>
&bull; PPI: <b>TRIM3</b> (String score 765) &mdash; AD 관련 E3 ubiquitin ligase와의 상호작용<br>
&bull; 스플라이싱 조절인자: SRSF5 (&darr;), RBFOX1 (&darr;), HNRNPK (&uarr;)<br>
&bull; 보존도: AD 아이소폼 PhyloP = 1.954 (highly conserved)
</span>
</div>

<div style='background:#f8fafc;border-left:4px solid #94a3b8;
padding:10px 16px;border-radius:6px;font-size:0.86rem;color:#475569;line-height:1.6'>
<b>BISECT Tier:</b> <code>tier2_functional_loss</code> &mdash; CT 아이소폼에서 확립된 운동 기능이
AD 조건에서 구조적으로 다른 아이소폼으로 교체되며 소실됨.
</div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Upload Guide
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("5. Upload 모드 — 내 데이터 분석 가이드")

with st.expander("CSV / NPZ 형식 및 업로드 방법"):
    st.markdown(
        """
사이드바에서 **Upload (내 데이터)** 모드를 선택하면 자체 데이터로 PRISM+BISECT 분석을 실행할 수 있습니다.
        """
    )

    tab_score, tab_bisect, tab_faq = st.tabs(["PRISM 점수 행렬", "BISECT 입력 CSV", "FAQ"])

    with tab_score:
        st.markdown(
            """
#### PRISM 점수 행렬 (score_matrix.npz / .csv)

PRISM 모델이 출력한 예측값을 그대로 업로드합니다.

**옵션 A — NPZ 형식** (권장):
```python
import numpy as np
# score_matrix: shape (n_isoforms, n_go_terms)
# isoform_ids:  list of transcript IDs
# gene_ids:     list of gene symbols (same length as isoform_ids)
# go_terms:     list of GO:xxxxxxx strings
np.savez('my_scores.npz',
         score_matrix=score_matrix,
         isoform_ids=isoform_ids,
         gene_ids=gene_ids,
         go_terms=go_terms)
```

**옵션 B — CSV 형식**:

| isoform_id | gene_id | GO:0007204 | GO:0006096 | ... |
|------------|---------|-----------|-----------|-----|
| ENST000001 | MYH7 | 0.823 | 0.041 | ... |
| ENST000002 | MYH7 | 0.102 | 0.651 | ... |

- 첫 번째 컬럼: `isoform_id` (전사체 ID)
- 두 번째 컬럼: `gene_id` (유전자 심볼)
- 이후 컬럼: GO term ID를 헤더로 사용, 값은 0.0–1.0 실수
            """
        )

    with tab_bisect:
        st.markdown(
            """
#### BISECT 아이소폼 쌍 CSV

BISECT Cases 페이지 및 Condition Analysis 페이지에서 아이소폼 간 비교를 실행하려면
아이소폼 쌍 정보가 필요합니다.

**필수 컬럼:**

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| `ct_transcript_id` | 정상(CT) 조건 아이소폼 ID | `ENST00000123456` |
| `ad_transcript_id` | 질병(AD) 조건 아이소폼 ID | `ENST00000789012` |
| `delta` | 발현 비율 변화 (AD − CT) | `-0.45` |
| `gene` | 유전자 심볼 | `KIF21B` |

**선택 컬럼 (있으면 자동 반영):**

| 컬럼명 | 설명 |
|--------|------|
| `dtu_p` | DTU p-value (satuRn 등) |
| `cell_type` | 세포 유형 |
| `domains_lost` / `domains_gained` | Pfam 도메인 변화 (`;` 구분) |
| `ppi_verdict` | PPI 검증 결과 (`SUPPORTED` / `NOT_SUPPORTED`) |
| `mechanism_type` | `alternative_splicing` / `alt_promoter` / `alt_polyA` |

**CSV 예시:**
```csv
gene,ct_transcript_id,ad_transcript_id,delta,cell_type,dtu_p
KIF21B,transcript293004,transcript292978,-0.855,Excitatory,3.81e-06
NDUFS4,transcript001.nnic,transcript002.nic,-0.621,Oligodendrocyte,1.2e-04
```
            """
        )

    with tab_faq:
        st.markdown(
            """
#### 자주 묻는 질문

**Q. 내 데이터에 GO term이 18개가 아니라 더 많습니다.**
A. PRISM은 임의의 GO term 집합을 지원합니다. 업로드 CSV의 컬럼 헤더가 `GO:xxxxxxx` 형식이면 자동 인식됩니다. Brain Expanded (41GO), Brain Extended (73GO) 패널도 사용 가능합니다.

**Q. 아이소폼 ID가 Ensembl ENST 형식이 아니라 커스텀 ID입니다.**
A. PRISM은 ID 형식을 강요하지 않습니다. 단, BISECT 도메인/PPI 검증은 Ensembl 기반이므로 커스텀 ID인 경우 해당 열이 공백으로 표시됩니다.

**Q. DTU 결과가 없어도 BISECT를 사용할 수 있나요?**
A. 네. `dtu_p` 컬럼을 생략하면 PRISM 점수와 delta 값만으로 Tier 분류가 이루어집니다. 이 경우 "단일조건" 모드로 처리됩니다.

**Q. 업로드 파일 크기 제한이 있나요?**
A. Streamlit 기본 200 MB 제한이 적용됩니다. 아이소폼 수가 수만 개를 초과하는 경우 NPZ 형식 사용을 권장합니다.

**Q. Brain 672GO 패널은 어떻게 사용하나요?**
A. 사이드바에서 **Brain — Full Module (672 GO)** 을 선택하세요. 이 패널은 UMAP / Module Landscape 분석에 최적화되어 있으며, 672개 GO BP term 전체를 사용합니다.
            """
        )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — 18 GO BP Terms Glossary
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("6. PRISM 훈련 18 GO BP Terms — 용어 설명")

st.markdown(
    "PRISM은 골격근(Skeletal Muscle) 조직에서 아래 18개 GO Biological Process term을 기반으로 훈련되었습니다. "
    "Brain zero-shot 분석도 동일한 18개 term 예측값을 사용합니다."
)

with st.expander("18개 GO term 전체 목록 및 생물학적 의미", expanded=True):
    terms = [
        ("GO:0007204", "Ca²⁺-mediated signaling",
         "세포 내 칼슘 이온 신호 전달 경로. 근육 수축 개시 및 세포 사멸에 핵심적."),
        ("GO:0045214", "Sarcomere organization",
         "근절(sarcomere) 구조 조립 및 유지. 근육 수축의 기본 단위 형성."),
        ("GO:0006941", "Striated muscle contraction",
         "횡문근(심근 + 골격근) 수축 과정. 근필라멘트 슬라이딩 메커니즘."),
        ("GO:0006914", "Autophagy",
         "세포 내 손상된 단백질/소기관 자가 분해 과정. 근육 항상성 유지."),
        ("GO:0043161", "Proteasome-mediated UPS",
         "유비퀴틴-프로테아좀 시스템을 통한 단백질 분해 경로."),
        ("GO:0007519", "Skeletal muscle tissue development",
         "골격근 조직 발달 전반. 근육 형성 및 분화 프로그램."),
        ("GO:0042692", "Muscle cell differentiation",
         "근원세포(myoblast)의 성숙한 근섬유(myocyte)로의 분화."),
        ("GO:0055074", "Calcium ion homeostasis",
         "세포 내 칼슘 이온 농도 항상성 유지. SR Ca²⁺ 사이클 포함."),
        ("GO:0007005", "Mitochondrial organization",
         "미토콘드리아 구조 형성, 분열, 융합, 위치 결정."),
        ("GO:0007517", "Muscle organ development",
         "근육 기관 전체 발달 과정. 심근 포함."),
        ("GO:0032006", "Regulation of TOR signaling",
         "mTOR 신호 경로 조절. 근육 단백질 합성/분해 균형에 핵심."),
        ("GO:0030048", "Actin filament-based movement",
         "액틴 필라멘트를 기반으로 한 세포/세포내 운동. Myosin과의 협력."),
        ("GO:0006096", "Glycolytic process",
         "포도당 → 피루브산 변환 경로(해당 과정). 근육의 주요 에너지원."),
        ("GO:0007268", "Chemical synaptic transmission",
         "화학 시냅스를 통한 신경 신호 전달. 신경근 접합부 포함."),
        ("GO:0007018", "Microtubule-based movement",
         "미세소관 기반 분자 모터(Kinesin, Dynein) 운동. 세포내 물질 수송."),
        ("GO:0031175", "Neuron projection development",
         "축삭 및 수상돌기 형성, 성장, 안내. 신경 연결 수립."),
        ("GO:0030182", "Neuron differentiation",
         "전구세포의 성숙한 뉴런으로의 분화. 뇌 발달에 필수."),
        ("GO:0000226", "Microtubule cytoskeleton organization",
         "미세소관 세포골격의 조립, 동적 불안정성, 공간적 조직화."),
    ]

    col_left, col_right = st.columns(2)
    mid = len(terms) // 2

    for i, (go_id, go_name, desc) in enumerate(terms):
        target_col = col_left if i < mid else col_right
        with target_col:
            st.markdown(
                f"""<div style='border:1px solid #e2e8f0;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;background:#fafbfc'>
                <div style='display:flex;justify-content:space-between;align-items:baseline'>
                <b style='font-size:0.9rem;color:#1e293b'>{go_name}</b>
                <code style='font-size:0.78rem;color:#64748b'>{go_id}</code>
                </div>
                <div style='font-size:0.83rem;color:#475569;margin-top:4px;line-height:1.5'>
                {desc}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Quick Reference Card
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("7. 빠른 참고 카드")

with st.expander("핵심 수치 및 기준값 요약"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
**PRISM 성능 (발표 기준)**

| 지표 | 값 |
|------|----|
| Muscle Macro AUPRC | **0.7022** |
| Brain Zero-shot AUPRC (18-term) | **0.5998** |
| Brain Zero-shot AUPRC (41-term) | **0.672** |
| 훈련 GO term 수 | 18 (BP) |
| 아키텍처 | ESM-2 640d → 256→128→64→sigmoid |
| 손실 함수 | Focal Loss (γ=2.0) |
            """
        )

    with c2:
        st.markdown(
            """
**BISECT 임계값 요약**

| 기준 | 값 |
|------|----|
| Tier 1 최소 AD score | ≥ 0.40 |
| Tier 1 최소 delta | ≥ +0.15 |
| Tier 2 최소 CT score | ≥ 0.40 |
| Tier 2 최대 delta | ≤ −0.15 |
| PRISM 고신뢰 임계 | ≥ 0.50 |
| Stage 2 통과 케이스 | `stage2_pass = True` |
            """
        )

    with c3:
        st.markdown(
            """
**데이터셋 요약**

| 항목 | 값 |
|------|----|
| 뇌 조직 케이스 (DTU) | 26개 |
| SRA 단일조건 케이스 | 57개 |
| 전체 BISECT PASS | 83개 |
| Complex I 수렴 케이스 | NDUFS4 / NDUFS7 / NDUFS8 |
| 예시 DLG1 delta | 5× gain (Chemical synaptic) |
| 예시 KIF21B delta | −0.870 (Motor function loss) |
            """
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='margin-top:40px;padding:16px 24px;background:#f1f5f9;
    border-radius:8px;font-size:0.82rem;color:#64748b;text-align:center'>
    PRISM+BISECT &middot; Lee et al. (2026) &middot; <i>Nature Machine Intelligence</i> (in review)<br>
    문의: seungwon.david.lee@gmail.com
    </div>
    """,
    unsafe_allow_html=True,
)
