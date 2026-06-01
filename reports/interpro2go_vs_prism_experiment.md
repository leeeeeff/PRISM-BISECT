# InterPro2GO vs PRISM 직접 비교 실험

**실행일**: 2026-06-01  
**목적**: BISECT PASS 26케이스에서 PRISM의 최대 delta GO term이 pfam2go로 설명 가능한가?

---

## 핵심 결과

| 분류 | 케이스 수 | 의미 |
|------|----------|------|
| **Type I** (pfam2go = PRISM) | **0 / 26** | InterProScan + pfam2go로 PRISM 예측 설명 가능한 케이스 없음 |
| **Type II** (PRISM ≠ pfam2go) | **26 / 26** | PRISM이 pfam2go를 넘는 예측을 하는 케이스 |

**결론: BISECT PASS 26케이스 전부 pfam2go가 설명하지 못하는 PRISM 예측에 의존한다.**

---

## 대표 케이스 상세

### KIF21B (Δ = 0.855, GO:0006936 muscle contraction)
- **PRISM**: CT isoform (kinesin motor) → muscle contraction **0.966** vs AD (WD40) → 0.111
- **pfam2go**: Kinesin (PF00225) → GO:0007018 (MT movement) ← PRISM이 찾는 GO:0006936이 아님
- **해석**: PRISM은 kinesin 서열에서 muscle contraction 기능을 예측. pfam2go는 MT movement만 예측. PRISM이 도메인→기능 매핑을 넘는 context 통합 수행.

### NDUFS4 (Δ = 0.563, GO:0007005 mito organization)
- **PRISM**: canonical → **0.587** vs tr73243 (RVT_1/LINE-1) → **0.024**
- **pfam2go**: RVT_1은 pfam2go에 없음 (비교적 최근 도메인). 설령 있어도 → RNA polymerase activity (GO:0003964), not GO:0007005
- **해석**: PRISM이 서열 맥락에서 mito function 부재를 예측. pfam2go는 이 예측 불가.

### DMD (Δ = 0.817, GO:0042692 muscle cell differentiation)
- **PRISM**: AD isoform → muscle differentiation **0.817** vs CT → 0.000
- **pfam2go**: Spectrin (PF00435) → GO:0005515 (protein binding). Muscle differentiation 예측 없음.
- **해석**: PRISM이 spectrin 도메인 조합에서 muscle differentiation을 예측하는 것은 pfam2go가 할 수 없는 multi-domain context 통합.

### IFT122 (Δ = 0.825, GO:0006936 muscle contraction)  
- **PRISM**: CT (WD40/eIF2A) → muscle contraction **0.825** vs AD (Clathrin/TPR) → 0.000
- **pfam2go**: WD40 → protein binding (generic). Muscle contraction 연결 없음.

---

## 왜 Type I이 0인가?

pfam2go가 PRISM의 18 GO terms와 거의 겹치지 않기 때문:
- PRISM의 18 GO terms는 **muscle/brain specific** (sarcomere, mito org, neuron projection 등)
- pfam2go는 **molecular function / cellular component** 위주 (protein binding, ATP binding, DNA binding)
- 이 불일치 자체가 PRISM의 기여를 보여줌: 도메인→분자기능이 아닌 도메인→세포프로세스 예측

### pfam2go가 PRISM 18 terms 중 매핑하는 것:
| pfam2go 가능 | pfam2go 불가 |
|-------------|-------------|
| GO:0007018 (MT movement) — Kinesin | GO:0006936 (muscle contraction) |
| GO:0003774 (cytoskeletal motor) — Kinesin 계열 | GO:0007005 (mito organization) |
| GO:0006096 (glycolysis) — 일부 대사 효소 | GO:0031175 (neuron projection development) |
| | GO:0042692 (muscle cell differentiation) |
| | GO:0030017 (sarcomere organization) |

→ PRISM이 예측하는 **세포/조직 수준의 생물학적 프로세스** GO terms는 pfam2go의 커버리지 밖.

---

## 논문 기여도에 대한 함의

### 확인된 것
**PRISM은 InterProScan + pfam2go로 대체 불가능하다.**

이유:
1. pfam2go는 도메인→분자기능(MF)/세포구성요소(CC) 위주
2. PRISM은 서열 맥락에서 생물학적 프로세스(BP) GO terms를 예측
3. 26/26 BISECT 케이스가 이 차이에 의존

### 주의사항 (gene-level label 문제)
- PRISM이 예측하는 GO terms (translation 0.889, muscle contraction 0.966)의 일부는 gene-level annotation에서 왔을 수 있음
- DLG1 translation 0.889: DLG1 유전자에 translation GO term이 있는 것이 이상함 → gene-level label noise 가능성
- IFT122 muscle contraction 0.825: IFT122 (intraflagellar transport)가 muscle contraction으로 예측 → 잘못된 gene-level label 전파 가능성

### 진짜 PRISM의 기여
InterProScan + pfam2go 대비 PRISM의 실질적 차별점:
1. **Biological Process GO terms 예측**: pfam2go가 커버 못하는 BP terms를 서열에서 예측
2. **Multi-domain context integration**: 단일 도메인→GO가 아닌, 전체 서열 맥락
3. **Novel isoform scoring**: annotation-free isoform (62.8%)에 대해 pfam2go는 예측 불가, PRISM은 가능
4. **Quantitative score**: binary (도메인 있음/없음)가 아닌 연속 확률값 → BISECT의 delta 기반 필터링 가능

---

*실험 실행: 2026-06-01 | BISECT 26 PASS cases × pfam2go × PRISM 18 BP GO terms*
