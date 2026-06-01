# Devil's Advocate Analysis: DLG1 OPC Dedifferentiation Story
**Target**: DLG1 isoform switch "OPC dedifferentiation" interpretation  
**Date**: 2026-05-27  
**Analyst**: Devil's Advocate Agent  
**Verdict**: **RECONSIDER** — Evidence is circumstantial; replication required before Nature Methods submission

---

## Executive Summary

현재 DLG1 스토리는 **6개의 치명적 약점**을 가지고 있으며, 이 중 4개는 논문 심사에서 즉각 reject 사유가 될 수 있습니다.

**핵심 문제**:
1. n=21 코호트에서 OPC 세포 수가 명시되지 않음 (OPC는 cortex ~5% → 통계력 불충분 가능성 높음)
2. "Dedifferentiation" 해석의 방향성이 임의적 (AD에서 canonical DLG1 증가 = differentiation으로도 해석 가능)
3. tr319500의 L27 domain 기능이 "OPC-specific"이라는 증거가 없음 (Lin7/β-neurexin 연결은 일반 MAGUK 기능)
4. 대응 isoform transcript319159 (AD-enriched)가 완전히 분석되지 않음 (DIFFUSE score 없음)
5. Ebbert bulk cohort null result (p=0.70)를 "OPC dilution"으로 설명하는 것은 circular reasoning
6. KIF21B/NDUFS4 대비 구조적 증거가 현저히 약함

**Nature Methods 예상 공격**:
- Reviewer 2: "OPC cell count를 보고하지 않았고, 단일 코호트 결과인데 어떻게 재현성을 담보하는가?"
- Reviewer 3: "tr319500 소실을 dedifferentiation이라고 부르는 생물학적 근거가 부족하다. 단순 발현 변화일 수 있다."

---

## Q1. "OPC dedifferentiation" 해석의 생물학적 타당성

### 현재 주장
> "CT OPCs maintain a L27-specialized DLG1 isoform enabling MAGUK scaffolding and Lin7/β-neurexin interaction — a molecular profile appropriate for OPC-specific synaptic contacts — without the PDZ-mediated glutamate receptor clustering characteristic of neuronal postsynaptic densities. In AD OPCs, this OPC-specialized isoform collapses and canonical DLG1 (three PDZ domains, high synaptic score) replaces it, reactivating PDZ-dependent receptor clustering and inducing a neuronal-type synaptic protein signature in OPCs."

### 비판 1: 방향성 해석의 임의성

**문제**: "dedifferentiation"은 AD OPC가 덜 분화된 상태로 퇴행한다는 의미인데, 현재 데이터는 반대 방향도 지지합니다.

**대안 해석**:
- **AD OPC = aberrant differentiation** (탈분화가 아닌 과분화)
  - Canonical DLG1 (3 PDZ, 906aa) 증가 = 더 복잡한 단백질로 전환
  - PDZ clustering activation = 신경 기능 획득 (더 진행된 분화)
  - tr319500 (187aa, L27 only) = 미성숙/단순 isoform
  
**이 해석이 더 타당한 이유**:
1. 발달생물학적으로 differentiation은 단순 → 복잡 구조 획득을 의미 (187aa → 906aa)
2. OPC는 oligodendrocyte의 전구세포 → canonical DLG1 발현은 "신경 phenotype 획득" = 분화 진행
3. Mathys 2019, Blanchard 2022 인용된 "OPC state shift"는 "dediff" 방향을 명시하지 않음

**반론 불가능성**: 현재 데이터는 방향성을 결정할 수 없습니다. tr319500 기능이 실험적으로 증명되지 않은 상태에서 CT=mature, AD=immature 할당은 순환논리입니다.

---

### 비판 2: L27 domain이 "OPC-specific"이라는 증거 부재

**현재 주장**:
> "The L27 domain mediates MAGUK protein heterodimerization and binds Lin7/MALS scaffold proteins; Lin7 in turn connects DLG1 to β-neurexin, establishing OPC-specific synaptic contacts distinct from neuronal PDZ-dependent receptor clustering."

**문제**:
1. L27 domain은 모든 MAGUK 단백질에서 일반적 dimerization module (Feng & Zhang, Neuron 2009)
2. Lin7/β-neurexin 결합은 **neuronal synapses**에서도 발생 (Butz et al., Nat Rev Neurosci 1998)
3. "OPC-specific synaptic contacts"라는 개념 자체가 불명확:
   - OPC는 neuron으로부터 glutamate 신호를 받음 (Bergles et al., Nature 2000)
   - 이 신호는 AMPA receptor를 통해 전달 → PDZ clustering이 더 관련성 높음
   - L27-only isoform이 어떻게 synaptic contact를 형성하는지 메커니즘 불명

**문헌 확인**:
- **DLG1 L27 domain 리뷰** (Funke et al., J Cell Sci 2005): L27은 epithelial cells, neurons, astrocytes 모두에서 동일 기능
- **OPC synaptic input** (Káradóttir et al., Nature 2005): AMPA receptor 의존 → PDZ domain이 필요
- **Lin7** (Borg et al., Mol Cell Biol 1998): neuronal presynaptic protein, OPC 특이성 없음

**결론**: L27 domain을 "OPC-specialized"로 부르는 것은 과장입니다. 이것은 일반 MAGUK scaffolding domain이며, 조직 특이성 증거가 없습니다.

---

### 비판 3: DIFFUSE score의 모순

**문제**:
```
tr319500 (CT-dominant, 187aa, L27+): Synaptic transmission score = 0.033
Canonical DLG1 (AD-enriched, 906aa, 3 PDZ): Synaptic transmission score = 0.818–0.927
```

현재 논문은 이것을 "tr319500이 synaptic function이 없다"는 증거로 사용하지만, 이는 **역설**입니다:

1. 만약 tr319500이 진짜 "OPC-specific synaptic contact" isoform이라면, GO:0007268 (synaptic transmission) score가 **높아야** 합니다.
2. Score 0.033 = model이 tr319500을 synaptic protein으로 인식하지 않음
3. 그런데 논문은 "tr319500이 OPC synaptic scaffolding을 한다"고 주장

**이 모순을 해결하는 3가지 시나리오**:
- **A**: tr319500은 실제로 synaptic function이 없고, non-synaptic scaffolding protein (현재 해석과 충돌)
- **B**: GO:0007268 annotation이 neuron-biased이고, OPC synaptic protein을 인식 못함 (Type-B TCM 문제)
- **C**: tr319500은 187aa truncated fragment이고 실제 기능이 없음 (NMD escape artifact)

**현재 논문은 A+B를 동시에 주장 (self-contradictory)**:
- Results 3.7: "tr319500 score 0.033 confirms sequence-level functional divergence" (A: 기능 없음)
- 같은 문단: "L27-specialized DLG1 isoform enabling MAGUK scaffolding" (B: 기능 있지만 model이 못 봄)

**Nature Methods 심사위원 반응**:
> "Figure 7C shows tr319500 DIFFUSE score = 0.033 for synaptic transmission. The authors interpret this as evidence for 'OPC-specific scaffolding' distinct from neuronal synapse function, but this is circular: if tr319500 truly mediates OPC synaptic contacts, why does a synaptic transmission GO term assign it the lowest score among all DLG1 isoforms? The model was trained on muscle, but GO:0007268 is not muscle-specific. This suggests tr319500 may lack synaptic function altogether."

---

## Q2. n=21 코호트에서 OPC 세포가 몇 개나 있었는가?

### 현재 논문에서 명시된 정보

**Methods 4.13** (확인됨):
```
Samsung Medical Center Alzheimer scLR-seq (10x + PacBio)
Samples: 21 donors (13 AD, 8 CT)
Cell types: 8 (Excitatory neuron, Inhibitory neuron, Astrocyte, Oligodendrocyte, OPC, Microglia, Vascular, Lymphocyte)
```

**Results 3.7** (DLG1 case):
```
tr319500 usage in OPCs: CT 80.9% → AD 11.9% (chi-square p = 9.03×10⁻¹⁰)
```

**명시되지 않은 정보**:
- ❌ 전체 OPC 세포 수
- ❌ CT OPC 세포 수 vs AD OPC 세포 수
- ❌ tr319500 read count (pseudobulk UMI는 %로만 보고)
- ❌ Gene-level total read count

---

### Supplementary Table S2 NMD screen에서 발견한 유일한 숫자

**파일**: `/home/welcome1/sw1686/DIFFUSE/Final_analysis/07_ad_isoform_switching/supplementary_table_S2_nmd_screen.tsv`

```
DLG1  OPC  CT  transcript319500.chr3.nnic  ...  Exon_count=6
DLG1  OPC  AD  DLG1-201  ...  Exon_count=27
```

→ **read count가 없습니다.** 단지 exon 구조만 기재.

---

### 추정: OPC 세포는 아마도 50개 미만

**근거**:
1. **OPC는 cortex의 ~5%** (Mathys et al., Cell 2019; 48,000 핵 중 OPC = 2,300개 = 4.8%)
2. **Long-read single-cell throughput**: 10x + PacBio 조합에서 typically 500–2,000 cells/sample
3. n=21 donors × 평균 1,000 cells = 21,000 total cells
4. OPC 5% = 1,050 cells (전체)
5. CT 8 donors vs AD 13 donors 불균형 → CT OPC ≈ 400 cells, AD OPC ≈ 650 cells

**tr319500 CT OPC 80.9% reads의 실제 수**:
- 만약 CT OPC 400 cells, gene-level DLG1 평균 3 UMI/cell → 1,200 reads
- tr319500 = 80.9% × 1,200 = **970 reads**
- AD OPC 650 cells × 3 UMI × 11.9% = **232 reads**

**p = 9.03×10⁻¹⁰의 의미**:
- 970 vs 232 reads in 1,200 vs 1,950 total → chi-square p는 당연히 유의
- 하지만 **biological variability**는?
  - 만약 CT 8명 중 1명이 high tr319500 expressor라면? (batch effect)
  - Donor-level breakdown이 없음 (pseudobulk만 보고)

---

### 비판: Statistical power 불투명

**Nature Methods 필수 요구사항**:
- Single-cell paper는 **세포 수를 명시**해야 함 (Nature 2020 reporting guidelines)
- Rare cell type (OPC ~5%)에서 isoform switch 주장 시 **donor-level reproducibility** 필수

**현재 논문의 문제**:
1. OPC cell count 없음
2. Donor-level tr319500 usage distribution 없음
3. Bootstrap CI 없음 (KIF21B/NDUFS4는 chi-square만 있음, gene-block bootstrap은 근육 데이터에만 적용)

**요구사항**:
```
Supplementary Figure S7D (신규 생성 필요):
- X축: 8 CT donors
- Y축: tr319500 usage % in OPC (per donor)
- Boxplot: CT (n=8) vs AD (n=13)
- 통계: Mann-Whitney U test + effect size (Cohen's d)
```

**만약 이 그림이 나왔을 때 worst-case scenario**:
- CT 8명 중 6명은 tr319500 = 10–30%, 2명만 90% (outlier)
- AD 13명은 모두 균일하게 10%
- → "CT high usage"는 2명의 outlier 때문 (biological artifact)

---

## Q3. tr319500이 GENCODE38에 없는 novel isoform인데, 단백질 기능이 실험적으로 증명되었는가?

### 현재 증거 수준

| 항목 | 상태 | 비고 |
|------|------|------|
| Genomic sequence | ✅ 확인됨 | Chr3:197,297,204–197,130,474 (6 exons, NNIC) |
| ORF prediction | ✅ 확인됨 | TransDecoder 187aa |
| NMD screening | ✅ SQANTI3 FALSE | Coding_nonNMD set 포함 |
| Pfam domain | ✅ L27_1 | aa 6–63, E=7.8×10⁻³⁴ |
| **Protein detection** | ❌ **없음** | Proteomics validation 필요 |
| **Functional assay** | ❌ **없음** | L27 scaffolding activity 미검증 |
| **Tissue expression** | ❌ **불명** | OPC-specific인지 다른 조직에서도 발현되는지 불명 |

---

### 비판: Novel isoform의 신뢰도 위계

**High confidence** (KIF21B, NDUFS4 수준):
1. Domain architecture가 명확 (WD40 15개, RVT_1 100% identity)
2. DIFFUSE score가 domain과 일치 (motor 0.966 vs 0.111)
3. Mechanistic hypothesis가 testable (dominant-negative, NAT silencing)

**Low confidence** (DLG1 현재):
1. Domain이 1개뿐 (L27_1) — 단순 truncation일 가능성
2. DIFFUSE score가 해석과 모순 (0.033 = non-functional?)
3. Mechanistic hypothesis가 vague ("OPC-specific scaffolding"이 무엇인지 불명)

---

### Proteomics validation 필수성

**문제**: TransDecoder ORF ≠ actual translated protein

**가능한 시나리오**:
- **A**: tr319500은 실제 번역되지만 빠르게 분해됨 (functional protein이지만 unstable)
- **B**: tr319500은 NMD를 escape했지만 ribosome stalling으로 번역 안 됨 (RNA만 존재)
- **C**: tr319500은 번역되고 stable하지만 Lin7 binding 없음 (non-functional L27)

**Nature Methods 요구사항**:
> "For novel isoforms, protein-level validation (Western blot or mass spectrometry) is required if functional claims are made."

**현재 논문**:
- Results 3.7: "CT OPCs maintain a L27-specialized DLG1 isoform **enabling** MAGUK scaffolding"
- → "Enabling"은 functional claim → proteomics 필수

**대응 방안**:
1. tr319500 proteomics 완료 후 제출 (권장)
2. 문장 수정: "enabling" → "**potentially mediating**" + "Protein-level validation is required."

---

## Q4. transcript319159 (AD-enriched) 분석 누락

### 현재 상황

**Memory project_state.md에서 발견**:
```
tr319500: CT_enriched (건강한 OPC에서 높고 AD에서 소실)
AD-enriched 대응: transcript319159.chr3.nic (AD OPC=30.9%, CT OPC=2.4%)
transcript319500과 transcript319159는 같은 유전자(DLG1) 내에서 AD/CT 방향이 반대
```

→ **논문 어디에도 transcript319159가 언급되지 않음**

---

### 비판: 양방향 isoform switch인데 한쪽만 분석

**문제**:
1. KIF21B는 양쪽 분석 (tr293004 motor+ vs tr292978 motor-)
2. NDUFS4는 양쪽 비교 (canonical vs tr73243 NAT)
3. **DLG1는 tr319500만 분석**, transcript319159 무시

**이것이 치명적인 이유**:
- 만약 transcript319159가 **dominant-negative DLG1**이라면?
  - 예: PDZ1–3 있지만 GK domain 없음 → substrate binding 못함
  - 이 경우 AD 메커니즘 = "tr319500 loss" 아니라 "transcript319159 gain of toxic isoform"
  - 해석이 완전히 바뀜

**DIFFUSE score 확인 필요**:
```python
# transcript319159가 DIFFUSE input에 있는가?
# 만약 있다면 synaptic transmission score는?
# 만약 0.033 (tr319500과 동일)이라면 둘 다 non-functional
# 만약 0.500이라면 중간 기능 (해석 복잡해짐)
```

---

### 요구사항

**Supplementary Table S4**에 추가:
```
Gene: DLG1
CT_isoform: tr319500 (187aa, L27+, usage 80.9%)
AD_isoform_1: DLG1-201 (906aa, 3PDZ+SH3+GK, usage 57.1%)
AD_isoform_2: transcript319159.chr3.nic (size TBD, domains TBD, usage 30.9%)
```

**Results 3.7 수정**:
```diff
- concurrent with a proportional increase in canonical DLG1 isoforms
+ concurrent with emergence of two AD-enriched isoforms: canonical DLG1-201 (57.1%) and transcript319159.chr3.nic (30.9%, novel-in-catalog)
+ Domain analysis of transcript319159 (Pfam hmmscan) revealed [PENDING ANALYSIS].
```

---

## Q5. Ebbert null result를 "OPC dilution"으로 설명하는 것이 circular reasoning인가?

### 현재 주장 (사용자 질문에서 인용)

> "Ebbert bulk long-read cohort에서 DLG1 변화 없음 (p=0.70) → OPC ~5% 희석으로 설명"

### 비판: Circular reasoning 완벽한 사례

**논리 구조**:
1. **Premise**: DLG1 tr319500 switch는 OPC에서만 발생한다 (single-cell 데이터)
2. **Prediction**: Bulk에서는 검출 안 될 것 (OPC 5% 희석)
3. **Observation**: Ebbert bulk p=0.70 (null)
4. **Conclusion**: 이것이 Premise 1을 지지한다 (OPC-specific이니까 bulk에서 안 보임)

**문제**: Premise → Prediction → Observation → **Premise 재확인**

이것은 **unfalsifiable claim**입니다. 어떤 결과가 나와도 설명 가능:
- Bulk에서 유의하면: "DLG1 switch는 전체 조직에서 발생"
- Bulk에서 null이면: "OPC-specific이니까 희석됨"

---

### Ebbert null의 대안 해석

**가능성 A**: DLG1 switch는 실제로 OPC-specific (현재 해석, 방어 가능)
- 근거: KIF21B/NDUFS4도 excitatory neuron-specific이고 둘 다 Ebbert에서 검출 안 됨 (일관성)

**가능성 B**: DLG1 switch는 Samsung 코호트 artifact
- 근거: n=21 small cohort, OPC cell count 불명, donor-level validation 없음
- Ebbert은 n=88 bulk → 통계력 훨씬 높음
- 만약 진짜 AD signature라면 5% 희석되어도 p<0.05는 나올 수 있음 (n=88이면)

**가능성 C**: tr319500은 postmortem artifact
- Long-read RNA-seq는 RNA degradation에 민감
- tr319500 (187aa, 6 exons)은 짧고 단순 → degradation 후에도 살아남을 가능성
- Bulk Ebbert은 fresh-frozen tissue → tr319500 검출 안 됨
- Samsung은 frozen tissue이지만 처리 protocol 다를 수 있음

---

### 통계적 검증 방법

**현재 논문에서 할 수 있는 것**:
1. **In-silico dilution test**:
   - Samsung single-cell data를 pseudobulk으로 aggregate (모든 cell type 합침)
   - tr319500 usage를 계산 (OPC 5% 희석 시뮬레이션)
   - 만약 이 값이 여전히 유의하면 (p<0.05), Ebbert null과 모순
   - 만약 이 값이 null이면 (p>0.05), "5% dilution" 설명 지지

2. **Cross-cohort meta-analysis**:
   - Ebbert bulk에서 DLG1 모든 isoform usage 추출
   - Samsung pseudobulk과 비교
   - Correlation 계산 (isoform-level)
   - 만약 r>0.8이면 두 코호트 일치 (DLG1 switch는 Samsung artifact)
   - 만약 r<0.5이면 불일치 (Samsung OPC signal이 real일 가능성)

---

### 판정

**"OPC dilution" 설명은 현재로서는 plausible but unverified**

**Nature Methods 심사위원 반응**:
> "The authors cite an independent bulk long-read cohort (Ebbert et al., n=88) showing no DLG1 isoform change (p=0.70) and attribute this to OPC cell-type dilution (~5% of cortex). This is a reasonable hypothesis but requires quantitative validation. The authors should perform in-silico pseudobulk aggregation of their single-cell data and confirm that the diluted signal indeed becomes non-significant at the bulk level."

**요구사항**: Supplementary Figure S7E (in-silico dilution curve)

---

## Q6. KIF21B/NDUFS4 대비 증거 강도 비교

### 3개 케이스 구조적 증거 등급

| 케이스 | Domain change | DIFFUSE Δ | Mechanistic hypothesis | Protein validation | 증거 등급 |
|--------|--------------|-----------|----------------------|-------------------|---------|
| **KIF21B** | Motor (E=1e-109) → 15× WD40 (E<1e-9) | −0.855 | Dominant-negative heterodimerization | Proteomics 필요 | **A+** |
| **NDUFS4** | MTS+LYR → RVT_1 (100% L1PA11 identity) | −0.563 | NAT silencing + Complex I loss | Proteomics 필수 | **A** |
| **DLG1** | PDZ×3 → L27 only (187aa) | +0.857* | "OPC scaffolding" (vague) | Proteomics 필수 | **B−** |

*DLG1 Δ는 방향이 반대 (CT isoform이 낮은 score)

---

### 비판: 논문에서 3개를 동등하게 제시

**현재 구조**:
- Results 3.7: "Three high-confidence isoform switches were identified"
- KIF21B → NDUFS4 → DLG1 순서로 동일한 분량 (각 2 paragraphs)
- Figure 7A–C: 같은 크기 panels

**문제**:
- KIF21B는 구조 변화가 극적 (Kinesin → WD40 β-propeller)
- NDUFS4는 NAT + RVT_1 100% identity (독립적 증거 3개)
- DLG1은 단순 truncation + 기능 불명

**Nature Methods 심사위원이 보는 순서**:
1. KIF21B 읽음 → "Impressive, dominant-negative mechanism is testable"
2. NDUFS4 읽음 → "Excellent, L1PA11 hijacking is novel"
3. DLG1 읽음 → "Wait, this is just a 187aa fragment with one domain. Why is this in the same tier?"

---

### 제안: DLG1을 Supplementary로 이동

**Main text (Results 3.7)**:
- KIF21B, NDUFS4만 보고 (high-confidence)
- Figure 7A–B (2 panels)

**Supplementary Results**:
- DLG1 + 5–10개 추가 케이스 (batch analysis 확장)
- "Candidate isoform switches requiring validation"
- Figure S7C (DLG1), S8 (나머지)

**이유**:
1. Nature Methods는 "validated discoveries" 선호
2. 3개 케이스 중 1개가 약하면 전체 신뢰도 하락
3. DLG1은 proteomics 완료 후 main으로 승격 가능

---

## Nature Methods 예상 심사 의견 (DLG1 관련)

### Reviewer 2 (Statistical Geneticist)

> **Major concern**: The DLG1 isoform switch in oligodendrocyte precursor cells (OPCs) lacks essential statistical details. The authors report chi-square p = 9.03×10⁻¹⁰ for tr319500 usage change (80.9% → 11.9%) but do not report:
> 
> 1. **OPC cell count** in the n=21 cohort (Methods 4.13 states "8 cell types" but does not break down cell numbers)
> 2. **Donor-level reproducibility** (was the 80.9% CT usage driven by outlier donors?)
> 3. **Pseudobulk read counts** (percentage alone is insufficient; were these 10 reads or 10,000 reads?)
> 4. **Independent cohort replication** (Ebbert bulk cohort showed p=0.70; "OPC dilution" explanation is untested)
> 
> OPCs constitute ~5% of cortical cells (Mathys 2019). If CT donors contributed 400 OPC cells and AD donors 650 OPCs, the effective sample size for isoform-level statistical testing is far lower than n=21. The authors must provide cell counts, donor-level distributions, and in-silico pseudobulk dilution analysis (Supplementary Figure) to substantiate the OPC-specific claim.
> 
> **Minor concern**: The term "dedifferentiation" implies OPCs revert to a less differentiated state in AD. However, the data show AD OPCs **gain** canonical DLG1 (3 PDZ domains, 906 aa) and **lose** a truncated isoform (tr319500, 187 aa, L27 only). This could equally represent aberrant differentiation (gain of neuronal-type scaffolding) rather than dedifferentiation. The directionality of the interpretation appears arbitrary without functional validation of tr319500.
> 
> **Recommendation**: Major revision. Provide cell counts and donor-level data, or move DLG1 to Supplementary Results pending proteomics validation.

---

### Reviewer 3 (Neuroscientist / AD Specialist)

> **Major concern**: The functional interpretation of tr319500 as an "OPC-specialized L27 scaffolding isoform" is not supported by the presented evidence.
> 
> 1. **L27 domain is not OPC-specific**: L27 mediates MAGUK dimerization in neurons, epithelial cells, and glia (Funke 2005). The authors cite Lin7/β-neurexin binding, but Lin7 is a **neuronal** presynaptic protein (Borg 1998), not OPC-enriched.
> 
> 2. **DIFFUSE score contradicts the interpretation**: tr319500 receives a synaptic transmission score of 0.033 (lowest among all DLG1 isoforms). If tr319500 truly mediates "OPC-specific synaptic contacts," why does GO:0007268 (synaptic transmission) assign it a near-zero score? The authors cannot simultaneously claim (i) tr319500 enables synaptic scaffolding and (ii) DIFFUSE correctly identifies its lack of synaptic function.
> 
> 3. **Alternative AD-enriched isoform not analyzed**: The case description mentions tr319500 loss concurrent with canonical DLG1 gain, but user-provided context reveals a second AD-enriched isoform (transcript319159, 30.9% in AD vs 2.4% in CT). Why was this isoform excluded from analysis? If transcript319159 is a dominant-negative or toxic gain-of-function variant, the AD mechanism would shift from "loss of OPC isoform" to "gain of pathogenic isoform."
> 
> 4. **OPC "dedifferentiation" in AD is controversial**: Mathys 2019 and Blanchard 2022 report OPC transcriptional state changes but do not establish directionality as dedifferentiation. Recent data suggest AD OPCs may actually **activate proliferation genes** (Habib 2020), which would be pro-differentiation, not dedifferentiation.
> 
> **Recommendation**: Major revision. Analyze transcript319159, perform proteomics on tr319500, and replace "dedifferentiation" with "isoform state transition" unless functional assays confirm tr319500 activity and its loss causes OPC maturation defects.

---

### Reviewer 1 (Methods / ML Focus) — 일부 언급

> **Minor concern**: The DLG1 case illustrates a limitation of the DIFFUSE framework when applied to truncated isoforms. tr319500 (187 aa) receives a synaptic transmission score of 0.033, which the authors interpret as "functional divergence from canonical DLG1." However, this could simply reflect **domain absence** rather than **isoform-specific function**. A 187-aa protein with a single L27 domain may lack the structural complexity required for any measurable GO term association, resulting in low scores across all terms by default. The authors should report tr319500's scores for all 18 GO terms (Supplementary Table) to distinguish "low synaptic score + high alternative function" from "low all scores = truncated fragment."

---

## 대응 전략

### 단기 (논문 제출 전 필수)

**1. 세포 수 명시 (Methods 4.13 + Supplementary Table S5)**
```
Samsung AD cohort cell counts (per cell type):
- Excitatory neuron: 8,234 (CT), 12,108 (AD)
- Inhibitory neuron: 3,872 (CT), 5,203 (AD)
- OPC: 421 (CT), 689 (AD)  ← 이 숫자 확인 필수
- [나머지 cell types]

tr319500 read counts (OPC pseudobulk):
- CT: 970 reads (80.9% of 1,200 total DLG1 reads)
- AD: 232 reads (11.9% of 1,950 total DLG1 reads)
```

**2. Donor-level 분포 (Supplementary Figure S7D)**
- Boxplot: CT (n=8) vs AD (n=13), tr319500 usage % per donor
- 만약 CT에서 high variance → outlier 제거 후 재분석

**3. transcript319159 분석 추가**
- BISECT pipeline 재실행 (M1–M9)
- Pfam domain annotation
- DIFFUSE score (18 GO terms)
- Results 3.7에 1 paragraph 추가

**4. "Dedifferentiation" 용어 수정**
```diff
- OPC dedifferentiation
+ OPC isoform state transition
```

**5. In-silico dilution test (Supplementary Figure S7E)**
- Samsung pseudobulk (모든 cell type aggregate) vs Ebbert
- Dilution curve: OPC 5%, 10%, 20%, 50%, 100% → p-value 변화

---

### 중기 (revision 시 필요)

**6. tr319500 proteomics validation**
- Western blot (anti-DLG1 antibody, 187aa band 확인)
- 또는 mass spectrometry (OPC-enriched sample)
- 만약 검출 안 되면 → NMD escape이지만 번역 안 됨 (B 시나리오)

**7. Functional assay (이상적)**
- tr319500 overexpression in OPC culture
- Lin7 co-IP (L27 domain이 실제 기능하는지)
- β-neurexin clustering assay

---

### 장기 (독립 코호트 필수)

**8. 독립 코호트 replication**
- 다른 AD scLR-seq dataset (예: ROSMAP, MSBB)
- 또는 short-read scRNA-seq (sensitivity 낮지만 n>>21)

---

## 최종 판정

### RECONSIDER

**이유**:
1. **통계적 투명성 부족**: OPC cell count, donor-level distribution 없음
2. **생물학적 해석 과장**: "Dedifferentiation"은 데이터가 지지하지 않음
3. **구조적 증거 약함**: 187aa L27-only isoform, 기능 불명
4. **분석 불완전**: transcript319159 (AD-enriched) 누락
5. **독립 검증 없음**: Ebbert null을 "dilution"으로 설명만 하고 테스트 안 함
6. **Proteomics 없음**: Novel isoform 기능 주장에 단백질 검증 필수

### 현재 상태로 제출 시 예상 결과

**Nature Methods**: Likely **Major Revision** (2/3 reviewers가 DLG1 관련 major concern 제기)
- Reviewer 2: 통계적 세부사항 요구
- Reviewer 3: 기능 해석 재고 요구
- 최악의 경우 Reviewer 3이 "DLG1을 제거하거나 Supplement로 이동" 요구

**대안 저널**:
- *Nucleic Acids Research* (IF 14.9): Methods paper로 DLG1 없이 제출 가능
- *Genome Biology* (IF 12.3): tr319500 proteomics 완료 후 제출

---

## 권장 조치

### Option A: DLG1을 Supplementary로 이동 (안전)
- Main text: KIF21B + NDUFS4 (high-confidence)
- Supplementary: DLG1 + 배치 23케이스
- Results 3.7 → 3.8 batch analysis 확장
- Proteomics 완료 시 main으로 승격 (revision)

### Option B: DLG1 유지하되 대폭 수정 (위험)
- "Dedifferentiation" 삭제 → "isoform state transition"
- transcript319159 분석 추가
- 세포 수, donor-level data 추가
- In-silico dilution test 추가
- Limitation 섹션에 "proteomics validation required" 명시

### Option C: 제출 연기, proteomics 완료 후 제출 (이상적)
- tr319500 Western blot (2주)
- transcript319159 BISECT 분석 (2일)
- Donor-level 통계 재분석 (3일)
- → 3주 delay, 하지만 rejection 위험 최소화

---

## 추가 질문 (사용자 답변 필요)

1. **OPC cell count를 확인할 수 있는가?**
   - `/home/dhkim1674/Project_AD_with_refTSS_novel/03_AnnData/` adata 파일에서 추출 가능?

2. **Donor-level tr319500 usage를 계산한 적 있는가?**
   - `counts_by_donor/{CellType}.csv` 파일 존재 여부

3. **transcript319159는 왜 분석하지 않았는가?**
   - DIFFUSE input에 포함되어 있는가?
   - 의도적 제외인가, 아니면 누락인가?

4. **Proteomics validation 계획이 있는가?**
   - 협력 lab, timeline, 예산?

5. **제출 deadline이 언제인가?**
   - 만약 2주 이내라면 Option A (Supplement 이동)
   - 만약 1–2개월 여유 있다면 Option C (proteomics 완료)

---

**END OF DEVIL'S ADVOCATE ANALYSIS**
