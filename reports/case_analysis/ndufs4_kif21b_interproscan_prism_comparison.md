# NDUFS4 & KIF21B: InterProScan vs PRISM vs 실험 근거 3-way 비교

**작성일**: 2026-06-01  
**목적**: Nature Methods Figure 후보 케이스 선정 및 주장 방어력 평가  
**출처**: BISECT PASS 케이스 + Agent B 분석

---

## Executive Summary

| 케이스 | InterProScan 관점 | PRISM 관점 | 실험 근거 | Figure 등급 |
|--------|-----------------|-----------|---------|------------|
| **KIF21B** | Kinesin domain 동일, C-terminal 차이 불명확 | ALE vs APA 기능 차이 예측 | 문헌 간접 지지 | **Main Figure 후보** |
| **NDUFS4** | N-terminal domain 소실 (도메인 변화 명확) | Alt promoter → Complex I assembly 차이 | Proteomics 필요 | Supplementary 권장 |

**핵심 판정**: KIF21B가 "PRISM beats domain annotation" 주장에 더 강한 케이스. NDUFS4는 domain 소실이 너무 명확해서 InterProScan도 탐지 가능.

---

## Case 1: NDUFS4

### 배경

NDUFS4 (NADH:Ubiquinone Oxidoreductase Subunit S4)는 미토콘드리아 Complex I의 assembly factor. Alt promoter 이소폼이 N-terminal mitochondrial targeting sequence(MTS)를 잃는 것으로 알려짐.

### 3-way 비교표

| 항목 | InterProScan 관점 | PRISM 예측 | 실험/문헌 근거 |
|------|-----------------|-----------|-------------|
| **Reference isoform (isoform 1)** | Pfam: NDUF_S4 domain (PF05699), MTS signal | ETC GO:0022900 high probability | Complex I assembly 기능 확립 (literature) |
| **Alt promoter isoform** | MTS domain ABSENT (N-terminal truncation) | ETC GO:0022900 lower probability | MTS 소실 → cytoplasmic localization 예상 |
| **도메인 차이** | **명확한 domain loss** — InterProScan이 탐지 가능 | PRISM이 추가로 예측하는 것: downstream assembly effect | 직접 실험 데이터 없음 |
| **PRISM의 추가 가치** | 제한적 — MTS 소실 자체는 이미 도메인 변화 | Assembly-related GO term 변화 예측 | 미검증 |

### 분석

**InterProScan도 이미 탐지한다**: MTS는 known signal peptide이며, SignalP / TargetP (InterProScan 포함)도 이 변화를 탐지 가능하다. "PRISM이 InterProScan이 못 찾는 것을 찾는다"는 주장에서 NDUFS4는 **반례**에 가깝다.

**PRISM의 실제 기여**: MTS 소실 이후 downstream protein folding과 Complex I assembly에 미치는 영향. 이것은 InterProScan이 못 보지만, 이 예측이 맞는지 실험 검증이 없다.

**Nature Methods 평가**: Supplementary 권장. Main figure로 사용 시 심사위원이 "InterProScan도 이미 N-terminal 차이를 탐지한다"고 반박할 것.

**필요한 추가 데이터**:
- Proteomics: alt promoter 이소폼 단백질 실제 발현 확인
- Blue native PAGE: Complex I assembly 차이 직접 측정

---

## Case 2: KIF21B

### 배경

KIF21B (Kinesin Family Member 21B)는 신경세포에서 축삭 수송에 관여. ALE(Alternative Last Exon) vs APA(Alternative Polyadenylation) 이소폼의 기능 차이가 관심 대상 (F70 발견).

### 3-way 비교표

| 항목 | InterProScan 관점 | PRISM 예측 | 실험/문헌 근거 |
|------|-----------------|-----------|-------------|
| **Reference isoform** | Pfam: Kinesin motor domain (PF00225), WD40 repeat | Microtubule-based movement GO:0007018 high | Kinesin-mediated transport (well established) |
| **ALE isoform** | **동일한 Kinesin + WD40 도메인** — C-terminal IDR 변화는 Pfam에 없음 | PRISM: microtubule function 감소, protein binding 증가 예측 | IDR의 LLPS tendency 차이 간접 지지 |
| **APA isoform** | **동일한 Kinesin + WD40 도메인** — 3' UTR 차이만 | PRISM: 예측값 ALE와 유사하지만 미세 차이 | APA가 단백질 서열 변화 없을 수 있음 (UTR만 변화시) |
| **도메인 차이** | **없음** — InterProScan 동일 판정 가능성 높음 | PRISM: embedding space에서 분리 가능 | 미검증 |
| **PRISM의 추가 가치** | **고** — 도메인이 동일한데 기능 차이 예측 | C-terminal IDR 변화가 신호 | 문헌: KIF21B C-terminal이 kinesin processivity 조절 |

### 분석

**InterProScan의 한계가 명확**: KIF21B ALE 이소폼의 C-terminal disordered tail 변화는 Pfam domain이 아니므로 InterProScan 기반 방법으로는 탐지 불가. 이것이 "PRISM이 찾는 것"의 핵심 사례가 될 수 있다.

**주의사항 — APA 구분 문제**: APA(Alternative Polyadenylation)가 단백질 서열을 바꾸지 않을 경우 (3' UTR만 변화), PRISM이 탐지하는 것은 서열 차이가 아니라 다른 무언가다. ALE와 APA를 명확히 구분해야 한다 (F70 발견의 핵심).

**ESM-2 embedding 분리 증거**: Layer probe 결과 (L7: IDR pattern encoding)와 연결하면, C-terminal IDR 변화가 L7에서 다른 표현을 만든다는 주장 가능.

**Nature Methods 평가**: **Main Figure 후보**. 단, 다음 조건:
1. ALE vs APA를 명확히 정의 (단백질 서열 변화 여부)
2. InterProScan 실제 실행 결과로 "domain 동일 판정" 확인
3. C-terminal IDR 변화의 실험적 중요성 문헌 인용

---

## 주장 방어력 비교

| 비교 항목 | NDUFS4 | KIF21B |
|---------|--------|--------|
| InterProScan도 탐지 가능한가? | **예** (MTS 소실) | **아니오** (IDR 변화) |
| PRISM 예측에 실험 근거 있는가? | 없음 (proteomics 필요) | 간접 지지 (문헌) |
| Domain annotation과 독립적인가? | **낮음** | **높음** |
| Figure-grade인가? | Supplementary | **Main Figure 후보** |
| 추가 필요 작업 | Proteomics | InterProScan 실행 확인 |

---

## Nature Methods 심사위원 예상 질문

### NDUFS4에 대해
1. "N-terminal MTS 소실은 InterProScan SignalP/TargetP로도 탐지된다. PRISM의 추가 기여는 무엇인가?"
2. "Alt promoter 이소폼이 실제로 세포에서 발현되는가? 질량분석 증거가 있는가?"

### KIF21B에 대해
1. "ALE와 APA의 구분이 중요하다. 두 이소폼 모두 단백질 서열이 다른가, 아니면 하나는 UTR만 다른가?"
2. "C-terminal IDR 변화가 기능에 영향을 준다는 직접 실험 증거가 없는데, PRISM 예측이 옳다는 근거는?"
3. "ESM-2가 IDR을 functional하게 표현한다는 독립적 증거가 있는가?"

---

## Figure 디자인 권장사항 (KIF21B Main Figure)

```
Panel A: 이소폼 구조 비교
  - Kinesin domain (공통) | WD40 (공통) | IDR tail (다름)
  - InterProScan annotation: 두 이소폼에서 동일한 도메인만 표시

Panel B: ESM-2 Embedding Space
  - UMAP/t-SNE: reference vs ALE 이소폼의 분리
  - 색깔: 도메인 있는 region vs IDR region contribution

Panel C: PRISM GO Term Prediction Delta
  - GO:0007018 (microtubule-based movement): reference vs ALE
  - 예측 차이 (delta) 시각화

Panel D: (Optional) 실험 검증
  - 문헌에서 KIF21B C-terminal 기능 관련 실험 인용
```

---

## Acceptance Path 시나리오

### Major Revision (가능성 높음)
- 요구사항: InterProScan 실제 실행 결과 추가 + ALE/APA 서열 차이 명확화
- 기간: 2–3개월

### Minor Revision (현재 데이터 최적화 시)
- 조건: Agent C 권고 사항 반영 (주장 수위 조정 + evidence code 분포 분석)
- KIF21B를 "conceptual case study"로 프레이밍 (검증 필요성 명시)

---

## 결론

**KIF21B를 Main Figure 케이스로 선택하되**, 다음 수정 사항을 반영한다:

1. "InterProScan이 못 찾는다"는 표현 → "Pfam domain annotation 범위 외의 기능 신호"로 완화
2. ALE vs APA 구분을 서열 수준에서 명확히 정의
3. InterProScan을 실제로 두 이소폼에 실행해서 "domain 동일 판정" 수치로 보여줌
4. Layer probe L7 결과와 연결: IDR region의 embedding 기여도

**NDUFS4는 Supplementary**로 보내고, proteomics 데이터 확보 후 main text 승격 검토.

---

*3-way Case Analysis — Agent B + 직접 작성 (2026-06-01)*
