# 비판적 재검토: "PRISM이 InterProScan이 못 찾는 것을 찾는다"

**작성일**: 2026-06-01  
**역할**: Nature Methods 심사위원 / Devils Advocate  
**평가 대상 주장**: "InterProScan은 알려진 도메인 유무만 보지만, PRISM은 IDR, Alt promoter, Novel exon combination, ALE/APA 같은 기능 신호를 ESM-2 기반으로 포착한다."

---

## 종합 판정 (선요약)

| 항목 | 평가 |
|------|------|
| **현재 주장 강도** | **3/10** (Nature Methods 기준) |
| **Reject 확률** | **70–80%** (현재 상태 제출 시) |
| **핵심 취약점** | 순환 논리 + 검증 부재 |
| **권고** | RECONSIDER — 주장 수위 조정 또는 실험 추가 필수 |

---

## 주요 반론 5가지

---

### 반론 1: 순환 논리 (CRITICAL)

**주장**: "PRISM이 InterProScan이 못 찾는 기능 신호를 학습한다"

**반론**: PRISM이 학습하는 GO term 레이블 자체가 InterPro 도메인 기반 전자 주석(IEA: Inferred from Electronic Annotation)에서 유래했을 가능성이 높다.

**구체적 문제**:
- UniProtKB GO annotation의 ~60–80%는 IEA (자동 추론)이며, 이 IEA의 상당수는 InterProScan 도메인 기반으로 부여됨
- 즉, PRISM이 "InterProScan보다 잘 찾는다"고 주장하려면, 학습 레이블이 InterProScan과 독립적이어야 한다
- SwissProt 17MB annotation file의 증거 코드(evidence code) 분포가 검증된 적 없음
- Gene-level annotation → 모든 isoform에 동일 레이블 부여 → 이소폼-특이 기능 차이가 레이블에 없음

**심사위원 질문**: "당신의 GO term 레이블 중 InterPro 기반 IEA가 몇 %인가? 그 비율이 높다면, PRISM은 InterProScan의 간접적 증류이지 독립적 발견이 아니다."

**최소 방어 요건**:
- 학습 데이터의 GO annotation evidence code 분포 분석 (EXP/TAS vs IEA 비율)
- IEA 제외 후 PRISM 성능 비교

---

### 반론 2: ESM-2 Pretraining 편향 (CRITICAL)

**주장**: "ESM-2가 IDR, novel motif 등 InterProScan이 못 보는 기능을 인코딩한다"

**반론**: ESM-2의 pretraining data(UniRef50)는 알려진 단백질로 가득하며, ESM-2가 "학습"하는 것이 "novel functional signal"인지 "도메인 패턴의 더 복잡한 재표현"인지 구별할 수 없다.

**구체적 문제**:
- Layer probe 결과에서 "L7: IDR pattern, L18: topology, L27: global context"를 발견했다고 주장
- 하지만 이 해석은 correlation이지 causation이 아님
- "Kinase domain 구별"과 "phosphorylation function 독립 인코딩"은 완전히 다른 주장
- ESM-2가 IDR을 functional하게 표현한다는 독립적 벤치마크가 없음 (Lin et al. 2023의 ESM-2 논문도 IDR에 대해 명시적 검증 없음)

**심사위원 질문**: "Layer probe에서 IDR 관련 패턴을 발견했다고 했는데, 이것이 IDR의 기능(예: phase separation, LLPS tendency)을 인코딩한다는 증거인가, 아니면 단지 낮은 complexity sequence를 구별하는 것인가?"

**최소 방어 요건**:
- IDR function 관련 벤치마크 (예: DisoRDPbind, MoRFpred 예측과 상관관계)
- ESM-2 embedding에서 domain region과 non-domain region의 predictive contribution 분리

---

### 반론 3: "PRISM이 다르게 예측" ≠ "PRISM이 맞다" (SERIOUS)

**주장**: "BISECT PASS 케이스에서 InterProScan이 동일하다고 보지만 PRISM이 다른 예측을 한다"

**반론**: BISECT PASS는 구조적/보존성 증거이지, 기능 차이의 직접 증거가 아니다. 검증되지 않은 예측 차이는 그냥 노이즈일 수 있다.

**구체적 문제**:
- KIF21B ALE vs APA: BISECT가 이를 "다르다"고 판정한 근거가 sequence conservation과 3' UTR 구조 차이인데, 이것이 **기능 차이**로 이어진다는 실험적 증거가 없음
- NDUFS4 alt promoter: mitochondrial targeting sequence 소실은 도메인 변화이며 InterProScan도 탐지 가능
- PTPRF 5-module: module 중 상당수가 Pfam 도메인 → InterProScan이 찾는 영역과 겹침
- AlphaFold pLDDT를 "실험적 증거"로 사용 불가 — 서열 기반 예측이므로 ESM-2와 독립적이지 않음

**심사위원 질문**: "당신이 PRISM이 '맞다'고 주장하는 케이스들 중, 실제 세포 실험으로 기능 차이가 검증된 케이스가 있는가? 있다면 몇 개인가?"

**최소 방어 요건**:
- 최소 3개 케이스에서 실험 검증 (단백질 발현, subcellular localization assay, 기능 assay)
- 또는: 기존 문헌에서 이미 검증된 isoform switch 케이스를 held-out test set으로 구성

---

### 반론 4: 불공정 비교 (SERIOUS)

**주장**: "PRISM vs InterProScan 비교로 PRISM의 우위를 보인다"

**반론**: InterProScan은 function prediction tool이 아닌 annotation tool이다. 이 비교는 공정하지 않으며, 진짜 비교 대상은 DIFFUSE, DeepFRI, CLEAN 같은 기존 function prediction 방법이다.

**구체적 문제**:
- Agent A 결과: Domain-only LR baseline AUPRC = 0.0006 (Glycolysis) → 이것은 InterProScan이 아닌, Pfam feature 기반 logistic regression
- "InterProScan이 못 찾는다"는 표현이 심사위원에게 misleading하게 읽힐 수 있음
- 실제 비교해야 할 것: PRISM vs DIFFUSE (Yao et al. 2022) — 이것이 논문의 원래 contribution 주장
- DeepFRI (Gligorijevic et al. 2021 Nature Comm), CLEAN (Yu et al. 2023 Science) 대비 우위는?

**심사위원 질문**: "왜 당신은 현재 state-of-the-art function prediction 방법들(DeepFRI, CLEAN, ESMFold + GNN)과 직접 비교하지 않았는가?"

**최소 방어 요건**:
- "InterProScan" 표현 대신 "domain-only baseline" 또는 "Pfam feature-based LR" 사용
- DIFFUSE 대비 성능 개선이 메인 contribution으로 유지되어야 함

---

### 반론 5: 데이터 품질 문제 (MODERATE)

**주장**: 근육 AUPRC 0.7022, 뇌 zero-shot 0.5998로 PRISM이 유효하다

**반론**: Gene-level annotation noise와 evaluation 설계 문제가 성능 수치를 과평가할 수 있다.

**구체적 문제**:
- 이소폼 레이블이 gene-level annotation에서 내려왔다면, 같은 유전자의 모든 이소폼이 동일 레이블 → training에서 gene identity만 학습해도 높은 성능 가능
- Brain zero-shot 15% 하락 원인: gene-level overfitting인가 vs. 진짜 tissue-specific function 차이인가? 불명확
- DIFFUSE Dataset#2 Phase B 벤치마크 실행 중 (세션 2026-05-28) — 완료 전 주장 방어 어려움
- Bootstrap CI (n=1000) 적용 여부 불명확

**심사위원 질문**: "뇌 zero-shot에서 15% 하락은 당신의 모델이 근육-특이 편향을 가졌다는 신호 아닌가? 이것이 'isoform-intrinsic' 특징을 학습했다는 주장과 모순되지 않는가?"

**최소 방어 요건**:
- Gene-level stratification 검증: 같은 유전자 이소폼들의 예측값 분산 분석
- Evidence code별 성능 분리 (IEA vs EXP label에서의 성능)
- Bootstrap CI 95% 구간 전 GO term에 대해 제시

---

## 각 반론의 최소 방어 요건 요약

| 반론 | 심각도 | 최소 방어 요건 | 난이도 |
|------|--------|--------------|--------|
| 순환 논리 | CRITICAL | Evidence code 분포 분석 + IEA 제외 재실험 | 중 |
| ESM-2 편향 | CRITICAL | IDR functional benchmark 또는 contribution 분리 | 고 |
| 검증 부재 | SERIOUS | 3개 케이스 실험 검증 또는 문헌 검증 held-out | 고 |
| 불공정 비교 | SERIOUS | 용어 수정 + DeepFRI/CLEAN 비교 추가 | 중 |
| 데이터 품질 | MODERATE | Gene-level overfitting 검증 + Bootstrap CI | 저 |

---

## 수정된 주장 제안

### Option A: Conservative (현재 데이터로 즉시 방어 가능)

> "PRISM의 ESM-2 기반 표현이 Pfam 도메인 기반 logistic regression baseline (AUPRC 0.0006–0.27) 대비 +0.37–0.84 AUPRC 개선을 보인다. 이는 서열 수준의 연속적 표현이 이산적 도메인 주석을 넘어서는 기능적 정보를 포착함을 시사한다."

- InterProScan 직접 언급 제거
- "novel signal discovery" 주장 약화
- 도메인 baseline과의 정량적 비교로 제한

### Option B: 실험 검증 후 (6–12개월 추가 작업)

> "PRISM은 InterProScan이 annotate하지 않는 서열 영역 (IDR, novel exon surface)에서 기능 예측이 가능하며, 이를 KIF21B [실험], NDUFS4 [proteomics], DLG1 [localization assay]로 검증한다."

- Proteomics validation (tr292978, tr73243 단백질 존재 확인)
- KIF21B transport assay
- NDUFS4 Complex I assembly mass spec

### Option C: 비교 대상 변경 (가장 안전한 경로)

> "PRISM은 DIFFUSE (Yao et al. 2022) 대비 근육 AUPRC +X%, 뇌 zero-shot AUPRC +Y% 개선한다. 특히 DIFFUSE가 gene-level reference에 의존하는 것과 달리, PRISM은 이소폼-내재적 서열 특징만으로 예측한다."

- InterProScan 비교 섹션 제거
- DIFFUSE 대비 improvement를 메인 contribution으로
- Zero-shot transfer를 두 번째 contribution으로

**권고: Option A + Option C 조합이 현재 가장 방어 가능한 경로**

---

## Tier S 필수 추가 실험 (논문 방어를 위해)

1. **Evidence code stratification** (2주): GO label 중 IEA vs EXP 비율 측정, IEA 제외 재실험
2. **Domain-matched pair evaluation** (1주): 같은 Pfam 구성을 가진 이소폼 쌍에서 PRISM 예측 차이 분포
3. **Gene-level overfitting 검증** (3일): Train gene / Test gene 완전 분리 후 재평가

---

*Devils Advocate 분석 — Claude Code Agent (2026-06-01)*
