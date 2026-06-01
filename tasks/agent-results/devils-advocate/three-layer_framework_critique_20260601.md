# Devil's Advocate: Three-Layer Framework Critique
**Target**: PRISM + BISECT "Three-layer annotation framework" (InterProScan → PRISM → BISECT)  
**Date**: 2026-06-01  
**Reviewer stance**: Nature Methods / Nature Machine Intelligence — Reject level 비판  
**Verdict**: **MAJOR REVISION** (framework는 유지하되, 주장 대폭 수정 필요)

---

## EXECUTIVE SUMMARY

논문이 제안하는 "3-layer annotation framework"는 다음과 같이 주장한다:
- Layer 1 (InterProScan): Pfam domain → MF GO terms, novel isoform 예측 불가
- Layer 2 (PRISM): ESM-2 embedding → 18 BP GO terms, 92.3% 케이스에서 InterProScan 너머
- Layer 3 (BISECT): Multi-evidence validation (AlphaFold + PPI + PhyloP + LINE)

**Critical assessment:**
1. **"비중복적" 주장의 온톨로지 트릭**: MF vs BP 차이를 활용한 인위적 분리
2. **92.3% 수치의 통계적 취약성**: n=26은 power 부족, selection bias 있음
3. **진짜 기여의 재정의 필요**: "도구 파이프라인"이 아닌 "isoform-specific functional representation"
4. **PRISM vs ESM-2 경계의 약점**: 일부 brain GO term에서 raw ESM-2가 더 좋음
5. **BISECT의 독립성 문제**: PRISM-flagged 케이스만 분석 → 순환 논리 위험

**Recommendation**: 
- **RECONSIDER** "3-layer framework"를 main contribution으로 제시하는 전략
- **PIVOT**: PRISM의 핵심 기여는 "task-specific functional representation transfer" + "isoform-specific discrimination"
- **BISECT**: 독립적 contribution으로 제시, PRISM과의 결합은 "application" 섹션으로 격하

**Reject Risk**: 
- Framework novelty: **HIGH** (기존 도구 조합)
- 92.3% 주장: **MEDIUM** (n=26은 약하지만 biological case studies가 보완)
- PRISM itself: **LOW** (isoform discrimination 증거는 견고)

---

## 1. 방법론적 기여 문제

### 1.1 "3-layer framework" = 도구 조합인가, 방법론 혁신인가?

**비판**:
"3-layer annotation framework"라는 명명은 마치 새로운 방법론 패러다임처럼 들리지만, 실제는:
- Layer 1: InterProScan (1999~현재 표준 도구, PMID: 10368463)
- Layer 2: PRISM (ESM-2 frozen embedding + 3-layer MLP + focal loss)
- Layer 3: BISECT (AlphaFold + STRING + PhyloP + custom scripts)

**Nature Methods가 원하는 것**: "substantially improved or new methods"
**이 논문이 제공하는 것**: 기존 도구들을 파이프라인으로 연결

**Occam's Razor 대안**:
1. InterProScan 단독으로 domain annotation 수행
2. ESM-2 embedding을 logistic regression으로 직접 GO 예측 (v10-B 제거)
3. AlphaFold pLDDT + domain annotation으로 isoform switch 검증 (PRISM 제거)

**핵심 질문**: 3개 도구를 순차 적용하는 것이 각각 독립적으로 실행하는 것 대비 무엇을 추가하는가?

**저자 반박 가능성**:
- Layer 1+2 결합: "92.3% 케이스에서 pfam2go가 못 잡는 BP GO term을 PRISM이 예측" (아래 1.2에서 재비판)
- Layer 2+3 결합: "PRISM score로 BISECT 우선순위 결정 → 계산 효율성 증가"
  - **Counter**: 이것은 "computational efficiency gain"이지 "methodological contribution"이 아님
  - **Counter2**: BISECT가 PRISM-flagged 케이스만 분석 → PRISM-independent validation 부재

**판정**: 
- **Reject risk: HIGH**
- Nature Methods는 "파이프라인 구축"을 method로 인정하지 않음 (이것은 workflow paper)
- 각 layer의 독립적 기여가 명확히 quantified되어야 함

---

### 1.2 "비중복적" 주장 = MF vs BP 온톨로지 트릭

**핵심 주장 (논문)**:
"In 24 of 26 BISECT-validated cases (92.3%), PRISM predicts a Biological Process GO term not captured by InterProScan+pfam2go domain annotation, demonstrating complementary and largely non-overlapping prediction spaces."

**비판 Level 1 — 당연한 결과**:
- pfam2go mapping은 Pfam domain → **Molecular Function** (MF) GO terms로 주로 매핑됨
  - 예: Kinesin domain (PF00225) → GO:0003777 (microtubule motor activity, MF)
  - 예: PDZ domain (PF00595) → GO:0030165 (PDZ domain binding, MF)
- PRISM은 **Biological Process** (BP) GO terms (18개)를 예측하도록 훈련됨
  - 예: GO:0007018 (microtubule-based movement, BP)
  - 예: GO:0007268 (chemical synaptic transmission, BP)
- MF와 BP는 GO 위계에서 **근본적으로 다른 수준**:
  - MF = "분자가 무엇을 하는가" (biochemical activity)
  - BP = "세포 수준에서 어떤 과정에 참여하는가" (cellular function)

**결론**: 92.3% "비중복성"은 **ontology space를 애초에 다르게 정의했기 때문**이지, PRISM이 InterProScan의 예측 능력을 초과해서가 아니다.

**Analogy**:
"우리 모델은 단백질의 3D 구조를 예측하고, AlphaFold는 서열 보존성을 예측하므로 92% 케이스에서 비중복적이다."
→ 이것은 애초에 다른 것을 예측하는 것이므로 당연히 겹치지 않는다.

**비판 Level 2 — GO 위계 무시**:
- MF GO term (motor activity)과 BP GO term (microtubule-based movement)은 **같은 생물학적 기능의 다른 층위**
- GO 온톨로지에서 BP는 종종 MF의 상위 개념:
  ```
  GO:0003777 (microtubule motor activity, MF)
    ↓ part_of
  GO:0007018 (microtubule-based movement, BP)
  ```
- KIF21B 케이스:
  - pfam2go: Kinesin domain → GO:0003777 (motor activity, MF)
  - PRISM: GO:0007018 (MT-based movement, BP)
  - 이것은 "비중복"이 아니라 **수렴** (같은 기능을 다른 수준에서 표현)
- 논문은 KIF21B와 CCAR1을 "Type I"로 분류했지만, 이것이 오히려 **대부분의 케이스가 Type I여야 한다는 반증**

**저자 반박 가능성**:
- "MF와 BP는 다른 수준이지만, pfam2go는 BP로 매핑 불가능하므로 PRISM이 필요하다"
  - **Counter**: Gene Ontology에는 MF → BP mapping이 있음 (is_a, part_of 관계)
  - InterProScan은 domain → MF만 예측하지만, GO 위계 traversal로 BP 추론 가능
  - 논문이 이 간단한 baseline을 비교하지 않은 것은 **missing critical control**

**Occam's Razor 대안**:
- pfam2go MF predictions → GO graph traversal → inferred BP predictions
- 이 simple rule-based approach와 PRISM을 비교해야 함
- 만약 PRISM이 이것보다 낫다면 진짜 기여, 그렇지 않으면 "우리는 MF를 BP로 자동 변환했다"는 사소한 기여

**판정**:
- **Reject risk: HIGH**
- "92.3% complementary" 주장은 **기만적 프레이밍** (deceptive framing)
- Reviewer가 GO ontology를 이해하면 즉각 반박 가능

---

### 1.3 Sample Size: n=26은 충분한가?

**통계적 문제**:
- 전체 클레임: "92.3% (24/26) 케이스에서 PRISM이 InterProScan 너머"
- Binomial test: n=26, observed=24/26=0.923
  - Null hypothesis: p=0.5 (random guess)
  - p-value < 0.001 → 통계적으로 유의함
- **하지만**: 이것은 "random보다 낫다"는 증명이지, "일반화 가능하다"는 증명이 아님

**Selection bias**:
- 26 케이스는 어떻게 선정되었는가?
  - 논문: "BISECT PASS cases (brain n=26)"
  - BISECT는 어떤 기준으로 케이스를 선정했는가?
    - AlphaFold pLDDT, domain change, PPI change, PhyloP, LINE-1 중 3/5 이상 만족
  - 이 기준 자체가 **"structural/domain change가 있는 케이스"를 선호**
  - 따라서 26 케이스는 "domain annotation이 informative한 케이스"일 가능성 높음
  - 역설적으로, 이런 케이스에서 pfam2go와 PRISM이 수렴해야 함 (KIF21B처럼)
  - 24/26이 Type II라는 것은 **pfam2go BP inference가 실패**했다는 뜻

**Statistical power**:
- 95% CI for 24/26: [75.7%, 99.1%] (Wilson score interval)
- 만약 진짜 population parameter가 80%라면 (not 92.3%):
  - n=26에서 24개 관찰 확률 = 6.8% (still possible)
- n=26은 **92.3% vs 80%를 구분할 power가 없음**
- 최소 필요 샘플: n=100 이상 (power 0.8 기준)

**Missing validation**:
- SRA 58 케이스는 왜 분석에서 제외되었는가?
  - 논문: "BISECT 84 PASS (brain 26 + SRA 58)"
  - 26 케이스 분석만 보고 → **cherry-picking 의심**
  - SRA 58 케이스에서 Type II 비율이 더 낮을 가능성 있음

**저자 반박 가능성**:
- "n=26은 proof-of-concept, SRA 58은 supplementary에 포함"
  - **Counter**: Main claim을 n=26으로 하면서 n=84를 숨기는 것은 selective reporting
- "BISECT validation 자체가 엄격하므로 n=26이어도 신뢰할 만함"
  - **Counter**: Validation 엄격성과 sample size adequacy는 별개 문제

**판정**:
- **Reject risk: MEDIUM**
- n=26은 약하지만, biological case studies (KIF21B, NDUFS4, DLG1)가 개별적으로 convincing
- Reviewer가 통계에 엄격하면 major revision 요구

---

## 2. PRISM 자체의 기여 재정의

### 2.1 진짜 기여: Isoform-specific Functional Discrimination

**현재 논문 프레이밍**: "3-layer framework"
**진짜 기여 (숨겨져 있음)**: PRISM이 gene-level annotation에서 훈련되었음에도 isoform-specific prediction을 생성

**강력한 증거 (논문에 있지만 주요 주장 아님)**:
1. Within-gene variance (0.00126) > Between-gene variance (0.00070)
   - 이것은 PRISM이 gene identity를 외우는 것이 아니라 isoform sequence features에 반응한다는 직접 증거
2. DLG1 canonical (906aa, 3 PDZ) vs novel (186aa, no PDZ): 27× score differential
   - Gene-level label은 둘 다 positive → 모델이 label을 override
3. IFT122 isoforms: 4,100× score differential within same gene

**이것이 왜 중요한가**:
- GO annotation은 gene-level only (UniProt/GOA에 isoform 구분 없음)
- 기존 모델들은 gene-level label을 학습 → 모든 isoform에 동일 score
- PRISM은 ESM-2 embedding이 isoform 구조를 구분하므로 → isoform-specific score 생성
- **이것은 "supervised learning with noisy labels" 문제의 성공 사례**

**Reframing 제안**:
- Main contribution: "PRISM learns isoform-specific functional representations from gene-level supervision"
- 증거: within-gene variance, DLG1/IFT122 case studies
- InterProScan 비교는 **secondary validation**, not main claim
- 3-layer framework는 **application**, not method

**판정**:
- **Reject risk: LOW** (이 프레이밍으로 전환하면)
- 이것은 진짜 methodological insight

---

### 2.2 약점: PRISM vs ESM-2 경계가 명확하지 않음

**문제 (Exp A 결과)**:
- PRISM-18 (learned representation) vs ESM-2-640 (raw embedding)
- 20 brain GO terms에서:
  - 12 terms: PRISM > ESM-2 (up to 10×)
  - 8 terms: ESM-2 > PRISM (예: potassium transport, GPCR signaling)
- 전체 평균: PRISM 0.169 vs ESM-2 0.055 (+3.1×)

**비판**:
- PRISM이 ESM-2보다 나은 이유가 **training GO terms와의 기능적 연관성**에 한정됨
- 연관 없는 GO terms에서는 오히려 ESM-2가 더 나음
- 이것은 PRISM이 "functional representation"을 학습했다기보다 **"18 GO terms의 선형 조합"**을 학습했다는 증거

**Occam's Razor 대안**:
- ESM-2 640-dim을 PCA로 18-dim으로 축소 → 비슷한 성능 가능성
- 만약 PCA-18이 PRISM-18과 비슷하면 → "task-specific learning" 주장 약화

**저자 반박 가능성**:
- "논문은 이 boundary를 명시적으로 보고함 (honest boundary)"
  - **맞음**: 논문이 ESM-2 우위 케이스를 숨기지 않고 보고한 것은 신뢰성 증가
- "PRISM의 적용 범위는 training GO terms와 연관된 biological space로 한정된다"
  - **맞음**: 이것은 limitation이지만, 명시적으로 인정하면 방어 가능

**판정**:
- **Reject risk: LOW** (limitation을 명시적으로 보고했으므로)
- 하지만 "task-specific representation transfer"라는 프레이밍은 약간 과장

---

### 2.3 Missing Baseline: ESM-2 Linear Probe가 진짜 baseline

**비판**:
- 논문은 "LR baseline (AUPRC 0.363) vs PRISM (0.7022)"를 비교
- 하지만 Exp A에서 ESM-2 640-dim linear probe가 일부 brain GO terms에서 PRISM보다 나음
- **질문**: muscle 18 GO terms에서도 ESM-2 linear probe가 PRISM에 근접하는가?

**Missing experiment**:
- ESM-2 640-dim → Logistic Regression (C=1.0, balanced) → 18 muscle GO terms
- 논문은 이것을 "LR baseline"이라고 하지만, 실제로는 무엇을 썼는지 불명확
  - Methods: "Logistic regression (C = 1.0, class_weight = 'balanced'; scikit-learn 1.3.0)"
  - 입력이 ESM-2 640-dim인지 명시적으로 확인 필요

**만약 ESM-2 LR이 이미 0.363이면**:
- PRISM (3-layer MLP + BN + focal loss)의 기여 = 0.7022 - 0.363 = +0.339 AUPRC
- 이것은 **architecture + loss function의 기여**이지, "feature representation"의 기여가 아님
- 따라서 "PRISM learns functional representation"보다는 "PRISM is better-tuned MLP"가 정확

**저자 반박 가능성**:
- "LR baseline은 ESM-2 640-dim 입력 사용, Methods에 명시됨"
  - **확인 필요**: manuscript Methods Section 2.7 참조
  - 만약 맞다면 PRISM의 +0.339 AUPRC gain은 legitimate
- "Architecture matters: BN + focal loss가 sparse GO annotation에 필수"
  - **맞음**: 하지만 이것은 "engineering contribution"이지 "scientific contribution"이 아님

**판정**:
- **Reject risk: LOW** (Methods가 명확하다면)
- 하지만 Nature Methods는 "hyperparameter tuning" 논문을 좋아하지 않음

---

## 3. BISECT의 독립성 문제

### 3.1 BISECT는 PRISM-independent validation인가?

**문제**:
- 논문 주장: "BISECT validates PRISM predictions"
- 실제 workflow:
  1. PRISM scores all isoform pairs → flag high-delta cases
  2. BISECT runs on PRISM-flagged cases only
  3. 84/84 PASS → 100% validation rate

**비판**:
- BISECT가 PRISM-flagged cases만 분석 → **selection bias**
- PRISM score가 낮은 케이스는 BISECT 분석 대상이 아님 → false negative rate 알 수 없음
- 순환 논리: "PRISM이 예측한 케이스를 BISECT가 검증했으므로 PRISM이 맞다"
  - **But**: PRISM이 놓친 케이스 중 BISECT가 검증할 만한 케이스는 몇 개인가?

**Missing control**:
- PRISM score 무관하게 모든 DTU significant isoform pairs를 BISECT 분석
- PRISM-flagged vs PRISM-missed 케이스 간 BISECT PASS rate 비교
- 만약 PRISM-missed 케이스의 PASS rate도 높으면 → PRISM의 기여 없음

**Analogy**:
"우리는 체온 37.5도 이상인 환자에게만 COVID 검사를 했고, 100% 양성이었으므로 체온이 COVID의 완벽한 지표다."
→ 체온이 낮지만 COVID인 환자를 놓쳤을 가능성 무시

**저자 반박 가능성**:
- "BISECT는 computationally expensive (AlphaFold structure, STRING PPI 등)"
  - **맞음**: 모든 케이스 분석은 비현실적
- "PRISM은 pre-screening 역할, BISECT는 causal validation 역할"
  - **맞음**: 하지만 이것은 "workflow efficiency gain"이지 "PRISM validation"이 아님
- "84/84 PASS는 PRISM pre-screening이 효과적이라는 증거"
  - **Counter**: PASS rate가 100%라는 것은 오히려 threshold가 너무 낮거나 BISECT 기준이 너무 관대하다는 의심

**판정**:
- **Reject risk: MEDIUM**
- 84/84 PASS는 suspicious (real world에서 100% success rate는 드묾)
- False positive rate를 측정하는 negative control 필요

---

### 3.2 BISECT 자체의 기여는 무엇인가?

**BISECT 구성 요소**:
1. AlphaFold pLDDT (structural confidence)
2. Pfam domain annotation (InterProScan)
3. STRING PPI network change
4. PhyloP vertebrate conservation
5. LINE-1 transposable element detection

**질문**: 이 중 어느 것이 novel인가?
- AlphaFold: 기존 도구
- Pfam: InterProScan (기존 도구)
- STRING: 기존 DB
- PhyloP: 기존 도구
- LINE-1 detection: custom script (potentially novel)

**BISECT의 진짜 기여**:
- **Integration of multiple evidence sources** → 이것은 "pipeline"이지 "method"가 아님
- **3/5 criteria threshold** → 이것은 heuristic rule
- **Case-by-case manual interpretation** → 이것은 "analysis", not automated method

**Occam's Razor 대안**:
- 각 evidence source를 독립적으로 사용
- Domain change (InterProScan)만으로도 대부분 케이스 설명 가능
- 예: KIF21B (motor domain loss), DLG1 (PDZ domain loss), IFT122 (WD40 loss)
- PRISM 없이도 동일한 결론 도출 가능

**저자 반박 가능성**:
- "BISECT는 multi-evidence integration을 자동화했다"
  - **Counter**: 논문에 automation 코드 없음, case-by-case interpretation 명시됨
- "PRISM이 BP GO term을 예측했기 때문에 biological process context가 추가됨"
  - **맞음**: 이것은 legitimate contribution
  - 하지만 이것은 PRISM의 기여이지 BISECT의 기여가 아님

**판정**:
- **Reject risk: HIGH** (BISECT를 method로 포장하면)
- BISECT는 "validation pipeline"으로 제시하는 것이 정직함

---

## 4. Nature Methods 적합성 평가

### 4.1 "Method" 기준 충족 여부

**Nature Methods acceptance criteria** (https://www.nature.com/nmeth/aims):
- "New methods or **substantial improvements to existing methods**"
- "Demonstrated performance gains over current state-of-the-art"
- "Broad applicability across biological domains"

**이 논문의 제출 전략별 평가**:

| 전략 | Method? | Performance gain? | Broad applicability? | Verdict |
|------|---------|-------------------|----------------------|---------|
| **A: 3-layer framework** | No (파이프라인) | Unclear (vs what?) | Yes (any tissue) | **REJECT** |
| **B: PRISM (isoform discrimination)** | Marginal (MLP tuning) | Yes (+91% vs LR) | Yes | **MAJOR REVISION** |
| **C: Task-specific representation** | No (transfer learning) | Yes (10× vs ESM-2) | Limited (related GO only) | **REJECT** |
| **D: BISECT multi-evidence** | No (integration) | N/A (no baseline) | Yes | **REJECT** |

**권장 전략**: **B (PRISM isoform discrimination)**
- Main claim: "First isoform-level GO predictor trained on gene-level labels"
- Evidence: within-gene variance, DLG1/IFT122 cases, 27× discrimination
- InterProScan comparison: secondary validation
- BISECT: application case study

---

### 4.2 경쟁 저널 대안

**만약 Nature Methods reject**:

| Journal | Impact Factor | 적합한 전략 | 가능성 |
|---------|--------------|------------|--------|
| **Nature Communications** | 16.6 | A (framework) + B (PRISM) | **High** |
| **Nucleic Acids Research** | 14.9 | B (PRISM) + biological cases | **High** |
| **Genome Biology** | 17.9 | A (framework) + AD discoveries | Medium |
| **Bioinformatics** | 5.8 | B (PRISM) technical | **High** |
| **PLOS Computational Biology** | 4.3 | C (representation transfer) | High |

**추천**: Nature Communications
- "Biological significance" 중시 (KIF21B/NDUFS4/DLG1 AD cases)
- "Tool/resource" 논문도 받음 (3-layer framework 가능)
- Impact factor 여전히 높음

---

## 5. 각 비판에 대한 Reject 가능성 + 방어 전략

### 비판 1: "3-layer framework = 도구 조합"
- **Reject 가능성**: **HIGH**
- **방어 전략**:
  - Framework를 main contribution이 아닌 "integrated workflow" 섹션으로 격하
  - PRISM isoform discrimination을 main contribution으로 전면 배치
  - 성공 가능성: **Medium** (contribution 재정의 필요)

---

### 비판 2: "92.3% = MF vs BP 온톨로지 트릭"
- **Reject 가능성**: **HIGH**
- **방어 전략**:
  1. **Baseline 추가**: pfam2go MF → GO graph traversal → inferred BP predictions
     - 이것과 PRISM 비교
     - PRISM이 여전히 나으면 legitimate
  2. **주장 수정**: "PRISM은 domain annotation 없이 BP를 예측한다" (not "InterProScan을 넘어선다")
  3. **Type I/II 재해석**: Type I (KIF21B)을 convergent validation으로 강조
  - 성공 가능성: **High** (baseline 추가하면)

---

### 비판 3: "n=26은 통계적으로 약함"
- **Reject 가능성**: **MEDIUM**
- **방어 전략**:
  1. **SRA 58 케이스 분석 추가**: brain 26 + SRA 58 = 84 전체 분석
     - Type I/II 비율이 유지되는지 확인
  2. **Bootstrap CI 추가**: 24/26의 95% CI [75.7%, 99.1%] 명시
  3. **Individual case studies 강조**: KIF21B/NDUFS4/DLG1의 biological validation이 statistical power보다 중요
  - 성공 가능성: **High** (SRA 분석 추가하면)

---

### 비판 4: "PRISM vs ESM-2 경계 불명확"
- **Reject 가능성**: **LOW**
- **방어 전략**:
  - 이미 논문에서 명시적으로 보고 (honest boundary)
  - Limitation section에 "PRISM은 training GO와 연관된 functional space에서만 ESM-2 대비 우위" 명시
  - 성공 가능성: **High** (이미 방어됨)

---

### 비판 5: "BISECT = PRISM-dependent validation (순환 논리)"
- **Reject 가능성**: **MEDIUM**
- **방어 전략**:
  1. **Negative control 추가**: PRISM score 낮은 케이스 중 random sampling → BISECT 분석
     - PASS rate가 84/84보다 낮으면 PRISM pre-screening 효과 입증
  2. **BISECT를 independent discovery로 재포지셔닝**:
     - BISECT가 AD isoform switches를 발견한 것이 main contribution
     - PRISM은 functional interpretation을 추가한 것
  - 성공 가능성: **Medium** (실험 추가 필요)

---

## 6. 대안 패러다임 (항상 3가지)

### 대안 1: Graph Neural Network on PPI
**아이디어**:
- Isoform을 node로, PPI를 edge로 하는 GNN
- Node feature = ESM-2 embedding + expression level
- GO term을 graph label로 학습

**왜 더 나을 수 있는가**:
- PPI context를 명시적으로 모델링 (PRISM은 ESM-2 embedding에 암묵적으로 포함)
- Isoform-specific PPI를 STRING에서 가져올 수 있음 (BISECT가 이미 사용)

**왜 시도 안 했는가**:
- Isoform-level PPI 데이터 부족 (대부분 gene-level)
- GNN은 MLP보다 복잡 → 해석 어려움

---

### 대안 2: AlphaFold Structure Embedding
**아이디어**:
- ESM-2 sequence embedding 대신 AlphaFold single_repr (structure embedding) 사용
- Structure가 function을 더 직접적으로 반영

**왜 더 나을 수 있는가**:
- Domain fold가 function의 직접 결정자
- AlphaFold pLDDT가 BISECT에서 유효했으므로 PRISM에서도 유효할 것

**왜 시도 안 했는가**:
- AlphaFold 구조 예측 비용 (36,748 isoforms × compute time)
- Novel isoforms의 경우 구조 예측 신뢰도 낮음

**저자 반박**:
- "AlphaFold는 canonical isoforms만 예측, novel isoforms 구조 없음"
- "ESM-2는 모든 isoforms에 적용 가능"
- **Counter**: ESMFold (ESM-2 기반 구조 예측) 사용 가능했을 것

---

### 대안 3: Retrieval-based Isoform Function Transfer
**아이디어**:
- Training set에서 가장 유사한 isoform K개 찾기 (ESM-2 embedding cosine similarity)
- K개의 GO terms를 majority vote로 예측

**왜 더 단순한가**:
- MLP training 필요 없음
- Interpretable (왜 이 GO term을 예측했는지 유사 isoform 제시 가능)

**왜 더 나을 수 있는가**:
- Novel isoforms의 97.4%가 known gene family → retrieval로 충분
- Zero-shot generalization (brain)이 이미 작동 → 유사 isoform 기반 예측으로 가능

**저자 반박**:
- "Retrieval은 training set에 없는 isoform에 취약"
  - **Counter**: 논문 자체가 97.4%가 known gene family라고 밝힘
- "MLP는 non-linear feature combination 학습"
  - **Counter**: 그렇다면 왜 일부 GO term에서 ESM-2 linear probe가 더 나은가?

**Occam's Razor 승리 가능성**: **Medium**
- Retrieval baseline을 추가하면 PRISM의 진짜 기여가 명확해질 것

---

## 7. 데이터 문제 우선 의심

### 7.1 GO Annotation Quality
**문제**:
- GO annotation source: UniProt/GOA gene-level annotations
- Evidence code 분포 (paper_defense_checklist.md 참조):
  - IEA (Inferred from Electronic Annotation): InterPro 기반 자동 주석
  - EXP (Experimental): 실험 검증

**순환 논리 위험**:
- 만약 training GO annotations의 대부분이 IEA (InterProScan 기반)이면:
  - PRISM이 학습한 것 = InterProScan의 예측 패턴
  - "PRISM이 InterProScan을 넘어선다"는 주장은 순환 논리

**Missing analysis**:
- Evidence code별 성능 분석
- IEA 제외 후 PRISM 재훈련 → 성능 변화 확인

**저자 반박 가능성**:
- "go_label_evidence_analysis.md에서 이미 분석 완료"
  - **확인 필요**: 해당 파일 읽어야 함
- 만약 EXP-only training에서도 성능 유지되면 순환 논리 기각

---

### 7.2 Train/Test Split Leakage
**문제**:
- Gene-stratified split: 모든 isoforms of a gene → same partition
- 하지만 gene family는?
  - 예: GABRB1, GABRB2, GABRB3 (같은 family, 비슷한 function)
  - 만약 GABRB1이 train, GABRB3가 test → gene family memorization 가능

**Missing analysis**:
- Gene family-stratified split
- Paralog genes를 같은 partition에 배치

**저자 반박 가능성**:
- "Sequence similarity가 높으면 당연히 function도 비슷 → 이것은 leakage가 아니라 legitimate generalization"
  - **맞음**: 하지만 "isoform-level" 주장과 모순
  - Isoform-level이 중요하다면 gene family-level도 중요

---

### 7.3 Evaluation Metric Appropriateness
**문제**:
- AUPRC primary metric (sparse labels) → 올바름
- 하지만 "isoform discrimination" 주장에는:
  - **Within-gene ranking correlation**이 더 직접적 metric
  - 예: Spearman correlation between PRISM score와 isoform expression ratio (high vs low expression isoform)

**Missing analysis**:
- Gene-wise ranking performance
- "Same gene, different isoforms" subset에서만 평가

**저자 반박 가능성**:
- "F39 실험에서 이미 within-gene ranking 검증 (Ridge Spearman 0.200)"
  - **Counter**: F39는 expression prediction이지 GO function prediction 아님

---

## 8. 최종 권장사항

### 8.1 논문 재구성 (Reject 회피 전략)

**현재 구조** (3-layer framework 중심):
```
Abstract: 3-layer framework (InterProScan → PRISM → BISECT)
Results:
  §1 PRISM muscle performance
  §2 PRISM brain zero-shot
  §3 BISECT case studies
Discussion: Framework integration
```

**권장 구조** (PRISM isoform discrimination 중심):
```
Abstract: 
  - PRISM learns isoform-specific GO functions from gene-level labels
  - Within-gene variance > between-gene variance
  - 27× discrimination (DLG1)
  - Zero-shot brain transfer
  - Validated by BISECT multi-evidence pipeline

Results:
  §1 Isoform-specific functional discrimination
     - Within/between-gene variance
     - DLG1, IFT122 case studies
     - pos_bias metric
  §2 PRISM muscle performance vs baselines
     - LR, RF, XGBoost comparison
     - Type-A/B classification
  §3 Task-specific representation transfer
     - PRISM-18 vs ESM-2-640 on 20 brain GO terms
     - Functional relatedness boundary
  §4 Zero-shot brain application
     - 7,903 novel isoforms coverage
     - 541 high-confidence predictions
  §5 BISECT integration: AD isoform switches
     - KIF21B, NDUFS4, DLG1 Tier A cases
     - PRISM pre-screening efficiency

Discussion:
  - PRISM vs InterProScan: complementary (not competitive)
  - Limitations: gene family generalization, task-specific transfer
  - BISECT provides causal validation
```

---

### 8.2 추가 실험 (Major Revision 대비)

**P1 — 필수**:
1. **pfam2go MF → BP inference baseline**
   - InterProScan domain → MF GO → GO graph traversal → BP GO
   - PRISM과 비교
   - 예상 결과: PRISM이 여전히 나음 (sequence context 활용)
   - 방어력: **HIGH**

2. **SRA 58 케이스 Type I/II 분석**
   - 현재 brain 26만 분석
   - 전체 84 케이스로 확장
   - 예상 결과: Type II 비율 약간 감소 (80~85%)
   - 방어력: **MEDIUM**

**P2 — 권장**:
3. **Retrieval baseline**
   - K-NN on ESM-2 embeddings → GO term majority vote
   - PRISM과 비교
   - 예상 결과: PRISM이 약간 나음 (non-linear combination)
   - 방어력: **HIGH** (Occam's Razor 대응)

4. **Evidence code 분석**
   - IEA vs EXP GO annotations 성능 비교
   - IEA 제외 후 재훈련
   - 예상 결과: 성능 약간 감소하지만 유지
   - 방어력: **MEDIUM** (순환 논리 대응)

---

### 8.3 주장 수정 (명확화)

**수정 전**:
"PRISM predicts a Biological Process GO term not captured by InterProScan+pfam2go, demonstrating complementary and largely non-overlapping prediction spaces."

**수정 후**:
"PRISM predicts Biological Process GO terms directly from sequence embeddings without requiring domain homology, providing functional annotation for novel isoforms where InterProScan produces no output. In 24 of 26 BISECT-validated cases (92.3%), PRISM's maximal functional prediction corresponds to a BP GO term, while pfam2go primarily annotates Molecular Function GO terms — reflecting the hierarchical structure of the Gene Ontology rather than competitive prediction spaces. The two approaches converge in cases where domain-to-function mapping is straightforward (e.g., KIF21B Kinesin domain → MT-based movement), providing mutual cross-validation."

---

## 9. VERDICT

### Overall Assessment

| Dimension | Score (1-10) | Rationale |
|-----------|--------------|-----------|
| **Novelty** | 4/10 | 3-layer framework = 도구 조합, PRISM MLP = 표준 architecture |
| **Rigor** | 7/10 | Bootstrap CI, gene-stratified split 올바름. 하지만 n=26 약함 |
| **Biological insight** | 8/10 | KIF21B/NDUFS4/DLG1 case studies compelling |
| **Technical execution** | 7/10 | ESM-2 embedding + focal loss 적절. Baselines 부족 |
| **Presentation** | 6/10 | "3-layer framework" 과장, "92.3% complementary" 오해 유발 |

**총점**: 32/50 = **64%**

---

### Reject Risk by Claim

| Claim | Reject Risk | 방어 가능성 |
|-------|------------|-----------|
| "3-layer annotation framework" | **HIGH** | LOW (contribution 재정의 필요) |
| "92.3% non-overlapping with InterProScan" | **HIGH** | MEDIUM (baseline 추가하면) |
| "PRISM learns isoform-specific functions" | **LOW** | HIGH (증거 견고) |
| "Task-specific representation transfer" | **MEDIUM** | HIGH (이미 boundary 명시) |
| "BISECT validates PRISM" | **MEDIUM** | MEDIUM (negative control 필요) |

---

### Final Recommendation

**Nature Methods 제출 시**:
- **MAJOR REVISION** 필요
- Main contribution을 "PRISM isoform discrimination"으로 재정의
- 3-layer framework를 application section으로 격하
- pfam2go BP inference baseline 추가
- SRA 58 케이스 분석 추가

**대안 저널**:
- **Nature Communications**: 현재 draft로 제출 가능 (biological impact 중시)
- **Nucleic Acids Research**: PRISM 중심으로 재구성 후 제출
- **Genome Biology**: AD discoveries 강조하면 가능

**Success Probability**:
- Nature Methods (현재 draft): **30%** (reject 위험)
- Nature Methods (수정 후): **60%** (major revision)
- Nature Communications (현재 draft): **70%**
- NAR (PRISM 중심): **80%**

---

## 10. 개별 질문 답변

### Q1: 이 프레임워크 자체가 진짜 방법론적 기여인가, 아니면 기존 도구를 파이프라인으로 묶은 것뿐인가?

**답**: **기존 도구를 파이프라인으로 묶은 것에 가깝다.**

- InterProScan: 1999년부터 표준 도구
- PRISM: ESM-2 (2023) + 3-layer MLP (표준 architecture) + focal loss (2017)
- BISECT: AlphaFold + STRING + PhyloP + custom scripts

**진짜 기여**: PRISM의 isoform-specific discrimination (within-gene variance analysis)
**약한 기여**: 3개 도구를 순차 적용하는 workflow

**Reject 가능성**: **HIGH** (framework를 main contribution으로 제시하면)

---

### Q2: "비중복적"이라는 주장이 MF vs BP 온톨로지 차이를 인위적으로 활용한 트릭인가?

**답**: **예, 상당 부분 그렇다.**

- pfam2go는 MF GO terms 예측 (motor activity, domain binding)
- PRISM은 BP GO terms 예측 (MT-based movement, synaptic transmission)
- MF와 BP는 GO hierarchy에서 **part_of** 관계 → 같은 기능의 다른 층위
- 92.3% "비중복"은 **ontology space를 다르게 정의했기 때문**

**방어 전략**: pfam2go MF → GO traversal → BP inference baseline 추가 필요

**Reject 가능성**: **HIGH** (현재 주장 유지 시)

---

### Q3: 92.3% 수치의 validity: 26개 BISECT validated 케이스가 충분한 샘플인가?

**답**: **통계적으로 약하다.**

- n=26, 92.3% → 95% CI [75.7%, 99.1%] (Wilson score)
- 진짜 population이 80%일 가능성 배제 못함
- SRA 58 케이스 (전체 84) 분석 누락 → **selective reporting 의심**

**방어 전략**: 전체 84 케이스 분석 추가

**Reject 가능성**: **MEDIUM** (biological case studies가 보완)

---

### Q4: PRISM 자체가 ESM-2보다 나은지 명확하지 않다 (일부 term에서 ESM-2가 더 좋음)는 점이 약점인가?

**답**: **약점이지만 정직하게 보고했으므로 방어 가능.**

- PRISM-18 > ESM-2-640: training GO와 연관된 12 brain GO terms
- ESM-2-640 > PRISM-18: 무관한 8 brain GO terms
- 이것은 PRISM이 "task-specific representation"임을 보임

**Limitation**: 명시적으로 인정하면 문제없음

**Reject 가능성**: **LOW**

---

### Q5: Nature Methods가 실제로 원하는 "methodological contribution"인가?

**답**: **현재 프레이밍으로는 아니다. 재구성 필요.**

**Nature Methods 기준**:
- "New methods or substantial improvements"
- PRISM MLP는 standard architecture
- 3-layer framework는 tool integration

**진짜 methodological contribution**:
- "Gene-level supervision에서 isoform-specific function learning"
- 이것은 "noisy label learning" 문제의 성공 사례
- Within-gene variance > between-gene variance가 핵심 증거

**권장**: 이 프레이밍으로 전환하면 Nature Methods 가능

**현재 draft의 Reject 가능성**: **60%**
**재구성 후 Reject 가능성**: **40%**

---

*Generated: 2026-06-01*  
*Agent: Devil's Advocate*  
*Target: PRISM + BISECT three-layer framework*  
*Stance: Maximum skepticism, Nature Methods standards*
