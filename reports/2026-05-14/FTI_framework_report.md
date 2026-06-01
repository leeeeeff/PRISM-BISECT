# FTI Framework 구현 보고서
**날짜**: 2026-05-14  
**저자**: DIFFUSE 프로젝트  
**목적**: SwissProt Training Data Quality 정량화 및 GO term별 시각화

---

## 1. 배경 및 목적

### 1.1 발견된 문제

2×2 ablation (SwissProt × Dim) 실험에서 GO term마다 SwissProt 학습 데이터의 효과가 **정반대**임이 확인됨.

| GO term | D256/P3-256 비율 (FTI) | SP 효과 |
|---------|----------------------|--------|
| GO:0006096 (glycolysis) | 1.87 | **필수** |
| GO:0003774 (myosin) | 1.03 | 약간 유익 |
| GO:0006941 (contraction) | 1.02 | 중립 |
| GO:0007204 (Ca²⁺) | 0.63 | 해로움 |
| GO:0030017 (sarcomere) | 0.48 | **매우 해로움** |

단순 "보존성" 1축으로는 GO:0007204 (보존됐지만 SP 해로움)를 설명 불가.  
→ **2D 프레임워크 필요**: Taxonomic Breadth (TBS) × Tissue Context Specificity (TCS)

### 1.2 논문 기여 목표

> "We introduce a two-dimensional framework for predicting cross-species training data quality (FTI) from GO term biological properties (TBS, TCS), enabling principled dataset curation for isoform function prediction."

---

## 2. FTI (Functional Transferability Index)

### 2.1 정의

SwissProt 데이터가 해당 GO term 예측에 실제로 도움이 되는 정도의 경험적 측정값.

```
FTI(GO, dim) = AUPRC(SP✓ model, human test set) / AUPRC(SP✗ model, human test set)
```

- FTI > 1: SwissProt 유익
- FTI = 1: 중립
- FTI < 1: SwissProt 해로움

### 2.2 실험 결과 (256d 기준)

```
FTI_256(GO:0006096) = 0.8331 / 0.4445 = 1.875
FTI_256(GO:0003774) = 0.5982 / 0.5830 = 1.026
FTI_256(GO:0006941) = 0.1567 / 0.1540 = 1.018
FTI_256(GO:0007204) = 0.1947 / 0.3109 = 0.626
FTI_256(GO:0030017) = 0.1366 / 0.2836 = 0.482
```

### 2.3 dim별 FTI 추이 (증폭 현상)

```
FTI_64(GO:X)  = AUPRC(v8b) / AUPRC(P3)     [64d 기준]
FTI_256(GO:X) = AUPRC(D256) / AUPRC(P3-256) [256d 기준]
FTI_512(GO:X) = AUPRC(D512) / AUPRC(P3-512) [512d 기준, 예정]
```

SP 효과는 dim이 커질수록 증폭됨:
- FTI > 1 인 GO term: dim 증가 → FTI 더 커짐
- FTI < 1 인 GO term: dim 증가 → FTI 더 작아짐 (더 해로워짐)

→ **"SP-aware dimension scaling"**: SP 효과에 따라 dim 전략을 GO별로 달리 적용해야 함.

---

## 3. TBS (Taxonomic Breadth Score)

### 3.1 개념

GO term에 대한 SwissProt annotation이 얼마나 넓은 taxonomic 범위에 분포하는지 측정.  
범위가 넓을수록 → 해당 기능이 진화적으로 보존됨 → SP 전이 신뢰도 높음.

### 3.2 수식

```
TBS(GO) = |{k ∈ K : ∃x ∈ SP+(GO) s.t. taxon(x) ∈ k}| / |K|

K = {Bacteria, Archaea, Fungi, Viridiplantae, Invertebrata, Vertebrata}
|K| = 6

SP+(GO) = {x : x ∈ SwissProt, GO ∈ annotations(x)}
taxon(x) = kingdom of organism that protein x belongs to
```

### 3.3 구현 방법

**데이터 소스**: `/home/welcome1/sw1686/DIFFUSE/hMuscle/data/raw_data/data/annotations/swissprot_annotations.txt`

**형식**: `GENE_SPECIES\tGO:XXXXXXX\tGO:XXXXXXX\t...`

**Species suffix → Kingdom 매핑 (UniProt entry name 규칙)**:

| Kingdom | UniProt suffix 예시 |
|---------|-------------------|
| Vertebrata | `_MOUSE` `_RAT` `_BOVIN` `_PIG` `_CHICK` `_XENLA` `_DANRE` `_MACMU` `_PANTR` 등 |
| Invertebrata | `_DROME` `_CAEEL` `_SCHMA` `_LOTGI` `_STRPU` `_CIOIN` 등 |
| Fungi | `_YEAST` `_SCHPO` `_CANAL` `_NEUCR` `_ASPFU` `_ASPNI` 등 |
| Viridiplantae | `_ARATH` `_ORYSJ` `_MAIZE` `_SOLTU` `_TOBAC` `_SOYBN` 등 |
| Bacteria | `_ECOLI` `_BACSU` `_STAAU` `_STRCO` `_MYCTU` `_SALTY` 등 |
| Archaea | `_METJA` `_PYRAE` `_SULSO` `_HALOAR` `_ARCFU` 등 |

**알고리즘**:
```python
for go_term in GO_TERMS:
    sp_proteins = [p for p in swissprot if go_term in annotations[p]]
    kingdoms_present = set()
    for prot in sp_proteins:
        species = prot.split('_')[-1]  # e.g., MOUSE, YEAST
        kingdom = species_to_kingdom(species)
        kingdoms_present.add(kingdom)
    TBS[go_term] = len(kingdoms_present) / 6
```

### 3.4 예측값 (문헌 기반 추정)

| GO term | 예상 Kingdoms | 예상 TBS |
|---------|-------------|---------|
| GO:0006096 (glycolysis) | Bacteria✓ Archaea✓ Fungi✓ Viridiplantae✓ Invertebrata✓ Vertebrata✓ | ~1.00 |
| GO:0003774 (myosin motor) | Fungi✓ Viridiplantae✓ Invertebrata✓ Vertebrata✓ | ~0.67 |
| GO:0006941 (striated contraction) | Invertebrata✓ Vertebrata✓ | ~0.33 |
| GO:0007204 (Ca²⁺ increase) | Fungi✓ Viridiplantae✓ Invertebrata✓ Vertebrata✓ | ~0.67 |
| GO:0030017 (sarcomere org.) | Invertebrata✓ Vertebrata✓ | ~0.33 |

**주목**: GO:0007204와 GO:0030017이 TBS만으로는 구분 안 됨 → TCS 필요

---

## 4. TCS (Tissue Context Specificity)

### 4.1 개념

해당 GO term의 human positive 단백질들이 얼마나 tissue-specific하게 발현되는지 측정.  
특히 **skeletal muscle 맥락**에서의 특이성이 핵심.  
TCS가 높을수록 → 근육 특이적 기능 → 다른 tissue의 SP 데이터가 noise가 됨.

### 4.2 수식 (Tau Index)

Yanai et al. (2005) 방법론을 tissue specificity 측정에 적용:

```
τ(gene) = Σᵢ₌₁ᴺ (1 - x̂ᵢ) / (N - 1)

x̂ᵢ = xᵢ / max(x₁, ..., xₙ)    # 조직 i에서의 정규화 발현량
N = 조직 수 (GTEx: 54개 또는 HPA 기준)
```

- τ = 0: 모든 조직에서 동일 발현 (ubiquitous)
- τ = 1: 단 하나의 조직에서만 발현 (perfectly tissue-specific)

**GO term-level TCS**:
```
TCS(GO) = mean_{g ∈ Human+(GO)} [τ(g)]

또는 Weighted version:
TCS(GO) = mean_{g ∈ Human+(GO)} [τ(g) × w(g)]
w(g) = skeletal_muscle_expr(g) / total_expr(g)  # 근육발현 가중치
```

### 4.3 Skeletal Muscle Specificity Index (SMSI)

TCS의 변형: 전체 tissue specificity가 아닌 skeletal muscle 특이성에 집중.

```
SMSI(gene) = expr_skeletal_muscle(gene) / mean_all_tissues(gene)

TCS_SMSI(GO) = mean_{g ∈ Human+(GO)} [SMSI(g)]
```

- SMSI >> 1: 근육특이적
- SMSI ≈ 1: ubiquitous
- SMSI < 1: 근육에서 낮게 발현

### 4.4 구현 방법

**Option A: Human Protein Atlas (HPA) API**
- Endpoint: `https://www.proteinatlas.org/api/search_download.php?search=...`
- 또는 bulk download: `https://www.proteinatlas.org/download/rna_tissue_consensus.tsv.zip`
- 제공 데이터: 54개 조직 × 각 단백질의 RNA consensus score

**Option B: GTEx v8 bulk RNA-seq**
- `https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz`
- 54개 tissue × 56,200 genes의 median TPM

**알고리즘**:
```python
for go_term in GO_TERMS:
    human_positives = get_human_positives(go_term, human_annotations)
    # gene_symbol → Ensembl ID 변환
    # HPA/GTEx에서 발현 데이터 조회
    tau_values = [compute_tau(gene, expression_matrix) for gene in human_positives]
    smsi_values = [compute_smsi(gene, expression_matrix) for gene in human_positives]
    TCS[go_term] = np.mean(tau_values)
    TCS_SMSI[go_term] = np.mean(smsi_values)
```

---

## 5. 2D FTI Prediction Model

### 5.1 최종 예측 모델

```
FTI(GO) = β₀ + β₁·TBS + β₂·TCS + β₃·(TBS × TCS) + ε

예측 방향:
  β₁ > 0: TBS 높음 → SP 유익 (보존된 기능)
  β₂ < 0: TCS 높음 → SP 해로움 (tissue-specific)
  β₃ < 0: TBS×TCS 교호 음수 (조직특이적이면 보존성도 도움 안 됨)
```

**현재 5개 GO term의 회귀 적합 계획** (P3-512 완료 후 FTI_512 추가하여 n=10 → 15개):

### 5.2 왜 이 모델이 GO:0007204를 설명하는가?

GO:0007204: TBS=0.67 (높음) + TCS=높음  
→ β₃·(TBS×TCS)의 음의 기여가 β₁·TBS의 양의 기여를 상쇄  
→ 결과적으로 FTI < 1 예측 → 실험 결과(0.63)와 일치

---

## 6. 시각화 계획

### 6.1 Figure 1: FTI Landscape Plot (핵심 그림)

```
FTI
2.0 │★ GO:0006096 (glycolysis, n=32)
    │  [Tier 1: SP Required]
1.5 │
    │
1.0 ├─────────────────────────────── [중립선 FTI=1]
    │      ○ GO:0003774    ○ GO:0006941
0.8 │        (myosin,n=102) (contraction,n=81)
    │       [Tier 2: SP Neutral]
0.6 │                    ● GO:0007204 (Ca²⁺,n=217)
    │                    ● GO:0030017 (sarcomere,n=129)
0.4 │                    [Tier 3: SP Harmful]
    └──────────────────────────────────────
    TBS  0.33          0.67          1.00
  [tissue-specific]            [pan-kingdom]
```

**점 크기**: Human positive 수 (n=32~217)  
**점 색상**: FTI 값 (연속 색상, red<1<blue)  
**배경 색상**: Tier 영역 (Tier 1/2/3 gradation)  
**오차 막대**: 64d/256d/512d 3개 측정값으로부터의 FTI 분산

### 6.2 Figure 2: 2D TBS × TCS Scatter (메커니즘 설명)

```
TCS(↑ = tissue-specific)
1.0 │              ● GO:0007204  ● GO:0030017
    │              [Tier 3: SP Harmful]
    │
0.5 │
    │                  ○ GO:0006941
    │     ○ GO:0003774
    │     [Tier 2: SP Neutral]
0.0 │                              ★ GO:0006096
    └──────────────────────────────────────────
    TBS  0.0       0.5             1.0
  [narrow]                      [broad]

배경 4분면:
- 우하단 (TBS↑, TCS↓): Tier 1 영역 (파랑)
- 좌상단 (TBS↓, TCS↑): Tier 3 영역 (빨강)
- 좌하단 (TBS↓, TCS↓): Tier 2 영역 (회색)
- 우상단 (TBS↑, TCS↑): Tier 3 영역 (빨강) ← GO:0007204 위치
```

**포인트**: 우상단이 Tier 3인 것이 핵심 인사이트.  
보존된 기능이라도 tissue-specific context면 SP가 해롭다.

### 6.3 Figure 3: dim별 FTI 증폭 패턴

```
FTI
2.0 │                              ● GO:0006096
    │                          ●
1.5 │                      ●
    │
1.0 ├─────────────────────────────────────────
    │              ○─○─○ GO:0003774, GO:0006941
0.8 │
    │  ●
0.6 │      ●       GO:0007204
    │          ●
0.5 │  ●
    │      ●       GO:0030017
0.4 │          ●
    └──────────────────────────────────────
       64d     256d    512d

선 색상: FTI > 1 파랑, FTI < 1 빨강
→ dim 증가 시 SP 효과가 증폭됨을 직접 시각화
```

### 6.4 Figure 4 (보조): SwissProt Dependency Anatomy

```
GO:0006096 ████████████████████████████ 87.6% SP dependency (human+: 32)
GO:0030017 ██████████████████████ 77.9% (human+: 129)
GO:0003774 ████████████████████ 79.6% (human+: 102)
GO:0006941 ████████████████████ 71.5% (human+: 81)
GO:0007204 ██████████████████ 71.4% (human+: 217)
```

SP dependency와 FTI의 관계:
- GO:0006096: SP dependency 87.6%, FTI 1.87 → 많이 의존하며 유익
- GO:0030017: SP dependency 77.9%, FTI 0.48 → 많이 의존하나 해로움

→ SP dependency 자체는 FTI를 예측하지 못함. TBS/TCS가 필요.

---

## 7. 구현 로드맵

### Phase A: TBS 정량화 (오늘, ~2시간)

**입력**: `swissprot_annotations.txt`  
**출력**: `tbs_results.json`, TBS vs FTI scatter plot

```python
# tbs_quantification.py
# 1. swissprot_annotations.txt 파싱
# 2. GO term별 protein 목록 추출
# 3. species suffix → kingdom 매핑
# 4. TBS = kingdoms / 6
# 5. 결과 저장 및 FTI vs TBS scatter 생성
```

### Phase B: TCS 정량화 (오늘, ~3시간)

**입력**: `human_annotations.txt`, HPA 또는 GTEx 발현 데이터  
**출력**: `tcs_results.json`, 2D TBS×TCS plot

```python
# tcs_quantification.py
# 1. human_annotations.txt에서 GO term별 positive 유전자 추출
# 2. HPA rna_tissue_consensus.tsv 다운로드 (또는 API)
# 3. 각 유전자의 tau index 계산
# 4. Skeletal muscle specificity index (SMSI) 계산
# 5. GO term별 TCS = mean(tau) 계산
# 6. 2D scatter 생성
```

### Phase C: 통합 시각화 (Phase A+B 완료 후)

**출력**: Figure 1~4 (논문 제출용 PNG/PDF)

---

## 8. 기대 수치 (검증 후 업데이트 예정)

| GO term | TBS (예상) | TCS (예상) | FTI_256 (실측) |
|---------|-----------|-----------|--------------|
| GO:0006096 | 0.83~1.00 | 0.1~0.3 | 1.875 |
| GO:0003774 | 0.50~0.67 | 0.2~0.4 | 1.026 |
| GO:0006941 | 0.17~0.33 | 0.5~0.7 | 1.018 |
| GO:0007204 | 0.50~0.67 | 0.6~0.8 | 0.626 |
| GO:0030017 | 0.17~0.33 | 0.7~0.9 | 0.482 |

**핵심 검증 목표**:
1. GO:0006096 TBS > GO:0030017 TBS (보존성 차이 확인)
2. GO:0007204 TCS > GO:0006096 TCS (tissue-specificity 차이 확인)
3. TBS + TCS 조합이 FTI를 R²>0.85 이상으로 예측 (5개 GO term 기준)

---

## 9. 논문 제출 전 필요 추가 작업 (Future)

1. GO term 15~20개로 확장 (검증 세트)
2. TBS 정량화에 evidence code 가중치 추가 (EXP > ISS > IEA)
3. TCS에 단백질 수준 발현 데이터 추가 (HPA protein score)
4. FTI 예측 모델 leave-one-GO-out 교차 검증
5. 다른 tissue context (liver, brain)에서 패턴 재현 확인

---

*Script files: `tbs_quantification.py`, `tcs_quantification.py`, `fti_visualization.py`*  
*Output dir: `/home/welcome1/sw1686/DIFFUSE/reports/2026-05-14/`*
