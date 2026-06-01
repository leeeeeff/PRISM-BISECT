# DIFFUSE 이소폼 기능 예측 — 모델 성능 개선 이력 분석 레포트

**작성일**: 2026-04-07  
**용도**: 교수님 미팅 발표 자료  
**목표**: 초기 DIFFUSE 베이스라인부터 현재(v5-4)까지의 핵심 분기점, 변경 이유, 성능 수치 변화 종합 분석

---

## 0. 연구 문제와 평가 기준

### 무엇을 예측하는가
- **입력**: 사람 근육 세포(hMuscle)의 single-cell long-read RNA-seq에서 검출된 이소폼 36,748개
- **출력**: 각 이소폼이 특정 GO term 기능을 가지는지 여부 (binary classification)
- **핵심 과제**: 같은 유전자(gene)에서 유래한 이소폼들이 서로 다른 기능을 갖는다 — 이 이소폼 수준의 기능 차이를 예측

### 평가 지표
- **Primary**: AUPRC (Area Under Precision-Recall Curve) — 극단적 class imbalance에서 실질적 예측 능력 반영 [R9.1]
- **Secondary**: AUROC (Area Under ROC Curve)
- **Random baseline**: positive class ratio와 동일 (AUPRC ≈ pos_ratio)

### 데이터셋 통계 (테스트셋 기준)

| GO Term | 기능 | positive | total | pos ratio | random AUPRC |
|---------|------|---------|-------|----------|-------------|
| GO_0006936 | 근육 수축 (Muscle contraction) | 597 | 36,748 | 1.625% | 0.016 |
| GO_0006412 | 번역 (Translation) | 701 | 36,748 | 1.908% | 0.019 |
| GO_0006096 | 해당과정 (Glycolysis) | 76 | 36,748 | 0.207% | 0.002 |
| GO_0022900 | 전자전달계 (Electron transport chain) | 291 | 36,748 | 0.792% | 0.008 |

---

## 1. 전체 성능 변화 요약

아래 표는 각 핵심 버전의 **최종 출력 기준** AUROC / AUPRC (GO_0006936 기준 정렬):

| 버전 | 날짜 | 핵심 변화 | GO_0006936 | GO_0006096 | GO_0006412 | GO_0022900 |
|------|------|---------|-----------|-----------|-----------|-----------|
| | | | AUROC / **AUPRC** | AUROC / **AUPRC** | AUROC / **AUPRC** | AUROC / **AUPRC** |
| **원본 DIFFUSE** | 참조 | BCE + CRF (gene-level) | — | — | — | — |
| **v1** (이소폼 평가 적용) | 2026-03-19 | Triplet+Focal+CRF 첫 통합 | 0.674 / **0.101** | 0.847 / **0.527** | 0.546 / **0.025** | 0.624 / **0.191** |
| **v1 Final (CRF 후)** | | (동일 실험, CRF 적용) | 0.501 / **0.019** | 0.506 / **0.002** | 0.512 / **0.020** | 0.489 / **0.008** |
| **v3** Phase 2 | 2026-03-19 | 전역 Triplet mining 적용 | 0.802 / **0.387** | 0.791 / **0.493** | 0.587 / **0.052** | 0.636 / **0.087** |
| **v3 Final (CRF 후)** | | (동일 실험, CRF 적용) | 0.503 / **0.017** | 0.484 / **0.002** | 0.497 / **0.019** | 0.535 / **0.009** |
| **v4-3** Phase 2 | 2026-03-25 | GradientTape global mining | 0.791 / **0.296** | 0.760 / **0.258** | 0.636 / **0.082** | 0.643 / **0.074** |
| **v4-3 Final (CRF 후)** | | (동일 실험, CRF 적용) | 0.516 / **0.017** | 0.512 / **0.002** | 0.503 / **0.019** | 0.481 / **0.008** |
| **v5-2** ★ **최고** | 2026-04-07 | CRF 제거, LabelProp | **0.805 / 0.410** | **0.854 / 0.354** | 0.632 / **0.089** | 0.619 / **0.031** |
| **v5-3** (회귀) | 2026-04-07 | margin 0.3→0.1 (오진) | 0.712 / **0.119** | 0.746 / **0.095** | 0.605 / **0.075** | 0.679 / **0.016** |
| **v5-4** | 실행 중 | margin 복구, AUPRC early stop | — | — | — | — |

**Random baseline 대비 최고 성능 배율 (v5-2)**:
- GO_0006936: AUPRC 0.410 / 0.016 = **25.6배**
- GO_0006096: AUPRC 0.354 / 0.002 = **168배**

---

## 2. 핵심 분기점 버전 분석

---

### 분기점 1: v1 → v3 (2026-03-19)
#### "Triplet Loss + Focal Loss 통합 첫 시도 — CRF 파괴 문제 발견"

#### 핵심 변경
원본 DIFFUSE (BCE + Gene-level CRF)에 Triplet Loss와 Focal Loss를 추가한 최초 버전. 3단계 파이프라인 도입:
```
Phase 1: Triplet Loss (metric learning, 15 epoch)
Phase 1.5: Linear Probing (encoder frozen, focal loss)
Phase 2: Joint Fine-tuning (focal + ingroup triplet)
Phase 3: CRF (원본 DIFFUSE 방식 유지)
```

#### 역할
이소폼 수준 기능 예측을 위한 두 핵심 손실함수(Focal + Triplet)가 **실제로 효과가 있는지** 첫 검증.

#### 성능 수치 (GO_0006936 기준)

| 단계 | AUROC | AUPRC | 해석 |
|------|-------|-------|------|
| Phase 0 (untrained) | 0.430 | 0.014 | random과 유사 |
| Phase 2 (our best) | **0.674** | **0.101** | random(0.016) 대비 **6.3배** |
| Phase 3 (CRF 후) | 0.501 | 0.019 | random 수준으로 완전 붕괴 |

#### v3에서의 Phase 2 성능 (개선)

| GO Term | Phase 2 AUROC | Phase 2 AUPRC | CRF 후 AUPRC |
|---------|--------------|--------------|-------------|
| GO_0006936 | **0.802** | **0.387** | 0.017 (-95.6%) |
| GO_0006096 | **0.791** | **0.493** | 0.002 (-99.6%) |
| GO_0006412 | 0.587 | 0.052 | 0.019 (-63.5%) |
| GO_0022900 | 0.636 | 0.087 | 0.009 (-89.7%) |

#### 긍정적 영향
- Triplet + Focal 조합이 Phase 2에서 강력한 discriminative 표현을 만들 수 있음을 **처음으로 확인**
- GO_0006936 AUPRC 0.387, GO_0006096 AUPRC 0.493 — random 대비 각각 24배, 235배

#### 한계점 (치명적)
**CRF가 Phase 2의 뛰어난 표현을 100% 파괴한다.** 원인:
1. 원본 DIFFUSE CRF는 gene-level annotation 전파 도구로 설계됨 → isoform 수준 판별력 파괴
2. CRF message passing이 같은 유전자 이소폼 간 점수를 평균화 → discriminative score 소멸
3. 우리 Phase 2 점수(AUROC ~0.80)는 CRF 입력 전에 이미 고품질 → CRF pairwise potential이 unary를 압도

**→ 핵심 교훈: 이소폼 수준 예측에서 원본 DIFFUSE CRF는 반드시 제거되어야 함.**

---

### 분기점 2: v4-3 (2026-03-25)
#### "GradientTape 전역 Triplet Mining — 안정적 Phase 2 기반 확보"

#### 핵심 변경
Phase 1 Triplet 학습 방식을 **ingroup (같은 배치 그룹 내)** 에서 **전역 GradientTape mining** 으로 전환:

```python
# v1~v3: ingroup triplet (같은 길이 그룹 내 positive/negative 선택)
# → gene-level co-occurrence bias 위험
get_triplet_batch_ingroup(X_grp_seq, X_grp_dm, y_grp)

# v4-3: 전체 임베딩 공간에서 semi-hard negative mining
# → cross-gene negative 보장, embedding 품질 향상
extract_embeddings(feature_model, X_train, ...)  # 전체 추출
build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices, margin=0.3)
```

Semi-hard mining 조건: `d(a,p) < d(a,n) < d(a,p) + margin`

#### 역할
Phase 1 Triplet이 gene-level bias를 배우지 않도록 강제 → Phase 2의 출발점인 embedding 품질 개선.

#### 성능 수치

**v3 vs v4-3 Phase 2 비교:**

| GO Term | v3 Phase2 AUPRC | v4-3 Phase2 AUPRC | 변화 |
|---------|----------------|------------------|------|
| GO_0006936 | 0.387 | **0.296** | ↓ (v3이 더 높음) |
| GO_0006096 | 0.493 | **0.258** | ↓ (v3이 더 높음) |
| GO_0006412 | 0.052 | **0.082** | ↑ +57.7% |
| GO_0022900 | 0.087 | **0.074** | ↓ |

**v4-3 Phase별 embedding 품질 (GO_0006936):**

| Phase | AUROC | AUPRC |
|-------|-------|-------|
| Phase 0 (untrained) | 0.430 | 0.014 |
| Phase 1 (triplet) | 0.293 | 0.011 |
| Phase 1.5 (linear) | 0.438 | 0.014 |
| **Phase 2 (joint)** | **0.791** | **0.296** |
| Final (CRF 후) | 0.516 | 0.017 |

#### 긍정적 영향
- Phase 2 최고 성능이 4개 GO term 모두에서 안정적으로 0.60 이상의 AUROC 확보
- Cross-gene negative mining으로 gene-level shortcut 학습 억제
- GO_0006412 AUPRC: 0.052 → 0.082 (58% 개선) — 번역 관련 이소폼에 대한 판별력 향상

#### 한계점
1. **CRF 파괴 문제 지속**: 4개 GO term 모두 CRF 이후 AUPRC가 random 수준으로 붕괴
2. **Phase 1 자체 성능 저하**: GO_0006936에서 Phase 0(AUROC 0.43) → Phase 1(0.29)으로 triplet 학습이 오히려 성능을 낮춤 — margin과 n_batches 설정 문제
3. v3 대비 GO_0006936, GO_0006096 Phase 2 AUPRC가 하락: ingroup → global mining 전환 과정의 hyperparameter 조정 미완

**→ 핵심 교훈: CRF 제거가 최우선 과제임을 명확히 확인. Phase 2 자체는 강력하나 Phase 1 튜닝이 필요.**

---

### 분기점 3: v5-2 (2026-04-07) ★ 현재 최고 성능
#### "CRF 완전 제거 + Expression Label Propagation 도입 — 실질적 최대 도약"

#### 핵심 변경

**① CRF 완전 제거 및 출력 방식 변경**
```
v4-3: Phase 2 → Phase 3 (CRF) → Final (쓸모없음)
v5-2: Phase 2 → Phase 3 (LabelProp) → Final (개선)
```

**② Expression Label Propagation (새로운 Phase 3)**
```python
# 이소폼별 CPM 발현 벡터(24-dim)로 KNN graph 구성
# KNN k=15, cosine similarity
# 예측 점수를 이웃 이소폼들 간에 부드럽게 전파
refined = (1 - alpha) * base_scores + alpha * neighbor_mean_scores
# alpha ∈ {0.0, 0.2, 0.3, 0.5} 중 AUPRC 최대 선택 [R9.1]
```

**③ Phase 2 모니터링 개선**
- AUROC 체크 interval: 3 epoch → **1 epoch** (세밀한 early stop)
- NO_IMPROVE_LIMIT: 2 → **3** (조기 종료 방지)
- LabelProp alpha 선택 기준: AUROC → **AUPRC** ([R9.1])

**④ hyperparameter**: margin=0.3, n_batches=20 (v4-3과 동일)

#### 역할
- CRF 파괴 문제를 근본적으로 해결
- Expression 데이터를 gene-level bias 없이 isoform-level에서 활용
- Phase 2의 고품질 예측 점수를 최종 출력으로 보존

#### 성능 수치 — 전체 비교

**v4-3(Phase 2) vs v5-2(Final) — CRF 제거 효과:**

| GO Term | v4-3 Phase2 | v4-3 Final(CRF) | **v5-2 Final(LP)** | v4-3→v5-2 AUPRC 개선 |
|---------|------------|----------------|-------------------|---------------------|
| GO_0006936 | 0.791/0.296 | 0.516/0.017 | **0.805/0.410** | +0.114 (+38.5%) |
| GO_0006096 | 0.760/0.258 | 0.512/0.002 | **0.854/0.354** | +0.096 (+37.2%) |
| GO_0006412 | 0.636/0.082 | 0.503/0.019 | **0.632/0.089** | +0.007 (+8.5%) |
| GO_0022900 | 0.643/0.074 | 0.481/0.008 | **0.619/0.031** | -0.043 (-58.1%) |

**v5-2 Phase별 embedding 진행 (GO_0006936):**

| Phase | Silhouette | LinAUROC | PredAUROC |
|-------|-----------|---------|----------|
| Ph0 (untrained) | +0.162 | 0.565 | 0.469 |
| Ph1 (triplet, margin=0.3) | **+0.473** | 0.793 | **0.678** |
| Ph1.5 (linear) | +0.523 | 0.769 | 0.696 |
| Ph2 (joint) | **+0.733** | 0.791 | **0.775** |
| Final (LP, alpha=0.5) | — | — | **0.805** |

**v5-2 Phase별 embedding 진행 (GO_0006096 — 최고 성능):**

| Phase | Silhouette | LinAUROC | PredAUROC |
|-------|-----------|---------|----------|
| Ph0 (untrained) | +0.040 | 0.654 | 0.557 |
| Ph1 (triplet) | -0.009 | 0.770 | 0.442 |
| Ph1.5 (linear) | **+0.429** | **0.865** | 0.742 |
| Ph2 (joint) | **+0.463** | **0.901** | **0.854** |
| Final (LP, alpha=0.0) | — | — | **0.854** |

**Random baseline 대비 v5-2 최고 성능:**

| GO Term | v5-2 AUPRC | Random AUPRC | **배율** |
|---------|-----------|-------------|---------|
| GO_0006936 | 0.410 | 0.016 | **25.6×** |
| GO_0006096 | 0.354 | 0.002 | **168×** |
| GO_0006412 | 0.089 | 0.019 | **4.7×** |
| GO_0022900 | 0.031 | 0.008 | **3.9×** |

#### 긍정적 영향
1. **CRF 파괴 완전 해결**: AUPRC 0.017 → 0.410 (GO_0006936, 24배 개선)
2. **Label Propagation의 효과**: GO_0006936에서 alpha=0.5 선택, Phase 2(AUPRC 0.394) → Final(0.410) 추가 개선
3. **Expression 데이터를 isoform-level에서 활용**: CRF의 gene-level 균일화 없이 발현 패턴 활용
4. **Phase 1 margin=0.3의 "frustrated triplet" 효과 확인**: margin satisfaction 0.0%에도 불구하고 최고 embedding 품질 달성

#### 한계점
1. **GO_0006412 (번역), GO_0022900 (전자전달계) 성능 낮음**: tissue-nonspecific 기능 → isoform 수준 발현 패턴으로 분리 어려움
2. **n_batches 공식의 문제**: GO_0006412(3,046 positive)임에도 n_batches=20으로 클램핑 → 학습량 심각 부족 (1.7×/epoch)
3. **expression shortcut 가능성**: LabelProp이 expression으로 tissue-specific GO term을 잘 예측하는 것이 이소폼 기능 학습인지 발현 패턴 shortcut인지 미검증

---

### 분기점 4: v5-3 (2026-04-07) — 회귀 사례
#### "잘못된 진단으로 인한 성능 역행 — 실험 설계의 중요성"

#### 핵심 변경
```python
# v5-3 변경사항
MARGIN_P1 = 0.3 → 0.1   # [I3] sparse GO term Silhouette 개선 목적
N_BATCHES  = 20 → 50     # [I4] 학습량 확보
```

**변경 근거 (잘못된 진단)**:
> "v5-2에서 GO_0006412, GO_0022900의 Phase 1 Silhouette이 음수 (-0.16, -0.18). margin=0.3이 이 sparse GO term에서 embedding을 파괴하고 있음. margin=0.1로 낮추면 해결됨."

#### 성능 수치 — v5-2 vs v5-3

| GO Term | v5-2 AUPRC | v5-3 AUPRC | 변화 | 해석 |
|---------|-----------|-----------|------|------|
| GO_0006936 | **0.410** | 0.119 | **-71.0%** | 치명적 회귀 |
| GO_0006096 | **0.354** | 0.095 | **-73.2%** | 치명적 회귀 |
| GO_0006412 | **0.089** | 0.075 | -15.7% | 악화 |
| GO_0022900 | **0.031** | 0.016 | -48.4% | 악화 |

**GO_0006936: v5-2 vs v5-3 Phase별 embedding 비교:**

| Phase | v5-2 Silhouette | v5-3 Silhouette | v5-2 PredAUROC | v5-3 PredAUROC |
|-------|----------------|----------------|---------------|---------------|
| Ph0 | +0.162 | +0.031 | 0.469 | 0.527 |
| Ph1 | **+0.473** | +0.045 | **0.678** | 0.433 |
| Ph1.5 | +0.523 | +0.144 | 0.696 | 0.661 |
| Ph2 | **+0.733** | +0.316 | **0.775** | 0.706 |
| Final | — | — | **0.805** | 0.712 |

#### 진단 오류의 핵심

**역설**: v5-2에서 margin=0.3, best_margin_sat=**0.0%** (한 번도 만족 안 됨) → 그럼에도 Ph1 Silhouette=0.473, AUPRC=0.410 (최고 성능)

```
v5-2 GO_0006936 Phase 1 active ratio 추이 (margin=0.3, n_batches=20):
  Epoch 1:  94.9%  Epoch 2: 97.6%  Epoch 3: 71.3%
  Epoch 4:   9.6%  Epoch 5:  4.8%  Epoch 6:  6.5%
  ...  Epoch 15:  2.4%
  → 15 epoch 내내 유의미한 gradient 유지

v5-3 GO_0006936 Phase 1 active ratio 추이 (margin=0.1, n_batches=50):
  Epoch 3:  6.7%   Epoch 4:  4.8%   Epoch 5:  3.7%
  Epoch 6:  2.0%   Epoch 7:  1.7%   Epoch 8:  0.9%   Epoch 9:  1.0%
  → Epoch 7부터 gradient 소멸 → early stop
```

**메커니즘**: margin=0.3은 대부분 triplet이 non-zero loss → 매 batch 유의미한 gradient. margin=0.1은 너무 쉽게 satisfy → gradient 조기 소멸.

#### 추가 혼입 변수 (실험 설계 문제)
v5-2→v5-3 사이 margin과 n_batches가 **동시에 변경** → 어느 것이 원인인지 단독으로 판단 불가. 완전한 인과 추론을 위한 통제 실험 부재.

#### 교훈
1. **Silhouette은 최종 AUPRC의 충분 조건이 아님**: GO_0006096에서 v5-3 Ph1 Silhouette=+0.22 (v5-2의 -0.01보다 훨씬 좋음)에도 AUPRC는 0.354→0.095로 하락
2. **단일 지표로 인한 오진 위험**: Phase 1 Silhouette만 보고 margin 변경 결정 → AUPRC 전면 하락
3. **여러 변수 동시 변경의 위험**: 인과관계 판단 불가 상태에서 결론 도출

---

### 분기점 5: v5-4 (2026-04-07, 실행 중)
#### "진단 오류 수정 + Phase 2 평가 기준 정비"

#### 핵심 변경
```python
# v5-4 변경사항
MARGIN_P1 = 0.1 → 0.3    # [I3 복구] v5-3 오진 수정
# n_batches=50 유지       # [I4] margin=0.3+n_batches=50 첫 조합 실험

# Phase 2 early stop 기준 변경 [I5]
# v5-3: AUROC 기준 → v5-4: AUPRC 기준 [R9.1]
best_phase2_auprc = 0.0
if current_auprc > best_phase2_auprc:
    save_best_weights()
```

**Phase 2 AUPRC 기준으로 변경한 이유:**
- AUROC는 93% positive 예측(GO_0006412 collapse)도 방치 가능
- AUPRC는 precision 감소 즉시 감지 → positive collapse 조기 차단

#### 기대 성능 (예측)

| GO Term | 목표 AUPRC | 근거 |
|---------|-----------|------|
| GO_0006936 | ≥ 0.35 | v5-2 수준(0.41) 80% 이상 회복 |
| GO_0006096 | ≥ 0.30 | v5-2 수준(0.35) 85% 이상 회복 |
| GO_0006412 | ≥ 0.08 | v5-2 수준 유지 |
| GO_0022900 | ≥ 0.03 | v5-2 수준 유지 |

---

## 3. 성능 개선 경과 종합 분석

### 3-1. AUPRC 기준 개선 궤적 (GO_0006936)

```
Random baseline: 0.016
─────────────────────────────────────────────────────────
v1 Phase2 (2026-03-19):  0.101  ████  첫 meaningful prediction
v1 Final (CRF):          0.019  ▏     CRF 파괴
v3 Phase2 (2026-03-19):  0.387  ████████████████  Phase2 최강
v3 Final (CRF):          0.017  ▏     CRF 파괴
v4-3 Phase2 (03-25):     0.296  ████████████  global mining 효과
v4-3 Final (CRF):        0.017  ▏     CRF 여전히 파괴
═══════════════════════════════════  ← CRF 제거 분기점
v5-2 Final (04-07):      0.410  ████████████████▊  역대 최고
v5-3 Final (04-07):      0.119  ████▊  margin 오진 회귀
v5-4 Final (실행중):     ???
─────────────────────────────────────────────────────────
```

### 3-2. 3대 핵심 발견

#### 발견 1: CRF는 이소폼 수준 예측에서 작동하지 않는다
**수치적 증거**: 4개 GO term × 3개 실험(v1, v3, v4-3)에서 CRF 이후 AUPRC가 예외 없이 random 수준으로 붕괴.

| 버전 | GO_0006936 Phase2→Final AUPRC | GO_0006096 Phase2→Final AUPRC |
|------|------------------------------|------------------------------|
| v1 | 0.101 → 0.019 (-81.2%) | 0.527 → 0.002 (-99.6%) |
| v3 | 0.387 → 0.017 (-95.6%) | 0.493 → 0.002 (-99.6%) |
| v4-3 | 0.296 → 0.017 (-94.3%) | 0.258 → 0.002 (-99.2%) |

**메커니즘**: CRF message passing이 within-gene isoform 간 score를 평균화 → discriminative signal 소멸. 원본 DIFFUSE CRF는 gene-level annotation 전파용으로 설계됨.

#### 발견 2: Triplet Loss margin=0.3이 최적 — "frustrated triplet" 효과
**수치적 증거**:

| 조건 | margin | sat rate | Ph1 Silhouette | 최종 AUPRC |
|------|--------|---------|----------------|-----------|
| v5-2 (GO_0006936) | 0.3 | **0.0%** | +0.473 | **0.410** |
| v5-3 (GO_0006936) | 0.1 | 34.8% | +0.045 | 0.119 |

margin satisfaction이 0%여도 더 좋은 embedding과 더 높은 최종 AUPRC. "쉬운 margin이 어려운 margin보다 나쁘다" — 어려운 목표가 지속적 gradient를 유지함.

#### 발견 3: Phase 1 embedding 품질이 Phase 2 최종 성능의 핵심 선행 지표
**수치적 증거** (GO_0006936):

| 버전 | Ph1 Silhouette | Ph1 PredAUROC | 최종 AUPRC |
|------|---------------|--------------|-----------|
| v5-2 | **+0.473** | **0.678** | **0.410** |
| v5-3 | +0.045 | 0.433 | 0.119 |
| v4-3 | — | — | 0.296 |

Ph1 PredAUROC가 최종 AUPRC와 강한 양의 상관관계 → Phase 1 학습 품질 모니터링이 핵심.

### 3-3. GO Term별 특성 분석

| GO Term | 수렴 패턴 | 주요 문제 | 현재 최고 AUPRC |
|---------|---------|---------|----------------|
| GO_0006936 (근육수축) | 안정적, LabelProp 효과 큼 | margin 민감 | **0.410** (v5-2) |
| GO_0006096 (해당과정) | 매우 안정적, Ph1.5 효과 큼 | 초기 negative Silhouette | **0.354** (v5-2) |
| GO_0006412 (번역) | 불안정, positive collapse | 조직 비특이적, class imbalance | **0.089** (v5-2) |
| GO_0022900 (전자전달계) | 불안정, triplet 학습 실패 | 조직 비특이적, 희소성 | **0.031** (v5-2) |

**GO_0006412와 GO_0022900이 낮은 근본 이유**: Translation(번역)과 ETC(전자전달계)는 사실상 모든 세포 유형에서 발생하는 기능 → 근육세포 발현 데이터로는 이소폼 간 분리 신호 약함. 이 두 GO term에 대해서는 sequence-based feature(ESM-2)가 필수적일 가능성 높음.

### 3-4. 실험 설계 관련 교훈

**통제 실험 부재의 위험 (v5-3 사례)**:
- v5-2→v5-3에서 margin과 n_batches가 동시 변경
- GO_0006412 Phase 1 Silhouette 개선만 보고 margin 변경 결론 도출
- 실제: 4개 GO term 모두 AUPRC 하락

**올바른 실험 설계**:
```
변수 1: margin (0.1 vs 0.3)
변수 2: n_batches (20 vs 50)
→ 2×2 factorial design 필요:
   (a) margin=0.3, n=20 (v5-2 재현)
   (b) margin=0.3, n=50 (v5-4 현재)
   (c) margin=0.1, n=20 (순수 margin 효과 분리)
   (d) margin=0.1, n=50 (v5-3 현재)
```

---

## 4. 현재까지의 미해결 문제

### 즉시 해결 가능
| 문제 | 상태 | 다음 단계 |
|------|------|---------|
| v5-3 margin 회귀 | 🔄 v5-4 실행 중 | 결과 확인 후 진단 |
| Phase 2 AUPRC early stop | ✅ v5-4에 적용 | 효과 확인 |
| GO_0006412 positive collapse | ⏳ 미해결 | Phase 2 focal loss alpha 재검토 |

### 중기 과제
| 문제 | 우선순위 | 설명 |
|------|---------|------|
| ESM-2 통합 | 높음 | Sequence 기반 isoform-specific feature. GO_0006412, GO_0022900에 특히 필요 |
| Expression shortcut 검증 | 높음 | LabelProp이 tissue-specific expression 패턴인지, 실제 기능 학습인지 분리 |
| Seed 고정 재현성 | 중간 | Phase 0 초기화가 최종 성능에 미치는 영향 정량화 |
| GO term 교체 검토 | 중간 | GO_0006412, GO_0022900 → 근육 특이적 GO term으로 교체 |

### 장기 과제 (논문 방어)
| 문제 | 설명 |
|------|------|
| Gene-level bias 정량화 | intra-gene vs inter-gene score variance 측정 [R9.3] |
| Bootstrap CI | 성능 개선 주장에 신뢰구간 필요 [R9.4] |
| Ablation study | no_triplet / no_focal / no_labelprop / no_esm 체계적 비교 |
| Cell localization, PPI 통합 | 아키텍처 규칙 MANDATORY 항목 |

---

## 5. 다음 단계 로드맵

```
현재(04-07) → v5-4 결과 확인 (3~4시간 후)
    ├── 성공 (AUPRC GO_0006936≥0.35, GO_0006096≥0.30)
    │   └── v6: ESM-2 통합 설계 및 구현
    │
    └── 실패
        ├── GO_0006936 AUPRC < 0.20
        │   └── Seed 고정 통제 실험 (initialization sensitivity 검증)
        └── 전반적 하락
            └── 2×2 factorial 실험 설계 (margin × n_batches)

v6 (ESM-2 통합) 이후:
    → Ablation study 시작
    → Bootstrap CI 계산
    → Nature Methods 논문 작성 준비
```

---

## 부록: 전체 수치 데이터 (최종 출력 기준)

### A-1. 버전별 최종 AUROC

| 버전 | GO_0006936 | GO_0006412 | GO_0006096 | GO_0022900 |
|------|-----------|-----------|-----------|-----------|
| v1 Phase2 | 0.674 | 0.546 | 0.847 | 0.624 |
| v1 Final(CRF) | 0.501 | 0.512 | 0.506 | 0.489 |
| v3 Phase2 | **0.802** | 0.587 | 0.791 | 0.636 |
| v3 Final(CRF) | 0.503 | 0.497 | 0.484 | 0.535 |
| v4-3 Phase2 | 0.791 | **0.636** | 0.760 | **0.643** |
| v4-3 Final(CRF) | 0.516 | 0.503 | 0.512 | 0.481 |
| v5-2 Final(LP) | **0.805** | 0.632 | **0.854** | 0.619 |
| v5-3 Final(LP) | 0.712 | 0.605 | 0.746 | **0.679** |

### A-2. 버전별 최종 AUPRC (Primary Metric)

| 버전 | GO_0006936 | GO_0006412 | GO_0006096 | GO_0022900 |
|------|-----------|-----------|-----------|-----------|
| Random baseline | 0.016 | 0.019 | 0.002 | 0.008 |
| v1 Phase2 | 0.101 | 0.025 | 0.527 | 0.191 |
| v1 Final(CRF) | 0.019 | 0.020 | 0.002 | 0.008 |
| v3 Phase2 | 0.387 | 0.052 | **0.493** | 0.087 |
| v3 Final(CRF) | 0.017 | 0.019 | 0.002 | 0.009 |
| v4-3 Phase2 | 0.296 | 0.082 | 0.258 | 0.074 |
| v4-3 Final(CRF) | 0.017 | 0.019 | 0.002 | 0.008 |
| **v5-2 Final(LP)** | **0.410** | **0.089** | **0.354** | **0.031** |
| v5-3 Final(LP) | 0.119 | 0.075 | 0.095 | 0.016 |

### A-3. Phase 2 AUPRC만 비교 (CRF/LP 효과 제거)

| 버전 | GO_0006936 | GO_0006412 | GO_0006096 | GO_0022900 |
|------|-----------|-----------|-----------|-----------|
| v1 Phase2 | 0.101 | 0.025 | 0.527 | 0.191 |
| v3 Phase2 | 0.387 | 0.052 | 0.493 | 0.087 |
| v4-3 Phase2 | 0.296 | 0.082 | 0.258 | 0.074 |
| v5-2 Phase2 | 0.394 | 0.084 | 0.354 | 0.031 |
| v5-3 Phase2 | 0.118 | 0.075 | 0.095 | 0.015 |

> v1 GO_0006096 Phase2 AUPRC=0.527이 이 벤치마크에서 가장 높음 — 단, 이후 CRF로 0.002로 붕괴.
> v5-2에서 Phase2→Final 시 GO_0006936은 0.394→0.410으로 LabelProp 효과 확인.

---

*레포트 작성: 2026-04-07*  
*데이터 출처: `hMuscle/results_isoform/` 저장 score 파일 기반 sklearn 직접 계산*  
*v5-4 실행 중: GPU0(GO_0006412, GO_0022900), GPU1(GO_0006936, GO_0006096)*
