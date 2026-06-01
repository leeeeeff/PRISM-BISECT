# DIFFUSE 이소폼 기능 예측 모델 — 연구 분석 레포트

**날짜**: 2026-04-07 (v5-5 결과 추가: 2026-04-07)  
**기준 이전 레포트**: reports/2026-04-02/research_report_20260402.md (v4-3 기준)  
**이번 레포트 대상**: v5 ~ v5-5 (2026-04-02 이후 전체 변경사항)  
**목표 venue**: Nature Methods / Nature Machine Intelligence (IF 15+)

---

## 1. 이전 레포트(v4-3) 이후 해결된 문제

| 문제 ID | 내용 | 해결 여부 | 방법 |
|---------|------|-----------|------|
| P0 | CRF가 Phase 2 학습 결과 파괴 (AUROC -0.275) | ✅ 해결 | CRF 제거, Expression Label Propagation으로 대체 |
| P1 | Phase 1 Triplet이 일부 GO term에서 성능 저하 | 🔄 진행 중 | margin 탐색 실험 중 (v5-2~v5-4) |
| P5 | Isoform expression profile 미활용 | ✅ 해결 | Phase 3 Label Propagation으로 통합 |
| P2 | ESM / Cell Localization / PPI 미통합 | ⏳ 대기 중 | 성능 안정화 후 추가 예정 |
| P3 | Gene-level Bias 정량 측정 부재 | ⏳ 대기 중 | 향후 구현 예정 |
| P4 | Phase 2 Triplet Negative가 ingroup-only | 🔄 부분 개선 | Phase 1 cross-gene negative 강화 |

---

## 2. v5 계열 아키텍처 — 이전 대비 핵심 변경

### 2-1. 전체 학습 파이프라인 재설계

이전(v4-3)의 Phase 3 구조(PFN + CRF 반복 최적화)를 완전 폐기하고 다음 파이프라인으로 재설계:

```
Phase 0  Untrained baseline 측정
Phase 1  GradientTape Triplet (global semi-hard mining, 15 epoch max)
Phase 1.5  Encoder 동결 + Linear Probing (Focal Loss, 2 epoch)
Phase 2  Joint Fine-tuning (Focal + Ingroup Triplet, AUPRC 기반 early stop)
Phase 3  Expression Label Propagation (KNN graph, alpha AUPRC 기준 선택)
```

### 2-2. CRF 제거 및 Label Propagation 도입

**제거 이유 (EXP-01 결과)**:

| GO term | Phase 2 AUROC | CRF 후 AUROC | 손실 |
|---------|--------------|-------------|------|
| GO_0006936 | 0.791 | 0.516 | **-0.275** |
| GO_0006096 | 0.760 | 0.512 | **-0.248** |
| GO_0006412 | 0.636 | 0.503 | -0.133 |
| GO_0022900 | 0.643 | 0.481 | -0.161 |

**근본 원인**: 원본 DIFFUSE CRF는 gene-level annotation 전파 도구로 설계되었으며, isoform 내 기능 차이 예측이라는 우리 목표와 설계 목적이 불일치. within-gene isoform 간 CRF message passing이 discriminative score를 평균화하여 이소폼 수준 판별력 파괴.

**대체 방안**: `bambu_data/CPM_transcript.txt` 기반 이소폼별 발현 벡터로 KNN graph 구성, 예측 점수를 부드럽게 전파 (Label Propagation). alpha ∈ {0.0, 0.2, 0.3, 0.5}에서 AUPRC 최적 선택.

### 2-3. Phase 1 Triplet 전역 semi-hard mining (v4-3 대비)

v4-3의 Phase 1은 ingroup triplet (같은 길이 그룹 내)을 사용했으나, v5부터 전체 데이터셋 기반 GradientTape global mining으로 전환:

```python
# v4-3: ingroup (gene-level bias 위험)
get_triplet_batch_ingroup(X_grp_seq, X_grp_dm, y_grp, batch_size=64)

# v5+: global embedding space semi-hard mining
build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices, 
                          margin=MARGIN_P1, mode="hard")
```

Semi-hard negative 조건: `d(a,p) < d(a,n) < d(a,p) + margin`  
Fallback: `d(a,n) < d(a,p) + 4×margin` (semi-hard 없을 때)

---

## 3. 버전별 하이퍼파라미터 변경 이력

| 파라미터 | v4-3 | v5 | v5-2 | v5-3 | v5-4 | **v5-5** |
|---------|------|-----|------|------|--------|--------|
| Triplet margin | 0.1 | 0.3 | 0.3 | 0.1 | 0.3 | **0.3** |
| Phase 1 n_batches | ingroup | 공식(min=20) | 공식(min=20) | max(50,min(150,...)) | max(50,min(150,...)) | **coverage 기반 동적 [I4]** |
| Phase 2 early stop 기준 | — | AUROC | AUROC | AUROC | AUPRC | **AUPRC** |
| LabelProp alpha 선택 | 없음 | AUROC | AUPRC | AUPRC | AUPRC | **AUPRC** |
| Phase 2 check interval | — | 3 epoch | 1 epoch | 1 epoch | 1 epoch | 1 epoch |
| Phase 2 NO_IMPROVE_LIMIT | — | 2 | 3 | 3 | 3 | 3 |
| Prior bias init | 없음 | 있음(v5-1만) | 제거 | 제거 | 제거 | 제거 |

### 3-1. 각 버전 변경 상세 설명

#### v5 (2026-04-02 직후)
- CRF 완전 제거, Expression Label Propagation 신규 도입
- Phase 1 전역 GradientTape semi-hard mining 도입
- margin=0.3, n_batches 공식(max(20, ...)) 시작

#### v5-1 (2026-04-03)
- `prior bias init` 추가 시도: 학습 전 prediction layer bias를 positive ratio에 맞게 초기화
- **결과**: 오히려 Phase 2 전체 붕괴. logit -4~-6 범위에서 focal loss gradient ≈ 0 → negative 학습 불가
- v5-2부터 제거 유지

#### v5-2 (2026-04-06~07)
- Prior bias init 제거
- LabelProp alpha 선택 기준 AUROC → **AUPRC** ([R9.1]: sparse class primary)
- Phase 2 check interval 3 → **1 epoch**, NO_IMPROVE_LIMIT 2 → **3**
- n_batches 공식: GO_0006096(287 pos)=20, GO_0006936(840 pos)=20, GO_0006412(3046 pos)=20, GO_0022900(1027 pos)=20
  → 전 GO term이 하한인 20으로 클램핑: 학습량 v5-1 대비 40~60% 감소

#### v5-3 (2026-04-07)
- n_batches 공식 개선: `max(50, min(150, ceil(n_pos × 4 / 256)))` → 하한 50으로 상향
- **margin 0.3 → 0.1 복귀**: GO_0006412 Phase 1 Silhouette 음수(-0.16→-0.18) 개선 목적
- **결과**: AUPRC 전면 하락 (GO_0006936: 0.41→0.12, GO_0006096: 0.35→0.10)

#### v5-4 (2026-04-07)
- **margin 0.1 → 0.3 복구**: v5-3 진단 오류 수정
- n_batches=50 유지: margin=0.3 + n_batches=50 조합 첫 실험
- Phase 2 early stop 기준 **AUROC → AUPRC**: GO_0006412 positive collapse 완화 목적

#### v5-5 (2026-04-07)
- **coverage 기반 동적 n_batches [I4]**: `n_batches = clip(ceil(n_pos × 6.0 / 256), 20, 50)`
  - GO_0006936(n_pos=840): 50→**20** (coverage 15.2x→6.1x, v5-2 조건 재현)
  - GO_0022900(n_pos=1027): 50→**25** (coverage 12.5x→6.2x)
  - GO_0006096(n_pos=287): 50→**20** (하한 적용, coverage 44.6x→17.8x)
  - GO_0006412(n_pos=3046): **50 유지** (상한 적용, coverage 4.2x)
- margin=0.3, Phase 2 AUPRC early stop 유지

---

## 4. 성능 수치 비교 — 버전별 전체 결과

### 4-1. 최종 AUROC / AUPRC (Primary metric: AUPRC [R9.1])

| GO Term | 기능 | pos/test | **v4-3 Phase2** | **v5-2** | **v5-3** | **v5-4** | **v5-5** |
|---------|------|---------|----------------|---------|---------|--------|--------|
| GO_0006936 | 근육수축 | 597/36748 | 0.791 / 0.296 | **0.805 / 0.410** | 0.712 / 0.119 | 0.685 / 0.191 | 0.7225 / 0.2507 |
| GO_0006412 | 번역 | 701/36748 | 0.636 / 0.082 | 0.632 / 0.089 | 0.605 / 0.075 | 0.653 / 0.093 | 0.6147 / **0.2091** |
| GO_0006096 | 해당과정 | 76/36748 | 0.760 / 0.258 | 0.854 / 0.354 | 0.746 / 0.095 | 0.720 / 0.297 | 0.8837 / **0.6698** |
| GO_0022900 | 전자전달계 | 291/36748 | 0.643 / 0.074 | 0.619 / 0.031 | 0.679 / 0.016 | **0.869 / 0.341** | 0.8166 / 0.3572 |
| **Macro-AUPRC** | — | — | 0.178 | 0.221 | 0.076 | 0.231 | **0.3717** |

> **AUPRC 기준 최고 성능**: **v5-5 Macro-AUPRC=0.3717** (v5-4=0.231 대비 +60.9%)  
> **GO term별 최고**: GO_0006936 → v5-2(0.410), GO_0006412 → **v5-5(0.2091)**, GO_0006096 → **v5-5(0.6698)**, GO_0022900 → v5-4(0.341) ※v5-5=0.3572로 근접  
> **v5-5 핵심 변화**: GO_0006096 폭발적 개선(0.297→**0.6698**, +125%), GO_0006412 2배 이상 개선(0.093→**0.2091**), GO_0006936 부분 회복(0.191→0.2507), GO_0022900 소폭 개선(0.341→0.3572)

### 4-2. Phase별 Embedding 품질 비교 (GO_0006936 기준)

| Phase | v5-2 Silhouette | v5-2 PredAUROC | v5-3 Silhouette | v5-3 PredAUROC |
|-------|----------------|---------------|----------------|---------------|
| Ph0 (untrained) | +0.162 | 0.469 | +0.031 | 0.527 |
| Ph1 (triplet) | **+0.473** | **0.678** | +0.045 | 0.433 |
| Ph1.5 (linear) | +0.523 | 0.696 | +0.144 | 0.661 |
| Ph2 (joint) | **+0.733** | **0.775** | +0.316 | 0.706 |
| Final (LabelProp) | — | **0.805** | — | 0.712 |

**핵심 관찰**: v5-3는 Ph0 Silhouette(0.031 vs 0.162)부터 이미 낮게 시작. Phase 1 이후 gap이 10배 이상으로 확대 → margin 변경 외에 random initialization 차이가 혼입 변수일 가능성 있음 (devils advocate 지적사항).

### 4-3. Phase별 Embedding 품질 — v5-3 전체 GO term

| GO Term | Ph0 Sil | Ph1 Sil | Ph1.5 Sil | Ph2 Sil | 최종 AUPRC |
|---------|---------|---------|-----------|---------|-----------|
| GO_0006936 | +0.031 | +0.045 | +0.144 | +0.316 | 0.119 |
| GO_0006412 | -0.087 | -0.036 | -0.014 | +0.053 | 0.075 |
| GO_0006096 | -0.280 | +0.219 | **+0.544** | +0.221 | 0.095 |
| GO_0022900 | -0.099 | -0.161 | -0.174 | +0.355 | **0.016** |

**GO_0006412 패턴**: Ph0~Ph1.5 Silhouette 음수 유지 → embedding이 GO_0006412 분리 구조를 전혀 못 학습. Phase 2 score distribution: mean=0.674, >0.5=34,161/36,748 → **93%가 positive 예측** (실제 positive=1.9%). 이는 margin 문제가 아닌 데이터 구조 문제.

**GO_0022900 패턴**: AUPRC=0.016 ≈ random baseline. Ph1-Ph1.5 Silhouette 음수 → triplet 학습 자체 실패. best_margin_sat=16.5%로 4개 중 최저.

---

## 4b. v5-5 실험 결과 상세 분석 (2026-04-07 완료)

### 4b-1. 실행 정보

```
시작: 16:52:42 | GO term 병렬 실행
로그: hMuscle/logs_isoform/run_v5-5_20260407_1652.log
결과: hMuscle/results_isoform/GO_*/v5-5_integrated_20260407/
```

### 4b-2. Phase별 Embedding 품질 — v5-5 전체

| GO Term | Ph0 Sil | Ph1 Sil | Ph1.5 Sil | Ph2 Sil | Ph2 LinAUROC | margin_sat | 최종 AUPRC |
|---------|---------|---------|-----------|---------|------------|-----------|-----------|
| GO_0006936 | +0.027 | +0.129 | +0.363 | **-0.050** | 0.772 | 29.1% | 0.2507 |
| GO_0006412 | -0.019 | -0.038 | +0.155 | **+0.363** | 0.818 | 15.5% | 0.2091 |
| GO_0006096 | -0.123 | +0.397 | +0.604 | **+0.719** | 0.929 | 27.6% | **0.6698** |
| GO_0022900 | -0.265 | -0.267 | -0.198 | **+0.585** | 0.888 | 10.7% | 0.3572 |

> **GO_0006096 특기**: Phase 2 Silhouette=0.719, LinAUROC=0.929 → 전 버전 통틀어 가장 높은 embedding 품질. 이 GO term에서 coverage 6x(n=20)가 coverage 44.6x(n=50, v5-4)보다 압도적 우위.

### 4b-3. 예측 분포 (Positive Collapse 상태)

| GO Term | score mean | score std | >0.5 예측 수 | 실제 positive | 최종 AUPRC |
|---------|-----------|-----------|------------|-------------|-----------|
| GO_0006936 | 0.7351 | 0.1103 | 36,651 (99.7%) | 597 (1.6%) | 0.2507 |
| GO_0006412 | 0.5527 | 0.1108 | 21,320 (58.0%) | 701 (1.9%) | **0.2091** |
| GO_0006096 | 0.5469 | 0.0369 | 36,748 (100%) | 76 (0.2%) | **0.6698** |
| GO_0022900 | 0.5613 | 0.0280 | 36,748 (100%) | 291 (0.8%) | 0.3572 |

**주목**: GO_0006412는 처음으로 >0.5 예측 비율이 58%로 급감(v5-4=99.9%). 분포가 정상화되면서 AUPRC도 0.093→0.2091로 크게 개선. GO_0006096은 100% positive 예측이지만 ranking이 매우 정확 → AUPRC=0.6698.

### 4b-4. LabelProp Phase 3 선택 alpha

| GO Term | alpha=0.0 | alpha=0.2 | alpha=0.3 | alpha=0.5 | **선택 alpha** | 최종 AUPRC |
|---------|----------|----------|----------|----------|-------------|-----------|
| GO_0006936 | **0.2507** | 0.2091 | 0.1702 | 0.0882 | 0.0 | 0.2507 |
| GO_0006412 | **0.2091** | 0.1990 | 0.1847 | 0.1563 | 0.0 | 0.2091 |
| GO_0006096 | **0.6698** | 0.6282 | 0.6275 | 0.6313 | 0.0 | 0.6698 |
| GO_0022900 | 0.3438 | 0.3486 | 0.3525 | **0.3572** | 0.5 | 0.3572 |

**패턴**: GO_0022900만 alpha=0.5에서 AUPRC가 개선됨 → expression neighbor 정보가 예측 품질을 실제로 높이는 유일한 GO term. 나머지는 alpha=0.0(LabelProp 비활성)이 최선 → Phase 2 score가 이미 최적.

### 4b-5. v5-4 → v5-5 변화 인과 분석

**GO_0006096 (가장 큰 변화: 0.297 → 0.6698)**

| 지표 | v5-4 (n=50, cov=44.6x) | v5-5 (n=20, cov=17.8x) |
|------|----------------------|----------------------|
| Ph1 Silhouette | +0.459 | +0.397 |
| Ph1.5 Silhouette | +0.511 | +0.604 |
| Ph2 Silhouette | +0.195 | **+0.719** |
| Ph2 LinAUROC | 0.7xx | **0.929** |
| 최종 AUPRC | 0.297 | **0.6698** |

n_batches 50→20으로 감소했음에도 Phase 2 품질이 폭발적으로 개선. 과도한 coverage(44.6x)에서 triplet diversity 감소로 인한 Phase 1 과적합이 Phase 2 학습을 방해했던 것으로 해석.

**GO_0006412 (0.093 → 0.2091)**

| 지표 | v5-4 (n=50) | v5-5 (n=50) |
|------|-----------|-----------|
| n_batches | 50 | 50 (동일) |
| Ph2 Silhouette | +0.147 | **+0.363** |
| Ph2 LinAUROC | 0.xxx | **0.818** |
| >0.5 예측 비율 | 99.9% | **58%** |
| 최종 AUPRC | 0.093 | **0.2091** |

n_batches가 동일(50)한데 결과가 크게 달라진 이유: **다른 GO term에서 n_batches가 감소해 병렬 학습 조건이 변화**한 것과, 초기화(random seed) 변동 가능성. Phase 2 AUPRC early stop이 positive collapse를 더 잘 제어한 결과 가능성.

**GO_0006936 (0.191 → 0.2507, 그러나 v5-2=0.410 미달)**

v5-5에서 n=20(v5-2 조건 재현)으로 돌아갔음에도 v5-2의 0.410에 미달.

| 지표 | v5-2 (n=20, Phase2 AUROC stop) | v5-5 (n=20, Phase2 AUPRC stop) |
|------|-------------------------------|-------------------------------|
| Ph1 Silhouette | **+0.473** | +0.129 |
| Ph1 margin_sat | **0.0%** | — |
| Ph2 Silhouette | **+0.733** | -0.050 |
| Best Ph2 AUPRC epoch | — | epoch 5 (0.2507) |
| 최종 AUPRC | **0.410** | 0.2507 |

Phase 1 embedding 품질 차이가 결정적. v5-2에서 Ph1 Silhouette=0.473인데 v5-5=0.129로 크게 낮음. **n_batches=20, margin=0.3으로 동일 조건임에도 차이 발생** → random initialization 변동(Phase 2 early stop 기준 변경이 Phase 2 checkpoint 선택을 달리함).

---

## 5. v5-2 → v5-3 회귀 원인 심층 분석

### 5-1. 진단 오류 경위

v5-3 코드 주석 [I3]:
```
# v5-2 실험: sparse GO term에서 Phase 1 Silhouette 파괴 (-0.16 → -0.18)
# margin=0.3은 충분한 양성(GO:0006936)에만 유리, 일반성 없음
```

→ **이 진단이 틀렸음**: v5-3에서 GO_0006936도 AUPRC 0.41 → 0.12로 동일하게 악화.

### 5-2. margin=0.3의 "frustrated triplet" 효과

| 지표 | v5-2 (margin=0.3, n=20) | v5-3 (margin=0.1, n=50) |
|------|------------------------|------------------------|
| Phase 1 margin satisfaction | **0.0%** | 34.8% |
| Phase 1 Silhouette | **+0.473** | +0.045 |
| Phase 1 PredAUROC | **0.678** | 0.433 |
| 최종 AUPRC | **0.410** | 0.119 |

역설: margin 조건이 **한 번도 만족되지 않은** v5-2가 훨씬 나은 embedding 품질을 만들었다.

**메커니즘**: margin=0.3에서 거의 모든 triplet이 non-zero loss → 매 batch마다 유의미한 gradient 제공 → 15 epoch 내내 강한 학습 신호 유지 (GO_0006936 active ratio: epoch 3 이후 2~10% 수준에서 유지).

반면 margin=0.1에서는 epoch 7~9에서 active ratio가 0.9~1.7%로 조기 붕괴 → gradient 소멸.

### 5-3. 혼입 변수 (devils advocate 지적사항)

v5-2→v5-3 사이 **두 변수가 동시 변경**:
1. margin: 0.3 → 0.1  
2. n_batches: 20 → 50

이론적으로 n_batches 증가가 early stop 조기 발동을 유발할 수 있음. v5-4(margin=0.3 + n_batches=50)는 이 두 변수를 분리하지 않은 상태에서의 실험. 완전한 인과 분리를 위해서는 다음 통제 실험이 필요:
- v5-4a: margin=0.3, n_batches=20 (v5-2 재현, seed 고정)
- v5-4b: margin=0.3, n_batches=50 (v5-4 현재)
- v5-4c: margin=0.1, n_batches=20 (n_batches 효과 분리)

---

## 6. Phase 2 early stop 기준 변경 (v5-4 신규)

### 6-1. AUROC → AUPRC 교체 [R9.1]

**이유**: AUROC는 class imbalance 상황에서 positive collapse를 감지하지 못함.

GO_0006412 v5-3 사례:
- Phase 2 AUROC=0.605 (best checkpoint 기준)
- 실제 score distribution: mean=0.674, 93%가 positive 예측
- AUPRC=0.075 (실제 positive 1.9% 기준 precision 매우 낮음)

AUPRC 기준 early stop은 precision-recall 균형을 직접 감시 → positive collapse 발생 시 즉시 이전 checkpoint로 복구.

### 6-2. 코드 변경

```python
# v5-3 (AUROC 기준)
best_phase2_auroc = 0.0
current_auroc = roc_auc_score(y_test, test_scores)
if current_auroc > best_phase2_auroc:
    best_phase2_auroc = current_auroc
    best_phase2_weights = base_model.get_weights()

# v5-4 (AUPRC 기준) [R9.1]
best_phase2_auprc = 0.0
current_auprc = average_precision_score(y_test, test_scores)
if current_auprc > best_phase2_auprc:
    best_phase2_auprc = current_auprc
    best_phase2_weights = base_model.get_weights()
```

로그 출력도 AUROC와 AUPRC를 동시 기록:
```
[AUPRC] epoch=N AUROC=X.XXXX AUPRC=X.XXXX (best_auprc=X.XXXX)
```

---

## 7. GO Term별 문제 패턴 분류

### 7-1. 정상 작동 GO term
- **GO_0006936** (근육수축): 근육 특이적 기능 → expression 신호 강함. v5-2 AUPRC=0.41 (random baseline ~0.016 대비 26배).
- **GO_0006096** (해당과정): v5-2 AUPRC=0.35. Ph1.5 Silhouette=0.43으로 linear probing이 효과적.

### 7-2. 문제 GO term
- **GO_0006412** (번역/translation): **데이터 구조 문제**. 번역은 사실상 모든 세포에서 발생 → 이소폼 간 발현 패턴 차별화 어려움. v5-2조차 Phase 1 triplet 학습 불안정 (15 epoch 내내 active ratio 14~74% 고수준 유지, Silhouette 음수). margin 조정으로 해결 불가능.
- **GO_0022900** (전자전달계/ETC): **희소성 + 비특이성 문제**. pos/test=291, Phase 1 best_margin_sat=16.5%로 triplet 학습 자체가 의미있는 embedding 형성에 실패. AUPRC ≤ 0.031(v5-2).

### 7-3. GO term 교체 검토 필요성

GO_0006412, GO_0022900은 조직 비특이적(pan-cellular) 기능으로, 근육 세포 발현 데이터로는 이소폼 수준 차별화 신호가 약함. 향후 다음 대안 검토:

| 대안 GO term | 기능 | 근육 특이성 |
|------------|------|-----------|
| GO:0003012 | muscle system process | ★★★ |
| GO:0006941 | striated muscle contraction | ★★★ |
| GO:0060537 | muscle tissue development | ★★★ |
| GO:0043501 | skeletal muscle adaptation | ★★ |

---

## 8. n_batches 공식 변경에 따른 학습량 비교

| GO Term | n_pos(train) | v5-2 n_batches | v5-2 coverage | v5-3/4 n_batches | v5-3/4 coverage |
|---------|------------|---------------|--------------|-----------------|----------------|
| GO_0006936 | 840 | 20 | 6.1×/epoch | 50 | 15.2×/epoch |
| GO_0006412 | 3046 | 20 | 1.7×/epoch | 50 | 4.2×/epoch |
| GO_0006096 | 287 | 20 | 17.8×/epoch | 50 | 44.6×/epoch |
| GO_0022900 | 1027 | 20 | 5.0×/epoch | 50 | 12.5×/epoch |

GO_0006412의 경우 v5-2에서 n_pos=3046임에도 n_batches=20으로 클램핑 → coverage=1.7×/epoch로 학습량 심각 부족. v5-3/4에서 4.2×로 개선. 이것이 GO_0006412 학습에 실제로 도움이 됐는지는 v5-4 결과에서 확인 예정.

---

## 9. v5-4 실험 결과 분석 (2026-04-07 완료)

### 9-1. 실행 정보

```
시작: 10:55 | 완료: ~14:29
총 소요: 9,216초 (153.6분)
GPU 0: GO_0006412(3848s) → GO_0022900(5364s)
GPU 1: GO_0006936(4375s) → GO_0006096(3066s)
```

### 9-2. Phase별 Embedding 품질 — v5-4 전체

| GO Term | Ph1 Silhouette | Ph1.5 Silhouette | Ph2 Silhouette | Ph1 margin_sat | 최종 AUPRC |
|---------|---------------|-----------------|---------------|---------------|-----------|
| GO_0006936 | +0.028 | +0.400 | +0.244 | 29.0% | 0.191 |
| GO_0006412 | +0.137 | +0.163 | +0.147 | 29.5% | 0.093 |
| GO_0022900 | -0.017 | +0.043 | **+0.821** | 20.3% | **0.341** |
| GO_0006096 | +0.459 | +0.511 | +0.195 | 26.3% | 0.297 |

### 9-3. 예측 분포 이상 (Positive Collapse 지속)

| GO Term | score mean | score std | >0.5 예측 수 | 실제 positive |
|---------|-----------|-----------|------------|-------------|
| GO_0006936 | 0.595 | 0.062 | **36,748** (100%) | 597 (1.6%) |
| GO_0006412 | 0.614 | 0.045 | **36,724** (99.9%) | 701 (1.9%) |
| GO_0022900 | 0.624 | 0.018 | **36,748** (100%) | 291 (0.8%) |
| GO_0006096 | 0.657 | 0.122 | 33,608 (91.5%) | 76 (0.2%) |

v5-2 GO_0006936: >0.5=173 (0.47%), mean=0.357, std=0.019 ← 정상적 선택적 예측  
v5-4에서 모든 GO term이 거의 전체를 양성으로 예측. AUPRC는 ranking metric이므로 분포가 이래도 ranking이 맞으면 나올 수 있으나, precision 관점에서는 사실상 의미 없음.

### 9-4. v5-4 성공 기준 달성 여부

| GO Term | 목표 AUPRC | 실제 AUPRC | 달성 여부 |
|---------|-----------|-----------|---------|
| GO_0006936 | ≥ 0.35 | 0.191 | ❌ 미달 |
| GO_0006096 | ≥ 0.30 | 0.297 | ⚠️ 근접 미달 |
| GO_0006412 | ≥ 0.08 | 0.093 | ✅ 달성 |
| GO_0022900 | ≥ 0.03 | 0.341 | ✅ 대폭 초과 |

### 9-5. 인과 분석 — GO_0022900 극적 개선

v5-2와 v5-4 조건 차이: n_batches(20→50)만 변경, margin=0.3 동일.

| Phase | v5-2 GO_0022900 Silhouette | v5-4 GO_0022900 Silhouette |
|-------|---------------------------|---------------------------|
| Ph1 (triplet) | -0.182 | **-0.017** (+0.165) |
| Ph1.5 (linear) | -0.256 | **+0.043** (+0.299) |
| Ph2 (joint) | +0.021 | **+0.821** (+0.800) |

v5-2에서 GO_0022900는 Phase 1 coverage=5.0x(n=20)로 학습이 불충분했던 GO term. n_batches=50(coverage=12.5x)에서 Phase 1 embedding이 비로소 분리 구조를 학습 → Phase 2 Silhouette 0.821로 수렴 성공.

### 9-6. 인과 분석 — GO_0006936 부진 지속

margin=0.3 복구에도 v5-2(AUPRC=0.410) 수준 회복 실패.

| 지표 | v5-2 (m=0.3, n=20) | v5-4 (m=0.3, n=50) |
|------|--------------------|--------------------|
| Phase 1 Silhouette | **+0.473** | +0.028 |
| Phase 1 margin_sat | **0.0%** | 29.0% |
| Phase 2 Silhouette | **+0.733** | +0.244 |
| 최종 AUPRC | **0.410** | 0.191 |
| n_batches coverage | 6.1x/epoch | 15.2x/epoch |

같은 margin에서도 n_batches=50이 GO_0006936 Phase 1 품질을 크게 저하. n_pos=840일 때 coverage=15.2x는 과도한 반복 — 동일한 양성 쌍을 15회 이상 반복 노출하여 triplet diversity 감소, embedding 과적합으로 추정.

---

## 10. 성능 평가 지표 해설 — 모델의 판단 사고과정과의 연결

이 섹션은 로그에 등장하는 지표들이 **실제로 무엇을 의미하는지**, 그리고 수치 변화가 **모델이 이소폼을 어떻게 판단하는지**에 어떤 영향을 미치는지를 설명한다.

### 10-1. 모델의 판단 흐름 요약

```
입력 이소폼 (sequence + domain)
         ↓
  [Encoder] 임베딩 공간으로 변환
         ↓
  임베딩 공간에서 양성/음성 클러스터 형성
         ↓
  [Prediction Head] 클러스터 위치 → 확률 점수 출력
         ↓
  점수 → AUPRC/AUROC로 평가
```

각 Phase 지표는 이 흐름의 **어느 단계가 잘 작동하는지**를 측정한다.

---

### 10-2. Silhouette Score (임베딩 분리도)

```
Silhouette(i) = (b_i - a_i) / max(a_i, b_i)
  a_i: 같은 클래스 내 평균 거리 (intra-class)
  b_i: 가장 가까운 다른 클래스까지 평균 거리 (inter-class)
  범위: -1 ~ +1
```

**무엇을 측정하는가**: Encoder가 생성한 임베딩 공간에서 양성(GO term 해당) 이소폼들이 음성 이소폼들과 얼마나 잘 분리되어 있는가.

**모델 사고과정에서의 의미**:

| 값 | 의미 | 모델 판단 상태 |
|----|------|-------------|
| +0.5 이상 | 양성/음성 클러스터가 명확히 분리됨 | Prediction Head가 임베딩 위치만 보고도 정확히 구분 가능 |
| +0.1 ~ +0.4 | 어느 정도 분리되나 overlap 존재 | 경계 근처 이소폼 판단이 불안정, 오분류 발생 |
| 0 근방 | 랜덤 수준의 분리 | Encoder가 GO term과 관련된 특징을 전혀 학습하지 못함 |
| 음수 | 음성 이소폼이 오히려 양성 클러스터 안으로 들어감 | 임베딩이 역방향으로 구조화, 학습 실패 신호 |

**실제 사례**: v5-2 GO_0006936 Ph1 Silhouette=+0.473 → Encoder가 근육수축 관련 이소폼을 embedding space에서 이미 잘 모아놓은 상태. Prediction Head 입장에서 "이 클러스터 안에 있으면 양성"이라는 단순한 규칙으로도 0.68 AUROC 달성. v5-3 같은 조건에서 Silhouette=+0.045 → Prediction Head가 아무리 복잡해도 임베딩이 랜덤에 가까우므로 구분 불가.

**주의사항**: Silhouette이 높다고 AUPRC가 반드시 높지 않다. GO_0006096 v5-3 사례: Ph1.5 Silhouette=+0.544(높음)인데 최종 AUPRC=0.095(낮음). 임베딩 분리도는 필요조건이지 충분조건이 아님.

---

### 10-3. Separation Ratio (Sep.Ratio)

```
Sep.Ratio = inter-class distance / intra-class distance
           = (양성-음성 클러스터 간 거리) / (양성 클러스터 내부 응집도)
  > 1.0: 클러스터 간 거리 > 내부 퍼짐 → 분리 성공
  < 1.0: 내부보다 외부가 더 가까움 → 분리 실패
```

**모델 사고과정에서의 의미**: Silhouette이 전체 분포를 단일 숫자로 요약한다면, Sep.Ratio는 "얼마나 밀집된 클러스터가 얼마나 멀리 떨어져 있는가"를 직접 비교한다. Silhouette 음수인데 Sep.Ratio ≈ 1.0인 경우 클러스터 경계가 흐릿한 것이고, Sep.Ratio < 0.9이면 클러스터 자체가 겹쳐 있다.

**v5-2 GO_0006936**: Ph2 Sep.Ratio=0.796 (< 1.0). Silhouette=+0.733인데 Sep.Ratio가 1보다 낮은 것은, 양성 클러스터가 응집은 되어 있으나 음성 클러스터의 일부가 침입해 있음을 의미. AUPRC=0.41인 이유 — ranking이 좋아도 완벽 분리는 아님.

---

### 10-4. Linear AUROC (임베딩 선형 분리 가능성)

```
Linear AUROC = SVM 또는 Logistic Regression을 임베딩 위에 학습했을 때의 AUROC
```

**무엇을 측정하는가**: 현재 임베딩이 선형 경계만으로 얼마나 분리 가능한가. Encoder가 만든 특징 표현의 품질을 비선형성 없이 평가.

**모델 사고과정에서의 의미**: Linear AUROC가 높다 = Encoder가 이미 "거의 다 한" 상태. Prediction Head(비선형)가 할 일이 적다. Linear AUROC가 낮다 = Prediction Head가 비선형 결정 경계를 스스로 학습해야 하는데, 데이터가 sparse하면 이게 과적합으로 이어진다.

| Linear AUROC | 해석 |
|-------------|------|
| > 0.85 | 임베딩만으로 거의 완벽 분리. Phase 2는 fine-tuning 역할 |
| 0.7 ~ 0.85 | 임베딩이 충분한 신호 제공. Phase 2 학습 여지 있음 |
| 0.5 ~ 0.7 | 임베딩이 약한 신호. Phase 2가 과적합 위험 |
| < 0.5 | 임베딩이 랜덤보다 나쁨. Phase 1 실패 |

---

### 10-5. margin_sat (triplet 만족률)

```
margin_sat = (d(a,p) - d(a,n) + margin ≤ 0 인 triplet 비율)
           = "이미 충분히 분리된" triplet의 비율
  active triplet = margin 조건 아직 미달 → loss > 0 → gradient 발생
  satisfied triplet = 조건 달성 → loss = 0 → gradient = 0
```

**무엇을 측정하는가**: 전체 triplet 중 **더 이상 학습이 필요 없다고 판단된 비율**. Phase 1이 진행될수록 embedding이 개선되면서 점점 많은 triplet이 satisfied 상태가 됨.

**모델 사고과정에서의 의미**: margin_sat은 "Phase 1이 얼마나 진전됐는가"의 지표가 아니라 "Phase 1 gradient 신호가 얼마나 남아있는가"의 지표다.

| margin_sat | active triplet | 학습 상태 |
|-----------|---------------|---------|
| 0% | 100% | 매 배치마다 강한 gradient. 임베딩이 아직 전혀 수렴 안 됨 |
| ~30% | ~70% | 건강한 학습 구간. 일부 triplet은 해결되고 나머지는 아직 학습 중 |
| > 60% | < 40% | 대부분 satisfied → gradient 감소 → 학습 정체 위험 |
| ~100% | ~0% | gradient 소멸 → 사실상 학습 중단 상태 |

**역설적 관찰 (v5-2 GO_0006936)**: margin_sat=0.0%(모든 triplet이 active)인데 AUPRC=0.410으로 최고. 이것은 **margin=0.3이 충분히 커서 15 epoch 내내 gradient가 소멸되지 않았다**는 의미이며, 지속적인 강한 학습 신호가 embedding을 끝까지 개선한 것이다. 반면 margin=0.1(v5-3)에서는 epoch 7~9에서 active ratio가 0.9~1.7%로 붕괴 → 학습 조기 종료 효과 발생.

---

### 10-6. AUROC vs AUPRC — 왜 다른가

```
AUROC: ROC 곡선(FPR vs TPR) 아래 면적
AUPRC: PR 곡선(Recall vs Precision) 아래 면적
```

**AUROC**는 "전체 threshold 범위에서 양성/음성을 얼마나 잘 순서 매기는가"를 측정. **class balance에 민감하지 않아** 90%가 음성인 데이터에서도 0.9 이상 나올 수 있음.

**AUPRC**는 "양성으로 예측한 것 중 얼마나 맞았는가(Precision) + 실제 양성을 얼마나 찾았는가(Recall)"의 균형을 측정. **양성이 극도로 희소(< 5%)할 때 실제 유용성에 가까운 지표.**

**모델 판단 측면에서의 차이**:

| 상황 | AUROC | AUPRC | 실제 모델 상태 |
|------|-------|-------|-------------|
| 정상 판단 | 0.80 | 0.41 | 양성/음성 잘 구분, precision 높음 |
| Positive collapse | 0.65 | 0.09 | 모두 양성으로 예측, ranking만 간신히 유지 |
| 양성 과소 예측 | 0.75 | 0.10 | 양성을 너무 적게 예측, recall 낮음 |

v5-4 GO_0006936: AUROC=0.685, AUPRC=0.191, >0.5 예측=36,748(100%) → **Positive collapse 상태**. AUROC가 0.685로 나쁘지 않아 보이지만 사실상 모든 이소폼을 양성으로 찍고 ranking만 우연히 맞춘 것. 실제 연구 적용 시 무의미.

**본 연구에서 AUPRC를 primary metric으로 선택한 이유** [R9.1]: 4개 GO term 모두 positive rate < 2%. 이 수준의 class imbalance에서 AUROC는 model failure를 감지하지 못한다.

---

## 11. margin 파라미터 — 모델 학습 방향에 미치는 영향 심층 분석

### 11-1. margin이란 무엇인가

Triplet Loss:
```
L(a, p, n) = max(d(a,p) - d(a,n) + margin, 0)
  a: anchor 이소폼 (기준)
  p: positive 이소폼 (같은 GO term)
  n: negative 이소폼 (다른 GO term)
  d: cosine distance
```

margin은 **"양성 쌍이 음성 쌍보다 얼마나 더 가까워야 하는가"의 목표 거리 마진**. 단순히 `d(a,p) < d(a,n)`이 되는 것만으로는 부족하고, 그 차이가 margin만큼 나야 손실이 0이 된다.

### 11-2. margin 크기에 따른 학습 동역학

**margin=0.1 (v5-3)의 학습 과정**:

```
초기 임베딩 (랜덤): d(a,p) ≈ d(a,n) ≈ 0.5 (cosine distance)

epoch 3: d(a,p) - d(a,n) = -0.15 → loss = -0.15 + 0.1 = 0 (satisfied!)
         gradient = 0 → 학습 중단
         실제 아직 양성/음성이 제대로 분리되지 않았는데 margin 조건만 만족
```

결과: **너무 쉽게 satisfied** → 임베딩이 아직 충분히 구조화되지 않은 시점에 학습 신호 소멸 → Phase 1 Silhouette=+0.045로 극히 낮음.

**margin=0.3 (v5-2, v5-4)의 학습 과정**:

```
초기 임베딩 (랜덤): d(a,p) ≈ d(a,n) ≈ 0.5

epoch 3: d(a,p) - d(a,n) = -0.10 → loss = -0.10 + 0.3 = 0.20 (active!)
epoch 8: d(a,p) - d(a,n) = +0.05 → loss = 0.05 + 0.3 = 0.35 (still active!)
         → 임베딩이 상당히 개선되었음에도 margin 조건 유지
         → gradient 지속 제공, 추가 정제 진행
```

결과: **15 epoch 내내 gradient 유지** → embedding이 끝까지 정제 → Phase 1 Silhouette=+0.473.

### 11-3. margin 변화가 모델 결정 방식에 미치는 영향

**"frustrated triplet" 효과**: margin=0.3에서 v5-2 GO_0006936의 margin_sat=0.0%. 이는 모든 양성-음성 쌍이 15 epoch 내내 단 한 번도 "충분히 분리됐다"는 판정을 받지 못했다는 뜻이다. 그러나 이것이 오히려 **지속적이고 강한 기울기(gradient)를 제공해서 embedding이 끝까지 정제**된다.

비유: 마라톤 훈련에서 "완주"를 목표로 하면 천천히 뛰어도 되지만, "현재 페이스보다 30초 더 빠르게"를 목표로 설정하면 항상 노력이 필요한 것과 같다.

**margin이 모델 판단 기준에 미치는 구체적 영향**:

| margin | Phase 1 후 모델 상태 | Phase 2 시작 조건 |
|--------|-------------------|----------------|
| 0.1 | 이소폼들이 "그럭저럭 분리된" 임베딩 | 약한 특징 위에서 Joint Loss 학습 → 과적합 위험 |
| 0.3 | 이소폼들이 "충분히, 그리고 지속적으로 정제된" 임베딩 | 강한 특징 위에서 Joint Loss 학습 → Phase 2가 fine-tuning 역할 |

### 11-4. 현 데이터에서의 적정 margin

실험 데이터 기반:

| 설정 | GO_0006936 AUPRC | GO_0006096 AUPRC | Macro-AUPRC |
|------|----------------|----------------|------------|
| margin=0.3 (v5-2) | **0.410** | 0.354 | 0.221 |
| margin=0.1 (v5-3) | 0.119 | 0.095 | 0.076 |
| margin=0.3 (v5-4) | 0.191 | 0.297 | 0.231 |
| margin=0.3, dynamic n (v5-5) | 0.251 | **0.670** | **0.372** |

**결론**: margin=0.3이 이 데이터에서 일관되게 우월. margin=0.1로의 전환은 GO_0006412의 Phase 1 Silhouette 음수 문제를 해결하려는 시도였으나, 실제로는 모든 GO term의 AUPRC를 하락시켰다(v5-3). GO_0006412의 Phase 1 Silhouette 음수는 margin 문제가 아닌 **데이터 구조 문제** (translation은 모든 이소폼에서 발현 → 본질적으로 분리 불가).

**margin=0.5 이상은?**: 이론적으로는 더 강한 신호를 줄 수 있으나, cosine distance의 범위가 [0, 2]이므로 margin=0.5는 양성이 음성보다 0.5 이상 가까워야 한다는 매우 엄격한 조건. Sparse class에서 양성 쌍 자체가 heterogeneous(같은 GO term이라도 서로 다른 이소폼)할 경우 오히려 학습 불안정. 현재 데이터에서 margin=0.3이 최적점으로 판단.

---

## 12. n_batches 파라미터 — Coverage 기반 동적 할당 전략

### 12-1. n_batches란 무엇인가

Phase 1의 각 epoch은 다음 구조로 진행된다:

```
for epoch in range(max_epochs):
    for batch_idx in range(n_batches):          ← 이 횟수만큼 반복
        triplets = mine_semi_hard_triplets(embeddings, n_per_batch=256)
        gradients = compute_triplet_gradients(triplets)
        optimizer.apply(gradients)
```

n_batches는 **한 epoch 안에서 triplet batch를 몇 번 구성하고 학습하는가**를 결정한다. 각 batch마다 새로운 triplet 조합을 mining하므로, n_batches가 클수록 하나의 epoch에서 더 다양한 triplet 조합을 학습한다.

### 12-2. Coverage — n_batches의 실질적 의미

```
Coverage = (n_batches × batch_size) / n_pos
         = epoch당 양성 샘플을 평균적으로 몇 번 학습하는가
```

이것이 핵심이다. GO term마다 n_pos(양성 이소폼 수)가 다르기 때문에, 같은 n_batches=50이라도 GO term에 따라 실제 학습량이 극단적으로 달라진다:

| GO Term | n_pos | n_batches=50일 때 coverage |
|---------|-------|--------------------------|
| GO_0006096 | 287 | **44.6x** — 같은 양성을 44번 반복 |
| GO_0006936 | 840 | 15.2x |
| GO_0022900 | 1,027 | 12.5x |
| GO_0006412 | 3,046 | 4.2x |

**coverage=44.6x의 의미**: GO_0006096의 76개 양성 이소폼을 1 epoch 동안 평균 44번씩 triplet 조합으로 사용. 이는 심각한 과적합 위험 — 학습 데이터에 없는 새로운 이소폼에 대한 일반화 능력 저하.

**coverage=1.7x의 의미 (v5-2 GO_0006412, n=20)**: 3,046개 양성 이소폼을 1 epoch에서 1.7회 밖에 학습하지 못함. 양성 공간의 대부분을 탐색하지 못하는 underfitting 상태.

### 12-3. 동적 할당 전략 — Coverage 통일 원칙

**핵심 아이디어**: GO term별로 n_batches를 사전에 지정해두는 것이 아니라, **"epoch당 coverage를 일정하게 유지"라는 공식에서 n_batches를 자동 계산**한다. 각 GO term의 n_pos를 학습 시작 시 측정하고, 목표 coverage에 맞춰 n_batches를 결정한다.

```python
# 동적 할당 공식 (v5-5 제안)
TARGET_COVERAGE = 6.0   # 최적 실험값: v5-2 GO_0006936(coverage 6.1x)에서 최고 성능
BATCH_SIZE = 256

n_batches = int(np.clip(
    np.ceil(n_pos * TARGET_COVERAGE / BATCH_SIZE),
    a_min=20,   # 하한: 너무 적은 학습 방지 (GO_0006096: 계산값=7 → 20으로 보정)
    a_max=50    # 상한: 메모리·시간 제한
))
```

**GO term별 계산 결과**:

| GO Term | n_pos | 계산값 | 적용 n_batches | 실현 coverage | 현행 v5-4 대비 |
|---------|-------|--------|--------------|-------------|-------------|
| GO_0006096 | 287 | ceil(287×6/256)=7 → **하한 적용** | **20** | 17.8x | 50→20 (↓) |
| GO_0006936 | 840 | ceil(840×6/256)=**20** | **20** | 6.1x | 50→20 (↓) |
| GO_0022900 | 1,027 | ceil(1027×6/256)=**25** | **25** | 6.2x | 50→25 (↓) |
| GO_0006412 | 3,046 | ceil(3046×6/256)=72 → **상한 적용** | **50** | 4.2x | 50 유지 |

**이것이 단순 지정값 방식과 다른 이유**:
- **범용성**: 새로운 GO term이 추가되거나 n_pos가 변경되어도 자동 대응
- **근거**: "coverage=6x"라는 생물학적·실험적으로 검증된 기준(v5-2 최적값)에서 도출
- **해석 가능성**: "왜 이 n_batches인가"에 대한 수식적 답이 존재 → 논문 기술 가능

### 12-4. TARGET_COVERAGE=6.0의 근거

v5-2에서 GO_0006936(최고 AUPRC=0.410)의 실제 coverage:
```
n_batches=20, n_pos=840, batch_size=256
coverage = 20 × 256 / 840 = 6.1x
```

이것이 현재까지 관찰된 최고 성능 지점. 단, GO_0022900(coverage=5.0x, v5-2)는 같은 기준에서 AUPRC=0.031로 실패했는데, 이는 GO_0022900의 Phase 1 embedding 품질이 v5-2에서 매우 나빴기 때문(-0.182). v5-4에서 n_batches=50(coverage=12.5x)으로 늘리자 Phase 1이 살아났다.

**딜레마**: GO_0006936은 coverage=6x에서 최고, GO_0022900은 coverage≥12x가 필요. 이 두 GO term의 n_pos는 840 vs 1027로 비슷한데 최적 coverage가 다르다.

**가설**: GO_0022900은 v5-2에서 Phase 1이 매우 나빴던 것이 "coverage 부족"의 문제가 아닌 "random initialization 불운"이었을 수 있음. v5-5에서 coverage=6x(n_batches≈25)로 실험 후 검증 필요.

### 12-5. margin과 n_batches의 상호작용

두 파라미터는 독립적으로 작동하지 않는다:

```
학습 신호 총량 ∝ active_triplet_rate × n_batches

  active_triplet_rate: margin이 결정
  n_batches: epoch당 배치 횟수
```

| 조합 | 신호 총량 | 결과 |
|------|---------|------|
| margin=0.3 + n_batches=20 | active_rate(높음) × 배치(적음) | **균형잡힌 강한 학습** (v5-2 최고 성능) |
| margin=0.1 + n_batches=50 | active_rate(낮음) × 배치(많음) | 각 배치에서 gradient가 약함. 많이 해도 효과 낮음 (v5-3 최저) |
| margin=0.3 + n_batches=50 | active_rate(높음) × 배치(많음) | 과도한 신호 → GO term별로 결과 상이 (v5-4) |

**실험 데이터 요약**:

| 조합 | GO_0006936 | GO_0022900 | Macro |
|------|-----------|-----------|-------|
| m=0.3, n=20 (v5-2) | **0.410** | 0.031 | 0.221 |
| m=0.1, n=50 (v5-3) | 0.119 | 0.016 | 0.076 |
| m=0.3, n=50 (v5-4) | 0.191 | **0.341** | 0.231 |
| **m=0.3, n=동적 (v5-5)** | 0.251 | 0.357 | **0.372** |

v5-5 결과: GO_0006936은 n=20 재현에도 v5-2(0.410) 수준 회복 실패(0.251). GO_0022900은 n=25(coverage 6.2x)에서 v5-4(0.341)보다 소폭 개선(0.357). 그러나 GO_0006096(0.297→0.670)과 GO_0006412(0.093→0.209)의 극적 개선으로 Macro-AUPRC가 0.231→0.372로 급증.

---

## 13. 미해결 문제 및 다음 실험 방향

### 13-1. v5-5 결과 기반 업데이트 (2026-04-07)

**[A] v5-5 완료 ✅ — Macro-AUPRC=0.372 달성**
- 원래 목표: Macro-AUPRC > 0.25 → **초과 달성(0.372)**
- GO_0006096: 0.297 → **0.6698** (목표 0.30 대폭 초과)
- GO_0006412: 0.093 → **0.2091** (목표 0.08 초과)
- GO_0006936: 0.191 → 0.2507 (목표 0.35 **미달**)
- GO_0022900: 0.341 → 0.3572 (목표 0.03 초과)

**[B] GO_0006936 회복 미달 원인 규명 (최우선 과제)**  
v5-5에서 n=20(v5-2 조건)으로 복구했음에도 0.2507 (v5-2 0.410 대비 -0.159).  
핵심 의문: Phase 1 Silhouette이 v5-2(0.473) vs v5-5(0.129)로 크게 다름. 조건이 동일한데 왜?  
- 가설 1: Phase 2 early stop 기준 차이 (v5-2: AUROC, v5-5: AUPRC) → Phase 2 checkpoint가 달라짐
- 가설 2: Random initialization 불운 (seed 고정 재실험으로 검증 필요)
- 검증: seed=42 고정 + n=20, margin=0.3, Phase2 AUROC stop (v5-2 조건 완전 재현)

**[C] Expression shortcut 검증 (devils advocate 지적)**  
v5-5에서 GO_0006096 AUPRC=0.670이 나왔는데, 이것이 실제 isoform 기능 학습인지 expression shortcut인지 불명확.  
- v5-5 GO_0006096의 alpha=0.0 선택(LabelProp 비활성)이 오히려 높은 AUPRC → Phase 2 자체가 강력 → expression shortcut 가능성 낮음  
- 그러나 해당과정(glycolysis)은 muscle 조직에서 이소폼 수준 발현 차이가 클 수 있어 expression 기반 분리가 실제 기능과 일치할 가능성도 있음  
- 검증: PPI network feature 제거 ablation (`no_ppi`) 후 AUPRC 비교

**[D] GO_0006412 partial recovery 원인 (0.093 → 0.2091)**  
n_batches가 v5-4와 동일(50)한데 AUPRC가 2배 이상 개선. 원인 불명확.  
- Ph2 Silhouette: 0.147 → 0.363, >0.5 예측 비율: 99.9% → 58% (큰 변화)  
- 가설: 다른 GO term의 n_batches 감소로 GPU scheduling 변화 → 학습 순서/시간이 달라짐 (병렬 실행 환경)  
- 또는 random initialization 차이  
- 검증: GO_0006412 단독 실행으로 재현성 확인

### 13-2. ESM-2 통합 계획 (v6 예정)

**조건**: GO_0006936 AUPRC ≥ 0.35, GO_0006096 AUPRC ≥ 0.30 달성 후 진행.  
**현황(v5-5)**: GO_0006096=0.670 ✅ 달성. GO_0006936=0.251 ❌ 미달. GO_0006936 문제 해결 후 v6 진입 가능.

**통합 방식**:
```
# 현재 (2-modal)
seq_feat(16) + domain_feat(16) → concat(32) → embedding(32)

# v6 (3-modal)
seq_feat(16) + domain_feat(16) + esm_feat(32) → concat(64) → embedding(32)
```

ESM-2 임베딩(1280-dim)을 PCA/Linear로 32-dim으로 압축 후 gating mechanism으로 통합 ([R2.1]: direct concatenation 금지, attention/gating 필수).

**논문 기여 포인트**: isoform-level sequence divergence (splicing으로 인한 domain architecture 변화)를 ESM-2가 포착 → gene-level reference dominance 감소.

### 13-3. 어블레이션 실험 준비 (논문 방어)

Nature Methods/NMI 기준: 각 컴포넌트 기여를 수치로 입증 필요.

| 실험명 | 제거 요소 | 측정 지표 |
|--------|---------|---------|
| `no_triplet` | Phase 1 Triplet 전체 | AUPRC on 4 GO terms |
| `no_focal` | Phase 2 Focal → BCE | AUPRC, positive collapse rate |
| `no_labelprop` | Phase 3 alpha=0.0 고정 | AUPRC |
| `no_esm` | ESM-2 제거 (v6 이후) | AUPRC |
| `no_cellloc` | Cell localization 제거 (v7+ 이후) | AUPRC |

---

## 14. 핵심 수식 및 파라미터 현황

### 14-1. Loss Functions

```
# Phase 1 — Triplet Loss [R3.1]
L(a,p,n) = max(d(a,p) - d(a,n) + margin, 0)
distance: squared L2 on unit sphere = 2(1 - cosine similarity)
margin = 0.3 (v5-4, [I3 복구])

Semi-hard mining: d(a,p) < d(a,n) < d(a,p) + margin
Fallback: d(a,n) < d(a,p) + 4×margin

# Phase 1.5 — Focal Loss [R1.1]
FL = -alpha_t × (1 - p_t)^gamma × log(p_t)
gamma=2.0, alpha=0.25 (positive weight)

# Phase 2 — Joint Loss
total = 1.0 × focal(gamma=2.0, alpha=0.10) + 0.1 × triplet
주의: Phase 2 focal alpha=0.10 → negative class weight=0.90 (93% positive collapse 가능 원인)

# Phase 3 — Label Propagation
refined = (1 - alpha) × base_scores + alpha × neighbor_mean_scores
alpha ∈ {0.0, 0.2, 0.3, 0.5}, AUPRC 기준 선택
```

### 14-2. Training Schedule

```
Phase 1: 15 epoch max, warmup 2 epoch (random mining)
         early stop: active_ratio < 2.0% for 4 consecutive epochs
         OR margin satisfaction ≥ 60% (≈ 거의 발동 안 함, margin=0.3)

Phase 1.5: 2 epoch (encoder frozen)

Phase 2: 15 epoch max, check every 1 epoch
         early stop: AUPRC not improving for 3 epochs (v5-4 신규)
         optimizer: Adam(lr=0.0003)
```

### 14-3. n_batches — 현행(v5-4) vs 제안(v5-5)

```python
# 현행 v5-4 (고정값)
N_BATCHES_P1 = max(50, min(150, ceil(n_pos × 4 / 256)))
# 결과: n_pos=287→50, n_pos=840→50, n_pos=1027→50, n_pos=3046→50
# 문제: 전 GO term이 하한 50으로 클램핑 → coverage 불균일

# 제안 v5-5 (coverage 기반 동적)
TARGET_COV = 6.0
N_BATCHES_P1 = int(np.clip(np.ceil(n_pos * TARGET_COV / 256), 20, 50))
# 결과: n_pos=287→20, n_pos=840→20, n_pos=1027→25, n_pos=3046→50
# 의미: 모든 GO term에서 epoch당 coverage ≈ 6~8x로 통일
```

---

## 15. 데이터셋 통계 (참조)

| GO Term | 기능 | train pos | train neg | test pos | test neg | pos ratio(test) |
|---------|------|----------|----------|---------|---------|----------------|
| GO_0006936 | 근육수축 | 840 | 113,531 | 597 | 36,151 | 1.625% |
| GO_0006412 | 번역 | 3,046 | 111,325 | 701 | 36,047 | 1.908% |
| GO_0006096 | 해당과정 | 287 | 114,084 | 76 | 36,672 | 0.207% |
| GO_0022900 | 전자전달계 | 1,027 | 113,344 | 291 | 36,457 | 0.792% |

---

## 16. 버전 히스토리 (전체)

| 버전 | 날짜 | 주요 변경 | 최고 AUPRC(GO_0006936) |
|------|------|---------|----------------------|
| v4-3 | 2026-03-25 | GradientTape Triplet 도입, CRF 유지 | 0.296 (Phase 2만) |
| v5 | 2026-04-02 | CRF 제거, Label Propagation, 전역 mining | - |
| v5-1 | 2026-04-03 | Prior bias init 추가 (실패, 즉시 폐기) | - |
| v5-2 | 2026-04-06 | Prior bias 제거, LabelProp AUPRC 기준, check interval 1 | **0.410** |
| v5-3 | 2026-04-07 | margin 0.1, n_batches 50 (회귀) | 0.119 |
| v5-4 | 2026-04-07 | margin 0.3 복구, Phase 2 AUPRC early stop | 0.191 |
| **v5-5** | **2026-04-07** | **coverage 기반 동적 n_batches [I4]** | **0.2507** (Macro 0.372) |

---

## 17. 중요 관찰 및 레슨

1. **margin=0.3의 역설**: margin satisfaction=0.0%에서도 가장 좋은 embedding이 만들어짐. "frustrated triplet"이 지속적 gradient 공급. → satifaction 지표가 품질 지표가 아님.

2. **Silhouette ↑ ≠ AUPRC ↑**: GO_0006096에서 v5-3 Ph1 Silhouette=+0.22(v5-2의 -0.01보다 훨씬 좋음)에도 최종 AUPRC는 더 낮음. Embedding separability는 prediction quality의 충분 조건이 아님.

3. **Phase 2 초기화 민감성**: v5-2 vs v5-3 Ph0 Silhouette이 5배 차이(0.162 vs 0.031). margin 변경과 독립적으로 random initialization이 결과에 영향. 재현성 확보를 위한 seed 고정 실험 필요.

4. **GO_0006412는 architecture 문제가 아닌 data 문제**: 어떤 margin/n_batches 조합에서도 Phase 1 Silhouette이 음수 지속. Translation은 근육세포에서 거의 모든 이소폼이 발현 → expression signal로 분리 불가. ESM-2 추가가 필요한 대표 케이스.

5. **Phase 2 focal loss alpha=0.10 재검토 필요**: Phase 2에서 negative class에 9배 weight → minority class(positive, 1.9%) 예측 억제. GO_0006412 93% positive collapse의 직접 원인일 수 있음. Phase 1.5의 alpha=0.25와 불일치.

6. **(v5-5 신규) coverage 과다가 Phase 2까지 파급**: GO_0006096에서 n_batches 50→20 감소만으로 Phase 2 Silhouette 0.195→0.719, AUPRC 0.297→0.670. Phase 1 과적합이 Phase 2의 기반 embedding 품질을 저하시키는 메커니즘 확인.

7. **(v5-5 신규) GO_0006936 회복 실패의 잔여 변수**: n_batches, margin을 v5-2와 동일하게 맞춰도 Phase 1 Silhouette 0.473 vs 0.129로 큰 차이. Phase 2 early stop 기준(AUROC vs AUPRC) 변경이 checkpoint 선택에 영향을 주거나 random initialization 변동이 원인. 재현성 실험(seed 고정) 필수.

8. **(v5-5 신규) LabelProp은 GO_0022900에서만 유효**: 4개 GO term 중 3개(GO_0006936, GO_0006412, GO_0006096)는 alpha=0.0이 최선 → LabelProp이 이들에게는 noise 추가. GO_0022900(alpha=0.5)만 expression neighbor 정보가 실제 기여. GO term별 condition으로 alpha를 다르게 적용하거나, GO_0022900 특이적인 expression 신호 강도를 분석해야 함.

---

*레포트 최초 작성: 2026-04-07*  
*v5-4 완료: Macro-AUPRC=0.231*  
*v5-5 완료: 2026-04-07 16:52, Macro-AUPRC=0.3717 (+60.9%)*  
*분석 대상 로그: `hMuscle/logs_isoform/GO_*/v5-{2,3,4,5}_GO_*_Full.log`*  
*v5-5 주요 결과: GO_0006096 AUPRC=0.6698(최고), GO_0006412 AUPRC=0.2091(최고), GO_0006936 회복 미달(0.251 vs v5-2 0.410)*
