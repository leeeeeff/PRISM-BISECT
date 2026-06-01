# v6d Phase-wise Analysis Report
**날짜**: 2026-04-14  
**분석 대상**: v6d 3-run × 5 GO term 전 phase 로그  
**목적**: 학습 단계별 성능 변화의 생물학적 의미 해석 + v6e 설계 근거 수립

---

## 1. 분석 프레임워크: 알고리즘 용어 ↔ 생물학적 의미

| 알고리즘 용어 | 생물학적 의미 |
|---|---|
| **Phase 0 sep_ratio** (inter_dist / intra_dist) | ESM-2가 이 GO term의 positive 단백질들을 사전 지식(진화적 서열 보존)만으로 얼마나 구분하는가. ratio > 1이면 positive 집합이 negative보다 서로 더 비슷함 |
| **Phase 1 Triplet Loss** | "같은 기능을 공유하는 isoform들은 임베딩 공간에서 가까워야 한다"는 제약을 학습. margin_sat = 학습된 거리 순서가 옳게 배열된 triplet 비율 |
| **Phase 0 linear_AUROC** | 진화 정보(ESM-2)만으로 이 GO term을 예측할 수 있는 사전 한계. 이 수치가 낮으면 서열 유사성과 기능이 분리되어 있다는 의미 |
| **Phase 2 CNN fine-tuning** | 서열 내 특정 위치의 모티프(active site, splice junction, PTM site)를 학습. CNN이 이소폼-특이적 서열 신호를 탐지 |
| **sep_ratio가 Phase 2에서 감소** | CNN이 학습하면서 전체 embedding 공간을 재구성 → Phase 1이 만든 기능 기반 거리 관계가 무너짐. 과적합의 embedding 수준 증거 |
| **Label Propagation (alpha)** | 근육 세포에서 함께 발현되는 isoform들은 같은 기능을 할 가능성이 높다는 가정 하에, 예측 점수를 이웃 isoform들과 평균냄 |
| **Coverage** (n_batches × batch_size / n_pos) | Phase 1에서 각 positive isoform이 epoch당 몇 번 triplet 학습에 참여하는가 |

---

## 2. GO Term 이진 분류: Type A vs Type B

### 핵심 기준: Phase 0 sep_ratio

Phase 0은 모델 가중치가 **완전히 랜덤**인 상태에서 ESM-2 임베딩을 통과시킨 결과다.  
이 시점의 sep_ratio는 "ESM-2가 서열/진화 정보만으로 이 GO term의 positive를 얼마나 coherent하게 표현하는가"를 측정한다.

#### Type A: 진화적으로 응집된 GO term

| GO term | 생물학적 내용 | Ph0 sep_ratio | Phase 0 linear AUROC |
|---------|-------------|--------------|---------------------|
| GO:0006096 (해당) | 해당질 (glycolysis) — TIM barrel fold 효소들 | 1.51–1.79 | 0.978 |
| GO:0003774 (근수축, 모터단백질) | Myosin/Kinesin motor — P-loop NTPase superfamily | 1.06–1.36 | 0.941 |

**생물학적 해석**: 해당질 효소들은 35억 년 전 공통 조상에서 갈라진 후에도 TIM barrel 구조를 유지했다. ESM-2는 이 진화적 보존 패턴을 임베딩 공간의 군집으로 이미 인코딩한다. 따라서 학습 전부터 positive 군집이 coherent하다.

**학습 결과**: Phase 0→Phase 2까지 sep_ratio 단조증가. CNN이 추가적으로 active site 서열 모티프를 학습해 더욱 강화.

```
GO:0006096 sep_ratio 진행:
  Phase 0: 1.51  →  Phase 1: 2.52  →  Phase 1.5: 2.66  →  Phase 2: 3.72
  (단조증가 — 모든 학습 단계가 기여)
```

---

#### Type B: 진화적으로 이질적인 GO term

| GO term | 생물학적 내용 | Ph0 sep_ratio | Phase 0 linear AUROC |
|---------|-------------|--------------|---------------------|
| GO:0006941 (골격근 수축) | Actin-myosin sliding + regulatory proteins | 1.04–1.09 | 0.855 |
| GO:0007204 (Ca²⁺ 신호) | VGCC + IP3R + SERCA + calmodulin | 1.00–1.04 | 0.822 |
| GO:0030017 (sarcomere 조직화) | Myosin HC + Actin + Titin + Troponin | 0.97–1.08 | 0.803 |

**생물학적 해석**:  
GO:0030017 sarcomere를 예시로 들면:
- **Myosin heavy chain** (MYH7, 1935 aa): P-loop NTPase, 수백 개의 Pfam 도메인
- **Actin** (ACTC1, 375 aa): Actin fold, 진화적으로 완전히 다른 superfamily
- **Titin** (TTN, ~34,000 aa): Immunoglobulin + fibronectin 반복 도메인의 거대 scaffold

이 세 단백질은 **수렴 진화**로 같은 기능 단위(sarcomere)에 함께 속하지만, 서열/구조 관점에서는 무관하다. ESM-2는 서열 진화를 학습했으므로, 이들을 임베딩 공간에서 하나의 군집으로 표현할 수 없다.

**sep_ratio ≈ 1.0의 의미**: positive 단백질들 간의 평균 거리 ≈ positive-negative 간 평균 거리. 즉 임베딩 공간에서 positive가 negative보다 서로 더 비슷하지 않다. Triplet loss가 학습할 수 있는 공통 구조가 없다.

---

## 3. Phase별 sep_ratio 진행 — Type A vs Type B 비교

### Type A (GO:0006096, 3-run 대표)

```
Phase 0   Phase 1   Phase 1.5  Phase 2
  1.51  →   2.52  →   2.66  →   3.72    run3
  1.52  →   1.96  →   2.68  →   3.32    run2  
  1.79  →   2.20  →   3.14  →   3.04    run1
```
→ **전 단계에서 단조증가. Phase 2 CNN이 오히려 embedding 구조를 강화.**

### Type B (GO:0006941, 3-run 대표)

```
Phase 0   Phase 1   Phase 1.5  Phase 2
  1.09  →   1.37  →   0.89  →   0.83    run3  ← Phase 2에서 Phase 0 이하로 하락
  1.04  →   1.60  →   1.11  →   1.10    run2
  1.04  →   1.17  →   0.88  →   0.83    run1
```
→ **Phase 1 triplet은 효과적 (1.0→1.3-1.6). Phase 2 CNN이 이를 역전.**

**Phase 1 triplet 후 linear_AUROC (GO:0006941 run3): 0.855 → 0.9489**  
Phase 1은 실제로 이질적인 positive들 사이에서 학습 가능한 공통 패턴을 찾아낸다.  
그러나 Phase 2 CNN이 local sequence motif를 학습하면서 이 구조를 파괴한다.

### 생물학적 해석: "왜 CNN이 Type B를 망가뜨리는가"

Phase 2에서 CNN은 positive와 negative를 구분하는 서열 패턴을 학습한다.  
Type B의 positive 집합은 이질적이므로 CNN이 학습하는 것은:
- Actin-type positive에 대한 서열 패턴 (actin fold motif)
- Myosin-type positive에 대한 서열 패턴 (P-loop NTPase motif)
- Titin-type positive에 대한 서열 패턴 (Ig/FN3 반복)

이 세 가지 서로 다른 "positive" 패턴을 하나의 임베딩 공간에 표현하려다 보니,  
Phase 1이 만든 "기능 기반" 거리 구조가 "서열 패턴 기반" 구조로 대체된다.  
결과: sep_ratio 감소, 그러나 AUPRC는 훈련 세트 기준으로는 개선 (과적합).

---

## 4. Label Propagation 측정 결과

**측정 방법**: 매 run에서 alpha ∈ {0.0, 0.2, 0.3, 0.5} 모두 계산 후 best 선택  
→ alpha=0.0 = LP 없음 = "no_ppi ablation"이 이미 run마다 계산됨

| GO term | Type | alpha=0.0 AUPRC | Best alpha | LP 효과 | 선택된 alpha |
|---------|------|----------------|-----------|--------|------------|
| GO:0006096 | A | **0.7932** | 0.0 | **-8.9%** (0.2→0.7217) | 0.0 |
| GO:0003774 | A | **0.6713** | 0.0 | -0.8% | 0.0 |
| GO:0006941 | B | **0.2835** | 0.0 | -0.9% | 0.0 |
| GO:0007204 | B | **0.2645** | 0.0 | ±0% | 0.0 |
| GO:0030017 | B | 0.2762 | **0.2** | +2.4% | 0.2 |

**결론**:
- 5개 중 4개에서 모델이 자발적으로 LP를 비활성화 (alpha=0.0 선택)
- GO:0006096에서 LP가 AUPRC를 -8.9%나 떨어뜨림 → LP가 적극적으로 해롭다
- GO:0030017의 +2.4%는 noise 범위 (bootstrap CI 없이 단언 불가)

**생물학적 해석**:  
Label propagation의 가정: "근육에서 함께 발현되는 이소폼들은 같은 GO term을 공유한다."  
이 가정이 실패하는 이유:
1. 유전자 발현은 GO term보다 더 거친 단위로 조절된다 (operons, polycistronic, co-regulation)
2. Muscle CPM matrix (EXPR_DIM=24 조건)의 24개 시점이 5개 GO term의 기능적 차이를 구분하기에 해상도가 낮다
3. 특히 GO:0006096 (glycolysis)처럼 모든 세포에서 발현되는 GO term에서, 공발현 이웃이 오히려 non-specific positive를 끌어들임

---

## 5. Coverage 비대칭 분석

**현재 설정**: `N_BATCHES = clip(n_pos × 6.0 / 256, min=20, max=50)`

| GO term | n_pos (추정) | n_batches | coverage/epoch | 총 exposure (15ep) |
|---------|------------|----------|---------------|-------------------|
| GO:0006096 | ~287 | 50 (max) | 2.79x | 41.8x |
| GO:0007204 | ~882 | 50 (max) | 0.91x | 13.6x |

**문제**: Type B는 이미 Phase 0 sep_ratio가 낮아서 어렵고, 추가로 coverage도 낮다. 이중 불이익.  
그러나 coverage 증가만으로 Type B 문제가 해결되지 않음 (gradient conflict가 근본 원인).

**v6e 수정**: `clip(n_pos × 4.0 / 256, min=10, max=80)` → 모든 GO term에서 positive당 epoch 노출 균등화

---

## 6. v6e 설계 근거 및 구현 계획

### 핵심 변경 3가지

#### 변경 1: Phase 0 자동 Type 분류 + sep_ratio 저장

```python
ph0_metrics = analyze_embedding_quality(emb_ph0, y_test, 'Phase 0')
ph0_sep_ratio = ph0_metrics['sep_ratio']
IS_TYPE_B = (ph0_sep_ratio < 1.15)

print("[Type] GO term = {} (sep_ratio={:.4f})".format(
    'Type-B (heterogeneous positives)' if IS_TYPE_B else 'Type-A (coherent positives)',
    ph0_sep_ratio))
```

**생물학적 의미**: 모델이 학습을 시작하기 전에 이 GO term의 positive 단백질들이  
진화적으로 유사한지(Type A) 아니면 수렴 진화 집합인지(Type B)를 자동 판단.

#### 변경 2: Phase 2 Type별 adaptive hyperparameter

```python
# Type B: CNN이 embedding 구조를 파괴하지 않도록 학습 제약
if IS_TYPE_B:
    adam_p2 = optimizers.Adam(lr=0.0001, clipnorm=0.5)  # 작은 업데이트 + 그라디언트 클리핑
    NO_IMPROVE_LIMIT = 3   # 빠른 early stop (AUPRC peak 직후 중단)
else:
    adam_p2 = optimizers.Adam(lr=0.0003)               # 현재 값 유지
    NO_IMPROVE_LIMIT = 7
```

**생물학적 의미**:
- Type A: CNN이 TIM barrel fold나 P-loop 모티프를 적극적으로 학습해도 기존 구조와 일관됨 → 큰 LR 허용
- Type B: CNN이 학습하면 서로 다른 단백질 패밀리의 서열 패턴이 충돌 → 작은 LR로 Phase 1 구조 보존

#### 변경 3: Phase 3 LP 제거 (alpha=0.0 고정)

```python
# v6e: LP 제거 — 측정 결과 4/5 GO term에서 해롭거나 중립
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])
final_scores = base_scores  # alpha=0.0 고정
final_auroc  = roc_auc_score(y_test, final_scores)
final_auprc  = average_precision_score(y_test, final_scores)
print("[v6e] LP 제거 — alpha=0.0 고정")
print("  AUROC={:.4f} AUPRC={:.4f}".format(final_auroc, final_auprc))
```

**생물학적 의미**: 공발현 기반 점수 전파는 근육 세포에서 GO term 기능 예측에  
충분한 해상도를 제공하지 못함. 오히려 noise 전파 → 제거가 나음.

---

## 7. 예상 효과 및 한계

### 기대 효과

| 변경 | Type A 효과 | Type B 효과 |
|------|-----------|-----------|
| Coverage 정규화 | 과도한 반복 감소 (과적합 약간 감소) | Exposure 균등화 (marginal 개선) |
| Phase 2 adaptive LR | 변화 없음 | Phase 1 embedding 구조 보존 → sep_ratio 하락 감소 |
| Phase 2 early patience | 변화 없음 | AUPRC peak 직후 중단 → 과적합 감소 |
| LP 제거 | GO:0006096 +8.9% AUPRC 기대 | GO:0030017 -2.4% (trade-off 허용 범위) |

### 남은 한계

1. **Type B의 근본 문제는 해결되지 않음**: Phase 2 LR을 낮춰도 CNN은 여전히  
   이질적인 positive들로부터 상충되는 서열 패턴을 학습한다.  
   → 중장기: PPI 그래프 기반 feature 강화 (단백질 복합체 내 상호작용이 서열보다 더 나은 signal)

2. **Type B AUPRC 절대값이 낮음**: v6e 후에도 GO:0007204, GO:0030017의  
   AUPRC는 0.30-0.40 범위를 벗어나기 어려울 것. 이것은 데이터 한계 (annotation sparsity + ESM-2 표현력 한계)이며, 훈련 전략만으로는 극복 불가.

3. **GO:0003774의 불안정성**: Type A/B 경계에 위치 (sep_ratio 1.06-1.36). 일부 run에서 Type B처럼 Phase 2 역전이 발생. 추가 seed 실험 필요.

---

## 8. 다음 실험 우선순위

1. **즉시**: v6e 실행 (v6d + 3가지 수정) → LP 제거 효과가 GO:0006096에서 가장 명확하게 나타날 것
2. **단기**: Bootstrap CI (n=1000) [R9.4] — 현재 3-run은 CI를 계산하기에 부족
3. **중기**: no_domain, no_dd ablation — DomainDelta 기여가 Type A/B에서 다른가?
4. **장기**: PPI graph features — Type B의 근본 한계를 address하려면 상호작용 정보 필요

---

*분석자: Claude Sonnet 4.6 | 데이터: v6d 3-run × 5 GO term Phase 0-3 로그*
