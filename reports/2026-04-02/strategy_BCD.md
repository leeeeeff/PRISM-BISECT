# Expression 통합 전략 B / C / D — 상세 설계
**날짜**: 2026-04-02  
**목적**: Co-expression / Expression 데이터를 이소폼 수준 기능 예측에 통합하는 세 가지 전략의 설계 원리, 실행 순서, 논문 기여 포지셔닝

---

## 전략 개요

| 전략 | 명칭 | 구현 시점 | 핵심 아이디어 |
|---|---|---|---|
| **B** | Expression as Input Feature | v5 (현재) | 발현 프로파일을 세 번째 입력 모달로 DNN에 직접 통합 |
| **C** | Expression-guided Triplet Mining | v5-1 | 발현 유사도 기반 hard negative 선택으로 gene-level bias 공격 |
| **D** | Isoform Co-expression GNN | v6 / future work | 이소폼 수준 co-expression 그래프 + GNN attention으로 pairwise context 학습 |

---

## 전략 B: Expression as Input Feature

### 핵심 원리

발현 패턴을 CRF의 외부 regularizer가 아닌, **DNN 내부의 학습 가능한 특징**으로 internalize.

```
seq_input  → Conv1D + PyramidPooling → Dense(16) → seq_feat   (16-dim)
domain_input → LSTM(16)              → domain_feat  (16-dim)
expr_input → Dense(16, relu) + Dropout(0.3)  → expr_feat   (16-dim)  ← NEW
                     ↓ concatenate(48-dim)
               Dense(16) → L2-normalize → embedding_out
                                        ↓
                              Dense(1, sigmoid) → prediction
```

### 데이터 사양 (확인 완료)

| 항목 | 값 |
|---|---|
| 파일 | `bambu_data/CPM_transcript.txt` |
| 전체 이소폼 수 | 392,585 |
| Test set (my_isoform_list_fixed) | 36,748 — **100% 존재** |
| Cell-type 컬럼 수 | 24 (D1~D15, N1~N16 샘플) |
| 희소성 (zero ratio) | ~83.5% |
| CPM 범위 | [0, 117,021] → log1p 후 [0, 11.67] |

### 전처리 설계

```python
# log1p 변환만 적용 (standardization 없이)
# 이유: training isoform은 expression = 0이므로,
#       standardization 시 -mean/std → 음수값 발생하여 train/test 불일치
expr_raw = CPM_matrix[isoform_order, :]  # (36748, 24)
X_test_expr = np.log1p(expr_raw)         # [0, 11.67] 범위
X_train_expr = np.zeros((n_train, 24))   # training isoform은 발현 데이터 없음
```

### Train/Test Mismatch 대응 전략: Expression Dropout

```
Training phase: X_train_expr = zeros (항상)
                → model이 "expression = 0일 때도 seq+domain으로만 판단"을 학습
Expr Dropout p=0.3:
                → training 중 expression branch를 30% 확률로 0으로 강제
                → test-time에 real expression이 들어왔을 때의 distribution shift 완화
                → 모델이 expression에만 over-reliant하지 않도록 정규화
Inference: X_test_expr = log1p(CPM) (실제 발현값)
                → expression이 추가 신호로 작동
```

### 생물학적 근거

같은 유전자의 이소폼 A1, A2가 있을 때:
- A1: 근육세포에서 높은 CPM, 신경세포에서 낮음
- A2: 근육세포에서 낮은 CPM, 신경세포에서 높음

gene-level annotation은 둘에게 동일하게 적용됨. 그러나 발현 프로파일은 달라서, 모델이 "근육 특이적 발현 패턴 → GO:0006936 가능성"을 학습할 수 있음.

### Gene-level Bias 관점

- 같은 유전자의 이소폼이라도 발현 벡터가 다르면 → 다른 embedding 생성
- `isoform_emb`이 expression channel을 포함하므로 gene-level 동일성에서 벗어남
- 아키텍처 규칙 준수: `isoform_emb` primary, PFN forward pass 서명 변경 없음

### CRF 제거 (동시 수행)

EXP-01에서 확인: CRF가 4개 GO term 전부에서 AUROC를 평균 -0.21 감소시킴.  
v5에서는 Phase 2 출력 또는 PFN 출력을 Final로 사용.

### v5 변경 사항 목록

| 항목 | v4-3 | v5 |
|---|---|---|
| 입력 모달 | seq + domain (2채널) | seq + domain + expr (3채널) |
| feature_model 출력 차원 | 32 | 48 |
| EMB_DIM | 32 | 48 |
| Phase 3 | Focal + PFN + CRF | Focal + PFN (CRF 제거) |
| Final output | CRF pos_prob | PFN score |
| 파일명 | v4-3_integrated_full_model.py | v5_integrated_full_model.py |

---

## 전략 C: Expression-guided Triplet Mining

### 왜 지금 안 하는가

1. **선행 조건**: B가 먼저 동작해야 C가 의미를 가짐  
   - C의 목적: "발현이 비슷해도 sequence/domain으로 기능을 구분하도록 강제"
   - 그런데 모델에 expression 채널이 없으면, triplet 선택 기준과 학습 목표가 불일치
   - B로 expression이 입력에 들어간 후에야 "발현 알고 있으니 sequence로 판단하라"가 의미있음

2. **Ablation 분리**: B와 C를 동시 도입 시 각 기여를 논문에서 분리 불가

3. **데이터 파이프라인**: C에 필요한 이소폼 간 발현 유사도 행렬이 B 구현 중 자연스럽게 만들어짐

### 설계 (v5-1에서 구현)

```python
# 이소폼 발현 유사도 행렬 사전계산
# X_test_expr: (36748, 24), log1p normalized
expr_sim = cosine_similarity(X_test_expr)  # (36748, 36748)

# Expression-guided Negative 선택
# 조건:
#   1. 다른 유전자 소속 (cross-gene, 기존 규칙 준수)
#   2. 발현 유사도 > threshold (cos_sim > 0.7)  ← 발현이 비슷한데 기능이 다른 케이스
#   3. 현재 embedding 거리가 가장 가까운 top-K에서 선택 (hard negative 유지)

def build_expr_guided_negatives(embeddings, expr_sim, gene_ids,
                                 pos_indices, neg_indices, batch_size):
    # 발현 유사도 높은 다른-유전자 이소폼 우선 negative로 선택
    # 모델이 "발현이 같아도 기능은 다를 수 있다"를 학습
```

### 논문 기여 포인트

> "기존 hard negative mining은 embedding 거리만 고려하므로, 발현 패턴이 유사한 different-function 이소폼을 hard negative로 활용하지 못한다. 우리의 expression-guided triplet mining은 발현 기반 confounding factor를 명시적으로 다루어 gene-level bias를 직접 공격한다."

---

## 전략 D: Isoform Co-expression GNN

### 왜 지금 안 하는가

1. **아키텍처 규칙 위반**: CLAUDE.md "PFN backbone — do not replace without ablation proof"  
   - B, C의 ablation 결과 없이는 GNN 도입 정당화 불가

2. **공학 비용**: 
   - 36748 × 36748 co-expression graph: ~10GB 메모리
   - PyTorch Geometric + Keras 혼합 통합 엔지니어링
   - Node212 GPU에서 OOM 위험

3. **논문 포지셔닝**: D는 별도 contribution 규모  
   - B+C로 v5 논문을 완성한 후, D는 후속 연구 또는 revision addition

### 설계 (v6 / future work)

```
이소폼 co-expression 그래프:
  - 노드: 36,748 hMuscle 이소폼
  - 엣지: 이소폼 간 Pearson correlation of log1p(CPM) > threshold
  - 엣지 가중치: correlation 값

GNN layer (attention):
  h_i = MLP(concat(h_i, Σ_j α_ij * h_j))
  α_ij = softmax(e_ij / Σ_k e_ik)  # attention weight
  e_ij = LeakyReLU(a^T [Wh_i || Wh_j])  # edge score

CRF와의 핵심 차이:
  - CRF: fixed theta, pairwise potential이 within-gene isoform 균일화
  - GNN: learned attention, within-gene vs cross-gene 구분 가능
  - GNN은 "이 이웃은 같은 유전자라 참고 불필요"를 학습할 수 있음
```

### 논문 기여 포인트 (미래)

> "CRF의 fixed pairwise regularization은 이소폼 수준 판별과 양립 불가능하다. GNN의 학습 가능한 attention이 co-expression 정보를 이소폼 수준 판별과 공존 가능한 방식으로 통합한다."

---

## 실행 로드맵

```
2026-04 (현재)
  → v5: 전략 B (Expression Input) + CRF 제거
  → ablation: no_expr, no_focal, no_triplet
  → 목표: GO_0006936 AUROC 0.80+

2026-05
  → v5-1: 전략 C (Expression-guided Triplet)
  → ablation: no_expr_triplet
  → 논문 Table 2: B vs B+C delta

2026-06~07
  → 전략 D 설계 검토 (B+C baseline 확립 후)
  → 또는 ESM/cell_loc/PPI 통합으로 방향 전환

2026-10~11
  → Nature Methods 제출 타겟
```

---

## 핵심 방어 논리 (논문 Introduction / Methods)

> "원본 DIFFUSE는 expression 데이터를 gene-level pairwise CRF regularization에 사용했다.
> 이 방식은 gene-level 기능 예측에는 유효했으나, isoform-level 판별에서는
> within-gene score 균일화를 야기하여 성능을 오히려 저하시킨다 (AUROC -0.27).
> 
> 본 연구는 expression을 이소폼 내재적 입력 피처로 재설계(전략 B)하고,
> expression 유사도 기반 triplet mining으로 gene-level confounding을 명시적으로
> 제거(전략 C)함으로써, 동일한 생물학적 정보를 gene-level bias 없이 활용한다."

---

*작성: 2026-04-02*  
*데이터 확인: bambu CPM 36,748 test isoform 100% 존재, 24 cell-type, log1p 최대 11.67*
