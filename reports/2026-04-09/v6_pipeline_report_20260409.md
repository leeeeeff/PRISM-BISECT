# v6_integrated_full_model — 로직 파이프라인 레포트

**작성일**: 2026-04-09  
**대상 파일**: `hMuscle/model/v6_integrated_full_model.py`  
**v5-5 대비 핵심 변경**: 서열 모달리티 교체 (Integer Embedding+CNN → ESM-2 640-dim MLP)  
**실행 중 GO terms**: GO:0006941, GO:0007204, GO:0030017, GO:0003774, GO:0006096

---

## 1. 전체 파이프라인 개요

```
[데이터 로딩]
    ├── ESM-2 임베딩 (6개 .npy)
    ├── Domain feature
    ├── Gene/Isoform ID
    └── Expression matrix (test only)
         ↓
[레이블 생성 + 업샘플]
    ├── generate_label × 2 (deterministic 2-call 방식)
    └── upsample × 1 (hstack trick, seed=42)
         ↓
[모델 구조]
    ├── ESM-2 MLP branch: 640→256→128→64
    ├── Domain LSTM branch: Embedding→LSTM(16)
    └── Fusion: concat[80] → Dense(32) → L2_norm → embedding[32]
         ↓
[4단계 학습]
    Phase 0 → Phase 1 → Phase 1.5 → Phase 2
         ↓
[Phase 3: Test-time Label Propagation]
    └── KNN on expression space → alpha 최적 선택
         ↓
[저장 및 시각화]
    UMAP / Score distribution / Embedding quality table
```

---

## 2. 데이터 로딩

### 2-1. 입력 데이터 구성

| 변수 | 파일 | 형상 | 설명 |
|------|------|------|------|
| `X_train_esm2` | `esm2_train_human.npy` | (31668, 640) | Human 단백질 ESM-2 임베딩 |
| `X_train_esm2_mask` | `esm2_train_human_mask.npy` | (31668, 1) | 서열 존재 여부 (1=있음, 0=없음) |
| `X_train_other_esm2` | `esm2_train_swissprot.npy` | (82703, 640) | SwissProt 단백질 ESM-2 임베딩 |
| `X_train_other_esm2_mask` | `esm2_train_swissprot_mask.npy` | (82703, 1) | SwissProt 서열 존재 여부 |
| `X_test_esm2` | `esm2_embeddings_t30_150M.npy` | (36748, 640) | BambuTx 이소폼 ESM-2 임베딩 |
| `X_test_esm2_mask` | `esm2_mask.npy` | (36748, 1) | Test 이소폼 서열 존재 여부 |
| `X_train_dm` | `human_domain_train.npy` | (31668, dm_dim) | Human 도메인 feature (정수 인덱스) |
| `X_train_other_dm` | `swissprot_domain_train.npy` | (82703, dm_dim) | SwissProt 도메인 feature |
| `X_test_dm` | `domain_matrix.npy` | (36748, dm_dim) | Test 도메인 feature |
| `X_test_expr` | `expr_matrix_fixed.npy` (캐시) | (36748, 24) | CPM 발현량 log1p 변환 (Phase 3용) |

**ESM-2 생성 방법**:
- Train: `compute_esm2_train_embeddings.py` — `esm2_t30_150M_UR50D` 모델, mean-pool
- Test: `compute_esm2_embeddings.py` — BambuTx TransDecoder ORF 서열 입력

**Mask 설계**: 서열 없는 이소폼은 mask=0 → ESM-2 feature zeroed out → domain feature만 기여  
**실측 coverage**: Human train=100%, SwissProt train=100%, Test=100%

### 2-2. 레이블 수집

```python
# human_annotations.txt, swissprot_annotations.txt에서 선택된 GO term 포함 항목 수집
positive_Gene = [gene_id for fname in ['human_annotations.txt', 'swissprot_annotations.txt']
                 for line in open(fname) if selected_go in line.split('\t')[1:]]
```

---

## 3. 레이블 생성 및 업샘플 파이프라인

### 3-1. generate_label 2-call 설계

v6의 핵심 데이터 파이프라인. `generate_label`이 **결정론적(deterministic)**임을 이용해  
동일한 gene ID 인수로 두 번 호출하면 동일한 `add_index`(추가 행 선택)를 보장.

```
Call ①: generate_label(X_train_esm2, X_train_dm, ...)
         → 반환: y_train, y_test, y_all, crf_bag_index, gene_index, gene_count,
                 _dummy_esm2 (버림), X_train_dm_comb
         역할: labels + domain feature 결합

Call ②: generate_label(X_train_esm2, X_train_esm2_mask, ...)
         → 반환: _, _, ..., X_train_esm2_comb, X_train_esm2_mask_comb
         역할: ESM-2 임베딩 + mask 결합
         보장: Call ①과 동일한 add_index → 행 정렬 일치
```

**왜 2번 호출하는가?**  
`generate_label`의 7번째 반환값은 `X_train_seq_comb`, 8번째는 `X_train_dm_comb`.  
ESM-2(640-dim)와 mask(1-dim)를 동시에 결합할 방법이 없으므로,  
mask를 dm 슬롯에 넣어 두 번째 호출로 추출.

### 3-2. upsample hstack trick

`upsample` 함수는 내부적으로 하나의 랜덤 인덱스 배열로 seq/dm을 동시에 oversampling.  
ESM-2(640)와 dm을 각각 호출하면 다른 랜덤 인덱스 → **행 불일치** 발생.  
해결: 모든 feature를 하나의 배열로 합친 후 upsample 1회 → 이후 분리.

```python
# 통합 (N, 640+1+dm_dim)
X_train_combined = np.hstack([
    X_train_esm2_comb,              # (N, 640) float32
    X_train_esm2_mask_comb,         # (N, 1)   float32
    X_train_dm_comb.astype(float32) # (N, dm_dim) int→float (dm_dim < 2^23 보장)
])

np.random.seed(42)  # 재현성 고정
X_train_combined_upsmp, _, y_train_upsmp, _ = upsample(y_train, gene_index, gene_count,
                                                         X_train_combined, placeholder, flag)

# 분리
X_train_esm2_upsmp      = X_train_combined_upsmp[:, :640].astype(float32)
X_train_esm2_mask_upsmp = X_train_combined_upsmp[:, 640:641].astype(float32)
X_train_dm_upsmp        = X_train_combined_upsmp[:, 641:].astype(int32)
```

### 3-3. Coverage 기반 동적 n_batches [I4]

Phase 1 triplet 배치 수를 positive 샘플 수에 비례하여 동적 결정:

```
n_batches = clip(ceil(n_pos_train × 6.0 / 256), min=20, max=50)
```

목표 coverage = positive 샘플 1개당 epoch당 평균 6.0회 노출.

---

## 4. 모델 구조

### 4-1. 전체 아키텍처

```
입력:
  esm2_input  (640,)  float32   ← ESM-2 mean-pool embedding
  esm2_mask   (1,)    float32   ← 서열 존재 여부 gate
  domain_input (dm_dim,) int32  ← 도메인 ID 시퀀스

ESM-2 MLP branch (isoform-specific, 먼저 계산):
  Dense(256, l2=1e-5) → ReLU → Dropout(0.2)
  → Dense(128, l2=1e-5) → ReLU
  → Dense(64, l2=1e-5, relu, name='esm2_feat')   ← 64-dim
  → Lambda(feat × mask, name='esm2_gated')         ← 64-dim (mask=0이면 zeroed)

Domain branch (gene context):
  Embedding(domain_emb_dim, 32, mask_zero=True)
  → LSTM(16, name='domain_feat')                   ← 16-dim

Fusion (EMB_DIM=80):
  concatenate([esm2_gated, domain_feat])            ← 80-dim
  feature_model 출력 ↑

Head:
  Dense(32, l2=1e-5) → ReLU → Dropout(0.2)
  → Lambda(L2_normalize, name='embedding_out')      ← 32-dim
  → Dense(1, sigmoid, name='prediction_out')        ← 스칼라
```

**압축비**: ESM-2 640 → 64 = **10×** (v5-5 CNN 대비 이론적 정보 보존 향상)  
**아키텍처 원칙 준수**: isoform_emb(ESM-2) 먼저 계산 후 gene context(domain) 결합 [R2.1]  
**직접 concatenation 금지 우회**: domain은 LSTM → 독립 계산 후 concat [R2.1]

### 4-2. 모델 객체 구성

| 객체 | 입력 | 출력 | 용도 |
|------|------|------|------|
| `feature_model` | [esm2, mask, dm] | concat[80] | Phase 1 triplet embedding 추출 |
| `base_model` | [esm2, mask, dm] | [embedding[32], pred[1]] | Phase 0/1.5/2 예측 및 저장 |
| `classification_model` | [esm2, mask, dm] | pred[1] | Phase 1.5 focal loss 학습 |
| `triplet_model` | 9 inputs (a/p/n × 3) | [triplet_loss, pred_a] | Phase 2 joint 학습 |

---

## 5. 손실 함수

### 5-1. Focal Loss [R1.1]

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

| 단계 | γ | α | 목적 |
|------|---|---|------|
| Phase 1.5 | 2.0 | 0.25 | Encoder frozen 상태에서 head 학습 |
| Phase 2 | 2.0 | 0.10 | Joint fine-tuning (α 낮춰 과대예측 억제) |

### 5-2. Triplet Loss [R3.1]

```
L(a, p, n) = max(d(a,p) - d(a,n) + margin, 0)
margin = 0.3 [I3], distance = squared L2
```

- Phase 1: GradientTape 직접 계산 (feature_model 가중치 업데이트)
- Phase 2: triplet_model.train_on_batch (joint 학습)
- Negative 전략: **cross-gene 필수** [R2.1], Phase 1 = semi-hard mining

### 5-3. Combined Loss (Phase 2)

```python
triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.10)],
    loss_weights=[0.1, 1.0])  # λ_triplet=0.1, λ_focal=1.0
```

---

## 6. 4단계 학습 프로세스

### Phase 0: Untrained Baseline

학습 없이 초기화된 모델의 예측값과 임베딩 저장. 이후 단계와 비교 기준점.

### Phase 1: Embedding-based Triplet (최대 15 epoch)

**목표**: ESM-2 공간에서 positive/negative 클러스터 분리 (margin_sat 개선)

```
Warmup (epoch 1~2): random triplet sampling
Main   (epoch 3~): semi-hard negative mining

Early stop 조건:
  ① active_rate < 2% 연속 4 epoch → triplet 포화 신호
  ② margin_sat(5 epoch마다) ≥ 60% → 충분한 분리 달성

Semi-hard 정의: d(a,p) < d(a,n) < d(a,p) + margin
  → 학습 효과 있는(active) 그러나 너무 어렵지 않은 negative
```

**Embedding refresh**: 10 batch마다 feature_model로 전체 임베딩 재계산 → 최신 embedding 기반 mining

### Phase 1.5: Linear Probing (2 epoch, encoder frozen)

**목표**: frozen encoder 위에서 head가 GO term 특이적 decision boundary 학습

```
feature_model 전체 layers: trainable=False
classification_model.compile(focal_loss γ=2.0, α=0.25)
배치 처리: BATCH_SIZE=512, 단순 순차 (ESM-2 고정 크기 → make_batch 불필요)
epoch 내 positive/negative 혼합 후 shuffle
```

### Phase 2: Joint Fine-tuning — Focal + Triplet (최대 15 epoch)

**목표**: 전체 모델 end-to-end 학습, AUPRC 최대화

```
feature_model 전체: trainable=True (unfreeze)
triplet_model로 학습 (focal + triplet joint)

배치 처리:
  outer loop: 512-sample 그룹
  inner loop: 그룹 내 64-sample triplet 배치 (get_triplet_batch_ingroup_v6)

[I5] AUPRC 기반 early stop [R9.1]:
  매 epoch 후 test AUPRC 계산
  no_improve_count ≥ 3 → best weights 복원 및 종료
```

### Phase 3: Test-time Expression Label Propagation [I2]

**목표**: hMuscle 발현 패턴으로 Phase 2 점수 보정

```python
for alpha in [0.0, 0.2, 0.3, 0.5]:
    refined = (1 - alpha) × base_scores + alpha × propagated_scores
    # propagated_scores: KNN(k=15) on log1p(CPM) cosine similarity

# AUPRC 기준 best alpha 선택 [R9.1]
```

**KNN 구성**: 코사인 유사도, 임계값 0.1 이하 엣지 제거 (noise 억제)

---

## 7. 임베딩 품질 지표 체계

각 Phase 후 자동 계산 및 출력:

| 지표 | 수식 / 방법 | 의미 |
|------|------------|------|
| Silhouette (cosine) | `silhouette_score(emb, labels, metric='cosine')` | >0 = 분리 가능 |
| Sep.Ratio | `inter_dist / intra_dist` | >1.0 = 양/음 클러스터 분리 |
| Centroid dist | `||mean(pos_emb) - mean(neg_emb)||` | ↑ = 분리 우수 |
| Linear AUROC | `LogisticRegression().fit(emb).predict_proba` | 임베딩 선형 분리성 |
| Margin sat% | `(d_an - d_ap > margin).mean()` | Phase 1 성공 기준 |

**v5-5 문제 지점**: GO:0006941 margin_sat=5.9% → v6에서 ESM-2로 개선 기대

---

## 8. 출력 저장 구조

```
hMuscle/results_isoform/GO_{term}/
└── v6_integrated_{YYYYMMDD}/
    ├── v6_integrated_training.log          ← 전체 학습 로그
    ├── v6_integrated_phase0_initial_untrained_embeddings.npy
    ├── v6_integrated_phase0_initial_untrained_scores.txt
    ├── v6_integrated_phase1_triplet_only_embeddings.npy
    ├── v6_integrated_phase1_5_linear_probing_embeddings.npy
    ├── v6_integrated_phase2_joint_focal_embeddings.npy
    ├── v6_integrated_{GO}_Final_LabelProp_scores.txt  ← 최종 예측
    ├── v6_integrated_{GO}_Final_LabelProp_labels.npy
    ├── v6_integrated_{GO}_BaseModel_weights.h5
    └── plots/
        ├── v6_integrated_{GO}_umap.png
        └── v6_integrated_{GO}_score_dist.png
```

---

## 9. v5-5 대비 변경 요약

| 항목 | v5-5 | v6 | 변경 이유 |
|------|------|-----|---------|
| 서열 입력 | int k-mer (가변 길이) | ESM-2 640-dim (고정) | 이소폼 수준 서열 차이 포착력 |
| 서열 인코더 | Embedding(8001,32)+Conv1D+PyramidPooling | Dense(256→128→64) MLP | 사전학습 특징 직접 활용 |
| Seq feature dim | 16-dim | 64-dim | 10× 압축 (기존 40×) |
| feature_model 출력 | 32-dim (16+16) | 80-dim (64+16) | ESM-2 정보 공간 확보 |
| 최종 embedding | 16-dim | 32-dim | 분류 경계면 표현력 |
| 배치 방식 | make_batch (길이 그룹화) | 순차 배치 (고정 크기) | 가변 길이 처리 불필요 |
| mask gating | 없음 | esm2_feat × mask | 서열 미존재 이소폼 처리 |
| upsample seed | 없음 | np.random.seed(42) | 재현성 보장 [Fix-DA] |

---

## 10. 현재 실험 진행 현황

| GO term | 이름 | v5-5 AUPRC | GPU | 목표 |
|---------|------|-----------|-----|------|
| GO:0006941 | striated muscle | 0.095 | GPU0 | margin_sat 5.9%→30%+ |
| GO:0007204 | Ca2+ signaling | 0.508 | GPU0 | 회귀 없음 확인 |
| GO:0030017 | sarcomere | 0.390 | GPU0 | 건강한 분포 유지 |
| GO:0003774 | motor activity | 0.205 | GPU1 | Phase transition 패턴 유지 |
| GO:0006096 | glycolysis | 0.670 | GPU1 | 최고성능 회귀 없음 확인 |

**평가 기준** [R9.1][R9.4]:  
- Primary: AUPRC (sparse class)  
- Secondary: AUROC  
- 개선 주장 시 bootstrap CI (n=1000) 필수
