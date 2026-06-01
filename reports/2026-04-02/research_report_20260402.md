# DIFFUSE 이소폼 기능 예측 모델 — 연구 분석 레포트
**날짜**: 2026-04-02  
**대상 모델**: v4-3_integrated_full_model.py  
**목표 venue**: Nature Methods / Nature Machine Intelligence (IF 15+)

---

## 1. 프로젝트 최종 목표

Long-read single-cell sequencing 데이터를 활용하여, **이소폼 수준의 GO term 기능 예측**에서 두 가지 근본 한계를 동시에 극복하는 것.

| 한계 | 극복 전략 |
|---|---|
| **데이터 불균형 / Mode Collapse** | Focal Loss + Triplet Loss + Bio-constrained Augmentation |
| **Gene-level Reference Dominance** | 이소폼 내재적 피처 우선, gene-level shortcut 패널티 |

---

## 2. 코드베이스 구조

```
hMuscle/
├── model/
│   ├── v4-3_integrated_full_model.py   ← 현재 최신 실험 버전 (733줄)
│   ├── integrated_full_model.py        ← 프로덕션 참조본
│   ├── pfn_model.py                    ← PFN Bridge 단독 모듈
│   ├── focal_model.py                  ← Focal Loss 단독 모듈
│   ├── triplet_model.py                ← Triplet Loss 단독 모듈
│   ├── utils_Full.py                   ← 데이터 라벨링/업샘플링
│   ├── crf.py                          ← CRF 추론/파라미터 학습
│   └── run_GPU_Full.py                 ← GPU 런처
├── results_isoform/
│   ├── GO_0006096/  GO_0006412/  GO_0006936/  GO_0022900/
│   └── evaluation.py                   ← AUROC/AUPRC/Gene_Top-k 계산
└── data/
    ├── bambu_data/counts_transcript.txt   ← hMuscle 이소폼 발현량
    ├── bambu_data/CPM_transcript.txt      ← hMuscle 이소폼 CPM
    └── raw_data/data/                     ← 원본 DIFFUSE 학습 데이터
```

### 핵심 데이터 파일 (수정 금지)
- `my_isoform_list_fixed.npy` — test isoform ID 목록
- `my_gene_list_fixed.npy` — test gene ID 목록
- `my_sequence_matrix_fixed.npy` — test sequence 행렬

---

## 3. v4-3 모델 아키텍처

### 3-1. 입력 (현재 — 설계 목표 대비 미통합 있음)

| 채널 | 현황 |
|---|---|
| Sequence (amino acid trigram) | ✅ 통합 |
| Protein Domain (CDD) | ✅ 통합 |
| ESM protein embedding | ❌ 미통합 (아키텍처 규칙 MANDATORY) |
| Cell localization | ❌ 미통합 |
| PPI network | ❌ 미통합 |
| Isoform expression profile | ❌ 미통합 (bambu_data 존재) |

### 3-2. 신경망 구조

```
seq_input → Embedding(8001, 32) → Conv1D(64, k=32) → PyramidPooling([1,2,4,8]) → Dense(16)  ← seq_feat
domain_input → Embedding → LSTM(16)                                                           ← domain_feat
                             ↓ concatenate(32)
                       Dense(16) → L2-normalize → embedding_out (32-dim)
                                                 ↓
                                       Dense(1, sigmoid) → prediction
```

### 3-3. 학습 3단계

| Phase | 내용 | Epoch |
|---|---|---|
| **Phase 0** | Untrained baseline 저장 | — |
| **Phase 1** | 임베딩 기반 Triplet (GradientTape). 전체 임베딩 추출 후 hard negative 선택. warmup 2 epoch(random) → hard negative | 15 |
| **Phase 1.5** | Encoder 동결 후 Linear Probing (Focal Loss) | 2 |
| **Phase 2** | Joint fine-tuning: Focal(주, weight=1.0) + Ingroup Triplet(보조, weight=0.1) | 5 |
| **Phase 3** | Focal + PFN + CRF 반복 최적화 | 10 |

### 3-4. Loss 함수

```
# Focal Loss
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
gamma=2.0, alpha=0.25

# Triplet Loss (Phase 1: GradientTape, Phase 2: ingroup)
L(a,p,n) = max(d(a,p) - d(a,n) + margin, 0)
margin=0.1 (L2 normalized 공간)

# Combined (Phase 2)
total_loss = 1.0 * focal_loss + 0.1 * triplet_loss
```

---

## 4. 현재 성능 평가 (EXP-01 포함)

### 4-1. Phase별 AUROC / AUPRC (v4-3, 2026-03-25 실행)

| GO term | pos/total | Phase0 | Phase1 (Triplet) | Phase1.5 (Linear) | Phase2 (Joint) | **Final (CRF후)** |
|---|---|---|---|---|---|---|
| GO_0006096 (Glycolysis) | 76/36748 | 0.516/0.002 | 0.483/0.002 | 0.723/0.134 | **0.760/0.258** | 0.512/0.002 |
| GO_0006412 (Translation) | 701/36748 | 0.500/0.019 | 0.457/0.017 | 0.506/0.019 | **0.636/0.082** | 0.503/0.019 |
| GO_0006936 (Muscle contraction) | 597/36748 | 0.430/0.014 | 0.293/0.011 | 0.438/0.014 | **0.791/0.296** | 0.516/0.017 |
| GO_0022900 (Electron transport) | 291/36748 | 0.405/0.006 | 0.432/0.007 | 0.546/0.009 | **0.643/0.074** | 0.481/0.008 |

### 4-2. EXP-01: no_crf vs CRF 직접 비교

| GO term | Phase2 (no_crf) AUROC | Final (CRF후) AUROC | **CRF 손실** | Phase2 AUPRC | Final AUPRC | **AUPRC 손실** |
|---|---|---|---|---|---|---|
| GO_0006096 | 0.760 | 0.512 | **-0.248** | 0.258 | 0.002 | **-0.256** |
| GO_0006412 | 0.636 | 0.503 | **-0.133** | 0.082 | 0.019 | **-0.063** |
| GO_0006936 | 0.791 | 0.516 | **-0.275** | 0.296 | 0.017 | **-0.279** |
| GO_0022900 | 0.643 | 0.481 | **-0.161** | 0.074 | 0.008 | **-0.066** |

**결론**: CRF는 4개 GO term 전부에서 성능을 파괴하고 있음. Phase 2의 우수한 표현(AUROC 최대 0.791)이 CRF 이후 랜덤 수준(0.48~0.52)으로 역행.

GO_0006936 (hMuscle 핵심 GO term): Phase 2 F1=0.394 → CRF 후 F1=0.019

---

## 5. 원본 DIFFUSE vs v4-3 CRF 동작 차이 분석

### 5-1. 구조적 차이

| | **원본 DIFFUSE** | **우리 v4-3** |
|---|---|---|
| DNN Loss | Binary Cross-Entropy | Focal + Triplet |
| 학습 구조 | DNN ↔ CRF **매 epoch 공동 최적화** (50 epoch) | Phase 1→2→3 **순차적** (CRF는 마지막에만) |
| theta 학습 | 매 epoch L-BFGS-B 갱신 | Phase 3 10 epoch만 |
| sigma | 고정 0.1 | centroid_dist 기반 동적 결정 |
| CRF 투입 점수 | 초기 BCE 점수 (약함, 넓은 분산) | Phase 2 고품질 점수 (AUROC ~0.79) |
| **평가 단위** | **gene-level** (`np.max()` per gene) | **isoform-level** |
| co-exp network | coexp_net_unified.npy (39375×39375, count 행렬) | coexp_net_bridged.npy (68416×68416, correlation) |

### 5-2. 원본 CRF가 성공한 3가지 이유

**① DNN-CRF 공동 훈련 (Co-calibration)**

원본은 50 epoch 동안 DNN과 CRF를 번갈아 최적화합니다. DNN은 순수한 이소폼 레이블이 아닌 CRF가 업데이트한 레이블로 학습하므로, DNN 출력이 CRF 친화적 점수로 수렴합니다. 우리 v4-3은 DNN을 독립적으로 강하게 학습(AUROC 0.79)시킨 후 CRF를 외부에서 적용하므로, calibration이 맞지 않습니다.

**② Gene-level 평가와 MIL 제약의 일치**

원본 평가는 gene-level max pooling AUROC입니다. CRF의 MIL 제약(positive gene의 이소폼 중 최소 1개 반드시 positive)이 이 평가 지표를 직접 최적화합니다. 유전자 내 이소폼 점수 균일화는 gene-level max pooling에서 성능에 무해합니다.

**③ 약한 DNN 점수가 CRF에게 개입 여지를 줌**

BCE 5 epoch만 학습한 약한 DNN 점수는 분산이 넓어 CRF pairwise potential이 균형있게 개입할 수 있습니다. 반면 우리 Phase 2 점수는 이미 고품질이어서, CRF pairwise potential이 unary를 압도합니다.

### 5-3. 우리 CRF 실패의 수식적 설명

CRF `massage_passing` 단계:
```
pairwise[1, i] = Σ_j (coexp[i,j] × q[0,j]) / Σ_j coexp[i,j]
```

`coexp_net_bridged.npy`가 같은 유전자 이소폼 간 높은 weight를 가지면 (gene-level co-expression 구조), 유전자 내 모든 이소폼의 q가 평균으로 수렴합니다. Phase 2에서 특정 이소폼이 높은 점수를 받았어도, 같은 유전자 다른 이소폼들이 CRF를 통해 비슷한 점수를 받게 됩니다.

**근본 원인**: 원본 DIFFUSE CRF는 **gene-level annotation 전파 도구**입니다. 우리 목표인 **이소폼 수준 내 기능 차이 예측**과 설계 목적이 다릅니다.

---

## 6. 문제와 한계 목록

### P0 — 치명적: CRF가 Phase 2 학습 결과를 파괴 (확인됨)
- AUROC 손실: 최대 -0.275 (GO_0006936)
- AUPRC 손실: 최대 -0.279 (GO_0006936)
- CRF 후 일부 GO term에서 R=1.0 (모든 이소폼 positive 예측) trivial solution 수렴

### P1 — Phase 1 Triplet 학습이 성능 저하 유발
- GO_0006096: Phase0 AUROC 0.516 → Phase1 0.483 (-0.033)
- GO_0006936: Phase0 AUROC 0.430 → Phase1 0.293 (-0.137)
- 원인 후보: margin=0.1이 L2 normalized 공간에서 trivial solution; stale embedding; cross-gene negative 미강제

### P2 — ESM / Cell Localization / PPI 미통합
- 아키텍처 규칙 MANDATORY 항목이 코드에 미반영
- 현재 입력: sequence + domain 2채널만

### P3 — Gene-level Bias 정량 측정 부재
- `evaluation.py`에 intra-gene vs inter-gene score variance 지표 없음
- 논문 방어 불가

### P4 — Phase 2 Triplet Negative가 ingroup-only
- `get_triplet_batch_ingroup`: 같은 길이 그룹 내 negative 선택 → intra-gene negative 혼입 가능
- Loss function 규칙 "Negative MUST be cross-gene" 위반 가능

### P5 — Isoform expression profile 미활용
- `bambu_data/CPM_transcript.txt` 존재하나 모델 입력으로 미통합
- Expression data의 기여가 원본 DIFFUSE에서 확인됨에도 이소폼 수준에서는 미사용

---

## 7. Co-expression / Expression 데이터의 이상적 통합 방안

### 7-1. 원본 방식의 한계

원본 DIFFUSE의 gene-level co-expression → CRF pairwise potential 방식은:
- "이 유전자가 어떤 기능군과 묶이는가" 정보 인코딩
- 같은 유전자 이소폼 간 기능 차이는 알 수 없음
- 이소폼 수준 평가에서 within-gene 균일화로 판별력 파괴

### 7-2. 이상적 통합 방안 (우선순위 순)

**방안 A: Expression profile을 세 번째 입력 모달로 통합 (즉시 실행 가능)**

```
seq_feat    → Conv1D + PyramidPooling → Dense(16)
domain_feat → LSTM(16)
expr_feat   → CPM vector (cell-type별) → Dense(16)   ← NEW
                      ↓ concatenate(48)
              Dense(16) → embedding_out → prediction
```

- 데이터: `bambu_data/CPM_transcript.txt` (이미 존재)
- 이소폼별 cell-type 발현 벡터를 내재적 피처로 학습
- Gene-level co-expression과 달리, 이소폼 수준 발현 패턴이 인코딩됨
- 아키텍처 규칙 준수: `isoform_emb` primary, PFN forward pass 변경 없음

**방안 B: Expression-guided Triplet Mining (Phase 1 개선과 결합)**

```
현재: hard negative = embedding 공간에서 가까운 negative
개선: co-expression 유사도 기반 negative 선택

Anchor: 이소폼 A1 (GO:0006936 positive)
Positive: 이소폼 B1 (다른 유전자, 같은 GO, 유사한 근육 발현 패턴)
Hard Negative: 이소폼 C1 (다른 유전자, 다른 GO, 근육에서도 발현)
```

- 발현 패턴이 유사해도 기능이 다른 케이스 → 모델이 발현만 보고 판단하지 못하도록 강제
- Gene-level bias의 직접 공격: "같은 발현 패턴 ≠ 같은 기능"을 학습
- 논문 기여: "expression-guided isoform-level triplet mining"

**방안 C: Isoform co-expression GNN (중장기)**

```
이소폼별 발현 상관관계 그래프 구축 (bambu_data 기반)
→ GNN attention으로 이웃 정보 선택적 집계
→ 기존 embedding에 context 추가
```

- CRF와의 차이: attention weight가 학습 가능 → within-gene 균일화 방지 가능
- 논문 기여: "isoform-aware pairwise regularization" (CRF 대체)

### 7-3. 논문 방어 논리

> "기존 DIFFUSE는 expression을 gene-level pairwise regularization(CRF)으로 사용했으나,
> 이는 이소폼 수준 판별을 방해한다. 우리는 expression을 isoform-intrinsic feature로
> 통합하여 동일한 생물학적 정보를 gene-level bias 없이 활용한다."

---

## 8. 실험 체크리스트

### STAGE 1 — CRF 제거 (즉시)

- [x] **EXP-01**: no_crf 검증 — Phase 2 출력 직접 사용 시 성능 확인 (**완료**, AUROC 최대 0.791 확인)
- [ ] **EXP-02**: CRF 파라미터 grid — sigma ∈ {0.001, 0.01, 0.05} (참고용 ablation)

### STAGE 2 — Triplet 학습 수정

- [ ] **EXP-03**: Cross-gene negative 강제 — `build_emb_triplet_inputs` neg pool을 다른 유전자 소속만으로 제한
- [ ] **EXP-04**: Margin 탐색 — {0.1, 0.3, 0.5} (L2 normalized 공간)
- [ ] **EXP-05**: Phase 1 ablation — `no_triplet` (Phase 1 제거) vs `no_phase1.5` (linear probing 제거)

### STAGE 3 — Expression 통합

- [ ] **EXP-06**: Expression profile 입력 통합 (방안 A) — `bambu_data/CPM_transcript.txt` 기반
- [ ] **EXP-07**: Expression-guided triplet mining (방안 B) — bambu_data 이소폼 발현 유사도 행렬 계산
- [ ] **EXP-08**: ESM 임베딩 통합 (아키텍처 규칙 MANDATORY)
- [ ] **EXP-09**: Cell localization 입력 추가
- [ ] **EXP-10**: PPI 네트워크 입력 추가

### STAGE 4 — Gene-level Bias 정량화

- [ ] **EXP-11**: `evaluation.py`에 intra-gene/inter-gene score variance 지표 추가
- [ ] **EXP-12**: `lambda_triplet` 조정 — {0.5, 1.0, 2.0}
- [ ] **EXP-13**: Train/test gene-level split 검증 (leakage 확인)

### STAGE 5 — Ablation (논문 방어)

| 실험명 | 제거 요소 |
|---|---|
| `no_triplet` | Triplet Loss 전체 |
| `no_focal` | Focal Loss (→ BCE) |
| `no_ppi` | PPI network input |
| `no_esm` | ESM embedding |
| `no_cellloc` | Cell localization |
| `no_expr` | Expression profile |

---

## 9. 다음 즉시 실행 항목

### v5 모델 설계 방향

1. **CRF 제거**: Phase 2 출력을 Final로 사용 (EXP-01에서 AUROC 0.791 이미 확인)
2. **Expression profile 통합**: `bambu_data/CPM_transcript.txt` → 세 번째 입력 모달
3. **Cross-gene negative 강제**: Phase 1 triplet mining 수정
4. **파일명**: `v5_integrated_full_model.py` (기존 파일 `_backup` 유지)

### 성능 목표 (v5)

| GO term | 현재 Phase2 AUROC | 목표 AUROC |
|---|---|---|
| GO_0006096 | 0.760 | 0.800+ |
| GO_0006412 | 0.636 | 0.700+ |
| GO_0006936 | 0.791 | 0.830+ |
| GO_0022900 | 0.643 | 0.700+ |

---

## 10. 버전 히스토리

| 버전 | 날짜 | 주요 변경 |
|---|---|---|
| model.py | 원본 | 원본 DIFFUSE (BCE, gene-level CRF) |
| model_fixed.py | — | hMuscle 데이터 경로/인자 수정 |
| v3_integrated | 2026-03-19 | Triplet + Focal 첫 통합 |
| v4_integrated | 2026-03-25 | 실패 (embedding collapse) |
| **v4-3_integrated** | **2026-03-25** | **임베딩 기반 GradientTape Triplet, 현재 최신** |
| v5_integrated | 예정 | CRF 제거, Expression 통합, Cross-gene negative |

---

*레포트 작성: 2026-04-02*  
*분석 대상 실행 로그: `results_isoform/GO_*/v4-3_integrated_260325/v4-3_integrated_training.log`*  
*EXP-01 실행 코드: 저장된 `phase2_joint_focal_scores.txt` 기반 즉시 계산*
