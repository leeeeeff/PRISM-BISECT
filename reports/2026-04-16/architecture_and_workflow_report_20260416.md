# DIFFUSE v6f 모델 아키텍처 및 워크플로우 종합 레포트

**작성일**: 2026-04-16  
**대상 파일**: `hMuscle/model/v6f_integrated_full_model.py`  
**모델 계보**: v4-3 → v5-x → v6 → v6h → v6d → v6e → v6f (current)

---

## 1. 문제 정의

### 1.1 과제
인간 근육(hMuscle) 장거리 단일세포 시퀀싱 데이터에서 식별된 **novel isoform들의 GO term 기능 예측**.
- 대상: BambuTx 형태의 새로운 전사체 (기존 DB에 없음)
- 방식: Few-shot / meta-learning (support set에서 학습 → query set 예측)
- 평가: GO:0006096, GO:0006941, GO:0007204, GO:0003774, GO:0030017 (5개 GO term)
- 주요 지표: AUPRC (Sparse label 조건 [R9.1])

### 1.2 두 가지 핵심 난제

| 난제 | 설명 | 생물학적 원인 |
|------|------|--------------|
| **Data Sparsity** | 5-15%의 sparse positive label, class imbalance 심각 | GO term 기능 annotation 자체가 불완전 |
| **Gene-level Reference Dominance** | 모델이 isoform 고유 특성 대신 gene 수준 feature에 의존 | 동일 gene의 isoform들이 서열 유사성 공유 → shortcut learning |

---

## 2. 데이터 파이프라인

### 2.1 입력 데이터

```
훈련 데이터:
  Human:      31,668개 단백질 (ESM-2 640차원 임베딩)
  SwissProt:  82,703개 단백질 (보강 데이터, DomainDelta=0으로 채움)

테스트 데이터:
  36,748개 BambuTx isoform (long-read 시퀀싱 결과)
```

| 입력 종류 | 차원 | 출처 | 역할 |
|----------|------|------|------|
| ESM-2 임베딩 | 640 | ESM-2 150M (frozen pre-trained) | 진화적 단백질 맥락 |
| ESM-2 mask | 1 | ESM-2 신뢰도 | ESM-2 coverage 가중치 |
| 아미노산 서열 | 1,500 | Long-read 시퀀싱 (last 1500 aa) | 위치 특이적 모티프 |
| Pfam domain | 가변 | Pfam DB annotation | 기능 도메인 조합 |
| DomainDelta | 251 | Pfam domain gain/loss (sign 변환) | 이소폼 특이적 도메인 변화 |

### 2.2 레이블 생성 (generate_label)

훈련/테스트 split 방식:
- `generate_label()`: 유전자 ID 기반으로 support/query 분리
- 규칙: isoform-level stratification (gene-level split 금지 [data-pipeline.md])
- positive_Gene: `human_annotations.txt` + `swissprot_annotations.txt`에서 해당 GO term을 가진 유전자 목록

### 2.3 업샘플링 (upsample)

```python
N_BATCHES_P1 = clip(n_pos × 6.0 / 256, min=10, max=80)
```

- positive class를 gene 단위로 균등 oversampling
- **Coverage**: 한 epoch에서 positive sample이 몇 번 노출되는지
  - `TARGET_COVERAGE=6.0`: positive 1개당 평균 6회 노출
  - min=10, max=80: GO term별 coverage 편차 제한
- SwissProt 데이터 포함: 풍부한 annotation 데이터로 prior knowledge 제공

---

## 3. 모델 아키텍처

### 3.1 전체 구조

```
                         입력층
    ┌──────────┬──────────┬──────────┬──────────┬──────────┐
    │ ESM-2    │ ESM-2    │ 서열     │ Pfam     │ Domain   │
    │ (640)    │ mask (1) │ (1500)   │ domain   │ Delta    │
    │          │          │          │ (가변)   │ (251)    │
    └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
         │          │          │          │          │
    ┌────▼─────┐    │    ┌────▼─────┐ ┌──▼──────┐ ┌─▼───────┐
    │ESM-2     │    │    │CNN       │ │Domain   │ │Domain   │
    │Branch    │    │    │Branch    │ │Branch   │ │Delta    │
    │640→64    │    │    │1500→32   │ │LSTM 16  │ │Branch   │
    │(3 Dense) │×   │    │(Conv1D×2)│ │         │ │251→16   │
    └────┬─────┘ mask    └────┬─────┘ └──┬──────┘ └─┬───────┘
         │gating              │          │           │
    esm2_gated[64]     cnn_feat[32] domain_feat[16] dd_feat[16]
         │                   │          │           │
         └─────────┬──────────┴──────────┴───────────┘
                   │  concat[128]
              ┌────▼────┐
              │Dense(64)│  ← feature_model 출력
              │relu     │
              └────┬────┘
                   │Dropout(0.2)
              ┌────▼────┐
              │L2_norm  │  ← embedding_layer (64차원 단위 구)
              └────┬────┘
              ┌────▼────┐
              │Dense(1) │
              │sigmoid  │  ← prediction_layer
              └─────────┘
```

### 3.2 각 Branch 상세

#### ESM-2 Branch (640 → 64)
```python
Dense(640→256, relu, L2_reg=1e-5) → Dropout(0.2)
→ Dense(256→128, relu, L2_reg=1e-5)
→ Dense(128→64, relu, L2_reg=1e-5)  → esm2_feat[64]
→ Lambda(esm2_feat × mask)           → esm2_gated[64]
```

**도입 목적**: ESM-2는 UniRef50으로 학습된 150M parameter 언어모델로 단백질의 진화적 맥락, 구조적 특성, 기능적 보존성을 640차원으로 인코딩. 이를 task-specific 64차원으로 투영.

**mask gating**: ESM-2가 존재하지 않는 isoform (novel sequence, mask=0)에서 ESM-2 신호를 제로로 만들어 CNN branch에 의존하게 함.

**Phase 역할**: Phase 1에서 triplet loss로 학습 → Phase 2에서 frozen (Phase 1이 만든 표현 보존)

**한계**:
- ESM-2는 mean-pooling 기반 → 위치 특이적 서열 모티프 손실
- Novel isoform에서 ESM-2 coverage가 낮으면 이 branch가 무력화됨
- 640차원 → 64차원 압축 과정에서 fine-grained 정보 손실 가능

#### CNN Branch (1,500 → 32)
```python
Embedding(8001, 32) → Conv1D(64, k=7, same) → Conv1D(32, k=5, same)
→ GlobalMaxPooling1D() → Dense(32, relu)  → cnn_feat[32]
```

**도입 목적**: ESM-2 mean-pool이 포착하지 못하는 **위치 특이적 서열 모티프** 학습. 특히 splice site, exon-specific motif 등이 target.

**GlobalMaxPooling**: 가변 길이 서열을 고정 차원으로 변환. 시퀀스 어디에 있든 가장 강한 신호를 추출 (위치 불변성).

**Phase 역할**: Phase 1에서 frozen → Phase 2에서만 학습 (focal loss 기반). ESM-2가 학습된 후 로컬 모티프 추가 학습.

**한계**:
- kernel size 7, 5: 중간 스케일 모티프 특화, 짧은 binding site나 긴 도메인은 놓칠 수 있음
- Phase 2에서 CNN이 Type-B GO term에서 **이질적 positive들의 sub-cluster별 서열 shortcut**을 학습 → embedding space 파괴 (핵심 문제)

#### Domain Branch (가변 → 16)
```python
Embedding(domain_emb_dim, 32, mask_zero=True) → LSTM(16) → domain_feat[16]
```

**도입 목적**: Pfam domain annotation을 LSTM으로 처리 → 도메인 **조합**과 **순서**를 학습. 단백질 기능은 개별 도메인보다 도메인 조합에 의해 결정되는 경우가 많음.

**Phase 역할**: Phase 1, Phase 2 모두 학습 (전 구간).

**한계**:
- SwissProt 데이터에서 domain annotation 품질이 불균등
- Novel isoform의 domain boundary 예측이 불완전할 수 있음 (Pfam은 이미 알려진 단백질 기반)

#### DomainDelta Branch (251 → 16) [v6d 신규]
```python
sign(domain_matrix[i] - domain_matrix[canonical_gene_i])
→ Dense(251→64, relu, L2_reg) → Dropout(0.2) → Dense(64→16, relu) → dd_feat[16]
```

**도입 목적**: Gene-level reference dominance 극복 [R2.1]. 유전자 내 canonical isoform (Pfam domain count 최대)과의 **도메인 gain/loss**를 sign 변환으로 인코딩.

- +1: canonical보다 이 isoform에 추가 도메인 존재
- 0: canonical과 동일
- -1: canonical보다 이 isoform에서 도메인 소실

**생물학적 의미**: 같은 유전자의 isoform들이 특정 도메인을 포함하거나 제외함으로써 기능이 달라지는 경우를 직접 인코딩.

**Phase 역할**: Phase 1, Phase 2 모두 학습.

**한계**:
- SwissProt 데이터는 isoform 정보 없음 → all zeros (정보 없음)
- sign 변환은 gain/loss 존재 여부만 포착, 개수 차이는 무시
- canonical 선택 기준(domain count 최대)이 생물학적으로 최선이 아닐 수 있음

#### Fusion Head
```python
concat[esm2_gated(64) + cnn_feat(32) + domain_feat(16) + dd_feat(16)] = 128차원
→ Dense(64, relu, L2_reg) → Dropout(0.2)
→ L2_normalize → embedding_layer[64]  # 단위 구(unit sphere) 위의 점
→ Dense(1, sigmoid) → prediction_layer
```

**도입 목적**: 4개 modality의 complementary 정보를 단일 64차원 embedding으로 통합. L2 정규화로 cosine similarity 기반의 triplet loss 적용 가능.

---

## 4. 학습 워크플로우 (4-Phase)

### Phase 0: Untrained Baseline 측정

**목적**: 학습 전 ESM-2 사전 표현의 품질 측정 및 GO term 분류.

**핵심 지표 — sep_ratio**:
```
sep_ratio = inter_dist / intra_dist
           = (positive-negative 평균 거리) / (positive 내부 평균 거리)
```

**GO term 자동 분류 [v6e-1]**:
```
sep_ratio ≥ 1.15 → Type-A (응집형)
  → positive들이 ESM-2 공간에서 이미 클러스터 형성
  → 진화적으로 관련된 단백질들이 공유하는 기능
  예: GO:0006096 (glycolysis, TIM barrel fold)

sep_ratio < 1.15 → Type-B (이질형)
  → positive들이 ESM-2 공간에서 분산됨
  → 수렴 진화(convergent evolution)로 무관한 단백질들이 동일 GO term 공유
  예: GO:0006941 (muscle contraction: myosin+actin+titin이 동일 label)
```

### Phase 1: Triplet Learning (15 epochs)

**학습 대상**: ESM-2 + Domain + DomainDelta branches (CNN frozen)

**알고리즘**:
```
L(a, p, n) = max(d(a,p) - d(a,n) + margin, 0)
margin = 0.3
distance = squared L2 (cosine proxy on unit sphere)
```

**Semi-hard negative mining** (warmup 2 epoch 이후):
```python
# Semi-hard: anchor-positive보다 가깝지만 margin 내에 있는 negative
semi_mask = (d_an > d_ap) & (d_an < d_ap + margin)
```

**목적**:
1. ESM-2 임베딩 공간에서 같은 GO term positive들을 응집
2. 다른 GO term sample들을 멀리 배치
3. CNN은 frozen → Phase 1에서 ESM-2 global context 순수하게 학습

**Early stop 조건**: active_rate < 2% 연속 4 epoch (triplet이 거의 활성화되지 않으면 이미 수렴)

**Biology**: 진화적으로 관련된 단백질들은 ESM-2 공간에서 이미 가깝게 있고, triplet은 이 신호를 강화. 이질적 단백질들(Type B)은 triplet이 sep_ratio를 1.0 → 1.2-1.4로 향상시키지만 완전한 응집은 불가.

**한계**:
- Cross-gene negative만 허용 [R2.1]: intra-gene negative 사용 시 gene-level bias 강화
- 256 batch × 50 batches = 12,800 triplet/epoch → positive 수가 적은 GO term에서 coverage 부족
- sep_ratio가 Phase 1에서 향상되어도 Phase 1.5에서 급락하는 현상 관찰 (미해결)

### Phase 1.5: Linear Probing (2 epochs)

**학습 대상**: classification head Dense(1)만 (feature_model 전체 frozen)

**목적**:
- Phase 1이 만든 임베딩 위에 분류 head 초기화
- Phase 2 진입 전 예측 점수를 의미 있는 범위로 조정
- Focal loss로 class imbalance 처리 시작

**설정**:
```python
focal_loss(gamma=2.0, alpha=0.25)  # hard negative에 집중
batch_size=512, epochs=2
```

**한계 (핵심 미해결 문제)**:
- 이론상 feature_model이 frozen이면 embedding이 변해선 안 됨
- 그러나 empirical observation: Phase 1 이후 sep_ratio가 Phase 1.5에서 급락
  - GO:0006941: 1.25 → 0.86 (-0.39)
  - GO:0003774: 1.58 → 1.01 (-0.57)
  - GO:0006096만 정상: 1.87 → 1.88
- 원인 불명확: 동시 실행(concurrent runs)에 의한 파일 덮어쓰기 race condition 가능성

### Phase 2: CNN Fine-tuning (15-25 epochs)

**학습 대상**: CNN + Domain + DomainDelta (ESM-2 frozen)

**목적**:
- Phase 1에서 frozen이었던 CNN으로 **위치 특이적 서열 모티프** 학습
- Focal loss로 sparse positive 조건에서 수렴

**Focal Loss 설정**:
```
FL(p_t) = -α_t × (1-p_t)^γ × log(p_t)
Phase 2: gamma=2.0, alpha=0.10
```
alpha=0.10: Phase 2에서 precision-recall 균형 (0.25보다 낮춰 recall 희생)

**GO term Type별 적응적 hyperparameter [v6e-1, v6f-1,2]**:

| 설정 | Type-A | Type-B |
|------|--------|--------|
| LR | 0.0003 | 0.0002 |
| clipnorm | 없음 | 0.5 |
| patience | 7 | 10 |
| max_epochs | 15 | 25 |

**Type-B 보수적 설정 근거**:
이질적 positive set에서 CNN이 각 sub-cluster별 서열 shortcut을 빠르게 학습하면 embedding space가 파괴됨 (sep_ratio Phase 0 이하로 역전). LR 낮춤 + gradient clipping으로 Phase 1 표현 보존 시도.

**Early stop**: AUPRC val 기준, no_improve_count ≥ patience이면 best weight 복원.

**한계**:
- Type-B에서 lr=0.0002가 충분한지 미검증 (v6e lr=0.0001은 GO:0006941 수렴 불가)
- Phase 1.5가 이미 embedding을 파괴한 상태에서 Phase 2가 시작됨 → 초기 AUPRC가 낮음
- CNN이 위치 특이적 모티프를 학습하지만 Type-B에서 GO term과 무관한 서열 패턴에 과적합 위험

### Phase 3: Label Propagation (제거됨, v6e부터)

**원래 목적**: 발현 공동변이(co-variation)를 PPI proxy로 사용하여 예측 점수 전파

**제거 근거**: 5개 GO term 중 4개에서 alpha=0.0 (LP 없음)이 최고 성능. GO:0006096에서 LP 적용 시 AUPRC -8.9%. 근육 co-expression network와 GO term annotation 간 상관이 충분하지 않음.

---

## 5. 평가 지표 및 기준

| 지표 | 우선순위 | 근거 |
|------|----------|------|
| AUPRC | Primary [R9.1] | Sparse class (positive < 50)에서 AUROC는 bias |
| AUROC | Secondary | 전체적인 판별력 |
| Macro-AUPRC | Overall [R9.5] | 5개 GO term 평균 (micro는 GO term 크기 bias) |
| Silhouette (cosine) | Diagnostic | Embedding 품질 |
| sep_ratio | Diagnostic | Phase 간 embedding 구조 변화 추적 |
| Linear AUROC | Diagnostic | Embedding의 linear separability |

**신뢰도 요구사항**: 개선 주장 시 bootstrap CI n=1000 필수 [R9.4]

---

## 6. 현재 버전 성능 (v6f 기준 reference: v6d)

| GO Term | 생물학적 기능 | Type | v6d AUPRC | v6e best | 현 상태 |
|---------|-------------|------|-----------|----------|---------|
| GO:0006096 | Glycolysis (TIM barrel) | A | 0.793 | 0.813 | +2.5% ✓ |
| GO:0007204 | Ca2+ signaling | B | 0.265 | 0.388 | +47% ✓✓ |
| GO:0030017 | Sarcomere organization | B | 0.276 | 0.332 | ~flat |
| GO:0006941 | Striated muscle contraction | B | 0.284 | 0.226 | -30% ✗ |
| GO:0003774 | Motor activity | A(경계) | 0.671 | 0.558 | -34% ✗ |
| **Macro-AUPRC** | | | **0.458** | **0.463** | +1% (best) |

---

## 7. 현재 핵심 문제점

### 문제 1: Phase 1.5 Embedding 파괴 (미해결, 근본 원인 불명)

**현상**: Phase 1 triplet이 sep_ratio를 향상시킨 직후, Phase 1.5 linear probing 2 epoch에서 급락.

```
GO:0006941: Ph1=1.25 → Ph1.5=0.86  (Δ=-0.39)
GO:0003774: Ph1=1.58 → Ph1.5=1.01  (Δ=-0.57)
GO:0006096: Ph1=1.87 → Ph1.5=1.88  (Δ=+0.01) ← 정상
```

**영향**: Phase 2가 이미 나빠진 embedding에서 시작 → 초기 AUPRC가 낮음 → 수렴 어려움

**가설**:
1. Concurrent runs가 같은 파일에 embedding을 덮어씀 (race condition)
2. feature_model frozen 선언에도 불구하고 optimizer state가 일부 weight 업데이트

**미해결 이유**: Phase 1.5 제거 실험(v6g) 미진행

---

### 문제 2: Type-B GO term 수렴 불안정 (GO:0006941)

**현상**:
- v6e: lr=0.0001, Phase 2 epoch1 AUPRC=0.150, 15 epoch 내 0.226 (v6d 0.284 미달)
- 원인: Phase 1.5 이후 시작점이 낮음 + lr이 충분하지 않음

**v6f 대응**: lr=0.0002, patience=10, max_epochs=25 → 검증 필요

**근본 원인**: Type-B의 이질적 positive set (myosin+actin+titin이 동일 GO term).
ESM-2 공간에서 서로 다른 cluster를 형성하는 단백질들을 단일 함수로 예측하는 것의 본질적 어려움.

---

### 문제 3: GO:0003774 High Variance (Type-A 경계)

**현상**:
- v6e 6-run AUPRC: 0.260 ~ 0.558 (range 0.30)
- v6d: 0.671 (단일 run이지만 현저히 높음)
- sep_ratio: 1.16 ~ 1.57 (run마다 크게 다름)

**원인**:
- sep_ratio 1.16 ≈ threshold(1.15) → Type 경계에 위치
- Phase 1.5에서 sep drop이 심한 run에서 Phase 2 수렴 불량
- v6e의 coverage 감소(6→4)가 Phase 1.5 학습 분포에 영향 가능성

**v6f 대응**: TARGET_COVERAGE 6 원복

---

### 문제 4: SwissProt DomainDelta 정보 부재

**현상**: SwissProt 훈련 데이터의 DomainDelta가 all zeros (82,703개 샘플).
SwissProt에는 isoform variant 정보가 없어 canonical 대비 delta 계산 불가.

**영향**: 훈련 데이터의 72%(82703/114371)가 DomainDelta=0 → DomainDelta branch가 이 데이터에서 정보를 제공하지 못함.

**현 대응**: 없음 (SwissProt에서 dd_feat=0)

---

### 문제 5: Gene-level Bias 완전 미해결

**현상**: Gene-level reference dominance가 여전히 남아 있을 가능성 (정량 측정 미진행).

**측정 방법** [mathematical-validation.md]:
```python
bias_score = 1 - (H(y|isoform_id) / H(y|gene_id))
# < 0.3 → gene-level 편향 심각
```

**v6f 대응**: 없음 (DomainDelta가 일부 완화하지만 정량 확인 필요)

---

### 문제 6: Triplet Active Ratio 미모니터링

**현황**: Phase 1 종료 시 active_rate을 출력하지만 Phase 2에서 triplet을 제거하여 측정 불가.

**규칙** [loss-functions.md]: active_ratio < 5% → hard negative mining 필요 [R3.2]

**v6f 대응**: Phase 2에 triplet 없음 → 해당 규칙 적용 불가

---

## 8. 설계 원칙 및 제약

### 8.1 Anti-Gene-Bias 규칙 [architecture.md]

- isoform_emb를 gene_emb보다 먼저 계산
- gene context는 attention/gating으로만 처리
- 직접 concatenation 금지 [R2.1]

→ **현재 구현**: DomainDelta가 isoform-specific 신호 역할. ESM-2 gating이 gene-level vs novel isoform 구분.

### 8.2 보호 파일 [data-pipeline.md]

```
my_isoform_list_fixed.npy  → 읽기 전용
my_gene_list_fixed.npy     → 읽기 전용
my_sequence_matrix_fixed.npy → 읽기 전용
```

### 8.3 버전 관리 [architecture.md]

```
Production: integrated_full_model.py
Latest:     v6f_integrated_full_model.py  ← 현재
Next:       v6g_ (Phase 1.5 제거 실험)
```

---

## 9. 알려진 설계 결함 및 미래 탐색 방향

| 결함 | 심각도 | 탐색 방향 |
|------|--------|----------|
| Phase 1.5 embedding 파괴 | 높음 | v6g: Phase 1.5 제거, Phase 1→2 직접 전환 |
| Type-B 수렴 불안정 | 높음 | GO term별 Phase 2 epoch 동적 조정 |
| SwissProt DomainDelta=0 | 중간 | SwissProt에서 DomainDelta branch 비활성화 |
| Gene-level bias 미측정 | 중간 | bias_score 정량화 후 IRM/Adversarial 검토 |
| CNN kernel 단일 스케일 | 낮음 | Multi-scale (k=3,7,11) — GO-term specific 효과 |
| Splicing delta 미활용 | 낮음 | v6f 안정화 후 v6g에 통합 |

---

## 10. 버전 이력 요약

| 버전 | 핵심 변경 | 결과 |
|------|----------|------|
| v4-3 | PFN backbone, multimodal (ESM-2+Domain) | baseline |
| v6h | Modality-role separation (Phase 1=ESM-2, Phase 2=CNN) | +10% |
| v6d | DomainDelta branch (Pfam gain/loss, 251차원) | +4-7% |
| v6e | Phase 0 Type 분류, adaptive lr, LP 제거 | GO:0007204 +47%, GO:0003774 -34% |
| **v6f** | lr 0.0001→0.0002, patience 6→10, coverage 4→6 복구 | **검증 중** |

---

*레포트 작성: 2026-04-16*  
*기반 데이터: v6e 6-run 결과 분석 + v6f 코드 검토*
