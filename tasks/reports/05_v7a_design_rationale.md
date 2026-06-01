# 레포트 05: v7a 설계 근거 (SupCon + Prototype Loss)
**작성일:** 2026-04-29  
**목표:** Phase 1 손실 함수 개선으로 희소 GO term에서 AUPRC +0.02 이상 달성

---

## 1. 왜 Phase 1 손실을 변경하는가

### 현재 Triplet의 한계 (진단)

**[R3.1] Triplet active ratio 지표:**
- GO:0006941 Phase 1: active_rate < 2% for 4 consecutive epochs → Early Stop
- 의미: 대부분의 triplet (a,p,n)이 이미 margin을 만족 → 더 이상 학습이 없음
- 결과: Phase 1이 충분히 훈련되지 않은 상태로 Phase 2 진입

**Triplet의 구조적 한계:**
- 한 번에 (anchor, positive, negative) 3개 샘플만 학습
- 같은 배치 내 다른 positive 쌍들의 정보를 활용하지 못함
- Sparse GO term에서 positive 수가 적어 triplet 다양성 부족

---

## 2. Supervised Contrastive Loss 선택 이유 [R3.3]

### 수식
```
L_i = log Σ_{a≠i} exp(z_i·z_a/τ) - (1/|P(i)|) Σ_{p∈P(i)} z_i·z_p/τ

여기서:
  z_i: L2-정규화 임베딩
  P(i): 배치 내 positive 샘플 집합 (자신 제외)
  τ=0.1: temperature (sharpness 조절)
```

### Triplet 대비 장점
| 항목 | Triplet | SupCon |
|------|---------|--------|
| 한 배치에서 학습하는 쌍 수 | N (triplet 수) | N_pos × (N_pos-1) / 2 |
| Positive 활용 | 하나씩 | 배치 내 전체 positive |
| Negative 활용 | 하나씩 | 배치 내 전체 negative |
| Warmup 필요성 | 필요 (random → semi-hard) | 불필요 |
| Hyperparameter | margin, batch_size | τ, batch_size |

### 희소 GO term에서의 이점
- Sparse positive (GO:0006941: 0.69%) → upsampling 후도 positive 절대수 적음
- Triplet: 각 배치에서 소수의 triplet만 생성 → 정보 효율 낮음
- SupCon: 같은 배치의 모든 positive 쌍을 동시에 밀어붙임 → 정보 효율 높음

### Temperature τ=0.1의 의미
- τ 낮음 → 유사도 차이에 더 민감 → harder한 학습
- τ=0.1은 SupCon 원 논문 (Khosla et al. NeurIPS 2020)에서 검증된 기본값
- τ가 너무 낮으면 수치 불안정 → 0.05 이하 주의

---

## 3. Prototype Contrastive Loss 설계 (v7a-Proto)

### 동기: Type-B GO term의 이질적 positive

GO:0006941, GO:0007204, GO:0030017 같은 Type-B GO term:
- Positive가 생물학적으로 이질적 서브그룹으로 구성
- 예: GO:0006941 = {구조 단백질} ∪ {칼슘 신호 단백질} ∪ {에너지 대사 단백질}
- 단일 centroid 기반 학습은 서브그룹 간 거리 최소화에 실패

### 수식 (CLEAN, Gong et al. Science 2023 기반)

```
L_proto = mean over positives:
  log Σ_j exp(sim(x_i, c_j)/τ) - sim(x_i, c_k(i))/τ

c_k(i): x_i에 가장 가까운 prototype
EMA update: c_k ← normalize(α·c_k + (1-α)·mean(assigned_positives_k))
```

### Gap Statistic으로 k 자동 선택 (Tibshirani 2001)

```
Gap(k) = E[log W_k^ref] - log W_k
k_opt = smallest k s.t. Gap(k) ≥ Gap(k+1) - std(k+1)
```

- Type-A (sep_ratio ≥ 1.15): k=1 예상 (단일 응집 그룹)
- Type-B (sep_ratio < 1.15): k=2-3 예상 (이질적 서브그룹)

---

## 4. Ablation 순서 및 설계 근거

### Exp 2a: v7a-SupCon (현재 실행 중)
**목적:** "Triplet → SupCon" 효과만 분리해서 측정
- Prototype 없이 순수 SupCon
- 결과 해석:
  - PASS (Δ>+0.02): loss 형태 자체가 개선
  - FAIL (Δ<0): Triplet이 오히려 이 데이터에 더 적합
- 의존성: 없음 (독립 실험)

### Exp 2b: v7a-Proto-k1 (SupCon PASS 후)
**목적:** "단일 prototype" 효과 측정
- k=1 강제: positive들의 평균 centroid를 prototype으로 학습
- Type-A GO term에서는 SupCon과 유사 성능 예상
- Type-B GO term에서는 k>1이 필요한지 확인
- 결과 해석:
  - Type-B에서 k=1 < k>1: 다중 prototype 필요성 확인 → Exp 2c
  - Type-B에서 k=1 >= k>1: 다중 prototype 불필요 → Exp 2c 간략화

### Exp 2c: v7a-Proto-kN (Exp 2b 후)
**목적:** Gap Statistic k 선택의 효과 측정
- k∈[1,5] 자동 선택
- 학습 안정성: L_diversity (epoch 15) < L_diversity (epoch 1) 확인

---

## 5. 배치 구성 설계

### SupCon 배치 (phase1_supcon_epoch_hybrid)
```python
n_pos_per_batch = max(8, min(N_pos, batch_size // 4))
n_neg_per_batch = batch_size - n_pos_per_batch
```

**min_pos_in_batch=8 이유:**
- SupCon은 |P(i)| ≥ 1 필요 (valid anchor 조건)
- 8개 positive → 7개 valid anchor → 28쌍의 positive pair
- GO:0006941 (0.69% positive): upsampling 후 N_pos 충분, 배치에서 보장 필요

### 기존 Triplet 배치 대비
- Triplet: embedding refresh every 10 batches (semi-hard mining)
- SupCon: embedding refresh 불필요 (배치 내에서 직접 계산)
- SupCon이 더 단순하고 계산 효율적

---

## 6. 하이퍼파라미터 선택

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| τ (temperature) | 0.1 | Khosla et al. NeurIPS 2020 기본값 |
| EMA decay α | 0.9 | CLEAN 논문 기본값 (prototype 안정성) |
| λ_diversity | 0.1 | diversity loss 비중 (prototype 분산 촉진) |
| k_max | 5 | Type-B GO term의 예상 서브그룹 수 상한 |
| n_refs (Gap Stat) | 10 | 계산 시간과 정확도 균형 |

---

## 7. 실패 시나리오 및 대응

| 시나리오 | 대응 |
|----------|------|
| SupCon AUPRC < Triplet | Triplet 유지, hard negative mining 강화 |
| Proto-k1 >> Proto-kN | k=1 고정, Gap Statistic 비활성화 |
| Proto 훈련 불안정 (loss 발산) | τ=0.2로 증가, EMA decay=0.95로 증가 |
| k_opt가 항상 k_max | k_max=3으로 축소 또는 Gap Statistic 버리고 k=2 고정 |

---

## 8. 구현 파일

| 파일 | 역할 |
|------|------|
| `prototype_contrastive.py` | SupCon + ProtoConLoss 전체 모듈 |
| `v7a_integrated_full_model.py` | v6g + Phase 1 SupCon 교체 |
| `run_GPU_v7a.py` | 2-GPU 실행 스크립트 |

### prototype_contrastive.py 함수 목록
- `supervised_contrastive_loss(embeddings, labels, temperature)` — SupCon loss 계산
- `phase1_supcon_epoch_hybrid(...)` — Phase 1 SupCon epoch
- `determine_k_gap_statistic(embeddings, k_max, n_refs)` — 최적 k 선택
- `PrototypeContrastiveLoss` — k prototype EMA 관리 클래스
  - `initialize_from_embeddings(pos_embeddings)` — k-means 초기화
  - `compute_loss(pos_embeddings)` — L_proto + L_diversity
  - `update_prototypes_ema(pos_embeddings_np)` — EMA 업데이트
  - `prototype_stats(embeddings_np, y_np)` — 진단 출력
- `phase1_proto_epoch_hybrid(...)` — Phase 1 Prototype epoch
