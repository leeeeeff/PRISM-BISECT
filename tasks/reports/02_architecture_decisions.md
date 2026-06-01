# 레포트 02: 아키텍처 설계 결정 이유
**작성일:** 2026-04-29  
**범위:** v6d → v6g → v7a 아키텍처 누적 변경사항과 각 결정 근거

---

## 1. 현재 아키텍처 전체 구조 (v6g 기준)

```
입력 1: ESM-2 (640d) + mask (1d)
         → Dense(256,relu) → Dense(128,relu) → Dense(64,relu)
         → gate(×mask) → esm2_gated[64]

입력 2: 아미노산 서열 (max 1500aa)
         → Embedding(8001,32) → Conv1D(64,k=7) → Conv1D(32,k=5)
         → GlobalMaxPool → Dense(32,relu) → cnn_feat[32]

입력 3: Pfam 도메인 매트릭스 (N_domains × 1024)
         → Embedding → LSTM(16) → domain_feat[16]

입력 4: DomainDelta (251d, sign{-1,0,+1}) [v6d 신규]
         → Dense(64,relu) → Dense(16,relu) → dd_feat[16]

Fusion: concat[64+32+16+16=128] → Dense(64,relu) → L2_norm → emb[64] → sigmoid
```

---

## 2. 각 모달리티 설계 근거

### ESM-2 (640d)
- **역할:** 전역 진화 맥락 (evolutionary context)
- **Phase 역할:** Phase 1에서 학습, Phase 2에서 frozen
- **근거:** ESM-2는 단백질 언어 모델 — 배열에서 구조/기능 정보를 포착. Phase 2에서 CNN이 ESM-2 공간을 재조직하면 Phase 1 Triplet이 구축한 metric space 파괴 → freeze
- **gate:** missing embedding(mask=0) isoform 처리 — missing → 0벡터, valid → 원래 값

### CNN (SEQ_LEN=1500aa)
- **역할:** 위치 특이적 로컬 모티프 (position-specific local motifs)
- **Phase 역할:** Phase 1에서 frozen, Phase 2에서 학습
- **근거:** ESM-2가 mean-pool 기반이라 위치 정보 손실 → CNN이 보완
- **SEQ_LEN 1500:** p95 커버 (전체 서열 6000aa의 마지막 1500aa 사용)
- **GlobalMaxPool:** 가변 길이 서열 처리, make_batch 불필요

### Domain LSTM (Pfam)
- **역할:** GO term 특이적 도메인 조합 학습
- **Phase 역할:** 전 구간 학습 가능
- **근거:** 도메인 순서가 중요한 경우 LSTM이 순서 정보 포착

### DomainDelta (251d) [v6d 신규]
- **역할:** Isoform-특이적 도메인 gain/loss 인코딩
- **수식:** `delta[i] = sign(domain_matrix[i] - domain_matrix[canonical_gene_i])`
- **canonical 정의:** gene 내 Pfam domain count 최대 isoform
- **sign transform 이유:** 절대값이 아닌 방향성 (이소폼이 canonical 대비 어떤 도메인을 잃었는가)
- **기여:** Gene-level reference dominance 극복 [R2.1] — isoform-intrinsic signal

---

## 3. 훈련 단계 설계 (v6g 기준)

| Phase | 학습 가중치 | 손실함수 | 목적 |
|-------|------------|---------|------|
| **Phase 0** | 없음 (진단) | - | sep_ratio 측정 → GO term Type 분류 |
| **Phase 1** | ESM-2 ✅, CNN ❌, Domain ✅, Delta ✅ | Triplet (→v7a: SupCon) | 임베딩 공간 구성 |
| ~~Phase 1.5~~ | ~~전체 동결~~ | ~~Focal~~ | **v6g에서 제거** |
| **Phase 2** | ESM-2 ❌, CNN ✅, Domain ✅, Delta ✅ | Focal | CNN 로컬 모티프 학습 |
| **Phase 3** | 없음 | - | LP 비활성화 (alpha=0.0 고정) |

### Phase 1.5 제거 근거 [v6g-1]
- **실측 데이터:** v6e 결과에서 Phase 1.5 전후 임베딩 max_diff=0.52-0.63
  - GO:0006941: max_diff=0.52 (frozen 선언에도 실제 변화)
  - GO:0003774: max_diff=0.63
- **sep_ratio 파괴:** Phase 1 구축한 분리도가 Phase 1.5에서 역전
  - GO:0006941: 1.25 → 0.86 (-0.39)
  - GO:0003774: 1.58 → 1.01 (-0.57)
- **성능:** v6f(Phase 1.5 포함) Macro-AUPRC=0.424 → v6g(제거) 0.465 (+0.041)
- **메커니즘 가설:** Linear Probe head의 선형 변환이 임베딩 공간의 effective metric 왜곡

---

## 4. GO Term Type A/B 분류 [v6e-1]

```
Phase 0 sep_ratio 측정
  ≥ 1.15 → Type-A: positive 집합이 진화적으로 응집
           (ESM-2가 이미 분리 가능한 수준)
           Phase 2: lr=0.0003, patience=7, max_epochs=15

  < 1.15 → Type-B: positive 집합이 이질적 (수렴 진화)
           (ESM-2가 분리 불가 → Phase 2 CNN이 embedding 역전 위험)
           Phase 2: lr=0.0002, clipnorm=0.5, patience=10, max_epochs=25
```

| GO Term | 일반적 sep_ratio | Type |
|---------|-----------------|------|
| GO:0006096 | 1.17-1.66 | A |
| GO:0003774 | 1.16-1.57 | A |
| GO:0006941 | 1.04-1.12 | B |
| GO:0007204 | 0.98-1.12 | B |
| GO:0030017 | 1.00-1.07 | B |

**주의:** GO:0006096은 1.145-1.148 run에서 Type-B 오분류 가능성 있음 (threshold=1.15 경계선)

---

## 5. Loss Function 설계

### Focal Loss [R1.1] (Phase 2)
```
FL(p_t) = -α_t(1-p_t)^γ · log(p_t)
γ=2.0, α=0.10 (class imbalance 대응)
```
- α=0.10 이유: positive class 비율이 매우 낮음 (GO:0006941: 0.69%)
  → α를 낮게 설정해 false positive 억제

### Triplet Loss [R3.1] (Phase 1, v6d~v6g)
```
L(a,p,n) = max(d(a,p) - d(a,n) + margin, 0)
margin=0.3, distance=cosine (sq_L2로 구현)
```
- Semi-hard negative mining: d(a,p) < d(a,n) < d(a,p) + margin
- Cross-gene negative 강제 [R2.1]: intra-gene negative 금지
- Warmup 2 epoch: random mode (초기 불안정 방지)
- Early stop: active_rate < 2% for 4 consecutive epochs

### Supervised Contrastive Loss [R3.3] (Phase 1, v7a)
```
L_i = log Σ_{a≠i} exp(z_i·z_a/τ) - (1/|P(i)|) Σ_{p∈P(i)} z_i·z_p/τ
τ=0.1, P(i) = batch 내 positive 샘플 (자신 제외)
```
- Triplet 대비: (a,p,n) 3-way → batch 내 전체 positive 쌍 동시 학습
- 배치 구성: n_pos=max(8, N_pos//4), n_neg=batch-n_pos (positive 비율 보장)

---

## 6. 모달리티별 학습 역할 분리 [v6h 원칙]

**문제 (v6 초기):** Phase 2에서 Triplet+Focal 동시 사용 시 ESM-2 공간 재조직
→ Phase 1.5 calibration 파괴 (GO:0007204 0.508→0.300)

**해결 원칙:**
- ESM-2: 전역 진화 맥락 → Phase 1 학습 완료 후 동결 (불변)
- CNN: 위치 특이적 로컬 → Phase 2에서만 학습
- Domain/Delta: GO term 특이적 → 전 구간 학습

---

## 7. 데이터 파이프라인 설계

### Upsampling (TARGET_COVERAGE=6.0)
- 훈련 데이터의 positive 비율을 TARGET_COVERAGE배로 증폭
- coverage clip: min=10, max=80 (min=20→10 개선, max=50→80 개선) [v6e-4]
- coverage=4.0 → GO:0006096 coverage 50% 감소 유발 → 6.0으로 원복 [v6f-3]

### SwissProt 보강 훈련 [기존 설계]
- Human + SwissProt ESM-2 임베딩을 함께 사용
- SwissProt: 고품질 GO annotation, 희소 GO term positive 보강 역할

### Sequence 처리
- SEQ_LEN=1500: 전체 서열(6000aa)의 마지막 1500aa truncation
- GlobalMaxPool: 가변 길이 대응, 패딩 불필요
- Embedding(8001,32): 20 아미노산 + 특수 토큰

---

## 8. 향후 계획된 변경 (v7a → v7b)

| 단계 | 변경 | 근거 |
|------|------|------|
| v7a-SupCon (진행중) | Phase 1: Triplet → SupCon | 배치 내 전체 positive 쌍 동시 학습 |
| v7a-Proto-k1 | Phase 1: Prototype Loss k=1 | 평균 prototype 학습 |
| v7a-Proto-kN | Phase 1: Prototype Loss k=gap stat | 이질적 positive subclustering |
| v7b | + acorde co-expression LP | 샘플 간 기능 연관성 활용 |
