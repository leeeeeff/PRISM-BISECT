# v6-Hybrid 설계 보고서

**작성일**: 2026-04-10  
**버전**: v6h_integrated (v6h)  
**목적**: v6 Phase 2 regression 진단 및 Modality-Role Separation 설계 근거 문서화

---

## 1. v6 Regression 진단

### 관찰된 문제

| GO term | v5-5 AUPRC | v6 Phase 1.5 | v6 Phase 2 | 변화 |
|---------|-----------|-------------|-----------|------|
| GO:0007204 | 0.508 | (개선) | 0.300 | **-0.208** |
| GO:0030017 | 0.390 | (개선) | 0.211 | **-0.179** |
| GO:0006941 | (낮음) | (낮음) | (낮음) | 변화 없음 |

### 근본 원인 분석

**원인 1: Phase 2 triplet이 Phase 1.5 임베딩 공간 파괴**

```
Phase 1.5: focal loss로 ESM-2 임베딩을 GO term 방향으로 정렬
           → AUPRC 개선 달성

Phase 2: joint loss (focal + triplet)
         → focal 0.003 → 0.000 (빠르게 포화)
         → triplet이 gradient 지배
         → ESM-2 임베딩 공간 재편성
         → GO term 방향 정렬 파괴
         → score drift → AUPRC 하락
```

**원인 2: ESM-2 초기화 편향 (GO:0030017)**

```
문제: sarcomere 단백질 (MYH, ACTN, TTN, TMOD)이
     ESM-2 공간에서 이미 조밀하게 클러스터링
     → Phase 0에서 pred_dist.std()=0.000 (100% collapse)

Phase 1.5: focal loss로 클러스터 분리 성공
Phase 2:   triplet이 cross-gene negative를 강하게 당기면서
           sarcomere 클러스터 내부 구조 파괴
```

**원인 3: Ca²⁺ 결합 위치 정보 손실 (GO:0007204)**

```
ESM-2 mean-pool: 전체 시퀀스 평균 → 위치별 정보 손실
EF-hand, RYR domain: 특정 위치의 모티프 패턴이 핵심
→ mean-pool 후 이 정보가 다른 도메인 신호에 희석
→ Phase 2 triplet이 전체 시퀀스 레벨 재조직화
→ 위치 특이적 신호 추가 손실
```

---

## 2. v6-Hybrid 설계 원칙

### Modality-Role Separation

| Modality | 역할 | 학습 단계 |
|---------|------|---------|
| **ESM-2** | Global context (전체 서열 의미론) | Phase 1만 학습, Phase 2 Frozen |
| **CNN** | Local motifs (위치 특이적 패턴) | Phase 1 Frozen, Phase 2만 학습 |
| **Domain** | Functional domain annotation | 전 단계 학습 |

### 핵심 아이디어

```
Phase 1 (ESM-2 주도):
  - ESM-2 + Domain + Head → triplet으로 전역 구조 학습
  - CNN은 frozen (초기화 랜덤 → 학습 전 기여 없음)
  - 목적: cross-gene negative로 gene-level shortcut 제거

Phase 1.5 (전체 frozen → Head 재조정):
  - encoder freeze, focal로 Head만 GO term 방향 정렬
  - ESM-2 임베딩 공간 보존

Phase 2 (CNN 주도):
  - ESM-2 frozen → Phase 1.5 임베딩 공간 보존
  - CNN + Domain + Head → focal loss로 위치 특이적 패턴 학습
  - triplet 없음 → ESM-2 공간 파괴 방지
  - 목적: GO:0007204 EF-hand, GO:0030017 sarcomere motif 포착
```

---

## 3. 아키텍처

### 입력 레이어

| 입력 | 차원 | 타입 | 설명 |
|------|------|------|------|
| `esm2_input` | 640 | float32 | ESM-2 mean-pool 임베딩 |
| `esm2_mask_input` | 1 | float32 | 0/1 (ESM-2 유효 여부) |
| `seq_input` | 1500 | int32 | 아미노산 서열 (SEQ_LEN=1500, p95 커버리지) |
| `domain_input` | dm_dim | int32 | InterPro domain token index |

### ESM-2 Branch (Phase 1 학습, Phase 2 Frozen)

```
esm2_input [640]
    → Dense(256, name='esm2_d1') → ReLU → Dropout(0.2)
    → Dense(128, name='esm2_d2') → ReLU
    → Dense(64, name='esm2_feat') → ReLU       [64]
    → Lambda(x[0]*x[1])([feat, mask])           [64] ← esm2_gated
```

학습 파라미터: `ESM2_TRAINABLE_NAMES = {'esm2_d1', 'esm2_d2', 'esm2_feat'}`

### CNN Branch (Phase 1 Frozen, Phase 2 학습)

```
seq_input [1500]
    → Embedding(8001, 32, name='cnn_emb')       [1500, 32]
    → Conv1D(64, kernel_size=7, padding='same', name='cnn_conv1')
    → Conv1D(32, kernel_size=5, padding='same', name='cnn_conv2')
    → GlobalMaxPooling1D(name='cnn_pool')        [32]
    → Dense(32, name='cnn_feat') → ReLU         [32]
```

학습 파라미터: `CNN_TRAINABLE_NAMES = {'cnn_emb', 'cnn_conv1', 'cnn_conv2', 'cnn_feat'}`

설계 근거:
- `kernel_size=7`: 아미노산 7-mer 모티프 (EF-hand: ~12aa, RYR: ~30aa 중 핵심 7aa 포착)
- `kernel_size=5`: 짧은 로컬 패턴 정제
- `GlobalMaxPooling`: 위치 불변 모티프 존재 여부 검출

### Domain Branch (전 단계 학습)

```
domain_input [dm_dim]
    → Embedding(domain_emb_dim, 32, name='dm_emb')
    → LSTM(16, name='domain_feat')               [16]
```

### Fusion & Head

```
concat = [esm2_gated(64), cnn_feat(32), domain_feat(16)]  → [112]
    → Dense(48) → ReLU → Dropout(0.2)
    → L2Normalize → embedding_layer [48]
    → Dense(1, sigmoid) → prediction
```

**EMB_DIM = 112** (64 ESM-2 + 32 CNN + 16 Domain)

---

## 4. 데이터 파이프라인

### SEQ_LEN=1500 선택 근거

| 통계 | 값 |
|------|-----|
| 최대 시퀀스 길이 | ~6000 |
| p95 | 1304 |
| p99 | ~2800 |
| 선택: SEQ_LEN | 1500 |

→ p95 이하 단백질은 손실 없이 커버, 메모리 효율 5× 향상

### hstack 레이아웃

```python
# 컬럼 인덱스
MASK_START = 640          # esm2(640)  | mask(1)
SEQ_START  = 641          # mask(1)    | seq(1500)
DM_START   = 2141         # seq(1500)  | dm(dm_dim)

X_combined = np.hstack([
    esm2_comb,            # [N, 640]
    mask_comb,            # [N, 1]
    seq_comb.astype(np.float32),   # [N, 1500]
    dm_comb.astype(np.float32)     # [N, dm_dim]
])
```

**메모리**: N=10,000 기준 → 10000 × 2141 × 4B ≈ 81MB (관리 가능)

### 3-call generate_label 설계

```
Call ①: labels, dm_comb  (domain → 라벨 기준 정렬)
Call ②: esm2_comb, mask_comb  (esm2+mask → 동일 add_index)
Call ③: _dummy, seq_comb  (esm2를 dummy key → seq 정렬)
```

결정성 보장: 동일 gene IDs → 동일 add_index → 동일 행 선택

### upsample 재현성

```python
np.random.seed(42)  # [Fix-DA] hstack 전체를 단일 upsample
X_combined_upsmp, y_upsmp = upsample(X_combined, labels_comb)
```

---

## 5. Phase별 학습 설정

### Phase 1: ESM-2 + Domain → Triplet

```python
set_cnn_trainable(feature_model, False)   # CNN frozen
# ESM-2 + Domain + Head trainable
# Loss: triplet (cross-gene negative, margin=0.3)
# 목적: global embedding structure 학습
optimizer: Adam(1e-4)
```

### Phase 1.5: Head 재조정

```python
for layer in feature_model.layers:
    layer.trainable = False               # 전체 frozen
# Loss: focal (γ=2, α=0.25)
# Head (classification_model) only
# 목적: Phase 1 embedding 보존하며 GO term 방향 정렬
optimizer: Adam(1e-4)
```

### Phase 2: CNN → Focal (ESM-2 Frozen)

```python
for layer in feature_model.layers:
    layer.trainable = True
set_esm2_trainable(feature_model, False)  # ESM-2 frozen
# Loss: focal only (γ=2, α=0.10) — triplet 없음
# CNN + Domain + Head trainable
# 목적: 위치 특이적 모티프 학습, ESM-2 임베딩 공간 보존
optimizer: Adam(5e-5)  # 작은 LR → Phase 1.5 방향 유지
```

**triplet 제거 이유**: Phase 2에서 triplet을 사용하면
- focal → 0 포화 후 triplet이 gradient 지배
- ESM-2 frozen이어도 CNN이 triplet에 맞게 재편성
- focal이 학습한 GO term 방향 파괴 위험

---

## 6. v6 vs v6-hybrid 비교

| 항목 | v6 | v6-hybrid |
|------|----|----|
| Phase 2 loss | focal + triplet | **focal only** |
| Phase 2 ESM-2 | trainable | **frozen** |
| CNN branch | 없음 | **있음 (Phase 2 전용)** |
| EMB_DIM | 80 (64+16) | **112 (64+32+16)** |
| 위치 특이적 모티프 | ESM-2 mean-pool (손실) | **CNN GlobalMaxPool (보존)** |
| Phase 2 regression 위험 | 높음 | **낮음** |

---

## 7. 예상 개선

### GO:0007204 (Ca²⁺-mediated signaling)

- v6 regression 원인: ESM-2 mean-pool에서 EF-hand 위치 정보 손실
- v6h 해결: CNN Conv1D(k=7)이 EF-hand 7-mer 패턴 직접 학습
- 예상: v6 0.300 → v6h 0.450+ (v5-5 0.508 회복 목표)

### GO:0030017 (sarcomere organization)

- v6 regression 원인: Phase 2 triplet이 sarcomere 클러스터 내부 구조 파괴
- v6h 해결: Phase 2 ESM-2 frozen → sarcomere 클러스터 보존, CNN이 추가 세분화
- 예상: v6 0.211 → v6h 0.380+ (v5-5 0.390 회복 목표)

### GO:0003774 (motor activity)

- v6 개선 유지 기대 (Phase 1 triplet 구조 동일)
- CNN이 myosin head domain 패턴 추가 학습 가능성

---

## 8. 실행 파일

```
hMuscle/model/v6h_integrated_full_model.py  ← 모델 본체
hMuscle/model/run_GPU_v6h.py               ← 실행 스크립트 (미생성)
hMuscle/model/v6_go_list.txt               ← GO term 목록 (공유)
```

### 실행 명령

```bash
conda activate isoform_env
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model
nohup python run_GPU_v6h.py > ../logs_isoform/run_v6h_$(date +%Y%m%d_%H%M).log 2>&1 &
tail -f ../logs_isoform/$(ls -t ../logs_isoform/ | head -1)
```

---

## 9. 향후 필요 실험

| 우선순위 | 실험 | 목적 |
|---------|------|------|
| Critical | `no_cross_gene_negative` ablation | Main Contribution 2 방어 |
| High | v6h vs v6 AUPRC 비교 (5 GO term) | Phase 2 regression 해결 확인 |
| High | Bootstrap CI (n=1000) [R9.4] | 개선 주장 통계 근거 |
| Medium | `no_phase1` ablation | Curriculum 인과성 방어 |
| Medium | `no_cnn` ablation | CNN branch 기여 수치화 |
