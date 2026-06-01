# v6 ESM-2 통합 설계 계획서

**날짜**: 2026-04-08  
**목적**: ESM-2 단백질 언어 모델의 구조적 임베딩을 v5-5에 통합하여 Go:0006936/0006941 계열의 seq+domain 2-modal 한계 극복  
**기반 모델**: v5-5_integrated_full_model.py  
**목표 버전**: v6_integrated_full_model.py

---

## 1. 설계 동기 — v5-5 분석에서 도출된 구체적 문제

### 1-1. 실패 패턴 분류 (로그 수치 기반)

| GO term | Phase 1 margin_sat | Phase 1 Sil | 실패 유형 | 2-modal 한계 |
|---------|-------------------|------------|---------|------------|
| GO:0006941 | **5.9%** | +0.681 | Phase 1 수렴 실패 | seq+domain이 striated vs non-striated 구분 불가 |
| GO:0006936 | — | 불안정 | initialization 민감 | 동일 수축 이소폼 간 서열 차이 미인코딩 |
| GO:0003774 | 27.4% | **-0.121** | Phase 1 Sil 음수 | motor domain 미세 서열 변이 포착 안 됨 |
| GO:0007204 | 22.9% | **-0.210** | Ph0 시작 음수 | Ca2+ 채널 TM domain 이소폼 차이 미인코딩 |

**공통 원인**: 현재 seq 인코더(k-mer Embedding + Conv1D + PyramidPool)는 **국소 서열 패턴** 만 포착. ESM-2는 자기지도 사전학습으로 **전역 구조 문맥(global structural context)** 을 인코딩 → 이소폼 간 기능 차이 구별에 필수.

### 1-2. ESM-2가 특히 기여할 생물학적 메커니즘

| GO term | ESM-2 기여 포인트 | 기대 개선 |
|---------|----------------|---------|
| GO:0006941/0006936 | Myosin heavy chain isoform (MYH1/2/4/7) 간 S1 motor domain 서열 변이 인코딩 | Phase 1 margin_sat 5.9% → 15%+ |
| GO:0003774 | Kinesin/dynein motor domain의 미세 서열 차이 — 같은 gene의 이소폼이라도 stalk domain 변이로 화물 특이성 다름 | Phase 1 Sil -0.121 → +0 이상 |
| GO:0007204 | RYR1/RYR2, SERCA1/2, CACNA1S/C 등 Ca2+ 채널 transmembrane helix 이소폼 변이 | Positive collapse 억제 |
| GO:0030017 | 이미 0.390. ESM-2는 **no_esm ablation control**로 기여도 정량화 | 기여도 측정 |

---

## 2. 데이터 파이프라인 — ESM-2 임베딩 사전 계산

### 2-1. 입력 데이터 현황 (이미 준비됨)

```
hMuscle/data/top30k_isoforms.pep        ← 36,776개 단백질 서열 (TransDecoder 결과)
hMuscle/model/my_isoform_list_fixed.npy ← 36,748개 이소폼 ID (BambuTx*)
```

- `top30k_isoforms.pep`에 36,748개 전체 coverage 확인 (100%)
- 각 이소폼별 최고 ORF 선택 기준: TransDecoder score 최대 (complete ORF 우선)

### 2-2. ESM-2 모델 선택

| 모델 | 파라미터 | 출력 dim | 4090 처리 시간 | 권장 |
|------|---------|---------|--------------|------|
| `esm2_t6_8M_UR50D` | 8M | 320 | ~20분 | ❌ 표현력 부족 |
| `esm2_t12_35M_UR50D` | 35M | 480 | ~45분 | 검토 |
| `esm2_t30_150M_UR50D` | 150M | 640 | ~90분 | ✅ **권장** |
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~4~6시간 | 검토 (성능 상한 확인용) |

**1차 개발**: `esm2_t30_150M_UR50D` (640-dim) — 처리 시간과 표현력 균형  
**성능 상한 확인**: `esm2_t33_650M_UR50D` (1280-dim) — ablation에서 상한 측정

### 2-3. 사전 계산 스크립트 설계

**파일**: `hMuscle/preprocessing/compute_esm2_embeddings.py`

```python
"""
compute_esm2_embeddings.py
--------------------------
목적: my_isoform_list_fixed.npy의 36748개 BambuTx 이소폼에 대해
     ESM-2 mean-pooled protein embeddings를 사전 계산하여 저장.

입력: top30k_isoforms.pep (TransDecoder 결과 단백질 서열)
출력: hMuscle/data/esm2_embeddings_t30_150M.npy  (shape: [36748, 640])
      hMuscle/data/esm2_isoform_index.npy         (순서: my_isoform_list_fixed.npy 와 동일)

실행:
  CUDA_VISIBLE_DEVICES=0 python compute_esm2_embeddings.py --model esm2_t30_150M_UR50D
  (GPU 0에서 ~90분, 결과 .npy에 저장)
"""

KEY STEPS:
1. pep 파일 파싱: 이소폼 ID → 가장 높은 score의 complete ORF 1개 선택
   - ORF type 우선순위: complete > 5prime_partial > 3prime_partial > internal
   - 동점 시 score 최대값 선택
   - 최대 길이 제한: 1022 aa (ESM-2 position embedding 한계) → 초과 시 N-terminal 1022aa truncation
2. my_isoform_list_fixed.npy 순서로 정렬 → 36748개 순서 index 생성
3. ESM-2 배치 추론 (batch_size=32, 512 token 기준 GPU VRAM ~8GB 사용)
4. 각 isoform: per-residue embedding → mean pooling → [dim] 벡터
5. 저장: np.save(emb_matrix, allow_pickle=False)
```

**처리 불가 이소폼 대응**:
- ORF 없는 경우 (<1%): 0-vector로 패딩 + mask flag 저장
- v6 모델에서 ESM-2 feature가 0인 경우 gate=0 처리 → seq+domain만 사용

### 2-4. Train 이소폼 ESM-2 임베딩

```
hMuscle/data/raw_data/data/sequences/human_sequence_train.npy: 31,668 NM_* 이소폼
→ UniProt에서 canonical 단백질 서열 다운로드 OR RefSeq 단백질 서열 사용
→ 저장: hMuscle/data/esm2_embeddings_train_t30_150M.npy (shape: [31,668, 640])
```

Train set은 RefSeq NM_ 이소폼이므로 Entrez API로 단백질 서열 일괄 취득 가능 (UniProt 매핑 또는 NCBI 직접).

---

## 3. v6 아키텍처 설계

### 3-1. 현재 v5-5 아키텍처 (참조)

```python
# Phase 1 Embedding space (32-dim) — Triplet loss 학습 대상
seq_input  → Embedding(8001,32) → Conv1D(64,32) → PyramidPool → Dense(32) → Dense(16) = seq_feat
domain_input → Embedding → LSTM(16) = domain_feat
concat = concatenate([seq_feat, domain_feat])     # 32-dim, feature_model 출력
                                                  # ← Triplet이 이 공간에서 작동
concat → Dense(16) → relu → Dropout → L2-norm = embedding_layer   # 16-dim
embedding_layer → Dense(1, sigmoid) = prediction
```

### 3-2. v6 아키텍처 — Gated ESM-2 Residual Fusion

#### 설계 원칙
- [R2.1] 직접 concatenation 금지 → **gating으로 ESM-2 통합**
- Triplet loss가 ESM-2 강화된 32-dim 공간에서 작동
- ESM-2 미활용 이소폼(mask=0)에서 gate=0 자동 처리

```python
# ── 기존 2-modal 브랜치 (변경 없음) ─────────────────────────────────
seq_input    = Input(shape=(None,), dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim,), dtype='int32', name='domain_input')

x1 = Embedding(input_dim=8001, output_dim=32)(seq_input)
x1 = Convolution1D(64, 32, activation='relu', padding='valid')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=l2(1e-5))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.2)(x1)
seq_feat = Dense(16, activation='relu')(x1)                       # 16-dim

x2 = Embedding(input_dim=domain_emb_dim, output_dim=32,
               input_length=dm_dim, mask_zero=True)(domain_input)
domain_feat = LSTM(16)(x2)                                        # 16-dim

concat_2modal = concatenate([seq_feat, domain_feat])              # 32-dim

# ── ESM-2 브랜치 (신규) ───────────────────────────────────────────────
esm2_input = Input(shape=(ESM2_DIM,), name='esm2_input')          # 640-dim (t30 기준)
esm2_mask  = Input(shape=(1,), name='esm2_mask')                  # 1 or 0 (유효 여부)

e = Dense(256, kernel_regularizer=l2(1e-5))(esm2_input)           # 640→256
e = Activation('gelu')(e)                                         # GELU (ESM-2와 동일 활성함수)
e = LayerNormalization(epsilon=1e-6)(e)
e = Dropout(0.1)(e)
e = Dense(64, kernel_regularizer=l2(1e-5))(e)                     # 256→64
e = Activation('gelu')(e)
e = LayerNormalization(epsilon=1e-6)(e)
esm2_proj = Dense(32, kernel_regularizer=l2(1e-5))(e)             # 64→32 [R2.1 통합 금지]

# Gated fusion: gate는 두 브랜치의 joint representation에서 계산
gate_input = concatenate([concat_2modal, esm2_proj])              # 64-dim
gate = Dense(32, activation='sigmoid', name='esm2_gate')(gate_input)  # 32-dim scalar gate

# mask 처리: ESM-2 없는 이소폼은 gate=0 강제
gate_masked = Multiply()([gate, esm2_mask])                       # mask=0 → gate=0 → seq+domain만

# Residual gated fusion
esm2_contrib  = Multiply()([gate_masked, esm2_proj])              # gate 적용 ESM-2 기여
seq_dm_contrib = Multiply()(                                      # (1-gate) 적용 seq+domain 기여
    [Lambda(lambda g: 1.0 - g)(gate_masked), concat_2modal])
fused = Add(name='fused_emb')([esm2_contrib, seq_dm_contrib])     # 32-dim

# feature_model_v6: Triplet loss가 이 공간에서 작동
feature_model_v6 = Model(
    inputs=[seq_input, domain_input, esm2_input, esm2_mask],
    outputs=fused,
    name='feature_model_v6')

# ── 분류 헤드 (v5-5와 동일 구조) ──────────────────────────────────────
x = Dense(16, kernel_regularizer=l2(1e-5))(fused)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer = Lambda(lambda a: K.l2_normalize(a, axis=1),
                         name='embedding_out')(x)                 # 16-dim L2 norm
prediction_layer = Dense(1, activation='sigmoid',
                         kernel_regularizer=l2(1e-5),
                         name='prediction_out')(embedding_layer)

base_model_v6 = Model(
    inputs=[seq_input, domain_input, esm2_input, esm2_mask],
    outputs=[embedding_layer, prediction_layer])
```

### 3-3. 아키텍처 선택 근거

| 결정 | 내용 | 근거 |
|------|------|------|
| Gate 입력 = concat(2modal, esm2_proj) | joint 64-dim에서 gate 계산 | 두 modality의 상호 정보 반영 |
| GELU 활성함수 | ESM-2 pre-training과 동일 | 표현 공간 호환성 |
| LayerNorm | ESM-2 내부 정규화와 일관성 | gradient 안정화 |
| 32-dim projection | EMB_DIM=32 유지 | Triplet 공간 차원 변경 없음, 기존 Phase 1 하이퍼파라미터 재사용 |
| Mask 입력 | 0-vector ESM-2 이소폼 처리 | gate=0 → seq+domain 전용 |

### 3-4. 학습 가능한 파라미터 추가 규모

```
ESM-2 Projector MLP:
  Dense(640→256): 640×256 + 256 = 164,096
  Dense(256→64):   256×64 + 64  =  16,448
  Dense(64→32):     64×32 + 32  =   2,080
Gate:
  Dense(64→32):     64×32 + 32  =   2,080
LayerNorm ×2:          32 + 32  =      64
──────────────────────────────────────────
총 신규 파라미터: ~185,000 (~185K)
기존 v5-5 파라미터: ~800K 추정
증가율: ~23%
```

---

## 4. 학습 파이프라인 변경사항

### 4-1. 데이터 로딩

```python
# v6에 추가되는 데이터 로딩 (기존 로딩 코드 하단에 추가)
ESM2_DIM     = 640  # esm2_t30_150M_UR50D
ESM2_EMBS    = '../data/esm2_embeddings_t30_150M.npy'       # test (36748, 640)
ESM2_EMBS_TR = '../data/esm2_embeddings_train_t30_150M.npy' # train (31668, 640)

X_test_esm2  = np.load(ESM2_EMBS).astype(np.float32)   # (36748, 640)
X_train_esm2 = np.load(ESM2_EMBS_TR).astype(np.float32) # (31668, 640)

# ESM-2 mask 생성 (0-vector → mask=0)
X_test_esm2_mask  = (np.linalg.norm(X_test_esm2,  axis=1, keepdims=True) > 0).astype(np.float32)
X_train_esm2_mask = (np.linalg.norm(X_train_esm2, axis=1, keepdims=True) > 0).astype(np.float32)
```

### 4-2. Phase 1 Triplet 입력 변경

현재 triplet model 입력: `[seq_a, dm_a, seq_p, dm_p, seq_n, dm_n]`  
v6 triplet model 입력: `[seq_a, dm_a, esm_a, mask_a, seq_p, dm_p, esm_p, mask_p, seq_n, dm_n, esm_n, mask_n]`

```python
# Phase 1 배치 구성 시 ESM-2 조회
def build_emb_triplet_inputs_v6(embeddings, y, pos_indices, neg_indices,
                                 X_seq, X_dm, X_esm2, X_mask, ...):
    # 기존 seq, dm 조회 + esm2, mask 조회 추가
    ...
    return (a_seq, a_dm, a_esm, a_mask,
            p_seq, p_dm, p_esm, p_mask,
            n_seq, n_dm, n_esm, n_mask), triplet_labels
```

### 4-3. Phase 1.5 ESM-2 처리

Phase 1.5 (Encoder frozen, 분류 헤드만 학습) 시 ESM-2 projector와 gate는 **frozen** 처리.

```python
# Phase 1.5에서 ESM-2 파라미터 동결
for layer in ['esm2_gate', 'dense_esm2_proj*']:
    model.get_layer(layer).trainable = False
```

### 4-4. Phase 2 — Gate 학습 활성화

Phase 2 joint training에서 gate와 ESM-2 projector를 **함께 학습**:

```python
# Phase 2 optimizer 설정
adam_p2 = optimizers.Adam(lr=0.0003)
# ESM-2 파라미터 unfreeze
for layer in base_model_v6.layers:
    layer.trainable = True
```

### 4-5. Gate Monitoring (새로운 진단 지표)

```python
# 학습 중 gate 값 모니터링 — ESM-2 기여도 추적
def log_gate_stats(base_model, X_seq, X_dm, X_esm2, X_mask, y):
    gate_model = Model(base_model.inputs, base_model.get_layer('esm2_gate').output)
    gates = gate_model.predict([X_seq, X_dm, X_esm2, X_mask], batch_size=512)
    print(f"  Gate | pos mean={gates[y==1].mean():.3f}  neg mean={gates[y==0].mean():.3f}"
          f"  overall mean={gates.mean():.3f}  std={gates.std():.3f}")
    # gate > 0.5인 이소폼 비율 (ESM-2 우세 구간)
    print(f"  Gate>0.5: {(gates>0.5).mean()*100:.1f}% isoforms")
```

Phase 1 완료 시, Phase 2 매 에폭마다 gate 통계 출력. **Gate가 전혀 열리지 않으면** (mean < 0.05) ESM-2 projector 학습률 10× 증가 처리.

---

## 5. 어블레이션 설계

### 5-1. 필수 어블레이션 (논문 Table 용)

| 실험명 | 설명 | 목적 |
|--------|------|------|
| `v5-5` | seq+domain 2-modal (기존) | baseline |
| `v6_esm2_t30` | seq+domain+ESM-2(150M) gated | 핵심 기여 |
| `v6_esm2_t33` | seq+domain+ESM-2(650M) gated | 성능 상한 |
| `v6_no_gate` | ESM-2 직접 concat (gating 없이) | gate 중요성 [R2.1] |
| `v6_esm2_only` | ESM-2 단독 (seq+domain 제거) | ESM-2 단독 성능 |
| `v6_frozen_esm2` | ESM-2 projector frozen, gate만 학습 | projector 학습 필요성 |

### 5-2. Phase별 어블레이션

| 실험명 | 설명 | 확인 가설 |
|--------|------|---------|
| `v6_phase1_only` | Phase 1만 ESM-2 사용, Phase 2는 seq+domain | Phase 1 발판 개선 기여도 |
| `v6_phase2_only` | Phase 2에서만 ESM-2 추가 | Phase 2 분류 헤드 기여도 |

---

## 6. 파일/디렉토리 구조

```
DIFFUSE/
├── hMuscle/
│   ├── preprocessing/
│   │   └── compute_esm2_embeddings.py   ← 신규 (사전 계산 스크립트)
│   ├── data/
│   │   ├── esm2_embeddings_t30_150M.npy      ← 신규 (36748, 640)
│   │   ├── esm2_embeddings_train_t30_150M.npy ← 신규 (31668, 640)
│   │   ├── esm2_embeddings_t33_650M.npy      ← 선택적 (성능 상한용)
│   │   └── esm2_isoform_index.npy             ← 신규 (순서 검증용)
│   └── model/
│       ├── v5-5_integrated_full_model.py   ← 기존 (수정 없음)
│       ├── v6_integrated_full_model.py     ← 신규 (ESM-2 통합)
│       └── run_GPU_v6.py                   ← 신규 (실행 스크립트)
└── reports/
    └── 2026-04-08/
        ├── generality_report_20260408.md   ← 기존 업데이트됨
        └── v6_esm2_design_plan.md          ← 이 문서
```

---

## 7. 구현 로드맵 (단계별)

### Step 0: 사전 조건 확인 (1시간)

```bash
# ESM-2 패키지 확인
conda activate isoform_env
python -c "import esm; print(esm.__version__)"
# 없으면:
pip install fair-esm

# GPU 메모리 확인 (batch_size=32, len=512 기준 ~8GB 필요)
nvidia-smi
```

### Step 1: ESM-2 임베딩 사전 계산 (~1.5시간, GPU 0 전용)

```bash
CUDA_VISIBLE_DEVICES=0 python compute_esm2_embeddings.py \
    --model esm2_t30_150M_UR50D \
    --pep_file ../data/top30k_isoforms.pep \
    --isoform_list my_isoform_list_fixed.npy \
    --output ../data/esm2_embeddings_t30_150M.npy \
    --batch_size 64 \
    --max_len 1022
```

완료 후 검증:
```python
import numpy as np
emb = np.load('../data/esm2_embeddings_t30_150M.npy')
assert emb.shape == (36748, 640), f"shape mismatch: {emb.shape}"
assert not np.isnan(emb).any(), "NaN in embeddings"
mask = (np.linalg.norm(emb, axis=1) > 0)
print(f"Valid embeddings: {mask.sum()}/36748 ({mask.mean()*100:.1f}%)")
```

### Step 2: v6 모델 코드 작성 (~3시간)

`v5-5_integrated_full_model.py`를 기반으로 `v6_integrated_full_model.py` 작성:

1. **추가 상수**: `ESM2_DIM = 640`, `ESM2_MODEL = 'esm2_t30_150M_UR50D'`
2. **데이터 로딩**: `X_test_esm2`, `X_train_esm2`, `mask` 배열 추가
3. **아키텍처**: §3-2 코드 블록 그대로 구현
4. **Triplet 배치 빌더**: `build_emb_triplet_inputs_v6()` (ESM-2 인덱스 조회 추가)
5. **Phase 1**: `phase1_embedding_triplet_epoch_v6()` — 입력 12개 (기존 6 → +6)
6. **Phase 1.5**: ESM-2 projector frozen
7. **Phase 2**: Gate monitoring 로그 추가
8. **Gate 진단**: `log_gate_stats()` Phase 2 매 에폭 출력

### Step 3: 단일 GO term 검증 실행 (~50분, GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 python -u v6_integrated_full_model.py GO:0006936 \
    > ../logs_isoform/GO_0006936/v6_GO_0006936_$(date +%Y%m%d_%H%M)_Full.log 2>&1
```

**성공 기준** (Go/No-Go 판단):
```
Phase 1 margin_sat > 10%       (v5-5: 불안정 시 5.9%)
Phase 2 AUPRC (epoch 5) > 0.15 (v5-5: 0.251 최종)
gate mean ∈ [0.1, 0.9]        (gate 활성화 확인)
```

### Step 4: 4개 원래 GO term 전체 실행 (~4시간, dual GPU)

```bash
nohup python run_GPU_v6.py > ../logs_isoform/run_v6_$(date +%Y%m%d_%H%M).log 2>&1 &
```

### Step 5: 8개 GO term 전체 어블레이션 (v6 안정화 후)

`no_gate`, `esm2_only`, `frozen_esm2` 등 어블레이션 배리언트 순차 실행.

---

## 8. 성능 기대값

### 8-1. GO term별 예측 (v5-5 대비)

| GO term | v5-5 AUPRC | v6 예측 AUPRC | 개선 근거 |
|---------|-----------|--------------|---------|
| GO:0006936 | 0.251 | **0.35~0.45** | MHC isoform 서열 차이 ESM-2 인코딩 → margin_sat 개선 |
| GO:0006941 | 0.095 | **0.20~0.30** | Phase 1 수렴 실패 원인 해소 (margin_sat 5.9% → 15%+) |
| GO:0003774 | 0.205 | **0.25~0.35** | Phase 1 Sil 음수 → ESM-2 motor domain 특이성 보완 |
| GO:0007204 | 0.507 | **0.50~0.55** | 이미 포화 상태, collapse 개선 가능성 |
| GO:0030017 | 0.390 | **0.38~0.42** | 이미 높음, ESM-2 기여 제한적 — ablation control |
| GO:0006096 | 0.670 | **0.65~0.68** | 이미 높음, expression shortcut 지배 |
| GO:0022900 | 0.357 | **0.35~0.40** | 중간 개선 예상 |
| GO:0006412 | 0.209 | **0.25~0.35** | housekeeping term, ESM-2 ribosome isoform 구별 |

**예상 Macro-AUPRC (8개)**: 0.322 → **0.38~0.42**

### 8-2. Gate 동작 예측

| GO term | 예상 gate mean | 의미 |
|---------|--------------|------|
| GO:0006941 | 0.3~0.5 | ESM-2 기여 중간 |
| GO:0006936 | 0.4~0.6 | ESM-2 기여 높음 |
| GO:0030017 | 0.1~0.2 | seq+domain 우세 (ESM-2 기여 낮음) |
| GO:0006096 | 0.1~0.3 | expression 신호 우세, ESM-2 보조 |

---

## 9. 위험 요소 및 대응책

| 위험 | 확률 | 대응 |
|------|------|------|
| Gate 비활성화 (mean < 0.05) | 중간 | ESM-2 projector lr 10×, gate bias 초기화 +0.5 |
| ESM-2 feature가 seq+domain을 완전히 대체 (gate → 1) | 낮음 | gate에 L2 패널티, target gate ∈ [0.3, 0.7] 정규화 검토 |
| train/test ESM-2 분포 불일치 | 낮음 | LayerNorm이 완충, 추가로 BatchNorm 검토 |
| 학습 속도 저하 (입력 12개) | 높음 | Phase 1 batch_size 절반으로 조정 (512→256) |
| ESM-2 임베딩 계산 OOM | 낮음 | batch_size=32 (길이 512 기준 ~6GB) |

---

## 10. 논문 서술 연결

```
§3 Methods — ESM-2 Integration:
"We extend the 2-modal feature model (seq+domain) of v5-5 with ESM-2 protein language model
embeddings using a gated residual fusion architecture. Given the pre-computed ESM-2 mean-pooled
embeddings e_i ∈ R^{640} for isoform i, a 3-layer MLP projects e_i to the embedding space
(R^{32}). A learned gate g = σ(W_g[z_i; e'_i]) ∈ (0,1)^{32}, where z_i is the seq+domain concat
embedding, determines the contribution of each modality. The fused representation
f_i = g ⊙ e'_i + (1-g) ⊙ z_i is passed to the Triplet loss in Phase 1, enabling ESM-2 structural
context to guide the initial embedding organization — directly addressing the Phase 1 margin_sat
failure observed in GO:0006941 (margin_sat=5.9%) and GO:0006936."
```

---

*설계 기반: v5-5 로그 분석 (2026-04-08)*  
*다음 단계: Step 0 (ESM-2 패키지 확인) → Step 1 (임베딩 계산) → Step 2 (모델 코드 작성)*  
*GO:0003779 Tier-2 실행 중 (PID: 2067519, GPU1, 완료 예상: ~02:45)*
