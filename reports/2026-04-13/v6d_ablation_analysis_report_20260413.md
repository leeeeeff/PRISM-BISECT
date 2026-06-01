# v6d 아키텍처 컴포넌트 분석 및 Ablation 설계 레포트

**작성일**: 2026-04-13  
**대상 모델**: v6d_integrated_full_model.py  
**목적**: Nature Methods / NMI 투고를 위한 각 컴포넌트 기여도 분리 및 Gene-level reference dominance 극복 주장 검증

---

## 0. 모델 아키텍처 개요

```
입력 → [ESM-2 branch  (640→64)]  ← 진화 보존 맥락
      [CNN   branch  (1500→32)]  ← 위치 특이적 서열 모티프
      [Domain LSTM   (→16)]      ← 도메인 조합/순서
      [DomainDelta   (251→16)]   ← 이소폼-특이적 도메인 변화
           ↓ concat[128]
      Dense(64) → L2_norm → emb[64] → sigmoid

훈련 단계:
  Phase 1  (triplet): ESM-2✅  CNN❌  Domain✅  DeltaBranch✅
  Phase 1.5 (frozen): ESM-2❌  CNN❌  Domain❌  DeltaBranch❌
  Phase 2   (focal):  ESM-2❌  CNN✅  Domain✅  DeltaBranch✅
  Phase 3   (label propagation): 학습 없음
```

---

## 1. ESM-2 Branch

### 1.1 생물학적 맥락

ESM-2(650M, t30)는 2억 5천만 개 이상의 단백질 서열로 사전 학습된 protein language model로, 진화적으로 보존된 위치별 잔기 맥락(residue context)을 인코딩한다.

**포착하는 신호 패턴:**
- **진화적 제약(evolutionary constraint)**: alignment-free하게 학습된 co-evolutionary 의존성. 기능적으로 중요한 위치(active site, binding interface)는 다른 잔기 환경보다 attention weight가 높다.
- **구조적 경향성(structural propensity)**: ESM-2 임베딩은 3D 구조 예측(AlphaFold)에 활용될 만큼 secondary structure, solvent accessibility를 암묵적으로 인코딩한다.
- **계통학적 정보(phylogenetic signal)**: 유사한 기능을 가진 isoform은 서열 공간에서 더 가까이 위치하는 경향이 있다.

**이소폼 예측에서의 역할:**
같은 gene 내 isoform들은 대부분 서열이 매우 유사(>80% identity)하지만, ESM-2는 잔기 맥락 변화를 민감하게 포착한다. Alternative exon이 새로운 결합 모티프를 도입하거나 구조적으로 불안정한 영역을 만드는 경우, ESM-2 임베딩 공간에서 거리가 유의하게 달라진다.

**학습 단계 설계 근거:**
Phase 1에서 triplet loss로 학습 → cross-gene negative를 사용해 "같은 기능 = 서열과 무관하게 임베딩 공간에서 가깝게" 학습. Phase 2에서 frozen → CNN이 독립적으로 position-specific signal을 학습할 수 있도록 ESM-2 임베딩 공간을 보존.

### 1.2 현재 구조

```python
# 640 → Dense(256) → Dense(128) → Dense(64) → Gate(×mask) → esm2_gated[64]
x_esm = Dense(256, name='esm2_d1')(esm2_input)  # L2 reg 1e-5
x_esm = Dense(128, name='esm2_d2')(x_esm)
esm2_feat  = Dense(64, name='esm2_feat')(x_esm)
esm2_gated = esm2_feat * esm2_mask              # mask=1: ESM 있음, 0: 없음
```

**mask gate 설계 이유**: Train set 중 일부 isoform은 ESM-2 임베딩 미적용(novel isoform 등) → mask=0으로 해당 branch 기여 차단. Novel isoform case에서 CNN, Domain, DomainDelta branch가 주도적으로 기여하도록 설계.

### 1.3 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_esm` | esm2_gated를 zeros로 강제 (`esm2_mask = 0`) | ESM-2가 전체 성능에 기여하는가? |
| `esm_only` | CNN, Domain, DomainDelta branch 제거 | ESM-2 단독 성능 (baseline upper bound) |
| `no_esm_gate` | gate 제거: `esm2_gated = esm2_feat` | ESM-2 mask gate의 novel isoform 처리 기여 |
| `esm_frozen_all` | Phase 1에서도 ESM-2 frozen | Phase 1 triplet 학습에서 ESM-2 적응이 필요한가? |

**구현 방법:**
```python
# no_esm: mask를 0으로 고정
esm2_mask_in = tf.zeros_like(esm2_mask_in)  # 모든 ESM-2 기여 차단

# no_esm_gate: gate 제거
esm2_gated = esm2_feat  # mask 무시

# esm_only: 다른 branch concat에서 제거
concat = concatenate([esm2_gated])  # esm2_gated[64]만 사용
```

**기대 결과 해석:**
- `no_esm` AUPRC가 크게 하락(>15%) → ESM-2가 주도적 신호 제공
- `esm_only` vs `no_esm` 차이가 작음 → CNN/Domain이 ESM-2와 중복 신호 제공 (redundancy 문제)
- `no_esm_gate` 차이 없음 → ESM-2 coverage가 100%라 mask gate 불필요 (데이터 확인 필요)

---

## 2. CNN Branch

### 2.1 생물학적 맥락

CNN branch는 아미노산 서열(1500 잔기, last 1500 truncation)에서 **위치 특이적 서열 모티프(position-specific sequence motif)**를 감지한다.

**포착하는 신호 패턴:**
- **기능 모티프(functional motif)**: Prosite 패턴, ELM (Eukaryotic Linear Motif) 등 단거리 선형 모티프. 예: RGD(integrin binding), KFERQ(chaperone-mediated autophagy), NLS(nuclear localization signal)
- **Alternative splicing 결과**: Alternative exon에 의해 도입/삭제되는 로컬 서열 패턴. Conv1D(k=7)는 약 7 잔기 윈도우, Conv1D(k=5)는 5 잔기 윈도우에서 모티프를 학습.
- **C-말단/N-말단 신호**: 분비 신호 펩타이드, GPI anchor, transmembrane helix는 특정 위치에 집중 → GlobalMaxPool이 전체 서열에서 가장 강한 모티프 신호를 추출.

**이소폼 예측에서의 역할:**
동일 gene 내 isoform 간 서열 차이는 주로 특정 exon의 포함/배제로 발생한다. Alternative exon이 특이적 모티프를 포함하는 경우(예: cardiac 특이적 exon이 Ca2+ binding 모티프를 포함), CNN이 이를 감지해 GO:0007204(Ca2+ signaling)와 연결.

**Phase 분리 설계 근거:**
Phase 1에서 CNN을 frozen → ESM-2와 Domain branch가 먼저 isoform-level 임베딩 공간을 구성. Phase 2에서 CNN 학습 → 이미 구성된 임베딩 공간 위에서 position-specific fine-tuning. 만약 Phase 1에서 CNN도 학습하면 CNN이 triplet gradient를 선점해 ESM-2 학습 방해(v6c Phase1a anti-pattern, 실험적으로 확인됨).

### 2.2 현재 구조

```python
# Embedding(8001, 32) → Conv1D(64, k=7) → Conv1D(32, k=5) → GlobalMaxPool → Dense(32)
x_seq = Embedding(8001, 32)(seq_input)           # 아미노산 vocab 8001 (패딩 포함)
x_seq = Conv1D(64, kernel_size=7, padding='same')(x_seq)  # 7-residue 모티프
x_seq = Conv1D(32, kernel_size=5, padding='same')(x_seq)  # 5-residue 모티프
x_seq = GlobalMaxPooling1D()(x_seq)              # 서열 전체에서 최강 신호
cnn_feat = Dense(32)(x_seq)                      # 32-dim CNN feature
```

**last 1500 truncation 근거**: 단백질 서열의 median 길이가 약 450 aa이나 일부 isoform은 매우 길다. p95 커버를 위해 1500으로 설정하되, 긴 서열은 **C-말단 1500 잔기**만 사용. C-말단이 통상 신호 서열, transmembrane domain 등 기능적으로 중요한 영역을 포함한다는 생물학적 근거.

### 2.3 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_cnn` | cnn_feat를 zeros로 강제 | CNN의 단독 기여 |
| `cnn_only` | ESM-2, Domain, DomainDelta 제거 | CNN 단독 성능 |
| `cnn_phase1_unfreeze` | Phase 1에서 CNN도 학습 | Phase 1 CNN freeze의 필요성 (v6c anti-pattern 재검증) |
| `cnn_k3_only` | Conv1D(k=7)+Conv1D(k=5) → Conv1D(k=3) | kernel size의 모티프 길이 영향 |
| `cnn_k5_k3_k1` | Multi-scale: k=5, k=3, k=1 | 더 짧은 모티프가 유효한가? |
| `cnn_truncate_n` | C-말단 대신 N-말단 1500 truncation | 어느 끝이 더 중요한가? |

**구현 방법:**
```python
# no_cnn
cnn_feat = tf.zeros((batch_size, 32))

# cnn_phase1_unfreeze (학습 단계 변경)
set_cnn_trainable(feature_model, True)   # Phase 1에서도 trainable
```

**기대 결과 해석:**
- `cnn_phase1_unfreeze`에서 GO:0006941 margin_sat이 급락 → v6c Phase1a anti-pattern 메커니즘 재확인 (paper에서 anti-pattern으로 보고 가능)
- `no_cnn` AUPRC 하락 < `no_esm` AUPRC 하락 → ESM-2 > CNN 기여도 (Phase 2가 ESM-2 없이 CNN만으로 하는 이유 설명)

---

## 3. Domain LSTM Branch

### 3.1 생물학적 맥락

Pfam domain은 단백질의 기능적 단위(functional unit)로, HMMER profile-based로 검출된다. Domain LSTM branch는 단순 domain 유무(presence/absence)가 아닌 **도메인들의 조합과 순서(domain architecture)**를 인코딩한다.

**포착하는 신호 패턴:**
- **Domain architecture**: 같은 도메인이 있더라도 N-말단부터 C-말단 방향으로의 순서가 다르면 기능이 다를 수 있다. 예: PH-GEF 순서 vs GEF-PH 순서는 다른 활성화 메커니즘. LSTM이 이 순서 의존성을 포착.
- **도메인 공존 패턴(domain co-occurrence)**: kinase domain과 SH2 domain의 공존 → 신호 전달. LSTM의 hidden state에 이전 도메인 정보가 누적되어 조합 패턴 학습.
- **Novel isoform domain disruption**: Alternative splicing이 domain 경계를 가로질러 발생하면 불완전한 도메인(truncated domain)이 생성. LSTM은 domain 서열의 단절을 시간 순서 이상(temporal anomaly)으로 감지.

**이소폼 예측에서의 역할:**
GO:0003774(myosin motor activity)의 경우 MYH 계열 isoform은 IQ motif(칼모듈린 결합) 개수와 배열이 isoform마다 다르다. LSTM이 IQ motif의 반복 횟수와 순서를 포착하여 myosin binding 기능을 예측하는 데 기여할 수 있다.

**전 구간 학습 설계 근거:**
Domain 정보는 이소폼 간 차이를 잘 반영하면서도 안정적인 구조적 특징이다. Phase 1에서는 triplet을 통한 유사도 구조 학습, Phase 2에서는 focal loss를 통한 분류 최적화 양쪽에서 모두 유용하다.

### 3.2 현재 구조

```python
# domain_input: int sequence (Pfam domain IDs, padding=0)
x_dm = Embedding(domain_emb_dim, 32, mask_zero=True)(domain_input)
domain_feat = LSTM(16)(x_dm)   # 도메인 순서 인코딩 → 16-dim
```

### 3.3 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_domain` | domain_feat = zeros | Domain LSTM 단독 기여 |
| `domain_bow` | LSTM → GlobalAvgPool (순서 무시) | LSTM의 순서 인코딩이 단순 BoW보다 유리한가? |
| `domain_only` | ESM-2, CNN, DomainDelta 제거 | Domain 단독 성능 |
| `domain_pretrained` | Pfam2Vec으로 domain embedding 초기화 | 사전 학습된 domain representation의 이점 |
| `domain_full_phase` | Phase 2에서 Domain LSTM frozen | Phase 2에서 Domain 학습이 필요한가? |

**구현 방법:**
```python
# domain_bow: LSTM → GlobalAvgPool
x_dm = Embedding(domain_emb_dim, 32, mask_zero=True)(domain_input)
x_dm_pool = tf.reduce_mean(x_dm, axis=1)           # BoW
domain_feat = Dense(16)(x_dm_pool)

# no_domain: LSTM 출력을 zeros로
domain_feat = tf.zeros((batch_size, 16))
```

**기대 결과 해석:**
- `domain_bow` ≈ `no_domain_lstm` → LSTM의 순서 인코딩 효과 없음 → Dense로 대체 고려
- `no_domain` AUPRC 하락이 GO:0003774에서 특히 크면 → Domain이 myosin binding에 핵심 신호

---

## 4. DomainDelta Branch

### 4.1 생물학적 맥락

DomainDelta는 **이소폼 i가 canonical 이소폼 대비 어떤 Pfam domain을 추가/상실했는가**를 인코딩한다.

```
domain_delta[i] = sign(domain_matrix[i] - domain_matrix[canonical_i])
```

- `+1`: 이소폼 i가 canonical보다 해당 domain 스코어 높음 (domain gain 또는 강화)
- `-1`: 이소폼 i가 canonical보다 해당 domain 스코어 낮음 (domain loss 또는 약화)
- `0`: 동일

**canonical 정의**: gene 내에서 Pfam nonzero count가 가장 많은 이소폼 (domain-richest isoform).

**포착하는 신호 패턴:**
- **Alternative exon에 의한 domain insertion/deletion**: 단순 domain 유무가 아니라 canonical 대비 변화. 예: 근육 특이적 exon이 Fibronectin type-III domain을 추가 → delta=+1 → 세포외기질 결합 기능 특이성
- **Domain boundary disruption**: alternative splicing이 domain 내부를 잘라 불완전한 domain 생성 → domain score 감소 → delta=-1
- **Isoform specialization signal**: canonical(가장 '완전한' domain을 가진 reference)과의 차이를 정규화된 비교 형태로 제공

**devils-advocate 지적에 대한 반론**:
canonical이 gene-level reference인 것은 사실이나, DomainDelta의 목적은 "gene context 제거"가 아니라 "gene context 내에서 해당 이소폼의 상대적 위치를 인코딩"하는 것이다. 이를 통해 모델이 "이 이소폼은 gene의 full-length version과 어떻게 다른가?"라는 질문에 답할 수 있게 된다. 이것은 gene-level bias를 제거하는 것이 아니라, gene-level reference를 명시적 신호로 활용하는 **complementary approach**다.

**따라서 논문 주장은 재정의 필요:**  
~~"Gene-level reference dominance 극복"~~ → **"Canonical-relative domain architecture delta as isoform-specific functional divergence signal"**

### 4.2 현재 구조

```python
# DomainDelta: Input(251, sign) → Dense(64) → Dense(16) → dd_feat[16]
x_dd = Dense(64, name='dd_dense1')(dd_input)   # L2 reg 1e-5
x_dd = Dropout(0.2)(x_dd)
dd_feat = Dense(16, name='dd_dense2')(x_dd)

# Phase 1: DomainDelta trainable (ESM-2와 함께 triplet 학습)
# Phase 2: DomainDelta trainable (CNN과 함께 focal 학습)
```

### 4.3 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_dd` | dd_feat = zeros | DomainDelta의 단독 기여 |
| `dd_raw_domain` | sign delta 대신 raw domain score 사용 | sign transform의 필요성 |
| `dd_log1p` | sign 대신 sign × log1p(abs(delta)) | 크기 정보 추가의 효과 |
| `dd_binary` | {0, 1} (domain 있으면 1, 없으면 0) | delta(상대값) vs absolute(절대값) 비교 |
| `dd_canonical_longest` | canonical = longest transcript (NCBI RefSeq 기준) | canonical 정의가 성능에 미치는 영향 |
| `dd_canonical_mane` | canonical = MANE Select transcript | 임상적으로 정의된 canonical의 효과 |
| `dd_frozen_phase2` | Phase 2에서 DomainDelta frozen | Phase 2에서 dd branch 학습이 필요한가? |
| `dd_phase1_frozen` | Phase 1에서 DomainDelta frozen | DomainDelta의 triplet 학습 기여 |

**구현 방법:**
```python
# no_dd: DomainDelta 제거
concat = concatenate([esm2_gated, cnn_feat, domain_feat])  # dd_feat 제거, EMB_DIM=112

# dd_raw_domain: sign 대신 raw score 사용 (단, 정규화 필요)
X_test_dd = domain_matrix[test_idx] / (domain_matrix[test_idx].max(axis=1, keepdims=True) + 1e-8)

# dd_binary: 0/1 절대값
X_test_dd = (domain_matrix[test_idx] > 0).astype(np.float32)

# dd_canonical_longest: canonical 재정의
# build_train_domain_delta.py에서 canonical 선택 기준 변경
canonical_local = np.argmax([len(seq) for seq in gene_seqs])  # 최장 서열
```

**기대 결과 해석:**
- `no_dd` AUPRC 하락 vs `no_domain` AUPRC 하락 비교 → DomainDelta vs Domain LSTM 상대 기여도
- `dd_raw_domain` ≈ `no_dd` → sign transform이 핵심 (정보 압축 효과)
- `dd_binary` < `dd_raw_domain` → domain score 크기가 의미 있음
- `dd_canonical_mane` > `dd_canonical_maxdomain` → 임상적 canonical이 생물학적으로 더 적합한 reference

---

## 5. Fusion Layer 및 훈련 전략

### 5.1 Fusion 설계

```python
concat = concatenate([esm2_gated[64], cnn_feat[32], domain_feat[16], dd_feat[16]])
# → [128-dim]
x = Dense(64)(concat) → L2_norm → emb[64] → sigmoid
```

**차원 배분의 함의:**
- ESM-2: 64/128 = 50% → 진화 보존 맥락이 절반 차지
- CNN: 32/128 = 25% → 위치 특이적 모티프
- Domain: 16/128 = 12.5% → 도메인 조합
- DomainDelta: 16/128 = 12.5% → 이소폼 특이적 변화

### 5.2 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `balanced_dim` | ESM[32] + CNN[32] + Domain[32] + Delta[32] = 128 | 차원 배분이 성능에 미치는 영향 |
| `no_l2_norm` | L2 normalization 제거 | triplet space와 classification space의 통합 효과 |
| `attention_fusion` | concat → cross-attention | attention 기반 modality weighting |

---

## 6. Phase 1 Triplet Loss

### 6.1 생물학적 맥락

Triplet loss는 "같은 GO 기능을 공유하는 이소폼은 임베딩 공간에서 가깝고, 다른 기능의 이소폼은 멀다"는 metric learning 목표를 실현한다.

**핵심 설계 선택: Cross-gene negative sampling [R2.1]**
```
negative = cross-gene isoform with different function
```
Intra-gene negative를 사용하면 모델이 gene identity로 negative를 구분하는 shortcut을 학습할 위험이 있다. Cross-gene negative는 이 shortcut을 차단하고 function-specific representation을 강제한다.

**Semi-hard negative mining [R3.2]:**
```python
semi_mask = (d_an > d_ap) & (d_an < d_ap + margin)
```
Easy negative(이미 충분히 먼 쌍)는 gradient 기여 없음. Hard negative(현재 positive보다 더 가까운 쌍)는 붕괴 위험. Semi-hard는 productive gradient를 최대화.

### 6.2 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_triplet` | Phase 1 제거, Phase 2만 실행 | Triplet loss의 임베딩 구조화 기여 |
| `intra_gene_neg` | negative를 same-gene isoform으로 교체 | Cross-gene negative의 gene-bias 방지 효과 |
| `hard_negative` | semi-hard → fully hard negative | Mining 전략 비교 |
| `margin_0.1` | margin=0.3 → margin=0.1 | margin 값 민감도 |
| `margin_1.0` | margin=0.3 → margin=1.0 | 큰 margin의 효과 (embedding collapse 방지) |
| `supcon` | Triplet → SupCon loss | 더 안정적인 contrastive learning [R3.3] |

**구현 방법:**
```python
# intra_gene_neg: negative를 같은 gene 내 다른 레이블 isoform으로
n_i = same_gene_diff_label_isoform(a_i, gene_dict, y)

# SupCon: temperature-scaled
loss = -log(sum(exp(z_i · z_j / τ) for j in same_class) /
            sum(exp(z_i · z_k / τ) for k in all))
```

**기대 결과 해석:**
- `intra_gene_neg` AUPRC >> `no_triplet` → triplet이 기여하지만 cross-gene neg이 중요하지 않음 (gene-bias 없음)
- `intra_gene_neg` AUPRC << `cross_gene_neg` → cross-gene negative가 gene-bias 방지에 핵심

---

## 7. Phase 2 Focal Loss 전략

### 7.1 생물학적 맥락

이소폼 기능 데이터는 극단적으로 불균형(positive: 0.1~5%, negative: 95~99.9%). Focal loss [R1.1]는 easy negative에 의한 loss 지배를 방지하고 소수 positive 샘플에 집중.

```
FL(p_t) = -α_t(1-p_t)^γ · log(p_t), γ=2, α=class-balanced
```

**Phase 2 설계:**
- ESM-2 frozen → 이미 학습된 function-specific embedding space 보존
- CNN + Domain + DomainDelta trainable → 이 세 modality가 fine-grained classification 담당
- ESM-2가 Phase 2에서도 학습하면 triplet으로 구성한 임베딩 공간이 파괴됨 (v6c 실험에서 확인)

### 7.2 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_focal` | Focal → BCE | Focal loss의 class imbalance 처리 효과 |
| `gamma_1` | γ=2 → γ=1 | γ 값 최적화 |
| `gamma_3` | γ=2 → γ=3 | 더 강한 hard example 집중 |
| `esm_phase2_unfreeze` | Phase 2에서 ESM-2 학습 | Phase 2 ESM-2 동결의 필요성 (v6c-fix3 재검증) |
| `no_phase1_5` | Phase 1.5(linear probing) 제거 | calibration step의 기여 |

---

## 8. Phase 3 Expression Label Propagation

### 8.1 생물학적 맥락

단일세포 RNA-seq 발현 데이터(24개 세포 타입)를 기반으로 발현 패턴이 유사한 이소폼끼리 레이블을 전파.

**신호 패턴:**
- 같은 조직/세포 타입에서 함께 발현되는 이소폼은 기능적으로 관련될 가능성 높음
- Muscle-specific 이소폼은 muscle cell type에서 발현 높음 → GO:0006941, 0003774 예측에 직접 기여
- Cell type-specific expression이 GO term prediction을 보완: 서열/도메인 단독으로 구분하기 어려운 경우 발현 유사도로 레이블 보정

**한계**: 발현 데이터가 없는 이소폼(novel isoform)은 label propagation 효과 없음 → Phase 3은 novel isoform에게 불리할 수 있음.

### 8.2 Ablation 설계

| 실험 코드 | 변경 내용 | 검증 가설 |
|-----------|-----------|-----------|
| `no_labelprop` | Phase 3 제거 | Label propagation의 AUPRC 기여 |
| `alpha_0.1` | alpha=0.3 → alpha=0.1 | 전파 강도 최적화 |
| `alpha_0.5` | alpha=0.3 → alpha=0.5 | 강한 전파의 효과 |
| `novel_isoform_only` | ESM-2 mask=0인 isoform만 평가 | Novel isoform에서 label propagation 효과 |

---

## 9. Gene-Bias 측정 분석

### 9.1 Gene-Bias Score

devils-advocate의 지적: AUPRC +10% 자체가 gene-bias 극복의 증거가 아님. 직접 측정 필요.

**구현:**
```python
def compute_gene_bias_score(y_pred, y_true, gene_ids):
    """
    bias_score = 1 - H(y|isoform_id) / H(y|gene_id)
    0: isoform과 gene이 동등한 예측력
    → 1: gene이 y를 완전히 결정 (isoform 무관) — 최악
    < 0: isoform이 gene보다 더 정보적 — 이상적
    """
    from scipy.stats import entropy
    
    # Gene-level entropy: gene별 평균 예측 점수의 분포
    gene_scores = {g: [] for g in gene_ids}
    for pred, gene in zip(y_pred, gene_ids):
        gene_scores[gene].append(pred)
    gene_means = np.array([np.mean(v) for v in gene_scores.values()])
    
    H_gene = entropy(np.histogram(gene_means, bins=50, density=True)[0] + 1e-8)
    H_isoform = entropy(np.histogram(y_pred, bins=50, density=True)[0] + 1e-8)
    
    return 1.0 - H_isoform / H_gene
```

**비교 계획:** v6h vs v6d bias_score per GO term → v6d가 낮으면 gene-bias 감소 확인.

### 9.2 Intra-Gene Discrimination Score

```python
def compute_intragene_disc(y_pred, y_true, gene_ids):
    """
    같은 gene 내 isoform 중 positive/negative가 함께 있는 경우:
    positive의 예측 점수 - negative의 예측 점수 (클수록 잘 구분)
    """
    scores = []
    for gene in np.unique(gene_ids):
        mask = gene_ids == gene
        gene_pred = y_pred[mask]
        gene_true = y_true[mask]
        
        if gene_true.sum() == 0 or (gene_true == 0).sum() == 0:
            continue  # positive/negative 모두 있는 gene만
        
        pos_mean = gene_pred[gene_true == 1].mean()
        neg_mean = gene_pred[gene_true == 0].mean()
        scores.append(pos_mean - neg_mean)
    
    return np.mean(scores), np.std(scores)
```

**기대:** v6d intra-gene disc > v6h → DomainDelta가 같은 gene 내 이소폼 구분에 기여.

### 9.3 Cross-Gene Generalization Score

```python
def compute_crossgene_generalization(embeddings, y_true, gene_ids):
    """
    다른 gene이지만 같은 GO function을 가진 isoform 쌍의 임베딩 유사도
    (높을수록 gene boundary를 넘어 기능 학습)
    """
    pos_emb = embeddings[y_true == 1]
    pos_genes = gene_ids[y_true == 1]
    
    scores = []
    for i in range(len(pos_emb)):
        for j in range(i+1, min(i+50, len(pos_emb))):
            if pos_genes[i] != pos_genes[j]:  # different gene
                sim = np.dot(pos_emb[i], pos_emb[j])
                scores.append(sim)
    
    return np.mean(scores)
```

**비교:** v6d cross-gene sim > v6h → ESM-2 triplet 학습이 gene-agnostic representation 구성.

---

## 10. 우선순위별 실험 계획

### Tier 1: 논문 투고 최소 조건 (즉시 실행 가능)

| 우선순위 | 실험 | 목적 | 예상 소요 |
|---------|------|------|---------|
| P1 | `no_dd` (5 GO term × 3 seed) | DomainDelta 기여 분리 | 6시간 |
| P1 | `no_domain` (5 GO term × 3 seed) | Domain LSTM 기여 분리 | 6시간 |
| P1 | Gene-bias score v6h vs v6d | 주장의 직접 증거 | 30분 |
| P1 | Intra-gene discrimination v6h vs v6d | 이소폼 구분 능력 측정 | 30분 |
| P2 | `dd_raw_domain` (5 GO term) | Sign transform 필요성 | 6시간 |
| P2 | `no_triplet` (5 GO term) | Triplet loss 기여 | 6시간 |
| P2 | Bootstrap CI (n=1000) for all metrics | 통계적 유의성 | 1시간 |

### Tier 2: Stronger Claim (2주 내)

| 우선순위 | 실험 | 목적 |
|---------|------|------|
| P3 | `intra_gene_neg` | Cross-gene negative의 gene-bias 방지 효과 |
| P3 | `dd_canonical_mane` | Canonical 정의 민감도 |
| P3 | `esm_phase2_unfreeze` | Phase 2 ESM-2 동결 필요성 재확인 |
| P4 | `no_esm` | ESM-2 단독 기여 |
| P4 | `no_focal` | Focal loss 기여 |
| P4 | `no_labelprop` | Phase 3 기여 |
| P5 | 5 GO term → 10 GO term (housekeeping 포함) | 일반화 검증 |

### Tier 3: Nature Methods Full Acceptance

| 실험 | 목적 |
|------|------|
| Novel isoform 전용 평가 (ESM-2 mask=0 subset) | DomainDelta가 novel isoform에서 유리한가? |
| Multi-tissue validation (brain, liver GO terms) | Muscle 이외 tissue 일반화 |
| Canonical 정의 ablation (maxdomain vs longest vs MANE) | 최적 canonical 정의 |
| k-NN retrieval baseline | DomainDelta 복잡성의 필요성 |
| Modality gradient norm 분석 | Modality 기여 imbalance 진단 |

---

## 11. 분석 스크립트 설계

### 11.1 Gene-Bias 분석 스크립트

```python
# results_isoform/gene_bias_analysis.py
"""
Usage: python gene_bias_analysis.py --go GO:0003774 --models v6h v6d
"""
import numpy as np
import os

def load_model_results(model_tag, go_term):
    safe_go = go_term.replace(':', '_')
    result_dir = f"../results_isoform/{safe_go}/{model_tag}_integrated_*/"
    scores_file = f"{result_dir}/*_phase2*_scores.txt"
    # gene_id, iso_id, score 로드
    ...

def gene_bias_score(scores, y_true, gene_ids): ...
def intragene_disc(scores, y_true, gene_ids): ...
def crossgene_generalization(embeddings, y_true, gene_ids): ...

if __name__ == '__main__':
    for go_term in ['GO:0006941', 'GO:0003774', 'GO:0006096', 'GO:0007204', 'GO:0030017']:
        for model in ['v6h', 'v6d']:
            results = load_model_results(model, go_term)
            bias = gene_bias_score(results.scores, results.y_true, results.gene_ids)
            disc = intragene_disc(results.scores, results.y_true, results.gene_ids)
            print(f"{model} {go_term}: bias={bias:.3f} disc={disc:.3f}")
```

### 11.2 Ablation Runner 스크립트

```python
# model/run_ablation.py
"""
Usage: python run_ablation.py --ablation no_dd --go_list v6_go_list.txt --n_seeds 3
"""
ABLATION_CONFIGS = {
    'no_dd':          {'dd_input': 'zeros'},
    'no_domain':      {'domain_input': 'zeros'},
    'no_esm':         {'esm_mask': 'zeros'},
    'dd_raw_domain':  {'dd_transform': 'raw'},
    'dd_binary':      {'dd_transform': 'binary'},
    'no_triplet':     {'skip_phase1': True},
    'no_focal':       {'loss': 'bce'},
    'no_labelprop':   {'skip_phase3': True},
    'intra_gene_neg': {'negative_sampling': 'intra_gene'},
}
```

---

## 12. 논문 기여 프레임 수정

devils-advocate 지적을 반영한 논문 주장 재정의:

### 수정 전 (취약)
> "DomainDelta overcomes gene-level reference dominance"

### 수정 후 (방어 가능)
> "Canonical-relative Pfam domain architecture delta (DomainDelta) captures isoform-specific functional divergence, improving rare isoform GO prediction across all tested GO categories"

**필요한 뒷받침:**
1. Intra-gene discrimination score v6d > v6h (DomainDelta가 같은 gene 내 이소폼을 더 잘 구분)
2. `no_dd` ablation AUPRC 하락 → DomainDelta의 독립 기여
3. `dd_canonical_mane` vs `dd_canonical_maxdomain` → canonical 정의에 따른 성능 변화 (생물학적 해석 가능)
4. Novel isoform subset에서 v6d > v6h → ESM-2 미적용 케이스에서 DomainDelta의 보완 역할

---

*레포트 작성: 2026-04-13*  
*모델 버전: v6d_integrated_full_model.py (margin_sat early stop 제거 후)*  
*다음 단계: Tier 1 ablation 우선 실행 (no_dd, no_domain, gene_bias_analysis)*
