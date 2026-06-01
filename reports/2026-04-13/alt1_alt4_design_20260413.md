# Alt 1 + Alt 4 설계 및 전처리 레포트
**작성일**: 2026-04-13  
**목적**: 동일 유전자 내 아이소폼 기능 분리를 위한 추가 특징 설계

---

## 배경: Within-gene Isoform 분리 문제

### 문제 정의
GO label이 gene-level → 같은 유전자의 모든 isoform이 동일 label.  
→ Within-gene hard negative mining: 불가 (0% 다른-label 페어)  
→ 모델이 isoform 특이적 패턴을 학습할 인센티브 부재.

### 5가지 대안 (전 세션 설계)

| 대안 | 방법 | 상태 |
|------|------|------|
| **Alt 1** | Domain Delta (isoform - canonical) | **전처리 완료** |
| Alt 2 | IRM (Invariant Risk Minimization) | 미정 |
| Alt 3 | Adversarial gene discriminator | 미정 |
| **Alt 4** | Splicing Pattern (Exon matrix) | **전처리 완료** |
| Alt 5 | DTU expression pattern | 보조 분석 (학습 제외) |

---

## Alternative 1: Domain Delta

### 수학적 정의
```
domain_delta[i] = domain_matrix[i] - domain_matrix[canonical_i]
```

**canonical 선택**: 유전자당 보유 도메인 수 (non-zero) 최대 isoform.

### 생물학적 근거

단백질 도메인(Pfam)은 기능의 직접 표현자:
- **EF-hand (PF00036)**: Ca2+ 결합 → GO:0007204 (Ca2+ signal)
- **Myosin head (PF00063)**: 모터 활동 → GO:0003774
- **Actin-binding (PF00143)**: 근절 조직화 → GO:0030017

alternative splicing으로 특정 domain이 포함/배제되면 기능이 변함.  
`domain_delta > 0`: canonical 대비 extra domain 보유  
`domain_delta < 0`: canonical 대비 domain 결여

### 전처리 결과

```
파일: hMuscle/results_isoform/features/domain_delta.npy
형태: (36748, 251) float32

통계:
  총 isoform:       36,748 (36,748 매핑, 0 누락)
  유전자 수:        12,709
  Non-zero delta:   22,937 (62.4%)  ← 유의미한 도메인 차이 존재
  Zero delta:       13,811 (canonical이거나 도메인 없음)

샘플:
  BambuTx10  | gene=ENSG00000204859.13 | Δ domain=0 (canonical)
  BambuTx100 | gene=ENSG00000162664.17 | Δ domain=4 (extra domains)
  BambuTx1001| gene=ENSG00000066827.16 | Δ domain=2 (extra domains)
```

### v6d 모델 통합 계획

```python
# 현재 domain branch (v6c)
domain_feat = LSTM(16)(Embedding(...)(domain_input))  # [16]

# v6d 추가: domain_delta branch
delta_input = Input(shape=(251,), dtype='float32', name='delta_input')
delta_feat  = Dense(32, activation='relu')(delta_input)
delta_feat  = Dense(16, activation='relu')(delta_feat)  # [16]

# Fusion 변경
concat = concatenate([f_esm_r2(64), f_cnn_r2(32), domain_feat(16), delta_feat(16)])
# EMB_DIM: 112 → 128
```

---

## Alternative 4: Splicing Pattern (Exon Matrix)

### 수학적 정의
```
exon_matrix[i, j]    = 1  if exon_j ∈ isoform_i's exons
                       = 0  otherwise
splicing_delta[i, j] = exon_matrix[i, j] - exon_matrix[canonical_i, j]
```

**exon_j**: 유전자의 j번째 exon (start 기준 정렬)  
**MAX_EXONS=50**: 유전자당 분산 최대 exon 50개 선택

### 생물학적 근거

Alternative splicing의 주요 기전:
- **Exon skipping**: 특정 exon 배제 → 기능 도메인 손실
- **Mutually exclusive exons**: 두 exon 중 하나만 포함 → 기능 전환
- **Alternative 5'/3' splice sites**: exon 길이 변화

`splicing_delta > 0`: canonical 대비 exon 추가 포함 (novel function 가능성)  
`splicing_delta < 0`: canonical 대비 exon 배제 (domain loss)

### 전처리 결과

```
GTF 파싱:
  파일: hMuscle/data/cleaned_annotations.gtf (256만 lines)
  총 transcript: 389,853 (BambuTx 4,194 + ENST 385,659)
  총 유전자:     79,633

Exon Universe:
  유전자당 exon: min=1, median=3, p95=49 → MAX_EXONS=50 적절
  MAX_EXONS 초과 시: isoform 간 분산 최대 exon 상위 50 선택

파일:
  exon_matrix.npy:    (36748, 50) int8
  splicing_delta.npy: (36748, 50) float32

통계:
  Exon 보유 isoform: 36,634/36,748 (99.7%)
  Non-zero delta:    13,706 (37.3%)
  Mean |delta|:      (splicing_stats.txt 참조)

샘플:
  BambuTx10  | exons=11 | Δ exons=0 (canonical)
  BambuTx100 | exons=11 | Δ exons=5
  BambuTx1001| exons=7  | Δ exons=13
```

### v6d 모델 통합 계획

```python
# splice branch
splice_input = Input(shape=(50,), dtype='float32', name='splice_input')
splice_feat  = Dense(32, activation='relu')(splice_input)
splice_feat  = Dense(16, activation='relu')(splice_feat)  # [16]

# Final fusion (v6d)
concat = concatenate([
    f_esm_r2(64),     # ESM-2 branch
    f_cnn_r2(32),     # CNN branch
    domain_feat(16),  # Pfam domain
    delta_feat(16),   # Alt 1: domain delta
    splice_feat(16),  # Alt 4: splicing pattern
])  # 총 144-dim
```

---

## 전처리 파일 목록

```
hMuscle/results_isoform/features/
├── domain_delta.npy         (36748, 251) float32  — Alt 1
├── iso_gene_map.txt         isoform_idx | isoform_id | gene_id
├── canonical_map.txt        gene_id | canonical_idx | canonical_id
└── splicing/
    ├── exon_matrix.npy      (36748, 50) int8       — Alt 4
    ├── splicing_delta.npy   (36748, 50) float32    — Alt 4 delta
    ├── exon_meta.npz        gene별 exon 메타
    └── splicing_stats.txt   요약 통계
```

---

## 도입 결정 기준 (v6c 결과 확인 후)

| 지표 | 임계값 | 도입 대안 |
|------|--------|---------|
| 같은 유전자 isoform 간 AUPRC 분산 | std > 0.1 | Alt 1 우선 |
| GO:0007204 / GO:0003774 AUPRC 저조 | < v6h | CNN이 domain 못 잡음 → Alt 1 |
| FP 비율 중 같은 gene 비율 | > 70% | → Alt 4 (splicing으로 분리) |
| v6c와 v6h 간 Δ AUPRC | 음수 (하락) | Alt 1+4 동시 도입 검토 |

---

## Alternative 5 (DTU Expression): 보조 분석 전략

학습에 포함하지 않고, 최종 결과 분석 시 활용:
- `counts_transcript.txt` (발현량)로 isoform별 발현 수준 계산
- 고발현 isoform vs 저발현 isoform의 예측 정확도 비교
- Novel isoform (BambuTx) 중 DTU 패턴 관련 정보 제공

---

*레포트 작성: 2026-04-13 | 전처리 스크립트: hMuscle/preprocessing/build_domain_delta.py, build_exon_matrix.py*
