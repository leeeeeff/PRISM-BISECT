# DIFFUSE 프로젝트 진행 레포트
**작성일**: 2026-04-13  
**작성 범위**: v6 계열 아키텍처 설계, v6c 구현 및 실행, Alt 1/4 전처리 완료

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [모델 버전 히스토리 및 성능 비교](#2-모델-버전-히스토리-및-성능-비교)
3. [v6h 분석 및 문제 진단](#3-v6h-분석-및-문제-진단)
4. [v6c 아키텍처 설계](#4-v6c-아키텍처-설계)
5. [v6c 구현 세부사항](#5-v6c-구현-세부사항)
6. [v6c 실행 현황 및 초기 결과](#6-v6c-실행-현황-및-초기-결과)
7. [Alt 1/4 전처리 완료](#7-alt-14-전처리-완료)
8. [핵심 이슈 및 버그 수정](#8-핵심-이슈-및-버그-수정)
9. [다음 단계 계획](#9-다음-단계-계획)

---

## 1. 프로젝트 개요

### 연구 목표
BambuTx 장기 읽기 single-cell 시퀀싱에서 검출된 신규 아이소폼(36,748개)에 대한 GO 기능 예측.

### 핵심 해결 과제
| 과제 | 설명 | 현재 접근 |
|------|------|-----------|
| **Data Sparsity / Mode Collapse** | 양성 샘플 극소수 (0.1~2%) | Focal Loss + Triplet Loss + AUPRC early stop |
| **Gene-level Reference Dominance [R2.1]** | 모델이 isoform 대신 gene 수준 패턴 학습 | Cross-gene triplet negative + isoform-specific features |

### 대상 GO Term (5개)
| GO Term | 기능 | 양성 비율(테스트) |
|---------|------|----------------|
| GO:0006941 | 골격근 수축 (skeletal muscle contraction) | 0.69% |
| GO:0007204 | 세포내 Ca2+ 신호 (calcium signaling) | - |
| GO:0030017 | 근절 조직화 (sarcomere organization) | - |
| GO:0003774 | 모터 활동 (motor activity) | 0.69% |
| GO:0006096 | 해당과정 (glycolysis) | - |

---

## 2. 모델 버전 히스토리 및 성능 비교

### AUPRC 비교표 (5개 GO Term)

| 버전 | GO:0006941 | GO:0007204 | GO:0030017 | GO:0003774 | GO:0006096 | 비고 |
|------|-----------|-----------|-----------|-----------|-----------|------|
| **v5-5** | 0.0947 | 0.5075 | 0.3904 | 0.2045 | 0.6698 | CNN+LSTM, baseline |
| **v6h** | **0.2854** | **0.3104** | **0.2798** | **0.5975** | **0.7954** | Modality-Role Separation |
| **v6c** | 0.0585† | 실행 중 | 실행 중 | 0.0183† | 실행 중 | Multi-CNN + Bidirectional Gating |

† 현재까지 확인된 값 (2026-04-13 기준, 미완료)

### 버전별 핵심 아키텍처

```
v4-3  PFN + Focal Loss + Triplet Loss (기본)
v5-5  v4-3 + CNN(k=3) + LSTM(32) + Gradient Clip
v6h   v5-5 + ESM-2 Phase 1 전담 / CNN Phase 2 전담 (Modality-Role Separation)
v6c   v6h + Multi-scale CNN(k=5,7,11) + Bidirectional Gating ×2
v6d   (예정) v6c + Domain Delta + Splicing Pattern
```

---

## 3. v6h 분석 및 문제 진단

### v6h 설계 의도
ESM-2가 global protein context를, CNN이 local sequence motif를 담당하는 역할 분리 전략.

### 발견된 문제: Phase 2 조기 종료
**진단**: v6h Phase 2에서 AUPRC가 2~3 epoch 이후 급격히 하락.

**근본 원인 분석**:
1. **CNN Cold-start**: v6h는 Phase 1에서 CNN 미사용 → Phase 2에서 CNN이 랜덤 초기화 상태로 시작
2. **alpha=0.10 score collapse**: Focal loss의 alpha 값이 너무 낮아 예측 점수가 0 근처로 몰림
3. **patience=3**: 너무 짧아 CNN이 수렴하기 전에 early stop 발동

**v6c 대응책**:
- Phase 1a CNN warmup 추가 (2 epoch focal, CNN only)
- alpha: 0.10 → 0.25
- patience: 3 → 7
- ESM-2 ↔ CNN Bidirectional Gating으로 Phase 1 공동 학습

### v5-5 vs v6h 역전 패턴 분석
```
GO:0007204 (EF-hand, Ca2+): v5-5=0.5075 > v6h=0.3104  ← v6h에서 악화
GO:0030017 (sarcomere):     v5-5=0.3904 > v6h=0.2798  ← v6h에서 악화
GO:0006941 (근수축):        v5-5=0.0947 < v6h=0.2854  ← v6h에서 개선
GO:0003774 (motor):         v5-5=0.2045 < v6h=0.5975  ← v6h에서 대폭 개선
```

**해석**: ESM-2 mean-pool은 positional 정보 손실 → EF-hand, sarcomere 같은 domain-specific motif 인식 실패. CNN이 이를 보완해야 하나 v6h는 Phase 분리로 상호작용 부재.

---

## 4. v6c 아키텍처 설계

### 설계 원칙
AlphaFold2 Evoformer 참조: 두 modality (ESM-2, CNN)가 독립 처리하되 **상호 정보 교환 (bidirectional gating)**.

### 아키텍처 다이어그램

```
입력 레이어
├── ESM-2 embedding (640) → Dense(256) → Dense(128) → Dense(64) [esm2_gated]
├── Sequence (1500, 3-gram k-mer)
│   └── Embedding(8001,32) → Conv1D(k=5) ┐
│                           → Conv1D(k=7) ├→ concat[96] → Conv1D(3)[64]
│                           → Conv1D(k=11)┘            → GlobalMaxPool → Dense(32) [cnn_feat]
└── Domain (251) → Embedding → LSTM(16) [domain_feat]

Bidirectional Gating (×2 rounds)
  Round 1:
    gate_e2c_r1 = sigmoid(Dense(32)(esm2_gated))
    f_cnn_r1   = cnn_feat  * gate_e2c_r1 + cnn_feat   [잔차]
    gate_c2e_r1 = sigmoid(Dense(64)(f_cnn_r1))
    f_esm_r1   = esm2_gated * gate_c2e_r1 + esm2_gated [잔차]
  Round 2: (동일 구조, f_esm_r1, f_cnn_r1 입력)

Fusion: concat[f_esm_r2(64) + f_cnn_r2(32) + domain_feat(16)] = 112-dim
→ Dense(48, relu) → L2_norm → Dense(1, sigmoid)
```

### 학습 단계 (5-Phase)

| Phase | 대상 레이어 | Loss | 목적 |
|-------|------------|------|------|
| **Phase 0** | (평가만) | - | 미학습 베이스라인 |
| **Phase 1a** | CNN 7개 레이어만 | Focal (α=0.25) | CNN cold-start 방지 |
| **Phase 1** | 전체 (Gradient Scaling 적용) | Triplet | ESM-2↔CNN 공동 학습 |
| **Phase 1.5** | Frozen (Head only) | Focal | Linear Probing |
| **Phase 2** | ESM-2 Frozen, CNN+Gate+Domain | Focal (patience=7, α=0.25) | 최종 분류 |
| **Phase 3** | - | - | Label Propagation |

### Gradient Scaling (Phase 1)

| 레이어 그룹 | Scale | 이유 |
|------------|-------|------|
| ESM-2 (esm2_d1/d2/feat) | ×0.2 | 사전학습 표현 보존 |
| CNN (cnn_emb/k5/k7/k11/reduce/pool/feat) | ×1.0 | 새 모듈 충분한 학습 |
| Gate (gate_e2c/c2e ×2) | ×0.5 | 안정적 gating 수렴 |

### 바이오 타당성 (bio-validator PASS 조건)

- Multi-scale CNN [k=5,7,11]: 5aa(활성부위), 7aa(모티프), 11aa(도메인 경계) 커버 → PASS
- Bidirectional gating residual: 정보 소실 방지 → PASS
- Cross-gene negative: intra-gene same-label 문제 회피 → PASS
- Per-residue ESM-2 cross-attention: 학습 데이터에 FASTA 없음 → 제외 (CONDITIONAL PASS)

---

## 5. v6c 구현 세부사항

### 파일

| 파일 | 경로 | 상태 |
|------|------|------|
| `v6c_integrated_full_model.py` | `hMuscle/model/` | 완료 (46KB) |
| `run_GPU_v6c.py` | `hMuscle/model/` | 완료 |

### 주요 상수 (v6h 대비 변경)

| 파라미터 | v6h | v6c | 변경 이유 |
|---------|-----|-----|----------|
| CNN 구조 | Conv1D(k=3,32) | Conv1D([k=5,7,11],32) → concat[96] → reduce[64] | Multi-scale motif |
| Bidirectional Gating | 없음 | 2 rounds × 2 gates | ESM-2↔CNN 상호작용 |
| Phase 1a | 없음 | 2 epochs CNN-only focal | CNN cold-start 방지 |
| Phase 2 patience | 3 | 7 | 충분한 수렴 기회 |
| Phase 2 alpha | 0.10 | 0.25 | score collapse 방지 |
| EMB_DIM | 112 | 112 (동일) | f_esm_r2(64)+f_cnn_r2(32)+domain(16) |

### 수정된 버그

1. **IndexedSlices TypeError** (Phase 1 gradient scaling):
   ```python
   # 수정 전: g * scale → IndexedSlices 연산 불가
   # 수정 후:
   if isinstance(g, tf.IndexedSlices):
       g = tf.convert_to_tensor(g)
   g_safe = tf.where(tf.math.is_finite(g), g * scale, tf.zeros_like(g))
   ```
   - 원인: `Embedding` 레이어는 sparse gradient (IndexedSlices) 반환 → dense로 변환 필요

---

## 6. v6c 실행 현황 및 초기 결과

### 실행 환경
- GPU 0: GO:0006941 → GO:0007204 → GO:0030017 순
- GPU 1: GO:0003774 → GO:0006096 순
- 실행 시작: 2026-04-13 05:32 (버그 수정 후)

### Phase별 진행 확인 (GO:0006941, 수정버전)

| Phase | 결과 | 비고 |
|-------|------|------|
| Phase 0 | Silhouette=-0.0643, Linear AUROC=0.7343 | 미학습 |
| Phase 1a | CNN warmup 2 epoch 완료 | - |
| Phase 1 | margin_sat=0.4%, active_rate < 2% (4 epoch early stop) | 트리플렛 수렴 어려움 |
| Phase 1.5 | Silhouette=+0.9804, Linear AUROC=0.7690 | Head 재조정 |
| Phase 2 | AUPRC best=0.0557 (Ep1), 이후 flat | 개선 없음 |
| Phase 3 | alpha=0.5 → AUPRC=0.0585 | Label Propagation |

### 현재 알려진 v6c 결과 (2026-04-13 기준)

| GO Term | v6c AUPRC | 상태 |
|---------|-----------|------|
| GO:0006941 | **0.0585** | 완료 |
| GO:0007204 | - | 실행 중 |
| GO:0030017 | - | 실행 중 |
| GO:0003774 | 0.0183 (Phase 2 진행 중) | 미완료 |
| GO:0006096 | - | 대기 중 |

### v6c 초기 성능 이슈

**GO:0006941 기준 v6h(0.2854) 대비 v6c(0.0585) 대폭 하락**.

의심 원인:
1. **Phase 1 triplet collapse**: margin_sat=0.4% (매우 낮음) → embedding 공간이 변별력 없이 수렴
2. **ESM-2 gradient scale 0.2**: 너무 공격적인 제약으로 Phase 1에서 pretrained 표현 파괴 가능성
3. **Bidirectional gating 불안정**: gate std가 낮아지며 gate가 ~0.5 상수로 수렴 (정보 전달 미흡)
4. **Multi-scale CNN feature 충돌**: 3개 Conv1D 출력의 concat이 Phase 1a 이후에도 미분화 상태

**참고**: Phase 0 Linear AUROC=0.8033 > Phase 2 최종 AUROC=0.5224 → 학습이 ESM-2 표현을 오히려 저하시킴.

---

## 7. Alt 1/4 전처리 완료

### Alternative 1: Domain Delta

**파일**: `hMuscle/results_isoform/features/domain_delta.npy`  
**형태**: (36748, 251) float32

**계산 로직**:
```
domain_delta[i] = domain_matrix[i] - domain_matrix[canonical_gene_i]
```

**Canonical isoform 결정**: 유전자당 domain 수 (non-zero) 최대인 isoform.

**결과 요약**:
```
총 isoform:       36,748
매핑 성공:        36,748 (100%)
유전자 수:        12,709
Non-zero delta:   22,937 (62.4%)
Zero delta:       13,811 (이미 canonical이거나 도메인 없음)
```

**도입 조건**:
- v6c에서 같은 유전자 isoform 간 AUPRC 분산 > 0.1
- GO:0007204 (EF-hand), GO:0003774 (motor domain) 계열 성능 부족 시

**전처리 스크립트**: `hMuscle/preprocessing/build_domain_delta.py`

---

### Alternative 4: Splicing Pattern (Exon Matrix)

**파일**:
- `hMuscle/results_isoform/features/splicing/exon_matrix.npy` — (36748, 50) int8
- `hMuscle/results_isoform/features/splicing/splicing_delta.npy` — (36748, 50) float32

**계산 로직**:
```
exon_matrix[i, j]    = 1 if exon_j ∈ isoform_i else 0
splicing_delta[i, j] = exon_matrix[i, j] - exon_matrix[canonical_i, j]
```

**결과 요약**:
```
GTF 파싱:           2,566,186 lines → 389,853 transcripts, 79,633 genes
Exon 보유 isoform:  36,634/36,748 (99.7%)
Non-zero delta:     13,706 (37.3%)
유전자당 exon:      min=1, median=3, p95=49 → MAX_EXONS=50 설정 적절
```

**가변길이 처리 전략**: MAX_EXONS=50 초과 유전자 → isoform 간 분산 최대 exon 상위 50개 선택.

**전처리 스크립트**: `hMuscle/preprocessing/build_exon_matrix.py`

---

### 전처리 파일 목록

```
hMuscle/results_isoform/features/
├── domain_delta.npy       (36748, 251) float32 — Alt 1
├── iso_gene_map.txt       isoform_idx → isoform_id → gene_id
├── canonical_map.txt      gene_id → canonical_isoform_idx
└── splicing/
    ├── exon_matrix.npy        (36748, 50) int8   — Alt 4
    ├── splicing_delta.npy     (36748, 50) float32 — Alt 4 delta
    ├── exon_meta.npz          gene별 exon 메타
    └── splicing_stats.txt     요약 통계
```

---

## 8. 핵심 이슈 및 버그 수정

### 이슈 1: SEQ_FILE 경로 오류 (v6h)
- **증상**: "sequence matrix not found"
- **원인**: `my_sequence_matrix_fixed.npy`가 `model/` 디렉토리에 위치 (root 소유), `../data/` 경로 참조 실패
- **수정**: `SEQ_FILE = 'my_sequence_matrix_fixed.npy'` (상대 경로)

### 이슈 2: IndexedSlices × float TypeError (v6c Phase 1)
- **증상**: `TypeError: unsupported operand type(s) for *: 'IndexedSlices' and 'float'`
- **원인**: `Embedding` 레이어의 sparse gradient를 직접 float 곱셈 불가
- **수정**: gradient scaling 전 `tf.convert_to_tensor(g)` 호출

### 이슈 3: Per-residue ESM-2 Cross-attention 불가
- **원인**: 학습 데이터 (human 31,668 + swissprot 82,703)에 raw FASTA 없음 → k-mer encoded만 존재
- **결정**: cross-attention 완전 제거, bidirectional gating으로 대체

### 이슈 4: Within-gene Hard Negative 불가
- **원인**: GO label이 gene-level → 같은 유전자 내 isoform 간 label 차이 없음 (0%)
- **결정**: cross-gene negative 유지, domain_delta/splicing_delta로 isoform 분리

---

## 9. 다음 단계 계획

### 즉시 (v6c 완료 후)
1. **v6c 5개 GO term 결과 수집** — Phase 2 AUPRC 비교표 완성
2. **v6c vs v6h 대비 분석** — 성능 하락 GO term 특정, 원인 규명
3. **Phase 1 collapse 진단** — margin_sat, active_rate, gate mean/std 분석

### 단기 (v6c 분석 후)
4. **v6c 재설계 또는 하이퍼파라미터 조정** 검토:
   - ESM-2 gradient scale 0.2 → 0.5 시도
   - Phase 1a warmup 2 → 5 epoch
   - Gate round 2 → 1 (단순화)
5. **v6d 구현**: v6c + domain_delta + splicing_feat
   - `concat = [f_esm_r2(64) + f_cnn_r2(32) + domain_feat(16) + domain_delta_feat(16) + splice_feat(16)]`
   - EMB_DIM: 112 → 144

### 중기 (v6d 결과 후)
6. **No-cross-gene-negative ablation** — Main Contribution 2 방어를 위한 필수 실험
7. **Bootstrap CI (n=1000)** — 개선 주장을 위한 통계적 신뢰구간 [R9.4]
8. **Alternative 5 (DTU expression)** — 보조 분석 도구로 최종 결과와 함께 적용

---

## 부록: 참조 규칙

| 규칙 | 내용 |
|------|------|
| [R1.1] | Focal Loss γ=2, γ<1 금지 |
| [R2.1] | Gene-level 직접 concatenation 금지, attention/gating 사용 |
| [R3.1] | Triplet margin=0.3, cross-gene negative |
| [R3.2] | active_ratio < 5% → hard negative mining |
| [R9.1] | Sparse class: AUPRC primary metric |
| [R9.4] | 개선 주장: bootstrap CI (n=1000) 필수 |

---

*레포트 생성: Claude Sonnet 4.6 | 2026-04-13*
