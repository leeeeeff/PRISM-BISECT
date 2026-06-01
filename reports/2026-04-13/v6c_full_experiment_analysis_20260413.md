# v6c 전체 실험 분석 보고서
Date: 2026-04-13

## 1. 실험 목적
v6h 기반 Multi-scale CNN (k=5,7,11) 도입 → isoform 기능 예측 성능 향상 시도

---

## 2. 실험 이력 (GO:0006941 기준)

### 2.1 v6c 수정 이력

| 버전 | 핵심 변경 | Phase1 margin_sat | Phase2 AUPRC | 상태 |
|------|-----------|-------------------|-------------|------|
| v6c original | Bidirectional Gating ×2 + Phase1a | ~0% | 0.055 | 실패 |
| v6c-fix1 | ESM-2 Dense frozen 해제 | 0.4% | 0.056 | 실패 |
| v6c-fix2 | ESM-2 Dense Phase1 trainable | 0.4% | 0.1158 | 부분 개선 |
| v6c-fix3 | + Domain LSTM Phase1 trainable | 14.6% | 0.0641 | 역행 |
| v6c-nogating | Bidirectional Gating 제거 | 11.1% | 0.1014 | 부분 개선 |
| v6c-nop1a | + Phase1a 완전 제거 | **64.0%** | **0.3146** | GO:0006941 WIN |
| v6h (reference) | 단순 CNN, Phase1a 없음 | 60.5% | 0.2854 | baseline |

### 2.2 핵심 발견 1: Bidirectional Gating의 Phase2 오염

**문제**: fix3에서 margin_sat=14.6%로 좋은 Phase1 임베딩을 만들었음에도
Phase2 AUPRC가 0.0641로 fix2(0.1158)보다 낮음

**원인 (Bidirectional Gating Paradox)**:
```
Phase2에서 CNN frozen→unfrozen 시:
f_esm_r2 = f_esm_r1 × sigmoid(Dense(random_CNN)) + f_esm_r1
→ 잘 훈련된 ESM-2 표현이 random CNN feature로 contamination
→ 좋은 Phase1 임베딩이 Phase2에서 파괴됨
```

**해결**: Gating 블록 완전 제거 → direct concat

---

### 2.3 핵심 발견 2: Phase1a CNN Warmup의 역설적 방해 효과

**문제**: Phase1a를 포함한 v6c-nogating (Phase1 margin_sat=11.1%) vs
Phase1a 없는 v6h (margin_sat=60.5%)의 5배 이상 차이

**원인 분석**:
```
Phase1a: CNN only focal warmup (2 epoch)
→ CNN이 GO function을 어느 정도 인코딩한 상태로 Phase1 진입
→ concat([random_ESM2(64) + learned_CNN(32) + random_domain(16)])
→ CNN signal이 triplet gradient를 희석
→ ESM-2가 강한 학습 신호를 받지 못함

v6h (Phase1a 없음):
→ CNN complete random init
→ 전체 임베딩 공간이 분산
→ ESM-2+Domain이 강한 triplet gradient 수신
→ margin_sat 60.5% → Phase2 AUPRC 0.2854
```

**검증 근거**:
- Phase1 warmup active ratio: v6c 0.2% vs v6h 78.9%
- Phase1a 제거 후: v6c-nop1a warmup active ratio 79.6% (v6h와 동일)
- Phase1 final margin_sat: v6c-nop1a 64.0% > v6h 60.5% (Multi-scale CNN 추가 기여)

**교훈 [ANTI-PATTERN]**:
> Phase1에서 일부 branch를 사전 학습하면 해당 branch가 triplet signal을 선점하여
> 다른 branch(ESM-2)의 학습을 방해할 수 있다.
> 특히 triplet loss는 전체 concat 임베딩 공간에서 계산되므로,
> 한 branch가 dominant해지면 gradient가 해당 branch로만 흐름.

---

## 3. Multi-scale CNN 전체 성능 평가 (5개 GO term)

### 3.1 결과 테이블

| GO term | 생물학적 기능 | v6h AUPRC | v6c-nop1a AUPRC | Δ AUPRC |
|---------|-------------|-----------|-----------------|---------|
| GO:0006941 | Skeletal muscle contraction | 0.2854 | 0.3146 | **+10.2%** |
| GO:0007204 | Ca2+ signaling | 0.3104 | 0.3201 | **+3.1%** |
| GO:0030017 | Actin filament organization | 0.2798 | 0.2286 | **-18.3%** |
| GO:0003774 | Motor activity | 0.5975 | 0.5364 | **-10.2%** |
| GO:0006096 | Glycolysis | 0.7954 | 0.7681 | **-3.4%** |
| **Macro 평균** | | **0.4537** | **0.4336** | **-4.4%** |

Phase1 margin_sat (모두 60%+):
- GO:0006941: 64.0% | GO:0007204: 73.9% | GO:0030017: 70.6%
- GO:0003774: 67.1% | GO:0006096: 78.9%

### 3.2 패턴 분석

**Multi-scale CNN이 도움이 된 GO term** (GO:0006941, GO:0007204):
- 근육 수축 / Ca2+ 신호전달
- Transmembrane domain, binding site 등 특정 sequence motif와 밀접 연관
- Multi-scale CNN(k=5,7,11)이 다양한 길이의 motif를 동시 포착 → 유리

**Multi-scale CNN이 방해된 GO term** (GO:0030017, GO:0003774, GO:0006096):
- Actin organization / Motor activity / Glycolysis
- 이 기능들은 서열보다 단백질 상호작용, 위치, 도메인 구성에 더 의존
- Multi-scale CNN이 오히려 ESM-2 표현을 방해했을 가능성

**핵심 결론**: Multi-scale CNN은 GO term specific improvement이며 universal enhancement 아님.
Macro-AUPRC 기준 v6h가 여전히 best model.

---

## 4. 방향 재설정: v6d 설계

### 4.1 근거

1. v6h가 현재 best model → 이 구조를 유지
2. Multi-scale CNN 방향 포기 → 단순 CNN (v6h 방식) 유지
3. 성능 개선 방향: **이소폼-특이적 특징 추가** (Alt 1 + Alt 4)

### 4.2 v6d 설계 (v6h 기반)

**추가 입력**:
- `domain_delta[251]`: 이소폼의 Pfam 도메인 구성 vs canonical 차이
- `splicing_delta[50]`: exon inclusion 패턴 vs canonical 차이

**전처리 (domain_delta 정규화)**:
```python
# 값 범위: [-16125, 16090] → 정규화 필요
# 옵션 1: sign transform → {-1, 0, 1} (gain/loss 이진화)
dd_sign = np.sign(domain_delta)
# 옵션 2: log transform
dd_log = np.sign(domain_delta) * np.log1p(np.abs(domain_delta))
```

**모델 구조 (추가분)**:
```
domain_delta[251] → Dense(64,relu) → Dropout(0.2) → Dense(16,relu) → dd_feat[16]
splicing_delta[50] → Dense(32,relu) → Dropout(0.2) → Dense(8,relu)  → spl_feat[8]

concat: [esm2_gated(64) + cnn_feat(32) + domain_feat(16) + dd_feat(16) + spl_feat(8)]
      = 136 (v6h의 112 + 24)
```

**Phase 설계**:
- Phase 1 (triplet): ESM-2 + Domain LSTM + dd_branch + spl_branch trainable, CNN frozen
- Phase 2 (focal): CNN trainable, ESM-2 frozen

**기대 효과**:
- domain_delta: canonical과 다른 Pfam 구성을 가진 이소폼 구분 (Alt 1)
- splicing_delta: exon 포함 패턴 차이를 통한 이소폼-특이적 기능 예측 (Alt 4)

**NMI 기여**:
> "Multi-modal isoform-intrinsic feature fusion: domain composition delta
> and splicing pattern delta against canonical isoform as additional
> isoform-discriminating signals" → Gene-level reference dominance 극복

### 4.3 검증 계획

1. GO:0006941로 v6d 우선 검증 (v6h AUPRC=0.2854 초과 목표)
2. 성공 시 5개 GO term 전체 실행
3. bootstrap CI [R9.4] 계산
4. Ablation: no_domain_delta / no_splice / no_both

---

## 5. 의사결정 트리

```
v6c 실험 결론
├── Multi-scale CNN: GO-specific, macro -4.4% → 포기
├── Phase1a 제거: 원칙 확립 → 이후 모든 버전 적용
├── Gating 제거: 원칙 확립 → 이후 모든 버전 적용
└── 다음: v6d = v6h + domain_delta + splice_feat
```

---

## 6. 데이터 준비 상태

| 파일 | 경로 | 상태 |
|------|------|------|
| domain_delta.npy | results_isoform/features/domain_delta.npy | ✅ (36748, 251) |
| splicing_delta.npy | results_isoform/features/splicing/splicing_delta.npy | ✅ (36748, 50) |
| iso_gene_map.txt | results_isoform/features/iso_gene_map.txt | ✅ |
| canonical_map.txt | results_isoform/features/canonical_map.txt | ✅ |

정규화 방향: `sign(domain_delta)` 또는 `sign * log1p(|domain_delta|)` — 검증 후 결정
