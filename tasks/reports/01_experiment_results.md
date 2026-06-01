# 레포트 01: 실험 성능 결과 이력
**작성일:** 2026-04-29  
**기준 모델:** v6g (현재 best, Macro-AUPRC 0.465)  
**평가 지표:** AUPRC (primary, sparse class [R9.1]) · AUROC (secondary)

---

## 1. 전체 버전별 Macro-AUPRC 비교

| 버전 | Macro-AUPRC | 변화 | 핵심 변경 |
|------|-------------|------|-----------|
| v6d  | 0.458 | 기준선 | DomainDelta branch 추가 |
| v6e  | 0.463 | +0.005 | Phase 0 Type A/B 분류, LP 제거 |
| v6f  | 0.424 | -0.039 ✗ | Type-B lr=0.0002, Phase 1.5 포함 |
| **v6g** | **0.465** | **+0.041** ✓ | Phase 1.5 제거 |
| v7a  | 진행 중 | TBD | Phase 1 Triplet → SupCon |

---

## 2. GO term별 AUPRC 상세

| GO Term | v6d | v6e | v6f | v6g | 비고 |
|---------|-----|-----|-----|-----|------|
| GO:0006941 | 0.284 | 0.226 | 0.194 | 0.121 | ↓ 지속 하락, 구조적 문제 |
| GO:0007204 | 0.265 | 0.388 | 0.309 | 0.591 | ⚠️ 재현성 의심 |
| GO:0030017 | 0.276 | 0.332 | 0.273 | 0.201 | 불안정 |
| GO:0003774 | 0.671 | 0.558 | 0.561 | 0.591 | 회복 |
| GO:0006096 | 0.793 | 0.813 | 0.783 | 0.823 | 안정적 |

---

## 3. GO term별 특이사항

### GO:0006941 (Striated Muscle Contraction)
- **v6g AUPRC: 0.121** — v6 시리즈 최저
- Class imbalance: positive 0.69% (훈련셋 기준)
- Phase 0 sep_ratio: 1.033 (Type-B 중 최저)
- 패턴: Phase 2 epoch 2에서 Acc가 0.699 → 0.978로 급등 → "all negative" shortcut
- 원인: 낮은 sep_ratio + 극단적 class imbalance → Phase 2 focal loss가 수렴 불능
- **현재 판단:** 구조적 패턴, v7a Prototype Loss로 개선 가능성 탐색 중

### GO:0007204 (cytosolic calcium ion concentration)
- **v6g AUPRC: 0.591** — 역대 최고
- ⚠️ **재현성 경고:** 이 run에서 sep_ratio=1.450으로 Type-A로 분류됨
  - 보통 GO:0007204 sep_ratio ~1.07 (Type-B)
  - Type-A 경로(lr=0.0003, patience=7)로 훈련됨
  - 다음 run에서 Type-B로 분류되면 0.3 근방으로 회귀 가능
- **v7 비교 기준:** v6g 0.591이 아닌 v6f 0.309 또는 v6e 0.388 사용 권장

### GO:0006096 (Glycolysis)
- **v6g AUPRC: 0.823** — v6 시리즈 최고
- Type-A 안정적 (sep_ratio 1.17-1.66 범위)
- v6e: coverage=4.0로 하향하여 0.783 감소, v6f/v6g coverage=6.0 복구

### GO:0003774 (Motor Activity)
- **v6g AUPRC: 0.591** — v6e(0.558) 대비 회복
- Phase 1.5 제거 효과: v6e에서 sep_ratio 1.58→1.01 파괴 → v6g Phase 1.5 없애 0.591 회복

### GO:0030017 (Sarcomere Organization)
- **v6g AUPRC: 0.201** — 불안정
- v6e 0.332, v6f 0.273, v6g 0.201: 트렌드 없음
- 데이터 특성: sparse + 이질적 positive → Phase 1.5 제거 혜택 제한적

---

## 4. 실험 설계 변수 정리

| 변수 | v6d | v6e | v6f/v6g |
|------|-----|-----|---------|
| Phase 1 loss | Triplet (margin=0.3) | Triplet | Triplet (→ v7a: SupCon) |
| Phase 1.5 | Linear Probe | Linear Probe | **없음** |
| Phase 2 lr (Type-A) | 0.0003 | 0.0003 | 0.0003 |
| Phase 2 lr (Type-B) | 0.0001 | 0.0001 | **0.0002** |
| Type-B patience | 6 | 6 | **10** |
| Type-B max_epochs | 15 | 15 | **25** |
| TARGET_COVERAGE | 6.0 | 4.0 | **6.0** |
| Label Propagation | alpha=0.0-0.3 | **alpha=0.0** | alpha=0.0 |

---

## 5. Phase별 Embedding Quality (v6g 기준, 대표 예시)

v6e 관찰 기준 (v6g는 Phase 1.5 없음):

| Phase | sep_ratio | 의미 |
|-------|-----------|------|
| Ph0 (untrained) | 0.98-1.87 | GO term마다 편차 큼 |
| Ph1 (triplet) | 1.03-1.66 | Triplet이 분리도 향상 |
| Ph1.5 (linear probe) | **0.86-1.88** | 일부 GO term에서 급락 (파괴) |
| Ph2 (focal) | 최종 성능 | CNN이 isoform-specific 모티프 학습 |

Phase 1.5에서 sep_ratio 급락 GO term:
- GO:0006941: 1.25 → 0.86 (-0.39)
- GO:0003774: 1.58 → 1.01 (-0.57)

---

## 6. Label Propagation 실험 결과 (v6d 기준, 제거 결정)

5개 GO term 중 4개에서 LP 적용 시 AUPRC 감소 또는 중립:
- GO:0006096: LP alpha=0.2 → AUPRC -8.9%
- 나머지 4개: alpha=0.0 (LP 없음)이 best
- **결론:** hMuscle 24 샘플 수는 LP에 불충분 → v6e부터 LP 고정 제거

---

## 7. v7a 예상 (실행 중)

ablation_schedule.md Exp 2a 기준:
- **acceptance:** Δ Macro-AUPRC > +0.02 (≥ 0.485)
- **falsification:** Δ Macro-AUPRC < 0 → SupCon 포기, Triplet 유지
- 비교 기준: v6g Macro-AUPRC 0.465 (단, GO:0007204는 v6f 0.309 기준)
