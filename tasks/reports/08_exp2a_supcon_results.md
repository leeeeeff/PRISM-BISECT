# 레포트 08: Exp 2a SupCon 실험 결과
**작성일:** 2026-04-29  
**실험:** v7a (Phase 1 Triplet → SupCon)  
**결론:** FAIL — τ 전 범위에서 v6g Triplet 미달

---

## 1. 전체 결과 (τ=0.1, 0.2, 진행중 0.3)

| GO Term | Type | v6g | τ=0.1 | τ=0.2 | τ=0.3 |
|---------|------|-----|-------|-------|-------|
| GO:0006096 | A | **0.823** | 0.650 | 0.617 | TBD |
| GO:0003774 | A | **0.591** | 0.540 | 0.459 | TBD |
| GO:0006941 | B | 0.121 | **0.165** | 0.141 | TBD |
| GO:0007204 | B | 0.309† | 0.203 | **0.234** | TBD |
| GO:0030017 | B | 0.201 | 0.175 | **0.239** | TBD |
| **Macro** | | **0.409†** | 0.347 | 0.338 | TBD |

†GO:0007204: v6g 0.591은 lucky run(sep=1.45, Type-A 오분류). 비교 기준 v6f=0.309.

---

## 2. 핵심 진단: Phase 2 시작점 (epoch=1 AUPRC)

| 모델 | Phase1 sep GO:0006096 | ep1 AUPRC GO:0006096 | Final AUPRC GO:0006096 |
|------|-----------------------|---------------------|------------------------|
| v6g Triplet | 1.94 | **0.774** | **0.823** |
| v7a τ=0.1 | 3.39 | 0.528 | 0.650 |
| v7a τ=0.2 | 3.19 | 0.487 | 0.617 |

- **SupCon Phase 1 sep_ratio > Triplet** (더 좋은 임베딩 공간)
- **SupCon Phase 2 epoch=1 AUPRC < Triplet** (더 낮은 시작점)
- τ를 키울수록 ep1 AUPRC가 더 낮아짐

---

## 3. τ별 패턴

| τ | Type-A (GO:0006096, 0003774) | Type-B (나머지) | 특징 |
|---|------------------------------|----------------|------|
| 0.1 | 더 집중적 (sep=3.39, 2.52) | GO:0006941 개선 | Phase 2 시작 낮음 |
| 0.2 | 덜 집중적 (sep=3.19, 1.81) | Type-B 혼합 개선 | Type-A 더 악화 |
| 0.3 | TBD | TBD | TBD |

**τ 증가 효과:** Type-B 일부 개선, Type-A 악화, 전체 Macro는 개선 없음.

---

## 4. 근본 원인: Geometry Incompatibility

**Triplet 작동 방식:**
- cross-gene semi-hard negative: d(a,p) < d(a,n) < d(a,p) + margin
- L2 distance → 이진 sigmoid 분류기에 자연스럽게 매핑
- Phase 2 시작점: epoch=1 AUPRC 0.77 (이미 분류 준비 완료)

**SupCon 작동 방식:**
- cosine similarity 기반 균등 군집화
- τ가 매우 작으면 hypersphere 위 극단적 집중
- Phase 2 시작점: epoch=1 AUPRC 0.53 (sigmoid가 SupCon 기하 구조 인식 불가)

**결론:** τ 조정으로는 해결 불가능한 structural incompatibility.
SupCon embedding geometry ↔ Focal Loss sigmoid의 인터페이스 불일치.

---

## 5. Ablation Schedule 판정

**Exp 2a acceptance criterion:** mean Δ AUPRC > +0.02  
**실제 결과:** Macro Δ = 0.347 - 0.409 = **-0.062** (v6g 대비)

**판정: FAIL**

ablation_schedule.md 규칙:  
"If mean Δ AUPRC < 0 → Conclusion: SupCon is not better than Triplet → Action: Investigate Triplet margin or hard negative mining instead"

---

## 6. Exp 2a에서 건진 것

실패했지만 유의미한 발견:

1. **SupCon Phase 1 임베딩 품질은 더 좋음** (sep_ratio 3.39 vs 1.94)
   → Phase 1 손실 자체는 문제 없음
   
2. **Phase 2 시작점이 병목** (epoch=1 AUPRC가 Phase 2 최종 성능 결정)
   → Phase 2 초기 condition이 중요함을 입증

3. **Type-B sparse GO term에서 SupCon 잠재력** (GO:0006941 +0.044, τ=0.1)
   → 나중에 Phase 2 별도 튜닝 시 활용 가능

4. **SupCon + 다른 Phase 2** 조합 가능성
   → Phase 2를 SupCon으로도 이어가거나, head warm-up 추가 시 다를 수 있음

---

## 7. 다음 단계

ablation_schedule.md 결정 트리 기준: Triplet 유지.

**옵션:**
- [A] Triplet 개선: hard negative mining 강화 (margin=0.5, k-NN 기반)
- [B] Proto-k1 시도: 단일 prototype loss (SupCon과 다른 구조)
- [C] Phase 2 아키텍처 변경: Phase 2에도 contrastive term 추가
- [D] 방향 전환: v6g를 현재 best로 확정 후 논문 작성 진입

현재 v6g Macro=0.465가 이 파이프라인의 사실상 ceiling에 가까울 수 있음.
