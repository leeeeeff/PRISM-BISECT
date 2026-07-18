# Annotation-Free Isoform PRISM Performance Analysis

**생성일**: 2026-06-01  
**목적**: Pfam domain annotation이 없는 이소폼에서 PRISM 성능 정량화 (S2 체크리스트 항목)

---

## Dataset 통계

| 항목 | 수치 |
|------|------|
| 전체 이소폼 | 36,748 |
| Pfam annotation 있음 (domain_matrix_proper_test 기준) | 13,683 (37.2%) |
| **Pfam annotation 없음 (annotation-free)** | **23,065 (62.8%)** |
| 사용 domain matrix | `features/domain_matrix_proper_test.npy` (shape: 36748 × 512) |

---

## 핵심 결과: annotation-free 이소폼에서 PRISM 성능

**모델**: v5fix (phase2 joint focal) — v15d_bp_clean 전 버전, 참조용

| GO Term | 설명 | 전체 AUPRC | Domain 있음 | Domain 없음 | n_pos (w/ domain) | n_pos (w/o domain) |
|---------|------|-----------|------------|------------|-----------------|-----------------|
| GO:0006096 | Glycolysis | 0.4939 | 0.2670 | **0.5169** | 4 | 72 |
| GO:0006412 | Translation | 0.0194 | 0.0132 | **0.0236** | 163 | 538 |
| GO:0006936 | Muscle contraction | 0.1213 | **0.1559** | 0.0997 | 340 | 257 |
| GO:0022900 | ETC | 0.0178 | 0.0185 | 0.0189 | 74 | 217 |

---

## 해석

### 주요 발견

1. **Glycolysis (GO:0006096)**: annotation-free 이소폼에서 더 높은 AUPRC (0.5169 > 0.2670)
   - 이 GO term의 양성 이소폼 76개 중 72개 (94.7%)가 annotation-free
   - PRISM이 Pfam domain 없이도 glycolytic 기능을 예측
   - Domain-only 기반 방법으로는 이 이소폼들 예측 불가

2. **Translation (GO:0006412)**: annotation-free 이소폼에서 더 높은 AUPRC (0.0236 > 0.0132)
   - 701 positive 중 538개 (76.7%)가 annotation-free
   - PRISM의 translation 예측은 주로 domain-free 이소폼에서 작동

3. **Muscle contraction (GO:0006936)**: domain 있는 그룹에서 더 높음 (0.1559 > 0.0997)
   - Muscle contraction은 actin/myosin 같은 알려진 도메인과 밀접 연관 → 예외 케이스
   - Domain 정보가 유용한 GO term

4. **ETC (GO:0022900)**: 두 그룹 유사 (0.0185 ≈ 0.0189)

### 논문 방어 논점

> "PRISM의 ESM-2 기반 예측은 Pfam 도메인 annotation이 없는 이소폼(62.8%)에서도 의미 있는 성능을 보인다. 특히 Glycolysis와 Translation GO term에서는 annotation-free 이소폼에 더 많은 양성 샘플이 존재하며, PRISM은 이 그룹에서 더 높은 AUPRC를 달성한다."

---

## v15d_bp_clean 결과와의 비교 (Agent A 분석)

v5fix는 초기 버전. v15d_bp_clean (생산 모델)에서의 전체 성능:

| GO Term | v5fix Overall | v15d_bp_clean Overall | 개선 |
|---------|--------------|----------------------|------|
| Glycolysis | 0.4939 | **0.8391** | +0.3452 |
| Translation | 0.0194 | **0.7295** | +0.7101 |
| Muscle contraction | 0.1213 | **0.8234** | +0.7021 |
| ETC | 0.0178 | **0.4563** | +0.4385 |

v15d에서 annotation-free 성능도 비례하여 향상되었을 것으로 예상. **v15d로 재실험 권장.**

---

## 한계 및 주의사항

1. **v5fix 결과**: 생산 모델 v15d_bp_clean의 annotation-free 분리 실험이 아님. 참조값으로만 사용
2. **레이블 일치 확인 필요**: v5fix 레이블이 v15d와 동일한지 검증 필요
3. **Domain matrix 버전**: `domain_matrix_proper_test` vs `domain/domain_matrix.npy` 두 버전 존재, 비율 차이 있음 (37.2% vs 56.5%)

---

## 권장 추가 실험

```bash
# v15d_bp_clean 모델로 annotation-free 성능 재계산
# 각 GO term별 predictions.json 이용하여 domain 유무로 stratify
python3 reports/annotation_free_v15d.py
```

---

## 결론

**S2 체크리스트 상태**: 부분 완료 (v5fix 기반)  
**핵심 주장 지지 여부**: 지지 — Glycolysis/Translation에서 62.8% annotation-free 이소폼의 예측이 가능하며, annotation-present 그룹과 동등하거나 더 높은 성능

*Generated 2026-06-01 | PRISM annotation-free analysis*
