# 레포트 03: GO Term 특성 분석
**작성일:** 2026-04-29  
**데이터셋:** hMuscle (36,748 isoforms, 5 GO terms)

---

## 1. GO Term 기본 통계

| GO Term | 이름 | Positive 비율 | Type | 특성 |
|---------|------|--------------|------|------|
| GO:0006096 | Glycolysis | ~3-5% | A | 진화적 보존, ESM-2 분리 가능 |
| GO:0003774 | Motor Activity | ~2-4% | A | Myosin 계열, 도메인 보존 |
| GO:0006941 | Striated Muscle Contraction | **0.69%** | B | 극단적 sparse, 수렴 진화 |
| GO:0007204 | Cytosolic Ca²⁺ Concentration | ~1-2% | B | 이질적 positive, 불안정 |
| GO:0030017 | Sarcomere Organization | ~1-3% | B | sparse + 이질적 |

---

## 2. GO:0006941 — 상세 분석 (가장 어려운 케이스)

### 생물학적 배경
- Striated muscle contraction: 골격근 · 심근의 수축 기능
- GO:0006941 positive: titin, myosin heavy chain, troponin 등 구조적 단백질 + 칼슘 신호 단백질 혼합
- 수렴 진화적 특성: 다양한 단백질 계열에서 독립적으로 진화 → 이질적 positive

### 실험적 관찰
```
Phase 0 sep_ratio: 1.033 (Type-B 중 최저)
Positive 비율: 0.69% (5 GO term 중 가장 낮음)
```

**v6g Phase 2 실패 패턴:**
```
Epoch 1: Acc=0.699, AUPRC=0.119 (random)
Epoch 2: Acc=0.978, AUPRC=0.094 → "all negative" shortcut
         Focal loss가 negative 샘플에 극도로 편향
```

### 근본 원인
1. **sep_ratio=1.033:** Phase 1 Triplet이 positive를 분리하지 못함
   → Phase 2가 discriminative signal 없이 시작
2. **0.69% positive:** Focal Loss α=0.10에도 클래스 불균형 극단적
   → "all negative" 예측이 loss 최소화 지름길
3. **두 조건 동시 충족 시 collapse 확실:** 분리 불가 embedding + 극단 불균형

### 다른 GO term과 비교

| GO Term | Phase 0 sep | Positive % | Phase 2 결과 |
|---------|------------|-----------|-------------|
| GO:0006941 | 1.033 | 0.69% | all-neg collapse |
| GO:0007204 | 1.07-1.45 | ~1-2% | 불안정하지만 학습 |
| GO:0030017 | 1.00-1.07 | ~1-3% | 부분 학습 |
| GO:0003774 | 1.16-1.57 | ~2-4% | 학습 성공 (Type-A) |
| GO:0006096 | 1.17-1.66 | ~3-5% | 학습 성공 (Type-A) |

### 개선 가능성
- **v7a SupCon:** 배치 내 전체 positive 쌍 동시 학습 → sep_ratio 개선 가능성
- **v7a-Proto:** k=1 단일 prototype → positive centroid 명시적 유지
- **장기:** 극단 class imbalance 전용 loss (asymmetric loss, query-based sampling)

---

## 3. GO:0007204 — 재현성 경고 분석

### 관찰
- v6g에서 AUPRC=0.591 (역대 최고)
- 이 run의 Phase 0 sep_ratio = **1.450** (비정상적으로 높음)
- 일반적 GO:0007204 sep_ratio: 0.98-1.12 (Type-B)

### 원인 추정
- **랜덤 초기화 운:** 특정 seed에서 초기 embedding이 우연히 Type-A 수준 분리
- sep_ratio=1.450 → Type-A 경로 (lr=0.0003, patience=7)로 훈련
- Type-B 경로(lr=0.0002)로 훈련했다면 결과 달랐을 것

### 판단
- 이 0.591은 **재현 불가능한 lucky run**일 가능성 높음
- v7 비교 시 GO:0007204 기준: v6g 0.591이 아닌 **v6f 0.309** 또는 **v6e 0.388** 사용
- 추후 v6g 3-run 실행으로 실제 mean AUPRC 측정 필요

---

## 4. GO:0006096 — 안정 케이스 분석

### 특성
- Glycolysis 경로: 잘 보존된 효소 집합 (hexokinase, PGK, GAPDH 등)
- 진화적 보존 강함 → ESM-2가 자연스럽게 분리 가능

### 실험 관찰
- Phase 0 sep_ratio 1.17-1.66: 초기부터 분리 양호
- Type-A 안정적 분류 (간헐적 경계선 run 제외)
- v6g AUPRC=0.823: v6 시리즈 최고

### TARGET_COVERAGE 민감성
- coverage=4.0 (v6e): AUPRC 0.813 → 0.783 (-3.8%)
  → positive 과도한 downsampling → 훈련 정보 손실
- coverage=6.0 복구 (v6f/v6g): 0.823 회복

---

## 5. GO:0003774 — Motor Activity 분석

### 특성
- Myosin, Dynein, Kinesin 계열: 도메인 구조 보존적
- Phase 0 sep_ratio 1.16-1.57: 경계선 근처 (Type-A/B 경계)

### Phase 1.5 취약성
- v6e에서 Phase 1.5가 sep_ratio 1.58 → 1.01로 파괴
  → v6e AUPRC 0.671 → 0.558 (급락)
- v6g (Phase 1.5 제거) → 0.591 회복

### DomainDelta 기여
- Motor domain gain/loss가 이소폼 기능 분화의 핵심 신호
- v6d에서 DomainDelta 추가 후 0.671 (이전 버전 대비 상승)

---

## 6. GO:0030017 — Sarcomere Organization (불안정)

### 특성
- Sarcomere 구조 단백질: titin, nebulin, alpha-actinin 등
- GO:0006941 (수축)과 생물학적 중복 있음

### 실험 관찰
- v6e 0.332 (최고), v6g 0.201 (최저)
- 버전 간 변동 폭: 0.131 — 다른 GO term보다 큰 불안정성

### 가설
- Type-B이지만 positive들의 이질성이 매우 높음
- 한 번의 run 결과로 판단하기 어려움
- 3-run 평균이 실제 성능 파악에 필요

---

## 7. 전체 GO Term 비교: 예측 난이도 순위

```
쉬움 ──────────────────────────────── 어려움
GO:0006096 > GO:0003774 >> GO:0007204 > GO:0030017 > GO:0006941
   0.823        0.591        (불안정)     (불안정)      0.121
```

주요 결정 요인:
1. Phase 0 sep_ratio (높을수록 쉬움)
2. Positive 클래스 비율 (높을수록 쉬움)  
3. Positive 집합의 진화적 응집성 (보존적 기능일수록 쉬움)

---

## 8. 향후 추가 GO Term 고려사항

현재 5개 GO term 외에 추가 검토 가능:
- GO:0006412 (Translation): ribosomes, 매우 보존적 → 쉬울 것으로 예상
- GO:0006936 (Muscle Contraction): GO:0006941 상위 term → 유사 특성
- GO:0022900 (Electron Transport): mitochondrial complex → 보존적

ablation_schedule.md의 5-term 세트:
GO:0006096, GO:0006412, GO:0006936, GO:0022900, GO:0003774
(현재 실험 GO term과 일부 차이 — 통일 필요 검토)
