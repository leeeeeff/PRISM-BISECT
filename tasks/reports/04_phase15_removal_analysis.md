# 레포트 04: Phase 1.5 제거 분석 보고서
**작성일:** 2026-04-29  
**결정:** Phase 1.5 (Linear Probing) 완전 제거 [v6g-1]  
**근거 유형:** 실측 임베딩 데이터 + 성능 비교

---

## 1. Phase 1.5 원래 설계 목적

Phase 1 (Triplet)이 구축한 임베딩 공간 위에 prediction head를 calibrate:
- Feature model (ESM-2 + Domain) 완전 동결
- Classification head만 Focal Loss로 학습
- 기대 효과: Triplet embedding 품질을 유지하면서 판별 경계 학습

---

## 2. 실측 데이터: 임베딩이 실제로 변화함

### 관찰 (v6e 실험)

Phase 1 완료 후 임베딩 `e_ph1`과 Phase 1.5 완료 후 임베딩 `e_ph15`를 비교:

| GO Term | max_diff | 의미 |
|---------|----------|------|
| GO:0006941 | 0.52 | frozen 선언에도 임베딩 변화 |
| GO:0003774 | 0.63 | 더 큰 변화 |

- `max_diff = max(|e_ph1 - e_ph15|)` — frozen이라면 0이어야 함
- **결론:** feature_model이 코드상 frozen 선언되었으나 실제 forward pass에서 임베딩이 변화

### 가능한 메커니즘
1. **Keras GradientTape vs compile 혼용:** Phase 1은 GradientTape (수동), Phase 1.5는 model.compile+fit (자동) → 학습 경로 차이
2. **BN/Dropout 상태:** BatchNorm의 running mean/var가 inference 시 업데이트될 수 있음
3. **Head의 역전파가 feature_model로 누수:** Keras 1.x 동결 메커니즘의 불완전성

---

## 3. sep_ratio 파괴 패턴 (v6e 관찰)

Phase 1이 구축한 분리도(sep_ratio)가 Phase 1.5에서 역전:

| GO Term | Ph1 sep | Ph1.5 sep | 변화 | 영향 |
|---------|---------|----------|------|------|
| GO:0006941 | 1.25 | 0.86 | **-0.39** | Phase 2 시작점 열악 |
| GO:0003774 | 1.58 | 1.01 | **-0.57** | 심각한 파괴 |
| GO:0006096 | 1.87 | 1.88 | +0.01 | 정상 (Type-A 영향 없음) |

**패턴:** Type-B GO term에서 Phase 1.5가 특히 해로움
- Type-B는 Phase 1에서 어렵게 구축한 분리도가 낮음 (1.0-1.2 수준)
- Phase 1.5가 이 약한 분리도를 더 파괴

---

## 4. 성능 비교

| 버전 | Phase 1.5 | Macro-AUPRC | 비고 |
|------|----------|-------------|------|
| v6d | 있음 | 0.458 | Phase 1.5 있는 버전들 |
| v6e | 있음 | 0.463 | |
| v6f | 있음 | **0.424** | v6 시리즈 최저 |
| **v6g** | **없음** | **0.465** | v6 시리즈 최고 |

v6f에서 최저를 기록한 이유: Phase 1.5가 있는 상태에서 lr=0.0002로 Phase 2를 더 공격적으로 학습 → Phase 1.5 파괴된 임베딩에 과도한 Phase 2 학습 = 이중 손상

---

## 5. 이론적 설명

### Linear Probe가 metric space를 왜곡하는 메커니즘

Phase 1 Triplet이 구축한 임베딩 공간:
```
intra_pos 거리 < margin + inter_pos_neg 거리
(positive들이 뭉치고, negative들이 떨어짐)
```

Phase 1.5 Linear Probe가 학습하는 것:
```
w_head · z = logit → sigmoid → label 예측
```

`w_head`의 선형 변환은 임베딩 공간에서 effective metric을 변경:
- 특정 방향(head가 중요하다고 학습한 방향)으로 거리 가중
- Triplet이 구축한 isotropic metric → anisotropic으로 변환
- 결과: sep_ratio 계산 시 이 왜곡된 metric을 반영 → 겉보기 sep_ratio 감소

### Phase 1.5가 효과 없는 이론적 이유 (이 데이터셋)
1. **Sparse positive:** Phase 1.5 배치에 positive가 거의 없음 → head가 trivial solution(all-negative) 학습
2. **Focal Loss with frozen embedding:** embedding이 discriminative하지 않으면 head가 할 수 있는 것 없음
3. **Phase 2와 역할 중복:** Phase 1.5와 Phase 2 모두 Focal Loss → Phase 1.5 제거해도 Phase 2가 담당

---

## 6. 결론 및 규칙

**결론:** Phase 1.5는 이 데이터셋에서:
1. 임베딩을 실제로 변화시킴 (frozen 선언 불완전)
2. Phase 1이 구축한 metric space를 파괴
3. 성능을 감소시킴 (v6f -0.039)

**v6g 결정:** Phase 1.5 완전 제거, Phase 1 → Phase 2 직행

**규칙 (향후 적용):**
- Phase 사이에 frozen + 다른 loss로 calibration 시도 시 임베딩 변화 실측 필수
- GradientTape(Phase 1)와 model.compile+fit(Phase 1.5/2) 혼용 시 freeze 효과 불신뢰
- sep_ratio 파괴 관찰 시 즉시 해당 Phase 제거 고려

---

## 7. 대안으로 검토되었으나 채택 안 된 방법들

| 대안 | 이유 |
|------|------|
| Phase 1.5에서 완전 새 모델 | 코드 복잡도 증가, Phase 1 가중치 재사용 불가 |
| Phase 1.5 learning rate 극소화 | 근본 문제(임베딩 변화) 해결 안 됨 |
| Phase 1 + Phase 2 동시 학습 | ESM-2 재조직 문제 (v6h 초기 실패 패턴) |
| Phase 1.5 제거 + Phase 2 loss 보강 | **채택**: v6g 방향 |
