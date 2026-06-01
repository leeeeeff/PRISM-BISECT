# DIFFUSE 프로젝트 현황 보고서
**작성일:** 2026-05-13  
**버전:** v1.0  
**대상:** 연구 방향 점검 및 다음 실험 우선순위 결정

---

## 1. 프로젝트 목표 및 평가 기준

### 목표
근육 세포 단백질 아이소폼의 GO term 기능 예측 (5개 GO term, isoform 해상도)

### 평가 GO terms
| GO term | 기능 | 타입 | 양성 비율 (test) |
|---------|------|------|----------------|
| GO:0007204 | Calcium signaling regulation | Type-B | 0.844% |
| GO:0030017 | Sarcomere assembly | Type-B | 1.230% |
| GO:0006941 | Regulation of muscle contraction | Type-B | 0.688% |
| GO:0003774 | Motor activity | Type-A | 0.446% |
| GO:0006096 | Glycolysis | Type-A | 0.207% |

**Primary metric:** Macro-AUPRC (5개 GO term 평균)  
**Type 분류 기준:** Phase 0 sep_ratio = inter_dist / intra_dist (ESM-2 untrained 기준)  
- Type-A (sep ≥ 1.15): 양성 클러스터 응집 → contrastive 학습 효과적  
- Type-B (sep < 1.15): 양성 이질적 → 모든 파이프라인 전략이 지금까지 어려움

---

## 2. 실험 이력 요약

### 2.1 버전별 성능 (Macro-AUPRC)

| 버전 | GO:0006941 | GO:0007204 | GO:0030017 | GO:0003774 | GO:0006096 | Macro |
|------|-----------|-----------|-----------|-----------|-----------|-------|
| v7c (baseline) | 0.085 | 0.122 | 0.158 | 0.420 | 0.790 | 0.315 |
| v8b (Unified Loss + Discrim. LR) | 0.118 | 0.146 | 0.157 | 0.569 | 0.795 | 0.357 |
| v8-1 (v8b + Bidirectional FiLM) | 0.092 | 0.135 | 0.179 | 0.602 | 0.808 | 0.363 |
| v8-2 (FiLM zeros-init, 기각) | 0.078 | 0.147 | 0.157 | 0.565 | 0.786 | 0.347 |
| v8-3c (Phase 3 balanced LR) | 0.059 | 0.081 | 0.155 | — | — | — |
| **Best ensemble** (GO별 최적 선택) | v8b | v8b | v8-1 | v8-1 | v8-1 | **0.370** |

### 2.2 주요 실험적 발견 (확립된 사실)

| ID | 발견 | 근거 실험 |
|----|------|----------|
| F7 | Phase 0 sep_ratio가 GO term 난이도 및 전략 결정 | v8b vs v8-1 비교 |
| F10 | ESM-2는 Type-B에서도 필수 (제거 시 AUPRC 0.122→0.012) | Diagnostic A |
| F13 | FiLM은 Type-A에만 유효, Type-B에서는 일관 악화 | v8-1, v8-2 |
| F16 | Phase 2가 Type-B 임베딩 정보를 손상 (GO:0006941 −0.027) | LinAUPRC oracle |
| F17 | Prediction head가 임베딩 정보를 충분히 추출 못함 (Head Gap +0.038~+0.102) | LinAUPRC oracle |
| F14 | Temperature scaling은 AUPRC를 변경하지 않음 (ranking 기반) | 실험 직접 확인 |

### 2.3 LinAUPRC Oracle 분석 결과 (2026-05-06)

Phase별 임베딩에서 balanced-LR로 추출 가능한 최대 AUPRC 측정 (5-fold CV):

| GO term | Ph0 (untrained) | Ph1 (metric) | Ph2 (unified) | Model | Head Gap | Ph2 손상 |
|---------|----------------|-------------|--------------|-------|----------|---------|
| GO:0006941 | 0.106 | **0.183** | 0.155 | 0.118 | +0.038 | −0.027 |
| GO:0007204 | 0.113 | **0.231** | 0.223 | 0.146 | +0.077 | −0.008 |
| GO:0030017 | 0.095 | **0.298** | 0.281 | 0.179 | +0.102 | −0.017 |

**해석:** Oracle은 64-dim Phase 1 임베딩에서 5-fold CV (테스트 분포 직접 사용)로 계산.

---

## 3. 신규 핵심 발견: ESM-2 Linear Probe Baseline (2026-05-13)

### 3.1 실험 설계

| | 학습 데이터 | 테스트 데이터 | 모델 | GPU |
|--|------------|-------------|------|-----|
| Option A | ESM-2 640-dim (human 31k + swissprot 83k) | ESM-2 640-dim | sklearn LR (balanced) | 불필요 |
| Option B | ESM-2 640-dim (human 31k only) | ESM-2 640-dim | sklearn LR (balanced) | 불필요 |

### 3.2 결과

| GO term | Option A (A+S LR) | Option B (H-only LR) | v8b (full pipeline) | Best ensemble | Oracle |
|---------|-----------------|-------------------|-------------------|--------------|--------|
| GO:0007204 | 0.267 | **0.414** | 0.146 | 0.146 | 0.231 |
| GO:0030017 | 0.307 | **0.561** | 0.157 | 0.179 | 0.298 |
| GO:0006941 | 0.294 | **0.312** | 0.118 | 0.118 | 0.183 |
| GO:0003774 | 0.765 | **0.825** | 0.569 | 0.602 | 0.656 |
| GO:0006096 | **0.855** | 0.695 | 0.795 | 0.808 | 0.733 |
| **Macro** | **0.497** | **0.561** | 0.357 | 0.370 | — |

> **Random baseline:** GO:0006941 ≈ 0.0069, GO:0007204 ≈ 0.0084, GO:0030017 ≈ 0.0123

### 3.3 수치 해석

**[발견 1] SwissProt이 muscle-specific GO term에 해롭다**
- GO:0006096 (glycolysis, cross-species conserved): A > B (0.855 vs 0.695) → SwissProt이 도움
- 나머지 4개 GO term: B > A → SwissProt가 distribution shift 유발
- Type-B 3개 GO term (GO:0006941 제외): B가 A보다 +37~+83% 우수

**[발견 2] 전체 파이프라인이 단순 LR보다 크게 열등하다**

| GO term | v8b AUPRC | Option B AUPRC | 배율 |
|---------|----------|---------------|------|
| GO:0007204 | 0.146 | 0.414 | +183% |
| GO:0030017 | 0.157 | 0.561 | +257% |
| GO:0006941 | 0.118 | 0.312 | +165% |
| GO:0003774 | 0.569 | 0.825 | +45% |
| Macro | 0.357 | 0.561 | +57% |

**[발견 3] 이전 Oracle이 과소 추정되었음**  
- 이전 Oracle은 Phase 1 64-dim 임베딩 + 테스트 분포 직접 사용 (cheating)
- 실제 ESM-2 640-dim + 정직한 train→test 예측이 oracle보다 높음
  - GO:0007204: Oracle 0.231 vs Option B 0.414 (+79%)
  - GO:0030017: Oracle 0.298 vs Option B 0.561 (+88%)
- **결론:** 기존 oracle이 이미 degraded된 Phase 1 임베딩(64-dim)을 기반으로 했기 때문에 원래 성능 상한을 심각하게 과소 추정했음

---

## 4. 현재 상황 진단: 파이프라인 실패 원인 분석

### 4.1 알려진 원인들

```
ESM-2 raw (640-dim, human-only) + LR = Macro 0.561  ← 현재 최고
ESM-2 raw (640-dim, human+swiss) + LR = Macro 0.497
Phase 0 (untrained, 64-dim) + CV oracle = Macro ~0.09
Phase 1 (metric, 64-dim) + CV oracle = Macro ~0.38
v8b full pipeline = Macro 0.357
```

### 4.2 원인 후보 분해

| 원인 | 증거 | 심각도 |
|------|------|--------|
| **① SwissProt distribution shift** | B >> A for 4 GO terms | 높음 |
| **② 640→64 dimension bottleneck** | Ph1 oracle(64-dim) 0.183 vs raw ESM-2(640-dim) 0.312 (같은 GO:0006941) | 높음 |
| **③ Phase 2 embedding 손상** | Ph1 oracle 0.183 → Ph2 oracle 0.155 (−0.027) | 중간 |
| **④ Prediction head calibration** | Ph2 oracle 0.155 → model 0.118 (−0.038) | 중간 |
| ~~⑤ ESM-2 자체 문제~~ | F10으로 기각: ESM-2 OFF → AUPRC 10배 악화 | 해당없음 |
| ~~⑥ Score compression~~ | F14로 기각: temperature scaling AUPRC 불변 | 해당없음 |

### 4.3 원인별 기여도 추정 (GO:0006941 기준)

```
ESM-2 640-dim human-only LR 목표치: 0.312
  - 현재 v8b:               0.118

Gap 분해:
  SwissProt 오염 제거:       +0.018  (A=0.294 → B=0.312)
  Dimension 회복 (64→640):   +0.129  (Ph1 oracle 0.183 → ESM-2 LR 0.312, 추정*)
  Phase 2 손상 해소:         +0.027  (Ph2 oracle 0.155 → Ph1 oracle 0.183)
  Head calibration 해소:     +0.038  (Model 0.118 → Ph2 oracle 0.155)

* 직접 비교 불가 (oracle은 테스트 분포 사용). 성분 분리를 위해 실험 필요.
```

**가장 큰 미해결 문제:** 왜 Phase 1 contrastive 학습이 ESM-2의 640-dim 정보를 64-dim으로 압축하면서 이렇게 많은 discriminative power를 잃는가?

---

## 5. 논문 기여도 현황 평가

### 5.1 현재 상태

- **현재 Best (ensemble 0.370) < ESM-2 LR baseline (0.561)**
- 이 상태로는 Nature Methods / NMI 제출 불가 — 복잡한 파이프라인이 단순 baseline보다 나쁨
- "DIFFUSE 방법론이 기여를 한다"는 핵심 클레임을 지지할 수 없음

### 5.2 기여 가능한 영역 (파이프라인이 LR보다 나은 경우)

| 항목 | 현황 | 논문 기여 가능성 |
|------|------|----------------|
| Type-A GO term (GO:0003774, GO:0006096) | v8-1이 LR Option A보다 우수 (0.808 vs 0.855 일부) | 제한적 |
| GO:0006096: model(0.808) > B(0.695) | 파이프라인이 LR보다 우수 | 유망 |
| GO:0003774: model(0.602) < B(0.825) | 파이프라인이 LR보다 열등 | 개선 필요 |
| Type-B 3개 GO term | 모두 파이프라인 << LR | 전면 재검토 필요 |

---

## 6. 우선 분석 목록 (Priority Queue)

### 6.1 즉각 실행 가능 (재훈련 불필요)

| 우선순위 | 실험 | 목적 | 예상 소요 |
|---------|------|------|----------|
| **P1** | Phase 1 train 임베딩 저장 → 전체 LR 파이프라인 (train→test) | 64-dim vs 640-dim 기여 분리 | 1일 |
| **P2** | ESM-2 640-dim 5-fold CV on train (human-only) | 실제 train-time oracle 확인 | 2시간 |

### 6.2 단기 실험 (1~2주)

| 우선순위 | 실험 | 목적 | 예상 소요 |
|---------|------|------|----------|
| **P3** | v8b human-only training (SwissProt 제거) | 파이프라인 자체가 human-only에서 개선되는지 확인 | 2일 |
| **P4** | ESM-2 640-dim → MLP head (frozen) | LR vs 비선형 head 비교 | 1일 |
| **P5** | 임베딩 차원 ablation (64, 128, 256, 640) | bottleneck 크기가 성능에 미치는 영향 | 3일 |

### 6.3 중기 실험 (전략적 재설계)

| 우선순위 | 실험 | 목적 | 예상 소요 |
|---------|------|------|----------|
| **P6** | End-to-end fine-tuning (ESM-2 직접 fine-tune on human only) | Phase 1/2 구분 없이 단순화 | 3일 |
| **P7** | PFN context mechanism ablation (PFN vs MLP) | PFN이 isoform context 정보를 실제로 사용하는지 | 3일 |
| **P8** | 전체 ablation table (7개 component: no_triplet/no_focal/no_ppi/no_esm/no_cellloc/no_isoform_specific/no_pfn) | Nature Methods 필수 조건 | 1~2주 |

### 6.4 결정 Gate

```
P3 결과 (v8b human-only) 분석 후:

  Case A: human-only v8b > Option B LR
    → 파이프라인에 가치 있음 → P5, P6, P8로 진행
    → 논문 기여 클레임: "human-only training + contrastive learning > ESM-2 LR"

  Case B: human-only v8b ≈ Option B LR
    → 파이프라인의 복잡성 정당화 불가 → 구조적 재설계 필요
    → P6 (end-to-end fine-tuning) + P7 (PFN ablation)으로 전환

  Case C: human-only v8b < Option B LR (현재 v8b 패턴 반복)
    → 근본적 설계 문제 → PFN 가정 재검토 필요
    → Devils-advocate 에이전트 호출 + 전면 재검토
```

---

## 7. 검증되어야 할 핵심 가설들

### 가설 H1: "SwissProt 제거만으로 파이프라인이 LR baseline을 회복할 수 있다"
- 검증: P3 (v8b human-only)
- 예측: v8b human-only > Option B LR? 현재로선 불명확
- **이 가설이 참이면:** 데이터 오염이 주요 원인 → 데이터 전략 변경으로 해결
- **이 가설이 거짓이면:** 아키텍처 자체가 문제 → 더 근본적 재설계 필요

### 가설 H2: "640→64 차원 압축이 Type-B discriminative power를 파괴한다"
- 검증: P5 (차원 ablation)
- 예측: 128-dim 또는 256-dim에서 급격한 성능 개선 가능성
- 함의: PFN 내부 임베딩 차원 증가 → 파라미터 수 증가 → 정당화 가능

### 가설 H3: "contrastive 학습 자체가 ESM-2 정보를 손상시킨다"
- 검증: Ph0 (untrained) 64-dim oracle vs Ph1 64-dim oracle 비교
  - GO:0006941: Ph0=0.106 → Ph1=0.183 → contrastive가 도움 (손상 아님)
  - **H3는 현재 데이터로는 기각 방향**
- 단, Ph1 (64-dim) vs raw ESM-2 (640-dim) 차이의 얼마가 차원 때문인지 불명확

### 가설 H4: "PFN의 in-context learning이 isoform-specific 정보를 실제로 사용한다"
- 검증: P7 (PFN vs MLP ablation)
- 현재 반증 신호: 단순 LR(context 없음)이 PFN(context 있음)보다 우수
- **이 가설이 거짓이면:** PFN을 MLP로 교체해도 성능 유지 → 아키텍처 간소화

---

## 8. 알려진 안티패턴 (즉시 거부 목록)

| 패턴 | 근거 |
|------|------|
| Phase 1 이전 특정 branch 사전 훈련 (Phase 1a) | AP1: active ratio 0.2% 붕괴 |
| ESM-2 제거/비활성화로 Type-B 개선 시도 | AP5: F10 실험 확인, AUPRC 10배 악화 |
| Phase 1.5 ("frozen repr + new head" 중간 단계) | AP6: sep_ratio 급락 실험 확인 |
| Type-B에 대한 FiLM 파라미터 추가 탐색 | AP13: v8-2에서 검증 완료 기각 |
| Temperature scaling으로 AUPRC 개선 시도 | AP11: F14 실험 확인, 불변 |
| "α 높이면 positive gradient 증가" 단순 논리 | AP12: F15 확인, 93.9% 이미 지배적 |

---

## 9. 요약

### 현재 위치
```
                           Macro-AUPRC
Random baseline:           ~0.005~0.012 (GO별 positive ratio)
v7c (baseline):             0.315
v8b/v8-1 best ensemble:     0.370
ESM-2 + LR (human+swiss):   0.497  ← 파이프라인 없이 달성
ESM-2 + LR (human-only):    0.561  ← 현재 사실상 SOTA

LinAUPRC Oracle (Ph1,64d):  ~0.38 (테스트 분포 cheating)
```

### 즉각 해결해야 할 핵심 질문 하나

> **"Human-only 데이터로 훈련된 DIFFUSE 파이프라인이 human-only ESM-2 + LR보다 나은가?"**

이 질문의 답이 향후 연구 방향 전체를 결정한다.

- YES → SwissProt 제거 + 파이프라인 유지 → 점진적 개선
- NO → 파이프라인의 핵심 설계 가정 재검토 필요

### 다음 즉시 행동
1. **v8b human-only training 실행** (P3) — 2일 소요, GPU 필요
2. **Phase 1 train 임베딩 저장 + 전체 LR 실험** (P1) — 1일 소요, CPU

---

*이 보고서는 2026-05-13 기준 실험 데이터를 기반으로 작성되었습니다.*  
*ESM-2 linear probe 실험 코드: `hMuscle/model/esm2_probe_baseline.py`*
