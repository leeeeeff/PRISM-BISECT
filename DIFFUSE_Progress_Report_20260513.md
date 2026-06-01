# DIFFUSE 중간 분석 보고서
**작성일:** 2026-05-13  
**상태:** P3 GPU 대기 중 (dorado 선점)  
**범위:** 오늘 세션에서 수행된 분석 전체

---

## 목차
1. [위기 진단 — ESM-2 LR baseline](#1-위기-진단)
2. [원인 분석 1 — v9-1 추가 feature ablation](#2-v9-1-feature-ablation)
3. [원인 분석 2 — ESM-2 dimension bottleneck 정량화](#3-esm-2-dimension-bottleneck)
4. [파이프라인 실패 원인 종합 분해](#4-원인-종합-분해)
5. [P3 실험 설계 현황](#5-p3-실험-설계)
6. [의사결정 트리](#6-의사결정-트리)

---

## 1. 위기 진단

> **단순 ESM-2 + Logistic Regression이 전체 파이프라인보다 57% 우수하다.**

### 1.1 핵심 수치 비교

| 모델 | GO:0007204 | GO:0030017 | GO:0006941 | GO:0003774 | GO:0006096 | **Macro** |
|------|-----------|-----------|-----------|-----------|-----------|---------|
| ESM-2 LR (human+swiss) | 0.267 | 0.397 | 0.294 | 0.716 | 0.855 | 0.506 |
| **ESM-2 LR (human-only)** | **0.414** | **0.561** | **0.312** | **0.825** | **0.695** | **0.561** |
| v8b 파이프라인 | 0.146 | 0.157 | 0.118 | 0.569 | 0.795 | 0.357 |
| v8-1 best ensemble | 0.146 | 0.179 | 0.118 | 0.602 | 0.808 | 0.370 |

- `v8b < ESM-2 LR (human+swiss)` — 복잡한 파이프라인이 더 단순한 방법에도 패배
- `ESM-2 human-only > ESM-2 human+swiss` — SwissProt이 muscle-specific GO term에 해로움
  - 예외: GO:0006096 (glycolysis, cross-species conserved) — swiss 쪽이 유리 (0.855 > 0.695)

### 1.2 이전 Oracle이 과소 추정된 이유

이전 LinAUPRC oracle (Phase 1, 64-dim, 5-fold CV on test) = ~0.38  
실제 ESM-2 train→test = 0.561

```
이전 oracle 한계:
  ① 64-dim으로 이미 정보 손실된 임베딩 기반
  ② 테스트 내 5-fold CV → test 분포를 보고 학습 (optimistic bias)
  ③ 이 두 효과가 상쇄되어 과소 추정

→ 기존 상한(0.38)은 잘못된 ceiling이었다.
→ 실제 ceiling ≈ 0.561 (ESM-2 human-only LR)
```

---

## 2. v9-1 Feature Ablation

> **domain_delta_v2, splicing_delta_v2 모두 ESM-2에 추가 정보를 제공하지 않는다.**

### 2.1 실험 설계

- **방법:** gene-stratified GroupKFold K=5 CV on test set (이소폼 누출 방지)
- **비교:**
  - A: ESM-2 640-dim 단독
  - B: ESM-2 640-dim + domain_delta_v2 (sign)
  - C: ESM-2 640-dim + domain_delta_v2 + splicing_delta_v2
- **코드:** `esm2_640dim_ablation.py`

### 2.2 결과

| GO term | A (ESM-2 only) | B (+domain_Δ) | C (+domain_Δ+splice_Δ) |
|---------|---------------|--------------|----------------------|
| GO:0007204 | 0.1261 | 0.1180 | 0.1048 |
| GO:0030017 | 0.1709 | 0.1670 | 0.1776 |
| GO:0006941 | 0.0423 | 0.0383 | 0.0299 |
| GO:0003774 | 0.5077 | 0.4955 | 0.4066 |
| GO:0006096 | 0.4069 | 0.4080 | 0.3899 |
| **Macro** | **0.2508** | **0.2454 (−0.005)** | **0.2218 (−0.029)** |

> ※ CV Macro(0.251)와 train→test Macro(0.562)의 괴리(0.45×)는 GroupKFold의 test-내 평가 한계.  
> 중요한 건 A/B/C 간 상대 비교 — B와 C 모두 A 이하.

### 2.3 결론

- `domain_delta_v2` (Δ=−0.005): ESM-2 대비 중립적 또는 미미한 손해
- `splicing_delta_v2` (Δ=−0.029 추가): 일관된 성능 저하
- **v9-1 설계 방향 (추가 feature 통합) 우선순위 하향** — feature들이 ESM-2와 중복이거나 노이즈

---

## 3. ESM-2 Dimension Bottleneck 정량화

> **현재 64-dim 파이프라인은 PCA 선형 상한의 63.6%를 달성하고 있다.  
> 90% 효율에 도달하려면 512-dim PCA가 필요하다.**

### 3.1 실험 설계

- **방법:** ESM-2 640-dim → PCA(d) → balanced LR, train(human 31K) → test
- **차원:** [32, 64, 96, 128, 192, 256, 384, 512, 640]
- **코드:** `esm2_dim_sweep.py` (오늘 완료)

### 3.2 PCA Explained Variance

| dim | 분산 보존율 | 의미 |
|-----|-----------|------|
| 32 | 85.18% | |
| 64 | 91.98% | 현재 파이프라인 target dim |
| 96 | 94.73% | |
| 128 | 96.19% | |
| 192 | 97.77% | 임계점 (분산 관점) |
| **256** | **98.61%** | **P3-256 실험 target** |
| 512 | 99.84% | |

> 90% 분산 도달 기준: 51 PCs / 95%: 101 PCs / 99%: 303 PCs  
> → 분산 보존율은 이미 64-dim에서 92%로 높지만, AUPRC는 전혀 다른 이야기

### 3.3 PCA LR Macro-AUPRC (선형 상한)

| dim | PCA Macro | % of 640-PCA ceiling |
|-----|-----------|---------------------|
| 64 | 0.1826 | 35.4% |
| 96 | 0.1886 | 36.6% |
| 128 | 0.2204 | 42.8% |
| 192 | 0.3195 | 62.0% |
| **256** | **0.3311** | **64.3%** |
| 384 | 0.4291 | 83.3% |
| **512** | **0.4690** | **91.0%** ← 90% 도달 |
| 640 | 0.5151 | 100% |
| ESM-2 LR ref | 0.5614 | — |

### 3.4 GO term별 포화점 분석

모든 5개 GO term이 640-dim에서도 포화되지 않음 (95% AUPRC 미도달).  
→ 더 많은 차원이 항상 더 나은 PCA 성능을 제공

Type-B 3개 term에서 특히 두드러짐:

| GO term | PCA 256 | PCA 640 | 256/640 비율 |
|---------|---------|---------|------------|
| GO:0007204 | 0.1656 | 0.3973 | **41.7%** |
| GO:0030017 | 0.2876 | 0.4206 | **68.4%** |
| GO:0006941 | 0.1224 | 0.2565 | **47.7%** |

→ Type-B에서는 256-dim PCA도 정보를 절반 이상 손실

### 3.5 핵심 발견: v8b 파이프라인 > PCA 256-dim

```
v8b 64-dim contrastive:  Macro = 0.357
PCA 256-dim (linear):    Macro = 0.331

→ 64-dim contrastive가 256-dim PCA를 +7.8% 상회
→ Contrastive 학습 효율 ≈ PCA 대비 4배 이상 (같은 dim에서)
```

이는 contrastive 학습이 단순 PCA보다 discriminative 정보를 훨씬 효율적으로 추출함을 의미.  
반면 `ESM-2 LR (human-only) = 0.561`은 여전히 v8b(0.357)보다 57% 우수.  
→ **이 gap의 원인이 SwissProt 오염인지, 차원 bottleneck인지를 P3/P3-256이 분리한다.**

### 3.6 Contrastive × Dimension 기대치

PCA 대비 4× 효율을 보수적으로 2×로 가정할 때:

| 파이프라인 dim | PCA 등가 | 기대 Macro |
|-------------|--------|----------|
| 64-dim (현재 v8b) | PCA 128-256 | ≈ 0.33~0.37 ← 실측 0.357 ✓ |
| 256-dim (P3-256) | PCA 512 | **≈ 0.42~0.47** |
| 512-dim (미설계) | PCA 640+ | **≈ 0.50+** |

---

## 4. 원인 종합 분해

### 4.1 현재까지 확립된 Gap 구성 (GO:0006941 기준)

```
목표:  ESM-2 LR human-only = 0.312
현재:  v8b pipeline          = 0.118
Gap                          = 0.194 (100%)

원인 분해:
  ① SwissProt 오염           +0.018  (9.3%)   [측정됨: A=0.294, B=0.312]
  ② 64→640 dim 손실          +0.129  (66.5%)  [추정: Ph1 oracle 0.183 → ESM-2 LR 0.312]
  ③ Phase 2 embedding 손상   +0.027  (13.9%)  [측정됨: LinAUPRC Ph1→Ph2]
  ④ Head calibration gap     +0.038  (19.6%)  [측정됨: LinAUPRC Ph2→model]
  (중복 포함 가능, 합이 100%를 초과할 수 있음)
```

> ② 의 "dim 손실"은 Phase 1 contrastive 학습 자체 한계 + 차원 bottleneck 복합 효과.  
> P3-256이 64→256 변경만으로 이 성분 얼마를 회복하는지를 측정한다.

### 4.2 각 원인의 실험적 상태

| 원인 | 측정 방법 | 상태 |
|------|---------|------|
| SwissProt 오염 | P3 (human-only training) | **대기 중** |
| Dim bottleneck | P3-256 vs P3 delta | **대기 중** |
| Phase 2 손상 | LinAUPRC oracle (F16) | 확립됨 |
| Head gap | LinAUPRC oracle (F17) | 확립됨 |
| v9-1 feature 기여 | esm2_640dim_ablation | **오늘 확립 — 기여 없음** |

---

## 5. P3 실험 설계

### 5.1 실험 구조

두 실험을 순차 실행하여 SwissProt 효과와 dim 효과를 분리:

```
v8b (current):          human+swiss, 64-dim  → Macro 0.357
  ↓ SwissProt 제거만
v8b-P3 (P3):            human-only,  64-dim  → Macro ?
  ↓ dim 64→256만 추가
v8b-P3-256 (P3-256):    human-only,  256-dim → Macro ?
```

### 5.2 모델 변경 사항

**v8b → v8b-P3 (6개 변경):**
- `X_train_other_esm2 = np.zeros((0, 640))` — SwissProt ESM-2 비활성화
- `X_train_other_seq/dm/dd` 동일하게 비활성화
- `positive_Gene` 소스를 `human_annotations.txt` 단독으로 변경
- `X_train_geneid_other = []`

**v8b-P3 → v8b-P3-256 (임베딩 차원만 변경):**
- `Dense(64)` → `Dense(256)` (ESM-2 projection 3단계)
- `head_dense64` → `head_dense256`
- `emb_dim=256` (PrototypeContrastiveLoss)
- `w_init.reshape(256, 1)` (prediction head warm init)

### 5.3 Gate 판단 기준

```
P3 결과 기준:
  Macro > 0.561  →  Case A: 파이프라인이 단독으로 LR 상회 (SwissProt이 주원인)
  Macro 0.37~0.56 →  Case B: 부분 개선 (SwissProt + dim 둘 다 문제)
  Macro < 0.37   →  Case C: SwissProt 제거가 역효과 → 구조 재검토

P3-256 결과 기준:
  Gate PASS: P3-256 Macro > 0.561 → dim 확장으로 LR baseline 도달
  Gate PARTIAL: 0.45~0.56 → 512-dim 실험 또는 Phase 2/Head 개선 병행
  Gate FAIL: < 0.45 → E2E fine-tuning 방향 검토
```

### 5.4 현재 상태

| 항목 | 상태 |
|------|------|
| `v8b-P3_integrated_full_model.py` | 작성 완료 |
| `v8b-P3-256_integrated_full_model.py` | 작성 완료 |
| `run_GPU_v8-P3.py` / `run_GPU_v8-P3-256.py` | 작성 완료 |
| `wait_and_run_P3.sh` (PID 1193207) | **폴링 중** — GPU 8GB 확보 시 자동 시작 |
| `analyze_P3_results.py` | 결과 분석 준비 완료 |
| GPU 현황 | GPU 0: 1.3GB 여유, GPU 1: 0.8GB 여유 (dorado 점유 중) |

---

## 6. 의사결정 트리

```
                        [P3 결과]
                           │
           ┌───────────────┼───────────────┐
        Case A           Case B           Case C
      Macro>0.561     0.37~0.56        Macro<0.37
           │               │               │
    [P3-256 결과]    [P3-256 결과]    [devils-advocate]
           │               │
     PASS(>0.561)    PARTIAL(>0.45)
           │               │
   ✓ 5→20 GO term    512-dim 실험
   Bootstrap CI      or Phase2 fix
```

### 6.1 Case A가 성립하면 (P3 > 0.561)

- **해석:** SwissProt 오염이 주 원인. 파이프라인 자체는 유효.
- **다음:** GO term 20개 확장 → Bootstrap CI (n=1000) → 논문 기여 클레임 수립

### 6.2 Case B + P3-256 PARTIAL (0.45~0.56)

- **해석:** 두 원인이 공존. Dim이 지배적.
- **다음 옵션:**
  - `v8b-P3-512` 실험 (PCA 90% 효율 dim)
  - Phase 2 손상 해소 (Phase 1 임베딩 고정 + head만 balanced LR fine-tune)
  - Head gap 해소 (F17): Phase 2 head를 `pos_weight` 기반으로 교체

### 6.3 Case B + P3-256 FAIL (< 0.45)

- **해석:** 파이프라인의 contrastive 설계가 근본적으로 dim 증가를 활용하지 못함
- **다음:** E2E fine-tuning (ESM-2 직접 fine-tune on human-only) 검토

### 6.4 Case C (P3 < v8b 0.357)

- **해석:** SwissProt 제거가 오히려 악영향 (데이터 양 감소가 지배적)
- **다음:** devils-advocate 에이전트 호출 → 파이프라인 기본 가정 재검토

---

## 부록: 오늘 생성된 파일 목록

| 파일 | 목적 | 상태 |
|------|------|------|
| `esm2_probe_baseline.py` | ESM-2 LR baseline 측정 | 완료 |
| `esm2_640dim_ablation.py` | v9-1 feature 기여 측정 | 완료, 결과 확립 |
| `esm2_dim_sweep.py` | PCA dim 선형 상한 측정 | 완료, 오늘 실행 |
| `esm2_dim_sweep_results.txt` | dim sweep 수치 결과 | 저장됨 |
| `v8b-P3_integrated_full_model.py` | SwissProt 제거 모델 | 대기 |
| `v8b-P3-256_integrated_full_model.py` | +dim 256 모델 | 대기 |
| `run_GPU_v8-P3.py` / `v8-P3-256.py` | 듀얼 GPU 런너 | 대기 |
| `wait_and_run_P3.sh` | GPU 여유 감시 + 자동 실행 | 폴링 중 |
| `analyze_P3_results.py` | 결과 자동 분석 + Gate 판단 | 준비됨 |

---

*보고서 기준: 2026-05-13 15:30 KST*
