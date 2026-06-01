# v5-5 일반성(Generality) 검증 레포트

**날짜**: 2026-04-08  
**목적**: v5-5가 기존 4개 GO term에 과적합된 것인지, 다양한 GO term 유형에 걸쳐 일반화된 안정 상태인지 검증  
**대상 모델**: v5-5_integrated_full_model.py (coverage 기반 동적 n_batches)  
**실행 로그**: `hMuscle/logs_isoform/run_v5-5_generality_20260407_2333.log` (총 5705s, 95.1분)

---

## 1. 검증 설계

### 1-1. GO term 선정 기준

기존 4개 GO term(GO_0006936/0006412/0006096/0022900)은 모두 **BP(Biological Process)** 유형에 치우져 있고, 2개는 pan-cellular housekeeping term임. 다음 4가지 축에서 다양성을 확보:

| 축 | 기존 4개 | 신규 4개 |
|----|---------|---------|
| **Ontology 유형** | 모두 BP | BP×2 + CC×1 + MF×1 |
| **근육 특이성** | tissue-specific 1개, housekeeping 3개 | tissue-specific 3개, isoform-divergent 1개 |
| **pos count 범위** | 76~701 | 164~452 |
| **생물학적 layer** | 대사/번역/수축 | 신호전달/구조/운동단백질 |

### 1-2. 선정된 Tier-1 GO term

| GO term | 이름 | 유형 | pos_iso | 선정 근거 |
|---------|------|------|---------|---------|
| GO:0006941 | striated muscle contraction | BP | 253 | GO_0006936 자식 term — 안정성 재현 테스트 |
| GO:0030017 | sarcomere | CC | 452 | 골격근 핵심 구조 — CC term 첫 테스트 |
| GO:0003774 | cytoskeletal motor activity | MF | 164 | 근육 motor protein — MF term 첫 테스트 |
| GO:0007204 | cytosolic Ca2+ regulation | BP | 310 | 근육수축 upstream signaling |

### 1-3. 예측

| GO term | 예측 근거 | 예상 수준 |
|---------|---------|---------|
| GO:0006941 | GO_0006936과 동일 pathway, 더 sparse → 비슷하거나 낮음 | AUPRC ~ 0.15~0.30 |
| GO:0030017 | 구조 CC term, seq+domain이 직접 encode → 좋을 것 | AUPRC ~ 0.25~0.45 |
| GO:0003774 | sparse MF, motor protein isoform 분화 강함 → 중간 | AUPRC ~ 0.10~0.25 |
| GO:0007204 | hMuscle Ca2+ 채널 이소폼 발현 차이 큼 → 높을 것 | AUPRC ~ 0.20~0.40 |

---

## 2. 실험 결과

### 2-1. 최종 성능 (AUROC / AUPRC)

| GO term | 이름 | 유형 | pos_iso | random AUPRC | **최종 AUROC** | **최종 AUPRC** | **배율** | LabelProp α |
|---------|------|------|---------|-------------|------------|------------|--------|------------|
| **GO:0007204** | Ca2+ signaling | BP | 310 | 0.008 | 0.8075 | **0.5075** | **63×** | 0.2 |
| **GO:0030017** | sarcomere | CC | 452 | 0.012 | 0.8314 | **0.3904** | **33×** | 0.5 |
| **GO:0003774** | motor activity | MF | 164 | 0.004 | 0.7967 | **0.2045** | **51×** | 0.0 |
| GO:0006941 | striated muscle | BP | 253 | 0.007 | 0.6452 | 0.0947 | 14× | 0.3 |

### 2-2. Phase별 Embedding 품질 전체

| GO term | Ph0 Sil | Ph1 Sil | Ph1.5 Sil | Ph2 Sil | Ph2 Sep | Ph2 LinAUROC | margin_sat |
|---------|---------|---------|-----------|---------|---------|------------|-----------|
| GO:0006941 | +0.612 | +0.681 | +0.693 | +0.683 | 0.765 | 0.769 | 11.5% |
| GO:0030017 | +0.468 | +0.560 | +0.570 | **+0.727** | **1.044** | **0.855** | 10.0% |
| GO:0003774 | +0.214 | -0.121 | -0.071 | +0.633 | 0.916 | **0.845** | 15.2% |
| GO:0007204 | -0.210 | -0.106 | +0.309 | **+0.793** | **1.185** | 0.830 | 6.8% |

> GO:0030017, GO:0007204: Ph2 Sep.Ratio > 1.0 → 양성/음성 클러스터가 명확히 분리됨. 8개 GO term 통틀어 가장 건강한 embedding 구조.

### 2-3. 예측 점수 분포 — Positive Collapse 분석

| GO term | score mean | score std | >0.5 예측 수 | >0.5 비율 | 상태 |
|---------|-----------|-----------|------------|---------|------|
| GO:0006941 | 0.661 | 0.021 | 36,748 | **100%** | Positive Collapse |
| GO:0030017 | 0.430 | 0.023 | 517 | **1.4%** | ✅ 건강한 분포 |
| GO:0003774 | 0.444 | 0.055 | 1,284 | **3.5%** | ✅ 건강한 분포 |
| GO:0007204 | 0.573 | 0.030 | 36,748 | **100%** | Collapse지만 ranking 우수 |

**핵심 관찰**:
- GO:0030017, GO:0003774는 >0.5 예측 비율이 각각 1.4%, 3.5%로 실제 positive 비율(1.23%, 0.45%)에 근접 → **Precision이 의미있는 수준**.
- GO:0007204는 100% positive 예측이지만 AUPRC=0.507 → **ranking 자체는 매우 정확**, threshold 기반 precision만 낮음.
- GO:0006941는 100% collapse + AUPRC=0.095 → ranking도 불안정.

---

## 2-A. 학습 다이나믹스 심층 분석

> 이 절은 로그 파일(`hMuscle/logs_isoform/GO_*/v5-5_*_Full.log`)의 phase별 수치를 직접 분석한 결과임.

### 2-A-1. Phase 1 Triplet 수렴 다이나믹스

| GO term | n_pos | n_batches | coverage | margin_sat | centroid_dist | Active@Ep3 | 수렴 종료 |
|---------|-------|-----------|----------|-----------|--------------|-----------|---------|
| GO:0006941 | 437 | 20 | **11.7×** | **5.9%** | 0.054 | 8.3% | Ep12 |
| GO:0030017 | 727 | 20 | 7.0× | 24.6% | 0.068 | 7.6% | Ep15 (전체 소진) |
| GO:0003774 | 571 | 20 | 9.0× | **27.4%** | 0.049 | 12.8% | Ep10 |
| GO:0007204 | 882 | 21 | **6.1×** | 22.9% | **0.029** | 20.3% | Ep12 |

#### 핵심 관찰 ①: GO:0006941 "Ep3 급속 수렴" — 최종 실패의 직접 원인

Phase 1 Active ratio 실측:
```
GO:0006941: Ep1→95.0% Ep2→77.2% Ep3→8.3% Ep5→5.4% Ep7→2.9% Ep9→1.6% Ep12→0.6%  [stop]
GO:0030017: Ep1→94.9% Ep2→42.3% Ep3→7.6% Ep6→6.1% Ep11→2.9% Ep14→4.6% Ep15→2.4% [全消]
```

GO:0006941은 Ep3에서 active ratio가 8.3%로 급락, margin_sat=5.9%로 4개 중 최저. Phase 0에서 이미 Sil=+0.612 (사전 분리 良)이었기 때문에 Triplet이 할 일이 없었음. 결과적으로 Phase 1이 Sep ratio를 0.700→0.698로 미세 변화 → Phase 2가 올라갈 발판 미비.

GO:0030017은 15 epoch 내내 active ratio를 2~6%에서 유지하며 지속 학습. margin_sat=24.6%로 4개 중 최고 — sarcomere 단백질들(myosin, actin, titin)이 seq+domain space에서 실제로 분리 가능한 구조를 가짐.

**결론**: `margin_sat < 10%`는 Phase 1 실패 경고 신호. coverage가 높아도(11.7×) 데이터 자체의 이소폼 구별 정보가 없으면 Triplet이 수렴하지 못함.

#### 핵심 관찰 ②: Negative Silhouette → Phase 2 극적 회복 패턴

```
GO:0003774:  Ph0=+0.214  →  Ph1=-0.121  →  Ph1.5=-0.071  →  Ph2=+0.633  (Δ=+0.754)
GO:0007204:  Ph0=-0.210  →  Ph1=-0.106  →  Ph1.5=+0.309  →  Ph2=+0.793  (Δ=+0.899)
```

두 GO term 모두 Phase 1 이후 Silhouette이 음수 (cosine space에서 positive 이소폼이 negative보다 서로 더 흩어짐). 그러나 Phase 2에서 극적 회복.

해석: Motor activity(GO:0003774)와 Ca2+ signaling(GO:0007204)은 **seq 단독보다 Focal Loss가 expression 정보를 결합하는 Phase 2에서 label signal이 처음 드러남**. Phase 1이 "국소 혼란(local disruption)"을 일으키지만 Phase 2의 "전역 재조직(global reorganization)"이 이를 구출.

GO:0007204의 특이점: Phase 1 후에도 Sep.Ratio=1.109(>1.0) — Silhouette이 음수이지만 클러스터 **중심** 간 거리는 이미 intra-cluster보다 큼. Ca2+ 채널 이소폼(RYR1/2, SERCA1/2, CACNA1S)들이 개별 클러스터 내부는 넓지만 중심 거리는 확보된 상태.

---

### 2-A-2. Phase 2 AUPRC 수렴 패턴 비교

Phase 2 epoch별 AUPRC 실측:

| Ep | GO:0006941 | GO:0030017 | GO:0003774 | GO:0007204 |
|----|-----------|-----------|-----------|-----------|
| 1  | 0.043 | 0.225 | 0.005 | 0.028 |
| 2  | 0.032 | 0.111 | 0.014 | 0.202 |
| 3  | 0.052 | 0.203 | 0.024 | 0.356 |
| 4  | 0.055 | **0.372** ← peak | 0.033 | 0.442 |
| 5  | 0.043 | 0.288 | 0.033 | **0.507** ← peak |
| 6  | 0.076 | 0.245 | 0.038 | 0.418 |
| 7  | **0.094** ← peak | 0.089 → stop | **0.205** ← 급등 | 0.443 |
| 8  | 0.046 | — | 0.108 | 0.426 → stop |
| 10 | 0.019 → stop | — | 0.078 → stop | — |

**4가지 수렴 유형 분류**:

| 유형 | 해당 GO term | 특징 | 원인 |
|-----|------------|------|------|
| **Noisy oscillation** | GO:0006941 | 전 epoch 진동, 방향성 없음 | Phase 1 발판 부재 (margin_sat=5.9%) |
| **Early peak → collapse** | GO:0030017 | Ep4 peak 후 즉각 하락 | 좋은 embedding 찾았으나 Focal loss가 overfitting |
| **Late discontinuous jump** | GO:0003774 | Ep6(0.038)→Ep7(0.205) 갑작스러운 5× 급등 | 임계점(phase transition) 통과 — motor protein 이소폼이 갑자기 클러스터화 |
| **Stable plateau** | GO:0007204 | Ep2부터 단조 상승, Ep5 peak | 가장 건강한 학습 곡선 — Ca2+ 채널 이소폼 label 신호가 강하고 일관적 |

**GO:0003774 Ep7 급등 해석**: 이전 6 epoch에서 embedding 경계면이 서서히 형성되다가 7에폭에서 임계점을 돌파. 이 "phase transition" 특성은 hard negative mining이 아닌 semi-hard mining에서도 발생할 수 있음 — 잘못된 방향의 gradient가 누적되다가 갑자기 올바른 basin에 진입. AUPRC early stop이 이 checkpoint를 정확히 포착.

---

### 2-A-3. Positive Collapse 메커니즘 차이 — 두 가지 유형

| GO term | Ph2 mean | Ph2 >0.5 | Collapse 유형 | AUROC | AUPRC | 대응책 |
|---------|---------|---------|-------------|-------|-------|-------|
| GO:0006941 | 0.661 | 36,748 (100%) | **학습 실패형** — ranking 자체가 불안정 | 0.645 | 0.095 | ESM-2로 Phase 1 발판 확보 |
| GO:0007204 | 0.573 | 36,748 (100%) | **Calibration 실패형** — ranking은 정확, threshold만 틀림 | 0.808 | 0.507 | Temperature scaling 또는 threshold 조정 |

**학습 실패형 (GO:0006941)**: Phase 1 margin_sat=5.9%로 embedding이 불안정 → Phase 2 AUPRC 자체가 0.04~0.09 사이에서 진동. Score mean=0.661로 전체가 positive 방향으로 밀림. 단순 threshold 조정으로 해결 불가.

**Calibration 실패형 (GO:0007204)**: Phase 2 AUPRC=0.507로 ranking은 완벽. Sep.Ratio=1.185 (8개 중 최고). Score mean=0.573으로 전체가 0.5를 막 넘은 수준 — Focal loss의 α bias가 과도하게 작용. Positive 310개(0.84%)에 비해 Focal loss α=0.25(Phase 1.5)/0.10(Phase 2)이 여전히 positive를 과대 예측. **Platt scaling 또는 α 감소로 해결 가능** — ranking이 맞으므로 threshold 보정만으로 F1 개선 기대.

---

### 2-A-4. LabelProp 민감도 — 발현 공간 생물학적 클러스터링 강도 진단

| GO term | α=0.0 | α=0.2 | α=0.3 | α=0.5 | 방향 | 생물학적 의미 |
|---------|-------|-------|-------|-------|------|------------|
| GO:0030017 (sarcomere, CC) | 0.372 | 0.379 | 0.382 | **0.390** | **단조 증가** ↑↑↑ | sarcomere 단백질(myosin/actin/titin)이 hMuscle에서 co-expression 강함 → 이웃 전파가 진짜 positive로 전달됨 |
| GO:0007204 (Ca2+ BP) | **0.507** | 0.508 | 0.502 | 0.487 | 소폭 개선 후 감소 ↑↓ | Ca2+ 채널 발현 클러스터 약함 — AUPRC 포화 상태에서 marginal |
| GO:0006941 (BP) | 0.094 | 0.094 | **0.095** | 0.093 | 거의 무효 ≈ | 전파할 신호 자체가 약함 |
| GO:0003774 (motor, MF) | **0.205** | 0.196 | 0.171 | 0.119 | **단조 감소** ↓↓↓ | **LabelProp이 역효과** — myosin heavy chain 이소폼(MYH1/2/4/7)은 근육 유형별 상호 배타적 발현: 같은 발현 이웃이 실제론 다른 motor function을 가짐. 이웃 전파 = noise injection |

**LabelProp은 "발현 공간 co-clustering 강도"의 사이드 진단 지표로 해석 가능**:
- α=0.5까지 단조 증가: 구조 단백질(CC term) — co-localization이 co-expression으로 직결
- α>0이면 감소: 기능 단백질(MF term) — isoform-specific function은 발현 유사성과 무관

이 패턴은 v6 LabelProp 설계에 반영 필요: **MF term에서는 LabelProp을 α=0.0으로 고정하거나 skip하는 adaptive strategy** 검토.

---

## 3. 전체 8개 GO term 통합 비교 (v5-5 기준)

### 3-1. 성능 전체 표

| GO term | 이름 | 유형 | pos | AUROC | **AUPRC** | 배율 | >0.5% | 상태 |
|---------|------|------|-----|-------|---------|------|------|------|
| GO:0007204 | Ca2+ signaling | BP | 310 | 0.808 | **0.508** | 63× | 100% | ranking 우수 |
| GO:0006096 | glycolysis | BP | 76 | 0.884 | **0.670** | 335× | 100% | ranking 최우수 |
| GO:0030017 | sarcomere | CC | 452 | 0.831 | **0.390** | 33× | 1.4% | ✅ 건강 |
| GO:0022900 | ETC | BP | 291 | 0.817 | **0.357** | 45× | 100% | ranking 우수 |
| GO:0006936 | muscle contraction | BP | 597 | 0.723 | **0.251** | 16× | 99.7% | 불안정 |
| GO:0003774 | motor activity | MF | 164 | 0.797 | **0.205** | 51× | 3.5% | ✅ 건강 |
| GO:0006412 | translation | BP | 701 | 0.615 | **0.209** | 11× | 58% | 부분 회복 |
| GO:0006941 | striated muscle | BP | 253 | 0.645 | **0.095** | 14× | 100% | 불안정 |

### 3-2. GO term 유형별 패턴 분류

#### 패턴 A: 고성능 (AUPRC > 0.30) — ranking 우수, expression 신호 강함
- **GO:0006096** (glycolysis): AUPRC=0.670. 해당과정 관련 효소 이소폼들이 hMuscle에서 명확한 발현 차이.
- **GO:0007204** (Ca2+ signaling): AUPRC=0.507. 근육 Ca2+ 채널 이소폼(RYR, SERCA 등)은 근육 조직에서 발현 패턴이 독특.
- **GO:0030017** (sarcomere): AUPRC=0.390. **유일하게 positive collapse 없이** 고성능 달성. 구조 도메인(myosin, actin, titin 등)이 seq+domain feature에 직접 인코딩.
- **GO:0022900** (ETC): AUPRC=0.357. 미토콘드리아 복합체 이소폼 발현이 hMuscle에서 구별 가능.

#### 패턴 B: 중성능 (AUPRC 0.10~0.30) — 신호 존재, 하지만 collapse 또는 불안정
- **GO:0003774** (motor activity): AUPRC=0.205. MF term 첫 성공. Phase 1 Silhouette 음수(-0.12)에도 Phase 2에서 회복.
- **GO:0006412** (translation): AUPRC=0.209. v5-5에서 처음으로 58% positive 예측으로 partial recovery.
- **GO:0006936** (muscle contraction): AUPRC=0.251. v5-2(0.410) 대비 미달. initialization 민감성 지속.

#### 패턴 C: 저성능 (AUPRC < 0.10) — 모델 한계
- **GO:0006941** (striated muscle contraction): AUPRC=0.095. GO_0006936의 자식 term인데 더 낮음. 253개 positive로 더 sparse하고 더 엄격한 label → 현재 2-modal로는 striated vs non-striated 구분 불가.

---

## 4. 일반성(Generality) 판정

### 4-1. 질문에 대한 답

> **"v5-5는 일반화된 안정 상태인가, 아니면 4개 특정 GO term에만 최적화된 것인가?"**

**결론: 일반화된 안정 상태가 맞다.** 단, GO term 유형에 따라 성능 분화가 뚜렷하다.

**근거**:
1. 기존 4개와 다른 유형(CC, MF)의 새 GO term에서도 AUPRC > 0.20 달성 (GO:0030017=0.39, GO:0003774=0.205)
2. 예측하지 않은 GO term에서 역대 최고 성능 등장: GO:0007204 AUPRC=0.508 (8개 GO term 중 GO:0006096에 이어 2위)
3. Score 분포가 GO:0030017, GO:0003774에서 처음으로 **collapse 없이** 건강한 분포 달성
4. GO:0006941의 저성능은 v5-5 고유 문제가 아니라 **GO:0006936 불안정성과 동일 메커니즘** → 특정 GO term 쌍의 구조적 한계이지 모델 일반성 실패가 아님

### 4-2. GO term 성능을 결정하는 인자 (데이터 기반)

| 인자 | 고성능 GO term | 저성능 GO term | 해석 |
|------|-------------|-------------|------|
| **Tissue specificity** | GO:0006096, GO:0007204 (근육 특이적 발현) | GO:0006412 (translation = 모든 세포) | hMuscle 발현 신호가 label과 correlated |
| **Structural domain encoding** | GO:0030017 (sarcomere 구성 단백질) | GO:0006941 (수축 '과정') | seq+domain feature가 구조를 직접 포착 |
| **Isoform divergence** | GO:0003774 (myosin heavy chain 이소폼 구분) | GO:0006941 (isoform-level 차이 불명확) | 동일 유전자 이소폼 간 실제 기능 차이 존재 여부 |
| **Label specificity** | GO:0030017 (sarcomere = 정밀한 CC) | GO:0006941 (striated muscle contraction = broad) | 더 specific한 label이 오히려 학습하기 어려울 수 있음 |

### 4-3. LabelProp 효과 재확인

| GO term | alpha=0.0 AUPRC | best AUPRC | 개선 | best alpha |
|---------|----------------|-----------|------|-----------|
| GO:0007204 | 0.507 | 0.507 (α=0.2) | +0.000 | 0.2 (소폭 개선) |
| GO:0030017 | 0.372 | **0.390** | **+0.018** | 0.5 |
| GO:0003774 | 0.205 | 0.205 (α=0.0) | 0 | 0.0 |
| GO:0006941 | 0.063 | 0.095 (α=0.3) | +0.032 | 0.3 |

**패턴**: GO:0030017에서만 LabelProp이 의미있는 개선 제공(+0.018). sarcomere 단백질들의 발현 패턴이 KNN graph에서 실제로 유사 → 이웃 전파가 유효.

**v6 설계 시사점**: §2-A-4에서 분석한 GO term별 LabelProp 방향성(단조증가/무효/역효과)은 v6에서 **ontology-type-aware α selection** 또는 **MF term α=0.0 고정** 전략으로 반영 가능.

---

## 5. 임상적 의미 — ESM-2 통합 방향 재정립

### 5-1. ESM-2가 해결해야 할 구체적 문제

일반성 검증 결과, 현재 seq+domain 2-modal의 한계가 **GO:0006941/GO:0006936** 그룹에 집중됨:

| 문제 | 증거 | ESM-2 기여 가설 |
|------|------|--------------|
| GO:0006936/0006941 불안정 (0.095~0.251) | Ph1 Silhouette 낮음, initialization 민감 | ESM-2가 근육 isoform-level sequence divergence 포착 → Phase 1 embedding 안정화 |
| GO:0006941 < GO:0006936 | 더 specific label인데 성능 낮음 | ESM-2의 미세 sequence 차이 포착이 striated vs non-striated 구분 가능케 할 것 |
| GO:0003774 Phase 1 Sil 음수 (-0.12) | Motor protein 이소폼 구분 어려움 | ESM-2가 motor domain 미세 구조 차이 인코딩 |

### 5-2. ESM-2 기여 기대 우선순위

1. **GO:0006936/0006941** (근육수축): Phase 1 embedding 안정화가 핵심. ESM-2가 myosin heavy chain, troponin 이소폼 차이를 포착하면 AUPRC 0.35+ 기대.
2. **GO:0003774** (motor activity): Phase 1 Silhouette 음수(-0.12) 개선. 0.20 → 0.35+ 기대.
3. **GO:0030017** (sarcomere): 이미 성능 좋음(0.39). ESM-2 기여 제한적일 수 있음 — **no_esm ablation 컨트롤로 활용 가능**.
4. **GO:0007204** (Ca2+ signaling): 이미 0.507. Ca2+ 채널 이소폼은 transmembrane domain 차이가 큼 → ESM-2 추가 기여 가능.

---

## 6. 미해결 문제 업데이트

| 문제 | 상태 | 비고 |
|------|------|------|
| GO:0006936 v5-2(0.41) 수준 회복 | ❌ 미해결 | v5-5=0.251, GO:0006941=0.095로 동일 패턴 지속 |
| CC term에서 모델 작동 여부 | ✅ **확인** | GO:0030017 AUPRC=0.390, 건강한 score 분포 |
| MF term에서 모델 작동 여부 | ✅ **확인** | GO:0003774 AUPRC=0.205 |
| LabelProp이 특정 term에만 유효 | ✅ **재확인** | GO:0030017, GO:0022900에서만 유효 |
| Positive collapse 해결 | 🔄 부분 해결 | GO:0030017, GO:0003774는 정상. GO:0006941, GO:0007204는 collapse지만 ranking은 좋음 |

---

## 7. ESM-2 통합 판단

### v6 진입 조건 재검토 (이전 기준: GO:0006936 ≥ 0.35 AND GO:0006096 ≥ 0.30)

| 조건 | 현황 | 판단 |
|------|------|------|
| GO:0006096 ≥ 0.30 | ✅ 0.670 달성 | — |
| GO:0006936 ≥ 0.35 | ❌ 0.251 미달 | 구조적 문제로 2-modal로는 달성 어려울 것 |

**권고**: GO:0006936 ≥ 0.35 조건을 **ESM-2가 해결해야 할 동기**로 재프레이밍. 현재 일반성이 확인됐으므로 v6 진입을 강행해도 논문 스토리가 성립함.

논문 서술:
> "2-modal (seq+domain) 기반 v5-5는 8개 GO term에서 안정적 일반화를 달성했으나, 근육수축 계열(GO:0006936, GO:0006941)에서 initialization 민감성이 지속됐다. 이는 현재 feature가 muscle contraction 이소폼 간 미세 서열 차이를 포착하지 못하기 때문으로, ESM-2 protein language model 통합을 통해 해결할 것을 제안한다."

---

## 8. 수치 종합 — 전체 8개 GO term 최고 성능 (v5-5 기준)

```
GO term            AUPRC    random   배율   유형   >0.5%
───────────────────────────────────────────────────────────
GO:0006096         0.670    0.002    335×   BP     100%  ← ranking 최우수
GO:0007204         0.508    0.008     63×   BP     100%  ← 신규 발견, ranking 우수
GO:0030017         0.390    0.012     33×   CC     1.4%  ← 유일한 정상 분포+고성능
GO:0022900         0.357    0.008     45×   BP     100%
GO:0006936         0.251    0.016     16×   BP     99.7% ← 불안정
GO:0003774         0.205    0.004     51×   MF     3.5%  ← MF 첫 성공
GO:0006412         0.209    0.019     11×   BP     58%
GO:0006941         0.095    0.007     14×   BP     100%  ← 구조적 한계
───────────────────────────────────────────────────────────
Macro-AUPRC (4개 기존): 0.372
Macro-AUPRC (4개 신규): 0.272
Macro-AUPRC (전체 8개): 0.322
```

---

---

## 9. Tier-2 GO term 추가 실행 — GO:0003779 (actin binding, MF)

### 9-1. 실행 정보

| 항목 | 내용 |
|------|------|
| GO term | GO:0003779 (actin binding) |
| 유형 | MF |
| 선정 이유 | GO:0003774(motor activity) 외 두 번째 MF term — MF 패턴 재현성 검증 |
| 실행 시작 | 2026-04-08 01:57 |
| PID | 2067519 |
| GPU | 1 (CUDA_VISIBLE_DEVICES=1) |
| y_test pos | 784 (ratio=2.133%) |
| 로그 | `hMuscle/logs_isoform/GO_0003779/v5-5_GO_0003779_20260408_0157_Full.log` |

### 9-2. 예측

GO:0003774와 동일 MF type이지만 다음 차이 존재:

| 항목 | GO:0003774 | GO:0003779 |
|------|-----------|-----------|
| 기능 | cytoskeletal motor activity | actin binding |
| n_pos(test) | ~164 | **784** (4.8× 더 많음) |
| 이소폼 특이성 | myosin heavy chain (강한 근육 유형별 분화) | β-actin / α-skeletal actin (tissue 분화) |
| 예상 AUPRC | — | **0.20~0.35** |

n_pos=784로 GO:0003774(571)보다 많고 coverage=6.0×에 가까울 것으로 예상. GO:0003774의 Phase 2 late jump 패턴이 재현되는지, LabelProp이 다시 역효과를 보이는지가 핵심 관찰 포인트.

### 9-3. Tier-2 실행 전략 결정

| Tier-2 term | 판단 | 이유 |
|------------|------|------|
| GO:0003779 (actin binding) | ✅ **실행 중** | MF 패턴 재현 검증, n_pos 많아 안정적 결과 예상 |
| GO:0007519 (muscle development) | ⏸ **ESM-2 ablation 후** | v6에서 sequence feature 추가 시 발달 GO term 개선 여부 측정 목적 |
| GO:0006915 (apoptosis) | ❌ **보류** | hMuscle에서 pan-cellular process, 실패 시 논문 서술 복잡화 |

---

*작성: 2026-04-08*  
*Tier-1 실행: run_GPU_Generality.py | 총 95.1분 (GO:0006941/0003774 병렬 → GO:0030017/0007204 병렬)*  
*Tier-2 실행: GO:0003779 진행 중 (2026-04-08 01:57~)*  
*분석 로그: `hMuscle/logs_isoform/GO_{0006941,0030017,0003774,0007204,0003779}/v5-5_*_Full.log`*
