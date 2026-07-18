# PRISM+BISECT — 교수님 미팅 핵심 요약

> 작성일: 2026-06-25 | 버전: v1.0

---

## 1. 앱 개요 (무엇을 보여주는가)

### 1.1 연구 배경 (한 줄 요약)
알츠하이머 환자 뇌에서 **어떤 유전자의 어떤 아이소폼이 기능을 잃는지**를
딥러닝(PRISM)으로 예측하고, 다중 생물학적 증거(BISECT)로 검증한 파이프라인.

### 1.2 앱 구성 (8개 페이지)

| 그룹 | 페이지 | 핵심 기능 |
|------|--------|-----------|
| **데이터셋 분석** | QC & Overview | PRISM 예측 점수 분포, 아이소폼 유형(ALE/TSS/AS) 시각화 |
| | Module Landscape | 44개 GO 기능 모듈 × 아이소폼 분포 히트맵 |
| | Functional Patterns | UMAP 군집 — AD vs Control 발현 패턴 비교 |
| | Condition Analysis | DTU (Differential Transcript Usage) 분포 |
| **타겟 분석** | 타겟 탐색 | 유전자/GO/아이소폼 검색 → PRISM 점수 즉시 조회 |
| | 시나리오 & 분석 | S1-S4 시나리오별 후보 유전자 랭킹 |
| | 개별 아이소폼 분석 | CT vs AD 이소폼 쌍 비교, 도메인 맵, PPI 네트워크 |
| | **핵심 케이스 스터디** | **101 BISECT 케이스 전체, 5-Tier 증거 분류, 수렴축 분석** |

---

## 2. PRISM 모델 성능

### 2.1 핵심 수치

| 지표 | 근육 (18 GO) | 뇌 Zero-shot (672 GO) |
|------|-------------|----------------------|
| **Macro AUPRC** | **0.7022** | **0.5998** |
| 비교 기준 (랜덤) | 0.056 | ~0.015 |
| 비교 기준 (유전자 중앙값) | 0.61 | — |
| 개선율 | +15% vs gene-level baseline | 타 조직 직접 이전 |

> **Zero-shot**: 근육 데이터로 훈련한 모델을 재훈련 없이 뇌 데이터에 적용.
> AUPRC 0.60은 672개 GO term 평균 — 근육 대비 교차 조직 일반화 가능성을 시사.

### 2.2 모델 구조

```
ESM-2 (150M, 640-dim) → Dense(256, ReLU) → BN → Dropout(0.3)
                       → Dense(128, ReLU) → Dropout(0.2)
                       → Dense(64, ReLU) → Sigmoid
Loss: Focal Cross-Entropy (γ=2.0)
Ensemble: N seeds 평균
```

---

## 3. BISECT 분석 — 101 케이스 핵심 통계

### 3.1 5-Tier 증거 분류

| Tier | 통계 방법 | 케이스 수 | 설명 |
|------|----------|-----------|------|
| **A-DR** | DRIMSeq + stageR (이중 FDR 보정) | **14** | 최강 통계 근거 |
| **A-BP** | Donor permutation 검증 | **5** | 독립 순열 검정 통과 |
| **B** | 독립 데이터셋 복제 | **1** (KIF21B) | 외부 데이터 교차 검증 |
| **C** | 탐색적 (Pooled χ²) | **80** | 가설 생성 단계 |
| **D** | 통계 미지지 | **1** (DLG1/OPC) | 생물학적 관련성은 높으나 DTU 미유의 |
| **합계** | — | **101** | — |

> **A-DR + A-BP = 19 케이스**: 전체의 18.8% — 엄격한 통계 기준을 통과한 확정 케이스.

### 3.2 A-DR 케이스 통계 (DRIMSeq+stageR, n=14)

- **stageR FDR 범위**: 2.67×10⁻²⁴ ~ 4.93×10⁻²
- **상위 유의**: ZNF736 (p=2.67e-24), ZNF582 (p=7.21e-11), USP1 (p=7.66e-5)
- **7개**: p < 0.01 (강한 근거)
- **14개 모두**: p < 0.05 (이중 FDR 보정 후)
- **이소폼 발현 차이(|Δ|)**: 최대 0.786 (ZNF736), 평균 0.46

| 유전자 | 세포유형 | stageR p | |Δ| | 주요 메커니즘 |
|--------|---------|----------|------|--------------|
| ZNF736 | Excitatory | 2.67e-24 | 0.786 | 8개 Zn-finger 도메인 소실, NNIC 아이소폼 |
| ZNF582 | Excitatory | 7.21e-11 | 0.310 | Alt-TSS, KRAB 전사억제자 소실 |
| USP1 | Oligodendrocyte | 7.66e-5 | — | 탈유비퀴틴화 효소 기능 변화 |
| NDUFAF5 | Excitatory | 1.43e-4 | 0.450 | Complex I 조립 인자, NMD 스위치 |
| DOCK10 | Microglia | 1.15e-3 | 0.573 | Minor exon; GEF 기능 조절 |
| SAMHD1 | Inhibitory | 7.68e-3 | — | NMD + tetramer domain 소실 |
| ERCC6L2 | Astrocyte | 1.11e-2 | — | DEAD-box + helicase 도메인 소실 |

### 3.3 A-BP 케이스 통계 (Donor permutation, n=5)

| 유전자 | 세포유형 | perm p | |Δ| | 비고 |
|--------|---------|--------|------|------|
| DOCK11 | Inhibitory | 0.0008 | 0.717 | 가장 강한 A-BP 근거 |
| NDUFS8 | Inhibitory | 0.0044 | 0.370 | NIC 아이소폼, 미토콘드리아 국소화 스위치 |
| NDUFS4 | Inhibitory | 0.0236 | 0.320 | Complex I NDUS4 도메인 소실 |
| NDUFS4 | Excitatory | 0.041 | 0.563 | 동일 유전자, 2개 세포유형 모두 유의 |
| NDUFS7 | Excitatory | 0.048 | 0.051 | 소량 차이, permutation으로만 검출 |

---

## 4. PRISM 기능 예측 — Tier × PRISM 분류

### 4.1 PRISM Tier (T1/T2/T3) 정의

| PRISM Tier | 기준 | 전체 (n=101) |
|-----------|------|-------------|
| **T1 스위치** | AF2에서 CT→AD 새 기능 구조 획득 (af_gained_confident) | 21 (20.8%) |
| **T2 소실** | AF2에서 CT 기능 구조 소실 (af_lost_confident) | 61 (60.4%) |
| **T3 구조만** | AlphaFold 구조 변화 없음 (RNA-level 메커니즘) | 19 (18.8%) |

> **A-DR 14개 중 대부분이 T3**: RNA-level 메커니즘(NMD, 핵 국소화 이상)이 주도하여
> 단백질 구조 변화 없이도 기능이 소실됨 — DRIMSeq으로만 검출 가능한 이유.

### 4.2 대표 케이스 PRISM 기능 예측

**ZNF736 (A-DR, Excitatory) — 전사 조절 기능 완전 소실**
- CT 아이소폼 TOP GO: `regulation of DNA-templated transcription` (score=0.534)
- AD 아이소폼 TOP GO: `reproductive process` (score=0.210) ← **무관한 기능으로 전환**
- 소실: `regulation of transcription by RNA pol II` Δ= **−0.446**
- 획득: `exocytosis` Δ=+0.196, `lipid metabolic process` Δ=+0.163
- 해석: KRAB + 8×Zn-finger 소실 → 전사억제 기능 완전 소실 → 비특이적 기능 신호

**NDUFS4 (A-BP, Excitatory) — 에너지 대사 → 지질 대사로 기능 전환**
- CT TOP GO: `cellular respiration` (score=0.597)
- AD TOP GO: `lipid biosynthetic process` (score=0.339)
- 소실: `cellular respiration` Δ= **−0.538**, `electron transport chain` Δ=−0.480
- 획득: `lipid biosynthetic process` Δ=+0.227, `regulation of lipid metabolic process` Δ=+0.214
- 해석: NDUS4 도메인 소실 → Complex I Q-module 불안정 → 에너지 생산 기능 소실

---

## 5. 6대 생물학적 수렴축

### 5.1 수렴축 요약

| 수렴축 | 유전자 | A-tier 수 | 핵심 메커니즘 |
|--------|--------|-----------|--------------|
| **Complex I Assembly** | NDUFAF5, NDUFS4(×2), NDUFS7, NDUFS8 | 5/7 (71%) | ETC 붕괴, 에너지 대사 → 지질 대사 전환 |
| **RNA Metabolism** | ZNF736, ZNF582, DDX19A, DIS3, CNOT11 | 5/5 (100%) | mRNA 분해/안정화 조절 소실 |
| **DNA Repair / 게놈 안정성** | SAMHD1, ERCC6L2, USP1, NOL8 | 4/4 (100%) | NMD, DEAD-box 헬리케이스 소실 |
| **DOCK-GEF (세포 이동)** | DOCK10 (Microglia), DOCK11 (Inhibitory) | 2/2 (100%) | GEF 도메인 minor exon 조절 |
| **KRAB-ZFP** | ZNF736, ZNF582, DCAF5 | 3/3 (100%) | KRAB 전사억제 구조 소실 |
| **Ub-Proteasome** | USP1, RPS3, AZIN1 | 3/3 (100%) | 단백질 분해 경로 조절 이상 |

> **핵심 통계**: 모든 수렴축에서 A-tier 비율이 높음 → 개별 우연이 아닌 경로 수준의 수렴.

### 5.2 Complex I 축 상세 (NDUFS4/7/8/NDUFAF5)

```
N-module         Q-module         Membrane arm
NDUFAF5(조립)    NDUFS7           NDUFS8
NDUFS4           (전자 릴레이)    (미토콘드리아 국소화)
   ↓  A-DR           ↓  A-BP          ↓  A-BP
   NMD switch    Δ=-0.051        Δ=-0.37 (perm p=0.004)
                 perm p=0.048    NIC 아이소폼 → 국소화 결함
```

**4개 유전자가 동일 복합체 내 서로 다른 위치에서 독립적으로 수렴** → 단일 통계 기술 오류 가능성 배제.

---

## 6. BISECT 메커니즘 분류

### 6.1 M14 이벤트 유형 (아이소폼 구조 변화)

| 이벤트 | 케이스 수 | 설명 |
|--------|---------|------|
| ALE (Alternative Last Exon) | 36 | 3' 끝 엑손 교체 → TTS 이동 |
| Exon Exchange | 25 | 내부 엑손 전환 |
| Major Exon Loss | 18 | 대형 엑손 소실 |
| Cassette Exon Cluster | 11 | 다중 엑손 클러스터 생략 |
| Retained Intron | 8 | 인트론 보유 → NMD 타겟 |

### 6.2 M15 NMD 스위치 (Nonsense-Mediated mRNA Decay)

- **21개 케이스**: AD 아이소폼이 NMD 타겟으로 전환 → 실제 단백질 감소
- **대표**: NDUFAF5 (A-DR), KIF21B (Tier B), RGS3, BSG, EGFR
- 이 케이스들에서 AlphaFold 구조 변화 없음 → T3로 분류 → RNA-level 메커니즘만 존재

### 6.3 M16 최종 메커니즘 분류

| 메커니즘 | 케이스 수 |
|---------|---------|
| ALE (Alternative Last Exon) | 32 |
| NMD Switch | 21 |
| Domain Loss | 15 |
| Major Truncation | 14 |
| Domain Gain | 8 |
| Exon Exchange | 5 |
| Retained Intron | 3 |

---

## 7. C18 클러스터 — 핵심 발견

### 7.1 C18 특성
- **위치**: Excitatory neuron, L4 IT 비정형 클러스터
- **세포 수**: 1,603개 (24개 도너)
- **AD 농축**: MWU p=0.00994 (Δ+4.57%) — **전체 30개 클러스터 중 유일하게 유의한 AD 농축**

### 7.2 C18 내 Complex I 이소폼 비율 (AD vs Control)

| 유전자 | C18 (AD−CT) | C10 L4 (AD−CT) | C11 L4 (AD−CT) | 해석 |
|--------|------------|----------------|----------------|------|
| NDUFS8 | **+0.500** | −0.338 | −0.103 | C18 특이적 AD 축적 (+0.72 차이) |
| ZNF736 | +0.500 | +0.500 | +1.000 | 전반적 AD 방향 일치 |
| DCAF5 | +1.000 | +0.756 | +0.608 | C10 유의 (MWU p=0.016) |
| RPS3 | +0.262 | +0.319 | — | 일관된 방향 |
| NDUFS7 | +0.044 | +0.025 | +0.096 | 소량 차이 (perm p=0.048과 일치) |

> **NDUFS8 C18 MWU p=0.31** (n=2,2 underpowered) — 방향은 명확하나 통계적 유의성 부족.
> **DCAF5 통계 유의**: C10 p=0.016, C11 p=0.038.

### 7.3 NDUFS8 C18 발견의 의미
NDUFS8 NIC 아이소폼이 정상 L4(C10/C11)에서는 AD에서 오히려 감소(-0.22)하지만,
AD 농축 C18 클러스터에서는 AD에서 증가(+0.50) → C18 세포가 NDUFS8 이소폼 전환에
특별히 취약한 세포 집단임을 시사.

---

## 8. 통계 방법론 요약 (방법론 질문 대비)

### 8.1 DTU 통계 계층

```
Tier A-DR: DRIMSeq (Dirichlet-Multinomial) + stageR (이중 FDR)
           → 도너를 생물학적 반복으로 처리 (N=24~30) → 위음성 방지
           → 유전자 수준 + 전사체 수준 FDR 동시 보정

Tier A-BP: Donor permutation (N=1,000회)
           → 도너 라벨 섞기 → 실제 관찰값 분위수 계산
           → 세포 수준 가변성에 독립적

Tier C:    Pooled χ² test (탐색적)
           → 도너별 반복 없음 → 위음성 가능성 있음 → 가설 생성용
```

### 8.2 왜 DRIMSeq인가?

기존 Pooled χ²는 **세포를 독립 반복으로 취급** → 도너 내 세포들이 상관됨에도 독립으로 가정 → 위양성 과다 (pseudoreplication).
DRIMSeq는 **도너를 반복 단위**로 설정 → 현실적 검정력 확보.

### 8.3 PRISM AUPRC 해석

- **AUPRC (Area Under Precision-Recall Curve)**: 불균형 클래스에 적합 (GO term당 positive 비율 < 5%)
- AUROC와 달리 false positive에 민감 → 보수적 지표
- 0.7022 vs 랜덤 0.056 → 실질적 예측력 존재

---

## 9. 한계 및 솔직한 인정

| 한계 | 내용 | 현재 대응 |
|------|------|-----------|
| **Tier C 가설** | 80개 케이스는 탐색적 (pseudoreplication) | 5-tier 시스템으로 명확히 구분 표시 |
| **C18 MWU 미유의** | NDUFS8 C18 p=0.31 (n=2,2) | 방향성은 보고, 유의성은 인정 안 함 |
| **Zero-shot 한계** | AUPRC 0.60 — 672개 term 중 일부는 낮음 | 방법론적 가능성 증명에 초점 |
| **T3 PRISM delta 소** | A-DR 대부분 delta_max < 0.1 | RNA-level 메커니즘 설명으로 대체 |
| **NNIC 아이소폼** | 신규 비정형 서열 — annotation 미확인 | ESMFold 직접 구조 예측으로 보완 |

---

## 10. 핵심 숫자 요약 (한눈에 보기)

```
┌─────────────────────────────────────────────────────────┐
│  PRISM 모델 성능                                         │
│    Macro AUPRC (근육): 0.7022                           │
│    Macro AUPRC (뇌, zero-shot): 0.5998                  │
│    GO terms: 18 (근육) / 672 (뇌)                       │
│                                                         │
│  BISECT 케이스 (총 101개)                               │
│    Tier A-DR: 14 (DRIMSeq+stageR)                       │
│    Tier A-BP:  5 (donor permutation)                    │
│    Tier B:     1 (독립 복제)                            │
│    Tier C:    80 (탐색적)                               │
│    Tier D:     1 (미지지)                               │
│                                                         │
│  6대 수렴축 (A-tier 케이스 비율)                        │
│    Complex I: 5/7 (71%)                                 │
│    RNA Metabolism: 5/5 (100%)                           │
│    DNA Repair: 4/4 (100%)                               │
│    DOCK-GEF: 2/2 (100%)                                 │
│                                                         │
│  C18 L4-IT 클러스터                                     │
│    AD 농축: MWU p=0.00994, Δ+4.57%                     │
│    NDUFS8 C18 이소폼 전환: AD+0.50 vs L4 평균−0.22     │
│    DCAF5 C10 유의: MWU p=0.016                         │
│                                                         │
│  NMD Switch 케이스: 21/101 (20.8%)                      │
│  Pfam 도메인 소실 케이스: 77/101 (76.2%)               │
└─────────────────────────────────────────────────────────┘
```

---

## 부록: 주요 용어 빠른 참조

| 용어 | 의미 |
|------|------|
| **아이소폼** | 같은 유전자에서 대안적 스플라이싱으로 만들어진 서로 다른 mRNA/단백질 |
| **DTU** | Differential Transcript Usage — 동일 유전자 내 아이소폼 비율 변화 |
| **PRISM** | ESM-2 임베딩 기반 아이소폼 기능 예측 딥러닝 모델 |
| **BISECT** | 15-모듈 생물학적 증거 수집 파이프라인 |
| **DRIMSeq** | Dirichlet-Multinomial 회귀 기반 DTU 검정 (도너=반복 단위) |
| **stageR** | 유전자 + 전사체 수준 이중 FDR 보정 프레임워크 |
| **NMD** | Nonsense-Mediated mRNA Decay — 조기 종결 코돈 포함 mRNA 분해 |
| **NNIC/NIC** | Novel Non-In-Catalogue / Novel In-Catalogue — 신규 발견 아이소폼 |
| **AUPRC** | Area Under Precision-Recall Curve — 불균형 데이터 분류 성능 지표 |
| **A-tier** | A-DR + A-BP = 최고 통계 근거 케이스 (전체 19개, 18.8%) |
| **C18** | Excitatory neuron L4 IT 비정형 클러스터 — 유일한 유의 AD 농축 클러스터 |
| **Zero-shot** | 타 조직(근육) 훈련 모델을 재훈련 없이 뇌에 직접 적용 |
