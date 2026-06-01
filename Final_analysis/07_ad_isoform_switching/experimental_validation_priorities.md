# Experimental Validation Priorities
**DIFFUSE AD Isoform Switch Findings — 2026-05-22**

---

## 원칙: 실험 선정 기준

각 실험의 우선순위는 다음 3가지를 곱한 값으로 결정:
- **결정력(Decisiveness)**: 이 실험이 YES/NO를 주는가, 아니면 힌트만 주는가
- **비용(Cost)**: 시간 + 재료 + 장비 (낮을수록 좋음)
- **영향력(Impact)**: 결과가 논문에서 몇 개 발견을 검증하는가

**우선순위 공식**: Priority = Decisiveness × Impact / Cost

---

## TIER 1: 존재 증명 — 가장 싸고, 가장 결정적

### [E1] tr292978 / tr73243 / tr319500 단백질 존재 확인 (proteomics)
**목표**: 3개 novel AD isoform이 실제 단백질로 번역되는가?

**실험**: Samsung AD 코호트 동결 bulk 뇌 조직 → TMT 정량 proteomics (LC-MS/MS)
- 분석: PTPRF-specific peptides, tr292978-specific peptides (C-terminal WD40 영역),
  tr73243 RVT_1-specific peptides (aa141-366), tr319500 specific peptides (L27 region)
- 참조 데이터: ProteomicsDB / AlzPED에 이미 있는 AD 뇌 proteome data

**예상 소요**: 4-8주, $5,000-15,000 (core facility)

**결정력**: ★★★★★ — 단백질이 없으면 모든 메커니즘 가설은 무효
**영향력**: 모든 3개 priority case + PTPRF AD isoform
**주의**: peptide uniqueness 확인 필수 (tr292978 WD40 영역이 canonical KIF21B와 겹치지 않아야)

---

### [E2] 독립 코호트 replication (RNA level)
**목표**: DTU 발견이 Samsung 코호트 특이적인가, 아니면 일반적인가?

**실험**: ROSMAP (Rush Alzheimer's Disease Center) snRNA-seq 데이터 → pseudo-bulk DTU testing
- 이미 공개 데이터: Mathys et al. 2019 (Cell), Blanchard et al. 2022 (Nat Neurosci)
- 분석: transcript-level quantification이 있는 long-read 데이터 필요
- 단기 대안: Seattle Alzheimer's Disease Brain Cell Atlas (SEA-AD) long-read snRNA-seq 데이터

**예상 소요**: 2-4주 (데이터 공개 시), 외부 공동연구 시 3-6개월

**결정력**: ★★★★★ — replication 없으면 모든 발견은 single-cohort finding
**영향력**: 전체 AD isoform switch section의 신뢰도

---

## TIER 2: 메커니즘 증명 — 핵심 가설 검증

### [E3] KIF21B tr292978 — KIF21B-201과의 heterodimerization 확인
**목표**: dominant-negative transport 모델 검증

**실험 A (Co-IP)**:
- HEK293 또는 iPSC-derived excitatory neuron에 tr292978-HA + KIF21B-201-GFP 과발현
- Anti-HA immunoprecipitation → anti-GFP western blot
- 예상 결과: tr292978-HA가 KIF21B-201-GFP를 pull-down → heterodimerization 확인

**실험 B (Single-molecule transport)**:
- In vitro kinesin motility assay: polarity-marked MTs on TIRF microscope
- KIF21B-201 alone vs. KIF21B-201 + tr292978 (1:1 molar ratio) → velocity + run length 측정
- 예상 결과: tr292978 존재 시 velocity 감소, run length 감소 (processivity 손상)

**예상 소요**: A = 4-6주; B = 8-12주 (TIRF setup 필요)

**결정력**: ★★★★☆ (Co-IP는 직접적; TIRF은 가장 강력)
**영향력**: KIF21B mechanism section 전체

---

### [E4] NDUFS4 tr73243 — LINE-1 antisense promoter 활성 확인
**목표**: L1PA3 ASP가 실제로 tr73243 전사를 구동하는가?

**실험 A (Luciferase reporter)**:
- L1PA3 서열 (-500 bp to +200 bp around chr5:53,685,456-53,686,732) → Luciferase reporter
- 형질전환: excitatory neurons (iPSC-derived) vs. fibroblasts
- 예상 결과: Excitatory neuron에서 AD 조건에서 Luciferase 활성 증가

**실험 B (ATAC-seq in AD neurons)**:
- iPSC-derived excitatory neurons (AD vs. CT donor)
- ATAC-seq: chr5:53,685,000-53,687,000 영역 open chromatin 확인
- 예상 결과: AD 뉴런에서 L1PA3 ASP 영역의 chromatin accessibility 증가

**예상 소요**: A = 6-8주; B = 8-12주

**결정력**: ★★★★☆
**영향력**: NDUFS4 mechanism의 "LINE-1 antisense promoter" 주장

---

### [E5] NDUFS4 tr73243 — 미토콘드리아 수입 실패 확인
**목표**: MTS 부재 예측이 실제로 미토콘드리아 국지화 실패로 이어지는가?

**실험**: GFP fusion import assay
- tr73243-GFP + NDUFS4-201-RFP → HeLa cell 또는 primary neuron 과발현
- Confocal: MitoTracker (미토콘드리아 marker)와의 co-localization 측정
- 예상 결과: NDUFS4-201-RFP는 MitoTracker와 overlap, tr73243-GFP는 분산

**예상 소요**: 4-6주

**결정력**: ★★★★★ (직접적, 영상으로 명확)
**영향력**: NDUFS4 mechanism의 핵심 — "MTS absent → Complex I 기여 불가"

---

### [E6] PTPRF AD isoform — 분비 확인
**목표**: ENST00000617451 (262 aa)이 실제로 분비되는가?

**실험**:
- 재조합 단백질 발현: aa1-262 with His-tag in HEK293
- 세포 배양 상청액 + 세포 용해물을 분리하여 anti-His western blot
- N-말단 시퀀싱으로 signal peptide 절단 위치(aa26) 확인
- 예상: 상청액에서 ~28 kDa 밴드 (236 aa 성숙 단백질)

**실험 B (CSF 검색)**:
- AD CSF 샘플에서 PTPRF Ig1-Ig2 특이적 peptide (aa32-225 영역) LC-MS/MS 검색
- 기존 공개 AD CSF proteomics 데이터 (Higginbotham et al. 2020, Nat Neurosci)

**예상 소요**: 4-6주

**결정력**: ★★★★★
**영향력**: PTPRF "secreted decoy" 모델의 전제 조건

---

### [E7] DLG1 tr319500 — Lin7 결합 확인
**목표**: tr319500이 L27 도메인을 통해 Lin7/MALS과 결합하는가? (OPC-specific 기능 증명)

**실험 A (Co-IP)**:
- OPC 세포주 (HOG 또는 Oli-neu) 또는 iPSC-derived OPC
- tr319500-HA 과발현 → anti-Lin7 immunoprecipitation → anti-HA western blot
- 예상: tr319500-HA가 Lin7 pulldown에서 검출됨 (L27 기능 확인)

**실험 B (OPC dedifferentiation recapitulation)**:
- tr319500 siRNA knockdown in OPC → RNA-seq
- Does knockdown of tr319500 shift OPC transcriptomic state toward "AD OPC" (canonical DLG1 up)?
- 예상: tr319500 감소 → canonical DLG1 증가 → OPC 탈분화 마커 출현

**예상 소요**: A = 4-6주; B = 10-16주

**결정력**: ★★★★☆
**영향력**: DLG1 OPC dedifferentiation 메커니즘

---

## TIER 3: 교차 케이스 / 테마 검증

### [E8] IFT122-KIF21B WD40 거울 패턴 — APC/C 결합 테스트
**목표**: tr292978 WD40이 실제로 APC/C와 상호작용하는가?

**실험**:
- Proximity ligation assay (PLA) in neurons: tr292978-HA vs. APC3 (APC/C subunit) antibody
- 또는 BioID proximity labeling: tr292978-BioID → streptavidin pulldown → LC-MS/MS (find APC/C subunits)
- 예상: tr292978가 APC3/ANAPC3, ANAPC4와 PLA 신호를 보임

**예상 소요**: 10-14주

**결정력**: ★★★☆☆ (PLA는 공간적 근접성만; pulldown이 더 결정적)
**영향력**: WD40 redistribution 가설 (투기적 현재)

---

### [E9] Spectrin 손실 테마 — DMD isoform switch functional consequence
**목표**: AD DMD isoform(604 aa, Spectrin-less)이 DAPC 복합체 조립을 저해하는가?

**실험**:
- iPSC-derived inhibitory neurons: 내생 DMD의 knockdown + 재발현 (ENST00000541735 vs. ENST00000682600)
- Dystrophin-associated glycoprotein complex (DGC) co-IP: β-dystroglycan, syntrophin pull-down
- 예상: Spectrin-less isoform은 DGC 조립 효율 감소

**예상 소요**: 16-24주 (iPSC differentiation + knockdown)

**결정력**: ★★★☆☆ (복잡한 실험 시스템)
**영향력**: Spectrin convergence theme

---

### [E10] PTPRF secreted fragment — 경쟁 결합 assay
**목표**: 분비된 AD PTPRF Ig 단편이 NGL-3/Slitrk 결합에서 경쟁하는가?

**실험**:
- 재조합 단백질 3종: AD-PTPRF-Ig (aa27-225), canonical PTPRF Ig1-3, NGL-3 ectodomain
- SPR (Surface Plasmon Resonance): NGL-3를 chip에 고정, 두 PTPRF 단백질의 결합 kinetics 측정
- 경쟁 assay: canonical PTPRF + 증가하는 농도의 AD fragment → NGL-3 결합 IC50

**예상 소요**: 12-16주 (단백질 생산 + SPR 설정)

**결정력**: ★★★★☆
**영향력**: PTPRF decoy 메커니즘

---

## 우선순위 매트릭스

| 실험 | 결정력 | 비용 | 영향력 | Priority Score | 추천 순서 |
|------|--------|------|--------|---------------|---------|
| E2 Replication (RNA) | ★★★★★ | 낮음 | 전체 | **최고** | **1순위** |
| E1 Proteomics (단백질 존재) | ★★★★★ | 중간 | 4개 case | **최고** | **2순위** |
| E5 tr73243 mito import (GFP) | ★★★★★ | 낮음 | NDUFS4 | **높음** | **3순위** |
| E6 PTPRF secretion | ★★★★★ | 낮음 | PTPRF | **높음** | **4순위** |
| E3A KIF21B co-IP | ★★★★☆ | 낮음 | KIF21B | **높음** | **5순위** |
| E7A DLG1 Lin7 co-IP | ★★★★☆ | 낮음 | DLG1 | **높음** | **6순위** |
| E4B ATAC-seq (L1PA3 ASP) | ★★★★☆ | 중간 | NDUFS4 | 중간 | 7순위 |
| E3B TIRF transport assay | ★★★★★ | 높음 | KIF21B | 중간 | 8순위 |
| E10 PTPRF NGL-3 SPR | ★★★★☆ | 높음 | PTPRF | 중간 | 9순위 |
| E8 APC/C BioID | ★★★☆☆ | 높음 | WD40 theme | 낮음 | 10순위 |
| E9 DMD DAPC co-IP | ★★★☆☆ | 매우 높음 | Spectrin | 낮음 | 11순위 |

---

## 제안하는 실험 로드맵 (18개월)

### Phase 1 (M1-3): 존재 증명
- E2: 공개 데이터 replication (in silico, 즉시 가능)
- E5: tr73243 GFP import assay (낮은 비용, 높은 결정력)
- E6: PTPRF recombinant secretion (낮은 비용, 핵심 전제)

### Phase 2 (M3-9): 메커니즘 핵심
- E1: TMT proteomics (AD bulk brain tissue → 3개 isoform 동시 확인)
- E3A: KIF21B co-IP (heterodimerization)
- E7A: DLG1 Lin7 co-IP (L27 기능)

### Phase 3 (M9-18): 세부 메커니즘
- E4B: ATAC-seq (L1PA3 chromatin accessibility)
- E3B: TIRF motility assay (transport velocity)
- E10: PTPRF SPR competition assay

### Phase 4 (M18+): 교차 케이스 / 테마
- E8: BioID (APC/C interaction)
- E9: DMD DAPC assembly

---

## 가장 빠른 논문 강화 경로

Nature Methods/NMI 리뷰어는 computational 발견에 대해 최소 1-2개의 실험 검증을 요구할 것.

**가장 싸고 빠르게 논문에 추가 가능한 실험 (4-6주):**

1. **E5 (tr73243 GFP import)**: 단순 형광 영상. 결과가 명확 (mito = red, tr73243-GFP = green).
   If GFP는 cytoplasmic → "MTS absent → Complex I 기여 불가" 주장이 실험으로 확인.

2. **E6 (PTPRF secretion western)**: HEK293 + 조건 배지. 1-2주면 western 결과 나옴.
   If band in media → "secreted Ig decoy" 전체 모델의 기반 확립.

이 두 실험은 논문 Fig. 8 (validation) 에 추가하여 computational → experimental의 가교 역할 가능.
