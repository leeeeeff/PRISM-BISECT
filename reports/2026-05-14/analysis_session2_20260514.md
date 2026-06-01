# 분석 레포트 Session 2 — 2026-05-14

작성: 자동 생성 (세션 종료 시점 기준)

---

## 1. 이번 세션 작업 목록

| 작업 | 상태 |
|------|------|
| Bootstrap CI 계산 (Gene-block, N=1000) | ✅ 완료 |
| UMAP 임베딩 시각화 (4 GO term × Type A/B) | ✅ 완료 |
| 클러스터 core 이소폼 선정 (centroid 최근접 Top-10) | ✅ 완료 |
| 문헌 탐색 (12개 대표 유전자 생물학적 검증) | ✅ 완료 |
| CORE_META 문헌 근거로 업데이트 | ✅ 완료 |
| fig_cluster_core_table.png 재생성 | ✅ 완료 |
| 전체 Figure 세트 완성 (21개 PNG) | ✅ 완료 |

---

## 2. Bootstrap CI 결과 (Gene-block, N=1000, seed=42)

### 방법론
- **Gene-block bootstrap**: 36,748개 이소폼이 아닌 12,709개 유전자 단위로 리샘플링
- **목적**: 같은 유전자 내 이소폼 간 label 누출(isoform leakage) 방지
- **CI**: 백분위법 [2.5%, 97.5%] — 95% confidence interval
- **유의성**: 단측 검정 (A > B), paired bootstrap Δ = AUPRC(A) − AUPRC(B)

### GO term별 95% CI 요약

#### GO:0007204 — Ca²⁺ Signaling (Type-B)

| 모델 | Point | 95% CI |
|------|-------|--------|
| v8b | 0.1462 | [0.1007, 0.2033] |
| **P3-512** | **0.4055** | **[0.3083, 0.5065]** |
| D256 | 0.1947 | [0.1397, 0.2689] |
| LR | 0.4138 | [0.3092, 0.5356] |

**유의성 검정:**
- P3-512 vs v8b: Δ=+0.2592, **p<0.001** ✓ (통계적으로 유의)
- P3-512 vs LR: Δ=−0.0083, p=0.607 (차이 없음 — LR과 동등)

#### GO:0030017 — Sarcomere (Type-B)

| 모델 | Point | 95% CI |
|------|-------|--------|
| v8b | 0.1570 | [0.1054, 0.2280] |
| **P3-512** | **0.3553** | **[0.2595, 0.4618]** |
| D256 | 0.1366 | [0.1013, 0.1796] |
| LR | 0.5609 | [0.4480, 0.6653] |

**유의성 검정:**
- P3-512 vs v8b: Δ=+0.1983, **p<0.001** ✓ (통계적으로 유의)

#### GO:0006096 — Glycolysis (Type-A)

| 모델 | Point | 95% CI |
|------|-------|--------|
| v8b | 0.7945 | [0.6092, 0.9082] |
| **D256** | **0.8331** | **[0.6665, 0.9406]** |
| P3-512 | 0.4939 | [0.2851, 0.7116] |
| LR | 0.6949 | [0.4662, 0.8823] |

**유의성 검정:**
- D256 vs LR: Δ=+0.1382, p=0.055 (**경계선, 유의하지 않음**)
- D256 vs v8b: Δ=+0.0386, p=0.114 (유의하지 않음)

> ⚠️ **논문 기술 주의**: GO:0006096 D256=0.8331 > LR=0.6950이지만 p=0.055 → "통계적 우위" 주장 불가. "trend toward superiority" 또는 "numerically higher, 95% CI includes zero" 표현 사용 필요.

#### GO:0003774 — Motor Activity (Type-A)

| 모델 | Point | 95% CI |
|------|-------|--------|
| v8b | 0.5686 | [0.4440, 0.7071] |
| **P3-512** | **0.6192** | **[0.4574, 0.7672]** |
| D256 | 0.5982 | [0.4551, 0.7350] |
| LR | 0.8253 | [0.7294, 0.9090] |

**유의성 검정:**
- P3-512 vs v8b: Δ=+0.0506, p=0.097 (유의하지 않음)

#### GO:0006941 — Muscle Contraction (Type-B)

| 모델 | Point | 95% CI |
|------|-------|--------|
| v8b | 0.1177 | [0.0691, 0.2041] |
| **P3-512** | **0.1954** | **[0.1096, 0.3153]** |
| D256 | 0.1567 | [0.0983, 0.2421] |
| LR | 0.3124 | [0.1954, 0.4391] |

### 핵심 통계적 결론

| 주장 | 통계적 지지 수준 |
|------|----------------|
| P3-512 >> v8b (Type-B GO terms: 0007204, 0030017) | **p<0.001** ★★★ |
| D256 > LR (GO:0006096) | p=0.055 *(경계선, △±SE만 보고)* |
| Selective ensemble > v8b (Macro) | 개별 GO 합산 — 논문에서 별도 검정 권장 |

---

## 3. UMAP 임베딩 시각화 결과

### 방법론
- **입력**: Phase-2 unified embeddings (36,748 이소폼 × dim)
  - Type-A (D256): 256-dim, SP✓
  - Type-B (P3-512): 512-dim, SP✗
- **전처리**: PCA 50d → UMAP 2d (cosine metric, n_neighbors=30, min_dist=0.1, seed=42)
- **서브샘플**: 음성 최대 5000개 + 전체 양성 (속도 최적화)
- **Cluster core**: positive centroid에서 L2 거리 최소 Top-10 이소폼

### GO term별 클러스터 구조 요약

| GO Term | Type | 양성 수 | Core 유전자 | 클러스터 분리도 |
|---------|------|---------|------------|--------------|
| GO:0006096 | A | 76 | PFKP, PFKL, PKM, PFKM | **뚜렷한 분리** — 단일 밀집 클러스터 |
| GO:0003774 | A | 164 | KIF26A, KIF21A, MYO10, KIF3C, MYH7 | 분산형 — 다중 motor 서브클러스터 |
| GO:0007204 | B | 310 | F2R, LPAR6, CX3CR1, APLNR, CMKLR1 | **확산형** — GPCR 다양성 반영 |
| GO:0030017 | B | 452 | ANK3, ACTN2, PPP1R12B, LDB3, MYPN | 중간 — 근육 단백질 중심 |

---

## 4. 문헌 검증 결과 — 클러스터 Core 대표 이소폼

### 4-1. GO:0006096 — Glycolysis (Type-A, D256, AUPRC=0.8331)

#### PFKP — Phosphofructokinase (Platelet isoform)
- **실제 조직**: 혈소판, 뇌 (non-muscle)
- **근육 관련성**: 부분적. 성인 골격근의 PFK는 PFKM이 지배적 (M4 homotetramer). PFKP는 발달 중 또는 근위축 시 일부 발현
- **GO:0006096 annotation 타당성**: 효소 기능 자체는 정확 (glycolysis 촉매)
- **핵심 참고문헌**: PMC12125951

#### PFKL — Phosphofructokinase (Liver isoform)
- **실제 조직**: 간, 신장 (hepatic L4 homotetramer)
- **근육 관련성**: 없음. 성인 근육에서 PFKM 독점. PFKL의 특이 구조(C-말단 도메인의 필라멘트 중합)는 PFKM에 없음
- **GO:0006096 annotation 타당성**: 촉매 반응은 동일하나 조직 맥락은 비근육
- **핵심 참고문헌**: PMC5551721, OMIM 171860

#### PKM — Pyruvate Kinase M (PKM2 splice isoform)
- **실제 조직**: PKM2 = 태아/증식 세포/종양; PKM1 = 성인 심근/뇌/골격근
- **근육 관련성**: PKM1이 성인 근육형 (exon 9 포함), PKM2는 exon 10 포함 (태아형)
- **핵심 발견**: 임베딩이 PKM1이 아닌 PKM2 이소폼(ENST00000319622.10)을 캡처 → SwissProt training corpus에 PKM2(태아/종양) annotation이 다수 포함됨을 시사
- **핵심 참고문헌**: PMC7061896, PMC11030359

**GO:0006096 결론**: 효소 기능 정체성(glycolysis)은 보존되나, 클러스터 core가 비근육 이소폼(PFKP platelet, PFKL liver, PKM2 fetal)으로 구성 → 부분적 Tissue Context Mixing 존재. 그럼에도 Type-A 구조 덕분에 AUPRC=0.8331 달성 (SP 조직 정보가 glycolysis 클러스터 통합에 기여).

---

### 4-2. GO:0007204 — Ca²⁺ Signaling (Type-B, P3-512, AUPRC=0.4055)

> **Tissue Context Mixing (TCM) 가설의 핵심 실증 증거**

#### F2R / PAR1 — Protease-Activated Receptor 1 (Thrombin Receptor)
- **실제 조직**: 혈소판, 혈관 내피세포, 백혈구, 신경
- **Ca²⁺ 기전**: Gαq → PLCβ → IP3 → ER Ca²⁺ 방출 + DAG/PKC
- **골격근 역할**: 없음 (골격근 고유 Ca²⁺ 신호: RYR1, SERCA, CaMKII)
- **TCM 증거**: SwissProt에 Ca²⁺ signaling annotation은 정당하지만, 조직 맥락은 platelet/vascular — 근육 Ca²⁺ 학습에 noise
- **핵심 참고문헌**: Nature 2000 (thrombin signalling), PMC18009

#### LPAR6 — Lysophosphatidic Acid Receptor 6
- **실제 조직**: 지방, 피부, 비장, 림프절, 편도, 골수, 충수 (immune/dermal)
- **Ca²⁺ 기전**: Gα12/13-Rho, cAMP/PKA, Ca²⁺-PKC
- **주요 생리 기능**: 모낭 형태형성 (TACE-TGFα-EGFR axis); 기능 상실 돌연변이 → 양털 모발/무모증
- **골격근 역할**: 없음 (HPA 확인)
- **핵심 참고문헌**: Nat Signal Transduct Targeted Ther 2020, OMIM 609239

#### CX3CR1 — CX3C Chemokine Receptor 1 (Fractalkine Receptor / GPR13)
- **실제 조직**: 단핵구, CD8⁺ T세포, NK세포, 조직대식세포, 소교세포, 수지상세포
- **Ca²⁺ 기전**: PLC/PKC 및 PI3K/AKT
- **HPA 데이터**: 골격근 단백질 발현 없음 (absent)
- **근육과의 간접 연관**: CX3CR1⁺ 대식세포가 근육 손상 후 침윤하여 재생 촉진 — 수용체 자체는 근세포에서 발현되지 않음
- **TCM 증거**: 면역세포 전용 Ca²⁺ receptor가 근육 Ca²⁺ 학습 corpus에 포함
- **핵심 참고문헌**: PMC5833124

**GO:0007204 결론 (TCM 완전 확인)**:
세 유전자 모두 Gαq 경로 Ca²⁺ signaling annotation은 SwissProt에 생물학적으로 정당하게 기재됨. 그러나 조직 맥락은 각각 혈소판/혈관(F2R), 림프/피부(LPAR6), 면역/CNS(CX3CR1)로 **골격근과 완전히 무관**. 임베딩 centroid가 **비근육 GPCR 공간에 고정**되어 근육 특이적 Ca²⁺ 단백질(RYR1, CASQ2, CaMKII)이 주변부에 위치 → AUPRC=0.4055의 직접 원인.

이것이 **TCS(Tau)=0.869, SMSI=1.63×** 의 의미: 근육에 Ca²⁺ 신호는 있지만(RYR1 등), SwissProt annotation corpus가 비근육 GPCRs로 압도적으로 편향되어 model이 잘못된 embedding space를 학습.

---

### 4-3. GO:0003774 — Motor Activity (Type-A, D256, AUPRC=0.5982)

#### KIF26A — Kinesin-26A (Non-motor kinesin)
- **핵심 특성**: motor domain에 ATPase 활성화 잔기 없음 → 비운동성 kinesin
- **실제 기능**: GDNF-Ret 신호 억제 (GRB2 결합); 미세소관 구성적 결합
- **실제 조직**: 신장, 췌장, 일부 뇌 영역 (neural/renal)
- **골격근 역할**: 없음; OMIM 613231 표현형 = 피질 이형성, 비정상 신경 이동
- **GO:0003774 annotation**: 구조적 kinesin 도메인 상동성 기반 — 실제 운동성은 없음
- **핵심 참고문헌**: OMIM 613231, Nat Neurosci 2009

#### KIF21A — Kinesin-21A (CFEOM1 locus)
- **핵심 특성**: plus-end 지향 운동 kinesin; 피질 미세소관 성장 억제
- **실제 기능**: 뇌신경 축삭 수송; 외안근 신경지배에 필수
- **CFEOM1**: KIF21A 자동억제 기능 장애 돌연변이 → 동안신경 발달 이상 → 외안근 마비
- **골격근 역할**: 없음 (운동신경 수준의 문제; 근육 자체에는 발현 없음)
- **핵심 참고문헌**: Sci Rep 2016, MedlinePlus KIF21A

#### MYO10 — Myosin-X (Unconventional)
- **핵심 특성**: MyTH4-FERM tail domain; 필로포디아 첨단 국재화
- **✅ 근육 관련성 확인됨**: 분화 근모세포에서 고발현; Myomaker/Myomixer 융합 단백질을 세포-세포 접촉 부위로 전달; 조건부 위성세포 knockout → 출생 후 근육 재생 심각 손상
- **DMD 마커**: 재생 근섬유에서 MYO10 높은 발현 (재생 지표)
- **GO:0003774 annotation**: 비전통적 actin 기반 운동성 — 근육 생물학과 직접 연결
- **핵심 참고문헌**: eLife 2021 (PMC8500716) — PMID 34519272

**GO:0003774 결론**: KIF26A/KIF21A는 신경계 중심으로 골격근과 직접 무관 (구조적 homology annotation), MYO10은 실제 근육 기능 확인. 클러스터 내 이질성이 AUPRC=0.5982의 한계를 설명.

---

### 4-4. GO:0030017 — Sarcomere Organization (Type-B, P3-512, AUPRC=0.3553)

#### ANK3 — Ankyrin-G
- **지배적 이소폼**: 270 kDa, 480 kDa — 축삭 초절편(AIS)/Ranvier 결절에서 Nav 채널 군집 (신경계)
- **근육 특이 이소폼**: AnkG107 (76-aa C-말단 삽입) — 코스타미어 및 신경근육접합부의 근소포체에 국재; dystrophin 및 β-dystroglycan 앵커링
- **결론**: 지배적 이소폼은 신경계; AnkG107은 진짜 근육 특이적 — GO:0030017 annotation은 소수 이소폼 기반
- **핵심 참고문헌**: PMID 15953600, PMID 11796721

#### ACTN2 — Alpha-Actinin-2 (Z-disc)
- **✅ 명확히 근육 특이적**: 심근 + 골격근으로만 발현 제한 (sarcomeric actinin 이소폼)
- **기능**: Z-disc에서 액틴 세필라멘트와 N-말단 titin 분자를 교차연결; 근절 기본 구조 앵커
- **임상**: HCM/DCM 돌연변이 (근절 구조 손상)
- **핵심 참고문헌**: OMIM 102573, Frontiers Physiol 2023

#### PPP1R12B — MYPT2 (Myosin Light Chain Phosphatase Regulatory Subunit 2)
- **✅ 횡문근 특이적**: 심근 > 골격근 발현 (MYPT1은 평활근 dominant)
- **기능**: PP1c 촉매 서브유닛을 심근 myosin regulatory light chain (RLC)으로 타겟팅; RLC 탈인산화 → DRX 상태 조절 → 심근 수축력 조절
- **임상**: PPP1R12B 녹아웃 → 압력 과부하 유발 비대 방어
- **주의**: CORE_META 초기 버전에서 "smooth muscle"로 잘못 기재됨 → 정확한 표현은 "striated muscle (cardiac > skeletal)"
- **핵심 참고문헌**: JBC 2024 PMID 38224947, PMC10851227

**GO:0030017 결론**: ACTN2와 PPP1R12B는 강한 근육 특이성 확인 (논문에 직접 인용 가능). ANK3는 근육 이소폼(AnkG107)이 존재하지만 지배적 이소폼은 신경계 → 임베딩이 혼합 표현 가능성.

---

## 5. Type-A vs Type-B 임베딩 구조 비교 해석

### Type-A (SP 도움이 되는 GO term)

| 특성 | GO:0006096 | GO:0003774 |
|------|-----------|-----------|
| AUPRC (D256) | 0.8331 | 0.5982 |
| 클러스터 형태 | 단일 밀집 | 다중 서브클러스터 |
| Core tissue | 부분 mixing | 혼합 (MYO10 제외) |
| SP 기여 | 다양한 종의 glycolytic enzyme annotation으로 클러스터 강화 | motor domain homology 광범위 수집 |

**해석**: Type-A에서 SP가 유익한 이유 — SwissProt annotation이 기능 그룹을 **종 횡단적으로 일관되게** 레이블링 (glycolysis는 모든 진핵생물에서 보존). 조직 mismatch가 있어도 기능 클러스터 자체는 형성됨.

### Type-B (SP가 해로운 GO term)

| 특성 | GO:0007204 | GO:0030017 |
|------|-----------|-----------|
| AUPRC (P3-512) | 0.4055 | 0.3553 |
| 클러스터 형태 | 확산형 (광범위 GPCR) | 중간 (근육 중심화) |
| Core tissue | **전체 비근육** (platelet/immune/CNS) | 혼합 (ACTN2 근육 ✓, ANK3 CNS 혼합) |
| SP 기여 | 비근육 Ca²⁺ GPCR annotation 대량 포함 → noise | 일부 도움 (스스로 tissue 정보 포함 시) |

**해석**: Type-B에서 SP가 해로운 이유 — SwissProt에서 GO:0007204 annotation이 **조직적으로 heterogeneous** (platelet, immune, CNS 모두 포함). 근육 특이 Ca²⁺ 단백질(RYR1, CaMKII)은 상대적으로 적은 annotation → 임베딩 centroid가 비근육 공간에 고정.

---

## 6. 전체 Figure 목록 (2026-05-14 최종)

### 연구 미팅용 (generate_meeting_figures.py)

| 번호 | 파일명 | 내용 | 크기 |
|------|--------|------|------|
| Fig 1 | `fig1_performance_history.png` | v7c→Selective 성능 진행 히스토리 바 차트 | 105 KB |
| Fig 2 | `fig2_sp_dim_ablation.png` | 2×2 SP×Dim 어블레이션 히트맵 | 142 KB |
| Fig 3 | `fig3_dim_scaling_law.png` | Dim Scaling Law (Type-A/B 비교) | 140 KB |
| Fig 4 | `fig4_fti_tier.png` | FTI Tier 분류 산점도 | 148 KB |
| Fig 5 | `fig5_2d_sp_dependency.png` | 2D TBS×TCS + Regression R²=0.992 | 208 KB |
| Fig 6 | `fig6_kingdom_smsi_panel.png` | Kingdom stacked bar + SMSI 바 플롯 | 171 KB |
| Fig 7 | `fig7_selective_ensemble.png` | Per-GO grouped bars + Selective Macro | 125 KB |
| Fig 8 | `fig8_ds_overview.png` | Domain/Splicing 피처 통계 개요 | 415 KB |
| Fig 9 | `fig9_isoform_resolution.png` | Isoform-level vs Gene-level 해상도 비교 | 209 KB |
| Fig 10 | `fig10_ds_vs_previous.png` | D/S 피처 이전 방식 대비 개선 | 180 KB |

### 심화 분석 (bootstrap + embedding)

| 파일명 | 내용 | 크기 |
|--------|------|------|
| `fig_bootstrap_ci.png` | Gene-block bootstrap 95% CI + 유의성 마커 | 251 KB |
| `fig_embedding_overview.png` | UMAP 4-panel 개요 (Type-A/B 전체) | 695 KB |
| `fig_embedding_GO_0006096.png` | Glycolysis UMAP + score distribution | 272 KB |
| `fig_embedding_GO_0003774.png` | Motor Activity UMAP + score distribution | 285 KB |
| `fig_embedding_GO_0007204.png` | Ca²⁺ Signaling UMAP + score distribution | 317 KB |
| `fig_embedding_GO_0030017.png` | Sarcomere UMAP + score distribution | 330 KB |
| `fig_cluster_core_table.png` | 문헌 검증 클러스터 core 대표 이소폼 테이블 | 317 KB |
| `fig_cluster_structure.png` | Centroid 거리 바이올린 플롯 | 189 KB |

**총 21개 PNG, 약 4.6 MB**

---

## 7. 핵심 발견 (논문 인용 가능 수준)

### F23: Tissue Context Mixing (TCM) — 실증적 확인

GO:0007204 (Ca²⁺ signaling) 양성 클러스터 centroid 최근접 이소폼:
- **F2R/PAR1**: 혈소판/혈관 내피 전용 thrombin receptor GPCR
- **LPAR6**: 림프/피부 특이 LPA receptor GPCR
- **CX3CR1**: 단핵구/소교세포 전용 fractalkine receptor GPCR

세 유전자 모두 Gαq→Ca²⁺ annotation은 생물학적으로 정당하지만 골격근과 무관.  
**결론**: SwissProt Ca²⁺ signaling annotation이 비근육 조직(혈소판, 면역, CNS)으로 편향 → 임베딩 centroid가 비근육 GPCR 공간에 고정 → AUPRC 저하.

이것이 **TCS×TBS 상호작용이 FTI를 예측하는 이유**의 세포 수준 확인:  
TBS(0.667) 동일한 GO:0006096과 GO:0007204에서 FTI가 1.874 vs 0.626으로 갈리는 이유 = TCS(SMSI: 5.63× vs 1.63×) 차이 → TCM 심각도 차이.

### F24: MYO10 — 근육 관련 Motor의 실증

GO:0003774 클러스터 core 중 MYO10이 유일하게 근육 직접 기능 확인:
- eLife 2021 (PMID 34519272): myoblast 융합 시 Myomaker/Myomixer를 cell-cell contact site로 전달
- Satellite cell KO → 출생 후 근육 재생 심각 손상
- DMD 재생 근섬유 마커

**결론**: 클러스터 core 분석이 단순 positive 분류를 넘어 **기능적으로 의미 있는 이소폼**을 발굴할 수 있음을 실증.

### F25: Bootstrap CI로 확인된 통계적 한계 명시

- **강한 주장 가능**: P3-512 vs v8b (Type-B, p<0.001)
- **논문에서 조심해야 할 주장**: D256 > LR (GO:0006096, p=0.055)
  - "numerically higher but not statistically significant (p=0.055)" 표현 사용
  - 또는 "trend toward superiority without reaching significance"

---

## 8. 논문 핵심 주장 체계 (최종 확정)

### Main Claim
> "SwissProt 전이학습이 근육 기능 예측에 미치는 영향은 GO term의 조직적 맥락 특이성에 따라 결정되며, 두 축(TBS × TCS)의 상호작용으로 정량 예측 가능하다."

### Supporting Claims (통계 지지 수준별)

| 주장 | 근거 | 신뢰도 |
|------|------|--------|
| P3-512가 v8b를 Type-B에서 유의하게 초과 | p<0.001, Δ≈+0.2 | ★★★ |
| D256이 GO:0006096에서 LR을 수치적으로 초과 | p=0.055, Δ=+0.138 | ★★ (trend) |
| FTI 2D framework (TBS+TCS+상호작용) R²=0.992 | n=5 해석적 모델 | ★★ (과적합 주의) |
| GO:0007204 TCM — embedding centroid가 비근육 GPCR에 고정 | 문헌 확인 3/3 | ★★★ |
| MYO10이 근육 기능 motor임을 임베딩이 캡처 | eLife 2021 | ★★★ |

---

## 9. 다음 단계 옵션

| 옵션 | 내용 | 기대 결과 | 선결 조건 |
|------|------|----------|----------|
| **A: P3-1024** | dim 512→1024 확장, SP✗ | Macro ~0.50~0.52 예상 | GPU ~2h, 과적합 위험 (GO:0006941 n=81) |
| **B: 논문 작성 진입** | 현재 수치로 framing | — | Figure 세트 완비됨 |
| **C: GO-conditioned projection** | FTI tier 기반 SP 비중 동적 조절 | ~0.50~0.55 | 설계 복잡도 중간 |

**현재 Selective best**: 0.4817 (85.8% of LR)  
**논문 진입 기준**: 0.52 이상 → 현재 0.47~0.52 구간  
**FTI framework 가치**: 독립적인 기여 가능 (p=0.055 이슈와 무관하게 설명 프레임워크)

---

*Report generated: 2026-05-14, Session 2 종료 시점*  
*생성 파일: reports/2026-05-14/ 디렉토리 내 21개 PNG + 2개 JSON + 본 레포트*
