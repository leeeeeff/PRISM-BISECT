# BISECT 총정리 — Biological Isoform-Switch Evidence Characterization Tool
**Version:** v1.1 (M1–M16) | **Date:** 2026-05-30 | **Cohort:** Samsung AD long-read scRNA-seq (13 AD / 8 CT)

---

## 1. 파이프라인 개요

```
Stage 1: DIFFUSE score + DTU filter (Dirichlet-multinomial, Bonferroni-adj)
  ↓ 53 candidates (5 cell types: Excitatory, Inhibitory, OPC, Oligodendrocyte, Astrocyte/Microglia)
Stage 2: Pfam domain change filter (HMMER 3.3.2, E < 0.01)
  ↓ 26 Stage 2 PASS (49%)
M10–M12: Structural / Interactomic / Evolutionary validation
  ↓ 13/26 M11 SUPPORTED (50%) → Tier A/B/C assignment
M14–M16: Regulatory context / Alternative promoter / APA
  ↓ 51/53 processed (exon data available)
```

**모듈 목록:**

| Module | Name | Key Output |
|--------|------|-----------|
| M1 | Stage 1 Filter | DIFFUSE delta, DTU p-value |
| M2 | Domain Annotation | Pfam domain set diff (CT vs AD) |
| M3 | Functional Motif Scan | GxGxxGKT, PDZ GLGF, LYR, WD40 |
| M5 | Sequence Extraction | ct_seq, ad_seq from FAA |
| M8 | Genomic Sequence Validation | UCSC REST API, 6-frame translation, SW alignment |
| M9 | NMD Screening | SQANTI3 NMD flag + 50-nt rule |
| M10 | AlphaFold Structural Confidence | pLDDT (AlphaFold DB / ESMAtlas API) |
| M11 | PPI Network Validation | STRING v12.0, combined score ≥ 700 |
| M12 | Evolutionary Conservation | UCSC phyloP 100-vertebrate (hg38) |
| M14 | Regulatory Context Evidence | Cell-type DEG, RBP motif density, L1 derepression |
| M15 | Alternative Promoter Usage | TSS displacement, ENCODE SCREEN cCRE (disabled) |
| M16 | Alternative Polyadenylation | TTS displacement, 3'UTR motif scan, miRNA seeds |

---

## 2. 전체 통계

### Stage 1 → Stage 2

| Metric | Value |
|--------|-------|
| Stage 1 candidates | 53 |
| Stage 2 PASS (domain change) | **26 (49%)** |
| Domain losses dominant | 21 cases |
| Domain gains | 8 cases |
| Complex exchanges | 3 cases |
| Most lost families | Ig folds (7), Spectrin (2), catalytic (3: Y_phosphatase/Kinesin/Fanconi_A) |

**Cell type distribution (Stage 1):**

| Cell Type | n | SUPPORTED | Rate |
|-----------|---|-----------|------|
| Excitatory | 18 | 4 | 57% (4/7 Stage2) |
| Inhibitory | 14 | 5 | **71%** (5/7 Stage2) |
| OPC | 3 | 1 | 50% |
| Oligodendrocyte | 12 | 1 | — |
| Astrocyte | 4 | 1 | 67% |
| Microglia | 2 | — | — |

### M10–M12 결과

| Tier | Cases | n |
|------|-------|---|
| **A** | KIF21B (Excitatory), PTPRF (Inhibitory), FANCA (Excitatory) | 3 |
| **B** | NDUFS4 (Excitatory), DLG1 (OPC), IFT122 (Excitatory), SYNE1 (Inhibitory), RGS3 (Astrocyte) | 5 |
| **C** | ADGRB2 (Inhibitory) | 1 |

### M14 결과 (53케이스)

| Mechanism | n (before M15) | n (after M15) |
|-----------|----------------|---------------|
| alternative_splicing | 2 | 2 |
| epigenetic_derepression | 1 | 1 |
| transcriptional | 50 | **21** |
| alternative_promoter | 0 | **29** |

Evidence strength: strong(2) + moderate(15) + correlative(31) + weak(5)

### M15 결과 (51 exon-characterized cases)

| TSS Class | n | % |
|-----------|---|---|
| alt_promoter_candidate (≥ 500 bp) | **29** | **57%** |
| same_promoter (< 100 bp) | 18 | 35% |
| tss_shift (100–500 bp) | 4 | 8% |

**TSS_diff 상위 케이스:**

| Case | TSS_diff (bp) | Interpretation |
|------|--------------|----------------|
| DMD_Inhibitory | 888,502 | Dp427b/c vs Dp71 brain promoters |
| PTPRF_Inhibitory | 60,574 | Distinct promoter in inhibitory neurons |
| FANCA_Excitatory | 10 bp | Same promoter (canonical splicing) |
| KIF21B_Excitatory | 34 bp | Same promoter (canonical splicing) |
| SYNE1_Inhibitory | > 50,000 | Distinct transcription units |

### M16 결과 (51 cases)

| APA Class | n | % |
|-----------|---|---|
| same_apa (< 500 bp) | 21 | **41%** |
| major_apa (≥ 5 kb) | **14** | **27%** |
| moderate_apa (500–5 kb) | 13 | 25% |
| minor_apa (< 500 bp) | 3 | 6% |

**Stability predictions (n=21 informative):**

| Predicted Stability | n | Representative Cases |
|--------------------|---|---------------------|
| AD_less_stable | **9** | SYNE1, ANKRD44, FRMD4A, FANCA, GOLGB1 |
| AD_more_stable | 6 | LRPPRC, CCAR1, ASXL3 |
| AD_escapes_miR-132_repression | **3** | **PTPRF, PML** |
| AD_suppressed_in_CT_neurons | 3 | IFI16, ZNF397 |

**Both alt_promoter + major_apa (7 cases = distinct transcription units):**
ANKRD44, ASXL3, FRMD4A, GOLGB1, IFI16, **PTPRF**, SYNE1

---

## 3. Tier A 케이스 상세

### 3.1 KIF21B — Excitatory Neuron

**핵심 메커니즘:** Kinesin motor → WD40 β-propeller gain-of-fold switch

| Module | Evidence |
|--------|----------|
| M2 | CT: Kinesin_motor; AD: WD40/ANAPC4_WD40/Nup160 |
| M3 | CT: P-loop GxGxxGKT(pos.87) + Switch-I(SSRSHA) + Switch-II(DLAGSE); AD: PDZ GVGF(pos.251) acquired |
| M10 | CT ESMAtlas pLDDT = **93.2** (P-loop ATPase); AD = **94.6** (WD40 scaffold) |
| M11 | SUPPORTED: TRIM3=765, SMO=694, STK36=691 |
| M12 | AD-specific exons phyloP = **4.067**; CT-specific = **3.842** (both very highly conserved) |
| M14 | mech=**alternative_splicing**; RBFOX1↓(logFC −0.156, padj 10⁻⁸⁵) + RBFOX3↓(−0.192, padj 10⁻⁷⁶); TDP-43 motif **174×** enriched |
| M15 | same_promoter (TSS_diff = **34 bp**) — canonical splicing, not promoter switch |
| M16 | **ALE** (not APA): TTS_diff = 28,492 bp = different terminal exon (CT=418aa, AD=710aa on minus strand); stability prediction artifact |

**Inter-case chain:** FANCA → R-loop → TDP-43 depletion → KIF21B CT exon loss (same AD excitatory neurons)

---

### 3.2 PTPRF — Inhibitory Neuron (LAR)

**핵심 메커니즘:** PTP catalytic domain loss → dominant-negative Liprin-α scaffolding; driven by alternative promoter

| Module | Evidence |
|--------|----------|
| M2 | CT: Y_phosphatase + DSPc; AD: Ig_3 + I-set + V-set (no catalytic domain) |
| M10 | Global pLDDT = 82.0 (identical); domain-level: CT Y_phosphatase=**72.3**/DSPc=**78.6**, AD Ig=**84.8–87** |
| M11 | SUPPORTED: PPFIA1=**997**, PPFIA3=**996**, CTNNB1=**982** (Liprin-α dominant-negative) |
| M12 | CT (PTP exons) phyloP = **4.341**; AD (Ig exons) = **2.835** |
| M14 | mech → **alternative_promoter** (reclassified); top TF: STAT1, SP3, SP1; evidence=correlative |
| M15 | **alt_promoter_candidate**: TSS_diff = **60,574 bp** → distinct promoter in AD inhibitory neurons |
| M16 | **major_apa**: TTS_diff = **43,557 bp**; AD loses miR-132 seed×2 + PAS; gains miR-9 seed → **AD_escapes_miR-132_repression** |

**5-module convergence:** M10 domain asymmetry + M11 PPI + M12 conservation + M15 alt_promoter + M16 miR-132 escape
**PTPRF = 패널 내 가장 다층적 증거 케이스**

---

### 3.3 FANCA — Excitatory Neuron

**핵심 메커니즘:** Fanconi_A domain loss → DNA repair pathway suppression → R-loop cascade → tau + TDP-43

| Module | Evidence |
|--------|----------|
| M2 | CT (1455aa): Fanconi_A domain; AD (297aa): Fanconi_A **ABSENT** |
| M10 | CT (UniProt O15360) pLDDT = 73.4; AD (novel NNIC) pLDDT: ESMFold pending |
| M11 | SUPPORTED: FANCF=999, FANCC=999, FANCE=999, FANCG=999, FANCM=999, BRCA1=995, UBE2T=998 |
| M12 | CT exons (33) phyloP = **1.321** (conserved); AD exon = **−0.493** (**only negative in panel**) |
| M14 | mech=transcriptional; top regulators: STAT1, SP3, KLF9; evidence=correlative |
| M15 | same_promoter (TSS_diff = **10 bp**) — canonical truncation, not promoter switch |
| M16 | **major_apa**: TTS_diff = **60,825 bp**; AD isoform predicted **less stable** (ARE enrichment) |

**Pathway chain:** FANCA switch → ICL repair failure → R-loop @ MAPT locus → ATM/ATR → CDK5 → tau hyperphosphorylation; parallel: R-loop → TDP-43 cytoplasmic mislocalization → KIF21B switch

---

## 4. 주요 생물학적 발견

### 4.1 기전 재분류: 53 케이스의 57%는 "alternative promoter" 전환

M14에서 "transcriptional" 분류된 50케이스 중 29케이스(58%)가 실제로 TSS displacement ≥ 500 bp.
→ AD-associated isoform switch의 다수가 **splicing 변화가 아닌 transcription unit 선택 전환**.
→ 치료 표적: coding sequence가 아닌 chromatin accessibility (promoter cis-elements).

### 4.2 DAPC 경로 수렴 (DMD + SNTG1)

두 개의 독립적인 inhibitory neuron switch가 DAPC에 수렴:
- DMD: phyloP=4.823 (패널 최고), 888kb TSS shift = Dp427b/c → Dp71 promoter 전환
- SNTG1: phyloP=4.558, PH2 domain 소실
→ AD inhibitory neuron에서 GABA-A receptor clustering 및 GABAergic synapse 조직화 체계적 붕괴

### 4.3 FANCA → TDP-43 → KIF21B 기전 연쇄 (동일 Excitatory neuron)

```
FANCA isoform switch (AD excitatory neuron)
  ↓ ICL repair pathway 억제
  ↓ R-loop accumulation @ MAPT + 기타 고전사 loci
  ↓ ATM/ATR → CDK5 → tau hyperphosphorylation (S202/T205, S396/S404)
  ↓ nuclear TDP-43 depletion (R-loop 해소 실패)
  ↓ KIF21B CT exon inclusion 상실 (TDP-43 motif 174× enriched)
  ↓ KIF21B: kinesin motor → WD40 scaffold (gain-of-fold)
```

### 4.4 PTPRF miR-132 escape (AD inhibitory neuron)

- CT isoform: miR-132 seed sites×2 보유 → miR-132에 의해 억제됨
- AD isoform: miR-132 seed 소실 → miR-132 불감성
- AD brain에서 miR-132 ≥5× 감소 → CT 이소폼 de-repression
- AD 이소폼은 miR-132 완전 불감 → translational output 추가 증폭
→ dominant-negative Liprin-α 단편의 protein-level amplification

### 4.5 NDUFS4 epigenetic derepression (LINE-1 메커니즘)

- L1PA3/L1PA11 (young LINE-1, pct_div < 15%) AD-specific RVT_1 exon 중첩
- DNMT3A↓ (−0.152, padj 10⁻¹⁹) → CpG methylation 소실 → L1 derepression
- RVT_1 도메인: L1PA11 ORF2p와 **100% 서열 동일성** (226/226 aa, score=1,153)
- SETDB2/SIRT1/TRIM28 보상적 상향 → 불충분한 억제 응답

---

## 5. 증거 계층 요약 테이블 (9 curated cases)

| Gene | Cell Type | Domain Change | M10 pLDDT | M11 | M12 AD | M12 CT | M14 Mechanism | M15 TSS | M16 APA | Tier |
|------|-----------|--------------|-----------|-----|--------|--------|---------------|---------|---------|------|
| **KIF21B** | Excitatory | Gain: WD40 | CT=93.2, AD=94.6 | SUPP | 4.067 | 3.842 | alt_splicing | same (34bp) | ALE 28kb | **A** |
| **PTPRF** | Inhibitory | Loss: Y_phosphatase | CT/AD=82.0 [dom-level] | SUPP | 2.835 | 4.341 | alt_promoter | alt (60.6kb) | major+miR132_escape | **A** |
| **FANCA** | Excitatory | Loss: Fanconi_A | CT=73.4 | SUPP | −0.493 | 1.321 | transcriptional | same (10bp) | major+less_stable | **A** |
| NDUFS4 | Excitatory | Gain: RVT_1 | CT=84.1 | SUPP | 2.263 | N/A | epigenetic | alt (?) | — | B |
| DLG1 | OPC | Loss: L27/PDZ(CT) | CT=N/A, AD=73.1 | SUPP | N/A | 2.507 | alt_splicing | — | — | B |
| IFT122 | Excitatory | Gain: WD40 | CT=84.6 | SUPP | 4.826 | 4.022 | alt_promoter | alt (?) | — | B |
| SYNE1 | Inhibitory | Loss: Spectrin | 76.6 | SUPP | 3.450 | 4.228 | alt_promoter | alt (>50kb) | major+less_stable | B |
| RGS3 | Astrocyte | Loss: C2+PDZ | — | SUPP | — | — | transcriptional | — | — | B |
| ADGRB2 | Inhibitory | Gain: HRM | — | SUPP* | 0.075 | — | alt_promoter | alt (4.6kb) | — | C |

*ADGRB2: STRING SUPPORTED하나 M12 HRM exon 저보존(phyloP=0.075) → Tier C

---

## 6. 잔여 이슈 및 검증 필요 항목

### 실험적 검증 필수

| 항목 | 케이스 | 실험 방법 |
|------|--------|---------|
| PTPRF alternative promoter 확인 | PTPRF_Inhibitory | scATAC-seq 또는 CAGE-seq |
| PTPRF miR-132 seed 기능 확인 | PTPRF_Inhibitory | Luciferase 3'UTR reporter |
| KIF21B WD40 ANAPC4 상호작용 | KIF21B_Excitatory | Co-IP, pull-down |
| FANCA isoform → R-loop | FANCA_Excitatory | S9.6 immunostaining |
| tr73243 (NDUFS4) proteomics | NDUFS4_Excitatory | mass spectrometry |
| tr292978 (KIF21B) proteomics | KIF21B_Excitatory | mass spectrometry |
| Independent cohort replication | KIF21B, NDUFS4, DLG1 | Ebbert MWU: KIF21B p=0.026 ✓ |

### 파이프라인 미완 항목

| 항목 | 상태 |
|------|------|
| ENCODE SCREEN cCRE API | 컴퓨트 노드에서 접근 불가 → 수동 검증 필요 |
| PolyASite 2.0 API | 404 → local FASTA scan으로 대체 (완료) |
| ESMFold local 설치 | CUDA openfold 빌드 필요 → API로 대체 (완료) |
| Bambu 버전 확인 | 외부 (data provider 문의 필요) |
| MitoFates/TargetP 2.0 | 웹 서버 접근 필요 |
| Zenodo DOI / GitHub URL | 제출 전 필요 |

### 논문 TODO (주요)

| 항목 | 위치 | 상태 |
|------|------|------|
| Results 3.5 M15/M16 통계 | main_paper_draft.md L.83 | ✅ 완료 |
| Results 3.X.2 PTPRF M15/M16 | main_paper_draft.md L.93 | ✅ 완료 |
| Discussion PTPRF 재프레이밍 | main_paper_draft.md L.197 | ✅ 완료 |
| Abstract 57% alt_promoter 추가 | main_paper_draft.md L.25 | ✅ 완료 |
| Supp Note S2 DMD Dp427b/c | main_paper_draft.md L.286 | ✅ 완료 |
| KIF21B ALE 설명 (not APA) | Results 3.5 | ✅ 완료 |
| Results 3.4 TARDBP 174× | main_paper_draft.md L.75 | ✅ 완료 |

---

## 7. 파일 경로

```
Final_analysis/pipeline_bioanalysis/
├── orchestrate.py               ← BISECT v1.1 M1–M9 통합
├── run_m14_batch.py             ← M14 배치 (53케이스)
├── run_m15_m16_batch.py         ← M15+M16 배치
├── recover_none_exons.py        ← NONE 케이스 exon 복구 (GTF 파싱)
├── config.yaml                  ← 전체 설정 (경로, 임계값)
├── modules/
│   ├── m14_regulatory_context.py
│   ├── m15_promoter_usage.py
│   └── m16_apa.py
├── outputs/
│   ├── [GENE_CELLTYPE]/
│   │   ├── analysis.json        ← 전체 증거 (M1-M16)
│   │   └── report.md            ← 케이스별 마크다운 리포트
│   ├── batch_summary_all53.tsv  ← M10-M12 배치 요약
│   ├── m14_batch_summary.tsv    ← M14 배치 요약
│   └── m15_m16_batch_summary.tsv ← M15+M16 배치 요약
├── main_paper_draft.md          ← 논문 초안 (Results + Discussion + Methods + Supp)
└── BISECT_design.md             ← 설계 문서
```

---

## 8. 핵심 발견 요약 (Nature Methods 기여도 관점)

1. **규모**: 53케이스 × 16 모듈 = 체계적 multi-evidence 분석. M11 STRING falsification 4회 실증.
2. **기전 재분류**: 57% = alternative promoter switch → "isoform switching"의 상당 부분이 실제로 transcription unit selection 전환임을 genome-scale로 최초 제시.
3. **Tier A 케이스**: KIF21B(gain-of-fold), PTPRF(5-module convergence), FANCA(R-loop cascade) — 각각 독립적이고 다른 기전 archetype 대표.
4. **Inter-case mechanistic chain**: FANCA→TDP-43→KIF21B — 단일 세포 타입에서 두 isoform switch가 mechanistic chain으로 연결됨. BISECT의 multi-case 통합 분석 능력 실증.
5. **Post-transcriptional amplification**: miR-132 escape (PTPRF) + AD miRNA microenvironment = protein level에서 AD 이소폼 선택적 증폭. splicing 연구에서 드물게 다루어지는 층위.
