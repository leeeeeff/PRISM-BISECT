# 심층 분석 대상 목록 (2026-06-25)

**기준**: 101개 BISECT 케이스 전수에서 생물학적 흥미도·통계적 신뢰도·기계론적 불확실성 기준으로 선정.
**분석 방법 분류**: [A] 시퀀스/도메인 | [B] 구조 예측 | [C] 문헌/DB 조회 | [D] 발현/공동발현 | [E] 게놈 컨텍스트 | [F] 기능 예측

---

## 1. Tier A-DR — 즉시 심층 분석 권장 (14개)

### 1-1. ERCC6L2 / Astrocyte ★★★

**핵심 발견**: CT 712 aa → AD 212 aa; 6개 SF2/SWI-SNF 도메인 전부 소실 (DEAD, ERCC3_RAD25_C, HDA2-3, Helicase_C, ResIII, SWI2_SNF2); 패널 내 최대 구조 파괴

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| AD 이소폼(212 aa) NMD 가능성 | [A] ERCC6L2-211 exon map → PTC 위치 계산, 최종 EJC까지 거리(>50 nt?) | NMD target이면 기능상실 가속화 |
| 212 aa 구조 안정성 | [B] ESMFold 212 aa vs 712 aa; pLDDT 비교; 212 aa가 독립 도메인으로 폴딩? | 안정 구조 → dominant negative 가능성 |
| 골수부전 표현형과 대조 | [C] ClinVar ERCC6L2 germline variants (aplastic anemia) → AD switch 비교 | 표현형 유사성 논거 |
| AD GWAS 근접성 | [E] ERCC6L2 chr2 locus → GWAS Catalog AD hits 근방 여부 | 유전적 연관 |
| 아스트로사이트 DNA 손상 마커 | [C] 문헌: γH2AX, 53BP1 in AD astrocytes | Switch와 genomic instability 연결 |
| Somatic mutation 공동 검토 | [C] Lee et al. Nature 2024 — AD 뇌 somatic mutation 증가 부위 | ERCC6L2 locus 체세포 돌연변이 빈도 |

---

### 1-2. NOL8 / Microglia ★★★

**핵심 발견**: AD 이소폼이 CT보다 198 aa 더 큼 (1,167 vs 969 aa); TTS +4,459 bp; 패널 내 유일하게 AD>CT 길이

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| 추가 198 aa 삽입 exon 동정 | [A] NOL8-208 vs NOL8-205 exon-level diff; Ensembl GTF 비교 | 어느 exon이 포함되는지 |
| 추가 서열 HMMER 스캔 | [A] 198 aa C-terminal extension만 추출 → Pfam/PANTHER scan (E<0.1) | 새 도메인 획득 여부 |
| APA mechanism vs exon inclusion | [E] DaPars2, APADB → chr1 NOL8 3′UTR APA site; short-read RNAseq 확인 | 실제 APA event vs 엑손 포함 구별 |
| Nucleolar stress → rRNA | [D] AD bulk RNAseq에서 RPS/RPL, rRNA processing 유전자 공동발현 | NOL8 switch와 ribosome biogenesis 연결 |
| Microglia activation context | [D] DAM marker (TREM2, SPP1, AXL) vs NOL8 switch 발현 패턴 | Homeostatic vs DAM microglia 비교 |

---

### 1-3. AZIN1 / Inhibitory ★★★

**핵심 발견**: 패널 내 최고 phyloP=2.348; CT=AD=448 aa (동일 길이); 내부 exon 교환만 발생; 폴리아민 합성 조절

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| 교환 exon 동정 | [A] AZIN1-201 vs AZIN1-216 exon map; 다른 exon 1개 특정 | 단백질 내 위치와 크기 |
| 교환 exon 구조적 위치 | [B] AZIN1 crystal structure PDB:3LK9 → 교환 exon 위치 매핑 | antizyme 결합면 변화 여부 |
| Polyamine axis 공동발현 | [D] ODC1, OAZ1/2/3, AZIN1/2, AMD1, SRM, SMS — AD vs CT 코호트 | 폴리아민 합성 경로 전체 변화 |
| RNA editing connection | [A] AZIN1 A-to-I editing 사이트 (p.Ser367Gly) → AD 샘플 editing level | Isoform switch + RNA editing 동시 변화? |
| Inhibitory subtype | [D] PV/SST/VIP/LAMP5 마커와 AZIN1 switch 상관 | 어떤 inhibitory subtype이 영향? |
| Phylo 2.348 → 기능 추론 | [C] ConSurf, UCSC conservation track → 해당 exon의 종간 보존 패턴 | 포유류 전체 보존 → 핵심 기능 exon |

---

### 1-4. DDX19A / Inhibitory ★★

**핵심 발견**: tss_diff=16,398 bp (거대 TSS 이동) + ad_cat=retained_intron + C-terminal 177 aa 소실; 메커니즘 이중성

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| TSS 이동 vs intron retention 구별 | [A] DDX19A-201 vs DDX19A-209 exon 구조 Ensembl 비교; 각 exon 1 위치 확인 | 두 메커니즘이 동일 이소폼에 공존하는지 |
| Helicase_C 도메인 소실 확인 | [A] DDX19A-209 (301 aa) HMMER Pfam 재스캔 | DEADc 유지, Helicase_C 소실 여부 |
| NMD 가능성 | [A] Retained intron → PTC 위치 계산, EJC 거리 | NMD target인지 여부 |
| Alternative promoter 실체 | [E] ENCODE cCRE, H3K4me3 ChIP-seq → chr11 +16 kb 위치에 promoter element | Alternative promoter 존재 확인 |
| mRNA export 기능 손실 | [C] DDX19A-NXF1 interaction domain 문헌 → Helicase_C 필수 여부 | Export 기능 소실 예측 |

---

### 1-5. NDUFS4 (Inhibitory) — RVT_1 도메인 획득 ★★

**핵심 발견**: CT=175 aa → AD=378 aa (AD가 2배 이상 큼); NDUS4 소실 + RVT_1 역전사효소 도메인 획득; mechanism=epigenetic_derepression

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| RVT_1 annotation 신뢰도 검증 | [A] NDUFS4 AD 이소폼 HMMER e-value; RVT_1 hit의 sequence alignment 확인 | False positive vs genuine LINE-1 derived |
| LINE-1 exon 포획 여부 | [E] AD 이소폼 신규 서열 RepeatMasker → LINE-1 element 위치 확인 | NDUFS4/Exc와 동일 LINE-1 exon poaching 기전 |
| Exc vs Inh NDUFS4 비교 | [A] Excitatory (transcript73243, 378 aa) vs Inhibitory (NDUFS4-204) — 같은 AD 이소폼인지? | 세포 유형별 다른 AD isoform 채용 여부 |
| Complex I assembly 영향 | [C] 문헌: NDUFS4 없이 N-module 조립 불가 → 2배 크기 이소폼의 의미 | 미토콘드리아 조립 방해 vs 새 기능 |
| CpG demethylation 근거 | [E] DNMT3A/TET2 expression AD vs CT (bulk RNAseq) → LINE exon 접근성 | Epigenetic derepression 기전 |

---

### 1-6. DOCK11 / Inhibitory ★★ (Tier A-BP, 가장 높은 perm significance)

**핵심 발견**: perm p=0.0008 (패널 최고); DHR-2 3-lobe + DOCK-C2 도메인 전체 소실; phyloP=3.25 (A-BP 중 최고); delta=0.717

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| DHR-2 소실 → Cdc42 GEF 활성 | [C] DHR-2 lobe A/B/C의 기능 문헌; CT 이소폼은 GEF active, AD 이소폼은? | GEF 기능 소실 → dendritic spine 유지 불가 |
| phyloP=3.25 exon 보존성 | [E] DHR-2 lobe 해당 exon들의 종간 phyloP; 포유류 보존 수준 | 기능적으로 필수적인 서열 |
| DRIMSeq에서 왜 미검출? | [D] DOCK11 in Samsung cohort: read depth per donor in Inhibitory | Reads sparsity vs real DTU 확인 |
| DOCK10 (Microglia) vs DOCK11 (Inhibitory) 비교 | [C] DOCK-family substrate specificity: DOCK10=Rac1/Cdc42, DOCK11=Cdc42 | Pan-cellular GEF 파괴의 세포 특이성 논거 |
| Inhibitory synapse 마커 | [D] GPHN, GABBR1, SYT2 등 inhibitory synapse 유전자와 DOCK11 switch 상관 | Inhibitory synapse loss와 연결 |

---

### 1-7. USP1 / Oligodendrocyte ★★

**핵심 발견**: stageR p=7.7×10⁻⁵ (Oligo 중 최고); tss_diff=745 bp (alt promoter); CT=AD=785 aa; FANCD2/PCNA DUB

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| Alt promoter 실체 확인 | [E] ENCODE cCRE, FANTOM5 CAGE → USP1 locus +745 bp에 H3K4me3/ATAC peak | Alternative promoter element 존재 여부 |
| N-terminal 서열 차이 | [A] USP1-201 vs USP1-202 N-terminal alignment → 어떤 아미노산이 다른지 | N-terminal 20-30 aa의 기능적 의미 |
| UAF1 결합 영향 | [C] USP1-UAF1 complex 구조 문헌 (PDB:3AL2) → N-terminal 역할 | N-terminal switch가 UAF1 결합 변화시키는지 |
| Oligodendrocyte DNA repair context | [C] OPC → Oligo 분화 시 DNA repair 요구사항; demyelination과 USP1 | 미엘린 재생 능력과 연결 |

---

### 1-8. DIS3 / Oligodendrocyte ★

**핵심 발견**: Toxin_R_bind_C 도메인 소실; phyloP=1.665; C-terminal 30 aa 소실 (958→928 aa); RNA exosome 촉매 서브유닛

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| Toxin_R_bind_C 기능 | [C] DIS3 crystal structure (PDB:2CB2, 2VNU) → Toxin_R_bind_C의 RNA 결합/게이팅 역할 | 도메인 소실이 3'→5' exonuclease activity에 미치는 영향 |
| RNA exosome complex stability | [C] DIS3-exosome subunit interaction map → C-terminal의 복합체 안정화 역할 | 도메인 소실 → exosome 해리 가능성 |
| Oligodendrocyte mRNA turnover | [D] MBP, PLP1, MAG mRNA stability in AD oligo | DIS3 switch → myelin protein mRNA 축적/소실 |
| DIS3 + CNOT11 convergence | [D] 두 유전자 공동발현 패턴 (Oligodendrocyte) | 같은 세포 유형에서 mRNA 분해 이중 파괴 |

---

### 1-9. DCAF5 / Excitatory ★

**핵심 발견**: MIOS_WD40 도메인 획득; WD40 보존; CT=942 aa, AD=941 aa (거의 동일 길이); CRL4 E3 ligase substrate receptor

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| MIOS_WD40 획득 → 새 기질 예측 | [C] MIOS (missing oocyte): GATOR2 complex 성분 → mTORC1 억제 활성화 | DCAF5 AD isoform이 mTOR pathway 기질 선택 변경? |
| CRL4-DCAF5 기질 목록 | [C] DCAF5 ubiquitylation substrate 문헌 (PRMT5, histone H3, ATRX) → MIOS WD40 영향 | 새 기질 예측 |
| WD40 vs MIOS_WD40 구조 차이 | [B] AlphaFold DCAF5-201 vs -209 β-propeller 비교; 기질 결합면 차이 | 단백질-단백질 인터페이스 변화 |

---

### 1-10. CNOT11 / Oligodendrocyte ★

**핵심 발견**: NIC 이소폼 (novel); TSS +113 bp, TTS +1,169 bp; 63 aa C-terminal 소실 (510→447 aa); CCR4-NOT scaffold

| 분석 요소 | 분석 방법 | 예상 결과 |
|-----------|-----------|-----------|
| C-terminal 63 aa의 복합체 기능 | [C] CNOT11 구조 (PDB:4CT4) → NOT10/NOT11 모듈 상호작용 면 | CCR4-NOT 복합체 조립 변화 |
| DIS3+CNOT11 → mRNA 반감기 | [D] mRNA decay signature (AU-rich 타겟) in AD oligo | 이중 파괴로 인한 mRNA 안정화 |
| NIC 이소폼 신뢰도 | [A] transcript223081.chr2.nic SQANTI3 분류 확인; full-splice-match 여부 | Novel 이소폼의 실체 |

---

## 2. Tier C — 뇌 Top 케이스 (Brain, 22개 중 상위)

### 2-1. ZCCHC17 / Oligodendrocyte ★★★

**멀티에비던스 최고 점수 (7점); 뇌 Tier C 1위**

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.965 (매우 큰 기능 점수 차이) |
| DTU p | 7.3×10⁻⁸ |
| domain lost | S1 (RNA-binding, 리보솜 RNA 안정화) |
| phyloP | 2.201 (매우 높은 보존도) |
| PPI | UNSUPPORTED |

**분석 방향**:
- [A] S1 도메인 소실 확인 + 잔여 서열 도메인 스캔 (pCI/SNRNP 복합체 결합 서열)
- [C] ZCCHC17 기능 문헌 (PABPN1 RNA granule, neuronal RNA processing)
- [D] AD 올리고덴드로사이트에서 CNOT11/DIS3와 공동 발현 — RNA 대사 수렴 클러스터
- [E] GWAS AD hit와 chr7 ZCCHC17 근접성

---

### 2-2. DMD / Inhibitory ★★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.919 |
| DTU p | 3.0×10⁻²² |
| domain lost | Spectrin + WW |
| domain gained | SOGA |
| phyloP | 5.625 (패널 내 최고 수준) |
| PPI | SUPPORTED (DAG1, SGCB, SNTA1) |

**분석 방향**:
- [A] DMD isoform pair: 어떤 exon이 Spectrin/WW 소실을 일으키는가 (BMP vs 뇌 특이적 프로모터)
- [B] DMD 뇌 특이적 이소폼 (Dp140/Dp71) 맥락: 뇌 Dp71 → 시냅스 안정화
- [C] Duchenne Muscular Dystrophy → 뇌 인지 결손 표현형: DMD Dp71 억제 뉴런 시냅스 기능
- [D] SNTG1과 공동 분석 (Tier C; DAG1 공유 partner)

---

### 2-3. SYNE1 / Inhibitory ★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.839 |
| DTU p | 3.1×10⁻²⁹ |
| domain lost | Spectrin |
| phyloP | 2.097 |
| PPI | SUPPORTED (EMD, LMNA, LMNB1) |

**분석 방향**:
- [A] SYNE1 (Nesprin-1): SUN-LINC complex 구성; Spectrin 소실 → LINC complex 해체
- [C] Cerebellar ataxia (SYNE1 mutations) → AD 억제뉴런 핵막 무결성
- [D] DMD/Spectrin axis: SYNE1+DMD 동시 Spectrin 소실 → 억제뉴런 세포골격 붕괴

---

### 2-4. FRMD4A / Excitatory ★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.787 |
| DTU p | 4.4×10⁻⁶ |
| domain lost | CUPID + FERM_C + FERM_M (FERM 도메인 3개 모두) |
| phyloP | 1.834 |
| PPI | UNSUPPORTED |

**분석 방향**:
- [C] FRMD4A: PARD3-aPKC-TIAM1 complex에서 apical polarity 조절; AD 연관 유전자 (GWAS hit Chr10)
- [A] FERM 3개 도메인 완전 소실 → 피질 흥분성 뉴런 극성 상실
- [E] FRMD4A AD GWAS locus: rs17125944 (2013 Lambert meta-GWAS) — 동일 유전자

---

### 2-5. ASXL3 / Excitatory ★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.749 |
| DTU p | 1.4×10⁻⁷ |
| domain lost | HARE-HTH + PHD_3 (크로마틴 조절) |
| phyloP | 1.265 |
| PPI | UNSUPPORTED |

**분석 방향**:
- [C] ASXL3: Polycomb repressive deubiquitinase (PR-DUB) complex; H2AK119ub1 제거
- [A] HARE-HTH + PHD 소실 → Polycomb 기반 H3K27me3 표적 유전자 derepression
- [D] ZNF736/ZNF582(KRAB-ZFP) + ASXL3(Polycomb) → 흥분성 뉴런 epigenetic repression 이중 축 파괴

---

### 2-6. FANCA / Excitatory ★★ (DNA repair 수렴)

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.946 (매우 큼) |
| DTU p | 2.2×10⁻¹² |
| domain lost | Fanconi_A |
| phyloP | -0.493 (accelerated evolution — novel exon) |
| PPI | SUPPORTED (FANCD2, BRCA1, BRCA2) |

**분석 방향**:
- [A] Fanconi_A 소실 → FANCD2 ubiquitylation 불가 → Fanconi anemia pathway 완전 차단
- [C] DNA repair 수렴 축: ERCC6L2(HR) + USP1(FANCD2 DUB) + FANCA(FA complex) + RPS3(BER) = 4개 독립 경로
- [E] Phylo=-0.493: accelerated evolution → 기능 획득? CT-specific NNIC exon이 빠르게 진화하는 신규 서열

---

### 2-7. RGS3 / Astrocyte ★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.807 |
| DTU p | 1.1×10⁻¹⁰ |
| domain lost | C2 + CEP76-C2 + PDZ + PDZ_2 + PDZ_6 (5개) |
| PPI | SUPPORTED (GNA11, GNA12, GNAI2) |

**분석 방향**:
- [C] RGS3: G-protein regulatory subunit; GAP activity for Gαi/Gαq; PDZ는 receptor anchoring
- [A] PDZ 5개 + C2 소실 → Gαi/Gαq inhibitory signaling loss → cAMP 상승 예측
- [D] Astrocyte reactivity 마커와 공동발현

---

### 2-8. PTPRS / Astrocyte ★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.789 |
| DTU p | 1.4×10⁻²⁹ |
| domain lost | SusE |
| domain gained | Ig_C17orf99 |
| PPI | SUPPORTED (NTRK3, PPFIA1, PPFIA4) |

**분석 방향**:
- [C] PTPRS (LAR family RPTP): 시냅스 조직화; HSPG co-receptor; CSPG 결합으로 축삭 재생 억제
- [A] SusE 소실 → CSPG binding 상실 → axon regeneration inhibition 감소 (역설적 AD gain-of-function?)
- [D] PTPRF (Inhibitory, Tier C) + PTPRS (Astrocyte, Tier C) → astrocyte-neuron synapse RPTP axis

---

### 2-9. LRPPRC / Oligodendrocyte ★★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.758 |
| DTU p | 1.2×10⁻⁴² |
| domain lost | MA3 + RPN7 |
| phyloP | -0.004 |
| PPI | UNSUPPORTED |

**분석 방향**:
- [C] LRPPRC: mitochondrial RNA binding; mt-mRNA stability + polyadenylation; SLIRP partner
- [A] MA3 (eIF4A binding) + RPN7 (proteasome lid) 소실 → mt-mRNA 안정화 기능 파괴
- [D] Complex I 축(NDUFAF5/NDUFS4/7/8) + LRPPRC(mt-mRNA processing) → 미토콘드리아 이중 파괴 in oligo

---

### 2-10. ZNF268 / Microglia ★

| 항목 | 값 |
|------|-----|
| PRISM delta | 0.795 |
| DTU p | 2.6×10⁻²⁴ |
| domain lost | DsrE |
| domain gained | KRAB |
| phyloP | 0.113 |

**분석 방향**:
- [A] KRAB 획득 + DsrE 소실 → 미세아교세포 이소폼 switch에서 ZNF 기능 전환
- [D] ZNF736(Exc)/ZNF582(Exc)/ZNF268(Microglia) → KRAB-ZFP axis가 여러 세포 유형에 걸쳐 교란

---

## 3. 교차 분석 (Cross-cutting Analyses) — 가장 고부가가치

### 3-A. RNA 대사 수렴 축 ★★★

**4개 케이스 + ZCCHC17**: DDX19A(Inh) + DIS3(Oligo) + CNOT11(Oligo) + NOL8(Micro) + ZCCHC17(Oligo)

| 분석 | 방법 | 핵심 질문 |
|------|------|-----------|
| RNA life-cycle 단계별 파괴 매핑 | [C] 핵 export(DDX19A) → 분해(CNOT11) → exosome(DIS3) → rRNA(NOL8) → snRNA(ZCCHC17) | RNA 처리 연속 단계 모두 파괴되는가? |
| 5개 유전자 공동발현 네트워크 | [D] Samsung 샘플 transcript-level 발현 → Pearson/WGCNA co-expression module | 서로 연결된 모듈 형성 여부 |
| mRNA half-life 변화 예측 | [D] DIS3+CNOT11 동시 손상 → AU-rich mRNA 표적 축적; bulk AD RNAseq | 불안정 mRNA 특이적 축적 |
| TDP-43/FUS 연계 | [C] TDP-43 타겟 mRNA와 DDX19A/DIS3 타겟 overlap | Proteinopathy와 isoform switch의 독립성 |

---

### 3-B. DNA 복구/게놈 안정성 수렴 축 ★★★

**4개 경로**: ERCC6L2(HR/SWI-SNF) + USP1(FA/TLS) + RPS3(BER) + FANCA(Tier C, FA complex)

| 분석 | 방법 | 핵심 질문 |
|------|------|-----------|
| 4가지 독립 경로 모두 파괴 확인 | [C] HR(ERCC6L2) + FA pathway(USP1+FANCA) + BER(RPS3) 경로 다이어그램 | 완전한 DNA repair 차단 가설 |
| Somatic mutation 공동 검토 | [C] Lee et al. 2024 somatic mutation 분포 → 4개 유전자 발현 세포 유형과 overlap | Switch → mutation 축적의 causal 방향 |
| GWAS 연관 | [E] 4개 유전자 locus → AD GWAS 근접성 | 유전적 선행 증거 |
| Cell type specificity | 표: Astrocyte(ERCC6L2), Oligo(USP1), Exc(RPS3/FANCA) → 다른 세포 유형에서 같은 axis | 범세포형 genome instability |

---

### 3-C. KRAB-ZFP/크로마틴 수렴 축 ★★

**3+ 케이스**: ZNF736(Exc, A-DR) + ZNF582(Exc, A-DR) + ZNF268(Micro, Tier C) + ASXL3(Exc, Tier C)

| 분석 | 방법 | 핵심 질문 |
|------|------|-----------|
| KAP1/TRIM28 결합 공유 여부 | [C] ZNF736/ZNF582/ZNF268 → 각 KRAB의 KAP1 결합 예측 (이미 ZNF736 확인) | 3개 KRAB-ZFP 동시 KAP1 해제 → transposon derepression |
| 공유 타겟 유전자 | [C] ZNF736/ZNF582 target gene prediction (STRING, literature) → overlap | 동일 유전자군 derepression |
| ASXL3(Polycomb) + ZNF(KRAB) | [C] H3K27me3(Polycomb) vs H3K9me3(KRAB) → 흥분성 뉴런 epigenetic dual loss | 두 repressor system 동시 실패 |
| Transposable element derepression | [D] L1, HERV-K expression in AD excitatory neurons (bulk RNAseq) | ZNF switch → TE derepression 직접 증거 |

---

### 3-D. Spectrin/세포골격 수렴 축 ★★

**3개 케이스**: DMD(Inh, Tier C) + SYNE1(Inh, Tier C) + SPTBN1(Muscle, Tier C)

| 분석 | 방법 | 핵심 질문 |
|------|------|-----------|
| Spectrin domain 소실 세 유전자 비교 | [A] 세 유전자의 소실 exon 위치 비교; 같은 Spectrin repeat? | 공유 취약 exon vs 독립적 소실 |
| 억제뉴런 세포골격 취약성 | [C] DMD Dp71 + Nesprin-1 → inhibitory synapse anchoring complex | GABAergic synapse 안정화 이중 파괴 |

---

## 4. 분석 방법 총괄 레퍼런스

| 코드 | 분석 방법 | 도구/리소스 | 소요 시간 |
|------|-----------|-------------|-----------|
| [A] | HMMER 재스캔 (E<0.01/0.1), exon map 비교, 서열 alignment | hmmer 3.3.2, Pfam 36, muscle/mafft | 즉시 |
| [B] | ESMFold/AlphaFold 구조 예측 비교 | hMuscle ESMFold, AF2 DB | ~30분/케이스 |
| [C] | 문헌/DB 조회 (UniProt, PDB, ClinVar, STRING, GWAS Catalog) | 웹 검색, db-fetcher 에이전트 | 1-2시간 |
| [D] | Samsung 코호트 공동발현 분석 (transcript-level) | pandas, scipy, seaborn; 기존 count matrix | ~1시간 |
| [E] | ENCODE cCRE, phyloP, RepeatMasker, GWAS locus | UCSC Genome Browser, Ensembl, ENCODE portal | ~30분 |
| [F] | NMD 예측 (PTC 위치 + EJC 거리 계산) | 기존 m6_nmd_screen 모듈 확장 | 즉시 |

---

## 5. 우선순위 매트릭스

| 순위 | 케이스 | Tier | 이유 | 즉시 가능 분석 |
|------|--------|------|------|----------------|
| 1 | ERCC6L2/Astrocyte | A-DR | 가장 극단적 구조 파괴; 임상 표현형(빈혈) 연계 | [B]+[C] |
| 2 | DOCK11/Inhibitory | A-BP | perm p=0.0008 최고; DHR-2 GEF 소실; phyloP=3.25 | [C]+[D] |
| 3 | NOL8/Microglia | A-DR | AD>CT 유일; 198 aa 추가 서열 정체 불명 | [A]+[E] |
| 4 | AZIN1/Inhibitory | A-DR | phyloP=2.348 최고; 동일 길이 내부 exon 교환 | [A]+[C] |
| 5 | DMD/Inhibitory | C | phyloP=5.625(전체 최고); PPI SUPPORTED; Spectrin+WW 소실 | [C] |
| 6 | FANCA/Excitatory | C | DNA repair 4-way convergence 완성; PPI SUPPORTED | [C] |
| 7 | RNA 대사 축 (4개) | A-DR/C | 범세포형 RNA homeostasis 붕괴 가설 검증 | [D] |
| 8 | DNA repair 축 (4개) | A-DR/A-BP/C | 4개 독립 경로 수렴; Nature Methods 수준 finding | [C] |
| 9 | ZCCHC17/Oligodendrocyte | C | Brain Tier C 최고 점수; phyloP=2.201 | [A]+[C] |
| 10 | KRAB-ZFP 축 (3개) | A-DR/C | ZNF736+ZNF582+ZNF268 수렴; TE derepression | [D]+[E] |

---

*생성일: 2026-06-25 | 기반: 101 BISECT 케이스 전수 분석*
