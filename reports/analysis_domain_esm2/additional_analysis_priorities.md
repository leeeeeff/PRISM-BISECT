# BISECT 추가 분석 우선순위 (2026-06-25)

101개 케이스 통합 완료 후, 신규 17개 A-DR/A-BP 케이스 중 추가 심층 분석이 가장 가치 있는 케이스와 분석 방향을 정리.

---

## Priority 1 — 가장 강력한 신규 시그널 (즉시 분석 권장)

### 1-A. ERCC6L2 / Astrocyte (A-DR, stageR p=1.11e-02)

**왜 중요한가:**
- CT 712 aa → AD 212 aa (30% 미만 잔류): 단백질 길이 절단 비율이 17개 케이스 중 최대
- 소실 도메인: DEAD, ERCC3_RAD25_C, HDA2-3, Helicase_C, ResIII, SWI2_SNF2 (6개 SF2/SWI-SNF 관련)
- ERCC6L2 germline mutation → 빈혈·골수부전(aplastic anemia). AD 아스트로사이트에서 같은 기전 재활성화 가설
- 절두 212 aa 이소폼은 SWI/SNF ATP가수분해 기능 전부 소실 → dominant-negative 가능성

**추가 분석 항목:**

| 분석 | 방법 | 기대 결과 |
|------|------|-----------|
| AD 이소폼(212 aa) NMD 가능성 | ERCC6L2-211 exon boundary → PTC 위치 확인, EJC distance 계산 | NMD target → 기능상실 가속화 |
| AF2 구조 예측 비교 | ESMFold/AF2 CT 712 aa vs AD 212 aa; pLDDT 비교 | 212 aa isoform이 안정적 구조를 갖는지 (dominant negative vs 분해) |
| Astrocyte-specific regulation | ENCODE cCRE, GTEx eQTL(brain astrocyte) → ERCC6L2 근방 AD-risk variants | ALS/AD GWAS hit 연결 가능성 |
| Germline vs somatic comparison | ClinVar ERCC6L2 variants → 골수부전 표현형과 AD 아스트로사이트 switch 비교 | 표현형 유사성 논거 |
| SAMHD1 co-analysis | SAMHD1도 DNA repair(innate immune) + NMD switch → 같은 아스트로사이트/억제뉴런 DNA integrity 축 | Figure 통합 가능 |

---

### 1-B. NOL8 / Microglia (A-DR, stageR p=3.85e-02)

**왜 중요한가:**
- AD 이소폼이 CT보다 198 aa **더 크다** (1,167 vs 969 aa): 17개 케이스 중 AD 이소폼이 CT보다 긴 유일한 케이스
- tts_diff=4,459 bp: APA(Alternative Poly-Adenylation) 이동이 주요 메커니즘 → 3'UTR extension 또는 추가 exon inclusion
- tss_diff=8 bp: TSS 거의 동일 → 추가 서열은 C-terminal 쪽 exon 포함
- NOL8: nucleolus 조직화, rRNA 가공. AD microglia에서 nucleolar stress → rDNA instability 연결
- phyloP=0.538: 보통 수준 보존 (생물학적 기능 암시)

**추가 분석 항목:**

| 분석 | 방법 | 기대 결과 |
|------|------|-----------|
| 추가 198 aa 서열 동정 | ERCC6L2-211 exon map 대응; NOL8-208 vs NOL8-205 exon diff | 어떤 exon이 포함되는지 |
| HMMER 신규 도메인 스캔 | 198 aa extension만 추출 → Pfam/PANTHER scan | 새 도메인 획득 여부 |
| Nucleolar stress marker | MIR upstream, RPS/RPL gene co-expression (bulk AD RNAseq) | NOL8 switch와 ribosome biogenesis 연결 |
| Microglia-specific context | 미세아교세포 AD activation (DAM) 마커와 NOL8 switch 상관 | Homeostatic vs DAM microglia에서 발현 차이 |
| APA mechanism 검증 | DaPars2, APADB annotation → NOL8 3'UTR APA site | 실제 APA event vs exon skipping 구별 |

---

### 1-C. AZIN1 / Inhibitory (A-DR, stageR p=4.36e-02)

**왜 중요한가:**
- phyloP=2.348: **17개 케이스 중 최고 보존도** → 기능적으로 매우 중요한 exon 변화
- CT=AD=448 aa: 단백질 길이 동일, Pfam 도메인 동일 → 내부 exon 교환
- AZIN1 기능: Antizyme inhibitor → polyamine 합성 탈억제 → spermine/spermidine 증가
- 폴리아민 대사가 AD에서 변화한다는 보고 존재 (spermine → tau aggregation 억제 vs AD에서 감소)

**추가 분석 항목:**

| 분석 | 방법 | 기대 결과 |
|------|------|-----------|
| 교환 exon 동정 | AZIN1-201 vs AZIN1-216 exon map 비교 | 어느 exon이 다른지, 단백질 내 위치 |
| 교환 exon의 기능적 역할 | 알려진 AZIN1 crystal structure (PDB:3LK9)와 교환 exon 위치 매핑 | antizyme 결합 면 변화 여부 |
| Polyamine pathway 상관 | 삼성 AD 샘플 bulk expression: ODC1, OAZ1/2/3, AZIN1/2, AMD1 co-expression 패턴 | AD에서 polyamine pathway 전체 변화 맥락 |
| RNA editing connection | AZIN1 A-to-I RNA editing (p.Ser367Gly) 알려진 유전자 → AD 샘플에서 editing level 확인 | 이소폼 switch와 RNA editing 동시 변화? |
| Inhibitory neuron specific | GABAergic neuron polyamine requirement → parvalbumin/somatostatin 마커 상관 | 어떤 inhibitory subtype이 영향받는지 |

---

## Priority 2 — 메커니즘 불명확 케이스 (명확화 필요)

### 2-A. DDX19A / Inhibitory (A-DR, stageR p=7.68e-03)

**현재 불확실성:**
- `tss_diff=16,398 bp`: 거대한 TSS 이동 → alternative promoter 가능성
- `ad_cat=retained_intron`: 그런데 같은 케이스가 mechanism=alternative_promoter로 기록됨
- CT=478 aa → AD=301 aa: C-terminal 177 aa 소실 (Helicase_C 도메인 소실 의심)
- 두 메커니즘이 공존? (alt promoter로 시작 + intron retention 포함)

**추가 분석 항목:**

| 분석 | 방법 | 기대 결과 |
|------|------|-----------|
| 이소폼 exon 구조 비교 | DDX19A-201 vs DDX19A-209 Ensembl exon map | TSS 차이 + intron retention 위치 명확화 |
| Helicase_C 도메인 상실 확인 | DDX19A-209 (301 aa) Pfam 스캔 | DEADc 유지, Helicase_C 소실 여부 |
| NMD 가능성 | Retained intron → PTC 위치, EJC distance | NMD target인지 여부 |
| mRNA export 기능 영향 | DDX19A은 NXF1/NXT1 mRNA export complex 협력자 → Helicase_C 없으면? | export 기능 소실 여부 문헌 확인 |
| TSS 위치 ENCODE | chr11:66.8 Mb 근방 (DDX19A) cCRE, H3K4me3 → 16 kb upstream에 별도 promoter 존재? | Alternative promoter 실체 확인 |

---

### 2-B. NDUFS4 / Inhibitory (A-BP, perm p=2.36e-02)

**왜 특이한가:**
- CT=175 aa → AD=378 aa: **AD 이소폼이 CT의 2배 이상** (유일한 케이스)
- NDUS4 도메인 소실 + **RVT_1 (reverse transcriptase) 도메인 획득**
- RVT_1은 LINE-1 retrotransposon ORF2p의 특징 → 매우 비정상적
- mechanism_type=epigenetic_derepression: BISECT가 epigenetic 기전으로 분류

**추가 분석 항목:**

| 분석 | 방법 | 기대 결과 |
|------|------|-----------|
| RVT_1 annotation 재검증 | NDUFS4-204 (378 aa) HMMER e-value 확인 | False positive vs genuine domain |
| Retrotransposon 연관성 | AD 뇌에서 LINE-1 de-repression (HERV-K) 보고 존재 → NDUFS4 locus LINE-1 insertion? | LINE-1/HERV 연관 메커니즘 |
| NDUFS4/Excitatory vs Inhibitory 비교 | Excitatory: NDUFS4-201 CT / transcript73243 AD; Inhibitory: NDUFS4-201 CT / NDUFS4-204 AD | 왜 cell type마다 다른 AD isoform? |
| Complex I assembly impact | NDUFS4 없이 N-module 조립 불가; AD isoform의 2배 크기는? | 미토콘드리아 조립 방해 vs 새 기능 획득 |
| SAMHD1 공동 분석 | SAMHD1(Inhibitory, A-DR)도 LINE-1 restriction factor → Inhibitory neuron LINE-1 derepression axis | Figure 통합 가능 |

---

### 2-C. USP1 / Oligodendrocyte (A-DR, stageR p=7.66e-05)

**왜 중요한가:**
- tss_diff=745 bp, tss_class=alt_promoter_candidate: alternative promoter switch
- CT=AD=785 aa: 단백질 길이 완전 동일 → N-terminal 아미노산 서열만 다를 가능성
- USP1은 FANCD2/PCNA deubiquitylation → Fanconi anemia pathway
- Oligodendrocyte-specific: OPC/myelin 유지와 DNA repair의 연결

**추가 분석 항목:**

| 분석 | 방법 | 기대 결과 |
|------|------|-----------|
| N-terminal 서열 비교 | USP1-201 vs USP1-202 N-terminal 20-30 aa alignment | 어떤 아미노산이 다른지 |
| Alternative promoter 실체 | ENCODE cCRE, FANTOM5 → TSP +745 bp 위치에 promoter element | H3K4me3, ATAC-seq peak 존재 여부 |
| UAF1(WD40) 결합면 영향 | USP1-UAF1 complex에서 N-terminal의 역할 문헌 조사 | N-terminal switch가 UAF1 결합 변화시키는지 |
| PCNA/FANCD2 기질 특이성 | N-terminal domain이 substrate specificity 결정하는지 → deubiquitylase 활성 변화 예측 | DNA repair 기능 보존 vs 변화 |
| OPC demyelination context | MS/AD 공통 demyelination → OPC의 DNA repair capacity가 미엘린 재생에 중요한지 | 임상적 의의 |

---

## Priority 3 — 교차 분석 (Cross-cutting Analyses)

### 3-A. RNA 대사 수렴 축 (4개 케이스)

**케이스:** DDX19A (mRNA export, Inhibitory) + DIS3 (RNA exosome, Oligo) + CNOT11 (CCR4-NOT deadenylase, Oligo) + NOL8 (rRNA processing, Microglia)

**핵심 발견:** AD 뇌에서 RNA 합성-성숙-수출-분해의 전체 사이클이 동시에 교란됨

**분석 방향:**

| 분석 | 방법 |
|------|------|
| 4개 유전자 공동 발현 네트워크 | WGCNA 또는 simple Pearson → AD vs CT 코호트에서 co-expression module 형성 여부 |
| RNA stability genome-wide impact | DIS3+CNOT11 동시 switch → 전체 mRNA 반감기 증가 예측 → bulk RNA-seq data의 transcript length bias |
| Cell type-specific vulnerability | DDX19A=Inhibitory, DIS3/CNOT11=Oligo, NOL8=Microglia → 같은 기능 축이 다른 cell type에서 독립 발현 → 범세포형 RNA homeostasis 붕괴 |
| Literature: RNA processing in AD | FUS/TDP-43 aggregation → RNA metabolism 연결 → 우리 발견이 protein-mediated vs isoform-level | 논문 Discussion §3.11 연결 |

---

### 3-B. DNA 복구/게놈 안정성 수렴 축 (3개 케이스)

**케이스:** ERCC6L2 (SWI/SNF helicase, Astrocyte) + USP1 (FANCD2 DUB, Oligodendrocyte) + RPS3 (BER endonuclease VIII, Excitatory)

**핵심 발견:** 세 가지 독립적 DNA 복구 경로가 각기 다른 세포 유형에서 동시 교란

**분석 방향:**

| 분석 | 방법 |
|------|------|
| DNA damage marker 상관 | γH2AX, 53BP1 IHC in AD brain → 우리 케이스 해당 세포 유형에서 증가하는지 문헌 확인 |
| GWAS locus overlap | ERCC6L2 chr2, USP1 chr1q, RPS3 chr11q → AD GWAS risk loci와 proximity | 유전적 증거 연결 |
| Pathway analysis | StringDB → 3개 유전자 네트워크 확장 → 공유 interaction partner | 공통 허브 발견 |
| AD somatic mutation burden | 최근 AD 뇌에서 somatic mutation 증가 보고 → 우리 DNA repair isoform switch가 원인? | causality 방향 논의 |

---

### 3-C. Oligodendrocyte 취약성 클러스터 (3개 케이스)

**케이스:** DIS3 + CNOT11 + USP1 모두 Oligodendrocyte

**왜 특이한가:** 6개 cell type 중 oligodendrocyte만 3개 A-DR 케이스를 가짐 (Excitatory=4, Oligo=3, Inhibitory=3, Micro=2, Astro=1, OPC=0)

**분석 방향:**

| 분석 | 방법 |
|------|------|
| Oligodendrocyte-specific DRIMSeq enrichment | 전체 transcriptome에서 Oligo DTU 빈도 vs 다른 cell type 비교 (background rate) | 진정한 enrichment vs sampling bias |
| Myelin maintenance connection | DIS3/CNOT11 → MBP/MAG mRNA stability → 미엘린 붕괴 | 간접 효과 경로 |
| OPC → Oligo differentiation | USP1 (FANCD2) → OPC DNA repair → Oligo maturation block | 발달 단계 영향 |
| MS vs AD comparison | MS 병변에서도 같은 Oligo isoform switch? → 미엘린 질환 공통 메커니즘 | 논문 breadth 확장 |

---

## 요약 우선순위 표

| 순위 | 케이스 | 이유 | 즉시 가능한 분석 |
|------|--------|------|-----------------|
| 1 | ERCC6L2/Astrocyte | 6 도메인 소실, 30% 길이, 골수부전 연결 | AF2 구조 예측, NMD check |
| 2 | NOL8/Microglia | AD isoform이 더 큼, 198 aa 추가 삽입 정체 불명 | exon map, HMMER 198aa |
| 3 | AZIN1/Inhibitory | phyloP 2.348 최고 보존, polyamine axis | exon 동정, AZIN1-201 vs -216 alignment |
| 4 | DDX19A/Inhibitory | 메커니즘 모호 (alt-promoter vs retained intron) | exon structure, helicase_C 재확인 |
| 5 | NDUFS4/Inhibitory | RVT_1 도메인 획득 (LINE-1?) — 가장 이상한 발견 | HMMER e-value 재검증, LINE-1 locus |
| 6 | RNA 대사 축 | DDX19A+DIS3+CNOT11+NOL8 수렴 | co-expression, pathway enrichment |
| 7 | DNA repair 축 | ERCC6L2+USP1+RPS3 수렴 | StringDB network, GWAS overlap |

---

*생성일: 2026-06-25 | 기반 데이터: BISECT 17개 신규 케이스 + 101 케이스 통합*
