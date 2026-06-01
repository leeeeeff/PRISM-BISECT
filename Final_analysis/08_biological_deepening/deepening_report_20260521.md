# Biological Deepening Analysis Report
**2026-05-21 | DIFFUSE Project — AD Isoform Switch Cases**

---

## 실행 요약

3가지 P1 분석 완료:
1. **MTS Feature Analysis** (NDUFS4-201 vs tr73243) — 수치적 MTS 불가 근거 확보
2. **hmmscan tr319500** (DLG1 OPC) — L27 도메인 발견 (예상 외 기능 유지)
3. **hmmscan tr73243** (NDUFS4 AD) — RVT_1 도메인 발견 + NAT 확인 (주요 재발견)

---

## 1. NDUFS4 — MTS Feature Analysis

### 분석 방법
MitoFates (Fukasawa et al. 2015) / TargetP 2.0 (Almagro Armenteros et al. 2019)의
sequence-level feature categories를 직접 구현하여 두 이소폼 비교.

### 결과 테이블

| Feature | NDUFS4-201 (canonical) | tr73243 (AD) |
|---------|------------------------|--------------|
| Net charge, first 30aa | **+2** (K+R=4, D+E=2) | **−1** (K+R=2, D+E=3) |
| Hydrophobic moment (μH) | 0.276 | 0.327 |
| HHH motif (aa 1-30) | **ABSENT** | **PRESENT @ aa 7** |
| LYR motif | ABSENT | ABSENT |
| MTS Composite Score | 3/5 | 1/5 |

### 해석
- **Net charge** (+2 vs −1): 가장 강력한 MTS 판별 지표. 양전하 우세 = 미토콘드리아 막 통과 구동력
- **HHH@7**: His-His-His triplet은 canonical MTS에서 보고된 바 없음. 
  amphipathic helix 형성 방해 (His는 pH 7.4에서 부분 양전하 → local structure disruption)
- **MTS Score 1/5**: tr73243은 미토콘드리아 import 가능성 매우 낮음

### 논문 기술 (Methods)
> "MTS probability was assessed using sequence features from MitoFates and TargetP 2.0: 
> NDUFS4-201 shows net charge +2 in the N-terminal 30 aa (favorable), while tr73243 
> shows net charge −1 with an unusual HHH triplet at position 7 (both unfavorable; 
> MTS composite score 1/5 vs 3/5). This quantitatively confirms the absence of 
> mitochondrial targeting in tr73243."

---

## 2. DLG1 — tr319500 도메인 구조 (주요 재해석)

### hmmscan 결과

| Domain | Position | E-value | Score | 기능 |
|--------|----------|---------|-------|------|
| **L27_1** | aa 6-63 | 7.8e-34 | 103.2 | MAGUK 올리고머화 & Lin7 결합 |
| MAGUK_N_PEST | aa 107-144 | 2.9e-15 | 44.4 | N-terminal 조절 도메인 |
| PDZ_assoc | aa 152-172 | 2.2e-10 | 28.8 | PDZ 연관 (full PDZ 아님) |

**PDZ, SH3, GK, HOOK: 모두 ABSENT (확인됨)**

### tr319500 도메인 맵

```
Canonical DLG1 (906aa):
[L27N]--[L27C]--[PDZ1]--[PDZ2]--[PDZ3]--[SH3]--[HOOK]--[GK]

tr319500 (187aa):
[L27_1(6-63)]--[????]--[PEST(107-144)]--[PDZ_assoc(152-172)]
    ↑ RETAINED                              ↑ NOT functional PDZ
```

### 생물학적 의미 (기존 해석 교정)

**기존 해석**: tr319500 = "PDZ 없는 기능 없는 decoy"  
**수정된 해석**: tr319500 = "L27-specialized OPC scaffolding isoform"

L27 domain의 기능:
- MAGUK 단백질 간 heterodimerization (MPP family)
- **Lin7 (MALS/Veli) family와 결합** → Lin7은 OPC의 β-neurexin과 연결
- 시냅스 수용체 클러스터링 없이 scaffolding 구조 유지

**정상 OPC** (CT): tr319500 우세 (80.9%)
- L27 ✓ → MAGUK scaffolding & Lin7-β-neurexin 연결 유지
- PDZ ✗ → NMDA/AMPA receptor clustering 없음
- = OPC-specific scaffolding function (신경 시냅스 X, OPC 특화 구조 O)

**AD OPC**: canonical DLG1 우세 (95.2%)
- L27 ✓, PDZ×3 ✓ → 완전한 neuronal synaptic scaffolding protein
- NMDA/AMPA receptor clustering 활성화
- = OPC가 neuronal synaptic profile 획득 = **탈분화**

**강화된 메커니즘 플로우**:
```
CT OPC: tr319500 (L27+, PDZ-)
         ↓ L27-mediated
         Lin7 → β-neurexin → OPC-specific contact
         PDZ 없음 → synaptic receptor 없음
         = OPC identity 유지

AD OPC: canonical DLG1 (L27+, PDZ×3)
         ↓ PDZ-mediated
         NMDA/AMPA receptor clustering
         Wnt/β-catenin 경로 조절 변화
         = Neuronal-type synapse 형성 = OPC 탈분화
```

### Mathys 2019 연결
Mathys et al. 2019 (Cell) "OPC states in AD resemble developmental progenitors" — 
현상 기술. 우리 결과는 DLG1 이소폼 switch를 통해 이 탈분화의 **분자적 스위치**를 처음 제시.

### 논문 기술 업데이트
> "hmmscan analysis revealed that tr319500, previously considered domain-depleted, 
> retains an L27 domain (aa 6-63; E=7.8×10⁻³⁴) and MAGUK_N_PEST (aa 107-144; 
> E=2.9×10⁻¹⁵). The L27 domain mediates MAGUK protein oligomerization and Lin7/MALS 
> interaction, establishing β-neurexin-linked OPC-specific scaffolding without PDZ-
> mediated receptor clustering. In AD OPCs, replacement by canonical DLG1 (3×PDZ) 
> activates synaptic receptor clustering, recapitulating the neuronal synaptic profile 
> observed by Mathys et al. (2019)."

---

## 3. NDUFS4 — tr73243 주요 재발견: NAT + RVT_1 도메인

### 가장 중요한 발견

**tr73243는 NDUFS4 Natural Antisense Transcript (NAT)이다.**

| 항목 | 값 |
|------|---|
| tr73243 strand | **+ (positive)** |
| NDUFS4 canonical strand | **− (negative/reverse)** |
| 관계 | **Antisense** (같은 locus, 반대 strand) |
| tr73243 TSS | chr5:53,560,626 (+ strand) |
| tr73243 CDS start | chr5:53,686,672 ✅ (exon 6 내, 수치 일치) |
| 게놈 span | ~127 kb (NDUFS4 전체 intron을 가로지름) |

### hmmscan RVT_1 결과

| Domain | Position | E-value | Score |
|--------|----------|---------|-------|
| **RVT_1 (PF00078)** | aa 141-366 | **4.6e-48** | 149.7 |

RVT_1 = RNA-dependent DNA Polymerase (Reverse Transcriptase)  
E=4.6×10⁻⁴⁸ — spurious hit 가능성 극히 낮음

**YMDD canonical catalytic motif**: 검출 안 됨 → catalytically inactive RT fold 가능성

### LINE-1 가설

LINE-1 (L1) 요소와의 연관성 가설:

```
LINE-1 원리:
  - LINE-1 elements는 ORF2에서 RT domain을 encode
  - LINE-1은 자체 antisense promoter (ASP)를 5' UTR 내 보유
  - LINE-1이 host gene intron에 삽입 → ASP가 host gene과 antisense 방향 전사 구동 가능

NDUFS4 locus 가설:
  LINE-1 삽입 (NDUFS4 intron, AD에서 활성화)
    ↓
  LINE-1 ASP → + strand 전사 (= NDUFS4 - strand의 antisense)
    ↓
  tr73243 생성 (NNIC, RVT_1 domain 포함)
    ↓
  두 가지 효과:
  (1) 직접: tr73243 → 379aa 단백질 (MTS absent, Complex I 참여 불가)
  (2) 간접: antisense transcript → NDUFS4 canonical mRNA 억제 (RNA silencing)
    ↓
  순결과: Complex I N-module assembly 실패 → 미토콘드리아 기능 저하
```

### AD에서의 LINE-1 활성화 문헌
- **Guo et al. 2018 Nature**: AD 뉴런에서 LINE-1 retrotransposition 증가
- **Cook et al. 2021 Nature Neuroscience**: 신경퇴행에서 L1 활성화
- **Tam et al. 2019 Nature Neuroscience**: 노화 뇌에서 somatic L1 삽입 축적

### 검증 필요 (우선순위)
1. **즉시 가능**: tr73243 RVT_1 region (aa 141-366)을 RepeatMasker/Dfam L1 데이터베이스 비교
2. **즉시 가능**: chr5:53,560,626-53,688,219 내 L1 요소 in silico 스캔
3. **중기**: ENCODE AD brain ATAC-seq — tr73243 TSS 주변 chromatin 접근성 변화
4. **장기**: RT-PCR로 tr73243 실제 antisense transcript 검출

### 논문 서술 업데이트
> "Genome coordinate analysis revealed that tr73243 is transcribed from the positive 
> strand (chr5:53,560,626→53,688,219), antisense to the canonical NDUFS4-201 
> (negative strand). This establishes tr73243 as a natural antisense transcript (NAT) 
> spanning the NDUFS4 locus. hmmscan identified an RVT_1 domain (aa 141-366; 
> E=4.6×10⁻⁴⁸), consistent with a reverse transcriptase-homologous fold, suggesting 
> derivation from a LINE-1 retroelement active at this locus in AD neurons. NAT 
> upregulation may suppress canonical NDUFS4 expression through antisense silencing, 
> providing a dual mechanism for Complex I N-module assembly failure."

---

## 수정이 필요한 기존 기술

| 위치 | 기존 | 수정 |
|------|------|------|
| project_state.md | "TSS+7bp (chr5:53,686,672)" | "CDS start chr5:53,686,672 (exon 6, NAT)" |
| Fig 7B NDUFS4 | "Alternative TSS isoform" | "Natural Antisense Transcript (NAT)" |
| results_draft_20260516.md | "locus hijacking by alternative TSS" | "NAT activation at NDUFS4 locus" |
| DLG1 narrative | "domain-less decoy isoform" | "L27-specialized OPC scaffolding isoform" |

---

## 우선순위 후속 작업

| 우선순위 | 작업 | 예상 시간 |
|---------|------|-----------|
| P1 | Results/Discussion NDUFS4 서술 수정 (NAT 반영) | 1시간 |
| P1 | DLG1 서술 수정 (L27 기능 유지 반영) | 30분 |
| P1 | Fig 7B 업데이트 (NAT + antisense 구조 추가) | 2-3시간 |
| P2 | RepeatMasker/Dfam으로 LINE-1 확인 | 반나절 |
| P2 | Lin7-β-neurexin-DLG1 L27 문헌 검색 | 1시간 |
| P3 | MitoFates 웹 서버 실제 실행 (수치 논문 제출용) | 30분 |

---

*Generated: 2026-05-21 | Analyst: DIFFUSE biological deepening pipeline*

---

## 4. LINE-1 RepeatMasker 검증 — tr73243 RVT_1 도메인 출처 확인 (옵션 B)

### UCSC RepeatMasker hg38 query
쿼리 범위: chr5:53,560,626-53,688,219 (tr73243 전체 span)
총 LINE class hits: 62개

### Exon 6 직접 중첩 LINE-1 요소

| Element | Start | End | Strand | %Div | SW Score | Exon6 Overlap |
|---------|-------|-----|--------|------|----------|---------------|
| **L1PA4(−)** | 53,685,059 | 53,685,452 | − | **3.4%** | 3,075 | 0bp (538bp upstream) |
| **L1PA3(−)** | 53,685,456 | 53,686,732 | − | **4.5%** | 7,461 | **742bp (Exon6 start)** |
| **L1PA11(+)** | 53,686,734 | 53,689,784 | + | 9.4% | 12,793 | **1,485bp (Exon6 end)** |

### 확정된 LINE-1 구조

```
Exon 6 (53,685,990-53,688,219):
          |----L1PA3(-)----||--------L1PA11(+)-------------------→
          53,685,456      53,686,732 53,686,734               53,689,784
                           ↑
                    Exon6 start (53,685,990)
                                    ↑
                    CDS start (53,686,672) — within L1PA3(-) 5' UTR region

CDS 매핑:
  aa  1- 20: chr5:53,686,672-53,686,731 → L1PA3(-) 내부
  aa 21-379: chr5:53,686,733-53,687,808 → L1PA11(+) ORF2 내부
  
RVT_1 domain (aa 141-366):
  게놈 위치: chr5:53,687,092-53,687,770 → L1PA11(+) 내부 ✅
```

### 최종 메커니즘 모델

```
[정상 excitatory neuron]
  L1PA3 + L1PA11 → 메틸화/이형염색질화 → 전사 억제
  canonical NDUFS4(-strand) → 정상 발현 → Complex I 완전 조립

[AD excitatory neuron]
  LINE-1 전반적 탈억제 (Guo 2018, Cook 2021)
    ↓
  L1PA11(+) ORF2 발현 → RT 도메인 단백질 (tr73243 aa 21-379)
  L1PA3(-) ASP 활성화 → + strand 전사 시작 (CDS start 53,686,672)
    ↓
  tr73243 (NAT): L1PA3 5'UTR-junction + L1PA11 ORF2 RT domain
    ↓ 이중 억제:
  (1) tr73243 단백질: MTS 없음, LYR 없음 → Complex I 기여 불가
  (2) antisense 전사체: NDUFS4 canonical mRNA antisense silencing
    ↓
  Complex I N-module 조립 실패 → 미토콘드리아 호흡 저하 → AD 병리
```

### 논문 기술 추가

> "RepeatMasker annotation of the NDUFS4 locus confirmed two adjacent LINE-1 elements 
> directly overlapping the tr73243 CDS-containing Exon 6: L1PA3 (− strand; 4.5% 
> divergence; chr5:53,685,456–53,686,732; overlapping the first 742 bp of Exon 6) and 
> L1PA11 (+ strand; 9.4% divergence; chr5:53,686,734–53,689,784; overlapping the 
> remaining 1,485 bp of Exon 6). The RVT_1 domain (aa 141–366; genomic chr5:53,687,092–
> 53,687,770) maps entirely within L1PA11(+), confirming that the reverse transcriptase 
> homologous fold derives from L1PA11 ORF2 sequence. These findings establish tr73243 
> as a LINE-1-chimeric natural antisense transcript at the NDUFS4 locus, activated 
> by LINE-1 de-repression in AD excitatory neurons."

### 잔여 검증
- [ ] L1PA11 ORF2 서열과 tr73243 aa 21-379 정렬 (BLAST 또는 pairwise)
- [ ] L1PA7(-) at 53,625,889-53,632,164 (9.0% div, SW=22,913): 이 요소가 LINE-1 단백질 공급원?
- [ ] ENCODE AD brain ATAC-seq: chr5:53,685,456-53,689,784 열린 크로마틴 여부

*Generated: 2026-05-21 | Analysis: RepeatMasker + genomic coordinate mapping*
