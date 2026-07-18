# 심층 분석 결과 (2026-06-25)

exon 구조 비교 (CT-only / AD-only / shared exon 분석) + NMD 수동 계산 + 메커니즘 분류.
기반 데이터: `analysis.json` ct_info/ad_info exon 좌표, domain_change, m6_nmd_screen.

---

## 1. ERCC6L2 / Astrocyte (Tier A-DR)

### Exon 구조
| 항목 | CT (ERCC6L2-201) | AD (ERCC6L2-211) |
|------|-----------------|-----------------|
| exon 수 | 14 | 2 |
| 공유 exon | **0개** — 완전히 다른 exon 조합 |
| 단백질 | 712 aa | 212 aa |
| Genomic span | chr2:95,875,618–95,968,840 (93 kb) | chr2:95,875,650–95,883,331 (7.5 kb) |

**AD-only exons**:
- Exon 1: 434 bp (95,875,650–95,876,084) — CT Exon 1과 미세하게 다른 끝 위치 (18 bp 짧음)
- Exon 2: 2,462 bp (95,880,869–95,883,331) — CT exon 2 (424 bp)를 훨씬 넘어 intron 영역까지 포함

### NMD 분석
- AD 이소폼 2 exons → 1개 junction → EJC 1개 (exon1/exon2 경계 근방)
- 212 aa protein = 636 nt CDS → stop codon이 exon 2 내부 (exon 2는 2,462 bp로 충분한 공간)
- **PTC는 유일한 EJC 하류(downstream)** → 50-nt 룰 위반 없음 → **NMD 미발생**
- ERCC6L2-211은 분해되지 않는 안정한 212 aa 단백질

### 메커니즘 분류
**Alternative Last Exon (ALE)**: CT의 exon 1-14 전체 스플라이싱 경로를 무시하고, 유전자 5' 끝 2개 exon만 사용하는 완전히 독립된 짧은 전사체. 공유 exon이 0개이므로 단순 exon skipping이 아님.

### 생물학적 의미
- 212 aa 절두 단백질은 ERCC6L2의 모든 SF2/SWI-SNF 촉매 도메인(DEAD, ERCC3_RAD25_C, HDA2-3, Helicase_C, ResIII, SWI2_SNF2)을 소실
- 단백질이 안정적으로 번역된다면 **dominant-negative** 가능성: 스캐폴드 역할은 하되 ATPase 활성 없음
- 골수부전 germline mutations(ERCC6L2 haploinsufficiency)와 동등한 기능 소실이 AD 아스트로사이트에서 isoform switch를 통해 발생

---

## 2. NOL8 / Microglia (Tier A-DR)

### Exon 구조
| 항목 | CT (NOL8-205) | AD (NOL8-208) |
|------|--------------|--------------|
| exon 수 | 13 | 17 (+4) |
| 공유 exon | 11개 |
| 단백질 | 969 aa | 1,167 aa (+198 aa) |
| CT-only | 2개 (5 bp + 178 bp) |
| AD-only | 6개 (528+79+70+126+271+44 bp = 1,118 bp) |

**AD-only exon cluster** (chr1:92,297,358–92,301,822, minus strand):
```
528 bp: 92,297,358–92,297,886
 79 bp: 92,298,257–92,298,336
 70 bp: 92,298,884–92,298,954
126 bp: 92,299,890–92,300,016
271 bp: 92,301,551–92,301,822
 44 bp: 마지막 exon 경계 변화
```
- 총 1,118 bp AD-only; CT-only exon 183 bp; net 추가: ~935 bp → ~312 aa (UTR 포함 시 ~198 aa 추가 단백질과 일치)
- AD exon cluster는 C-terminal 영역(minus strand → 낮은 게놈 좌표 = 3' mRNA)에 위치

### 메커니즘 분류
**Cassette Exon Cluster Inclusion (C-term extension)**: 5개 신규 exon이 cluster로 포함되어 C-terminal이 연장. APA가 아닌 exon inclusion 이벤트.

### 생물학적 의미
- 추가 198 aa는 NOL8 뉴클레올라 복합체 내 새 상호작용 면을 형성 가능 → gain-of-interaction 또는 dominant interference
- CT-only exon (178 bp)는 AD에서 제거: alternative internal exon skipping + downstream exon gain이 동시 발생
- **후속 분석 필요**: 추가 198 aa의 HMMER 스캔 (새 도메인 획득 여부)

---

## 3. AZIN1 / Inhibitory (Tier A-DR)

### Exon 구조
| 항목 | CT (AZIN1-201) | AD (AZIN1-216) |
|------|---------------|---------------|
| exon 수 | 12 | 13 (+1) |
| 공유 exon | 10개 |
| 단백질 | 448 aa | 448 aa (동일!) |
| Strand | − |

**차이점** (minus strand → 높은 게놈 좌표 = 5' mRNA):
```
[5' 방향] CT exon: 102,863,807–102,864,163 (356 bp)
          AD exon: 102,863,807–102,864,192 (385 bp)  → 3' 경계 29 bp 연장
[중간]    AD extra exon: 102,849,987–102,850,144 (157 bp) — CT에 없음
[3' 방향] CT exon: 102,826,308–102,828,678 (2,370 bp)
          AD exon: 102,826,111–102,828,678 (2,567 bp) → 5' 경계 197 bp 연장
```

### NMD 분석
- 단백질 길이 동일(448 aa) → CDS 내 차이는 exon boundary shift로 상쇄됨
- 157 bp AD extra exon: 157 ÷ 3 = 52.3 → **non-divisible by 3** → frameshift 또는 UTR
  - 단백질이 동일하므로 UTR에 위치하거나 frame이 다른 exon boundary shift로 보정
- **NMD 없음** (단백질 길이 동일)

### 메커니즘 분류
**Alternative splice site + cassette exon (UTR-level)**: 단백질 코딩 서열은 보존, 5'UTR 또는 3'UTR 구조가 다름. phyloP=2.348의 고보존 AD-specific exon은 UTR regulatory element일 가능성.

### 생물학적 의미
- 단백질 동일 → **번역 효율, mRNA 안정성, 번역 개시 속도**가 이소폼별로 다를 수 있음
- AD-specific exon의 phyloP=2.348: UTR regulatory element가 포유류 전체에서 보존 → 기능적으로 매우 중요한 번역/안정성 제어 서열
- Polyamine 합성 속도를 mRNA 레벨에서 조절하는 post-transcriptional regulatory switch

---

## 4. DDX19A / Inhibitory (Tier A-DR)

### Exon 구조
| 항목 | CT (DDX19A-201) | AD (DDX19A-209) |
|------|----------------|----------------|
| exon 수 | 12 | 5 |
| 공유 exon | 3개 (N-terminal 초기 exon) |
| 단백질 | 478 aa | 301 aa |
| CT-only | 9개 (중간 coding exons) |
| AD-only | 2개 (대형 "super-exon") |

**AD-only exons**:
```
2,961 bp: 70,363,301–70,366,262 → CT의 6개 소형 exon (145+48+50+135+92+102 bp = 572 bp) + introns 포함
1,451 bp: 70,371,925–70,373,376 → CT의 마지막 coding exon (1,458 bp)과 거의 동일 (7 bp 짧음)
```

### 메커니즘 분류
**Retained Intron → "Super-exon" 형성**: AD 이소폼에서 CT의 6개 소형 exon 사이 intron들이 보존되어 하나의 2,961 bp 대형 exon을 형성. 이 retained intron 내 PTC → 177 aa C-terminal 소실.

**TSS 이동 (16,398 bp)**: 별도 promoter에서 시작하여 N-terminal 3개 exon 공유 후 retained intron 경로 진입.

### NMD 가능성
- AD super-exon (2,961 bp) 내 PTC 위치에 따라 결정
- CT 마지막 exon과 공유 exon이 3개이므로 EJC가 2개 존재 가능
- 만약 PTC가 마지막 EJC에서 50 nt 이상 upstream → NMD 대상
- **현재 m6 모듈이 None 반환 → M6 업그레이드 필요** (파이프라인 모듈화 섹션 참고)

---

## 5. DOCK11 / Inhibitory (Tier A-BP)

### Exon 구조
| 항목 | CT (ENST00000276204) | AD (ENST00000632573) |
|------|---------------------|---------------------|
| exon 수 | 53 | 3 |
| 공유 exon | **0개** |
| 단백질 | 2,077 aa | ~0 aa (추정) |

**AD 이소폼 특성**:
```
Exon 1: 114 bp (118,683,103–118,683,217)
Exon 2:  17 bp (118,685,520–118,685,537)  ← 매우 짧음
Exon 3: 459 bp (118,685,688–118,686,147)
총: 590 bp
```
- CT 유전자 전체(118,495,898–118,686,163 = 190 kb) 중 마지막 2 kb만 사용
- 3 exons 합계 590 bp → 최대 ~196 aa 코딩 가능
- Pipeline에서 ad_seq=0 aa: 3 exon이 정상 ORF를 형성하지 못함 (early stop 또는 non-coding)
- **AD 이소폼이 단백질을 만들지 않을 가능성** → DHR-2 GEF 기능 완전 소실 (loss-of-function, 단순 감소가 아닌 CT isoform의 전체 소실)

### 생물학적 의미
- perm p=0.0008, phyloP=3.25: 통계적 신뢰도 + 진화적 보존 모두 최고
- CT 이소폼(2,077 aa, 53 exon)이 전체 제거되고 3 exon 짧은 전사체로 대체
- 억제 뉴런에서 Cdc42-GEF 기능 완전 제거 → dendritic spine 형성 불가 → GABAergic synapse 불안정

---

## 6. FANCA / Excitatory (Tier C, DNA repair 축)

### Exon 구조
| 항목 | CT (ENST00000389301) | AD (ENST00000389302) |
|------|---------------------|---------------------|
| exon 수 | 43 | 11 |
| 공유 exon | 9개 (N-terminal) |
| 단백질 | 1,455 aa | 297 aa (20% 잔류) |
| CT-only | 34개 (중간/C-terminal coding exons) |
| AD-only | 2개 |

- CT 전체 1455 aa에서 첫 9개 공유 exon = N-terminal ~297 aa
- AD 이소폼은 N-terminal만 있고 Fanconi_A 도메인 전체 소실
- 도메인 소실 → FANCD2 모노유비퀴틴화 불가 → FA pathway 차단
- ERCC6L2(Astrocyte) + USP1(Oligodendrocyte) + FANCA(Excitatory): 3 cell type × 3 FA/DNA repair 유전자

---

## 7. DMD / Inhibitory (Tier C)

### Exon 구조
| 항목 | CT | AD |
|------|----|----|
| exon 수 | 32 | 17 |
| 공유 exon | 12개 |
| 단백질 | 1,115 aa | 604 aa |
| CT-only | 20개 (중간 Spectrin 영역) |
| AD-only | 5개 |

**AD-only exons** (chr X, minus strand, 낮은 좌표 = 3' mRNA):
```
31,119,239–31,121,930 (2,691 bp)  ← CT exon과 12 bp 차이 (alt 5' splice site)
31,169,443–31,169,601 (  158 bp)
31,172,348–31,172,413 (   65 bp)
31,173,539–31,173,604 (   65 bp)
31,266,810–31,266,967 (  157 bp)
```
- CT-only exon 중 가장 큰 것: 31,119,228–31,121,930 (2,702 bp, CT) → AD는 31,119,239–31,121,930 (2,691 bp, 11 bp 짧은 alt splice)
- 20 CT-only exon은 Spectrin+WW domain 영역 → AD에서 이 영역 대신 5개 소형 exon으로 대체
- **Mechanism**: Dp140 → Dp71 상당 isoform 전환? 또는 뇌 특이적 internal promoter 전환

---

## 8. ZCCHC17 / Oligodendrocyte (Tier C, Brain 최고 점수)

### Exon 구조
| 항목 | CT (ENST00000615916) | AD (ENST00000616393) |
|------|---------------------|---------------------|
| exon 수 | 7 | 7 (동일) |
| 공유 exon | 5개 |
| 단백질 | 263 aa | 179 aa (−84 aa) |

**차이**:
```
CT exon 2: 31,296,982–31,297,213 (231 bp)
AD exon 2: 31,296,982–31,297,075 ( 93 bp) → 138 bp 짧은 alt 3' splice site

CT exon X: 31,337,175–31,337,275 (100 bp) → AD에 없음
AD exon Y: 31,310,044–31,310,164 (120 bp) → CT에 없음
```
- S1 domain (RNA-binding) 소실 → 84 aa 짧은 단백질
- Mechanism: **Alternative 3' splice site + exon exchange** in C-terminal coding region

---

## 종합 메커니즘 분류

| 케이스 | 메커니즘 | 특이점 |
|--------|---------|--------|
| ERCC6L2 | Alternative Last Exon (ALE) | 공유 exon 0개; stable 212 aa; NOT NMD |
| NOL8 | Cassette Exon Cluster Inclusion | AD 이소폼이 CT보다 큼; 5 exon cluster |
| AZIN1 | Alt splice site + UTR exon | 단백질 동일; phyloP=2.348 UTR regulation |
| DDX19A | Retained Intron "super-exon" | 9 exon → 1 super-exon; TSS +16 kb |
| DOCK11 | Alternative Last Exon (ALE) | 공유 exon 0개; AD transcript 3' tail only |
| FANCA | Alternative Termination / exon skipping | 43→11 exon; N-terminal only |
| DMD | Internal exon exchange | Dp140→Dp71 상당; Spectrin/WW → SOGA |
| ZCCHC17 | Alt 3' splice site + exon exchange | S1 domain 84 aa 소실 |

---

*생성일: 2026-06-25 | 분석 방법: BISECT analysis.json exon 좌표 비교 + 수동 NMD 계산*
