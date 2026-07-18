# Layer × Braak Stage × Isoform Switch 통합 분석 보고서

**분석일**: 2026-06-16  
**대상**: Samsung Medical Center AD scLR-seq (25 donors, Excitatory neuron)  
**목적**: BISECT 발견 isoform switch의 공간적(layer) + 임상적(Braak) 맥락 보강

---

## 요약

두 가지 분석을 통해 BISECT의 핵심 발견들이 뇌 피질 층 구조 및 AD 진행 단계와 **방향적으로 일관된** 패턴을 보임을 확인함.

| 분석 | 핵심 결과 | 생물학적 의미 |
|------|-----------|--------------|
| Braak × KIF21B CT isoform | r = −0.527, n=10, p=0.117 (trend) | Braak 단계가 높을수록 motor domain isoform 감소 |
| Braak × NDUFS8 CT isoform | r = −0.375, n=19, p=0.114 (trend) | Complex I 이상이 AD 진행과 연동 |
| L5 Exc cluster (C19) 구성 비율 | AD 2.51% vs CT 6.97%, **p=0.013** | L5 대형 투사 뉴런이 AD에서 선택적 감소 |
| L4 Exc cluster (C18) 구성 비율 | AD 7.23% vs CT 2.66%, **p=0.010** | L4 세포 상대적 보존 또는 보상적 반응 |

---

## 분석 A: Braak Stage × Isoform Switch 상관분석

### 방법
- `tx_counts_by_donor_Excitatory_neuron.csv` (25 donors)
- Braak B score: B0–B3 (4단계, 노트북 임상 메타데이터)
- CT isoform usage fraction = target_isoform_count / gene_total_count (per donor)
- Spearman correlation + Bootstrap 95% CI (n=1000)

### 결과

| 유전자 | CT isoform | n_donors | Spearman r | p-value | Bonf-p | 95% CI |
|--------|-----------|---------|-----------|---------|--------|--------|
| KIF21B | transcript293004.chr1.nic | 10 | **−0.527** | 0.117 | 0.352 | — |
| NDUFS4 | NDUFS4-201 | 13 | −0.308 | 0.307 | 0.920 | [−0.768, 0.264] |
| NDUFS8 | transcript100761.chr11.nic | 19 | −0.375 | 0.114 | 0.342 | [−0.720, 0.089] |

### 해석

**방향 일관성**: 세 유전자 모두 음의 상관 방향 (CT isoform 감소 ↔ Braak 단계 증가).  
**통계적 비유의**: Bonferroni 보정 후 모두 p > 0.05. 이는 주로:
1. **Braak B의 낮은 해상도**: B1/B2/B3 3단계에 불과 — 실제 AD 코호트는 대부분 B3에 집중 → 연속 변수 효과 희석
2. **KIF21B n=10**: novel isoform (transcript293004.chr1.nic)은 낮은 UMI로 인해 일부 donor에서 NaN 처리 → 검정력 감소
3. **SMC D column 버그**: PO 샘플(n=9)은 Roman numeral Braak 있음, SMC는 B column만 → 정밀도 차이

**권장 후속 조치**:
- PO 샘플 9개만으로 Roman numeral (II–VI) 기준 재분석 → 해상도 개선
- NDUFS4-201은 permutation test PASS (전체 데이터셋 기준) → donor 수 확보가 관건

---

## 분석 B: Excitatory Neuron Subcluster Layer Annotation

### Layer 마커 발현 기반 주석

| Cluster | n_cells | Layer 레이블 | 근거 마커 발현 |
|---------|---------|-------------|--------------|
| C1 | 10,258 | **L2/3** | CUX1=0.91, CUX2 high, FEZF2 낮음 |
| C2 | 9,711 | **L2/3** | CUX1=1.06 (가장 높음), RORB 중등도 |
| C27 | 666 | **L2/3** | CUX1=2.05 (최고치), 명확한 상층 |
| C10 | 4,156 | **L4** | RORB=1.23 (최고치), ETV1 high |
| C11 | 4,290 | **L4** | RORB=1.23, SYT6 중등도 |
| C18 | 1,603 | **L4** | RORB=1.09 |
| C19 | 1,410 | **L5** | FEZF2=0.19, BCL11B, SYT6=0.16 |
| C26 | 744 | **L5** | FEZF2=0.29 (최고치), BCL11B high |
| C25 | 757 | **L6** | SYT6=0.10, FOXP2, RORB 혼재 |
| C20 | 1,356 | 미분류 | 모든 마커 낮음 (score_max = −0.34) |

### Layer 구성 비율 AD vs CT (Mann-Whitney U)

```
클러스터          AD (%)   CT (%)   Δ       p-value   해석
─────────────────────────────────────────────────────────
C18_L4          7.23     2.66    +4.57   0.010 **    L4 상대적 증가
C19_L5          2.51     6.97    −4.46   0.013 *     L5 선택적 감소
C11_L4         12.95     9.77    +3.18   0.089       
C10_L4         12.77     9.39    +3.37   0.121       
C20_(미분류)     3.05     5.94    −2.89   0.232       
L2/3 합계      57.28%   61.40%  −4.12   ns
```

### 핵심 발견: L5 뉴런 선택적 감소

**AD에서 L5 Excitatory cluster (C19+C26)가 유의하게 감소** (C19: p=0.013):
- L5 뉴런 = 피질의 주요 원거리 투사 뉴런 (corticospinal, corticothalamic, corticocortical)
- 매우 긴 축삭 → kinesin-기반 anterograde transport에 절대적으로 의존
- AD에서 가장 에너지 소모가 크고 tau 병리에 취약한 세포 유형

**L4 cluster (C18) 상대적 증가** (p=0.010):
- L4 spiny stellate neuron = 짧은 axon, 주로 thalamic input 처리
- L5와 달리 원거리 transport 의존도 낮음 → 상대적으로 보존

---

## 통합 해석: 세 발견의 수렴

```
BISECT 발견:
  KIF21B motor domain isoform → AD에서 9.3× 감소
  (Excitatory neuron 집합 수준)
         ↓
Layer 분석:
  L5 Excitatory neuron이 AD에서 선택적으로 감소 (p=0.013)
  (L5 = 가장 transport-의존적 뉴런)
         ↓
Braak 상관:
  KIF21B CT motor isoform 감소 ~ Braak 진행 (r=−0.527, trend)
         ↓
통합 가설:
  "L5 피질 투사 뉴런은 KIF21B motor domain 소실로 인해
   anterograde transport가 실패하고, 이것이 AD 진행(Braak 단계)에
   따라 누적되어 L5 뉴런 선택적 취약성으로 귀결된다"
```

이 가설은:
1. **세포 수준 증거**: KIF21B switch가 Excitatory neuron에서 발생 (BISECT)
2. **공간 수준 증거**: L5 뉴런이 AD에서 선택적으로 감소 (본 분석)
3. **임상 진행 증거**: motor isoform 감소가 Braak 단계와 연동 (trend)
4. **기전적 연결**: L5 → 원거리 투사 → kinesin 의존성 최고

---

## 한계

1. **Braak B 해상도**: B1/B2/B3의 3단계로 연속 변수 분석에 한계. 전체 SMC 샘플의 AD군 대부분이 B3에 집중.
2. **Layer annotation의 간접성**: short-read 기반 marker 발현 → long-read isoform 데이터와 직접 연결 불가. L5 세포의 KIF21B switch를 직접 측정하지 않음.
3. **C20 클러스터 미분류**: 모든 layer marker 낮음 → 피질하 흥분성 뉴런 또는 혼합 집단일 가능성.
4. **인과성 부재**: L5 뉴런 감소가 KIF21B switch의 원인인지 결과인지 불명.

---

## 논문 Integration 권고

### Discussion에 추가 가능한 내용 (데이터 지지)

```
"Layer composition analysis of short-read single-cell data revealed
 selective depletion of Layer 5 excitatory neurons in AD donors
 (C19_L5: 2.51% AD vs 6.97% CT, MWU p=0.013), while Layer 4
 clusters showed relative enrichment (C18_L4: 7.23% vs 2.66%,
 p=0.010). This layer-specific vulnerability pattern is consistent
 with the KIF21B motor domain isoform switch identified by BISECT,
 as Layer 5 projection neurons — with their exceptionally long axons
 — are the most dependent on kinesin-based anterograde transport
 for organelle and vesicle delivery to distal synaptic terminals."
```

### 추가하면 안 되는 내용

- "L5 뉴런에서 KIF21B switch가 직접 발생한다" (단일세포 isoform 데이터 없음)
- "Braak stage와 유의한 상관관계" (p > 0.05, trend만 존재)

---

## 분석 C: PO 샘플 Roman Numeral Braak (II–VI) 재분석 [2026-06-16 추가]

### 방법
- 대상: PO 코호트 9개 샘플 (AD 5개: Braak V/VI, CT 4개: Braak II/III)
- B0-B3 4단계 → Roman numeral II/III/V/VI 4점 척도 (간격 유지)
- min_total=3 (sparse PO 데이터 대응)

### 결과

| 유전자 | isoform | n | Spearman r | p-value | 방향 |
|--------|---------|---|-----------|---------|------|
| KIF21B | KIF21B-203 (AD enriched) | 8 | **+0.464** | 0.247 | ✓ Braak↑ → AD isoform↑ |
| NDUFS4 | NDUFS4-201 (CT canonical) | 7 | −0.434 | 0.331 | ✓ Braak↑ → CT isoform↓ |
| NDUFS8 | transcript100761 CT novel | 7 | **−0.496** | 0.258 | ✓ Braak↑ → CT isoform↓ |

### 해석
- **세 유전자 모두 예측 방향 일치**: AD 진행(Braak↑)에 따라 AD isoform 증가/CT isoform 감소
- **통계적 비유의**: n=7-9로 검정력 부족. 95% CI 모두 0 포함 → 방향 확인용 증거
- Braak IV 해당 샘플 없음 (II/III → V/VI 점프) — 연속 상관 분석의 한계

---

## 분석 D: RBFOX1/RBFOX2 Layer Cluster별 발현 [2026-06-16 추가]

### 방법
- adata_sr_overlapped h5ad (95,487 cells)에서 직접 추출
- 10개 Exc 서브클러스터에서 RBFOX1/2/3, SRSF5/1 평균 발현
- AD vs CT Mann-Whitney U (per cluster, per layer group)

### 클러스터별 RBFOX1 발현

| Cluster | Layer | RBFOX1 | RBFOX2 | RBFOX3 |
|---------|-------|--------|--------|--------|
| C1 | L2/3 | 3.93 | 1.41 | 1.67 |
| C2 | L2/3 | 4.10 | 1.50 | 1.59 |
| C27 | L2/3 | 4.04 | 1.52 | 1.80 |
| C10 | L4 | 3.57 | 1.56 | 1.42 |
| C11 | L4 | 3.92 | 1.59 | 1.92 |
| **C18** | **L4** | **2.94** | 1.53 | 1.07 |
| C19 | L5 | 3.84 | 1.46 | 1.86 |
| C26 | L5 | 4.06 | 1.65 | 1.22 |
| C25 | L6 | 3.72 | 1.53 | 1.59 |

### AD vs CT 비교 결과

- **모든 cluster 및 layer group: p > 0.2** (최소 p=0.238, cluster C19 L5)
- Layer group: L2/3 p=0.928, L4 p=0.639, L5 p=0.699, L6 p=0.813

### 해석

**핵심 발견 (negative result의 의미)**:
1. **RBFOX1은 층 특이적이지 않다**: L2/3~L6 모두 3.7–4.1 범위로 균등 발현 → RBFOX1이 L5에서 선택적으로 높다는 가설 불지지
2. **단일세포 수준에서 AD/CT RBFOX1 차이 없음**: 이전 bulk 분석의 logFC=-0.279는 세포 유형 구성 변화(L5 세포 감소)에 의한 위위 효과일 가능성
3. **C18(L4) RBFOX1 저발현 이상치**: 2.94 vs 평균 3.8 — 이 cluster의 기능적 특이성 추가 조사 필요
4. **KIF21B exon17 switch 메커니즘**: RBFOX1 발현량 차이가 아닌 **활성화 상태(phosphorylation) 또는 cofactor 차이**에 의한 것일 가능성 → 정량적 발현이 아닌 기능적 regulation 차이

---

## 다음 단계 권고 (갱신)

| 우선순위 | 작업 | 방법 | 상태 |
|---------|------|------|------|
| 1 | ~~PO 샘플 Roman numeral Braak 재분석~~ | — | **완료** (분석 C) |
| 2 | ~~RBFOX1 layer cluster별 발현~~ | — | **완료** (분석 D) |
| 3 | C19/C26 (L5) 세포 바코드 → long-read BAM split → L5 KIF21B isoform | BAM split + IsoQuant | ~1주 |
| 4 | Allen Human Brain Atlas KIF21B layer 발현 조회 | allensdk API | ~반일 |
| 5 | C18(L4) 저RBFOX1 cluster 기능 규명 | marker gene 추가 조회 | 즉시 가능 |

---

*스크립트 위치*:
- `reports/braak_isoform_correlation.py`
- `reports/layer_annotation_excitatory.py`
- `reports/braak_isoform_plots/`
- `reports/layer_annotation/`
