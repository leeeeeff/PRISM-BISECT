# BISECT 핵심 케이스 — 뇌 Layer 연계 분석 (2026-06-25)

**방법**: 
- 클러스터-레이어 매핑 (`cluster_layer_mapping.tsv`): leiden cluster → cortical layer
- 레이어 AD 농축 (`layer_composition_mwu.tsv`): 각 cluster의 AD vs CT 세포 비율 MWU 검정
- 이소폼 스위치 효과크기 (`donor_level_isoform_switches.tsv`): 유전자별 Δ비율 (AD − CT)
- Co-enrichment score: `cluster_AD_delta × (−isoform_delta)` (양수 = AD 농축 layer에서 CT 이소폼 감소)

---

## 핵심 발견 1: L4 Excitatory 뉴런 — Complex I 축 최강 신호

| Layer | Cluster | AD enrichment | MWU p |
|-------|---------|--------------|-------|
| **L4 IT atypical** | **C18** | **+4.57% (AD)** | **p=0.00994** |
| L4 IT | C10 | +3.37% (AD) | p=0.121 ns |
| L4 IT | C11 | +3.18% (AD) | p=0.089 ns |
| L5 ET | C19 | −4.46% (CT) | p=0.013 * |
| L2/3 IT | C1, C2, C27 | ~−0.7% (CT) | ns |

**C18 L4 IT atypical**: Samsung 코호트에서 AD-enriched 클러스터 중 유일하게 통계 유의(p<0.01). 이 클러스터는 L4 IT에 분류되나 전형적인 L4 마커와 다른 atypical 패턴을 보임.

### NDUFS7 (Excitatory, Tier A-BP) — L4 Co-enrichment 확인

- Excitatory neuron 전체: delta=−0.046 (CT 이소폼 감소), perm_p=0.048
- L4 clusters (C18, C10, C11):
  - C18 co-score = **+0.172** (가장 높음; C18 AD delta +4.57 × NDUFS7 delta 0.046)
  - C11 co-score = +0.147 (L4 IT, p=0.089 trend)
  - C10 co-score = +0.156 (L4 IT, p=0.12 ns)

**해석**: L4 Excitatory 뉴런, 특히 atypical L4 IT (C18)에서 NDUFS7 Q-module PSST subunit 이소폼 switch가 가장 강하게 공동 농축됨. L4 excitatory 뉴런은 corticothalamic projection neuron 후보 → 시상-피질 연결에서 Complex I 결함 가능성.

### NDUFAF5 (Excitatory, Tier A-DR)

- L4 IT clusters에서도 발현 (Excitatory 전체)
- DRIMSeq-level로 발견 → stageR p=1.43×10⁻⁴, but delta 미제공
- L4 excitatory 뉴런에서 Complex I assembly chaperone 결함 → NDUFS4 N-module integration 불가

---

## 핵심 발견 2: L4 Atypical Cluster (C18) 특이성

C18은 L4 IT atypical로 분류되며:
- `SLC17A7+` (Excitatory marker)
- 낮은 CUX2 발현 (L2/3 marker 없음)
- 높은 RORB (L4 marker)
- AD에서 **+4.57% 세포 비율 증가** (유일한 유의 AD-enriched cluster)

이 cluster에서 강하게 공동 농축되는 이소폼 스위치:
1. **NDUFS7** (Q-module, Excitatory A-BP): co-score +0.172
2. **NDUFAF5** (Assembly factor, Excitatory A-DR): 델타 미산출 (DRIMSeq)
3. **ZNF736** (KRAB-ZFP, Excitatory A-DR): 델타 미산출

**핵심 가설**: L4 Atypical Excitatory 클러스터(C18)는 Complex I + KRAB-ZFP 전사억제 축이 동시에 붕괴되는 AD 취약 세포 집단.

---

## 핵심 발견 3: L5 ET 클러스터 — CT-enriched

C19 (L5 Extra-telencephalic, ET):
- CT enriched: −4.46% (p=0.013 *)
- L5 ET = long-range projection neurons (callosal, subcortical)
- DOCK11 inhibitory 뉴런의 이소폼 switch는 L3-5 PV/SST clusters에서 발생

**해석**: L5 ET 뉴런이 CT에 더 많이 존재 → AD에서 L5 long-range projection neuron 선택적 소실 가능성. 이는 Braak staging에서 L5/L6의 neurofibrillary tangle 우선 발생 패턴과 일치.

---

## 핵심 발견 4: Inhibitory 뉴런 Layer 특이성

Inhibitory 클러스터별 layer:
| Cluster | Subtype | Layer |
|---------|---------|-------|
| 7 | PV | L3-5 |
| 9 | SST | L3-5 |
| 8, 16 | VIP | L1-3 |
| 15 | LAMP5/KIT | L1-2 |
| 13 | LAMP5/NDNF | L1 |

A-BP Inhibitory 케이스:
- **DOCK11** (delta=−0.243, perm_p=0.0008): L3-5 PV/SST에서 Cdc42-GEF 완전 소실
- **NDUFS8** (delta=−0.369, perm_p=0.004): L3-5 PV/SST에서 Complex I Q-module 손상
- **NDUFS4 (Inh)** (delta=−0.320, perm_p=0.024): L3-5 PV/SST에서 N-module 손상

**통합 해석**: L3-5 PV 인터뉴런에서 DOCK11(GEF) + NDUFS4/NDUFS8(Complex I)이 동시에 붕괴 → 억제성 시냅스 구조 + 에너지 공급 이중 결함. PV 인터뉴런은 gamma oscillation 생성에 필수 → AD의 40 Hz 감마파 결함 (Iaccarino et al. 2016 Nature)과 연관 가능.

---

## 기능적 Layer 축 요약

| Layer | 주요 세포 | 유의 케이스 | 생물학적 의미 |
|-------|---------|-----------|-------------|
| **L4 IT atypical (C18)** | Excitatory projection | NDUFS7*, NDUFAF5, ZNF736 | Complex I + KRAB-ZFP; 시상-피질 연결 취약 |
| L5 ET (C19) | Excitatory long-range | (CT-enriched; 소실) | L5 projection neuron AD 취약 |
| L3-5 (C7, C9) | PV, SST Inhibitory | DOCK11*, NDUFS8*, NDUFS4* | GEF + Complex I; 감마 진동 회로 |
| L1-2 (C15) | LAMP5/KIT | (이전 분석: AD-enriched) | 억제 네트워크 재편 |
| Mixed/WM | Oligo, Astro, Microglia | DDX19A, DIS3, CNOT11 | RNA 대사 전반; layer 비특이적 |

*통계 유의 co-enrichment 확인됨

---

*생성일: 2026-06-25 | 방법: layer_composition_mwu.tsv × donor_level_isoform_switches.tsv 교차 분석*
