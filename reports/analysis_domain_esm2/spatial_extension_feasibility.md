# Long-read Spatial Transcriptomics 확장 가능성 — 비판적 타당성 분석

**작성일**: 2026-06-16  
**목적**: PRISM+BISECT 발견의 공간 전사체 연동 연구 가능성 평가  
**결론 선요약**: 기술적으로 타당하며 생물학적 가치가 있으나, AD-specific spatial long-read 공개 데이터가 현재 부재하므로 즉시 실행 불가. 단기적으로는 벌크 long-read AD 코호트(Nat Biotech 2024)를 통한 재현성 검증이 현실적이며, spatial 연동은 중기(2–3년) 로드맵으로 설정해야 한다.

---

## 1. 연구 질문의 명확화

### 현재 PRISM+BISECT가 아는 것

| 축 | 현재 해상도 | 데이터 소스 |
|----|-------------|-------------|
| 세포 유형 | 단일세포 수준 (excitatory neuron, oligodendrocyte 등) | LR scRNA-seq (뇌 단일세포) |
| 아이소폼 | 전장 서열 해상도 (exon-level) | IsoQuant, 10,817 novel isoforms |
| 조건 | AD vs CT (컨디션 비교) | BISECT 84 PASS cases |
| 공간 | **없음** | — |

### Spatial이 추가해줄 수 있는 것

```
"어떤 세포 유형에서 스위치가 일어남"
            +
"그 세포들이 뇌 조직의 어느 위치에 있을 때 스위치가 발생함"
            ↓
"AD의 공간적 전파 경로와 isoform switch 패턴이 연동되어 있는가"
```

이것은 재현성 확인이 아니라 **새로운 생물학적 차원**이다. AD가 entorhinal cortex → hippocampal CA1 → association cortex 순서로 전파된다는 Braak staging과 isoform switch의 공간 패턴이 일치한다면, BISECT 발견이 단순한 상관 관계를 넘어 **AD 진행 메커니즘을 공간적으로 추적할 수 있는 분자 지표**가 된다.

---

## 2. 핵심 후보 분석

### 2.1 KIF21B — 가장 강력한 공간 스토리

**현재 BISECT 발견 요약:**
- CT isoform: exon17 포함 (motor domain 온전), PRISM GO:microtubule-based movement score = **0.975** (전체 데이터셋 100th percentile)
- AD isoform: WD40-repeat (motor domain 소실), score = 0.105
- 전환 배율: 9.3×, DTU p = 9.3×10⁻⁸ (padj = 1.54×10⁻¹⁰)
- Trans-regulator: SRSF5 (logFC −0.279), RBFOX1 (logFC +0.180), M8 regulatory module
- 독립 재현: Ebbert long-read 코호트 MWU p = 0.026, delta = −0.30

**Spatial이 보여줄 수 있는 것:**

*층위 1 — 뇌 영역 수준 (tissue spatial, ~10 μm 해상도)*
```
예측 가설:
  AD 신경퇴행 순서: 해마 CA1 ≫ 내후각피질 2층 > 전두엽 연합 피질 > 1차 시각 피질

  Spatial 데이터로 테스트:
  피질 층(layer)별 KIF21B motor isoform 비율
  → Layer II/III (AD 조기 취약) < Layer V/VI (AD 후기)이면
    "isoform switch가 AD 취약성 지형을 반영함"
```

*층위 2 — 뉴런 내부 수준 (sub-cellular, <1 μm 필요)*
```
사용자 질문의 핵심: 
  "motor domain 온전 isoform이 dendrite 말단에서 역할하고,
   WD40 isoform이 Golgi 주변에서 역할하는가?"

  생물학적 근거:
  - 완전한 kinesin-2 motor → processive movement along MT → 수지상돌기 원위부 도달
  - WD40-only isoform → 모터 부재 → soma/Golgi 주변 억류
  - AD: motor isoform 감소 → 화물(미토콘드리아, vesicle)이 원위부 미도달
         → dystrophic neurite와 synaptic terminal 에너지 고갈

⚠️ 비판적 평가:
  이 층위는 현재 어떤 spatial 기술로도 검증 불가
  Spl-ISO-Seq 최고 해상도(220 nm, Spl-ISO-Seq2)도 subcellular localization 구분 불가
  → smFISH/ORCA + 고해상도 현미경 필요 (별도 실험 계획 필요)
```

**Spatial 스토리의 강점**: 모터 단백질의 기능이 본질적으로 공간적(어디서 어디로 운반)이므로, spatial transcriptomics 적용의 생물학적 정당성이 가장 명확함.

---

### 2.2 NDUFS4/7/8 (Complex I 삼각수렴) — 에너지 지형도 가설

**현재 발견:** AD에서 Complex I 조립 관련 3개 유전자가 동시에 isoform switch. 미토콘드리아 클러스터 확정.

**Spatial 가설:**
```
ATP 소비 지형: 시냅스 말단 > 수초 노드 > 세포체

  Spl-ISO-Seq로 테스트 가능한 것:
  V1 시각 피질 4C층 (thalamic input이 집중, ATP 수요 최고) vs 1층 비교
  → NDUFS4 isoform 비율이 층별로 다른가?

  AD 적용 예측:
  에너지 수요가 높은 층에서 dysfunction isoform이 먼저 축적됨
  = Complex I 기능 저하가 에너지 수요 지형을 따라 공간적으로 전파
```

**비판적 평가:**
- 미토콘드리아 이상이 AD에서 공간적으로 진행된다는 것은 이미 단백질 수준에서 알려져 있음 (COX 염색 등)
- Spatial isoform 데이터가 추가하는 것: 어떤 specific isoform이 관여하는가 → PRISM의 기여
- 단, 이 주장이 KIF21B보다 mechanistic 스토리가 약함 (수렴이 흥미롭지만 공간적 예측력이 낮음)

---

### 2.3 DLG1 — 시냅스 층 특이성

**현재 발견:** BISECT 5× enrichment, 시냅스 scaffold 기능 연관.

**Spatial 가설:**
```
DLG1 (SAP97): 시냅스후막 scaffolding
  - N-terminal L27 domain 포함 isoform → PSD 앵커링 → 시냅스 유지
  - L27 domain 없는 isoform → 확산성 → 시냅스 기능 약화

  AD에서 시냅스 소실은 layer-specific:
  피질 Layer II/III dendrites에서 조기 spine loss
  → DLG1 synaptic isoform이 이 층에서 선택적으로 감소?
```

**비판적 평가:**
- DLG1 isoform 기능 구분이 KIF21B보다 문헌에서 덜 정립됨
- PRISM score 해석이 GO term 기반이므로 "시냅스 scaffold"라는 구체적 위치 예측에 한계
- Spatial story는 가능하나 KIF21B와 NDUFS4/7/8에 비해 3순위

---

## 3. 공개 데이터 현황 — 비판적 평가

### 현재 존재하는 것

| 데이터셋 | 조직 | Long-read | Spatial | AD | 툴 호환성 |
|---------|------|-----------|---------|-----|----------|
| Spl-ISO-Seq (Nat Comm 2025) | 인간 시각 피질 | ✅ ONT | ✅ ~10μm | ❌ 비AD (발달) | ✅ Spl-IsoQuant ≈ IsoQuant |
| Spl-ISO-Seq2 (bioRxiv 2025) | 마우스 성체 뇌 | ✅ | ✅ 220nm | ❌ 마우스 | ✅ |
| Nat Biotech 2024 (AD 전두엽) | 인간 전두엽 피질 | ✅ ONT | ❌ 벌크 | ✅ AD 6명+CT 6명 | 확인 필요 |
| GSE158450 (Nat Comm 2021) | 마우스 해마/전전두엽 | ✅ PacBio | ❌ scRNA-seq | ❌ 마우스 | 부분 호환 |
| PRJNA664117 | 인간 피질 | ✅ PacBio | ❌ 벌크 | ❌ | 부분 호환 |

**결론**: **AD + 인간 뇌 + Long-read + Spatial을 동시에 충족하는 공개 데이터는 현재 존재하지 않는다.** 이는 기술적 한계가 아니라 이 분야가 2025–2026년에 막 형성되고 있기 때문이다.

### Transcript ID 호환성 기술 분석

```
Reference isoforms (ENST IDs):
  → 동일 GENCODE 버전 사용 시 완전 직접 매핑 가능 ✅
  → 대부분의 확인된 isoform이 여기 해당

Novel isoforms (IsoQuant 자동 부여 ID):
  → ID 자체는 데이터셋 간 불일치 (run-specific 증분 ID)
  → 매핑 방법: GTF exon coordinate intersect (bedtools)
  → 동일 exon boundary를 가지면 동일 isoform으로 판정 가능

Spl-IsoQuant → IsoQuant 호환:
  → Spl-IsoQuant은 IsoQuant의 직계 fork
  → 동일 알고리즘, 동일 GTF 파싱 로직
  → 매핑 정확도 가장 높음 ✅
```

---

## 4. 비판적 평가 — 이 연구 확장이 논문에 기여하는가?

### 기여하는 경우 (긍정)

1. **BISECT의 예측 능력 입증**: "우리 모델이 공간적 맥락을 갖는 아이소폼 switch를 예측할 수 있음" → 단순한 AD 마커 목록이 아닌, 공간적으로 해석 가능한 메커니즘 제시
2. **KIF21B 스토리 완성**: motor domain 소실 → dendritic transport 실패 → synaptic terminal 기능 저하라는 인과 체인이 공간 데이터로 시각화되면 강력한 Figure
3. **Nature Methods 기준 충족**: "방법론적 기여 + 생물학적 발견"이 함께 있어야 함. Spatial 연동은 BISECT의 생물학적 발견이 독립적 맥락을 가짐을 보여줌

### 기여하지 않는 경우 (한계)

1. **인과성 부재**: Spatial correlation은 인과 관계를 증명하지 않음. Layer II/III에서 KIF21B motor isoform이 감소해도 "KIF21B switch가 AD를 유발함"은 별도 실험 필요
2. **해상도의 근본 한계**: Sub-cellular (Golgi vs. dendrite tip) 질문은 현재 어떤 spatial LR 기술로도 답할 수 없음. 논문에서 이를 주장하면 reviewer 즉시 지적
3. **데이터 부재에 의한 추론 위험**: 공개 데이터가 없는 상태에서 "spatial 분석이 가능할 것이다"는 추론을 논문에 포함하면 speculation으로 분류됨

### 종합 판단

| 시나리오 | 실현 조건 | 논문 기여도 |
|---------|----------|------------|
| Nat Biotech 2024 AD 코호트로 KIF21B 재현 | SRA 접근 + IsoQuant reanalysis | 독립 코호트 재현 → Methods에 추가 ★★★ |
| Spl-ISO-Seq 시각 피질에서 층별 isoform 분포 | 현재 데이터 재분석 | Supporting evidence → 생물학적 타당성 ★★ |
| AD 환자 spatial LR-seq 신규 생산 | 자체 실험 or 협업 | 완전한 spatial 검증 → 별도 논문 ★★★★ |
| Subcellular localization (Golgi vs. dendrite) | smFISH 실험 필요 | 가장 강력한 증거 but 별도 논문 수준 |

---

## 5. 합리적 로드맵

### 단기 (현재 논문에 포함 가능)

```
1. Nat Biotech 2024 (SRA 접근):
   → KIF21B exon17 isoform ratio: AD vs CT
   → NDUFS4/7/8 Complex I isoform direction 재현
   → 결과: "독립 bulk long-read 코호트에서 유전자-수준 재현"
   → 논문 내 위치: Results §4.x 또는 Supplementary

2. 현존 Spl-ISO-Seq 데이터 (연락 or 출시 대기):
   → KIF21B isoform 층별 분포 (비AD 상태)
   → 결과: "KIF21B isoform이 피질 층에 따라 공간적으로 차별 발현됨"
   → 논문 내 위치: Discussion의 기전 지지 증거
```

### 중기 (후속 논문 또는 revision)

```
3. Spl-ISO-Seq 저자 협업 타진:
   → AD 샘플에 Spl-ISO-Seq 적용 공동 연구 제안
   → BISECT 발견 목록 제공 → 그들의 spatial 기술로 검증
   → 예상 소요: 6–12개월

4. Nature Neuro or Cell Reports 후속 논문 타겟:
   → "PRISM+BISECT predicts spatially-resolved isoform switches in AD"
   → KIF21B spatial validation이 핵심 Figure
```

### 장기 (3년+ 전망)

```
5. AD 환자 뇌 조직 직접 Spl-ISO-Seq 적용:
   → NIH Neurobiobank or ROSMAP 뇌 은행 샘플
   → Multiple Braak stages × multiple brain regions
   → 이것이 완성되면 BISECT spatial 전파 지도 완성
```

---

## 6. 논문에서의 언어 처리 권고

### 현재 논문에 포함 가능한 표현

```markdown
(Discussion)
"The isoform switches identified by BISECT in AD — particularly the
 motor-domain loss in KIF21B and the Complex I assembly factor
 convergence in NDUFS4/7/8 — are mechanistically predicted to show
 spatially organized expression patterns across cortical layers,
 consistent with the known Braak staging trajectory of AD pathology.
 Emerging spatial long-read technologies (Spl-ISO-Seq; Nat Comm 2025)
 now offer the resolution required to directly test this prediction,
 representing a natural extension of the PRISM+BISECT framework."
```

### 포함하면 안 되는 표현

```
❌ "spatial 데이터에서 확인되었다" (데이터 없음)
❌ "KIF21B motor isoform은 dendrite에, WD40 isoform은 Golgi에 위치한다"
   (sub-cellular localization은 현재 어떤 데이터로도 확인 불가)
❌ "spatial 패턴이 Braak staging과 일치한다" (실제 측정 없이 주장 불가)
```

---

## 7. 결론

Long-read spatial transcriptomics와의 연동은 **생물학적으로 타당하고 방법론적으로 실현 가능하지만, 현재 이를 직접 실행할 공개 데이터가 존재하지 않는다.** 이는 PRISM+BISECT의 한계가 아니라 **이 연구가 해당 분야의 기술 발전보다 앞서 있음**을 나타낸다.

현실적인 전략은:
1. **즉시**: Nat Biotech 2024 AD long-read 코호트로 유전자 수준 재현
2. **단기**: Spl-ISO-Seq 시각 피질 데이터에서 층별 분포로 생물학적 타당성 지지
3. **중기**: Spl-ISO-Seq 저자 협업 또는 별도 spatial 논문으로 완전 검증

KIF21B는 공간 스토리가 가장 명확한 후보이며, motor domain isoform의 공간적 분포 차이가 AD의 dendritic transport 실패 메커니즘과 직결된다는 점에서 향후 가장 강력한 validation target이다.

---

*참조 논문:*
- Spl-ISO-Seq: Nat Comm 2025, DOI 10.1038/s41467-025-63301-9 (PMC12397408)
- Spl-ISO-Seq2: bioRxiv 2025.06.25.661563
- AD long-read (Nat Biotech 2024): DOI 10.1038/s41587-024-02245-9
- Longcell (Nat Comm 2025): DOI 10.1038/s41467-025-60902-2
- Cell-type resolved human cortex: bioRxiv 2025.11.25.690524
