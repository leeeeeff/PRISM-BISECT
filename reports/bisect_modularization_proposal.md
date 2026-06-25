# BISECT 심층 분석 모듈화 제안 (2026-06-25)

현재 심층 분석(exon 구조 비교, NMD 수동 계산, 메커니즘 분류)을 파이프라인에 통합하기 위한 설계 논의.

---

## 현재 BISECT 모듈 현황 (M1–M13)

| 모듈 | 기능 | 현재 한계 |
|------|------|-----------|
| M1 | 서열 추출 | NIC/NNIC transcript 종종 누락 |
| M2 | Pfam/HMMER 도메인 | E<0.01 임계값; 일부 경계 도메인 미검출 |
| M3–M5 | (내부) | — |
| M6 | NMD screen | **PTC/EJC 거리 계산 실패율 높음** (새 케이스 대부분 None) |
| M7 | 서열 검증 | transcript pair 비교 |
| M8 | Regulatory context | TF/RBP 조절 인자 추론 |
| M9 | Promoter usage | TSS 비교 |
| M10 | APA analysis | TTS 비교 |
| M11 | AlphaFold | pLDDT 비교 |
| M12 | PPI (STRING) | combined_score 기반 |
| M13 | Conservation | phyloP 평균 |

**주요 공백**: Exon-level 구조 비교 모듈 없음; NMD 계산 신뢰도 낮음; 메커니즘 자동 분류 없음; 교차 케이스 수렴 검출 없음.

---

## 신규 모듈 제안

### M14 — Exon Structure Comparator

**목적**: CT vs AD 이소폼의 exon 구조를 정량적으로 비교하여 스플라이싱 이벤트 유형을 자동 분류.

**입력**:
- `ct_info.exons`, `ad_info.exons`: genomic exon 좌표 리스트
- `strand`, `chrom`

**출력**:
```python
{
  "shared_exons": [...],         # 공유 exon 좌표
  "ct_only_exons": [...],        # CT 특이적 exon
  "ad_only_exons": [...],        # AD 특이적 exon
  "event_type": "retained_intron | cassette_exon | alt_splice_site | ale | alt_tss | complex",
  "event_details": {
    "n_ct_only": int,
    "n_ad_only": int,
    "n_shared": int,
    "ct_only_total_bp": int,
    "ad_only_total_bp": int,
    "ct_unique_in_coding": bool,  # CDS 내 CT-only 여부
    "ad_unique_in_coding": bool,
  },
  "super_exon_detected": bool,   # CT 다수 exon → AD 1개 대형 exon
  "ale_detected": bool,          # shared=0 또는 5'만 공유
}
```

**분류 로직**:
```
shared == 0 → Alternative Last Exon (ALE) 또는 Alternative TSS
shared 존재 + ad_only 대형 exon (>500 bp, CT의 여러 exon 포함) → Retained Intron
ad 단백질 > ct 단백질 → Cassette Exon Cluster Inclusion
단백질 동일 + ad_only exon → Alternative Splice Site or UTR exon
```

**구현 위치**: `modules/m14_exon_comparator.py`
**의존성**: `ct_info`, `ad_info` (M1에서 제공됨)
**소요 시간**: O(n) exon 수 기준, 즉시

---

### M15 — NMD Calculator v2

**목적**: 현재 M6의 NMD 예측이 신규 케이스에서 대부분 None을 반환하는 문제 해결.

**현재 문제**: `m6_nmd_screen`이 PTC 위치를 계산하지 못하는 경우:
- NIC/NNIC transcript의 CDS 시작 위치 불명확
- 새 케이스(ERCC6L2, AZIN1, NOL8 등) 모두 `ptc=None, ejc_dist=None`

**입력**:
- exon 좌표 (M14 output 또는 ct_info/ad_info)
- 단백질 서열 (M1 output)
- CDS 길이 (aa × 3)

**출력**:
```python
{
  "n_exon_junctions": int,       # 총 EJC 수
  "ejc_positions_nt": [...],     # mRNA 좌표 기준 EJC 위치
  "cds_end_nt": int,             # mRNA 좌표 기준 stop codon 위치
  "ptc_to_last_ejc_distance": int,  # 핵심: PTC → 마지막 EJC 거리
  "nmd_predicted": bool,         # >50 nt → True
  "nmd_confidence": "high|low",  # CDS 시작 추정 신뢰도
  "notes": str,
}
```

**계산 로직**:
```
1. 단백질 길이로 CDS 길이 추정 (aa × 3 + 3 stop)
2. exon 좌표에서 spliced mRNA 좌표 시스템 구성
3. EJC = 각 exon junction에서 downstream 20-24 nt
4. PTC 위치 = CDS 길이 위치 (ATG 추정 필요)
5. 최종 EJC에서 PTC까지 거리
```

**구현 위치**: `modules/m15_nmd_v2.py`
**의존성**: M1 (서열), M14 (exon 구조)
**소요 시간**: 즉시

---

### M16 — Mechanism Classifier

**목적**: M9(TSS), M10(APA), M14(exon 구조), M15(NMD), M2(도메인) 결과를 통합하여 생물학적 메커니즘을 자동 분류.

**입력**: 기존 모듈 출력 통합

**출력**:
```python
{
  "primary_mechanism": "nmd_switch | domain_loss | domain_gain | ale | 
                         retained_intron | cassette_exon | alt_tss | 
                         alt_splice_site | minor_exon | complex",
  "secondary_mechanism": [...],   # 복합 메커니즘
  "mechanism_evidence": {
    "nmd": bool,
    "domain_change": bool,
    "exon_count_change": int,
    "protein_length_change_aa": int,
    "tss_shift_bp": int,
    "tts_shift_bp": int,
    "super_exon": bool,
    "ale": bool,
  },
  "narrative": str,  # 자동 생성 1-2 문장 요약
}
```

**분류 우선순위**:
```
1. NMD → nmd_switch (protein 소실)
2. 도메인 손실 + 단백질 단축 > 200 aa → major_truncation
3. 도메인 소실 → domain_loss
4. 도메인 획득 → domain_gain
5. shared_exon==0 + 단백질 단축 → ale
6. super_exon 검출 → retained_intron
7. AD 단백질 > CT → cassette_exon_inclusion
8. 단백질 동일 + TSS 이동 > 500 bp → alt_tss
9. 단백질 동일 + 도메인 변화 없음 → minor_exon or alt_splice_site
```

**구현 위치**: `modules/m16_mechanism_classifier.py`
**의존성**: M2, M6/M15, M9, M10, M14
**소요 시간**: O(1)

---

### M17 — Cross-Case Convergence Detector

**목적**: 현재 파이프라인은 케이스를 독립적으로 분석하므로 교차 케이스 수렴(RNA 대사 축, DNA 복구 축 등)을 자동 감지하지 못함.

**입력**: 전체 `bisect_cases.json` (배치 실행 후)

**출력**:
```python
{
  "convergence_axes": [
    {
      "axis_name": "RNA metabolism",
      "genes": ["DDX19A", "DIS3", "CNOT11", "NOL8", "ZCCHC17"],
      "cell_types": {"DDX19A": "Inhibitory", "DIS3": "Oligodendrocyte", ...},
      "go_terms": ["GO:0006396", "GO:0006402"],
      "axis_significance": float,   # hypergeometric p for GO enrichment
      "notes": str,
    },
    ...
  ],
  "domain_family_clusters": [
    {"family": "Spectrin", "genes": ["DMD", "SYNE1", "SPTBN1"], ...},
    {"family": "KRAB", "genes": ["ZNF736", "ZNF582", "ZNF268"], ...},
  ],
}
```

**구현 방법**:
- GO term 공유 기반 hypergeometric 검정 (PRISM의 18 GO term 활용)
- Pfam 도메인 family 그룹핑 (같은 domain superfamily에서 소실/획득)
- 세포 유형 조합 패턴 (같은 축이 다른 cell type에서)

**구현 위치**: `scripts/convergence_detector.py` (post-processing)
**의존성**: bisect_cases.json 전체
**소요 시간**: ~1분

---

## 구현 우선순위

| 우선순위 | 모듈 | 이유 | 난이도 |
|---------|------|------|--------|
| 1 | M14 Exon Comparator | 현재 심층 분석의 핵심; 재현 가능한 exon 구조 기술 | ★★☆ |
| 2 | M15 NMD Calculator v2 | M6 실패율 높음; ERCC6L2/DDX19A 등 핵심 케이스에서 필수 | ★★☆ |
| 3 | M16 Mechanism Classifier | M14+M15 의존; 자동화 가능한 논리 있음 | ★★★ |
| 4 | M17 Convergence Detector | post-processing; 논문 Discussion에서 강력한 발견 | ★★☆ |

---

## 설계 원칙 (기존 파이프라인과 호환)

```python
# 신규 모듈 인터페이스 규칙
def run_m14(case_data: dict, config: dict) -> dict:
    """
    Returns: {"m14_exon_comparator": {...결과...}}
    모든 신규 모듈은 단일 dict 반환, 키는 모듈명으로 통일.
    """
    ...

# orchestrate.py에 통합 위치
# Stage 1 후: M14 (exon structure) → M15 (NMD v2) → M16 (classification)
# Stage 2 후 (배치): M17 (convergence)
```

---

## M14 구현 스케치 (즉시 실행 가능)

```python
# modules/m14_exon_comparator.py
def run_m14(case: dict, config: dict) -> dict:
    ct_info = case.get("ct_info", {}) or {}
    ad_info = case.get("ad_info", {}) or {}
    
    ct_exons = [tuple(e) for e in ct_info.get("exons", [])]
    ad_exons = [tuple(e) for e in ad_info.get("exons", [])]
    
    ct_set = set(ct_exons)
    ad_set = set(ad_exons)
    
    shared   = sorted(ct_set & ad_set)
    ct_only  = sorted(ct_set - ad_set)
    ad_only  = sorted(ad_set - ct_set)
    
    ct_only_bp = sum(e[1]-e[0] for e in ct_only)
    ad_only_bp = sum(e[1]-e[0] for e in ad_only)
    
    # Detect super-exon (AD exon spans multiple CT exons)
    super_exon = False
    for ae in ad_only:
        contained = [ce for ce in ct_only if ce[0] >= ae[0] and ce[1] <= ae[1]]
        if len(contained) >= 3:
            super_exon = True
            break
    
    # ALE: no shared or only 5' shared
    ale = len(shared) == 0
    
    # Classify event
    ct_aa = len(case.get("ct_seq", {}).get("seq", "") if isinstance(case.get("ct_seq"), dict) else case.get("ct_seq", "") or "")
    ad_aa = len(case.get("ad_seq", {}).get("seq", "") if isinstance(case.get("ad_seq"), dict) else case.get("ad_seq", "") or "")
    
    if ale:
        event_type = "ale"
    elif super_exon:
        event_type = "retained_intron"
    elif ad_aa > ct_aa and len(ad_only) >= 3:
        event_type = "cassette_exon_cluster"
    elif ct_aa == ad_aa and len(ct_only) + len(ad_only) <= 4:
        event_type = "alt_splice_site"
    elif len(ct_only) > 5:
        event_type = "major_exon_loss"
    else:
        event_type = "complex"
    
    return {
        "m14_exon_comparator": {
            "n_shared": len(shared),
            "n_ct_only": len(ct_only),
            "n_ad_only": len(ad_only),
            "ct_only_bp": ct_only_bp,
            "ad_only_bp": ad_only_bp,
            "net_bp_change": ad_only_bp - ct_only_bp,
            "super_exon_detected": super_exon,
            "ale_detected": ale,
            "event_type": event_type,
            "ct_only_exons": [list(e) for e in ct_only],
            "ad_only_exons": [list(e) for e in ad_only],
        }
    }
```

---

## 논의 포인트

### Q1: M14를 Stage 1과 Stage 2 중 어디에 배치?
**추천: Stage 1 이후, Stage 2 전**
- Stage 1은 delta + dtu_p 필터만 수행
- M14는 exon 구조 정보를 제공하여 **Stage 2 도메인 분석의 맥락 제공**
- "super_exon 검출" → retained intron → Stage 2에서 NMD 우선 확인하도록 유도

### Q2: M14가 현재 M9(TSS)/M10(APA)와 어떻게 다른가?
- M9: promoter 위치 비교 (TSS diff bp)
- M10: poly-A 위치 비교 (TTS diff bp)
- M14: **mRNA 내부 exon 구조** 비교 (splicing 이벤트 유형)
- 세 모듈이 complementary: M9→M14→M10 = TSS → exon body → TTS

### Q3: M17 Convergence Detector는 실시간 vs 배치 중 어떤 방식?
**추천: 배치 post-processing**
- 개별 케이스 파이프라인과 독립적으로 실행
- `python scripts/convergence_detector.py --input bisect_cases.json`
- 실행 주기: 신규 케이스 통합 시마다 (현재처럼 merge_new_bisect_cases.py 실행 후)

### Q4: 기존 84개 케이스를 M14로 소급 적용할 수 있나?
**가능**: analysis.json에 ct_info/ad_info가 이미 있으므로 retroactive 적용 가능
- `scripts/apply_m14_retroactive.py` → 기존 모든 analysis.json에 m14 결과 추가
- bisect_cases.json에 `event_type` 필드 추가 → 앱 필터로 활용

---

*생성일: 2026-06-25*
