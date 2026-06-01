# Paper-Check: NMI / Nature Methods 기여도 구조 분석

**작성일**: 2026-04-10  
**대상**: v6 프레임워크 전체 기여 주장 구조  
**기준**: Nature Machine Intelligence / Nature Methods 심사 기준  

---

## 1. 기여 주장별 평가

### Claim 1: "Novel isoform function prediction in long-read RNA-seq context"

**판정: Main ✅ — 단, ground truth 정당화 필요**

| | 내용 |
|--|------|
| **강점** | BambuTx 36,748 isoform → 기존 UniProt에 없는 신규 서열 대상. DeepGO/CAPE/원본 DIFFUSE 모두 known protein 대상 → 명확한 공백 존재. Long-read RNA-seq 보편화로 2023년 이후 생물학적 수요 급증. |
| **예상 공격** | "BambuTx isoform에 gold standard label이 없다. y_test가 gene-level GO 상속이면 isoform-level ground truth가 아니다." |
| **보완 필요** | 문헌 확립된 isoform-specific function 케이스 제시 필수 (예: RYR1 vs RYR2, SERCA1 vs SERCA2, MYH isoforms). GO annotation이 실제 isoform-specific function을 반영하는 사례를 생물학적으로 정당화해야 함. |

---

### Claim 2: "Gene-level shortcut 식별 및 해결 [R2.1]"

**판정: Main ✅ — `no_cross_gene_negative` ablation 없으면 Supporting으로 격하**

| | 내용 |
|--|------|
| **강점** | 원본 DIFFUSE CRF가 within-gene isoform score를 평균화하여 판별 신호 소멸시킴을 실험적으로 확인. Cross-gene negative [R2.1]이 gene-level shortcut을 구조적으로 차단. Isoform 예측 맥락에서 이 문제를 명시적으로 정의한 선행 논문 없음. |
| **예상 공격** | "IRM/DRO 문헌(spurious correlation)과 구조적으로 동일 문제. Domain generalization의 특수 케이스 아닌가?" |
| **차별점** | IRM은 환경(domain) 변화에 대한 불변성 문제. 이 연구는 동일 gene 내 isoform 간 판별이라는 생물학적 구조 특화 문제. 프레이밍이 다르지만 IRM 비교 실험이 없으면 심사자가 일반화할 위험 있음. |
| **즉시 필요** | ablation: `no_cross_gene_negative` (cross-gene 제약 제거 시 AUPRC 하락 수치) |

---

### Claim 3: "4단계 커리큘럼의 인과 구조"

**판정: Supporting — Main claim으로 내세우면 공격받음**

| | 내용 |
|--|------|
| **강점** | GO:0003774 Ep7 phase transition이 Phase 1→2 순서의 인과성을 지지하는 실증 데이터. Phase별 Silhouette trajectory로 임베딩 안정화 과정 추적 가능. |
| **예상 공격** | "Curriculum learning (Bengio et al., 2009)의 특수 케이스. 순서의 인과성은 이 데이터에서만 성립할 수 있다. ablation 없이는 사후 합리화." |
| **권장 위치** | "우리 프레임워크를 효과적으로 학습하기 위한 훈련 전략"으로 Methods에 기술. ablation으로 각 phase 기여를 수치화하면 Supporting contribution으로 방어 가능. |

---

### Claim 4: "Test-time expression label propagation [I2]"

**판정: Methods 기술 수준 — Contribution 주장 제외 권장**

| | 내용 |
|--|------|
| **강점** | GO:0030017 AUPRC +0.018 실증. Long-read co-expression을 inductive 추론에 활용하는 아이디어. |
| **약점** | Zhu & Ghahramani 2002의 고전 방법 적용. GO:0003774에서 역효과 (단조 감소). GO:0007204에서 개선 미미 (+0.000). 일관된 효과 없음. |
| **권장** | "선택적 후처리 모듈"로 위치. 특정 GO term 유형(tissue-specific, co-expression 신호 강한 경우)에서만 권장하도록 조건 명시. |

---

## 2. 기여도 재구조화 (권장안)

```
Main Contribution:
  [C1] 문제 정의: Long-read RNA-seq 기반 novel isoform function prediction
       - 기존 툴의 공백 명확히 (DeepGO, CAPE, 원본 DIFFUSE 비교)
       - BambuTx → TransDecoder → ESM-2 전체 파이프라인

  [C2] Isoform-specific representation learning:
       - Gene-level shortcut 문제 formal definition
       - Cross-gene negative triplet [R2.1]이 핵심 solution
       - 필수 ablation: no_cross_gene_negative AUPRC 비교

Supporting Contribution:
  [C3] ESM-2 기반 isoform 임베딩이 sparse GO term에서 효과적
       (margin_sat 개선: v5-5 5.9% → v6 목표 30%+)
  [C4] Expression-guided test-time refinement
       (효과 있는 GO term 유형 한정 — tissue-specific BP/CC)

Engineering (Methods 기술):
  - 4단계 학습 파이프라인 (ablation으로 보완)
  - hstack upsample synchronized 처리
  - Coverage 기반 동적 n_batches [I4]
```

---

## 3. 치명적 미비 사항 (우선순위순)

| 우선순위 | 미비 사항 | 심각도 | 필요 실험 |
|---------|---------|--------|---------|
| 1 | `no_cross_gene_negative` ablation 없음 | **Critical** | Claim C2 전체 무너짐 |
| 2 | Isoform-level ground truth 정당화 없음 | **Critical** | 평가 타당성 공격 |
| 3 | 기존 툴 직접 비교 없음 (DeepGO, IsoFun) | **High** | "왜 이 방법이 필요한가" |
| 4 | Bootstrap CI 미수행 [R9.4] | **High** | 개선 주장 통계 근거 없음 |
| 5 | `no_phase1`, `no_phase1_5` ablation 없음 | Medium | Curriculum 방어 |
| 6 | LabelProp 역효과 GO term 설명 없음 | Medium | GO:0003774 하락 해명 |
| 7 | IRM/SupCon 비교 없음 | Medium | Gene-shortcut solution 독자성 |

---

## 4. 종합 판정

| Claim | 판정 | 생존 조건 |
|-------|------|---------|
| Novel isoform function prediction | **Main ✅** | Isoform-specific ground truth 케이스 제시 |
| Gene-level shortcut 해결 | **Main ✅** | `no_cross_gene_negative` ablation 수치 |
| 4단계 커리큘럼 | **Supporting** | `no_phase1`, `no_phase1_5` ablation |
| ESM-2 통합 | **Supporting** | margin_sat 개선 수치 (v6 결과 후) |
| LabelProp | **Methods** | Contribution 주장 제외 권장 |

**결론**: Main contribution 2개는 방어 가능한 구조. 단, `no_cross_gene_negative` ablation이 없으면 Reviewer 1에서 reject 확정. v6 실험 완료 직후 ablation 설계 착수 필요.
