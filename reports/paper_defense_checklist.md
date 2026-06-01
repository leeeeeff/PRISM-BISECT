# PRISM 논문 방어 체크리스트

**생성일**: 2026-06-01  
**기반**: 3-agent 병렬 분석 (A: 정량화, B: 케이스 분석, C: 비판적 재검토) + 논문 draft 실제 검토

---

## 핵심 발견: 주장 재정의 필요

> **기존 프레이밍**: "InterProScan이 못 찾는 것을 PRISM이 찾는다"
>
> **실제 논문 내용** (results_draft 검토 후):  
> - KIF21B: tr293004 (kinesin motor domain 있음) vs tr292978 (없음) → **InterProScan도 이 차이를 탐지함**  
> - PRISM 기여: "domain annotation 없이 sequence embedding만으로 동일한 결론 도달" = 독립적 교차검증  
> - 진짜 "InterProScan 너머" 사례: Pfam annotation 없는 50% 이소폼, LINE-1 유래 tr73243
>
> **수정된 주장**: "PRISM은 도메인 데이터베이스 없이 서열만으로 기능을 예측하며, 도메인 분석과의 수렴이 교차검증을 제공한다."

---

## S 우선순위 — 논문 심사 방어 필수

### S1: GO 레이블 Evidence Code 분포 분석
- **목표**: IEA(전자 주석, 종종 InterPro 기반) vs EXP(실험적) 비율 측정
- **파일**: `hMuscle/data/raw_data/data/annotations/swissprot_annotations.txt`
- **현황**: 파일 형식 확인됨 (protein_id + GO terms). Evidence code 컬럼 별도 파일 필요
- **위험**: IEA 비율 높으면 순환 논리 반론 나옴
- **담당**: 스크립트 작성 후 실행
- [ ] evidence code 포함 GAF 파일 탐색 또는 UniProt API 조회
- [ ] IEA vs EXP 비율 측정 및 보고
- [ ] IEA 제외 후 성능 변화 테스트

### S2: "Pfam annotation 없는 이소폼" 전용 PRISM 성능 분석
- **목표**: 50% annotation-free 이소폼에서 PRISM이 실제로 맞는지 정량화
- **이것이 진짜 "InterProScan 너머" 주장의 핵심**
- **파일**: `hMuscle/results_isoform/features/pfam_to_int_mapping.json` + PRISM predictions
- [ ] annotation-free 이소폼 추출
- [ ] 해당 이소폼에서만 PRISM AUPRC 계산
- [ ] annotation-present vs annotation-free 성능 비교

### S3: 주장 표현 수정 (논문 전체)
- **목표**: "InterProScan이 못 찾는다" → "도메인 annotation 없이도" 로 수정
- [x] `reports/manuscript_full_english.md` — **이미 올바른 표현 사용** ("convergence", "independently supported", "cross-validation", "without access to domain annotations")
- [x] `reports/discussion_draft_20260516.md` — 동일. "InterProScan" 단어 자체가 논문에 없음
- [x] `reports/introduction_draft_20260516.md` — 수정 불필요
- **결론**: S3 완료 (수정 불필요). 논문 draft는 이미 방어 가능한 표현 사용 중.

---

## A 우선순위 — 주장 강화

### A1: KIF21B 프레이밍 명확화
- **현상**: draft에 이미 Pfam-A hmmscan 결과 포함 (tr293004: motor domain, tr292978: WD40만)
- **결론**: 논문 이미 올바른 표현 사용 중 (lines 697-701 results_draft)
  - "PRISM v15d_bp_clean functional scores **independently recapitulate** this domain-level distinction"
  - "solely from ESM-2 sequence embeddings, without access to domain annotations"
  - "the **convergence** between the sequence-derived functional prediction and the experimentally derived domain architecture provides **cross-validation**"
- [x] 표현 확인 완료 — 추가 수정 불필요

### A2: tr73243 (LINE-1 유래 NAT) — 진짜 "domain presence ≠ functional consequence" 케이스
- **발견**: PRISM Mito.org score: canonical NDUFS4 0.587 → tr73243 **0.024** (Δ = −0.563)
- **핵심 논점**:
  - InterProScan: tr73243에서 **RVT_1 domain 발견** (올바름, Pfam E=4.6×10⁻⁴⁸)
  - InterProScan의 한계: "RVT_1 있음" → Complex I 기능 손실 예측 불가
  - PRISM: Complex I assembly GO:0007005에서 0.024 → 기능 손실 올바르게 예측
  - 이유: MTS 부재 + LYR motif 부재 + 98.3% sequence divergence를 서열 맥락으로 통합
- **이것이 올바른 주장**: "domain presence를 알아도 functional consequence는 서열 맥락이 필요"
- [x] PRISM 예측값 확인 (results_draft line 524, 750)
- [ ] Discussion에 이 논점을 명시적으로 추가 ("domain presence vs. functional consequence" 구분)

### A3: Domain-matched pair 분석 (Agent A 제안)
- 같은 Pfam 구성을 가진 이소폼 쌍에서 PRISM 예측 차이 분포
- 이것이 "domain 이상의 정보를 포착한다"는 직접 증거
- [ ] `reports/interproscan_prism_comparison.py` 기반으로 domain-matched pair 서브셋 분석

---

## B 우선순위 — 통계 강화

### B1: Bootstrap CI 전 GO term 적용 확인
- [ ] 현재 결과에서 95% CI 제시 여부 확인
- [ ] 없으면 bootstrap_ci 스크립트 실행

### B2: Gene-level overfitting 검증
- [ ] Train gene / Test gene 완전 분리 실험
- [ ] 같은 유전자 이소폼들의 예측값 분산 측정

### B3: InterProScan 실제 실행 확인
- [ ] `hMuscle/results_isoform/features/pfam_db/Pfam-A.hmm` 존재 확인 (methods draft에 언급)
- [ ] 주요 케이스 (KIF21B, NDUFS4, DLG1) hmmscan 결과 summary table 생성

---

## C 우선순위 — 논문 완성도

### C1: Evidence Code 결과를 Methods에 추가
- [ ] GO annotation 출처 및 evidence code 분포를 Methods에 명시

### C2: Supplementary Table — Domain vs No-Domain 성능
- [ ] Pfam annotation 있는 그룹 vs 없는 그룹 성능 분리 표

### C3: Figure 업데이트
- [ ] KIF21B Figure에 "PRISM score와 domain architecture 수렴" 명시

---

## 파일 목록

| 파일 | 상태 |
|------|------|
| `reports/interproscan_prism_comparison_result.md` | ✅ Agent A 완료 |
| `reports/case_analysis/ndufs4_kif21b_interproscan_prism_comparison.md` | ✅ Agent B 완료 |
| `reports/challenge_interproscan_claim.md` | ✅ Agent C 완료 |
| `reports/paper_defense_checklist.md` | ✅ 이 파일 |
| `reports/go_label_evidence_analysis.md` | ✅ S1 완료 (Agent S1) |
| `reports/annotation_free_analysis_result.md` | ✅ S2 완료 (v5fix 기반, v15d 재실험 필요) |
| S3: 논문 표현 수정 | ✅ 수정 불필요 (이미 올바른 표현) |
| A2: Discussion에 "domain presence vs. functional consequence" 추가 | ⬜ 미완 |
| B1: v15d로 annotation-free 성능 재계산 | ⬜ 미완 (권장) |
| `reports/interpro2go_vs_prism_experiment_v2.md` | ✅ v1 GO term 오류 수정 완료 (2026-06-01) |
| `reports/gene_label_noise_analysis.md` | ✅ Agent A — gene memorization 기각 완료 |
| `reports/prism_vs_gene_memorization_analysis.md` | ✅ Agent C — variance 분석, memorization 기각 완료 |

## 긴급 수정: v1 interpro2go 보고서 GO term 오류

**발견일**: 2026-06-01 (Agent A/C 병렬 분석 결과)

v1 보고서(`interpro2go_vs_prism_experiment.md`)의 모든 GO term 레이블이 틀렸음:
- GO:0006936, GO:0006412 → v15d_bp_clean에 없는 GO IDs
- 실제: GO:0006941 (Muscle contraction), GO:0007268 (Synaptic transmission), GO:0007018 (MT movement)

**수정된 결과**: Type I = 2/26 (KIF21B, CCAR1), Type II = 24/26  
**수정 파일**: `reports/interpro2go_vs_prism_experiment_v2.md` ✅

**Gene-level memorization 기각 증거**:
- Within-gene variance (0.00126) > Between-gene variance (0.00070)
- DLG1 novel 186aa isoform: synaptic transmission 0.033 vs canonical 0.88 (29배 차이)
- IFT122 isoforms: MT movement 0.0001~0.83 (4,100배 차이)

---

*Generated 2026-06-01 | PRISM paper defense analysis*
