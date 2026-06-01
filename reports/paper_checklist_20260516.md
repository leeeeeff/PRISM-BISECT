# Paper Completion Checklist — DIFFUSE v10-B
**생성: 2026-05-16 | 근거: paper-critic 17개 지적사항 + 기존 TODO**

---

## 🔴 FATAL 방어 실험 (Nature Methods 통과를 위한 필수)

- [x] **F1. gene-mean baseline 계산** ✓ 2026-05-17
  - v10-B per-isoform scores → gene 평균으로 대체 → pos_bias 비교
  - 추가: LR pos_bias, random predictor pos_bias, shuffled-label v10-B pos_bias
  - 목표: gene-mean = 0 < random ≈ 1.0 < v10-B > 1.0 구조 확인
  - 스크립트: `/tmp/v10_posbias_controls.py` (PYTHONUNBUFFERED=1, flush=True)
  - 결과 예정: `reports/gene_mean_baseline/posbias_controls_{ts}.json`

- [x] **F2. pos_bias negative controls (shuffled-label)** ✓ F1에 포함 완료
  - GO:0006941: shuf=0.245, GO:0007005: shuf=0.282, GO:0006914: shuf=0.193
  - 모든 v10-B 값이 shuffled floor (≈0.24) 초과 확인

---

## 🟠 MAJOR 수정 (리뷰어 필수 요구사항)

- [x] **M1. sep_cosine 주장 격하** ✓ (results_draft_20260516.md Section 3.2 수정)
  - "prospective classifier" → "post-hoc characterisation"
  - "external validation required" 문구 추가

- [x] **M2. NMD 스크리닝 top 이소폼 대칭 적용** ✓ (2026-05-17 완료)
  - 결과: 23/126 제외 (18.3%) — 원래 14/126 (11.1%)에서 9 추가
  - 스크립트: `/tmp/nmd_screening_symmetric.py`
  - 결과: `reports/nmd_screening_symmetric_20260516.json`
  - Supp Table S2 업데이트: `reports/supp_table_s2_nmd_screening.md` ✓
  - Methods 4.10, Results 3.4 텍스트 반영 ✓

- [x] **M3. Introduction 재작성: 선행 방법 추가** ✓ (introduction_draft_20260516.md Para 4)
  - IsoformSwitchAnalyzeR, SQANTI, APPRIS 인용 추가
  - novelty 재프레이밍: "첫 unified DL framework for isoform-level GO prediction"

- [x] **M4. Discussion 5.3 어조 수정** ✓ (discussion_draft_20260516.md Section 5.3)
  - "computational predictions requiring experimental validation" 명시
  - DMD/PINK1 = "recapitulates known biology" vs NDUFAF6 = "novel prediction" 분리

- [x] **M5. gene-mean baseline을 Methods 4.9에 추가** ✓ 2026-05-17
  - 3가지 negative control 상세 추가 (gene-mean=0, random≈0.9, shuffled≈0.24)
  - Results 3.3 및 Discussion 5.1 대폭 업데이트

- [x] **M6. Ablation table 생성** ✓ 2026-05-17
  - ablation_results_20260517_1537.json → Supp Table S4 완전 업데이트
  - 핵심 결과: no_focal = Type-B macro Δ = −0.070 (가장 중요한 component)
  - Methods 4.11 실제 수치로 업데이트 완료

---

## 🟡 MINOR 수정 (리뷰어 요구 가능성 있음)

- [x] **Mi1. TOR signaling 포함/제외 macro 모두 보고** ✓ (이미 완료)
  - Table 1: Type-B macro (11 terms) = 0.685, Type-B macro (excl. TOR, 10 terms) = 0.693
  - Table 1 footnote: "0.685 (with TOR) vs 0.693 (without TOR), Δ = 0.008" 포함됨

- [x] **Mi2. ESM-2 임베딩 세부 정보 추가** ✓ 2026-05-17
  - Methods 4.2: facebook/esm2_t30_150M_UR50D, layer 30, mean pooling over sequence length 추가 (**모델명 오류 수정: t33_650M→t30_150M**)

- [x] **Mi3. Seed stability full table** ✓ (supp_table_s3_seed_stability.md 생성)
  - 5 seeds × 13 GO terms AUPRC 값 포함
  - Type-B macro CV = 0.9% (안정적)

- [x] **Mi4. Bootstrap CI 방법 명확화** ✓ (methods_draft_20260516.md Section 4.6)
  - "training set held fixed; test-set genes resampled with replacement (n=500)"

- [x] **Mi5. pos_bias p-value 추가** ✓ 2026-05-17
  - posbias_pvalues_20260517_1548.json 생성 완료
  - 결과: 11/13 terms q < 0.05 vs. shuffled floor; Ca2+ signaling (q=0.307), Glycolysis (q=0.189) n.s.
  - Table 2 완전 업데이트: 95% CI + q-value 컬럼 추가
  - Methods 4.12 실제 수치 업데이트 완료

- [x] **Mi6. NIPSNAP1/TAFAZZIN isoform 상세 보고** ✓ 2026-05-17
  - NIPSNAP1: 1 isoform (ENST00000216121.12), score=0.819, GO:0007600/0019233/0050877 있음
  - TAFAZZIN: 4 isoforms, top=ENST00000601016.6, score=0.934 (GO:0007005) / 0.990 (GO:0006941)
  - Results 3.4 업데이트 완료 (isoform count, score, Barth syndrome 문헌 추가)

- [x] **Mi7. Data availability statement** ✓ 2026-05-17 (draft 완료)
  - methods_draft_20260516.md에 "Data availability" 섹션 추가
  - Zenodo DOI + GitHub URL 제출 전 확보 필요 (TODO 표시)

- [x] **Mi8. Multiple testing correction 완성** ✓ (Mi5에 포함)
  - pos_bias BH correction: posbias_pvalues_20260517_1548.json (q_vs_shuffled_BH, q_vs_random_BH)
  - Pearson r permutation test: DEFER (r=−0.72 correlation는 충분히 보고됨; permutation test는 minor)

---

## ✍️ 논문 텍스트 작업

- [x] Results draft 완료 (results_draft_20260516.md)
- [x] Methods draft 완료 (methods_draft_20260516.md)
- [x] Introduction draft 완료 (introduction_draft_20260516.md)
- [x] Discussion draft 완료 (discussion_draft_20260516.md)
- [x] Supplementary Table S2 완료 + symmetric 업데이트 (supp_table_s2_nmd_screening.md) ✓ 2026-05-17
- [x] **Introduction 재작성** ✓ M3 완료 (IsoformSwitchAnalyzeR/SQANTI/APPRIS 인용)
- [x] **Discussion 5.3 수정** ✓ M4 완료 (computational predictions 명시)
- [x] **Discussion 5.1 재작성** ✓ neg controls 반영 (2026-05-17)
- [x] **Discussion 5.2 업데이트** ✓ sep_cosine "post-hoc" 수정 (2026-05-17)
- [x] **Abstract 최종 검토** ✓ (2026-05-17: GABARAPL1→DMD, BNIP3 제거, sep_cosine 표현 수정)
- [x] **Citation list 1차 완성** ✓ 2026-05-17
  - Introduction PMID 오류 3건 수정: Cruz-Jentoft(*Lancet*→*Age Ageing* 30312372), APPRIS(30239896→34755864), IsoformSwitchAnalyzeR(30988129→30989184)
  - Introduction 누락 인용 9건 citation list에 추가 (A1-A9: Lexell, Hepple&Rice, Masiero, Sousa-Victor, Sandri, Pan, Baralle, Guo2021, Hao2024)
  - CLEAN, SpliceAI, DIFFUSE-tool, Yao2022 추가
  - **⚠️ 제출 전 PubMed 확인 필요 (7건)**: DeepFRI PMID 불일치(34039969 vs 34210978), Beaudart저널, Hepple&Rice 내용, Sandri저널, Guo2021 channelopathy, Hao2024, CLEAN/Yao2022

---

## 📊 Figure 작업

- [x] Fig 1: Architecture (fig1_architecture.pdf)
- [x] Fig 2: Isoform switch, DMD Panel A, NDUFAF6 Panel C (fig2_isoform_switch.pdf)
- [x] Fig 3: sep_cosine 2-panel (fig3_sepcosine_2panel.pdf)
- [x] Supp Fig 1: LOOCV threshold (suppfig_loocv_threshold.pdf)
- [x] Supp Fig 2: Seed stability (suppfig_seed_stability.pdf)
- [x] **Fig 1 polish**: Panel A title overlap 수정 ✓ 2026-05-17
  - "A" label 위치 0.065→0.025 (Isoforms label과 분리)
  - 650M→150M params, pos_bias>1.0→shuffled≈0.24, 5-fold CV→80/20 split
  - Macro pos_bias=1.130→1.064, bar_data 실제 값으로 업데이트
- [x] **Fig 2 Panel B**: x-axis label 가독성 ✓ 2026-05-17
  - xticklabels fontsize=9.5→8.5, rotation=10 추가
  - 이소폼 라벨 y=-0.12→-0.07, ylim 바닥 -0.35→-0.42
- [x] **Supp Fig 3**: 전체 isoform-switch 후보 78개 시각화 ✓ 2026-05-17
  - suppfig3_isoform_candidates.pdf (bubble chart + ranked lollipop)
  - /tmp/suppfig3_isoform_candidates.py 생성

---

## 🔬 실험 작업

- [x] v10-B 13 GO term 평가 완료
- [x] NMD 스크리닝 완료 (nmd_screening_20260516.json)
- [x] ESM-RF baseline 완료 (rf_results_20260516_1439.json)
- [x] **F1: gene-mean baseline** ✓ posbias_controls_20260517_1433.json
- [x] **F2: shuffled-label control** ✓ F1에 포함 완료
- [x] **M2: 대칭 NMD 스크리닝** (top+bot) ✓ 23/126 제외
- [x] **M6: ablation 실험** ✓ 2026-05-17 완료
  - ablation_results_20260517_1537.json 생성 완료
  - Supp Table S4 실제 수치로 채움 (S4A AUPRC + S4B Δ + Interpretation)
  - Methods 4.11 구체적 수치 업데이트 완료

---

## 진행 상황 요약

| 카테고리 | 완료 | 미완료 | 진행률 |
|----------|------|--------|--------|
| FATAL 방어 | 2 | 0 | **100%** ✅ |
| MAJOR 수정 | 6 | 0 | **100%** ✅ |
| MINOR 수정 | 8 | 0 | **100%** ✅ |
| 논문 텍스트 | 10 | 0 | **100%** ✅ |
| Figure | 8 | 0 | **100%** ✅ |
| 실험 | 6 | 1 | **86%** |
| **전체** | **42** | **1** | **98%** |

**완료한 것 (2026-05-17 전체 세션):**
- M6 ✅, Mi1–Mi8 ✅, Fig 1 polish ✅, Fig 2 Panel B ✅, Supp Fig 3 ✅
- Supp Table S1 ✅, Supp Table S4 ✅, Supp Note 1 ✅
- Citation list 확장 (Introduction 누락 9건 + Discussion 5건 추가, PMID 오류 3건 수정)
- Abstract 오류 수정: SwissProt 언급 제거 → "ESM-2 embeddings"로 수정
- Discussion 5.1 pos_bias 수치 5-seed ensemble 기준으로 일관화
- pos_bias 5-seed(Table 2) vs seed=42(bootstrap CI) 불일치 문서화 (Table 2 footnote)

**남은 작업 (제출 직전 필수, 현재 환경에서 불가):**
1. **Bambu 버전** — 데이터 생성 환경에서 `packageVersion("bambu")` 실행
2. **PubMed 확인 7건** — DeepFRI PMID(34039969 vs 34210978), Beaudart저널, Hepple&Rice내용, Sandri저널, Guo2021/Hao2024/CLEAN 제목 확인
3. **Zenodo DOI + GitHub URL** — 제출 직전 확보
4. **Abstract version 선택** — V1/V2/V3 중 target journal에 따라 결정
