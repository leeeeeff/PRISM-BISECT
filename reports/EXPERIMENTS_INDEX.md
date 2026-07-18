# reports/ 실험 디렉토리 인덱스

> **이 문서는 내비게이션 전용이다 — 어떤 파일도 이동하지 않았다.**
> `reports/` 최상위의 ~165개 실험 디렉토리는 `natcomm_v0.md` 본문(Methods 인용 5곳), `hMuscle/model/*.py` 스크립트 212개, memory 노트 66곳에서
> 정확한 상대경로로 참조되고 있어 물리적 이동이 위험(2026-07-19 조사 결과, [[session_20260719_repo_reorg]]).
> 대신 이 인덱스로 "무엇이 어디 있는지"를 문서화한다. 느슨한 파일 재배치는 이미 완료됨 — `manuscript_archive/`, `manuscript_drafts/`,
> `meetings/`, `analysis_brain3d/`, `analysis_celltype_layer/`, `analysis_domain_esm2/` 참조.
>
> 표에 없는 새 실험 디렉토리가 생기면 이 문서에 한 줄 추가(디렉토리 생성과 같은 세션에).

---

## 1. 모델 버전 히스토리 (v10 ~ v20b)

PRISM 아키텍처의 순차적 버전 실험. `hMuscle/model/v*.py` 스크립트의 실행 산출물.

| 디렉토리 | 내용 |
|---|---|
| `v10_isoform_delta/`, `v10_mlp/`, `v10_splice_real/` | v10 계열 초기 아이소폼-델타 특징 실험 |
| `v11_attention/`, `v11_canonical/`, `v11_deviation/`, `v11_slim/` | v11 attention/정규화 변형 시리즈 |
| `v12_sequential/` | v12 순차 인코딩 실험 |
| `v13_mil/`, `v13_pcgrad/` | v13 multiple-instance-learning + PCGrad 그래디언트 수술 |
| `v14_domain_aux/` | v14 도메인 보조손실 실험 |
| `v15_bp_clean/`, `v15_bp_clean_noIEA/`, `v15_switch_dtu/`, `v15_unified/`, `v15d_brain_eval/`, `v15d_splice/` | **v15d_bp_clean = 초기 production 계열**(§2 18-BP 모델). noIEA=IEA 라벨 제외 ablation, brain_eval=muscle→brain 최초 zero-shot, splice=FAIL(AP15) |
| `v16/`, `v16b/`, `v16c/`, `v16_brain/`, `v16_brain_switch/` | v16 4-Stream Gated Residual Fusion — **전부 FAIL**(gate collapse, F71) |
| `v17_brain_model/`, `v17_two_stage/`, `v17b_two_stage/`, `v17c_two_stage/` | v17 초기 Two-Stage 계열 — Two-Stage T FAIL(triplet collapse) |
| `v17d_bootstrap/`, `v17d_delta/`, `v17d_muscle/` | v17d δ_layer(L30−L15) 도입, muscle 단독 ablation |
| `v17e_ablation/` | v17e ablation 시리즈 |
| `v17f_abl_no_tpsi/`, `v17f_abl_tpsi_256/`, `v17f_abl_tpsi_640/` | **v17f\* = T_ψ 제거 ablation, 현재 BEST 모델**(0.734 muscle / 0.647 true-brain) |
| `v17f_b2_bootstrap/`, `v17f_bootstrap/`, `v17f_star_bootstrap/` | v17f/v17f* bootstrap CI 계산 |
| `v17f_bp/`, `v17f_bp_cc_eval/` | v17f BP/CC 카테고리 확장 평가(279-GO) |
| `v17f_capacity_baselines/` | δ_layer capacity(폭/깊이) 대조군 — [[finding-v17f-head-capacity-gap]] |
| `v17f_gene_mean_baseline/` | gene-mean oracle 계산(macro AUPRC 0.803/0.795) |
| `v17f_l3_recovery/`, `v17f_l4_cellstate/` | L3/L4 층 신호 복구 실험 |
| `v17f_layer_breakdown/`, `v17f_layer_delta/`, `v17f_layer_scan/`, `v17f_layer_search/` | per-layer Fisher/attribution 스캔 — F3/F4 산출물 |
| `v17f_reclassify/` | GO term 카테고리 재분류(BP/CC 재평가) |
| `v17f_selective_weight/` | selective_weight 가중치 방식(v17f* 최종안, DR 0.5245) |
| `v17f_splice_diagnostic/` | splice-feature-only 대조 모델 |
| `v17g_hybrid/` | v17g hybrid delta 실험 |
| `v18a_rna_loc/`, `v18b_celltype/`, `v18_brain_prototype/` | v18 RNA+LOC/cell-type 멀티모달 — **전부 FAIL**(L4 경계, 세션 20260628) |
| `v19_curve/`, `v19_swiglu/` | v19 curve-sweep/SwiGLU 활성화 실험 |
| `v20b/`, `v20b_brain/`, `v20b_domain_evidence/`, `v20b_pca_interp/` | v20b PCA 8축 해석 + domain-evidence null(기각) — [[approach-v20b-domain-evidence-null]] |
| `v20_cache/`, `v20_curve_gospecific/`, `v20_swiglu_nodelta/` | v20 계열 curve/cache 실험 |
| `v_expanded_gomf/` | 279-GO(BP+MF+CC) 확장 훈련 — **PRISM zero-shot 행 삭제 사유가 된 스크립트**([[session_20260719_comparative_analysis_truebrain]]) |

## 2. exp_* 명명 실험 시리즈 (알파벳 코드)

| 디렉토리 | 내용 |
|---|---|
| `exp_a_cond_domain_aux/`, `exp_a_scale/` | Exp A: PRISM vs ESM-2 조건부 도메인 보조손실 |
| `exp_B1_within_gene_ranking/`, `exp_B2_window_ablation/`, `exp_B3_bootstrap_ci/`, `exp_B3b_paired_bootstrap/`, `exp_b_probing/` | Exp B: within-gene 랭킹 window ablation + bootstrap |
| `exp_C1_layer_probe_279/`, `exp_c_decomposition/`, `exp_c_dr_auc_fix/` | Exp C: 279-GO 층별 probe + Domain-Ranking AUC 버그 수정 |
| `exp_d_finetune/` | Exp D: ESM-2 fine-tuning 비교(D0/D1/D2) — [[session_20260628_exp_abc]] |
| `exp_e_sota/` | Exp E: SOTA 비교(DeepFRI/DeepGoPlus/k-NN/gene-mean) — 이번 세션에서 true-brain 재통합 완료 |
| `exp_f_plm_scale/` | Exp F: PLM 스케일(8M~650M+ProtT5+Ankh) δ_layer 일반화 — [[finding-cross-plm-generalization]] |
| `exp_go_prototype/` | GO centroid-prototype 방식 프로토타입 |
| `exp_g_uniprot/`, `exp_h_uniprot_eval/` | UniProt 51쌍 isoform-pair 벤치마크(v1→v2) — natcomm_v0.md 직접 인용, **이동 금지** |
| `exp_isoform_discrimination/` | 아이소폼 판별력 3-metric(CV/length-AUC/top1-agreement) — muscle 버전. true-brain 버전은 `truebrain_rerun_20260714/exp_isoform_discrimination_truebrain/` |

## 3. True-brain 재실행 & SOTA 벤치마크

| 디렉토리 | 내용 |
|---|---|
| `truebrain_rerun_20260714/` | **2026-07-14 true-brain(63,994-isoform) 전면 재실행 세션의 모든 산출물** — v17f*, BLAST→GOA, isopretEM, DeepGoPlus, Domain LR 등 하위 폴더 다수. natcomm_v0.md §Comparative analysis가 직접 의존 |
| `sota_final_benchmark/` | k-NN 등 SOTA 벤치마크 "v6" 권위 스크립트 산출물(자체 muscle 버전) |
| `domain_ranking_validation/` | Domain-Ranking AUC 계산(0.630 muscle) — natcomm_v0.md 직접 인용 |
| `novel_go_matched_null/` | Table 1 matched-null 통계(9-14× enriched) — natcomm_v0.md 직접 인용 |
| `novel_go_acquisition_20260714/`, `novel_go_validation/` | non-domain GO acquisition 사례 발굴/검증 |
| `benchmark_external/` | DeepFRI/isopretEM 등 외부 툴 설치·실행 스크립트 위치(`isopret/`) |
| `alphafold_validation/` | AlphaFold pLDDT 구조 신뢰도 교차검증(phase4) |
| `feature_attribution/` | ESM-2 probe AUROC, β_domain 등 feature attribution 실험 |
| `xgb_baseline/`, `gene_mean_baseline/`, `domain_baseline/` | 비교용 baseline 모델(XGBoost/gene-mean/domain) |

## 4. 뇌 조직 / 세포유형 / 층(layer) 분석

| 디렉토리 | 내용 |
|---|---|
| `brain_alzheimer/`, `brain_fiu/`, `brain_flow/` | AD 뇌 코호트 분석(brain_flow=trajectory-flow 서사 탐색, 세션 20260708) |
| `brain_meshes/` | 3D 뇌 시각화용 해부학적 메시(glb 등) — `analysis_brain3d/`의 스크립트가 참조 가능성, 이동 전 확인 필요 |
| `c18_barcodes/` | C18(L4) 클러스터 세포 바코드 전량 추출(110개 파일) |
| `c18_deep_dive/` | C18 아이소폼 비율 심층 분석 |
| `celltype_composition/` | 세포유형 구성비 분석 |
| `cohort_batch_check/` | Samsung/SRA 코호트 배치 효과 점검 |
| `inhibitory_subtypes/`, `all_subtypes/` | 억제성/전체 세포 subtype 분류 |
| `layer_annotation/`, `layer_probe/` | 피질층(layer) 주석 + probe 분석 |
| `allen_kif21b/` | Allen Brain Atlas 대조 KIF21B 층별 발현 |
| `bisect_celltype/` | BISECT 결과의 세포유형별 breakdown |
| `case_analysis/` | 개별 유전자 케이스 분석 모음 |
| `samsung_dtu/` | Samsung 코호트 DTU(differential transcript usage) 결과 |
| `lamp5_kit_validation/` | LAMP5/KIT AD-enriched 검증 산출물(스크립트는 `analysis_celltype_layer/lamp5_kit_validation.py`로 이동) |
| `rbfox1_layer/` | RBFOX1 층별 발현 산출물(스크립트는 `analysis_celltype_layer/rbfox1_layer_expression.py`로 이동) |
| `prss12_pathway/` | PRSS12 경로 분석 산출물(스크립트는 `analysis_celltype_layer/prss12_pathway_analysis.py`로 이동) |
| `ebbert_replication/` | Ebbert 코호트 NDUFS4 복제 검증 산출물(스크립트는 `scripts/adhoc_analysis/ebbert_replication.py`로 이동) |

## 5. BISECT / 아이소폼 라벨·해상도

| 디렉토리 | 내용 |
|---|---|
| `bisect_isoform_labels/`, `bisect_prism_scores/` | BISECT 케이스별 라벨·PRISM 점수 저장 |
| `isoform_resolution/`, `isoform_resolution_full/` | 아이소폼 해상도 경계 종합 분석 — [[approach-isoform-resolution-boundary]] |
| `within_gene/`, `within_gene_layer_divergence/`, `within_gene_metrics_all_domains/` | within-gene 발산/layer-divergence/전 도메인 지표 |
| `pos_bias_coding/`, `posbiascontrol/` | pos_bias(within-gene 판별력) 계산 + 대조군 |
| `true_motif_level/` | motif-level 기능방향 판별 실험(FAIL, 0.448) |
| `typeAB_classifier/` | Type A/B(motif-dep vs domain-dep) 분류기 |
| `label_propagation_novel/` | novel GO term label propagation 실험 |
| `stage0_esmfold_audit/` | ESMFold pLDDT Stage0 게이트 감사 — [[finding-stage0-esmfold-pldct]] |
| `reranking/` | 예측 재순위화(simple reranking λ=0.02) 실험 |
| `tbs_tcs_13terms/` | TBS/TCS 13-term 세부 GO 분석 |

## 6. 근육/희소성(sarcopenia) 및 기타 진단

| 디렉토리 | 내용 |
|---|---|
| `sarcopenia_eval/`, `sarcopenia_novel/` | 근감소증 관련 평가·novel 발굴(sarcopenia_novel은 빈 디렉토리) |
| `muscle_labelgap/`, `logs_labelgap/` | muscle 라벨 격차 진단 로그 |
| `fluid_stage1/`, `fluid_stage2/`, `fluid_stage3/` | fluid(체액) 단계별 실험 3종 |
| `phase5_novel/` | Phase5 novel 실험(빈 디렉토리) |
| `curve_sweep/` | 학습곡선/하이퍼파라미터 sweep(48개 파일) |
| `consensus_idr/` | consensus IDR(intrinsically disordered region) 분석(빈 디렉토리) |
| `bootstrap_ci/` | 범용 bootstrap CI 계산 스크립트/로그(빈 디렉토리 — 산출물은 각 실험 폴더 내부에 개별 저장) |
| `diagnostics/` | 범용 진단 스크립트 모음 |
| `braak_isoform_plots/` | Braak stage 상관 플롯 |
| `go_acquisition_type3/` | GO acquisition Type 3(mixed) 분류 산출물 |
| `ablation/` | 범용 ablation 결과 모음 |

## 7. Figures

| 디렉토리 | 내용 |
|---|---|
| `figures/` | 초기 figure 산출물 |
| `figures_20260519/` | 2026-05-19 시점 figure 세트 |
| `figures_v2/` | NatComm v2 figure 세트(F1~F8, Okabe-Ito 팔레트) — 세션 20260709 |

## 8. 백업 및 기타

| 디렉토리 | 내용 |
|---|---|
| `_backup_20260709/` | 2026-07-09 시점 전체 백업 스냅샷 |
| 날짜 폴더 (`2026-04-02` ~ `2026-05-15`, 11개) | 초기 세션별 원시 로그/산출물(날짜순 자연 정렬, 별도 정리 불필요) |

## 9. 2026-07-19 세션에 새로 만든 느슨한-파일 정리 폴더

`manuscript_archive/`, `manuscript_drafts/`, `meetings/`, `analysis_brain3d/`, `analysis_celltype_layer/`, `analysis_domain_esm2/` — 상세는 [[session_20260719_repo_reorg]] 참조.

---

## 경고 — 이동 시 반드시 재확인할 것

아래 디렉토리는 **재현성 인용이 직접 걸려 있어** 향후에도 이동 금지(또는 이동 시 반드시 참조 패치 동반):
- `exp_g_uniprot/`, `exp_h_uniprot_eval/`, `exp_f_plm_scale/`, `domain_ranking_validation/`, `novel_go_matched_null/` — `natcomm_v0.md` 본문 직접 인용
- `truebrain_rerun_20260714/`, `v_expanded_gomf/` — 다수 memory 노트 + 본문 §6/§Comparative analysis 인용
- `v17f_*` 계열 전체, `sota_final_benchmark/` — `hMuscle/model/*.py` 스크립트 다수가 상대경로로 재사용
