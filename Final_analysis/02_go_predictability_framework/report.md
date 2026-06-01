# Section 02: GO Predictability Framework

## Key Question
What structural properties of GO terms determine whether deep learning (v15d) outperforms linear models (LR)?

## Data Sources
- Case analysis: reports/case_analysis/case_analysis_20260518_0043.json
- TypeAB classifier: reports/typeAB_classifier/typeAB_results_20260516_1251.json
- TBS/TCS analysis: reports/tbs_tcs_13terms/tbs_tcs_results_20260518_0014.json

## Key Findings
### Type-A/B Classification (sep_cosine threshold)
- Sep_cosine threshold=0.060 (corrected from original incorrect 1.0-2.5 range)
- LOOCV accuracy: 13/13 (100%) on 13-term dataset
- Type-A (sep_cosine >= 0.060): Glycolysis (0.737), Motor activity (0.167) — LR comparable/better
- Type-B (sep_cosine < 0.060): 11 terms — v15d consistently superior
- Pearson r(sep_cosine, ΔAUPRC) = -0.601, p=0.030 (from case analysis)

### 3-Case Quantitative Framework (pc1_var_ratio)
- pc1_var_ratio is the strongest predictor: r=-0.765, p=0.002
- Case 1 (pc1>0.35): Motor activity (0.407), Glycolysis (0.323) — LR sufficient, ΔAUPRC < 0
- Case 2 (0.28≤pc1<0.35): TOR signaling, Skel. muscle dev, Sarcomere org — moderate DL advantage
- Case 3 (pc1<0.28): 8 terms — largest DL advantage, ΔAUPRC range 0.287-0.475
- Case 3 includes: Ca2+homeostasis (Δ=0.475), Muscle organ dev (Δ=0.465), Muscle cell diff (Δ=0.421)

### TBS/TCS Negative Result (annotation quality)
- TBS vs ΔAUPRC: r=-0.143, p=0.640 — NO significant correlation
- TCS vs ΔAUPRC: r=-0.100, p=0.744 — NO significant correlation
- Counter-example: Motor activity (TBS=0.833) vs Ca2+ homeostasis (TBS=0.833) — same TBS but ΔAUPRC differs by 0.49
- Structural heterogeneity (pc1_var_ratio), NOT annotation quality, determines model advantage

## Figures
- fig02_1_typeAB: Scatter sep_cosine vs ΔAUPRC, Type-A (blue)/B (orange), threshold line, r and p annotation (89×89mm)
- fig02_2_3case: 2-panel — pc1_var_ratio vs ΔAUPRC scatter + violin/box by case category (183×80mm)
- fig02_3_tbs_tcs: 2-panel scatter — TBS vs ΔAUPRC and TCS vs ΔAUPRC, both show no correlation, Motor/Ca2+ counterexample annotated (183×80mm)

## Interpretation
The predictability of a GO term by deep learning versus linear models is governed by the structural heterogeneity of positive-class ESM-2 embeddings (quantified by pc1_var_ratio). When positives form a compact, linearly separable cluster (high pc1_var_ratio), LR is sufficient. When positives are structurally diverse (low pc1_var_ratio, multiple protein families share the function), ESM-2's nonlinear feature space is required. This framework is directly falsifiable and contributes a novel mechanistic understanding to the "when is deep learning needed for GO prediction" question — a key methodological contribution for Nature Methods.
