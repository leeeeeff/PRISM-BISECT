# Section 01: Muscle Performance

## Key Question
How does v15d_bp_clean (ESM-2 MLP) compare against LR, RF, and XGBoost baselines on muscle isoform GO function prediction?

## Data Sources
- Muscle biopsy isoforms (●): hMuscle, ~10,000 isoforms
- GO annotations (◇): UniProt/SwissProt Swiss-Prot human
- Bootstrap CI: gene-block resampling n=1,000 (reports/bootstrap_ci/20260515_0240/)
- XGB results: reports/xgb_baseline/xgb_results_20260518_1050.json
- Bias check: reports/xgb_baseline/xgb_bias_check_20260518_1120.json
- 18-term eval: reports/v15_bp_clean/cross_go_18go_20260519_1914.json

## Key Findings
- Macro AUPRC (13 terms): v15d=0.6935, XGB=0.7384, RF=0.6258, LR=0.4239
- Type-B terms (11/13): v15d macro=0.685 vs LR macro=0.363 — delta=+0.322
- Type-A terms (2/13, Motor activity + Glycolysis): LR slightly better (LR=0.760, v15d=0.742)
- Bootstrap CI (5 terms): v15d significantly better than LR on Ca2+ signaling (p<0.001), Sarcomere org (p<0.001), Muscle contraction (p<0.001)
- XGB CI overlap: v15d falls within XGB 95% CI for all 13/13 terms (statistical non-inferiority)
- XGB gene-level bias: mean_bias_xgb=0.026 vs mean_bias_lr=0.080 — XGB lower bias despite higher point estimates
- 18-term extended: macro AUPRC=0.702; Sarcomere org highest (0.867), TOR signaling lowest (0.496)
- Neuroscience terms (5 new): MT-based movement=0.740, Synaptic transmission=0.667

## Figures
- fig01_1_baseline_comparison: Grouped dot plot, 13 GO terms × 4 models, Type-A/B colored y-axis (89×130mm)
- fig01_2_bootstrap_ci: Forest plot, 5 terms, v15d vs LR with 95% CI, p-value annotations (89×100mm)
- fig01_3_xgb_challenge: 2-panel — CI overlap (XGB bar + v15d dot) + bias comparison bar chart (183×90mm)
- fig01_4_18go_extended: Dot plot, 18 terms, muscle (blue) vs neuro-related (orange), macro AUPRC line (89×110mm)

## Interpretation
v15d_bp_clean substantially outperforms LR on Type-B GO terms (heterogeneous functional classes). XGB achieves similar performance but relies less on gene-level features (lower bias). The performance advantage of v15d over LR is largest for structurally heterogeneous terms where ESM-2 embedding captures isoform-intrinsic features that linear models miss.
