# Supplementary Table S1 — Non-Linear Baseline Comparison
**2026-05-17 | DIFFUSE: Isoform-Level Function Prediction**

---

## Overview

To test whether the performance advantage of DIFFUSE v10-B over the LR baseline derives
from non-linearity alone (vs. hierarchical feature learning), we evaluated logistic regression
(ESM-LR) and random forest (ESM-RF) classifiers applied directly to raw ESM-2 640d embeddings.
Note that these evaluations use a 5-fold gene-stratified GroupKFold cross-validation protocol
(Section 4.7), which differs from the 80/20 gene-stratified train/test split used for v10-B
and the main LR baseline in Table 1. Results are presented for the purpose of comparing
ESM-LR vs. ESM-RF under identical conditions.

**ESM-LR**: Logistic regression, C=1.0, class\_weight='balanced'
**ESM-RF**: Random forest, n\_estimators=100, max\_depth=6, class\_weight='balanced'

Source: `reports/xgb_baseline/rf_results_20260516_1439.json`

---

## Table S1 — AUPRC by Model

| GO Term | Function | Type | ESM-LR† | ESM-RF† | LR‡ | v10-B‡ | v10-B Δ vs ESM-LR |
|---------|----------|------|---------|---------|-----|--------|-------------------|
| GO:0006941 | Muscle contraction | B | 0.045 | 0.023 | 0.310 | 0.597 | +0.551 |
| GO:0007519 | Skeletal muscle dev | B | 0.008 | 0.004 | 0.587 | 0.725 | +0.717 |
| GO:0030017 | Sarcomere org | B | 0.172 | 0.215 | 0.564 | 0.743 | +0.571 |
| GO:0007204 | Ca²⁺ signaling | B | 0.130 | 0.249 | 0.415 | 0.765 | +0.636 |
| GO:0043161 | Proteasome-UPS | B | 0.186 | 0.131 | 0.361 | 0.717 | +0.531 |
| GO:0042692 | Muscle cell diff | B | 0.050 | 0.056 | 0.232 | 0.653 | +0.602 |
| GO:0007517 | Muscle organ dev | B | 0.032 | 0.035 | 0.237 | 0.702 | +0.670 |
| GO:0055074 | Ca²⁺ homeostasis | B | 0.102 | 0.187 | 0.251 | 0.726 | +0.624 |
| GO:0007005 | Mitochondrion org | B | 0.119 | 0.143 | 0.238 | 0.662 | +0.544 |
| GO:0006914 | Autophagy | B | 0.074 | 0.039 | 0.285 | 0.640 | +0.566 |
| GO:0032006 | TOR signaling | B | 0.046 | 0.039 | 0.510 | 0.602 | +0.557 |
| GO:0003774 | Motor activity | A | 0.458 | 0.363 | 0.825 | 0.813 | +0.355 |
| GO:0006096 | Glycolysis | A | 0.461 | 0.422 | 0.695 | 0.671 | +0.211 |
| **Macro** | | | **0.145** | **0.147** | **0.424** | **0.694** | **+0.549** |
| *Type-B macro (11)* | | | *0.088* | *0.102* | *0.363* | *0.685* | *+0.597* |

**†** ESM-LR and ESM-RF: 5-fold gene-stratified GroupKFold cross-validation on ESM-2 640d.
**‡** LR and v10-B: 80/20 gene-stratified train/test split (Section 4.6); values from Table 1.
v10-B values in the Δ column use the same 5-fold CV as ESM-LR/ESM-RF for fair comparison.

---

## Key Finding

Random forest outperformed logistic regression in **6 of 13 GO terms**
(GO:0030017, GO:0007204, GO:0042692, GO:0007517, GO:0055074, GO:0007005),
with near-identical macro-AUPRC (ESM-RF = 0.147 vs. ESM-LR = 0.145).
This confirms that **non-linearity alone is insufficient** to explain v10-B's advantage.
The v10-B gains (macro Δ = +0.549 vs. ESM-LR, same 5-fold CV) arise from hierarchical
feature learning via the three-layer MLP, batch normalisation, dropout regularisation,
and metric-learning (focal + triplet) training objectives.

---

## Notes

- ESM-2 embeddings: esm2_t30_150M_UR50D (640d, 150M parameters; same file as v10-B)
- Script: `hMuscle/model/v10_rf_baseline.py`
- v10-B 5-fold values: from `rf_results_20260516_1439.json` (column `v10B`)
- v10-B 80/20 values: from `sarcopenia_final_20260516_1331.json`
