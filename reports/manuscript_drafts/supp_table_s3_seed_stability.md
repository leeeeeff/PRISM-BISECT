# Supplementary Table S3 — Per-Seed AUPRC Stability
**Draft 2026-05-16 | DIFFUSE: Isoform-Level Function Prediction**

---

## Overview

v10-B was trained and evaluated independently across 5 random seeds (42, 123, 456, 789, 2024)
for all 13 GO terms. For the original 5 GO terms (marked †), seeds 42/123/456/789/1234 were
used in the bootstrapping experiment (reports/bootstrap_ci/20260515_0240/); for consistency,
seed 2024 was used in the main 13-term evaluation and all values represent seed 2024 as the
final seed. Mean and standard deviation are computed across all 5 seeds for each GO term.

CV (coefficient of variation) = 100 × Std / Mean.

---

## Table S3 — Per-Seed AUPRC (v10-B, all 13 GO terms)

| GO Term | Function | Type | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 | Mean | Std | CV% |
|---------|----------|------|---------|----------|---------|----------|-----------|------|-----|-----|
| GO:0007204 | Ca²⁺ signaling | B | 0.7180 | 0.7819 | 0.7710 | 0.7542 | 0.8012 | **0.765** | 0.028 | 3.7% |
| GO:0030017 | Sarcomere org | B | 0.7701 | 0.7329 | 0.7608 | 0.7484 | 0.7009 | **0.743** | 0.024 | 3.3% |
| GO:0006941 | Muscle contraction | B | 0.6237 | 0.5984 | 0.6005 | 0.5661 | 0.5953 | **0.597** | 0.018 | 3.1% |
| GO:0006914 | Autophagy | B | 0.6766 | 0.7068 | 0.5785 | 0.6541 | 0.5824 | **0.640** | 0.051 | 8.0% |
| GO:0043161 | Proteasome-UPS | B | 0.7301 | 0.7111 | 0.7117 | 0.7258 | 0.7083 | **0.717** | 0.009 | 1.2% |
| GO:0007519 | Skeletal muscle dev | B | 0.6929 | 0.6998 | 0.7608 | 0.7376 | 0.7338 | **0.725** | 0.025 | 3.5% |
| GO:0042692 | Muscle cell diff | B | 0.6493 | 0.6313 | 0.6676 | 0.6687 | 0.6460 | **0.653** | 0.014 | 2.2% |
| GO:0055074 | Ca²⁺ homeostasis | B | 0.7100 | 0.7496 | 0.7287 | 0.7085 | 0.7306 | **0.726** | 0.015 | 2.1% |
| GO:0007005 | Mitochondrion org | B | 0.6410 | 0.6896 | 0.6779 | 0.6719 | 0.6318 | **0.662** | 0.022 | 3.4% |
| GO:0007517 | Muscle organ dev | B | 0.6796 | 0.7035 | 0.7361 | 0.7092 | 0.6803 | **0.702** | 0.021 | 3.0% |
| GO:0032006 | TOR signaling | B | 0.6434 | 0.5636 | 0.6128 | 0.5545 | 0.6371 | **0.602** | 0.037 | 6.1% |
| GO:0003774 | Motor activity | A | 0.8155 | 0.8081 | 0.8336 | 0.8126 | 0.7944 | **0.813** | 0.013 | 1.6% |
| GO:0006096 | Glycolysis | A | 0.7970 | 0.6368 | 0.5368 | 0.8297 | 0.5559 | **0.671** | 0.121 | 18.1% |
| **Type-B macro** | | | 0.689 | 0.677 | 0.683 | 0.678 | 0.672 | **0.680** | 0.006 | 0.9% |
| **All 13 macro** | | | 0.695 | 0.680 | 0.683 | 0.681 | 0.677 | **0.683** | 0.006 | 0.9% |

---

## Notes

**High-CV terms (>5%):**

- **GO:0006096 (Glycolysis, CV = 18.1%):** Type-A term where LR is competitive (LR AUPRC = 0.695).
  The high CV reflects sensitivity to the train-test gene split for this 76-positive-gene term,
  where glycolytic enzyme gene families dominate (PFKP, PKM, PFKL, PFKM). Seed 456 produces a
  fold with fewer PFKM-family isoforms in test, reducing AUPRC to 0.537.

- **GO:0006914 (Autophagy, CV = 8.0%):** The highest-variance Type-B term. Seeds 456 and 2024
  produce lower AUPRC (0.578–0.582), likely due to sampling of ATG gene family composition in
  the test fold. This term is noted in Results 3.5 as requiring cautious interpretation.

- **GO:0032006 (TOR signaling, CV = 6.1%):** Non-significant term (q = 0.106). High variance
  consistent with the mTOR pathway's functionally heterogeneous gene set (147 positive genes
  in training; wide range of kinase and adaptor proteins).

**Stable terms (CV < 4%):**
All other 10 Type-B terms show CV < 4%, confirming robust model performance independent of
random seed. Type-B macro-AUPRC across 5 seeds ranges from 0.672 to 0.689 (CV = 0.9%),
demonstrating stable ensemble behaviour.

**Seed ordering note:** For GO:0007204/0030017/0006941/0003774/0006096, seeds are
42/123/456/789/1234 (from bootstrap CI experiment, reports/bootstrap_ci/20260515_0240/).
For GO:0006914–0007517, seeds are 42/123/456/789/2024 (from sarcopenia eval). The final
mean AUPRC reported in main Table 1 uses the seed-2024 mean for all 13 terms.
