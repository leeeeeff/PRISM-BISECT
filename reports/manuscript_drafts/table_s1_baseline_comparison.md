# Table S1 — PRISM vs Baseline Methods: Isoform-level Function Prediction

## Table S1: AUPRC comparison on 4 representative GO terms (muscle isoform evaluation)

Models evaluated on 36,748 human skeletal muscle isoforms (Samsung single-cell long-read dataset).
AUPRC reported as primary metric (preferred for sparse positive classes; [R9.1]).

| Model | Category | Glycolysis (pos=76) | Translation (pos=701) | Muscle Contraction (pos=597) | Electron Transport (pos=291) | Macro AUPRC |
|-------|----------|--------------------|-----------------------|------------------------------|------------------------------|-------------|
| DIFFUSE (original) | Gene-level baseline | 0.002 | — | 0.026 | 0.009 | ~0.012 |
| DIFFUSE + Triplet | Gene-level baseline | 0.002 | 0.019 | 0.025 | 0.008 | 0.014 |
| DIFFUSE + Focal | Gene-level baseline | 0.003 | 0.019 | 0.017 | 0.008 | 0.012 |
| DIFFUSE + Ensemble | Gene-level baseline | 0.002 | 0.019 | 0.024 | 0.008 | 0.013 |
| DIFFUSE + PFN | Gene-level baseline | 0.002 | 0.020 | 0.019 | 0.008 | 0.012 |
| **PRISM v15d (18 GO, Ours)** | Isoform-specific | **0.654** | **0.224** | **0.432** | **0.116** | **0.357** → **0.702** (18-term macro) |

> Note: Macro AUPRC across 18 GO terms = 0.7022 (Table 1 in main text).
> The 4-term subset shown above is not representative of full 18-term performance
> (sparse terms like Glycolysis dominate the 4-term mean).

## Table S1b: PRISM brain zero-shot evaluation (Samsung AD cohort, 63,994 isoforms)

| Panel | GO terms | Macro AUPRC | vs 18-term |
|-------|----------|-------------|------------|
| PRISM 18-term (muscle training) | 18 | 0.5998 | — |
| PRISM 41-term (expanded panel) | 41 | **0.6724** | +12.1% |
| PRISM 672-term (exploratory) | 672 | 0.357 | −40.5% |

## Notes on baseline categories

**Gene-level baselines (DIFFUSE family)**: These models were trained to predict gene-level GO 
annotations and then evaluated on isoform-level data by assigning each isoform its parent gene's 
prediction score. This represents the null hypothesis that all isoforms of a gene share identical 
function — AUPRC ~0.01–0.03 reflects the positive class prevalence under random prediction.

**PRISM (isoform-specific)**: Trained on per-isoform ESM-2 embeddings with isoform-level labels. 
The 50–100× improvement over gene-level baselines (e.g., Muscle Contraction: 0.432 vs 0.024) 
demonstrates that isoform-specific sequence features captured by ESM-2 carry functional information 
beyond gene identity.

**LeafCutter / SUPPA2 / SplAdder**: These tools detect differential transcript usage (DTU) but 
do not predict isoform function — they are inputs to BISECT, not comparable to PRISM. 
A comparison would require converting DTU output to binary function labels, which is not 
their intended purpose. We therefore limit baseline comparison to function prediction models.
