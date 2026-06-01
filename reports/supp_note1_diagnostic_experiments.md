# Supplementary Note 1 — Phase 1 Diagnostic Experiments (D1–D3)
**2026-05-17 | DIFFUSE v10-B**

---

## Overview

Prior to designing DIFFUSE v10-B, three diagnostic experiments (D1–D3) were performed to
identify the architectural bottleneck and determine whether isoform-specific features (domain
delta, splicing delta) could improve upon ESM-2 embeddings alone. These experiments guided
the decision to adopt the ESM-2-only deep MLP architecture (v10-B) and to discard additional
feature modalities.

---

## D1 — ESM-2 640d → MLP (without PFN compression)

**Motivation**: The prior best model (v8b PFN, macro AUPRC = 0.357) performed substantially
below the ESM-2 logistic regression (LR) baseline (macro = 0.424). D1 tested whether the
PFN bottleneck (640d → 64d) was the primary limiting factor.

**Setup**: ESM-2 640-dimensional embeddings (esm2_t30_150M_UR50D) passed directly to a
three-layer MLP (Dense 256 → Dropout 0.3 → Dense 128 → Dense 64 → L2 normalisation →
sigmoid), trained with BinaryFocalCrossentropy (γ = 2.0, class-balanced α) and evaluated
by 5-fold gene-stratified GroupKFold cross-validation.

**Result**: D1 macro AUPRC = 0.690 — substantially above the LR baseline (0.424).

**Gate conclusion** (Gate 1 PASS): The PFN compression from 640d to 64d was the primary
bottleneck. No additional isoform-specific features were needed to establish a strong baseline;
a simple MLP on raw ESM-2 embeddings sufficed.

---

## D2 — ESM-2 Δ = ESM2(iso) − ESM2(canonical) → LR

**Motivation**: If gene-level information dominates ESM-2 absolute embeddings (because all
isoforms of a gene share most of their sequence), subtracting the canonical isoform embedding
should remove gene-level bias and expose isoform-specific signal.

**Setup**: For each isoform *i* of gene *g*, the delta embedding was computed as:
**Δᵢ = ESM-2(isoform i) − ESM-2(canonical isoform of g)**
where the canonical isoform was selected as the GENCODE principal isoform (MANE Select or
longest CDS). The 640-dimensional Δ vector was used as input to logistic regression with
gene-stratified evaluation.

**Result**: D2 macro AUPRC ≈ 0 (near-random performance across all GO terms).

**Interpretation**: The ESM-2 embedding difference between isoforms of the same gene is
dominated by sequence-length effects and noise rather than functionally meaningful structural
differences. Subtracting the canonical embedding destroys the gene-family discriminative
information that drives AUPRC without recovering meaningful isoform-specific signal.

**Conclusion**: The canonical-differential representation was abandoned. ESM-2 absolute
embeddings are the correct input format; gene-level bias in the embeddings is addressed
by the triplet loss objective during training rather than by input-space subtraction.

---

## D3 — ESM-2 Δ + Domain delta (D_sign) + Splicing delta (S) → LR

**Motivation**: Test whether isoform-specific features (domain gain/loss signs, exon-level
splicing deltas) provide additional signal over the ESM-2 delta representation.

**Setup**: Input was the concatenation of:
- ESM-2 delta (640d, as in D2)
- Domain delta sign (251d binary: +1 = domain gained, −1 = lost, 0 = unchanged, relative to canonical)
- Splicing delta v2 (150d: per-exon Δ PSI values, sorted by exon number)

Total: 1,041-dimensional vector → LogisticRegression (class-balanced, C=1.0).

**Result**: D3 macro AUPRC ≈ 0 (near-random, comparable to D2).

**Interpretation**: Given that D2 ≈ 0, D3 could not recover performance either. The D/S
features alone cannot compensate for the destroyed gene-level discriminative information.
A separate ablation (hMuscle/model/esm2_640dim_ablation.py) confirmed that adding D/S features
to ESM-2 absolute embeddings (640d + D/S) slightly *decreased* performance, indicating that
the features encode information already captured by ESM-2 and add noise under gene-stratified
evaluation.

**Conclusion**: D/S features were excluded from v10-B. The final architecture uses ESM-2 640d
absolute embeddings as the sole input, with domain-level functional information implicitly
encoded by ESM-2 pretraining.

---

## Summary

| Experiment | Input | Model | Macro AUPRC | Conclusion |
|-----------|-------|-------|-------------|------------|
| D1 | ESM-2 640d (absolute) | Deep MLP | 0.690 | Gate 1 PASS — PFN was bottleneck |
| D2 | ESM-2 Δ (canonical-subtracted) | LR | ≈ 0.00 | Delta representation invalid |
| D3 | ESM-2 Δ + D_sign + S | LR | ≈ 0.00 | D/S insufficient without gene-level signal |
| **v10-B** | **ESM-2 640d (absolute)** | **Deep MLP** | **0.685 (Type-B, 11 terms)** | **Final model** |

All experiments used gene-stratified 5-fold GroupKFold cross-validation (Section 4.7) and
AUPRC as the primary metric.

**Script references**:
- D1: `hMuscle/model/v10_diagnostic.py`
- D2/D3: `hMuscle/model/esm2_640dim_ablation.py`
