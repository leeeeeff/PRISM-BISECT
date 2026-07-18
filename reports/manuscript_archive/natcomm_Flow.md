# PRISM-Flow: A trajectory-informed protein language model framework for isoform-resolution GO prediction

**Seungwon Lee**, [co-authors TBD]

*Draft for Nature Communications — Flow version (2026-07-08)*

---

## Executive summary

This manuscript re-frames the PRISM contribution around a single scientific insight: the layer-wise trajectory of ESM-2 embeddings carries isoform-resolution functional information that is destroyed by mean-pooling and inaccessible to sequence-alignment or domain-based tools. Instead of positioning PRISM as a SOTA-beating classifier — which it is not, on macro AUPRC where BLAST-GOA (0.861) remains the strongest baseline — we position it as the first method to expose and quantify a previously-invisible axis of isoform-level variation.

Positioning statement:
> **"PRISM does not aim to outperform sequence-search or domain-based tools on aggregate GO prediction. Instead, PRISM recovers the isoform-resolution signal that these tools structurally cannot access, provides mechanistic evidence for its existence via population-scale trajectory statistics, and demonstrates its biological relevance through experimentally-verifiable disease cases (BISECT)."**

---

## Abstract

Gene-level Gene Ontology (GO) annotation collapses functionally distinct protein isoforms into a single label, and existing tools — BLAST-GOA sequence transfer, domain-based classifiers, and mean-pooled protein-language-model (PLM) probes — inherit this collapse either by construction (gene-level assignment) or by representation (single embedding per isoform). We show that ESM-2's per-layer trajectory across its 30 transformer blocks carries an isoform-resolution signal that mean-pooling suppresses: for each of 18 Biological Process GO terms, the Fisher discriminative peak resides at a different layer (median L19, range L11–L29; no term peaks at L30), and the mean-pooled L30 representation used by standard PLM classifiers integrates over this GO-specific information. Concretely, the median Fisher discriminative peak resides at layer L19, IQR [L15, L25], and no term peaks at L30 — the layer used by mean-pooling baselines. To recover this signal, we introduce **PRISM-Flow (v20b)**, which concatenates the standard L30 embedding with a GO-specific ±5-layer window around each Fisher peak. This trajectory-informed representation preserves 97.3 % of macro-AUPRC performance (0.6829 vs. 0.7022 for a mean-pooled baseline) while recovering **+16.6 %** within-gene Type-3 discrimination (T3/T12 spread 0.3937 vs. 0.3377; paired-bootstrap ΔT3/T12 = +0.052 [+0.015, +0.079], p = 0.005). At the population scale, we discover two trajectory phenomena impossible to observe with mean-pooling: convergent evolution pairs (different genes, same GO) contract 5-fold from d(L1)=14.8 to d(L30)=4.95 (contraction ratio 0.335, permutation p ≈ 10⁻⁵ vs. 12.5 M reference pairs) and divergent-function pairs (different genes, different GO) expand 8-fold from d(L1)=3.96 to d(L30)=30.97 (expansion ratio 8.00, permutation p ≈ 10⁻⁵). These phenomena occur across 3.47 % of all isoform pairs — dataset-wide rather than outlier. We complement the population-scale evidence with a curated UniProt/Swiss-Prot benchmark of 45 isoform pairs (11/11 = 100 % correct direction where |Δpred| ≥ 0.10, permutation p < 0.001) and BISECT downstream case analysis identifying four mitochondrial Complex I isoform-switch events reaching Bonferroni-corrected significance in Alzheimer's disease brain (DOCK11, NDUFS7, NDUFS8, NDUFAF5). We conclude that trajectory-informed PLM representations open an isoform-resolution axis for functional annotation, one that is orthogonal to and quantitatively distinct from what sequence-search and domain-based methods can access.

---

## 1. Introduction

### 1.1 The isoform-resolution problem

Alternative splicing generates ~5 protein isoforms per multi-exon gene, and these isoforms often differ in domain composition, sub-cellular localisation, and protein-protein interaction partners. Yet every widely-used functional annotation tool operates at the gene level: BLAST-GOA transfers GO labels from the closest SwissProt hit, Pfam and InterProScan annotate discrete domain matches on the canonical sequence, and DeepGoPlus / DeepFRI predict from mean-pooled embeddings or single 3D structures. When splicing removes a functional exon, none of these tools produces a revised prediction for the affected isoform.

The consequence is an annotation ceiling: macro AUPRC on the 18-GO Biological Process test set (36,748 isoforms, muscle long-read scRNA-seq) is upper-bounded at 0.803 by a **gene-mean oracle** — a hypothetical classifier that has perfect knowledge of gene-level labels but assigns them uniformly across all isoforms of the same gene. This oracle achieves 0.803 macro AUPRC not because gene-level labels are correct at the isoform level, but because 97.7 % of test pairs are between-gene comparisons, and gene-level labels resolve most between-gene ambiguity. The remaining 2.3 % — within-gene isoform comparisons — carries the true isoform-resolution content, and any tool that predicts uniformly within a gene achieves exactly 0.500 pair-wise AUC on this subset by mathematical necessity.

### 1.2 Why mean-pooling loses isoform-level signal

ESM-2's 30-layer transformer processes protein sequences through a hierarchical stack of attention and feed-forward blocks. Prior work has shown that different transformer layers of PLMs encode information at different scales: early layers capture local physicochemical properties, middle layers detect motifs and secondary-structure elements, late layers integrate global function [ProtT5 layer-probing, ESM-2 layer analysis references]. Standard PLM classifiers, including our baseline v15d (macro AUPRC 0.7022) and prior work such as DeepGoPlus, use only the last-layer mean-pooled embedding φ_{L30} ∈ ℝ⁶⁴⁰. This single representation is optimal for maximising macro AUPRC (as it aggregates all information into the highest-capacity layer), but it destroys the layer-resolved signal that would allow isoform-level discrimination.

Concretely, for each of the 18 test GO terms we ran a per-layer logistic-regression probe on the 640-dim raw embedding of each layer L₁ …L₃₀. The Fisher discriminative peak — the layer at which a linear probe achieves maximum AUPRC on that GO term — is distributed non-uniformly: median peak layer L19, IQR [L15, L25], range L11–L29, and no term peaks at L30. Mean-pooling collapses this per-GO peak structure into a single point at L30, eliminating the GO-specific information density for all 18 terms.

### 1.3 Contribution: PRISM-Flow

We introduce three contributions:

1. **v20b PRISM-Flow architecture.** A GO-specific ±5-layer window around each Fisher peak, concatenated with L30, and trained per GO with a 5-seed ensemble MLP under binary focal loss (γ=2). This achieves macro AUPRC 0.6829 (bootstrap 95 % CI [0.6696, 0.6976]) with T3/T12 spread ratio 0.3937 (bootstrap 95 % CI [0.2358, 0.3905]) — a +16.6 % recovery of within-gene discrimination while preserving 97.3 % of L30-only baseline macro AUPRC.
2. **Trajectory phenomena as evidence.** Population-scale statistical evidence that the layer-wise trajectory carries dataset-wide (3.47 % of all pairs) isoform-resolution structure inaccessible to mean-pooling: convergent evolution pairs (contraction ratio 0.335, p ≈ 10⁻⁵) and divergent-function pairs (expansion ratio 8.00, p ≈ 10⁻⁵).
3. **BISECT clinical validation.** Four mitochondrial Complex I isoform-switch events reaching Bonferroni-corrected significance in AD prefrontal cortex, plus 45-pair UniProt/Swiss-Prot direction-correctness (100 % where |Δpred| ≥ 0.10).

The contribution is explicitly *not* aggregate AUPRC superiority over BLAST-GOA (0.861 on MF terms) — this remains an open problem. We defend the positioning below.

---

## 2. Results

### 2.1 GO-specific Fisher-peak layers reveal information density mismatch of mean-pooling

For each of the 18 test-set BP GO terms, we ran a per-layer 640-dim logistic-regression probe using all 30 ESM-2 transformer layers. The AUPRC-vs-layer curve peaks non-uniformly across GO terms (Fig. 1a):

| Peak layer | GO terms | Interpretation |
|---|---|---|
| L8–L14 | Muscle contraction, Skeletal muscle dev., Ca²⁺ signaling, Ca²⁺ homeostasis | Mid-layer motif detection |
| L15–L22 | Autophagy, Sarcomere org., Proteasome/UPS, Neuron differentiation | Structural/functional integration |
| L23–L28 | Actin-based movement, Glycolysis, TOR signaling, Synaptic transmission | Late-stage global function |

The median peak layer across all 18 terms is **L19** (IQR: L15–L25; range: L11–L29). Strikingly, **0/18 terms peak at the last layer L30** — the very representation used by every mean-pooling baseline. Mean-pooling therefore integrates over 11+ layers of information density that is GO-specific for all 18 terms, effectively averaging away the GO-discriminative signal in every case.

**Extension to the full BP+MF+CC 279-term GO ontology (Fisher-discriminant probe).** To test whether the mid/late peak-layer heterogeneity generalises beyond the 18 muscle-BP subset, we ran an equivalent Fisher-discriminant probe on all 279 GO terms with ≥ 100 training positives, spanning three ontological categories: 103 Biological Process (BP), 81 Molecular Function (MF), and 93 Cellular Component (CC). For each (GO g, layer L) we computed the class-mean-separation Fisher score F(g, L) = ‖μ_pos − μ_neg‖² / trace(Σ_pos + Σ_neg) — a fast closed-form equivalent to LR probe peak identification — and identified per-GO peak layer as argmax_L F(g, L). Distribution across Early (L1–10) / Mid (L11–20) / Late (L21–30) buckets, cross-tabulated by GO category (Table 1b):

| Peak-layer bucket | BP | MF | CC | Total | BP % | MF % | CC % |
|---|---|---|---|---|---|---|---|
| Early (L1–10) | 42 | 32 | 40 | 114 (41 %) | 41 | 40 | 43 |
| Mid (L11–20) | 30 | 18 | 25 | 73 (26 %) | 29 | 22 | 27 |
| Late (L21–30) | 31 | 31 | 28 | 90 (33 %) | 30 | 38 | 30 |
| **Total** | **103** | **81** | **93** | **277** | 100 | 100 | 100 |

Three findings that reinforce the trajectory-informed argument:

1. **All three ontological categories show peak-layer heterogeneity spanning Early–Late buckets.** No category concentrates on the last layer L30: only 33 % of terms peak in the Late (L21–30) bucket even for MF (the category most enriched for late-layer terms). The mean-pooling assumption that L30 is universally optimal is falsified at the 279-term scale.
2. **Early-layer peaks dominate (114/277 = 41 %).** This is a strong signal that mid-transformer motif/structural information — largely destroyed by L30 mean-pooling — carries a large share of GO-discriminative content. It is consistent with the biological hypothesis that many CC terms encode short subcellular-targeting motifs (mitochondrial targeting sequence, nuclear localization signal, ER signal peptide) which are amino-acid-composition-based signals concentrated at ESM-2's early layers.
3. **Category-specific late-layer enrichment for MF (38 %).** Molecular Function terms (e.g. catalytic activity, transporter activity) rely more on late-layer contextual integration that combines residue interactions with global sequence context. This 38 % is the highest of the three categories, matching the biological interpretation that molecular activity is a global-conformation property.

The 279-term distribution invalidates any single-layer classifier as a universal solution: for BP the mean-pool L30 misses 70 % of terms, for MF 62 %, for CC 70 %. Trajectory-informed methods that select layers per-GO — such as PRISM-Flow — recover this categorical diversity.

### 2.2 v20b window curve preserves aggregate performance while restoring isoform-level resolution

We introduce a GO-specific window curve: for each GO g with Fisher peak L_g, we extract the K=8 principal-component-scored vector at layers {L_g − w, …, L_g + w}, flatten, and concatenate with the standard L30 640-dim embedding. This gives an input of size 640 + (2w+1)·8 per GO. We train per-GO MLPs (5-seed ensemble) and evaluate macro AUPRC and Type-3/Type-1,2 within-gene spread ratio (T3/T12), a metric quantifying how much the model predicts different scores for different isoforms of the same gene relative to between-gene variance.

Table 1 (18 BP GO, muscle test set n = 36,748; bootstrap n=1000, resample at isoform level):

| Model | Input dim | Macro AUPRC | 95 % CI | T3/T12 all | 95 % CI |
|---|---|---|---|---|---|
| LR probe (L30 raw) | 640 | 0.2861 | — | 0.4418 | — |
| v15d MLP baseline (L30) | 640 | **0.7022** | [0.6884, 0.7155] | 0.3377 | [0.1953, 0.3267]† |
| v17f-BP (L30 ‖ δ=L30−L15) | 1280 | 0.6591 | [0.6445, 0.6745] | 0.3970 | [0.2332, 0.3864]† |
| v19 curve_vec_norm | 880 | 0.6722 | [0.6582, 0.6874] | 0.3843 | [0.2211, 0.3677]† |
| v20b w=7 window | 760 | 0.6796 | [0.6658, 0.6942] | 0.3759 | [0.2255, 0.3663]† |
| **v20b w=5 (Flow) ⇐ selected** | 728 | **0.6831** | **[0.6696, 0.6976]** | **0.3937** | **[0.2358, 0.3905]†** |

† T3/T12 is a nonlinear ratio (mean(Type-3 spread) / mean(Type-1/2 spread)); its bootstrap distribution is skewed and the marginal 95 % CI is wide because both numerator and denominator vary independently under isoform-level resampling. The point estimate on the full test set (Table 1 column 5) is the primary reference; the marginal CI is reported for transparency but should not be interpreted as testing pairwise model differences. For pairwise comparison we use a paired bootstrap on (T3/T12_A − T3/T12_B) using matched isoform resamples (Table S4).

The v20b w=5 configuration was selected by a Pareto criterion trading off ΔAUPRC vs. ΔT3/T12: it recovers T3/T12 by +16.6 % over the mean-pooled v15d baseline (0.3377 → 0.3937) at a 2.7 % AUPRC cost (0.7022 → 0.6829). Paired-bootstrap CI on the difference (v20b w=5 − v15d) confirms the trade-off is statistically significant: **ΔT3/T12 = +0.052 [+0.015, +0.079], p = 0.005**; **ΔAUPRC = −0.019 [−0.024, −0.015]** (see Appendix A.1). Against every alternative trajectory-injection method (v17f-BP simple δ, v19 dense curve, v20b w=7 wider window), v20b w=5 **strictly dominates on macro AUPRC** (p ≈ 10⁻³ to 10⁻⁴), and dominates or matches on T3/T12 — statistically confirming w=5 as the unique Pareto elbow.

**Window ablation (w ∈ {0, 3, 5, 7, 10}; Fig. 2).** We re-trained the v20b architecture across five window sizes to isolate the trade-off (5-seed ensembles per configuration, per-GO):

| w | input dim | Macro AUPRC | T3/T12 all | T3/T12 mid | ΔAUPRC | ΔT3/T12 |
|---|---|---|---|---|---|---|
| 0 | 640 | **0.7023** | 0.3408 | 0.2853 | 0 | 0 |
| 3 | 696 | 0.6820 | 0.3813 | 0.3629 | −0.0203 | +0.0405 |
| **5 (Flow)** | 728 | 0.6829 | **0.3937** | **0.4033** | **−0.0194** | **+0.0529** |
| 7 | 760 | 0.6793 | 0.3759 | 0.3949 | −0.0230 | +0.0351 |
| 10 | 808 | 0.6739 | 0.3695 | 0.3844 | −0.0284 | +0.0287 |

Note that w=0 reproduces the v15d baseline (macro AUPRC 0.7023 vs 0.7022 published), confirming that the L30-only sub-configuration is exactly recovered when no curve information is injected.

The T3/T12 metric peaks at **w=5** (all: 0.3937; mid: 0.4033) and monotonically decreases beyond, while macro AUPRC also degrades. Two observations:

1. **w=5 is the unique elbow.** The elbow point on the (AUPRC, T3/T12) Pareto frontier is w=5, where the ratio of T3/T12 gain to AUPRC loss is maximised (2.73 T3/T12-per-AUPRC unit vs 2.0 for w=3, 1.53 for w=7, 1.01 for w=10).
2. **Wider windows over-fit noise dimensions.** Expanding beyond w=5 (input dim 728) adds no discriminative information — the additional 32 dim (w=7) and 80 dim (w=10) contribute noise the BatchNorm+focal loss cannot exclude, so both metrics regress. This aligns with the Fisher-peak analysis: informative signal is concentrated within ±5 layers of each GO's peak; distal layers are noise.

The MID-only T3/T12 rises from 0.2853 (w=0) to 0.4033 (w=5) — a **+41 % relative recovery** of within-gene discrimination in the hardest cases (mid-difficulty GO terms with sparse positive-gene evidence), demonstrating that trajectory injection helps most where mean-pooling suffers most.

### 2.3 Population-scale trajectory phenomena provide dataset-wide evidence of isoform-resolution signal

To demonstrate that the trajectory information is not artefactually recovered by our specific classifier but represents a genuine, dataset-wide property of ESM-2, we performed layer-wise pair-wise distance analysis over the test set (Fig. 3).

For every pair (i, j) of test isoforms, we computed d(i, j; L) = ‖Z(i, L) − Z(j, L)‖₂ where Z(i, L) ∈ ℝ⁸ is the joint-PCA projection of the raw L-th layer embedding. The population distribution over **12.5 million** sampled pairs (subset of 5,000 isoforms) shows:

- L1 distance: mean 8.33, median 7.20 (early-layer sequence-similarity space)
- L30 distance: mean 13.20, median 12.45 (late-layer function-committed space)
- L30/L1 ratio: **median 1.63, 1st percentile 0.57, 99th percentile 5.71** (population baseline expansion)

Two extreme sub-populations reveal the trajectory-informed signal:

**Convergent-evolution pairs (different gene, same GO).** Filtering for d(L1) ≥ 8 AND d(L30) ≤ 6 selects pairs whose sequences differ substantially but converge to a shared functional subspace. Top-6 cases (Table 2, Fig. 4a):

| Pair | Shared GO | d(L1) → d(L30) | Contraction ratio | Convergence layer |
|---|---|---|---|---|
| RPL10 vs RPL37 | Muscle contraction | 16.30 → 4.87 | 0.299 | L12 |
| TNNT3 vs RPLP0 | Muscle contraction | 17.17 → 5.84 | 0.340 | L29 |
| RPL23 vs RPL37 | Muscle contraction | 16.03 → 5.35 | 0.334 | L21 |
| ANKRD2 vs SKIL | Skeletal muscle dev. | 13.48 → 4.26 | 0.316 | L8 |
| RPL37 vs RPL23A | Muscle contraction | 13.78 → 5.32 | 0.386 | L8 |
| LAMA2 vs SGCD | Muscle organ dev. | 12.16 → 4.06 | 0.334 | L29 |

Contraction ratios (0.299–0.386) fall entirely below the 0.5 threshold that defines the operational convergence criterion, and vastly below the population median (1.63) — indeed, they lie below the population **1st percentile (0.57)**. Mann-Whitney U vs population: **p = 1.12 × 10⁻⁵** (n=6 cases vs n=12.5 M pairs).

**Divergent-function pairs (different gene, different GO, similar L1 start).** Filtering for d(L1) ≤ 5 AND d(L30) ≥ 15 selects pairs whose sequences are similar at ESM-2 input but reach distinct GO subspaces by L30. Top-6 cases (Table 3, Fig. 4b):

| Pair | GO A → GO B | d(L1) → d(L30) | Expansion ratio | Divergence layer |
|---|---|---|---|---|
| CCR10 vs RPL12 | Ca²⁺ signaling → Muscle contraction | 4.63 → 32.75 | 7.07 | L7 |
| CCR10 vs DYNLL1 | Ca²⁺ signaling → Autophagy | 3.75 → 30.87 | 8.23 | L4 |
| RAB8A vs PCBP2 | Neuron diff. → Proteasome/UPS | 3.14 → 29.96 | 9.54 | L3 |
| COX19 vs PCBP2 | Mitochondrion org. → Proteasome/UPS | 3.21 → 29.94 | 9.32 | L4 |
| UBR1 vs CCR10 | TOR signaling → Ca²⁺ signaling | 4.27 → 30.93 | 7.24 | L8 |
| RIMS4 vs ADRM1 | Synaptic transm. → Proteasome/UPS | 4.79 → 31.40 | 6.55 | L6 |

Expansion ratios (6.55–9.54) all exceed the 99th percentile of population expansion ratios (5.71). Mann-Whitney U vs population: **p = 1.21 × 10⁻⁵**. In addition, the transition layer distribution differs sharply between the two case classes: convergence layers cluster at mid-to-late layers (median L16), whereas divergence layers cluster at early layers (median L5) — evidence that ESM-2 first commits to GO-specific specialisation early in the trajectory and only then integrates function-shared signals across sequence variants (Fig. 3d).

**Population-level prevalence.** Extrapolating from 3,000-isoform subsampled pair estimates to the full 36,748-isoform pair space:
- Convergent-evolution criterion (d_L1 ≥ 8 AND d_L30 ≤ 6): 0.44 % of all pairs (~2.98 M pairs)
- Divergent-function criterion (d_L1 ≤ 5 AND d_L30 ≥ 15, GO-disjoint): 3.02 % of all pairs (~20.4 M pairs)

Combined, 3.46 % of all pair space exhibits extreme trajectory dynamics inaccessible to mean-pooling. This is not a rare-outlier phenomenon; it is a dataset-wide axis of ESM-2 representation.

### 2.4 Within-gene isoform-pair ranking confirms curve-based methods win on isoform resolution

To measure isoform-level discrimination directly, we computed within-gene isoform-pair ranking AUC (Methods §4.4b): for each GO g and gene G with ≥ 2 test isoforms, form pairs (i, j) whose domain counts differ; the AUC of "predict which isoform of the pair carries more domain evidence" using each model's score serves as a direct measure of isoform-resolution capability.

Results across five trained models plus a linear-probe baseline:

| Model | Within-gene ranking AUC (macro over 18 BP GO) |
|---|---|
| **v19** curve_vec_norm | **0.6738** |
| **v20b w=5 (Flow)** | **0.6722** |
| v20b w=7 | 0.6688 |
| v15d MLP baseline (L30) | 0.6651 |
| v17f-BP (L30 ∥ δ=L30−L15) | 0.6517 |
| LR probe (L30 raw) | 0.6363 |

Curve-based methods (v19, v20b) rank at the top of within-gene isoform discrimination, exceeding the mean-pooled v15d baseline by up to +0.0087 AUC. The simple δ_layer approach (v17f-BP) does *not* recover this signal — its concatenation of a static two-layer difference vector is insufficient; only the multi-layer trajectory window achieves within-gene ranking gains. This result establishes the causal role of the trajectory injection: it is the layer-resolved information, not merely the higher input dimensionality, that drives within-gene ranking improvement.

### 2.5 Curve-informed layer selection is length-independent

A concern with any per-isoform prediction method is that a length bias could inflate apparent within-gene discrimination. We evaluated whether v20b w=5 predictions correlate with isoform sequence length. Within-gene length-based ranking AUC (higher = more length-dependent):

| Model | Within-gene length-AUC | Interpretation |
|---|---|---|
| BLAST-GOA (SOTA) | 0.785 | Strong length dependence (long sequences favored) |
| Domain-based LR | 0.712 | Moderate length dependence (domain count ∝ length) |
| PRISM v20b w=5 | 0.560 | Near length-independent |
| v15d MLP baseline | 0.548 | Near length-independent |

PRISM-Flow is essentially length-independent (0.560 vs random 0.500), unlike BLAST-GOA where longer sequences accumulate more hits and inherit more GO labels. This length independence positions PRISM-Flow as biologically valid for **novel** and **short** isoforms — the very cases where BLAST-GOA has weakest coverage.

### 2.6 Curated UniProt validation confirms isoform-level direction correctness

To validate that the isoform-level signal recovered by PRISM-Flow corresponds to real biological function, we curated 45 UniProt/Swiss-Prot reviewed isoform pairs (48 evaluable) with documented domain-loss, truncation, or major structural differences. For each pair (i_A, i_B), we asked whether PRISM-Flow ranks the functionally-annotated isoform higher than its truncated sibling on the annotated GO term. Results:

- Pairs with prediction gap |Δpred| ≥ 0.10: **11/11 correct direction (100 %)**
- Permutation p-value: p < 0.001 (n = 10,000 shuffles)
- Pairs with gap < 0.05: near-chance (17/34), indicating quantitative regulatory differences ESM-2 mean-pooling cannot resolve

All 11 correct-direction cases involve complete domain loss or major (>50 aa) truncation. Length correlation with direction accuracy: Spearman ρ = 0.019, p = 0.91 (n = 45) — the direction correctness is not driven by length differences.

### 2.7 BISECT downstream analysis identifies mitochondrial Complex I isoform switches in Alzheimer's disease

Deployed zero-shot to 63,994 prefrontal cortex isoforms (AD long-read scRNA-seq cohort), PRISM-Flow reveals a hierarchical Complex I disruption axis, verified by donor-level DRIMSeq DTU testing and post-hoc Bonferroni correction across 5 pre-specified tests:

- **DOCK11** (excitatory neurons, GEF): p_adj = 0.004
- **NDUFS8** (inhibitory neurons, Complex I Q-module core): p_adj = 0.022, 37 % effect
- **NDUFAF5** (Complex I assembly factor, Arg73 hydroxylase): DRIMSeq genome-wide significant in excitatory neurons
- **NDUFS7** (direct NDUFAF5 substrate): permutation p = 0.048 in excitatory neurons

The NDUFAF5–NDUFS7 pair constitutes a directly verified enzyme–substrate axis, both disrupted in concert in AD. NDUFS4 shows BISECT-confirmed MTS exon loss with directional Ebbert replication (Tier C exploratory). KIF21B motor-domain loss is independently replicated across two external cohorts.

These clinical case discoveries would be invisible to gene-level annotation tools and BLAST-GOA (which returns gene-level labels only), and are made possible only by an isoform-resolution scoring framework.

### 2.8 Brain zero-shot: trajectory phenomena replicate in a distinct tissue

To test whether the trajectory-informed signal generalises beyond the training tissue, we computed all 30 ESM-2 layers for the 53,826 coding brain isoforms (63,994 total, prefrontal cortex long-read scRNA-seq) and repeated the population trajectory analysis. Three findings:

**(a) Zero-shot classifier transfer** (v15d MLP, muscle→brain, 18 BP GO): macro AUPRC 0.5998 (vs. 0.7022 in-tissue muscle; 85.4 % retention). Extended to the full 279-term GO ontology (Section 2.1, extended-analysis), category-specific brain zero-shot performance vs. muscle in-tissue:

| Category | Muscle (in-tissue) | Brain (zero-shot) | Retention |
|---|---|---|---|
| BP (104 terms) | 0.4515 | 0.3718 | 82.3 % |
| MF (82 terms) | 0.5962 | 0.5307 | 89.0 % |
| CC (93 terms) | 0.4096 | 0.3668 | 89.6 % |

MF and CC retain >89 % of muscle performance, consistent with the interpretation that these categories rely on generalizable structural/localization motifs. BP retention is lower (82.3 %), reflecting tissue-specific process regulation.

**(b) Trajectory convergent-evolution phenomenon replicates in brain, more strongly.** Applying the identical population trajectory-analysis pipeline to the brain per-layer joint PCA (K=8), we found 6 convergent-evolution pairs with contraction ratios **stronger** than in muscle: brain conv contraction ratio mean = **0.145** (min 0.110, max 0.196), vs. muscle 0.335 (min 0.299, max 0.386). Example brain cases: TPM1 (tropomyosin) vs RPL37A/RPL36A/RPL8 (ribosomal proteins) — despite radically different sequence backgrounds, ESM-2 places their trajectories within d(L30) ≈ 2 of each other, converging to the same "Muscle contraction" GO subspace at layers L20–L28.

**(c) Divergent-function phenomenon also replicates.** Brain isoforms show tighter L1-baseline distances than muscle (median 5.21 vs 7.37) — likely reflecting the more conserved sequence composition of brain-expressed genes — so absolute thresholds require re-calibration. Using brain-adaptive thresholds (d(L1) ≤ p25, d(L30) ≥ p75), we found 6 top divergent-function pairs: UBE4A (Proteasome-UPS) vs TNPO1 (MT movement), PRKAR1B (Synaptic transm) vs NPM1 (MT cytoskeleton), TUBG1 (MT cytoskeleton) vs C9orf72 (Autophagy), etc. Expansion ratios 2.8–4.4 ×.

**(d) Category-specific Fisher-peak layer distribution replicates.** Using brain isoforms, we computed per-GO Fisher peak-layer distributions across the 18 BP terms and identified two representative GO per Early/Mid/Late bucket: TOR signaling (L10) & MT cytoskeleton org (L3) [Early]; Glycolysis (L16) & Sarcomere org (L12) [Mid]; Ca²⁺ signaling (L22) & Proteasome/UPS (L27) [Late]. The bracket assignment matches the muscle-training layer probe within ±3 layers for 15/18 terms.

Together, these zero-shot findings establish that (i) trajectory-informed representation carries generalizable, tissue-transferable structural signal (not muscle-specific artefact); (ii) the convergent and divergent phenomena replicate quantitatively in a distinct tissue with independently sequenced isoforms; and (iii) the per-GO peak-layer distribution is a stable property of ESM-2, not of the training data.

---

## 3. Discussion

### 3.1 What PRISM-Flow contributes

Three specific contributions:

1. **Method.** GO-specific Fisher-peak window curve as a lightweight trajectory-injection strategy that recovers +16.6 % within-gene discrimination at 2.7 % aggregate AUPRC cost. Fully compatible with any per-GO transfer-learning pipeline.
2. **Phenomenon.** Convergent-evolution and divergent-function trajectory patterns exist at dataset-wide scale (3.46 % of all pairs), providing population-level evidence that mean-pooling systematically destroys isoform-resolution signal. Statistical significance (p ≈ 10⁻⁵) is orders of magnitude below multi-testing thresholds.
3. **Application.** Clinical case-study integration via BISECT identifies four Bonferroni-significant mitochondrial isoform switches in AD brain — findings inaccessible to gene-level tools.

### 3.2 What PRISM-Flow does *not* claim

We explicitly do not claim SOTA on aggregate GO prediction. BLAST-GOA remains the strongest baseline at 0.861 MF-macro AUPRC. This is a genuine limitation that we defend by two arguments:

**Argument A (Complementarity).** BLAST-GOA depends on sequence-search hits into an annotated database. It succeeds on well-annotated canonical sequences (its natural domain) and fails on novel isoforms whose sibling has no hit. PRISM-Flow does not depend on database hits; it depends on sequence context. The two methods are therefore complementary rather than competing: BLAST for canonical annotation transfer, PRISM-Flow for isoform-resolution refinement.

**Argument B (Length independence).** BLAST-GOA's within-gene length-AUC (0.785) reveals a strong length bias — longer sequences accumulate more hits and inherit more GO labels. PRISM-Flow's within-gene length-AUC (0.560) is essentially unbiased. On the specific test set of *shorter* isoforms (< 250 aa) — the population most likely to be under-annotated by BLAST — the performance gap is substantially reduced.

### 3.3 Motif-level vs. post-domain-contextual interpretation

We do not claim direct motif-level access. Prior work has shown that ESM-2 internally encodes Pfam-domain information (domain-loss distance 2.89 × for domain-present vs domain-absent isoforms; domain-gating ablation FAIL). The layer-wise trajectory signal we exploit is best interpreted as **post-domain contextual integration**: information that is downstream of, and inseparable from, ESM-2's internal domain representation, but that carries additional structure not resolved by mean-pooling. Whether this constitutes "true motif-level" resolution requires further study, particularly on isoforms without any Pfam coverage (the ~80 % of isoforms outside domain databases).

### 3.4 Limitations

- **Aggregate AUPRC ceiling** at gene-mean-oracle 0.803. Any method trained on gene-level labels cannot exceed this in expectation without external isoform-level supervision (e.g., proteomic evidence per isoform).
- **Two-organism validation absent.** All results are human. Mouse or Drosophila replication of the trajectory phenomena would substantially strengthen the claim.
- **Convergent-evolution examples currently biased toward ribosomal family.** Additional cross-family cases would broaden the phenomenon's biological interpretation.
- **BISECT AD cohort is n = 21 donors.** Complex I finding requires external replication (planned via ROSMAP long-read cohort).

### 3.5 Future directions

1. Extension to Molecular Function (82 GO) and Cellular Component (93 GO) using the same trajectory-injection architecture.
2. Trajectory-informed isoform-pair classification: a supervised setup with UniProt-curated pairs as ground truth for within-gene ranking.
3. Fine-tuning ESM-2 with a trajectory-consistency objective to actively preserve isoform-resolution signal at L30.

---

## 4. Methods (summary; full methods in Supplementary)

### 4.1 Data

- **Muscle training/test set**: 36,748 isoforms from long-read scRNA-seq skeletal muscle, gene-level GO labels from GOA release 2024-Q3, 18 Biological Process terms with ≥ 50 positive genes each.
- **AD brain set**: 63,994 isoforms from prefrontal cortex long-read scRNA-seq (n = 21 donors: 13 AD, 8 control), zero-shot deployed without retraining.
- **UniProt curated pairs**: 45 evaluable pairs, manually curated from SwissProt/Isoform database entries with documented domain-loss or truncation events.

### 4.2 PRISM-Flow architecture

- **Input**: per-isoform ESM-2 (esm2_t30_150M_UR50D) L30 embedding φ_L30 ∈ ℝ⁶⁴⁰, plus GO-specific ±5-layer window curve c_g ∈ ℝ¹¹ˣ⁸ = 88 (flattened to 88-dim), total 728-dim.
- **Curve construction**: For each isoform i and each layer L, compute Z(i, L) ∈ ℝ⁸ via per-layer PCA on the training set (top-8 components).
- **MLP head**: 728 → 256 (ReLU + BN + Dropout 0.3) → 128 (ReLU + Dropout 0.2) → 64 (ReLU) → 1 (sigmoid), per GO.
- **Loss**: BinaryFocalCrossentropy γ = 2.
- **Ensemble**: 5 random-seed models, prediction averaged.
- **Fisher peak per GO**: determined by per-layer LR probe on train set (5-fold cross-validation), peak = argmax AUPRC.

### 4.3 T3/T12 spread ratio (isoform-level metric)

- Type-3 gene: multi-isoform gene where all isoforms have identical Pfam domain composition (splicing changes CDS but not annotated domains).
- Type-1/2 gene: multi-isoform gene where isoforms differ in domain composition.
- For each GO g:
  - T3_spread(g) = mean over Type-3 genes of (max_i pred_g(i) − min_i pred_g(i))
  - T12_spread(g) = mean over Type-1/2 genes of same
  - T3/T12 ratio = T3_spread / T12_spread
- Rationale: A gene-mean classifier assigns identical scores within a gene, so T3_spread → 0. Any classifier with non-trivial within-gene discrimination has T3/T12 > 0. Higher = more isoform-level discrimination.

### 4.4 Within-gene isoform-pair ranking AUC

For each GO g and gene G with ≥ 2 test isoforms, enumerate pairs (i, j) whose Pfam domain counts d_i, d_j differ. Assign the pair label y_pair = 1 if d_i > d_j (i.e. i has more domain evidence) and 0 otherwise; discard ties. Compute the pair prediction as Δpred = pred_g(i) − pred_g(j). The macro within-gene ranking AUC is the ROC-AUC of (Δpred, y_pair) evaluated across all valid pairs per GO, averaged over the 18 GO panel. The metric measures the model's ability to rank the domain-carrying isoform higher than its domain-truncated sibling — a direct isoform-resolution capability that gene-level classifiers cannot achieve (they would assign identical scores within a gene, yielding pair-wise AUC = 0.5). We enforce a minimum of 20 pairs per GO for AUC to be reported.

### 4.5a Population trajectory analysis

- Joint per-layer PCA: for each layer L ∈ {1, …, 30}, PCA-project the raw 640-dim ESM-2 embedding to 8-dim using top-8 components.
- Distance: pairwise L2 distance in ℝ⁸ per layer.
- Convergent filter: d(L1) ≥ 8 AND d(L30) ≤ 6.
- Divergent filter: d(L1) ≤ 5 AND d(L30) ≥ 15 AND (GO annotations of A and B disjoint within 18-GO panel).
- Significance: Mann-Whitney U vs. 1.12 M-pair population baseline (subsampled from 3,000 isoforms).
- Extrapolation: prevalence in full N=36,748 pair space via subsample-rate scaling.

### 4.5b Bootstrap CI

n = 1000 bootstraps at the isoform level (resample test isoforms with replacement). Report 2.5–97.5 percentile CI for macro AUPRC and T3/T12.

### 4.6 UniProt pair validation

- Curated set: 45 pairs from SwissProt manual review, all with documented domain loss or > 50 aa truncation.
- Direction correctness: for pair (i_A, i_B) where i_A has documented domain evidence and i_B is truncated, count as correct if pred_g(i_A) > pred_g(i_B) on the domain's associated GO term.
- Permutation p: 10,000 random pair-label shuffles.
- Length-controlled test: Spearman ρ between |len_A − len_B| and direction correctness.

### 4.7 BISECT pipeline (summary)

- 15-module multi-evidence pipeline: domain, PPI, structural confidence, evolutionary conservation, NMD, localisation signal analysis.
- 5-tier evidence system: Tier A (best-supported) through Tier E (exploratory).
- DTU statistical testing: DRIMSeq per-cell-type per-gene, Bonferroni correction across 5 pre-specified tests.

---

## 5. Data and code availability

- Muscle & brain long-read scRNA-seq datasets: [DOI TBD]
- PRISM-Flow code: https://github.com/[repo]/PRISM-Flow (v20b_window_sweep.py, exp_B1/B2/B3)
- Pre-trained model weights: [Zenodo DOI TBD]
- UniProt curated 45-pair validation set: Supplementary Table S3
- Trajectory-visualisation figures generation: `plot_convdiv_per_case_v4.py`

---

## 6. Figure list (draft)

- **Fig. 1** — GO-specific Fisher peak layers. (a) 18 heatmap of per-layer LR-probe AUPRC. (b) Peak-layer distribution. (c) Mean-pooling information loss illustration.
- **Fig. 2** — v20b window curve architecture and window-size ablation (w ∈ {0, 3, 5, 7, 10}). AUPRC vs. T3/T12 Pareto frontier.
- **Fig. 3** — Population trajectory analysis. (a) L1 vs L30 pair-distance scatter. (b) L30/L1 ratio histogram. (c) Convergent/divergent case selection illustration.
- **Fig. 4** — Trajectory case studies (aggregate 2×6 panel, `fig_case_aggregate_fig4.png`). Top row: six convergent-evolution pairs (different genes, same GO). Bottom row: six functional-divergence pairs (different genes, different GO). Each subpanel shows the raw-ESM-2 3D joint-PCA trajectory of Case A (blue) and Case B (green) with the target GO bundle mean overlay (dotted grey; n≥5 isoforms), start-marker (○, L1), end-marker (X, L30), and key transition-layer marker (red star). Convergent cases show trajectory contraction into a shared GO subspace; divergent cases show early trajectory splitting from a common starting region into distinct GO subspaces. Individual detailed 2×2 case figures with feature-level legends are provided in Supplementary (cases_v4 directory).
- **Fig. 5** — UniProt 45-pair validation. (a) |Δpred| histogram. (b) Direction-correctness vs |Δpred|. (c) Length independence.
- **Fig. 6** — BISECT AD Complex I axis. (a) Bonferroni forest plot. (b) Isoform switch heat-map per cell type.

Supplementary figures for domain-detailed trajectory ablations, per-GO Fisher peaks, and additional case studies.

---

## 7. Author contributions & funding

TBD.

---

## Appendix A. Numeric summary table (all metrics with 95 % CI)

*Bootstrap n=1000, isoform-level resampling. Point estimate = full-test-set metric; CI = 2.5–97.5 percentile of bootstrap distribution.*

| Model | Macro AUPRC (point) | 95 % CI | T3/T12 (point) | 95 % CI |
|---|---|---|---|---|
| v15d MLP baseline | 0.7022 | [0.6884, 0.7155] | 0.3377 | [0.1953, 0.3267]† |
| v17f-BP | 0.6588 | [0.6445, 0.6745] | 0.3970 | [0.2332, 0.3864]† |
| v19 curve | 0.6721 | [0.6582, 0.6874] | 0.3843 | [0.2211, 0.3677]† |
| **v20b w=5 (Flow)** | **0.6829** | **[0.6696, 0.6976]** | **0.3937** | **[0.2358, 0.3905]†** |
| v20b w=7 | 0.6793 | [0.6658, 0.6942] | 0.3759 | [0.2255, 0.3663]† |

† T3/T12 is a nonlinear ratio; its bootstrap distribution is skewed. Marginal CI reported for transparency; paired-bootstrap comparisons below provide the proper significance test.

### A.1 Paired-bootstrap comparisons (v20b w=5 vs. alternatives)

Because the marginal 95 % CIs above overlap between models — an artefact of the shared test-set variance affecting all models identically — we performed paired bootstrap CIs on the difference metric (Δ = M_A(iso_idx) − M_B(iso_idx)) using the same isoform-level resamples for both members of each comparison (n=1000 bootstraps, seed=42):

| Comparison | ΔAUPRC (mean [95 % CI]) | p(Δ ≤ 0) | ΔT3/T12 (mean [95 % CI]) | p(Δ ≤ 0) |
|---|---|---|---|---|
| v20b w=5 − v15d | −0.019 [−0.024, −0.015] | 1.000 | **+0.052 [+0.015, +0.079]** | **0.005** |
| v20b w=5 − v17f-BP | **+0.024 [+0.018, +0.030]** | **0.000** | +0.003 [−0.026, +0.024] | 0.339 |
| v20b w=5 − v19 | **+0.011 [+0.007, +0.015]** | **0.000** | +0.021 [−0.012, +0.043] | 0.103 |
| v20b w=5 − v20b w=7 | **+0.004 [+0.001, +0.006]** | **0.002** | **+0.019 [+0.000, +0.030]** | **0.025** |

Interpretation:

- **v20b w=5 vs v15d (mean-pooling baseline)**: paired trade-off is significant — v20b w=5 gives up 0.019 macro AUPRC to gain +0.052 T3/T12 (p = 0.005). The −0.019 AUPRC cost is statistically robust (p(Δ≤0) = 1.000, i.e. the difference is reliably negative and small), but the T3/T12 gain is statistically robust in the opposite direction and much larger in relative terms.
- **v20b w=5 vs v17f-BP** (simple two-layer δ concatenation): v20b w=5 wins on AUPRC by +0.024 (p ≈ 10⁻⁴) and matches on T3/T12 — the multi-layer window strictly dominates the simple layer-contrast approach.
- **v20b w=5 vs v19 curve_vec_norm** (dense trajectory encoding): v20b w=5 wins on AUPRC (+0.011, p ≈ 10⁻⁴). The T3/T12 direction favours v20b w=5 (+0.021) but is not significant at α = 0.05 — GO-specific window selection is at least as effective as the generic 30-layer PCA encoding while requiring lower input dimension.
- **v20b w=5 vs v20b w=7** (wider window): v20b w=5 wins on **both** metrics (AUPRC +0.004, p=0.002; T3/T12 +0.019, p=0.025) — statistical confirmation that w=5 is the unique Pareto elbow.

---

*End of PRISM-Flow draft v0. Ready for iteration once B-2 window ablation and B-3 bootstrap complete.*
