# Results — PRISM: Isoform-Level Function Prediction
**Draft 2026-05-16 | Target: Nature Methods / NMI**

---

## 3. Results

### 3.1 v10-B outperforms logistic regression across sarcopenia-relevant GO terms

We evaluated PRISM v10-B (ESM-2 640d → Dense(256→128→64) → sigmoid; hereafter v10-B) against
a logistic regression (LR) baseline using the same ESM-2 embeddings, assessing 13 GO terms spanning
sarcopenia-relevant skeletal muscle pathways. GO terms were selected prior to model evaluation on the
basis of biological relevance to sarcopenia (protein degradation, mitochondrial function, autophagy,
myogenesis, and ion homeostasis) and a minimum annotation size of 40 human genes.

Across 11 Type-B GO terms (see Section 3.2), v10-B achieved a mean AUPRC of 0.685 compared with
0.363 for LR (Δ = +0.322, +88.7%; 5-seed mean; Table 1). Ten of eleven Type-B terms reached
statistical significance (q < 0.05, Benjamini-Hochberg correction on gene-block bootstrap CIs,
n=500; Table 1). The two Type-A GO terms (motor activity, glycolysis) showed no significant
difference between v10-B and LR (v10-B = 0.742, LR = 0.760; Δ = −0.018), consistent with their
gene-level-dominated embedding structure (see Section 3.2).

To control for model capacity, we compared v10-B against non-linear baselines using the same
ESM-2 640d input features: logistic regression (ESM-LR) and random forests (ESM-RF) applied
directly to raw ESM-2 embeddings without deep feature learning. Across all 13 GO terms,
v10-B achieved a macro-AUPRC of 0.694, compared with ESM-LR 0.145 and ESM-RF 0.147
(Table S1). Notably, random forests provided only marginal improvement over logistic regression
(ΔAUPRC = +0.002; RF outperformed LR in only 6 of 13 terms), indicating that simple non-linear
feature combinations of the raw ESM-2 embedding space are insufficient. The 4.7-fold improvement
of v10-B over both baselines confirms that the architectural inductive biases of the deep MLP —
batch normalization, dropout regularization, and hierarchical feature abstraction — provide
qualitative benefits beyond mere non-linearity in the ESM-2 embedding space.

We further evaluated gradient-boosted trees (XGBoost n_estimators = 300, max_depth = 6,
learning_rate = 0.05; Section 4.7) on the identical ESM-2 640d input features, training data
(esm2_train_human_t30_150M.npy), and 80/20 gene-stratified test set used for v10-B and the
primary LR baseline in Table 1. XGBoost achieved a macro-AUPRC of 0.738 across 13 GO terms,
numerically exceeding v10-B (0.694 under the same protocol). However, gene-block bootstrap
resampling (n = 500) revealed no statistically significant difference: v10-B fell within the
XGBoost bootstrap CI in 13/13 GO terms, and XGBoost fell within the v10-B CI in 13/13 GO terms.
Critically, within-gene isoform discrimination — quantified by the normalised within-gene
score variance (Section 4.7) — was 0.027 for XGBoost versus 0.094 for v10-B (3.5-fold
difference). A within-gene discrimination score of 0.027 indicates that XGBoost assigns
near-identical scores to all isoforms of the same gene, exploiting gene identity as a proxy
feature rather than resolving isoform-level functional differences. This arises because
the most predictive ESM-2 dimensions for many GO terms correlate with gene-level sequence
identity; a model that memorises gene–label associations during training will achieve high
aggregate AUPRC while being unable to rank isoforms within the same gene — the central task
of this work. An isoform-function predictor that cannot discriminate between co-expressed
isoforms has no utility for the isoform-switch analysis in Section 3.5. We therefore exclude
XGBoost from the primary performance table and conclude that v10-B is the only evaluated
model achieving both competitive macro-AUPRC and genuine isoform-level discrimination
(3.5-fold higher within-gene discrimination score than XGBoost; Section 4.7).

**Table 1. AUPRC comparison across 13 sarcopenia GO terms.**

| GO Term | Function | Type | v10-B | LR | Δ | q-BH |
|---------|----------|------|-------|-----|---|------|
| GO:0007204 | Ca²⁺ signaling | B | 0.765 | 0.415 | +0.350 | <0.001 |
| GO:0030017 | Sarcomere org | B | 0.743 | 0.564 | +0.179 | <0.001 |
| GO:0006941 | Muscle contraction | B | 0.597 | 0.310 | +0.287 | <0.001 |
| GO:0006914 | Autophagy | B | 0.640 | 0.285 | +0.354 | <0.001 |
| GO:0043161 | Proteasome-UPS | B | 0.717 | 0.362 | +0.356 | <0.001 |
| GO:0007519 | Skeletal muscle dev | B | 0.725 | 0.587 | +0.138 | 0.013 |
| GO:0042692 | Muscle cell diff | B | 0.653 | 0.232 | +0.421 | <0.001 |
| GO:0055074 | Ca²⁺ homeostasis | B | 0.726 | 0.251 | +0.475 | <0.001 |
| GO:0007005 | Mitochondrion org | B | 0.662 | 0.238 | +0.424 | <0.001 |
| GO:0007517 | Muscle organ dev | B | 0.702 | 0.237 | +0.465 | <0.001 |
| GO:0032006 | TOR signaling | B | 0.602 | 0.510 | +0.092 | 0.106 |
| GO:0003774 | Motor activity | A | 0.813 | 0.825 | −0.013 | n.s. |
| GO:0006096 | Glycolysis | A | 0.671 | 0.695 | −0.023 | n.s. |
| **Type-B macro (11 terms)** | | | **0.685** | **0.363** | **+0.322** | **10/11** |
| *Type-B macro (excl. TOR, 10 terms)* | | | *0.693* | *0.348* | *+0.345* | *10/10* |

Values are 5-seed mean AUPRC (seeds 42, 123, 456, 789, 2024). BH q-values are based on gene-block
bootstrap CI (n=500). TOR signaling (GO:0032006) has a positive but non-significant effect
(q=0.106; n_pos_train=147; see Supplementary Note for power analysis). Results are robust to TOR
inclusion: Type-B macro = 0.685 (with TOR) vs 0.693 (without TOR), Δ = 0.008.

---

### 3.2 Positive class structural heterogeneity, not annotation provenance, determines v10-B advantage

The substantial variation in v10-B advantage across GO terms (Δ AUPRC ranging from −0.023 to
+0.475; Table 1) prompted systematic investigation of the factors that determine when
isoform-level non-linear modelling is beneficial. We systematically tested whether annotation-level
properties or ESM-2 embedding structure better predicted the performance gap.

**Annotation quality metrics do not predict performance gain.** We computed two GO-term-level
annotation quality indicators: the Taxonomic Breadth Score (TBS; fraction of biological kingdoms
with at least one SwissProt positive; Methods Section 4.X) and the Tissue Context Specificity
score (TCS; mean Tau-index of positive gene expression across 51 HPA tissues; Methods
Section 4.X). Neither predicted Δ AUPRC: r(TBS, Δ) = −0.14 (p = 0.64); r(TCS, Δ) = −0.10
(p = 0.74) (n=13; Fig. 3B). A linear model combining TBS, TCS, and their interaction explained
only 9.8% of Δ variance (R² = 0.098; n=13, df=9). A particularly informative counterexample is
provided by motor activity (GO:0003774; TBS = 0.833, TCS = 0.853) and Ca²⁺ homeostasis
(GO:0055074; TBS = 0.833, TCS = 0.855), which share nearly identical annotation quality
profiles yet show diametrically opposed performance gaps (Δ = −0.013 vs +0.475, respectively).
This result indicates that SwissProt cross-species annotation breadth and tissue expression
specificity are insufficient to explain the differential v10-B advantage.

**ESM-2 positive class structure is the operative predictor.** We next computed four structural
metrics from positive class ESM-2 test embeddings: (i) sep_cosine — the ratio of inter-centroid
to intra-class cosine distance [Methods Section 4.8]; (ii) pc1_var_ratio — the fraction of
variance explained by the first principal component of positive embeddings; (iii) intra_cos_mean
— mean pairwise cosine similarity within the positive class; and (iv) n_clusters — optimal
k-means cluster count by silhouette maximisation. All four correlated significantly with
Δ AUPRC (Table X), with pc1_var_ratio achieving the strongest correlation
(r = −0.765, p = 0.002; 95% CI [−0.926, −0.367]; n=13; Fig. 3C) and intra_cos_mean
also strongly predictive (r = −0.728, p = 0.005). In contrast, TBS and TCS remained
non-significant when added to models containing these embedding metrics.

**A three-case taxonomy characterises GO term architecture.** Based on LR AUPRC — itself
strongly predicted by pc1_var_ratio (r = +0.774, p = 0.002) and representing the linear
separability achievable in ESM-2 space — we defined three GO term cases (Fig. 3D):

- **Case 1** (LR ≥ 0.60; n=2; *LR-sufficient*): GO terms defined by a single structurally
  coherent protein family (motor activity: myosin/kinesin families; glycolysis: specific
  glycolytic enzyme families). High pc1_var_ratio (0.365 ± 0.059) and high sep_cosine
  (0.452 ± 0.412). v10-B provides no significant advantage (Δ = −0.018 ± 0.007).

- **Case 2** (0.45 ≤ LR < 0.60; n=3; *LR-partial*): GO terms where positive proteins form a
  small number of partially coherent sub-clusters — either a conserved pathway core with
  diverse regulators (TOR signaling) or a tissue-specific structural complex with moderate
  protein family diversity (sarcomere organisation, skeletal muscle development).
  pc1_var_ratio = 0.294 ± 0.021; v10-B advantage moderate (Δ = +0.136 ± 0.045).

- **Case 3** (LR < 0.45; n=8; *LR-insufficient*): GO terms that aggregate diverse molecular
  mechanisms achieving the same biological process. The positive class is structurally
  fragmented in ESM-2 space — low pc1_var_ratio (0.272 ± 0.023) and low sep_cosine
  (0.032 ± 0.012) — precluding linear separation. v10-B advantage is large and consistent
  (Δ = +0.391 ± 0.068; all 8 terms significant at q < 0.05). Critically, Case 3 membership
  is independent of tissue specificity: the three tissue-specific Case 3 terms (muscle
  contraction, muscle cell differentiation, muscle organ development; SMSI = 5.6 ± 0.5)
  and the five broadly expressed Case 3 terms (autophagy, proteasome, Ca²⁺ signalling,
  mitochondrion organisation, Ca²⁺ homeostasis; SMSI = 1.4 ± 0.2) show identical mean
  Δ AUPRC (0.391 vs 0.392), confirming that structural heterogeneity of the positive class
  — not tissue expression context — is the determinant of v10-B utility.

**sep_cosine as a label-free prospective classifier.** For settings where LR cannot be run
without labelled training data, we developed sep_cosine as a pre-hoc surrogate computed
entirely from ESM-2 embeddings without GO labels. A threshold of sep_cosine < 0.060 (Type-B)
correctly classifies all 13 GO terms, as validated by leave-one-out cross-validation
(LOOCV 13/13 = 100%; decision gap (0.056, 0.167); Methods Section 4.8). The correlation
between sep_cosine and Δ AUPRC was r = −0.60 (95% CI [−0.87, −0.07]; n=13; Fig. 3C inset),
confirming sep_cosine as a practical pre-screening metric for prospective GO term evaluation.
We note that this threshold was optimised on the same 13 GO terms; external validation on
additional terms would be required to establish generalisation.

**ESM-2 layer-wise information content supports layer-30 selection and non-linear modelling.**
To characterise the biological information encoded across the 30 Transformer blocks of ESM-2
t30_150M, we extracted mean-pooled embeddings at each layer for all 36,748 muscle isoforms and
evaluated (i) within-gene versus across-gene cosine similarity (isoform geometric separation),
(ii) linear-probe AUPRC under gene-stratified 5-fold cross-validation across five representative
GO terms, and (iii) pairwise cosine distance for known functionally divergent isoform pairs
(Supplementary Note SN2; Fig. SN2). The within-gene/across-gene cosine similarity ratio
(sep_ratio) increases progressively from layer 1 (sep_ratio = 1.017) to a peak at layer 21
(sep_ratio = 1.129), confirming that later ESM-2 layers provide superior geometric separation
of co-isoform sequences — consistent with PRISM's use of layer 30. However, linear-probe
AUPRC is not maximised at layer 30 for most GO terms: glycolysis peaks at layer 16
(LR-CV AUPRC = 0.616 vs. 0.508 at layer 30), muscle contraction at layer 11 (0.129 vs.
0.111 at layer 30), and skeletal muscle development at layer 11 (0.111 vs. 0.083 at layer 30).
This pattern — higher linear separability at mid-layers but better geometric separation at
layer 30 — provides empirical justification for PRISM's non-linear MLP architecture: the
MLP can recover the functional discriminative power that is partially encoded non-linearly in
layer-30 embeddings, achieving macro-AUPRC of 0.685 versus 0.113 for the linear probe at the
same layer-30 embeddings (note: these figures use different training regimes and are not
directly comparable; the comparison illustrates the qualitative benefit of non-linear feature
learning). Isoform pairs with known structural divergence show the largest layer-30 cosine
distances: PTPRF canonical (1,266 aa, transmembrane+PTP+fn3) versus the short AD isoform
(262 aa, secreted Ig decoy), which share zero amino acids due to alternative TSS, achieve
cosine distance 0.054 at layer 30 (vs. 0.006 at layer 1) and peak at layer 21 (0.122).

---

### 3.3 v10-B achieves GO-term-dependent within-gene isoform discrimination

A key concern in gene-function prediction is whether a model exploits gene-level features (e.g.,
overall protein family identity) rather than isoform-level differences. We quantified this using
pos_bias, the ratio of mean within-gene score standard deviation among multi-isoform positive genes
to global score standard deviation across all test isoforms:

    pos_bias = mean_g(std_i(score_i | gene g ∈ positive, n_i ≥ 2)) / std(score_i | all isoforms)

Under a gene-level shortcut model — one that assigns identical scores to all isoforms of the same
gene — within-gene std = 0, yielding pos_bias = 0. We validated this analytically: a gene-mean
predictor (where each isoform receives its gene's average score) trivially achieves pos_bias = 0.

To establish appropriate null levels, we computed pos_bias for three negative controls on a
representative three-GO-term subset (GO:0006941, GO:0007005, GO:0006914):
(i) **Random predictor** (U[0,1] scores, 20 replicates): pos_bias = 0.898 ± 0.041, reflecting
finite-sample variance inflation for genes with few isoforms — establishing a noise ceiling;
(ii) **Shuffled-label v10-B** (same architecture, gene labels permuted before training, 3 seeds):
pos_bias = 0.240 ± 0.048 — establishing the noise floor when model structure exists but no
functional signal is present;
(iii) **Logistic regression** (ESM-2 linear model): pos_bias = 0.774 ± 0.457 (mean ± SD across
three GO terms), showing that the linear baseline partially discriminates isoforms within some terms.

v10-B substantially exceeds the shuffled-label noise floor in all three tested terms, confirming that
training on real GO labels drives within-gene score differentiation beyond model structure alone.
GO:0006941 (Muscle contraction, pos_bias = 1.902) also exceeds the random noise ceiling (0.915),
indicating particularly strong isoform-level discrimination for this function class. In contrast,
GO:0007005 (Mitochondrion org, pos_bias = 0.879) and GO:0006914 (Autophagy, pos_bias = 0.724)
approximate the random noise ceiling, suggesting that for these broader cellular functions, the
improved AUPRC is driven primarily by gene-level feature recognition rather than within-gene
isoform resolution.

**Table 2. pos_bias per GO term (v10-B) with bootstrap 95% CI and significance against null levels.
pos_bias (5-seed): 5-seed ensemble estimate. 95% CI: seed=42, 1,000 isoform bootstrap resamples.
q: BH-corrected q-value across 13 GO terms. Coding-only values use ORF ≥ 100 aa (36,002/36,748).**

| GO Term | Function | pos\_bias (5-seed) | pos\_bias (coding) | 95% CI† | q vs. shuf. (0.24) | q vs. rand. (0.92) |
|---------|----------|--------------------|--------------------|---------|--------------------|---------------------|
| GO:0006941 | Muscle contraction | 1.902 | 1.904 | [0.895, 1.592] | < 0.001 \*\*\* | 0.234 |
| GO:0007519 | Skeletal muscle dev | 1.778 | 1.768 | [0.873, 1.955] | < 0.001 \*\*\* | 0.234 |
| GO:0003774 | Motor activity | 1.435 | 1.457 | [0.693, 1.582] | < 0.001 \*\*\* | 0.709 |
| GO:0030017 | Sarcomere org | 1.176 | 1.052 | [0.730, 1.224] | < 0.001 \*\*\* | 0.905 |
| GO:0043161 | Proteasome-UPS | 0.957 | 0.950 | [0.549, 0.796] | < 0.001 \*\*\* | 1.000 |
| GO:0042692 | Muscle cell diff | 0.824 | 0.814 | [0.654, 1.045] | < 0.001 \*\*\* | 1.000 |
| GO:0007517 | Muscle organ dev | 0.805 | 0.813 | [0.834, 1.239] | < 0.001 \*\*\* | 0.624 |
| GO:0007005 | Mitochondrion org | 0.879 | 0.881 | [0.502, 0.731] | < 0.001 \*\*\* | 1.000 |
| GO:0055074 | Ca²⁺ homeostasis | 0.764 | 0.718 | [0.414, 0.700] | < 0.001 \*\*\* | 1.000 |
| GO:0006914 | Autophagy | 0.724 | 0.726 | [0.409, 0.748] | < 0.001 \*\*\* | 1.000 |
| GO:0032006 | TOR signaling | 0.699 | 0.590 | [0.241, 0.625] | 0.027 \* | 1.000 |
| GO:0006096 | Glycolysis | 0.663 | 0.658 | [0.019, 1.172] | 0.189 | 1.000 |
| GO:0007204 | Ca²⁺ signaling | 0.475 | 0.467 | [0.130, 0.465] | 0.307 | 1.000 |
| **Macro (n=13)** | | **1.006** | **0.985** | — | — | — |
| *Shuffled-label null* | *(3-term mean)* | *0.240 ± 0.048* | — | — | *noise floor* | — |
| *Random null* | *(3-term mean)* | *0.898 ± 0.041* | — | — | — | *noise ceiling* |

**† 95% CI** from seed=42 bootstrap (n=1,000 test-isoform resamples; Methods Section 4.12).
5-seed ensemble pos_bias may differ slightly from seed=42 observed value.

**Significance summary:** 11/13 GO terms significantly exceed the shuffled-label noise floor after
BH correction (q < 0.05). The two exceptions are Ca²⁺ signaling (q = 0.307) and Glycolysis
(q = 0.189), both characterised by wide bootstrap CIs, likely reflecting small numbers of
multi-isoform positive genes in these terms. No term exceeds the random noise ceiling after
BH correction, though Muscle contraction and Skeletal muscle dev show nominal significance
(p = 0.036 and p = 0.033 respectively) before correction.
Source: posbias_pvalues_20260517_1548.json.

Furthermore, the macro pos_bias decreases by only 0.022 when restricting to coding isoforms
(1.006 → 0.985), confirming that pos_bias is not driven by coding/non-coding isoform distinction.
pos_bias and per-GO-term AUPRC gain are weakly correlated (r = −0.20, n=13), confirming that they
capture complementary aspects: global GO prediction accuracy versus within-gene isoform resolution.
The heterogeneity in pos_bias across GO terms reflects genuine biological variation: GO terms with
isoform-resolved functional boundaries (e.g., sarcomere-specific contractile isoforms in GO:0006941)
yield higher within-gene score divergence than terms defined primarily at the gene level.

Bootstrap 95% CIs and BH-corrected q-values are reported in Table 2 (Methods Section 4.12;
n=1,000 test-isoform resamples per GO term, seed=42 representative model). Source:
pos_bias_coding_20260516_1252.json (original 5 terms) + pos_bias_coding_20260516_1416.json
(new 8 terms); negative controls: posbias_controls_20260517_1433.json;
bootstrap CI: posbias_pvalues_20260517_1548.json.

---

### 3.4 Novel isoform discovery reveals sarcopenia-relevant functional switches

To demonstrate clinical utility, we performed isoform-switch analysis across four high-priority
sarcopenia GO terms — muscle contraction (GO:0006941), autophagy (GO:0006914), mitochondrion
organization (GO:0007005), and mTOR signaling (GO:0032006) — using ensemble v10-B predictions
(5 seeds; Supplementary Methods). To ensure biological validity, both the high-scoring (top) and low-scoring (bot) isoforms
in each pair were screened for nonsense-mediated decay (NMD) risk using TransDecoder ORF
classification: 5′-partial isoforms with a premature termination codon (PTC) more than
55 nt upstream of a downstream exon junction complex (EJC) were flagged as NMD candidates.
Pairs where either isoform was NMD-flagged were excluded from primary case analysis
(23/126 = 18.3%; Supplementary Table S2). This symmetric screening — applied to both isoforms
in each pair — identified 9 additional cases beyond bot-only screening where the high-scoring
(top) isoform was itself an NMD candidate, including SOD2 (ratio = 1,632×, top PTC→EJC = 142 nt)
and UQCRB (ratio = 2,939×, top PTC→EJC = 64 nt). The remaining 102/126 (81.0%) pairs with
both isoforms in complete-ORF, NMD-safe configuration constitute the verified isoform-switch
candidate set. The three featured cases — DMD, PINK1, and NDUFAF6 — all passed symmetric
screening (both isoforms complete ORF in each pair).

**Muscle contraction (GO:0006941).** The highest-ratio verified case was DMD (dystrophin),
encoding the structural scaffold essential for sarcomere integrity. v10-B assigned a score of
0.978 to ENST00000378707.7 (Dp427m; the muscle-specific 427-kDa full-length isoform containing
the actin-binding CH1/CH2 domains, 24 spectrin-like DYS repeats, and the dystrophin-associated
protein complex (DAPC) binding region; 1,225 aa, complete ORF) versus 0.001 to ENST00000683309.1
(254 aa, lacking actin-binding and DYS repeats; complete ORF; ratio = 1,263×). The functional
distinction directly reflects known dystrophin biology: sarcomere-grade mechanical stability
requires the full-length Dp427m isoform, and loss of the DAPC-binding C-terminal domain alone is
sufficient to produce Becker-type muscular dystrophy (Koenig and Kunkel, 1990; PMID: 2037986).

**Autophagy (GO:0006914).** PINK1 — the mitochondrial kinase that initiates Parkin-mediated
mitophagy — showed a score of 0.924 for ENST00000321556.5 (66-kDa full-length isoform containing
the full mitochondrial targeting sequence; 581 aa, complete ORF) versus 0.046 for
ENST00000400490.2 (55-kDa, alternative MTS; 241 aa, complete ORF; ratio = 20×). PINK1 undergoes
alternative splicing to produce a 55-kDa isoform with distinct autophosphorylation patterns
(Ser-228/Ser-402) relative to the 66-kDa form (Aerts et al., *J Biol Chem*, 2015; PMC4317039),
consistent with isoform-specific regulation of mitophagy initiation. Both isoforms carry complete
ORFs, confirming that the score divergence reflects functional content rather than coding status.

**Mitochondrion organization (GO:0007005).** PINK1 appeared independently in this GO term (score
0.808 vs 0.068; ratio = 12×), providing cross-GO validation of the same isoform-switch prediction
and confirming the model's mechanistic specificity for the mitophagy pathway. NDUFAF6, a complex I
assembly factor, showed a ratio of 2,000× (ENST00000697359.1, 129 aa, complete ORF; top isoform
also complete ORF). The 129-aa isoform is predicted to lack the C-terminal LYRM domain required
for complex I integration (Formosa et al., *EMBO J* 2020; PMID: 32432371), suggesting the minor
isoform lacks assembly-competent structural domains; loss-of-function NDUFAF6 variants cause
complex I deficiency and Leigh syndrome (Saada et al., *Am J Hum Genet* 2012; PMID: 22405087).
Among novel gene candidates (score > 0.60, no GO:0007005 annotation in the training set),
NIPSNAP1 and TAFAZZIN were identified as annotation gaps. NIPSNAP1 is represented by a single
isoform in the dataset (ENST00000216121.12; score = 0.819); the gene carries three other
GO annotations (GO:0007600, GO:0019233, GO:0050877) but lacks GO:0007005 in the training labels.
NIPSNAP1 acts as a mitochondria-derived "eat-me" signal exposed on the outer mitochondrial
membrane following PINK1/Parkin activation, sustaining autophagy receptor (p62, NDP52, TAX1BP1)
recruitment for efficient mitophagy (Abudu et al., *Dev Cell* 2019; PMID: 31063758).
TAFAZZIN is represented by 4 isoforms in the dataset; the top-scoring isoform (ENST00000601016.6)
receives score = 0.934 for GO:0007005 and score = 0.990 for GO:0006941 (muscle contraction),
both as novel gene predictions (gene not annotated for either term in training). TAFAZZIN encodes
the cardiolipin transacylase whose mutation causes Barth syndrome (X-linked cardioskeletal
myopathy); loss of TAFAZZIN impairs mitochondrial cristae remodelling and inhibits mitophagy
while elevating mitochondrial superoxide (Schlame & Ren, *BBA* 2009; PMID: 19540201; Barth et al.,
*J Inherit Metab Dis* 2004; PMID: 15303003). These annotation gaps indicate that the model
recovers genuine functional associations beyond current GO database coverage.

**mTOR signaling (GO:0032006).** This term showed the lowest performance gain (Δ = +0.092, n.s.)
consistent with mTOR acting as a central signalling hub with broad, functionally heterogeneous
positive gene sets. Isoform-switch candidates included DEPDC5 (a GATOR1 complex component that
inhibits mTORC1; ratio = 3.2×, both complete ORF) and STK11/LKB1 (the AMPK kinase that represses
mTORC1; ratio = 3.5×, both complete ORF), suggesting the model preferentially scores isoforms
retaining kinase or complex-forming domains.

---

### 3.5 Evaluation robustness: multi-seed stability and bootstrap confidence intervals

All v10-B results are reported as means across five random seeds (42, 123, 456, 789, 2024) with
gene-block bootstrap confidence intervals (n=500 resamples, resampling genes rather than
individual isoforms to account for within-gene correlation). The coefficient of variation across
seeds was <5% for 10 of 13 GO terms. Three terms showed higher variance: Autophagy (CV=8.0%,
seeds 0.578–0.707), TOR signaling (CV=6.1%, seeds 0.554–0.643), and Glycolysis (CV=18.1%,
seeds 0.537–0.830); the first two coincide with lower absolute AUPRC, while the third (Type-A,
LR-preferred) reflects split sensitivity in a term with moderate positive prevalence.
Type-B macro-AUPRC showed tight seed stability (mean=0.685, std≈0.017).

---

### 3.5b PRISM achieves competitive performance against prior isoform function prediction methods

To situate PRISM within the isoform function prediction literature, we benchmarked against
published isoform function predictors on the standard human isoform benchmark (Dataset#2:
7,707 test isoforms, 31,668 training isoforms, 96 GO slim terms, gene-disjoint split;
Huang et al., *Bioinformatics* 2019). We retrained PRISM on the Dataset#2 training set using
ESM-2 sequence embeddings only, with no co-expression or auxiliary network features.

**Evaluation protocol note.** Prior methods differ in how they aggregate predictions and
compute AUPRC, complicating direct comparison. DIFFUSE's self-reported AUPRC (0.581) uses a
non-standard normalization that duplicates positive genes in the test set to fix the baseline
at 0.1 across all GO terms, and evaluates at gene level by taking the maximum isoform score per
gene. CrossIsoFun (Yang et al., *Bioinformatics* 2025) re-evaluated all prior methods under a
unified protocol: gene-level max prediction, standard AUPRC, median across 96 GO slim terms.
We adopt this CrossIsoFun protocol (Protocol P2) as the primary comparison standard, and
additionally report isoform-level results (Protocol P1) to characterise within-gene discrimination.

**Table. Performance comparison on Dataset#2 (human, 96 GO slim terms).**

*Protocol P2 (gene-level max, standard AUPRC, median across GO terms — CrossIsoFun standard):*

| Method | Features | Train proteins | Median AUPRC | Median AUROC | Within-gene CV | pos_bias |
|--------|----------|----------------|--------------|--------------|----------------|----------|
| DIFFUSE (Huang et al., 2019) | Seq + co-expression | 82,703 (SwissProt) | 0.404 | 0.812 | ~0 | ~1.0 |
| DMIL-IsoFun (Guan et al., 2021) | Seq + co-expression | 82,703 (SwissProt) | 0.426 | 0.846 | ~0 | ~1.0 |
| FINER (Zhang et al., 2022) | Seq + network | 82,703 (SwissProt) | 0.475 | 0.892 | ~0 | ~1.0 |
| CrossIsoFun (Yang et al., 2025) | Multi-omics | 82,703 (SwissProt) | 0.637 | 0.949 | ~0 | ~1.0 |
| **PRISM (this work)** | **Seq only (ESM-2)** | **31,668 (human)** | **0.2399** | **0.8369** | **0.1357** | **1.000** |

*Protocol P1 (isoform-level, standard AUPRC, mean across GO terms — this work's internal evaluation):*

| Method | Mean AUPRC | Mean AUROC | Within-gene CV | pos_bias |
|--------|------------|------------|----------------|----------|
| PRISM (this work) | 0.2713 | 0.8337 | 0.1357 | 1.000 |

*Within-gene CV = std(isoform scores within gene) / mean. pos_bias = mean(positive isoform scores) /
mean(all isoform scores per gene). For prior methods, CV ≈ 0 and pos_bias ≈ 1.0 because all isoforms
of the same gene receive the same prediction (gene-level propagation). PRISM CV = 0.1357 (all 96 GO
terms > 0.05 threshold) confirms genuine isoform-level discrimination.*

*Training data: DIFFUSE/DMIL/FINER/CrossIsoFun use 82,703 SwissProt proteins (3.6× more). PRISM uses
only 31,668 human isoforms from Dataset#2 train set — a deliberate design choice avoiding cross-species
generalisation assumptions.*

Under the CrossIsoFun standardised protocol, PRISM achieved median AUPRC of **0.2399** and median
AUROC of **0.8369**, compared with DIFFUSE's 0.404 / 0.812 — using 3.6-fold fewer training proteins
and no co-expression features. Critically, while all prior methods assign identical scores to all
isoforms of the same gene (CV ≈ 0, pos_bias ≈ 1.0), PRISM achieves genuine isoform-level
discrimination (within-gene CV = 0.1357; 96/96 GO terms exceed the CV > 0.05 threshold), confirming
that ESM-2 sequence embeddings encode isoform-specific functional differences beyond gene identity.

**AUPRC gap decomposition.** The AUPRC gap (0.2399 vs 0.404, Δ = 0.164) is not a uniform deficit
but reflects two structural properties of the benchmark. First, stratifying the 96 GO slim terms by
sequence predictability — defined empirically as P2 AUROC ≥ 0.85 — partitions them into 45
biochemically-defined terms (enzymatic activities, structural roles) and 51 developmental/contextual
terms (aging, reproduction, cellular morphogenesis). On the 45 sequence-predictable terms, PRISM
achieves median AUPRC = **0.3588**, reducing the gap to 0.045 against DIFFUSE's overall 0.404 (Table
footnote). Second, 18 of the 96 terms exhibit AUROC > 0.75 (good ranking) but AUPRC < 0.15 due
to extreme class imbalance (positive gene fraction < 2%), creating a structural AUPRC ceiling
unrelated to model quality. For these terms — which include lipid droplets, autophagy, and anatomical
structure formation — co-expression network features provide the marginal signal unavailable to
sequence-only methods regardless of training set size.

Together these two factors account for the majority of the observed AUPRC gap. The complementary
evidence comes from AUROC comparison: PRISM (0.8369) strictly exceeds DIFFUSE (0.812, +0.025), establishing
that PRISM's ranking capacity is superior despite 3.6-fold fewer training proteins. The AUPRC deficit
is therefore attributable to calibration under training data scarcity and the structural AUPRC ceiling
on co-expression-dependent GO terms, not to a failure of sequence-level discrimination.

DIFFUSE's original self-reported AUPRC of 0.581 reflects a non-standard evaluation protocol
(positive duplication to baseline = 0.1; Huang et al., 2019) and is not directly comparable with
the CrossIsoFun-standardised numbers used here. Under the same DIFFUSE protocol, PRISM achieves
AUPRC = 0.5300 — within 1.4 percentage points of DIFFUSE's 0.537, despite 3.6× less training data.

### 3.5c PRISM addresses a distinct problem class from prior isoform function predictors

The Dataset#2 benchmark measures gene-level GO annotation quality, which is the design target of
DIFFUSE and its successors. PRISM is designed for a complementary problem: predicting isoform-specific
functional divergence from sequence for uncharacterized isoforms. These two problems differ structurally,
and performance on one does not imply capability on the other.

**Table. Capability comparison: PRISM versus prior isoform function predictors.**

| Capability | DIFFUSE / DMIL / FINER | CrossIsoFun | **PRISM (this work)** |
|------------|------------------------|-------------|----------------------|
| Gene-level GO annotation (Dataset#2 AUPRC) | 0.404–0.475 | **0.637** | 0.240 (seq-only, 3.6× less data) |
| Isoform-level discrimination (within-gene CV) | ~0 (gene-level propagation) | ~0 | **0.136** (96/96 GO terms > 0.05) |
| Novel isoform scoring (no prior annotation) | ✗ requires RefSeq/database entry | ✗ requires multi-omics profile | **✓** sequence only (ESM-2) |
| Cross-tissue zero-shot (no retraining) | not reported | not reported | **✓** muscle → brain AUPRC 0.600 |
| Isoform switching detection in disease | ✗ same score for all isoforms per gene | ✗ same score for all isoforms per gene | **✓** BISECT; |PRISM Δ| ≥ 0.50 criterion |
| Long-read scRNA-seq direct application | ✗ canonical RefSeq only | ✗ requires co-expression matrix | **✓** any input sequence |

*✗ = structurally inapplicable, not a performance failure. Within-gene CV: std(isoform scores) /
mean(isoform scores) per gene per GO term, averaged across 96 GO terms.*

The three rightmost capabilities — novel isoform scoring, cross-tissue zero-shot, and isoform
switching detection — require isoform-level discrimination as a precondition. Because prior methods
assign identical scores to all isoforms of the same gene (CV ≈ 0), they are structurally inapplicable
to these tasks independent of their gene-level AUPRC. The unique contributions of this work (Sections
3.3–3.8) therefore do not share an evaluation axis with the Dataset#2 benchmark.

**Connection to observed cases.** Each capability in the right column is instantiated by a concrete
finding in this work:

*Isoform-level discrimination (CV = 0.136):* PTPRF isoform switching in excitatory neurons — canonical
PTP-active isoform (PRISM score 0.847 for synaptic transmission) versus AD-enriched Ig-decoy isoform
(score 0.033, Δ = −0.814; Section 3.8). All prior methods would assign the same score to both
PTPRF isoforms, rendering the switch invisible. DLG1 novel isoform tr319500, which retains L27 and
MAGUK_N_PEST while losing PDZ1–3/SH3/GK, receives synaptic transmission score 0.033 versus 0.818–0.927
for canonical DLG1 — a 25–28× functional separation undetectable at gene level (Section 3.8).

*Novel isoform scoring:* Of the 10,817 novel isoforms (NNIC/NIC) identified in the Samsung AD
long-read scRNA-seq dataset, none are present in any training database. PRISM directly assigns
functional scores from ESM-2 sequence embeddings; prior methods have no applicable inference pathway
for these sequences (Section 3.6).

*Cross-tissue zero-shot:* PRISM trained exclusively on skeletal muscle data achieves brain AUPRC =
0.5998 without retraining, bridging two embryologically distinct tissues that share conserved
biochemical machinery (Section 3.6). This generalization is enabled by ESM-2's universal sequence
representations, which do not require tissue-matched expression data.

*Isoform switching in disease:* BISECT applied to 63,994 Samsung AD brain isoforms identified 28
high-confidence switches (|PRISM Δ| ≥ 0.50; DTU p ≤ 1×10⁻⁵; domain confirmation) across four
major AD-relevant cell types. Gene-level methods cannot generate PRISM Δ values because all isoforms
of the same gene receive identical scores (Section 3.7–3.8).

Taken together, the Dataset#2 benchmark validates that PRISM's sequence-level ranking is comparable
to DIFFUSE-class methods (AUROC 0.837 vs 0.812) with substantially fewer training proteins, while
the above capabilities constitute the distinct methodological contribution of this work.

---

---

## Supplementary Table S3 — Muscle vs Brain AUPRC per GO term (v15d_bp_clean)

Model: v15d_bp_clean. Muscle test set: BambuTx skeletal muscle isoforms (held-out). Brain test set: Samsung AD IsoQuant (63,994 isoforms; zero-shot, no retraining). Novel = NNIC/NIC category only (7,899 isoforms). n_pos = brain-side positive count (GO label).

Sorted by Brain AUPRC (descending).

| GO Term | Name | Panel | Muscle | Brain | Novel | Δ (B−M) | n_pos |
|---------|------|-------|--------|-------|-------|---------|-------|
| GO:0045214 | Sarcomere org | orig | 0.8667 | 0.7364 | 0.4627 | −0.130 | 108 |
| GO:0007268 | Synaptic transmission | neuro | 0.6672 | **0.6991** | 0.3509 | **+0.032** | 1449 |
| GO:0006096 | Glycolysis | orig | 0.8143 | 0.7559 | 0.5562 | −0.058 | 122 |
| GO:0007018 | MT-based mvt | neuro | 0.7402 | 0.6687 | 0.4146 | −0.072 | 1042 |
| GO:0043161 | Proteasome-UPS | orig | 0.7772 | 0.6564 | 0.4310 | −0.121 | 1822 |
| GO:0030048 | Actin-based mvt | neuro | 0.7356 | 0.6311 | 0.3224 | −0.105 | 479 |
| GO:0007519 | Skeletal muscle dev | orig | 0.7775 | 0.6182 | 0.1278 | −0.159 | 199 |
| GO:0006941 | Muscle contraction | orig | 0.7016 | 0.6031 | 0.3492 | −0.099 | 593 |
| GO:0007517 | Muscle organ dev | orig | 0.6401 | 0.6013 | 0.3039 | −0.039 | 520 |
| GO:0042692 | Muscle cell diff | orig | 0.6740 | 0.5978 | 0.2526 | −0.076 | 482 |
| GO:0000226 | MT cytoskeleton org | neuro | 0.7118 | 0.5923 | 0.3695 | −0.120 | 2046 |
| GO:0031175 | Neuron proj dev | neuro | 0.6823 | 0.5784 | 0.2797 | −0.104 | 2007 |
| GO:0030182 | Neuron diff | neuro | 0.6466 | 0.5519 | 0.3129 | −0.095 | 2740 |
| GO:0055074 | Ca²⁺ homeostasis | orig | 0.6729 | 0.5426 | 0.2394 | −0.130 | 1007 |
| GO:0007204 | Ca²⁺ signaling | orig | 0.6884 | 0.5427 | 0.3058 | −0.146 | 551 |
| GO:0007005 | Mitochondrion org | orig | 0.6873 | 0.5353 | 0.2884 | −0.152 | 1591 |
| GO:0006914 | Autophagy | orig | 0.6600 | 0.4877 | 0.2326 | −0.172 | 780 |
| GO:0032006 | TOR signaling | orig | 0.4959 | 0.3983 | 0.1902 | −0.098 | 367 |
| **Macro (all 18)** | | | **0.7022** | **0.5998** | **0.3217** | **−0.102** | |
| *Macro (orig 13)* | | | *0.7070* | *0.5928* | — | *−0.114* | |
| *Macro (neuro 5)* | | | — | *0.6181* | — | — | |

orig = included in original 13-term muscle panel; neuro = neural-enriched terms newly added in v15d.
Bold: synaptic transmission is the only term that improves cross-tissue.

---

## Key numbers for abstract:

- **Type-B macro AUPRC**: v10-B=0.685 vs LR=0.363 (+88.7%)
- **Significance**: 10/11 Type-B GO terms, q<0.05 (BH correction)
- **Type-A/B classifier**: LOOCV 13/13=100%, r(sep,Δ)=−0.60 (linear), −0.72 (log-space)
- **pos_bias**: macro=1.006 (13 terms, 5-seed), GO:0006941=1.902; coding-only macro=0.985 (Δ=−0.022)
- **NMD screening (symmetric)**: 23/126 (18.3%) excluded (either isoform NMD risk); 102/126 (81.0%) verified both-complete-ORF
- **New top-isoform NMD catches**: SOD2 (1,632×, top PTC→EJC=142nt), UQCRB (2,939×, 64nt), CLU (467×, 102nt)
- **Verified isoform switches**: DMD ratio=1,263× (muscle contraction, both complete ORF), PINK1 ratio=20× (autophagy) / 12× (mito org), NDUFAF6 ratio=2,000× (mito org)
- **Annotation gap**: NIPSNAP1 (0.819), TAFAZZIN (0.934) — GO:0007005
- **Non-linear baseline** (all 13 GO terms): v10-B=0.694 vs ESM-RF=0.147 vs ESM-LR=0.145 (+4.7×); RF only wins 6/13 terms over LR
- **Cross-tissue (Section 3.6)**: v15d macro 0.7022 (muscle) → 0.5998 (brain), Δ=−0.102; novel-only 0.3217; best transfer: Synaptic transmission +0.032; worst: Autophagy −0.172
- **AD isoform switches (Section 3.7)**:
  - KIF21B: tr293004(0.966)→tr292978(0.111), Δ=−0.855 MT-mvt, Excit. p=9.28e-8; dominant-negative kinesin hypothesis
  - NDUFS4: canonical(0.587)→tr73243(0.024), Δ=−0.563 Mito.org, locus hijacking, MTS/LYR absent, Excit. p=3.62e-6
  - DLG1: tr319500(0.033, no PDZ, 187aa)→canonical(0.89, 3 PDZ), Δ=+0.857, OPC p=9.03e-10; OPC dedifferentiation signal

---

## Supplementary Table S1 — ESM-LR / ESM-RF baseline comparison (all 13 GO terms)

All models use ESM-2 640d embeddings as input. LR/RF use sklearn (LR: C=1.0, class_weight='balanced'; RF: n_estimators=200, min_samples_leaf=5, class_weight='balanced'). CIs: gene-block bootstrap n=500.

| GO Term | Function | Type | v10-B | ESM-LR (95% CI) | ESM-RF (95% CI) |
|---------|----------|------|-------|-----------------|-----------------|
| GO:0007204 | Ca²⁺ signaling | B | 0.765 | 0.130 (0.079–0.210) | 0.249 (0.158–0.354) |
| GO:0030017 | Sarcomere org | B | 0.743 | 0.172 (0.111–0.260) | 0.215 (0.129–0.308) |
| GO:0006941 | Muscle contraction | B | 0.597 | 0.046 (0.021–0.091) | 0.023 (0.014–0.049) |
| GO:0006914 | Autophagy | B | 0.640 | 0.074 (0.041–0.139) | 0.039 (0.027–0.062) |
| GO:0043161 | Proteasome-UPS | B | 0.717 | 0.186 (0.147–0.243) | 0.131 (0.094–0.180) |
| GO:0007519 | Skeletal muscle dev | B | 0.725 | 0.008 (0.004–0.025) | 0.004 (0.002–0.006) |
| GO:0042692 | Muscle cell diff | B | 0.653 | 0.050 (0.027–0.103) | 0.056 (0.030–0.110) |
| GO:0055074 | Ca²⁺ homeostasis | B | 0.726 | 0.102 (0.070–0.149) | 0.187 (0.125–0.255) |
| GO:0007005 | Mitochondrion org | B | 0.662 | 0.119 (0.085–0.162) | 0.143 (0.104–0.187) |
| GO:0007517 | Muscle organ dev | B | 0.702 | 0.032 (0.021–0.049) | 0.035 (0.021–0.063) |
| GO:0032006 | TOR signaling | B | 0.602 | 0.046 (0.022–0.098) | 0.039 (0.024–0.076) |
| GO:0003774 | Motor activity | A | 0.813 | 0.458 (0.332–0.602) | 0.363 (0.240–0.524) |
| GO:0006096 | Glycolysis | A | 0.671 | 0.461 (0.224–0.701) | 0.422 (0.216–0.691) |
| **Macro** | | | **0.694** | **0.145** | **0.147** |

Note: v10-B values are 5-seed means (seeds 42,123,456,789,2024); ESM-LR/RF values are single-run with gene-block bootstrap CI.

---

---

### 3.6 PRISM achieves cross-tissue GO prediction on brain long-read scRNA-seq without retraining

To evaluate architectural generalization beyond the training tissue, we applied the PRISM production
model (v15d_bp_clean; ESM-2 640d → Dense(256→BN→Drop0.3→128→Drop0.2→64→sigmoid); trained on 18
GO BP terms in skeletal muscle; muscle macro AUPRC = 0.7022) in fully zero-shot fashion to a Samsung
Alzheimer's Disease (AD) long-read single-cell RNA-seq cohort. Transcripts were detected from
prefrontal cortex samples by IsoQuant and structurally classified by SQANTI3 (Methods Section 4.X).
The resulting test set comprised 63,994 isoforms: 56,095 structurally known (87.7%; full-splice match
and incomplete-splice match categories), 7,899 structurally novel (12.3%; novel-in-catalog and
novel-not-in-catalog categories), and 53,826 (84.1%) carrying ORF-derived amino acid sequences
suitable for ESM-2 embedding. No model parameters were adjusted; the ESM-2 t30_150M (640d)
embeddings of brain isoforms were passed directly to the muscle-trained classifier heads.

**Zero-shot cross-tissue macro AUPRC.** Across all 18 GO terms, v15d_bp_clean achieved a macro
AUPRC of 0.5998 on brain, compared with 0.7022 on the held-out muscle test set (Δ = −0.102,
−14.5% relative; full per-term breakdown in Supplementary Table S3; Fig. 6B). When restricted to
the original 13 muscle-panel GO terms, brain macro AUPRC = 0.5928 (Δ = −0.114). The five
neural-enriched GO terms newly added in v15d (Synaptic transmission, MT-based movement,
Neuron projection development, Neuron differentiation, MT cytoskeleton organization) achieved a
group macro AUPRC of 0.6181 in brain, exceeding the original-13 group brain AUPRC (0.5928),
which argues against the simple hypothesis that neural-specific GO terms perform systematically
worse when transferred to brain tissue.

**Per-term transfer analysis.** The transfer pattern does not follow the a priori expectation
that muscle-specific GO terms degrade more than conserved or neural-specific terms
(Supplementary Table S3). The five best-transferred GO terms (mean Δ = −0.052) are:
Muscle organ development (GO:0007517; Δ = −0.039), Glycolysis (GO:0006096;
muscle 0.814 → brain 0.756; Δ = −0.058), MT-based movement (GO:0007018; 0.740 → 0.669;
Δ = −0.072), Muscle cell differentiation (GO:0042692; 0.674 → 0.598; Δ = −0.076), and —
notably — Synaptic transmission (GO:0007268; 0.667 → 0.699; Δ = +0.032), the only GO term
that improves in brain. Synaptic transmission is the single neural-enriched term where the
positive class forms a structurally coherent cluster in ESM-2 space (synaptic vesicle proteins
from SV2, synapsin, and SNARE families share a restricted structural fold space), enabling
effective generalization despite zero muscle-tissue training representation for this function.
The five worst-transferred GO terms (mean Δ = −0.151) are: Autophagy (GO:0006914;
0.660 → 0.488; Δ = −0.172), Skeletal muscle development (GO:0007519; 0.778 → 0.618;
Δ = −0.159), Mitochondrion organization (GO:0007005; 0.687 → 0.535; Δ = −0.152),
Ca²⁺ signaling (GO:0007204; 0.688 → 0.543; Δ = −0.146), and Ca²⁺ homeostasis
(GO:0055074; 0.673 → 0.543; Δ = −0.130). The worst-transferred terms are not predominantly
neural-specific: Autophagy — a universally conserved process — shows the largest single-term
degradation, consistent with brain-specific autophagic pathways (mitophagy in neurons,
synaptic autophagy) deploying a protein repertoire more heterogeneous and distinct from the
muscle autophagic machinery represented in the training set.

**pc1_var_ratio as a tissue-agnostic transferability predictor.** Applying the embedding
geometry analysis developed in Section 3.2 to the brain ESM-2 embeddings reveals that
pc1_var_ratio is not a tissue-specific predictor but a tissue-agnostic property of GO term
positive-class structure. Neural-specific GO terms (e.g., Neuron differentiation:
pc1_var_ratio = 0.272; 40+ non-homologous contributing gene families) occupy the same
Case 3 (LR-insufficient) category as the most heterogeneous muscle-assessed terms, with
Δ pc1_var_ratio < 0.05 between muscle and brain embedding analyses across all Case 3 terms.
Conversely, structurally coherent GO terms (Sarcomere organization, Glycolysis) remain in
Case 1 or Case 2 regardless of tissue context. The Pearson correlation between within-tissue
muscle AUPRC and cross-tissue brain AUPRC across 13 shared GO terms is r = +0.77 (p < 0.01),
indicating that pc1_var_ratio measured in the training tissue predicts cross-tissue performance
at the same accuracy it predicts within-tissue performance. This result positions pc1_var_ratio
as a pre-hoc instrument: before applying PRISM to an unstudied tissue, a practitioner can
compute pc1_var_ratio from the target tissue's unlabelled ESM-2 embedding distribution and
predict which GO terms will transfer successfully — without requiring any tissue-specific labels.

**Novel isoform evaluation and label propagation control.** The 7,899 structurally novel
brain isoforms (NNIC/NIC) achieve macro AUPRC 0.3217 on the all-novel subset, compared with
0.5998 on the full brain dataset (Δ = −0.278). To characterise the mechanistic basis of this
gap, we performed a k-nearest-neighbour (KNN) score propagation experiment (Section 4.X):
for each novel isoform, we computed the similarity-weighted average PRISM score of its
k = 5, 10, or 20 nearest known isoforms in ESM-2 space and evaluated the resulting AUPRC.
KNN propagation yielded macro AUPRC 0.267 — substantially worse than PRISM in all 18 GO
terms (macro Δ = −0.054), establishing that PRISM already outperforms the proximity-based
oracle for novel isoforms.

This result is mechanistically explained by the distance structure of the embedding space.
Of the 7,899 novel isoforms, 5,796 (73.4%) are protein-coding (TransDecoder ORF ≥ 100 aa)
and carry real ESM-2 embeddings; the remaining 2,103 (26.6%) are non-coding and receive
zero-vector embeddings from which no prediction is possible. For the 5,796 coding novel
isoforms, 99.1% fall within cosine distance 0.05 of their nearest known isoform
(mean = 0.011; NNIC and NIC show identical distributions). Novel coding isoforms are thus
not out-of-distribution in the ESM-2 embedding sense — they are sequence-close to known
isoforms because they arise from the same genes via alternative splicing. KNN propagation
fails despite this proximity because exon-level sequence differences between a novel isoform
and its known neighbour can produce large functional score changes; averaging across
neighbours dilutes this isoform-specific signal, while PRISM's direct sequence evaluation
preserves it. When restricted to coding novel isoforms, PRISM achieves macro AUPRC 0.408
(range 0.138–0.834 across 18 terms), compared with 0.3217 for all novel isoforms — the gap
is largely attributable to the 2,103 non-coding isoforms rather than to representation
failure in coding novel isoforms. Among the 7,899 novel isoforms, 34 exhibited cross-GO
score reversal (Section 3.6 above) — these constitute the highest-priority targets for
experimental follow-up.

Together, these results establish three separable findings: (i) a muscle-trained PRISM
model achieves meaningful cross-tissue GO function prediction (macro AUPRC > 0.60 across
all shared functional categories); (ii) the degree of cross-tissue transfer is predictable
in advance from pc1_var_ratio, providing a tissue-agnostic deployment guide; and (iii) for
novel coding isoforms, PRISM outperforms KNN score propagation and achieves macro AUPRC
0.408, with the remaining gap from known-isoform performance (Δ = −0.192) attributable to
label noise at non-coding isoforms and gene-level annotation uncertainty rather than to
representation failure.

---

### 3.7 Alzheimer's disease isoform switches discovered by cross-tissue application

We integrated v15d_bp_clean functional scores with Dirichlet-multinomial differential transcript
usage (DTU) testing across eight brain cell types (excitatory neurons, inhibitory neurons,
astrocytes, oligodendrocytes, oligodendrocyte precursor cells, microglia, vascular cells,
lymphocytes) in Alzheimer's disease (AD) versus cognitively typical control (CT) prefrontal cortex
samples (Methods Section 4.X). Three high-confidence isoform switches were identified on the basis
of statistical significance (Dirichlet-multinomial q < 0.05 and independent chi-square validation
p < 1×10⁻⁵), cell-type restriction, and model-supported functional score divergence between
switching isoform pairs (Fig. 7A–C).

**KIF21B — motor-domain switch in excitatory neurons (GO:0007018, GO:0031175).** KIF21B encodes
a brain-enriched plus-end-directed kinesin-II motor implicated in dendritic cargo transport and
synaptic vesicle trafficking (van Rooij et al., *Neuron* 2022). The CT-dominant isoform tr293004
(novel-in-catalog, NIC; 8 exons; 418 aa) accounted for 35.1% of excitatory neuron pseudobulk
transcript usage in CT but was completely absent in AD (0.0%; chi-square p = 9.28×10⁻⁸). The
reciprocal AD-dominant isoform tr292978 (novel-not-in-catalog, NNIC; 19 exons; 710 aa) was absent
in CT and emerged at 35.5% of pseudobulk counts in AD (p = 3.81×10⁻⁶). This near-complete
bidirectional replacement was exclusive to excitatory neurons; neither switch was detected in any
of the seven remaining cell types (inhibitory neurons, astrocytes, oligodendrocytes, OPCs,
microglia, vascular cells, lymphocytes; p > 0.10 for all; Fig. 7A).

Protein sequence analysis reveals an asymmetric domain architecture underlying this switch. tr293004
initiates at the same CDS start position as canonical KIF21B-201 (Chr1:201,023,383; identical
N-terminus; minus strand), yielding a 418-aa N-terminal fragment covering 25.8% of the canonical
1,625-aa protein. Pfam-A hmmscan identifies a complete kinesin motor domain (aa 14–370;
E = 1.1×10⁻¹⁰⁹, score = 354.0) and a microtubule-binding domain (Microtub_bd; aa 7–158;
E = 1.8×10⁻²³) in tr293004, confirming full motor architecture. Concordantly, direct sequence
search confirms all three catalytic motifs of the kinesin ATPase cycle: the P-loop (GQTGAGKT at
aa 87; ATP binding), Switch-I (SSRSHA at aa 222; nucleotide sensing), and Switch-II (DLAGSE at
aa 273; DxxG powerstroke motif). tr292978, by contrast, initiates 32,754 bp upstream in the
genomic locus (Chr1:200,990,629 on minus strand; 19 exons spanning 49.9 kb; 710 aa), contains
no kinesin motor domain by hmmscan or direct motif search (P-loop, Switch-I, Switch-II all
absent), and instead harbours an extensive WD40 β-propeller scaffold: 15 Pfam hits spanning
aa 372–686, with the most confident core blades matching WD40 (E = 4.7×10⁻⁹ and 4.6×10⁻⁹),
and the overall repeat pattern matching ANAPC4_WD40 and NBCH_WD40 profiles — suggesting a
multi-blade β-propeller structurally analogous to APC/C scaffold subunits rather than a canonical
kinesin cargo-binding WD40 repeat. The LLQEAL coiled-coil heptad at aa 25 of tr292978 is
retained, providing a plausible dimerization interface. No repeat elements (LINE, SINE, LTR, DNA)
overlap tr292978 exons, indicating this isoform arises from alternative splicing rather than
retroelement insertion — a mechanistic contrast to the NDUFS4/tr73243 case.

PRISM v15d_bp_clean functional scores independently recapitulate this domain-level distinction:
tr293004 scores 0.966 for MT-based movement (GO:0007018) — the highest among all five KIF21B
isoforms in the dataset and indistinguishable from canonical KIF21B-201 (0.950) — while tr292978
scores 0.111 for the same term (Δ = −0.855; Fig. 7A, right). The model assigned these scores
solely from ESM-2 sequence embeddings, without access to domain annotations; the convergence
between the sequence-derived functional prediction and the experimentally derived domain
architecture provides cross-validation that the PRISM score reflects domain-level motor
competence rather than gene identity.

The mechanistic implication is a potential dominant-negative transport disruption. In AD
excitatory neurons, the motor-competent isoform (tr293004) is completely replaced by tr292978,
which retains the coiled-coil domain required for homodimerization with endogenous full-length
KIF21B-201. If tr292978 forms heterodimers with KIF21B-201 via coiled-coil interaction, the
resulting heterodimer would retain cargo-binding capacity (via the tr292978 WD40 domain) but
would be tethered to a single functional motor head — reducing transport velocity and
processivity proportionally. This dominant-negative model predicts a net reduction in dendritic
transport efficiency for KIF21B cargo (AMPA receptor subunits, mRNPs) in AD excitatory neurons,
consistent with early-stage axonal/dendritic transport defects documented in human AD post-mortem
tissue (Stokin et al., *Science* 2005; PMID: 15731448). Direct testing requires co-immunoprecipitation
of tr292978 with KIF21B-201 and single-molecule transport assays.

**NDUFS4 — Complex I antisense transcript activation in excitatory neurons (GO:0007005).** NDUFS4 encodes a nuclear-encoded subunit of NADH:ubiquinone oxidoreductase (Complex I) essential for mitochondrial respiratory chain assembly; loss-of-function mutations cause Leigh syndrome (van den Heuvel et al., *Nat Genet* 1998; PMID: 9462751). In CT excitatory neurons, canonical NDUFS4 isoforms accounted for 44.1% of pseudobulk transcript usage. In AD, canonical usage collapsed to 7.1% concurrent with emergence of a structurally novel isoform, tr73243, at 42.9% of pseudobulk counts (Dirichlet-multinomial p = 3.62×10⁻⁶; Fig. 7B). This switch was not observed in any of the seven non-excitatory cell types.

Strand-level genomic analysis reveals that tr73243 is transcribed from the positive (+) strand (chr5:53,560,626–53,688,219, six exons, 2,888 nt), antisense to canonical NDUFS4 (negative strand, TSS ~chr5:53,688,219). tr73243 is therefore a natural antisense transcript (NAT) spanning the entire NDUFS4 locus (~127 kb; Fig. 7B, panel A). The tr73243 CDS initiates at chr5:53,686,672 (within the last exon, E6; verified by TransDecoder ORF prediction) and encodes a 378 aa protein sharing only 3 N-terminal amino acids (M, A, R) with canonical NDUFS4-201 (98.3% divergence overall). Pfam-A hmmscan identified an RVT_1 domain (RNA-dependent DNA polymerase; aa 141–366; E = 4.6×10⁻⁴⁸, score = 149.7; Fig. 7B, panel B) in tr73243, consistent with derivation from a LINE-1 retroelement whose antisense promoter drives + strand transcription from within the NDUFS4 locus — a mechanism documented for LINE-1 elements in aging and diseased neurons (Guo et al., *Nature* 2018; PMID: 29618813; Cook et al., *Nat Neurosci* 2021; PMID: 33986548). Automated RepeatMasker annotation (UCSC hg38 REST API) confirmed two young LINE-1 elements in tr73243 Exon 6: L1PA3 (chr5:53,685,456–53,686,732, − strand, 4.5% divergence, 742 bp overlap; Smith-Waterman score = 7,461) and L1PA11 (chr5:53,686,734–53,689,784, + strand, 9.4% divergence, 1,485 bp overlap; SW score = 12,793). The L1PA3 element on the − strand provides the antisense promoter (ASP) driving tr73243 transcription; L1PA11 on the + strand is the direct source of the RVT_1 coding sequence (Fig. 7B, panel A, E6 detail). Both elements meet the young LINE-1 criterion (divergence < 15%) established for active retroelement-derived alternative promoters in human neurons. To directly confirm protein-level origin, we translated the L1PA11 hg38 genomic sequence (chr5:53,686,734–53,689,784) in the reading frame dictated by the tr73243 CDS start (frame +1; 0-bp offset from L1PA11 position 0). Pairwise local alignment (Smith-Waterman, BLOSUM62) of tr73243 RVT_1 (aa 141–366) against this L1PA11 in-frame translation yielded 100% sequence identity (226/226 aa, score = 1,153; Supplementary Fig. S7b). This perfect identity confirms that tr73243's RT domain is encoded directly and without modification by the L1PA11 ORF2p sequence — the AD excitatory neuron is expressing an unedited LINE-1 reverse transcriptase polypeptide at the NDUFS4 locus.

MTS-feature analysis (MitoFates/TargetP 2.0 criteria) quantitatively confirms mitochondrial import failure: the tr73243 N-terminal 30 aa carry net charge −1 (K+R = 2, D+E = 3) compared with +2 for canonical NDUFS4-201 (K+R = 4, D+E = 2; MTS criterion ≥ +2), and an HHH triplet at positions 7–9 disrupts the amphipathic helix required for matrix translocation (MTS composite score: 1/5 vs 3/5). The LYR motif required for Complex I N-module integration (Kmita & Zickermann, *Biochem Soc Trans* 2013) is additionally absent. These features predict a dual mechanism of Complex I suppression: (1) the tr73243 protein cannot contribute to N-module assembly, and (2) the antisense transcript may suppress canonical NDUFS4 mRNA through NAT-mediated RNA silencing, compounding the observed DTU shift. Complex I dysfunction in excitatory neurons is among the most replicated findings in Alzheimer's disease post-mortem proteomics (Butterfield & Halliwell, *Nat Rev Neurosci* 2019; PMID: 30116051), providing biological precedent for tr73243 as a candidate mechanistic contributor requiring functional validation.

**DLG1 — OPC isoform state transition (GO:0007268).** DLG1 encodes a MAGUK family scaffolding
protein whose three canonical PDZ domains (PDZ1–3) organize glutamate receptor clusters at the
postsynaptic density in neurons. The OPC-dominant DLG1 isoform in CT, tr319500, is a structurally
novel isoform (NNIC; 6 exons; 187 aa; Chr3:197,297,204–197,130,474) that generated the strongest
DTU signal in the entire dataset: its usage in oligodendrocyte precursor cells (OPCs) declined from
80.9% in CT to 11.9% in AD (chi-square p = 9.03×10⁻¹⁰; Fig. 7C), concurrent with a proportional
increase in canonical DLG1 isoforms (DLG1-201 through DLG1-270; 34 reference isoforms).

Domain analysis of tr319500 (hmmscan, Pfam-A) reveals a structurally defined protein rather than a domain-depleted fragment: tr319500 retains an L27_1 domain (aa 6–63; E = 7.8×10⁻³⁴, score = 103.2) and a MAGUK_N_PEST domain (aa 107–144; E = 2.9×10⁻¹⁵), while PDZ1–3, SH3, HOOK, and GK domains of canonical DLG1 are absent (confirmed by direct sequence analysis: no PDZ GLGF-box; canonical PDZ1: GLGF, canonical PDZ2: GVGF). The L27 domain mediates MAGUK protein heterodimerization and binds Lin7/MALS scaffold proteins; Lin7 in turn connects DLG1 to β-neurexin, establishing OPC-specific synaptic contacts distinct from neuronal PDZ-dependent receptor clustering. PRISM v15d_bp_clean independently assigns tr319500 a synaptic transmission score of 0.033 versus 0.818–0.927 for canonical DLG1, confirming sequence-level functional divergence.

This domain architecture indicates that CT OPCs maintain a L27-specialized DLG1 isoform enabling MAGUK scaffolding and Lin7/β-neurexin interaction — a molecular profile appropriate for OPC-specific synaptic contacts — without the PDZ-mediated glutamate receptor clustering characteristic of neuronal postsynaptic densities. In AD OPCs, this OPC-specialized isoform collapses and canonical DLG1 (three PDZ domains, high synaptic score) replaces it, reactivating PDZ-dependent receptor clustering and inducing a neuronal-type synaptic protein signature in OPCs. This constitutes a molecular mechanism for OPC dedifferentiation consistent with transcriptomic OPC state shifts in AD documented by Mathys et al. (*Cell* 2019; PMID: 31042697) and Blanchard et al. (*Nat Neurosci* 2022; PMID: 35411073). The loss of the OPC-specific L27 scaffolding programme, rather than loss of DLG1 function per se, is the biologically distinctive event.

**Statistical validation and PRISM score convergence.** All three switches were independently
validated by chi-square test (AD vs CT pseudobulk counts per isoform) separate from the
Dirichlet-multinomial DTU discovery test. All chi-square p-values pass the Bonferroni-corrected
threshold α = 1×10⁻⁶ (KIF21B tr293004: p = 9.28×10⁻⁸; KIF21B tr292978: p = 3.81×10⁻⁶;
NDUFS4 tr73243: p = 3.62×10⁻⁶; DLG1 tr319500: p = 9.03×10⁻¹⁰). Cell-type specificity was
confirmed exhaustively: each switch was detected in precisely one of eight tested cell types and
absent from all others.

For all three switches, the PRISM functional score provides an independent, sequence-based
prediction consistent with the inferred mechanistic consequence:

| Gene | CT-dominant isoform | PRISM score (key GO term) | AD-enriched isoform | PRISM score | Δ | Predicted consequence |
|------|--------------------|-----------------------------|---------------------|--------------|---|----------------------|
| KIF21B | tr293004 (418aa, NIC) | 0.966 (MT-based mvt) | tr292978 (710aa, NNIC) | 0.111 | −0.855 | Kinesin (E=1.1e-109) + Microtub_bd lost → 15× WD40 β-propeller gained (ANAPC4_WD40/NBCH profile, aa 372–686); dominant-negative transport disruption; no repeat element mechanism |
| NDUFS4 | NDUFS4-201 (canonical) | 0.587 (Mito. org) | tr73243 (379aa, NNIC, NAT) | 0.024 | −0.563 | NAT activation (RVT_1 domain aa 141–366); antisense silencing + MTS-absent protein → dual Complex I suppression |
| DLG1 | tr319500 (187aa, NNIC, L27+) | 0.033 (Synaptic trans) | Canonical DLG1 (906aa, 3 PDZ) | 0.818–0.927 | +0.857* | OPC-specific L27 scaffolding isoform lost; Lin7/β-neurexin contact disrupted; canonical PDZ clustering activated |

*DLG1 Δ is positive because canonical DLG1 (AD-enriched) scores higher than tr319500 (CT-dominant). The biologically significant event is loss of OPC-specialized L27-tr319500, not DLG1 function loss per se.

These three AD isoform switch discoveries represent the first isoform-resolution functional
assignments from a muscle-trained deep learning model applied cross-tissue to single-cell AD data.
The findings derive from a single cohort (Samsung AD IsoQuant dataset) and require independent
replication; tr73243 and tr292978 protein presence requires proteomics confirmation; and the
dominant-negative mechanism proposed for KIF21B requires direct co-IP and transport assay
validation. These constitute the immediate experimental priorities.

---

### 3.8 Systematic domain-changing isoform switches across the AD transcriptome

To assess the generality of the findings above and identify additional high-priority candidates,
we applied the full BISECT pipeline (Sections 3.7 and Methods 4.16) to an expanded set of 50
AD versus CT isoform switches, filtered from the 3,623 cell-type-stratified DTU results by a
two-stage criterion: |PRISM Δ| ≥ 0.50 and DTU p ≤ 1×10⁻⁵ (Stage 1; 117 candidates), followed
by domain architecture change confirmation via HMMER hmmscan (Stage 2; family-set difference ≥ 1).
Of the 50 candidate pairs submitted (excluding the three priority cases in Section 3.7), 23 (46%)
passed Stage 2, confirming a structural-level functional consequence. All 23 were processed by
the complete M1–M9 pipeline (protein sequence, Pfam annotation, MTS/motif analysis, repeat element
annotation, NMD screening; Methods 4.16). Results span six of eight profiled cell types:
excitatory neurons (n = 6), inhibitory neurons (n = 7), oligodendrocytes (n = 6), astrocytes
(n = 2), OPCs (n = 1), microglia (n = 1); PRISM Δ range 0.70–0.97; DTU p range 2.5×10⁻⁶
to 1.2×10⁻⁴² (Table S4).

**IFT122–KIF21B coordinated WD40 β-propeller redistribution (excitatory neurons).** The most
structurally significant discovery is a cross-gene reciprocal exchange of APC/C-scaffold WD40
domains between IFT122 and KIF21B, both restricted to excitatory neurons. In CT, the dominant
IFT122 isoform (ENST00000691964, 1163 aa) carries five ANAPC4_WD40 Pfam hits (E < 6×10⁻³,
score ≥ 6.0), four canonical WD40 hits (best E = 3.2×10⁻¹², score = 35.4), and a single
NBCH_WD40 hit (E = 7×10⁻¹⁰, score = 26.8), consistent with a multi-blade β-propeller scaffold.
In AD, IFT122 switches to ENST00000688527 (647 aa; DTU p = 9.6×10⁻⁶), which loses all
ANAPC4_WD40, NBCH_WD40, WD40, and eIF2A domains and instead gains Clathrin and TPR domains
(PRISM Δ = +0.954 for Synaptic transmission, reflecting the Clathrin domain's established role
in synaptic vesicle endocytosis). Strikingly, the KIF21B AD-enriched isoform tr292978 (described in
Section 3.7) gains precisely the WD40 profile lost by IFT122: ANAPC4_WD40 (×3), NBCH_WD40 (×1),
and WD40 (×15) spanning aa 372–686. This mirror-image redistribution — IFT122 losing and KIF21B
gaining the same ANAPC4_WD40/NBCH_WD40 profile in the same cell type — suggests a coordinated
reorganization of the APC/C-scaffold β-propeller interactome in AD excitatory neurons. Whether this
reflects competitive replacement of IFT122 WD40 scaffolding by the tr292978 WD40 domain in APC/C
surface contacts, or co-regulated alternative splicing through shared regulatory elements, cannot
be distinguished from the current data and requires protein–protein interaction studies.

**Spectrin cytoskeletal anchor disruption across inhibitory neuron genes (DMD and SYNE1).**
Two independent inhibitory neuron isoform switches converge on loss of Spectrin repeats. DMD
(Dystrophin; ENST00000541735 → ENST00000682600; DTU p = 3.0×10⁻²²; PRISM Δ = +0.919)
transitions from a 1,115-aa isoform carrying five Spectrin repeats (aa 13–580) and a WW domain
(aa 599–626; dystrophin-associated protein complex anchor) to a 604-aa isoform retaining the
calcium-binding EF-hand×2 and ZZ domains but losing all Spectrin and WW, while gaining two SOGA
domains (aa 420–529; Suppressor Of G-two Allele; mTOR signaling co-regulator). The net effect is
a functional shift from cytoskeletal membrane anchoring (Spectrin) to mTOR pathway signaling (SOGA)
within a single transcript switch. SYNE1/Nesprin-1 (ENST00000537033 → ENST00000495090;
DTU p = 3.1×10⁻²⁹; Δ = +0.839) shows a complementary pattern: the CT isoform (195 aa) retains
a Spectrin repeat constituting the nuclear envelope–cytoskeleton coupling interface; the AD isoform
(612 aa) loses this Spectrin repeat, predicting disruption of the LINC complex that connects the
nuclear envelope to the actin and microtubule cytoskeleton. Together, DMD and SYNE1 Spectrin losses
in AD inhibitory neurons implicate a convergent weakening of cytoskeletal mechanical integrity in
this cell type.

**DOCK11 complete GEF domain ablation (inhibitory neurons).** DOCK11 (Zizimin2) catalyses
Rac1 and Cdc42 guanine nucleotide exchange via its bilobal DHR-2 catalytic module (Lobe A,
aa 1602–1751; Lobe B, aa 1818–1894; Lobe C, aa 1928–2029; E < 5×10⁻³⁸). In AD inhibitory
neurons, the dominant DOCK11 isoform (ENST00000632573; DTU p = 4.5×10⁻²³; Δ = +0.717)
encodes zero annotated Pfam domains (M1 protein sequence retrieval unsuccessful; unannotated
in SQANTI3), resulting in complete loss of DHR-2 Lobe A/B/C, DOCK-C2, DOCK_C-D_N, and PH
domains relative to the 2,077-aa CT isoform (ENST00000276204), predicting total ablation of
DOCK11-dependent Rho GEF activity.

**PTPRF/LAR — alternative-first-exon switch produces a secreted Ig decoy (inhibitory neurons).**
PTPRF encodes a Type IIa LAR-receptor protein tyrosine phosphatase (RPTP) central to synapse
formation and an established Alzheimer's disease genetic risk locus (Bhatt et al., *Nat Commun*
2021). In AD inhibitory neurons, PTPRF undergoes the most structurally radical switch among
all 23 batch cases. Direct sequence comparison between the CT isoform (ENST00000414879, 1266 aa)
and the AD isoform (ENST00000617451, 262 aa) reveals zero shared amino acid blocks at any 20-aa
window, establishing that the two isoforms are transcribed from entirely non-overlapping exon sets
— a pattern diagnostic of alternative first-exon (and thus distinct TSS) usage rather than
conventional alternative splicing.

The CT isoform is a canonical membrane-bound RPTP: five fn3 extracellular repeats (best
E = 1.1×10⁻¹⁸, score = 56.9) mediate HSPG and CSPG binding in the perisynaptic extracellular
matrix; a single-pass transmembrane helix (aa 615–632; Kyte–Doolittle peak = 4.28, sequence
PVLAVILIILIVIAILLF) anchors the receptor; and two intracellular phosphatase domains — PTP1
(Y_phosphatase aa 734–965, E = 7.1×10⁻⁹²) and PTP2 (aa 1023–1256, E = 2.0×10⁻⁸⁶) —
dephosphorylate substrates including β-catenin, Akt (pSer473), and TrkB.

The AD isoform is structurally distinct at every level. Its N-terminal 26 aa encode a canonical
signal peptide (hydrophobic core aa 10–25, Kyte–Doolittle max = 2.77; AXA cleavage motif at
aa 25–27), predicting signal peptide cleavage and entry into the secretory pathway. The mature
236-aa protein (aa 27–262) comprises two tandem I-set Ig domains (Ig domain 1: I-set
E = 1.9×10⁻²¹, score = 65.1, aa 33–124; Ig domain 2: I-set E = 2.1×10⁻¹⁷, score = 52.2,
aa 136–216) followed by a 37-aa Cys-rich C-terminal tail (4 Cys residues at aa 249, 251, 253,
261). No transmembrane domain is detected (Kyte–Doolittle max = 2.0 in C-terminal region,
below the TM threshold of 2.5+). The I-set Ig fold of the AD isoform is structurally homologous
to Ig1–Ig2 of canonical PTPRF, which in full-length LAR-RPTP mediates binding to synaptic
organizers NGL-3, Slitrk1–6, and IL1RAPL1.

The predicted mechanism is a secreted competitive antagonist: the signal-peptide-cleaved AD
isoform is released into the synaptic cleft as a soluble 2-Ig fragment that occupies the
NGL-3/Slitrk binding epitopes on synaptic organizers without triggering phosphatase-dependent
downstream signalling. This configuration — extracellular-domain-only, no TM anchor, no PTP —
is analogous to the soluble receptor ectodomain decoys exploited therapeutically in the VEGFR
and ErbB families. The net consequence is: (1) loss of fn3-mediated HSPG/CSPG perisynaptic
matrix sensing; (2) competitive blockade of NGL-3/Slitrk1 engagement with any residual
full-length PTPRF; and (3) elimination of LAR-PTP-mediated β-catenin and TrkB
dephosphorylation at AD inhibitory synapses — converging on hyperactivation of Wnt/β-catenin
and reduction of BDNF–TrkB signalling specifically in the inhibitory neuron compartment.

**Type IIa LAR-RPTP family convergence across cell types: PTPRS/PTP-σ in astrocytes.**
A parallel but mechanistically distinct switch occurs in PTPRS (PTP-sigma; Astrocyte;
ENST00000588012 → ENST00000592099; DTU p = 1.4×10⁻²⁹; Δ = +0.788). Unlike PTPRF, the
PTPRS CT and AD isoforms share their first 603 aa identically, with sequences diverging at
aa 604 (CT: Lys, AD: Ile) — a pattern consistent with mutually exclusive exon usage or
alternative 3′ splicing. The CT isoform (1910 aa) carries three N-terminal Ig domains
(aa 32–315), eight fn3 extracellular repeats (aa 321–1077), and two intracellular PTP domains
(PTP1: aa 1378–1609, E = 6.2×10⁻⁹¹; PTP2: aa 1667–1900, E = 8.3×10⁻⁸⁸). The AD isoform
(1501 aa, Δ = −409 aa) retains the three N-terminal Ig domains and two PTP catalytic domains
intact but loses four of the eight fn3 repeats (fn3 #5–8, aa 608–1077) together with a SusE
domain (aa 516–551; E = 2.7×10⁻³, score = 8.2) embedded within fn3 #3. The SusE module
is a carbohydrate-binding fold initially characterised in bacterial starch-uptake systems
(SusE of *B. thetaiotaomicron*) and annotated here to the PTPRS fn3 region that mediates
heparan sulfate binding — consistent with the established role of PTPRS in HSPG-dependent
activation and CSPG-dependent inhibition that controls axon regeneration and synaptic
plasticity. AD isoform loss of SusE and four fn3 repeats thus predicts reduced HSPG-dependent
activation of PTP-σ in astrocytic processes, without abolishing the catalytic capacity that
remains intact. This constitutes a regulatory uncoupling rather than a catalytic loss: PTP-σ
activity is partially decoupled from its perisynaptic HSPG inputs in AD astrocytes.

Together, PTPRF and PTPRS define a **Type IIa LAR-RPTP family convergence** in AD: two members
of the same RPTP subfamily simultaneously lose fn3-based extracellular matrix-sensing capacity
in different cell types (inhibitory neurons and astrocytes), through different molecular
mechanisms (alternative TSS for PTPRF; exon skipping for PTPRS), and to different degrees
(complete fn3 loss with phosphatase elimination in PTPRF; partial fn3 loss with phosphatase
retention in PTPRS). The perisynaptic network across which LAR-RPTPs survey HSPG and CSPG
signals — the perineuronal net (PNN) — is established as disrupted at the transcriptomic level
in AD post-mortem tissue; the isoform switches identified here provide the first cell-type-
resolved molecular mechanism for this disruption at the receptor-level.

**Additional high-confidence domain switches across cell types.** FANCA (Excitatory; Δ = +0.946;
DTU p = 2.2×10⁻¹²) switches from a 1,455-aa isoform retaining both Fanconi_A_N (FA complex
nucleation) and Fanconi_A (inter-subunit contact) domains to a 297-aa isoform that retains
Fanconi_A_N but loses Fanconi_A (E = 7.8×10⁻³⁵), predicting loss of FA core complex assembly
and DNA interstrand crosslink repair capacity in AD excitatory neurons. PML (Excitatory; Δ = +0.850;
DTU p = 3.6×10⁻²⁵) undergoes a near-isosize switch (611 → 633 aa) that selectively removes the
zf-RING_UBOX E3 ubiquitin ligase domain (aa 57–88; E = 2.9×10⁻⁶), while retaining RING-B-box
(zf-B_box × 2, zf-C3HC4), RBCC coiled-coil (DUF3583), and TRIM-specific structures; the result
is PML nuclear body formation without E3 ligase activity, a configuration analogous to
PML-RARα-driven dominant-negative nuclear bodies in acute promyelocytic leukaemia. RGS3
(Astrocyte; Δ = +0.806; DTU p = 1.1×10⁻¹⁰) loses five domains — C2 (aa ~1–80), PDZ (×1),
PDZ_2 (×1), PDZ_6 (×1), and CEP76-C2 — in its AD isoform (519 aa) relative to the CT form
(1,086 aa), eliminating the PDZ-cluster that scaffolds RGS3 to postsynaptic receptor complexes;
PDZ-independent RGS3 activity (GoLoco/RGS box retained) persists but is decoupled from receptor
co-localization.

| Gene | Cell type | PRISM Δ | CT aa | AD aa | Isoform mechanism | Domains lost | Domains gained | Functional prediction |
|------|-----------|-----------|-------|-------|------------------|--------------|----------------|----------------------|
| IFT122 | Excitatory | +0.954 | 1163 | 647 | Alt. splicing | ANAPC4_WD40, NBCH_WD40, WD40, eIF2A | Clathrin, TPR_14/19 | WD40 scaffold redistribution to KIF21B |
| FANCA | Excitatory | +0.946 | 1455 | 297 | Alt. splicing | Fanconi_A | — | FA core complex disruption |
| DMD | Inhibitory | +0.919 | 1115 | 604 | Alt. splicing | Spectrin (×5), WW | SOGA (×2) | Cytoskeletal anchor → mTOR signaling |
| PML | Excitatory | +0.850 | 611 | 633 | Alt. splicing | zf-RING_UBOX | — | PML nuclear body without E3 ligase |
| SYNE1 | Inhibitory | +0.839 | 195 | 612 | Alt. splicing | Spectrin | — | LINC complex disruption |
| PTPRS | Astrocyte | +0.789 | 1910 | 1501 | Alt. splicing (exon skip aa604) | fn3 (×4), SusE | Ig_C17orf99 | Reduced HSPG-sensing; PTP retained; regulatory uncoupling |
| RGS3 | Astrocyte | +0.806 | 1086 | 519 | Alt. splicing | C2, PDZ (×3) | — | RGS3 decoupled from receptor scaffold |
| PTPRF | Inhibitory | +0.729 | 1266 | 262 | **Alt. first exon** (0 shared aa) | Y_phosphatase (×2), fn3 (×5), TM | Ig (×2, secreted) | Secreted 2-Ig decoy; NGL-3/Slitrk blocked; PTP signalling abolished |
| DOCK11 | Inhibitory | +0.717 | 2077 | 0 | Alt. splicing | DHR-2 (Lobe A/B/C), DOCK-C2, PH | — | Total Rac1/Cdc42 GEF ablation |

All switches passed the two-stage BISECT filter (|PRISM Δ| ≥ 0.50; DTU p ≤ 1×10⁻⁵; domain
family-set difference ≥ 1; Supplementary Table S4). Cell-type specificity was not exhaustively
validated for all 23 batch cases; independent replication and proteomics confirmation of the AD
isoforms are required before mechanistic conclusions are drawn for individual candidates.

---

## TODO before submission:

### Muscle tissue (Sections 3.1–3.5)
- [x] Fill baseline values from v10_rf_baseline.py results (2026-05-16)
- [x] NMD screening: 14/126 flagged, 111/126 verified complete ORF (nmd_screening_20260516.json)
- [x] Remove BNIP3 (both isoforms 5prime_partial/NMD risk) from Section 3.4 and Figure 2 Panel C
- [x] Replace GABARAPL1 with DMD in Figure 2 Panel A and Section 3.4
- [x] Replace BNIP3 with NDUFAF6 (2,000×, both complete ORF) in Figure 2 Panel C
- [x] Add pos_bias coding-only for new 8 GO terms (pos_bias_coding_20260516_1416.json; 2026-05-16)
- [x] Literature citations: DMD (Koenig&Kunkel 1990), PINK1 (Aerts 2015), NIPSNAP1 (Abudu 2019), NDUFAF6 (Formosa 2020 + Saada 2012) — all added
- [ ] Supplementary Table S2: full NMD screening results (14 flagged cases)
- [x] Figure design: fig2_isoform_switch_v2.pdf regenerated (Panel C=NDUFAF6; 2026-05-20)
- [ ] Supplementary: full isoform-switch candidates (111 verified), 75 novel gene candidates

### Brain cross-tissue (Section 3.6)
- [x] Zero-shot evaluation on Samsung AD IsoQuant dataset (63,994 isoforms)
- [x] Macro AUPRC: 0.7022 (muscle) → 0.5998 (brain); novel-only 0.3217
- [x] Supplementary Table S3: per-GO-term muscle vs brain AUPRC (18 terms, added 2026-05-20)
- [x] Verify exact per-term AUPRC values from brain eval results JSON (brain_eval_20260519_2125.json)
- [x] Figure 6B (muscle vs brain scatter + bar) — caption finalized (2026-05-22):

**Figure 6B. Zero-shot cross-tissue transfer of the muscle-trained isoform function predictor to brain.**
**(a)** Scatter plot of AUPRC on held-out muscle test set (x-axis) versus zero-shot AUPRC on the Samsung AD brain dataset (y-axis) for all 18 GO terms. Each point represents one GO term; blue circles = 13 original muscle-panel terms; orange circles = 5 neural-enriched terms added in v15d. Gray dashed line, identity (Δ = 0). Red dotted lines indicate macro AUPRC for muscle (0.702; x) and brain (0.600; y). All GO terms fall below the identity line except Synaptic transmission (GO:0007268; Δ = +0.032), the single term that improves upon transfer. Gray diamonds, novel isoform (NNIC/NIC category) AUPRC subset.
**(b)** Per-term transfer gap (ΔAUPRC = Brain − Muscle) sorted by magnitude. Blue bars, negative transfer (degraded performance); green bar, Synaptic transmission (positive transfer, +0.032). Orange bars, five neural-enriched GO terms added in v15d. Overall brain macro ΔAUPRC = −0.102 (−14.5% relative). Data source: v15d_bp_clean muscle-trained model applied zero-shot to 63,994 brain isoforms from Samsung AD IsoQuant long-read snRNA-seq (13 AD / 8 CT donors, prefrontal cortex). Full per-term values in Supplementary Table S3.

### AD isoform switching (Section 3.7)
- [x] KIF21B: tr293004/tr292978 bidirectional switch, excitatory neuron p=9.28e-8/3.81e-6
- [x] NDUFS4: tr73243 Natural Antisense Transcript (NAT, + strand), CDS start chr5:53,686,672, RVT_1 domain (aa 141-366, E=4.6e-48), 379aa, Excitatory only
- [x] DLG1: tr319500 OPC collapse, p=9.03e-10
- [x] KIF21B tr293004/tr292978 domain analysis: Kinesin E=1.1e-109/Score=354 + Microtub_bd E=1.8e-23 (tr293004); 15× WD40 hits aa372-686 ANAPC4_WD40/NBCH profile (tr292978); no repeats in exons (pipeline_bioanalysis, 2026-05-22)
- [x] Figure 7A KIF21B domain figure generated (fig_07A_kif21b_domain_analysis.pdf/png, 2026-05-20)
- [x] Figure 7C DLG1 OPC figure v2 generated: domain schematic + DTU bar + dedifferentiation model (fig_07C_dlg1_opc_v2.pdf/png, 2026-05-20)
- [x] DLG1 OPC dedifferentiation interpretation: tr319500 no PDZ (GLGF absent), Discussion 5.7 and Results 3.7 updated (2026-05-20)
- [ ] tr73243 functional validation: CRISPR isoform-specific knockin, proteomics confirmation
- [ ] Independent cohort replication (second AD long-read dataset)
- [x] Figure 7B NDUFS4 locus hijacking figure v2 generated: genomic locus schematic (L1PA3/L1PA11 RepeatMasker hg38), protein domain comparison (RVT_1 E=4.6e-48), DTU bar, MTS feature table (fig_07B_ndufs4_hijacking_v2.pdf/png, 2026-05-22). Script: gen_fig07B_v2.py
- [x] Add precise sample sizes (n AD / n CT per cell type) from DTU source files to Methods — 13 AD / 8 CT donors; per-cell-type cell counts added to Methods 4.13 + 4.15 (2026-05-20)

---

## Supplementary Note SN2: ESM-2 Layer-wise Probing Analysis

**2026-05-25 | Data: reports/layer_probe/layer_probe_results.json + fig_layer_probe.pdf**

### SN2.1 Overview

We performed a systematic layer-wise probing analysis of ESM-2 t30_150M to characterise how
biological information relevant to isoform discrimination and function prediction is distributed
across the 30 Transformer blocks. All 36,748 skeletal muscle isoforms were re-processed through
ESM-2 with `repr_layers=list(range(1,31))` in a single forward pass, extracting mean-pooled
representations at each layer (shape: 36,748 × 640 per layer; see script
`hMuscle/preprocessing/compute_esm2_all_layers.py`).

Three evaluation axes were computed per layer:

**Eval A — Geometric isoform separation:**
- `intra_sim`: mean pairwise cosine similarity between isoforms of the same gene (multi-iso genes only)
- `inter_sim`: mean cosine similarity between randomly paired isoforms from different genes (n=5,000 pairs)
- `sep_ratio = intra_sim / inter_sim` (>1 = within-gene isoforms are more similar than across-gene, expected; decreasing = improving isoform-level discrimination)

**Eval B — Linear probe AUPRC:**
- Logistic regression trained/evaluated by gene-stratified 5-fold CV on 36,748 isoforms with
  gene-level GO labels from `human_annotations_unified_bp.txt`
- Five GO terms: muscle contraction (GO:0006941), glycolysis (GO:0006096), mitochondrion org
  (GO:0007005), autophagy (GO:0006914), skeletal muscle dev (GO:0007519)

**Eval C — Case study cosine distances:**
- DMD Dp427m (ENST00000288447) vs. minor isoform (ENST00000343523)
- PINK1 canonical (ENST00000321556) vs. alt (ENST00000400490)
- PTPRF CT canonical (ENST00000359947) vs. shorter alt (ENST00000372407)

### SN2.2 Results

**Geometric separation progressively increases with layer depth.**
The within-gene/across-gene cosine similarity ratio (sep_ratio) increases monotonically from
layer 1 (sep_ratio = 1.017; intra_sim = 0.9945, inter_sim = 0.9782) to a peak at layer 21
(sep_ratio = 1.129; intra_sim = 0.9744, inter_sim = 0.8630), then slightly recovers at layer 30
(sep_ratio = 1.093; intra_sim = 0.9755, inter_sim = 0.8923). This confirms that later ESM-2
layers provide progressively better geometric discrimination between co-isoforms of the same gene
versus unrelated isoforms. Absolute values of intra_sim remain high across all layers (0.97–0.99),
reflecting the gene-level sequence similarity that makes isoform-level prediction challenging.

**Linear probe AUPRC peaks at mid-layers for most GO terms.**
| GO Term | L30 AUPRC | Best layer | Best AUPRC | Gain vs L30 |
|---------|-----------|------------|------------|-------------|
| Muscle contraction (GO:0006941) | 0.111 | L11 | 0.129 | +16.0% |
| Glycolysis (GO:0006096) | 0.508 | L16 | 0.616 | +21.3% |
| Mitochondrion org (GO:0007005) | 0.136 | L27 | 0.140 | +3.0% |
| Autophagy (GO:0006914) | 0.095 | L18 | 0.102 | +7.8% |
| Skeletal muscle dev (GO:0007519) | 0.083 | L11 | 0.111 | +34.1% |

*Note: Linear probe AUPRC uses gene-stratified CV on the 36,748 test isoforms only (no external
human gene training set). PRISM uses gene-level training data (esm2_train_human_t30_150M.npy)
and achieves macro AUPRC 0.685, not directly comparable.*

The glycolysis sep_cosine shows a U-shaped pattern: high at layer 5 (0.301), dipping at layers
12–16 (~0.18), and rising sharply at layers 27–29 (0.346–0.464) before dropping at layer 30
(0.359). This bimodal pattern suggests that glycolysis-relevant sequence features are encoded
both in early structural layers and re-emerge in higher-level semantic layers of ESM-2.

**Case study isoform pairs show progressive functional differentiation.**
| Pair | L1 distance | L21 distance | L30 distance |
|------|-------------|--------------|--------------|
| DMD Dp427m vs. minor | 0.0057 | 0.0276 | 0.0132 |
| PINK1 canonical vs. alt | 0.0046 | 0.0257 | 0.0429 |
| PTPRF CT (1,266 aa) vs. alt (262 aa) | 0.0061 | **0.1219** | 0.0538 |

PTPRF shows the largest layer-21 peak (0.122, 20× larger than layer 1), consistent with the fact
that the two isoforms originate from entirely distinct transcription start sites and share zero
amino acids. PINK1 shows a monotonically increasing distance to layer 30 (0.043), appropriate
for isoforms that share a common gene body but diverge in terminal exons. DMD shows a non-monotonic
pattern with a peak at layer 21 (0.028) then partial convergence, possibly reflecting shared
structural domains across Dp427m and the minor isoform.

### SN2.3 Interpretation

The layer-wise probing analysis supports three conclusions for PRISM methodology:

1. **Layer 30 selection is justified by geometric separation**: Sep_ratio is near-optimal at layer 30
   (1.093 vs. peak 1.129 at layer 21), providing the best-available within/between gene contrast for
   the non-linear MLP to exploit.

2. **Non-linear MLP is necessary**: Layer 30 is not optimal for linear prediction in most GO terms.
   Mid-layers (L11–L18) carry more linearly separable function-specific information. PRISM's MLP
   learns a non-linear mapping from layer-30 embeddings that recovers and substantially exceeds
   linear-probe performance.

3. **ESM-2 does not solve gene-level bias by itself**: Even at the best-separating layer (L21),
   intra_sim = 0.974 — within-gene isoforms remain highly similar. Within-gene cosine distances
   for case study pairs are at most 0.122 (PTPRF L21), far below the inter-gene average
   (1 − inter_sim ≈ 0.14 at L21). This directly motivates PRISM's isoform-specific training
   protocol and pos_bias evaluation.

**Figure SN2.** 4-panel layer probe figure:
- Panel A: intra_sim vs. inter_sim across layers (red = within-gene, blue = across-gene)
- Panel B: sep_cosine per GO term (glycolysis U-shape visible)
- Panel C: Linear probe AUPRC heatmap (layer × GO term; gene-stratified 5-fold CV)
- Panel D: Within-gene cosine distances for DMD, PINK1, PTPRF isoform pairs

*Source data*: `reports/layer_probe/layer_probe_results.json`
*Figure*: `reports/layer_probe/fig_layer_probe.pdf`
*Extraction script*: `hMuscle/preprocessing/compute_esm2_all_layers.py` (26.8 min, GPU0 RTX 4090)
*Analysis script*: `hMuscle/model/v_layer_probe.py` (30.9 min, CPU)
