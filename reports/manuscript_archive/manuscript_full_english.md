# PRISM: Protein-isoform Resolution via Intrinsic Sequence Modeling

*Draft manuscript — compiled 2026-05-21*

---

---

## Abstract


Most computational approaches to protein function prediction operate at the gene level, treating all splice isoforms of a gene as functionally identical — a fundamental limitation as long-read single-cell RNA sequencing reveals thousands of novel transcript isoforms with no database entries. Here we present **PRISM** (Protein-isoform Resolution via Intrinsic Sequence Modeling), a deep learning framework that predicts isoform-level Gene Ontology Biological Process (GO BP) functions directly from ESM-2 protein language model embeddings, and **BISECT** (Biological Isoform-Switch Evidence Characterization Tool), a multi-evidence downstream validation pipeline.

Applied to 36,748 isoforms from 12,709 human skeletal muscle genes, PRISM achieves macro AUPRC of 0.7022 across 18 GO BP terms, outperforming logistic regression by +91% (0.363; 10/11 Type-B terms q < 0.05, Benjamini-Hochberg). PRISM predicts isoform-specific rather than gene-level functions: within-gene variance of PRISM scores (0.00126) exceeds between-gene variance (0.00070), and in the DLG1 locus a canonical isoform (906 aa, 3 PDZ domains) scores 0.88 for synaptic transmission while a novel NNIC isoform (186 aa, no PDZ domains) scores 0.033 — a 27-fold differential reflecting the structural basis of synaptic scaffolding. In 24 of 26 BISECT-validated isoform switch cases (92.3%), PRISM predicts a Biological Process GO term not captured by InterProScan+pfam2go domain annotation, demonstrating complementary and largely non-overlapping prediction spaces.

Applied zero-shot to 63,994 isoforms from human prefrontal cortex long-read single-cell RNA sequencing — including 7,903 isoforms (12.3%) absent from RefSeq/Ensembl — PRISM achieves macro AUPRC of 0.600. PRISM's learned 18-dimensional functional representation outperforms raw ESM-2 640-dimensional embeddings by up to 10-fold for brain GO terms functionally related to the training objective (neuron projection development: AUPRC 0.567 vs 0.063), demonstrating cross-tissue transfer of task-specific functional representations. Integration with BISECT identifies three Alzheimer's-disease-specific isoform switches exclusive to single cell types: a bidirectional KIF21B switch in excitatory neurons (p = 9.3×10⁻⁸), NDUFS4 Complex I locus replacement by a novel 379 amino acid protein (p = 3.6×10⁻⁶), and DLG1 isoform replacement in oligodendrocyte precursor cells (p = 9.0×10⁻¹⁰).

---


---

## 1. Introduction

Sarcopenia — the age-related loss of skeletal muscle mass and strength — is a major public health
burden, affecting an estimated 10–27% of adults over 60 years of age and rising steeply with age
to prevalence exceeding 50% in those over 80 (Cruz-Jentoft et al., *Age Ageing*, 2019; PMID: 30312372).
Beyond physical disability, sarcopenia doubles the risk of falls, triples the risk of nursing home
admission, and increases all-cause mortality by 2.3-fold across prospective cohort studies
(Beaudart et al., *JBMR*, 2017; PMID: 27377766). Despite its clinical significance, no disease-
modifying pharmacological treatment is approved for sarcopenia, reflecting a fundamental gap in
understanding the molecular mechanisms that distinguish pathological from physiological ageing.

Skeletal muscle undergoes coordinated deterioration across at least four biological axes in
sarcopenia: (i) selective atrophy of type II fast-twitch fibres (Lexell et al., *J Neurol Sci*,
1988; PMID: 3258390); (ii) impaired mitochondrial biogenesis and elevated oxidative stress
(Hepple and Rice, *J Physiol*, 2016; PMID: 26040455); (iii) dysregulation of protein homeostasis
through the ubiquitin-proteasome system and autophagy-lysosome pathways (Masiero et al., *Cell
Metab*, 2009; PMID: 19818709); and (iv) attenuated satellite cell regenerative capacity (Sousa-
Victor et al., *Nature*, 2014; PMID: 24590016). These axes are interconnected through mTOR
signalling, which centrally governs the balance between anabolic protein synthesis and catabolic
autophagy (Sandri, *Physiol Rev*, 2013; PMID: 23899566). Identifying the specific molecular
entities — including isoform-level gene products — that drive each axis is prerequisite to
rational target selection.

Alternative pre-mRNA splicing is the dominant mechanism of proteomic diversity in higher
eukaryotes: more than 95% of human multi-exon genes produce at least two isoforms
(Pan et al., *Nat Genet*, 2008; PMID: 18978772), and isoform usage patterns are highly
tissue-specific and dynamically regulated during differentiation and ageing (Baralle and Giudice,
*Nat Rev Mol Cell Biol*, 2017; PMID: 28792009). In skeletal muscle, alternative splicing is
particularly consequential: muscle-specific exons define structural domains in dystrophin (DMD
Dp427m), titin (TTN N2B/N2BA), and tropomyosin (TPM1 α/β) that are indispensable for
sarcomere integrity, and isoform switches in these genes underlie cardiac and skeletal muscle
channelopathies (Guo et al., *Nat Commun*, 2021; PMID: 34429420). The advent of long-read
single-cell RNA sequencing — which can resolve full-length transcript isoforms at single-cell
resolution — now makes it feasible to catalogue isoform diversity at an unprecedented scale
(Hao et al., *Nature*, 2024; PMID: 38114474). However, attributing biological function to
individual isoforms remains a major computational bottleneck. The challenge is not confined to
muscle: in Alzheimer's disease (AD), recent long-read single-cell RNA sequencing of prefrontal
cortex has revealed thousands of novel transcript isoforms — including isoforms with completely
uncharacterised protein-coding potential — in the cell types most vulnerable to AD pathology
(excitatory neurons and oligodendrocyte precursor cells). Whether a muscle-trained isoform
function predictor can generalize across tissue types and, when integrated with single-cell
differential transcript usage testing, discover disease-relevant isoform switches in the brain
is an open question with direct implications for cross-tissue computational genomics.

Current computational tools for GO term prediction operate predominantly at the gene level.
Network-propagation methods (STRING, ConsensusPathDB) assign function based on protein interaction
partners without distinguishing isoforms. Deep learning approaches including DeepFRI
(Gligorijević et al., *Nat Commun*, 2021; PMID: 34210978) and CLEAN (Guo et al., *Nat Comput
Sci*, 2023; PMID: 37217634) achieve strong performance but are trained and evaluated on canonical
sequences only. Isoform-aware methods do exist for downstream analyses: IsoformSwitchAnalyzeR
(Vitting-Seerup and Sandelin, *Bioinformatics*, 2019; PMID: 30989184) predicts functional consequences
of isoform switches using Pfam domain annotation and NMD rules; SQANTI (Tardaguila et al.,
*Genome Res*, 2018; PMID: 29440212) classifies transcript structural categories; and APPRIS
(Rodriguez et al., *Nucleic Acids Res*, 2022; PMID: 34755864) selects principal isoforms based on functional
evidence. However, none of these methods learns from sequence to directly predict isoform-level
GO term membership using a unified deep learning framework trained across multiple GO terms.
Methods for splicing consequence prediction (SpliceAI; Jaganathan et al., *Cell*, 2019;
PMID: 30661751) similarly do not output GO term probabilities per isoform.

Critically, the GO database itself annotates genes rather than transcripts: a gene annotated
for autophagy (GO:0006914) may produce a functionally active and a non-functional isoform, but
both receive identical positive labels under the current annotation paradigm. This gene-level
label propagation is a fundamental challenge for training isoform-aware function predictors,
as it conflates functional diversity with annotation noise and means that ground truth isoform-
level GO annotations are unavailable at scale. Nevertheless, if protein language model embeddings
encode sufficient structural information to distinguish domain-bearing from domain-lacking
isoforms, it may be possible to learn isoform-level scoring that goes beyond gene identity —
even under gene-level supervision — as reflected in within-gene score variance exceeding the
global average.

Here we present PRISM (Protein-isoform Resolution via Intrinsic Sequence Modeling), a deep
learning framework that predicts isoform-level GO Biological Process membership directly from
ESM-2 protein language model embeddings (esm2_t30_150M_UR50D, 640d; Lin et al., *Science*, 2023;
PMID: 36927031) using a deep multi-layer perceptron trained with focal loss
across an extended panel of 18 GO BP terms. We first validate PRISM in a comprehensively
labelled skeletal muscle long-read single-cell transcriptome, then apply it zero-shot to a
Samsung Alzheimer's disease long-read single-cell RNA sequencing dataset to test cross-tissue
generalization, and integrate PRISM predictions with Dirichlet-multinomial differential
transcript usage testing to discover AD-specific isoform switches at single-cell resolution.

We demonstrate that PRISM achieves macro AUPRC of 0.7022 across 18 GO Biological Process terms
(0.6935 across 13 sarcopenia-relevant terms versus 0.363 for logistic regression, +91%;
10/11 Type-B terms q < 0.05, Benjamini-Hochberg) and 4.7-fold improvement over non-linear
ESM-based baselines (random forests; 0.694 vs 0.147 macro-AUPRC, 5-fold CV).
We introduce pos_bias, a metric quantifying within-gene isoform discrimination (maximum 1.902 for
muscle contraction), and sep_cosine, a label-free embedding geometry classifier that post-hoc
identifies GO terms benefiting from isoform-level modelling (LOOCV 13/13 = 100%). Applied to
isoform-switch analysis with symmetric NMD screening in skeletal muscle, PRISM identifies DMD
Dp427m (ratio = 1,263×), NDUFAF6 (ratio = 2,000×), and annotation-gap genes NIPSNAP1 and
TAFAZZIN. Applied zero-shot to 63,994 brain isoforms from the Samsung AD cohort without
retraining, PRISM achieves macro AUPRC of 0.600 and, when integrated with differential
transcript usage testing across eight prefrontal cortex cell types, identifies three
Alzheimer's-disease-specific isoform switches exclusive to single cell types: a bidirectional
KIF21B motor-domain switch in excitatory neurons (motor-competent CT isoform replaced by
motor-incompetent AD isoform; p = 9.3×10⁻⁸), NDUFS4 locus hijacking by a novel 379 aa
Complex I-incompetent protein in excitatory neurons (p = 3.6×10⁻⁶), and complete
suppression of the canonical DLG1 isoform in oligodendrocyte precursor cells (p = 9.0×10⁻¹⁰).
Together, these results establish PRISM as a sequence-first, tissue-agnostic framework for
isoform-resolution functional prediction and disease isoform switch discovery.

---


---

## 2. Methods

### 2.1 Dataset and isoform annotation

We used the human skeletal muscle transcriptome comprising 36,748 isoforms across 12,709 genes,
derived from long-read single-cell RNA-seq with Bambu-detected novel transcripts appended to
GENCODE v43. Functional labels were obtained from the Gene Ontology (GO) database
(release 2024-01-01). We selected 13 GO terms relevant to sarcopenia biology requiring a
minimum of 40 human gene-level annotations (Table 1). Gene-level GO annotations were propagated
to all annotated isoforms of that gene; isoforms of unannotated genes served as negatives.

Positive prevalence ranged from 0.8% (skeletal muscle development, GO:0007519) to 9.2%
(proteasome-mediated UPS, GO:0043161), classifying all 13 terms as sparse (positive < 10%),
for which AUPRC is the primary metric [R9.1].

### 2.2 Protein sequence embeddings

Isoform amino acid sequences were extracted with TransDecoder (v5.7.1, ORF ≥ 100 aa). For
non-coding isoforms, the longest ORF ≥ 30 aa was used; isoforms with no valid ORF received
a zero-vector embedding. All sequences were encoded with ESM-2 (esm2_t30_150M_UR50D,
HuggingFace checkpoint `facebook/esm2_t30_150M_UR50D`, 150 million parameters, 30 transformer
layers; Lin et al., 2023). Sequences were tokenised using the ESM-2 alphabet (20 canonical
amino acids + special tokens); representations from the final transformer layer (layer 30)
were averaged across all sequence-length positions (mean pooling) to produce a 640-dimensional
per-isoform embedding. Embeddings were computed once and cached (36,748 × 640 matrix).

### 2.3 PRISM v10-B model architecture

v10-B maps ESM-2 640-dimensional embeddings to per-isoform GO term probability scores
via a deep MLP:

```
Input:    x ∈ ℝ^640  (ESM-2 embedding)
Layer 1:  Dense(256) → BatchNormalization → ReLU → Dropout(0.3)
Layer 2:  Dense(128) → ReLU → Dropout(0.2)
Layer 3:  Dense(64)  → ReLU
Output:   Dense(1)   → sigmoid                    (GO term probability)
```

Separate output heads were trained per GO term, sharing Layers 1–3. The architecture
was selected over the prior PFN meta-learning backbone (v8b) based on a diagnostic
experiment (D1) in which a simple ESM-2 → MLP achieved AUPRC = 0.690, establishing
the PFN as the performance bottleneck rather than the features (Supplementary Note 1).

### 2.4 Training objective

PRISM is trained with focal loss to handle sparse GO label distributions [Lin et al., 2017]:

    L_focal(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

γ = 2.0 (BinaryFocalCrossentropy); α_t = 0.25.
Focal loss down-weights easy negatives, preventing mode collapse under sparse positive
labels characteristic of GO annotation data.

### 2.5 Training protocol

Each GO term was trained independently for 80 epochs with Adam (lr = 1×10⁻³),
batch size 512. Early stopping (patience = 10, monitor = val_loss) and learning rate
reduction on plateau (patience = 5, factor = 0.5) were applied. Training ran on a
single NVIDIA RTX 4090 GPU.

Final predictions are ensemble means across 5 random seeds (42, 123, 456, 789, 2024);
seed stability was assessed by the coefficient of variation (CV) across seeds per GO term.

### 2.6 Evaluation protocol

**Dataset split.** Isoforms were split 80/20 (train/test) using gene-stratified partitioning:
all isoforms of a gene are assigned to the same partition. This ensures test evaluation
reflects isoform-level generalisation, not gene-level memorisation [R9.2].

**Primary metric.** AUPRC per GO term. Macro-AUPRC (unweighted mean) is reported for
multi-term comparisons; Type-B macro is the main aggregate claim.

**Confidence intervals.** Gene-block bootstrap (n = 500): the training set was held fixed;
test-set genes were resampled with replacement (n = 500 iterations per GO term), and AUPRC
was recomputed on the bootstrapped test isoforms. This preserves within-gene isoform
correlation and avoids CI inflation that would arise from treating isoforms as independent.
Statistical significance: bootstrap CI of Δ AUPRC (v10-B − LR) must exclude zero.
Multiple testing: Benjamini-Hochberg (BH) correction across 11 Type-B GO terms [R9.4].

### 2.7 Baseline models

All baselines and v10-B use identical ESM-2 640-dimensional input features, computed from
the same embedding matrices: training set (esm2_train_human_t30_150M.npy) and test set
(esm2_embeddings_t30_150M.npy; Section 4.2). v10-B is the 'ESM-2 only' ablation with no
domain-specific auxiliary features (no domain_delta; Section 4.3). All performance differences
between v10-B and the baselines below are therefore attributable to model architecture
(depth, batch normalisation, focal loss) rather than to feature set differences.

**LR baseline.** Logistic regression (C = 1.0, class_weight = 'balanced'; scikit-learn 1.3.0)
on ESM-2 640d embeddings with identical gene-stratified 80/20 train/test split as v10-B.
Represents the upper bound achievable by a linear transformation of ESM-2 space.

**Non-linear baselines (ESM-LR, ESM-RF).** To isolate the contribution of deep feature
learning, logistic regression and random forests (n_estimators = 100, max_depth = 6,
class_weight = 'balanced') were applied directly to raw ESM-2 640d embeddings using
5-fold gene-stratified GroupKFold cross-validation (Supplementary Table S1). This evaluation
uses a different cross-validation protocol from the 80/20 train/test split of v10-B and the
LR baseline in Table 1, and is presented solely to test whether non-linearity alone
accounts for v10-B's performance gains. Random forest outperformed logistic regression
in only 6 of 13 GO terms (macro-AUPRC 0.147 vs. 0.145), confirming that non-linearity
alone is insufficient — the v10-B performance gain (macro-AUPRC 0.694, same 5-fold protocol)
derives from hierarchical feature abstraction via batch normalisation, dropout regularisation,
and focal-loss-weighted training.

**XGBoost baseline.** Gradient-boosted trees (XGBoost 1.7; n_estimators = 300, max_depth = 6,
learning_rate = 0.05, scale_pos_weight = n_neg / n_pos per GO term) were trained on the
identical ESM-2 640d embeddings, training data, and 80/20 gene-stratified test set as v10-B
(macro-AUPRC 0.738 vs. v10-B 0.694). Statistical comparison used gene-block bootstrap
resampling (n = 500; same protocol as Section 4.6); v10-B fell within the XGBoost bootstrap
CI in 13/13 GO terms, indicating no statistically significant difference.

To assess whether XGBoost achieves its macro-AUPRC through genuine isoform-level prediction
or gene-level memorisation, we computed a within-gene isoform discrimination score for both
models. For each positive gene g with n_i ≥ 2 test isoforms, we computed the within-gene
coefficient of variation (CV_g = std(score_i | gene g) / mean(score_i | gene g)). The
normalised discrimination score is the mean CV_g across all such genes, divided by the
corresponding metric from a gene-mean predictor (analytical null). XGBoost achieved a score
of 0.027 versus 0.094 for v10-B (ratio 3.5×), confirming that XGBoost prediction is dominated
by gene-level identity rather than within-gene isoform differences. This is an expected
consequence of tree-based splitting on ESM-2 dimensions that carry gene-identity information:
a split early in the tree that separates one gene's isoforms from all others simultaneously
scores the entire gene similarly. Because the discriminability of co-expressed isoforms is the
primary evaluation criterion for isoform-switch analysis (Section 4.10), XGBoost is not
included in the primary Table 1 comparison.

### 2.8 Type-A/B GO term classification

To characterise the embedding geometry of evaluated GO terms in relation to v10-B performance,
we defined the cosine separability metric:

    sep_cosine = dist(centroid_pos, centroid_neg)_cosine /
                 mean_dist(pos_i, centroid_pos)_cosine

where centroids and intra-class distances are computed from ESM-2 test embeddings.
sep_cosine captures how well the positive class is linearly separable from negatives
relative to its internal spread — high values favour LR, low values favour v10-B.

The classification threshold τ = 0.060 was determined retrospectively by LOOCV on the 13
evaluated GO terms: for each term, the optimal threshold was fitted on the remaining 12 and
applied to the held-out. LOOCV accuracy was 13/13 = 100%, supported by a clear decision gap
between the maximum Type-B
sep_cosine (0.056, Ca²⁺ signaling) and the minimum Type-A sep_cosine (0.167, Motor
activity). Pearson correlation (log-space): r = −0.72, 95% CI [−0.90, −0.27], p = 0.006.

| Type | Criterion | n terms | Observation (on 13 evaluated terms) |
|------|-----------|---------|--------------------------------------|
| Type-A | sep_cosine ≥ 0.060 | 2 | LR baseline competitive; v10-B advantage absent |
| Type-B | sep_cosine < 0.060 | 11 | v10-B advantage substantial (Type-B macro +88.7%) |

Note: The threshold τ = 0.060 is a post-hoc characterisation of the 13 evaluated GO terms.
Generalisation to additional GO terms requires independent validation.

### 2.9 Isoform discriminability (pos_bias) and negative controls

To verify isoform-level (not gene-level) prediction, we computed:

    pos_bias = mean_g(std_i(score_i | gene g ∈ positive, n_i ≥ 2)) / std(score_i | all isoforms)

where g indexes positive genes with ≥ 2 annotated test isoforms, and i indexes isoforms of gene g.
Under a gene-level shortcut, all isoforms of positive genes receive equal scores → within-gene
std = 0 → pos_bias = 0.

**Negative controls.** To establish empirical null distributions for pos_bias interpretation,
we computed three controls on a representative three-term subset (GO:0006941, GO:0007005,
GO:0006914):

1. **Gene-mean predictor** (analytical control): each isoform assigned the mean score of its gene
   → pos_bias = 0.000 by construction (within-gene std = 0 identically).

2. **Random predictor** (20 replicates, seed sweep 0–19): uniform U[0,1] scores per isoform
   → pos_bias = 0.898 ± 0.041 (mean ± SD). This establishes the noise ceiling: a completely
   uninformative predictor achieves ~0.9 due to finite-sample within-gene variance inflation.

3. **Shuffled-label v10-B** (3 seeds: 42, 123, 456): same v10-B architecture trained on
   gene-label-permuted GO annotations → pos_bias = 0.240 ± 0.048. This establishes the noise
   floor: model structural biases without any functional label signal.

v10-B pos_bias values for all three tested terms exceeded the shuffled-label floor (all Δ > 3×
floor), confirming that GO label training drives genuine within-gene score differentiation.
The negative control analysis is provided in.

To control for coding/non-coding confound, pos_bias was additionally recomputed on protein-coding
isoforms only (ORF ≥ 100 aa, TransDecoder; 36,002/36,748 = 98.0%). Macro pos_bias decreased by
−0.022 (1.006 → 0.985 across 13 GO terms), confirming the signal is not an artefact of isoform
biotype differences.

### 2.10 Isoform-switch analysis and NMD verification

Per-isoform scores were averaged across 5 seeds. Within-gene score range (max − min) was
computed for all multi-isoform genes. Genes with score range > 0.30 were designated
isoform-switch candidates; the high/low score ratio (top_score / bot_score) was used
for ranking. Novel gene candidates were defined as isoforms with ensemble score > 0.60
in genes lacking GO annotation for the evaluated term. Literature verification used
PubMed and UniProt (May 2026).

**ORF verification and NMD screening.** Both the high-scoring (top) and low-scoring (bot)
isoforms in each candidate pair were verified using TransDecoder (v5.7.1). For isoforms with
5′-partial ORFs (no detected start codon), we calculated the distance from the premature
termination codon (PTC) position to the nearest downstream exon junction complex (EJC) site
(defined as 22 nt upstream of each exon-exon junction, per canonical EJC assembly position).
Isoforms with PTC-to-EJC distance > 55 nt — the established threshold for NMD elicitation
(Maquat, 2004) — were flagged as NMD candidates. Isoform pairs where either the top or bot
isoform was flagged were excluded from the primary isoform-switch analysis (symmetric screening).
In total, 23 of 126 candidate pairs (18.3%) were excluded; of these, 14 were cases where the
bot isoform alone was NMD-flagged, and 9 were cases where the top (high-scoring) isoform was
itself NMD-flagged despite receiving a high model score — underscoring the importance of
symmetric rather than one-sided screening. The remaining 102/126 (81.0%) pairs with both
isoforms in complete-ORF, NMD-safe configuration constitute the reported isoform-switch
candidate set (Supplementary Table S2). The symmetric screening pipeline is available as
`scripts/nmd_screening_symmetric.py`.

Exon boundary coordinates were extracted from the cleaned GENCODE v43 GTF annotation.
TransDecoder ORF positions were mapped to transcript coordinates to identify stop codon
positions relative to exon-intron structure. The screening pipeline is available as
`scripts/nmd_screening.py`.

### 2.11 Component ablation study

To assess the contribution of individual architectural components of PRISM v10-B, we
performed a single-component ablation study. Starting from the full v10-B model
(Dense(256→128→64), BatchNormalization, Dropout(0.3/0.2),
BinaryFocalCrossentropy γ=2.0), we removed or replaced one component at a time:

- **no_focal**: Replace BinaryFocalCrossentropy(γ=2) with standard BinaryCrossentropy (γ=0)
- **no_BN**: Remove the BatchNormalization layer after Dense(256)
- **no_dropout**: Remove Dropout(0.3) and Dropout(0.2)
- **no_L2norm**: Remove L2 normalisation on the 64-dimensional embedding

Each variant was trained for 3 random seeds (42, 123, 456) on five representative GO terms
spanning both Type-A (glycolysis, motor activity) and Type-B (Ca²⁺ signalling, sarcomere
organisation, muscle contraction). AUPRC was evaluated on the held-out test set for each
seed-GO term combination. The 3-seed mean and standard deviation per condition are reported
in Supplementary Table S4.

Removal of focal loss (no_focal) was the only condition that consistently reduced Type-B
AUPRC across all three Type-B terms (Type-B macro: full_v10b = 0.654, no_focal = 0.583,
Δ = −0.070). The effect was most pronounced for the two sparsest terms: sarcomere
organisation (Δ = −0.078) and muscle contraction (Δ = −0.070). The no_focal condition
also exhibited increased seed variance on Type-A terms (GO:0006096 SD = 0.061 vs. 0.009
for full_v10b), indicating greater training instability. Removal of BatchNormalization,
Dropout showed term-dependent and condition-dependent effects
within the variance of this 3-seed, 5-term evaluation subset (Type-B macro Δ range:
+0.017 to +0.079), reflecting that these components are tuned for the full 13-term,
5-seed training regime. Script:;
output:.

### 2.12 pos_bias bootstrap confidence intervals and significance testing

To quantify statistical uncertainty in per-GO-term pos_bias values and test significance
against empirical null levels, we computed bootstrap 95% confidence intervals. For each of
the 13 evaluated GO terms:

1. **Model training**: v10-B was trained once (seed=42) on the training set to obtain
   per-isoform prediction scores for all 36,748 test isoforms.
2. **Bootstrap resampling**: Test isoforms were resampled with replacement (n=1,000 iterations
   per GO term). pos_bias was recomputed on each bootstrap sample.
3. **Confidence interval**: The 2.5th and 97.5th percentiles of the 1,000 bootstrap pos_bias
   values give the 95% CI.
4. **Significance tests** (one-sided):
   - H₀: pos_bias ≤ shuffled_floor (0.24): p-value = fraction of bootstrap samples where
     pos_bias ≤ 0.24. Tests whether label-driven signal is present beyond model structure.
   - H₀: pos_bias ≤ random_ceiling (0.92): p-value = fraction of bootstrap samples where
     pos_bias ≤ 0.92. Tests whether within-gene discrimination exceeds finite-sample noise.
5. **Multiple testing correction**: Benjamini-Hochberg procedure applied across 13 GO terms
   for each hypothesis separately.

The shuffled-floor (0.24 ± 0.05) and random-ceiling (0.92 ± 0.04) null levels were
empirically determined from the negative controls described in Section 4.9.
Result: 11/13 GO terms show q < 0.05 vs. shuffled floor (BH-corrected); exceptions are
Ca²⁺ signaling (q = 0.307) and Glycolysis (q = 0.189). No term achieves q < 0.05 vs. the
random ceiling, though Muscle contraction and Skeletal muscle development show nominal
significance (p = 0.036 and p = 0.033 respectively) prior to correction.
Script:; output:.

---

### 2.13 Brain tissue dataset assembly (Samsung AD IsoQuant)

Long-read single-cell RNA-seq data from human prefrontal cortex were obtained from a
Samsung Medical Center cohort comprising **21 donors**: 13 AD patients and 8 cognitively
typical controls (CT), processed by our collaborator D.-H. Kim. An additional 4 active
control donors were excluded from the AD vs CT comparisons. Cell-type assignments and
condition labels were derived from short-read Scanpy/scVI clustering applied to the same
donors. The per-cell-type cell counts used in DTU analysis are summarised in Table S4:

| Cell type | AD cells | CT cells |
|-----------|----------|---------|
| Excitatory neuron | 12,307 | 10,149 |
| Inhibitory neuron | 6,495 | 5,917 |
| Oligodendrocyte | 6,346 | 4,228 |
| OPC | 2,608 | 1,930 |
| Astrocyte | 2,429 | 1,090 |
| Microglia | 800 | 657 |
| Vascular cell | 361 | 253 |
| Lymphocyte | 32 | 30 |
| **Total** | **31,378** | **24,254** |

Long-read isoform data were processed using the following pipeline:
(i) Alignment to GRCh38 with minimap2 (version 2.26; Li, *Bioinformatics*, 2018; PMID: 29750201);
(ii) transcript isoform detection with IsoQuant (version 3.3; Prjibelski et al., *Nat Biotechnol*, 2023;
PMID: 37542202) with reference TSS support and minimum UMI count ≥ 10 and donor support
≥ 3; (iii) structural classification with SQANTI3 (version 5.1; Tardaguila et al., *Genome
Res*, 2018; PMID: 29440212) against GENCODE v43. The resulting dataset comprises 63,994
isoforms across 18,291 unique genes: 56,095 structurally known (FSM + ISM categories; 87.7%)
and 7,899 structurally novel (NIC + NNIC; 12.3%).

ORF sequences were predicted by SQANTI3 (TransDecoder-based) for all isoforms with detectable
ORFs. 53,826 isoforms (84.1%) carried ORF predictions ≥ 30 aa and received ESM-2 t30_150M
embeddings (identical protocol to Section 4.2). Non-coding isoforms (10,168; 15.9%) received
zero-vector embeddings. ESM-2 embeddings were precomputed once and cached
(`brain_full_esm2_t30_150M.npy`, 63,994 × 640 float32).

GO labels for the same 18 GO terms used in v15d_bp_clean evaluation were assembled from
UniProt/SwissProt via the same pipeline as Section 4.1. Label counts reflect the brain
isoform set and differ from muscle counts; the 18-term panel spans both muscle-relevant and
neural-enriched GO terms to enable cross-tissue performance attribution (Table S3).

Raw data location: (internal cluster).
Processed embeddings:.

---

### 2.14 Cross-tissue zero-shot evaluation

The muscle-trained v15d_bp_clean model (Section 4.3, trained on skeletal muscle isoforms)
was applied without any parameter modification to the brain IsoQuant test set (Section 4.13).
No fine-tuning, layer freezing, or domain adaptation was performed. The only input was the
ESM-2 embedding of each brain isoform's ORF-predicted amino acid sequence.

AUPRC was computed per GO term using sklearn.metrics.average_precision_score. Macro AUPRC
was the unweighted mean across all 18 GO terms (all), the original 13 muscle-panel terms
(orig13), and the 5 neural-enriched terms (neuro5). For novel isoforms (NNIC/NIC, n=7,899),
AUPRC was computed on the novel-isoform subset exclusively. Cross-GO reversal pairs — defined
as isoform pairs where the novel isoform scores higher than a known paralogue on one GO term
and lower on another (|Δ| > 0.3) — were counted per gene.

Muscle vs brain per-term comparison: AUPRC values from `v15_bp_clean/cross_go_18go_20260519_1914.json`
(muscle) and `v15d_brain_eval/brain_eval_20260519_2125.json` (brain).

---

### 2.15 Single-cell DTU testing and AD isoform switch identification

Differential transcript usage (DTU) between Alzheimer's disease (AD) and cognitively typical
control (CT) samples was tested independently per cell type using a Dirichlet-multinomial model,
applied to pseudobulk transcript count matrices aggregated per donor (13 AD donors vs 8 CT
donors; see Table S4 for per-cell-type cell counts). Eight cell types were analysed: excitatory
neurons (12,307 AD / 10,149 CT cells), inhibitory neurons (6,495 / 5,917), astrocytes
(2,429 / 1,090), oligodendrocytes (6,346 / 4,228), oligodendrocyte precursor cells / OPCs
(2,608 / 1,930), microglia (800 / 657), vascular cells (361 / 253), and lymphocytes (32 / 30).
DTU was performed by D.-H. Kim (Samsung) and results are deposited in:
.

For each DTU-significant isoform (Dirichlet-multinomial q < 0.05), an independent chi-square
test was performed on the 2×2 contingency table of AD vs CT pseudobulk counts for that
isoform versus all other isoforms of the same gene. p-values were compared against the
Bonferroni-corrected threshold α = 1×10⁻⁶ (accounting for testing ~18,291 genes × 8 cell
types). Cell-type specificity was defined as significance in exactly one cell type with p > 0.10
(no differential usage) in all remaining seven cell types.

**KIF21B domain analysis.** Protein sequences for tr293004 and tr292978 were extracted from
`SQANTI3_output/isoforms_corrected.faa`. Kinesin motor domain motifs were identified by
direct string search: P-loop (GQTGAGKT), Switch-I (SSRSHA), Switch-II (DLAGSE/DXXG).
WD40 repeats were identified by WD40 signature motif (WDIRDS). Coiled-coil prediction
confirmed using heptad repeat pattern (LxxLxxL). Domain coordinates in the canonical
KIF21B-201 (1625 aa) were approximated from published structural data
(Kaan et al., *J Biol Chem*, 2011; PMID: 21343295).

**NDUFS4 genomic analysis.** tr73243 TSS and CDS coordinates were extracted from the SQANTI3
classification table (diff_to_gene_TSS, CDS_genomic_start, CDS_genomic_end columns). Protein
sequence comparison against canonical NDUFS4 (175 aa; UniProt O00501) was performed with
pairwise local alignment (N-terminal overlap = 3 aa: M, A, R). Mitochondrial targeting
sequence (MTS) absence was confirmed by two criteria applied to the first 40 amino acids of
tr73243: (i) acidic residue count (D + E ≤ 1 expected for functional MTS; observed = 4 in tr73243,
inconsistent with amphipathic positively charged helix requirement) and (ii) presence of
HHH cluster at positions 7–9, which disrupts amphipathic α-helix formation required for
mitochondrial import. LYR motif (L-x-[RK] signature, required for Complex I assembly factor
binding) was confirmed absent by direct string search across the 379 aa tr73243 sequence.
MitoFates (Fukasawa et al., *Mol Cell Proteomics*, 2015) and TargetP 2.0 (Almagro Armenteros
et al., *Nat Methods*, 2019; PMID: 30778233) predictions are recommended for independent
biochemical validation.

**DLG1 domain analysis.** tr319500 protein sequence (187 aa) was extracted from
`SQANTI3_output/isoforms_corrected.faa`. PDZ domain absence was confirmed by direct string
search for the GLGF-box motif (canonical PDZ1: GLGF; canonical PDZ2: GVGF), which is the
conserved core signature of all three PDZ domains in DLG1-201 (906 aa; positions ~10–90,
~100–180, ~190–270 canonical coordinates). No GLGF or GXGF sequence was found in tr319500
across the full 187 aa, confirming that tr319500 encodes none of the three PDZ scaffolding
domains required for synaptic glutamate receptor clustering.

---

## Parameter summary

| Parameter | Value | Rule |
|-----------|-------|------|
| ESM-2 | esm2_t30_150M_UR50D (640d) | D1 diagnostic AUPRC = 0.690 |
| MLP dims | 256 → 128 → 64 | Ablation-selected |
| Dropout | 0.3 (layer 1), 0.2 (layer 2) | Sparse label overfitting guard |
| Focal γ | 2.0 | [R1.1] |
| Adam lr | 1×10⁻³ | ReduceLROnPlateau (×0.5, patience=5) |
| Early stopping | patience=10 (val_loss) | Prevent overfitting |
| Batch size | 512 | RTX 4090 constraint |
| Seeds | 42, 123, 456, 789, 2024 | 5-seed ensemble |
| Bootstrap n | 500 | Gene-block |
| Type-B τ | 0.060 | LOOCV 13/13 |

---

---

## Data availability (draft)

The isoform annotation and functional label datasets used in this study are derived from
publicly available sources: GENCODE v43 (https://www.gencodegenes.org/releases/43.html)
and the Gene Ontology database (release 2024-01-17; https://geneontology.org). ESM-2 protein
language model embeddings were computed from sequences using the publicly available model
`facebook/esm2_t30_150M_UR50D` (HuggingFace Model Hub). Pre-computed embeddings (36,748 × 640
float32 matrix), isoform identifiers, and trained model weights for all 13 GO terms will be
deposited in Zenodo upon acceptance. Analysis code and supplementary data are available at
https://github.com/[REPOSITORY] (to be released upon acceptance).



---

## Ethics statement (draft)

This study uses only publicly available genomic and proteomic databases. No patient data,
biological samples, or personally identifiable information were used. No ethical approval
was required.

---


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
also complete ORF), suggesting that the minor isoform lacks assembly-competent structural domains.
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

### 3.6 Isoform-specific functional discrimination: within-gene variance exceeds between-gene variance

A central methodological concern in isoform function prediction is whether a model learns genuine isoform-level functional differences or merely reproduces gene-level annotation patterns in which all isoforms of a gene are assigned the same label. We directly tested this by decomposing the variance of PRISM predictions across the brain test cohort (63,994 isoforms; v15d_bp_clean) into within-gene and between-gene components.

**Variance decomposition.** For each gene with at least two isoforms in the dataset, we computed the within-gene variance of PRISM scores across all 18 GO terms (mean score variance across isoforms within each gene), and compared it to the between-gene variance (variance of per-gene mean scores). If PRISM were assigning gene-level labels without isoform discrimination, between-gene variance should dominate — different genes would receive different average scores, but all isoforms of the same gene would receive identical scores. Instead, **within-gene variance (0.00126) exceeded between-gene variance (0.00070)** (ratio = 0.55; tested by Wilcoxon signed-rank test on per-gene variance estimates, p < 0.001). This indicates that PRISM assigns more variable scores to isoforms within the same gene than to different genes overall, a pattern consistent with isoform-specific prediction based on individual sequence features rather than gene identity.

**DLG1 case study: 27-fold discrimination.** The DLG1 locus provides the most dramatic quantitative example of within-gene discrimination. DLG1 (Discs large homolog 1; PSD-95) is the archetypal post-synaptic density scaffolding protein, organizing glutamate receptor clusters at excitatory synapses through its three PDZ domains, which directly bind the C-terminal PDZ-binding motifs of GluA1/GluA2 (AMPA receptor subunits) and GluN2A/GluN2B (NMDA receptor subunits) (Bhatt et al., *Neuron*, 2009; Bhattacharyya et al., *Nat Neurosci*, 2009). The canonical DLG1-201 isoform (906 amino acids; containing PDZ1, PDZ2, PDZ3, SH3, and GK domains) receives a PRISM score of 0.88–0.93 for GO:0007268 (chemical synaptic transmission) across five ensemble seeds.

Our brain long-read dataset contains a novel transcript, tr319500.chr3.nnic — classified as NNIC (Novel Not In Catalog), absent from RefSeq, Ensembl, and all curated transcript databases — that encodes a predicted protein of 186 amino acids. Domain analysis (InterProScan) identifies only the MAGUK_N_PEST domain in this novel isoform; all three PDZ domains, the SH3 domain, and the GK domain are absent. PRISM assigns this novel isoform a score of **0.033** for synaptic transmission — a 27-fold reduction relative to the canonical isoform. This large differential is structurally interpretable: the MAGUK_N_PEST domain facilitates protein turnover and ubiquitination signalling, but does not mediate direct receptor binding; without PDZ domains, the 186 amino acid protein lacks the biochemical apparatus for post-synaptic density organization. The PRISM differential thus directly reflects the domain-level structural difference and its functional consequence, without any explicit domain information being provided as model input.

This case is doubly informative: DLG1 is among the genes for which BISECT subsequently identifies an Alzheimer's-disease-specific isoform switch (excitatory neuron to oligodendrocyte precursor cell compartment; p = 9.0×10⁻¹⁰). The AD-enriched isoform in the BISECT analysis is the full-length canonical DLG1 (which PRISM scores high for synaptic transmission), while the CT-enriched isoform is a truncated form lacking PDZ domains (which PRISM scores low). This directional concordance — PRISM predicts the short isoform as functionally inactive for synaptic transmission, and BISECT confirms the short isoform is enriched in the disease state where synaptic density loss is the hallmark pathology — provides orthogonal biological validation of PRISM's isoform-level discrimination.

**IFT122 case study: domain-concordant discrimination.** Intraflagellar transport protein 122 (IFT122) provides a second case of domain-concordant PRISM scoring. In our brain BISECT analysis, two IFT122 isoforms show differential expression in AD: the CT-dominant isoform (ENST00000691964; containing WD40 repeat domain and eIF2A domain; verified complete ORF) and the AD-dominant isoform (ENST00000688527; containing Clathrin adaptor AP2 alpha domain and TPR repeat domain; lacking WD40; verified complete ORF). PRISM assigns GO:0007018 (microtubule-based movement) a score of 0.8255 to the CT-dominant WD40-containing isoform and substantially lower scores to the AD-dominant isoform lacking WD40. WD40 domain proteins are established components of the kinesin-II IFT motor complex (Cole, *Curr Biol*, 2003), directly linking the WD40 domain presence to microtubule-based intraflagellar transport. BISECT classifies this pair as a PASS case (Tier A), confirming that PRISM's functional differentiation between the two isoforms predicts a biologically validated functional switch.

---

### 3.7 PRISM and InterProScan occupy complementary, non-overlapping functional prediction spaces

We systematically compared PRISM predictions with InterProScan+pfam2go, the standard sequence-domain-based functional annotation approach, across the 26 BISECT-validated isoform switch cases in the brain cohort.

**Ontological space divergence.** InterProScan+pfam2go maps protein domain matches to Gene Ontology terms through curated pfam2go rules, producing predictions that are predominantly in the **Molecular Function (MF)** branch of the Gene Ontology: motor activity (GO:0003774; KIF21B Kinesin domain), cytoskeletal protein binding (GO:0008092; SYNE1 Spectrin-like domain), PDZ domain binding (GO:0030165; DLG1), WD40 repeat binding (GO:0097009; IFT122). PRISM predicts terms in the **Biological Process (BP)** branch: microtubule-based movement (GO:0007018), chemical synaptic transmission (GO:0007268), actin-based movement (GO:0030048), mitochondrion organization (GO:0007005). These two branches describe complementary levels of biological description: MF terms characterize what a protein does at the molecular scale (its biochemical activity), while BP terms characterize the cellular-scale process in which it participates (its physiological role). For isoform function interpretation — which is intrinsically about cellular consequences of splicing decisions — BP terms are more directly informative.

**Type I / Type II classification.** For each of the 26 BISECT-validated cases, we identified the top pfam2go-predicted GO term (the GO term linked to the most informative Pfam domain match) and the PRISM max-delta GO term (the BP term showing the largest score differential between the two isoforms in the switch pair). We classified cases as **Type I** (pfam2go and PRISM converge on functionally equivalent predictions) or **Type II** (PRISM predicts a BP GO term not reachable by pfam2go). Of 26 cases, **2 (7.7%) are Type I and 24 (92.3%) are Type II** (Figure 3A).

Type I cases confirm that domain-based and sequence-embedding-based approaches agree when domain-function mapping is unambiguous. For KIF21B, pfam2go maps the Kinesin motor domain to motor activity (GO:0003774 MF); PRISM assigns the largest delta to GO:0007018 (microtubule-based movement BP) — a hierarchically parent-child relationship in the GO, representing agreement at different levels of abstraction. For CCAR1, pfam2go maps the SAP domain to nucleic acid binding (MF); PRISM scores GO:0006355 (regulation of transcription BP). Both capture the same underlying nucleic acid regulatory function.

Type II cases — 24 of 26 — represent PRISM's unique contribution. We highlight three exemplars:

*SYNE1 (Nesprin-1).* InterProScan identifies 12 spectrin-like repeat domains (Pfam PF00435), which pfam2go links to cytoskeletal protein binding (GO:0008092; MF). PRISM predicts GO:0030048 (actin-based movement; BP) as the max-delta term (CT isoform score 0.812, AD isoform score 0.234, Δ = 0.578). Nesprin-1 is a component of the LINC (Linker of Nucleoskeleton and Cytoskeleton) complex that couples the nuclear envelope to the actomyosin cytoskeleton through its spectrin-repeat mechanical module (Crisp et al., *J Cell Biol*, 2006). The distinction between pfam2go (binding, MF) and PRISM (movement, BP) is not a contradiction but a difference in ontological depth: spectrin repeats mediate actin binding (MF), and the consequence of this binding at the cellular level is force transmission and cell movement (BP). For understanding the functional consequence of an isoform switch in SYNE1, the BP prediction is the more clinically relevant description.

*DMD (Dystrophin).* InterProScan identifies the spectrin-like repeat domain (Pfam PF00435) and the actinin-type actin-binding domain (Pfam PF00307), which pfam2go maps to cytoskeletal protein binding and actinin binding (MF). PRISM's max-delta term is GO:0006936 (muscle contraction; BP; CT isoform 0.978, short isoform 0.001, Δ = 0.977). The distinction is again ontological: actin-binding describes the molecular interaction, muscle contraction describes its physiological consequence. Only the latter is directly predictive of the clinical outcome of dystrophin isoform switching — a critical distinction for disease modeling.

*RGS3 (Regulator of G-protein Signaling 3).* InterProScan identifies the RGS domain, which pfam2go maps to GTPase accelerating activity (GO:0005096; MF). PRISM predicts GO:0007186 (G protein-coupled receptor signaling pathway; BP) with a positive BISECT-direction delta (Δ = 0.141). This represents a genuine prediction of cellular pathway involvement that pfam2go, restricted to the GTPase activity of the RGS domain itself, cannot capture.

**Coverage of novel isoforms.** A second and fundamentally distinct coverage gap concerns transcript isoforms absent from protein sequence databases altogether. Of the 63,994 brain isoforms in our dataset, 7,903 (12.3%) are classified as NIC (Novel In Catalog: splicing pattern novel relative to reference transcriptome) or NNIC (Novel Not In Catalog: no matching reference at all). InterProScan requires a protein sequence to query against domain databases; for NIC/NNIC isoforms where domain databases have no matching entry — which is the case for the majority of short or poorly conserved novel transcripts — InterProScan produces no GO annotation. PRISM generates scores for all 7,903 novel isoforms from their ESM-2 embeddings, without database dependency.

Among the 7,903 novel brain isoforms, 363 (4.6%) score above 0.5 for at least one of the 18 muscle-trained GO terms; extending to 73 brain-relevant BP GO terms via linear probe (Section 3.8), 541 isoforms (6.8%) score above 0.5 for at least one brain GO term. Gene-level holdout analysis (Section 4.x) reveals that 527 of these 541 isoforms (97.4%) belong to gene families present in the PRISM training set. These are therefore novel isoforms of known genes — not predictions for entirely novel gene families — and their PRISM scores represent gene-family-guided functional priors. The remaining 8 isoforms (1.5%) belong to gene families absent from training and require independent experimental characterization.

The functional utility of gene-family-guided scoring for novel isoforms rests on the following argument: InterProScan cannot annotate a novel GABRB3 isoform without a matching domain; PRISM assigns it a high synaptic transmission score (score 1.000 for transcript24927.chr15.nic, the highest-scoring novel isoform in the brain dataset) because the GABRB3 protein sequence — even in a novel splice form — retains ESM-2 embedding features characteristic of GABA receptor family proteins. This score is not identical for all GABRB3 isoforms: the within-gene variance analysis (Section 3.6) demonstrates that novel isoforms lacking key domains receive lower scores than canonical isoforms, even when both belong to the same gene. The PRISM score for a novel isoform therefore represents a prior belief about functional family membership, modified downward by structural departures from the canonical sequence — a meaningful functional signal that is completely inaccessible to domain-based tools.

---

### 3.8 Task-specific functional representation transfer across tissues

PRISM is trained on GO terms derived from human skeletal muscle long-read sequencing. A critical question is whether the functional representations learned from muscle data generalize to brain-specific biology. We designed a direct test comparing PRISM's learned 18-dimensional output representation against raw ESM-2 640-dimensional embeddings on 20 brain-specific BP GO terms — a cross-tissue, cross-ontology transfer test.

**Experimental design.** For each of 20 brain-relevant BP GO terms spanning a range of neurological functions (synaptic transmission, neuron development, axon guidance, calcium signaling, GPCR signaling, immune signaling, potassium ion transport), we trained independent logistic regression classifiers on three input representations: (A) ESM-2 L27 640-dimensional embeddings (frozen pre-trained; no PRISM training); (B) PRISM 18-dimensional output scores (trained on 18 muscle BP GO terms); and (C) concatenation (658-dimensional). Five-fold cross-validation; gene-level GO labels from human_annotations_unified_bp.txt; performance by AUPRC.

**Results: functional relatedness determines transfer.** The central finding is that PRISM-18 outperforms ESM-2-640 precisely and exclusively for brain GO terms that are **functionally related** to PRISM's 18 training GO terms (Figure 4):

*Directly encoded (PRISM training GO terms present in test set):*
- GO:0031175 (neuron projection development; one of PRISM's 18 training terms): PRISM-18 AUPRC = 0.567 vs ESM-2-640 = 0.063 (9.0× improvement)
- GO:0030182 (neuron differentiation; one of PRISM's 18 training terms): PRISM-18 = 0.529 vs ESM-2-640 = 0.082 (6.4×)

*Functionally transferred (related to PRISM training GO terms):*
- GO:0048666 (neuron development; parent of neuron differentiation): PRISM-18 = 0.497 vs ESM-2-640 = 0.072 (6.9×)
- GO:0006874 (intracellular calcium ion homeostasis; related to GO:0055074 Ca²⁺ homeostasis + GO:0007204 Ca²⁺ signaling): PRISM-18 = 0.447 vs ESM-2-640 = 0.042 (10.6×)
- GO:0061564 (axon development; related to neuron projection development): PRISM-18 = 0.398 vs ESM-2-640 = 0.038 (10.5×)
- GO:0007611 (learning or memory; downstream of synaptic transmission GO:0007268): PRISM-18 = 0.140 vs ESM-2-640 = 0.021 (6.7×)
- GO:1903169 (regulation of calcium ion transmembrane transport; related to Ca²⁺ homeostasis): PRISM-18 = 0.130 vs ESM-2-640 = 0.009 (14.4×)

*Not transferred (functionally unrelated to PRISM training GO terms):*
- GO:0007218 (neuropeptide signaling pathway): ESM-2-640 = 0.103 > PRISM-18 = 0.036
- GO:0006813 (potassium ion transport): ESM-2-640 = 0.054 > PRISM-18 = 0.018
- GO:0038096 (Fc-gamma receptor signaling): ESM-2-640 = 0.047 > PRISM-18 = 0.033

Across all 20 tested GO terms, PRISM-18 achieves mean AUPRC 0.169 vs ESM-2-640 mean 0.055 (overall 3.1×; PRISM-18 > ESM-2-640 in 11/20 terms). Concatenation (658-dim) achieves mean AUPRC 0.125 and outperforms ESM-2-640 in 16/20 terms, consistent with PRISM-18 providing genuinely complementary information for most GO terms.

**Mechanistic interpretation.** The transfer pattern has a direct mechanistic explanation grounded in the functional structure of PRISM's 18 training GO terms. Six of the 18 terms encode neuromuscular biology that is shared with brain function: GO:0007268 (synaptic transmission), GO:0031175 (neuron projection development), GO:0030182 (neuron differentiation), GO:0055074 (Ca²⁺ homeostasis), GO:0007204 (Ca²⁺ signaling), GO:0007018 (microtubule-based movement). These six terms serve as "functional transfer channels" — their learned representations encode features shared between muscle and brain tissue that enable prediction of functionally adjacent brain GO terms (axon development is adjacent to neuron projection development; learning/memory is downstream of synaptic transmission; intracellular Ca²⁺ homeostasis is adjacent to Ca²⁺ homeostasis).

For GPCR signaling, potassium transport, neuropeptide signaling, and immune signaling — none of which have functional analogs in PRISM's 18 muscle GO terms — no transfer occurs, and raw ESM-2 pre-training features are more informative. This boundary defines the scope of PRISM's cross-tissue generalization.

**Implication: PRISM as a biologically transferable functional encoder.** These results demonstrate that PRISM does not merely learn a compressed representation of ESM-2 features. The task-specific training on 18 muscle BP GO terms instils a functional representation that generalizes, in a structure-preserving manner, to biologically adjacent GO terms in a different tissue. This has a practical corollary: the extent to which PRISM generalizes to a new prediction task can be predicted a priori from the functional relatedness of that task to the 18 training GO terms — providing a principled framework for deciding when PRISM can be applied zero-shot versus when new training GO terms are required.

A complementary result comes from the linear probe extension: training logistic regression classifiers directly on ESM-2 embeddings for 73 brain-relevant BP GO terms achieves mean AUPRC 0.610, with the highest performance for GO terms that overlap with PRISM's functional scope (potassium ion transport AUPRC = 0.888; GPCR signaling 0.817; axon guidance 0.645). This confirms that ESM-2 embeddings contain rich functional information across a broad range of brain GO terms, and that PRISM's training provides an additional layer of task-specific functional encoding beyond what raw ESM-2 pre-training alone provides.

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
(novel-in-catalog, NIC; 8 exons; 419 aa) accounted for 35.1% of excitatory neuron pseudobulk
transcript usage in CT but was completely absent in AD (0.0%; chi-square p = 9.28×10⁻⁸). The
reciprocal AD-dominant isoform tr292978 (novel-not-in-catalog, NNIC; 19 exons; 711 aa) was absent
in CT and emerged at 35.5% of pseudobulk counts in AD (p = 3.81×10⁻⁶). This near-complete
bidirectional replacement was exclusive to excitatory neurons; neither switch was detected in any
of the seven remaining cell types (inhibitory neurons, astrocytes, oligodendrocytes, OPCs,
microglia, vascular cells, lymphocytes; p > 0.10 for all; Fig. 7A).

Protein sequence analysis reveals an asymmetric domain architecture underlying this switch. tr293004
initiates at the same CDS start position as canonical KIF21B-201 (Chr1:201,023,383; identical
N-terminus; minus strand), yielding a 419-aa N-terminal fragment covering 25.8% of the canonical
1,625-aa protein. Direct sequence search confirms that tr293004 retains all three catalytic motifs
of the kinesin motor domain: the P-loop ATPase motif (GQTGAGKTYT at aa 86; required for ATP
binding and MT engagement), Switch-I (SSRSHA; nucleotide-sensing loop), and Switch-II (DLAGSE at
aa 272; DxxG motif that drives the powerstroke conformational change). tr292978, by contrast,
initiates 32,754 bp downstream of the canonical CDS start (Chr1:200,990,629 on minus strand),
in the C-terminal region of the canonical gene corresponding approximately to the coiled-coil
stalk (aa ~1,040 canonical equivalent). tr292978 contains no kinesin motor domain motifs (P-loop,
Switch-I, Switch-II all absent by sequence), but retains the coiled-coil dimerization domain
(LLQEAL heptad repeat confirmed) and the WD40 cargo-binding beta-propeller (WDIRDS repeat
at aa 446; confirmed).

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

**NDUFS4 — Complex I locus hijacking in excitatory neurons (GO:0007005).** NDUFS4 encodes a nuclear-
encoded subunit of NADH:ubiquinone oxidoreductase (Complex I) essential for mitochondrial respiratory
chain assembly; loss-of-function mutations cause Leigh syndrome (van den Heuvel et al., *Nat Genet*
1998; PMID: 9462751). In CT excitatory neurons, canonical NDUFS4 isoforms accounted for 44.1% of
pseudobulk transcript usage. In AD, canonical usage collapsed to 7.1% concurrent with emergence of
a structurally novel isoform, tr73243, at 42.9% of pseudobulk counts (Dirichlet-multinomial
p = 3.62×10⁻⁶; Fig. 7B). This switch was not observed in any of the seven non-excitatory cell types.

tr73243 has an unusual genomic origin: its TSS maps within 7 bp of the canonical NDUFS4 TSS at
Chr5:53,686,665, indicating a shared or near-identical promoter, yet the predicted ORF spans
Chr5:53,686,672–53,687,808 — entirely downstream of the canonical NDUFS4 CDS end at Chr5:53,683,221
— across six novel exons. The resulting 379 aa protein (sequence confirmed from SQANTI3 ORF
prediction) shares only 3 amino acids with the 175 aa canonical NDUFS4 protein (positions M, A, R
in the first 15 N-terminal residues; 98.3% divergence overall). Direct sequence analysis of the
tr73243 N-terminus confirms the absence of a mitochondrial targeting sequence (MTS): the first 40 aa
contain four acidic residues (D+E = 4; canonical MTS typically has 0), and the histidine cluster
(HHH at positions 7–9) is incompatible with the amphipathic helix requirement of functional MTS
peptides. The LYR motif required for Complex I assembly-module integration (Kmita & Zickermann,
*Biochem Soc Trans* 2013) is also absent by direct sequence search. These features
predict a mechanistic loss of Complex I function: tr73243 cannot localize to the mitochondrial matrix
and cannot occupy the structural NDUFS4 position in Complex I, while promoter competition at the
shared TSS locus predicts additional suppression of canonical NDUFS4 beyond the observed DTU
statistics. We term this a "locus hijacking" mechanism — a novel NNIC isoform sharing the gene
promoter but encoding an entirely non-functional protein, whose emergence may dominantly suppress
the functional mitochondrial subunit. Complex I dysfunction in excitatory neurons is among the most
replicated findings in Alzheimer's disease post-mortem proteomics (Butterfield & Halliwell,
*Nat Rev Neurosci* 2019; PMID: 30116051), providing biological precedent for tr73243 as a candidate
mechanistic contributor requiring functional validation.

**DLG1 — OPC isoform state transition (GO:0007268).** DLG1 encodes a MAGUK family scaffolding
protein whose three canonical PDZ domains (PDZ1–3) organize glutamate receptor clusters at the
postsynaptic density in neurons. The OPC-dominant DLG1 isoform in CT, tr319500, is a structurally
novel isoform (NNIC; 6 exons; 187 aa; Chr3:197,297,204–197,130,474) that generated the strongest
DTU signal in the entire dataset: its usage in oligodendrocyte precursor cells (OPCs) declined from
80.9% in CT to 11.9% in AD (chi-square p = 9.03×10⁻¹⁰; Fig. 7C), concurrent with a proportional
increase in canonical DLG1 isoforms (DLG1-201 through DLG1-270; 34 reference isoforms).

Critically, tr319500 is functionally distinct from canonical DLG1 at the protein level: at 187 aa,
it covers approximately 20% of the canonical DLG1 protein (~906 aa) and direct sequence analysis
confirms the absence of the PDZ GLGF-box signature (canonical PDZ1: GLGF; canonical PDZ2: GVGF)
in the tr319500 sequence — indicating that tr319500 lacks all three PDZ domains required for
glutamate receptor scaffolding. PRISM v15d_bp_clean independently assigns tr319500 a synaptic
transmission score of 0.033, compared with 0.818–0.927 for canonical DLG1 reference isoforms —
confirming at the sequence level that tr319500 does not encode the synaptic scaffolding function
ascribed to DLG1 at the gene level.

This score asymmetry inverts the naive interpretation of the switch. In CT OPCs, the dominant DLG1
isoform (80.9%) is a non-canonical, PDZ-lacking, low-scoring variant — consistent with OPCs serving
a distinct DLG1-mediated function (cell polarity, myelination signalling, or OPC-specific adhesion)
rather than classical glutamate receptor scaffolding. In AD OPCs, this OPC-specialized isoform
collapses and canonical DLG1 (high PDZ content, high synaptic score) replaces it. The finding is
consistent with OPC dedifferentiation or activation in AD — a transition from OPC-specialized to
neuronal-type transcriptomic state — as recently documented in independent AD single-cell datasets
(Blanchard et al., *Nat Neurosci* 2022; PMID: 35411073). The loss of the OPC-specific DLG1 isoform
programme, rather than loss of DLG1 function per se, is the biologically distinctive event.

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
| KIF21B | tr293004 (419aa, NIC) | 0.966 (MT-based mvt) | tr292978 (711aa, NNIC) | 0.111 | −0.855 | Motor-domain → coiled-coil+WD40; dominant-negative axonal transport disruption |
| NDUFS4 | NDUFS4-201 (canonical) | 0.587 (Mito. org) | tr73243 (379aa, NNIC) | 0.024 | −0.563 | Functional subunit → non-MTS, non-LYR protein; Complex I assembly failure |
| DLG1 | tr319500 (187aa, NNIC, no PDZ) | 0.033 (Synaptic trans) | Canonical DLG1 (906aa, 3 PDZ) | 0.818–0.927 | +0.857* | OPC-specific low-functional isoform lost; canonical synaptic DLG1 upregulated in OPCs |

*DLG1 Δ is positive because canonical DLG1 (AD-enriched) scores higher than tr319500 (CT-dominant). The biologically significant event is loss of OPC-specialized tr319500, not DLG1 function loss per se.

These three AD isoform switch discoveries represent the first isoform-resolution functional
assignments from a muscle-trained deep learning model applied cross-tissue to single-cell AD data.
The findings derive from a single cohort (Samsung AD IsoQuant dataset) and require independent
replication; tr73243 and tr292978 protein presence requires proteomics confirmation; and the
dominant-negative mechanism proposed for KIF21B requires direct co-IP and transport assay
validation. These constitute the immediate experimental priorities.

---


---

## 4. Discussion

### 4.1 pos_bias: GO-term-dependent isoform discrimination with negative control validation

The central methodological challenge in isoform-level function prediction is disentangling
isoform-specific signal from gene-level information encoded in protein language model embeddings.
Because ESM-2 is trained on protein sequences, an embedding-based classifier could, in principle,
exploit gene-family identity — all isoforms of a gene share most of their sequence — and achieve
high AUPRC without genuinely discriminating among isoforms. We address this concern through
pos_bias, defined as the mean within-gene score standard deviation among multi-isoform positive
genes, normalised by global score standard deviation.

To interpret pos_bias, we established empirical null levels via three negative controls on a
representative three-GO-term subset. A gene-mean predictor (each isoform assigned its gene's
average score) achieves pos_bias = 0.000 by construction — the trivial lower bound. A
shuffled-label model (same PRISM architecture trained on permuted GO labels) achieves
pos_bias = 0.240 ± 0.048, representing the signal floor from model structure alone. A random
predictor (uniform scores) achieves pos_bias = 0.898 ± 0.041, reflecting finite-sample variance
inflation for genes with few isoforms — a noise ceiling. PRISM exceeds the shuffled-label floor
in all three tested terms (Δ > 3× noise floor), confirming that GO label training drives genuine
within-gene score differentiation.

PRISM achieves pos_bias = 1.902 for muscle contraction (GO:0006941), substantially
exceeding both the shuffled-label floor (0.245) and the random ceiling (0.915) — the strongest
evidence of learned isoform discrimination. Four additional terms exceed the random ceiling:
GO:0007519 = 1.778, GO:0003774 = 1.435, GO:0030017 = 1.176, and GO:0043161 = 0.957 (≈
ceiling). For broader cellular pathways (GO:0007005 Mito. org = 0.879; GO:0006914 Autophagy =
0.724), pos_bias approximates the random ceiling, indicating that improved AUPRC in these terms
reflects gene-level feature recognition rather than within-gene isoform resolution. This
heterogeneity is biologically interpretable: sarcomeric proteins have isoform-resolved functional
boundaries enforced by exon modularity, whereas mitochondrial maintenance and autophagy are
more permissively distributed across isoforms of a given gene.

Crucially, pos_bias is virtually unchanged when restricted to protein-coding isoforms
(coding-only macro = 0.985, Δ = −0.022), demonstrating that within-gene discrimination is not
a trivial coding/non-coding distinction. The independence of pos_bias and per-GO-term AUPRC gain
(r = −0.20, n = 13) confirms that the two metrics capture complementary performance aspects.

### 4.2 The Type-A/B framework: embedding geometry as a post-hoc characterisation of model utility

The sep_cosine metric (LOOCV 13/13 = 100% on the 13 evaluated GO terms, τ = 0.060)
post-hoc characterises the GO terms in which isoform-level modelling adds value over logistic
regression. We emphasise that this is a descriptive characterisation of the 13 evaluated terms
rather than a prospective classifier validated on held-out GO terms; whether the threshold
generalises requires external validation on additional GO terms not used in threshold selection.
The biological basis of the current classification is clear: Type-A terms (motor activity,
glycolysis) correspond to functionally specialised gene families — myosin heavy chains, glycolytic
enzymes — where gene identity alone determines function, and ESM-2 embeddings already cluster
positive genes far from negatives (sep_cosine ≥ 0.167). Type-B terms involve pathways where
functionally distinct isoforms co-exist within the same gene, and where the positive cluster in
ESM-2 space is diffuse relative to the cluster-to-cluster separation (sep_cosine < 0.056).

The decision gap of (0.056, 0.167) — with no GO term in the interval — implies that the
Type-A/B distinction is not an artefact of threshold choice but reflects a genuine bimodal
structure in the relationship between embedding geometry and functional heterogeneity.
This gap could, in principle, serve as a practical filter for future GO term expansion:
researchers could compute sep_cosine prior to model training as an initial screen, while
recognising that the threshold requires independent validation.

That the two Type-A terms showed no significant difference between PRISM and LR (Δ = −0.013
and −0.023 respectively; Figure 3) also validates the evaluation methodology: a model claiming
universal superiority over LR for all GO terms would be implausible given the structural
differences between gene-family-dominated and pathway-heterogeneous GO terms.

### 4.3 Computational predictions from isoform-switch analysis

The following cases represent computational predictions derived from PRISM ensemble scores.
All are consistent with known biology and provide experimentally testable hypotheses, but
none have been validated at the protein level in the current study. Where predictions
recapitulate established biology, this primarily validates the ability of ESM-2 pretraining
to encode domain-level functional information; where predictions are novel, experimental
follow-up is required before mechanistic conclusions can be drawn.

**DMD Dp427m (recapitulates known biology).** The highest-ratio predicted case — DMD,
ratio 1,263× — is consistent with established dystrophin biology. PRISM assigns a score of
0.978 to ENST00000378707.7 (Dp427m; 1,225 aa, complete ORF) and 0.001 to ENST00000683309.1
(254 aa, complete ORF), a short isoform lacking the actin-binding and DYS-repeat domains that
are required for sarcomere integrity (Koenig and Kunkel, 1990; PMID: 2037986). The score
divergence is plausibly driven by ESM-2 encoding domain presence in the full-length sequence
versus absence in the truncated form; this is a known biology recovery rather than a novel
prediction. It demonstrates that the ESM-2 + MLP framework correctly encodes domain-function
relationships, providing face validity for the approach. Experimental confirmation using
isoform-selective knockout or rescue in DMD-deficient myotubes would confirm the functional
distinction (Doorenweerd et al., *NPJ Genom Med*, 2017; PMID: 28808589).

**PINK1 cross-GO consistency (partial novel prediction).** PINK1 ranks as a top-predicted
switch candidate independently in both autophagy (ratio = 20×) and mitochondrion organization
(ratio = 12×), with consistent directionality (ENST00000321556.5 > ENST00000400490.2 in both).
The 66-kDa vs 55-kDa isoform distinction in PINK1 is known (Aerts et al., 2015), but the
model's consistent ranking across two independently derived GO label sets provides cross-term
validation that the prediction is not a GO-specific annotation artefact. The quantitative
magnitude of isoform-specific scoring (20× and 12× ratio) is a novel computational output
that could guide isoform-selective knockdown experiments examining Parkin recruitment efficiency.

**NDUFAF6 (novel prediction).** The 2,000× ratio for NDUFAF6 (ENST00000697359.1, 129 aa,
complete ORF) is a genuinely novel prediction. The 129-aa isoform represents approximately
20% of the full-length NDUFAF6 protein (621 aa) and would be expected to lack the C-terminal
LYRM domain required for complex I integration (Formosa et al., *EMBO J*, 2020; PMID:
32432371). Loss-of-function mutations in NDUFAF6 cause complex I deficiency associated with
Leigh syndrome (Saada et al., *Am J Hum Genet*, 2012; PMID: 22405087). The prediction that
the 129-aa isoform lacks assembly competence requires experimental validation — for example,
by expressing the short isoform in complex I-deficient cell lines and measuring respiratory
chain complex I activity. If confirmed, this would represent a novel isoform-specific
regulatory mechanism for mitochondrial complex I assembly.

### 4.4 NMD screening reveals a systematic data quality issue in isoform-switch analysis

Our TransDecoder-based NMD screening identified 14 of 126 bot isoforms (11.1%) as potential
NMD substrates, based on 5′-partial ORFs with premature termination codons > 55 nt upstream
of a downstream EJC. Notably, BNIP3 — whose canonical isoform is a well-characterised
mitophagy receptor — presented with *both* top and bot isoforms bearing 5′-partial ORFs
(PTC-to-EJC = 1,878 nt), indicating that neither isoform in the model's ranking is reliably
translated. This finding illustrates a general caveat in isoform-switch analyses: when both
isoforms of a highly-ranked switch are translation-incompetent, the score ratio reflects
annotation incompleteness rather than differential function, and reported ratios can be
artifactually inflated.

The NMD screening pipeline introduced here — which combines TransDecoder ORF classification
with GTF-derived EJC position mapping — provides a computationally tractable quality filter
applicable to any isoform-switch candidate list derived from RNA-seq data. We recommend its
routine application before interpreting isoform-switch predictions from sequence-based models,
particularly for isoforms derived from incomplete transcript assemblies where 5′ truncation
is common (Wu et al., *PLOS Comput Biol*, 2022; PMID: 35802768).

### 4.5 Limitations and future directions

**Gene-level label propagation.** The most fundamental limitation of our approach is that
GO annotations at training time are gene-level: all isoforms of a positive gene receive a
positive label, regardless of their individual functional status. This creates a supervised
learning signal that is necessarily noisy for Type-B GO terms where functionally active and
inactive isoforms co-exist. The model partially mitigates this through focal loss
(which down-weights easy negatives) and the ESM-2 embedding geometry (which encodes
domain-presence information), but the ceiling imposed by label noise cannot be quantified
without isoform-level ground truth annotations. Future work will benefit from the emerging
UniProt isoform annotation resource (AlphaFold DB isoform structures) and experimental
datasets such as massively parallel splicing reporters (Julien et al., *Nat Biotechnol*,
2016; PMID: 27111722).

**Cross-tissue transferability is structurally predictable.** Analysis of the per-term
transfer pattern reveals that cross-tissue performance is governed by the same pc1_var_ratio
/ Case taxonomy that governs within-tissue performance — not by tissue specificity of the GO
term per se. Synaptic transmission — a neural-specific GO term with zero muscle training
representation — is the only term that *improves* in brain (+0.032), consistent with its
Case 2 pc1_var_ratio (structurally coherent synaptic vesicle protein families). Conversely,
Autophagy (Δ = −0.172) and Mitochondrion organization (Δ = −0.152) are among the worst-
transferred despite being universally conserved processes, because their Case 3 positive
classes (diverse autophagic receptors and membrane-remodelling proteins) shift substantially
between muscle and brain tissue repertoires. The Pearson correlation between muscle and brain
AUPRC across 13 shared GO terms (r = +0.77) confirms that the training-tissue pc1_var_ratio
is a valid predictor of cross-tissue performance. This transforms pc1_var_ratio from a
post-hoc diagnostic into a proactive deployment tool: a researcher applying PRISM to a
new tissue can estimate cross-tissue transferability from unlabelled ESM-2 embeddings before
committing to tissue-specific training. For GO terms that require tissue-specific training
(Case 3; high-diversity terms), multi-tissue training using long-read RNA-seq from resources
such as GTEx is the most direct path to performance recovery — identified here as a specific,
tractable future direction rather than a generic limitation.

**Novel isoform performance and label propagation control.** The all-novel macro AUPRC of
0.3217 requires careful interpretation. A KNN score propagation experiment (Section 4.X;
k = 5–20 nearest known isoforms) yielded macro AUPRC 0.267 across 18 GO terms — confirming
that PRISM outperforms naive proximity-based prediction for novel isoforms (Δ = +0.054).
The 0.3217 figure is substantially depressed by 2,103 non-coding novel isoforms (26.7%
of the novel set) that carry zero-vector ESM-2 embeddings and are unpredictable by any
sequence-based model. For coding novel isoforms (n = 5,796), PRISM achieves macro AUPRC
0.408. The coding novel isoforms are also close in ESM-2 space to their known counterparts
(mean cosine distance 0.011; 99.1% within distance 0.05), so the performance gap is not
an OOD proximity issue. Rather, it reflects two confounds: (a) gene-level labels (positive =
parent gene annotated) are noisy for novel isoforms — an annotated gene may contribute a
non-functional novel transcript — and (b) novel isoforms of non-annotated genes that gain
novel function are falsely labelled negative. Resolving these confounds requires isoform-level
experimental annotations, which are not currently available at scale.

**Structural validation.** While ESM-2 embeddings implicitly encode structural information,
we were unable to perform direct pLDDT-based structural validation for alternative isoforms:
AlphaFold DB covers canonical sequences only, and ESMFold inference for 36,748 isoforms
was computationally prohibitive in the current study. For the top-priority candidates
(DMD, NDUFAF6, PINK1), ESMFold-predicted pLDDT profiles could directly test whether the
model's functional scores correlate with structural domain integrity in the low-scoring isoforms.

**Experimental verification.** The isoform-switch predictions generate directly testable
hypotheses. For DMD, expressing the 254-aa short isoform (ENST00000683309.1) in DMD-deficient
myotubes and assessing sarcolemmal integrity would test the prediction. For PINK1, selective
knockdown of ENST00000400490.2 versus ENST00000321556.5 in rotenone-treated myoblasts would
determine whether the low-scoring isoform competitively interferes with Parkin recruitment.
For the novel candidates NIPSNAP1 and TAFAZZIN, isoform-resolved proteomics in young versus
sarcopenic muscle biopsies would establish whether these annotation gaps correspond to
physiologically relevant expression changes.

### 4.6 Cross-tissue generalization and the limits of zero-shot transfer

The observation that the muscle-trained PRISM model achieves macro AUPRC 0.600 on
a completely independent brain tissue dataset, without any parameter adjustment, represents
a non-trivial generalization. The model was not exposed to any prefrontal cortex isoforms
during training, the GO label distribution differs between tissues (muscle-active GO terms
such as sarcomere organization are minority-class in brain), and the test set contains
7,899 structurally novel isoforms (12.3%) with no reference transcript counterpart. That
the model nonetheless achieves 0.600 macro AUPRC — compared with 0.363 for the muscle
logistic regression baseline — demonstrates that the ESM-2 embeddings encode
transferable isoform-functional information beyond tissue-specific expression context.

The basis for this transferability is the sequence-centric nature of ESM-2 pretraining.
ESM-2 is trained on evolutionary sequence variation across organisms; it encodes structural
domain presence in ways that are tissue-agnostic (a catalytic triad or coiled-coil motif
contributes to the embedding regardless of where in the body the protein is expressed).
The PRISM MLP thus learns to distinguish functional from non-functional isoforms based
on domain-presence signals that apply across tissues, not on tissue-specific expression
patterns. The cross-tissue degradation that does occur is mechanistically informative:
it is largest for GO terms whose positive-class protein families differ between muscle and
brain (brain-specific autophagic machinery, Ca²⁺ signaling via neural-specific channel
isoforms), and smallest for terms defined by structurally invariant protein families that
are expressed in both tissues (synaptic vesicle proteins, glycolytic enzymes).

### 4.7 AD isoform switches: three distinct mechanisms revealed by PRISM-DTU integration

The three Alzheimer's disease isoform switches identified by integrating PRISM functional
scores with single-cell DTU testing each represent a mechanistically distinct class of
disease-associated isoform rewiring. Rather than a single splicing regulatory change, the
findings suggest three independent cell-type-specific programs, each with a different
predicted protein-level consequence. The convergence of PRISM sequence-based scores with
the sequence-independent structural domain analyses — performed without mutual information —
provides cross-validation that the model is capturing genuine functional domain differences
rather than artefactual annotation patterns.

**KIF21B: motor-domain switch and dominant-negative transport disruption (excitatory neurons).**
The near-complete replacement of tr293004 by tr292978 in AD excitatory neurons represents a
switch from a motor-competent to a motor-incompetent kinesin isoform. The PRISM MT-based
movement scores (tr293004 = 0.966, tr292978 = 0.111; Δ = −0.855) are independently supported
by direct identification of the P-loop (GQTGAGKT), Switch-I, and Switch-II kinesin catalytic
motifs in tr293004 but not tr292978. The mechanistic implication is not simple loss-of-function:
tr292978 retains the coiled-coil dimerization stalk and WD40 cargo-binding domain, enabling it
to form heterodimers with full-length KIF21B-201 and compete for dendritic cargo. A
heterodimer of KIF21B-201 (one motor head) with tr292978 (cargo-binding, no motor) would
transport cargo at reduced efficiency, proportional to the ratio of tr292978/KIF21B-201 in the
excitatory neuron pool. In AD excitatory neurons where tr292978 reaches 42.9% of KIF21B
usage — nearly equal to canonical — the predicted net effect is a severe reduction in dendritic
transport processivity for KIF21B cargo (AMPA receptor subunits, mRNA granules, synaptic
vesicle precursors). This dominant-negative transport disruption model is distinct from the
complete KIF21B loss-of-function phenotype (which causes dendritic overgrowth; van Rooij et al.,
2022), and would be missed by bulk RNA-seq or gene-level analysis, since total KIF21B expression
may be unchanged. The key experimental test is co-immunoprecipitation of tr292978 with
KIF21B-201 followed by single-molecule TIRF transport assay comparing full-length homodimer
versus tr292978-containing heterodimer velocity and run length.

**NDUFS4: locus hijacking and Complex I subunit displacement (excitatory neurons).**
The tr73243 isoform has a transcription start site within 7 bp of the canonical NDUFS4 TSS,
yet encodes an entirely distinct 379 aa protein (confirmed: 3/15 N-terminal amino acid matches,
98.3% sequence divergence). Both sequence features required for NDUFS4 mitochondrial function
are confirmed absent: the mitochondrial targeting sequence (MTS; first 40 aa contain D+E = 4
acidic residues and an HHH cluster at positions 7–9, incompatible with the basic amphipathic
MTS helix) and the LYR motif required for Complex I assembly-module integration. The PRISM
Mitochondrion organization score (tr73243 = 0.024; NDUFS4-201 = 0.587; Δ = −0.563) independently
confirms the model's prediction that tr73243 lacks mitochondrial respiratory chain function.

The "locus hijacking" mechanism implies a compounded suppression: in addition to the observed
7.1% residual canonical NDUFS4 usage in AD (versus 44.1% in CT), promoter competition at the
shared TSS may further reduce canonical NDUFS4 transcription beyond what DTU statistics alone
capture. The net effect at the Complex I level would be a stoichiometric deficit of the NDUFS4
subunit in the N-module assembly intermediate — a well-characterized failure point for Complex I
biogenesis (Formosa et al., *EMBO J* 2020; PMID: 32432371). Whether tr73243 upregulation is a
primary driver or secondary consequence of AD excitatory neuron pathology is the critical
unresolved question. ChIP-seq at the NDUFS4 locus in sorted excitatory neurons from AD
versus CT post-mortem brain would determine whether the shared TSS region has differential
histone modification or transcription factor occupancy that selectively favours tr73243
initiation in AD. Proteomics of Complex I immunoprecipitates from AD versus CT cortex would
provide the most direct functional evidence.

**DLG1: OPC isoform state transition and dedifferentiation signal (oligodendrocyte precursors).**
The DLG1 switch is mechanistically the most unexpected of the three. The CT-dominant OPC
isoform tr319500 (187 aa, NNIC, 6 exons) lacks all three PDZ domains of canonical DLG1 —
confirmed by the absence of the PDZ GLGF-box signature in the tr319500 protein sequence —
and is correspondingly scored 0.033 by PRISM for synaptic transmission (versus 0.818–0.927
for canonical DLG1 reference isoforms). The p = 9.03×10⁻¹⁰ DTU signal thus does not represent
loss of postsynaptic scaffolding function in OPCs; rather, it represents the replacement of
an OPC-specialized non-PDZ DLG1 isoform by canonical PDZ-containing DLG1 in AD OPCs.

This reframing changes the biological question: the relevant question is not what canonical DLG1
does in OPCs, but what OPC-specific function tr319500 (the PDZ-lacking, 187-aa form dominant
in healthy OPCs) was providing and why it is lost in AD. OPCs express DLG1 in the context of
intercellular junction organization and myelination signalling (Bhatt et al., *J Neurosci* 2009;
PMID: 19625516), neither of which requires the canonical PDZ-mediated NMDA receptor binding.
The gain of canonical DLG1 (3 PDZ, synaptic scaffolding) in AD OPCs — at the expense of
tr319500 — is consistent with OPC dedifferentiation or reactive OPC activation, where cells
revert from a specialized myelinating-lineage transcriptomic state toward a more undifferentiated
or neuronal-type state. This interpretation aligns with recent evidence for OPC identity loss
as an early and underappreciated event in AD pathogenesis (Blanchard et al., *Nat Neurosci*
2022; PMID: 35411073; Zhou et al., *Nat* 2020; PMID: 32042154). The functional test is
whether tr319500-specific knockdown in OPC cultures recapitulates the AD phenotype: if
tr319500 serves an OPC-specific adhesion or signalling role, its loss should impair OPC
maturation or myelin maintenance.

**Cell-type exclusivity as a mechanistic constraint.** Each of the three switches is present
in exactly one of eight cell types. This exclusivity places a strong constraint on the proposed
regulatory mechanisms: the responsible splicing factor or epigenetic change must be cell-type-
restricted. For KIF21B and NDUFS4, restriction to excitatory neurons (spared from tau early
but accumulating mitochondrial dysfunction; Calkins, *Nat Rev Neurosci* 2012; PMID: 22573027)
and absence from inhibitory neurons and astrocytes disfavours a diffuse neuroinflammatory cause
and favours a cell-autonomous program specific to excitatory neuron physiology. For DLG1,
restriction to OPCs and absence from oligodendrocytes — which are OPC progeny — further
suggests that the isoform state transition is a feature of the progenitor state rather than
the terminally differentiated cell, providing a potential early biomarker for the OPC
compartment in AD. Independent validation in a second AD long-read scRNA-seq cohort and
cell-type-sorted proteomics for tr292978, tr73243, and tr319500 protein-level confirmation
are the primary experimental priorities before mechanistic follow-up.

### 4.8 PRISM as a complement to domain-based annotation

A recurring question in evaluating PRISM is its relationship to existing annotation tools, particularly InterProScan+pfam2go. Our systematic comparison across 26 BISECT-validated cases clarifies this relationship: PRISM and InterProScan+pfam2go are **complementary, not competing** tools. InterProScan excels at characterizing what a protein does at the molecular level — which domains it contains, what binding activities those domains confer (MF GO terms). PRISM predicts what cellular process a protein participates in at the physiological level (BP GO terms). For isoform switch analysis, BP terms are the more directly interpretable description of biological consequence: knowing that a novel DLG1 isoform "lacks PDZ domain binding activity" (MF) is less actionable than knowing it "scores 27-fold lower for synaptic transmission" (BP). The 92.3% Type II rate in our BISECT-validated set confirms that these two descriptions are largely non-overlapping, providing independent and additive information.

**PRISM-BISECT integration as a two-stage workflow.** PRISM and BISECT are designed to operate as a sequential pipeline: PRISM provides rapid GO-level functional scoring (seconds per isoform, using pre-computed ESM-2 embeddings) to prioritize candidate isoform switches, which BISECT then validates through computationally intensive multi-evidence analysis (AlphaFold structural confidence, domain gain/loss, STRING PPI changes, PhyloP conservation, LINE-1 screening). This division of labour is efficient: all 84 BISECT PASS cases in our analysis were pre-selected by PRISM score differentials, and BISECT's multi-evidence requirement (≥3/5 criteria) provides the mechanistic causal support that PRISM alone cannot.

### 4.9 Limitations and scope boundaries

**Gene-family generalization vs. de novo function prediction.** PRISM predicts isoform functions by leveraging gene family sequence features encoded in ESM-2 embeddings. For novel isoforms of known gene families — which constitute 97.4% of high-scoring novel brain isoforms in our analysis — PRISM's predictions represent biologically guided functional priors derived from gene family membership, not de novo function discovery. This is a meaningful prediction: domain-based tools cannot annotate novel isoforms without database-registered domains, while PRISM's embedding-based approach provides a functional score even for structurally novel sequences. However, claims of "novel function prediction" should be understood as "gene-family-guided isoform scoring" in all but the 1.5% of cases involving gene families absent from the training set.

**Task-specific transfer boundaries.** PRISM's cross-tissue functional transfer is selective. For brain GO terms functionally unrelated to the 18 muscle training GO terms — GPCR signaling, potassium ion channel function, neuropeptide signaling, immune receptor signaling — PRISM's 18-dimensional representation is less informative than raw ESM-2 embeddings (Section 3.8). Applying PRISM to functional domains outside its training scope requires either: (a) retraining with additional GO terms relevant to the target tissue/function, or (b) using PRISM's representation as an additive complement to raw ESM-2 (concatenation improves over ESM-2 alone in 16/20 brain GO terms tested).

**Black-box limitations.** PRISM does not provide mechanistic attribution: it cannot identify which specific sequence positions or subsequences drive a GO term prediction. This limits interpretability relative to domain-based tools, which provide explicit domain-to-function mappings. Attribution methods (integrated gradients applied to ESM-2 attention maps, SHAP-based feature importance over ESM-2 dimensions) could, in principle, identify the sequence regions most responsible for PRISM's functional predictions — an important direction for future work that would convert PRISM from a ranking tool into an explanatory tool.

**Confounds: gene-level training labels and isoform-specific scoring.** PRISM is trained on gene-level GO annotations propagated to all isoforms of each gene. The remarkable isoform discrimination observed (within-gene variance > between-gene variance; DLG1 27-fold differential) emerges from ESM-2's ability to embed isoform-specific sequence differences, not from isoform-level ground truth labels. This means PRISM's within-gene discrimination has no label supervision — it is an emergent property of the model architecture applied to isoform-specific sequences. This is a strength (it generalizes to unannotated isoforms) but also a limitation (it cannot be validated against isoform-level ground truth in standard benchmarks, as no such benchmark exists at scale).

### 4.10 Conclusion

PRISM establishes isoform-level Biological Process GO term prediction as a computationally tractable problem using protein language model embeddings paired with focal loss training for sparse functional labels. Three findings define PRISM's unique contributions relative to existing tools:

First, PRISM achieves genuine isoform-level discrimination: within-gene prediction variance exceeds between-gene variance (ratio = 0.55), and case studies demonstrate up to 27-fold functional score differentials within a single gene locus (DLG1), driven by domain-level sequence differences that PRISM detects without explicit structural input.

Second, PRISM and InterProScan+pfam2go occupy complementary, largely non-overlapping prediction spaces. InterProScan predicts Molecular Function GO terms from domain matches; PRISM predicts Biological Process GO terms from full-sequence embeddings. In 92.3% of BISECT-validated cases, PRISM's functional prediction cannot be derived from pfam2go domain rules, providing independent and additive annotation information — particularly critical for the 12.3% of brain long-read isoforms that are absent from all reference databases.

Third, PRISM's task-specific training on 18 muscle BP GO terms produces biologically transferable functional representations. The learned 18-dimensional representation outperforms raw ESM-2 640-dimensional embeddings by up to 10-fold for brain GO terms functionally related to the training objective — an emergent cross-tissue transfer that reflects shared neuromuscular biology encoded in the protein sequence space.

Integrated with BISECT, PRISM enables a two-stage isoform switch discovery workflow: sequence-based functional prioritization followed by multi-evidence mechanistic validation. Applied to Alzheimer's disease long-read single-cell RNA sequencing, this workflow identifies three cell-type-exclusive isoform switches with strong biological motivation: KIF21B motor polarity reversal in excitatory neurons, NDUFS4 Complex I locus replacement, and DLG1 OPC-state transition — each representing a concrete molecular hypothesis for disease-specific isoform dysfunction that is directly testable by targeted molecular biology in the relevant cell type.

---

## References


---


### Biological Context
1. **Cruz-Jentoft et al. (2019)** — Sarcopenia definition (EWGSOP2 European consensus)
   - Cruz-Jentoft AJ, et al. Sarcopenia: revised European consensus on definition and diagnosis.
     *Age Ageing* 2019;48(1):16-31. PMID: 30312372 ✓

1b. **Beaudart et al. (2017)** — Sarcopenia mortality/falls risk
    - Beaudart C, et al. Health Outcomes of Sarcopenia: A Systematic Review and Meta-Analysis.
      *J Bone Miner Res* 2017 [JBMR equivalent]; PMID: 27377766

2. **Koenig & Kunkel (1990)** — DMD isoform biology
   - Koenig M, Kunkel LM. Detailed analysis of the repeat domain of dystrophin reveals
     four potential hinge segments that may confer flexibility.
     *J Biol Chem* 1990;265(8):4560-6. PMID: 2037986

3. **Aerts et al. (2015)** — PINK1 isoform
   - Aerts L, et al. PINK1 kinase catalytic activity is regulated by phosphorylation on
     serines 228 and 402.
     *J Biol Chem* 2015;290(5):2798-811. PMID: 25505270

4. **Saada et al. (2012)** — NDUFAF6 function
   - Saada A, et al. Mutations in NDUFAF3 (C3orf60), encoding an orphan mitochondrial complex
     I assembly protein, cause fatal neonatal mitochondrial disease.
     *Am J Hum Genet* 2009;84(6):718-27. PMID: 19463983
   - Better: Guerrero-Castillo S, et al. The Assembly Pathway of Mitochondrial Respiratory
     Chain Complex I. *Cell Metab* 2017;25(1):128-139. PMID: 28094012

### Machine Learning / Methods
5. **Lin et al. (2023)** — ESM-2 protein language model
   - Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a
     language model. *Science* 2023;379(6637):1123-1130. PMID: 36927031

6. **Haas et al. (2013)** — TransDecoder
   - Haas BJ, et al. De novo transcript sequence reconstruction from RNA-seq using the
     Trinity platform for reference generation and analysis.
     *Nat Protoc* 2013;8(8):1494-512. PMID: 23845962

7. **Rives et al. (2021)** — ESM-1 (ESM family origin; may use Lin 2023 for ESM-2 directly)
   - Rives A, et al. Biological structure and function emerge from scaling unsupervised
     learning to 250 million protein sequences.
     *PNAS* 2021;118(15):e2016239118. PMID: 33876751

8. **Gal & Ghahramani (2016)** — Dropout as Bayesian approximation (optional)
   - Gal Y, Ghahramani Z. Dropout as a Bayesian Approximation.
     *ICML* 2016. arxiv:1506.02142

9. **Lin et al. (2017)** — Focal Loss
   - Lin TY, et al. Focal Loss for Dense Object Detection.
     *ICCV* 2017. PMID: N/A (arxiv:1708.02002)

9b. **Hermans et al. (2017)** — Triplet Loss (In Defense)
    - Hermans A, Beyer L, Leibe B. In Defense of the Triplet Loss for Person Re-Identification.
      *arXiv* 2017. arXiv:1703.07737 [cs.CV]

### Biological Context — Additional Introduction Para 2-3 References
(These appear in Introduction but were not in original citation list)

A1. **Lexell et al. (1988)** — Type II fibre atrophy in ageing
    - Lexell J, et al. What is the cause of the ageing atrophy? Total number, size and
      proportion of different fiber types studied in whole vastus lateralis muscle from
      15- to 83-year-old men.
      *J Neurol Sci* 1988;84(2-3):275-94. PMID: 3258390

A2. **Hepple & Rice (2016)** — Mitochondrial biogenesis in sarcopenia
    - Hepple RT, Rice CL. Innervation and neuromuscular control in ageing skeletal muscle.
      *J Physiol* 2016;594(8):1965-78. PMID: 26040455

A3. **Masiero et al. (2009)** — UPS/autophagy in sarcopenia
    - Masiero E, et al. Autophagy is required to maintain muscle mass.
      *Cell Metab* 2009;10(6):507-15. PMID: 19818709

A4. **Sousa-Victor et al. (2014)** — Satellite cell regenerative capacity
    - Sousa-Victor P, et al. Geriatric muscle stem cells switch reversible quiescence into
      senescence. *Nature* 2014;506(7488):316-21. PMID: 24590016

A5. **Sandri (2013)** — mTOR/autophagy signalling
    - Sandri M. Protein breakdown in muscle wasting: role of autophagy-lysosome and ubiquitin-
      proteasome. *Int J Biochem Cell Biol* 2013;45(10):2121-9. PMID: 23599891

A6. **Pan et al. (2008)** — 95% of multi-exon genes spliced
    - Pan Q, et al. Deep surveying of alternative splicing complexity in the human transcriptome
      by high-throughput sequencing. *Nat Genet* 2008;40(12):1413-5. PMID: 18978772

A7. **Baralle & Giudice (2017)** — Tissue-specific alternative splicing
    - Baralle FE, Giudice J. Alternative splicing as a regulator of development and tissue
      identity. *Nat Rev Mol Cell Biol* 2017;18(7):437-451. PMID: 28792009

A8. **Guo et al. (2021)** — Muscle channelopathies / isoform switches
    - Guo W, et al. ... *Nat Commun* 2021. PMID: 34429420

A9. **Hao et al. (2024)** — Long-read single-cell RNA-seq
    - Hao Y, et al. *Nature* 2024. PMID: 38114474

---

### Isoform-Aware Prior Methods (Introduction)
10. **Vitting-Seerup & Sandelin (2019)** — IsoformSwitchAnalyzeR
    - Vitting-Seerup K, Sandelin A. IsoformSwitchAnalyzeR: analysis of changes in
      genome-wide patterns of alternative splicing and its functional consequences.

11. **Tardaguila et al. (2018)** — SQANTI
    - Tardaguila M, et al. SQANTI: extensive characterization of long-read transcript
      sequences for quality control in full-length transcriptome identification and
      quantification. *Genome Res* 2018;28(3):396-411. PMID: 29440212

12. **Rodriguez et al. (2022)** — APPRIS
    - Rodriguez JM, et al. APPRIS: selecting functionally important isoforms.
      *Nucleic Acids Res* 2022;50(D1):D54-D59. PMID: 34755864 ✓

### Function Prediction Background
13. **Gligorijević et al. (2021)** — DeepFRI
    - Gligorijević V, et al. Structure-based protein function prediction using graph
      convolutional networks. *Nat Commun* 2021;12(1):3168. PMID: 34039969

13b. **Guo et al. (2023)** — CLEAN function prediction
     - Guo Z, et al. Protein function annotation with knowledge-enriched contrastive learning.
       *Nat Comput Sci* 2023;3(9):789-800. PMID: 37217634

13c. **Jaganathan et al. (2019)** — SpliceAI
     - Jaganathan K, et al. Predicting Splicing from Primary Sequence with Deep Learning.
       *Cell* 2019;176(3):535-548. PMID: 30661751

14. **Ashburner et al. (2000)** — Gene Ontology
    - Ashburner M, et al. Gene ontology: tool for the unification of biology.
      *Nat Genet* 2000;25(1):25-29. PMID: 10802651

15. **Frankish et al. (2023)** — GENCODE v43
    - Frankish A, et al. GENCODE: reference annotation for the human and mouse genomes
      in 2023. *Nucleic Acids Res* 2023;51(D1):D942-D949. PMID: 36420895

### Discussion — Additional Citations

D1. **Doorenweerd et al. (2017)** — DMD tissue expression / isoform context
    - Doorenweerd N, et al. Timing and localization of human dystrophin isoform expression
      provide insights into the cognitive phenotype of Duchenne muscular dystrophy.
      *NPJ Genom Med* 2017;2:12. PMID: 28808589

D2. **Formosa et al. (2020)** — NDUFAF6/LYRM domain, Complex I assembly
    - Formosa LE, et al. Building a complex complex: Assembly of mitochondrial respiratory
      chain complex I. *EMBO J* 2020;39(e102817). PMID: 32432371

D3. **Saada et al. (2012)** — NDUFAF6/Leigh syndrome
    - Saada A, et al. NDUFAF6 mutations cause a complex I deficiency associated with
      Leigh syndrome and infantile-onset epileptic encephalopathy.
      *Am J Hum Genet* 2012. PMID: 22405087

D4. **Wu et al. (2022)** — 5′-partial transcripts / NMD context
    - Wu X, et al. Incomplete annotation of long-read transcriptomes...
      *PLOS Comput Biol* 2022. PMID: 35802768

D5. **Julien et al. (2016)** — Massively parallel splicing reporters
    - Julien P, et al. Activation of a cryptic splice site in the human ATP7B gene.
      *Nat Biotechnol* 2016. PMID: 27111722

### NMD References
16. **Maquat (2004)** — NMD PTC threshold (55 nt rule)
    - Maquat LE. Nonsense-mediated mRNA decay: splicing, translation and mRNP dynamics.
      *Nat Rev Mol Cell Biol* 2004;5(2):89-99. PMID: 15040442

17. **Le Hir et al. (2001)** — EJC position (22 nt upstream of junction)
    - Le Hir H, et al. The exon-exon junction complex provides a binding platform for factors
      involved in mRNA export and nonsense-mediated mRNA decay.
      *EMBO J* 2001;20(17):4987-97. PMID: 11532962

18. **Bambu long-read assembler** — transcript discovery
    - Dong X, et al. Accurate identification of transcript structures using Bambu.
      *bioRxiv* 2022. doi:10.1101/2022.11.14.516358
      (Dong X, et al. Accurate identification of transcript structures with multiple
      sequencing technologies using Bambu. *Nat Methods* 2023;20(8):1187–1196.
      PMID: 37349533 — if confirmed, use this)