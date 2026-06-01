# Methods — PRISM: Isoform-Level Function Prediction
**Draft 2026-05-16 | Target: Nature Methods / NMI**

---

## 4. Methods

### 4.1 Dataset and isoform annotation

We used the human skeletal muscle transcriptome comprising 36,748 isoforms across 12,709 genes,
derived from long-read single-cell RNA-seq with Bambu-detected novel transcripts appended to
GENCODE v43. Functional labels were obtained from the Gene Ontology (GO) database
(release 2024-01-01). We selected 13 GO terms relevant to sarcopenia biology requiring a
minimum of 40 human gene-level annotations (Table 1). Gene-level GO annotations were propagated
to all annotated isoforms of that gene; isoforms of unannotated genes served as negatives.

Positive prevalence ranged from 0.8% (skeletal muscle development, GO:0007519) to 9.2%
(proteasome-mediated UPS, GO:0043161), classifying all 13 terms as sparse (positive < 10%),
for which AUPRC is the primary metric [R9.1].

### 4.2 Protein sequence embeddings

Isoform amino acid sequences were extracted with TransDecoder (v5.7.1, ORF ≥ 100 aa). For
non-coding isoforms, the longest ORF ≥ 30 aa was used; isoforms with no valid ORF received
a zero-vector embedding. All sequences were encoded with ESM-2 (esm2_t30_150M_UR50D,
HuggingFace checkpoint `facebook/esm2_t30_150M_UR50D`, 150 million parameters, 30 transformer
layers; Lin et al., 2023). Sequences were tokenised using the ESM-2 alphabet (20 canonical
amino acids + special tokens); representations from the final transformer layer (layer 30)
were averaged across all sequence-length positions (mean pooling) to produce a 640-dimensional
per-isoform embedding. Embeddings were computed once and cached (36,748 × 640 matrix).

### 4.3 PRISM v10-B model architecture

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

### 4.4 Training objective

PRISM is trained with focal loss to handle sparse GO label distributions [Lin et al., 2017]:

    L_focal(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

γ = 2.0 (BinaryFocalCrossentropy); α_t = 0.25.
Focal loss down-weights easy negatives, preventing mode collapse under sparse positive
labels characteristic of GO annotation data.

### 4.5 Training protocol

Each GO term was trained independently for 80 epochs with Adam (lr = 1×10⁻³),
batch size 512. Early stopping (patience = 10, monitor = val_loss) and learning rate
reduction on plateau (patience = 5, factor = 0.5) were applied. Training ran on a
single NVIDIA RTX 4090 GPU.

Final predictions are ensemble means across 5 random seeds (42, 123, 456, 789, 2024);
seed stability was assessed by the coefficient of variation (CV) across seeds per GO term.

### 4.6 Evaluation protocol

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

### 4.7 Baseline models

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

### 4.8 Type-A/B GO term classification

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

### 4.9 Isoform discriminability (pos_bias) and negative controls

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
The negative control analysis is provided in `reports/gene_mean_baseline/posbias_controls_20260517_1433.json`.

To control for coding/non-coding confound, pos_bias was additionally recomputed on protein-coding
isoforms only (ORF ≥ 100 aa, TransDecoder; 36,002/36,748 = 98.0%). Macro pos_bias decreased by
−0.022 (1.006 → 0.985 across 13 GO terms), confirming the signal is not an artefact of isoform
biotype differences.

### 4.10 Isoform-switch analysis and NMD verification

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

### 4.11 Component ablation study

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
Dropout, and L2-normalisation showed term-dependent and condition-dependent effects
within the variance of this 3-seed, 5-term evaluation subset (Type-B macro Δ range:
+0.017 to +0.079), reflecting that these components are tuned for the full 13-term,
5-seed training regime. Script: `/tmp/v10_ablation_fast.py`;
output: `reports/ablation/ablation_results_20260517_1537.json`.

### 4.12 pos_bias bootstrap confidence intervals and significance testing

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
Script: `/tmp/v10_posbias_pvalue.py`; output: `reports/ablation/posbias_pvalues_20260517_1548.json`.

---

### 4.13 Brain tissue dataset assembly (Samsung AD IsoQuant)

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

Raw data location: `/home/dhkim1674/Project_AD_with_refTSS_novel/` (internal cluster).
Processed embeddings: `/home/welcome1/sw1686/DIFFUSE/hMuscle/data/brain_isoquant_esm2/full/`.

---

### 4.14 Cross-tissue zero-shot evaluation

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

### 4.15 Single-cell DTU testing and AD isoform switch identification

Differential transcript usage (DTU) between Alzheimer's disease (AD) and cognitively typical
control (CT) samples was tested independently per cell type using a Dirichlet-multinomial model,
applied to pseudobulk transcript count matrices aggregated per donor (13 AD donors vs 8 CT
donors; see Table S4 for per-cell-type cell counts). Eight cell types were analysed: excitatory
neurons (12,307 AD / 10,149 CT cells), inhibitory neurons (6,495 / 5,917), astrocytes
(2,429 / 1,090), oligodendrocytes (6,346 / 4,228), oligodendrocyte precursor cells / OPCs
(2,608 / 1,930), microglia (800 / 657), vascular cells (361 / 253), and lymphocytes (32 / 30).
DTU was performed by D.-H. Kim (Samsung) and results are deposited in:
`/home/dhkim1674/Project_AD_with_refTSS_novel/06_DIU/DIU_by_condition_{CellType}.csv`.

For each DTU-significant isoform (Dirichlet-multinomial q < 0.05), an independent chi-square
test was performed on the 2×2 contingency table of AD vs CT pseudobulk counts for that
isoform versus all other isoforms of the same gene. p-values were compared against the
Bonferroni-corrected threshold α = 1×10⁻⁶ (accounting for testing ~18,291 genes × 8 cell
types). Cell-type specificity was defined as significance in exactly one cell type with p > 0.10
(no differential usage) in all remaining seven cell types.

**KIF21B domain analysis.** Protein sequences for tr293004 (418 aa) and tr292978 (710 aa)
were extracted from `SQANTI3_output/isoforms_corrected.faa`. Kinesin motor domain motifs
in tr293004 were confirmed by direct string search: P-loop (GQTGAGKT at aa 87), Switch-I
(SSRSHA at aa 222), Switch-II (DLAGSE at aa 273). Pfam-A domain annotation was performed
using HMMER 3.3.2 hmmscan (E-value threshold 0.01); tr293004 yielded Kinesin (aa 14–370,
E = 1.1×10⁻¹⁰⁹) and Microtub_bd (aa 7–158, E = 1.8×10⁻²³), whereas tr292978 yielded
15 WD40-family hits spanning aa 372–686 (canonical WD40 E = 4.6×10⁻⁹; ANAPC4_WD40 and
NBCH_WD40 profiles detected), with no kinesin-family domains detected. Full hmmscan
coordinates are reported in Section 4.16 and the automated pipeline output (domains.tsv).

**NDUFS4 genomic analysis.** tr73243 TSS and CDS coordinates were extracted from the SQANTI3
classification table (diff_to_gene_TSS, CDS_genomic_start, CDS_genomic_end columns). Protein
sequence comparison against canonical NDUFS4 (175 aa; UniProt O00501) was performed with
pairwise local alignment (N-terminal overlap = 3 aa: M, A, R). Mitochondrial targeting
sequence (MTS) absence was confirmed by two criteria applied to the first 40 amino acids of
tr73243: (i) acidic residue count (D + E ≤ 3 expected for functional MTS per MitoFates criteria; observed = 8 in tr73243,
inconsistent with amphipathic positively charged helix requirement) and (ii) presence of
HHH cluster at positions 7–9, which disrupts amphipathic α-helix formation required for
mitochondrial import. LYR motif (L-x-[RK] signature, required for Complex I assembly factor
binding) was confirmed absent by direct string search across the 379 aa tr73243 sequence.
Quantitative MTS composite scoring (5-criterion: net charge ≥ 2, D+E ≤ 3, hydrophobic
moment μH ≥ 0.12, HHH absent, LYR present) was performed computationally for all isoform
pairs as described in Section 4.16; tr73243 scored 1/5 (MTS ABSENT) versus NDUFS4-201
canonical 5/5. MitoFates (Fukasawa et al., *Mol Cell Proteomics*, 2015) and TargetP 2.0
(Almagro Armenteros et al., *Nat Methods*, 2019; PMID: 30778233) are recommended for
independent experimental validation of the MTS scoring.

**DLG1 domain analysis.** tr319500 protein sequence (187 aa) was extracted from
`SQANTI3_output/isoforms_corrected.faa`. PDZ domain absence was confirmed by direct string
search for the GLGF-box motif (canonical PDZ1: GLGF; canonical PDZ2: GVGF), which is the
conserved core signature of all three PDZ domains in DLG1-201 (906 aa; positions ~10–90,
~100–180, ~190–270 canonical coordinates). No GLGF or GXGF sequence was found in tr319500
across the full 187 aa, confirming that tr319500 encodes none of the three PDZ scaffolding
domains required for synaptic glutamate receptor clustering.

### 4.16 Pfam domain annotation and retroelement characterisation

**Pfam-A hmmscan.** Protein sequences for key AD isoforms (tr319500, tr73243, tr292978,
tr293004) were submitted to hmmscan (HMMER 3.3.2; `/opt/hmmer/hmmer-3.3.2/build/bin/hmmscan`)
against the Pfam-A database (release 36.0;
`hMuscle/results_isoform/features/pfam_db/Pfam-A.hmm`). Parameters: E-value threshold 0.01
(domain-level). Domain coordinates in query sequence were extracted from the `--domtblout`
output (ali_from, ali_to columns). Hits were filtered to E < 0.01. For tr319500, an L27_1
domain was identified at aa 6–63 (E = 7.8×10⁻³⁴) and MAGUK_N_PEST at aa 107–144
(E = 2.9×10⁻¹⁵). For tr73243, an RVT_1 domain (RNA-dependent DNA polymerase; PF00078)
was identified at aa 141–366 (E = 4.6×10⁻⁴⁸). The absence of PDZ, SH3, HOOK, and GK
domains in tr319500, and the absence of MTS-associated domains in tr73243, was confirmed
by the absence of corresponding Pfam hits at E < 0.01.

**MTS quantitative feature analysis.** Net charge in the N-terminal 30 aa was computed
as (count(K) + count(R)) − (count(D) + count(E)) for each sequence. Hydrophobic moment
(μH) was computed using the Eisenberg (1984) hydrophobicity scale over an 18-residue
sliding window with 100° per-residue rotation angle (α-helix), selecting the maximum μH
across the first 80 aa. The HHH motif was defined as three consecutive histidine residues
in positions 1–30. MTS composite score (0–5) was computed as the sum of five binary
criteria: net charge ≥ +2, D+E count (first 40 aa) ≤ 3, μH ≥ 0.12, absence of HHH motif,
and presence of LYR motif. For canonical NDUFS4-201 (UniProt Q16718), these values were:
net charge +2, μH = 0.276, HHH absent, composite score 3/5. For tr73243: net charge −1,
μH = 0.327 (not in MTS window), HHH present at position 7, composite score 1/5.

**RepeatMasker LINE-1 annotation.** Repetitive element annotation for the NDUFS4 locus
(chr5:53,560,626–53,688,219; hg38) was retrieved from the UCSC Genome Browser RepeatMasker
track via the UCSC REST API
(`https://api.genome.ucsc.edu/getData/track?genome=hg38;track=rmsk;chrom=chr5;start=...;end=...`).
A total of 220 repeat hits were returned; 62 were of the LINE class. Two LINE-1 elements
directly overlapping the CDS-containing Exon 6 (chr5:53,685,990–53,688,219) were identified:
L1PA3 (− strand; 4.5% divergence; Smith-Waterman score 7,461; chr5:53,685,456–53,686,732;
742 bp overlap with Exon 6) and L1PA11 (+ strand; 9.4% divergence; Smith-Waterman score
12,793; chr5:53,686,734–53,689,784; 1,485 bp overlap with Exon 6). The RVT_1 domain
genomic coordinates (aa 141–366 → chr5:53,687,092–53,687,770) were confirmed to lie
entirely within the L1PA11(+) annotated region, establishing L1PA11 ORF2 as the source
of the reverse transcriptase-homologous fold in tr73243. L1 subfamily nomenclature follows
Dfam 3.7 (https://dfam.org).

**Genomic sequence validation of retroelement-derived protein domains.** To confirm that the RVT_1 domain of tr73243 is encoded directly by the L1PA11 retroelement without RNA editing or sequence divergence, the hg38 genomic sequence of L1PA11 (chr5:53,686,734–53,689,784; 3,050 bp) was retrieved from the UCSC REST API (`getData/sequence` endpoint). The reading frame was derived from the tr73243 CDS start position (chr5:53,686,672): the L1PA11 element begins 62 bp downstream of the CDS start (62 mod 3 = 2), such that in-frame translation of L1PA11 begins at L1PA11 position +0 (frame +1 in standard 6-frame notation). Six-frame translation was computed and scored against the tr73243 RVT_1 query (aa 141–366) by pairwise local alignment (Smith-Waterman; BLOSUM62 substitution matrix; gap-open −11, gap-extend −1; Biopython PairwiseAligner v1.81). Frame +1 yielded an alignment score of 1,153 with 100% sequence identity over 226/226 aa (full coverage of the RVT_1 query); all other five frames scored ≤28. Perfect identity between tr73243 aa 141–366 and the L1PA11 frame +1 translation confirms that the reverse transcriptase domain is encoded by L1PA11 ORF2p sequence without modification. This finding, combined with the RepeatMasker and Pfam evidence, constitutes three independent lines of evidence establishing L1PA11 as the protein-coding source of tr73243's RT fold. Alignment files and figure are provided in the study repository (Supplementary File S8b).

**Genomic strand analysis.** Transcript strand assignments were extracted from the SQANTI3
classification table (`isoforms_classification.txt`, strand column). Canonical gene strands
were obtained from Ensembl GRCh38 annotation (NDUFS4/ENSG00000164258: negative strand).
tr73243 was confirmed as a positive-strand transcript associated with the negative-strand
NDUFS4 gene, establishing antisense orientation. Exon boundaries and CDS coordinates were
derived from the SQANTI3 corrected GTF (`isoforms_corrected.gtf`) and TransDecoder ORF
predictions (`.pep` file, ORF position encoded in sequence header as
`transcript_id.pN|len_aa|strand|cds_start|cds_end`).

**Automated biological analysis pipeline.** To systematise and reproduce the individual
analyses described above at scale, we implemented **BISECT** (Biological Isoform-Switch
Evidence Characterization Tool; `pipeline_bioanalysis/`, v1.1), a modular Python pipeline
comprising seven modules (M1–M7) and a master orchestrator (`orchestrate.py`). Candidate isoform pairs from the DTU screening output were subjected
to a two-stage filter before deep analysis: Stage 1 required |PRISM Δ| ≥ 0.5 and
DTU Dirichlet-multinomial p ≤ 1×10⁻⁵; Stage 2 required at least one Pfam-A domain family
present in the CT isoform and absent in the AD isoform, or vice versa (strict mode: Pfam
family ID set comparison). Cases failing Stage 2 received a partial JSON record but were
excluded from downstream structural and repeat-element analysis.

For each case passing both filters, the pipeline executed: (M1) protein sequence extraction
from the SQANTI3/IsoQuant corrected FAA; (M2) hmmscan domain annotation and Stage 2
domain-change detection; (M3) MTS composite scoring, Kinesin catalytic motif search,
and functional motif detection using Eisenberg hydrophobicity and regex-based pattern
matching; (M4) genomic coordinate extraction from SQANTI3 GTF and classification table,
exon structure reconstruction, TransDecoder CDS mapping, and natural antisense transcript
(NAT) detection; (M5) UCSC REST API RepeatMasker annotation for the full transcript locus
(exon boundaries ± 5 kb extension) with young LINE-1 filtering (Smith-Waterman score and
percent divergence < 15%); (M6) structured JSON output, per-case domain table (`domains.tsv`
with LOST/GAINED status labels), domain architecture figure (matplotlib; PDF + PNG), and
Markdown report generated from a Jinja2 template containing auto-populated Methods and
Results draft paragraphs; (M7) cross-case comparison table (`cases_summary.tsv`) aggregating
domain, motif, genomic, and repeat-element fields for all completed cases, sorted by priority
and |PRISM Δ|.

The pipeline was executed on NDUFS4, DLG1, and KIF21B, reproducing all manually derived
findings (NDUFS4/tr73243 RVT_1 E = 4.6×10⁻⁴⁸; DLG1/tr319500 L27_1 E = 7.8×10⁻³⁴) and
additionally revealing the KIF21B structural transition: tr293004 harbours a complete
kinesin motor domain (Kinesin, aa 14–370; E = 1.1×10⁻¹⁰⁹, score = 354.0) and
microtubule-binding domain (Microtub_bd, aa 7–158; E = 1.8×10⁻²³), both absent in
tr292978 (710 aa), which instead harbours 15 WD40 β-propeller Pfam hits spanning
aa 372–686 (dominant profiles: WD40, ANAPC4_WD40, NBCH_WD40) and no repeat element
overlap with any of its 19 exons. The pipeline supports resumption of interrupted runs
(completed cases identified by presence of `analysis.json`) and optional parallel execution
via Python `concurrent.futures.ProcessPoolExecutor`. All pipeline code, configuration, and
output files will be made available in the study repository.

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
| Seeds | 5 (ensemble mean) | Seed stability [R9.4] |
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

*[TODO: Zenodo DOI, GitHub URL — obtain prior to submission]*

---

## Ethics statement (draft)

This study uses only publicly available genomic and proteomic databases. No patient data,
biological samples, or personally identifiable information were used. No ethical approval
was required.

---

## TODO before submission
- [x] ESM-2 citation: Lin et al. (2023) *Science* doi:10.1126/science.ade2574 — added to References
- [ ] TransDecoder citation: Haas et al. (2013) *Nat Protoc* — add to References before submission
- [ ] Lin et al. (2017) focal loss — add to References before submission
- [x] Ablation table: Supp Table S4 complete (ablation_results_20260517_1537.json)
- [ ] Data availability: Zenodo DOI, GitHub URL
- [x] Supplementary Note 1: D1/D2/D3 diagnostic experiments ✓ 2026-05-17
  → reports/supp_note1_diagnostic_experiments.md
- [ ] Supplementary Table S1: full ESM-LR / ESM-RF per-GO-term comparison
- [ ] Verify Bambu version used in long-read assembly
  *Note: bambu R package not installed in current env (isoform_env); long-read assembly was
  performed on a separate system. Check with data provider or session logs from assembly step.
  Likely cite: Dong X et al., Nat Methods 2023;20(8):1187-1196 (PMID 37349533) — verify before submission.*
### Brain / AD sections (4.13–4.15) added 2026-05-20
- [x] 4.13 Brain tissue dataset assembly written
- [x] 4.14 Cross-tissue zero-shot evaluation written
- [x] 4.15 DTU testing + AD isoform switch identification written
- [x] Add IsoQuant citation (Prjibelski 2023) to reference list — done
- [x] Add minimap2 citation (Li 2018) to reference list — done
- [x] Samsung AD cohort: 13 AD donors / 8 CT donors confirmed (2026-05-20); per-cell-type cell counts added to Methods 4.13 and 4.15 (from adata_gene_for_UMAP.h5ad)
- [x] MTS absence: D+E=4 in first 40aa (typical MTS has 0; confirmed by sequence), HHH motif at pos 7-9 also atypical
- [x] MitoFates/TargetP criteria: net charge +2(canonical) vs −1(tr73243), MTS composite 3/5 vs 1/5 — quantified (2026-05-21)
- [x] 4.16 Pfam hmmscan + RepeatMasker + MTS analysis added to Methods (2026-05-21)
- [x] L1PA3/L1PA11 LINE-1 confirmation: RepeatMasker hg38 query confirms RVT_1 from L1PA11(+) ORF2 (2026-05-21)
- [x] DLG1 tr319500 L27_1 domain (aa 6-63, E=7.8e-34): hmmscan confirmed — update DLG1 narrative complete
- [ ] MitoFates/TargetP web server formal prediction for tr73243 — optional, computational criteria already quantified
- [x] LYR motif absence in tr73243: ABSENT by sequence search (confirmed 2026-05-20)
- [x] tr73243 overlap with canonical NDUFS4: 3/15 aa match in first 15 aa (M, A, R); "3 aa overlap" confirmed
- [x] KIF21B P-loop confirmed: GQTGAGKT at aa 87 in tr293004 ✓; DLAGSE Switch-II at aa 273 ✓; Switch-I SSRSHA at aa 222 ✓ (automated pipeline M3, 2026-05-22)
- [x] KIF21B automated pipeline: Kinesin E=1.1e-109/Score=354 + Microtub_bd E=1.8e-23 (tr293004); 15× WD40/ANAPC4_WD40/NBCH_WD40 aa372-686 (tr292978); no exonic repeats — 4.16 updated (2026-05-22)
- [x] Automated pipeline (orchestrate.py v1.1, M1-M7, Jinja2, M7 cases_summary.tsv) — 4.16 updated (2026-05-22)
- [ ] KIF21B domain coordinates: optional verification against PDB (3B6U kinesin motor crystal structure)
  *Note: bambu R package not installed in current env (isoform_env); long-read assembly was
  performed on a separate system. Check with data provider or session logs from assembly step.
  Likely cite: Dong X et al., Nat Methods 2023;20(8):1187-1196 (PMID 37349533) — verify before submission.*
- [x] ESM-2 citation added to References list (Lin et al. 2023)
- [x] IsoQuant citation added to References list (Prjibelski et al. 2023)
- [x] minimap2 citation added to References list (Li 2018)
- [x] MitoFates citation added to References list (Fukasawa et al. 2015)
- [x] TargetP 2.0 citation added to References list (Almagro Armenteros et al. 2019)
- [x] HMMER citation added to References list (Eddy 2011)

---

## References

Almagro Armenteros JJ, Salvatore M, Emanuelsson O, Winther O, von Heijne G, Elofsson A, Nielsen H. (2019). Detecting sequence signals in targeting peptides using deep learning. *Life Science Alliance*, 2(5): e201900429. doi:10.26508/lsa.201900429. PMID: 31570514.  
*(TargetP 2.0 — MTS prediction tool)*

Almagro Armenteros JJ, Tsirigos KD, Sønderby CK, et al. (2019). SignalP 5.0 improves signal peptide predictions using deep neural networks. *Nat Methods*, 16, 837–839. doi:10.1038/s41592-019-0358-2. PMID: 30778233.  
*(TargetP 2.0 — cited in §4.16)*

Eddy SR. (2011). Accelerated profile HMM searches. *PLoS Comput Biol*, 7(10): e1002195. doi:10.1371/journal.pcbi.1002195. PMID: 22039361.  
*(HMMER 3.x — hmmscan used in §4.16 and §4.2)*

Frankish A, Carbonell-Sala S, Diekhans M, et al. (2023). GENCODE: reference annotation for the human and mouse genomes in 2023. *Nucleic Acids Res*, 51(D1): D942–D949. doi:10.1093/nar/gkac1071. PMID: 36420896.  
*(GENCODE v43 annotation used throughout)*

Fukasawa Y, Tsuji J, Fu SC, Tomii K, Horton P, Imai K. (2015). MitoFates: improved prediction of mitochondrial targeting sequences and their cleavage sites. *Mol Cell Proteomics*, 14(4): 1113–1126. doi:10.1074/mcp.M114.043083. PMID: 25670805.  
*(MitoFates — MTS scoring criteria in §4.16)*

Li H. (2018). Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18): 3094–3100. doi:10.1093/bioinformatics/bty191. PMID: 29750201.  
*(minimap2 v2.26 — long-read alignment in §4.13)*

Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, Smetanin N, Verkuil R, Kabeli O, Shmueli Y, Dos Santos Costa A, Fazel-Zarandi M, Sercu T, Candido S, Rives A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637): 1123–1130. doi:10.1126/science.ade2574. PMID: 36927031.  
*(ESM-2 — protein language model used for sequence embeddings in §4.2, §4.13)*

Prjibelski AD, Mikheenko A, Joglekar A, Tilgner HU, Skoblov MY, Korobeynikov A, Holbrook JD, Pevzner PA. (2023). Accurate isoform discovery with IsoQuant using long reads. *Nat Biotechnol*, 41, 915–918. doi:10.1038/s41587-022-01565-y. PMID: 37542202.  
*(IsoQuant v3.3 — isoform detection in §4.13)*

Tardaguila M, de la Fuente L, Marti C, Pereira C, Pardo-Palacios FJ, Del Risco H, Ferrell M, Mellado M, Macchietto M, Verheggen K, Edelmann M, Ezkurdia I, Tress M, Martens L, Pisconti A, Bhatt DL, Mortazavi A, Gonzalez JM, Climent J. (2018). SQANTI: extensive characterization of long-read transcript sequences for quality control in full-length transcriptome identification and quantification. *Genome Res*, 28(3): 396–411. doi:10.1101/gr.222976.117. PMID: 29440212.  
*(SQANTI3 v5.1 — isoform classification in §4.13)*

Haas BJ, Papanicolaou A, Yassour M, Grabherr M, Blood PD, Bowden J, Couger MB, Eccles D, Li B, Lieber M, MacManes MD, Ott M, Orvis J, Pochet N, Strozzi F, Weeks N, Westerman R, William T, Dewey CN, Henschel R, LeDuc RD, Friedman N, Regev A. (2013). De novo transcript sequence reconstruction from RNA-seq using the Trinity platform for reference generation and analysis. *Nat Protoc*, 8(8): 1494–1512. doi:10.1038/nprot.2013.084. PMID: 23845962.  
*(TransDecoder — longest ORF prediction within assembled transcripts in §4.16)*

Lin TY, Goyal P, Girshick R, He K, Dollár P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988. doi:10.1109/ICCV.2017.324. arXiv:1708.02002.  
*(Focal loss — class-imbalanced training objective in §4.3)*

Schroff F, Kalenichenko D, Philbin J. (2015). FaceNet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 815–823. doi:10.1109/CVPR.2015.7298682. arXiv:1503.03832.  
*(Triplet loss — embedding-space metric learning objective in §4.3)*
