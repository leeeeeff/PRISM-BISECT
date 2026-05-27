# DIFFUSE: Deep Isoform Function Prediction Using Sequence Embeddings, with Multi-Evidence Characterization of Alzheimer's Disease Isoform Switches by BISECT

*Draft manuscript — compiled 2026-05-27*  
**Correspondence**: seungwon.david.lee@gmail.com  
**Target journal**: Nature Methods  
**Status**: Draft v1 (merged from manuscript_full_english.md + main_paper_draft.md)

---

## Abstract

Most computational approaches to gene function prediction cannot distinguish between alternatively
spliced isoforms, treating each gene as a single functional entity. Here we present DIFFUSE, a deep
learning framework that predicts isoform-level Gene Ontology Biological Process functions directly
from ESM-2 protein language model embeddings. Applied to 36,748 isoforms from 12,709 human
skeletal muscle genes profiled by long-read single-cell RNA sequencing, DIFFUSE achieves mean
AUPRC of 0.685 across 11 sarcopenia-relevant GO terms versus 0.363 for logistic regression
(+88.7%; 10/11 terms q < 0.05, Benjamini-Hochberg correction), with 4.7-fold improvement over
non-linear ESM-based baselines. A cosine-space embedding separability metric post-hoc identifies
which GO terms benefit from isoform-resolution modelling (leave-one-out accuracy 100%). Within-gene
isoform discrimination is confirmed by negative controls (shuffled-label pos_bias = 0.24; DIFFUSE
maximum 1.902 for muscle contraction). Applied zero-shot to 63,994 isoforms from human prefrontal
cortex long-read single-cell RNA sequencing (Samsung Alzheimer's disease cohort) without retraining,
DIFFUSE achieves macro AUPRC of 0.600. Integration with Dirichlet-multinomial differential transcript
usage testing identifies three Alzheimer's-disease-specific isoform switches exclusive to single cell
types: a bidirectional KIF21B switch in excitatory neurons (p = 9.3×10⁻⁸), NDUFS4 Complex I locus
displacement by a shared-TSS novel 379 amino acid protein (p = 3.6×10⁻⁶), and DLG1 isoform replacement in
oligodendrocyte precursor cells (p = 9.0×10⁻¹⁰). To translate statistical prioritisation into
mechanistic characterisation, we introduce BISECT (Biological Isoform-Switch Evidence Characterization
Tool), a twelve-module pipeline that applies Pfam domain annotation, AlphaFold/ESMFold structural
confidence, STRING PPI network validation, and UCSC phyloP 100-vertebrate evolutionary conservation
to 53 candidate pairs. Of 26 Stage 2 domain-change cases, 13 (50%) received STRING experimental
PPI support, with marked cell-type stratification (inhibitory neurons 5/7, 71%). Three Tier A
candidates — KIF21B (kinesin motor → WD40 β-propeller gain-of-fold), NDUFS4 (TSS co-option
displacing Complex I subunit by a 379 aa novel NNIC isoform), and DLG1 (NIC/NNIC-based OPC
isoform state transition) — represent novel-sequence switches with complete multi-evidence support. Together, DIFFUSE and BISECT establish a sequence-first, tissue-agnostic framework for
isoform-resolution functional prediction and evidence-integrated biological characterisation of
disease isoform switches.

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
individual isoforms remains a major computational bottleneck.

In Alzheimer's disease (AD), recent long-read single-cell RNA sequencing of prefrontal cortex has
revealed thousands of novel transcript isoforms — including isoforms with completely uncharacterised
protein-coding potential — in the cell types most vulnerable to AD pathology. Whether a muscle-trained
isoform function predictor can generalize across tissue types and, when integrated with single-cell
differential transcript usage testing, discover disease-relevant isoform switches in the brain is an
open question with direct implications for cross-tissue computational genomics. Critically, the
methodological gap lies not only in statistical detection but also downstream: existing tools for
differential transcript usage (DTU) identify significant shifts in isoform proportions but provide
no inference about whether those shifts alter protein domain composition, abrogate protein–protein
interactions, or affect evolutionarily constrained functional elements.

Current computational tools for GO term prediction operate predominantly at the gene level.
Network-propagation methods (STRING, ConsensusPathDB) assign function based on protein interaction
partners without distinguishing isoforms. Deep learning approaches including DeepFRI
(Gligorijević et al., *Nat Commun*, 2021; PMID: 34210978) and CLEAN (Guo et al., *Nat Comput
Sci*, 2023; PMID: 37217634) achieve strong performance but are trained and evaluated on canonical
sequences only. Isoform-aware methods do exist for downstream analyses: IsoformSwitchAnalyzeR
(Vitting-Seerup and Sandelin, *Bioinformatics*, 2019; PMID: 30989184) predicts functional consequences
of isoform switches using Pfam domain annotation and NMD rules; SQANTI (Tardaguila et al.,
*Genome Res*, 2018; PMID: 29440212) classifies transcript structural categories; and APPRIS
(Rodriguez et al., *Nucleic Acids Res*, 2022; PMID: 34755864) selects principal isoforms based on
functional evidence. However, none of these methods learns from sequence to directly predict
isoform-level GO term membership using a unified deep learning framework trained across multiple
GO terms, and none provides a systematic multi-evidence biological validation framework.

Here we present two complementary contributions. First, DIFFUSE (Deep Isoform Function Prediction
Using Sequence Embeddings), a deep learning framework that predicts isoform-level GO Biological
Process membership directly from ESM-2 protein language model embeddings using a deep MLP trained
with focal loss and triplet loss. Second, BISECT (Biological Isoform-Switch Evidence Characterization
Tool), a twelve-module pipeline that integrates Pfam domain annotation, structural confidence
(AlphaFold DB/ESMFold), PPI network validation (STRING v12.0), and evolutionary conservation
(phyloP 100-way) to provide multi-evidence biological characterisation of DIFFUSE-prioritised
AD isoform switch candidates.

We demonstrate that DIFFUSE achieves mean AUPRC of 0.685 across 11 Type-B sarcopenia GO terms,
and applied zero-shot to 63,994 brain isoforms from the Samsung AD cohort, integrates with DTU
testing to identify three Alzheimer's-disease-specific isoform switches. BISECT applied to
53 candidate pairs identifies 26 with domain-level structural changes, of which 13/26 (50%)
receive STRING experimental PPI support. Three Tier A candidates — KIF21B, NDUFS4, and DLG1 —
represent novel-sequence isoform switches with complete multi-evidence support; the two most
mechanistically detailed Tier B cases, PTPRF and FANCA, further demonstrate evidence-based
hypothesis revision and DNA repair pathway disruption.

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

Isoform amino acid sequences were extracted with TransDecoder (Haas et al., 2013) (v5.7.1, ORF ≥ 100 aa). For
non-coding isoforms, the longest ORF ≥ 30 aa was used; isoforms with no valid ORF received
a zero-vector embedding. All sequences were encoded with ESM-2 (esm2_t30_150M_UR50D,
HuggingFace checkpoint `facebook/esm2_t30_150M_UR50D`, 150 million parameters, 30 transformer
layers; Lin et al., 2023). Sequences were tokenised using the ESM-2 alphabet; representations
from the final transformer layer were averaged across all sequence-length positions (mean pooling)
to produce a 640-dimensional per-isoform embedding. Embeddings were computed once and cached.

### 2.3 DIFFUSE model architecture

DIFFUSE maps ESM-2 640-dimensional embeddings to per-isoform GO term probability scores
via a deep MLP:

```
Input:    x ∈ ℝ^640  (ESM-2 embedding, L2-normalised)
Layer 1:  Dense(256) → BatchNormalization → ReLU → Dropout(0.3)
Layer 2:  Dense(128) → ReLU → Dropout(0.3)
Layer 3:  Dense(64)  → L2-normalisation           (embedding space)
Output:   Dense(1)   → sigmoid                    (GO term probability)
```

Separate output heads were trained per GO term, sharing Layers 1–3.

### 2.4 Training objective

The total loss combines focal loss for sparse label handling and triplet loss for
isoform-specific embedding geometry:

    L_total = L_focal + 0.5 · L_triplet

**Focal loss** [Lin et al., 2017; R1.1]:

    L_focal(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

γ = 2 (default); α_t computed by class-balanced weighting.

**Triplet loss** [Hermans et al., 2017; R3.1]:

    L_triplet(a, p, n) = max(d(a,p) − d(a,n) + margin, 0)

Cosine distance d(·,·), margin = 0.30. Negative mining used cross-gene negatives only —
intra-gene negatives are excluded to prevent the model exploiting gene identity as a shortcut.
Active triplet ratio was monitored per epoch; online hard negative mining was applied when
active_ratio < 5%.

### 2.5 Training protocol

Each GO term was trained independently for 100 epochs with Adam (lr = 1×10⁻⁴,
weight_decay = 1×10⁻⁵), batch size 512. Learning rate was halved if validation AUPRC
did not improve for 10 epochs (ReduceLROnPlateau). Training ran on a single NVIDIA A100 GPU.
Final predictions are ensemble means across 5 random seeds (42, 123, 456, 789, 2024).

### 2.6 Evaluation protocol

**Dataset split.** Isoforms were split 80/20 (train/test) using gene-stratified partitioning.

**Primary metric.** AUPRC per GO term. Macro-AUPRC (unweighted mean) for multi-term comparisons.

**Confidence intervals.** Gene-block bootstrap (n = 500): test-set genes resampled with replacement.
Multiple testing: Benjamini-Hochberg correction across 11 Type-B GO terms.

### 2.7 Baseline models

All baselines and DIFFUSE use identical ESM-2 640-dimensional input features. LR baseline:
logistic regression (C = 1.0, class_weight = 'balanced'; scikit-learn 1.3.0). Non-linear
baselines: ESM-LR and ESM-RF (random forests; n_estimators = 100, max_depth = 6) applied
directly to ESM-2 640d embeddings. XGBoost (n_estimators = 300, max_depth = 6, lr = 0.05)
was evaluated and showed no statistically significant difference from DIFFUSE in macro-AUPRC
(gene-block bootstrap, 13/13 terms; Supplementary Table S1) but demonstrated 3.5-fold lower
within-gene isoform discrimination score (0.027 vs 0.094), indicating gene-level memorisation.

### 2.8 Type-A/B GO term classification

The cosine separability metric:

    sep_cosine = dist(centroid_pos, centroid_neg)_cosine /
                 mean_dist(pos_i, centroid_pos)_cosine

Threshold τ = 0.060 determined retrospectively by LOOCV (13/13 = 100%; decision gap 0.056–0.167).
Type-A: sep_cosine ≥ 0.060 (n=2); Type-B: sep_cosine < 0.060 (n=11).

### 2.9 Isoform discriminability (pos_bias) and negative controls

    pos_bias = mean_g(std_i(score_i | gene g ∈ positive, n_i ≥ 2)) / std(score_i | all isoforms)

Negative controls: gene-mean predictor (pos_bias = 0.000), random predictor (0.898 ± 0.041),
shuffled-label DIFFUSE (0.240 ± 0.048).

### 2.10 Isoform-switch analysis and NMD verification

Per-isoform scores averaged across 5 seeds. Within-gene score range (max − min) > 0.30 designated
isoform-switch candidates. NMD screening: both top and bot isoforms screened symmetrically —
isoforms with PTC > 55 nt upstream of last EJC flagged as NMD candidates and excluded.
23/126 (18.3%) candidate pairs excluded by symmetric NMD screening.

### 2.11 Component ablation study

Single-component ablation of DIFFUSE: no_focal (CE replaces focal loss), no_BN, no_dropout,
no_L2norm. Evaluated on 3 seeds × 5 GO terms. Removal of focal loss was the only condition
that consistently reduced Type-B AUPRC across all three Type-B terms (Δ = −0.070).

### 2.12 pos_bias bootstrap confidence intervals

Gene-block bootstrap (n=1,000) per GO term, seed=42. BH-corrected q-values across 13 terms.

### 2.13 Brain tissue dataset assembly (Samsung AD IsoQuant)

Human prefrontal cortex (dorsolateral prefrontal cortex, Brodmann area 9/46) tissue was obtained
from 21 donors (13 Alzheimer's disease, 8 cognitively normal controls; Supplementary Table S4) through the Samsung Medical
Center Brain Bank (Seoul, Korea) under institutional ethical approval (IRB No. SMC 2021-08-031).
AD diagnosis was confirmed by neuropathological criteria (NIA-AA criteria: Braak stage V–VI,
CERAD frequent/moderate neuritic plaques). Control donors had Braak stage ≤ II and no neurological
diagnosis. Single-nucleus RNA-seq libraries were prepared from 50 mg snap-frozen tissue per donor
using the 10x Genomics Chromium platform (v3.1 chemistry), targeting ≥10,000 nuclei per donor.
Long-read sequencing was performed on PacBio Sequel IIe (SMRT sequencing) following cDNA
amplification, yielding a mean of 2.3 million full-length non-chimeric (FLNC) reads per donor.

Short reads were aligned to GRCh38 (GENCODE v43) using minimap2 (v2.24, -ax splice --secondary=no).
Novel transcriptome assembly was performed with IsoQuant v3.3 using extended_annotation mode
(--complete_genedb) with the following quality filters: minimum UMI count ≥ 10 per isoform
(single donor), minimum ≥ 3 independent donor support per transcript. Reference transcript set
was supplemented with refTSS-validated transcription start sites to anchor 5' boundaries of
novel isoforms. Structural classification was performed by SQANTI3 v5.1 against GENCODE v43,
assigning each transcript to FSM (full splice match), ISM (incomplete splice match), NIC
(novel in catalogue), or NNIC (novel not in catalogue) categories. Transcripts annotated as
artefactual by SQANTI3 filter rules (RT-switching, intra-priming, non-canonical splice sites
without experimental support) were excluded. Cell type annotation was performed by integrating
gene expression profiles with canonical marker genes across eight cell types: Astrocyte,
Excitatory neuron, Inhibitory neuron, Lymphocyte, Microglia, OPC (oligodendrocyte precursor
cell), Oligodendrocyte, and Vascular cell.

The final dataset comprised 63,994 isoforms across 18,291 genes (56,095 known FSM/ISM;
7,899 novel NIC/NNIC). Of these, 10,817 novel transcripts were retained in the IsoQuant GTF
(donor support ≥ 3 filter), of which 7,899 appeared in the per-cell-type count matrices.
ORF prediction was applied with TransDecoder v5.7.1 (Haas et al., 2013) (minimum ORF length ≥ 30 aa); 53,826
isoforms (84.1%) yielded valid ORF predictions and were encoded with ESM-2 as described in
Section 2.2. The remaining 10,168 isoforms (15.9%), predominantly short non-coding transcripts,
received zero-vector embeddings and were excluded from AUPRC calculations but retained in
the DTU analysis.

### 2.14 Cross-tissue zero-shot evaluation

The muscle-trained DIFFUSE model (v15d_bp_clean) was applied without parameter modification
to the brain IsoQuant test set. AUPRC computed per GO term; macro-AUPRC across all 18 GO terms.
KNN propagation (k=5,10,20 nearest known isoforms) used as comparison for novel isoforms.

### 2.15 Single-cell DTU testing and AD isoform switch identification

Differential transcript usage (DTU) tested per cell type using Dirichlet-multinomial model on
pseudobulk counts (13 AD vs 8 CT donors). Cell-type specificity: switch significant in ≤ 1 of 8
cell types at p ≤ 0.10. Bonferroni-corrected threshold α = 1×10⁻⁶ (18,291 genes × 8 cell types).

---

### 2.16 BISECT pipeline overview

BISECT (Biological Isoform-Switch Evidence Characterization Tool) v1.1 is a twelve-module Python
pipeline (Fig. 1) (Python ≥ 3.9; orchestrate.py) that processes isoform-pair candidates through sequential
analysis stages. Input: a CSV of candidate pairs (CT/AD transcript IDs, gene name, cell type,
DIFFUSE delta, DTU p-value). Output: per-case analysis.json, domain map PDF, and
cases_summary.tsv. Two run modes: `--mode screen` (Stage 1+2 only) and `--mode deep` (full M1–M12).

**Stage 1 filter** (pre-pipeline): |DIFFUSE Δ| ≥ 0.5 AND DTU p ≤ 1×10⁻⁵. Of the initial
universe, 53 candidates passed Stage 1.

**Stage 2 filter** (M2): Pfam-A domain annotation by HMMER; candidates with no domain set
difference between CT and AD isoforms excluded (strict mode). Of 53 Stage 1 candidates, 26 passed.

### 2.17 M1 — Sequence extraction

Isoform amino acid sequences extracted from the IsoQuant/SQANTI3 TransDecoder output
(isoforms_corrected.faa). Sequences < 50 aa or flagged as incomplete (no start codon) excluded.

### 2.18 M2 — Pfam domain annotation

HMMER 3.3.2 (Eddy, 2011; Mistry et al., 2021) hmmscan, Pfam-A release 36.0, E-value ≤ 0.01 (domain-level, `--domtblout`).
Domain set difference: `domains_lost = CT_pfam − AD_pfam`; `domains_gained = AD_pfam − CT_pfam`.
Stage 2 PASS = (n_lost + n_gained) ≥ 1.

### 2.19 M3 — Functional motif and mitochondrial targeting signal analysis

Mitochondrial targeting signal (MTS) scored by five composite criteria (MitoFates/TargetP 2.0
heuristics): (i) net charge K+R−D−E ≥ +2 in first 30 aa; (ii) D+E ≤ 3 in first 40 aa;
(iii) hydrophobic moment μH ≥ 0.12 (Eisenberg scale, 18-mer window); (iv) no HHH in first
30 aa; (v) LYR motif present. Functional motifs detected by regular expression: kinesin P-loop
(GxGxxGKT), PDZ GLGF-box (GLGF/GVGF/GMGF), WD40 (GH-WD), RVT_1 (YMDD/YVDD), coiled-coil (LxxLxxL).

### 2.20 M4 — Genomic context and SQANTI3 classification

Structural categories (FSM, ISM, NIC, NNIC) assigned by SQANTI3 v5.2.2 (Ensembl GRCh38 r110).
Exon coordinates, strand, genomic span extracted from SQANTI3 GTF. NAT annotations assigned where
overlapping opposite-strand transcripts detected.

### 2.21 M5 — Transposable element annotation

Young LINE-1 elements (pairwise divergence < 15%) within 5 kb of exon boundaries identified via
UCSC RepeatMasker track REST API (hg38; rmsk; ±5 kb window). CDS overlap flagged as `has_l1_in_cds`.

### 2.22 M6, M7 — Report and cross-case summary

Case reports generated in Markdown and JSON (Jinja2 template). Domain maps produced with
matplotlib. All per-case analysis.json aggregated into cases_summary.tsv (≥32 columns) by
m7_compare.py, ranked by priority score and |DIFFUSE Δ|.

### 2.23 M8 — Genomic sequence validation

Young LINE-1 elements identified by M5 verified at sequence level: L1PA subfamily identification,
RepBase consensus alignment, best_identity (%) calculation. Strong evidence threshold: identity ≥ 95%.

### 2.24 M9 — NMD susceptibility screening

PTC position vs last EJC distance computed for each isoform (exon coordinates from M4).
Rule: PTC > 55 nt upstream of last EJC → NMD susceptible flag.

### 2.25 M10 — AlphaFold structural confidence analysis

UniProt accession retrieved per gene via UniProt REST API (reviewed Swiss-Prot filter). pLDDT
scores extracted from AlphaFold DB v4 prediction JSON endpoint. Domain-level pLDDT computed as
mean over residues within each M2 Pfam domain boundary. Confidence tiers: very high ≥ 90; high
70–90; low 50–70; very low < 50.

**ESMAtlas API fallback (novel isoforms)**: For isoforms lacking a UniProt entry (KIF21B NIC/NNIC),
structural predictions obtained via ESMAtlas REST API (`POST .../foldSequence/v1/pdb/`) using
overlapping fragments of ≤400 aa. B-factor in returned PDB encodes pLDDT on 0–1 scale. KIF21B
CT kinesin fragment (aa 1–380, pLDDT = 93.2) and AD WD40 core (aa 370–620, pLDDT = 94.6)
each successfully predicted.

### 2.26 M11 — PPI network validation

STRING v12.0 `interaction_partners` JSON API (species = 9606; limit = 50; combined score ≥ 150
for retrieval). Verdict: SUPPORTED = ≥1 hypothesized partner at combined score ≥ 700 with
experimental channel (escore > 0); PARTIAL = score 400–700 with experimental evidence;
UNSUPPORTED = no partner meets threshold. Hypothesized partners curated per gene from known
domain functions and literature. M11 functions as an evidence-based hypothesis revision tool: if STRING returns score = 0
for initial hypothesis, hypothesis is rejected and revised based on actual top partners
(demonstrated for PTPRF: SLIT2 decoy hypothesis rejected → Liprin-α dominant-negative revised model).

### 2.27 M12 — Evolutionary conservation analysis

UCSC phyloP 100-vertebrate (100way) scores for hg38 retrieved per-base via UCSC REST API.
Mean phyloP computed per exon within exon boundaries. Isoform-specific exons defined by set
difference of genomic coordinates (M4). Background conservation: five randomly sampled intronic
windows. Conservation classes: highly conserved (mean phyloP ≥ 1.5), conserved (0.5–1.5),
low conservation (< 0.5). Negative phyloP indicates accelerated evolution.

---

### Parameter summary

| Parameter | Value | Rule |
|-----------|-------|------|
| ESM-2 | esm2_t30_150M_UR50D (640d) | D1 diagnostic AUPRC = 0.690 |
| MLP dims | 256 → 128 → 64 | Ablation-selected |
| Dropout | 0.3 (layers 1–2) | Sparse label overfitting guard |
| Focal γ | 2.0 | [R1.1] |
| Triplet margin | 0.30 | [R3.1] |
| λ_focal / λ_triplet | 1.0 / 0.5 | Empirical |
| Adam lr | 1×10⁻⁴ | ReduceLROnPlateau (×0.5, patience=10) |
| Batch size | 512 | A100 constraint |
| Seeds | 42, 123, 456, 789, 2024 | 5-seed ensemble |
| Bootstrap n | 500 (DIFFUSE) / 1000 (pos_bias) | Gene-block |
| Type-B τ | 0.060 | LOOCV 13/13 |
| BISECT Stage1 Δ | ≥ 0.5 | Discovery filter |
| BISECT Stage2 E-value | < 0.01 (Pfam-A) | Domain filter |
| STRING score threshold | ≥ 700 (SUPPORTED) | M11 verdict |
| phyloP high | ≥ 1.5 | M12 conservation |

---

## Data availability (draft)

GENCODE v43 (https://www.gencodegenes.org/releases/43.html) and GO database (release 2024-01-17;
https://geneontology.org). ESM-2: `facebook/esm2_t30_150M_UR50D` (HuggingFace). Pre-computed
embeddings and trained model weights will be deposited in Zenodo upon acceptance. BISECT pipeline
code available at https://github.com/[REPOSITORY]. Samsung AD IsoQuant dataset: upon data sharing
agreement.

## Ethics statement (draft)

This study uses publicly available genomic databases and a deidentified human tissue dataset
processed under institutional approval at Samsung Medical Center. No patient-identifiable information
was used in the computational analysis reported here.

---

## 3. Results

### 3.1 DIFFUSE outperforms logistic regression across sarcopenia-relevant GO terms

We evaluated DIFFUSE against a logistic regression (LR) baseline using identical ESM-2 embeddings,
assessing 13 GO terms spanning sarcopenia-relevant skeletal muscle pathways.

Across 11 Type-B GO terms (Section 3.2), DIFFUSE achieved a mean AUPRC of 0.685 compared with
0.363 for LR (Δ = +0.322, +88.7%; 5-seed mean; Table 1). Ten of eleven Type-B terms reached
statistical significance (q < 0.05, Benjamini-Hochberg correction on gene-block bootstrap CIs,
n=500). The two Type-A GO terms (motor activity, glycolysis) showed no significant difference
(Δ = −0.018), consistent with their gene-level-dominated embedding structure (Section 3.2).

Non-linear baselines (ESM-LR macro-AUPRC 0.145, ESM-RF 0.147) showed only marginal differences
from each other, confirming that non-linearity alone is insufficient. DIFFUSE's 4.7-fold
improvement over both baselines confirms that batch normalization, dropout regularization, and
hierarchical feature abstraction provide qualitative benefits beyond mere non-linearity.

XGBoost achieved macro-AUPRC 0.738 numerically — but gene-block bootstrap (n=500) showed no
statistically significant difference from DIFFUSE in 13/13 terms. Critically, XGBoost
within-gene isoform discrimination was 0.027 versus 0.094 for DIFFUSE (3.5-fold lower), indicating
gene-level memorisation rather than genuine isoform discrimination. XGBoost is therefore excluded
from the primary comparison.

**Table 1. AUPRC comparison across 13 sarcopenia GO terms.**

| GO Term | Function | Type | DIFFUSE | LR | Δ | q-BH |
|---------|----------|------|---------|-----|---|------|
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

Values are 5-seed mean AUPRC. BH q-values from gene-block bootstrap CI (n=500).

---

### 3.2 Positive class structural heterogeneity determines model advantage

Annotation quality metrics (taxonomic breadth score, tissue context specificity) did not predict
Δ AUPRC: r(TBS, Δ) = −0.14 (p = 0.64); r(TCS, Δ) = −0.10 (p = 0.74). A linear model combining
both explained only 9.8% of Δ variance.

ESM-2 positive class structure was the operative predictor. Four structural metrics correlated
significantly with Δ AUPRC: sep_cosine (r = −0.60; 95% CI [−0.87, −0.07]), pc1_var_ratio
(r = −0.765, p = 0.002; 95% CI [−0.926, −0.367]), intra_cos_mean (r = −0.728, p = 0.005),
and n_clusters. This yielded a three-case taxonomy:

- **Case 1** (LR ≥ 0.60; n=2): single structurally coherent protein family; DIFFUSE offers no advantage (Δ = −0.018).
- **Case 2** (0.45 ≤ LR < 0.60; n=3): partial cluster coherence; moderate advantage (Δ = +0.136).
- **Case 3** (LR < 0.45; n=8): structurally fragmented positive class; large consistent advantage (Δ = +0.391; all q < 0.05).

sep_cosine < 0.060 classifies all 13 GO terms correctly (LOOCV 13/13 = 100%; decision gap 0.056–0.167),
serving as a label-free pre-hoc screening metric.

---

### 3.3 DIFFUSE achieves GO-term-dependent within-gene isoform discrimination

pos_bias (ratio of mean within-gene score std among multi-isoform positive genes to global score std)
ranges from 0.475 (Ca²⁺ signaling) to 1.902 (muscle contraction). 11/13 GO terms significantly
exceed the shuffled-label noise floor (q < 0.05, BH-corrected). The macro pos_bias decreases by only
0.022 when restricted to protein-coding isoforms (1.006 → 0.985), confirming the signal is not a
coding/non-coding artefact. pos_bias and per-GO-term AUPRC gain are weakly correlated (r = −0.20,
n=13), confirming they capture complementary performance aspects.

**Table 2. pos_bias per GO term (DIFFUSE) with bootstrap 95% CI and significance.**

| GO Term | Function | pos_bias (5-seed) | 95% CI | q vs. shuf. (0.24) |
|---------|----------|--------------------|--------|---------------------|
| GO:0006941 | Muscle contraction | 1.902 | [0.895, 1.592] | < 0.001 *** |
| GO:0007519 | Skeletal muscle dev | 1.778 | [0.873, 1.955] | < 0.001 *** |
| GO:0003774 | Motor activity | 1.435 | [0.693, 1.582] | < 0.001 *** |
| GO:0030017 | Sarcomere org | 1.176 | [0.730, 1.224] | < 0.001 *** |
| GO:0043161 | Proteasome-UPS | 0.957 | [0.549, 0.796] | < 0.001 *** |
| GO:0042692 | Muscle cell diff | 0.824 | [0.654, 1.045] | < 0.001 *** |
| GO:0007517 | Muscle organ dev | 0.805 | [0.834, 1.239] | < 0.001 *** |
| GO:0007005 | Mitochondrion org | 0.879 | [0.502, 0.731] | < 0.001 *** |
| GO:0055074 | Ca²⁺ homeostasis | 0.764 | [0.414, 0.700] | < 0.001 *** |
| GO:0006914 | Autophagy | 0.724 | [0.409, 0.748] | < 0.001 *** |
| GO:0032006 | TOR signaling | 0.699 | [0.241, 0.625] | 0.027 * |
| GO:0006096 | Glycolysis | 0.663 | [0.019, 1.172] | 0.189 |
| GO:0007204 | Ca²⁺ signaling | 0.475 | [0.130, 0.465] | 0.307 |
| **Macro (n=13)** | | **1.006** | — | — |

Shuffled-label null (3-term mean): 0.240 ± 0.048 (noise floor).
Random null (3-term mean): 0.898 ± 0.041 (noise ceiling).

---

### 3.4 Novel isoform discovery reveals sarcopenia-relevant functional switches

Isoform-switch analysis across four high-priority sarcopenia GO terms (muscle contraction,
autophagy, mitochondrion organization, mTOR signaling) with symmetric NMD screening identified
three verified cases:

**DMD (muscle contraction):** Score 0.978 (Dp427m; 1,225 aa, complete ORF) vs 0.001 (254 aa;
ratio = 1,263×). Functional distinction directly reflects known dystrophin biology.

**PINK1 (autophagy + mitochondrion org):** Score 0.924 vs 0.046 (ratio = 20×; autophagy) and
0.808 vs 0.068 (ratio = 12×; mitochondrion org) — cross-GO validation confirming model
mechanistic specificity.

**NDUFAF6 (mitochondrion org):** Ratio = 2,000×. Novel prediction: 129-aa isoform expected to
lack C-terminal LYRM domain required for Complex I integration.

**Novel annotation-gap genes:** NIPSNAP1 (score = 0.819, mitophagy "eat-me" signal; lacks GO:0007005
training label) and TAFAZZIN (score = 0.934 and 0.990 for two GO terms; cardiolipin transacylase
in Barth syndrome) identified as functional annotation gaps.

---

### 3.5 Evaluation robustness: multi-seed stability and bootstrap confidence intervals

Coefficient of variation across 5 seeds < 5% for 10 of 13 GO terms. Three terms showed higher
variance (Autophagy CV = 8.0%, TOR signaling CV = 6.1%, Glycolysis CV = 18.1%). Type-B
macro-AUPRC showed tight seed stability (mean = 0.685, std ≈ 0.017).

---

### 3.6 DIFFUSE achieves cross-tissue GO prediction on brain scRNA-seq without retraining

Applied zero-shot to 63,994 brain isoforms from the Samsung AD cohort, DIFFUSE achieved macro
AUPRC of 0.5998 (muscle held-out: 0.7022; Δ = −0.102, −14.5% relative; Supplementary Table S3).

Per-term analysis: the five best-transferred terms (mean Δ = −0.052) include Synaptic transmission
(+0.032, the only term that *improves* in brain — structurally coherent synaptic vesicle families).
The five worst-transferred terms (mean Δ = −0.151) include Autophagy (Δ = −0.172) and Mitochondrion
organization (Δ = −0.152), consistent with brain-specific autophagic machinery diverging from the
muscle repertoire.

pc1_var_ratio is tissue-agnostic: Pearson correlation between within-tissue muscle AUPRC and
cross-tissue brain AUPRC is r = +0.77 (p < 0.01), validating pc1_var_ratio as a pre-hoc
deployment guide for unstudied tissues.

Novel isoform evaluation: 7,899 structurally novel brain isoforms (NNIC/NIC) achieved macro AUPRC
0.3217. KNN propagation (k=5–20 nearest known isoforms) yielded macro AUPRC 0.267 — confirming
DIFFUSE outperforms proximity-based oracle for novel isoforms. Coding novel isoforms (n=5,796)
achieved macro AUPRC 0.408.

---

### 3.7 Alzheimer's disease isoform switches discovered by cross-tissue application

Integration with Dirichlet-multinomial DTU testing identified three high-confidence isoform switches
(q < 0.05, independent chi-square p < 1×10⁻⁵, cell-type restricted, model-supported).

**KIF21B — motor-domain switch in excitatory neurons (GO:0007018, GO:0031175).**
CT isoform tr293004 (NIC; 419 aa; 35.1% CT usage) was completely absent in AD (0.0%; p = 9.28×10⁻⁸).
AD isoform tr292978 (NNIC; 711 aa) absent in CT, emerged at 35.5% in AD (p = 3.81×10⁻⁶).
Exclusive to excitatory neurons; absent in all 7 remaining cell types.

tr293004 retains all three kinesin catalytic motifs: P-loop GQTGAGKT (aa 86), Switch-I SSRSHA
(aa 222), Switch-II DLAGSE (aa 273). tr292978 lacks all motor domain motifs but retains coiled-coil
(LLQEAL heptad) and WD40 cargo-binding domain (WDIRDS at aa 446). DIFFUSE scores: tr293004 = 0.966
(MT-based movement), tr292978 = 0.111 (Δ = −0.855). Mechanistic implication: bidirectional isoform switch eliminating the kinesin motor domain and gaining a WD40 β-propeller scaffold, with consequent transport disruption via heterodimer formation with full-length KIF21B-201.

**NDUFS4 — Complex I displacement by shared-TSS novel transcript in excitatory neurons (GO:0007005).**
Canonical NDUFS4: 44.1% CT usage → 7.1% AD usage. Novel tr73243 (NNIC; 379 aa): absent CT →
42.9% AD (p = 3.62×10⁻⁶). tr73243 TSS maps within 7 bp of canonical NDUFS4 TSS but encodes an
entirely distinct protein (98.3% divergence). MTS absent (first 40 aa: D+E = 4, HHH cluster at
positions 7–9). LYR motif absent. DIFFUSE scores: tr73243 = 0.024, NDUFS4-201 = 0.587 (Δ = −0.563).
Mechanism: TSS co-option — shared promoter (7 bp TSS proximity), functionally non-overlapping protein product.

**DLG1 — OPC isoform state transition (GO:0007268).**
CT-dominant tr319500 (NNIC; 187 aa; 80.9% CT OPC usage) declined to 11.9% in AD (p = 9.03×10⁻¹⁰).
Canonical DLG1 (906 aa, 3 PDZ domains) reciprocally increased. tr319500 lacks all PDZ GLGF-box
signatures; DIFFUSE score 0.033 vs canonical DLG1 0.818–0.927. Interpretation: loss of OPC-specialized
non-PDZ isoform, not loss of DLG1 function per se — consistent with OPC dedifferentiation in AD.

| Gene | CT-dominant isoform | DIFFUSE score | AD isoform | DIFFUSE score | Δ | Chi-sq p |
|------|--------------------|--------------|-----------|--------------|----|----------|
| KIF21B | tr293004 (419aa, NIC) | 0.966 | tr292978 (711aa, NNIC) | 0.111 | −0.855 | 9.3×10⁻⁸ |
| NDUFS4 | NDUFS4-201 (175aa) | 0.587 | tr73243 (379aa, NNIC) | 0.024 | −0.563 | 3.6×10⁻⁶ |
| DLG1 | tr319500 (187aa, NNIC) | 0.033 | DLG1-201 (906aa) | 0.818–0.927 | +0.857 | 9.0×10⁻¹⁰ |

---

### 3.8 BISECT multi-evidence characterization of AD isoform switches

#### 3.8.1 Stage 2 domain filtering identifies 26 functionally distinct isoform pairs

BISECT was applied to all 53 Stage 1 candidates. Pfam domain annotation (M2; HMMER 3.3.2,
E < 0.01, Pfam-A r36.0) identified domain set changes in 26/53 cases (49%). Domain losses
predominated (21 cases with ≥1 domain lost) over pure domain gains (8 cases with ≥1 domain
gained), with 3 cases showing concurrent loss and gain. Most frequently lost domain families:
Immunoglobulin-fold modules (Ig_3, I-set, V-set; 7 cases), Spectrin repeats (2 cases), and
catalytic domains (Y_phosphatase in PTPRF, Kinesin_motor in KIF21B, Fanconi_A in FANCA).

#### 3.8.2 Multi-evidence validation reveals cell-type-selective functional stratification

Application of M10–M12 to 26 Stage 2 PASS cases yielded 13/26 (50%) SUPPORTED by STRING PPI
at high confidence (combined score ≥ 700, experimental channel escore > 0) (Fig. 3; Supplementary Table S2).
Full case-by-case comparison of BISECT outputs against IsoformSwitchAnalyzeR — the current
standard for isoform-switch consequence annotation — is provided in Supplementary Table S5.
The observed rates across cell types were: Inhibitory neurons 5/7 (71%), Astrocytes 2/3 (67%),
Excitatory neurons 4/7 (57%), OPCs 1/2 (50%), Oligodendrocytes 1/6 (17%), Microglia 0/1 (0%).
Given the small per-cell-type sample sizes (1–7 cases), these rates are not statistically
distinguishable from the overall SUPPORTED rate of 50% (binomial test, all p > 0.10), and should
be interpreted as descriptive rather than inferential. The apparent trend — inhibitory neurons and
astrocytes showing higher STRING coverage — is consistent with what is biologically known: inhibitory
neurons have dense PPI networks related to GABA synaptic machinery (PPFIA, DLG, DMD/DAPC components),
which are well represented in STRING experimental interaction databases. However, this is a hypothesis
for why higher SUPPORTED rates might be observed in larger samples, not a demonstrated cell-type
specificity effect in the present panel.

Four cases initially classified UNSUPPORTED were reclassified SUPPORTED after hypothesis expansion:
DMD and SNTG1 (DAPC partners DAG1/SNTA1 not in initial hypothesis set), PTPRS (Liprin-α partners
PPFIA1/NTRK3 absent), BSG (SLC16A1 MCT1 transporter absent). The UNSUPPORTED verdicts (13/26)
represent genuine negative signal where STRING experimental evidence is absent for all tested hypotheses.

Evolutionary conservation (M12) spanned from phyloP = −0.493 (FANCA AD exon, accelerated evolution)
to phyloP = 4.826 (IFT122, very highly conserved). Three cases showed AD-specific exons with
negative phyloP: FANCA (−0.493), BSG (−0.473), IFI16 (−0.089) — all accelerated evolution.

#### 3.8.3 Tier A: Functionally reprogrammed isoforms with complete multi-evidence support

Three cases received Tier A classification: KIF21B, NDUFS4, and DLG1 — each involving at least one novel NIC/NNIC isoform with complete STRING PPI support (M11 SUPPORTED).

*KIF21B* (excitatory neurons): ESMFold structural predictions for both novel isoforms — CT kinesin
motor (Fig. 2a) (aa 1–380; pLDDT = 93.2) and AD WD40 β-propeller core (aa 370–620; pLDDT = 94.6) — both
exceed the very-high-confidence threshold (≥90), establishing that the switch is not loss-of-structure
but gain-of-fold between two independently ordered architectures. STRING confirmed AD WD40 domain
complement links to TRIM3 (765), STK36 (691), SMO (694) — non-canonical kinesin partners. Both
exon sets show very high conservation (AD phyloP = 4.067; CT phyloP = 3.842), indicating both
architectures maintained under purifying selection across 100 vertebrates.

*NDUFS4* (excitatory neurons): NNIC tr73243 (379 aa) acquires RVT_1 domain with no Complex I interactors. CT isoform (UniProt O43181, pLDDT = 84.1) is a stable Complex I assembly subunit (STRING: NDUFS6 = 999, NDUFA12 = 999, NDUFAF2 = 996); LYR motif absent from tr73243 confirms functional decoupling from Complex I assembly. AD RVT_1 exons: mean phyloP = 2.263 (highly conserved), indicating this novel sequence is under independent functional constraint in a different cellular context. The TSS of tr73243 maps within 7 bp of the canonical NDUFS4 promoter (shared-TSS co-option), consistent with competitive promoter usage as a mechanism for canonical NDUFS4 suppression beyond observed DTU.

*DLG1* (OPC): CT-specific NNIC tr319500 (187 aa) declines from 80.9% CT OPC usage to 11.9% in AD (p = 9.03×10⁻¹⁰). AlphaFold (Q12959, pLDDT = 73.1) and STRING (GRIN2B = 992, CASK = 980, DLGAP1 = 952) confirm canonical DLG1-201 as the AD-predominant isoform. CT-specific exons mean phyloP = 2.507, indicating the sequence lost in the AD switch is evolutionarily constrained. The NNIC classification of tr319500 identifies it as a novel OPC-specialized scaffold isoform; its loss — replaced by canonical PDZ-domain DLG1 — is consistent with OPC state reversion rather than global DLG1 loss of function.

#### 3.8.4 Tier B: Domain-loss isoforms with consistent functional predictions

Five candidates — PTPRF, FANCA, IFT122, SYNE1, RGS3 — were classified Tier B on the basis of
FSM isoform classification with consistent M11 support plus mechanistically coherent domain-change narratives.

*PTPRF* (Fig. 2b) (inhibitory neurons): Domain-level pLDDT from the canonical AlphaFold model (UniProt P10586) reveals that the CT isoform retains phosphatase catalytic domains (Y_phosphatase pLDDT = 72.3; DSPc = 78.6) absent from the AD isoform, which contains only extracellular Ig-fold modules (pLDDT = 84–87). STRING: PPFIA1 = 997, PPFIA3 = 996, CTNNB1 = 982. M11 contradicted the initial SLIT2 decoy hypothesis (STRING score = 0 for SLIT2–PTPRF) and revised the model to Liprin-α dominant-negative: the AD isoform sequesters synaptic scaffold partners without phosphatase output. CT PTP exons more conserved (phyloP = 4.341) than AD Ig exons (2.835), consistent with the ancestral phosphatase-active isoform under stronger purifying selection.

*FANCA* (excitatory neurons): AD isoform (297 aa, FSM) loses Fanconi_A domain required for core complex scaffolding. STRING: FANCF = 999, FANCC = 999, FANCE = 999, BRCA1 = 995, UBE2T = 998 — the entire Fanconi Anemia repair complex is confirmed for the CT isoform. M12 evidence: CT exons (33 exons) mean phyloP = 1.321; AD exon = −0.493 (only negative phyloP in the panel), indicating accelerated evolution inconsistent with functional constraint. The primary functional consequence supported by multi-evidence data is selective suppression of DNA interstrand crosslink repair in AD excitatory neurons. One speculative but testable model is that FANCA loss promotes R-loop accumulation — supported by the established link between Fanconi Anemia pathway deficiency and R-loop formation (Groh et al. 2014; Walker et al. 2021) — which in turn activates ATM-mediated DNA damage signalling. We hypothesize that ATM activation in this context may engage CDK5, a kinase with established roles in tau phosphorylation in neurodegeneration, and that downstream consequences could include tau hyperphosphorylation and TDP-43 mislocalization. However, the complete causal chain from FANCA isoform switching in AD neurons to tau pathology has not been established, and each step in this pathway would require independent experimental support. Experimental validation of FANCA-R-loop interactions in AD neurons would be required to test this model.

*IFT122*: Strongest STRING support in the panel — four intraflagellar transport partners at maximum
confidence (IFT140 = 999, WDR35 = 999, IFT43 = 999, WDR19 = 999). Both AD-specific (phyloP = 4.826)
and CT-specific (phyloP = 4.673) exons are the most conserved in the entire panel.

*SYNE1*: AD isoform loses Spectrin repeat domains, disrupting LINC complex.
STRING: SUN1 = 999, SUN2 = 999, LMNA = 996. CT Spectrin exons more conserved (phyloP = 4.228)
than AD exons (3.450).

*RGS3*: AD isoform retains only RGS GAP domain; loses C2 and three PDZ domains.
STRING: EFNB2 = 996, GNAI2 = 953. CT multi-domain exons: mean phyloP = 2.687 (conserved).

#### 3.8.5 Tier C: Evidence divergence as calibration

*ADGRB2*: AD isoform gains HRM domain. STRING returns RANBP2 (900), RANGAP1, XPOT, KPNB1 — coherent
nuclear transport cluster. However, AD HRM exon phyloP = 0.075 (low conservation), inconsistent
with functional constraint. This M11/M12 divergence flags ADGRB2 for lower experimental priority,
demonstrating that multi-evidence divergence is itself an interpretable signal.

#### 3.8.6 Pathway-level convergence: DAPC remodelling and LAR-RPTP parallels

Two independent inhibitory neuron switches converge on the dystrophin-associated protein complex (Fig. 4):
*DMD* (phyloP AD = 4.823; STRING: DAG1 = 999, SNTA1 = 998, SNTG1 = 992) and *SNTG1*
(phyloP AD = 4.558; STRING: DMD = 992, SNTA1 = 938) — mutually confirming each other as top
STRING partners. Together they represent the first evidence of systematic DAPC remodelling through
alternative splicing in AD, with implications for inhibitory synapse GABAergic maintenance.

A structural parallel to PTPRF (inhibitory neurons) was identified in astrocytes: *PTPRS*, a LAR-RPTP
family member, shows CT-specific exons with mean phyloP = 3.919 and STRING PPFIA1 = 985,
LRRC4B = 989 — the same Liprin-α network as PTPRF. Convergent disruption of the LAR-RPTP
Liprin-α scaffold in both inhibitory neurons and astrocytes suggests a cell-type-distributed
mechanism for synaptic scaffold organisation impairment in AD.

#### 3.8.7 M3 motif analysis reveals sub-domain functional evidence

M3 regular-expression scanning detected mechanistically informative motif-level evidence:

*KIF21B*: CT isoform contains all three canonical kinesin mechanochemical elements (P-loop at aa 87,
Switch-I at aa 222, Switch-II at aa 273). None detected in AD WD40 isoform. Additionally, AD isoform
carries GVGF PDZ-type motif at position 251 — absent from CT kinesin — potentially enabling
WD40 scaffold to recruit PDZ-domain-containing postsynaptic density proteins.

*FANCA*: LYR tripeptide at position 358 of CT isoform (1455 aa) — absent from AD isoform (297 aa)
— mediates LYREX module interactions in Fanconi Anemia complex assembly. Complements M2-level
Fanconi_A domain loss with sequence-resolution interaction element loss.

RepeatMasker analysis (M5): no young LINE-1 elements (divergence < 15%) overlapping CDS in any of
the 26 Stage 2 cases, arguing against transposable element-mediated exonisation in this panel.
MTS composite scoring (M3): SYNE1 CT isoform scored highest in the panel (composite = 4; net charge
= +3; μH = 0.253), warranting investigation of mitochondria-associated nuclear envelope organisation.

---

## 4. Discussion

### 4.1 pos_bias: GO-term-dependent isoform discrimination with negative control validation

The central methodological challenge in isoform-level function prediction is disentangling
isoform-specific signal from gene-level information encoded in protein language model embeddings.
pos_bias addresses this through empirically validated null levels: gene-mean predictor (0.000),
shuffled-label model (0.240 ± 0.048), random predictor (0.898 ± 0.041). DIFFUSE exceeds the
shuffled-label floor in all three tested terms (Δ > 3× noise floor), confirming GO label training
drives genuine within-gene score differentiation.

DIFFUSE achieves pos_bias = 1.902 for muscle contraction (GO:0006941), substantially exceeding
both the shuffled-label floor and the random ceiling — strongest evidence of learned isoform
discrimination. Crucially, pos_bias is virtually unchanged when restricted to protein-coding
isoforms (macro = 0.985, Δ = −0.022).

### 4.2 The Type-A/B framework: embedding geometry as post-hoc characterisation of model utility

sep_cosine (LOOCV 13/13 = 100% on 13 evaluated GO terms, τ = 0.060) characterises GO terms where
isoform-level modelling adds value. This is a descriptive characterisation rather than a prospective
classifier validated on held-out GO terms; generalisation requires external validation. The decision
gap of (0.056, 0.167) implies genuine bimodal structure rather than threshold sensitivity.

### 4.3 Computational predictions from isoform-switch analysis

DIFFUSE muscle isoform-switch predictions are computational predictions consistent with known
biology, providing experimentally testable hypotheses. DMD Dp427m (ratio = 1,263×) recapitulates
known dystrophin biology. PINK1 cross-GO consistency (ratio = 20× and 12×) provides cross-term
validation. NDUFAF6 (ratio = 2,000×) is a novel prediction: 129-aa isoform expected to lack LYRM
domain required for Complex I integration.

### 4.4 NMD screening reveals a systematic data quality issue

23/126 (18.3%) candidate pairs excluded by symmetric NMD screening. BNIP3 — both isoforms with
5′-partial ORFs (PTC-to-EJC = 1,878 nt) — exemplifies the failure mode where score ratio reflects
annotation incompleteness rather than differential function. Symmetric NMD screening (both isoforms)
identified 9 additional cases where the *top* (high-scoring) isoform was itself an NMD candidate.
This pipeline is recommended as routine quality filter for any isoform-switch candidate list.

### 4.5 Limitations and future directions

**Gene-level label propagation and isoform-level annotation refinement.** DIFFUSE does not claim to
predict isoform function from first principles independent of gene-level knowledge. Rather, its
contribution is more precisely framed as *isoform-level annotation refinement*: given that
gene-level functional databases (UniProt, GO) annotate at the gene level despite the fact that
distinct isoforms produce structurally different protein products, DIFFUSE infers which isoforms
within a gene are most likely to carry a given function, based on their sequence-encoded structural
features. This is a meaningful distinction — gene-level databases cannot, by design, resolve which
splice variant actually executes a catalytic function or localises to the correct compartment.

This framing is supported by empirical evidence. The pos_bias metric (Section 3.3) demonstrates
that for Type-B GO terms, DIFFUSE's within-gene isoform score discrimination significantly exceeds
the shuffled-label noise floor (shuffled-label pos_bias = 0.240 ± 0.048; DIFFUSE maximum = 1.902
for muscle contraction; 11/13 terms q < 0.05). This means DIFFUSE is not simply memorising
gene identity: it differentially scores isoforms of the same gene based on their sequence content.
The gene-level label is the training signal, but the learned representation goes beyond reproducing it.

A genuine limitation remains: without isoform-level experimental ground truth (e.g., isoform-specific
knockout phenotypes, or direct measurements of which splice variant executes a given function), the
ceiling of isoform-level prediction accuracy cannot be quantified. Future work incorporating
UniProt isoform-specific functional annotations, massively parallel splicing reporters (Julien et al.,
*Nat Biotechnol*, 2016), or isoform-selective loss-of-function screens would allow direct validation
of isoform-level predictions and benchmarking against isoform-resolution ground truth.

**Cross-tissue transferability.** Cross-tissue performance governed by pc1_var_ratio, not tissue
specificity of GO term. Autophagy (Δ = −0.172) is worst-transferred despite being universally
conserved — brain-specific autophagic machinery diverges from muscle repertoire. Multi-tissue
training using GTEx long-read RNA-seq is the most direct path for Case 3 GO terms.

**Structural validation.** AlphaFold DB covers canonical sequences only; ESMFold for 36,748 isoforms
was computationally prohibitive. For top-priority candidates, ESMFold pLDDT profiles could test
whether functional scores correlate with structural domain integrity in low-scoring isoforms.

### 4.6 Cross-tissue generalization and the limits of zero-shot transfer

The muscle-trained DIFFUSE model achieving macro AUPRC 0.600 on a completely independent brain
tissue dataset represents a non-trivial generalization. ESM-2 pretraining on evolutionary sequence
variation encodes structural domain presence in tissue-agnostic ways — a catalytic triad or coiled-coil
motif contributes to the embedding regardless of tissue expression context. Cross-tissue degradation
is mechanistically informative: largest for GO terms with divergent muscle-brain positive-class protein
families; smallest for structurally invariant families expressed in both tissues.

### 4.7 AD isoform switches: three distinct mechanisms revealed by DIFFUSE-DTU integration

The three AD isoform switches each represent a mechanistically distinct class:

**KIF21B:** Motor-competent (tr293004) replaced by motor-incompetent (tr292978) retaining
coiled-coil and WD40. Dominant-negative transport disruption: heterodimers with KIF21B-201
reduce dendritic transport processivity, predicted to affect AMPA receptor subunits and mRNA
granule delivery — missed by gene-level analysis if total KIF21B expression is unchanged.

**NDUFS4:** TSS co-option — tr73243 shares the NDUFS4 promoter (7 bp TSS proximity)
but encodes a 379-aa protein with no MTS, no LYR motif, 98.3% sequence divergence. Compounded
suppression: promoter competition may reduce canonical NDUFS4 transcription beyond observed DTU
statistics. Complex I dysfunction in excitatory neurons is among the most replicated findings in
AD post-mortem proteomics.

**DLG1:** The switch is not loss of DLG1 function but loss of OPC-specialized tr319500 (PDZ-lacking,
low-scoring). Canonical PDZ-containing DLG1 replacing it is consistent with OPC state reversion —
we hypothesize a shift from myelinating-lineage toward undifferentiated transcriptomic state, in
line with broad OPC transcriptional alterations reported in AD single-nucleus profiling
(Mathys et al., *Nature* 2019; PMID: 31042697).

### 4.8 BISECT: structured evidence integration for hypothesis generation and revision

BISECT's key methodological contribution is structured evidence integration that prevents
confirmation bias: by requiring PPI network concordance (M11), evolutionary conservation (M12),
and structural confidence (M10) as separate evidence layers, BISECT ensures that candidate
mechanisms are not accepted on the basis of literature analogy alone. This forces evidence-based
iteration before a mechanistic hypothesis is accepted.

The PTPRF case demonstrates this process directly. An initial hypothesis — that the AD isoform
retaining only extracellular Ig-fold modules might function as a SLIT2 decoy receptor — was
contradicted by PPI network evidence: STRING returned a score of 0 for SLIT2–PTPRF interaction.
Rather than accepting the literature-motivated hypothesis, BISECT required revision based on the
actual top STRING partners: PPFIA1 = 997, PPFIA3 = 996. The revised model — that the AD isoform
sequesters Liprin-α scaffold partners without phosphatase output, acting as a dominant-negative
scaffold — emerged directly from this evidence-based iteration. This is hypothesis generation and
revision through sequential evidence integration, not post-hoc annotation. The methodological value
is that it prevents confirmation bias: a hypothesis motivated by mechanistic plausibility is
subjected to PPI network, structural, and evolutionary evidence layers before being reported.

Tier C case ADGRB2 demonstrates the framework's calibration capacity: multi-evidence divergence
(M11 SUPPORTED with RANBP2 = 900 vs M12 low conservation phyloP = 0.075) is itself an
interpretable signal that flags candidates for lower experimental priority.

At the pathway level, BISECT identifies convergence that statistical detection alone cannot reveal:
independent DAPC disruption in inhibitory neurons (DMD + SNTG1), parallel LAR-RPTP Liprin-α
scaffold disruption across cell types (PTPRF inhibitory + PTPRS astrocyte), and accelerated
evolution at two oligodendrocyte metabolic switches (BSG, IFI16). These convergences emerge only
from cross-case systematic comparison (M7), not from individual case analysis.

### 4.9 Conclusion

DIFFUSE establishes isoform-level GO term prediction as computationally tractable when protein
language model embeddings are paired with appropriate loss functions for sparse functional labels
and within-gene contrastive geometry. The Type-A/B classification framework, pos_bias metric,
and symmetric NMD quality-screening pipeline introduced here are general tools applicable beyond
skeletal muscle and sarcopenia.

Applied cross-tissue to Alzheimer's disease long-read single-cell data, DIFFUSE achieves meaningful
zero-shot generalization (macro AUPRC 0.600) and, integrated with DTU testing, identifies three
AD-specific isoform switches. BISECT v1.1, applied to 53 statistical candidates, provides
multi-evidence biological characterisation that converts the statistical signal into mechanistic
hypotheses with direct experimental testability and identifies pathway-level convergences invisible
to case-by-case analysis. Together, DIFFUSE and BISECT demonstrate that isoform-level functional
prediction and mechanistic characterisation are not tissue-specific problems: the same framework
discovers and characterises isoform switches in both musculoskeletal and neurodegenerative disease
contexts from a model trained on a single tissue type.

---

## References

*(References from manuscript_full_english.md retained in full; additional BISECT-specific
references listed below.)*

### Core references (DIFFUSE)
1. Cruz-Jentoft AJ et al. Sarcopenia: revised European consensus on definition and diagnosis. *Age Ageing* 2019;48(1):16-31. PMID: 30312372
2. Lin Z et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 2023;379(6637):1123-30. PMID: 36927031
3. Lin TY et al. Focal Loss for Dense Object Detection. *ICCV* 2017. arXiv:1708.02002
4. Hermans A et al. In Defense of the Triplet Loss for Person Re-Identification. *arXiv* 2017. arXiv:1703.07737
5. Pan Q et al. Deep surveying of alternative splicing complexity. *Nat Genet* 2008;40(12):1413-5. PMID: 18978772
6. Tardaguila M et al. SQANTI. *Genome Res* 2018;28(3):396-411. PMID: 29440212
7. Maquat LE. Nonsense-mediated mRNA decay. *Nat Rev Mol Cell Biol* 2004;5(2):89-99. PMID: 15040442
8. Gligorijević V et al. Structure-based protein function prediction using graph convolutional networks. *Nat Commun* 2021;12(1):3168. PMID: 34039969
9. Prjibelski AD et al. Accurate long-read transcriptome quantification and differential analysis with IsoQuant. *Nat Biotechnol* 2023. PMID: 37542202
10. Mathys H et al. Single-cell transcriptomic analysis of Alzheimer's disease. *Nature* 2019;570:332-337. PMID: 31042697

### BISECT-specific references
11. Jumper J et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 2021;596:583-9. PMID: 34265844
12. Varadi M et al. AlphaFold Protein Structure Database. *Nucleic Acids Res* 2022;50(D1):D439-44. PMID: 34791371
13. Szklarczyk D et al. STRING v12.0: protein–protein association networks with increased coverage. *Nucleic Acids Res* 2023. PMID: 36370105
14. Hubisz MJ et al. PHAST and RPHAST: phylogenetic analysis with space/time models. *Brief Bioinform* 2011;12(1):41-51. PMID: 21278375
15. van den Heuvel L et al. Demonstration of a new pathogenic mutation in human complex I deficiency. *Nat Genet* 1998;18:195-7. PMID: 9462751
16. Formosa LE et al. Building a complex complex: Assembly of mitochondrial respiratory chain complex I. *EMBO J* 2020;39:e102817. PMID: 32432371
17. Lee Y et al. Oligodendroglia metabolically support axons and contribute to neurodegeneration. *Cell* 2012;151(7):1535-48. PMID: 23260143
18. Lu Q et al. δ-Catenin, an adhesive junction-associated protein, is related to cortactin and p120. *J Biol Chem* 2001 [Corrected: Lu M et al. *Cell* 2001;105(1):69-79. PMID: 11301003]
19. Bhatt DL et al. DLG1 in oligodendrocytes. *J Neurosci* 2009. PMID: 19625516
20. Groh M et al. R-loops associated with triplet repeat expansions promote gene silencing in *Friedreich ataxia* and fragile X syndrome. *PLoS Genet* 2014; PMID: 24415955 [correct citation for R-loop/Groh: Groh M et al. 2017, *Nat Commun*; Walker et al. 2021, *Nucleic Acids Res*]
21. Mistry J et al. Pfam: The protein families database in 2021. *Nucleic Acids Res* 2021. PMID: 33125078
22. Eddy SR. Accelerated Profile HMM Searches. *PLOS Comput Biol* 2011. PMID: 22039361
23. Haas BJ et al. De novo transcript sequence reconstruction from RNA-seq using the Trinity platform for reference generation and analysis. *Nat Protoc* 2013. PMID: 23845962

---

## Figure Legends

### Figure 1 — BISECT pipeline overview

**Figure 1 | BISECT pipeline architecture and case funnel.** *(a)* Schematic of the twelve-module
BISECT pipeline. Input: 53 candidate CT/AD isoform pairs filtered by |DIFFUSE Δ| ≥ 0.5 and
DTU p ≤ 10⁻⁵ (Stage 1). Stage 2 Pfam domain-change filter (M2; HMMER 3.3.2; E < 0.01) reduces
to 26 cases. M3–M9 apply sequence, motif, genomic context, and quality-control analysis. M10–M12
biological validation modules: structural confidence (AlphaFold DB/ESMFold), PPI network (STRING
v12.0), and evolutionary conservation (phyloP 100way). Output: per-case evidence tier (A/B/C)
with evidence-based PPI hypothesis revision. *(b)* Scatter plot of DIFFUSE Δ vs STRING combined
score for all 26 Stage 2 PASS cases, coloured by M11 verdict. Tier A cases annotated.

### Figure 2 — Isoform switch domain maps

**Figure 2 | Domain architecture changes at Tier A isoform switches.** Scaled protein domain maps
for three Tier A candidates. Top track: CT-predominant isoform; bottom track: AD-predominant
isoform. Domain blocks coloured by Pfam family (Kinesin in teal, WD40 in green, Complex I subunit
in orange, RVT_1 in purple, PDZ in blue). Novel junctions indicated by dashed borders.
*(a)* KIF21B: 419-aa CT kinesin motor fragment vs 711-aa AD WD40 β-propeller (NIC → NNIC).
*(b)* NDUFS4: CT 175-aa Complex I subunit (LYR motif, NDUFS4 core) vs AD 379-aa NNIC tr73243
(RVT_1 domain; no Complex I signature; shared-TSS co-option). *(c)* DLG1: CT 187-aa NNIC
tr319500 (no PDZ domains, OPC-specialized) vs AD 906-aa canonical DLG1-201 (3 PDZ + MAGUK scaffold).

### Figure 3 — Multi-evidence heatmap (26-case BISECT panel)

**Figure 3 | BISECT multi-evidence validation across 26 Stage 2 AD isoform switch candidates.**
*(a)* Heatmap of 26 Stage 2 PASS cases × 5 evidence columns: |DIFFUSE Δ| (purple), domains lost
(red), domains gained (blue), M11 PPI verdict (green/grey binary), AD-specific exon mean phyloP
(diverging blue-red). Rows ordered Tier A → B → C, within tier by |Δ| descending. Cell-type chips
shown at left. *(b)* Cell-type stratification of M11 SUPPORTED rate across all 26 Stage 2 cases.
*(c)* Per-residue ESMAtlas pLDDT traces for KIF21B CT kinesin (red, pLDDT = 93.2) and AD WD40
(blue, pLDDT = 94.6). *(d)* NDUFS4 evidence summary: DIFFUSE score, STRING score, phyloP, MTS
composite.

### Figure 4 — Pathway convergence network

**Figure 4 | Pathway convergence of 13 M11-SUPPORTED isoform switches.** *(a)* Manual network
layout of 13 SUPPORTED cases in four biological pathway clusters: WD40/motor redistribution
(orange, IFT122 and KIF21B), Spectrin/DAPC complex (blue, DMD/SNTG1/SYNE1), Phosphatase &
G-protein signalling (green, PTPRF/PTPRS/RGS3/ADGRB2), Organelle & specialised function
(purple, FANCA/NDUFS4/BSG/DLG1). Node size proportional to |DIFFUSE Δ|; node colour = cell
type; Tier A cases shown with additional ring. Selected STRING edges and external interaction
partners shown. FANCA and BSG annotated with negative phyloP scores. *(b)* M11 SUPPORTED rate
by cell type (all 26 Stage 2 cases).

---

## Supplementary Information

### Supplementary Table S1 — ESM-LR / ESM-RF baseline comparison (all 13 GO terms)

All models use ESM-2 640d embeddings. LR: C=1.0, class_weight='balanced'. RF: n_estimators=200,
min_samples_leaf=5, class_weight='balanced'. CIs: gene-block bootstrap n=500.

| GO Term | Function | Type | DIFFUSE | ESM-LR (95% CI) | ESM-RF (95% CI) |
|---------|----------|------|---------|-----------------|-----------------|
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

### Supplementary Table S2 — M10–M12 evidence summary (13 SUPPORTED cases)

| Gene | Cell Type | Domain Change | M10 CT pLDDT | M10 AD pLDDT | M11 Verdict | Top STRING Partners (score) | M12 AD phyloP | M12 CT phyloP |
|------|-----------|--------------|-------------|-------------|------------|---------------------------|-------------|-------------|
| KIF21B | Excitatory | Lost: Kinesin; Gained: WD40×9 | 93.2 (ESMAtlas)† | 94.6 (ESMAtlas)† | SUPPORTED | TRIM3=765, SMO=694, STK36=691 | 4.067 | 3.842 |
| PTPRF | Inhibitory | AD lacks PTP domain | 82.0 [PTP:72.3]* | 82.0 [Ig:85.0]* | SUPPORTED | PPFIA1=997, PPFIA3=996, CTNNB1=982 | 2.835 | 4.341 |
| FANCA | Excitatory | Lost: Fanconi_A | 74.9 (O15360)* | 74.9* | SUPPORTED | FANCF=999, FANCC=999, BRCA1=995 | **−0.493** | 1.321 |
| NDUFS4 | Excitatory | Gained: RVT_1 | 84.1 (O43181)† | N/A | SUPPORTED | NDUFS6=999, NDUFA12=999, NDUFAF2=996 | 2.263 | N/A§ |
| DLG1 | OPC | Gained: L27_1, MAGUK, PDZ | N/A§ | 73.1 (Q12959) | SUPPORTED | GRIN2B=992, CASK=980, LIN7C=956 | N/A§ | 2.507 |
| IFT122 | Excitatory | Lost: WD40/eIF2A; Gained: Clathrin/TPR | 82.9 (Q9HBG6)* | 82.9* | SUPPORTED | IFT140=999, WDR35=999, IFT43=999 | 4.826 | 4.673 |
| SYNE1 | Inhibitory | Lost: Spectrin | 76.6 (Q8NF91)* | 76.6* | SUPPORTED | SUN1=999, SUN2=999, LMNA=996 | 3.450 | 4.228 |
| RGS3 | Astrocyte | Lost: C2+PDZ×3 | 55.0 (P49796)* | 55.0* | SUPPORTED | EFNB2=996, GNAI2=953, GNAI3=956 | N/A§ | 2.687 |
| ADGRB2 | Inhibitory | Gained: HRM | 61.4 (O60241)* | 61.4* | SUPPORTED | RANBP2=900, MAGI1=693 | 0.075 | 1.935 |
| DMD | Inhibitory | Lost: DAPC-anchor exons | 75.3* | 75.3* | **SUPPORTED** | DAG1=999, SNTA1=998, SNTB1=996 | 4.823 | 3.350 |
| SNTG1 | Inhibitory | Lost: PH2 domain | 68.1* | 68.1* | **SUPPORTED** | DMD=992, SNTA1=938, SNTB1=874 | 4.558 | N/A§ |
| PTPRS | Astrocyte | Lost: SusE; Gained: Ig | 81.5* | 81.5* | **SUPPORTED** | NTRK3=999, PPFIA1=985, LRRC4B=989 | N/A§ | 3.919 |
| BSG | Oligodendrocyte | Lost: Ig_5, V-set | 86.1* | 86.1* | **SUPPORTED** | SLC16A1=999, PPIA=999, CD44=985 | −0.473 | −0.051 |

† KIF21B: ESMAtlas segmental (CT aa 1–380; AD aa 370–620). NDUFS4: isoforms share same UniProt.
§ N/A: no isoform-specific exons detectable on indicated side.
* Both isoforms map to same UniProt; domain-level pLDDT from canonical model.
**Bold** SUPPORTED = reclassified after hypothesis update.

### Supplementary Table S3 — Muscle vs Brain AUPRC per GO term (v15d_bp_clean)

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

orig = original 13-term muscle panel; neuro = neural-enriched terms in v15d.
Bold: synaptic transmission is the only term that improves cross-tissue.

### Supplementary Table S4 — Samsung AD cohort: cell-type composition per donor group

| Cell Type | AD donors (n=13) | CT donors (n=8) | Total nuclei (est.) | DTU-tested isoforms |
|-----------|-----------------|-----------------|--------------------|--------------------|
| Excitatory neuron | 13/13 | 8/8 | ~148,000 | 40,198 |
| Inhibitory neuron | 13/13 | 8/8 | ~42,000 | 28,341 |
| Oligodendrocyte | 13/13 | 8/8 | ~95,000 | 35,672 |
| OPC | 12/13 | 8/8 | ~18,000 | 22,104 |
| Astrocyte | 13/13 | 8/8 | ~38,000 | 31,887 |
| Microglia | 13/13 | 8/8 | ~22,000 | 19,543 |
| Vascular cell | 11/13 | 7/8 | ~8,000 | 11,229 |
| Lymphocyte | 10/13 | 6/8 | ~4,000 | 6,820 |
| **Total** | **13 AD** | **8 CT** | **~375,000** | **63,994 (total unique)** |

Cell type annotation based on canonical marker gene expression (excitatory: CAMK2A, SLC17A7; inhibitory: GAD1/2; oligodendrocyte: MBP, PLP1; OPC: PDGFRA, CSPG4; astrocyte: GFAP, AQP4; microglia: CX3CR1, TMEM119; vascular: CLDN5, FN1; lymphocyte: CD3E, CD8A). Donors per cell type reflects nuclei passing minimum 50-cell threshold for pseudobulk DTU testing. DTU-tested isoforms: genes with ≥10 total counts across all donors for that cell type. Isoform counts per cell type are not mutually exclusive (same isoform can be expressed in multiple cell types). AD diagnosis: NIA-AA criteria, Braak stage V–VI; CT: Braak stage ≤ II, no neurological diagnosis. SMC = Samsung Medical Center (donors SMC027, SMC030, SMC033, SMC052 are control group); PO = additional post-mortem control donors (PO13, PO15, PO20, PO23).

### Supplementary Table S5 — BISECT vs IsoformSwitchAnalyzeR: analysis of all 26 Stage 2 PASS cases

IsoformSwitchAnalyzeR (ISA; Vitting-Seerup and Sandelin, *Bioinformatics*, 2019) was used as a reference comparator because it represents the current state-of-the-art for isoform-switch consequence annotation. Both tools apply Pfam domain annotation (M2/ISA-Pfam) as a core analysis layer; however, ISA does not incorporate PPI network validation (BISECT M11), AlphaFold structural confidence (BISECT M10), or evolutionary conservation scoring (BISECT M12). BISECT additionally applies a DIFFUSE functional score pre-filter (Stage 1: |Δ| ≥ 0.5, DTU p ≤ 1×10⁻⁵) that ISA lacks, targeting candidates with large functional score changes at stringent DTU significance. The 26 Stage 2 PASS cases below represent all cases where Pfam domain annotation detects ≥1 domain loss or gain; ISA would detect the same 26 domain-change events given equivalent input. BISECT uniquely stratifies these 26 into SUPPORTED (n=13, 50%) and UNSUPPORTED (n=13, 50%) based on multi-evidence integration. **Bold = Tier A (novel NIC/NNIC isoform + SUPPORTED). Italic = Tier B (known FSM/ISM + SUPPORTED). Plain = Tier C (UNSUPPORTED).**

| Gene | Cell type | CT category | AD category | Pfam domains lost (n) | Pfam domains gained (n) | ISA: domain change | BISECT M11 verdict | Top PPI partner | STRING score | BISECT M10 AD pLDDT | BISECT M12 AD phyloP | Tier |
|------|-----------|-------------|-------------|----------------------|-------------------------|-------------------|--------------------|----------------|-------------|---------------------|---------------------|------|
| **KIF21B** | Excitatory | NIC | NNIC | DUF5082; Kinesin; Microtub_bd (3) | ANAPC4_WD40; NBCH_WD40; Nup160; WD40 (4) | YES | **SUPPORTED** | TRIM3 | 765 | ESMFold 93.2/94.6 | 4.067 | **A** |
| **NDUFS4** | Excitatory | FSM | NNIC | — (0) | RVT_1 (1) | YES | **SUPPORTED** | NDUFB5 | 999 | NA (novel isoform) | 2.263 | **A** |
| **DLG1** | OPC | NNIC | FSM | L27_1; MAGUK_N_PEST; PDZ_assoc (3) | — (0) | YES | **SUPPORTED** | GRIA1 | 999 | 73.05 | NA | **A** |
| *PTPRF* | Inhibitory | FSM | FSM | Arylsulfotran_N; DSPc; Y_phosphatase; fn3 (10) | C2-set_2; I-set; Ig_5; V-set (8) | YES | *SUPPORTED* | PPFIA1 | 997 | 81.99 | 2.835 | *B* |
| *FANCA* | Excitatory | FSM | FSM | Fanconi_A (1) | — (0) | YES | *SUPPORTED* | FANCF | 999 | 74.86 | −0.493 | *B* |
| *SYNE1* | Inhibitory | FSM | FSM | Spectrin (1) | — (0) | YES | *SUPPORTED* | SUN2 | 999 | 76.60 | 3.450 | *B* |
| *IFT122* | Excitatory | FSM | FSM | ANAPC4_WD40; NBCH_WD40; WD40; eIF2A (4) | Clathrin; TPR_14; TPR_19 (3) | YES | *SUPPORTED* | WDR35 | 999 | 82.86 | 4.826 | *B* |
| *RGS3* | Astrocyte | FSM | FSM | C2; CEP76-C2; PDZ; PDZ_2; PDZ_6 (5) | — (0) | YES | *SUPPORTED* | EFNB2 | 996 | 54.97 | NA | *B* |
| *ADGRB2* | Inhibitory | FSM | FSM | — (0) | HRM (1) | YES | *SUPPORTED* | RANBP2 | 900 | 61.39 | 0.075 | *B* |
| *DMD* | Inhibitory | FSM | FSM | Spectrin; WW (2) | SOGA (1) | YES | *SUPPORTED* | DAG1 | 999 | 76.35 | 4.823 | *B* |
| *SNTG1* | Inhibitory | FSM | FSM | — (0) | PH (1) | YES | *SUPPORTED* | DMD | 992 | 82.34 | 4.558 | *B* |
| *PTPRS* | Astrocyte | FSM | FSM | SusE (1) | Ig_C17orf99 (1) | YES | *SUPPORTED* | NTRK3 | 999 | 81.49 | NA | *B* |
| *BSG* | Oligodendrocyte | FSM | FSM | Ig_5; V-set (2) | — (0) | YES | *SUPPORTED* | SLC16A1 | 999 | 86.11 | −0.473 | *B* |
| ANKRD44 | Oligodendrocyte | FSM | FSM | — (0) | GATase_7 (1) | YES | UNSUPPORTED | PPP6R1 | 964 | 90.58 | 3.712 | C |
| ASXL3 | Excitatory | FSM | FSM | HARE-HTH; PHD_3 (2) | — (0) | YES | UNSUPPORTED | BAP1 | 917 | 39.37† | 2.827 | C |
| CCAR1 | Inhibitory | FSM | FSM | — (0) | S1-like (1) | YES | UNSUPPORTED | PRPF40A | 935 | 69.22 | 4.204 | C |
| DOCK11 | Inhibitory | FSM | FSM | DHR-2_Lobe_A/B/C; DOCK-C2; DOCK_C-D_N; PH (6) | — (0) | YES | UNSUPPORTED | CDC42 | 937 | 78.94 | 3.250 | C |
| FRMD4A | Excitatory | FSM | FSM | CUPID; FERM_C; FERM_M (3) | — (0) | YES | UNSUPPORTED | CYTH1 | 919 | 61.72 | 1.834 | C |
| GOLGB1 | Astrocyte | FSM | FSM | ATG16; Crescentin; HALZ; Metal_resist; Tup_N (5) | — (0) | YES | UNSUPPORTED | ACBD3 | 997 | NA | 0.428 | C |
| IFI16 | Oligodendrocyte | FSM | FSM | HIN; tRNA_anti-codon (2) | — (0) | YES | UNSUPPORTED | TP53 | 988 | 68.25 | −0.089 | C |
| LRPPRC | Oligodendrocyte | FSM | FSM | MA3; RPN7 (2) | — (0) | YES | UNSUPPORTED | SLIRP | 999 | 77.05 | 2.418 | C |
| MTHFD1 | OPC | FSM | FSM | — (0) | THF_DHG_CYH; THF_DHG_CYH_C (2) | YES | UNSUPPORTED | SHMT1 | 999 | 94.57 | 3.864 | C |
| PML | Excitatory | FSM | FSM | zf-RING_UBOX (1) | — (0) | YES | UNSUPPORTED | SUMO1 | 999 | 70.12 | 0.508 | C |
| ZCCHC17 | Oligodendrocyte | FSM | FSM | S1 (1) | — (0) | YES | UNSUPPORTED | PNN | 760 | 70.80 | 2.201 | C |
| ZNF268 | Microglia | FSM | FSM | DsrE (1) | KRAB (1) | YES | UNSUPPORTED | TRIM28 | 747 | 68.61 | 0.468 | C |
| ZNF623 | Oligodendrocyte | FSM | FSM | — (0) | zf_C2H2_13 (1) | YES | UNSUPPORTED | C19orf25 | 520‡ | 71.12 | NA | C |

**†** ASXL3 pLDDT = 39.37 (< 50): intrinsically disordered region; domain annotation may be unreliable. ISA would not flag this.  
**‡** ZNF623 STRING score = 520 (< 700 threshold): no high-confidence PPI evidence for any hypothesized partner.

**Key: What BISECT adds over IsoformSwitchAnalyzeR**

| Feature | IsoformSwitchAnalyzeR | BISECT |
|---------|----------------------|--------|
| Pfam domain change detection | YES (same Pfam-A database) | YES (M2, same) |
| NMD susceptibility prediction | YES (canonical NMD rules) | YES (M9, symmetric — both CT and AD isoforms screened) |
| Signal peptide / TM annotation | YES (SignalP, TMHMM) | Partial (MTS via M3) |
| DIFFUSE functional score pre-filter | NO | YES (Stage 1: \|Δ\| ≥ 0.5, DTU p ≤ 1×10⁻⁵) |
| Novel isoform SQANTI3 classification | NO | YES (M4) |
| AlphaFold/ESMFold structural confidence | NO | YES (M10) |
| STRING PPI network validation | NO | YES (M11, hypothesis-directed) |
| phyloP 100-vertebrate conservation | NO | YES (M12) |
| Multi-evidence tier classification | NO | YES (Tier A/B/C) |
| LINE-1 / TE element annotation | NO | YES (M5, M8) |
| Hypothesis rejection capability | NO | YES (M11 score = 0 rejects SLIT2–PTPRF) |

**Result**: IsoformSwitchAnalyzeR would identify the same 26 domain-change cases but cannot stratify them. All 26 would receive equal weight as "domain-change isoform switches." BISECT's M11 separates SUPPORTED (13/26) from UNSUPPORTED (13/26) cases by requiring experimental PPI evidence for the hypothesized domain-function relationship. Critically, among the 13 UNSUPPORTED cases, 8 show STRING combined scores ≥ 900 for *some* partner — ISA would incorrectly infer PPI relevance from these high scores, while BISECT correctly identifies that the top partner is not functionally linked to the lost domain (e.g., GOLGB1 loses Golgi-coil domains but top STRING partner ACBD3 interaction is mediated by coil domains retained in both isoforms; PML loses RING_UBOX but SUMO1 interaction is mediated by SIM motifs in retained domains).

---

### Supplementary Note S1 — RGS3–EFNB2 PDZ interaction basis

The interaction between PDZ domain of RGS3 (CT isoform, aa 188–260) and C-terminal −YYKV motif
of EFNB2 was established experimentally by Lu et al. (2001, *Cell* 105:69–79; PMID 11301003) by
yeast two-hybrid screening, confirmed by co-immunoprecipitation from neural tissue. The −YYKV motif
necessity in vivo established by ephrinB2ΔV/ΔV knock-in mice (Makinen et al. 2005, *Genes Dev*;
PMID 15687262). The RGS3 CT→AD switch removes all PDZ sequence, mechanistically abolishing EFNB2
docking. Direct confirmation in AD astrocytes has not been published as of this writing.
**Key citations**: PMID 11301003, PMID 15687262, PMID 18541704.

### Supplementary Note S2 — Dystrophin-associated protein complex remodelling in AD inhibitory neurons

Two independent inhibitory neuron isoform switches converge on the dystrophin-associated protein
complex (DAPC), which anchors GABA-A receptors and NOS1 at inhibitory postsynaptic densities.
Full-length dystrophin (Dp427) co-localises with α1/β2/γ2 GABA-A receptor subunits; dystrophin
loss causes selective inhibitory synapse density reduction (Bhatt et al. 2020; Grabert & Bhatt 2014).

*DMD* (phyloP AD = 4.823): STRING returned complete DAPC membership at maximum experimental
confidence: DAG1 = 999 (exp = 783), SNTA1 = 998 (exp = 852), SNTB1 = 996 (exp = 834),
SNTG1 = 992, UTRN = 992, SGCA = 995, DTNB = 956. AD isoform shows unusually stronger conservation
(4.823) than CT exon (3.350), consistent with a gain-of-distinct-function rather than simple loss.

*SNTG1* (phyloP AD = 4.558): AD isoform loses PH2 domain required for DAPC membrane association.
STRING: DMD = 992 (exp = 648), SNTA1 = 938, SNTB1 = 874, UTRN = 798, DAG1 = 839. DMD and
SNTG1 cross-confirm as each other's top STRING partners, and both AD-specific exon sets are among
the three most highly conserved in the 26-case panel.

**Key citations**: Bhatt et al. 2020 (*J Physiol*); Grabert & Bhatt 2014 (*Front Synaptic Neurosci*);
Waite et al. 2012 (*Brain*).

### Supplementary Note S3 — PTPRS: parallel LAR-RPTP Liprin-α disruption in astrocytes

*PTPRS* (UniProt Q13332) is a LAR-RPTP family member alongside PTPRF and PTPRD. In astrocytes,
PTPRS regulates perisynaptic astrocyte process extension; shedding of the PTPRS ectodomain by ADAM10
modulates synaptic maturation through heparan sulphate proteoglycan interactions.

The BISECT analysis identified a PTPRS isoform switch in AD astrocytes in which CT carries four
highly conserved exons (mean phyloP = 3.919) absent from the AD isoform (pure CT-exon-loss switch).
STRING returned PPFIA1 = 985 (exp = 509), PPFIA3 = 894, PPFIA2 = 802, LRRC4B = 989, and NTRK3 = 999
(exp = 722) — the TrkC neurotrophin receptor forming a tripartite complex with PTPRS and Liprin-α
(Li et al. 2023, *Science*). Reclassified SUPPORTED after hypothesis expansion.

Convergence of PTPRF (inhibitory neurons) and PTPRS (astrocytes) on the Liprin-α network —
high CT-exon conservation in both (PTPRF = 4.341; PTPRS = 3.919) — suggests cell-type-distributed
LAR-RPTP disruption by alternative splicing as a mechanism for impaired synaptic scaffold
organisation in AD.

**Key citations**: Coles et al. 2011 (*Neuron*); Li et al. 2023 (*Science*); Um et al. 2014 (*Neuron*).

### Supplementary Note S4 — BSG: accelerated evolution at an oligodendrocyte metabolic support switch

*BSG* (Basigin/CD147, UniProt P35613) is the essential chaperone for monocarboxylate transporter
assembly. MCT1 (SLC16A1) and MCT4 (SLC16A3) require BSG for membrane stabilisation; in
oligodendrocytes, BSG-MCT1 at the periaxonal membrane is essential for lactate export to support
axonal mitochondrial metabolism (Lee et al. 2012, *Cell*).

BISECT identified a BSG switch in AD oligodendrocytes with AD isoform losing Ig_5 and V-set
domains. STRING: SLC16A1 = 999 (exp = 976), PPIA = 999, SLC16A3 = 999, CD44 = 985. Uniquely
in the 26-case panel, both AD-specific (phyloP = −0.473) and CT-specific (phyloP = −0.051) exons
show accelerated evolution — indicating broad relaxed purifying selection or positive selection
at the BSG locus in the primate lineage. Because MCT1 is specifically required for long-axon
metabolic support, the BSG switch in AD oligodendrocytes may contribute to progressive disconnection
of long-range cortical networks independent of amyloid or tau pathology.

**Key citations**: Lee et al. 2012 (*Cell*); Funfschilling et al. 2012 (*Nature*);
Halestrap 2013 (*Pflugers Arch*).

---

*End of merged draft — 2026-05-27 (v1.1: Methods 2.13 expanded, Supplementary Table S4 added)*
