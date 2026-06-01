# Introduction — PRISM: Isoform-Level Function Prediction
**Draft 2026-05-16 | Target: Nature Methods / NMI**

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
PMID: 36927031) using a deep multi-layer perceptron trained with focal loss (BinaryFocalCrossentropy, γ = 2.0)
across an extended panel of 18 GO BP terms. We first validate PRISM in a comprehensively
labelled skeletal muscle long-read single-cell transcriptome, then apply it zero-shot to a
Samsung Alzheimer's disease long-read single-cell RNA sequencing dataset to test cross-tissue
generalization, and integrate PRISM predictions with Dirichlet-multinomial differential
transcript usage testing to discover AD-specific isoform switches at single-cell resolution.

We demonstrate that PRISM achieves macro AUPRC of 0.7022 across 18 GO BP terms (0.6935 across
13 sarcopenia-relevant terms), compared with 0.363 for standard logistic regression
(+91%; 10/11 Type-B terms q < 0.05, Benjamini-Hochberg correction).
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
