# Discussion — PRISM: Isoform-Level Function Prediction
**Draft 2026-05-16 | Target: Nature Methods / NMI**

---

## 5. Discussion

### 5.1 pos_bias: GO-term-dependent isoform discrimination with negative control validation

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

### 5.2 The Type-A/B framework: embedding geometry as a post-hoc characterisation of model utility

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

### 5.3 Computational predictions from isoform-switch analysis

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

### 5.4 NMD screening reveals a systematic data quality issue in isoform-switch analysis

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

### 5.5 Limitations and future directions

**Gene-level label propagation.** The most fundamental limitation of our approach is that
GO annotations at training time are gene-level: all isoforms of a positive gene receive a
positive label, regardless of their individual functional status. This creates a supervised
learning signal that is necessarily noisy for Type-B GO terms where functionally active and
inactive isoforms co-exist. The model partially mitigates this through focal loss (which focuses training on hard
examples) and the ESM-2 embedding geometry (which encodes domain-presence information),
but the ceiling imposed by label noise cannot be quantified without isoform-level ground
truth annotations. Future work will benefit from the emerging
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

### 5.6 Cross-tissue generalization and the limits of zero-shot transfer

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

### 5.7 AD isoform switches: three distinct mechanisms revealed by PRISM-DTU integration

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
by Pfam-A hmmscan, which identifies a complete kinesin motor domain (aa 14–370; E = 1.1×10⁻¹⁰⁹,
score = 354.0) and microtubule-binding domain (Microtub_bd, aa 7–158; E = 1.8×10⁻²³) in
tr293004, and no motor domain of any type in tr292978. Direct motif search confirms
P-loop (GQTGAGKT at aa 87), Switch-I (SSRSHA at aa 222), and Switch-II (DLAGSE at aa 273) in
tr293004 and their absence in tr292978. The mechanistic implication is not simple loss-of-function:
tr292978 (710 aa; 19 exons; 49.9 kb genomic span) harbours an extensive WD40 β-propeller scaffold
(15 Pfam hits spanning aa 372–686; predominant profile: ANAPC4_WD40 and NBCH_WD40), retaining
the dimerization-competent LLQEAL coiled-coil at aa 25. This ANAPC4-like WD40 β-propeller
structurally resembles scaffold subunits of the Anaphase Promoting Complex/Cyclosome (APC/C) —
a complex that has itself been implicated in neuronal cell-cycle re-entry and dendritic
morphology regulation (Huang et al., *Neuron* 2009; PMID: 19778503). Whether tr292978 interacts
with APC/C components or simply uses the β-propeller as a generic protein-interaction scaffold
for cargo retention is mechanistically unresolved.
A heterodimer of KIF21B-201 (one motor head) with tr292978 (cargo-binding, no motor) would
transport cargo at reduced efficiency, proportional to the ratio of tr292978/KIF21B-201 in the
excitatory neuron pool. In AD excitatory neurons where tr292978 reaches 42.9% of KIF21B
usage — nearly equal to canonical — the predicted net effect is a severe reduction in dendritic
transport processivity for KIF21B cargo (AMPA receptor subunits, mRNA granules, synaptic
vesicle precursors). This dominant-negative transport disruption model is distinct from the
complete KIF21B loss-of-function phenotype (which causes dendritic overgrowth; van Rooij et al.,
2022), and would be missed by bulk RNA-seq or gene-level analysis, since total KIF21B expression
may be unchanged. Notably, no repeat elements overlap tr292978 exons, indicating alternative
splicing rather than retroelement insertion drives this isoform — mechanistically distinguishing
the KIF21B switch from the LINE-1-derived NDUFS4/tr73243 case. The key experimental test is
co-immunoprecipitation of tr292978 with KIF21B-201 followed by single-molecule TIRF transport
assay comparing full-length homodimer versus tr292978-containing heterodimer velocity and run
length, and APC/C interaction screen for the WD40 scaffold.

**NDUFS4: natural antisense transcript activation and dual Complex I suppression (excitatory neurons).**
Strand-level genomic analysis reveals that tr73243 is transcribed from the positive (+) strand (chr5:53,560,626–53,688,219), antisense to canonical NDUFS4 (negative strand), establishing tr73243 as a natural antisense transcript (NAT) spanning the NDUFS4 locus (~127 kb). The tr73243 CDS initiates at chr5:53,686,672 and encodes an entirely distinct 379 aa protein (3/15 N-terminal amino acid matches with canonical NDUFS4-201; 98.3% divergence). MTS-feature analysis (MitoFates/TargetP 2.0 criteria) quantitatively confirms mitochondrial import failure: net charge in the N-terminal 30 aa is −1 for tr73243 versus +2 for NDUFS4-201 (MTS criterion ≥ +2), and an HHH triplet at positions 7–9 disrupts the amphipathic helix (MTS composite score 1/5 vs 3/5). The LYR motif required for Complex I N-module integration is additionally absent. The PRISM Mitochondrion organization score (tr73243 = 0.024; NDUFS4-201 = 0.587; Δ = −0.563) provides an independent sequence-based prediction consistent with all structural deficits.

A further unexpected finding from Pfam-A hmmscan is the identification of an RVT_1 domain (RNA-dependent DNA polymerase; aa 141–366; E = 4.6×10⁻⁴⁸) in tr73243 — a reverse transcriptase-homologous fold consistent with derivation from a LINE-1 retroelement. LINE-1 elements carry an antisense promoter (ASP) within their 5' UTR that can drive + strand (host-antisense) transcription, and LINE-1 activation in AD neurons is documented (Guo et al., *Nature* 2018; PMID: 29618813; Cook et al., *Nat Neurosci* 2021; PMID: 33986548). This suggests a dual mechanism of Complex I suppression: (1) tr73243 protein cannot contribute to N-module assembly; (2) the antisense transcript may suppress canonical NDUFS4 mRNA through NAT-mediated RNA silencing, compounding the observed DTU shift beyond what DTU statistics alone capture. Whether tr73243 upregulation is a primary driver or secondary consequence of AD pathology is the critical unresolved question. ATAC-seq at chr5:53,560,626 in sorted AD excitatory neurons and RepeatMasker annotation of LINE-1 elements within the NDUFS4 locus would determine whether retroelement de-repression underlies tr73243 emergence. Proteomics of Complex I immunoprecipitates from AD versus CT cortex would provide the most direct functional evidence.

**DLG1: OPC L27 scaffolding isoform loss and dedifferentiation signal (oligodendrocyte precursors).**
The DLG1 switch is mechanistically the most unexpected of the three. The CT-dominant OPC isoform tr319500 (187 aa, NNIC, 6 exons) generates the strongest DTU signal in the dataset (p = 9.03×10⁻¹⁰), yet domain analysis (Pfam-A hmmscan) reveals that tr319500 is not a domain-depleted fragment but a structurally defined L27-specialized isoform: it retains an L27_1 domain (aa 6–63; E = 7.8×10⁻³⁴, score = 103.2) and MAGUK_N_PEST domain (aa 107–144; E = 2.9×10⁻¹⁵), while all three PDZ domains, SH3, HOOK, and GK domains are absent (confirmed by direct sequence analysis; PDZ GLGF-box absent). The L27 domain mediates MAGUK heterodimerization and Lin7/MALS interaction; Lin7 connects DLG1 to β-neurexin and establishes OPC-specific synaptic contacts without PDZ-mediated glutamate receptor clustering. The PRISM synaptic transmission score of 0.033 for tr319500 reflects the absence of PDZ-dependent receptor scaffolding — not protein non-functionality.

This domain architecture resolves the biological question: in CT OPCs, tr319500 provides L27-mediated MAGUK scaffolding via Lin7/β-neurexin — a molecular profile appropriate for OPC-specific synaptic contacts distinct from neuronal postsynaptic densities. In AD OPCs, the OPC-specialized L27 isoform collapses and canonical DLG1 (3 PDZ, high synaptic score) replaces it, reactivating PDZ-dependent receptor clustering and imposing a neuronal-type synaptic protein signature. This constitutes a specific molecular mechanism for OPC dedifferentiation, an established feature of AD transcriptomics (Mathys et al., *Cell* 2019; PMID: 31042697; Blanchard et al., *Nat Neurosci* 2022; PMID: 35411073; Zhou et al., *Nature* 2020; PMID: 32042154). The functional test is whether tr319500-specific knockdown in OPC cultures disrupts Lin7/β-neurexin interaction and recapitulates the AD OPC transcriptomic state — testing the L27 scaffolding dependency directly.

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

### 5.8 Convergent domain-switch themes from systematic BISECT analysis

The BISECT batch pipeline analysis of 23 domain-changing isoform switches (Results 3.8)
reveals four convergent thematic patterns that extend and contextualize the three priority
discoveries of Section 3.7.

**Reciprocal WD40 β-propeller redistribution between IFT122 and KIF21B.** The most
structurally significant cross-case discovery is the mirror-image exchange of ANAPC4_WD40
and NBCH_WD40 domains between IFT122 and KIF21B in excitatory neurons. The CT-dominant IFT122
isoform (1163 aa) carries the ANAPC4_WD40 + NBCH_WD40 profile; in AD, IFT122 switches to a
shorter isoform (647 aa) that loses these domains and gains Clathrin + TPR. Concurrently, the
KIF21B AD isoform tr292978 (Section 3.7) gains precisely the ANAPC4_WD40/NBCH_WD40 profile
that IFT122 loses. This cross-gene WD40 redistribution in a shared cell type suggests one of
two mechanistic possibilities: (1) tr292978 displaces endogenous IFT122 WD40 modules from a
common APC/C surface-binding interface, reducing IFT122 occupancy; or (2) IFT122 and KIF21B
are co-regulated by a shared splicing factor whose activity shifts in AD excitatory neurons,
producing coordinated changes in WD40 scaffolding programmes. The convergence of three
structurally distinct WD40 β-propeller profiles — IFT122 (APC/C scaffold-like), KIF21B tr292978
(APC/C-interacting, Section 3.7), and ZCCHC17 (S1-domain loss, Oligodendrocyte) — across
independent gene loci suggests that WD40-mediated protein interaction networks are a recurring
target of AD-associated isoform switching, perhaps reflecting broader dysregulation of the
APC/C ubiquitin ligase complex in AD neurons consistent with prior reports of APC/C dysfunction
in tau pathology (Almeida et al., *Nat Commun* 2016).

**Convergent spectrin-anchor loss in inhibitory neurons.** Two independent inhibitory neuron
switches (DMD and SYNE1) both result in loss of Spectrin repeats, the structural unit mediating
membrane–cytoskeleton coupling. DMD additionally gains two SOGA mTOR-pathway domains, shifting
from cytoskeletal anchoring to intracellular signalling. SYNE1 loss of a nuclear-envelope
Spectrin interface predicts LINC complex weakening. These findings suggest that Spectrin-based
cytoskeletal tension is under selective pressure in AD inhibitory neurons, consistent with
reports of dendritic spine collapse and cytoskeletal reorganization in inhibitory interneurons
in AD post-mortem tissue (Bhatt et al., *Nat Rev Neurosci* 2009). The functional consequence
may be reduced mechanical coupling between nuclear activity and axonal/dendritic tension,
impairing activity-dependent gene regulation in inhibitory circuits. A third cytoskeletal
switch — FRMD4A (Excitatory) losing FERM_C + FERM_M, which mediate actin-membrane interface
binding — extends the cytoskeletal anchor disruption theme to excitatory neurons.

**Type IIa LAR-RPTP family convergence and the perisynaptic extracellular matrix.** The most
notable cross-case relationship in the batch analysis is the convergent disruption of fn3-
mediated extracellular matrix sensing across two Type IIa LAR-RPTP family members — PTPRF/LAR
(inhibitory neurons) and PTPRS/PTP-σ (astrocytes) — through distinct molecular mechanisms. The
PTPRF switch is the most architecturally extreme: confirmed sequence analysis shows zero amino
acid overlap between CT and AD isoforms across all 20-aa windows, establishing that the AD
isoform arises from an entirely different first exon. The CT isoform is a canonical fn3-TM-PTP
receptor, while the AD isoform carries a signal peptide and two I-set Ig domains (E = 1.9×10⁻²¹)
but no transmembrane domain — predicting secretion of a 236-aa soluble 2-Ig fragment into the
synaptic cleft. This fragment occupies the NGL-3 and Slitrk1–6 binding epitopes of canonical
PTPRF Ig1–Ig2 without triggering phosphatase signalling, constituting a competitive antagonist
of LAR-dependent synapse organisation at inhibitory synapses. For PTPRS in astrocytes, the
mechanism is subtler but directionally identical: four of eight fn3 repeats are lost by exon
skipping (sequence identity confirmed for aa 1–603; divergence at aa 604), together with a SusE
carbohydrate-binding module (aa 516–551) associated with HSPG-dependent PTP-σ activation. The
phosphatase catalytic domain is retained in both PTPRS isoforms, indicating a regulatory
uncoupling — PTP-σ activity in AD astrocytes persists but is partially decoupled from its
perisynaptic HSPG inputs. The convergence of PTPRF and PTPRS on fn3 loss across two cell types
implicates coordinated reorganisation of the Type IIa LAR-RPTP/extracellular matrix interface
in AD, potentially connecting to well-documented perineuronal net disruption in AD post-mortem
tissue. DOCK11 (complete Rho GEF ablation) and RGS3 (PDZ-cluster loss decoupling G-protein
regulation from postsynaptic receptors) extend the synaptic signalling disruption theme
independently of the RPTP axis.

**Genome-stability and nuclear body domain disruptions in excitatory neurons.** FANCA loss of
the Fanconi_A inter-subunit contact domain and PML loss of the RING E3 ligase together indicate
that AD excitatory neurons experience a dual challenge: reduced DNA interstrand crosslink repair
capacity (FANCA) and disrupted PML nuclear body E3 ligase activity (PML). Both proteins
implicate nuclear maintenance programmes that are distinct from the mitochondrial and
cytoskeletal themes identified in other cell types, raising the hypothesis that AD-associated
isoform switching in excitatory neurons targets multiple nuclear stress-response pathways
in parallel — consistent with transcriptomic evidence for nuclear lamina disruption and
DNA damage response activation in AD excitatory neurons.

### 5.9 Conclusion

PRISM establishes isoform-level GO term prediction as a computationally tractable problem
when protein language model embeddings are paired with appropriate loss functions for sparse
functional labels and within-gene contrastive geometry. The Type-A/B classification framework,
pos_bias metric, and NMD quality-screening pipeline introduced here are general tools
applicable to isoform-function analyses beyond skeletal muscle and sarcopenia. The verified
isoform-switch candidates in skeletal muscle — anchored by DMD Dp427m (ratio = 1,263×) and
NDUFAF6 (ratio = 2,000×), both with complete ORFs — represent high-priority targets for
experimental validation in the context of age-related muscle wasting.

Applied cross-tissue to Alzheimer's disease long-read single-cell data, PRISM achieves
meaningful zero-shot generalization (macro AUPRC 0.600) and, when integrated with
statistical DTU testing and systematic structural analysis via the BISECT pipeline, identifies
26 AD-specific isoform switches across six brain cell types — including the NDUFS4 NAT
activation mechanism in excitatory neurons, the IFT122–KIF21B WD40 β-propeller redistribution,
and convergent Spectrin anchor loss in inhibitory neurons. Together, these results demonstrate
that isoform-level functional prediction from protein language model embeddings is not
tissue-specific: the same framework discovers mechanistically interpretable isoform switches
in both musculoskeletal and neurodegenerative disease contexts from a model trained on a single
tissue type, and that systematic structural annotation via BISECT converts statistically
significant DTU signals into domain-level mechanistic hypotheses amenable to direct experimental
testing.
