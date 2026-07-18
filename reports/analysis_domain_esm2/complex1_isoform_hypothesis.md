# Isoform-Switch-Driven Disruption of Complex I Biogenesis as a Mechanistic Basis for Mitochondrial Subunit Depletion in Alzheimer's Disease

**Seungwon Lee**¹, [Co-authors TBD]

¹ GIST (Gwangju Institute of Science and Technology)

*Correspondence*: seungwon.david.lee@gmail.com

---

## Abstract

Reduced Complex I activity and subunit abundance in Alzheimer's disease (AD) brain are well-documented at the proteomic level, yet the upstream mechanisms remain incompletely resolved. We propose that AD-specific alternative splicing — driven by documented U1 snRNP dysfunction — introduces assembly-incompetent isoforms at multiple sequential stages of Complex I biogenesis, generating stalled intermediates whose constitutive degradation explains the observed protein-level reduction of specific subunits. Using single-cell long-read RNA-sequencing data from 63,994 AD prefrontal cortex isoforms and BISECT multi-evidence characterisation, we identify four cell-type-specific isoform switches affecting NDUFAF5, NDUFS7, NDUFS8, and NDUFS4 that collectively disrupt Q-module formation (stage 1), intermediate assembly (stage 2), and terminal holoenzymatic maturation (stage 3) — each switch independently sufficient to stall the assembly pathway. Crucially, the affected genes are not randomly distributed but occupy sequential, non-redundant positions in the Complex I assembly cascade. This convergent disruption is consistent with a mechanistic explanation for the cell-type-selective Complex I deficiency observed in AD excitatory and inhibitory neurons. We frame this as a testable mechanistic hypothesis, not a causal proof, and outline experiments to discriminate isoform-driven assembly failure from alternative explanations including oxidative damage and transcriptional suppression.

*Keywords*: Alzheimer's disease, Complex I, alternative splicing, isoform switch, mitochondrial assembly, NDUFAF5, NDUFS4, single-cell long-read sequencing

---

## 1. Introduction

Mitochondrial dysfunction is a central feature of Alzheimer's disease (AD) pathophysiology. Among the five OXPHOS complexes, Complex I (NADH:ubiquinone oxidoreductase; CI) shows the most consistent and quantitatively largest reduction in AD brain [1,2]. Proteomics of postmortem AD cortex reveals significantly decreased abundance of CI subunits from both the peripheral arm (N-module, Q-module) and membrane arm (P-module), with effect sizes that exceed those of Complexes II–V [1]. CI activity measured enzymatically in AD mitochondria is reduced by 25–45% depending on brain region and disease stage [2]. These observations establish CI depletion as a reproducible and quantitatively substantial feature of the AD molecular phenotype.

The dominant explanatory framework attributes CI depletion to: (i) mitochondrial DNA (mtDNA) mutations and deletions accumulating with age and exacerbated in AD, (ii) oxidative damage to CI subunit proteins by reactive oxygen species (ROS), and (iii) transcriptional suppression of nuclear-encoded CI genes secondary to upstream pathology [3]. While each mechanism has experimental support, they share a common weakness: they predict global or stochastic CI subunit reduction without explaining why specific subunits and specific cell types are preferentially affected. Proteomics data reveal that CI reduction in AD is not uniform across subunits — N-module and Q-module peripheral arm components show disproportionate loss — and single-cell transcriptomics shows that CI pathway dysregulation is concentrated in excitatory neurons, with a partially distinct pattern in inhibitory neurons [4,5].

Here we propose an additional, mechanistically specific explanation: AD-associated alternative splicing events introduce isoforms at multiple sequential steps of the CI assembly cascade that are assembly-incompetent, generating stalled intermediates that are proteolytically degraded, thereby reducing steady-state abundance of specific subunits. This hypothesis is grounded in: (1) documented U1 snRNP dysfunction in AD producing cell-type-specific aberrant splicing [6,7], (2) single-cell long-read RNA-seq evidence identifying isoform switches in CI assembly factor NDUFAF5 and subunits NDUFS7, NDUFS8, and NDUFS4 in AD prefrontal cortex, and (3) established biochemistry of the CI assembly cascade that maps each switch to a distinct, non-redundant assembly stage.

We emphasise that this is a **mechanistic hypothesis consistent with observational data**, not a causal proof. The available data are cross-sectional, and interventional experiments are required to definitively establish causal direction. Nonetheless, we argue that the convergence of (i) the splicing observations, (ii) the structural positions of affected genes, and (iii) the cell-type specificity of both the splicing changes and the downstream CI depletion constitutes sufficiently coherent evidence to warrant formalisation as a testable hypothesis and to motivate targeted experimental follow-up.

---

## 2. Background: Complex I Subunit Reduction in AD Brain

Two landmark proteomic studies directly motivate the hypothesis developed here.

Adav et al. (2019) performed quantitative proteomics of synaptoneurosomes from AD and control prefrontal cortex and identified consistent reduction in CI subunits of the peripheral arm — particularly NDUFS2, NDUFS3, NDUFS4, NDUFS7, and NDUFS8 — alongside relative preservation of membrane arm (ND1–ND6) subunits [1]. The specificity of peripheral arm depletion is notable: it is most parsimoniously explained by an assembly defect rather than global mitochondrial damage, because the membrane arm assembles first and is less exposed to soluble proteases.

Bai et al. (2021) integrated transcriptomics, proteomics, and network analysis of AD multi-omics data and identified a mitochondria-splicing network dysregulation axis in which CI-related genes are co-regulated with splicing factors — including components of the U1 snRNP — and that this co-dysregulation is more pronounced in AD than in other neurodegenerative diseases [2]. The co-occurrence of splicing factor dysregulation with CI subunit reduction in the same analytical module is a key observation: it suggests that the splicing machinery disruption is mechanistically upstream of, not merely coincident with, CI depletion.

---

## 3. U1 snRNP Dysfunction as an Upstream Driver of AD-Specific Splicing Pathology

The mechanistic link between AD pathology and aberrant alternative splicing is established through U1 snRNP dysfunction. Chen et al. (2022) demonstrated that tau pathology impairs U1 snRNP function through direct interaction, producing widespread splicing changes including cryptic exon inclusion and intron retention that are detectable at early Braak stages [6]. Farhadieh et al. (2023) extended this to cell-type resolution, showing that aberrant splicing events in AD are concentrated in excitatory neurons and that the splicing changes are both cell-autonomous (not attributable to altered cell-type composition) and enriched for genes with mitochondrial functions [7].

These findings establish the upstream context for the hypothesis: in AD excitatory and inhibitory neurons, U1-dependent splicing fidelity is compromised beginning at pre-symptomatic stages, and the affected transcripts are disproportionately enriched for mitochondrial assembly and function genes. Within this landscape, CI assembly transcripts are among the most structurally vulnerable to U1 dysfunction because many contain long introns with weak U1 binding sites that are particularly sensitive to reduced U1 activity [6].

---

## 4. Cell-Type-Specific Isoform Switches in Complex I Genes: Observational Evidence

Using BISECT (Biological Isoform-Switch Evidence Characterization Tool) applied to 63,994 isoforms from an AD prefrontal cortex single-nucleus long-read RNA-seq cohort (13 AD, 8 control donors; DRIMSeq donor-level DTU testing; see companion paper [REF]), we identified four isoform switches in CI-related genes that satisfy the following criteria: (a) statistically supported by DRIMSeq FDR or permutation testing, (b) concordant across independent sequencing batches, (c) cell-type specific, and (d) carrying predicted functional consequences from BISECT structural modules.

### 4.1 NDUFAF5 — NMD-Sensitive Isoform Switch in Excitatory Neurons

NDUFAF5 (C20orf7) is an early-stage CI assembly factor that hydroxylates NDUFS7 at Arg73, a post-translational modification required for the stable formation of the ~200-kDa Q-module intermediate [8]. Loss of NDUFAF5 hydroxylase activity prevents the 200-kDa intermediate from forming, permanently stalling assembly at the NDUFS2/NDUFS3 peripheral arm nucleus before Q-module consolidation.

In AD prefrontal cortex excitatory neurons, we observe a genome-wide significant (DRIMSeq FDR < 0.05) switch from the canonical 345-aa CT-dominant NDUFAF5 isoform to a 267-aa AD-dominant isoform carrying a premature termination codon (PTC) located >2,675 nucleotides upstream of the final exon-junction complex — a configuration meeting established NMD sensitivity criteria. The AD isoform truncation eliminates the C-terminal methyltransferase-like domain required for hydroxylase activity. The functional consequence of this switch is predicted to be complete abolition of NDUFS7 Arg73 hydroxylation in excitatory neurons carrying the AD isoform.

### 4.2 NDUFS7 — Cassette Exon Switch at the FeS-Cluster Interface in Excitatory Neurons

NDUFS7 (PSST) is the direct enzymatic substrate of NDUFAF5 and a core Q-module subunit carrying an iron-sulfur (FeS) cluster essential for electron transfer [8]. Following successful Arg73 hydroxylation, NDUFS7 is incorporated into the ~200-kDa intermediate; without hydroxylation, NDUFS7 is not stably incorporated.

We observe an additional, independent isoform switch in NDUFS7 in excitatory neurons (permutation p = 0.048; batch-concordant across PO and SMC batches). BISECT M1 identifies a cassette exon cluster at the FeS-cluster binding interface as the switch event — independently of the NDUFAF5 hydroxylation requirement. This constitutes a **directly verified enzyme–substrate pair**: NDUFAF5 (assembly factor, hydroxylase) and NDUFS7 (substrate, target subunit) both undergo AD-specific isoform switches in the same cell type (excitatory neurons), generating a double disruption of the same assembly step from two independent molecular mechanisms.

### 4.3 NDUFS8 — Mitochondrial Targeting Sequence Disruption in Inhibitory Neurons

NDUFS8 (TYKY) is a core Q-module subunit carrying two [4Fe-4S] clusters and is incorporated into the ~86-kDa intermediate via a mechanism mediated by assembly factor NDUFAF6 [9]. Incorporation of NDUFS8 represents the transition from the early Q-module nucleus to the expanding peripheral arm assembly.

In AD inhibitory neurons, we observe a statistically supported isoform switch (permutation p_adj = 0.022; 37% effect size; batch-concordant) from the canonical NDUFS8 isoform to a novel NIC isoform (IsoQuant-assembled; absent from GENCODE38). BISECT M9 identifies a disrupted mitochondrial targeting sequence (MTS) in the AD isoform: the canonical N-terminal TLLWTELFR amphipathic helix is replaced by a proline-disrupted PVLPTG segment that impairs MTS helix formation. BISECT M1 confirms by HMMER scan that both isoforms retain complete Fer4-family FeS cluster coordinating domains (E < 0.001; 7 coordinating cysteines shared). The predicted functional consequence is a localization switch — FeS cluster-competent NDUFS8 protein stranded in the cytoplasm — mechanistically distinct from domain loss but converging on depletion of TYKY subunit from the mitochondrial matrix.

### 4.4 NDUFS4 — Import Failure and Terminal Maturation Block in Both Neuron Types

NDUFS4 is an N-module accessory subunit whose incorporation at a late assembly stage serves a dual function: (i) mitochondrial import via its N-terminal MTS, and (ii) triggering NDUFA12-mediated displacement of assembly chaperone NDUFAF2 from the NDUFS4-free enzyme, thereby licensing terminal holoenzymatic maturation [10]. Without NDUFS4, NDUFA12 cannot complete this substitution and the complex is arrested as the NDUFS4-free enzyme intermediate.

In AD excitatory and inhibitory neurons, we observe a nominally significant isoform switch (excitatory: permutation p = 0.041; inhibitory: p = 0.024; uncorrected; batch-concordant in both batches) from canonical NDUFS4-201 (175 aa; MTS-bearing) to a novel NIC isoform (tr73243; 379 aa) in which the MTS-encoding exon 1 is absent (BISECT M1 PASS; M9 TargetP-2.0 MTS score drop confirmed). The absence of the MTS predicts cytoplasmic sequestration of otherwise folded NDUFS4 protein. Additionally, even for any NDUFS4 that reaches the mitochondrial matrix, the isoform-level exon 1 loss disrupts the protein's N-terminal structural context required for proper positioning within the NDUFS4 binding site, potentially further impairing NDUFA12/NDUFAF2 exchange.

---

## 5. A Three-Stage Assembly Cascade Disrupted by AD-Specific Isoform Switches

The biochemistry of CI biogenesis proceeds through a defined, modular sequence in which intermediates must successfully transit each stage before proceeding [8,9,10]. The four isoform switches described above map to three distinct, non-redundant stages of this cascade:

```
Stage 1: NDUFS2/NDUFS3 nucleus formation
         ↓ [NDUFAF7 methylates NDUFS2-Arg85]
         ↓ [NDUFAF5 hydroxylates NDUFS7-Arg73]  ← NDUFAF5 switch (excitatory)
                                                    NDUFS7 switch (excitatory)
         ~200-kDa Q-module intermediate
         ↓
Stage 2: Peripheral arm expansion
         ↓ [NDUFAF6 mediates NDUFS8 incorporation]  ← NDUFS8 switch (inhibitory)
         ~315-kDa intermediate → ~830-kDa intermediate
         ↓
Stage 3: Terminal N-module maturation
         ↓ [NDUFS4 incorporation]                   ← NDUFS4 switch (both)
         ↓ [NDUFA12 displaces NDUFAF2]
         Mature holoenzyme
```

**Stage 1 disruption (excitatory neurons):** NDUFAF5 switch eliminates hydroxylase activity → NDUFS7 Arg73 unmodified → 200-kDa intermediate cannot form → NDUFS7, NDUFS8, and all downstream Q-module subunits are never stably incorporated → constitutively degraded in excitatory neurons. The independent NDUFS7 isoform switch constitutes an additional, cell-autonomous Stage 1 disruption even if NDUFAF5 hydroxylase activity were restored.

**Stage 2 disruption (inhibitory neurons):** NDUFS8 MTS switch mislocalises TYKY protein → even if Stage 1 proceeds normally, the ~86-kDa → ~315-kDa transition is blocked in inhibitory neurons → 86-kDa intermediate stalls → NDUFS8 and downstream subunits not delivered to the growing complex.

**Stage 3 disruption (both neuron types):** NDUFS4 MTS switch prevents N-module import and secondarily blocks NDUFA12/NDUFAF2 exchange → even if Stages 1–2 complete normally, the NDUFS4-free enzyme cannot mature → fully assembled 830-kDa intermediate accumulates as a dead-end intermediate that is ultimately degraded.

This cascade architecture predicts a quantitatively graded depletion pattern: subunits incorporated early (NDUFS2, NDUFS3) may show smaller reductions than subunits incorporated at or after the blocked stage, because early-stage intermediates are disrupted but early-incorporating subunits can still form partial assemblies. Subunits critically dependent on intact later-stage assembly (NDUFS4, NDUFA12) should show the strongest protein-level reductions in cells carrying the Stage 3 isoform switch — a prediction testable against the Adav 2019 proteomics data by stratifying reported subunit reductions by their assembly stage position.

### 5.1 Connecting Assembly Failure to Observed Subunit Depletion

Unassembled CI subunits do not accumulate in mammalian cells. Stalled assembly intermediates are substrates for the mitochondrial protease system (LONP1, ClpXP, AAA-proteases), which degrades incompletely assembled complexes to recycle cofactors and prevent reactive intermediate accumulation [11]. The assembly-incompetent isoforms described here would generate: (i) cytoplasmic NDUFS8 and NDUFS4 protein degraded by cytoplasmic proteasome, and (ii) stalled mitochondrial intermediates (200-kDa, 86-kDa, NDUFS4-free enzyme) degraded by mitochondrial matrix proteases.

The net result — measurable by quantitative proteomics — is reduced steady-state abundance of the specific subunits whose assembly is blocked, without necessarily reducing mRNA levels. This prediction is consistent with the observation in Adav 2019 that CI subunit reductions in AD are more pronounced at the protein level than at the mRNA level for the peripheral arm subunits, whereas the membrane arm subunits (ND1–ND6, assembled first and most stably) show smaller or non-significant reductions [1]. The cell-type specificity of the isoform switches further predicts that bulk-tissue proteomics should show a signal that is diluted by the contribution of non-affected cell types (astrocytes, microglia, oligodendrocytes) in which these switches are absent or less pronounced.

---

## 6. Cell-Type Selectivity as a Mechanistic Constraint

A central challenge for any hypothesis about CI depletion in AD is explaining the cell-type specificity of the phenotype. Global mechanisms (oxidative damage, mtDNA mutations) predict pan-cellular CI depletion. The isoform-switch hypothesis predicts depletion specifically in cells where the isoform switches are active — a testable, stronger prediction.

Our observations show:
- Stage 1 switches (NDUFAF5, NDUFS7): excitatory neurons
- Stage 2 switch (NDUFS8): inhibitory neurons
- Stage 3 switch (NDUFS4): both neuron types

This pattern is consistent with the observation in single-cell AD transcriptomics that excitatory neurons show the largest CI-pathway dysregulation, followed by inhibitory neurons, with non-neuronal cell types showing smaller effects [5]. The two-stage specificity — excitatory neurons with Q-module biogenesis failure, inhibitory neurons with intermediate assembly failure — also explains why CI depletion has a particularly severe impact on excitatory neuronal energetics: excitatory neurons have the highest OXPHOS demand and the least metabolic reserve to compensate for partial CI loss.

Moreover, the cell-type specificity of the splicing changes themselves argues against a purely transcriptional suppression model: if CI subunit genes were simply transcriptionally downregulated, we would expect reduced mRNA and protein levels for all CI subunits across all cell types, rather than the cell-type-specific isoform ratio changes we observe in long-read single-cell data.

---

## 7. The Hypothesis Formalised

We propose the following mechanistic model, framed as a causal hypothesis compatible with the available observational evidence:

> **AD-associated U1 snRNP dysfunction generates cell-type-specific alternative splicing events in CI assembly factor NDUFAF5 (excitatory neurons) and core subunits NDUFS7 (excitatory), NDUFS8 (inhibitory), and NDUFS4 (both), introducing assembly-incompetent isoforms at three sequential, non-redundant stages of the CI biogenesis cascade. Stalled assembly intermediates are constitutively degraded by the mitochondrial and cytoplasmic proteasome systems, reducing steady-state abundance of CI peripheral arm subunits in a cell-type-selective and assembly-stage-specific pattern consistent with the quantitative CI subunit depletion documented in AD brain proteomics.**

This hypothesis makes the following empirically distinguishing predictions relative to alternative explanations:

| Prediction | Alternative explanation | Test |
|-----------|------------------------|------|
| Subunit reduction strongest in cells with active isoform switch | Oxidative damage: uniform | snRNA-seq paired with spatial proteomics |
| Protein reduction without proportional mRNA reduction | Transcriptional suppression | RNA:protein ratio per cell type |
| Reduction largest for stage-specific subunits, not all CI | Global mtDNA damage: all subunits | Subunit-stratified proteomics by assembly stage |
| Rescue by splice-switching antisense oligonucleotide | Upstream pathology (amyloid/tau) | ASO targeting NDUFAF5/NDUFS8 switch junction |
| Intermediate accumulation (200-kDa, 86-kDa) in AD neurons | N/A | BN-PAGE of AD neuron mitochondria |

---

## 8. Limitations

The evidence presented here is **observational and cross-sectional**. The following limitations must be acknowledged:

**Causal directionality is unproven.** The isoform switches are identified in the same tissue at the same disease stage as the CI subunit depletion. We cannot exclude that CI depletion (caused by other mechanisms) induces secondary splicing changes as a consequence. Longitudinal data or stage-stratified analysis (Braak I–VI) would help establish temporal ordering.

**Effect size and penetrance.** The isoform switches are nominally significant (NDUFAF5 is genome-wide significant; NDUFS7, NDUFS8, NDUFS4 are permutation-significant but not Bonferroni-corrected) and affect a fraction of transcripts — not all NDUFAF5 transcripts become the NMD-sensitive form. The quantitative contribution of isoform-driven assembly failure to total CI depletion, relative to other mechanisms (oxidative damage, mtDNA mutations), is not established.

**Technical constraints.** Single-nucleus RNA-seq captures nuclear RNA, including unspliced pre-mRNA, inflating intron retention estimates. Novel isoform reconstruction from long reads has reconstruction uncertainty, particularly for low-abundance transcripts. The NDUFS8 NIC isoform is entirely absent from GENCODE38; while we validate it through BISECT module analysis, additional validation by targeted RT-PCR and protein detection would strengthen the claim.

**No intervention data.** The hypothesis is not tested by genetic manipulation, splice-correcting ASOs, or iPSC-differentiated neuron experiments. Without such interventional data, the causal claim remains a mechanistic inference from structural and statistical evidence.

**Confounding by cell composition.** Despite cell-type deconvolution, snRNA-seq cannot fully exclude cell proportion shifts contributing to apparent within-cell-type transcript ratio changes. The multi-batch concordance (PO and SMC batches) and donor-level permutation testing partially address this.

---

## 9. Wet Lab Experimental Validation Design

We propose a six-tier experimental programme organised by causal proximity: Tier 1 confirms the isoform-level molecular events; Tier 2 establishes the biochemical consequences at the assembly level; Tier 3 tests causal necessity through splice correction; Tier 4 addresses temporal ordering; Tier 5 validates the upstream driver; Tier 6 tests therapeutic translatability.

---

### Tier 1 — Confirm isoform switches produce functional protein loss

**Experiment T1-A: Isoform-specific immunodetection of NDUFAF5**

*Rationale*: The AD-enriched NDUFAF5 isoform carries a premature termination codon predicted to trigger NMD. If NMD is active, the truncated protein will be absent; if NMD is incomplete, a short 267-aa truncated product may be detectable. Either outcome confirms loss of full-length hydroxylase activity.

*Protocol*:
1. Raise polyclonal antibodies against two NDUFAF5 epitopes: (a) shared N-terminal peptide (residues 50–100; detects all isoforms), (b) C-terminal peptide (residues 280–345; present only in canonical 345-aa isoform, absent in 267-aa AD isoform)
2. Cell fractionation: excitatory neuron-enriched nuclear fractions from fresh-frozen AD and CT prefrontal cortex (NeuN+/CaMKII+ immunopanning or FACS)
3. Western blot: N-terminal antibody detects total NDUFAF5; C-terminal antibody detects canonical isoform
4. NMD inhibition control: treat iPSC-excitatory neurons with cycloheximide (CHX; 100 µg/mL, 4 h) to block NMD and reveal NMD substrate accumulation
5. Mass spectrometry confirmation: isoform-discriminating tryptic peptide from C-terminal extension of NDUFAF5-201 vs absence in AD isoform

*Expected result*: Reduced C-terminal NDUFAF5 signal in AD excitatory neurons; CHX treatment rescues 267-aa truncated form in AD iPSC neurons confirming NMD degradation.

*Controls*: CT excitatory neurons (no switch); non-neuronal cell types (no predicted switch); CHX vehicle.

---

**Experiment T1-B: NDUFS8 MTS functional validation**

*Rationale*: The AD NIC NDUFS8 isoform carries a proline-disrupted N-terminal sequence predicted to fail mitochondrial import. Import failure produces cytoplasmic NDUFS8 protein that is biochemically competent (FeS domains intact) but mislocalised.

*Protocol*:
1. Subcellular fractionation: mitochondrial (Mito) vs cytoplasmic (Cyto) fraction from AD and CT inhibitory neurons (SST+/PV+ FACS-sorted)
2. Western blot: anti-NDUFS8 (commercial antibody, Abcam ab196161) in Mito vs Cyto fractions; Mito marker (HSP60), Cyto marker (GAPDH)
3. In vitro import assay: synthesise both NDUFS8 isoforms by cell-free translation (TNT system) with [35S]-methionine labelling; incubate with isolated mitochondria; protease (PK) protection assay
4. Immunofluorescence: co-localisation of anti-NDUFS8 with MitoTracker in AD vs CT neurons

*Expected result*: CT NDUFS8 in Mito fraction; AD-enriched NIC NDUFS8 shifted to Cyto fraction; in vitro import of CT isoform is PK-protected (imported), AD isoform is PK-sensitive (cytoplasmic).

---

**Experiment T1-C: Direct NDUFS7 Arg73 hydroxylation by targeted MS**

*Rationale*: The most direct test of the NDUFAF5→NDUFS7 enzyme-substrate relationship is to measure the specific PTM (Arg73 hydroxylation) rather than inferring it from assembly phenotypes.

*Protocol*:
1. Immunoprecipitate NDUFS7 from excitatory neuron-enriched mitochondria (AD and CT)
2. Trypsin digest; enrich for hydroxylated peptides (IMAC or TiO₂)
3. Targeted LC-MS/MS: MRM/PRM for Arg73-containing tryptic peptide (ALXXXXXR; search for +16 Da hydroxylation on R73)
4. Parallel: in vitro reaction using purified recombinant NDUFAF5-WT (345 aa), NDUFAF5-truncated (267 aa), and recombinant NDUFS7; measure Arg73 hydroxylation by MS
5. Quantify: hydroxylation stoichiometry (modified/total Arg73 peptide) in AD vs CT

*Expected result*: Reduced Arg73 hydroxylation in AD excitatory neuron mitochondria; recombinant NDUFAF5-truncated fails to hydroxylate NDUFS7 in vitro.

---

### Tier 2 — Demonstrate assembly intermediate accumulation

**Experiment T2-A: BN-PAGE of AD vs CT cortical mitochondria**

*Rationale*: Stalled assembly intermediates are the key mechanistic intermediate between isoform switches and subunit depletion. Direct visualisation by BN-PAGE is the most definitive assembly-level evidence.

*Protocol*:
1. Isolate mitochondria from fresh-frozen AD (n=6) and CT (n=6) prefrontal cortex (Percoll gradient; < 4°C throughout)
2. Digitonin solubilisation (4 g/g protein)
3. BN-PAGE (3–12% gradient)
4. Western transfer; probe sequentially with: anti-NDUFAS (complex I assembly marker), anti-NDUFS7 (Q-module subunit), anti-NDUFS8 (Q-module TYKY), anti-NDUFAF2 (expected: retained in AD), anti-NDUFA12 (expected: absent in NDUFS4-free enzyme fraction), anti-NDUFS4 (canonical subunit level), anti-CI-OXPHOS antibody cocktail
5. In-gel CI activity staining (NADH tetrazolium reductase assay)
6. Quantify: ratio of intermediate bands (≤315 kDa) to holoenzyme band (~980 kDa)

*Expected result*: AD shows: (a) accumulation of ~200-kDa band (NDUFAF5 hydroxylation failure → early intermediate), (b) ~86-kDa band (NDUFAF6/NDUFS8 assembly block), (c) high-MW band containing NDUFAF2 but lacking NDUFA12 (NDUFS4-free enzyme), (d) reduced in-gel CI activity, (e) quantitatively less complete holoenzyme.

*Controls*: Inhibitor-treated cells (rotenone for CI inhibition; not an assembly block control); NDUFAF5 siRNA knockdown as positive control for ~200-kDa stall.

---

**Experiment T2-B: NDUFA12/NDUFAF2 ratio in CI immunoprecipitate**

*Rationale*: The NDUFS4-free enzyme retains NDUFAF2 and lacks NDUFA12. Quantifying this ratio in immunoprecipitated CI complexes directly tests the terminal maturation failure prediction.

*Protocol*:
1. Immunoprecipitate CI using anti-NDUFS2 antibody (subunit present in all intermediates and holoenzyme) from AD and CT mitochondrial lysates
2. TMT-labelled quantitative proteomics of immunoprecipitate: quantify NDUFA12/NDUFAF2 molar ratio
3. Alternatively: SDS-PAGE + Western with anti-NDUFA12 and anti-NDUFAF2 on the same blot; normalise to NDUFS2 as loading reference
4. Compare: AD excitatory neuron fraction vs CT; AD inhibitory neuron fraction vs CT; non-neuronal cells (expected: no difference if no NDUFS4 switch)

*Expected result*: Significantly reduced NDUFA12/NDUFAF2 ratio in AD excitatory and inhibitory neurons; preserved ratio in astrocytes and microglia.

---

### Tier 3 — Test causal necessity: splice-correcting ASO rescue

**Experiment T3-A: NDUFAF5 splice-switching ASO in AD iPSC-excitatory neurons**

*Rationale*: If the NDUFAF5 isoform switch is causally necessary for CI assembly failure, then correcting the splice site should rescue CI biogenesis. This is the critical causal test.

*Protocol*:
1. Generate or obtain AD patient iPSC-derived excitatory neurons carrying the NDUFAF5 switch (confirm switch frequency ≥ 20% by long-read RT-PCR)
2. Design splice-switching ASO (20-mer phosphorothioate/2'-MOE; Ionis Pharmaceuticals design rules) targeting the aberrant 5' splice site that generates the truncated/NMD-sensitive NDUFAF5 isoform
3. Transfect at 100 nM (gymnotic or lipofection); n=6 biological replicates per condition
4. Readouts at 7 days post-transfection:
   - Primary: Long-read sequencing to confirm NDUFAF5-201 restoration (isoform ratio)
   - Secondary: NDUFS7 Arg73 hydroxylation by targeted MS (T1-C protocol)
   - Secondary: CI enzymatic activity (Seahorse XF Mito Stress Test; NADH:ubiquinone oxidoreductase assay)
   - Secondary: BN-PAGE (T2-A protocol) for intermediate resolution
   - Secondary: Total NDUFS7/NDUFS8/NDUFS4 protein level (Western; test whether rescue of splice corrects protein level)
5. Controls: scramble ASO; WT iPSC neurons; CHX pre-treatment (NMD inhibition alone as partial positive control)

*Primary expected result*: ASO correction of NDUFAF5 splicing → restoration of Arg73 hydroxylation → reduction in ~200-kDa intermediate → increase in CI activity. This would establish causal sufficiency of the splice change.

---

**Experiment T3-B: CRISPR-forced canonical splicing isogenic pair**

*Rationale*: ASO rescue is transient; CRISPR modification of the splice site provides a stable, isogenic comparison.

*Protocol*:
1. Identify the intronic sequence element responsible for the aberrant splice site activation using minigene reporter assay
2. CRISPR base-edit (CBE or ABE) to disrupt the cryptic 5' splice site while preserving adjacent coding sequence
3. Clone into AD patient iPSC; differentiate to excitatory neurons in parallel with unmodified isogenic line
4. Same readout battery as T3-A
5. Additional: RNA-seq to check for off-target splicing effects of CRISPR edit

---

### Tier 4 — Temporal ordering: Braak stage stratification

**Experiment T4: Isoform switch frequency vs Braak stage**

*Rationale*: If isoform switches cause CI depletion, switch frequency should precede or co-occur with CI protein reduction, not appear only in end-stage disease.

*Protocol*:
1. Obtain Braak stage I–VI postmortem prefrontal cortex snRNA-seq data (or generate from banked tissue n≥3/stage)
2. Quantify NDUFAF5 canonical/AD-isoform ratio and CI subunit protein levels (spatial proteomics or TMT proteomics) at each Braak stage
3. Fit mixed-effects model: isoform ratio ~ Braak stage + age + sex + RIN
4. Expected: isoform switch detectable at Braak II–III; CI protein depletion measurable at Braak III–IV (switch precedes depletion by at least one stage)

---

### Tier 5 — Upstream driver: U1 snRNP dysfunction

**Experiment T5: U1-70K cleavage correlates with NDUFAF5 switch**

*Rationale*: If the NDUFAF5 switch is downstream of U1 snRNP dysfunction, N40K fragment abundance should positively correlate with NDUFAF5 AD-isoform frequency across donors.

*Protocol*:
1. In the Samsung cohort (n=21 donors with banked tissue): measure U1-70K full-length vs N40K fragment by Western (anti-U1-70K antibody)
2. Correlate N40K/full-length ratio with NDUFAF5 canonical/AD-isoform ratio (from existing snRNA-seq data) per donor
3. Expected: Spearman ρ > 0.5 (N40K ↑ correlates with NDUFAF5 AD-isoform ↑)
4. Mechanistic test: transfect N40K-expressing vector into CT iPSC-excitatory neurons; measure NDUFAF5 splicing shift at 10, 21, 30 days

---

### Tier 6 — Therapeutic target validation

**Experiment T6: CI activity rescue by combined ASO + NDUFS8 import rescue**

*Rationale*: Multiple assembly stages are disrupted simultaneously. Correcting only one may be insufficient for meaningful CI recovery. Testing combinatorial rescue quantifies the contribution of each switch to total CI dysfunction.

*Protocol*:
1. AD iPSC-excitatory neurons treated with: (a) NDUFAF5-ASO alone, (b) NDUFS8-ASO (restoring MTS-containing isoform) alone, (c) combined NDUFAF5-ASO + NDUFS8-ASO, (d) scramble control
2. CI activity (Seahorse OCR; Complex I-linked respiration: basal, oligomycin, FCCP, Antimycin A/Rotenone)
3. Expected: partial rescue by individual ASOs; additive rescue by combination; combination rescue approaches WT levels if the isoform switches are the dominant CI depletion mechanism

---

### Summary experimental timeline

| Tier | Experiment | Timeline | Key readout | Tests |
|------|-----------|----------|-------------|-------|
| T1-A | NDUFAF5 isoform-specific Western + NMD inhibition | 6 months | C-terminal NDUFAF5 loss in AD | Isoform → protein depletion |
| T1-B | NDUFS8 subcellular localisation | 6 months | Cytoplasmic mislocalization | MTS disruption → import failure |
| T1-C | NDUFS7 Arg73 hydroxylation MS | 9 months | Reduced PTM in AD excitatory | Enzyme-substrate pair |
| T2-A | BN-PAGE intermediates | 12 months | ~200-kDa, ~86-kDa, NDUFS4-free | Stalled assembly intermediates |
| T2-B | NDUFA12/NDUFAF2 ratio | 9 months | Reduced NDUFA12/NDUFAF2 in AD | Terminal maturation failure |
| T3-A | ASO rescue (NDUFAF5) | 18 months | CI activity restoration | Causal necessity (splice→CI) |
| T3-B | CRISPR isogenic | 24 months | Stable isogenic CI comparison | Causal necessity (permanent) |
| T4 | Braak stage stratification | 18 months | Switch precedes depletion | Temporal ordering |
| T5 | U1-70K correlation | 12 months | N40K correlates with switch | Upstream driver |
| T6 | Combined ASO rescue | 24 months | Additive CI recovery | Therapeutic potential |

**Critical path**: T1-C (Arg73 hydroxylation) is the most direct and fastest test of the core molecular claim (NDUFAF5→NDUFS7 enzyme-substrate pair). T3-A (ASO rescue) is the decisive causal test. Both should be prioritised above all other experiments.

---

## 10. Relationship to Existing Hypotheses of AD Mitochondrial Dysfunction

The isoform-switch hypothesis is not mutually exclusive with existing mechanisms. It is likely that CI depletion in AD arises from multiple converging mechanisms operating at different timescales and with different cell-type specificities:

1. **Early stage (pre-symptomatic):** U1 snRNP dysfunction (triggered by tau pathology or primary splicing factor changes) begins to shift NDUFAF5, NDUFS7, NDUFS8, and NDUFS4 splicing ratios in specific neuron types. CI biogenesis efficiency declines gradually.

2. **Intermediate stage:** Reduced CI assembly throughput increases mitochondrial ROS production (partial CI inhibition generates superoxide at FeS cluster N2), creating an oxidative environment that further damages incoming subunits — a feed-forward amplification loop.

3. **Late stage (symptomatic):** mtDNA mutations accumulate, transcriptional suppression of CI genes may occur secondarily, and global mitochondrial membrane potential decline compounds the assembly defects. At this stage, multiple mechanisms are co-active.

The isoform-switch mechanism is most testable and most specific to early/intermediate stages when CI depletion is initiating rather than terminal. Testing it at early Braak stages is therefore scientifically most informative.

---

## 11. Discussion

The core novelty of this hypothesis is mechanistic specificity. Previous CI depletion hypotheses explain *that* CI is depleted but not *how* cell type, subunit, and assembly stage specificity emerges. The isoform-switch model makes all three specificity claims:
- **Cell-type specificity**: switch frequencies are neuron-type-selective, not pan-cellular
- **Subunit specificity**: the affected genes occupy distinct, non-redundant assembly stage positions
- **Stage specificity**: each switch predicts failure at a specific assembly checkpoint, generating a distinct stalled intermediate

The model also provides an upstream driver (U1 snRNP dysfunction) with independent literature support and an established connection to tau pathology — making it compatible with the amyloid/tau cascade without requiring AD pathology to directly damage mitochondrial proteins.

The strongest internal evidence for the hypothesis is the **enzyme–substrate pair**: NDUFAF5 (the hydroxylase) and NDUFS7 (its direct substrate) both show AD-specific isoform switches in the same cell type, disrupting the same biochemical step from two independent molecular mechanisms. This co-disruption is unlikely under a random splicing noise model: the probability of two genes at the same pathway step both showing same-direction AD-enriched isoform changes in the same cell type, by chance, is low. The same logic extends to the three-stage convergence of the full four-gene set.

The weakest point of the hypothesis is the absence of intervention data. Until splice-correcting ASOs rescue CI assembly in AD neurons, the model remains at the level of "mechanistically coherent hypothesis" rather than "experimentally validated mechanism." We believe the convergence of observational evidence is sufficient to justify the experimental investment, and propose the ASO rescue experiment as the most direct and achievable test within a 2-year timeline.

---

## 12. Conclusion

We propose that AD-specific alternative splicing — mediated by U1 snRNP dysfunction — disrupts CI biogenesis at three sequential assembly stages through cell-type-specific isoform switches in NDUFAF5, NDUFS7, NDUFS8, and NDUFS4. The resulting stalled assembly intermediates are degraded, reducing steady-state CI subunit abundance in a cell-type-selective pattern consistent with proteomics observations. This hypothesis provides a mechanistic link between the well-established splicing pathology of AD and the well-established mitochondrial CI depletion of AD, connecting two previously parallel lines of evidence through a structurally grounded assembly cascade model. Future interventional experiments targeting the splice sites identified here will test whether restoring canonical isoform expression is sufficient to rescue CI assembly and bioenergetic competence in AD neurons.

---

## References

1. Adav SS, Park JE, Sze SK. Quantitative profiling brain proteomes revealed mitochondrial dysfunction in Alzheimer's disease. *Mol Brain*. 2019;12(1):8. PMID: 30760317.

2. Bai B, Wang X, Li Y, et al. Deep multilayer brain proteomics identifies molecular networks in Alzheimer's disease progression. *Neuron*. 2021;105(6):975–991.e7. PMID: 31926610.

3. Wang W, Zhao F, Ma X, Perry G, Zhu X. Mitochondria dysfunction in the pathogenesis of Alzheimer's disease: recent advances. *Prog Neurobiol*. 2020;193:101877. PMID: 32461161.

4. Mathys H, Davila-Velderrain J, Peng Z, et al. Single-cell transcriptomic analysis of Alzheimer's disease. *Nature*. 2019;570(7761):332–337. PMID: 31042697.

5. Allen M, Wang X, Serie DJ, et al. Gene expression, methylation and neuropathology correlations at progressive supranuclear palsy risk loci. *Acta Neuropathol*. 2018;136(5):709–723.

6. Chen PC, et al. Alzheimer's disease-associated U1 snRNP splicing dysfunction causes neuronal hyperexcitability and cognitive impairment. *Nature Aging*. 2022;2:923–940. PMID: 36636325. doi:10.1038/s43587-022-00290-0. PMC: PMC9833817.

7. Farhadieh ME, Ghaedi K. Analyzing alternative splicing in Alzheimer's disease postmortem brain: a cell-level perspective. *Front Mol Neurosci*. 2023;16:1237874. PMID: 37799732. doi:10.3389/fnmol.2023.1237874. PMC: PMC10548223.

8. Rhein VF, Carroll J, Ding S, Fearnley IM, Walker JE. NDUFAF5 hydroxylates NDUFS7 at an early stage in the assembly of human complex I. *J Biol Chem*. 2016;291(28):14851–14860. PMID: 27226531. doi:10.1074/jbc.M116.734970.

9. Sung AY, Alston CL, et al. Systematic analysis of NDUFAF6 in complex I assembly and mitochondrial disease. *Nature Metabolism*. 2024;6:1128–1142. PMID: 38720117. doi:10.1038/s42255-024-01039-2. PMC: PMC11395703.

10. Yin Z, Agip AN, Bridges HR, Hirst J. Structural insights into respiratory complex I deficiency and assembly from the mitochondrial disease-related ndufs4−/− mouse. *EMBO J*. 2024;43(2):225–249. doi:10.1038/s44318-023-00001-4. PMC: PMC10897435.

11. Lavdovskaia E, Kolander E, Treunite E, et al. The human Obg protein GTPBP10 is involved in mitoribosomal biogenesis. *Nucleic Acids Res*. 2018;46(16):8471–8482.

---

*Draft v0.1 — 2026-07-02*  
*Status: Hypothesis paper — awaiting full citations for refs 6, 7, 9, 10 and interventional experimental validation*
