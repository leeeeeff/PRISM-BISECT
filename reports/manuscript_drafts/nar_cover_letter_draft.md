# Cover Letter Draft — NAR Methods Online

**To the Editors of Nucleic Acids Research,**

We are pleased to submit our manuscript, *"PRISM: Protein-isoform Resolution via Intrinsic Sequence Modeling, with Multi-Evidence Characterization of Alzheimer's Disease Isoform Switches by BISECT"*, for consideration as an original article in *Nucleic Acids Research*.

## The Problem We Address

Most computational tools for protein function prediction treat all isoforms of a gene as functionally identical — a critical blind spot as long-read single-cell RNA sequencing routinely discovers thousands of novel transcript isoforms absent from all existing databases. At present, InterProScan can annotate only 37.2% of expressed isoforms (those with recognizable Pfam domains); the remaining 62.8% receive no functional annotation. This gap is not merely incomplete — it is systematically biased toward constitutively expressed canonical isoforms, leaving disease-relevant alternative isoforms functionally uncharacterised.

## What We Present

We present two complementary computational tools:

**PRISM** (Protein-isoform Resolution via Intrinsic Sequence Modeling) is a deep learning framework that predicts isoform-level Biological Process GO terms directly from ESM-2 protein language model embeddings, without requiring domain database matches. On 36,748 human skeletal muscle isoforms, PRISM achieves macro AUPRC 0.7022 across 18 GO BP terms — a 45-fold improvement over a domain-presence logistic regression (AUPRC 0.0156), which represents the data-driven upper bound on pfam2go-style annotation. Crucially, PRISM achieves macro AUPRC 0.6844 on the 62.8% of isoforms invisible to domain-based tools — confirming that ESM-2 sequence context encodes sufficient biological information independent of domain architecture.

**BISECT** (Biological Isoform-Switch Evidence Characterization Tool) is a fifteen-module evidence integration pipeline that applies PRISM scores, AlphaFold structural confidence, STRING PPI networks, evolutionary conservation, and regulatory architecture analysis to characterise isoform switches detected by differential transcript usage analysis. Applied to 83 high-confidence cases from an Alzheimer's disease prefrontal cortex cohort and multi-tissue SRA data, BISECT identifies three Tier 1 AD-specific functional reversals: motor polarity disruption (KIF21B), mitochondrial Complex I locus replacement (NDUFS4), and post-synaptic scaffolding loss (DLG1).

## Key Technical Contributions

1. **Isoform-resolution without domain annotation**: PRISM predicts functional GO terms for isoforms lacking any Pfam domain hit (62.8% of tested isoforms), a capability impossible for existing domain-based tools including InterProScan, SIFTS, and pfam2go.

2. **Cross-tissue zero-shot transfer**: A model trained exclusively on muscle achieves macro AUPRC 0.672 on brain isoforms (41-term panel) without any brain supervision, demonstrating biologically grounded generalisation.

3. **Novel isoform coverage**: PRISM scores 7,899 NIC/NNIC brain isoforms absent from all databases, of which 541 receive high-confidence functional predictions (score > 0.5 for ≥1 GO term) — the first computational annotations for these sequences.

4. **Structured evidence integration**: BISECT's multi-module design prevents confirmation bias by requiring independent PPI, structural, and evolutionary evidence for each candidate mechanism. Three AD-specific isoform switches converge on a mechanistically coherent mitochondrial–cytoskeletal–synaptic axis.

5. **Complementary to InterProScan**: In 92.3% of BISECT-characterised cases, PRISM predicts a Biological Process GO term not recoverable from InterProScan+pfam2go, establishing non-overlapping prediction spaces rather than competing approaches.

## Why NAR Methods Online

This work introduces a new computational methodology for isoform-resolution functional annotation, with immediate utility for any research group performing long-read single-cell or bulk RNA sequencing. Both tools are implemented as a Streamlit-based interactive analysis platform with built-in BISECT case viewer, PRISM score visualisation, and upload mode for user-provided long-read sequencing data. Source code, pre-trained model weights, and a Docker image will be made publicly available upon acceptance. The scope aligns with NAR's Methods Online track: a new computational method addressing a clearly defined biological problem, with rigorous quantitative validation and practical availability for the community.

## Ethical and Data Availability Statements

The Samsung Alzheimer's disease long-read single-cell RNA-seq dataset was generated under institutional ethics approval. SRA validation data are publicly available. All analysis code and pre-trained model weights will be deposited in Zenodo upon acceptance.

We confirm that this manuscript has not been published elsewhere and is not under consideration at another journal. All authors have approved the submission.

Sincerely,

Seungwon Lee  
[Institution]  
seungwon.david.lee@gmail.com

---

*Word count (main text): approximately 9,200 words*  
*Figures: 5 main + supplementary*  
*Supplementary Tables: S1–S10*
