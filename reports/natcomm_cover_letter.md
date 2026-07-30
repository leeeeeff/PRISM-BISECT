# Cover Letter — Nature Communications

**To the Editors of Nature Communications,**

We are pleased to submit our manuscript, *"Sequence-first isoform-resolution function prediction maps the limits of protein language models and reveals a Complex I enzyme–substrate cascade disrupted in Alzheimer's disease"*, for consideration as an original Article in *Nature Communications*.

## The Core Problem

Alternative splicing generates proteomic diversity that existing annotation tools cannot resolve. InterProScan and pfam2go — the field standard — annotate canonical protein sequences and assign identical GO terms to all isoforms of a gene, leaving ~80% of expressed isoforms functionally uncharacterised (our data: 80.9% of isoforms carry no Pfam domain). When Alzheimer's disease-associated splicing changes remove a mitochondrial targeting sequence or introduce a premature stop codon, no existing computational tool propagates this into a revised isoform-level functional prediction. This representational gap is not a minor limitation — it systematically occludes the biological consequences of disease-associated splicing.

## What We Present

We introduce two integrated tools:

**PRISM** (Protein-isoform Resolution via Intrinsic Sequence Modeling) trains a lightweight multi-label classifier on top of frozen ESM-2 protein language model embeddings, without requiring domain database matches. Despite training on gene-level GO labels only, PRISM spontaneously discriminates isoforms within the same gene: the learned 18-dimensional representation shows within-gene prediction variance (0.00126) exceeding between-gene variance (0.00070), a reversal of raw ESM-2's gene-dominated structure (within/between ratio = 0.23). A layer-contrast delta architecture (δ_layer = L30 − L15), which captures how individual sequences resolve from local to global encoding across ESM-2 transformer layers, achieves macro AUPRC **0.734** (95% CI: 0.723–0.747) across 82 Molecular Function GO terms on held-out muscle isoforms (0.647 in true-brain zero-shot transfer) — the highest of eleven benchmarked methods, above ESM-2 fine-tuning, k-NN retrieval, DeepFRI, and DeepGoPlus. Importantly, the paper's central claim is *not* that PRISM out-predicts prior tools: this macro metric is dominated by gene identity (a gene-mean oracle reaches 0.803), and a training-free embedding-distance baseline matches PRISM on within-gene isoform ranking. The contribution is instead to *delineate*, and biologically *apply*, the boundary of what protein language models can and cannot resolve at isoform level — a boundary intrinsic to the ESM-2 representation, which we map. On a curated benchmark of 51 UniProt/Swiss-Prot reviewed isoform pairs (48 evaluable; kinase domain loss, large truncations, transcription-factor activation-domain loss, E3-ligase architecture), all 11 high-confidence pairs (prediction gap ≥ 0.10) are correctly directed (permutation p < 0.001, n = 10,000), every one involving complete domain loss or major truncation. Per-locus domain-ranking AUC reaches 0.630 [0.613, 0.646] on held-out muscle and 0.775 in true-brain zero-shot, versus a mathematical null of 0.500 — within-gene isoform discrimination that no gene-level classifier can produce.

**BISECT** (Biological Isoform-Switch Evidence Characterization Tool) is a 15-module pipeline integrating DRIMSeq donor-level DTU statistics, PRISM functional scores, domain architecture analysis, PPI disruption, NMD prediction, and subcellular localisation signals into a six-tier evidence hierarchy to characterise disease-associated isoform switches.

## The Key Biological Finding

Applied without retraining to 63,994 Alzheimer's disease prefrontal cortex isoforms, PRISM and BISECT together identify a **hierarchical Complex I disruption cascade** structured around a directly verified enzyme–substrate pair. NDUFAF5 — a hydroxylase assembly factor that obligatorily hydroxylates NDUFS7 at Arg73 during early Q-module intermediate formation (Rhein et al., *J Biol Chem*, 2016) — reaches genome-wide DRIMSeq significance in excitatory neurons via an NMD-switching isoform, and NDUFS7 independently shows a concordant isoform switch in the same excitatory neuron population (permutation p = 0.048). This constitutes a directly verified enzyme–substrate pair, both disrupted in concert in the same cell type in AD — a mechanistic specificity exceeding any prior computational analysis of disease-associated splicing. Downstream disruption is nominated for NDUFS8 (inhibitory neurons; permutation p_adj = 0.022, above the strict Bonferroni threshold of 0.010; 37% effect; batch-concordant across 8 independent donors; replication-required) and NDUFS4 (both neuron types; p < 0.05 uncorrected; replication-required), and the synaptic GEF DOCK11 independently achieves Bonferroni-corrected significance (p_adj = 0.004; 5 pre-specified candidates). KIF21B motor domain loss is independently replicated in an external Ebbert et al. cohort, validating the cross-cohort robustness of BISECT-nominated isoform switches.

## Fit for Nature Communications

This work sits at the intersection of protein language model methodology, long-read single-cell transcriptomics, and Alzheimer's disease biology — a profile that fits Nature Communications' multidisciplinary scope. The methodological contribution (annotation-free, sequence-first isoform function prediction; a metric reframe demonstrating that the field-standard macro AUPRC measures gene identity rather than isoform-specific function, with orthogonal within-gene metrics proposed in its place; and a rigorous characterisation of the representational boundary of protein-language-model isoform resolution) and the biological discovery (hierarchical enzyme–substrate Complex I cascade in AD excitatory neurons) are mutually reinforcing: the framework exists to make discoveries like this, and the discovery validates its biological relevance beyond benchmark metrics. Both PRISM and BISECT will be made publicly available as open-source software with an interactive web application.

We confirm that this manuscript has not been published elsewhere and is not under consideration at another journal.

Sincerely,

Seungwon Lee  
GIST (Gwangju Institute of Science and Technology)  
seungwon.david.lee@gmail.com

---

*Main text: ~4,500 words | Figures: 4 main | Supplementary: Tables S1–S4, Notes S1–S2*
