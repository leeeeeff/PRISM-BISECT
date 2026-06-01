# Section 00: Data and Pipeline

## Key Question
What are the input data characteristics and how does the end-to-end pipeline transform long-read sequencing data into isoform function predictions?

## Data Sources
- hMuscle biopsy long-read scRNA-seq (●): ~10,000 isoforms
- Samsung AD IsoQuant scRNA-seq (●): 63,994 isoforms (7,899 novel, 56,095 known)
- UniProt/SwissProt (◇): protein-level GO annotation
- Gene Ontology BP (◇): 18 BP terms as prediction labels

## Key Findings
- Brain: 63,994 isoforms total; 7,899 novel (IsoQuant "transcript" prefix), 56,095 known; ~37,846 coding, ~26,148 non-coding
- Brain SQANTI3 categories: FSM ~29,169, ISM ~26,926, NIC ~4,344, NNIC ~3,555
- Muscle: ~10,000 isoforms, all known, ~7,200 coding
- ESM-2 t30_150M → 640-dim embeddings (sole feature input to model)
- GO label sparsity: most terms < 1,000 positives in muscle test; sparse (n<50) evaluated by AUPRC
- 18 BP GO terms used in v15d_bp_clean; binary label matrix from UniProt Swiss-Prot × GO

## Figures
- fig00_1_pipeline: End-to-end 5-stage pipeline flowchart (89×140mm)
- fig00_2_input_stats: Input data statistics — stacked bar (coding/novel) + SQANTI3 categories (183×80mm)
- fig00_3_go_label_dist: GO term positive counts, muscle vs brain, sparse threshold line (89×120mm)

## Interpretation
Long-read sequencing data (●) → ESM-2 protein embeddings → MLP (v15d_bp_clean) → GO BP predictions. External GO annotations (◇) provide supervision labels. The pipeline is tissue-agnostic by design; the two datasets enable cross-tissue validation.
