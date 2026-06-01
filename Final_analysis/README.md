# DIFFUSE — Final Analysis Repository

**Project**: Isoform function prediction via ESM-2 deep learning  
**Target journal**: Nature Methods / Nature Machine Intelligence  
**Last updated**: 2026-05-19

---

## Data Provenance Convention

All figures and reports use two icons to mark data origin:

| Icon | Meaning |
|------|---------|
| ● | **Original long-read data** — hMuscle biopsy (long-read bulk scRNA-seq) or Samsung AD scRNA-seq (IsoQuant, single-cell long-read) |
| ◇ | **External database** — UniProt/SwissProt, Gene Ontology, AlphaFold DB, GRCh38 genome |

---

## Model Versions

| Model | GO terms | Architecture | Macro AUPRC (muscle) | Status |
|-------|----------|--------------|----------------------|--------|
| v10-B | 13 BP terms | ESM-2 640d → Dense(256→128→64) | 0.685 (Type-B) | Reference |
| **v15d_bp_clean** | **18 BP terms** | Same + BN + Dropout | **0.7022** | **PRODUCTION** |
| v11-A | 13 | + Gene-context attention | 0.571 | Ablation (negative) |
| v11-B | 13 | + Isoform deviation | 0.575 | Ablation (negative) |
| v19 | 18 | SwiGLU + delta embed | 0.546 | Ablation (negative) |
| v20 | 18 | SwiGLU only | 0.676 | Ablation (negative) |

**Primary model for all analyses: v15d_bp_clean**

---

## Directory Structure

```
Final_analysis/
├── 00_data_and_pipeline/          # Input data statistics + E2E pipeline
│   ├── 00A_input_data_statistics/ # Isoform counts, ESM-2 coverage, GO distribution
│   ├── 00B_pipeline_overview/     # End-to-end flowchart
│   └── 00C_go_label_distribution/ # Per-term positive class sizes
│
├── 01_muscle_performance/         # MUSCLE TISSUE — benchmarking [●]
│   ├── 01A_baseline_comparison/   # v15d vs LR vs RF vs XGB on 13/18 GO terms
│   ├── 01B_statistical_validation/ # Bootstrap CI, 5-seed stability
│   ├── 01C_xgboost_challenge/     # XGB point estimate > v15d → CI overlap analysis
│   └── 01D_18go_extended/         # 18 BP GO term evaluation
│
├── 02_go_predictability_framework/ # WHY the model works / doesn't work
│   ├── 02A_typeAB_classification/  # sep_cosine threshold, LOOCV 13/13
│   ├── 02B_3case_quantitative/    # pc1_var_ratio r=-0.765 framework
│   ├── 02C_annotation_quality_negative/ # TBS/TCS null result
│   └── 02D_embedding_geometry/    # UMAP, cluster analysis
│
├── 03_architecture_decisions/     # MODEL DESIGN — what was tried and why
│   ├── 03A_flat_mlp_rationale/   # v15d architecture diagram
│   ├── 03B_gene_context_ablation/ # v11-A (attention), v11-B (deviation)
│   ├── 03C_embedding_variants/    # v19 (SwiGLU+delta), v20 (SwiGLU)
│   └── 03D_negative_catalogue/    # All rejected approaches
│
├── 04_isoform_level_resolution/   # PROOF: model sees isoforms, not just genes
│   ├── 04A_within_gene_discrimination/ # pos_bias=1.2
│   ├── 04B_coding_noncoding_detection/ # 17-485× coding vs non-coding
│   └── 04C_structural_validation/      # AlphaFold pLDDT attempt
│
├── 05_muscle_isoform_discovery/   # MUSCLE TISSUE — novel biology [●]
│   ├── 05A_phase5_candidates/    # 100 genes, 48 switches
│   ├── 05B_literature_validated/ # TPM1, DMD, ANK2
│   └── 05C_annotation_gaps/      # DYNC2I1/2, MYO5C
│
├── 06_brain_cross_tissue/         # BRAIN TISSUE — generalization [●]
│   ├── 06A_data_assembly/        # Samsung AD IsoQuant 63,994 isoforms
│   ├── 06B_performance_analysis/ # Muscle 0.702 → Brain 0.600
│   └── 06C_novel_isoform_prediction/ # Novel 7,899 isoforms, AUPRC 0.322
│
├── 07_ad_isoform_switching/       # BRAIN/AD BIOLOGY — key discoveries [●]
│   ├── 07A_kif21b_switch/        # Excitatory: MT-movement → Autophagy
│   ├── 07B_ndufs4_hijacking/     # Excitatory: Complex I → novel 379aa protein
│   ├── 07C_dlg1_opc/             # OPC: scaffold 81% → 12% (p=9e-10)
│   ├── 07D_statistical_summary/  # Forest plot of all 6 key isoforms
│   └── 07E_domain_structure/     # Protein domain analysis
│
├── 08_synthesis/                  # PAPER NARRATIVE
│   ├── 08A_key_claims/           # 5 claims + evidence map
│   ├── 08B_negative_results/     # Methodological value of failures
│   └── 08C_limitations/          # Known limits + future directions
│
└── figures_consolidated/          # ALL FIGURES CENTRALIZED
    ├── existing/                  # Copied from reports/ (25 files)
    └── generated/                 # New figures from this analysis run
```

---

## Key Numbers Quick Reference

### Muscle Performance (v15d_bp_clean, 18 GO terms)
- Macro AUPRC: **0.7022** vs LR 0.363 vs RF 0.147
- Type-B (11 terms): AUPRC advantage +88.7% over LR
- XGB: 0.7384 (CI overlap with v15d: 13/13 = statistically non-inferior)

### GO Term Framework
- sep_cosine threshold: 0.060, LOOCV accuracy: 13/13
- pc1_var_ratio vs ΔAUPRC: r=-0.765, p=0.002
- TBS/TCS: r<-0.15, p>0.6 (null result)

### Isoform Resolution
- pos_bias (v15d): 1.196 (1.0 = no within-gene discrimination)
- Coding/non-coding ratio: 17–485×

### Cross-tissue (Brain)
- Macro AUPRC: 0.5998 (Δ=-0.102 from muscle)
- Novel isoform only: 0.3217

### AD Isoform Switches
| Case | Gene | Cell type | Key switch | p-value |
|------|------|-----------|-----------|---------|
| 1 | KIF21B | Excitatory | tr293004 CT→0, tr292978 0→AD | 9.3e-8, 3.8e-6 |
| 2 | NDUFS4 | Excitatory | Canonical 44→7%, tr73243 0→43% | 2.6e-4, 3.6e-6 |
| 3 | DLG1 | OPC | tr319500 81→12% | 9.0e-10 |
