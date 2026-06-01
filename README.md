# PRISM-BISECT

**PRISM** (Protein-isoform Resolution via Intrinsic Sequence Modeling) and **BISECT** (Biological Isoform-Switch Evidence Characterization Tool) — a two-stage framework for isoform-level function prediction and disease-associated isoform-switch characterization from long-read single-cell RNA sequencing data.

> Manuscript in preparation. Target: *Nature Methods* / *Nature Machine Intelligence*.

---

## Overview

Isoform-level function prediction is fundamentally limited by two problems: **data sparsity / mode collapse** under sparse GO annotation, and **gene-level reference dominance** where models rely on gene identity rather than isoform-intrinsic sequence features. PRISM addresses both through ESM-2 protein language model embeddings and focal loss training. BISECT then applies a 15-module evidence pipeline to characterize isoform switches discovered by PRISM in Alzheimer's disease (AD) long-read single-cell data.

### Key Performance

| Dataset | Macro AUPRC | Setting |
|---------|-------------|---------|
| Human skeletal muscle (18 BP GO terms) | **0.7022** | Supervised |
| Human brain AD/CT (zero-shot transfer) | **0.5998** | Zero-shot |
| DIFFUSE baseline (Yao et al., *Nat Methods* 2022) | 0.4981 | Supervised |

### Tier A AD Isoform Switches (BISECT)

| Gene | Cell type | Switch | PRISM Δ | Mechanism |
|------|-----------|--------|---------|-----------|
| **KIF21B** | Excitatory | NIC (418aa) → NNIC (710aa) | −0.855 | WD40 β-propeller gain; kinesin motor loss |
| **NDUFS4** | Excitatory | CT (175aa) → NNIC tr73243 (378aa) | −0.563 | LINE exon poaching via shared promoter (13 bp TSS) |
| **DLG1** | OPC | NNIC tr319500 (186aa) → canonical (926aa) | +0.857 | Paradoxical synaptic scaffold gain in AD |
| **PTPRF** | Astrocyte | Alt-promoter switch (60,574 bp TSS shift) | −0.621 | Liprin-α binding loss; C2/PDZ domain reduction |

---

## Repository Structure

```
PRISM-BISECT/
├── hMuscle/
│   ├── model/                  # PRISM model versions (v3 → v20)
│   │   ├── v15d_bp_clean.py    # Production model (Macro AUPRC 0.7022)
│   │   ├── v15d_unified.py     # Unified training script
│   │   └── v15f_layer_select.py # ESM-2 layer selection experiments
│   ├── preprocessing/          # ESM-2 embedding computation, data prep
│   └── results_isoform/        # Evaluation scripts and summary CSVs
│
├── Final_analysis/
│   ├── pipeline_bioanalysis/   # BISECT pipeline
│   │   ├── orchestrate.py      # Main pipeline runner (15 modules)
│   │   ├── modules/            # m1_extract_seq → m15_compare
│   │   ├── templates/          # Jinja2 case report templates
│   │   ├── BISECT_case_report.md       # Full 26-case PASS report
│   │   ├── TierA_mechanism_synthesis.md # Tier A deep-dive (4 cases)
│   │   └── cases_input_sra.csv         # SRA input case list
│   ├── 00_data_and_pipeline/ → 08_synthesis/  # Section analysis reports
│   └── 07_ad_isoform_switching/        # KIF21B, NDUFS4, PTPRF analyses
│
├── reports/
│   └── manuscript_merged_v1.md         # Full manuscript draft
│
└── CLAUDE.md                           # Project research rules
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/leeeeeff/PRISM-BISECT.git
cd PRISM-BISECT

# Create conda environment
conda create -n isoform_env python=3.9
conda activate isoform_env

# Install dependencies
pip install tensorflow==2.12 fair-esm torch biopython pandas numpy scikit-learn
pip install jinja2 requests scipy

# For BISECT domain annotation (HMMER required separately)
# Pfam-A.hmm database: https://www.ebi.ac.uk/interpro/download/pfam/
```

---

## Usage

### PRISM — Isoform Function Prediction

```bash
cd hMuscle

# Pre-compute ESM-2 embeddings (requires GPU)
conda activate isoform_env
python preprocessing/compute_esm2_all_layers.py

# Train production model (v15d)
python model/v15d_bp_clean.py

# Full training run with logging
nohup python run_GPU_Full.py > logs_isoform/run_$(date +%Y%m%d_%H%M).log 2>&1 &

# Evaluate
python results_isoform/evaluation.py
```

**Architecture (v15d_bp_clean):**
```
ESM-2 (esm2_t30_150M_UR50D, 640-dim, layer 30)
  → Dense(256, ReLU) → BatchNorm → Dropout(0.3)
  → Dense(128, ReLU) → Dropout(0.2)
  → Dense(64, ReLU)
  → Dense(18, sigmoid)        # 18 BP GO terms
Loss: BinaryFocalCrossentropy(γ=2.0)
```

### BISECT — Isoform Switch Characterization

```bash
cd Final_analysis/pipeline_bioanalysis

# Edit config.yaml with local paths
cp config.yaml.example config.yaml

# Run full pipeline on a case list
python orchestrate.py --cases cases_input_sra.csv --output outputs/

# Run on a single case
python orchestrate.py --gene KIF21B --cell_type Excitatory

# Batch run
bash run_batch_remaining.sh
```

**BISECT Pipeline (15 modules):**

| Module | Function |
|--------|----------|
| M1 | Sequence extraction & ORF translation |
| M2 | HMMER domain annotation (Pfam-A) |
| M3 | NMD susceptibility screening |
| M4 | Genomic coordinate mapping |
| M5 | Isoform-specific motif detection |
| M6 | NMD gate filtering |
| M7 | Sequence-level validation |
| M8 | Regulatory context classification |
| M9 | Promoter usage (TSS proximity) |
| M10 | Alternative polyadenylation (APA) |
| M11 | AlphaFold2 structural confidence |
| M12 | PPI network analysis (STRING) |
| M13 | Evolutionary conservation (phyloP100way) |
| M14 | Case report generation |
| M15 | Cross-case comparative analysis |

---

## Data Availability

| Data | Source | Access |
|------|--------|--------|
| Human skeletal muscle long-read scRNA-seq | In-house (ONT) | Available upon request |
| Human brain AD/CT long-read scRNA-seq | In-house (IsoQuant GTF, 10,817 novel isoforms) | Available upon request |
| SRA public validation cohort (42 samples) | NCBI SRA | `cases_input_sra.csv` |
| ESM-2 embeddings | Meta AI (fair-esm) | `preprocessing/compute_esm2_all_layers.py` |
| Pfam-A HMM database | EMBL-EBI InterPro | https://www.ebi.ac.uk/interpro/download/pfam/ |

> **Note**: Raw sequencing data, processed count matrices, model checkpoints (`hMuscle/saved_models/`), and training data (`hMuscle/data/`) are not included in this repository. Available upon reasonable request.

---

## Reproducibility

All numeric results in the manuscript were verified against per-case `analysis.json` pipeline outputs. Key files:

- `Final_analysis/pipeline_bioanalysis/outputs/run_summary_20260531_1425.md` — BISECT run summary (84 PASS / 121 cases)
- `Final_analysis/pipeline_bioanalysis/outputs/supplementary_table_S_bisect_121cases.tsv` — Full case table
- `hMuscle/results_isoform/total_performance_summary.csv` — PRISM evaluation summary

---

## Citation

```bibtex
@article{lee2026prism,
  title   = {PRISM: isoform-level function prediction via protein language model embeddings
             reveals disease-associated alternative splicing in Alzheimer's disease},
  author  = {Lee, Seungwon and ...},
  journal = {Nature Methods},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

Code: MIT License.
Data and manuscript text: All rights reserved pending publication.
