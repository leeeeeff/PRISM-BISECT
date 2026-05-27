# DIFFUSE + BISECT

**DIFFUSE**: Deep Isoform Function Prediction Using Sequence Embeddings  
**BISECT**: Biological Isoform-Switch Evidence Characterization Tool

> Lee S et al. "DIFFUSE: Deep Isoform Function Prediction Using Sequence Embeddings, with
> Multi-Evidence Characterization of Alzheimer's Disease Isoform Switches by BISECT."
> *Nature Methods* (under review), 2026.

---

## Overview

| Component | Input | Output | Key metric |
|-----------|-------|--------|------------|
| **DIFFUSE** | ESM-2 protein embeddings (640d) | GO term probability per isoform | Macro AUPRC 0.685 (muscle), 0.600 (brain zero-shot) |
| **BISECT** | CT/AD isoform pairs + DIFFUSE Δ scores | Multi-evidence tier classification (A/B/C) | 13/26 Stage 2 cases SUPPORTED by STRING PPI |

DIFFUSE is trained on 36,748 isoforms from human skeletal muscle long-read scRNA-seq and applied
zero-shot to 63,994 isoforms from human prefrontal cortex (Alzheimer's disease cohort).
BISECT applies 12 sequential analysis modules (Pfam domain annotation → AlphaFold structural
confidence → STRING PPI validation → phyloP conservation) to prioritize disease-relevant isoform
switches discovered by DIFFUSE + DTU integration.

---

## Repository Structure

```
DIFFUSE/
├── hMuscle/
│   ├── model/
│   │   ├── v15d_bp_clean.py          # Production DIFFUSE model (v15d)
│   │   ├── integrated_full_model.py  # Latest model
│   │   ├── run_GPU_Full.py           # Main training entry point
│   │   └── results_isoform/
│   │       ├── evaluation.py         # AUPRC/AUROC evaluation
│   │       └── triplet_analysis.py
│   ├── data/                         # Processed inputs (not included in repo)
│   └── saved_models/                 # Model checkpoints (not included in repo)
│
├── Final_analysis/
│   └── pipeline_bioanalysis/         # BISECT pipeline
│       ├── orchestrate.py            # Main BISECT entry point
│       ├── config.yaml.example       # Configuration template (copy → config.yaml)
│       ├── cases_input.csv           # Example input (53 AD isoform switch candidates)
│       ├── modules/                  # M1–M12 analysis modules
│       │   ├── m1_extract_seq.py
│       │   ├── m2_hmmscan.py
│       │   ├── m3_motif_analysis.py
│       │   ├── m4_genomic_coords.py
│       │   ├── m5_repeatmasker.py
│       │   ├── m6_report.py
│       │   ├── m7_compare.py
│       │   ├── m8_seq_validation.py
│       │   ├── m9_nmd_screen.py
│       │   ├── m10_alphafold.py
│       │   ├── m11_ppi.py
│       │   └── m12_conservation.py
│       ├── figA_pipeline.R           # Figure 1 (pipeline schematic)
│       ├── figB_cases.R              # Figure 2 (domain maps)
│       ├── figC_heatmap.R            # Figure 3 (multi-evidence heatmap)
│       └── figD_network.R            # Figure 4 (pathway network)
│
└── reports/
    └── manuscript_merged_v1.md       # Full manuscript draft
```

---

## Installation

### 1. DIFFUSE (Python + PyTorch)

```bash
conda create -n diffuse_env python=3.9
conda activate diffuse_env
pip install -r requirements.txt
```

**GPU requirement**: NVIDIA GPU with ≥16 GB VRAM recommended (training); inference runs on CPU.

### 2. BISECT (Python + external tools)

```bash
conda activate diffuse_env
pip install -r requirements_bisect.txt
```

BISECT additionally requires the following external tools:

| Tool | Version | Required for | Install |
|------|---------|--------------|---------|
| HMMER | ≥ 3.3.2 | M2 (Pfam domain scan) | `conda install -c bioconda hmmer` |
| Pfam-A.hmm | 36.0 | M2 | [ftp.ebi.ac.uk/pub/databases/Pfam](https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz) |
| TransDecoder | ≥ 5.7.1 | M1 (ORF prediction) | `conda install -c bioconda transdecoder` |
| RepeatMasker | ≥ 4.1.5 | M5 (TE annotation) | `conda install -c bioconda repeatmasker` |
| minimap2 | ≥ 2.24 | M4 (alignment) | `conda install -c bioconda minimap2` |

After installing HMMER, index the Pfam database:

```bash
hmmpress /path/to/Pfam-A.hmm
```

Set paths in `config.yaml` (copy from `config.yaml.example` and edit).

---

## Quick Start

### DIFFUSE: Isoform GO term prediction

```bash
conda activate diffuse_env
cd hMuscle

# Train from scratch (GPU)
python run_GPU_Full.py

# Evaluate on muscle test set
python results_isoform/evaluation.py \
    --model saved_models/v15d_bp_clean.pt \
    --output results_isoform/eval_output/
```

**Inference on new isoforms** (e.g., brain IsoQuant FASTA):

```python
from model.v15d_bp_clean import DIFFUSEModel
import torch, numpy as np

# Load pre-trained model
model = DIFFUSEModel.load('saved_models/v15d_bp_clean.pt')

# ESM-2 embeddings (640d, mean-pooled) for your isoforms
embeddings = np.load('your_esm2_embeddings.npy')  # shape: (N, 640)

scores = model.predict(embeddings)   # shape: (N, n_go_terms)
```

### BISECT: Multi-evidence isoform switch characterization

```bash
cd Final_analysis/pipeline_bioanalysis

# Copy and edit configuration
cp config.yaml.example config.yaml
# Edit config.yaml: set paths.faa, paths.pfam_db, paths.output_dir

# Screen mode (Stage 1+2 only, fast)
python orchestrate.py --mode screen

# Full deep analysis (all 12 modules)
python orchestrate.py --mode deep --workers 4

# Single case
python orchestrate.py --case KIF21B --mode deep --force

# Dry run (check inputs without running)
python orchestrate.py --dry-run
```

Output per case: `outputs/<GENE>_<CELLTYPE>/`
- `analysis.json` — full evidence summary
- `domain_map.pdf` — Pfam domain architecture visualization
- `report.md` — human-readable case report

Cross-case summary: `outputs/cases_summary_<TIMESTAMP>.tsv`

---

## Reproducing Paper Results

### Figure 1 (AUPRC comparison table)

```bash
cd hMuscle
python results_isoform/evaluation.py --model saved_models/v15d_bp_clean.pt
```

Expected: Macro AUPRC = 0.685 (Type-B terms), LR baseline = 0.363.

### Figure 3 (BISECT multi-evidence heatmap)

```bash
cd Final_analysis/pipeline_bioanalysis
python orchestrate.py --mode deep --workers 4
Rscript figC_heatmap.R
```

### Figure 4 (Pathway convergence network)

```bash
Rscript figD_network.R
# Output: outputs/figD_network.pdf, figD_network.png
```

### Ablation study

```bash
cd hMuscle
# no_triplet, no_focal, no_ppi, no_esm, no_cellloc, no_isoform_specific
python run_GPU_Full.py --ablation no_triplet
python run_GPU_Full.py --ablation no_focal
```

---

## Data Availability

| Data | Status | Location |
|------|--------|----------|
| Human skeletal muscle scLR-seq | Available on request | Samsung Medical Center |
| Samsung AD prefrontal cortex scLR-seq | Available on request (IRB SMC 2021-08-031) | Samsung Medical Center |
| ESM-2 embeddings (muscle, 36,748 isoforms) | Zenodo [LINK] | Upon acceptance |
| ESM-2 embeddings (brain, 63,994 isoforms) | Zenodo [LINK] | Upon acceptance |
| Pre-trained DIFFUSE model weights (v15d) | Zenodo [LINK] | Upon acceptance |
| BISECT output JSON (53 cases) | This repository | `Final_analysis/pipeline_bioanalysis/outputs/` |

Transcript sequences are derived from GENCODE v43 (known isoforms) and IsoQuant v3.3 novel transcripts.
No patient-identifiable information is included in this repository.

---

## Citation

```bibtex
@article{lee2026diffuse,
  title={DIFFUSE: Deep Isoform Function Prediction Using Sequence Embeddings,
         with Multi-Evidence Characterization of Alzheimer's Disease Isoform Switches by BISECT},
  author={Lee, Seungwon and others},
  journal={Nature Methods},
  year={2026},
  note={Under review}
}
```

---

## License

Code: MIT License  
Data: CC BY 4.0 (upon acceptance)  
Model weights: CC BY-NC 4.0

---

## Contact

Seungwon Lee — seungwon.david.lee@gmail.com
