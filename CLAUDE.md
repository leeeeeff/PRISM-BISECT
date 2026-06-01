# DIFFUSE project — PRISM + BISECT Isoform Function Prediction Research

## Model Names
- **PRISM** (Protein-isoform Resolution via Intrinsic Sequence Modeling): DL prediction model
- **BISECT** (Biological Isoform-Switch Evidence Characterization Tool): downstream analysis pipeline

## Project
Building PRISM to overcome two core limitations of DIFFUSE baseline (Yao et al., Nat Methods 2022):
- **Data Sparsity / Mode Collapse**: focal loss + bio-constrained training
- **Gene-level Reference Dominance**: isoform-intrinsic sequence features via ESM-2

## Active Codebase
```
hMuscle/
├── model/              ← v15d_bp_clean (production), v15f_layer_select.py
├── results_isoform/    ← evaluation.py
├── data/               ← processed inputs
├── src/                ← shared utilities
├── preprocessing/      ← data prep (compute_esm2_all_layers.py)
└── saved_models/       ← checkpoints
```

## Architecture (PRISM v15d_bp_clean — confirmed production)
- Input: ESM-2 (esm2_t30_150M_UR50D, 640-dim) pre-computed embeddings
- Model: Dense(256, ReLU) → BN → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → sigmoid
- Loss: BinaryFocalCrossentropy (γ=2.0)
- Ensemble: N seeds averaged
- GO terms: 18 BP terms
- Performance: Macro AUPRC 0.7022 (muscle), 0.5998 (brain zero-shot)

## Environment
- Conda: `conda activate isoform_env`
- GPU run: `python run_GPU_Full.py`
- Main model: `v15d_bp_clean.py` / `v15d_unified.py`
- Eval: `results_isoform/evaluation.py`

## Imports
@.claude/rules/architecture.md
@.claude/rules/loss-functions.md
@.claude/rules/mathematical-validation.md

## Research Checklist
- [ ] Reduces gene-level bias?
- [ ] Safe under sparse isoform data?
- [ ] Bio-validity guaranteed if synthetic?
- [ ] Contributes to novel case discovery?
- [ ] Defensible as NMI/Nature Methods contribution?
@.claude/rules/anti-local-minima.md
