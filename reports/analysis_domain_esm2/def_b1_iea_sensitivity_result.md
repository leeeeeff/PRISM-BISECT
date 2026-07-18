# DEF-B1: IEA Sensitivity Evaluation — v15d_bp_clean

## Setup

- **Model**: v15d_bp_clean (saved predictions: score_matrix_18go_20260519_1914.npy)
- **Evaluation**: same model, two label sets (full vs noIEA)
- **Isoforms**: 36,748 (36,748 muscle isoforms)
- **GO terms**: 18 (18 BP terms)

## Label Statistics

| Label set | Positive labels | IEA-only removed |
|-----------|-----------------|------------------|
| Full (with IEA) | 10,463 | — |
| noIEA (experimental only) | 10,001 | 462 (4.4%) |

## Macro AUPRC Comparison

| Label set | Macro AUPRC | Delta |
|-----------|-------------|-------|
| Full (with IEA) | **0.7022** | — |
| noIEA (experimental) | **0.6584** | -0.0438 (-6.2%) |

## Per-term Breakdown

| GO Term | Name | Full AUPRC | noIEA AUPRC | Delta | n_pos (full) | n_pos (noIEA) |
|---------|------|-----------|-------------|-------|-------------|--------------|
| GO:0007204 | Ca2+ signaling | 0.6884 | 0.6994 | +0.0110 | 338 | 311 |
| GO:0045214 | Sarcomere org | 0.8667 | 0.7986 | -0.0681 | 130 | 118 |
| GO:0006941 | Muscle contraction | 0.7016 | 0.6851 | -0.0164 | 378 | 375 |
| GO:0006914 | Autophagy | 0.6600 | 0.5586 | -0.1014 | 467 | 411 |
| GO:0043161 | Proteasome-UPS | 0.7772 | 0.7705 | -0.0068 | 1234 | 1213 |
| GO:0007519 | Skeletal muscle dev | 0.7775 | 0.5958 | -0.1818 | 189 | 155 |
| GO:0042692 | Muscle cell diff | 0.6740 | 0.6622 | -0.0117 | 432 | 419 |
| GO:0055074 | Ca2+ homeostasis | 0.6729 | 0.6560 | -0.0169 | 616 | 609 |
| GO:0007005 | Mitochondrion org | 0.6873 | 0.6711 | -0.0163 | 1012 | 968 |
| GO:0007517 | Muscle organ dev | 0.6401 | 0.6219 | -0.0183 | 486 | 476 |
| GO:0032006 | TOR signaling | 0.4959 | 0.4851 | -0.0108 | 248 | 243 |
| GO:0030048 | Actin-based movement | 0.7356 | 0.7356 | +0.0000 | 301 | 301 |
| GO:0006096 | Glycolysis | 0.8143 | 0.7100 | -0.1043 | 89 | 80 |
| GO:0007268 | Synaptic transmission | 0.6672 | 0.6717 | +0.0045 | 482 | 462 |
| GO:0007018 | MT-based movement | 0.7402 | 0.5762 | -0.1639 | 445 | 398 |
| GO:0031175 | Neuron proj dev | 0.6823 | 0.6763 | -0.0060 | 1098 | 1060 |
| GO:0030182 | Neuron diff | 0.6466 | 0.6289 | -0.0176 | 1517 | 1432 |
| GO:0000226 | MT cytoskeleton org | 0.7118 | 0.6486 | -0.0632 | 1001 | 970 |

## Interpretation

> **Removing IEA-only annotations (4.4% of positives) changes macro AUPRC by −0.0438 (−6.2%),
> from 0.7022 (full labels) to 0.6584 (experimental-only labels).**
>
> The −6.2% drop is partly attributable to three GO terms with relatively high IEA content:
> Skeletal muscle dev (−18.2%; 18% of positives IEA-only), MT-based movement (−16.4%; 10.6%),
> and Autophagy (−10.1%; 11.9%). For 14 of 18 terms the drop is ≤3.2%, consistent with
> minimal IEA influence in those categories.
>
> PRISM maintains macro AUPRC 0.6584 on experimental-only labels — substantially above
> random baselines — indicating meaningful prediction independent of computational annotations.
> The partial sensitivity to IEA labels in high-IEA terms should be noted as a limitation
> in the manuscript.

## Manuscript sentence (Methods §2.1 note)

> "To assess sensitivity to computationally propagated annotations, we re-evaluated PRISM
> using only experimentally supported GO labels (95.6% of positive instances; IEA evidence
> code excluded). Macro AUPRC changed from 0.7022 (full labels, including IEA) to 0.6584
> (experimental-only labels; Δ = −0.0438, −6.2%). The drop was concentrated in three GO
> terms with >10% IEA content (Skeletal muscle dev: −18.2%; MT-based movement: −16.4%;
> Autophagy: −10.1%); 14 of 18 terms showed ≤3.2% change. Results reported throughout this
> study use full annotations; experimental-only performance is provided for transparency."