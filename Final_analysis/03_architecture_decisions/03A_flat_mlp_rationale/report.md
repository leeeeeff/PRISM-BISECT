# 03A: Flat MLP Rationale

## Key Question
Why does a flat 3-layer MLP (v15d_bp_clean) outperform gene-context-aware and activation-modified architectures?

## Data Sources
- v11-A attention: reports/v11_attention/v11a_partial_20260518_0133.json  (● hMuscle)
- v11-B deviation: reports/v11_deviation/v11b_results_20260518_0142.json  (● hMuscle)
- v19 SwiGLU+delta: reports/v19_swiglu/v19_results_20260519_1935.json    (● hMuscle)
- v20 SwiGLU-only: reports/v20_swiglu_nodelta/v20_results_20260519_2013.json (● hMuscle)

## Methodology
Macro AUPRC (18 GO terms) compared across 5 architectural variants. All trained on hMuscle long-read single-cell data with identical ESM-2 640d inputs (except v19 which uses 1280d delta embeddings).

## Key Findings

| Model | Architecture | Macro AUPRC |
|-------|-------------|-------------|
| v15d (baseline) | Flat MLP, ReLU, 640d | 0.7022 |
| v20 | SwiGLU, 640d | 0.6763 |
| v11-B | Isoform deviation | 0.5744 |
| v11-A | Gene-context attention | ~0.571 |
| v19 | SwiGLU + delta, 1280d | 0.5462 |

- Flat MLP outperforms gene-context attention by +0.131 macro AUPRC
- SwiGLU activation alone costs -0.026 AUPRC relative to ReLU
- Adding delta embeddings (isoform minus gene mean, 1280d) costs a further -0.130 AUPRC

## Figure
03A_architecture_comparison.pdf/.png
- Panel a: Text-schematic of v15d_bp_clean (ESM-2 640d to Dense 256, BN+Drop 0.3, Dense 128, Drop 0.2, Dense 64, sigmoid).
- Panel b: Macro AUPRC bar chart with error bars. Dashed line at v15d = 0.7022.

## Biological Interpretation
GO annotations are gene-level (UniProt assigns function to gene products, not isoforms). Architectures that model gene context (v11-A mean-embedding attention; v11-B isoform-minus-gene-mean delta) cause over-fitting to gene identity rather than isoform-intrinsic sequence features. The flat MLP forces reliance solely on ESM-2 sequence embeddings, encoding actual protein sequence differences between isoforms. SwiGLU gating introduces capacity that overfits sparse GO labels. Architecture simplicity is a methodological contribution: when labels are gene-level, simpler architectures generalize better to isoform resolution.
