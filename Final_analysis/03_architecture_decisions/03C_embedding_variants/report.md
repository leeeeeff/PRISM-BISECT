# 03C: Embedding Variants Ablation

## Key Question
Do SwiGLU activation and delta embeddings (isoform minus gene mean) improve over standard ReLU with 640d ESM-2 embeddings?

## Data Sources
- v19 (SwiGLU + delta): reports/v19_swiglu/v19_results_20260519_1935.json  (● hMuscle, 18 terms)
- v20 (SwiGLU only): reports/v20_swiglu_nodelta/v20_results_20260519_2013.json  (● hMuscle, 18 terms)
- v15d baseline: 0.7022 macro AUPRC (reference from v20 JSON baselines field)

## Methodology
Macro AUPRC over 18 GO terms. v20 ablates delta embedding (640d input, SwiGLU). v19 adds delta embedding (1280d input = ESM-2 isoform + ESM-2 gene mean, SwiGLU). Both use residual connections and [512, 256, 128, 64] hidden layers.

## Key Findings

| Variant | Input dim | Activation | Macro AUPRC | vs v15d |
|---------|-----------|-----------|-------------|---------|
| v15d | 640 | ReLU | 0.7022 | — |
| v20 | 640 | SwiGLU | 0.6763 | -0.026 |
| v19 | 1280 | SwiGLU | 0.5462 | -0.156 |

- SwiGLU activation effect: -0.026 (v20 vs v15d)
- Delta embedding effect: -0.130 (v19 vs v20)
- v19 generates 134 isoform reversal genes; v20 generates 159

Per-term patterns: v20 is competitive with v15d on Type-A terms (Motor activity 0.693, Glycolysis 0.792) but underperforms on Type-B terms. v19 uniformly degrades across all 18 terms.

## Figure
03C_embedding_variants.pdf/.png
- Panel a: Three-bar macro AUPRC comparison with annotated effect brackets.
- Panel b: Per-term AUPRC bar chart for v20 vs v19 (first 12 terms), with v15d dashed reference.

## Biological Interpretation
Delta embeddings encode the deviation of each isoform from its gene mean. This should theoretically capture isoform-specific features. However, performance degrades by 0.130, suggesting: (1) gene-mean subtraction destroys protein-level sequence information that ESM-2 encodes in the absolute embedding space; (2) doubling input dimensionality (1280d) with the same label density causes overfitting. The SwiGLU gating mechanism alone (-0.026) may overfit sparse GO labels. Standard ESM-2 640d with ReLU (v15d) is optimal because it preserves the full ESM-2 representation geometry.
