# 03B: Gene-Context Ablation Detail

## Key Question
Do gene-context-aware modifications (attention over gene mean, isoform deviation embedding) help or hurt per-term AUPRC?

## Data Sources
- v11-A partial results: reports/v11_attention/v11a_partial_20260518_0133.json  (● hMuscle, 2 terms)
- v11-B full results: reports/v11_deviation/v11b_results_20260518_0142.json  (● hMuscle, 13 terms)
- v15d reference: v10b_ref field in both JSONs

## Methodology
Per-term AUPRC comparison. v11-A tested on Ca2+ signaling (GO:0007204) and Sarcomere org (GO:0030017) only (partial run). v11-B tested on all 13 Type-B GO terms. Baseline is v10b_ref (equivalent to v15d_bp_clean performance on those terms). Delta AUPRC computed as model minus baseline per term.

## Key Findings

### v11-A (gene-context attention)
| Term | v15d (v10b_ref) | v11-A | Delta | p |
|------|----------------|-------|-------|---|
| Ca2+ signaling | 0.7653 | 0.6514 | -0.114 | 0.01 |
| Sarcomere org | 0.7426 | 0.6537 | -0.089 | 0.002 |

pos_bias dropped by 66-75%: v11-A injected gene-level bias, destroying within-gene discrimination.

### v11-B (isoform deviation)
- Type-B macro: v10b = 0.6847, v11b = 0.5744 — delta = -0.110
- All 11 Type-B terms show negative delta (range: -0.071 to -0.140)
- Type-A terms (Motor activity, Glycolysis) also degrade: delta -0.071 and +0.069

### Summary
Both gene-context approaches consistently degrade AUPRC across all tested GO terms. Maximum degradation: -0.140 (Muscle cell differentiation).

## Figure
03B_gene_context_ablation.pdf/.png
- Panel a: Paired dot-line plot — v15d (blue circles), v11-A (purple squares), v11-B (pink triangles) per term.
- Panel b: Delta AUPRC horizontal bar chart. All bars negative = consistent degradation.

## Biological Interpretation
The consistent degradation across all terms indicates that gene context is not beneficial for isoform function prediction when labels are gene-level GO terms. The v11-A mean-context approach creates a shortcut: the model learns to predict based on gene membership rather than isoform-specific features. The v11-B deviation approach similarly disrupts the ESM-2 embedding space in ways that reduce discriminative power. Negative finding is a methodological contribution: explicitly encoding gene context worsens isoform-level prediction.
