# 04A: Within-Gene Discrimination (pos_bias)

## Key Question
Does v15d_bp_clean discriminate between isoforms within the same gene, or does it simply classify genes?

## Data Sources
- pos_bias raw bias scores: reports/xgb_baseline/v10b_bias_score_20260518_1130.json  (● hMuscle)
- pos_bias coding validation: reports/pos_bias_coding/pos_bias_coding_20260516_1252.json  (● hMuscle)
- Within-gene pairwise AUROC: reports/within_gene/pairwise_20260515_1624.json  (● hMuscle)

## Methodology
pos_bias = variance of predicted scores among isoforms of positive-class genes / variance among isoforms of negative-class genes. Values >1.0 indicate that positive-class gene isoforms show greater within-gene prediction variance than expected, demonstrating genuine isoform-level discrimination.

Within-gene pairwise AUROC: for each gene with multiple isoforms, compute whether higher GO-annotated isoforms receive higher scores. Cross-validated with 5-fold CV.

## Key Findings

### pos_bias per GO term (v15d / v10b equivalent)
| GO Term | pos_bias | Interpretation |
|---------|---------|---------------|
| Muscle contraction (GO:0006941) | 1.902 | Strong isoform discrimination |
| Motor activity (GO:0003774) | 1.435 | Strong |
| Skeletal muscle dev (GO:0007519) | 1.514 | Strong (from sarcopenia eval) |
| Sarcomere org (GO:0030017) | 1.176 | Moderate |
| Muscle organ dev (GO:0007517) | 1.093 | Moderate |
| Muscle cell diff (GO:0042692) | 1.035 | Borderline |
| Ca2+ signaling (GO:0007204) | 0.475 | Below threshold |
| Glycolysis (GO:0006096) | 0.663 | Below threshold |

### Within-gene pairwise AUROC (from pairwise_20260515_1624.json)
- Full model AUROC: 0.780 (95% CI: 0.774-0.787, p=0.0)
- ESM-2 only AUROC: 0.736 (95% CI: 0.728-0.743, p=0.0)
- vs. random (0.5): highly significant

### pos_bias: coding vs non-coding validation
- Coding isoforms: n=36,002 (97.97%)
- Non-coding isoforms: n=746 (2.03%)
- pos_bias_all vs pos_bias_coding: delta = -0.022 (negligible)
- Verdict: pos_bias is NOT driven by coding/non-coding status

## Figure
04A_within_gene_discrimination.pdf/.png
- Panel a: Bar chart of pos_bias values per GO term. Dashed reference at 1.0. Blue = v15d (pos_bias >= 1.0), gray = below threshold.
- Panel b: Conceptual schematic showing Gene A (positive, high within-gene variance) vs Gene B (negative, all near zero).

## Biological Interpretation
pos_bias > 1.0 for 5/8 Type-B GO terms indicates that the model captures genuine isoform-level functional differences within the same gene — not merely classifying genes. The strongest discrimination occurs for muscle contraction (1.902) and motor activity (1.435), biological processes where isoform diversity is functionally critical (e.g., fast vs slow myosin isoforms, cytoplasmic vs kinesin motors). The within-gene pairwise AUROC of 0.780 confirms this finding quantitatively. Importantly, the pos_bias signal is independent of coding/non-coding isoform status (delta = -0.022), ruling out the trivial explanation that the model detects non-coding isoforms by their shorter ORF characteristics.
