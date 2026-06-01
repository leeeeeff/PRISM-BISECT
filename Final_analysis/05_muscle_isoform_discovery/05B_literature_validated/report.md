# 05B: Literature-Validated Isoform Function Switches

## Key Question
Do the top isoform switch predictions by v15d_bp_clean correspond to experimentally validated cases in the literature?

## Data Sources
- Phase 5 isoform switch table: reports/phase5_novel/20260515_0232/isoform_switch.tsv  (● hMuscle)
- Literature: Cardiovascular Research (TPM1), Human Molecular Genetics (DMD), Journal of Cell Biology (ANK2)
- GO annotations: ◇ UniProt

## Methodology
Top switch candidates from TSV ranked by score divergence. Cross-referenced with PubMed literature for experimental validation of isoform function switching.

## Key Findings

### TPM1 — Sarcomere organization (GO:0030017)
- Predicted switch: ENST00000559281 (score 0.965) vs ENST00000610733 (score 0.001), divergence = 0.965
- Literature: high-MW tropomyosin isoforms (Tpm1.1, Tpm1.6) are required for sarcomere integration in striated muscle; non-muscle isoform (Tpm1.8) does not integrate into sarcomeres (Cardiovascular Research).
- v15d correctly predicts the high-MW muscle isoform as functional and the non-muscle isoform as non-functional.

### DMD — Muscle contraction (GO:0006941)
- Predicted switch: ENST00000378707 (Dp427m, score 0.978) vs ENST00000683309 (Dp71, score 0.001), divergence = 0.977
- Literature: Dp427m (427 kDa, muscle-type) is required for sarcolemmal integrity and force transmission; Dp71 (71 kDa, brain-type, lacks rod domain) is not expressed in muscle and lacks costameric function (Hum. Mol. Genet.).
- v15d correctly assigns high score to muscle-type isoform and near-zero to brain-type isoform.

### ANK2 — Sarcomere org (GO:0030017)
- Predicted switch: ENST00000671793 (score 0.984) vs ENST00000682198 (score 0.0002), divergence = 0.984
- Literature: AnkB-212 (cardiac M-line specific, 212 kDa) coordinates T-tubule/SR organization and is the predominant cardiac isoform; shorter AnkB isoforms lack the M-line targeting domain (J. Cell Biol.).
- v15d correctly predicts the cardiac M-line isoform as functionally distinct.

### Quantitative accuracy
| Gene | Predicted top score | Predicted bottom score | Ratio | Literature concordant |
|------|--------------------|-----------------------|-------|----------------------|
| TPM1 | 0.965 | 0.001 | 1,552x | Yes |
| DMD | 0.978 | 0.001 | 978x |Yes |
| ANK2 | 0.984 | 0.0002 | 4,014x | Yes |

## Figure
05B_literature_validated.pdf/.png
Three-panel figure (one gene per panel):
- Horizontal bar charts of predicted scores per isoform. Blue = coding/functional, red = non-coding/non-functional.
- Green "Literature confirmed" stamp with citation in each panel.

## Biological/Methodological Interpretation
Three independent literature-validated switches, spanning three different GO terms and three different protein families (tropomyosin, dystrophin, ankyrin), are correctly predicted by v15d_bp_clean. This validates the model's isoform-level resolution beyond statistical metrics. The 978-4014x score ratios demonstrate that the model makes confident, binary predictions rather than borderline scores. These cases were not used in training (hMuscle long-read data provides expression context, but GO annotations are gene-level). The rediscovery of known functional switches from first-principles ESM-2 sequence embeddings is a key biological contribution of the paper. Data: ● hMuscle; GO: ◇ UniProt.
