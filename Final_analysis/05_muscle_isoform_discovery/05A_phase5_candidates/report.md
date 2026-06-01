# 05A: Phase 5 Novel Isoform Switch Candidates

## Key Question
How many isoform function switches does v15d_bp_clean predict in hMuscle long-read data, and how do they distribute across GO terms?

## Data Sources
- Phase 5 summary: reports/phase5_novel/20260515_0232/phase5_summary.json  (● hMuscle)
- Isoform switch table: reports/phase5_novel/20260515_0232/isoform_switch.tsv  (● hMuscle)

## Methodology
Novel isoform threshold: predicted score > 0.5 (n=100 candidates across 5 GO terms, 20 per term).
Isoform switch threshold: within-gene score range > 0.3 (n=48 switches).
Score divergence = max_isoform_score - min_isoform_score per gene.

## Key Findings

### Summary
- Total novel candidates: 100 (20 per GO term)
- Total isoform switches: 48
- Switch threshold: within-gene divergence > 0.3

### Per-GO-term breakdown
| GO Term | Novel candidates | Isoform switches |
|---------|-----------------|-----------------|
| Glycolysis (GO:0006096) | 20 | 1 |
| Motor activity (GO:0003774) | 20 | 7 |
| Ca2+ signaling (GO:0007204) | 20 | 7 |
| Sarcomere org (GO:0030017) | 20 | 20 |
| Muscle contraction (GO:0006941) | 20 | 13 |

Sarcomere organization has the highest switch count (20/20 candidates show switches), reflecting high isoform diversity in structural muscle proteins.

### Score divergence distribution (48 switches)
- Median divergence: ~0.79
- High divergence (>0.5): 41/48 (85%)
- Extreme divergence (>0.9): ~25/48 (from TSV analysis)
- Top example: ANK2 GO:0006941 divergence = 0.941, ratio = 940,952x

### Top switches by divergence
1. KIF2A (GO:0003774): divergence 0.994, ratio 415x
2. KIF20B (GO:0003774): divergence 0.992, ratio 274x
3. DTNA (GO:0006941): divergence 0.991, ratio 746,690x
4. KIF21A (GO:0003774): divergence 0.988, ratio 214x
5. ANK2 (GO:0030017): divergence 0.984, ratio 4,014x

## Figure
05A_phase5_candidates.pdf/.png
- Panel a: Grouped bar chart — novel candidates (blue) and isoform switches (green) per GO term. Value labels on switch bars.
- Panel b: Histogram of within-gene score divergence for all 48 switches. Green = high divergence (>0.5), gray = moderate. Dashed line at 0.5 threshold.

## Biological Interpretation
The Sarcomere organization term has 100% switch rate (all 20 candidates show isoform switches), consistent with the known biology that sarcomere proteins (titin, tropomyosin, myosin binding proteins) undergo extensive alternative splicing to tune muscle fiber mechanics. The Motor activity term has strong switches in KIF2A, KIF20B, KIF21A — kinesin family members where isoform diversity regulates microtubule-based transport direction and cargo specificity. High score divergence (>0.5 in 85% of cases) indicates these are not borderline predictions but strong binary isoform switches. Data: ● hMuscle long-read biopsy sequencing.
