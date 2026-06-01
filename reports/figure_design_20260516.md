# Figure Design Specification — DIFFUSE Nature Methods
**2026-05-16**

---

## Figure 1 — Method Overview & Type-A/B Classifier

**Layout**: 2-panel (a + b)

**Panel a** — Model Architecture
- Schematic: ESM-2 640d → Dense(256, BN, DO) → Dense(128, DO) → Dense(64) → sigmoid
- Input: 36,748 test isoforms (GTEx/BambuTx long-read muscle)
- Compare to LR (same input, 1-layer linear)
- Annotation: "v10-B (MLP)" vs "LR baseline"
- Style: clean block diagram, no PFN (removed), no CNN

**Panel b** — Type-A/B Classifier
- X-axis: sep_cosine (inter-centroid / intra-positive distance ratio)
- Y-axis: AUPRC gain (v10-B − LR, Δ)
- 13 points: Type-A (red triangle), Type-B (blue circle), sig=*** vs n.s. (marker fill)
- Threshold line at sep_cosine=0.060 (LOOCV-estimated)
- Correlation: r = −0.60 [95% CI −0.87, −0.07]
- Labels on key points: GO:0006096 (right), GO:0003774, GO:0007204
- Data: already in reports/sarcopenia_eval/pos_bias_scatter.png (pos_bias version; redo with sep_cosine)

**Caption draft**:
*"(a) v10-B architecture. ESM-2 protein embeddings (640d) are passed through a 3-layer MLP. (b) GO term type classification. sep_cosine, the cosine-space embedding separability ratio, predicts whether v10-B outperforms LR (Pearson r = −0.60; LOOCV accuracy 12/13). Type-B terms (sep_cosine < 0.060, blue) show consistent v10-B advantage; Type-A terms (red) are equally served by LR."*

---

## Figure 2 — 13 GO Term AUPRC Comparison (Main Result)

**Layout**: Grouped bar chart or heatmap-style dot plot

**Design choice**: Dot plot with CI bars (more Nature Methods style)
- Y-axis: GO terms (sorted by Δ, descending)
- X-axis: AUPRC (0.0–0.9)
- Two dots per GO term: v10-B (blue) and LR (orange)
- Error bars: 95% bootstrap CI
- Significance annotation: q<0.001 (***), q<0.05 (*), n.s.
- Background band: Type-A terms (gray band), Type-B terms (white)
- Right-side column: Δ values with arrow

**Alternatively**: Heatmap
- Rows: 13 GO terms, Cols: v10-B | LR | Δ | sig
- Color: AUPRC by sequential colormap, Δ by diverging (red=positive)

**Data**: Table 1 from results_draft_20260516.md (complete)

**Caption draft**:
*"AUPRC comparison across 13 sarcopenia-relevant GO terms. v10-B (blue) and logistic regression (LR, orange) use identical ESM-2 640-dimensional embeddings as input. Error bars represent 95% bootstrap CI (gene-block resampling, n=500). Type-B terms (10/11 q<0.05, BH correction) show consistent v10-B advantage; Type-A terms (GO:0003774, GO:0006096) show no significant difference, consistent with their high sep_cosine values (Figure 1b)."*

---

## Figure 3 — pos_bias Analysis

**Layout**: 3-panel (a + b + c)

**Panel a** — pos_bias bar chart per GO term
- X-axis: GO terms (13), sorted by pos_bias
- Y-axis: pos_bias value
- Color: Type-A (red), Type-B (blue)
- Reference line at pos_bias=1.0 (isoform discrimination threshold)
- Annotation: GO:0006941 = 1.902 (highest)

**Panel b** — coding vs all pos_bias comparison
- Scatter: x=pos_bias_all, y=pos_bias_coding (5 original GO terms)
- Diagonal line: y=x
- All points near diagonal → coding/non-coding not the driver
- Text: "Δ = −0.022 (coding-only vs all)"

**Panel c** — pos_bias vs Δ AUPRC (generated, reports/sarcopenia_eval/pos_bias_scatter.png)
- r = −0.20, confirming independent information

**Caption draft**:
*"(a) pos_bias per GO term. Values >1.0 indicate that v10-B discriminates among isoforms within positive genes more than it separates positive from negative genes globally. (b) pos_bias is unchanged when restricting to protein-coding isoforms only (98% of isoforms; Δ = −0.022, Δ<0.05 threshold), ruling out a coding/non-coding artefact. (c) pos_bias and AUPRC gain are weakly correlated (r = −0.20), confirming independent information."*

---

## Figure 4 — GABARAPL1 Isoform Switch (Main Case Study)

**Layout**: 3-panel (a + b + c)

**Panel a** — Protein domain cartoon
- GABARAPL1 canonical (ENST00000266458.10): full-length with GABARAP domain
- GABARAPL1 alternative (ENST00000541960.5): truncated / domain-absent
- Highlight: ATG8-family ubiquitin fold, N-terminal α-helix
- Annotation: v10-B score 0.989 vs 0.0004

**Panel b** — Score distribution
- Violin/box: all GABARAPL1 isoforms (n=2)
- Points: individual isoform scores
- Red dot: high (0.989), Blue dot: low (0.0004)
- Add PINK1 comparison: top=0.924, bot=0.046

**Panel c** — BambuTx validation (if available)
- If BambuTx has GABARAPL1 isoforms: show expression ratio N vs D
- Otherwise: literature-based validation (aging muscle expression decline)

**Caption draft**:
*"(a) GABARAPL1 isoform structure. The canonical isoform (ENST00000266458.10) retains the full ATG8/GABARAP ubiquitin-fold domain required for autophagosome membrane association. The alternative isoform (ENST00000541960.5) lacks key structural elements. (b) v10-B predicts an extreme functional difference (score ratio = 2222×). (c) [validation]."*

---

## Figure 5 — Sarcopenia Novel Isoform Network

**Layout**: Summary figure (could be Supplementary main text)

**Design**: 3-row summary
- Row 1: Autophagy cascade — GABARAPL1 isoform switch, ATG12 switch, PINK1 overlap
- Row 2: Mito org — PINK1, BNIP3 switch, SOD2 switch, NIPSNAP1 novel
- Row 3: Novel gene annotation gaps — NIPSNAP1, TAFAZZIN with GO term context

OR: **Scatter summary**
- X: GO term (grouped by pathway)
- Y: score_range (isoform switch magnitude)
- Size: ratio
- Color: sarcopenia_candidate (highlight) vs other
- Star: GABARAPL1

**Data**: reports/sarcopenia_novel/20260516_1343/isoform_switch.tsv

---

## Supplementary Figures

**Supp Fig 1** — Type-A/B LOOCV threshold stability
- LOOCV accuracy vs threshold (0.02–0.80 range)
- Shows stable plateau at threshold 0.045–0.080
- Data: /tmp/typeab_loocv.py output

**Supp Fig 2** — All 78 isoform-switch candidates (table)
- Sorted by score_range
- Color: sarcopenia_candidate

**Supp Fig 3** — All 75 novel gene candidates (table)

**Supp Fig 4** — Seed stability across 13 GO terms
- Box plots of 5-seed AUPRC per GO term
- Shows TOR signaling (higher variance) vs stable terms

**Supp Fig 5** — XGB/RF baseline comparison
- Bar chart: LR vs GBT vs RF vs v10-B macro AUPRC
- [XGB_TBD]

---

## Priority order for figure creation

1. Figure 2 (main result, Table 1 dot plot) — DATA READY
2. Figure 1b (Type-A/B classifier sep_cosine vs Δ) — DATA READY (need sep_cosine version of scatter)
3. Figure 3a (pos_bias bar chart) — DATA READY
4. Figure 4 (GABARAPL1) — needs domain cartoon (can use Pfam/InterPro API)
5. Figure 5 (novel isoform summary) — DATA READY
6. Figure 1a (architecture schematic) — manual drawing
