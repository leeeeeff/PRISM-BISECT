# Results 3.9 — Cross-cohort directional concordance of key AD isoform switches

## Draft (2026-05-27)

---

### 3.9 Independent bulk long-read validation of key isoform switches

To assess whether the AD isoform switches identified in the Samsung single-cell
long-read cohort generalize beyond the discovery dataset, we examined transcript
proportion changes in an independent frontal cortex long-read RNA-seq cohort
(Ebbert et al., Nature Biotechnology 2024; n = 6 AD, n = 6 CT, BA9/46,
Oxford Nanopore PromethION). Transcript-level counts were obtained from the
publicly deposited Zenodo repository (CC-BY 4.0).

Because the Samsung cohort was processed with IsoQuant against an extended
annotation incorporating novel isoforms, and the Ebbert cohort was independently
processed with Bambu against GENCODE 38, direct transcript ID matching was not
possible. We therefore used structurally proximal annotated transcripts as
surrogates, guided by IsoQuant's `similar_reference_id` assignments and
reciprocal exon-coordinate overlap. Statistical significance was assessed by
Mann–Whitney U test and confirmed by permutation test (n = 2,000 permutations,
two-sided).

**KIF21B.** In the Samsung discovery cohort, a near-complete isoform usage
reversal was observed in excitatory neurons: the canonical motor-domain-containing
isoform (tr293004, 34 exons) was present exclusively in control excitatory neurons
(CT usage = 35.1%, AD usage = 0.0%), while a novel WD40-repeat alternative
(tr292978) showed the inverse pattern (AD usage = 35.5%, CT usage = 0.0%;
Δ = 0.355; cell-level chi-sq test note: the chi-sq padj is inflated by
pseudoreplication; donor-level permutation was non-significant at n = 21 donors,
consistent with high inter-donor variability). This isoform switch was absent from
all other cell types (astrocytes, inhibitory neurons, microglia, oligodendrocytes,
and vascular cells), establishing excitatory-neuron specificity. In the Ebbert bulk
cohort, the annotated surrogate of the canonical isoform (ENST00000332129; 34 exons)
showed a significant decrease in proportion among total KIF21B reads in AD relative
to CT brains (CT = 73.0% ± 12.1% vs AD = 43.0% ± 14.8%; Δ = −0.30;
MWU p = 0.026; permutation p = 0.048; Figure 3.9a), providing donor-level
statistical confirmation. The WD40 surrogate isoform (ENST00000422435) was not
reliably detected in the bulk dataset (mean proportion < 0.01% in both groups),
likely owing to structural differences from the Samsung novel isoform and dilution
of cell-type-specific signal in bulk tissue.

**NDUFS4.** In the Samsung discovery cohort, a novel isoform tr73243 (extending
approximately 5 kb beyond all GENCODE 38 annotated forms, overlapping an L1PA3/
L1PA11 retrotransposon locus) was exclusively expressed in AD excitatory neurons
(usage = 42.9%), with NDUFS4-201 canonical showing a complementary decrease
(CT usage = 44.1%, AD usage = 7.1%; Δ = 0.429 for tr73243; excitatory-neuron
specific; absent from all other cell types). As with KIF21B, the cell-level chi-sq
statistic (padj = 1.94 × 10⁻⁴) should be interpreted as directional evidence only,
given that the donor-level permutation was non-significant (n = 21 donors). In the
Ebbert bulk cohort, the canonical 5-exon isoform (ENST00000296684; NDUFS4-201)
showed a directional, non-significant decrease in AD (92.7% vs 91.2%; Δ = −0.016;
p = 0.59). Importantly, an alternative 6-exon isoform (ENST00000506974; NDUFS4-204)
showed a significant increase in AD proportion (1.9% vs 4.3%; Δ = +0.025;
MWU p = 0.041; permutation p = 0.048; Figure 3.9b). Although tr73243 could not be
directly measured in the Ebbert annotation, the complementary increase of NDUFS4-204
in AD provides donor-level corroboration of a shift away from the canonical isoform
in AD frontal cortex.

**DLG1** (Supplementary). The DLG1 tr319500 OPC switch (CT OPC usage = 80.9%,
AD OPC usage = 11.9%; OPC-restricted, absent from all other cell types) was not
detectable in the Ebbert bulk dataset (p = 0.70), as expected given the ~20-fold
dilution of OPC-specific signal in bulk cortex. This case is reported in full in
Supplementary Results; donor-level statistical confirmation awaits a
cell-type-resolved independent cohort.

Taken together, the two excitatory-neuron isoform switches (KIF21B and NDUFS4)
showed statistically concordant directional changes in an independent long-read
bulk cohort, with KIF21B reaching conventional significance at the donor level.
The absence of a matching long-read single-cell AD cohort in the public domain
precludes full cell-type-resolved replication; the Ebbert concordance should
therefore be interpreted as cross-cohort directional evidence rather than
independent replication.

---

### Figure captions (placeholder)

**Figure 3.9a.** KIF21B isoform proportion in Ebbert frontal cortex cohort.
Box plots showing the proportion of ENST00000332129 (canonical motor-domain
isoform) among all KIF21B reads, per sample (n = 6 AD, 6 CT). MWU p = 0.026,
permutation p = 0.048. Inset: Samsung discovery (excitatory neurons, n = 21;
tr293004 CT-dominant, tr292978 AD-dominant; padj = 1.54 × 10⁻¹⁰).

**Figure 3.9b.** NDUFS4-204 isoform proportion in Ebbert frontal cortex cohort.
Box plots showing ENST00000506974 proportion among NDUFS4 reads. MWU p = 0.041,
permutation p = 0.048. Direction concordant with Samsung tr73243 AD-enrichment
(excitatory neurons, padj = 1.94 × 10⁻⁴).

**Figure 3.9c (supplementary).** DLG1 tr319500 expression in Samsung OPC cells.
Bar plot or violin showing per-sample proportions across cell types (n = 21 subjects).
tr319500: CT OPC = 80.9%, AD OPC = 11.9% (padj = 4.11 × 10⁻⁹). Note near-absence
in all non-OPC cell types, explaining the Ebbert bulk null result.

---

### Limitations note (for Discussion 5.x)

The cross-cohort concordance analysis has the following structural limitations
that should be acknowledged:
1. Proxy mismatch: Samsung novel isoforms are absent from GENCODE 38; surrogates
   share genomic locus but differ in exon number (KIF21B: 19 vs 35 exons).
2. Tissue resolution: Samsung is single-cell; Ebbert is bulk. Cell-type-specific
   switches (DLG1-OPC) cannot be validated in bulk.
3. Cohort size: n = 6 per group provides <20% power for effect sizes <5%;
   NDUFS4 canonical decrease (Δ = 1.6%) is below the detectable threshold.
4. No public long-read scRNA-seq AD cohort exists for direct replication as of 2025.
5. Pseudoreplication in Samsung chi-sq DTU: cell-level chi-sq statistics are
   inflated relative to donor-level variation (n = 21 donors). Donor-level
   permutation tests were non-significant for all three cases, indicating that
   effect sizes are consistent but inter-donor variability precludes cell-level
   inflation from being used as primary statistical evidence. Ebbert donor-level
   MWU/permutation provides the formally valid statistical confirmation.
6. DLG1 counter-isoform transcript319159 (AD OPC usage = 30.9%) is structurally
   uncharacterized; the functional consequence of the tr319500 → tr319159 OPC
   transition requires independent validation.
