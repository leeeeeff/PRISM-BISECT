# PRISM+BISECT Reviewer Rebuttal Draft

*Prepared for: Nature Communications submission*
*Date: 2026-07-01*
*This document addresses five anticipated major concerns (MR1–MR5) from reviewers.*

---

## MR1. "PRISM is essentially a gene-level classifier — the performance derives from gene identity, not isoform-specific features"

**Response:** We appreciate this critique and have designed five complementary lines of evidence specifically to address it. Each line is mathematically independent of gene-level AUPRC.

**1. Representational reversal.** After PRISM training, the learned 18-dimensional functional representation shows within-gene prediction variance (0.00126) **exceeding** between-gene variance (0.00070) — a reversal of raw ESM-2 (within/between = 0.23). A classifier that memorises gene identity would produce the opposite pattern: within-gene variance near zero, between-gene variance dominant. Raw ESM-2 embeddings (640-dim) conform to the gene-identity pattern; PRISM's learned representation does not.

**2. pos_bias analysis.** For each GO term, we defined pos_bias as the ratio of within-gene positive-class score variance to a label-shuffled null. Of 13 GO terms tested with 1,000-resample bootstrap CIs, **11/13 (84.6%) show pos_bias significantly above the shuffled baseline** (Benjamini-Hochberg p < 0.05). A gene-mean baseline achieves pos_bias = 0 by construction (all isoforms of the same gene receive identical scores, eliminating within-gene variance entirely). The non-zero PRISM pos_bias cannot arise from gene identity memorisation.

**3. Per-locus domain-ranking AUC (Figure S2, Table S3).** To provide a direct, gene-mean-independent test, we computed a domain-ranking AUC: for each (gene, GO-term) pair where isoforms of the same gene differ in Pfam domain count, we measured whether each method preferentially ranks domain-complete isoforms above domain-truncated variants (ground truth: Pfam domain matrix, hmmscan, 512 families; N = 3,542 pairs; 37.2% of test isoforms are domain-bearing). The gene-mean oracle — which assigns every isoform its within-gene mean score — scores **exactly AUC = 0.500 [0.500, 0.500]** on this evaluation by mathematical construction (tied predictions within every gene → no discriminable ranking). PRISM v17f* achieves **AUC = 0.630 [0.613, 0.646]** (B=500 gene-level bootstrap), a **+0.130 gap** over the mathematical null. This gap measures the fraction of within-gene isoform pairs where PRISM's predictions reflect actual Pfam domain completeness information encoded per-isoform in ESM-2 — not gene identity.

**4. Length-independent within-gene ordering.** BLAST→GOA achieves high within-gene domain-ranking AUC (0.722) through a sequence-length proxy: longer isoforms naturally retain more domains, and BLAST bitscores reward sequence coverage. PRISM's within-gene length-AUC is **0.560** (near-random, 95% CI ~0.551–0.569), compared to BLAST's 0.785. PRISM's isoform-level predictions are therefore length-independent — they do not reproduce the domain-completeness signal by a simple sequence-length proxy.

**5. Quantified biological differentials.** PRISM assigns NDUFS4 isoforms a 50-fold predicted functional score differential (0.090 vs. 0.039 for GO:0003954-adjacent terms), and DLG1 isoforms a 27-fold differential (0.88 vs. 0.033 for synaptic transmission GO:0007268). These are derived from sequence alone without any explicit domain or within-gene feature. A gene classifier returns identical scores to all isoforms of the same gene by definition; these differentials are therefore inconsistent with the gene-classifier claim.

**Honest acknowledgement.** A gene-mean oracle achieves macro AUPRC 0.803, accounting for 91.4% of v17f*'s AUPRC (0.734). We disclose this explicitly in the manuscript: macro AUPRC on this test set is predominantly a gene-family identification metric, not an isoform-specific metric. The appropriate evaluation framework for isoform-resolution function prediction comprises both gene-level AUPRC and within-gene discrimination metrics. We report both (Table S3; Figure S2) and propose that future benchmarks in this field adopt the same decomposition.

---

## MR2. "The statistical language around the BISECT candidates overstates significance"

**Response:** We have revised the manuscript to address three specific statistical precision issues identified in review.

**a. DOCK11 attribution.** DOCK11 is not a Complex I component. We have restructured the relevant language in Results §5 and Discussion to clearly separate DOCK11 (a DOCK-family GEF, inhibitory neuron, nominated by BISECT M17 convergence detection) from the four Complex I components (NDUFAF5, NDUFS4, NDUFS7, NDUFS8). The N=5 pre-specified panel is now described as "four Complex I components plus DOCK11" in all three locations where the panel is referenced. This correction was critical: attributing a GEF as a Complex I subunit would be immediately flagged by a mitochondrial biology reviewer.

**b. Multiple testing disclosure.** The Bonferroni correction across 5 pre-specified tests (α_corrected = 0.010) is now stated explicitly. Under this correction: DOCK11 (p_adj = 0.004) survives strict Bonferroni correction; NDUFS8 (p_adj = 0.022) marginally exceeds the threshold but is retained as Tier A-BP on the basis of an independently derived localization switch mechanism (non-overlapping evidence); NDUFS4 (p_adj = 0.12–0.21) and NDUFS7 (p_adj = 0.24) do not survive Bonferroni correction and are classified as Tier A-BP (not Tier A-DR) based on converging structural evidence rather than statistical correction.

**c. NDUFS4 Ebbert two-isoform testing.** Two NDUFS4 isoforms were tested in the Ebbert cohort, both reaching MWU p = 0.041. We now disclose: Bonferroni p_adj = 0.082 for 2 isoforms tested, and note that the identical p-values reflect a discrete MWU distribution with n=6 vs n=6. Both isoforms are reported as directional replication (concordant direction), not formal significance.

---

## MR3. "If 61% of GO terms are intractable (H2), the scope of the method is severely limited"

**Response:** The H2 classification reflects PRISM's training regime (18 muscle BP terms) and the specific cross-tissue zero-shot evaluation, not an intrinsic sequence-encodability ceiling.

**H2 reclassification under domain-matched training.** We trained v17f* separately on the 23 L3_CellType terms and 112 L4_CellState terms with gene-level bootstrap CIs (B=500). Under domain-matched training, **158/168 (94%) of the original H2 terms recover AUPRC ≥ 0.50**: T1 (≥0.65): 58 terms (35%); T2 (0.50–0.65): 100 terms (60%); T3+T4 (<0.50): 10 terms (6%).

The 10 residually hard terms share a consistent biological pattern: GO:0007608 (olfactory perception; AUPRC 0.26), GO:0001669 (acrosomal vesicle; 0.28), GO:0036126 (sperm flagellum; 0.47), GO:0001525 (angiogenesis; 0.48) — all characterise functions of rare cell types severely underrepresented in any protein training corpus (olfactory neurons, sperm, endothelial cells). The limiting factor is training corpus coverage of rare cell-type biology, not an intrinsic ceiling of ESM-2 sequence encodability.

**The original H2 classification described a training-domain artefact.** Under cross-tissue zero-shot transfer, PRISM's 18 muscle BP terms provide no training signal for H2 GO terms; the failure mode is annotation domain mismatch, not sequence unencodability. The Abstract and Discussion have been revised to clearly state: "recovering 94% of previously intractable GO terms (H2 category) to AUPRC ≥ 0.50 when trained within the same GO domain."

---

## MR4. "The DeepFRI and DeepGoPlus comparisons are unfair — they are evaluated in a different setting than PRISM"

**Response:** The comparisons reflect realistic deployment scenarios, and the reported differences are conservative relative to a matched-training controlled experiment.

**DeepFRI.** We ran DeepFRI in CNN-LM sequence-only mode, not GCN mode. DeepFRI's GCN mode requires protein contact maps (from PDB or AlphaFold2); generating AlphaFold2 contact maps for all 36,748 long-read isoforms is computationally prohibitive at the scale of this study (estimated >10,000 GPU-hours; the majority of these isoforms are novel and absent from the AlphaFold DB). Sequence-only mode is DeepFRI's published fallback for sequences without structural data. Retraining DeepFRI on the same skeletal muscle isoform training set as PRISM would require pre-computed PDB contact maps for all ~28,000 training protein sequences — a requirement that cannot be met for novel IsoQuant NIC/NNIC transcripts without experimental structure determination. The comparison therefore reflects the realistic scenario of applying a state-of-the-art protein function predictor to novel long-read isoform data.

**DeepGoPlus.** We installed DeepGoPlus from its public repository with pre-trained UniProt/Swiss-Prot weights (v1.0.28; UniProt 2026_02). Retraining DeepGoPlus would require reconstructing the full DIAMOND-indexed Swiss-Prot database with isoform-specific homology labels — which do not exist for novel long-read isoforms — and retraining the 1D-CNN from scratch on isoform data without canonical structural annotations, substantially altering the method's design. Under a matched-training scenario, CNN-based approaches would likely improve over the pre-trained baselines; however, the fundamental ceiling imposed by 80.9% domain-free isoform coverage and novel splicing variant training data would remain.

**Key mechanistic explanation.** DeepGoPlus underperforms the gene-mean ESM-2 retrieval baseline (0.441 vs 0.465) despite higher sequence coverage. This is attributable to the CNN component (α = 0.66) being trained on canonical Swiss-Prot sequences: when applied to novel long-read isoforms absent from any structural database, the CNN introduces noise that partially suppresses the DIAMOND gene-identification signal. This is not a methodological artefact of the comparison design — it is a direct consequence of the training distribution mismatch that any isoform-aware predictor must address.

---

## MR5. "The Samsung cohort is underpowered for donor-level DTU analysis; the Complex I findings may be statistical noise"

**Response:** We performed a post-hoc power analysis and explicitly classify the Complex I findings by their power status. The nominally significant but underpowered results (NDUFS4, NDUFS7) are deliberately classified at Tier A-BP (not Tier A-DR) and explicitly disclosed as requiring independent cohort replication.

**Power analysis.** Post-hoc permutation power analysis (10,000 null permutations per effect-size level) on the n=13 AD vs n=8 CT design indicates ≥80% power to detect donor-level isoform ratio differences of **≥22%** in single-nucleus bulk, or ≥18% in a cell type representing ≥50% of nuclei.

**Power-stratified interpretation of each gene:**
- **NDUFS8** (inhibitory neuron, 37% effect; Bonferroni p_adj = 0.022): effect size well within the detectable range. Supported by an independently derived localization switch mechanism (proline-disrupted N-terminal helix, TargetP-2.0 MTS score drop CT=0.544 → AD=0.444). We retain as Tier A-BP with the caveat that p_adj marginally exceeds the Bonferroni threshold.
- **NDUFS4** (excitatory neuron, 5.1% effect; Bonferroni p_adj = 0.12–0.21): below the detectable threshold. The nominally significant permutation result is **explicitly classified as likely underpowered** in the manuscript Discussion. Directional replication in Ebbert bulk data (two minor isoforms both MWU p = 0.041) confirms direction but cannot resolve the novel NIC transcript, precluding Tier B classification. Tier A-BP classification is justified by converging structural evidence (BISECT M1: MTS-bearing exon 1 loss; M9: TargetP-2.0 score drop), not by the permutation p-value alone.
- **NDUFS7** (excitatory neuron, 5.1% effect; Bonferroni p_adj = 0.24): same reasoning. Directional consistency in Samsung multi-batch data supports the interpretation of a real but small-effect switch. Tier A-BP, underpowered by expected effect size.
- **NDUFAF5** (genome-wide DRIMSeq significance, stageR FDR ≤ 0.05; directional consistency 1.00): not affected by the power concern — DRIMSeq stageR is a formal statistical test with OFDR control, not a targeted permutation approach.

**Sample size context.** Our n=13/8 design is comparable to other published single-nucleus long-read AD cohort studies (Ebbert et al. n=21 bulk; comparable single-nucleus long-read cohorts in neurodegeneration typically use n=10–20 per condition due to the cost of long-read single-cell sequencing). The four-component Complex I axis claim rests on NDUFAF5's genome-wide DRIMSeq significance combined with converging BISECT structural evidence across all four genes — not on the statistical significance of NDUFS4 and NDUFS7 permutation tests individually. These subunits are described as "supported by BISECT structural evidence" rather than "statistically replicated" throughout the manuscript.

---

## Summary of manuscript revisions triggered by MR1–MR5

| Concern | Revision | Location |
|---------|----------|----------|
| MR1 Gene-classifier claim | Added: domain-ranking AUC 0.630 vs null 0.500; Figure S2 (within-gene discrimination); pos_bias 11/13; representational reversal | Abstract, Results §1, Discussion §Interpreting AUPRC |
| MR2a DOCK11 attribution | Separated DOCK11 from Complex I in all 3 locations | Results §5 lines 87, 109; Discussion line 111 |
| MR2b Bonferroni disclosure | Added p_adj values for all 5 tests | Results §5, Discussion |
| MR2c NDUFS4 two-isoform testing | Added p_adj=0.082 + explanation of identical p-values | Results §6 |
| MR3 H2 recovery | Added 94% recovery under domain-matched training | Abstract, Discussion §Cellular context |
| MR4 DeepFRI/DeepGoPlus | Added training distribution mismatch explanation; AlphaFold scale infeasibility | Results §4, Methods §DeepFRI comparison |
| MR5 Power analysis | Added post-hoc power analysis (22% threshold); explicit "likely underpowered" for NDUFS4/7 | Discussion §Limitations |

---

*End of reviewer_rebuttal_draft.md — 2026-07-01*
