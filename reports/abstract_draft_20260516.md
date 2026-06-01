# Abstract Draft — PRISM
**2026-05-17 (updated) | Target: Nature Methods / Nature Machine Intelligence**
**Word limit: ~200 words (Nature Methods)**

---

## Version 1 (full-length, ~210 words)

Understanding how alternative splicing generates functionally distinct protein isoforms from a single gene
is a central challenge in cell biology. We present PRISM, a deep learning framework that predicts
isoform-level biological process functions from ESM-2 protein language model embeddings, enabling
isoform-resolution functional annotation of long-read single-cell RNA sequencing data. Applied to
36,748 isoforms from 12,709 human genes profiled in skeletal muscle by long-read single-cell RNA
sequencing, PRISM v10-B — a three-layer MLP trained on gene-level GO annotations from
ESM-2 embeddings — achieves a mean AUPRC of 0.685 across 11 sarcopenia-relevant GO terms,
compared with 0.363 for logistic regression using the same input features (+88.7%; 10/11 terms
q < 0.05, Benjamini-Hochberg correction). Negative controls confirm that within-gene isoform
discrimination exceeds a shuffled-label noise floor in all tested GO terms (shuffled pos_bias ≈ 0.24;
v10-B max = 1.902 for muscle contraction).
A cosine-space separability metric (sep_cosine) post-hoc characterises which GO terms benefit from
non-linear modelling, with leave-one-out cross-validation accuracy of 100% across 13 GO terms.
Applying PRISM to isoform-switch analysis — with symmetric NMD screening of both isoforms in each
candidate pair — reveals extreme functional divergence in DMD (muscle contraction score ratio 1,263×),
PINK1 (independently validated in autophagy and mitochondrion organisation), and NDUFAF6 as a novel
mitochondrial complex I prediction. PRISM recovers novel functional gene associations, including
NIPSNAP1 and TAFAZZIN for mitochondrion organisation, demonstrating biological discovery potential
beyond current GO database coverage.

---

## Version 2 (condensed, ~180 words — recommended)

Alternative splicing generates functionally distinct protein isoforms, yet most function-prediction
methods operate at the gene level. Here we present PRISM, a deep learning framework that predicts
isoform-level biological process functions directly from protein language model embeddings. Applied
to 36,748 isoforms from 12,709 human skeletal muscle genes profiled by long-read single-cell RNA
sequencing, PRISM achieves mean AUPRC of 0.685 across 11 sarcopenia-relevant GO terms versus
0.363 for logistic regression (+88.7%; 10/11 terms q < 0.05 after Benjamini-Hochberg correction).
A cosine-space embedding separability metric post-hoc characterises which GO terms benefit from
non-linear modelling (leave-one-out accuracy 100%). Within-gene isoform discrimination confirmed
by negative controls (gene-mean null = 0; shuffled-label null ≈ 0.24; v10-B pos_bias up to 1.902
for muscle contraction). Isoform-switch analysis with symmetric NMD screening reveals extreme
functional divergence in DMD (ratio 1,263×) and cross-GO-term consistency in PINK1 mitophagy
isoforms. PRISM further identifies NIPSNAP1 and TAFAZZIN as annotation-gap genes for
mitochondrion organisation, extending functional annotation beyond current GO database coverage.

---

## Version 3 (Nature Methods brief, ~150 words)

Most gene function prediction methods cannot distinguish alternatively spliced isoforms.
We present PRISM, which applies a three-layer deep neural network to ESM-2 protein
embeddings to predict isoform-level GO Biological Process functions in long-read single-cell
RNA sequencing data. In skeletal muscle isoform data (36,748 isoforms, 12,709 genes), PRISM
achieves +88.7% AUPRC gain over logistic regression across sarcopenia-relevant GO terms
(mean 0.685 vs 0.363; 10/11 terms significant after FDR correction). A cosine-space
embedding separability metric characterises which GO terms benefit from isoform-resolution
prediction (leave-one-out accuracy 100%). Symmetric NMD screening of isoform-switch candidates
validates biological plausibility: DMD (ratio 1,263×) and PINK1 (autophagy and mitochondrion
organisation) recapitulate known biology, while NDUFAF6 and annotation-gap genes NIPSNAP1 and
TAFAZZIN represent testable novel predictions for sarcopenia-relevant pathways.

---

## Key numbers (for checking — 2026-05-28 업데이트):
- 36,748 isoforms, 12,709 genes; training = gene-level GO labels on ESM-2 embeddings of skeletal muscle isoforms (no SwissProt)
- Production model (PRISM v15d_bp_clean): macro AUPRC **0.7022** (18 GO terms), **0.6935** (13 sarcopenia GO terms)
- LR comparison basis (13 GO): 0.6935 vs 0.363 = +91% (use this in abstract; replaces old 0.685/+88.7%)
- Significance: 10/11 q<0.05 (BH correction)
- sep_cosine LOOCV: 13/13 = 100% (post-hoc, NOT prospective)
- pos_bias: gene-mean=0, shuffled_floor≈0.24, random_ceiling≈0.92, v10-B max=1.902 (GO:0006941); macro=1.006 (13 terms, 5-seed ensemble)
- NMD screening: 23/126 excluded (18.3%) — symmetric (both isoforms)
- Switch cases: DMD (1,263×), PINK1 (autophagy + mito org), NDUFAF6 (2,000×, novel)
- NIPSNAP1 score: 0.819, TAFAZZIN score: 0.934

## Key fixes from 2026-05-16 → 2026-05-17:
- GABARAPL1 (2,222×) → excluded (NMD_RISK), replaced by DMD (1,263×) ✅
- BNIP3 → excluded (both isoforms 5'-partial) ✅
- sep_cosine "prospectively" → "post-hoc characterises" ✅
- pos_bias > 1.0 → pos_bias negative controls cited explicitly ✅
- NMD screening: "both isoforms screened" (symmetric) ✅

## TODO for final abstract:
- [ ] Choose between Version 1/2/3 based on target journal
- [ ] Count words in each version
- [ ] Confirm NIPSNAP1/TAFAZZIN isoform scores are correct (0.819, 0.934)
- [x] RF baseline: "4.7× over ESM-LR and ESM-RF" — can add to Version 2 if space allows

---

## Version 4 (updated 2026-05-25 — includes cross-tissue + AD switching + BISECT batch; ~200 words)

Most isoform function-prediction methods operate at the gene level. Here we present PRISM, a
deep learning framework that predicts isoform-level GO Biological Process functions from ESM-2
protein language model embeddings. Applied to 36,748 isoforms from 12,709 human skeletal muscle
genes profiled by long-read single-cell RNA sequencing, PRISM achieves macro AUPRC of 0.7022
across 18 GO Biological Process terms (0.6935 across 13 sarcopenia-relevant terms versus 0.363
for logistic regression, +91%; 10/11 Type-B terms q < 0.05, Benjamini-Hochberg correction).
A cosine-space separability metric identifies GO terms suited for non-linear isoform-resolution
modelling (leave-one-out accuracy 100%; pos_bias maximum 1.902 for muscle contraction;
shuffled-label baseline 0.24). Applied zero-shot to 63,994 human prefrontal cortex isoforms
(Samsung Alzheimer's disease cohort) without retraining, PRISM achieves macro AUPRC of 0.600, demonstrating cross-tissue generalization. Integration with
Dirichlet-multinomial differential transcript usage testing identifies three AD-specific isoform
switches exclusive to single cell types: a bidirectional KIF21B switch in excitatory neurons
(p = 9.3×10⁻⁸), NDUFS4 locus hijacking by a novel 379-aa protein (p = 3.6×10⁻⁶), and DLG1
isoform replacement in oligodendrocyte precursor cells (p = 9.0×10⁻¹⁰). BISECT domain-change
analysis of 50 additional AD candidates identifies 23 domain-altering events across six cell
types (Table S4), including LAR-RPTP fibronectin domain loss in PTPRF and PTPRS and WD40
β-propeller redistribution between IFT122 and KIF21B in excitatory neurons.
