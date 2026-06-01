# Section 08C: Known Limitations and Future Directions

## Limitation 1: GO Annotation Paradigm Mismatch

**Issue**: GO annotations describe gene-level function. The model is trained with isoform-to-GO mapping that inherits this gene-level structure.

**Consequence**: 
- Model achieves AUPRC ~0.97 on positive-class genes' coding isoforms (score saturation)
- Cannot distinguish WHICH isoform of a positive gene performs the function
- This is the theoretical ceiling of the current approach

**Evidence**: AlphaFold H1 failure (score range ≤0.015 within positive genes), pos_bias saturation in Case 1 GO terms.

**Future direction**: Isoform-level functional annotation datasets (e.g., isoform-specific knockdown experiments, mass spectrometry isoform identification with activity assays) would enable a fundamentally different training target.

---

## Limitation 2: SwissProt Dependency for Type-A Terms

**Issue**: GO:0006096 (Glycolysis) performance is highly seed-dependent (std=0.121). 87.6% of GO:0006096 positive annotations derive from SwissProt.

**Consequence**: 
- Removing SwissProt completely collapses GO:0006096 AUPRC (AP15)
- The model's performance on highly curated, historically annotated terms depends on Swiss-Prot coverage

**Evidence**: Bootstrap CI seed 456: AUPRC=0.537 < LR for GO:0006096.

**Future direction**: Incorporate broader annotation sources (UniProt TrEMBL with evidence filtering, PhosphoSitePlus, isoform-specific databases).

---

## Limitation 3: Novel Isoform OOD Generalization Gap

**Issue**: Novel isoforms (NNIC/NIC, first detected in the long-read dataset) show substantially lower prediction AUPRC.

**Quantification**:
- All brain isoforms: Macro AUPRC 0.5998
- Novel brain isoforms only: Macro AUPRC 0.3217 (Δ=-0.278)

**Consequence**: The model's predictions for completely novel splice variants are less reliable, which is precisely where the most biologically interesting predictions would be.

**Evidence**: 34 cross-GO reversal genes detected in novel brain isoforms — these predictions have higher uncertainty.

**Future direction**: 
1. Training on a mix of known and novel isoforms from multiple long-read datasets
2. Uncertainty quantification (Monte Carlo Dropout or conformal prediction) for novel isoforms
3. Transfer learning from pre-trained splicing models (SpliceAI, Pangolin)

---

## Limitation 4: Family Diversity Is a Training-Time Problem

**Issue**: GO terms with high protein family diversity (e.g., Neuron differentiation: 40+ non-homologous gene families) produce scattered ESM-2 embeddings that cannot form a coherent decision boundary.

**Quantification**: pc1_var_ratio for Case 3 terms = 0.272 (lowest), ΔAUPRC = +0.391 (highest advantage, but starting from LR=0.291).

**Consequence**: Even with brain-specific training data, if the GO term itself spans multiple unrelated protein families, the upper bound AUPRC would remain limited.

**Evidence**: Family diversity analysis shows no improvement from changing the test set alone (06D finding).

**Future direction**: 
1. GO term decomposition (split broad terms into mechanistic sub-terms)
2. Hierarchical prediction (predict functional module first, then specific GO term)
3. Multi-label learning with GO term hierarchy (information content weighting)

---

## Limitation 5: Cross-Tissue Performance Degradation

**Issue**: Macro AUPRC drops 0.102 from muscle to brain (0.702 → 0.600).

**Cause (analyzed)**:
- Muscle-trained ESM-2 embeddings reflect muscle-specific sequence contexts
- Brain-enriched GO terms (Synaptic transmission, Neuron differentiation) have different embedding geometries in brain tissue

**Consequence**: Tissue-specific isoform predictions may require tissue-specific training.

**Future direction**: Fine-tuning v15d on brain tissue long-read data (keeping GO label matrix consistent). This requires careful train/test split design to avoid AD/CT label leakage.

---

## Limitation 6: AD Biological Findings Are Associative

**Issue**: The 3 AD isoform switches (KIF21B, NDUFS4, DLG1) are detected in a single cohort (Samsung AD IsoQuant), validated by statistical test (Dirichlet-multinomial) but not by experimental validation.

**Consequence**: 
- Findings are correlative, not causal
- Single-cohort discovery (n=7 AD, n=17 CT excitatory neuron samples approximation)
- tr73243 (NDUFS4, 379aa) function is completely unknown

**Future direction**: 
1. Independent replication in a second AD long-read cohort
2. Functional characterization of tr73243 (what does the novel protein do?)
3. CRISPR isoform-specific editing to establish causality
4. Proteomics to confirm protein-level presence of novel isoforms

---

## Summary

| Limitation | Severity | Addressable? | Priority |
|-----------|----------|-------------|---------|
| GO paradigm mismatch | Fundamental | Requires new data | Long-term |
| SwissProt dependency | Moderate | Augment annotations | Short-term |
| Novel isoform OOD gap | High for discovery | Better training data | Medium-term |
| Family diversity ceiling | Fundamental | Better GO structure | Long-term |
| Cross-tissue degradation | Moderate | Tissue-specific FT | Short-term |
| AD findings: associative | High | Independent cohort | Short-term |
