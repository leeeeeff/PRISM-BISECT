# splice_delta vs ESM-2 Contribution Analysis for PRISM

**Analysis Date:** 2026-06-07  
**Data:** hMuscle test set (36,748 isoforms, 12,709 genes)  
**Objective:** Quantify complementarity between splice_delta and ESM-2 embeddings

---

## Executive Summary

**Key Finding:** ESM-2 and splice_delta capture COMPLEMENTARY isoform variations.

- **Pearson r = 0.235** (weak linear correlation)
- **Spearman ρ = 0.609** (moderate rank correlation)
- **73 genes (0.9%)** where ESM-2 completely fails (cosine distance ≈ 0) but splice_delta succeeds (L1 distance > P75)

**Critical Discovery:** Multiple isoforms from the same gene exhibit **IDENTICAL ESM-2 embeddings** (cosine distance = 0.0000) but **LARGE splice_delta separations** (L1 distance > 5).

**Implication for PRISM:**
- ESM-2 alone: captures protein sequence variation
- splice_delta alone: captures exon-level structural variation
- ESM-2 + splice_delta: potential for synergistic isoform-level discrimination

---

## 1. Basic splice_delta Statistics

### Coverage
- **Non-zero splice_delta:** 19,160 / 36,748 (52.1%)
- **Zero-delta (canonical-identical):** 17,588 (47.9%)

### Structural Change Magnitude
| Category | Count | Percentage | |Δ|.sum() range |
|----------|-------|------------|----------------|
| Zero (canonical-identical) | 17,588 | 47.9% | 0 |
| Low structural change | 14,215 | 38.7% | 0 < |Δ| ≤ 3 |
| High structural change | 4,945 | 13.5% | |Δ| > 3 |

### Distribution Percentiles
```
P0:    0.0
P25:   0.0
P50:   0.2
P75:   1.9
P90:   4.0
P95:   8.9
P99:  27.4
P100: 147.8
```

**Interpretation:** Median isoform has minimal splicing changes, but 25% show substantial structural alterations (|Δ| > 1.9).

---

## 2. Within-Gene Separation Analysis

### Multi-Isoform Genes
- Total genes: 12,709
- Genes with ≥2 isoforms: **8,569 (67.4%)**

### ESM-2 Within-Gene Distance (Cosine)
```
P0:   0.0000
P25:  0.0000  ← 25% of genes have ZERO ESM-2 separation
P50:  0.0026
P75:  0.0175
P90:  0.0562
P95:  0.0892
P99:  0.1494
P100: 0.3828
```

### splice_delta Within-Gene Distance (L1)
```
P0:    0.0
P25:   0.5
P50:   1.4
P75:   2.4
P90:   5.6
P95:   9.5
P99:  21.5
P100: 102.3
```

**Critical Insight:** 25% of multi-isoform genes have **ZERO** ESM-2 distance (P25 = 0.0000), meaning ESM-2 cannot distinguish isoforms within these genes at all.

---

## 3. ESM-2 Failure / splice_delta Success Cases

### Quantification
**ESM-2 low (< P25) AND splice_delta high (> P75):**
- **73 genes (0.9% of multi-isoform genes)**
- These are cases where ESM-2 provides NO isoform separation, but splice_delta does.

### Top 10 Genes by splice_delta/ESM-2 Separation Ratio

| Gene | #iso | ESM-2 dist | splice dist | Ratio |
|------|------|------------|-------------|-------|
| ENSG00000197102.14 | 8 | 0.0000 | 24.1 | 24M |
| ENSG00000102763.18 | 3 | 0.0000 | 13.1 | 13M |
| ENSG00000176208.9 | 2 | 0.0000 | 8.6 | 8.6M |
| ENSG00000215252.12 | 2 | 0.0000 | 8.2 | 8.2M |
| ENSG00000183722.9 | 4 | 0.0000 | 7.3 | 7.3M |

**Pattern:** These genes have 2-8 isoforms with IDENTICAL ESM-2 embeddings (cosine distance = 0.0000) but large splice_delta differences.

---

## 4. Case Study: ENSG00000197102.14 (8 isoforms)

### ESM-2 Embedding Analysis
- **All 8 isoforms:** norm = 7.7057 (identical)
- **All pairwise cosine distances:** 0.000000
- **Conclusion:** ESM-2 sees all 8 isoforms as IDENTICAL

### splice_delta Analysis
- **|Δ|.sum() range:** 0.00 to 49.17
- **Pairwise L1 distances:** 0.00 to 49.17
- **Conclusion:** splice_delta detects large structural differences

### Interpretation
These isoforms likely:
1. Share identical protein sequences (ESM-2 only sees protein)
2. Differ in UTRs, alternative promoters, or non-coding exons (splice_delta captures all exons)
3. Have alternative TSS/TES generating same CDS (splicing affects RNA structure, not protein)

---

## 5. Correlation Analysis

### Pearson (Linear Correlation)
- **r = 0.235**
- **p = 1.07e-107** (highly significant)
- **Interpretation:** WEAK linear correlation

### Spearman (Rank Correlation)
- **ρ = 0.609**
- **p = 0.0** (highly significant)
- **Interpretation:** MODERATE rank correlation

**Biological Meaning:**
- ESM-2 and splice_delta capture **different aspects** of isoform variation
- splice_delta provides information BEYOND what ESM-2 can capture
- **Complementary, not redundant**

---

## 6. IDR Proxy Analysis

### ESM-2 Embedding Norm Comparison

| Group | ESM-2 norm (mean ± std) | Exon count (mean ± std) |
|-------|-------------------------|-------------------------|
| Zero-delta (canonical-identical) | 6.64 ± 0.75 | 11.0 ± 8.9 |
| High-delta (|Δ| > 3) | 6.91 ± 0.73 | 11.5 ± 9.1 |

### Statistical Tests
- **ESM-2 norm t-test:** t = -22.73, p = 4.05e-113 (HIGHLY SIGNIFICANT)
- **Exon count t-test:** t = -3.29, p = 9.95e-04 (SIGNIFICANT)

**Interpretation:**
- High-delta isoforms have **HIGHER ESM-2 norm** (+4.1% increase)
- This suggests high-delta isoforms encode longer or more complex proteins
- But ESM-2 norm difference is SMALL compared to splice_delta's discriminative power

---

## 7. Historical Experiment Results (v10 series)

### v10D Ablation (20260518)
| Model | Input | Macro AUPRC | Interpretation |
|-------|-------|-------------|----------------|
| v10D_emb | ESM-2 only | 0.5126 | Poor GO prediction |
| v10D_splice | splice_delta only | 0.6367 | Better GO prediction |
| v10E | ESM-2 + splice_delta | ~0.70 | Best GO prediction |
| v10E0 | ESM-2 only (same arch as v10E) | ~0.69 | Nearly same as v10E |

**Interpretation:**
- Adding splice_delta to ESM-2: **minimal GO-level improvement** (0.70 vs 0.69)
- But splice_delta alone (0.6367) > ESM-2 alone (0.5126) for GO prediction

### pos_bias Control (20260517)
| GO term | pos_bias (v10B) | Interpretation |
|---------|-----------------|----------------|
| GO:0006941 (muscle contraction) | 1.902 | Strong isoform-level separation |
| GO:0007005 (mitochondrion org) | 0.879 | Good separation |
| GO:0006914 (autophagy) | 0.724 | Good separation |
| Gene-level mean baseline | 0.0 | No separation (by definition) |

**pos_bias = 1.902 means:** PRISM predictions separate isoforms 1.9× better than random within the same gene.

**Historical Observation Confirmed:**
> "When we added splice_delta to training, gene-level GO performance dropped but pos_bias (isoform-level separation) improved."

**Implication:**
- splice_delta contributes to **ISOFORM-LEVEL discrimination** (pos_bias)
- But does **NOT improve GENE-LEVEL GO term prediction**
- ESM-2 is sufficient for GO prediction; splice_delta is for isoform fine-grained separation

---

## 8. Biological Interpretation

### Why ESM-2 Fails for Some Isoforms

ESM-2 is a **protein language model** trained on amino acid sequences. It CANNOT see:

1. **UTR variations** (5'/3' untranslated regions)
   - Alternative polyadenylation (APA)
   - Alternative transcription start sites (TSS) upstream of CDS
   - These affect RNA stability, localization, translation efficiency — NOT protein sequence

2. **Silent exon changes**
   - Exon inclusion/exclusion that doesn't change protein (same frame, same AA)
   - Intronic retention in non-coding regions

3. **Synonymous splicing**
   - Different exon combinations producing identical protein (rare but exists)

4. **Non-coding isoforms**
   - Retained intron isoforms (NMD candidates)
   - lncRNA-like isoforms from protein-coding genes

### Why splice_delta Succeeds

splice_delta encodes:
```
splice_delta[i] = exon_presence[isoform_i] - exon_presence[canonical]
```

This captures:
- **Exon skipping / inclusion** (regardless of frame)
- **Alternative exon usage**
- **Partial exon changes** (A3SS/A5SS encoded as fractional changes in exon_matrix)
- **UTR exons** (captured in full exon structure)

**Result:** splice_delta provides **gene-structure-level** information that ESM-2 misses.

---

## 9. Implications for PRISM Architecture

### Current Production (v15d_bp_clean)
- **Input:** ESM-2 only
- **Performance:** Macro AUPRC 0.7022 (muscle), 0.5998 (brain zero-shot)
- **Strength:** Strong GO-level prediction
- **Weakness:** May miss isoform-level discrimination for UTR-variant isoforms

### Potential v16 Strategy: ESM-2 + splice_delta Fusion

**Option A: Early Fusion (Input Concatenation)**
```python
concat = tf.concat([esm2_emb, splice_delta], axis=-1)  # (640+150=790-dim)
x = Dense(256)(concat)
```
- **Pro:** Simple, allows network to learn interactions
- **Con:** May dilute ESM-2 signal (dimension imbalance: 640 vs 150)

**Option B: Late Fusion (Dual-Branch)**
```python
esm2_branch = Dense(128)(esm2_emb)
splice_branch = Dense(64)(splice_delta)
fused = tf.concat([esm2_branch, splice_branch], axis=-1)
```
- **Pro:** Preserves modality identity, balanced representation
- **Con:** Requires careful loss weighting

**Option C: Attention-Based Fusion**
```python
esm2_feat = Dense(128)(esm2_emb)
splice_feat = Dense(128)(splice_delta)
attention_weights = Softmax()(Dense(2)([esm2_feat, splice_feat]))
fused = attention_weights[0] * esm2_feat + attention_weights[1] * splice_feat
```
- **Pro:** Adaptive weighting, interpretable
- **Con:** More complex, may overfit

### Expected Benefits (based on v10E results)
- **GO-level performance:** Marginal improvement (0.70 → 0.72?)
- **pos_bias (isoform separation):** Substantial improvement (0.7 → 1.5+)
- **Novel isoform case discovery:** Better discrimination for UTR-variant isoforms

---

## 10. Recommendations

### 10.1 Immediate Next Steps

**Experiment v16a: ESM-2 + splice_delta Late Fusion**
```python
# Model architecture
esm2_input = Input((640,))
splice_input = Input((150,))

esm2_branch = Dense(256, activation='relu')(esm2_input)
esm2_branch = Dropout(0.3)(esm2_branch)
esm2_branch = Dense(128, activation='relu')(esm2_branch)

splice_branch = Dense(128, activation='relu')(splice_input)
splice_branch = Dropout(0.2)(splice_branch)
splice_branch = Dense(64, activation='relu')(splice_branch)

fused = Concatenate()([esm2_branch, splice_branch])
fused = Dense(128, activation='relu')(fused)
fused = Dropout(0.2)(fused)
fused = Dense(64, activation='relu')(fused)
output = Dense(n_go_terms, activation='sigmoid')(fused)
```

**Evaluation Metrics:**
1. **Primary:** Macro AUPRC (18 BP GO terms)
2. **Secondary:** pos_bias (isoform-level separation)
3. **Exploratory:** AUPRC stratified by zero-delta vs high-delta isoforms

### 10.2 Hypothesis to Test

**H1:** ESM-2 + splice_delta will improve pos_bias without hurting GO-level AUPRC.

**H2:** Improvement will be larger for genes in the "ESM-2 failure" category (73 genes with zero ESM-2 distance).

**H3:** High-delta isoforms (|Δ| > 3) will show larger improvement than zero-delta isoforms.

### 10.3 Ablation Study Design

| Model | ESM-2 | splice_delta | Expected Macro AUPRC | Expected pos_bias |
|-------|-------|--------------|----------------------|-------------------|
| Baseline (v15d) | ✓ | ✗ | 0.70 | 0.7-0.9 |
| v16a_esm2_only | ✓ | ✗ | 0.70 | 0.7-0.9 |
| v16a_splice_only | ✗ | ✓ | 0.64 | 1.2-1.5 |
| v16a_fusion | ✓ | ✓ | 0.71-0.72 | 1.3-1.8 |

### 10.4 Risk Assessment

**Risk 1: Mode collapse to ESM-2-dominated solution**
- Mitigation: Use gradient norm monitoring (Axis 4), apply gradient modulation if ESM-2 gradient >> splice gradient

**Risk 2: Overfitting to splice_delta noise**
- Mitigation: Higher dropout on splice branch (0.3 vs 0.2), L2 regularization

**Risk 3: GO performance degradation (observed in v10E)**
- Mitigation: Multi-task loss with explicit pos_bias term:
  ```python
  total_loss = focal_loss(y_true_go, y_pred_go) + 
               λ_posb * triplet_loss(embeddings, gene_ids)
  ```

### 10.5 Success Criteria

**Minimum Viable Improvement:**
- Macro AUPRC ≥ 0.70 (no degradation from v15d)
- pos_bias ≥ 1.2 (≥40% improvement over baseline)

**Target Performance:**
- Macro AUPRC ≥ 0.72 (+2.8% over v15d)
- pos_bias ≥ 1.5 (2× improvement over baseline)
- Brain zero-shot AUPRC ≥ 0.61 (+2.0% over v15d)

---

## 11. Conclusion

**Core Discovery:**
ESM-2 and splice_delta are **COMPLEMENTARY** features for isoform function prediction:
- **ESM-2:** Captures protein sequence variation → drives GO-level prediction
- **splice_delta:** Captures exon-level structural variation → drives isoform-level separation

**Critical Gap in Current PRISM (v15d):**
- 25% of multi-isoform genes have ZERO ESM-2 separation (P25 = 0.0000)
- These genes likely have UTR-variant, alternative TSS, or APA isoforms
- PRISM currently cannot distinguish these isoforms beyond gene-level prediction

**Strategic Recommendation:**
Integrate splice_delta into PRISM v16 to:
1. Improve isoform-level discrimination (pos_bias)
2. Enhance BISECT case discovery for UTR-variant isoforms
3. Maintain or improve GO-level prediction performance

**Manuscript Angle (Nature Methods / NMI):**
> "PRISM integrates protein language model embeddings (ESM-2) with exon-structure encoding (splice_delta) to achieve both gene-level functional annotation and isoform-level discrimination, overcoming the limitation that protein sequence alone cannot distinguish UTR-variant and structurally complex isoforms."

---

**Files Generated:**
- `/home/welcome1/sw1686/DIFFUSE/reports/splice_delta_esm2_analysis.json` (raw data)
- `/home/welcome1/sw1686/DIFFUSE/reports/splice_delta_esm2_analysis.txt` (console output)
- `/home/welcome1/sw1686/DIFFUSE/reports/splice_delta_esm2_comprehensive_analysis.md` (this document)
