# Section 08B: Methodological Value of Negative Results

## Key Question
Which approaches were tested and failed, and what does each failure tell us about the problem structure?

## Negative Results Catalogue (13 approaches)

### Category 1: Training Dynamics (v6-era experiments)

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **F1**: Phase1 branch pre-training | Triplet active ratio 0.2% → gradient dominance | Pre-training a single branch monopolizes gradient flow |
| **F2**: Cross-modality gating | AUPRC 0.064 (catastrophic) | Phase2 random CNN corrupts Phase1 ESM-2 representations |
| **F5**: Phase2 CNN on Phase1 embeddings | centroid_cos degrades 0.81→0.91 | CNN focal loss overwrites metric geometry |

**Lesson**: Multi-phase training is unstable when modalities are at different learning stages.

---

### Category 2: Gene-Context Modifications (v11)

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **v11-A** Gene-context self-attention | Type-B macro Δ=-0.114 (Ca2+), -0.089 (Sarcomere) | Attention injects gene identity as shortcut |
| **v11-B** Isoform deviation (x_i − gene_mean) | 0/11 Type-B terms improved | Removing gene info also removes useful signal |

**Lesson**: GO annotations are gene-level constructs. Any model modification that emphasizes gene identity — either by adding gene context (v11-A) or explicitly removing it (v11-B) — degrades performance. The optimal strategy is to treat each isoform independently (flat MLP).

**Methodological contribution**: This negative result definitively addresses the "gene-level shortcut" concern: the model cannot learn gene-level shortcuts in the flat architecture, and attempting to address this concern (v11-B) paradoxically worsens performance.

---

### Category 3: Embedding and Activation Modifications (v19/v20)

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **v19** SwiGLU + delta embedding (1280d) | 0.5462 (−0.156) | Concatenating delta embedding adds noise |
| **v20** SwiGLU activation only (640d) | 0.6763 (−0.026) | SwiGLU marginally worse than ReLU |
| **delta only** (v19−v20) | −0.130 | The additional delta information is harmful |

**Lesson**: The 640-dimensional ESM-2 embedding already encodes isoform-specific functional information. Adding relative information (isoform − gene mean) as an additional channel degrades rather than improves predictions, possibly because it confounds ESM-2's intrinsic isoform representation.

---

### Category 4: Data Augmentation (v8-era)

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **Label propagation (LP)** | GO:0006096 AUPRC -8.9%, 4/5 GO terms degraded | Muscle co-expression network ≠ GO function similarity |
| **Zero-imputation BiGRU** | pos_bias 1.196→1.116 (−0.080) | Zero-imputed splice features train BiGRU on noise |
| **FiLM conditioning** | Type-B performance degraded | Cell localization FiLM overwrites ESM-2 representations |

**Lesson**: Data augmentation and additional modalities consistently fail when the auxiliary data (co-expression, splice, cell localization) is structurally misaligned with the GO label ontology.

---

### Category 5: Annotation Quality Hypotheses

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **TBS vs ΔAUPRC** | r=-0.143, p=0.640 | Tissue breadth score ≠ predictability |
| **TCS vs ΔAUPRC** | r=-0.100, p=0.744 | Tissue conservation ≠ predictability |
| **SMSI** | r<0.01 | Sarcopenia-muscle specificity index ≠ predictability |

**Counterexample**: Motor activity (TBS=0.833, TCS=0.853) vs Ca²⁺ homeostasis (TBS=0.833, TCS=0.855) — identical annotation quality metrics, ΔAUPRC differs by 0.488.

**Lesson**: The difficulty of a GO term is determined not by annotation breadth or conservation, but by the structural heterogeneity of the positive class in embedding space (pc1_var_ratio r=-0.765, p=0.002).

---

### Category 6: Validation Attempts

| Finding | Result | Interpretation |
|---------|--------|----------------|
| **AlphaFold H1** (score↔pLDDT) | 0/6 genes pass | Score saturation: all coding isoforms of positive-class genes ≥0.95 |
| **Multi-scale CNN** | 5-term inconsistent improvement | CNN kernels are GO-term-specific, not universal |

**Lesson**: AlphaFold H1 failure is itself informative — it reveals score saturation (ESM-2 strongly encodes gene-level function), and demonstrates that within-positive-gene coding isoforms are essentially equivalent in the model's view.

---

## Summary Table

| Category | n approaches | n negative | Core insight |
|----------|-------------|------------|--------------|
| Training dynamics | 3 | 3 | Multi-phase training is unstable |
| Gene-context | 2 | 2 | GO = gene-level; context always hurts |
| Embedding variants | 2 | 2 | ESM-2 640d is optimal |
| Data augmentation | 3 | 3 | Auxiliary data misaligned with GO |
| Annotation quality | 3 | 3 | Heterogeneity, not quality, matters |
| Validation | 2 | 1 partial | Score saturation is informative |
| **Total** | **15** | **14/15** | |

**Methodological contribution**: 14/15 negative results converge on a single insight: the optimal architecture for gene-level GO annotation prediction is a flat, isoform-independent MLP on ESM-2 embeddings, without gene context, auxiliary modalities, or complex activation functions.
