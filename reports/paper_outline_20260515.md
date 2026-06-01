# DIFFUSE v10 — Paper Outline Draft
# Target: Nature Methods / Nature Machine Intelligence
# Last updated: 2026-05-15

---

## Title (candidates)

A: "Deep learning of isoform-level protein function using ESM-2 embeddings and muscle long-read single-cell RNA sequencing"

B: "Isoform-resolved functional prediction reveals expression-dominant splicing patterns in human skeletal muscle"

C: "A sequence-based framework for isoform-level functional annotation of novel long-read transcripts"

→ 현재 선호: B (within-gene ranking contribution 강조) 또는 C (novel isoform focus)

---

## Abstract (key numbers)

- Dataset: 36,748 human skeletal muscle isoforms (14 N + 10 D long-read sc-seq samples)
- Method: ESM-2 based MLP (v10-B) for GO-level isoform function prediction
- Main result: Macro AUPRC +27.8% vs LR baseline (p<0.001 × 3/5 GO terms; bootstrap CI)
- Isoform-level validation: within-gene pairwise AUROC 0.692 (p<0.001)
- Novel isoform prediction: prospective AUROC 0.581 for BambuTx isoforms (p<0.001)
- Discovery: PDE4B novel isoform 10.6× upregulated in disease; TPM1/DMD/ANK2 isoform switches

---

## 1. Introduction

### 1.1 문제 정의
- Functional annotation은 gene-level → isoform-level gap 존재
- Alternative splicing이 기능 다양성의 주요 메커니즘 (인용 필요)
- 기존 방법: 단백질 서열 기반 function prediction은 gene-level에 편중

### 1.2 Long-read sc-seq의 기회
- 우리 데이터: 392,585 transcripts, 4,641 BambuTx novel isoforms
- 74.4% isoform coverage with within-gene expression ratios
- 45% of BambuTx novel isoforms co-dominant (ratio > 0.3) → 생물학적으로 active

### 1.3 핵심 도전
- Gene-level annotation만 존재 → isoform-level label 부재
- within-gene label variance = 0 for GO annotations
- Novel isoform = no existing function annotation

### 1.4 Contribution 목록
1. v10-B: ESM-2 MLP이 GO-term isoform function prediction에서 LR +27.8% (p<0.001)
2. Within-gene pairwise ranking: D/S delta features, AUROC 0.692
3. Prospective BambuTx validation: novel isoform ranking, AUROC 0.581
4. Novel isoform discovery: PDE4B disease-specific upregulation, TPM1/DMD/ANK2

---

## 2. Methods

### 2.1 Dataset

**Test isoforms (36,748)**
- Source: Human skeletal muscle long-read sc-seq
  - 14 normal (N) + 10 disease (D) samples
  - BAMBU assembly → 36,748 isoforms (including 1,550 BambuTx novel)
- GO labels: 5 muscle-relevant GO terms (GO:0006096, 0003774, 0007204, 0030017, 0006941)
  - Positive instances: 76–452 per term (test set)

**Train isoforms (31,668)**
- RefSeq NM_ human isoforms + SwissProt cross-species (GO:0006096 only)
- Gene-stratified split: test genes ∩ train genes = ∅

### 2.2 Features

**ESM-2 (640d) — Main feature**
- Model: ESM-2 t30_150M (30 layers, 150M params)
- Computed per-isoform protein sequence
- Captures protein-level functional identity

**Domain delta (512d) — Supplementary**
- Pfam-A binary presence/absence matrix (top-512 domains)
- Test: ENST-based HMMER annotation (37.2% coverage)
- Train: hmmscan-based (90.4% coverage after hmmscan)
- delta = isoform − canonical presence

**Splice delta (150d) — Supplementary**
- Per-exon usage difference vs MANE canonical
- 150-dimensional (exon position ordered)

### 2.3 Model Architecture (v10-B)

```
Input: ESM-2 embedding (640d)
  → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(128, ReLU) + Dropout(0.2)
  → Dense(64, ReLU)
  → Dense(1, Sigmoid)

Training:
  Loss: BinaryFocalCrossEntropy(gamma=2, α=balanced)
  Optimizer: Adam, lr=1e-4
  Epochs: 50, early stopping (patience=10)
  Positive weighting: class_weight='balanced'
```

### 2.4 Evaluation

**GO prediction**
- Gene-stratified GroupKFold (5-fold)
- Primary: AUPRC (sparse class metric)
- Bootstrap CI: gene-block resampling, n=1,000

**Within-gene pairwise ranking**
- All (iso_i, iso_j) pairs within same gene
- Feature: diff(D/S feature)
- Metric: pairwise AUROC (random baseline = 0.5)
- Gene-stratified 80:20 split, 5 seeds

**Prospective BambuTx validation**
- Train: non-BambuTx isoforms only
- Test: (BambuTx_i, known_j) pairs within same gene
- Metric: pairwise AUROC with gene-block bootstrap

---

## 3. Results

### 3.1 Baseline Comparison: v10-B vs LR

| GO term | v10-B | LR | Δ | p |
|---------|-------|-----|---|---|
| GO:0006096 | 0.671 | 0.695 | −0.023 | n.s. |
| GO:0003774 | 0.813 | 0.825 | −0.013 | n.s. |
| GO:0007204 | 0.765 | 0.415 | **+0.303** | <0.001 |
| GO:0030017 | 0.743 | 0.564 | **+0.207** | <0.001 |
| GO:0006941 | 0.597 | 0.310 | **+0.314** | <0.001 |
| **Macro** | **0.718** | **0.562** | **+0.156** | — |

Key finding: ESM-2 MLP significantly outperforms LR for 3/5 GO terms (p<0.001), particularly for isoform-discriminative terms (Ca²⁺ signaling, sarcomere, muscle contraction).

**Figure 1**: AUPRC bar plot per GO term, v10-B vs LR with bootstrap CI. Inset: PFN vs MLP diagnostic (F26: PFN degradation confirmed).

### 3.2 Isoform-Level Discrimination (pos_bias)

pos_bias_score = within_gene_score_std(positive_genes) / global_score_std

| GO term | pos_bias (v10-B) | pos_bias (gene-consensus) |
|---------|-----------------|--------------------------|
| GO:0006096 | 0.550 | 0.000 |
| GO:0003774 | 1.198 | 0.000 |
| GO:0007204 | 0.403 | 0.000 |
| GO:0030017 | 0.898 | 0.000 |
| GO:0006941 | 1.732 | 0.000 |
| **Macro** | **0.956** | **0.000** |

v10-B assigns distinct scores to isoforms within the same positive gene (pos_bias=0.956), confirming genuine isoform-level resolution. Gene-consensus model (same embedding for all isoforms) gives pos_bias=0 trivially.

**Figure 2**: pos_bias per GO term. Example: GO:0006941 pos_bias=1.73 — within positive muscle contraction genes, isoforms are discriminated more than globally.

### 3.3 Within-Gene Expression Ratio Validation

Using N-sample within-gene expression ratios from our long-read data as ground truth:

**Feature correlation with expression ratio:**

| Feature | Global Spearman | Within-gene Spearman | Non-canonical Spearman |
|---------|----------------|---------------------|------------------------|
| |splice_delta| | −0.700 | −0.512 | **−0.310** |
| |domain_delta| | −0.251 | −0.368 | **−0.211** |
| |ESM-2| | −0.047 | −0.133 | **−0.068** |

Pairwise ranking (gene-stratified CV):

| Model | AUROC | 95%CI | p |
|-------|-------|-------|---|
| D/S only | 0.692 | [0.685, 0.699] | <0.001 |
| ESM-2 | 0.736 | [0.728, 0.743] | <0.001 |
| Full | 0.780 | [0.774, 0.787] | <0.001 |

**Non-canonical validation** (canonical excluded):
- D/S (0.592) ≈ ESM-2 (0.587) → splice/domain features carry genuine isoform-specific expression information beyond canonical-dominance

**Figure 3**: 
(a) Spearman correlation barplot (global vs within-gene vs non-canonical)
(b) Pairwise ranking AUROC comparison (D/S, ESM-2, Full, with CI)
(c) Violin: within-gene ratio distribution by gene type

### 3.4 Prospective Novel Isoform Validation (BambuTx)

1,225 BambuTx novel isoforms with expression data:
- 551 (45%) co-dominant (ratio > 0.3)

Pairwise ranking (trained on known, tested on BambuTx):

| Model | Prospective AUROC | 95%CI | p |
|-------|-----------------|-------|---|
| D/S only | 0.581 | [0.557, 0.604] | <0.001 |
| ESM-2 | 0.662 | [0.641, 0.686] | <0.001 |
| Full | 0.645 | [0.622, 0.667] | <0.001 |

Spearman(predicted dominance, actual ratio) = 0.122, p<0.001

**Figure 4**: Prospective validation schematic + AUROC comparison.

### 3.5 Disease-Specific Novel Isoform Discovery

**Primary finding: PDE4B (BambuTx85)**
- Normal muscle: ratio = 0.22 (minor isoform)
- Disease muscle: ratio = 0.57 (co-dominant, **10.6× raw count increase**)
- PDE4B: cAMP phosphodiesterase expressed in skeletal muscle
- PDE4 inhibitors under investigation for muscular dystrophy
- Implication: disease-specific alternative splicing of PDE4B modulates cAMP signaling

**Secondary finding: MINDY3 (BambuTx1114)**
- Normal muscle: BambuTx1114 = dominant (ratio 0.73)
- Disease muscle: BambuTx1114 = minor (ratio 0.44), canonical isoforms increase
- MINDY3: Lys-48 deubiquitinase, highly expressed in skeletal muscle
- Implication: loss of this novel isoform may affect muscle proteostasis

**Literature-confirmed known switches (supplementary):**
- TPM1: sarcomere-competent vs non-competent isoforms
- DMD: Dp427m (muscle) vs Dp71 (brain)
- ANK2: AnkB-212 cardiac M-line specific (Camors 2015)

**Figure 5**: 
(a) Expression ratio heatmap: N vs D for top 10 BambuTx switch genes
(b) PDE4B isoform expression bar (N vs D, per transcript)
(c) Model framework schematic (v10-B + D/S features)

---

## 4. Discussion

### 4.1 Why ESM-2 MLP outperforms PFN
- PFN: 640→32→predict (compression bottleneck)
- MLP: 640→256→128→64→predict (preserves information)
- D1_MLP diagnostic: Macro 0.69 >> v8b-PFN 0.36

### 4.2 Gene-level vs Isoform-level information
- GO annotations are gene-level → ESM-2 absolute (gene-level function) is appropriate for GO prediction
- ESM-2 delta (canonical-differential) → kills GO prediction (Macro ~0.01)
- D/S features → within-gene expression prediction (genuine isoform-level signal)
- The two tasks require different feature spaces → validated here

### 4.3 Limitations
- GO labels are gene-level: cannot directly validate isoform-specific function
- pos_bias metric: consistent within positive genes, but ground truth functional equivalence unknown
- Prospective BambuTx: expression ≠ function (expressed ≠ functionally important)
- Disease type: samples labeled D/N (disease muscle tissue); specific diagnosis not publicly disclosed

### 4.4 Future directions
- Train splicing delta: requires shared exon cluster space (1-2 weeks)
- ESMFold for isoform-specific 3D structure prediction
- Functional validation of PDE4B BambuTx85 (RT-PCR, protein-level)

---

## 5. Supplementary

### S1. PFN diagnostic (D1_MLP: 0.689 >> v8b-PFN: 0.357)
### S2. Bootstrap CI tables (all GO terms, all models)
### S3. within-gene ratio distribution statistics
### S4. BambuTx isoform switch full table (10 candidates)
### S5. Annotation gap genes (DYNC2I1/2, MYO5C)
### S6. Coding vs non-coding discrimination (17-485×)

---

## Key Numbers for Abstract/Title

- 36,748 isoforms, 14 N + 10 D samples
- Macro AUPRC +27.8% vs LR (p<0.001 for 3/5 GO terms)
- pos_bias macro = 0.956 (genuine isoform resolution)
- Within-gene pairwise AUROC 0.692 (p<0.001, n=20,096 pairs)
- Prospective BambuTx AUROC 0.581 (p<0.001)
- PDE4B BambuTx85: 10.6× upregulation in disease
- splice_delta 150d alone: AUROC 0.647 ≈ ESM-2 640d (0.665)

---

## Result Files Reference

| Figure | Data File |
|--------|-----------|
| Fig 1 (GO pred) | reports/bootstrap_ci/20260515_0240/ |
| Fig 2 (pos_bias) | reports/v10_mlp/v10_bias_results_*.json |
| Fig 3 (within-gene) | reports/within_gene/pairwise_*.json |
| Fig 4 (BambuTx prospective) | reports/within_gene/prospective_bambu_*.json |
| Fig 5 (disease switch) | reports/within_gene/prospective_bambu_*.json |
| S1 (PFN diag) | reports/diagnostics/ |
| S2 (bootstrap) | reports/bootstrap_ci/ |

---

## Revision Checklist (before submission)

- [ ] Bootstrap CI on within-gene pairwise AUROC (currently in-sample bootstrap)
- [ ] Confirm disease type of D samples (contact data source)
- [ ] PDE4B BambuTx85 sequence characterization (splice site, protein domain)
- [ ] Gene-stratified CV for prospective AUROC (currently gene-block bootstrap)
- [ ] MANE canonical citation (O'Leary et al. 2016)
- [ ] ESMFold validation for PDE4B BambuTx85 structure prediction
