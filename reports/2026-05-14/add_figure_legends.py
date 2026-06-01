"""
add_figure_legends.py — 2026-05-14
===================================
모든 Figure PNG 하단에 publication-quality legend를 추가한다.
원본 파일을 덮어쓴다 (스크립트로 언제든 재생성 가능).
"""

import os, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

OUT_DIR = '/home/welcome1/sw1686/DIFFUSE/reports/2026-05-14'

# ─── Legend definitions ───────────────────────────────────────────────────────
# Each entry: (filename, figure_number, title, legend_text)
# Bold text indicated by **...**; italics by _..._
LEGENDS = {

# ── Meeting figures ────────────────────────────────────────────────────────────

'fig1_performance_history.png': (
    'Fig. 1',
    'Model performance progression across development stages.',
    (
        '**Metric**: Macro-AUPRC (unweighted mean across 5 GO terms: '
        'GO:0006096, GO:0003774, GO:0007204, GO:0030017, GO:0006941). '
        'AUPRC (Area Under the Precision-Recall Curve) is used as the primary metric '
        'due to severe class imbalance (positive rate 0.2–1.2%). '
        'Each bar represents a distinct model configuration; '
        'the **gray dashed line** denotes the ESM-2 640-dim Logistic Regression (LR) '
        'human-only baseline (Macro-AUPRC = 0.5614), serving as the upper reference. '
        '**v7c** (0.315): baseline multimodal PFN with 64-dim projection. '
        '**v8b** (0.357): Unified Loss (focal + triplet), human-only training signal for Type-B. '
        '**D256** (0.384): dimension expanded to 256 with SwissProt retained (SP+). '
        '**P3-512** (reported per GO): SwissProt removed (SP−), dimension = 512. '
        '**Selective Ensemble** (0.4817, **85.8% of LR**): optimal SP×Dim configuration '
        'selected independently per GO term — P3-512 for Type-B terms, D256 for Type-A (GO:0006096).'
    )
),

'fig2_sp_dim_ablation.png': (
    'Fig. 2',
    '2×2 ablation study: SwissProt (SP) × embedding dimension (Dim).',
    (
        'Each cell shows **Macro-AUPRC** for the corresponding SP×Dim configuration. '
        '**Rows**: SwissProt training data included (SP+) or excluded (SP−). '
        '**Columns**: projection dimension 64 or 256. '
        'The right panel shows the **per-GO-term breakdown** for all four conditions. '
        '**Key interaction (SP×Dim)**: (1) At 64d, SP+ and SP− perform similarly on Type-B GO terms '
        '(GO:0007204, GO:0030017), but SP+ is essential for GO:0006096 (Type-A, human positives = 32). '
        '(2) At 256d, the effect of SP is amplified: SP− gains +0.116 AUPRC on GO:0007204 '
        'while SP− loses −0.389 on GO:0006096. '
        'This interaction motivates GO-term-specific SP strategy (Selective Ensemble). '
        '**LR reference** (ESM-2 640-dim, human-only): Macro = 0.5614.'
    )
),

'fig3_dim_scaling_law.png': (
    'Fig. 3',
    'Embedding dimension scaling law under SP− condition (SwissProt excluded).',
    (
        '**Left panel**: AUPRC vs. embedding dimension (64d → 256d → 512d) '
        'for Type-B GO terms (GO:0007204, GO:0030017, GO:0006941) under SP− training. '
        'All three terms show monotonic improvement with dimension, '
        'with diminishing returns: GO:0007204 gains +91% (64→256d) then +30% (256→512d). '
        'At 512d, GO:0007204 reaches 0.4055 ≈ LR 0.4140 (**97.9% of LR**). '
        '**Right panel**: Type-A GO terms (GO:0006096, GO:0003774) for comparison — '
        'GO:0006096 collapses under SP− regardless of dimension '
        '(SP− 64d: 0.464; SP− 256d: 0.445; SP− 512d: 0.494 vs. SP+ 256d D256: 0.833), '
        'confirming that Type-A terms require SwissProt training signal. '
        '**Dashed lines**: LR baseline per GO term. '
        '**Interpretation**: Dimension scaling amplifies the benefit of SP removal for Type-B, '
        'while the harm of SP removal for Type-A persists independently of dimension.'
    )
),

'fig4_fti_tier.png': (
    'Fig. 4',
    'Functional Transferability Index (FTI) and GO term tier classification.',
    (
        '**FTI** is defined as AUPRC(SP+, 256d) / AUPRC(SP−, 256d), '
        'quantifying the net benefit of SwissProt transfer for each GO term. '
        'FTI > 1: SP helps (Type-A); FTI < 1: SP hurts (Type-B); FTI ≈ 1: neutral. '
        '**Left scatter**: FTI at 64d vs. 256d for all 5 GO terms. '
        'The dashed line (FTI = 1) separates SP-beneficial from SP-harmful regimes. '
        'Note that the SP×Dim interaction moves GO:0007204 from FTI≈1.0 (64d) to FTI=0.626 (256d), '
        'indicating that the harm of SP is **amplified** at higher dimensions for Type-B terms. '
        '**Right bar chart**: FTI values at 256d by tier. '
        'Tier 1 (SP Required): FTI > 1.5; Tier 2 (SP Neutral): 0.9–1.5; Tier 3 (SP Harmful): FTI < 0.9. '
        'GO:0006096 FTI = 1.874 (Tier 1); GO:0007204 FTI = 0.626 (Tier 3).'
    )
),

'fig5_2d_sp_dependency.png': (
    'Fig. 5',
    '2D mechanistic framework: Taxonomic Breadth Score (TBS) × Tissue Context Specificity (TCS) predict FTI.',
    (
        '**TBS** (x-axis): fraction of 6 biological kingdoms (Bacteria, Archaea, Fungi, Invertebrate, '
        'Vertebrate, Plant) with at least one SwissProt+ protein annotated for the GO term. '
        'Computed from local SwissProt annotation file. '
        '**TCS** (y-axis): mean Tau index (τ) of GO-positive proteins across 51 tissues (HPA atlas). '
        'τ = 1: tissue-specific; τ = 0: ubiquitously expressed. '
        '**Bubble size / color**: FTI (red = high / blue = low). '
        'The **2D regression model** (right panel) fits: '
        'FTI = 14.631 + 60.237×TBS − 12.058×TCS − 75.350×(TBS×TCS), **R² = 0.992** (n=5). '
        'The interaction term (β₃ = −75.350) explains the apparent paradox: '
        'GO:0007204 and GO:0006096 share TBS = 0.667, yet FTI = 0.626 vs. 1.874 — '
        'the difference is driven by TCS (0.869 vs. 0.849) amplified through the interaction. '
        '**Caution**: n=5 model; R² reflects analytical fit, not out-of-sample generalization.'
    )
),

'fig6_kingdom_smsi_panel.png': (
    'Fig. 6',
    'SwissProt annotation breadth by biological kingdom and skeletal muscle specificity index (SMSI).',
    (
        '**Left (stacked bar)**: number of SwissProt+ proteins per GO term, '
        'broken down by biological kingdom (Bacteria, Fungi, Invertebrate, Vertebrate, Plant, Archaea). '
        'GO:0003774 (Motor Activity) has the broadest taxonomic representation '
        '(TBS = 0.833, 5 kingdoms), reflecting the deep evolutionary conservation of motor proteins. '
        '**Center (bubble scatter)**: FTI vs. TBS; bubble area encodes number of human+ proteins. '
        'TBS alone has Pearson r = 0.358 (insufficient to predict FTI), '
        'motivating the 2D TBS×TCS framework. '
        '**Right (bar)**: Skeletal Muscle Specificity Index (SMSI) = '
        'mean nTPM(skeletal muscle) / mean nTPM(all 51 tissues) for GO-positive proteins (HPA). '
        'SMSI > 1: muscle-enriched; SMSI < 1: muscle-depleted. '
        'GO:0030017 SMSI = 13.71×: sarcomere proteins are highly muscle-enriched. '
        'GO:0007204 SMSI = 1.63×: Ca²⁺ signaling proteins are NOT muscle-enriched — '
        'explaining the training corpus tissue mixing that harms model performance.'
    )
),

'fig7_selective_ensemble.png': (
    'Fig. 7',
    'Per-GO-term performance comparison and Selective Ensemble construction.',
    (
        '**Left (grouped bar)**: AUPRC for each GO term across all model configurations '
        '(v8b, P3, P3-256, P3-512, D256) and the LR baseline. '
        'Bold borders indicate the best pipeline model per GO term. '
        '**Color coding**: blue family = SP− models (P3 variants); orange = SP+ D256. '
        '**Right (waterfall)**: Selective Ensemble construction. '
        'The Selective Ensemble assigns the best model independently per GO term: '
        'P3-512 for GO:0007204, GO:0030017, GO:0006941, GO:0003774; D256 for GO:0006096. '
        'Selective Macro-AUPRC = **(0.4817)**, compared to v8b baseline (0.357) and LR (0.561). '
        'The ensemble achieves **85.8% of LR** performance, '
        'up from 63.5% (v8b) — a +34.7% relative improvement over the starting model. '
        '**Previous selective best** (P3-256 + D256): 0.4365 → P3-512 upgrade yields +0.045 gain.'
    )
),

'fig8_ds_overview.png': (
    'Fig. 8',
    'Domain Delta (D) and Splicing Delta (S) feature overview and statistics.',
    (
        '**Domain Delta (D)**: isoform-level deviation from its canonical transcript in domain composition vector '
        '(251-dimensional; PFAM domain family counts). '
        'Computed as Δ_D(iso) = domain_vec(iso) − domain_vec(canonical). '
        'Non-zero entries indicate isoform-specific domain gain/loss events. '
        '**Splicing Delta (S)**: isoform-level deviation in exon usage pattern '
        '(150-dimensional; relative exon inclusion/exclusion). '
        'Computed as Δ_S(iso) = exon_vec(iso) − exon_vec(canonical). '
        '**Top-left pie**: fraction of isoforms with ≥1 non-zero D or S element. '
        '**Top-center histogram**: distribution of isoforms per gene (log-scale); '
        'median = [shown in figure], max = 63 isoforms/gene. '
        '**Bottom panels**: D and S magnitude distributions across all 36,748 isoforms. '
        'Sparse distribution (most isoforms near zero) reflects the specificity of '
        'alternative splicing events — only functionally diverged isoforms show large Δ values. '
        '**Table inset**: feature dimensionality and sparsity statistics.'
    )
),

'fig9_isoform_resolution.png': (
    'Fig. 9',
    'Isoform-level resolution: distinguishability of isoforms within the same gene.',
    (
        '**Left**: Comparison of within-gene isoform diversity (intra-gene standard deviation of D/S features) '
        'vs. between-gene diversity. '
        'If D/S features captured only gene-level information, intra-gene std would approach zero. '
        'The observed intra-gene std demonstrates **isoform-intrinsic signal** beyond gene identity. '
        '**Center scatter**: per-gene intra-gene D std (x) vs. S std (y). '
        'Genes with high D×S diversity are isoform-functionally heterogeneous. '
        '**Right**: fraction of isoform pairs within the same gene that are D/S-distinguishable '
        '(Euclidean distance > threshold). '
        '**78.8%** of within-gene isoform pairs are distinguishable — '
        'validating that D/S features provide isoform-level resolution beyond gene-level embeddings. '
        'Threshold defined as the 10th percentile of between-gene isoform pair distances.'
    )
),

'fig10_ds_vs_previous.png': (
    'Fig. 10',
    'Isoform resolution depth: current D/S features vs. prior gene-level approaches.',
    (
        '**Left panels**: example genes illustrating isoform-level heterogeneity captured by D/S. '
        'Each dot = one isoform; x-axis = D magnitude (domain change); y-axis = S magnitude (exon change). '
        'Isoforms in the top-right quadrant represent large-scale structural divergence from canonical. '
        '**Right (summary bar)**: fraction of genes where at least one non-canonical isoform '
        'differs substantially from the canonical transcript (D or S > threshold). '
        '**Comparison with gene-level baseline**: gene-level ESM-2 embeddings assign identical features '
        'to all isoforms of the same gene; D/S features differentiate isoforms within a gene. '
        'The current pipeline uses per-isoform ESM-2 + D + S, enabling function prediction '
        'at single-isoform resolution — a capability not achievable with gene-level representations. '
        'This is particularly critical for genes with isoform-specific pathological variants '
        '(e.g., PKM1 vs. PKM2, ANK3 AnkG107 muscle isoform).'
    )
),

# ── Bootstrap CI ───────────────────────────────────────────────────────────────

'fig_bootstrap_ci.png': (
    'Fig. S1',
    'Gene-block bootstrap 95% confidence intervals for AUPRC across GO terms and model configurations.',
    (
        '**Method**: Gene-block bootstrap resampling (N = 1,000 iterations, seed = 42). '
        'Resampling unit = gene (n = 12,709 genes, 36,748 isoforms total), '
        'preventing isoform-level label leakage within the same gene. '
        'Point estimates and 95% CIs computed via percentile method [2.5th, 97.5th percentile]. '
        '**Models shown**: v8b (baseline), P3-512 (SP−, 512d), D256 (SP+, 256d), '
        'LR (ESM-2 640-dim Logistic Regression, human-only). '
        '**Significance markers**: *** p < 0.001; ** p < 0.01; * p < 0.05; ns = not significant '
        '(one-sided paired bootstrap test, H₀: model A ≤ model B). '
        '**Key statistical findings**: '
        '(1) P3-512 vs. v8b: Δ = +0.259 (GO:0007204, p < 0.001), Δ = +0.198 (GO:0030017, p < 0.001) — '
        'statistically robust improvement for Type-B terms. '
        '(2) D256 vs. LR (GO:0006096): Δ = +0.138, p = 0.055 — trend toward superiority, '
        'CI includes zero; reported as "numerically higher" without significance claim. '
        '(3) Wide CIs reflect sparse positive sets (n = 32–452 positives per GO term).'
    )
),

# ── Embedding overview ─────────────────────────────────────────────────────────

'fig_embedding_overview.png': (
    'Fig. S2',
    'UMAP embedding space overview: Phase-2 unified embeddings for all four GO terms.',
    (
        '**Method**: Phase-2 unified embeddings (per-isoform, 36,748 isoforms) '
        'were dimensionality-reduced via PCA (50 components) followed by UMAP '
        '(n_neighbors = 30, min_dist = 0.1, cosine metric, seed = 42). '
        'For speed, up to 5,000 negative isoforms were randomly subsampled; all positives retained. '
        '**Color**: red/orange = positive (GO-annotated); light gray = negative. '
        '**Star markers**: cluster core representatives (top-10 isoforms nearest to positive centroid '
        'in original embedding space). '
        '**Panel layout** (clockwise from top-left): '
        'GO:0006096 Glycolysis (Type-A, D256, AUPRC = 0.8331); '
        'GO:0003774 Motor Activity (Type-A, D256, AUPRC = 0.5982); '
        'GO:0007204 Ca²⁺ Signaling (Type-B, P3-512, AUPRC = 0.4055); '
        'GO:0030017 Sarcomere (Type-B, P3-512, AUPRC = 0.3553). '
        'Type-A terms show more compact positive clusters; '
        'Type-B terms show diffuse distributions reflecting heterogeneous SwissProt training contexts.'
    )
),

'fig_embedding_GO_0006096.png': (
    'Fig. S3a',
    'Embedding space: GO:0006096 Glycolysis (Type-A, D256 model, AUPRC = 0.8331).',
    (
        '**Left**: UMAP 2D projection of Phase-2 unified embeddings (D256 model, SP+, 256-dim). '
        'Red dots = positive isoforms (n = 76); gray dots = negatives (subsampled). '
        'Star markers = cluster core (top-3 genes: **PFKP**, **PFKL**, **PKM**). '
        'The glycolysis positive cluster forms a **compact, well-separated group** — '
        'consistent with the enzymatic function being conserved across taxa '
        '(TBS = 0.667, 4 kingdoms) and the SP+ training providing diverse cross-species '
        'glycolytic enzyme annotations. '
        '**Right**: prediction score distribution (AUPRC = 0.8331 > LR = 0.6949). '
        '**Cluster core biology**: PFKP (platelet/brain PFK isoform), PFKL (liver/kidney PFK isoform), '
        'PKM (PKM2 fetal splice form) — all glycolytic enzymes, though muscle-dominant PFKM is absent '
        'from core top-3, reflecting partial Tissue Context Mixing in the training corpus. '
        'Notably, PFKM appears at rank 4 in the full core list, confirming it is captured.'
    )
),

'fig_embedding_GO_0003774.png': (
    'Fig. S3b',
    'Embedding space: GO:0003774 Motor Activity (Type-A, D256 model, AUPRC = 0.5982).',
    (
        '**Left**: UMAP 2D projection of Phase-2 unified embeddings (D256 model, SP+, 256-dim). '
        'Orange dots = positive isoforms (n = 164); gray dots = negatives (subsampled). '
        'Star markers = cluster core (top genes: **KIF26A**, **KIF21A**, **MYO10**, **KIF3C**, **MYH7**). '
        'The motor activity positive cluster shows **multiple sub-clusters** — '
        'reflecting the functional and structural heterogeneity of motor proteins '
        '(kinesins, myosins, unconventional motors). '
        '**Right**: prediction score distribution (AUPRC = 0.5982). '
        '**Cluster core biology**: '
        'KIF26A (non-motor kinesin, neural/renal), KIF21A (CFEOM1 kinesin, oculomotor neurons), '
        'MYO10 (**muscle-validated**: myoblast fusion motor, DMD regeneration marker [eLife 2021]), '
        'KIF3C (cilia/neural transport), MYH7 (slow myosin heavy chain, cardiac/Type-I fiber). '
        'MYH7 and MYO10 confirm that the cluster captures genuine muscle motor biology; '
        'KIF26A/KIF21A reflect structural motor-domain homology annotations.'
    )
),

'fig_embedding_GO_0007204.png': (
    'Fig. S3c',
    'Embedding space: GO:0007204 Ca²⁺ Signaling (Type-B, P3-512 model, AUPRC = 0.4055). '
    'Critical evidence for Tissue Context Mixing (TCM).',
    (
        '**Left**: UMAP 2D projection of Phase-2 unified embeddings (P3-512 model, SP−, 512-dim). '
        'Blue dots = positive isoforms (n = 310); gray dots = negatives (subsampled). '
        'Star markers = cluster core (top genes: **F2R**, **LPAR6**, **CX3CR1**, APLNR, CMKLR1). '
        'The Ca²⁺ signaling cluster shows a **diffuse, heterogeneous distribution** — '
        'consistent with the GO term spanning diverse cellular contexts. '
        '**Right**: prediction score distribution (AUPRC = 0.4055 ≈ LR = 0.4138, p = 0.607). '
        '**Cluster core biology — Tissue Context Mixing evidence**: '
        'F2R/PAR1 (thrombin receptor, Gαq→Ca²⁺, **platelet/vascular**; no skeletal muscle role); '
        'LPAR6 (LPA receptor 6, GPCR, **lymphoid/skin/adipose**; hair follicle morphogenesis); '
        'CX3CR1 (fractalkine receptor, GPCR, **immune/CNS only**; HPA: absent from skeletal muscle). '
        '**All three centroid-proximal isoforms are non-muscle GPCRs**, directly confirming that '
        'the SwissProt GO:0007204 training corpus is dominated by non-muscle Ca²⁺ signaling contexts. '
        'Muscle-specific Ca²⁺ effectors (RYR1, CASQ2, CaMKII) are peripherally located in the cluster, '
        'consistent with SMSI = 1.63× (Ca²⁺ proteins are not muscle-enriched in SwissProt).'
    )
),

'fig_embedding_GO_0030017.png': (
    'Fig. S3d',
    'Embedding space: GO:0030017 Sarcomere Organization (Type-B, P3-512 model, AUPRC = 0.3553).',
    (
        '**Left**: UMAP 2D projection of Phase-2 unified embeddings (P3-512 model, SP−, 512-dim). '
        'Purple dots = positive isoforms (n = 452); gray dots = negatives (subsampled). '
        'Star markers = cluster core (top genes: **ANK3**, **ACTN2**, **PPP1R12B**, LDB3, MYPN). '
        '**Right**: prediction score distribution (AUPRC = 0.3553). '
        '**Cluster core biology**: '
        'ANK3/Ankyrin-G (dominant isoforms at axon initial segment/nodes of Ranvier; '
        'muscle-specific AnkG107 isoform anchors dystrophin at costamere [PMID:15953600]); '
        'ACTN2/α-Actinin-2 (**cardiac + skeletal muscle only**; Z-disc actin/titin cross-linker; '
        'HCM/DCM mutations [OMIM:102573]); '
        'PPP1R12B/MYPT2 (**striated muscle-specific** MLCP regulatory subunit; '
        'dephosphorylates cardiac MyRLC [PMID:38224947]); '
        'LDB3/ZASP (Z-disc LIM domain protein, cardiac/skeletal specific); '
        'MYPN/Myopalladin (sarcomere scaffold, cardiac/skeletal). '
        'Unlike GO:0007204, the sarcomere cluster core is predominantly muscle-specific, '
        'consistent with SMSI = 13.71× (sarcomere proteins are highly muscle-enriched). '
        'Lower AUPRC (0.355 vs. LR 0.561) reflects limited training signal under SP− and small positive set.'
    )
),

# ── Cluster core table ────────────────────────────────────────────────────────

'fig_cluster_core_table.png': (
    'Fig. S4',
    'Literature-validated cluster core representatives: top-3 isoforms nearest to positive centroid '
    'in Phase-2 embedding space, per GO term.',
    (
        'Cluster core isoforms were identified as the top-3 (by unique gene symbol) isoforms '
        'with minimum L2 distance to the positive centroid in the original (pre-UMAP) embedding space. '
        '**Red entries (⚠)**: genes whose dominant tissue context is non-muscle — '
        'provide direct evidence of **Tissue Context Mixing (TCM)** in SwissProt annotations. '
        '**Green entries (✓)**: genes with confirmed muscle-specific or muscle-relevant function. '
        'Literature was searched via PubMed, OMIM, Human Protein Atlas (HPA), '
        'and primary research articles for each gene (see text). '
        '**Type-A (GO:0006096, GO:0003774)**: PFKP/PFKL/PKM are glycolytic enzymes (function correct) '
        'but represent non-muscle isoforms (platelet, liver, fetal), reflecting partial TCM; '
        'MYO10 (GO:0003774) is confirmed muscle-relevant (myoblast fusion motor). '
        '**Type-B (GO:0007204)**: all three centroid genes (F2R, LPAR6, CX3CR1) are GPCRs '
        'with non-muscle tissue contexts — strongest TCM evidence. '
        '**Type-B (GO:0030017)**: ACTN2 and PPP1R12B are striated muscle-specific; '
        'ANK3 shows isoform-specific muscle expression (AnkG107 at costamere).'
    )
),

# ── Cluster structure ─────────────────────────────────────────────────────────

'fig_cluster_structure.png': (
    'Fig. S5',
    'Positive cluster internal structure: centroid distance distributions by proximity tier.',
    (
        '**x-axis**: L2 distance from the positive cluster centroid (original embedding space). '
        'Isoforms are partitioned into three tiers: '
        '**Core** (top-10 nearest to centroid); '
        '**Sub** (11th–40th nearest); '
        '**Peripheral** (remaining positives). '
        'Each violin shows the distribution of centroid distances within each tier, per GO term. '
        '**Interpretation**: '
        'Compact, well-separated distributions (low centroid distance, narrow violin) '
        'indicate a tightly organized positive cluster — characteristic of Type-A GO terms (GO:0006096). '
        'Wide, high-distance distributions indicate diffuse positive clusters — '
        'characteristic of Type-B GO terms (GO:0007204, GO:0030017), '
        'reflecting heterogeneous SwissProt training contexts. '
        '**Biological implication**: cluster compactness correlates with AUPRC — '
        'compact clusters are more discriminable from negatives; '
        'diffuse clusters reflect the tissue context mixing problem addressed by TCM framework.'
    )
),

# ── TBS / TCS analysis figures ────────────────────────────────────────────────

'tbs_fti_scatter.png': (
    'Fig. S6a',
    'TBS (Taxonomic Breadth Score) vs. FTI (Functional Transferability Index): 1D model failure.',
    (
        '**x-axis**: TBS = fraction of 6 biological kingdoms with ≥1 SwissProt-annotated protein '
        'for the GO term (computed from local SwissProt annotation file). '
        '**y-axis**: FTI = AUPRC(SP+, 256d) / AUPRC(SP−, 256d). '
        'FTI > 1: SwissProt transfer is beneficial; FTI < 1: SwissProt transfer is harmful. '
        '**Pearson r = 0.358** — TBS alone is insufficient to predict FTI. '
        '**Critical counterexample**: GO:0007204 and GO:0006096 share identical TBS = 0.667 '
        '(both annotated in 4 of 6 kingdoms) yet have opposite FTI values '
        '(0.626 vs. 1.874), demonstrating that taxonomic breadth alone cannot explain '
        'SwissProt transfer effectiveness. This motivates the addition of TCS as a second axis. '
        'Each point represents one of the 5 evaluated GO terms; '
        'point labels indicate GO term identity and SMSI value.'
    )
),

'tbs_kingdom_heatmap.png': (
    'Fig. S6b',
    'SwissProt annotation coverage by GO term and biological kingdom.',
    (
        '**Rows**: GO terms (5 evaluated). '
        '**Columns**: biological kingdoms (Bacteria, Archaea, Fungi, Invertebrate, Vertebrate, Plant). '
        '**Cell values**: number of SwissProt-annotated proteins with GO term × kingdom combination. '
        'Empty cells (0 proteins) are shown in white. '
        'Color intensity reflects annotation count (log-scale recommended for visualization). '
        '**TBS** (rightmost column): fraction of kingdoms with ≥1 annotated protein '
        '(= number of non-zero columns / 6). '
        'GO:0003774 (Motor Activity) has the broadest taxonomic coverage (TBS = 0.833, 5 kingdoms), '
        'reflecting the ancient evolutionary origin of cytoskeletal motors. '
        'GO:0030017 (Sarcomere) and GO:0006941 (Muscle Contraction) are restricted to '
        'Invertebrate + Vertebrate (TBS = 0.333), as sarcomere-type muscle organization '
        'is absent in plants and most fungi.'
    )
),

'tbs_tcs_scatter.png': (
    'Fig. S6c',
    '2D TBS × TCS framework: joint prediction of FTI from taxonomic breadth and tissue specificity.',
    (
        '**x-axis**: TBS (Taxonomic Breadth Score; 0–1). '
        '**y-axis**: TCS (Tissue Context Specificity) = mean Tau index (τ) of GO-positive proteins '
        'across 51 human tissues (Human Protein Atlas rna_tissue_consensus, N = 20,151 genes). '
        'τ = Σᵢ(1 − x̂ᵢ)/(N−1) where x̂ᵢ = normalized tissue expression; '
        'τ → 1: highly tissue-specific; τ → 0: ubiquitously expressed. '
        '**Bubble size**: proportional to FTI value. '
        '**Bubble color**: red = FTI > 1 (SP helpful); blue = FTI < 1 (SP harmful). '
        '**2D regression surface**: FTI = 14.631 + 60.237×TBS − 12.058×TCS − 75.350×(TBS×TCS); '
        '**R² = 0.992** (n = 5; analytical fit). '
        'The interaction term (β₃ = −75.350): when both TBS and TCS are high, '
        'FTI drops sharply — explained by muscle-specific functions (high TCS) being '
        'overwhelmed by cross-tissue SwissProt annotations (high TBS). '
        'SMSI annotations shown as secondary labels.'
    )
),

'tcs_smsi_barplot.png': (
    'Fig. S6d',
    'Tissue Context Specificity (TCS) and Skeletal Muscle Specificity Index (SMSI) per GO term.',
    (
        '**Top panel (TCS bar)**: mean Tau index (τ) of GO-positive proteins (HPA, 51 tissues). '
        'All 5 GO terms show high TCS (τ = 0.849–0.919), indicating that GO-positive proteins '
        'tend to be tissue-specifically expressed. '
        'TCS alone has Pearson r = −0.625 with FTI (negative correlation: '
        'higher tissue specificity → lower FTI, but n=5 p = 0.259, insufficient). '
        '**Bottom panel (SMSI bar)**: mean nTPM(skeletal muscle) / mean nTPM(all 51 tissues) '
        'for GO-positive proteins. '
        'SMSI > 1: GO-positive proteins are enriched in skeletal muscle relative to average tissue. '
        'GO:0030017 (Sarcomere) SMSI = 13.71× — sarcomere proteins are strongly muscle-enriched. '
        'GO:0007204 (Ca²⁺ Signaling) SMSI = 1.63× — Ca²⁺ signaling proteins show minimal '
        'muscle enrichment, reflecting the predominance of non-muscle Ca²⁺ signaling molecules '
        '(GPCRs, second messengers) in the SwissProt annotation corpus.'
    )
),

}

# ─── Rendering ────────────────────────────────────────────────────────────────

def wrap_legend(text, width=145):
    """Wrap legend text preserving **bold** markers."""
    # Simple line-wrap (bold/italic markers preserved for display as-is)
    wrapped = textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped

def add_legend(fname, fig_num, title, legend_text, dpi=150):
    fpath = os.path.join(OUT_DIR, fname)
    if not os.path.exists(fpath):
        print(f'  [SKIP] {fname} — not found')
        return

    img = mpimg.imread(fpath)
    img_h, img_w = img.shape[:2]
    aspect = img_w / img_h

    # Figure width in inches (fixed); image height proportional
    fig_w = 14.0
    img_panel_h = fig_w / aspect

    # Legend text rendering
    legend_body = wrap_legend(legend_text, width=150)
    n_lines = legend_body.count('\n') + 1
    # Estimate legend panel height
    legend_panel_h = max(1.6, 0.155 * n_lines + 0.5)

    fig_h = img_panel_h + legend_panel_h + 0.15

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # ── image panel ──
    ax_img = fig.add_axes([0, legend_panel_h / fig_h, 1.0, img_panel_h / fig_h])
    ax_img.imshow(img)
    ax_img.axis('off')

    # ── legend panel ──
    ax_leg = fig.add_axes([0, 0, 1.0, (legend_panel_h) / fig_h])
    ax_leg.set_facecolor('#fafafa')
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')

    # separator line
    ax_leg.axhline(y=0.97, color='#888888', lw=0.8, xmin=0.01, xmax=0.99)

    # figure number + title (bold)
    header = f'{fig_num}  |  {title}'
    ax_leg.text(
        0.012, 0.91, header,
        transform=ax_leg.transAxes,
        fontsize=9.0, fontweight='bold', color='#111111',
        va='top', ha='left',
        wrap=False,
    )

    # legend body (regular)
    # Replace **...** with unicode bold lookalike — matplotlib text renderer
    # doesn't support inline bold in a single text object, so we strip ** markers
    # and render the whole body in normal weight (paper legend convention).
    body_clean = legend_body.replace('**', '')

    ax_leg.text(
        0.012, 0.78, body_clean,
        transform=ax_leg.transAxes,
        fontsize=7.9, color='#222222',
        va='top', ha='left',
        linespacing=1.45,
        wrap=False,
    )

    plt.savefig(fpath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    sz = os.path.getsize(fpath) // 1024
    print(f'  [OK]   {fname:50s}  {sz:4d} KB')


# ─── Main ────────────────────────────────────────────────────────────────────
print('=' * 65)
print(' Adding publication legends to all figures')
print('=' * 65)

for fname, (fig_num, title, legend_text) in LEGENDS.items():
    add_legend(fname, fig_num, title, legend_text)

print()
print('Done.')
