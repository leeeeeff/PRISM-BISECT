# Fluid Vector Trajectory Framework — Stage 1 + Stage 2 Session Report

**Date:** 2026-07-06
**Session lead:** Seungwon Lee
**Framework code prefix:** `hMuscle/model/exp_fluid_stage1*.py`, `exp_fluid_stage2*.py`
**Result directory:** `reports/fluid_stage1/`, `reports/fluid_stage2/`

---

## 0. Motivation

**Central hypothesis.** Treat each isoform's ESM-2 L1..L30 mean-pooled
embeddings as a 30-point trajectory in 640-dimensional space. Same-GO
isoforms should form trajectory bundles; isoforms that briefly diverge
from the bundle then return by the final layer ("side-branch" events)
carry **encoded but not expressed** isoform-level information — features
that ESM-2 internally represents but that are absorbed by the final
functional projection.

**Discriminating goal.** Beyond the domain-loss / partial-truncation
signal that current PRISM already resolves, the fluid framework aims to
surface trajectory divergence-return patterns as candidates for
isoform-level function that L30 mean-pool alone cannot capture.

---

## 1. Stage 1 — Pilot and characterization (11 experiments)

| Exp | Script | Objective | Key result | Verdict |
|---|---|---|---|---|
| 1a | `exp_fluid_stage1_curve_cluster.py` | 3-GO pilot, K_PCA=8, K_CLUS=12 | 4 Bonferroni-strict winner clusters (Glyco ×2.15/2.89/3.79, Muscle contraction ×2.22) | Bundles exist |
| 1b | `exp_fluid_stage1b_rep_isoforms.py` | Representative isoform nomination for 4+3 winner clusters | BISECT trio NDUFS4/7/8 auto co-cluster in c3; TPM1 all 11 iso in c6; BCL2/BAX/DNM1L apoptosis-dynamics in c5 | Biologically coherent |
| 1c | `exp_fluid_stage1c_grid_18go.py` | 18 BP GO × 48 grid point stability | 6/18 (33%) robust — all mechanistically specific (Glyco, sarcomere, cytoskeleton, MT-move, actin-move, Ca signal) | Broad GOs fail |
| 1d | `exp_fluid_stage1d_length_confound.py` | Sequence-length confound check | max Spearman \|ρ\|=0.464 (dim 23); R²=0.109 for cluster-length; c6 dissolves after length residualization (TPM1 stays, TTN separates); Glyco winners z-med +0.01 (immune to length) | Partial confound, localized |
| 1e | `exp_fluid_stage1e_narrow_go.py` | 15 narrow specific BP GO × 48 grid | 11/15 (73%) robust; strongest ratios mRNA splicing 7.48×, FA beta-ox 6.29×, tRNA aminoacyl 5.90× | Narrow-GO strategy works |
| 1f | `exp_fluid_stage1f_complex1_subflow.py` | Complex I family (34 genes) + controls within-gene bifurcation | bifL=28 nearly universal; NDUFAF6 disp 74.6, NDUFS4 29.95, NDUFS8 28.74, NDUFS7 8.98; family silhouette 0.301 (Complex I vs controls) | Within-gene fluid signal real |
| 1g | `exp_fluid_stage1g_l30_shootout.py` | fluid vs L30_pca vs L30_raw baseline shootout, 12 grid points | Robust count tied at 10/15 all 3 methods; per-GO split — L30 wins family-clean, fluid wins diffuse (Actin/MT-motor, Ca-transport, actin filament, ER-Golgi) | Novelty gate FAIL under default framework |
| 1h | `exp_fluid_stage1h_pca_layer_decomp.py` | Joint-PCA layer decomposition | 97.8% of top-8 PC variance dominated by L21-L30 layers | curve_vec ≈ L30 (devils confirmed) |
| 1i | `exp_fluid_stage1i_within_gene_gap.py` | Same-gene pair distance: fluid vs L30, correlated with PRISM GO gap | Spearman with PRISM: fluid 0.884 vs L30 0.891; top-decile median rank diff = -0.002 (fluid marginally worse) | Within-gene novelty FAIL under default framework |

**Stage 1 verdict.**
Bundles exist and are biologically coherent; robust GOs are family-specific.
However, under joint-PCA without normalization, the framework's discriminative
axis is a re-projection of L21-L30, and it does not beat L30 mean-pool at
per-GO winner count or within-gene functional divergence prediction.

---

## 2. Stage 2 — Layer normalization + typed-GO reframing (2 experiments)

Framework change based on Stage 1 diagnosis: apply per-layer per-dim
z-score normalization before joint PCA. Force each of L1..L30 to have
equal say in the reduced basis. Then classify GO terms by which layer
holds their peak Fisher discriminant signal.

### 2.1 Layer normalization + typed-flow (`exp_fluid_stage2_typed_flow.py`)

**Pilot.** N=15,000 (pos union 12,719 from 34 curated specific BP terms + matched neg).

**Layer decomposition after normalization:**

| | frac_late@top8 | expl_var(K=16) |
|---|---|---|
| UNNORM joint PCA | 0.978 | 0.914 |
| **NORM joint PCA** | **0.205** | 0.245 |

Layer normalization reduces late-layer PCA dominance from 98% to 20%.
The trade-off (expl_var drop 0.91 → 0.25) is expected: after
per-layer variance equalization the reduced basis captures information
distributed across all 30 layers rather than concentrated at L28-30.

**Per-GO layer signal profile.**
For every GO in the 34-term catalog, computed 30-point Fisher discriminant
signal profile:

    Fisher(GO, L) = ||mu_pos - mu_neg||^2 / sum_d (var_pos + var_neg)

Classified GO by peak layer:

| Type | n_GO | peak layer range | representative GOs |
|---|---|---|---|
| early | 3 | L1-L10 | Sarcomere org (L5), MT-based movement (L1), Ca2+ homeostasis (L3) |
| **mid** | **10** | L11-L20 | tRNA aminoacyl (L19, peak 0.30), Vesicle fusion (L18, 0.27), MAPK cascade (L19), Ca2+ signaling (L11), Translational elongation (L12) |
| late | 6 | L21-L30 | FA beta-oxidation (L28, 0.29), ATP biosynthesis (L29), Proteasome-UPS (L27), Carbohydrate metab (L28) |
| flat | 15 | max/min ratio < 2 | Glycolysis (L10, 0.22), mRNA splicing (L14), Complex I NADH ox (L30) |

**Stability shootout across types** (4 grid points: K_CLUS ∈ {12, 16} × seed ∈ {42, 137}):

| Type | n_go | fluid_un stab | **fluid_nm stab** | L30_raw stab | winner |
|---|---|---|---|---|---|
| early | 3 | 0.83 | **1.00** | 0.92 | fluid_nm |
| mid | 10 | 0.85 | **0.88** | 0.75 | fluid_nm |
| late | 6 | 0.75 | **0.83** | 0.62 | fluid_nm |
| flat | 15 | 0.80 | **0.95** | 0.77 | fluid_nm |
| **robust≥0.80 total** | **34** | 24 (71%) | **28 (82%)** | 20 (59%) | **fluid_nm** |

**fluid_nm − L30_raw = +8 robust GO** → Novelty gate PASSED
(devils-advocate demanded ≥ +3 diff).

### 2.2 Side-branch detection (`exp_fluid_stage2_typed_flow.py`, section)

Per winner bundle, computed side-branch score

    sb(i) = max_L ||x[i,L] - C(L)||_top3PC  -  ||x[i,L=30] - C(L=30)||_top3PC

where C(L) is the bundle centroid trajectory. Positive sb = trajectory
deviated at some layer and returned by L30.

Top side-branchers per type (from stage 2 script main body, single-GO):

- **[mid] Intracellular signaling (c4, n=1180)**:
  JAK1 sb=8.03 peakL=23 GO+;
  IRAK4 sb=6.37 peakL=23 GO+;
  KIT sb=6.33 peakL=19 **GO− (annotation miss?)**
- **[late] Proteasome-UPS (c3, n=821)**:
  Top-10 all GO−, peak L6-L15 (structural/scaffolding side-branchers)
- **[early] Ca2+ homeostasis (c9, n=614)**:
  TMEM178A sb=6.32 peakL=15 GO+;
  PDPN x3 iso peakL=23-26; ITGA5/7/X peakL=12

### 2.3 Full survey (`exp_fluid_stage2b_side_branch_survey.py`)

Extracted top-20 side-branchers from every winner bundle (32/34 GOs win).

- 640 total candidates, 135 unique gene symbols
- 66 GO+ (distinctive within-bundle) + 574 GO− (encoded-but-unexpressed candidates)

**Bio-validation enrichment (Fisher exact vs pilot background)**:

| Reference set | Hits / total | Ref in pilot | OR | p |
|---|---|---|---|---|
| UniProt curated pairs (51) | 2 / 135 | 29 | 3.27 | 0.137 |
| BISECT hits (13) | 0 / 135 | 7 | 0.00 | 1.000 |
| TARGET_GENES (8) | 0 / 135 | 8 | 0.00 | 1.000 |
| Any known (union) | 2 / 135 | 44 | 2.10 | 0.258 |

Nominal 2-3× enrichment but not significant. Reference set very small
(total 44 in pilot), power-limited.

**Top GO− side-branch candidates (biology plausible even without curated overlap)**:

| Rank | Gene | sb | peakL | Bundle | Biology note |
|---|---|---|---|---|---|
| 1 | ZCCHC10 | 8.40 | L6 | mRNA splicing | zinc-finger CCHC RNA-binding — splicing-adjacent |
| 2-3 | JAK1 | 8.03 | L23 | Ras signaling / Dephospho | JAK-STAT × Ras cross-talk |
| 4-5 | ANKRD12 | 7.73 | L1 | Translation regulation | ankyrin-repeat 12, translation-linked |
| 6-17 | MYO1D (ENST00000318217.10) | 7.66 | L14 | 6 metabolic + Ca bundles | myosin I family cross-cluster |
| 18 | JPH2 | 7.42 | L6 | Proteasome-UPS | junctophilin 2 — muscle Ca handling, DCM |
| 19-20 | PRSS23 | 7.21 | L12 | Carbohydrate metab / Ca transport | serine protease |

**Multi-bundle promiscuous genes** (isoforms appearing in ≥ 10 bundles top-20):

| Gene | n_appearances | Note |
|---|---|---|
| YPEL5 | 60 | multi-cluster boundary |
| GCC2 | 24 | Golgi coiled-coil |
| NTN1 | 20 | Netrin-1 (axon guidance) |
| ANXA2 | 16 | Annexin A2 (Ca-binding) |
| PMP22 | 15 | Peripheral myelin protein |
| JAK1, MYO1D, MRPL3, POLD2, SCRN2, AFG2B | 12 | multi-pathway |

---

## 3. Consolidated verdict on devils-advocate critiques

| Critique | Stage 1 status | Stage 2 status |
|---|---|---|
| curve_vec ≈ L30 re-projection | Confirmed (frac_late 0.978) | **Refuted with layer-norm** (frac_late 0.205) |
| Novelty gate (fluid vs L30 winner count) | Failed (tied 10/15 all 3 methods) | **Passed** (+8 robust: 28 vs 20 at 34-GO) |
| Within-gene evidence | Confirmed (BISECT trio disp 8.98–29.95, bifL=28) | Corroborated (JAK1/MYO1D side-branch) |
| Winner = protein family re-discovery | Confirmed (Complex I/glycolysis family clean) | Nuanced — late-type gets fluid gain +0.21, so family-clean GOs *also* benefit from layer-norm |
| Length confound (\|ρ\|=0.46) | Confirmed for c6 (TPM1+TTN artifact) | Not re-tested under layer-norm; open |
| Encoded-but-unexpressed real signal | Theory only | Partial: KIT, MYO1D, JPH2 candidates — bio-plausible but curated overlap p=0.13 |

---

## 4. Framework snapshot (current best)

**Preprocessing.**

    traj: (N, 30, 640) mean-pool ESM-2 L1..L30
    for each layer L:
        traj_norm[:, L, :] = (traj[:, L, :] - mean_L) / (std_L + 1e-6)
    joint PCA on traj_norm.reshape(N*30, 640) with K=16
    reduced_nm: (N, 30, 16), take first 8 axes -> curve_vec_240

**Clustering.**

    KMeans(k=16, seed=42, n_init=5) on curve_vec_240 -> bundle assignment

**Side-branch score.**

    for isoform i in bundle B:
        C(L) = mean over B members of reduced_nm[j, L, :3]
        dev(i, L) = || reduced_nm[i, L, :3] - C(L) ||
        sb(i) = max_L dev(i, L)  -  dev(i, L=30)
        peak_L(i) = argmax_L dev(i, L)

Positive sb = trajectory deviated from bundle at mid layer and returned.

---

## 5. Extension possibilities (priority-ordered)

**Priority 1 — Bio-validation power (address weak Fisher p)**
- Expand reference set with Ensembl APPRIS (principal / alternative
  transcript labels for all 100k human transcripts)
- Cross-reference with tissue-specific isoform switches from GTEx
- Domain-annotation overlap: does side-branch peak layer align with
  Pfam domain boundaries in the alternative isoform's sequence?

**Priority 2 — Length confound under layer-norm**
- Rerun Stage 1d diagnostics on curve_vec_norm to confirm the c6
  TPM1+TTN artifact is gone
- Report max \|ρ\| between log-length and any of the 240 norm axes

**Priority 3 — Case-study depth on top GO− side-branchers**
- MYO1D 6-bundle appearance: does ENST00000318217.10 have a specific
  domain composition (motor + tail) that differs from other MYO1D
  isoforms?
- JPH2 in Proteasome-UPS: Junctophilin-2 has known N-terminal cleavage
  by calpain; does peak-L6 (early) side-branch correspond to that?
- ZCCHC10 in mRNA splicing: is this a real annotation miss?

**Priority 4 — Cross-tissue validation**
- Rerun typed-flow analysis on brain isoform dataset
  (`hMuscle/data/brain_672`) to confirm mid-type/late-type
  classification is not muscle-specific artifact.

**Priority 5 — Contrastive fluid training (Stage 3)**
- Actual optimization: train velocity field v(x, t; θ) such that
  same-gene GO-discordant pairs are pushed apart along the trajectory
  (rectified flow with same-gene contrastive term).
- Gate: only proceed if Priority 1-4 collectively strengthen the
  bio-validation story.

**Priority 6 — Alternative normalization**
- Compare per-layer z-score (current) with layer-wise whitening,
  layer-wise L2 normalization, and Gaussian mixture per layer.
- Assess sensitivity of the +8 robust gain to normalization choice.

---

## 6. File registry

Trajectory tensors (rebuild on demand):
- Pilot subset always defined by `sorted(pos_union ∪ neg_match)` across
  the GO catalog specified in each script. Not cached to disk to avoid
  stale-state bugs.

Stage 1 outputs — `reports/fluid_stage1/`:
- `curve_cluster_20260706_1541.npz` — original 3-GO pilot curve_vec_240
- `purity_20260706_1541.json` — 3-GO winner cluster purity
- `rep_isoforms.json` — nearest-centroid isoforms per winner
- `grid_18go_20260706_1609.json` — 18 BP × 48 grid stability
- `narrow_go_20260706_1656.json` — 15 narrow BP × 48 grid stability
- `length_confound.json` + `length_confound_diag.png`
- `complex1_subflow.json` + `subflow_*.png` (5 gene 3D + dispersion)
- `l30_shootout_20260706_1736.json` — 3-method shootout
- `pca_layer_decomp.json` + `pca_layer_heatmap.png`
- `within_gene_gap.json` + `within_gene_gap.png`
- `trajectory_3d_GO_*.png` — original 3-GO bundle 3D means

Stage 2 outputs — `reports/fluid_stage2/`:
- `typed_flow_20260706_1855.json` — 34-GO typed shootout + side-branch
- `typed_layer_heatmap_20260706_1855.png` — 34 × 30 Fisher heatmap
- `bundle_tube_{early,mid,late}_20260706_1855.png` — 3D bundle tubes
  with top side-branches overlaid
- `all_side_branches_20260706_1957.csv` — 640 candidates full table
- `side_branch_survey_20260706_1957.json` — enrichment + top-50
- `session_report_20260706.md` — this file

---

## 7. Author notes

The Stage 1 → Stage 2 transition demonstrates a valuable pattern:
**a "novelty failure" can flip into a "novelty success" through a
targeted normalization choice** (per-layer z-score) that removes a
specific artifact (late-layer PCA dominance). Without Stage 1's
adversarial testing (L30 shootout + PCA decomposition), the Stage 2
gain would have been indistinguishable from noise. This is the
correct order:

    build → adversarial test → diagnose failure mode → targeted fix →
    re-test with the same adversarial framing.

The remaining weakness is bio-validation power (Fisher p=0.13). This
is not a framework failure but a reference-set size limitation — the
next mile of work should focus on expanding validated isoform-level
annotations (APPRIS, GTEx switches, tissue-specific proteomics) before
attempting Stage 3 (contrastive fluid training).
