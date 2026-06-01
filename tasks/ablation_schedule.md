# Ablation Schedule: v6g → v7b Experiments

**Purpose:** Systematic validation of each architectural change with clear acceptance/falsification criteria.

**Evaluation Protocol (per [R9.1]):**
- **Primary Metric:** AUPRC (sparse GO terms, positive < 50)
- **Secondary Metric:** AUROC (reference only)
- **Statistical Test:** Bootstrap CI (n=1000), Δ AUPRC must be > 0.02 for "improvement" claim
- **GO Term Set:** 5 core terms (GO:0006096, GO:0006412, GO:0006936, GO:0022900, GO:0003774)

---

## Experiment 0: Baseline Validation (v6f → v6f-replicate)

### Hypothesis
v6f results are reproducible with same random seed.

### Variant
Run v6f again with `np.random.seed(42)` fixed (already set in v6f line 622).

### Primary Metric
AUPRC per GO term (5 terms).

### Acceptance Criterion
- Replicated AUPRC within ±0.01 of original v6f for all 5 GO terms
- If not: random seed is NOT controlling all randomness → fix before v7 experiments

### Falsification
If any GO term AUPRC differs by > 0.02 → abort, fix non-determinism first.

### Estimated Runs
2 runs (original + replicate).

### Dependencies
None (prerequisite for all ablations).

---

## Experiment 1: v6g Bias Score Diagnostic

### Hypothesis
DomainDelta branch reduces gene-level bias (bias_score > 0.3).

### Variant
**v6g** (v6f + compute_bias_score in evaluation.py).

### Primary Metric
- bias_score (from evaluation.py)
- AUPRC (unchanged from v6f, this is diagnostic only)

### Acceptance Criterion
- bias_score > 0.3 for ≥ 4/5 GO terms → DomainDelta is effective
- bias_score > 0.5 for ≥ 2/5 GO terms → strong isoform-specificity

### Falsification
If bias_score < 0.3 for ≥ 3/5 GO terms:
- **Conclusion:** Gene-level dominance is severe
- **Action:** v7a MUST add adversarial gene-decorrelation (Gradient Reversal Layer [R2.4])

### Estimated Runs
1 run (5 GO terms in parallel).

### Dependencies
Experiment 0 (baseline validation).

---

## Experiment 2a: v7a-SupCon (Supervised Contrastive Baseline)

### Hypothesis
Supervised Contrastive Loss (all positives, no prototypes) improves over Triplet Loss.

### Variant
**v7a-SupCon:** Replace Phase 1 Triplet with SupCon [R3.3]:
```
L = Σ_i (-1/|P(i)|) · Σ_{p∈P(i)} log[exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ)]
```
where P(i) = all positives in batch, τ=0.1.

### Primary Metric
Δ AUPRC (v7a-SupCon vs v6f), per GO term.

### Acceptance Criterion
- Mean Δ AUPRC > +0.02 across 5 GO terms (bootstrap CI lower bound > 0)
- Improves ≥ 3/5 GO terms individually

### Falsification
If mean Δ AUPRC < 0:
- **Conclusion:** SupCon is not better than Triplet for this task
- **Action:** Investigate triplet margin or hard negative mining instead

### Estimated Runs
3 runs (mean ± std per GO term).

### Dependencies
Experiment 0.

---

## Experiment 2b: v7a-Proto-k1 (Single Prototype)

### Hypothesis
Single prototype (k=1) is sufficient for Type-A GO terms, but insufficient for Type-B.

### Variant
**v7a-Proto-k1:** Force k=1 prototype (mean of all positives), regardless of GO term type.

### Primary Metric
- AUPRC per GO term
- Compare Type-A (sep_ratio ≥ 1.15) vs Type-B (sep_ratio < 1.15)

### Acceptance Criterion
**Type-A GO terms (e.g., GO:0006096):**
- v7a-Proto-k1 AUPRC ≥ v7a-SupCon AUPRC - 0.01 (no significant loss)

**Type-B GO terms (e.g., GO:0003774, GO:0006941):**
- v7a-Proto-k1 AUPRC < v7a-Proto-kN AUPRC (k>1 is better for heterogeneous)

### Falsification
If v7a-Proto-k1 outperforms v7a-Proto-kN on Type-B terms:
- **Conclusion:** Gap statistic k-selection is wrong, or Type-B classification is incorrect
- **Action:** Revise sep_ratio threshold or disable multi-prototype

### Estimated Runs
3 runs.

### Dependencies
Experiment 2a (need SupCon baseline).

---

## Experiment 2c: v7a-Proto-kN (Multi-Prototype, Gap Statistic)

### Hypothesis
Gap statistic selects optimal k for Type-B GO terms, improving AUPRC over k=1.

### Variant
**v7a-Proto-kN:** Use `determine_k_gap_statistic()` to select k ∈ [1, 5] per GO term.

### Primary Metric
- AUPRC per GO term
- k selected (log during Phase 1)
- Prototype diversity loss L_diversity (should decrease during training)

### Acceptance Criterion
**Type-B GO terms:**
- v7a-Proto-kN AUPRC > v7a-Proto-k1 AUPRC by ≥ 0.02 (for ≥ 2/3 Type-B terms)
- Gap statistic selects k > 1 for Type-B terms

**Type-A GO terms:**
- v7a-Proto-kN AUPRC ≥ v7a-Proto-k1 AUPRC - 0.01 (no harm)

**Training Stability:**
- L_diversity (epoch 15) < L_diversity (epoch 1) for all GO terms (prototypes separate)

### Falsification
If v7a-Proto-kN does not improve over v7a-Proto-k1 for Type-B terms:
- **Conclusion:** Multi-prototype hypothesis is false for this task
- **Action:** Use SupCon or Triplet instead, abandon prototype approach

### Estimated Runs
5 runs (1 per GO term, to measure k selection).

### Dependencies
Experiments 2a, 2b (need k=1 and SupCon baselines).

---

## Experiment 3a: v7b-LP Sensitivity (acorde α Grid Search)

### Hypothesis
acorde co-expression network enables beneficial label propagation (vs v6f where LP was disabled).

### Variant
**v7b-LP:** Use acorde network (ci_threshold=0.3), test α ∈ [0.1, 0.2, 0.3, 0.5].

### Primary Metric
Δ AUPRC (v7b-LP best α vs v7a-Proto-kN) per GO term.

### Acceptance Criterion
- LP improves AUPRC on ≥ 3/5 GO terms (by ≥ 0.01 each)
- Best α is consistent across GO terms (≥ 3/5 select same α)

### Falsification
If LP improves < 3/5 GO terms:
- **Conclusion:** n_samples=24 is insufficient for reliable co-expression, even with bootstrap CI
- **Action:** Disable LP in production (α=0.0), flag as "requires > 50 samples for LP"

### Estimated Runs
5 runs (1 per GO term, 4 α values each → 20 sub-runs total).

### Dependencies
- Experiment 2c (need v7a-Proto-kN baseline)
- acorde network preprocessing (4 hours)

---

## Experiment 3b: v7b-LP Network Sensitivity (ci_threshold Grid Search)

### Hypothesis
ci_threshold=0.3 is optimal (edge reliability vs density tradeoff).

### Variant
**v7b-LP:** Test ci_threshold ∈ [0.2, 0.3, 0.4, 0.5] with best α from Exp 3a.

### Primary Metric
- AUPRC per GO term (for each ci_threshold)
- Network edge density (from stats files)

### Acceptance Criterion
- ci_threshold=0.3 gives best mean AUPRC across ≥ 3/5 GO terms
- If not, select optimal ci_threshold and document

### Falsification
If all ci_thresholds perform similarly (Δ AUPRC < 0.01):
- **Conclusion:** LP benefit is insensitive to network density
- **Action:** Use ci_threshold=0.3 (default)

### Estimated Runs
4 runs (4 ci_threshold values, 5 GO terms each → 20 sub-runs).

### Dependencies
Experiment 3a (need optimal α).

---

## Experiment 4: Overall v6f → v7b Comparison

### Hypothesis
v7b (Prototype + acorde LP) significantly improves over v6f.

### Variant
**Final comparison:** v6f vs v7b (with best α, ci_threshold from Exp 3).

### Primary Metric
- Macro-AUPRC (mean across 5 GO terms)
- Per-term AUPRC (all 5 terms)
- Bootstrap CI (n=1000) on mean AUPRC

### Acceptance Criterion
- v7b mean AUPRC > v6f mean AUPRC by ≥ 0.03 (bootstrap CI lower bound > 0.01)
- v7b improves ≥ 4/5 GO terms individually (by ≥ 0.02 each)
- No GO term degrades by > 0.05

### Falsification
If v7b mean AUPRC improvement < 0.02:
- **Conclusion:** v7 changes do not provide meaningful benefit
- **Action:** Revert to v6f, investigate alternative axes (e.g., IRM [R2.2], Spectral Norm [R7.3])

### Estimated Runs
10 runs (5 v6f + 5 v7b, for robust CI).

### Dependencies
All previous experiments.

---

## Experiment 5: Dataset A Generalization (Optional, Post-v7)

### Hypothesis
v7b performance generalizes to Shaw et al. 2019 benchmark (Dataset A).

### Variant
**v7b-DatasetA:** Run v7b on 39,375 isoforms, 96 GO slim terms.

### Primary Metric
- Macro-AUPRC (mean across 96 GO terms)
- Compare to Shaw et al. reported baseline

### Acceptance Criterion
- v7b macro-AUPRC ≥ Shaw baseline - 0.02
- If v7b > Shaw baseline → strong generalization claim

### Falsification
If v7b macro-AUPRC < Shaw baseline - 0.05:
- **Conclusion:** v7b is overfitted to hMuscle, does not generalize
- **Action:** Regularization (dropout, L2) or domain adaptation required

### Estimated Runs
3 runs (due to large dataset, each run ~6 hours).

### Dependencies
Dataset A preprocessing (18 hours).

---

## Ablation Summary Table

| Exp | Variant | Purpose | Primary Metric | Runs | Time (GPU hours) | Dependencies |
|-----|---------|---------|----------------|------|------------------|--------------|
| 0 | v6f-replicate | Reproducibility | AUPRC | 2 | 4 | - |
| 1 | v6g | Bias diagnostic | bias_score | 1 | 2 | Exp 0 |
| 2a | v7a-SupCon | SupCon baseline | Δ AUPRC | 3 | 6 | Exp 0 |
| 2b | v7a-Proto-k1 | Single prototype | AUPRC | 3 | 6 | Exp 2a |
| 2c | v7a-Proto-kN | Multi-prototype | AUPRC | 5 | 10 | Exp 2a,2b |
| 3a | v7b-LP (α) | LP α tuning | Δ AUPRC | 5 | 10 | Exp 2c |
| 3b | v7b-LP (ci) | LP network tuning | AUPRC | 4 | 8 | Exp 3a |
| 4 | v6f vs v7b | Final comparison | Macro-AUPRC | 10 | 20 | All above |
| 5 | v7b-DatasetA | Cross-dataset | Macro-AUPRC | 3 | 18 | Dataset A prep |

**Total GPU Time:** ~84 hours (~3.5 days on 2 GPUs)

---

## Execution Plan (Batched for Efficiency)

### Batch 1 (Day 1): Baselines
```bash
# Run in parallel on GPU 0 and GPU 1
GPU=0: Exp 0 (v6f-replicate GO:0006096, GO:0006412, GO:0006936)
GPU=1: Exp 0 (v6f-replicate GO:0022900, GO:0003774)
# Then Exp 1 (v6g) on all 5 GO terms
```

### Batch 2 (Day 2): Prototype Ablations
```bash
GPU=0: Exp 2a (v7a-SupCon) all GO terms
GPU=1: Exp 2b (v7a-Proto-k1) all GO terms
# Analyze, then run Exp 2c based on results
```

### Batch 3 (Day 3): Label Propagation
```bash
# Preprocessing (CPU): build acorde networks (4 hours)
GPU=0: Exp 3a (α grid) GO:0006096, GO:0006412, GO:0006936
GPU=1: Exp 3a (α grid) GO:0022900, GO:0003774
# Then Exp 3b (ci grid) on best α
```

### Batch 4 (Day 4): Final Comparison
```bash
# Run v6f and v7b with best hyperparameters (10 runs total)
GPU=0,1: Exp 4 (5 GO terms × 2 variants)
```

### Batch 5 (Optional, Days 5-6): Dataset A
```bash
# Preprocessing: 18 hours (CPU + GPU for ESM-2)
# Inference: Exp 5 (v7b on 96 GO terms)
```

---

## Decision Tree (Adaptive Ablation)

```
START → Exp 0 (reproducibility)
  ↓
  PASS? → Exp 1 (bias_score)
  ↓         ↓
  FAIL      bias_score < 0.3 for ≥3/5 terms?
  ↓         ↓
  FIX       YES → Add Gradient Reversal to v7a (modify plan)
  SEED      ↓
            NO → Proceed to Exp 2a
            ↓
        Exp 2a (SupCon)
          ↓
          PASS (Δ AUPRC > 0.02)?
          ↓               ↓
          YES             NO → Investigate Triplet instead
          ↓                   (halt prototype track)
        Exp 2b (Proto-k1)
          ↓
        Exp 2c (Proto-kN)
          ↓
          PASS (k>1 helps Type-B)?
          ↓               ↓
          YES             NO → Use k=1 for all
          ↓
        Exp 3a (LP α)
          ↓
          PASS (improves ≥3/5)?
          ↓               ↓
          YES             NO → Disable LP (α=0)
          ↓
        Exp 3b (LP ci)
          ↓
        Exp 4 (Final v6f vs v7b)
          ↓
          PASS (Δ AUPRC > 0.03)?
          ↓               ↓
          YES → DEPLOY    NO → REVERT to v6f
```

---

## Logging Requirements (per experiment)

Each run MUST log:
1. **Version tag** (e.g., v7a-SupCon)
2. **GO term ID**
3. **Random seed** (np.random.seed, tf.random.seed)
4. **Hyperparameters** (α, ci_threshold, k, tau, etc.)
5. **Training curves** (loss, AUPRC per epoch)
6. **Final metrics** (AUPRC, AUROC, bias_score if applicable)
7. **Embedding quality** (silhouette, sep_ratio, linear_AUROC)
8. **Runtime** (wall time, GPU memory peak)

**Log format:** CSV (one row per run) + detailed training log (TXT).

**Example CSV:**
```
version,go_term,seed,k,alpha,ci_threshold,auprc,auroc,bias_score,runtime_sec
v7a-Proto-kN,GO:0006096,42,1,0.0,0.0,0.6542,0.8123,0.42,1834
```

---

## Falsification Summary (What Would Make Us Abandon v7?)

| Experiment | Falsification Trigger | Consequence |
|------------|----------------------|-------------|
| Exp 0 | AUPRC non-reproducible (> 0.02 diff) | Halt, fix randomness |
| Exp 1 | bias_score < 0.3 for ≥3/5 terms | Add adversarial loss to v7a |
| Exp 2a | SupCon Δ AUPRC < 0 | Abandon contrastive, keep Triplet |
| Exp 2c | Proto-kN no better than k=1 | Use k=1 only, simpler |
| Exp 3a | LP improves < 3/5 terms | Disable LP (α=0), insufficient data |
| Exp 4 | v7b Δ AUPRC < 0.02 | **REVERT TO v6f**, v7 not viable |

**Critical Falsification (Exp 4):** If v7b does not improve mean AUPRC by ≥ 0.02 with bootstrap CI, the entire v7 architecture is rejected. We would then explore alternative axes (IRM, domain adaptation, synthetic augmentation [R5.1]).

---

## Output Artifacts (per experiment)

1. **Results CSV:** `ablation_results.csv` (all runs, all metrics)
2. **Per-GO Reports:** `GO_XXXXXX_v7a_report.txt` (training log, metrics, embedding stats)
3. **Comparison Plots:** `v6f_vs_v7b_auprc_comparison.png` (bar plot, 5 GO terms)
4. **Statistical Tests:** `bootstrap_ci_v6f_v7b.txt` (CI lower/upper bounds)
5. **Final Recommendation:** `v7_ablation_conclusion.md` (deploy v7b? revert to v6f?)

---

**Estimated Total Time:** 4-5 days (parallel GPU execution).

**Deliverable to User:** After Batch 4 completes, provide `v7_ablation_conclusion.md` with deployment recommendation.
