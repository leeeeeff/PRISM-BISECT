# v7 Implementation Plan
**Priority Order:** D1.1 diagnostic → v6g (bias_score) → v7a (Prototype Contrastive) → v7b (acorde LP) → Dataset A

---

## Phase 0: Prerequisites (Estimated: 2 hours)

### 0.1 Verify Diagnostic Report Findings
**Files to check:**
```bash
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/
# Check Phase 1.5 embedding files for GO:0006941
python3 -c "
import numpy as np
ph0 = np.load('GO_0006941/v6e_integrated_20260415/v6e_integrated_phase0_initial_untrained_embeddings.npy')
ph15 = np.load('GO_0006941/v6e_integrated_20260415/v6e_integrated_phase1_5_linear_probing_embeddings.npy')
print('Phase 0 mean:', ph0.mean(), 'std:', ph0.std())
print('Phase 1.5 mean:', ph15.mean(), 'std:', ph15.std())
print('Embeddings changed:', not np.allclose(ph0, ph15, atol=1e-6))
"
```

**Expected Outcome:**
- If embeddings are **identical** → freeze works, sep_ratio artifact confirmed
- If embeddings **differ** → freeze is broken (contradicts D1.1 PASS verdict, escalate)

### 0.2 Create Backup
```bash
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model/
cp v6f_integrated_full_model.py v6f_integrated_full_model_backup_20260428.py
```

---

## Phase 1: v6g — Add Bias Score Diagnostic (Estimated: 3 hours)

**Purpose:** Quantify gene-level bias to validate DomainDelta effectiveness before v7a changes.

### 1.1 Modify `evaluation.py`

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/evaluation.py`

**Add function after line 54:**

```python
def compute_bias_score(df):
    """
    Compute gene-level bias score.
    
    Args:
        df: DataFrame with columns [GeneID, IsoformID, Score, Label]
    
    Returns:
        bias_score: float in [0,1]
          0 = predictions are gene-level (isoform adds no info)
          1 = predictions are isoform-specific (gene adds no info)
    """
    from scipy.stats import entropy
    import numpy as np
    
    # H(y | isoform)
    iso_entropy_terms = []
    for iso in df['IsoformID'].unique():
        iso_df = df[df['IsoformID'] == iso]
        if len(iso_df) == 0:
            continue
        p_iso = len(iso_df) / len(df)
        labels = iso_df['Label'].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        if n_pos + n_neg > 0:
            # Binary entropy
            p_1 = n_pos / (n_pos + n_neg + 1e-10)
            p_0 = n_neg / (n_pos + n_neg + 1e-10)
            local_H = -p_1 * np.log2(p_1 + 1e-10) - p_0 * np.log2(p_0 + 1e-10)
            iso_entropy_terms.append(p_iso * local_H)
    
    H_y_given_iso = sum(iso_entropy_terms)
    
    # H(y | gene) - same pattern
    gene_entropy_terms = []
    for gene in df['GeneID'].unique():
        gene_df = df[df['GeneID'] == gene]
        if len(gene_df) == 0:
            continue
        p_gene = len(gene_df) / len(df)
        labels = gene_df['Label'].values
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        if n_pos + n_neg > 0:
            p_1 = n_pos / (n_pos + n_neg + 1e-10)
            p_0 = n_neg / (n_pos + n_neg + 1e-10)
            local_H = -p_1 * np.log2(p_1 + 1e-10) - p_0 * np.log2(p_0 + 1e-10)
            gene_entropy_terms.append(p_gene * local_H)
    
    H_y_given_gene = sum(gene_entropy_terms)
    
    bias_score = 1.0 - (H_y_given_iso / (H_y_given_gene + 1e-10))
    
    return {
        'bias_score': bias_score,
        'H_y_given_iso': H_y_given_iso,
        'H_y_given_gene': H_y_given_gene
    }
```

**Modify `calculate_metrics()` to call `compute_bias_score()`:**

```python
def calculate_metrics(df, k_list=[1, 3]):
    # ... existing code ...
    
    # Add bias score
    bias_metrics = compute_bias_score(df)
    metrics.update(bias_metrics)
    
    return metrics
```

### 1.2 Add Embedding Stability Diagnostic to v6g

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/model/v6g_integrated_full_model.py`

**Create v6g by copying v6f:**
```bash
cp v6f_integrated_full_model.py v6g_integrated_full_model.py
```

**Add after Phase 1.5 completion (after line 891 in v6f):**

```python
# [v6g] Embedding stability diagnostic
print("\n[v6g-diagnostic] Phase 1.5 Embedding Stability Check")
ph1_emb_path = os.path.join(SAVE_DIR, '{}_phase1_triplet_only_embeddings.npy'.format(VER_TAG))
ph15_emb_path = os.path.join(SAVE_DIR, '{}_phase1_5_linear_probing_embeddings.npy'.format(VER_TAG))
if os.path.exists(ph1_emb_path):
    ph1_emb = np.load(ph1_emb_path)
    ph15_emb = emb_by_phase['Ph1.5(linear)']
    
    emb_diff = np.abs(ph1_emb - ph15_emb)
    print("  Max embedding change: {:.6f}".format(emb_diff.max()))
    print("  Mean embedding change: {:.6f}".format(emb_diff.mean()))
    print("  Embeddings identical (tol=1e-5): {}".format(np.allclose(ph1_emb, ph15_emb, atol=1e-5)))
    
    # Log to file
    with open(os.path.join(SAVE_DIR, 'v6g_phase15_stability.txt'), 'w') as f:
        f.write("Phase 1.5 Embedding Stability Diagnostic\n")
        f.write("Max change: {:.6f}\n".format(emb_diff.max()))
        f.write("Mean change: {:.6f}\n".format(emb_diff.mean()))
        f.write("Identical (tol=1e-5): {}\n".format(np.allclose(ph1_emb, ph15_emb, atol=1e-5)))
```

**Change version tag:**
Line 120: `VER_TAG = "v6g_integrated"`

### 1.3 Create Run Script

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/model/run_GPU_v6g.py`

Copy `run_GPU_v6f.py` and modify:
```python
VERSION_TAG  = 'v6g'
MODEL_SCRIPT = "v6g_integrated_full_model.py"
```

### Validation

**Run on one GO term:**
```bash
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model/
CUDA_VISIBLE_DEVICES=0 python v6g_integrated_full_model.py GO:0006096
```

**Check outputs:**
1. `v6g_phase15_stability.txt` should show max change < 1e-5 (confirming freeze works)
2. Training log should include bias_score in embedding quality analysis
3. If bias_score > 0.3 → DomainDelta is helping
4. If bias_score < 0.3 → need adversarial loss in v7a

---

## Phase 2: v7a — Prototype Contrastive Loss (Estimated: 12 hours)

**Purpose:** Replace Triplet Loss with CLEAN-inspired prototype-based contrastive learning for better gradient stability and Type-B GO term handling.

### 2.1 Create Prototype Contrastive Module

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/model/prototype_contrastive.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""
Prototype-based Contrastive Loss (CLEAN-inspired)
Reference: [R3.3] Supervised Contrastive Loss + Prototype Networks [R5.3]

Key improvements over Triplet Loss:
1. Stable gradient (all positives contribute, not just hardest)
2. Prototype diversity regularization
3. Gap statistic for optimal k selection (Type-B GO terms)
"""
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def determine_k_gap_statistic(embeddings, k_max=5, n_refs=10, random_state=42):
    """
    Gap statistic to determine optimal number of prototypes.
    
    Args:
        embeddings: (n_samples, d) array
        k_max: maximum k to test
        n_refs: number of reference distributions
    
    Returns:
        optimal_k: int in [1, k_max]
    """
    if len(embeddings) < k_max:
        return 1
    
    n, d = embeddings.shape
    gaps = []
    
    # Compute W_k (within-cluster dispersion) for data
    W_k_data = []
    for k in range(1, k_max + 1):
        if k == 1:
            center = embeddings.mean(axis=0, keepdims=True)
            W = np.sum((embeddings - center) ** 2)
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            W = 0.0
            for i in range(k):
                cluster_i = embeddings[labels == i]
                if len(cluster_i) > 0:
                    center_i = cluster_i.mean(axis=0)
                    W += np.sum((cluster_i - center_i) ** 2)
        W_k_data.append(np.log(W + 1e-10))
    
    # Compute W_k for reference (uniform random)
    emb_min = embeddings.min(axis=0)
    emb_max = embeddings.max(axis=0)
    W_k_refs = []
    
    for _ in range(n_refs):
        ref = np.random.uniform(emb_min, emb_max, size=(n, d))
        W_k_ref = []
        for k in range(1, k_max + 1):
            if k == 1:
                center = ref.mean(axis=0, keepdims=True)
                W = np.sum((ref - center) ** 2)
            else:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(ref)
                W = 0.0
                for i in range(k):
                    cluster_i = ref[labels == i]
                    if len(cluster_i) > 0:
                        center_i = cluster_i.mean(axis=0)
                        W += np.sum((cluster_i - center_i) ** 2)
            W_k_ref.append(np.log(W + 1e-10))
        W_k_refs.append(W_k_ref)
    
    # Gap = E[log(W_k_ref)] - log(W_k_data)
    W_k_refs = np.array(W_k_refs)  # (n_refs, k_max)
    E_log_W_ref = W_k_refs.mean(axis=0)
    gaps = E_log_W_ref - np.array(W_k_data)
    
    # Find first k where Gap(k) >= Gap(k+1) - sd(k+1)
    sd_k = W_k_refs.std(axis=0) * np.sqrt(1 + 1.0/n_refs)
    
    optimal_k = 1
    for k_idx in range(len(gaps) - 1):
        if gaps[k_idx] >= gaps[k_idx + 1] - sd_k[k_idx + 1]:
            optimal_k = k_idx + 1
            break
    
    return optimal_k


def initialize_prototypes(embeddings, y, go_type='A', phase0_sep_ratio=1.0, k_max=5):
    """
    Initialize prototypes based on GO term type.
    
    Args:
        embeddings: (n, d) positive class embeddings
        y: labels (used to filter positives)
        go_type: 'A' (coherent) or 'B' (heterogeneous)
        phase0_sep_ratio: from Phase 0 embedding analysis
        k_max: maximum prototypes for Type-B
    
    Returns:
        prototypes: (k, d) array
        k: number of prototypes
    """
    pos_emb = embeddings[y == 1]
    
    if len(pos_emb) == 0:
        return np.zeros((1, embeddings.shape[1])), 1
    
    # Type A: single prototype (mean)
    if go_type == 'A' or phase0_sep_ratio >= 1.15:
        prototype = pos_emb.mean(axis=0, keepdims=True)
        return prototype, 1
    
    # Type B: gap statistic for k
    k = determine_k_gap_statistic(pos_emb, k_max=k_max)
    
    if k == 1:
        prototype = pos_emb.mean(axis=0, keepdims=True)
        return prototype, 1
    
    # k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(pos_emb)
    prototypes = kmeans.cluster_centers_  # (k, d)
    
    print("  [ProtoInit] Type-B: k={} prototypes (gap statistic)".format(k))
    
    return prototypes, k


class PrototypeContrastiveLoss:
    """
    Prototype-based contrastive loss with diversity regularization.
    
    L_proto = L_contrast + λ_div · L_diversity
    
    L_contrast = -log[ exp(-d(z, μ_pos)) / Σ_j exp(-d(z, μ_j)) ]
    L_diversity = -log(min_{i≠j} ||μ_i - μ_j||)  (encourage prototype separation)
    """
    
    def __init__(self, prototypes, tau=0.1, lambda_div=0.01):
        """
        Args:
            prototypes: (k, d) TensorFlow variable
            tau: temperature
            lambda_div: diversity regularization weight
        """
        self.prototypes = prototypes  # TensorFlow Variable
        self.tau = tau
        self.lambda_div = lambda_div
        self.k = prototypes.shape[0]
    
    def compute_loss(self, z_batch, y_batch):
        """
        Args:
            z_batch: (batch, d) embeddings
            y_batch: (batch,) labels {0, 1}
        
        Returns:
            loss: scalar TensorFlow tensor
        """
        # Normalize
        z_norm = tf.math.l2_normalize(z_batch, axis=1)
        proto_norm = tf.math.l2_normalize(self.prototypes, axis=1)
        
        # Cosine similarity: (batch, k)
        sim = tf.matmul(z_norm, proto_norm, transpose_b=True)  # (batch, k)
        
        # For positive samples: maximize similarity to ANY prototype
        # For negative samples: minimize similarity to ALL prototypes
        
        pos_mask = tf.cast(y_batch == 1, tf.float32)  # (batch,)
        neg_mask = tf.cast(y_batch == 0, tf.float32)
        
        # Positive loss: -log[ max_j exp(sim_j/tau) / Σ_j exp(sim_j/tau) ]
        # Equivalent to: attract to nearest prototype
        exp_sim = tf.exp(sim / self.tau)  # (batch, k)
        
        # For each positive sample, attract to nearest prototype
        max_sim = tf.reduce_max(sim, axis=1)  # (batch,)
        log_sum_exp = tf.reduce_logsumexp(sim / self.tau, axis=1)  # (batch,)
        
        pos_loss = -((max_sim / self.tau) - log_sum_exp)  # (batch,)
        pos_loss = pos_loss * pos_mask
        
        # Negative loss: push away from all prototypes
        # -log[ 1 / (1 + Σ_j exp(sim_j/tau)) ]
        neg_loss = tf.reduce_logsumexp(sim / self.tau, axis=1)  # (batch,)
        neg_loss = neg_loss * neg_mask
        
        # Average over batch
        n_pos = tf.reduce_sum(pos_mask) + 1e-6
        n_neg = tf.reduce_sum(neg_mask) + 1e-6
        
        L_contrast = (tf.reduce_sum(pos_loss) / n_pos + 
                      tf.reduce_sum(neg_loss) / n_neg)
        
        # Diversity regularization: penalize prototypes that are too close
        if self.k > 1:
            # Pairwise distances between prototypes
            proto_dists = tf.matmul(proto_norm, proto_norm, transpose_b=True)  # (k, k)
            # Mask diagonal
            mask = 1.0 - tf.eye(self.k)
            proto_dists = proto_dists * mask
            # Min distance (exclude diagonal)
            min_dist = tf.reduce_min(proto_dists + (1.0 - mask) * 999.0)
            # Diversity loss: encourage min_dist to be large
            L_diversity = -tf.math.log(min_dist + 1e-6)
        else:
            L_diversity = 0.0
        
        total_loss = L_contrast + self.lambda_div * L_diversity
        
        return total_loss, L_contrast, L_diversity


def update_prototypes_ema(prototypes, new_embeddings, y, momentum=0.99):
    """
    EMA update of prototypes.
    
    Args:
        prototypes: (k, d) current prototypes (NumPy)
        new_embeddings: (n, d) new batch embeddings
        y: (n,) labels
        momentum: EMA momentum
    
    Returns:
        updated_prototypes: (k, d)
    """
    pos_emb = new_embeddings[y == 1]
    
    if len(pos_emb) == 0:
        return prototypes  # No update
    
    k = prototypes.shape[0]
    
    if k == 1:
        # Single prototype: simple EMA with mean
        new_proto = pos_emb.mean(axis=0, keepdims=True)
        updated = momentum * prototypes + (1 - momentum) * new_proto
        return updated
    
    # Multi-prototype: assign to nearest, then EMA update
    # Compute distances to prototypes
    dists = np.sum((pos_emb[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2, axis=2)  # (n_pos, k)
    assignments = np.argmin(dists, axis=1)  # (n_pos,)
    
    updated = prototypes.copy()
    for i in range(k):
        cluster_i = pos_emb[assignments == i]
        if len(cluster_i) > 0:
            new_proto_i = cluster_i.mean(axis=0)
            updated[i] = momentum * prototypes[i] + (1 - momentum) * new_proto_i
    
    return updated
```

### 2.2 Modify v6f → v7a

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/model/v7a_integrated_full_model.py` (NEW)

**Copy v6f and modify:**

**1. Import prototype module (after line 95):**
```python
from prototype_contrastive import (
    initialize_prototypes, 
    PrototypeContrastiveLoss,
    update_prototypes_ema
)
```

**2. Replace Phase 1 triplet training (lines 766-837) with:**

```python
# ==========================================================================
# PHASE 1: Prototype Contrastive Learning (v7a)
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 1: Prototype Contrastive (ESM-2+Domain, CNN frozen)")
print("    [v7a] Replaces Triplet Loss with CLEAN-inspired prototypes")
print("=" * 55)

# CNN frozen
set_cnn_trainable(feature_model, False)

# Initialize prototypes based on Phase 0 embeddings
ph0_emb = emb_by_phase['Ph0(untrained)']
ph0_labels = y_test

proto_np, k_proto = initialize_prototypes(
    ph0_emb, ph0_labels, 
    go_type='B' if IS_TYPE_B else 'A',
    phase0_sep_ratio=ph0_sep_ratio,
    k_max=5
)

# TensorFlow variable for prototypes
prototypes_var = tf.Variable(proto_np.astype(np.float32), trainable=False, name='prototypes')
proto_loss_fn = PrototypeContrastiveLoss(prototypes_var, tau=0.1, lambda_div=0.01)

PHASE1_EPOCHS = 15
BATCH_SIZE_P1 = 256

for epoch in range(PHASE1_EPOCHS):
    print('Phase 1 - Epoch: {}/{}'.format(epoch + 1, PHASE1_EPOCHS))
    
    indices = np.arange(len(X_train_esm2_upsmp))
    np.random.shuffle(indices)
    
    ep_losses = []
    ep_contrast = []
    ep_diversity = []
    
    for start in range(0, len(X_train_esm2_upsmp), BATCH_SIZE_P1):
        idx = indices[start:start + BATCH_SIZE_P1]
        
        with tf.GradientTape() as tape:
            # Forward pass
            raw_emb = feature_model(
                [X_train_esm2_upsmp[idx].astype(np.float32),
                 X_train_mask_upsmp[idx].astype(np.float32),
                 X_train_seq_upsmp[idx],
                 X_train_dm_upsmp[idx],
                 X_train_dd_upsmp[idx].astype(np.float32)], 
                training=True)
            
            z_batch = tf.math.l2_normalize(raw_emb, axis=1)
            y_batch = y_train_upsmp[idx]
            
            loss, L_contrast, L_diversity = proto_loss_fn.compute_loss(z_batch, y_batch)
        
        grads = tape.gradient(loss, feature_model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, feature_model.trainable_variables)
                          if g is not None]
        if grads_and_vars:
            adam_p1.apply_gradients(grads_and_vars)
        
        ep_losses.append(float(loss.numpy()))
        ep_contrast.append(float(L_contrast.numpy()) if isinstance(L_contrast, tf.Tensor) else float(L_contrast))
        ep_diversity.append(float(L_diversity.numpy()) if isinstance(L_diversity, tf.Tensor) else float(L_diversity))
    
    print("  -> Loss: {:.4f} | Contrast: {:.4f} | Diversity: {:.4f}".format(
        np.mean(ep_losses), np.mean(ep_contrast), np.mean(ep_diversity)))
    
    # EMA update prototypes every epoch
    if epoch % 1 == 0:
        train_emb = extract_embeddings_hybrid(
            feature_model, X_train_esm2_upsmp, X_train_mask_upsmp, 
            X_train_seq_upsmp, X_train_dm_upsmp, X_train_dd_upsmp)
        proto_np = update_prototypes_ema(proto_np, train_emb, y_train_upsmp, momentum=0.99)
        prototypes_var.assign(proto_np.astype(np.float32))
        print("  [ProtoUpdate] EMA updated (momentum=0.99)")
    
    # Validation metrics every 5 epochs
    if (epoch + 1) % 5 == 0:
        compute_margin_stats_hybrid(
            feature_model, X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
            y_test, margin=MARGIN_P1)

# Save Phase 1 results
s1 = save_phase_results_hybrid(
    "phase1_prototype_contrast", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(proto)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_prototype_contrast_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(proto)'] = s1
analyze_embedding_quality(emb_by_phase['Ph1(proto)'], y_test, 'Phase 1 Prototype')
```

**3. Update VER_TAG (line 120):**
```python
VER_TAG = "v7a_integrated"
```

### 2.3 Ablation Variants

Create three variants for ablation:

**v7a-SupCon:** Replace prototype loss with standard SupCon (all positives, no prototypes)
**v7a-Proto-k1:** Force k=1 (single prototype, even for Type-B)
**v7a-Proto-kN:** Use gap statistic k (multi-prototype for Type-B)

**Files:**
- `v7a_SupCon_integrated_full_model.py`
- `v7a_Proto_k1_integrated_full_model.py`
- `v7a_Proto_kN_integrated_full_model.py` (same as v7a)

### Validation

**Run ablation on GO:0006941 (Type-B, known to benefit from multi-prototype):**

```bash
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model/
CUDA_VISIBLE_DEVICES=0 python v7a_SupCon_integrated_full_model.py GO:0006941 &
CUDA_VISIBLE_DEVICES=1 python v7a_Proto_k1_integrated_full_model.py GO:0006941 &
wait
CUDA_VISIBLE_DEVICES=0 python v7a_Proto_kN_integrated_full_model.py GO:0006941
```

**Acceptance Criterion:**
- v7a-Proto-kN AUPRC > v7a-SupCon AUPRC (by ≥ 0.02)
- v7a-Proto-kN AUPRC > v7a-Proto-k1 AUPRC (by ≥ 0.02) for Type-B GO terms
- Prototype diversity loss decreases during training (L_diversity: epoch1 > epoch15)

---

## Phase 3: v7b — acorde Percentile Co-expression + LP (Estimated: 12 hours)

**Purpose:** Replace unreliable log1p+cosine co-expression network with statistically robust acorde method.

### 3.1 Create Preprocessing Script

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/preprocessing/build_coexpression_percentile.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""
acorde-inspired co-expression network construction.

Reference: acorde (Crow et al., Bioinformatics 2018)
- Rank-transform per sample (percentile correlation)
- Bootstrap confidence intervals (n=1000)
- Retain edges only if CI lower bound > threshold
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
import argparse
import os


def percentile_transform(X_expr):
    """
    Rank-transform each sample to percentiles.
    
    Args:
        X_expr: (n_isoforms, n_samples) expression matrix
    
    Returns:
        X_percentile: (n_isoforms, n_samples) percentile-transformed
    """
    n_iso, n_samples = X_expr.shape
    X_percentile = np.zeros_like(X_expr, dtype=np.float32)
    
    for s in range(n_samples):
        sample = X_expr[:, s]
        # Rank within sample
        ranks = sample.argsort().argsort()  # 0 to n_iso-1
        X_percentile[:, s] = ranks / (n_iso - 1.0)  # 0 to 1
    
    return X_percentile


def bootstrap_correlation_ci(X_expr, n_bootstrap=1000, ci_level=0.9, random_state=42):
    """
    Compute pairwise correlation with bootstrap CI.
    
    Args:
        X_expr: (n_isoforms, n_samples)
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level (0.9 → 90% CI)
    
    Returns:
        corr_matrix: (n_isoforms, n_isoforms) median correlation
        ci_lower: (n_isoforms, n_isoforms) CI lower bound
    """
    np.random.seed(random_state)
    n_iso, n_samples = X_expr.shape
    
    # For large n_iso, compute only k-NN pairs to save memory
    # Here we compute sparse correlation (only top-k per isoform)
    
    # Full correlation (for smaller datasets)
    if n_iso < 5000:
        print("  [acorde] Full correlation matrix (n={})...".format(n_iso))
        corr_bootstrap = []
        
        for b in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_expr[:, idx]
            
            # Spearman correlation
            corr_b, _ = spearmanr(X_boot.T)  # (n_iso, n_iso)
            corr_bootstrap.append(corr_b)
        
        corr_bootstrap = np.array(corr_bootstrap)  # (n_bootstrap, n_iso, n_iso)
        
        corr_median = np.median(corr_bootstrap, axis=0)
        alpha = (1 - ci_level) / 2.0
        ci_lower = np.percentile(corr_bootstrap, alpha * 100, axis=0)
        
        return corr_median, ci_lower
    
    else:
        # k-NN sparse network (memory-efficient for large n_iso)
        print("  [acorde] k-NN sparse network (n={}, k=50)...".format(n_iso))
        k = 50
        
        # Use median correlation to find k-NN
        corr_approx, _ = spearmanr(X_expr.T)
        corr_approx = np.nan_to_num(corr_approx, nan=0.0)
        
        # For each isoform, keep only top-k neighbors
        top_k_idx = np.argsort(-corr_approx, axis=1)[:, :k]
        
        # Bootstrap CI for selected pairs only
        corr_median = np.zeros((n_iso, n_iso), dtype=np.float32)
        ci_lower = np.zeros((n_iso, n_iso), dtype=np.float32)
        
        for i in range(n_iso):
            neighbors = top_k_idx[i]
            
            # Bootstrap for each neighbor
            for j in neighbors:
                if i == j:
                    continue
                
                corr_ij_boot = []
                for b in range(n_bootstrap):
                    idx = np.random.choice(n_samples, n_samples, replace=True)
                    Xi = X_expr[i, idx]
                    Xj = X_expr[j, idx]
                    corr_b, _ = spearmanr(Xi, Xj)
                    corr_ij_boot.append(corr_b if not np.isnan(corr_b) else 0.0)
                
                corr_median[i, j] = np.median(corr_ij_boot)
                alpha = (1 - ci_level) / 2.0
                ci_lower[i, j] = np.percentile(corr_ij_boot, alpha * 100)
        
        return corr_median, ci_lower


def compute_percentile_coexpr_network(X_expr, n_bootstrap=1000, ci_threshold=0.3, ci_level=0.9):
    """
    Build co-expression network with acorde method.
    
    Args:
        X_expr: (n_isoforms, n_samples) log-transformed CPM
        n_bootstrap: bootstrap samples
        ci_threshold: retain edges if CI lower bound > threshold
        ci_level: confidence level
    
    Returns:
        W_reliable: (n_isoforms, n_isoforms) sparse adjacency matrix
        stats: dict with edge count, mean correlation, etc.
    """
    n_iso, n_samples = X_expr.shape
    
    print("  [acorde] Input: {} isoforms × {} samples".format(n_iso, n_samples))
    
    # Step 1: Percentile transform
    X_pct = percentile_transform(X_expr)
    
    # Step 2: Bootstrap correlation CI
    corr_median, ci_lower = bootstrap_correlation_ci(
        X_pct, n_bootstrap=n_bootstrap, ci_level=ci_level)
    
    # Step 3: Threshold by CI lower bound
    W_reliable = (ci_lower > ci_threshold).astype(np.float32)
    W_reliable *= corr_median  # Weight by correlation
    
    # Statistics
    n_edges = (W_reliable > 0).sum() / 2  # Symmetric, count once
    mean_corr = corr_median[W_reliable > 0].mean() if (W_reliable > 0).any() else 0.0
    
    stats = {
        'n_edges': int(n_edges),
        'edge_density': float(n_edges) / (n_iso * (n_iso - 1) / 2),
        'mean_corr': float(mean_corr),
        'ci_threshold': ci_threshold
    }
    
    print("  [acorde] Edges: {} | Density: {:.4f} | Mean corr: {:.4f}".format(
        stats['n_edges'], stats['edge_density'], stats['mean_corr']))
    
    return W_reliable, stats


def acorde_label_propagation(base_scores, W_reliable, alpha=0.3, max_iter=10):
    """
    Label propagation on acorde network.
    
    Args:
        base_scores: (n,) model predictions
        W_reliable: (n, n) adjacency matrix (symmetric)
        alpha: propagation weight
        max_iter: maximum iterations
    
    Returns:
        refined_scores: (n,)
    """
    n = len(base_scores)
    
    # Row-normalize W
    row_sum = W_reliable.sum(axis=1, keepdims=True)
    W_norm = W_reliable / (row_sum + 1e-10)
    
    # Iterative propagation
    scores = base_scores.copy()
    
    for it in range(max_iter):
        prop_scores = W_norm @ scores
        scores_new = (1 - alpha) * base_scores + alpha * prop_scores
        
        delta = np.abs(scores_new - scores).max()
        scores = scores_new
        
        if delta < 1e-6:
            print("  [LP] Converged at iteration {}".format(it + 1))
            break
    
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpm_path', type=str, 
                        default='../data/bambu_data/CPM_transcript.txt')
    parser.add_argument('--isoform_list', type=str,
                        default='../model/my_isoform_list_fixed.npy')
    parser.add_argument('--output_dir', type=str,
                        default='../model/')
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    parser.add_argument('--ci_threshold', type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Load expression
    print("[acorde] Loading CPM matrix...")
    df = pd.read_csv(args.cpm_path, sep='\t', index_col=0)
    expr_df = df.drop(columns=['GENEID']).astype(float)
    
    isoform_list = np.load(args.isoform_list, allow_pickle=True)
    iso_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in isoform_list]
    
    # Align isoforms
    X_expr_full = []
    for iso in iso_str:
        if iso in expr_df.index:
            X_expr_full.append(np.log1p(expr_df.loc[iso].values))
        else:
            X_expr_full.append(np.zeros(expr_df.shape[1]))
    
    X_expr_full = np.array(X_expr_full, dtype=np.float32)  # (n_iso, n_samples)
    
    print("[acorde] Expression matrix: {}".format(X_expr_full.shape))
    
    # Sensitivity analysis: grid search
    ci_thresholds = [0.2, 0.3, 0.4, 0.5]
    alphas = [0.1, 0.2, 0.3, 0.5]
    
    for ci_thr in ci_thresholds:
        W_reliable, stats = compute_percentile_coexpr_network(
            X_expr_full, 
            n_bootstrap=args.n_bootstrap,
            ci_threshold=ci_thr)
        
        # Save network
        output_path = os.path.join(
            args.output_dir, 
            'coexpr_acorde_ci{:.1f}.npy'.format(ci_thr))
        np.save(output_path, W_reliable)
        
        print("  [Saved] {}".format(output_path))
        
        # Save stats
        stats_path = os.path.join(
            args.output_dir,
            'coexpr_acorde_ci{:.1f}_stats.txt'.format(ci_thr))
        with open(stats_path, 'w') as f:
            for k, v in stats.items():
                f.write("{}: {}\n".format(k, v))
    
    print("\n[Done] Sensitivity analysis complete.")
    print("  Evaluate: Try alpha ∈ {0.1, 0.2, 0.3, 0.5} × ci ∈ {0.2, 0.3, 0.4, 0.5}")
```

### 3.2 Modify v7a → v7b

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/model/v7b_integrated_full_model.py`

**Copy v7a and modify Phase 3 (lines 1006-1038 in v6f):**

```python
# ==========================================================================
# PHASE 3: Test-time Label Propagation (acorde network) [v7b]
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 3: Label Propagation (acorde percentile network)")
print("    [v7b] Re-enabled with bootstrap-validated edges")
print("=" * 55)

preds_base  = base_model.predict(
    [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
    batch_size=256, verbose=0)
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])

# Load pre-computed acorde network (from preprocessing)
ACORDE_NETWORK_PATH = 'coexpr_acorde_ci0.3.npy'  # Default ci_threshold=0.3

if os.path.exists(ACORDE_NETWORK_PATH):
    W_acorde = np.load(ACORDE_NETWORK_PATH)
    
    # Sensitivity analysis: try multiple alphas
    alpha_grid = [0.1, 0.2, 0.3, 0.5]
    best_auprc = 0.0
    best_alpha = 0.0
    best_scores = base_scores.copy()
    
    for alpha in alpha_grid:
        from preprocessing.build_coexpression_percentile import acorde_label_propagation
        lp_scores = acorde_label_propagation(base_scores, W_acorde, alpha=alpha)
        
        if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
            auprc = average_precision_score(y_test, lp_scores)
            print("  [LP] alpha={:.1f} AUPRC={:.4f}".format(alpha, auprc))
            
            if auprc > best_auprc:
                best_auprc = auprc
                best_alpha = alpha
                best_scores = lp_scores
    
    final_scores = best_scores
    final_auroc = roc_auc_score(y_test, final_scores) if (y_test == 1).sum() > 0 else float('nan')
    final_auprc = best_auprc
    
    print("\n  [v7b] Best LP: alpha={:.1f} AUPRC={:.4f}".format(best_alpha, final_auprc))
    
else:
    print("  [Warning] acorde network not found: {}".format(ACORDE_NETWORK_PATH))
    print("  [Warning] Run preprocessing/build_coexpression_percentile.py first")
    final_scores = base_scores
    final_auroc = roc_auc_score(y_test, final_scores) if (y_test == 1).sum() > 0 else float('nan')
    final_auprc = average_precision_score(y_test, final_scores) if (y_test == 1).sum() > 0 else float('nan')

score_by_phase['Final(acorde_LP)'] = final_scores
```

**Update VER_TAG:**
```python
VER_TAG = "v7b_integrated"
```

### 3.3 Preprocessing Pipeline

**Run before v7b experiments:**

```bash
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/preprocessing/
python build_coexpression_percentile.py \
    --cpm_path ../data/bambu_data/CPM_transcript.txt \
    --isoform_list ../model/my_isoform_list_fixed.npy \
    --output_dir ../model/ \
    --n_bootstrap 1000 \
    --ci_threshold 0.3
```

**Expected Runtime:** ~4 hours for n_isoforms=36,748, n_bootstrap=1000 (sparse k-NN mode)

**Outputs:**
- `coexpr_acorde_ci0.2.npy`
- `coexpr_acorde_ci0.3.npy`
- `coexpr_acorde_ci0.4.npy`
- `coexpr_acorde_ci0.5.npy`
- `coexpr_acorde_ci{X}_stats.txt` for each

### Validation

**Run v7b on GO:0006096 (known to degrade with old LP):**

```bash
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model/
CUDA_VISIBLE_DEVICES=0 python v7b_integrated_full_model.py GO:0006096
```

**Falsification Criterion:**
If acorde LP improves AUPRC on < 3/5 GO terms → conclude LP is not viable for hMuscle (insufficient n_samples=24).

**Acceptance Criterion:**
acorde LP improves AUPRC on ≥ 3/5 GO terms (by ≥ 0.01 each).

---

## Phase 4: Dataset A Evaluation (Estimated: 18 hours)

**Purpose:** Cross-dataset validation on Shaw et al. 2019 benchmark.

### 4.1 Data Acquisition

**Steps:**
1. Download Shaw et al. 2019 supplementary data
2. Extract 39,375 isoform sequences
3. Extract 96 GO slim term annotations
4. Generate ESM-2 embeddings (if not provided)

**Script:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/preprocessing/prepare_dataset_A.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""
Prepare Dataset A (Shaw et al. 2019) for evaluation.

Inputs:
  - Shaw isoform sequences (FASTA)
  - Shaw GO annotations (TSV)

Outputs:
  - dataset_A_isoforms.npy (39375,)
  - dataset_A_go_annotations.npy (39375, 96)
  - dataset_A_esm2_embeddings.npy (39375, 640)
  - dataset_A_pfam_domains.npy (39375, max_domain_len)
"""
# Implementation: ~4 hours
# (Omitted for brevity — standard data loading + ESM-2 inference)
```

### 4.2 Adapt Evaluation Pipeline

**File:** `/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/evaluation_dataset_A.py` (NEW)

```python
# -*- coding: utf-8 -*-
"""
Evaluation on Dataset A (96 GO slim terms, macro-averaged).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_dataset_A(predictions, annotations):
    """
    Args:
        predictions: (n_isoforms, 96) predicted scores
        annotations: (n_isoforms, 96) binary labels
    
    Returns:
        macro_metrics: dict {macro_AUROC, macro_AUPRC, per_term_AUPRC}
    """
    n_terms = 96
    aurocs = []
    auprcs = []
    
    for t in range(n_terms):
        y_true = annotations[:, t]
        y_pred = predictions[:, t]
        
        if y_true.sum() > 0 and (y_true == 0).sum() > 0:
            aurocs.append(roc_auc_score(y_true, y_pred))
            auprcs.append(average_precision_score(y_true, y_pred))
    
    return {
        'macro_AUROC': np.mean(aurocs),
        'macro_AUPRC': np.mean(auprcs),
        'n_terms_evaluated': len(auprcs),
        'per_term_AUPRC': auprcs
    }


# Main evaluation loop: run v7b on Dataset A, collect predictions
# (Implementation: ~2 hours)
```

### 4.3 Run Full Pipeline

```bash
# Step 1: Preprocess Dataset A
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/preprocessing/
python prepare_dataset_A.py --shaw_fasta path/to/shaw.fasta --output_dir ../data/

# Step 2: Run v7b inference on Dataset A
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model/
python v7b_dataset_A_inference.py  # Multi-label inference

# Step 3: Evaluate
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/
python evaluation_dataset_A.py
```

**Expected Outputs:**
- `dataset_A_v7b_macro_AUPRC.txt` (single number)
- `dataset_A_v7b_per_term.csv` (96 rows, one per GO term)

### Validation

**Baseline Comparison:**
Compare v7b macro-AUPRC to Shaw et al. reported performance.

**Acceptance Criterion:**
v7b macro-AUPRC ≥ Shaw baseline (or within -0.02 if computational differences exist).

---

## File Structure Summary

```
hMuscle/
├── model/
│   ├── v6f_integrated_full_model.py  (baseline)
│   ├── v6g_integrated_full_model.py  (bias_score diagnostic)
│   ├── v7a_integrated_full_model.py  (prototype contrastive)
│   ├── v7a_SupCon_integrated_full_model.py  (ablation)
│   ├── v7a_Proto_k1_integrated_full_model.py  (ablation)
│   ├── v7b_integrated_full_model.py  (acorde LP)
│   ├── prototype_contrastive.py  (NEW module)
│   ├── run_GPU_v6g.py
│   ├── run_GPU_v7a.py
│   └── run_GPU_v7b.py
├── preprocessing/
│   ├── build_coexpression_percentile.py  (NEW)
│   └── prepare_dataset_A.py  (NEW)
├── results_isoform/
│   ├── evaluation.py  (modified: add compute_bias_score)
│   └── evaluation_dataset_A.py  (NEW)
└── tasks/
    ├── diagnostic_report.md  (this report)
    ├── v7_implementation_plan.md  (this plan)
    └── ablation_schedule.md  (next deliverable)
```

---

## Implementation Timeline (Sequential)

| Phase | Task | Estimated Time | Dependencies |
|-------|------|----------------|--------------|
| 0 | Verify diagnostics | 2 hours | - |
| 1 | v6g (bias_score) | 3 hours | Phase 0 |
| 2.1 | prototype_contrastive.py | 4 hours | - |
| 2.2 | v7a main | 4 hours | 2.1 |
| 2.3 | v7a ablations | 4 hours | 2.2 |
| 3.1 | build_coexpression_percentile.py | 6 hours | - |
| 3.2 | v7b main | 2 hours | 2.2, 3.1 |
| 3.3 | Run preprocessing | 4 hours | 3.1 |
| 4.1 | Dataset A data prep | 8 hours | - |
| 4.2 | Dataset A evaluation | 4 hours | 4.1 |
| 4.3 | Full run | 6 hours | 4.2 |

**Total:** ~47 hours (~6 working days for one engineer)

**Critical Path:** Phase 2 (v7a) → Phase 3 (v7b) → Phase 4 (Dataset A)

---

## Risk Mitigation

### Risk 1: Prototype Loss Instability
**Symptom:** Training loss NaN or exploding gradients  
**Mitigation:** 
- Reduce tau (0.1 → 0.05)
- Add gradient clipping (clipnorm=1.0)
- Reduce lambda_div (0.01 → 0.001)

### Risk 2: acorde Network Too Sparse
**Symptom:** n_edges < 1000, LP has no effect  
**Mitigation:**
- Lower ci_threshold (0.3 → 0.2)
- Increase n_bootstrap (1000 → 2000 for better CI)
- Fallback: disable LP (alpha=0.0)

### Risk 3: Dataset A Data Unavailable
**Symptom:** Shaw et al. supplementary data not downloadable  
**Mitigation:**
- Contact authors for data access
- Use alternative benchmark (e.g., DeepIsoFun dataset)
- Defer Dataset A to post-publication

---

**Next Deliverable:** `ablation_schedule.md` with detailed experiment specifications and acceptance/falsification criteria for each variant.
