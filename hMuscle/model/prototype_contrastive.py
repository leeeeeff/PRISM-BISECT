# -*- coding: utf-8 -*-
# ============================================================================
# prototype_contrastive.py  (v7a 모듈)
#
# 두 가지 Phase 1 대체 손실 함수:
#   1) SupConLoss       — Supervised Contrastive (Khosla et al. NeurIPS 2020)
#   2) ProtoConLoss     — CLEAN-style Prototype Contrastive (Gong et al. Science 2023)
#
# SupCon 수식 [ablation Exp 2a]:
#   L_i = log Σ_{a≠i} exp(z_i·z_a/τ) − (1/|P(i)|) Σ_{p∈P(i)} z_i·z_p/τ
#   P(i) = 배치 내 positive 샘플 (i 자신 제외)
#   z: L2-정규화 임베딩, τ=0.1
#
# ProtoConLoss 수식 [ablation Exp 2b/2c]:
#   L_proto = -log[exp(sim(x_i, c_k(i))/τ) / Σ_j exp(sim(x_i, c_j)/τ)]
#   c_k(i) = x_i에 가장 가까운 프로토타입
#   EMA update: c_k ← α·c_k + (1-α)·mean(assigned positives)
#
# Gap Statistic [Tibshirani et al. 2001]:
#   k_opt = smallest k s.t. Gap(k) ≥ Gap(k+1) − std(k+1)
#
# 사용법:
#   from prototype_contrastive import (
#       supervised_contrastive_loss,
#       phase1_supcon_epoch_hybrid,
#       determine_k_gap_statistic,
#       PrototypeContrastiveLoss,
#   )
# ============================================================================

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


# ─── 1. Supervised Contrastive Loss ──────────────────────────────────────────

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    SupCon loss (Khosla et al., NeurIPS 2020).

    L_i = log Σ_{a≠i} exp(z_i·z_a/τ) − (1/|P(i)|) Σ_{p∈P(i)} z_i·z_p/τ

    규칙 [R3.3]: Triplet active ratio < 5% 또는 SupCon Δ AUPRC > +0.02 시 교체 검토.

    Parameters
    ----------
    embeddings : tf.Tensor, shape (N, D), L2-정규화 완료
    labels     : tf.Tensor, shape (N,), binary {0, 1}
    temperature: float, default 0.1

    Returns
    -------
    scalar tf.Tensor (float32)
    """
    N = tf.shape(embeddings)[0]
    labels = tf.cast(labels, tf.float32)
    eye = tf.eye(tf.cast(N, tf.int32), dtype=tf.float32)

    # 코사인 유사도 / 온도
    sim = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature  # (N, N)

    # 대각 제외: log Σ_{a≠i} exp(sim_ia/τ)
    sim_no_diag = sim - 1e9 * eye
    log_denom = tf.reduce_logsumexp(sim_no_diag, axis=1)  # (N,)

    # Positive pair 마스크 (두 샘플 모두 label=1, 자기 자신 제외)
    pos_mask = (tf.expand_dims(labels, 1) * tf.expand_dims(labels, 0)) * (1.0 - eye)
    n_pos = tf.reduce_sum(pos_mask, axis=1)  # (N,)

    # 평균 positive 유사도 (분자)
    mean_pos_sim = tf.reduce_sum(sim * pos_mask, axis=1) / (n_pos + 1e-10)

    # L_i = log_denom_i − mean_pos_sim_i
    per_anchor = log_denom - mean_pos_sim

    # valid anchor: positive 가 1개 이상 있는 샘플만
    valid = tf.cast(n_pos > 0, tf.float32)
    n_valid = tf.reduce_sum(valid)

    loss = tf.cond(
        n_valid > 0,
        lambda: tf.reduce_sum(per_anchor * valid) / n_valid,
        lambda: tf.constant(0.0),
    )
    return loss


def phase1_supcon_epoch_hybrid(
        feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
        optimizer, temperature=0.1, batch_size=256, n_batches=50,
        min_pos_in_batch=8):
    """
    Phase 1 SupCon 1 epoch: 균형 배치 샘플링 → SupCon loss → gradient update.

    Triplet 대비 차이:
      - triplet (a, p, n) → 배치 전체 positive 쌍을 동시에 학습
      - active_rate 대신 n_valid_anchors (positive ≥ 2인 배치 비율) 반환

    Parameters
    ----------
    feature_model   : Keras Model (embedding layer 출력)
    X_esm2 ~ X_dd  : numpy arrays (upsampled train set)
    y               : numpy array (0/1 labels, upsampled)
    optimizer       : Keras optimizer (adam_p1 등)
    temperature     : SupCon τ (default 0.1)
    batch_size      : 배치 크기 (default 256)
    n_batches       : epoch 당 배치 수 (default 50)
    min_pos_in_batch: 배치 내 최소 positive 수 (default 8, 희소 GO term 대응)

    Returns
    -------
    (avg_loss, avg_n_valid) : (float, float)
      avg_n_valid: 배치 당 평균 valid anchor 수 (≥1 positive를 가진 anchor)
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    if len(pos_indices) < 2:
        print("  [SupCon] WARNING: < 2 positives, skipping epoch")
        return 0.0, 0

    # 배치 구성: positive 비율 보장
    n_pos_per_batch = max(min_pos_in_batch, min(len(pos_indices), batch_size // 4))
    n_neg_per_batch = batch_size - n_pos_per_batch
    replace_pos = (len(pos_indices) < n_pos_per_batch)
    replace_neg = (len(neg_indices) < n_neg_per_batch)

    batch_losses = []
    valid_counts = []

    for _ in range(n_batches):
        pos_idx = np.random.choice(pos_indices, n_pos_per_batch, replace=replace_pos)
        neg_idx = np.random.choice(neg_indices, n_neg_per_batch, replace=replace_neg)
        batch_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(batch_idx)
        batch_y = y[batch_idx]

        with tf.GradientTape() as tape:
            raw_emb = feature_model(
                [X_esm2[batch_idx].astype(np.float32),
                 X_mask[batch_idx].astype(np.float32),
                 X_seq[batch_idx],
                 X_dm[batch_idx],
                 X_dd[batch_idx].astype(np.float32)],
                training=True)
            emb = tf.math.l2_normalize(raw_emb, axis=1)
            batch_labels = tf.constant(batch_y, dtype=tf.float32)
            loss = supervised_contrastive_loss(emb, batch_labels, temperature=temperature)

        grads = tape.gradient(loss, feature_model.trainable_variables)
        g_v = [(g, v) for g, v in zip(grads, feature_model.trainable_variables)
               if g is not None]
        if g_v:
            optimizer.apply_gradients(g_v)

        batch_losses.append(float(loss.numpy()))
        # valid anchor 수: positive 가 ≥2이면 각 positive는 valid
        n_batch_pos = int((batch_y == 1).sum())
        valid_counts.append(max(0, n_batch_pos - 1))  # 자신 제외

    return np.mean(batch_losses), float(np.mean(valid_counts))


# ─── 2. Gap Statistic k-Selection ────────────────────────────────────────────

def determine_k_gap_statistic(embeddings, k_max=5, n_refs=10, random_state=42):
    """
    Gap statistic (Tibshirani et al. 2001) — 최적 k ∈ [1, k_max] 선택.

    선택 기준: Gap(k) ≥ Gap(k+1) − std(k+1) 를 만족하는 가장 작은 k

    Parameters
    ----------
    embeddings   : numpy array (N, D), L2-정규화 추천
    k_max        : 최대 프로토타입 수 (default 5)
    n_refs       : 레퍼런스 분포 샘플 수 (default 10)
    random_state : int

    Returns
    -------
    k_opt : int ∈ [1, k_max]
    """
    n_samples = len(embeddings)
    if n_samples < k_max * 3:
        return 1

    np.random.seed(random_state)
    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)

    def inertia(data, k):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=100)
        km.fit(data)
        return km.inertia_

    gaps = []
    gap_stds = []
    for k in range(1, k_max + 1):
        wk_log = np.log(inertia(embeddings, k) + 1e-10)
        ref_logs = []
        for _ in range(n_refs):
            ref_data = np.random.uniform(mins, maxs, embeddings.shape)
            ref_logs.append(np.log(inertia(ref_data, k) + 1e-10))
        gap = float(np.mean(ref_logs)) - wk_log
        sdk = float(np.std(ref_logs)) * np.sqrt(1.0 + 1.0 / n_refs)
        gaps.append(gap)
        gap_stds.append(sdk)
        print("  [Gap] k={}: gap={:.4f} std={:.4f}".format(k, gap, sdk))

    # 가장 작은 k s.t. Gap(k) >= Gap(k+1) - std(k+1)
    for k in range(1, k_max):
        if gaps[k - 1] >= gaps[k] - gap_stds[k]:
            print("  [Gap] k_opt={} (selected)".format(k))
            return k
    print("  [Gap] k_opt={} (max)".format(k_max))
    return k_max


# ─── 3. Prototype Contrastive Loss ───────────────────────────────────────────

class PrototypeContrastiveLoss:
    """
    CLEAN-style Prototype Contrastive Loss (Gong et al. Science 2023).

    Phase 1에서 Triplet/SupCon 대신 사용.

    손실 구조:
      L_proto = mean over positive anchors:
                  log Σ_j exp(sim(x_i, c_j)/τ) − sim(x_i, c_k(i))/τ
              where c_k(i) = nearest prototype to x_i

      L_diversity = (1/(k(k-1))) Σ_{j≠m} sim(c_j, c_m)   [k > 1 시]

      Total = L_proto + λ_div · L_diversity

    EMA update:
      c_k ← normalize(α · c_k + (1-α) · mean(assigned positives))

    Parameters
    ----------
    n_prototypes : int, number of prototypes k
    emb_dim      : int, embedding dimension (default 64)
    temperature  : float, τ (default 0.1)
    ema_decay    : float, α (default 0.9)
    lambda_div   : float, diversity loss weight (default 0.1)
    """

    def __init__(self, n_prototypes=1, emb_dim=64,
                 temperature=0.1, ema_decay=0.9, lambda_div=0.1):
        self.k          = n_prototypes
        self.D          = emb_dim
        self.tau        = temperature
        self.alpha      = ema_decay
        self.lambda_div = lambda_div
        self.prototypes = None   # tf.Variable (k, D), 학습 외 EMA 관리

    def initialize_from_embeddings(self, pos_embeddings):
        """
        k-means로 positive 임베딩에서 프로토타입 초기화.

        Parameters
        ----------
        pos_embeddings : numpy array (N_pos, D), L2-정규화 권장
        """
        if self.k == 1 or len(pos_embeddings) < self.k:
            center = pos_embeddings.mean(axis=0)
            norm = np.linalg.norm(center) + 1e-10
            centers = np.tile(center / norm, (self.k, 1)).astype(np.float32)
        else:
            km = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            km.fit(pos_embeddings)
            centers = km.cluster_centers_.astype(np.float32)
            norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10
            centers = centers / norms

        self.prototypes = tf.Variable(centers, name='prototypes', trainable=False)
        print("  [Proto] k={} initialized | N_pos={}".format(self.k, len(pos_embeddings)))

    def compute_loss(self, pos_embeddings):
        """
        L_proto + L_diversity 계산.

        Parameters
        ----------
        pos_embeddings : tf.Tensor (N_pos, D), L2-정규화 완료

        Returns
        -------
        (proto_loss, diversity_loss) : tuple of scalar tf.Tensor
        """
        if self.prototypes is None:
            return tf.constant(0.0), tf.constant(0.0)

        protos = tf.math.l2_normalize(self.prototypes, axis=1)  # (k, D)

        # anchor × prototype 유사도 / τ: (N_pos, k)
        sim = tf.matmul(pos_embeddings, protos, transpose_b=True) / self.tau

        # 분모: log Σ_j exp(sim_ij)
        log_denom = tf.reduce_logsumexp(sim, axis=1)   # (N_pos,)

        # 분자: nearest prototype sim
        sim_nearest = tf.reduce_max(sim, axis=1)        # (N_pos,)

        proto_loss = tf.reduce_mean(log_denom - sim_nearest)

        # Diversity loss (k > 1)
        if self.k > 1:
            proto_sim = tf.matmul(protos, protos, transpose_b=True)
            off_diag  = 1.0 - tf.eye(self.k)
            diversity_loss = tf.reduce_sum(proto_sim * off_diag) / float(self.k * (self.k - 1))
        else:
            diversity_loss = tf.constant(0.0)

        return proto_loss, diversity_loss

    def update_prototypes_ema(self, pos_embeddings_np):
        """
        EMA 프로토타입 업데이트 (numpy, gradient 외부).

        c_k ← normalize(α·c_k + (1-α)·mean(assigned_positives_k))

        Parameters
        ----------
        pos_embeddings_np : numpy array (N_pos, D), L2-정규화 완료
        """
        if self.prototypes is None or len(pos_embeddings_np) == 0:
            return

        protos = self.prototypes.numpy()  # (k, D)
        protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10)

        # 각 positive를 nearest prototype에 할당
        sims = pos_embeddings_np @ protos_n.T      # (N_pos, k)
        assignments = sims.argmax(axis=1)           # (N_pos,)

        new_protos = protos.copy()
        for ki in range(self.k):
            assigned = pos_embeddings_np[assignments == ki]
            if len(assigned) > 0:
                new_c = assigned.mean(axis=0)
                new_c = new_c / (np.linalg.norm(new_c) + 1e-10)
                blended = self.alpha * protos[ki] + (1.0 - self.alpha) * new_c
                new_protos[ki] = blended / (np.linalg.norm(blended) + 1e-10)

        self.prototypes.assign(new_protos.astype(np.float32))

    def prototype_stats(self, embeddings_np, y_np):
        """
        프로토타입 진단: 각 prototype에 몇 개의 positive가 할당되었는지 출력.

        Parameters
        ----------
        embeddings_np : numpy array (N, D)
        y_np          : numpy array (N,) binary
        """
        if self.prototypes is None:
            return
        pos_emb = embeddings_np[y_np == 1]
        if len(pos_emb) == 0:
            return
        protos = self.prototypes.numpy()
        protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10)
        sims = pos_emb @ protos_n.T
        assignments = sims.argmax(axis=1)
        for ki in range(self.k):
            n_assigned = (assignments == ki).sum()
            print("  [Proto] c_{}: {}/{} positives assigned".format(ki, n_assigned, len(pos_emb)))


# ─── 4. Phase 1 ProtoConLoss epoch ───────────────────────────────────────────

def phase1_proto_epoch_hybrid(
        feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
        optimizer, proto_loss_fn,
        batch_size=256, n_batches=50, min_pos_in_batch=8,
        do_ema_update=True):
    """
    Phase 1 Prototype Contrastive 1 epoch.

    1) positive 배치 샘플링
    2) L_proto + λ_div·L_diversity 계산
    3) gradient update
    4) (선택) EMA prototype 업데이트

    Parameters
    ----------
    proto_loss_fn : PrototypeContrastiveLoss instance (초기화 완료)
    do_ema_update : bool, epoch 마지막에 EMA 업데이트 수행 여부

    Returns
    -------
    (avg_proto_loss, avg_div_loss) : (float, float)
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    if len(pos_indices) < 2:
        return 0.0, 0.0

    n_pos_per_batch = max(min_pos_in_batch, min(len(pos_indices), batch_size // 2))
    replace_pos = (len(pos_indices) < n_pos_per_batch)

    proto_losses = []
    div_losses   = []
    all_pos_emb  = []   # EMA용 positive 임베딩 수집

    for _ in range(n_batches):
        pos_idx   = np.random.choice(pos_indices, n_pos_per_batch, replace=replace_pos)
        batch_idx = pos_idx  # proto loss는 positive 위주

        with tf.GradientTape() as tape:
            raw_emb = feature_model(
                [X_esm2[batch_idx].astype(np.float32),
                 X_mask[batch_idx].astype(np.float32),
                 X_seq[batch_idx],
                 X_dm[batch_idx],
                 X_dd[batch_idx].astype(np.float32)],
                training=True)
            emb = tf.math.l2_normalize(raw_emb, axis=1)
            p_loss, d_loss = proto_loss_fn.compute_loss(emb)
            total = p_loss + proto_loss_fn.lambda_div * d_loss

        grads = tape.gradient(total, feature_model.trainable_variables)
        g_v = [(g, v) for g, v in zip(grads, feature_model.trainable_variables)
               if g is not None]
        if g_v:
            optimizer.apply_gradients(g_v)

        proto_losses.append(float(p_loss.numpy()))
        div_losses.append(float(d_loss.numpy()))
        all_pos_emb.append(emb.numpy())

    if do_ema_update and all_pos_emb:
        stacked = np.vstack(all_pos_emb)
        proto_loss_fn.update_prototypes_ema(stacked)

    return float(np.mean(proto_losses)), float(np.mean(div_losses))
