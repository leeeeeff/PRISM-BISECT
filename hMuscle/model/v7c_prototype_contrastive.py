# -*- coding: utf-8 -*-
# ============================================================================
# v7c_prototype_contrastive.py
#
# v7b prototype_contrastive.py 에서 변경:
#
# [v7c-1] PrototypeContrastiveLoss.compute_loss — neg_embeddings 추가
#   기존 분모: Σ_j exp(sim(x_i, proto_j)/τ)             ← prototype만
#   v7c 분모: Σ_j exp(sim(x_i, proto_j)/τ)
#            + Σ_{n∈neg} exp(sim(x_i, x_n)/τ)            ← negative 포함
#   근거: prototype이 positive 공간에 위치하더라도 negative가
#         drift-in 하는 것을 막는 repulsion gradient 필요.
#         분모 확장은 기존 loss 구조를 유지하면서 negative repulsion을
#         별도 λ 없이 자연스럽게 통합.
#
# [v7c-1] phase1_proto_epoch_hybrid — negative 배치 샘플링 추가
#   기존: batch_idx = pos_idx  (positive 전용)
#   v7c: pos_raw + neg_raw 모두 GradientTape 내부에서 feature_model 실행
#        → gradient가 positive(당김)와 negative(밀어냄) 양방향으로 전달
#
# [v7c-2] save_prototype_assignments — Phase 1 완료 후 prototype 할당 저장
#   생물학적 검증용: 각 positive가 어느 prototype에 할당되었는지 기록
#   UniProt/InterPro 어노테이션과 교차하여 prototype의 생물학적 의미 검증
#
# 변경 없는 항목:
#   - SupConLoss, phase1_supcon_epoch_hybrid
#   - determine_k_gap_statistic
#   - PrototypeContrastiveLoss.__init__, initialize_from_embeddings
#   - update_prototypes_ema, prototype_stats
# ============================================================================

import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


# ─── 1. Supervised Contrastive Loss (unchanged from v7b) ─────────────────────

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    N = tf.shape(embeddings)[0]
    labels = tf.cast(labels, tf.float32)
    eye = tf.eye(tf.cast(N, tf.int32), dtype=tf.float32)
    sim = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature
    sim_no_diag = sim - 1e9 * eye
    log_denom = tf.reduce_logsumexp(sim_no_diag, axis=1)
    pos_mask = (tf.expand_dims(labels, 1) * tf.expand_dims(labels, 0)) * (1.0 - eye)
    n_pos = tf.reduce_sum(pos_mask, axis=1)
    mean_pos_sim = tf.reduce_sum(sim * pos_mask, axis=1) / (n_pos + 1e-10)
    per_anchor = log_denom - mean_pos_sim
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
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    if len(pos_indices) < 2:
        return 0.0, 0
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
        g_v = [(g, v) for g, v in zip(grads, feature_model.trainable_variables) if g is not None]
        if g_v:
            optimizer.apply_gradients(g_v)
        batch_losses.append(float(loss.numpy()))
        n_batch_pos = int((batch_y == 1).sum())
        valid_counts.append(max(0, n_batch_pos - 1))
    return np.mean(batch_losses), float(np.mean(valid_counts))


# ─── 2. Gap Statistic k-Selection (unchanged) ────────────────────────────────

def determine_k_gap_statistic(embeddings, k_max=5, n_refs=10, random_state=42):
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
    for k in range(1, k_max):
        if gaps[k - 1] >= gaps[k] - gap_stds[k]:
            print("  [Gap] k_opt={} (selected)".format(k))
            return k
    print("  [Gap] k_opt={} (max)".format(k_max))
    return k_max


# ─── 3. Prototype Contrastive Loss ───────────────────────────────────────────

class PrototypeContrastiveLoss:
    """
    v7c 수정: compute_loss에 neg_embeddings 추가 [v7c-1]

    손실 구조:
      L_proto = mean over positive anchors:
        log [Σ_j exp(sim(x_i, proto_j)/τ) + Σ_{n∈neg} exp(sim(x_i, x_n)/τ)]
              − sim(x_i, proto_k(i))/τ

      L_diversity = (1/(k(k-1))) Σ_{j≠m} sim(proto_j, proto_m)

      Total = L_proto + λ_div · L_diversity
    """

    def __init__(self, n_prototypes=1, emb_dim=64,
                 temperature=0.1, ema_decay=0.9, lambda_div=0.1):
        self.k          = n_prototypes
        self.D          = emb_dim
        self.tau        = temperature
        self.alpha      = ema_decay
        self.lambda_div = lambda_div
        self.prototypes = None

    def initialize_from_embeddings(self, pos_embeddings):
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

    def compute_loss(self, pos_embeddings, neg_embeddings=None):
        """
        [v7c-1] neg_embeddings가 제공되면 분모에 포함.

        Parameters
        ----------
        pos_embeddings : tf.Tensor (N_pos, D), L2-정규화 완료
        neg_embeddings : tf.Tensor (N_neg, D) or None  [v7c-1]
            None이면 v7b와 동일한 동작 (prototype만 분모)

        Returns
        -------
        (proto_loss, diversity_loss)
        """
        if self.prototypes is None:
            return tf.constant(0.0), tf.constant(0.0)

        protos = tf.math.l2_normalize(self.prototypes, axis=1)  # (k, D)

        # positive × prototype 유사도 / τ: (N_pos, k)
        sim_pos_proto = tf.matmul(pos_embeddings, protos, transpose_b=True) / self.tau

        if neg_embeddings is not None:
            # [v7c-1] positive × negative 유사도 / τ: (N_pos, N_neg)
            # 두 tensor 모두 GradientTape 안에서 feature_model을 거치므로
            # gradient가 pos(prototype 방향으로 당김)와 neg(밀어냄) 모두에 전달됨
            sim_pos_neg = tf.matmul(pos_embeddings,
                                    tf.transpose(neg_embeddings)) / self.tau
            all_sims  = tf.concat([sim_pos_proto, sim_pos_neg], axis=1)  # (N_pos, k+N_neg)
            log_denom = tf.reduce_logsumexp(all_sims, axis=1)             # (N_pos,)
        else:
            # fallback: v7b와 동일
            log_denom = tf.reduce_logsumexp(sim_pos_proto, axis=1)

        # 분자: nearest prototype sim
        sim_nearest = tf.reduce_max(sim_pos_proto, axis=1)  # (N_pos,)
        proto_loss  = tf.reduce_mean(log_denom - sim_nearest)

        # Diversity loss (unchanged)
        if self.k > 1:
            proto_sim     = tf.matmul(protos, protos, transpose_b=True)
            off_diag      = 1.0 - tf.eye(self.k)
            diversity_loss = tf.reduce_sum(proto_sim * off_diag) / float(self.k * (self.k - 1))
        else:
            diversity_loss = tf.constant(0.0)

        return proto_loss, diversity_loss

    def update_prototypes_ema(self, pos_embeddings_np):
        if self.prototypes is None or len(pos_embeddings_np) == 0:
            return
        protos   = self.prototypes.numpy()
        protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10)
        sims     = pos_embeddings_np @ protos_n.T
        assignments = sims.argmax(axis=1)
        new_protos  = protos.copy()
        for ki in range(self.k):
            assigned = pos_embeddings_np[assignments == ki]
            if len(assigned) > 0:
                new_c   = assigned.mean(axis=0)
                new_c   = new_c / (np.linalg.norm(new_c) + 1e-10)
                blended = self.alpha * protos[ki] + (1.0 - self.alpha) * new_c
                new_protos[ki] = blended / (np.linalg.norm(blended) + 1e-10)
        self.prototypes.assign(new_protos.astype(np.float32))

    def prototype_stats(self, embeddings_np, y_np):
        if self.prototypes is None:
            return
        pos_emb  = embeddings_np[y_np == 1]
        if len(pos_emb) == 0:
            return
        protos   = self.prototypes.numpy()
        protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10)
        sims     = pos_emb @ protos_n.T
        assignments = sims.argmax(axis=1)
        for ki in range(self.k):
            n_assigned = (assignments == ki).sum()
            print("  [Proto] c_{}: {}/{} positives assigned".format(ki, n_assigned, len(pos_emb)))


# ─── 4. Phase 1 ProtoConLoss epoch [v7c-1 수정] ──────────────────────────────

def phase1_proto_epoch_hybrid(
        feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
        optimizer, proto_loss_fn,
        batch_size=256, n_batches=50, min_pos_in_batch=8,
        do_ema_update=True):
    """
    [v7c-1] negative 배치 추가: GradientTape 안에서 pos + neg 모두 forward
    → compute_loss에 neg_embeddings 전달 → negative repulsion gradient 활성화

    변경:
      - neg_idx 샘플링 추가 (n_neg = n_pos)
      - pos_raw, neg_raw 각각 별도 feature_model 호출 (동일 tape)
      - proto_loss_fn.compute_loss(pos_emb, neg_emb) 호출
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    if len(pos_indices) < 2:
        return 0.0, 0.0

    n_pos_per_batch = max(min_pos_in_batch, min(len(pos_indices), batch_size // 2))
    n_neg_per_batch = min(len(neg_indices), n_pos_per_batch)   # [v7c-1] pos와 동수
    replace_pos = (len(pos_indices) < n_pos_per_batch)
    replace_neg = (len(neg_indices) < n_neg_per_batch)

    proto_losses = []
    div_losses   = []
    all_pos_emb  = []  # EMA용

    for _ in range(n_batches):
        pos_idx = np.random.choice(pos_indices, n_pos_per_batch, replace=replace_pos)
        neg_idx = np.random.choice(neg_indices, n_neg_per_batch, replace=replace_neg)  # [v7c-1]

        with tf.GradientTape() as tape:
            # positive forward
            pos_raw = feature_model(
                [X_esm2[pos_idx].astype(np.float32),
                 X_mask[pos_idx].astype(np.float32),
                 X_seq[pos_idx],
                 X_dm[pos_idx],
                 X_dd[pos_idx].astype(np.float32)],
                training=True)
            # [v7c-1] negative forward (동일 tape → gradient 전달)
            neg_raw = feature_model(
                [X_esm2[neg_idx].astype(np.float32),
                 X_mask[neg_idx].astype(np.float32),
                 X_seq[neg_idx],
                 X_dm[neg_idx],
                 X_dd[neg_idx].astype(np.float32)],
                training=True)

            pos_emb = tf.math.l2_normalize(pos_raw, axis=1)
            neg_emb = tf.math.l2_normalize(neg_raw, axis=1)  # [v7c-1]

            p_loss, d_loss = proto_loss_fn.compute_loss(pos_emb, neg_emb)  # [v7c-1]
            total = p_loss + proto_loss_fn.lambda_div * d_loss

        grads = tape.gradient(total, feature_model.trainable_variables)
        g_v = [(g, v) for g, v in zip(grads, feature_model.trainable_variables)
               if g is not None]
        if g_v:
            optimizer.apply_gradients(g_v)

        proto_losses.append(float(p_loss.numpy()))
        div_losses.append(float(d_loss.numpy()))
        all_pos_emb.append(pos_emb.numpy())

    if do_ema_update and all_pos_emb:
        stacked = np.vstack(all_pos_emb)
        proto_loss_fn.update_prototypes_ema(stacked)

    return float(np.mean(proto_losses)), float(np.mean(div_losses))


# ─── 5. Prototype Biological Validation [v7c-2] ──────────────────────────────

def save_prototype_assignments(proto_loss_fn, embeddings_np, y_np,
                                save_dir, ver_tag, go_term):
    """
    [v7c-2] Phase 1 완료 후 prototype 할당 저장 (생물학적 검증용).

    저장 파일:
      {ver_tag}_{safe_go}_proto_assignments.npy  — 각 test 샘플의 prototype 인덱스
      {ver_tag}_{safe_go}_proto_embeddings.npy   — L2 정규화된 prototype 벡터 (k, D)

    사후 분석 예시:
      assignments = np.load('proto_assignments.npy')  # (N_test,)
      pos_mask    = y_np == 1
      # UniProt family와 교차 → prototype이 생물학적 subgroup을 반영하는지 확인

    Parameters
    ----------
    proto_loss_fn : PrototypeContrastiveLoss (prototypes 초기화 완료)
    embeddings_np : numpy array (N_test, D), Phase 1 후 L2-정규화 임베딩
    y_np          : numpy array (N_test,), binary labels
    save_dir      : str, 결과 저장 경로
    ver_tag       : str, e.g. 'v7c_integrated'
    go_term       : str, e.g. 'GO:0006941'
    """
    if proto_loss_fn is None or proto_loss_fn.prototypes is None:
        return

    protos   = proto_loss_fn.prototypes.numpy()
    protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10)

    sims        = embeddings_np @ protos_n.T  # (N_test, k)
    assignments = sims.argmax(axis=1)         # (N_test,)

    safe_go = go_term.replace(':', '_')
    np.save(os.path.join(save_dir,
            '{}_{}_proto_assignments.npy'.format(ver_tag, safe_go)), assignments)
    np.save(os.path.join(save_dir,
            '{}_{}_proto_embeddings.npy'.format(ver_tag, safe_go)), protos_n)

    # 요약 출력
    pos_mask   = y_np == 1
    pos_assign = assignments[pos_mask]
    print("  [BioVal] Prototype assignments saved (k={}) — {} test positives".format(
        proto_loss_fn.k, pos_mask.sum()))
    for ki in range(proto_loss_fn.k):
        n   = (pos_assign == ki).sum()
        pct = 100.0 * n / (pos_mask.sum() + 1e-10)
        print("  [BioVal]   proto_{}: {}/{} ({:.1f}%)".format(
            ki, n, pos_mask.sum(), pct))
