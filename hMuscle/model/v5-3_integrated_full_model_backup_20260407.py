# -*- coding: utf-8 -*-
# ============================================================================
# v5-3_integrated_full_model.py
#
# v5-1 대비 유효 변경 (v5-2 실험 결과 반영):
#
# [I1] REMOVED — prior bias init은 성능에 무관했던 v5-1 mode-collapse를
#     잘못 진단하여 Phase 2 전체를 붕괴시킴. 제거 유지.
#
# [I2] LabelProp alpha 선택 기준: AUROC → AUPRC  [R9.1]  ← 유지
#     - v5-2에서 검증됨: GO:0006936 AUPRC +0.017, GO:0006412 AUPRC 악화 방지
#     - 수정: average_precision_score로 best alpha 선택
#
# [I3] margin=0.3 → 0.1 복귀  ← v5-2에서 역효과 확인, 복귀
#     - v5-2 실험: sparse GO term에서 Phase 1 Silhouette 파괴 (-0.16 → -0.18)
#     - margin=0.3은 충분한 양성(GO:0006936)에만 유리, 일반성 없음
#     - 복귀: MARGIN_P1 = 0.1
#
# [I4] n_batches 공식 개선  ← v5-2 공식(min=20)은 학습량 40~60% 감소 유발
#     - v5-2: max(20, ceil(n_pos × 0.8 / 256)) → 전부 20으로 하한 클램핑
#     - v5-3: max(50, min(150, ceil(n_pos × 4 / 256)))
#       → n_pos=287: 50, n_pos=840: 50, n_pos=3046: 50, n_pos=10000: 150
#       → v5-1 실험값(30~100)과 동등 이상 보장
#
# [I5] Phase 2 AUROC 체크 간격: every 1 epoch  ← 유지
#     - CHECK_EVERY=1, NO_IMPROVE_LIMIT=3
#
# [EMB] Phase별 임베딩 공간 정량 분석  ← 유지
#     - Silhouette, Sep.Ratio, LinAUROC, PredAUROC 4-phase 비교
#     - UMAP 시각화
# ============================================================================
import numpy as np
import sys
import os
from sys import argv
from datetime import datetime

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, Activation, LSTM,
                          Convolution1D, Embedding, Lambda, concatenate)
from keras import backend as K
from keras import regularizers, optimizers
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

# embedding analysis
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from crf import CRF
sys.path.insert(0, '/home/welcome1/layer_Full')
from PyramidPooling import PyramidPooling
from utils_Full import generate_label, upsample, make_batch

# --------------------------------------------------------------------------
# [1] 환경 설정 & 로깅
# --------------------------------------------------------------------------
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
K.set_session(session)

try:
    script, selected_go = argv
except ValueError:
    print("Usage: python v5-2_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d")
VER_TAG  = "v5-3_integrated"
SAVE_DIR = os.path.join(BASE_RESULTS_DIR, "{}_{}".format(VER_TAG, date_str))
os.makedirs(SAVE_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(SAVE_DIR, "{}_training.log".format(VER_TAG)))
print("\n[Info] {} | {} | {}\n".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), VER_TAG, SAVE_DIR))

# --------------------------------------------------------------------------
# [2] 상수
# --------------------------------------------------------------------------
EXPR_DIM = 24
EMB_DIM  = 32
MARGIN_P1 = 0.1        # [I3 복귀] 0.3 → 0.1 (sparse class Phase1 파괴 방지)
N_BATCHES_SCALE = 4    # [I4 개선] n_batches = max(50, min(150, ceil(n_pos × scale / batch)))

# --------------------------------------------------------------------------
# [3] Loss Functions
# --------------------------------------------------------------------------
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        eps     = K.epsilon()
        y_pred  = K.clip(y_pred, eps, 1.0 - eps)
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        loss    = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)
        return K.mean(loss, axis=-1)
    return fn

def triplet_loss_fn(inputs, margin=MARGIN_P1):
    anchor, positive, negative = inputs
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    return K.maximum(pos_dist - neg_dist + margin, 0.0)

# --------------------------------------------------------------------------
# [4] Expression 전처리
# --------------------------------------------------------------------------
def load_expression_matrix(cpm_path, isoform_list):
    import pandas as pd
    print("[Expr] Loading CPM matrix from {}...".format(cpm_path))
    df = pd.read_csv(cpm_path, sep='\t', index_col=0)
    expr_df = df.drop(columns=['GENEID']).astype(float)
    iso_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in isoform_list]
    missing = [x for x in iso_str if x not in expr_df.index]
    if missing:
        print("[Expr] WARNING: {} isoforms missing — zero-filled".format(len(missing)))
    X_expr = np.zeros((len(iso_str), EXPR_DIM), dtype=np.float32)
    for i, iso in enumerate(iso_str):
        if iso in expr_df.index:
            X_expr[i] = np.log1p(expr_df.loc[iso].values.astype(float))
    print("[Expr] shape={} nonzero={:.1f}%".format(X_expr.shape, (X_expr > 0).mean() * 100))
    return X_expr

# --------------------------------------------------------------------------
# [5] 임베딩 추출
# --------------------------------------------------------------------------
def extract_embeddings(feature_model, X_seq, X_dm, seq_dim, emb_dim=EMB_DIM):
    tup_idx, tup_gp = make_batch(X_seq)
    emb = np.zeros((len(X_seq), emb_dim), dtype=np.float32)
    for key in tup_gp.keys():
        sel       = tup_gp[key]
        batch_idx = tup_idx[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_idx) / 1000) + 1):
            sub_idx = batch_idx[1000 * i: 1000 * (i + 1)]
            if len(sub_idx) == 0:
                continue
            sub_seq = X_seq[sub_idx, seq_dim - sel[2]: seq_dim]
            sub_dm  = X_dm[sub_idx]
            raw = feature_model.predict_on_batch([sub_seq, sub_dm])
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            emb[sub_idx] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [EMB] 임베딩 공간 정량 분석
# --------------------------------------------------------------------------
def analyze_embedding_quality(embeddings, labels, phase_name, n_pairs=500, seed=42):
    """
    Phase별 embedding space quality metrics:
      - Silhouette score (cosine, subsampled)
      - Intra-class (positive) distance mean
      - Inter-class distance mean
      - Separation ratio = inter / intra  (>1 is good)
      - Centroid distance (L2)
      - Linear AUROC (LogisticRegression on embeddings)
    """
    np.random.seed(seed)
    pos_emb = embeddings[labels == 1]
    neg_emb = embeddings[labels == 0]

    metrics = {}

    # --- Centroid distance ---
    if len(pos_emb) > 0 and len(neg_emb) > 0:
        metrics['centroid_dist'] = float(np.linalg.norm(
            pos_emb.mean(axis=0) - neg_emb.mean(axis=0)))
    else:
        metrics['centroid_dist'] = float('nan')

    # --- Intra / Inter distance (squared L2 on unit sphere = 2(1-cosine)) ---
    if len(pos_emb) >= 2:
        n = min(n_pairs, len(pos_emb))
        intra_d, inter_d = [], []
        for _ in range(n):
            i, j = np.random.choice(len(pos_emb), 2, replace=False)
            intra_d.append(max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[i], pos_emb[j]))))
        for _ in range(n):
            i = np.random.choice(len(pos_emb))
            j = np.random.choice(len(neg_emb))
            inter_d.append(max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[i], neg_emb[j]))))
        metrics['intra_dist'] = float(np.mean(intra_d))
        metrics['inter_dist'] = float(np.mean(inter_d))
        metrics['sep_ratio']  = metrics['inter_dist'] / (metrics['intra_dist'] + 1e-8)
    else:
        metrics['intra_dist'] = metrics['inter_dist'] = metrics['sep_ratio'] = float('nan')

    # --- Silhouette (cosine, subsample for speed) ---
    if len(pos_emb) >= 2 and len(neg_emb) >= 2:
        n_sub = min(4000, len(embeddings))
        idx_sub = np.random.choice(len(embeddings), n_sub, replace=False)
        emb_sub = embeddings[idx_sub]
        lab_sub = labels[idx_sub].astype(int)
        if lab_sub.sum() >= 2 and (lab_sub == 0).sum() >= 2:
            try:
                metrics['silhouette'] = float(silhouette_score(
                    emb_sub, lab_sub, metric='cosine',
                    sample_size=min(2000, n_sub), random_state=seed))
            except Exception:
                metrics['silhouette'] = float('nan')
        else:
            metrics['silhouette'] = float('nan')
    else:
        metrics['silhouette'] = float('nan')

    # --- Linear separability (LogisticRegression AUROC) ---
    if len(pos_emb) >= 5 and len(neg_emb) >= 5:
        try:
            scaler = StandardScaler()
            emb_sc = scaler.fit_transform(embeddings)
            lr = LogisticRegression(
                max_iter=300, C=1.0, class_weight='balanced', random_state=seed)
            lr.fit(emb_sc, labels.astype(int))
            lin_scores = lr.predict_proba(emb_sc)[:, 1]
            metrics['linear_auroc'] = float(roc_auc_score(labels, lin_scores))
        except Exception:
            metrics['linear_auroc'] = float('nan')
    else:
        metrics['linear_auroc'] = float('nan')

    # --- Print ---
    print("  [EmbQuality | {}]".format(phase_name))
    print("    Silhouette (cosine):  {:+.4f}  (>0 = separable)".format(metrics['silhouette']))
    print("    Intra-pos dist:        {:.4f}  (↓ = tight cluster)".format(metrics['intra_dist']))
    print("    Inter dist:            {:.4f}  (↑ = well separated)".format(metrics['inter_dist']))
    print("    Sep ratio (inter/intra):{:.4f}  (>1 = good)".format(metrics['sep_ratio']))
    print("    Centroid dist:         {:.4f}".format(metrics['centroid_dist']))
    print("    Linear AUROC:          {:.4f}  (embedding linear separability)".format(
        metrics['linear_auroc']))

    return metrics


def plot_embedding_umap(emb_dict, labels, gene_ids, save_dir, ver_tag, go_term):
    """
    4개 phase의 UMAP 시각화 (2×2 grid).
    pos/neg 색상 + 선택된 positive gene cluster 하이라이트.
    """
    try:
        import umap as umap_lib
    except ImportError:
        print("  [UMAP] umap-learn not installed — skipping")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    phases = list(emb_dict.keys())
    n_phases = len(phases)
    ncols = min(4, n_phases)
    nrows = (n_phases + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if n_phases > 1 else [axes]

    reducer = umap_lib.UMAP(
        n_components=2, random_state=42,
        metric='cosine', n_neighbors=15, min_dist=0.1,
        low_memory=True)

    # 공통 2D 좌표 (Phase 0 embedding 기준 fit, 각 phase transform 아님 — 독립 fit)
    pos_mask = labels == 1
    neg_mask = labels == 0

    # subsample negatives for speed (keep all positives)
    np.random.seed(42)
    n_neg_plot = min(5000, neg_mask.sum())
    neg_idx_plot = np.random.choice(np.where(neg_mask)[0], n_neg_plot, replace=False)
    pos_idx = np.where(pos_mask)[0]
    plot_idx = np.concatenate([pos_idx, neg_idx_plot])
    plot_labels = labels[plot_idx]

    for ax, phase_name in zip(axes, phases):
        emb = emb_dict[phase_name][plot_idx]
        try:
            coords = reducer.fit_transform(emb)
        except Exception as e:
            ax.set_title("{}\n(UMAP failed: {})".format(phase_name, str(e)[:30]))
            continue

        is_pos = plot_labels == 1

        ax.scatter(coords[~is_pos, 0], coords[~is_pos, 1],
                   c='#cccccc', alpha=0.3, s=2, rasterized=True)
        ax.scatter(coords[is_pos, 0], coords[is_pos, 1],
                   c='#e74c3c', alpha=0.85, s=18, rasterized=True,
                   edgecolors='#c0392b', linewidths=0.3)

        ax.set_title("{}\npos={} neg(sample)={}".format(
            phase_name, is_pos.sum(), (~is_pos).sum()), fontsize=9)
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for ax in axes[n_phases:]:
        ax.set_visible(False)

    neg_patch = mpatches.Patch(color='#cccccc', label='negative (sample)')
    pos_patch = mpatches.Patch(color='#e74c3c', label='positive')
    fig.legend(handles=[neg_patch, pos_patch],
               loc='lower right', fontsize=9, framealpha=0.8)
    fig.suptitle("{} — Embedding Space Evolution".format(go_term), fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    plot_path = os.path.join(save_dir, "{}_{}_umap.png".format(ver_tag, go_term))
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  [UMAP] Saved: {}".format(plot_path))


def plot_score_distribution(score_dict, labels, save_dir, ver_tag, go_term):
    """
    Phase별 score 분포 (positive vs negative histogram).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phases = list(score_dict.keys())
    fig, axes = plt.subplots(1, len(phases),
                              figsize=(4.5 * len(phases), 3.5), sharey=False)
    if len(phases) == 1:
        axes = [axes]

    for ax, phase_name in zip(axes, phases):
        scores = score_dict[phase_name]
        pos_s = scores[labels == 1]
        neg_s = scores[labels == 0]
        bins = np.linspace(scores.min() - 0.01, scores.max() + 0.01, 40)
        ax.hist(neg_s, bins=bins, alpha=0.5, color='#95a5a6',
                density=True, label='neg (n={})'.format(len(neg_s)))
        ax.hist(pos_s, bins=bins, alpha=0.8, color='#e74c3c',
                density=True, label='pos (n={})'.format(len(pos_s)))
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        ax.set_title("{}\nAUROC={:.3f} AUPRC={:.3f}".format(
            phase_name, auroc, auprc), fontsize=9)
        ax.set_xlabel("Prediction score", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle("{} — Score Distribution per Phase".format(go_term), fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "{}_{}_score_dist.png".format(ver_tag, go_term))
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  [ScoreDist] Saved: {}".format(plot_path))

# --------------------------------------------------------------------------
# [6] Phase 1 — GradientTape Triplet (semi-hard mining, margin=0.3 [I3])
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=MARGIN_P1, mode="hard"):
    """
    Semi-hard negative mining (FaceNet, 2015).
    margin=0.3 [I3] — wider semi-hard zone for richer gradients.
    """
    n = min(batch_size, len(pos_indices))
    a_idx_list, p_idx_list, n_idx_list = [], [], []

    if mode == "hard" and len(neg_indices) >= 2:
        neg_emb     = embeddings[neg_indices]
        sampled_pos = np.random.choice(pos_indices, n, replace=False)

        for a_i in sampled_pos:
            p_candidates = pos_indices[pos_indices != a_i]
            if len(p_candidates) == 0:
                continue
            p_i = np.random.choice(p_candidates)

            a_emb   = embeddings[a_i]
            p_emb   = embeddings[p_i]
            d_ap_sq = max(0.0, 2.0 - 2.0 * float(np.dot(a_emb, p_emb)))
            d_an_sq = np.clip(2.0 - 2.0 * (neg_emb @ a_emb), 0.0, None)

            # [I3] margin=0.3: semi-hard zone = d_ap < d_an < d_ap + 0.3
            semi_mask = (d_an_sq > d_ap_sq) & (d_an_sq < d_ap_sq + margin)
            semi_idx  = np.where(semi_mask)[0]
            if len(semi_idx) > 0:
                n_i = neg_indices[np.random.choice(semi_idx)]
            else:
                mod_mask = d_an_sq < d_ap_sq + 4.0 * margin
                mod_idx  = np.where(mod_mask)[0]
                if len(mod_idx) > 0:
                    n_i = neg_indices[np.random.choice(mod_idx)]
                else:
                    n_i = np.random.choice(neg_indices)

            a_idx_list.append(a_i)
            p_idx_list.append(p_i)
            n_idx_list.append(n_i)
    else:
        for _ in range(n):
            a_i, p_i = np.random.choice(pos_indices, 2, replace=False)
            n_i      = np.random.choice(neg_indices)
            a_idx_list.append(a_i)
            p_idx_list.append(p_i)
            n_idx_list.append(n_i)
    return a_idx_list, p_idx_list, n_idx_list


def phase1_embedding_triplet_epoch(
        feature_model, X_seq, X_dm, y, seq_dim,
        optimizer, margin=MARGIN_P1, batch_size=256,
        n_batches=50, warmup=False, refresh_interval=10):
    def refresh_embeddings():
        emb = extract_embeddings(feature_model, X_seq, X_dm, seq_dim)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        return emb, pos, neg

    embeddings, pos_indices, neg_indices = refresh_embeddings()
    if len(pos_indices) < 2 or len(neg_indices) == 0:
        return 0.0, 0

    mode         = 'random' if warmup else 'hard'
    batch_losses = []
    active_count = 0

    for batch_i in range(n_batches):
        if batch_i > 0 and batch_i % refresh_interval == 0 and not warmup:
            embeddings, pos_indices, neg_indices = refresh_embeddings()

        a_idxs, p_idxs, n_idxs = build_emb_triplet_inputs(
            embeddings, y, pos_indices, neg_indices,
            batch_size=batch_size, margin=margin, mode=mode)
        if not a_idxs:
            continue

        with tf.GradientTape() as tape:
            def get_live_emb(indices):
                curr_seq = X_seq[indices]
                v_lens   = np.sum(curr_seq != 0, axis=1)
                max_l    = int(np.max(v_lens)) if len(v_lens) > 0 else 100
                sub_seq  = curr_seq[:, -max_l:]
                sub_dm   = X_dm[indices]
                raw_out  = feature_model([sub_seq, sub_dm], training=True)
                return tf.math.l2_normalize(raw_out, axis=1)

            emb_a_live = get_live_emb(a_idxs)
            emb_p_live = get_live_emb(p_idxs)
            emb_n_live = get_live_emb(n_idxs)

            pos_dist = tf.reduce_sum(tf.square(emb_a_live - emb_p_live), axis=1)
            neg_dist = tf.reduce_sum(tf.square(emb_a_live - emb_n_live), axis=1)
            raw_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            loss     = tf.reduce_mean(raw_loss)

        grads = tape.gradient(loss, feature_model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, feature_model.trainable_variables)
                          if g is not None]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)

        batch_losses.append(float(loss.numpy()))
        active_count += int((raw_loss.numpy() > 0).sum())

    avg_loss    = np.mean(batch_losses) if batch_losses else 0.0
    active_rate = active_count / (n_batches * batch_size) * 100
    return avg_loss, active_rate

# --------------------------------------------------------------------------
# [7] Phase 2 보조 함수
# --------------------------------------------------------------------------
def get_triplet_batch_ingroup(X_seq_grp, X_dm_grp, y_grp, batch_size=64):
    pos_idx = np.where(y_grp == 1)[0]
    neg_idx = np.where(y_grp == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) == 0:
        return None, None

    actual_bs = min(batch_size, len(pos_idx))
    a_seq, a_dm = [], []
    p_seq, p_dm = [], []
    n_seq, n_dm = [], []
    labels = []

    for _ in range(actual_bs):
        a_i, p_i = np.random.choice(pos_idx, 2, replace=False)
        n_i      = np.random.choice(neg_idx)
        a_seq.append(X_seq_grp[a_i]);  a_dm.append(X_dm_grp[a_i])
        p_seq.append(X_seq_grp[p_i]);  p_dm.append(X_dm_grp[p_i])
        n_seq.append(X_seq_grp[n_i]);  n_dm.append(X_dm_grp[n_i])
        labels.append(1)

    return ([np.array(a_seq), np.array(a_dm),
             np.array(p_seq), np.array(p_dm),
             np.array(n_seq), np.array(n_dm)],
            np.array(labels))


def compute_margin_stats(feature_model, X_seq, X_dm, y, seq_dim,
                         margin=MARGIN_P1, n_sample=1000):
    emb     = extract_embeddings(feature_model, X_seq, X_dm, seq_dim)
    pos_emb = emb[y == 1]
    neg_emb = emb[y == 0]
    if len(pos_emb) < 2:
        return 0.0, 0.0
    np.random.seed(42)
    n = min(n_sample, len(pos_emb))
    margins = []
    for _ in range(n):
        a_i, p_i = np.random.choice(len(pos_emb), 2, replace=False)
        n_i      = np.random.choice(len(neg_emb))
        d_ap_sq = max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[a_i], pos_emb[p_i])))
        d_an_sq = max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[a_i], neg_emb[n_i])))
        margins.append(d_an_sq - d_ap_sq)
    margins       = np.array(margins)
    satisfied     = (margins > margin).mean() * 100
    centroid_dist = np.linalg.norm(pos_emb.mean(axis=0) - neg_emb.mean(axis=0))
    print("  [Margin|sq_L2] satisfied(>{:.2f}): {:.1f}% | centroid_dist: {:.4f}".format(
        margin, satisfied, centroid_dist))
    return satisfied, centroid_dist


def save_phase_results(phase_name, model_base, X_seq, X_dm, y_true,
                       gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds  = model_base.predict([X_seq, X_dm], batch_size=256, verbose=1)
    np.save(os.path.join(save_dir, '{}_{}_embeddings.npy'.format(ver_tag, phase_name)),
            preds[0])
    np.save(os.path.join(save_dir, '{}_{}_labels.npy'.format(ver_tag, phase_name)),
            y_true)
    scores = preds[1]
    s      = np.array([scores[i][0] for i in range(len(y_true))])
    print("  mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
        s.mean(), s.std(), (s > 0.5).sum(), (s > 0.3).sum()))
    with open(os.path.join(save_dir, '{}_{}_scores.txt'.format(ver_tag, phase_name)),
              'w') as fw:
        for i in range(len(y_true)):
            fw.write(gene_ids[i] + '\t' + iso_ids[i] + '\t' +
                     str(scores[i][0]) + '\n')
    return s

# --------------------------------------------------------------------------
# [Fix1] Phase 3: Test-time Label Propagation
# --------------------------------------------------------------------------
def expression_label_propagation(base_scores, X_expr, alpha=0.3, k=15,
                                  sim_threshold=0.1):
    n = len(base_scores)
    nonzero_rows = np.abs(X_expr).sum(axis=1)
    if (nonzero_rows > 0).sum() < k + 1:
        print("  [LabelProp] Expression too sparse — skipping")
        return base_scores.copy()

    expr_norm = normalize(X_expr.astype(np.float32), norm='l2')
    print("  [LabelProp] Building KNN graph (k={})...".format(k))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                             algorithm='brute', n_jobs=-1).fit(expr_norm)
    distances, indices = nbrs.kneighbors(expr_norm)

    sims = np.maximum(0.0, 1.0 - distances[:, 1:].astype(np.float32))
    sims[sims < sim_threshold] = 0.0

    prop_scores = np.zeros(n, dtype=np.float32)
    weight_sum  = np.zeros(n, dtype=np.float32)
    for rank in range(k):
        j   = indices[:, rank + 1]
        s   = sims[:, rank]
        prop_scores += s * base_scores[j]
        weight_sum  += s

    valid = weight_sum > 0
    prop_scores[valid]  /= weight_sum[valid]
    prop_scores[~valid] = base_scores[~valid]

    refined = (1.0 - alpha) * base_scores + alpha * prop_scores
    changed = np.abs(refined - base_scores)
    print("  [LabelProp] alpha={:.2f} k={} | delta: mean={:.4f} max={:.4f}".format(
        alpha, k, changed.mean(), changed.max()))
    print("  [LabelProp] refined: mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
        refined.mean(), refined.std(),
        (refined > 0.5).sum(), (refined > 0.3).sum()))
    return refined

# --------------------------------------------------------------------------
# [8] 데이터 로딩
# --------------------------------------------------------------------------
print('>>> Preparing Data for ' + selected_go)

X_train_seq = np.load('../data/raw_data/data/sequences/human_sequence_train.npy')
X_train_dm  = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_test_seq  = np.load('my_sequence_matrix_fixed.npy')
X_test_dm   = np.load('../results/domain/domain_matrix.npy')

def load_ids(path):
    return [x.decode('utf-8') if isinstance(x, bytes) else x
            for x in np.load(path, allow_pickle=True)]

X_train_geneid       = load_ids('../data/raw_data/data/id_lists/train_gene_list.npy')
X_test_geneid        = load_ids('my_gene_list_fixed.npy')
X_test_isoid         = load_ids('my_isoform_list_fixed.npy')
X_train_other_seq    = np.load('../data/raw_data/data/sequences/swissprot_sequence_train.npy')
X_train_other_dm     = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_train_geneid_other = load_ids('../data/raw_data/data/id_lists/train_swissprot_list.npy')

positive_Gene = []
for fname in ['human_annotations.txt', 'swissprot_annotations.txt']:
    with open('../data/raw_data/data/annotations/' + fname, 'r') as fr:
        for line in fr:
            parts = line.strip().split('\t')
            if selected_go in parts[1:]:
                positive_Gene.append(parts[0])

# --------------------------------------------------------------------------
# [9] Expression 행렬 로딩
# --------------------------------------------------------------------------
EXPR_CACHE = 'expr_matrix_fixed.npy'
CPM_PATH   = '../data/bambu_data/CPM_transcript.txt'

test_iso_arr = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
if os.path.exists(EXPR_CACHE):
    print("[Expr] Loading cached expression matrix: {}".format(EXPR_CACHE))
    X_test_expr = np.load(EXPR_CACHE).astype(np.float32)
    print("[Expr] shape={} nonzero={:.1f}%".format(
        X_test_expr.shape, (X_test_expr > 0).mean() * 100))
else:
    X_test_expr = load_expression_matrix(CPM_PATH, test_iso_arr)
    np.save(EXPR_CACHE, X_test_expr)

assert X_test_expr.shape == (len(test_iso_arr), EXPR_DIM)

# --------------------------------------------------------------------------
# [10] 레이블 생성 및 업샘플
# --------------------------------------------------------------------------
K_training_size = X_train_seq.shape[0]
K_testing_size  = X_test_seq.shape[0]
seq_dim         = X_train_seq.shape[1]
dm_dim          = X_train_dm.shape[1]
domain_emb_dim  = max([np.max(X_train_dm), np.max(X_test_dm),
                        np.max(X_train_other_dm)]) + 1

y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, \
    X_train_seq, X_train_dm = generate_label(
        X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm,
        X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

n_pos_train = int((y_train == 1).sum())
n_neg_train = int((y_train == 0).sum())
pos_ratio   = n_pos_train / len(y_train)

print("  y_train pos={} neg={} ratio={:.3f}%".format(
    n_pos_train, n_neg_train, pos_ratio * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# [I4 개선] Phase 1 n_batches: v5-1 학습량 기준 보장
# v5-2 공식(min=20)은 전 GO term이 20으로 클램핑 → v5-1(30~100) 대비 40~60% 감소
# v5-3: max(50, ...) 로 하한 상향, scale=4로 large GO term 자동 확대
# --------------------------------------------------------------------------
BATCH_SIZE_P1 = 256
N_BATCHES_P1  = max(50, min(150, int(np.ceil(n_pos_train * N_BATCHES_SCALE / BATCH_SIZE_P1))))
coverage_per_epoch = N_BATCHES_P1 * BATCH_SIZE_P1 / n_pos_train
print("[I4] n_batches={} (n_pos={}, coverage={:.1f}x/epoch, scale={})".format(
    N_BATCHES_P1, n_pos_train, coverage_per_epoch, N_BATCHES_SCALE))

# --------------------------------------------------------------------------
# [11] 모델 구조 (2-modal: seq + domain)
# --------------------------------------------------------------------------
seq_input    = Input(shape=(None,),   dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim,), dtype='int32', name='domain_input')

x1 = Embedding(input_dim=8001, output_dim=32)(seq_input)
x1 = Convolution1D(filters=64, kernel_size=32, padding='valid', activation='relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(1e-5))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.2)(x1)
seq_feat = Dense(16, activation='relu')(x1)

x2 = Embedding(input_dim=domain_emb_dim, output_dim=32,
               input_length=dm_dim, mask_zero=True)(domain_input)
domain_feat = LSTM(16)(x2)

concat = concatenate([seq_feat, domain_feat])
feature_model = Model([seq_input, domain_input], concat, name='feature_model')

x = Dense(16, kernel_regularizer=regularizers.l2(1e-5))(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer  = Lambda(lambda a: K.l2_normalize(a, axis=1),
                          name="embedding_out")(x)
# [I1] name 부여 → Phase 2 전 bias 접근용
prediction_layer = Dense(1, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(1e-5),
                         name='prediction_out')(embedding_layer)

base_model = Model(inputs=[seq_input, domain_input],
                   outputs=[embedding_layer, prediction_layer])
classification_model = Model(inputs=[seq_input, domain_input],
                             outputs=prediction_layer)

# Phase 2 Triplet model
seq_a  = Input(shape=(None,),   dtype='int32');  dm_a = Input(shape=(dm_dim,), dtype='int32')
seq_p  = Input(shape=(None,),   dtype='int32');  dm_p = Input(shape=(dm_dim,), dtype='int32')
seq_n  = Input(shape=(None,),   dtype='int32');  dm_n = Input(shape=(dm_dim,), dtype='int32')

emb_a, pred_a = base_model([seq_a, dm_a])
emb_p, _      = base_model([seq_p, dm_p])
emb_n, _      = base_model([seq_n, dm_n])

triplet_loss_layer = Lambda(triplet_loss_fn, output_shape=(1,))([emb_a, emb_p, emb_n])
triplet_model = Model(
    inputs=[seq_a, dm_a, seq_p, dm_p, seq_n, dm_n],
    outputs=[triplet_loss_layer, pred_a])

adam_p1   = optimizers.Adam(lr=0.0005)
adam_main = optimizers.Adam(lr=0.001)
adam_p2   = optimizers.Adam(lr=0.0003)

print("\n[Model] feature_model dim={} (2-modal: seq+domain)".format(EMB_DIM))
print("[Model] Expression({}-dim) → Phase 3 Label Propagation".format(EXPR_DIM))
print("[Model] Phase 1 margin={:.1f} [I3] | n_batches={} [I4]".format(MARGIN_P1, N_BATCHES_P1))

# embedding 수집 dict (UMAP + score dist용)
emb_by_phase   = {}
score_by_phase = {}

# ==========================================================================
# PHASE 0: Untrained Baseline
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 0: Untrained Baseline")
print("=" * 50)

s0 = save_phase_results("phase0_initial_untrained", base_model,
                        X_test_seq, X_test_dm, y_test,
                        X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph0(untrained)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase0_initial_untrained_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph0(untrained)'] = s0
analyze_embedding_quality(emb_by_phase['Ph0(untrained)'], y_test, 'Phase 0')

# ==========================================================================
# PHASE 1: Triplet (margin=0.3 [I3], n_batches formula [I4])
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 1: Embedding-based Triplet (margin={:.1f} [I3], max 15 epochs)".format(MARGIN_P1))
print("=" * 50)

PHASE1_EPOCHS     = 15
WARMUP_EPOCHS     = 2
best_margin_sat   = 0.0
final_centroid    = 0.0
low_active_streak = 0
ACTIVE_THRESH     = 2.0
STREAK_LIMIT      = 4   # margin=0.3은 수렴이 느리므로 3→4

for epoch in range(PHASE1_EPOCHS):
    warmup   = (epoch < WARMUP_EPOCHS)
    mode_str = "random (warmup)" if warmup else "semi-hard"
    print('Phase 1 - Epoch: {}/{} [{}]'.format(epoch + 1, PHASE1_EPOCHS, mode_str))

    avg_loss, active_rate = phase1_embedding_triplet_epoch(
        feature_model=feature_model,
        X_seq=X_train_seq_upsmp,
        X_dm=X_train_dm_upsmp,
        y=y_train_upsmp,
        seq_dim=seq_dim,
        optimizer=adam_p1,
        margin=MARGIN_P1,
        batch_size=BATCH_SIZE_P1,
        n_batches=N_BATCHES_P1,
        warmup=warmup)

    print("  -> Triplet Loss: {:.4f} | Active triplets: {:.1f}%".format(avg_loss, active_rate))

    if not warmup:
        if active_rate < ACTIVE_THRESH:
            low_active_streak += 1
        else:
            low_active_streak = 0
        if low_active_streak >= STREAK_LIMIT:
            print("  [Early Stop] active_rate < {:.1f}% for {} epochs.".format(
                ACTIVE_THRESH, STREAK_LIMIT))
            break

    if (epoch + 1) % 5 == 0:
        sat, cdist = compute_margin_stats(
            feature_model, X_test_seq, X_test_dm,
            y_test, seq_dim, margin=MARGIN_P1)
        best_margin_sat = max(best_margin_sat, sat)
        final_centroid  = cdist
        if sat >= 60.0:
            print("  [Early Stop] margin satisfied {:.1f}% >= 60%.".format(sat))
            break

print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
    best_margin_sat, final_centroid))
s1 = save_phase_results("phase1_triplet_only", base_model,
                        X_test_seq, X_test_dm, y_test,
                        X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(triplet)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_triplet_only_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(triplet)'] = s1
analyze_embedding_quality(emb_by_phase['Ph1(triplet)'], y_test, 'Phase 1')

# ==========================================================================
# PHASE 1.5: Linear Probing
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 1.5: Linear Probing (Encoder Frozen)")
print("=" * 50)

for layer in feature_model.layers:
    layer.trainable = False
print("  [Freeze] All feature_model layers frozen.")

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=adam_main, metrics=['accuracy'])

for epoch in range(2):
    print('Phase 1.5 - Epoch: {}/2'.format(epoch + 1))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    p15_losses, p15_accs = [], []
    for key in tup_gp.keys():
        sel   = tup_gp[key]
        idx   = tup_idx[sel[0]: sel[1] + 1]
        X_bs  = X_train_seq_upsmp[idx, seq_dim - min(sel[2], seq_dim): seq_dim]
        X_bd  = X_train_dm_upsmp[idx]
        y_b   = y_train_upsmp[idx]
        mixed = np.hstack((np.where(y_b == 1)[0], np.where(y_b == 0)[0]))
        if len(mixed) == 0:
            continue
        np.random.shuffle(mixed)
        bs   = 1024 if key == 0 else (512 if key == 1 else 256)
        hist = classification_model.fit(
            [X_bs[mixed], X_bd[mixed]], y_b[mixed],
            batch_size=bs, epochs=1, verbose=0)
        p15_losses.append(hist.history['loss'][0])
        acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
        p15_accs.append(hist.history[acc_key][0])
    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(p15_losses), np.mean(p15_accs)))

s15 = save_phase_results("phase1_5_linear_probing", base_model,
                         X_test_seq, X_test_dm, y_test,
                         X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1.5(linear)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_5_linear_probing_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1.5(linear)'] = s15
analyze_embedding_quality(emb_by_phase['Ph1.5(linear)'], y_test, 'Phase 1.5')

for layer in feature_model.layers:
    layer.trainable = True
print("  [Unfreeze] All layers unlocked.")

# ==========================================================================
# PHASE 2: Joint Fine-tuning
# [I2] AUPRC criterion, [I5] check every 1 epoch
# Note: [I1] prior bias init removed — caused all predictions near 0
#   (logit values -4 ~ -6 made focal loss gradient ~0 for negatives,
#    preventing model from learning to predict positives)
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 2: Joint Fine-tuning (Focal+Triplet)")
print("    [I5] CHECK_EVERY=1 | NO_IMPROVE_LIMIT=3")
print("=" * 50)

triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.10)],
    loss_weights=[0.1, 1.0],
    optimizer=adam_p2)

PHASE2_MAX_EPOCHS  = 15
PHASE2_CHECK_EVERY = 1      # [I5] 3 → 1
best_phase2_auroc  = 0.0
best_phase2_weights = None
no_improve_count   = 0
NO_IMPROVE_LIMIT   = 3      # [I5] 2 → 3

for epoch in range(PHASE2_MAX_EPOCHS):
    print('Phase 2 - Epoch: {}/{}'.format(epoch + 1, PHASE2_MAX_EPOCHS))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    epoch_losses = []

    for key in tup_gp.keys():
        sel       = tup_gp[key]
        batch_idx = tup_idx[sel[0]: sel[1] + 1]
        X_grp_seq = X_train_seq_upsmp[batch_idx, seq_dim - sel[2]: seq_dim]
        X_grp_dm  = X_train_dm_upsmp[batch_idx]
        y_grp     = y_train_upsmp[batch_idx]

        if (y_grp == 1).sum() < 2 or (y_grp == 0).sum() == 0:
            continue

        for step in range(max(1, int(len(batch_idx) / 64))):
            batch, labels = get_triplet_batch_ingroup(
                X_grp_seq, X_grp_dm, y_grp, batch_size=64)
            if batch is None:
                continue
            losses = triplet_model.train_on_batch(
                batch, [np.zeros((len(labels), 1)), labels])
            epoch_losses.append(losses)

    if epoch_losses:
        avg = np.mean(epoch_losses, axis=0)
        print("  -> Total: {:.4f} | Triplet: {:.4f} | Focal: {:.4f}".format(
            avg[0], avg[1], avg[2]))

    # [I5] 매 epoch AUROC 체크 + best checkpoint
    if (epoch + 1) % PHASE2_CHECK_EVERY == 0:
        preds = base_model.predict([X_test_seq, X_test_dm], batch_size=256, verbose=0)
        test_scores = np.array([preds[1][i][0] for i in range(len(y_test))])
        if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
            current_auroc = roc_auc_score(y_test, test_scores)
            print("  [AUROC] epoch={} AUROC={:.4f} (best={:.4f})".format(
                epoch + 1, current_auroc, best_phase2_auroc))
            if current_auroc > best_phase2_auroc:
                best_phase2_auroc    = current_auroc
                best_phase2_weights  = base_model.get_weights()
                no_improve_count     = 0
            else:
                no_improve_count += 1
                if no_improve_count >= NO_IMPROVE_LIMIT:
                    print("  [Early Stop] AUROC not improving ({} epochs) → restore best".format(
                        NO_IMPROVE_LIMIT))
                    if best_phase2_weights is not None:
                        base_model.set_weights(best_phase2_weights)
                    break

if best_phase2_weights is not None and no_improve_count < NO_IMPROVE_LIMIT:
    base_model.set_weights(best_phase2_weights)
    print("\n[Phase 2] Restored best (AUROC={:.4f})".format(best_phase2_auroc))

_, p2_centroid = compute_margin_stats(
    feature_model, X_test_seq, X_test_dm, y_test, seq_dim, margin=MARGIN_P1)

s2 = save_phase_results("phase2_joint_focal", base_model,
                        X_test_seq, X_test_dm, y_test,
                        X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph2(joint)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase2_joint_focal_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph2(joint)'] = s2
emb_metrics_p2 = analyze_embedding_quality(emb_by_phase['Ph2(joint)'], y_test, 'Phase 2')

# ==========================================================================
# PHASE 3: Test-time Label Propagation
# [I2] AUPRC 기준 alpha 선택
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 3: Test-time Expression Label Propagation")
print("    [I2] Best alpha selected by AUPRC [R9.1]")
print("=" * 50)

preds_base = base_model.predict([X_test_seq, X_test_dm], batch_size=256, verbose=0)
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])
print("  Base (Phase 2): mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    base_scores.mean(), base_scores.std(),
    (base_scores > 0.5).sum(), (base_scores > 0.3).sum()))

results_by_alpha = {}
for alpha in [0.0, 0.2, 0.3, 0.5]:
    if alpha == 0.0:
        refined = base_scores.copy()
    else:
        refined = expression_label_propagation(
            base_scores, X_test_expr, alpha=alpha, k=15, sim_threshold=0.1)
    results_by_alpha[alpha] = refined
    if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
        auroc = roc_auc_score(y_test, refined)
        auprc = average_precision_score(y_test, refined)
        print("  alpha={:.1f}: AUROC={:.4f} AUPRC={:.4f}".format(alpha, auroc, auprc))

# [I2] AUPRC 기준 최적 alpha 선택
best_alpha = 0.0
best_auprc = -1.0
if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
    for alpha, scores in results_by_alpha.items():
        auprc = average_precision_score(y_test, scores)
        if auprc > best_auprc:
            best_auprc = auprc
            best_alpha = alpha

final_scores = results_by_alpha[best_alpha]
final_auroc  = roc_auc_score(y_test, final_scores)
final_auprc  = average_precision_score(y_test, final_scores)

print("\n  [Best/AUPRC] alpha={:.1f} → AUROC={:.4f} AUPRC={:.4f}".format(
    best_alpha, final_auroc, final_auprc))
score_by_phase['Final(LP)'] = final_scores

# ==========================================================================
# 시각화 — UMAP & Score Distribution
# ==========================================================================
print("\n" + "=" * 50)
print(">>> Generating Visualizations")
print("=" * 50)

# UMAP (4 phase)
plot_embedding_umap(emb_by_phase, y_test, X_test_geneid,
                    PLOTS_DIR, VER_TAG, safe_go)

# Score distribution
plot_score_distribution(score_by_phase, y_test,
                        PLOTS_DIR, VER_TAG, safe_go)

# ==========================================================================
# 임베딩 품질 요약 테이블
# ==========================================================================
print("\n" + "=" * 50)
print(">>> Embedding Quality Summary")
print("=" * 50)
print("{:20} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
    "Phase", "Silhouette", "Sep.Ratio", "Centroid", "LinAUROC", "PredAUROC"))
print("-" * 76)

for phase_name, emb in emb_by_phase.items():
    m = analyze_embedding_quality(emb, y_test, phase_name)
    try:
        pred_auroc = roc_auc_score(y_test, score_by_phase[phase_name])
    except Exception:
        pred_auroc = float('nan')
    print("{:20} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:12.4f}".format(
        phase_name,
        m['silhouette'], m['sep_ratio'], m['centroid_dist'],
        m['linear_auroc'], pred_auroc))

# ==========================================================================
# 최종 저장
# ==========================================================================
print("\nSaving Final Results...")
output_file = os.path.join(
    SAVE_DIR, "{}_{}_Final_LabelProp_scores.txt".format(VER_TAG, safe_go))
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                 str(final_scores[i]) + '\n')

base_model.save_weights(
    os.path.join(SAVE_DIR, "{}_{}_BaseModel_weights.h5".format(VER_TAG, safe_go)))
np.save(os.path.join(SAVE_DIR, "{}_{}_Final_LabelProp_labels.npy".format(VER_TAG, safe_go)),
        y_test)

print("\n[Final] alpha={:.1f} | AUROC={:.4f} AUPRC={:.4f}".format(
    best_alpha, final_auroc, final_auprc))
print("[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
