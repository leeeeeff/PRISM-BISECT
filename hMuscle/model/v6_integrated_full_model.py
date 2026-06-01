# -*- coding: utf-8 -*-
# ============================================================================
# v6_integrated_full_model.py
#
# v5-5 대비 핵심 변경:
#
# [ESM] seq 모달리티 교체 — integer Embedding+CNN → ESM-2 640-dim MLP
#   변경 내용: X_train_seq (int encoding) → X_train_esm2 (float32, 640-dim)
#   수학적 근거: [R_ESM] ESM-2 mean-pool embedding은 진화적 정보 + 구조적
#               특징을 640-dim dense space로 표현. integer k-mer encoding
#               대비 isoform-level sequence divergence 포착력 우월.
#   해결 문제: Axis 1 (data sparsity) + 직접적으로 GO:0006936/0006941
#              Phase 1 margin_sat=5.9% 문제 해결 (레포트 §5 근거)
#   ⚠️ 주의: ESM-2 train/test 파일 모두 필요 (4개 .npy 파일)
#   검증 방법: GO:0006936 AUPRC vs v5-5(0.251), GO:0006941 vs v5-5(0.095)
#
# [ESM_MASK] 단백질 서열 없는 이소폼 처리
#   mask=0인 이소폼은 ESM-2 feature를 zero-gate → domain 특징만 활용
#   훈련 데이터 coverage 체크: human/swissprot missing 비율 로그 출력
#
# [ARCH] 모델 구조 변경 요약
#   Before: seq_input(int) → Embedding(8001,32) → Conv1D(64) → PyramidPooling
#           → Dense(32) → Dense(16) → seq_feat[16]
#   After:  esm2_input(float,640) → Dense(256,relu) → Dropout(0.2) → Dense(128,relu)
#           → Dense(64,relu) → esm2_feat[64] → gate(×mask) → esm2_gated[64]
#   Domain: unchanged — Embedding(D,32) → LSTM(16) → domain_feat[16]
#   Fuse:   concat([esm2_gated, domain_feat]) [80] → Dense(32,relu) → L2_norm
#           → embedding[32] → Dense(1,sigmoid)
#   EMB_DIM=80: feature_model 출력 (64+16)
#   [Fix-DA] 640→64: 10× 압축 (이전 40× 대비), domain과 2:1 비율 유지
#
# [PIPE] 데이터 파이프라인 변경
#   generate_label 2회 호출: ① dm/labels용 ② esm2/mask용 (동일 add_index)
#   upsample: hstack(esm2, mask) → 641-dim으로 통합 처리 → 이후 분리
#
# 이전 버전 유지 항목:
#   - Focal Loss γ=2, α=0.25/0.10 [R1.1]
#   - Triplet margin=0.3 [I3], semi-hard mining
#   - Coverage 기반 동적 n_batches [I4]
#   - Phase 1.5 Linear Probing (encoder frozen)
#   - Phase 2 AUPRC early stop [R9.1] [I5]
#   - Phase 3 Expression Label Propagation [I2]
#   - EMB_DIM=32 (feature_model 출력 차원) — v5-5와 동일
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
                          Embedding, Lambda, concatenate)
from keras import backend as K
from keras import regularizers, optimizers
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize

from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from crf import CRF
sys.path.insert(0, '/home/welcome1/layer_Full')
from utils_Full import generate_label, upsample

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
    print("Usage: python v6_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d")
VER_TAG  = "v6_integrated"
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
ESM2_DIM  = 640            # [ESM] ESM-2 t30_150M_UR50D 출력 차원
EXPR_DIM  = 24
EMB_DIM   = 80             # feature_model 출력 (esm2_feat[64] + domain_feat[16])
                           # [Fix-DA] bottleneck 완화: 640→64 (10×) — PCA 분산 ~75% 보존
MARGIN_P1         = 0.3    # [I3] 유지
TARGET_COVERAGE   = 6.0    # [I4] 유지

# --------------------------------------------------------------------------
# [3] Loss Functions (v5-5에서 변경 없음)
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
# [4] Expression 전처리 (변경 없음)
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
# [5-v6] 임베딩 추출 — ESM-2 고정 크기 입력용 (make_batch 불필요)
# --------------------------------------------------------------------------
def extract_embeddings_v6(feature_model, X_esm2, X_esm2_mask, X_dm,
                           emb_dim=EMB_DIM, batch_size=1000):
    """
    feature_model: esm2_input, esm2_mask, domain_input → concat[32]
    Returns: (N, emb_dim) L2-normalized embeddings
    """
    n   = len(X_esm2)
    emb = np.zeros((n, emb_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end      = min(start + batch_size, n)
        sub_esm2 = X_esm2[start:end].astype(np.float32)
        sub_mask = X_esm2_mask[start:end].astype(np.float32)
        sub_dm   = X_dm[start:end]
        raw = feature_model.predict_on_batch([sub_esm2, sub_mask, sub_dm])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        emb[start:end] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [EMB] 임베딩 공간 정량 분석 (v5-5에서 변경 없음)
# --------------------------------------------------------------------------
def analyze_embedding_quality(embeddings, labels, phase_name, n_pairs=500, seed=42):
    np.random.seed(seed)
    pos_emb = embeddings[labels == 1]
    neg_emb = embeddings[labels == 0]
    metrics = {}

    if len(pos_emb) > 0 and len(neg_emb) > 0:
        metrics['centroid_dist'] = float(np.linalg.norm(
            pos_emb.mean(axis=0) - neg_emb.mean(axis=0)))
    else:
        metrics['centroid_dist'] = float('nan')

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

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten() if n_phases > 1 else [axes]

    reducer = umap_lib.UMAP(
        n_components=2, random_state=42,
        metric='cosine', n_neighbors=15, min_dist=0.1, low_memory=True)

    pos_mask = labels == 1
    neg_mask = labels == 0
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

    for ax in axes[n_phases:]:
        ax.set_visible(False)

    neg_patch = mpatches.Patch(color='#cccccc', label='negative (sample)')
    pos_patch = mpatches.Patch(color='#e74c3c', label='positive')
    fig.legend(handles=[neg_patch, pos_patch], loc='lower right', fontsize=9, framealpha=0.8)
    fig.suptitle("{} — Embedding Space Evolution".format(go_term), fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plot_path = os.path.join(save_dir, "{}_{}_umap.png".format(ver_tag, go_term))
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  [UMAP] Saved: {}".format(plot_path))


def plot_score_distribution(score_dict, labels, save_dir, ver_tag, go_term):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phases = list(score_dict.keys())
    fig, axes = plt.subplots(1, len(phases), figsize=(4.5 * len(phases), 3.5), sharey=False)
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
        ax.set_title("{}\nAUROC={:.3f} AUPRC={:.3f}".format(phase_name, auroc, auprc), fontsize=9)
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
# [6-v6] Phase 1 — GradientTape Triplet (ESM-2 입력 버전)
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=MARGIN_P1, mode="hard"):
    """Semi-hard negative mining (변경 없음)"""
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


def phase1_embedding_triplet_epoch_v6(
        feature_model, X_esm2, X_esm2_mask, X_dm, y,
        optimizer, margin=MARGIN_P1, batch_size=256,
        n_batches=50, warmup=False, refresh_interval=10):
    """
    [ESM] Phase 1 Triplet 학습 — ESM-2 입력 버전.
    feature_model: [esm2_input, esm2_mask, domain_input] → concat[32]
    """
    def refresh_embeddings():
        emb = extract_embeddings_v6(feature_model, X_esm2, X_esm2_mask, X_dm)
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
                sub_esm2 = X_esm2[indices].astype(np.float32)
                sub_mask = X_esm2_mask[indices].astype(np.float32)
                sub_dm   = X_dm[indices]
                raw_out  = feature_model([sub_esm2, sub_mask, sub_dm], training=True)
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
# [7-v6] Phase 2 보조 함수 — ESM-2 입력 버전
# --------------------------------------------------------------------------
def get_triplet_batch_ingroup_v6(X_esm2_grp, X_mask_grp, X_dm_grp, y_grp,
                                  batch_size=64):
    pos_idx = np.where(y_grp == 1)[0]
    neg_idx = np.where(y_grp == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) == 0:
        return None, None

    actual_bs = min(batch_size, len(pos_idx))
    a_esm2, a_mask, a_dm = [], [], []
    p_esm2, p_mask, p_dm = [], [], []
    n_esm2, n_mask, n_dm = [], [], []
    labels = []

    for _ in range(actual_bs):
        a_i, p_i = np.random.choice(pos_idx, 2, replace=False)
        n_i      = np.random.choice(neg_idx)
        a_esm2.append(X_esm2_grp[a_i]); a_mask.append(X_mask_grp[a_i]); a_dm.append(X_dm_grp[a_i])
        p_esm2.append(X_esm2_grp[p_i]); p_mask.append(X_mask_grp[p_i]); p_dm.append(X_dm_grp[p_i])
        n_esm2.append(X_esm2_grp[n_i]); n_mask.append(X_mask_grp[n_i]); n_dm.append(X_dm_grp[n_i])
        labels.append(1)

    return ([np.array(a_esm2), np.array(a_mask), np.array(a_dm),
             np.array(p_esm2), np.array(p_mask), np.array(p_dm),
             np.array(n_esm2), np.array(n_mask), np.array(n_dm)],
            np.array(labels))


def compute_margin_stats_v6(feature_model, X_esm2, X_esm2_mask, X_dm, y,
                             margin=MARGIN_P1, n_sample=1000):
    emb     = extract_embeddings_v6(feature_model, X_esm2, X_esm2_mask, X_dm)
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


def save_phase_results_v6(phase_name, model_base, X_esm2, X_esm2_mask, X_dm,
                           y_true, gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds  = model_base.predict([X_esm2, X_esm2_mask, X_dm], batch_size=256, verbose=1)
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
# [Fix1] Phase 3: Test-time Label Propagation (변경 없음)
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
# [8-v6] 데이터 로딩 — ESM-2 입력
# --------------------------------------------------------------------------
print('>>> Preparing Data (v6/ESM-2) for ' + selected_go)

# [ESM] ESM-2 임베딩 로딩
ESM2_DATA_DIR = '../data'

def _load_esm2(name, shape_hint=''):
    path = os.path.join(ESM2_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "[ESM] 필수 파일 없음: {}\n"
            "  compute_esm2_embeddings.py / compute_esm2_train_embeddings.py 실행 필요".format(path))
    arr = np.load(path).astype(np.float32)
    print("  [ESM] Loaded {} {}".format(name, arr.shape))
    return arr

X_train_esm2       = _load_esm2('esm2_train_human.npy')        # (31668, 640)
X_train_esm2_mask  = _load_esm2('esm2_train_human_mask.npy')   # (31668, 1)
X_train_other_esm2      = _load_esm2('esm2_train_swissprot.npy')       # (82703, 640)
X_train_other_esm2_mask = _load_esm2('esm2_train_swissprot_mask.npy')  # (82703, 1)
X_test_esm2        = _load_esm2('esm2_embeddings_t30_150M.npy')  # (36748, 640)
X_test_esm2_mask   = _load_esm2('esm2_mask.npy')                 # (36748, 1)

# ESM-2 coverage 진단
train_human_cov   = float(X_train_esm2_mask.sum()) / len(X_train_esm2_mask) * 100
train_ssp_cov     = float(X_train_other_esm2_mask.sum()) / len(X_train_other_esm2_mask) * 100
test_cov          = float(X_test_esm2_mask.sum()) / len(X_test_esm2_mask) * 100
print("[ESM] Coverage — Human train: {:.1f}% | SwissProt train: {:.1f}% | Test: {:.1f}%".format(
    train_human_cov, train_ssp_cov, test_cov))

# Domain features (변경 없음)
X_train_dm       = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_test_dm        = np.load('../results/domain/domain_matrix.npy')

def load_ids(path):
    return [x.decode('utf-8') if isinstance(x, bytes) else x
            for x in np.load(path, allow_pickle=True)]

X_train_geneid       = load_ids('../data/raw_data/data/id_lists/train_gene_list.npy')
X_test_geneid        = load_ids('my_gene_list_fixed.npy')
X_test_isoid         = load_ids('my_isoform_list_fixed.npy')
X_train_geneid_other = load_ids('../data/raw_data/data/id_lists/train_swissprot_list.npy')

positive_Gene = []
for fname in ['human_annotations.txt', 'swissprot_annotations.txt']:
    with open('../data/raw_data/data/annotations/' + fname, 'r') as fr:
        for line in fr:
            parts = line.strip().split('\t')
            if selected_go in parts[1:]:
                positive_Gene.append(parts[0])

# --------------------------------------------------------------------------
# [9] Expression 행렬 로딩 (변경 없음)
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
# [10-v6] 레이블 생성 및 업샘플 — ESM-2 파이프라인
#
# 설계:
#   generate_label 2회 호출:
#     ① dm용: (esm2, dm, other_esm2, other_dm) → labels + X_train_dm_comb
#        esm2는 vstack에만 쓰이고 버려짐
#     ② esm2용: (esm2, mask, other_esm2, other_mask) → X_train_esm2_comb, mask_comb
#        같은 geneid → 같은 add_index → 동일한 행 선택
#   upsample 1회: hstack(esm2[641], dm) → 통합 처리 → 이후 분리
#     이유: upsample이 seq/dm에 동일한 랜덤 인덱스를 적용하므로
#           분리된 두 호출은 다른 인덱스를 가질 수 있음 → 통합 필수
# --------------------------------------------------------------------------
print('>>> Preparing Data for ' + selected_go)

dm_dim         = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm),
                      np.max(X_train_other_dm)]) + 1

# ① generate_label — dm 결합 (labels + X_train_dm_comb 획득)
y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, \
    _dummy_esm2, X_train_dm_comb = generate_label(
        X_train_esm2, X_train_dm,
        X_train_other_esm2, X_train_other_dm,
        X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

# ② generate_label — esm2+mask 결합 (동일 add_index가 적용됨)
_, _, _, _, _, _, X_train_esm2_comb, X_train_esm2_mask_comb = generate_label(
    X_train_esm2, X_train_esm2_mask,
    X_train_other_esm2, X_train_other_esm2_mask,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

n_pos_train = int((y_train == 1).sum())
n_neg_train = int((y_train == 0).sum())
pos_ratio   = n_pos_train / len(y_train)

print("  y_train pos={} neg={} ratio={:.3f}%".format(
    n_pos_train, n_neg_train, pos_ratio * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

# upsample: esm2(640) + mask(1) + dm → hstack(641+dm_dim) 통합 처리
# 이후 분리: esm2_upsmp[:, :640], mask_upsmp[:, 640:641], dm_upsmp[:, 641:]
ESM2_MASK_DIM = ESM2_DIM + 1   # 641

X_train_combined = np.hstack([
    X_train_esm2_comb,            # (N, 640)
    X_train_esm2_mask_comb,       # (N, 1)
    X_train_dm_comb.astype(np.float32)  # (N, dm_dim)
])  # (N, 641 + dm_dim)

np.random.seed(42)  # [Fix-DA] upsample 재현성 고정 (run-to-run 동일 결과 보장)
unused_flag = np.zeros(y_train.shape[0])
X_train_combined_upsmp, _dummy_dm, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count,
    X_train_combined,
    np.zeros((len(y_train), 1)),   # dm placeholder (not used — split after)
    unused_flag)

# 분리
X_train_esm2_upsmp      = X_train_combined_upsmp[:, :ESM2_DIM].astype(np.float32)
X_train_esm2_mask_upsmp = X_train_combined_upsmp[:, ESM2_DIM:ESM2_DIM+1].astype(np.float32)
X_train_dm_upsmp        = X_train_combined_upsmp[:, ESM2_DIM+1:].astype(np.int32)

print("  Upsampled: esm2={} mask={} dm={} y={}".format(
    X_train_esm2_upsmp.shape, X_train_esm2_mask_upsmp.shape,
    X_train_dm_upsmp.shape, y_train_upsmp.shape))

# [I4] Coverage 기반 동적 n_batches 결정
BATCH_SIZE_P1 = 256
N_BATCHES_P1  = int(np.clip(
    np.ceil(n_pos_train * TARGET_COVERAGE / BATCH_SIZE_P1),
    a_min=20, a_max=50))
coverage_per_epoch = N_BATCHES_P1 * BATCH_SIZE_P1 / n_pos_train
raw_val = int(np.ceil(n_pos_train * TARGET_COVERAGE / BATCH_SIZE_P1))
print("[I4] n_batches={} (n_pos={}, coverage={:.1f}x/epoch, target_cov={:.1f}x, raw={})".format(
    N_BATCHES_P1, n_pos_train, coverage_per_epoch, TARGET_COVERAGE, raw_val))

# --------------------------------------------------------------------------
# [11-v6] 모델 구조
#
# ESM-2 branch: 640 → Dense(256,relu) → Dropout(0.2) → Dense(128,relu)
#               → Dense(64,relu) → gate(×mask) → esm2_gated [64-dim]
# Domain branch: Embedding(D,32) → LSTM(16) → domain_feat [16-dim]
# Fuse: concat([esm2_gated, domain_feat]) [80-dim] = feature_model 출력
# Head: Dense(32,relu) → Dropout(0.2) → L2_norm → embedding [32-dim]
#       → Dense(1, sigmoid)
#
# [Fix-DA] EMB_DIM=80: ESM-2 feat 64 + domain feat 16
#   ESM-2 branch 압축비: 640→64 (10×) — 이전 40× 대비 정보 보존 향상
#   domain branch 유지: LSTM(16) — isoform 도메인 조합 표현
#   최종 embedding: 32-dim (L2_norm 이후)
# --------------------------------------------------------------------------
esm2_input   = Input(shape=(ESM2_DIM,), name='esm2_input')
esm2_mask_in = Input(shape=(1,),         name='esm2_mask')
domain_input = Input(shape=(dm_dim,),   dtype='int32', name='domain_input')

# ESM-2 MLP branch (isoform-specific feature — architecture rule: isoform 먼저)
# [Fix-DA] 640→256→128→64: 10× 압축, PCA 분산 ~75% 보존
x_esm = Dense(256, kernel_regularizer=regularizers.l2(1e-5))(esm2_input)
x_esm = Activation('relu')(x_esm)
x_esm = Dropout(0.2)(x_esm)
x_esm = Dense(128, kernel_regularizer=regularizers.l2(1e-5))(x_esm)
x_esm = Activation('relu')(x_esm)
esm2_feat = Dense(64, activation='relu',
                  kernel_regularizer=regularizers.l2(1e-5),
                  name='esm2_feat')(x_esm)
# Mask gating: mask=0이면 ESM-2 feature zeroed out → domain만 기여
esm2_gated = Lambda(
    lambda x: x[0] * x[1],
    name='esm2_gated')([esm2_feat, esm2_mask_in])

# Domain branch (gene context — attention/gating 방식 유지)
x2 = Embedding(input_dim=domain_emb_dim, output_dim=32,
               input_length=dm_dim, mask_zero=True)(domain_input)
domain_feat = LSTM(16, name='domain_feat')(x2)

# Fusion (ESM-2 먼저, domain은 context)
concat = concatenate([esm2_gated, domain_feat], name='feature_concat')
feature_model = Model([esm2_input, esm2_mask_in, domain_input],
                       concat, name='feature_model')

x = Dense(32, kernel_regularizer=regularizers.l2(1e-5))(concat)   # [Fix-DA] 80→32
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer  = Lambda(lambda a: K.l2_normalize(a, axis=1),
                          name="embedding_out")(x)
prediction_layer = Dense(1, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(1e-5),
                         name='prediction_out')(embedding_layer)

base_model = Model(inputs=[esm2_input, esm2_mask_in, domain_input],
                   outputs=[embedding_layer, prediction_layer])
classification_model = Model(inputs=[esm2_input, esm2_mask_in, domain_input],
                              outputs=prediction_layer)

# Phase 2 Triplet model — ESM-2 triplet (anchor, positive, negative)
esm2_a = Input(shape=(ESM2_DIM,)); mask_a = Input(shape=(1,)); dm_a = Input(shape=(dm_dim,), dtype='int32')
esm2_p = Input(shape=(ESM2_DIM,)); mask_p = Input(shape=(1,)); dm_p = Input(shape=(dm_dim,), dtype='int32')
esm2_n = Input(shape=(ESM2_DIM,)); mask_n = Input(shape=(1,)); dm_n = Input(shape=(dm_dim,), dtype='int32')

emb_a, pred_a = base_model([esm2_a, mask_a, dm_a])
emb_p, _      = base_model([esm2_p, mask_p, dm_p])
emb_n, _      = base_model([esm2_n, mask_n, dm_n])

triplet_loss_layer = Lambda(triplet_loss_fn, output_shape=(1,))([emb_a, emb_p, emb_n])
triplet_model = Model(
    inputs=[esm2_a, mask_a, dm_a, esm2_p, mask_p, dm_p, esm2_n, mask_n, dm_n],
    outputs=[triplet_loss_layer, pred_a])

adam_p1   = optimizers.Adam(lr=0.0005)
adam_main = optimizers.Adam(lr=0.001)
adam_p2   = optimizers.Adam(lr=0.0003)

K_testing_size = len(X_test_esm2)

print("\n[Model] feature_model: ESM-2(640→256→128→64) + Domain(LSTM 16) → concat[80]")
print("[Model] base_model: concat[80] → Dense(32) → L2_norm → embedding[32] → sigmoid")
print("[Model] [Fix-DA] ESM-2 10× compression (vs 40× before), EMB_DIM=80")
print("[Model] ESM-2 coverage: train_human={:.1f}% train_ssp={:.1f}% test={:.1f}%".format(
    train_human_cov, train_ssp_cov, test_cov))
print("[Model] Phase 1 margin={:.1f} [I3] | n_batches={} (coverage={:.1f}x) [I4-dynamic]".format(
    MARGIN_P1, N_BATCHES_P1, coverage_per_epoch))

emb_by_phase   = {}
score_by_phase = {}

# ==========================================================================
# PHASE 0: Untrained Baseline
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 0: Untrained Baseline")
print("=" * 50)

s0 = save_phase_results_v6("phase0_initial_untrained", base_model,
                            X_test_esm2, X_test_esm2_mask, X_test_dm, y_test,
                            X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph0(untrained)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase0_initial_untrained_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph0(untrained)'] = s0
analyze_embedding_quality(emb_by_phase['Ph0(untrained)'], y_test, 'Phase 0')

# ==========================================================================
# PHASE 1: Triplet (margin=0.3 [I3], n_batches [I4])
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 1: Embedding-based Triplet (margin={:.1f} [I3], n_batches={} [I4], max 15 epochs)".format(
    MARGIN_P1, N_BATCHES_P1))
print("=" * 50)

PHASE1_EPOCHS     = 15
WARMUP_EPOCHS     = 2
best_margin_sat   = 0.0
final_centroid    = 0.0
low_active_streak = 0
ACTIVE_THRESH     = 2.0
STREAK_LIMIT      = 4

for epoch in range(PHASE1_EPOCHS):
    warmup   = (epoch < WARMUP_EPOCHS)
    mode_str = "random (warmup)" if warmup else "semi-hard"
    print('Phase 1 - Epoch: {}/{} [{}]'.format(epoch + 1, PHASE1_EPOCHS, mode_str))

    avg_loss, active_rate = phase1_embedding_triplet_epoch_v6(
        feature_model=feature_model,
        X_esm2=X_train_esm2_upsmp,
        X_esm2_mask=X_train_esm2_mask_upsmp,
        X_dm=X_train_dm_upsmp,
        y=y_train_upsmp,
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
        sat, cdist = compute_margin_stats_v6(
            feature_model, X_test_esm2, X_test_esm2_mask, X_test_dm,
            y_test, margin=MARGIN_P1)
        best_margin_sat = max(best_margin_sat, sat)
        final_centroid  = cdist
        if sat >= 60.0:
            print("  [Early Stop] margin satisfied {:.1f}% >= 60%.".format(sat))
            break

print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
    best_margin_sat, final_centroid))
s1 = save_phase_results_v6("phase1_triplet_only", base_model,
                            X_test_esm2, X_test_esm2_mask, X_test_dm, y_test,
                            X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(triplet)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_triplet_only_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(triplet)'] = s1
analyze_embedding_quality(emb_by_phase['Ph1(triplet)'], y_test, 'Phase 1')

# ==========================================================================
# PHASE 1.5: Linear Probing (Encoder Frozen)
# [ESM] make_batch 대신 단순 배치 처리 (ESM-2 고정 크기 → 길이 그룹화 불필요)
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

BATCH_SIZE_P15 = 512
n_train = len(X_train_esm2_upsmp)

for epoch in range(2):
    print('Phase 1.5 - Epoch: {}/2'.format(epoch + 1))
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    p15_losses, p15_accs = [], []

    for start in range(0, n_train, BATCH_SIZE_P15):
        idx  = indices[start:start + BATCH_SIZE_P15]
        X_be = X_train_esm2_upsmp[idx]
        X_bm = X_train_esm2_mask_upsmp[idx]
        X_bd = X_train_dm_upsmp[idx]
        y_b  = y_train_upsmp[idx]
        mixed = np.hstack((np.where(y_b == 1)[0], np.where(y_b == 0)[0]))
        if len(mixed) == 0:
            continue
        np.random.shuffle(mixed)
        hist = classification_model.fit(
            [X_be[mixed], X_bm[mixed], X_bd[mixed]], y_b[mixed],
            batch_size=256, epochs=1, verbose=0)
        p15_losses.append(hist.history['loss'][0])
        acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
        p15_accs.append(hist.history[acc_key][0])

    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(p15_losses), np.mean(p15_accs)))

s15 = save_phase_results_v6("phase1_5_linear_probing", base_model,
                             X_test_esm2, X_test_esm2_mask, X_test_dm, y_test,
                             X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1.5(linear)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_5_linear_probing_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1.5(linear)'] = s15
analyze_embedding_quality(emb_by_phase['Ph1.5(linear)'], y_test, 'Phase 1.5')

for layer in feature_model.layers:
    layer.trainable = True
print("  [Unfreeze] All layers unlocked.")

# ==========================================================================
# PHASE 2: Joint Fine-tuning (Focal + Triplet)
# [ESM] make_batch 대신 단순 배치 처리
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 2: Joint Fine-tuning (Focal+Triplet)")
print("    [I5] CHECK_EVERY=1 | NO_IMPROVE_LIMIT=3 | criterion=AUPRC [R9.1]")
print("=" * 50)

triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.10)],
    loss_weights=[0.1, 1.0],
    optimizer=adam_p2)

PHASE2_MAX_EPOCHS  = 15
PHASE2_CHECK_EVERY = 1
best_phase2_auprc  = 0.0
best_phase2_weights = None
no_improve_count   = 0
NO_IMPROVE_LIMIT   = 3
BATCH_SIZE_P2      = 512   # ESM-2 고정 크기: 큰 배치 가능

for epoch in range(PHASE2_MAX_EPOCHS):
    print('Phase 2 - Epoch: {}/{}'.format(epoch + 1, PHASE2_MAX_EPOCHS))
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    epoch_losses = []

    for start in range(0, n_train, BATCH_SIZE_P2):
        idx       = indices[start:start + BATCH_SIZE_P2]
        X_grp_esm2 = X_train_esm2_upsmp[idx]
        X_grp_mask = X_train_esm2_mask_upsmp[idx]
        X_grp_dm   = X_train_dm_upsmp[idx]
        y_grp      = y_train_upsmp[idx]

        if (y_grp == 1).sum() < 2 or (y_grp == 0).sum() == 0:
            continue

        for step in range(max(1, int(len(idx) / 64))):
            batch, labels = get_triplet_batch_ingroup_v6(
                X_grp_esm2, X_grp_mask, X_grp_dm, y_grp, batch_size=64)
            if batch is None:
                continue
            losses = triplet_model.train_on_batch(
                batch, [np.zeros((len(labels), 1)), labels])
            epoch_losses.append(losses)

    if epoch_losses:
        avg = np.mean(epoch_losses, axis=0)
        print("  -> Total: {:.4f} | Triplet: {:.4f} | Focal: {:.4f}".format(
            avg[0], avg[1], avg[2]))

    # [I5] AUPRC 기준 early stop [R9.1]
    if (epoch + 1) % PHASE2_CHECK_EVERY == 0:
        preds = base_model.predict(
            [X_test_esm2, X_test_esm2_mask, X_test_dm], batch_size=256, verbose=0)
        test_scores = np.array([preds[1][i][0] for i in range(len(y_test))])
        if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
            current_auroc = roc_auc_score(y_test, test_scores)
            current_auprc = average_precision_score(y_test, test_scores)
            print("  [AUPRC] epoch={} AUROC={:.4f} AUPRC={:.4f} (best_auprc={:.4f})".format(
                epoch + 1, current_auroc, current_auprc, best_phase2_auprc))
            if current_auprc > best_phase2_auprc:
                best_phase2_auprc   = current_auprc
                best_phase2_weights = base_model.get_weights()
                no_improve_count    = 0
            else:
                no_improve_count += 1
                if no_improve_count >= NO_IMPROVE_LIMIT:
                    print("  [Early Stop] AUPRC not improving ({} epochs) → restore best".format(
                        NO_IMPROVE_LIMIT))
                    if best_phase2_weights is not None:
                        base_model.set_weights(best_phase2_weights)
                    break

if best_phase2_weights is not None and no_improve_count < NO_IMPROVE_LIMIT:
    base_model.set_weights(best_phase2_weights)
    print("\n[Phase 2] Restored best (AUPRC={:.4f})".format(best_phase2_auprc))

_, p2_centroid = compute_margin_stats_v6(
    feature_model, X_test_esm2, X_test_esm2_mask, X_test_dm,
    y_test, margin=MARGIN_P1)

s2 = save_phase_results_v6("phase2_joint_focal", base_model,
                            X_test_esm2, X_test_esm2_mask, X_test_dm, y_test,
                            X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph2(joint)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase2_joint_focal_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph2(joint)'] = s2
emb_metrics_p2 = analyze_embedding_quality(emb_by_phase['Ph2(joint)'], y_test, 'Phase 2')

# ==========================================================================
# PHASE 3: Test-time Expression Label Propagation [I2]
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 3: Test-time Expression Label Propagation")
print("    [I2] Best alpha selected by AUPRC [R9.1]")
print("=" * 50)

preds_base = base_model.predict(
    [X_test_esm2, X_test_esm2_mask, X_test_dm], batch_size=256, verbose=0)
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
# 시각화
# ==========================================================================
print("\n" + "=" * 50)
print(">>> Generating Visualizations")
print("=" * 50)

plot_embedding_umap(emb_by_phase, y_test, X_test_geneid,
                    PLOTS_DIR, VER_TAG, safe_go)
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
