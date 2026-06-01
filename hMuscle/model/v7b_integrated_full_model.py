# -*- coding: utf-8 -*-
# ============================================================================
# v7b_integrated_full_model.py  (v6g + Type-Adaptive Phase 1 + Prototype Warm Init)
#
# v7b 핵심 변경 (v6g 대비):
# [v7b-1] Phase 1 — Type별 비대칭 Loss:
#   Type-A (sep_ratio ≥ 1.15, coherent):
#     Triplet Loss (L2 geometry 유지 → Phase 2 sigmoid 호환성 보장)
#     L(a,p,n) = max(d_L2(a,p) - d_L2(a,n) + margin, 0), margin=0.3
#   Type-B (sep_ratio < 1.15, heterogeneous):
#     Prototype Contrastive Loss (CLEAN, Science 2023)
#     k = Gap Statistic (Tibshirani 2001), k_max=5
#     L_proto + λ_div · L_diversity (EMA prototype 업데이트)
#   근거: Type-B positive는 생물학적으로 이질적 서브그룹 (심근/골격근 isoform 혼재).
#         단일 cluster로 강제하는 Triplet/SupCon 모두 margin_sat ~54% → 약한 gradient.
#         k prototype이 서브그룹을 별도 대표점으로 유지 → Phase 2에 다중 positive 공간 전달.
#
# [v7b-2] Phase 1→2 Interface — Prototype Warm Init (NEW):
#   Phase 1 종료 후 embedding 공간에서 prototype / centroid 방향으로
#   Phase 2 prediction head (Dense(1)) weights 초기화.
#   w_init = normalize(pos_rep - neg_centroid)  (scale=2.0)
#   b_init = -0.5
#   근거: SupCon (v7a) 실험에서 epoch=1 AUPRC=0.53 vs Triplet 0.77 격차의
#         원인이 Phase 2 cold start임을 확인.
#         Warm init으로 epoch=1 시작점을 높여 Phase 2 수렴 가속.
#
# v6g 변경사항 유지:
# [v6g-1] Phase 1.5 제거 (Phase 1 → Phase 2 직행)
#
# v6f 변경사항 유지:
# [v6f-1] Type-B Phase 2 LR=0.0002 | [v6f-2] patience=10, max_epochs=25
# [v6f-3] TARGET_COVERAGE=6        | [v6f-4] NO_IMPROVE_LIMIT 변수
#
# v6e 원본 (유지):
# [v6e-1] Phase 0 Type 분류 (sep_ratio < 1.15 → Type-B)
# [v6e-2] Adaptive Phase 2 lr/clipnorm/patience
# [v6e-3] LP 제거 (alpha=0.0 고정)
# [v6e-4] coverage clip: min=10, max=80
#
# [ARCH] 모델 구조 (v6d 동일)
#   ESM-2: 640 → Dense(256) → Dense(128) → Dense(64) → gated → esm2_gated[64]
#   CNN:   Embedding → Conv1D(64) → Conv1D(32) → GlobalMaxPool → Dense(32) → cnn_feat[32]
#   Domain: Embedding → LSTM(16) → domain_feat[16]
#   DomainDelta: sign(251) → Dense(64) → Dense(16) → dd_feat[16]
#   Fusion: concat[128] → Dense(64, relu) → L2_norm → emb[64] → Dense(1, sigmoid)
#
# [TRAIN] 단계별 trainable 범위
#   Phase 1  (Triplet or Proto): ESM-2 ✅ | CNN ❌(frozen) | Domain ✅ | Delta ✅
#   Phase 2  (Focal):            ESM-2 ❌(frozen) | CNN ✅ | Domain ✅ | Delta ✅
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
                          Embedding, Lambda, concatenate,
                          Conv1D, GlobalMaxPooling1D)
from keras import backend as K
from keras import regularizers, optimizers
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression

from crf import CRF
sys.path.insert(0, '/home/welcome1/layer_Full')
from utils_Full import generate_label, upsample

# [v7b] Proto-kN (Type-B) + Gap Statistic 임포트
from prototype_contrastive import (
    determine_k_gap_statistic,
    PrototypeContrastiveLoss,
    phase1_proto_epoch_hybrid,
)

# --------------------------------------------------------------------------
# [1] 환경 설정
# --------------------------------------------------------------------------
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
K.set_session(session)

try:
    script, selected_go = argv[0], argv[1]
except (IndexError, ValueError):
    print("Usage: python v7b_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go
if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d_%H%M")
VER_TAG  = "v7b_integrated"
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

log_path = os.path.join(SAVE_DIR, "{}_{}_Full.log".format(VER_TAG, safe_go))
sys.stdout = Logger(log_path)
print("\n[Info] {} | {} | {}\n".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), VER_TAG, SAVE_DIR))

# --------------------------------------------------------------------------
# [2] 상수
# --------------------------------------------------------------------------
ESM2_DIM            = 640
SEQ_LEN             = 1500
EXPR_DIM            = 24
DD_DIM              = 251
EMB_DIM             = 128
MARGIN_P1           = 0.3          # Triplet margin (Type-A)
TARGET_COVERAGE     = 6.0
SEP_RATIO_THRESHOLD = 1.15         # Type-A/B 분류 기준

# Prototype hyperparameters (Type-B)
PROTO_K_MAX         = 5            # Gap Statistic 최대 k
PROTO_TEMPERATURE   = 0.1          # τ for prototype contrastive loss
PROTO_EMA_DECAY     = 0.9          # α for EMA prototype update
PROTO_LAMBDA_DIV    = 0.1          # diversity loss weight
WARM_INIT_SCALE     = 2.0          # prediction_out 초기화 scale [v7b-2]

ESM2_TRAINABLE_NAMES  = {'esm2_d1', 'esm2_d2', 'esm2_feat'}
CNN_TRAINABLE_NAMES   = {'cnn_emb', 'cnn_conv1', 'cnn_conv2', 'cnn_feat'}
DELTA_TRAINABLE_NAMES = {'dd_dense1', 'dd_dense2'}

def set_esm2_trainable(model, trainable):
    for layer in model.layers:
        if layer.name in ESM2_TRAINABLE_NAMES:
            layer.trainable = trainable

def set_cnn_trainable(model, trainable):
    for layer in model.layers:
        if layer.name in CNN_TRAINABLE_NAMES:
            layer.trainable = trainable

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
# [5] 임베딩 추출 — 4-modality
# --------------------------------------------------------------------------
def extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd,
                               emb_dim=None, batch_size=512):
    # [v7b-fix2] emb_dim=None이면 모델 출력 크기 자동 감지 (64 or 128)
    n   = len(X_esm2)
    emb = None
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        raw  = feature_model.predict_on_batch([
            X_esm2[start:end].astype(np.float32),
            X_mask[start:end].astype(np.float32),
            X_seq[start:end],
            X_dm[start:end],
            X_dd[start:end].astype(np.float32)])
        if emb is None:
            actual_dim = raw.shape[1]
            emb = np.zeros((n, actual_dim), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        emb[start:end] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [EMB] 임베딩 품질 분석
# --------------------------------------------------------------------------
def analyze_embedding_quality(embeddings, labels, phase_name, n_pairs=500, seed=42):
    np.random.seed(seed)
    pos_emb = embeddings[labels == 1]
    neg_emb = embeddings[labels == 0]
    metrics = {}

    metrics['centroid_dist'] = float(np.linalg.norm(
        pos_emb.mean(axis=0) - neg_emb.mean(axis=0))) if len(pos_emb) > 0 and len(neg_emb) > 0 else float('nan')

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
        n_sub   = min(4000, len(embeddings))
        idx_sub = np.random.choice(len(embeddings), n_sub, replace=False)
        emb_sub = embeddings[idx_sub]
        lab_sub = labels[idx_sub].astype(int)
        try:
            metrics['silhouette'] = float(silhouette_score(
                emb_sub, lab_sub, metric='cosine',
                sample_size=min(2000, n_sub), random_state=seed)) if lab_sub.sum() >= 2 and (lab_sub==0).sum() >= 2 else float('nan')
        except Exception:
            metrics['silhouette'] = float('nan')
    else:
        metrics['silhouette'] = float('nan')

    if len(pos_emb) >= 5 and len(neg_emb) >= 5:
        try:
            scaler = StandardScaler()
            emb_sc = scaler.fit_transform(embeddings)
            lr = LogisticRegression(max_iter=300, C=1.0, class_weight='balanced', random_state=seed)
            lr.fit(emb_sc, labels.astype(int))
            lin_scores = lr.predict_proba(emb_sc)[:, 1]
            metrics['linear_auroc'] = float(roc_auc_score(labels, lin_scores))
        except Exception:
            metrics['linear_auroc'] = float('nan')
    else:
        metrics['linear_auroc'] = float('nan')

    print("  [EmbQuality | {}]".format(phase_name))
    print("    Silhouette (cosine):    {:+.4f}".format(metrics['silhouette']))
    print("    Sep ratio (inter/intra): {:.4f}".format(metrics['sep_ratio']))
    print("    Centroid dist:           {:.4f}".format(metrics['centroid_dist']))
    print("    Linear AUROC:            {:.4f}".format(metrics['linear_auroc']))
    return metrics

# --------------------------------------------------------------------------
# [6-v7b] Phase 1 Type-A: Triplet Loss (v6g 동일)
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=MARGIN_P1, mode="hard"):
    """Cross-gene semi-hard negative triplet 구성. [R3.1]"""
    n = min(batch_size, len(pos_indices))
    a_idx_list, p_idx_list, n_idx_list = [], [], []

    if mode == "hard" and len(neg_indices) >= 2:
        neg_emb     = embeddings[neg_indices]
        sampled_pos = np.random.choice(pos_indices, n, replace=False)
        for a_i in sampled_pos:
            p_candidates = pos_indices[pos_indices != a_i]
            if len(p_candidates) == 0:
                continue
            p_i   = np.random.choice(p_candidates)
            a_emb = embeddings[a_i]
            p_emb = embeddings[p_i]
            d_ap_sq = max(0.0, 2.0 - 2.0 * float(np.dot(a_emb, p_emb)))
            d_an_sq = np.clip(2.0 - 2.0 * (neg_emb @ a_emb), 0.0, None)
            semi_mask = (d_an_sq > d_ap_sq) & (d_an_sq < d_ap_sq + margin)
            semi_idx  = np.where(semi_mask)[0]
            if len(semi_idx) > 0:
                n_i = neg_indices[np.random.choice(semi_idx)]
            else:
                mod_idx = np.where(d_an_sq < d_ap_sq + 4.0 * margin)[0]
                n_i = neg_indices[np.random.choice(mod_idx)] if len(mod_idx) > 0 \
                      else np.random.choice(neg_indices)
            a_idx_list.append(a_i); p_idx_list.append(p_i); n_idx_list.append(n_i)
    else:
        for _ in range(n):
            a_i, p_i = np.random.choice(pos_indices, 2, replace=False)
            n_i      = np.random.choice(neg_indices)
            a_idx_list.append(a_i); p_idx_list.append(p_i); n_idx_list.append(n_i)
    return a_idx_list, p_idx_list, n_idx_list


def phase1_triplet_epoch_hybrid(
        feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
        optimizer, margin=MARGIN_P1, batch_size=256,
        n_batches=50, warmup=False, refresh_interval=10):
    """Phase 1 Triplet 1 epoch — ESM-2+Domain 학습, CNN frozen. [R3.1]"""
    def refresh_embeddings():
        emb = extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd)
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
                raw = feature_model(
                    [X_esm2[indices].astype(np.float32),
                     X_mask[indices].astype(np.float32),
                     X_seq[indices],
                     X_dm[indices],
                     X_dd[indices].astype(np.float32)], training=True)
                return tf.math.l2_normalize(raw, axis=1)

            emb_a = get_live_emb(a_idxs)
            emb_p = get_live_emb(p_idxs)
            emb_n = get_live_emb(n_idxs)

            pos_dist = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
            neg_dist = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
            raw_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            loss     = tf.reduce_mean(raw_loss)

        grads = tape.gradient(loss, feature_model.trainable_variables)
        g_v = [(g, v) for g, v in zip(grads, feature_model.trainable_variables)
               if g is not None]
        if g_v:
            optimizer.apply_gradients(g_v)

        batch_losses.append(float(loss.numpy()))
        active_count += int((raw_loss.numpy() > 0).sum())

    avg_loss    = np.mean(batch_losses) if batch_losses else 0.0
    active_rate = active_count / (n_batches * batch_size) * 100
    return avg_loss, active_rate


def compute_margin_stats_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
                                 margin=MARGIN_P1, n_sample=1000):
    """Triplet margin 만족률 및 centroid 거리 진단."""
    emb     = extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd)
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
        d_ap = max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[a_i], pos_emb[p_i])))
        d_an = max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[a_i], neg_emb[n_i])))
        margins.append(d_an - d_ap)
    margins       = np.array(margins)
    satisfied     = (margins > margin).mean() * 100
    centroid_dist = np.linalg.norm(pos_emb.mean(axis=0) - neg_emb.mean(axis=0))
    print("  [Margin|sq_L2] satisfied(>{:.2f}): {:.1f}% | centroid_dist: {:.4f}".format(
        margin, satisfied, centroid_dist))
    return satisfied, centroid_dist

# --------------------------------------------------------------------------
# [7-v7b] Phase 1→2 Prototype Warm Init (NEW) [v7b-2]
# --------------------------------------------------------------------------
def prototype_warm_init(base_model, embeddings_np, y_np, proto_fn=None,
                         scale=WARM_INIT_SCALE):
    """
    [v7b-2] Phase 1 종료 후 Phase 2 head 초기화.

    prediction_out Dense(1) 가중치를 positive / negative centroid 방향으로 설정.
    Type-A: pos_rep = positive centroid (L2-normalized embedding space)
    Type-B: pos_rep = mean(k prototypes) from proto_fn

    Parameters
    ----------
    base_model    : Keras Model (outputs [embedding_layer, prediction_layer])
    embeddings_np : numpy (N, 64) — Phase 1 종료 시점의 L2-normalized embedding
    y_np          : numpy (N,) binary labels
    proto_fn      : PrototypeContrastiveLoss instance (Type-B) or None (Type-A)
    scale         : float, weight magnitude (default 2.0)

    Returns
    -------
    True if successful, False otherwise
    """
    pos_emb = embeddings_np[y_np == 1]
    neg_emb = embeddings_np[y_np == 0]

    if len(pos_emb) < 2 or len(neg_emb) < 2:
        print("  [WarmInit] Skipped — insufficient pos/neg samples")
        return False

    # positive representative
    if proto_fn is not None and proto_fn.prototypes is not None:
        # Type-B: EMA prototypes의 중심을 사용
        protos = proto_fn.prototypes.numpy()  # (k, 64)
        norms  = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10
        protos_n = protos / norms
        pos_rep = protos_n.mean(axis=0)
        print("  [WarmInit] Type-B: using mean of {} prototypes as pos_rep".format(
            proto_fn.k))
    else:
        # Type-A: positive centroid
        pos_rep = pos_emb.mean(axis=0)
        print("  [WarmInit] Type-A: using positive centroid as pos_rep")

    pos_rep = pos_rep / (np.linalg.norm(pos_rep) + 1e-10)

    # negative representative: centroid
    neg_rep = neg_emb.mean(axis=0)
    neg_rep = neg_rep / (np.linalg.norm(neg_rep) + 1e-10)

    # 방향 벡터: neg → pos (in L2-normalized 64-dim space)
    w_dir = pos_rep - neg_rep
    w_norm = np.linalg.norm(w_dir) + 1e-10
    w_dir = w_dir / w_norm

    # 초기화 품질 진단: pos / neg centroid와의 dot product
    dot_pos = float(np.dot(w_dir, pos_rep))
    dot_neg = float(np.dot(w_dir, neg_rep))
    print("  [WarmInit] w_dir · pos={:.4f} | w_dir · neg={:.4f} | sep={:.4f}".format(
        dot_pos, dot_neg, dot_pos - dot_neg))

    # prediction_out Dense(1) 가중치 설정
    # kernel shape: (64, 1), bias shape: (1,)
    for layer in base_model.layers:
        if layer.name == 'prediction_out':
            w_init = (w_dir * scale).reshape(64, 1).astype(np.float32)
            b_init = np.array([-scale * 0.25], dtype=np.float32)  # 초기 threshold
            layer.set_weights([w_init, b_init])

            # 검증: pos/neg에 적용 시 예상 sigmoid 값
            logit_pos = float(np.dot(w_dir * scale, pos_rep) - scale * 0.25)
            logit_neg = float(np.dot(w_dir * scale, neg_rep) - scale * 0.25)
            sig_pos   = 1.0 / (1.0 + np.exp(-logit_pos))
            sig_neg   = 1.0 / (1.0 + np.exp(-logit_neg))
            print("  [WarmInit] prediction_out initialized | scale={:.1f}".format(scale))
            print("  [WarmInit] Expected sigmoid — pos_centroid={:.3f} | neg_centroid={:.3f}".format(
                sig_pos, sig_neg))
            return True

    print("  [WarmInit] WARNING: 'prediction_out' layer not found — random init 유지")
    return False

# --------------------------------------------------------------------------
# [8] Phase 3: Test-time Label Propagation (비활성)
# --------------------------------------------------------------------------
def expression_label_propagation(base_scores, X_expr, alpha=0.3, k=15,
                                  sim_threshold=0.1):
    n = len(base_scores)
    if (np.abs(X_expr).sum(axis=1) > 0).sum() < k + 1:
        print("  [LabelProp] Expression too sparse — skipping")
        return base_scores.copy()
    expr_norm = normalize(X_expr.astype(np.float32), norm='l2')
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                             algorithm='brute', n_jobs=-1).fit(expr_norm)
    distances, indices = nbrs.kneighbors(expr_norm)
    sims = np.maximum(0.0, 1.0 - distances[:, 1:].astype(np.float32))
    sims[sims < sim_threshold] = 0.0
    prop_scores = np.zeros(n, dtype=np.float32)
    weight_sum  = np.zeros(n, dtype=np.float32)
    for rank in range(k):
        j = indices[:, rank + 1]
        s = sims[:, rank]
        prop_scores += s * base_scores[j]
        weight_sum  += s
    valid = weight_sum > 0
    prop_scores[valid]  /= weight_sum[valid]
    prop_scores[~valid] = base_scores[~valid]
    refined = (1.0 - alpha) * base_scores + alpha * prop_scores
    changed = np.abs(refined - base_scores)
    print("  [LabelProp] alpha={:.2f} k={} | delta: mean={:.4f} max={:.4f}".format(
        alpha, k, changed.mean(), changed.max()))
    return refined

# --------------------------------------------------------------------------
# [9] save 유틸
# --------------------------------------------------------------------------
def save_phase_results_hybrid(phase_name, model_base, X_esm2, X_mask, X_seq, X_dm, X_dd,
                               y_true, gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds  = model_base.predict([X_esm2, X_mask, X_seq, X_dm, X_dd], batch_size=256, verbose=1)
    np.save(os.path.join(save_dir, '{}_{}_embeddings.npy'.format(ver_tag, phase_name)),
            preds[0])
    np.save(os.path.join(save_dir, '{}_{}_labels.npy'.format(ver_tag, phase_name)),
            y_true)
    scores = preds[1]
    s = np.array([scores[i][0] for i in range(len(y_true))])
    print("  mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
        s.mean(), s.std(), (s > 0.5).sum(), (s > 0.3).sum()))
    with open(os.path.join(save_dir, '{}_{}_scores.txt'.format(ver_tag, phase_name)),
              'w') as fw:
        for i in range(len(y_true)):
            fw.write(gene_ids[i] + '\t' + iso_ids[i] + '\t' + str(scores[i][0]) + '\n')
    return s

# --------------------------------------------------------------------------
# [10] 데이터 로딩
# --------------------------------------------------------------------------
print('>>> Preparing Data (v7b) for ' + selected_go)

ESM2_DATA_DIR = '../data'

def _load_esm2(name):
    path = os.path.join(ESM2_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError("[ESM] 필수 파일 없음: {}".format(path))
    arr = np.load(path).astype(np.float32)
    print("  [ESM] Loaded {} {}".format(name, arr.shape))
    return arr

X_train_esm2            = _load_esm2('esm2_train_human.npy')
X_train_esm2_mask       = _load_esm2('esm2_train_human_mask.npy')
X_train_other_esm2      = _load_esm2('esm2_train_swissprot.npy')
X_train_other_esm2_mask = _load_esm2('esm2_train_swissprot_mask.npy')
X_test_esm2             = _load_esm2('esm2_embeddings_t30_150M.npy')
X_test_esm2_mask        = _load_esm2('esm2_mask.npy')

train_human_cov = float(X_train_esm2_mask.sum()) / len(X_train_esm2_mask) * 100
test_cov        = float(X_test_esm2_mask.sum()) / len(X_test_esm2_mask) * 100
print("[ESM] Coverage — Human train: {:.1f}% | Test: {:.1f}%".format(
    train_human_cov, test_cov))

SEQ_PATH_TRAIN  = '../data/raw_data/data/sequences/human_sequence_train.npy'
SEQ_PATH_OTHER  = '../data/raw_data/data/sequences/swissprot_sequence_train.npy'
SEQ_PATH_TEST   = 'my_sequence_matrix_fixed.npy'

X_train_seq       = np.load(SEQ_PATH_TRAIN)[:, -SEQ_LEN:]
X_train_other_seq = np.load(SEQ_PATH_OTHER)[:, -SEQ_LEN:]
X_test_seq        = np.load(SEQ_PATH_TEST)[:, -SEQ_LEN:]
print("  [Seq] train={} other={} test={}".format(
    X_train_seq.shape, X_train_other_seq.shape, X_test_seq.shape))

X_train_dm       = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_test_dm        = np.load('../results/domain/domain_matrix.npy')

DD_TRAIN_PATH  = '../results_isoform/features/train_domain_delta_sign.npy'
DD_SW_PATH     = '../results_isoform/features/swissprot_domain_delta_sign.npy'
DD_TEST_PATH   = '../results_isoform/features/domain_delta.npy'
X_train_dd       = np.load(DD_TRAIN_PATH).astype(np.float32)
X_train_other_dd = np.load(DD_SW_PATH).astype(np.float32)
X_test_dd        = np.sign(np.load(DD_TEST_PATH)).astype(np.float32)
print("  [v6d] domain_delta sign: train={} sw={} test={}".format(
    X_train_dd.shape, X_train_other_dd.shape, X_test_dd.shape))

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

EXPR_CACHE  = 'expr_matrix_fixed.npy'
CPM_PATH    = '../data/bambu_data/CPM_transcript.txt'
test_iso_arr = np.load('my_isoform_list_fixed.npy', allow_pickle=True)

if os.path.exists(EXPR_CACHE):
    X_test_expr = np.load(EXPR_CACHE).astype(np.float32)
    print("[Expr] Loaded cache: {}".format(X_test_expr.shape))
else:
    X_test_expr = load_expression_matrix(CPM_PATH, test_iso_arr)
    np.save(EXPR_CACHE, X_test_expr)

assert X_test_expr.shape == (len(test_iso_arr), EXPR_DIM)

# --------------------------------------------------------------------------
# [11] 레이블 생성 및 업샘플
# --------------------------------------------------------------------------
print('>>> Generating Labels for ' + selected_go)

dm_dim         = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm),
                      np.max(X_train_other_dm)]) + 1

y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, \
    _dummy, X_train_dm_comb = generate_label(
        X_train_esm2, X_train_dm,
        X_train_other_esm2, X_train_other_dm,
        X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

_, _, _, _, _, _, X_train_esm2_comb, X_train_mask_comb = generate_label(
    X_train_esm2, X_train_esm2_mask,
    X_train_other_esm2, X_train_other_esm2_mask,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

_, _, _, _, _, _, _dummy2, X_train_seq_comb = generate_label(
    X_train_esm2, X_train_seq,
    X_train_other_esm2, X_train_other_seq,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

_, _, _, _, _, _, _dummy3, X_train_dd_comb = generate_label(
    X_train_esm2, X_train_dd,
    X_train_other_esm2, X_train_other_dd,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

n_pos_train = int((y_train == 1).sum())
print("  y_train pos={} neg={} ratio={:.3f}%".format(
    n_pos_train, int((y_train == 0).sum()), n_pos_train / len(y_train) * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

MASK_START = ESM2_DIM
SEQ_START  = ESM2_DIM + 1
DM_START   = ESM2_DIM + 1 + SEQ_LEN

X_train_combined = np.hstack([
    X_train_esm2_comb,
    X_train_mask_comb,
    X_train_seq_comb.astype(np.float32),
    X_train_dm_comb.astype(np.float32),
    X_train_dd_comb.astype(np.float32),
])

DD_START = DM_START + X_train_dm_comb.shape[1]

np.random.seed(42)
unused_flag = np.zeros(y_train.shape[0])
X_combined_upsmp, _dummy_dm, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count,
    X_train_combined,
    np.zeros((len(y_train), 1)),
    unused_flag)

X_train_esm2_upsmp = X_combined_upsmp[:, :ESM2_DIM].astype(np.float32)
X_train_mask_upsmp = X_combined_upsmp[:, MASK_START:SEQ_START].astype(np.float32)
X_train_seq_upsmp  = X_combined_upsmp[:, SEQ_START:DM_START].astype(np.int32)
X_train_dm_upsmp   = X_combined_upsmp[:, DM_START:DD_START].astype(np.int32)
X_train_dd_upsmp   = X_combined_upsmp[:, DD_START:].astype(np.float32)

print("  Upsampled: esm2={} mask={} seq={} dm={} dd={} y={}".format(
    X_train_esm2_upsmp.shape, X_train_mask_upsmp.shape,
    X_train_seq_upsmp.shape, X_train_dm_upsmp.shape,
    X_train_dd_upsmp.shape, y_train_upsmp.shape))

BATCH_SIZE_P1 = 256
N_BATCHES_P1  = int(np.clip(
    np.ceil(n_pos_train * TARGET_COVERAGE / BATCH_SIZE_P1), a_min=10, a_max=80))
print("[I4] n_batches={} (n_pos={}, coverage={:.1f}x)".format(
    N_BATCHES_P1, n_pos_train,
    N_BATCHES_P1 * BATCH_SIZE_P1 / n_pos_train))

# --------------------------------------------------------------------------
# [12] 모델 구조 (v6d 동일)
# --------------------------------------------------------------------------
esm2_input   = Input(shape=(ESM2_DIM,), name='esm2_input')
esm2_mask_in = Input(shape=(1,),         name='esm2_mask')
seq_input    = Input(shape=(SEQ_LEN,),  dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim,),   dtype='int32', name='domain_input')
dd_input     = Input(shape=(DD_DIM,),   name='dd_input')

x_esm = Dense(256, kernel_regularizer=regularizers.l2(1e-5), name='esm2_d1')(esm2_input)
x_esm = Activation('relu')(x_esm)
x_esm = Dropout(0.2)(x_esm)
x_esm = Dense(128, kernel_regularizer=regularizers.l2(1e-5), name='esm2_d2')(x_esm)
x_esm = Activation('relu')(x_esm)
esm2_feat  = Dense(64, activation='relu',
                   kernel_regularizer=regularizers.l2(1e-5),
                   name='esm2_feat')(x_esm)
esm2_gated = Lambda(lambda x: x[0] * x[1], name='esm2_gated')([esm2_feat, esm2_mask_in])

x_seq = Embedding(8001, 32, mask_zero=False, name='cnn_emb')(seq_input)
x_seq = Conv1D(64, kernel_size=7, padding='same', activation='relu', name='cnn_conv1')(x_seq)
x_seq = Conv1D(32, kernel_size=5, padding='same', activation='relu', name='cnn_conv2')(x_seq)
x_seq = GlobalMaxPooling1D(name='cnn_pool')(x_seq)
cnn_feat = Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-5),
                 name='cnn_feat')(x_seq)

x_dm = Embedding(input_dim=domain_emb_dim, output_dim=32,
                  input_length=dm_dim, mask_zero=True,
                  name='dm_emb')(domain_input)
domain_feat = LSTM(16, name='domain_feat')(x_dm)

x_dd = Dense(64, activation='relu',
             kernel_regularizer=regularizers.l2(1e-5),
             name='dd_dense1')(dd_input)
x_dd = Dropout(0.2)(x_dd)
dd_feat = Dense(16, activation='relu',
                kernel_regularizer=regularizers.l2(1e-5),
                name='dd_dense2')(x_dd)

concat = concatenate([esm2_gated, cnn_feat, domain_feat, dd_feat], name='feature_concat')
feature_model = Model([esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
                       concat, name='feature_model')

x = Dense(64, kernel_regularizer=regularizers.l2(1e-5), name='head_dense64')(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer  = Lambda(lambda a: K.l2_normalize(a, axis=1),
                           name='embedding_out')(x)
prediction_layer = Dense(1, activation='sigmoid',
                          kernel_regularizer=regularizers.l2(1e-5),
                          name='prediction_out')(embedding_layer)

base_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=[embedding_layer, prediction_layer])
classification_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=prediction_layer)
# [v7b-fix] Phase 1 Type-B Proto-kN용: 64-dim L2-normalized embedding 출력 모델
# feature_model(128-dim) 대신 embedding_model(64-dim)을 phase1_proto_epoch_hybrid에 전달
# 근거: PrototypeContrastiveLoss는 emb_dim=64로 초기화 → 128-dim 입력 시 MatMul shape 불일치
embedding_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=embedding_layer,
    name='embedding_model')

adam_p1   = optimizers.Adam(lr=0.0005)
adam_main = optimizers.Adam(lr=0.001)
adam_p2   = optimizers.Adam(lr=0.0003)

K_testing_size = len(X_test_esm2)

print("\n[Model v7b] ESM-2(640→64) + CNN(1500→32) + Domain(LSTM 16) + DomainDelta(251→16) → concat[128]")
print("[Model v7b] Head: Dense(64) → L2_norm → emb[64] → Dense(1,sigmoid)")
print("[Model v7b] Phase 1: Type-A=Triplet [v7b-1] | Type-B=Proto-kN [v7b-1]")
print("[Model v7b] Phase 1→2: Prototype Warm Init [v7b-2]")

emb_by_phase   = {}
score_by_phase = {}

# ==========================================================================
# PHASE 0: Untrained Baseline + Type 분류
# ==========================================================================
print("\n" + "=" * 58)
print(">>> PHASE 0: Untrained Baseline + GO term Type Classification")
print("=" * 58)

s0 = save_phase_results_hybrid(
    "phase0_initial_untrained", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph0(untrained)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase0_initial_untrained_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph0(untrained)'] = s0
ph0_metrics = analyze_embedding_quality(emb_by_phase['Ph0(untrained)'], y_test, 'Phase 0')

ph0_sep_ratio = ph0_metrics.get('sep_ratio', float('nan'))
IS_TYPE_B = (ph0_sep_ratio < SEP_RATIO_THRESHOLD) if not np.isnan(ph0_sep_ratio) else False
go_type_str = 'Type-B (heterogeneous, sep={:.4f})'.format(ph0_sep_ratio) if IS_TYPE_B \
              else 'Type-A (coherent, sep={:.4f})'.format(ph0_sep_ratio)
print("\n[v6e-1] GO term classification: {}".format(go_type_str))

if IS_TYPE_B:
    print("  -> Phase 1: Proto-kN (Gap Statistic, k_max={}) [v7b-1]".format(PROTO_K_MAX))
    print("  -> Phase 2: lr=0.0002, clipnorm=0.5, patience=10, max_epochs=25 [v6f-1,2]")
    print("  -> Bio: positive set contains evolutionarily unrelated proteins")
else:
    print("  -> Phase 1: Triplet (margin={:.2f}, semi-hard) [v7b-1]".format(MARGIN_P1))
    print("  -> Phase 2: lr=0.0003, patience=7, max_epochs=25 (standard)")
    print("  -> Bio: positive set shares evolutionary origin — ESM-2 coherent")

# [v7b-1] Type-B: Gap Statistic으로 k 결정 (Phase 0 embedding 사용)
proto_loss_fn = None
PROTO_K = 1  # default

if IS_TYPE_B:
    print("\n[v7b-1] Type-B: Determining k via Gap Statistic...")
    ph0_pos_emb = emb_by_phase['Ph0(untrained)'][y_test == 1]
    if len(ph0_pos_emb) >= PROTO_K_MAX * 3:
        PROTO_K = determine_k_gap_statistic(
            ph0_pos_emb, k_max=PROTO_K_MAX, n_refs=10, random_state=42)
    else:
        PROTO_K = 1
        print("  [Gap] Insufficient positives (N={}) — k=1 forced".format(len(ph0_pos_emb)))

    print("[v7b-1] Selected k={} for Type-B Proto-kN".format(PROTO_K))

    proto_loss_fn = PrototypeContrastiveLoss(
        n_prototypes=PROTO_K,
        emb_dim=64,
        temperature=PROTO_TEMPERATURE,
        ema_decay=PROTO_EMA_DECAY,
        lambda_div=PROTO_LAMBDA_DIV)

    # Phase 0 embedding으로 prototype 초기화
    proto_loss_fn.initialize_from_embeddings(ph0_pos_emb)

# ==========================================================================
# PHASE 1: Type-Adaptive Contrastive Loss [v7b-1]
# ==========================================================================
print("\n" + "=" * 58)
if IS_TYPE_B:
    print(">>> PHASE 1: Proto-kN [v7b-1] — ESM-2+Domain (CNN frozen)")
    print("    k={} | τ={} | λ_div={} | EMA α={}".format(
        PROTO_K, PROTO_TEMPERATURE, PROTO_LAMBDA_DIV, PROTO_EMA_DECAY))
else:
    print(">>> PHASE 1: Triplet [v7b-1] — ESM-2+Domain (CNN frozen)")
    print("    margin={} | n_batches={} | semi-hard".format(MARGIN_P1, N_BATCHES_P1))
print("=" * 58)

set_cnn_trainable(feature_model, False)
print("  [Phase1] CNN frozen | ESM-2 + Domain + DeltaBranch trainable")

PHASE1_EPOCHS = 15

if IS_TYPE_B:
    # ── Type-B: Proto-kN ──────────────────────────────────────────────
    LOSS_THRESH  = 0.01
    STREAK_LIMIT = 4
    low_loss_streak = 0
    best_proto_loss = float('inf')

    for epoch in range(PHASE1_EPOCHS):
        print('Phase 1 - Epoch: {}/{} [Proto-k{}]'.format(epoch + 1, PHASE1_EPOCHS, PROTO_K))

        avg_proto, avg_div = phase1_proto_epoch_hybrid(
            feature_model=embedding_model,  # [v7b-fix] 64-dim embedding 출력
            X_esm2=X_train_esm2_upsmp,
            X_mask=X_train_mask_upsmp,
            X_seq=X_train_seq_upsmp,
            X_dm=X_train_dm_upsmp,
            X_dd=X_train_dd_upsmp,
            y=y_train_upsmp,
            optimizer=adam_p1,
            proto_loss_fn=proto_loss_fn,
            batch_size=BATCH_SIZE_P1,
            n_batches=N_BATCHES_P1,
            min_pos_in_batch=8,
            do_ema_update=True)

        print("  -> Proto Loss: {:.4f} | Div Loss: {:.4f}".format(avg_proto, avg_div))

        # 진단: prototype 할당 현황 (5 epoch마다)
        if (epoch + 1) % 5 == 0:
            # [v7b-fix] base_model output[0] = 64-dim embedding_layer
            preds_tmp = base_model.predict(
                [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
                batch_size=256, verbose=0)
            test_emb_tmp = preds_tmp[0]  # 64-dim L2-normalized
            proto_loss_fn.prototype_stats(test_emb_tmp, y_test)
            sep_tmp = test_emb_tmp[y_test == 1].mean(axis=0)

        # Early stop
        if avg_proto < LOSS_THRESH:
            low_loss_streak += 1
        else:
            low_loss_streak = 0
        if low_loss_streak >= STREAK_LIMIT:
            print("  [Early Stop] Proto loss < {:.3f} for {} epochs.".format(
                LOSS_THRESH, STREAK_LIMIT))
            break

    print("\n[Phase 1 Final] k={} Proto-kN complete".format(PROTO_K))

else:
    # ── Type-A: Triplet ───────────────────────────────────────────────
    ACTIVE_RATE_THRESH = 2.0   # active ratio < 2% for streak → early stop
    STREAK_LIMIT       = 3
    low_active_streak  = 0
    best_margin_sat    = 0.0
    best_centroid_dist = 0.0

    for epoch in range(PHASE1_EPOCHS):
        print('Phase 1 - Epoch: {}/{} [Triplet]'.format(epoch + 1, PHASE1_EPOCHS))
        warmup = (epoch == 0)

        avg_loss, active_rate = phase1_triplet_epoch_hybrid(
            feature_model=embedding_model,  # [v7b-fix2] 64-dim 통일
            X_esm2=X_train_esm2_upsmp,
            X_mask=X_train_mask_upsmp,
            X_seq=X_train_seq_upsmp,
            X_dm=X_train_dm_upsmp,
            X_dd=X_train_dd_upsmp,
            y=y_train_upsmp,
            optimizer=adam_p1,
            margin=MARGIN_P1,
            batch_size=BATCH_SIZE_P1,
            n_batches=N_BATCHES_P1,
            warmup=warmup)

        print("  -> Triplet Loss: {:.4f} | Active Rate: {:.1f}%".format(avg_loss, active_rate))

        if active_rate < ACTIVE_RATE_THRESH:
            low_active_streak += 1
        else:
            low_active_streak = 0
        if low_active_streak >= STREAK_LIMIT and epoch >= 5:
            print("  [Early Stop] Active rate < {:.1f}% for {} epochs.".format(
                ACTIVE_RATE_THRESH, STREAK_LIMIT))
            break

        if (epoch + 1) % 5 == 0:
            sat, cdist = compute_margin_stats_hybrid(
                embedding_model, X_test_esm2, X_test_esm2_mask,
                X_test_seq, X_test_dm, X_test_dd, y_test)  # [v7b-fix2]
            best_margin_sat    = max(best_margin_sat, sat)
            best_centroid_dist = max(best_centroid_dist, cdist)

    print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
        best_margin_sat, best_centroid_dist))

# Phase 1 embedding 저장 및 분석
s1 = save_phase_results_hybrid(
    "phase1_contrastive", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(contrastive)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_contrastive_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(contrastive)'] = s1
ph1_metrics = analyze_embedding_quality(emb_by_phase['Ph1(contrastive)'], y_test, 'Phase 1')

# ==========================================================================
# [v7b-2] PHASE 1→2 INTERFACE: Prototype Warm Init
# ==========================================================================
print("\n" + "=" * 58)
print(">>> PHASE 1→2: Prototype Warm Init [v7b-2]")
print("=" * 58)

ph1_emb = emb_by_phase['Ph1(contrastive)']
warm_ok = prototype_warm_init(
    base_model=base_model,
    embeddings_np=ph1_emb,
    y_np=y_test,
    proto_fn=proto_loss_fn if IS_TYPE_B else None,
    scale=WARM_INIT_SCALE)

if warm_ok:
    # warm init 적용 후 epoch=1 시작점 확인용 초기 예측
    preds_warm = base_model.predict(
        [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
        batch_size=256, verbose=0)
    warm_scores = np.array([preds_warm[1][i][0] for i in range(len(y_test))])
    if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
        warm_auprc = average_precision_score(y_test, warm_scores)
        warm_auroc = roc_auc_score(y_test, warm_scores)
        print("  [WarmInit] Pre-Phase2 AUROC={:.4f} AUPRC={:.4f}".format(
            warm_auroc, warm_auprc))
else:
    print("  [WarmInit] Falling back to random initialization")

# [v6g-1] Phase 1.5 건너뜀 (warm init이 대체)
print("\n[v6g-1/v7b] Phase 1.5 SKIPPED — Prototype Warm Init [v7b-2] applied instead")

n_train = len(X_train_esm2_upsmp)

# ==========================================================================
# PHASE 2: CNN Fine-tuning (ESM-2 frozen, Focal Loss)
# ==========================================================================
print("\n" + "=" * 58)
print(">>> PHASE 2: CNN + DeltaBranch Fine-tuning")
print("    [v6e] ESM-2 FROZEN | CNN+Domain+Delta TRAIN | Focal only")
print("    [v7b] Warm Init applied [v7b-2]")
print("=" * 58)

for layer in feature_model.layers:
    layer.trainable = True
set_esm2_trainable(feature_model, False)
print("  [Phase2] ESM-2 frozen | CNN + Domain + DeltaBranch trainable")

if IS_TYPE_B:
    adam_p2       = optimizers.Adam(lr=0.0002, clipnorm=0.5)
    NO_IMPROVE_LIMIT = 10
    print("  [v6f-1] Type-B: lr=0.0002, clipnorm=0.5, patience={}".format(NO_IMPROVE_LIMIT))
else:
    adam_p2       = optimizers.Adam(lr=0.0003)
    NO_IMPROVE_LIMIT = 7
    print("  [v6g] Type-A: lr=0.0003, patience={} (standard)".format(NO_IMPROVE_LIMIT))

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.10),
    optimizer=adam_p2, metrics=['accuracy'])

PHASE2_MAX_EPOCHS  = 25
BATCH_SIZE_P2      = 512
best_phase2_auprc  = 0.0
best_phase2_weights = None
no_improve_count   = 0

for epoch in range(PHASE2_MAX_EPOCHS):
    print('Phase 2 - Epoch: {}/{}'.format(epoch + 1, PHASE2_MAX_EPOCHS))
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    ep_losses, ep_accs = [], []

    for start in range(0, n_train, BATCH_SIZE_P2):
        idx   = indices[start:start + BATCH_SIZE_P2]
        mixed = np.hstack((np.where(y_train_upsmp[idx] == 1)[0],
                           np.where(y_train_upsmp[idx] == 0)[0]))
        if len(mixed) == 0:
            continue
        np.random.shuffle(mixed)
        hist = classification_model.fit(
            [X_train_esm2_upsmp[idx[mixed]],
             X_train_mask_upsmp[idx[mixed]],
             X_train_seq_upsmp[idx[mixed]],
             X_train_dm_upsmp[idx[mixed]],
             X_train_dd_upsmp[idx[mixed]]],
            y_train_upsmp[idx[mixed]],
            batch_size=256, epochs=1, verbose=0)
        ep_losses.append(hist.history['loss'][0])
        acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
        ep_accs.append(hist.history[acc_key][0])

    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(ep_losses) if ep_losses else 0,
        np.mean(ep_accs)   if ep_accs   else 0))

    preds = base_model.predict(
        [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
        batch_size=256, verbose=0)
    test_scores = np.array([preds[1][i][0] for i in range(len(y_test))])
    if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
        current_auroc = roc_auc_score(y_test, test_scores)
        current_auprc = average_precision_score(y_test, test_scores)
        print("  [AUPRC] epoch={} AUROC={:.4f} AUPRC={:.4f} (best={:.4f})".format(
            epoch + 1, current_auroc, current_auprc, best_phase2_auprc))
        if current_auprc > best_phase2_auprc:
            best_phase2_auprc   = current_auprc
            best_phase2_weights = base_model.get_weights()
            no_improve_count    = 0
        else:
            no_improve_count += 1
            if no_improve_count >= NO_IMPROVE_LIMIT:
                print("  [Early Stop] AUPRC not improving ({} epochs)".format(NO_IMPROVE_LIMIT))
                break

if best_phase2_weights is not None:
    base_model.set_weights(best_phase2_weights)
    print("\n[Phase 2] Restored best (AUPRC={:.4f})".format(best_phase2_auprc))

s2 = save_phase_results_hybrid(
    "phase2_cnn_focal", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph2(cnn_focal)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase2_cnn_focal_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph2(cnn_focal)'] = s2
analyze_embedding_quality(emb_by_phase['Ph2(cnn_focal)'], y_test, 'Phase 2')

# ==========================================================================
# PHASE 3: LP 비활성 [v6e-3]
# ==========================================================================
print("\n" + "=" * 58)
print(">>> PHASE 3: [v6e-3] Label Propagation REMOVED (alpha=0.0)")
print("=" * 58)

preds_base  = base_model.predict(
    [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
    batch_size=256, verbose=0)
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])

final_scores = base_scores.copy()
final_auroc  = roc_auc_score(y_test, final_scores) \
               if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else float('nan')
final_auprc  = average_precision_score(y_test, final_scores) \
               if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else float('nan')
print("\n  [v6e-3] Final (no LP): AUROC={:.4f} AUPRC={:.4f}".format(
    final_auroc, final_auprc))

if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
    try:
        lp_02 = expression_label_propagation(base_scores, X_test_expr, alpha=0.2)
        lp_auprc = average_precision_score(y_test, lp_02)
        delta = lp_auprc - final_auprc
        print("  [Diag] LP alpha=0.2 AUPRC={:.4f} (delta={:+.4f}, NOT applied)".format(
            lp_auprc, delta))
    except Exception:
        pass

score_by_phase['Final(noLP)'] = final_scores

# ==========================================================================
# 임베딩 품질 요약
# ==========================================================================
print("\n" + "=" * 58)
print(">>> Embedding Quality Summary")
print("=" * 58)
print("{:25} {:>10} {:>10} {:>10} {:>12}".format(
    "Phase", "Silhouette", "Sep.Ratio", "LinAUROC", "PredAUROC"))
print("-" * 69)
for phase_name, emb in emb_by_phase.items():
    m = analyze_embedding_quality(emb, y_test, phase_name)
    try:
        pred_auroc = roc_auc_score(y_test, score_by_phase[phase_name])
    except Exception:
        pred_auroc = float('nan')
    print("{:25} {:10.4f} {:10.4f} {:10.4f} {:12.4f}".format(
        phase_name, m['silhouette'], m['sep_ratio'],
        m['linear_auroc'], pred_auroc))

# ==========================================================================
# 최종 저장
# ==========================================================================
print("\nSaving Final Results...")
output_file = os.path.join(
    SAVE_DIR, "{}_{}_Final_scores.txt".format(VER_TAG, safe_go))
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                 str(final_scores[i]) + '\n')

base_model.save_weights(
    os.path.join(SAVE_DIR, "{}_{}_BaseModel_weights.h5".format(VER_TAG, safe_go)))
np.save(os.path.join(SAVE_DIR, "{}_{}_Final_labels.npy".format(VER_TAG, safe_go)),
        y_test)

# v7b 전용: prototype 저장 (Type-B)
if IS_TYPE_B and proto_loss_fn is not None and proto_loss_fn.prototypes is not None:
    np.save(os.path.join(SAVE_DIR, "{}_{}_prototypes.npy".format(VER_TAG, safe_go)),
            proto_loss_fn.prototypes.numpy())
    print("  [Proto] Saved k={} prototypes".format(PROTO_K))

print("\n[Final] GO={} | Type={} | Phase1={} | k={} | WarmInit={}".format(
    selected_go, go_type_str,
    'Proto-kN' if IS_TYPE_B else 'Triplet',
    PROTO_K if IS_TYPE_B else 1,
    'OK' if warm_ok else 'FAIL'))
print("[Final] AUROC={:.4f} AUPRC={:.4f}".format(final_auroc, final_auprc))
print("[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
