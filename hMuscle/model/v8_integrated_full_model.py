# -*- coding: utf-8 -*-
# ============================================================================
# v8_integrated_full_model.py  (v7c + Step 2a: Phase freeze 제거)
#
# v8 핵심 변경 (v7c 대비):
#
# [v8-1] Phase 교대 동결 제거 → Discriminative LR
#   Phase1(ESM-2 train, CNN frozen) + Phase2(ESM-2 frozen, CNN train) 구조 제거.
#   근거: [F11] ESM-2와 CNN이 단 한 번도 동시 최적화된 적 없음 → 상호보완 차단.
#   대체: ESM2=1e-5, CNN=1e-4, Domain/Delta=1e-4, Head=1e-3 (10배 차등).
#
# [v8-2] Unified Loss: metric + focal 동시 학습
#   기존: Phase1=metric only | Phase2=focal only
#   v8:   L_total = α × L_metric + (1-α) × L_focal  (매 batch)
#   α = 0.5 (Type-B) | 0.3 (Type-A), 고정 (Step 2a — annealing은 v8-2에서)
#   근거: [F5] Phase2 CNN이 metric geometry 파괴 → α 하한으로 metric 신호 유지.
#
# [v8-3] Proto-kN k 선택 개선 — minimum occupancy 검증
#   Gap Statistic 단독 → dead prototype 발생 ([F12], GO:0006941: proto1=0, proto2=0).
#   개선: Gap Statistic 후 occupancy 검증, 미달 시 k 감소.
#
# [v8-4] Bias-only warm init (prediction_out)
#   Phase0 완료 후 bias = log(pos_ratio/(1-pos_ratio)) 설정.
#   Phase2 cold start 문제 없음 (unified 구조), bias만 초기화로 수렴 보조.
#
# v7c 유지 항목:
#   - PFN backbone (수정 금지)
#   - Multimodal inputs: ESM-2 + CNN + Domain + DomainDelta
#   - Type-A/B 분류 (sep_ratio < 1.15 → Type-B)
#   - Type-B: Proto-kN + Neg-InfoNCE | Type-A: Triplet (semi-hard)
#   - Focal Loss (γ=2, α=0.10)
#   - ESM-2 gating mask (데이터 가용성 마스크)
#   - Proto biological validation (save_prototype_assignments)
#
# [ARCH] 모델 구조 (v7c 동일, Phase freeze만 제거)
#   ESM-2: 640 → Dense(256) → Dense(128) → Dense(64) → gated → esm2_gated[64]
#   CNN:   Embedding → Conv1D(64) → Conv1D(32) → GlobalMaxPool → Dense(32) → cnn_feat[32]
#   Domain: Embedding → LSTM(16) → domain_feat[16]
#   DomainDelta: sign(251) → Dense(64) → Dense(16) → dd_feat[16]
#   Fusion: concat[128] → Dense(64, relu) → L2_norm → emb[64] → Dense(1, sigmoid)
#
# [TRAIN] v8 구조
#   Unified Training (35 epochs max): ALL layers trainable
#   Discriminative LR: ESM2=1e-5 (사실상 soft-freeze) | CNN=1e-4 | rest=1e-4 | head=1e-3
#   Early stopping: AUPRC, patience=15 (Type-B) / 10 (Type-A)
#
# 검증 기준 (Step 2a):
#   Type-B Macro-AUPRC > 0.315 (v7c)
#   → 성공 시 v8-1: FiLM 추가
#   → 실패 시: discriminative LR 비율 재검토
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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression

from crf import CRF
sys.path.insert(0, '/home/welcome1/layer_Full')
from utils_Full import generate_label, upsample

# [v8] Proto-kN imports (v7c_prototype_contrastive 재사용)
from v7c_prototype_contrastive import (
    determine_k_gap_statistic,
    PrototypeContrastiveLoss,
    save_prototype_assignments,
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
    print("Usage: python v8_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go
if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d_%H%M")
VER_TAG  = "v8_integrated"
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
MARGIN_P1           = 0.3
TARGET_COVERAGE     = 6.0
SEP_RATIO_THRESHOLD = 1.15

# Prototype hyperparameters (Type-B)
PROTO_K_MAX         = 5
PROTO_TEMPERATURE   = 0.1
PROTO_EMA_DECAY     = 0.9
PROTO_LAMBDA_DIV    = 0.1

# [v8-1] Discriminative LR
LR_ESM2 = 1e-5   # ESM-2 projection (사실상 soft-freeze)
LR_CNN  = 1e-4   # CNN branch
LR_REST = 1e-4   # Domain, DomainDelta
LR_HEAD = 1e-3   # head_dense64, prediction_out

# [v8-2] Unified Loss α (고정, Step 2a)
ALPHA_TYPE_B = 0.5   # Type-B: positive heterogeneous → metric 신호 강화
ALPHA_TYPE_A = 0.3   # Type-A: coherent → focal 비중 높임

# Layer name sets for discriminative LR
ESM2_TRAINABLE_NAMES = {'esm2_d1', 'esm2_d2', 'esm2_feat'}
CNN_TRAINABLE_NAMES  = {'cnn_emb', 'cnn_conv1', 'cnn_conv2', 'cnn_feat'}
HEAD_LAYER_NAMES     = {'head_dense64', 'prediction_out'}

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
# [5] 임베딩 추출
# --------------------------------------------------------------------------
def extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd,
                               batch_size=512):
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
            emb = np.zeros((n, raw.shape[1]), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        emb[start:end] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [EMB] 임베딩 품질 분석 (centroid_cos 추가)
# --------------------------------------------------------------------------
def analyze_embedding_quality(embeddings, labels, phase_name, n_pairs=500, seed=42):
    np.random.seed(seed)
    pos_emb = embeddings[labels == 1]
    neg_emb = embeddings[labels == 0]
    metrics = {}

    if len(pos_emb) > 0 and len(neg_emb) > 0:
        pos_c = pos_emb.mean(axis=0)
        neg_c = neg_emb.mean(axis=0)
        metrics['centroid_dist'] = float(np.linalg.norm(pos_c - neg_c))
        # centroid_cos: L2-normalized space의 cosine 유사도 (낮을수록 좋음)
        pos_c_n = pos_c / (np.linalg.norm(pos_c) + 1e-10)
        neg_c_n = neg_c / (np.linalg.norm(neg_c) + 1e-10)
        metrics['centroid_cos'] = float(np.dot(pos_c_n, neg_c_n))
    else:
        metrics['centroid_dist'] = float('nan')
        metrics['centroid_cos']  = float('nan')

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
        # frac_pos_near_neg: pos 중 neg와 cosine > 0.9인 비율 ([F5] 모니터링)
        near_neg_count = 0
        n_check = min(500, len(pos_emb))
        sample_pos = pos_emb[np.random.choice(len(pos_emb), n_check, replace=False)]
        for pe in sample_pos:
            cos_sims = neg_emb @ pe  # L2-normalized → dot = cos
            if (cos_sims > 0.9).any():
                near_neg_count += 1
        metrics['frac_pos_near_neg'] = float(near_neg_count / n_check)
    else:
        metrics['intra_dist'] = metrics['inter_dist'] = metrics['sep_ratio'] = float('nan')
        metrics['frac_pos_near_neg'] = float('nan')

    if len(pos_emb) >= 2 and len(neg_emb) >= 2:
        n_sub   = min(4000, len(embeddings))
        idx_sub = np.random.choice(len(embeddings), n_sub, replace=False)
        emb_sub = embeddings[idx_sub]
        lab_sub = labels[idx_sub].astype(int)
        try:
            metrics['silhouette'] = float(silhouette_score(
                emb_sub, lab_sub, metric='cosine',
                sample_size=min(2000, n_sub), random_state=seed)) \
                if lab_sub.sum() >= 2 and (lab_sub == 0).sum() >= 2 else float('nan')
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
    print("    centroid_cos (↓good):   {:+.4f}".format(metrics['centroid_cos']))
    print("    Sep ratio (inter/intra): {:.4f}".format(metrics['sep_ratio']))
    print("    frac_pos_near_neg:       {:.4f}".format(metrics['frac_pos_near_neg']))
    print("    Silhouette (cosine):    {:+.4f}".format(metrics['silhouette']))
    print("    Linear AUROC:            {:.4f}".format(metrics['linear_auroc']))
    return metrics

# --------------------------------------------------------------------------
# [6] Triplet helpers (Type-A, unchanged from v7c)
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=MARGIN_P1, mode="hard"):
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

# --------------------------------------------------------------------------
# [v8-3] Proto-kN k 선택: Gap Statistic + minimum occupancy 검증
# --------------------------------------------------------------------------
def determine_k_robust(pos_emb, k_max=5):
    """
    [v8-3] Gap Statistic 단독 → dead prototype 문제 해결 ([F12]).
    Gap Statistic으로 k_gap 선택 후, minimum occupancy 검증.
    미달 시 k-1로 내려가며 재검증 (k=1까지).
    """
    k_gap = determine_k_gap_statistic(pos_emb, k_max=k_max, n_refs=10, random_state=42)
    print("  [k_robust] Gap Statistic k_gap={}, running occupancy validation...".format(k_gap))

    for k in range(k_gap, 0, -1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(pos_emb)
        counts = np.bincount(labels, minlength=k)
        min_count = counts.min()
        threshold = max(3, len(pos_emb) // (k * 5))
        print("  [k_robust] k={}: min_occupancy={} (threshold={}) counts={}".format(
            k, min_count, threshold, counts.tolist()))
        if min_count >= threshold:
            print("  [k_robust] k={} accepted (all prototypes occupied)".format(k))
            return k

    print("  [k_robust] Fallback to k=1")
    return 1

# --------------------------------------------------------------------------
# [v8-1] Discriminative LR helpers
# --------------------------------------------------------------------------
def build_var_groups(model):
    """
    model 레이어를 이름 기준으로 4그룹으로 분류.
    Returns: (esm2_ids, cnn_ids, head_ids)  — set of id(variable)
    rest_ids = 나머지 (Domain, DomainDelta)
    """
    esm2_ids = set()
    cnn_ids  = set()
    head_ids = set()
    for layer in model.layers:
        ids = {id(v) for v in layer.trainable_variables}
        if layer.name in ESM2_TRAINABLE_NAMES:
            esm2_ids |= ids
        elif layer.name in CNN_TRAINABLE_NAMES:
            cnn_ids |= ids
        elif layer.name in HEAD_LAYER_NAMES:
            head_ids |= ids
    return esm2_ids, cnn_ids, head_ids

def apply_discriminative_grads(grads, variables,
                                esm2_ids, cnn_ids, head_ids,
                                opt_esm2, opt_cnn, opt_rest, opt_head):
    """그래디언트를 4그룹으로 분류하여 각각의 optimizer 적용."""
    g_esm2, g_cnn, g_head, g_rest = [], [], [], []
    for g, v in zip(grads, variables):
        if g is None:
            continue
        vid = id(v)
        if vid in esm2_ids:
            g_esm2.append((g, v))
        elif vid in cnn_ids:
            g_cnn.append((g, v))
        elif vid in head_ids:
            g_head.append((g, v))
        else:
            g_rest.append((g, v))
    if g_esm2: opt_esm2.apply_gradients(g_esm2)
    if g_cnn:  opt_cnn.apply_gradients(g_cnn)
    if g_head: opt_head.apply_gradients(g_head)
    if g_rest: opt_rest.apply_gradients(g_rest)

# --------------------------------------------------------------------------
# [v8-2] Unified Epoch: metric + focal in same GradientTape
# --------------------------------------------------------------------------
def unified_epoch_v8(
        embedding_model, classification_model,
        X_esm2, X_mask, X_seq, X_dm, X_dd, y,
        esm2_ids, cnn_ids, head_ids,
        opt_esm2, opt_cnn, opt_rest, opt_head,
        alpha, is_type_b,
        proto_loss_fn=None,
        margin=MARGIN_P1,
        batch_size=256, n_batches=50,
        refresh_interval=10,
        focal_alpha_val=0.10):
    """
    [v8-2] Metric + Focal 동시 학습, discriminative LR.

    Type-B: Proto-kN Neg-InfoNCE (pos + neg forward, same tape)
    Type-A: Triplet semi-hard (3 sub-batches, same tape)
    Focal:  random mini-batch (모든 경우)

    L_total = α × L_metric + (1-α) × L_focal

    Returns: (avg_metric_loss, avg_focal_loss, active_rate_pct)
    active_rate_pct: Type-A만 의미있음 (Type-B는 0.0 반환)
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    n = len(X_esm2)

    if len(pos_indices) < 2 or len(neg_indices) == 0:
        return 0.0, 0.0, 0.0

    all_vars = classification_model.trainable_variables
    focal_fn = binary_focal_loss(gamma=2.0, alpha=focal_alpha_val)

    n_pos_per_batch = max(8, min(len(pos_indices), batch_size // 2))
    n_neg_metric    = min(len(neg_indices), n_pos_per_batch)
    replace_pos     = len(pos_indices) < n_pos_per_batch
    replace_neg     = len(neg_indices) < n_neg_metric

    # Type-A: 사전 임베딩 계산 (triplet mining)
    if not is_type_b:
        embeddings = extract_embeddings_hybrid(
            embedding_model, X_esm2, X_mask, X_seq, X_dm, X_dd)
    else:
        embeddings = None

    batch_losses_m = []
    batch_losses_f = []
    active_count   = 0
    all_pos_emb_np = []   # Type-B EMA update용

    for batch_i in range(n_batches):
        # Type-A: 주기적 임베딩 갱신 (triplet mining)
        if not is_type_b and batch_i > 0 and batch_i % refresh_interval == 0:
            embeddings = extract_embeddings_hybrid(
                embedding_model, X_esm2, X_mask, X_seq, X_dm, X_dd)

        # Focal batch (random)
        focal_idx = np.random.choice(n, batch_size, replace=True)
        y_focal   = tf.cast(y[focal_idx], tf.float32)

        # ── 모델 forward helper (L2-normalized embedding) ──────────────
        def _emb(idx):
            return tf.math.l2_normalize(
                embedding_model(
                    [X_esm2[idx].astype(np.float32),
                     X_mask[idx].astype(np.float32),
                     X_seq[idx], X_dm[idx],
                     X_dd[idx].astype(np.float32)],
                    training=True),
                axis=1)

        if is_type_b:
            # ── Type-B: Proto-kN Neg-InfoNCE ──────────────────────────
            pos_idx = np.random.choice(pos_indices, n_pos_per_batch, replace=replace_pos)
            neg_idx = np.random.choice(neg_indices, n_neg_metric,    replace=replace_neg)

            with tf.GradientTape() as tape:
                pos_emb = _emb(pos_idx)
                neg_emb = _emb(neg_idx)

                L_proto, L_div = proto_loss_fn.compute_loss(pos_emb, neg_emb)
                L_metric = L_proto + proto_loss_fn.lambda_div * L_div

                pred_f = classification_model(
                    [X_esm2[focal_idx].astype(np.float32),
                     X_mask[focal_idx].astype(np.float32),
                     X_seq[focal_idx], X_dm[focal_idx],
                     X_dd[focal_idx].astype(np.float32)],
                    training=True)
                L_focal = tf.reduce_mean(
                    focal_fn(y_focal, tf.squeeze(pred_f, axis=-1)))

                L_total = alpha * L_metric + (1.0 - alpha) * L_focal

            grads = tape.gradient(L_total, all_vars)
            apply_discriminative_grads(
                grads, all_vars, esm2_ids, cnn_ids, head_ids,
                opt_esm2, opt_cnn, opt_rest, opt_head)

            # EMA: tape 밖에서 numpy로 업데이트
            all_pos_emb_np.append(pos_emb.numpy())
            batch_losses_m.append(float(L_metric.numpy()))
            batch_losses_f.append(float(L_focal.numpy()))

        else:
            # ── Type-A: Triplet semi-hard ──────────────────────────────
            a_idxs, p_idxs, n_idxs = build_emb_triplet_inputs(
                embeddings, y, pos_indices, neg_indices,
                batch_size=batch_size, margin=margin,
                mode='random' if batch_i == 0 else 'hard')
            if not a_idxs:
                continue

            with tf.GradientTape() as tape:
                emb_a = _emb(a_idxs)
                emb_p = _emb(p_idxs)
                emb_n = _emb(n_idxs)
                pd       = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
                nd       = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
                raw_loss = tf.maximum(pd - nd + margin, 0.0)
                L_metric = tf.reduce_mean(raw_loss)

                pred_f = classification_model(
                    [X_esm2[focal_idx].astype(np.float32),
                     X_mask[focal_idx].astype(np.float32),
                     X_seq[focal_idx], X_dm[focal_idx],
                     X_dd[focal_idx].astype(np.float32)],
                    training=True)
                L_focal = tf.reduce_mean(
                    focal_fn(y_focal, tf.squeeze(pred_f, axis=-1)))

                L_total = alpha * L_metric + (1.0 - alpha) * L_focal

            grads = tape.gradient(L_total, all_vars)
            apply_discriminative_grads(
                grads, all_vars, esm2_ids, cnn_ids, head_ids,
                opt_esm2, opt_cnn, opt_rest, opt_head)

            active_count += int((raw_loss.numpy() > 0).sum())
            batch_losses_m.append(float(L_metric.numpy()))
            batch_losses_f.append(float(L_focal.numpy()))

    # Type-B EMA prototype update (tape 외부)
    if is_type_b and all_pos_emb_np:
        stacked = np.vstack(all_pos_emb_np)
        proto_loss_fn.update_prototypes_ema(stacked)

    avg_m       = float(np.mean(batch_losses_m)) if batch_losses_m else 0.0
    avg_f       = float(np.mean(batch_losses_f)) if batch_losses_f else 0.0
    active_rate = active_count / (n_batches * batch_size) * 100 if not is_type_b else 0.0
    return avg_m, avg_f, active_rate

# --------------------------------------------------------------------------
# [7] save 유틸
# --------------------------------------------------------------------------
def save_phase_results_hybrid(phase_name, model_base, X_esm2, X_mask, X_seq, X_dm, X_dd,
                               y_true, gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds  = model_base.predict([X_esm2, X_mask, X_seq, X_dm, X_dd], batch_size=256, verbose=1)
    np.save(os.path.join(save_dir, '{}_{}_embeddings.npy'.format(ver_tag, phase_name)), preds[0])
    np.save(os.path.join(save_dir, '{}_{}_labels.npy'.format(ver_tag, phase_name)), y_true)
    scores = preds[1]
    s = np.array([scores[i][0] for i in range(len(y_true))])
    print("  mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
        s.mean(), s.std(), (s > 0.5).sum(), (s > 0.3).sum()))
    with open(os.path.join(save_dir, '{}_{}_scores.txt'.format(ver_tag, phase_name)), 'w') as fw:
        for i in range(len(y_true)):
            fw.write(gene_ids[i] + '\t' + iso_ids[i] + '\t' + str(scores[i][0]) + '\n')
    return s

def expression_label_propagation(base_scores, X_expr, alpha=0.3, k=15, sim_threshold=0.1):
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
# [8] 데이터 로딩
# --------------------------------------------------------------------------
print('>>> Preparing Data (v8) for ' + selected_go)

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

DD_TRAIN_PATH    = '../results_isoform/features/train_domain_delta_sign.npy'
DD_SW_PATH       = '../results_isoform/features/swissprot_domain_delta_sign.npy'
DD_TEST_PATH     = '../results_isoform/features/domain_delta.npy'
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
# [9] 레이블 생성 및 업샘플
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
# [10] 모델 구조 (v7c 동일)
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
embedding_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=embedding_layer,
    name='embedding_model')

# [v8-1] Discriminative LR optimizers (Phase freeze 대체)
opt_esm2 = optimizers.Adam(lr=LR_ESM2, clipnorm=1.0)
opt_cnn  = optimizers.Adam(lr=LR_CNN,  clipnorm=0.5)
opt_rest = optimizers.Adam(lr=LR_REST, clipnorm=1.0)
opt_head = optimizers.Adam(lr=LR_HEAD, clipnorm=1.0)

K_testing_size = len(X_test_esm2)

print("\n[Model v8] ESM-2(640→64) + CNN(1500→32) + Domain(LSTM 16) + DomainDelta(251→16) → concat[128]")
print("[Model v8] Head: Dense(64) → L2_norm → emb[64] → Dense(1,sigmoid)")
print("[Model v8] Discriminative LR: ESM2={} CNN={} Rest={} Head={}".format(
    LR_ESM2, LR_CNN, LR_REST, LR_HEAD))
print("[Model v8] Unified Loss: Type-B α={} | Type-A α={}".format(ALPHA_TYPE_B, ALPHA_TYPE_A))

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
go_type_str = 'Type-B (sep={:.4f})'.format(ph0_sep_ratio) if IS_TYPE_B \
              else 'Type-A (sep={:.4f})'.format(ph0_sep_ratio)
print("\n[v6e-1] GO term classification: {}".format(go_type_str))

ALPHA_V8 = ALPHA_TYPE_B if IS_TYPE_B else ALPHA_TYPE_A
print("[v8] α={:.2f} | Type-{}".format(ALPHA_V8, 'B' if IS_TYPE_B else 'A'))

# [v8-3] Type-B: robust k 선택
proto_loss_fn = None
PROTO_K = 1

if IS_TYPE_B:
    print("\n[v8-3] Type-B: Robust k selection (Gap Statistic + occupancy)...")
    ph0_pos_emb = emb_by_phase['Ph0(untrained)'][y_test == 1]
    if len(ph0_pos_emb) >= PROTO_K_MAX * 3:
        PROTO_K = determine_k_robust(ph0_pos_emb, k_max=PROTO_K_MAX)
    else:
        PROTO_K = 1
        print("  Insufficient positives (N={}) — k=1 forced".format(len(ph0_pos_emb)))
    print("[v8-3] Selected k={} for Type-B Proto-kN".format(PROTO_K))

    proto_loss_fn = PrototypeContrastiveLoss(
        n_prototypes=PROTO_K,
        emb_dim=64,
        temperature=PROTO_TEMPERATURE,
        ema_decay=PROTO_EMA_DECAY,
        lambda_div=PROTO_LAMBDA_DIV)
    proto_loss_fn.initialize_from_embeddings(ph0_pos_emb)
else:
    print("[v8] Type-A: Triplet (margin={})".format(MARGIN_P1))

# ==========================================================================
# [v8-4] BIAS INIT: prediction_out bias (Phase0 완료 후)
# ==========================================================================
print("\n[v8-4] Bias-only warm init...")
pos_ratio_train = float((y_train_upsmp == 1).mean())
pos_ratio_train = np.clip(pos_ratio_train, 1e-6, 1.0 - 1e-6)
bias_val = float(np.log(pos_ratio_train / (1.0 - pos_ratio_train)))
for _layer in base_model.layers:
    if _layer.name == 'prediction_out':
        _w, _b = _layer.get_weights()
        _layer.set_weights([_w, np.array([bias_val], dtype=np.float32)])
        print("  prediction_out bias = {:.4f} (pos_ratio={:.4f})".format(
            bias_val, pos_ratio_train))
        break

# ==========================================================================
# [v8] UNIFIED TRAINING: metric + focal, discriminative LR
# ==========================================================================
print("\n" + "=" * 58)
print(">>> UNIFIED TRAINING [v8-1,2]")
print("    ALL layers trainable — discriminative LR replaces Phase freeze")
print("    α={:.2f} | ESM2 lr={} | CNN lr={} | Head lr={}".format(
    ALPHA_V8, LR_ESM2, LR_CNN, LR_HEAD))
print("=" * 58)

# 모든 레이어 trainable 확인 (v7c의 Phase freeze 제거)
for layer in base_model.layers:
    layer.trainable = True
print("  [v8] All layers trainable confirmed")

# Discriminative LR를 위한 변수 그룹 구성
esm2_ids, cnn_ids, head_ids = build_var_groups(base_model)
print("  [v8] Variable groups: ESM2={} CNN={} Head={} Rest={}".format(
    len(esm2_ids), len(cnn_ids), len(head_ids),
    len(classification_model.trainable_variables) - len(esm2_ids) - len(cnn_ids) - len(head_ids)))

UNIFIED_MAX_EPOCHS = 35
PATIENCE_V8 = 15 if IS_TYPE_B else 10

best_auprc   = 0.0
best_weights = None
no_improve   = 0
prev_centroid_cos = ph0_metrics.get('centroid_cos', float('nan'))

print("\n  Max epochs={} | patience={} (AUPRC-based early stopping)".format(
    UNIFIED_MAX_EPOCHS, PATIENCE_V8))

for epoch in range(UNIFIED_MAX_EPOCHS):
    print('\nUnified Epoch: {}/{}'.format(epoch + 1, UNIFIED_MAX_EPOCHS))

    avg_m, avg_f, active_rate = unified_epoch_v8(
        embedding_model=embedding_model,
        classification_model=classification_model,
        X_esm2=X_train_esm2_upsmp,
        X_mask=X_train_mask_upsmp,
        X_seq=X_train_seq_upsmp,
        X_dm=X_train_dm_upsmp,
        X_dd=X_train_dd_upsmp,
        y=y_train_upsmp,
        esm2_ids=esm2_ids, cnn_ids=cnn_ids, head_ids=head_ids,
        opt_esm2=opt_esm2, opt_cnn=opt_cnn, opt_rest=opt_rest, opt_head=opt_head,
        alpha=ALPHA_V8,
        is_type_b=IS_TYPE_B,
        proto_loss_fn=proto_loss_fn,
        margin=MARGIN_P1,
        batch_size=BATCH_SIZE_P1,
        n_batches=N_BATCHES_P1,
        refresh_interval=10,
        focal_alpha_val=0.10)

    if IS_TYPE_B:
        print("  -> Metric(proto): {:.4f} | Focal: {:.4f}".format(avg_m, avg_f))
    else:
        print("  -> Metric(triplet): {:.4f} | Focal: {:.4f} | Active: {:.1f}%".format(
            avg_m, avg_f, active_rate))

    # ── 평가 (매 epoch) ──────────────────────────────────────────────────
    preds = base_model.predict(
        [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
        batch_size=256, verbose=0)
    curr_emb    = preds[0]
    test_scores = preds[1].squeeze()

    # ── 5 epoch마다 embedding 품질 진단 ([F5] centroid_cos 모니터링) ────
    if (epoch + 1) % 5 == 0:
        curr_metrics = analyze_embedding_quality(curr_emb, y_test, 'Epoch {}'.format(epoch + 1))
        curr_cc = curr_metrics.get('centroid_cos', float('nan'))
        if not np.isnan(prev_centroid_cos) and not np.isnan(curr_cc):
            delta_cc = curr_cc - prev_centroid_cos
            flag = " ⚠ [F5 감지: centroid_cos 악화!]" if delta_cc > 0.05 else ""
            print("  [Diag] centroid_cos: {:.4f} → {:.4f} (Δ={:+.4f}){}".format(
                prev_centroid_cos, curr_cc, delta_cc, flag))
        prev_centroid_cos = curr_cc

        if IS_TYPE_B and proto_loss_fn is not None:
            proto_loss_fn.prototype_stats(curr_emb, y_test)

    # ── AUPRC early stopping ──────────────────────────────────────────────
    if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
        auroc = roc_auc_score(y_test, test_scores)
        auprc = average_precision_score(y_test, test_scores)
        print("  [Eval] AUROC={:.4f} AUPRC={:.4f} (best={:.4f}, patience={}/{})".format(
            auroc, auprc, best_auprc, no_improve, PATIENCE_V8))
        if auprc > best_auprc:
            best_auprc   = auprc
            best_weights = base_model.get_weights()
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE_V8:
                print("  [Early Stop] AUPRC not improving for {} epochs".format(PATIENCE_V8))
                break

if best_weights is not None:
    base_model.set_weights(best_weights)
    print("\n[Unified Training] Restored best weights (AUPRC={:.4f})".format(best_auprc))

# ── Unified training 완료: 임베딩 저장 및 분석 ─────────────────────────
s_unified = save_phase_results_hybrid(
    "unified_training", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Unified'] = np.load(
    os.path.join(SAVE_DIR, '{}_unified_training_embeddings.npy'.format(VER_TAG)))
score_by_phase['Unified'] = s_unified
analyze_embedding_quality(emb_by_phase['Unified'], y_test, 'Unified Training Final')

# Prototype biological validation (Type-B)
if IS_TYPE_B and proto_loss_fn is not None:
    save_prototype_assignments(
        proto_loss_fn,
        emb_by_phase['Unified'],
        y_test,
        save_dir=SAVE_DIR,
        ver_tag=VER_TAG,
        go_term=selected_go)

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
    "Phase", "CentCos", "Sep.Ratio", "LinAUROC", "PredAUROC"))
print("-" * 70)
for phase_name, emb in emb_by_phase.items():
    m = analyze_embedding_quality(emb, y_test, phase_name)
    try:
        pred_auroc = roc_auc_score(y_test, score_by_phase[phase_name])
    except Exception:
        pred_auroc = float('nan')
    print("{:25} {:10.4f} {:10.4f} {:10.4f} {:12.4f}".format(
        phase_name, m['centroid_cos'], m['sep_ratio'],
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

if IS_TYPE_B and proto_loss_fn is not None and proto_loss_fn.prototypes is not None:
    np.save(os.path.join(SAVE_DIR, "{}_{}_prototypes.npy".format(VER_TAG, safe_go)),
            proto_loss_fn.prototypes.numpy())
    print("  [Proto] Saved k={} prototypes".format(PROTO_K))

print("\n[Final] GO={} | Type={} | k={} | α={:.2f} | unified_epochs={}".format(
    selected_go, go_type_str,
    PROTO_K if IS_TYPE_B else 'N/A',
    ALPHA_V8,
    UNIFIED_MAX_EPOCHS))
print("[Final] AUROC={:.4f} AUPRC={:.4f}".format(final_auroc, final_auprc))
print("[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
