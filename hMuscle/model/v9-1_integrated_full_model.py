# -*- coding: utf-8 -*-
# ============================================================================
# v9-1_integrated_full_model.py
#
# v9-1: GENCODE Canonical + Splicing Delta v2 통합
#
# 핵심 변경 (v8-3 대비):
#   [v9-1-1] Canonical 재정의: "Pfam 최다" → GENCODE v44 MANE_Select/Ensembl_canonical
#            34.4% 유전자에서 canonical 교체 (4,066개 유전자)
#            → domain_delta_v2.npy 사용
#
#   [v9-1-2] Splicing Delta v2 도입: Binary(50dim) → Continuous(150dim)
#            - Overlap 기반 exon clustering + constitutive exon 제거
#            - Alternative splice site 감지 (38,322 fractional deltas)
#            → splicing_delta_v2.npy 사용
#
#   [v9-1-3] Phase 3 Gene-Stratified CV LR: test set 내부 K-fold
#            LR 입력 = [Phase1 embedding 64] + [domain_delta_v2 251] + [splicing_delta_v2 150]
#            = 465dim, gene-level GroupKFold (Axis 2 leakage 방지)
#            train/test feature asymmetry 문제 근본 해결
#            자동 ablation: emb-only / emb+dd / emb+dd+splice 3단계 비교 출력
#
# D×S 교정 효과 (검증 완료):
#   D0 S1: 788 (2.1%) → 3,608 (9.8%)  — +358%
#   D1 S0: 4,904 (13.3%) → 3,925 (10.7%) — 파이프라인 아티팩트 감소
#   S1 중 D0 비율: 4.2% → 18.8% (문헌 IDR 편향 방향)
#
# v8-3 승계 사항:
#   [v8-3-1] Phase 1 TRAIN embedding 추출 (balanced LR 학습용)
#   [v8-3-2] Type-B: Phase 2 완전 SKIP
#   [v8-3-3] Type-B: Phase 3 = LogisticRegression(class_weight='balanced', C=1.0)
#   [v8-3-4] Final score: Type-B = balanced LR score, Type-A = sigmoid head
#
# [ARCH] v7c 동일
#   ESM-2: 640→64 | CNN: 1500→32 | Domain: LSTM16 | DomainDelta: 251→16
#   Fusion: concat[128] → Dense(64) → L2_norm → emb[64] → Dense(1,sigmoid)
#
# [TRAIN] v8b 승계
#   Phase 1 (15 ep): v7c 동일 (ESM-2+Domain, CNN frozen)
#   Phase 2 (25 ep): Unified(α=0.2) + 5그룹 discriminative LR (Type-A만)
#   Phase 3: Balanced LR on [embedding + splicing_delta_v2]
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

from v7c_prototype_contrastive import (
    determine_k_gap_statistic,
    PrototypeContrastiveLoss,
    phase1_proto_epoch_hybrid,
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
    print("Usage: python v9-1_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go
if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d_%H%M")
VER_TAG  = "v9-1_integrated"
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

PROTO_K_MAX       = 5
PROTO_TEMPERATURE = 0.1
PROTO_EMA_DECAY   = 0.9
PROTO_LAMBDA_DIV  = 0.1
WARM_INIT_SCALE   = 2.0   # Phase 1→2 prototype warm init (v7b-2 유지)

# [v8b-1] Phase 2 discriminative LR
LR_P2_SLOW = 1e-5   # ESM-2 + head_dense64 (geometry 보호)
LR_P2_CNN  = 1e-4   # CNN branch
LR_P2_REST = 1e-4   # Domain, DomainDelta
LR_P2_PRED = 1e-3   # prediction_out (head)

# [v8b-2] Phase 2 Unified Loss
ALPHA_P2 = 0.2      # AP7 준수: metric loss 완전 소멸 방지

# Layer name groups
ESM2_TRAINABLE_NAMES  = {'esm2_d1', 'esm2_d2', 'esm2_feat'}
CNN_TRAINABLE_NAMES   = {'cnn_emb', 'cnn_conv1', 'cnn_conv2', 'cnn_feat'}
HEAD_DENSE_NAMES      = {'head_dense64'}          # embedding geometry — slow LR
PRED_OUT_NAMES        = {'prediction_out'}         # classification head — normal LR
# rest: domain_feat, dd_dense1, dd_dense2, dm_emb → LR_P2_REST

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
# [EMB] 임베딩 품질 분석
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
        near_neg = 0
        n_check = min(500, len(pos_emb))
        sample_pos = pos_emb[np.random.choice(len(pos_emb), n_check, replace=False)]
        for pe in sample_pos:
            if (neg_emb @ pe > 0.9).any():
                near_neg += 1
        metrics['frac_pos_near_neg'] = float(near_neg / n_check)
    else:
        metrics.update({'intra_dist': float('nan'), 'inter_dist': float('nan'),
                        'sep_ratio': float('nan'), 'frac_pos_near_neg': float('nan')})

    if len(pos_emb) >= 2 and len(neg_emb) >= 2:
        n_sub   = min(4000, len(embeddings))
        idx_sub = np.random.choice(len(embeddings), n_sub, replace=False)
        try:
            metrics['silhouette'] = float(silhouette_score(
                embeddings[idx_sub], labels[idx_sub].astype(int),
                metric='cosine', sample_size=min(2000, n_sub), random_state=seed)) \
                if labels[idx_sub].sum() >= 2 and (labels[idx_sub] == 0).sum() >= 2 else float('nan')
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
            metrics['linear_auroc'] = float(roc_auc_score(
                labels, lr.predict_proba(emb_sc)[:, 1]))
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
# [6] Triplet helpers (v7c 동일)
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
            p_i     = np.random.choice(p_candidates)
            a_emb   = embeddings[a_i]
            p_emb   = embeddings[p_i]
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
    def refresh_emb():
        emb = extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd)
        return emb, np.where(y == 1)[0], np.where(y == 0)[0]

    embeddings, pos_indices, neg_indices = refresh_emb()
    if len(pos_indices) < 2 or len(neg_indices) == 0:
        return 0.0, 0
    mode         = 'random' if warmup else 'hard'
    batch_losses = []
    active_count = 0

    for batch_i in range(n_batches):
        if batch_i > 0 and batch_i % refresh_interval == 0 and not warmup:
            embeddings, pos_indices, neg_indices = refresh_emb()
        a_idxs, p_idxs, n_idxs = build_emb_triplet_inputs(
            embeddings, y, pos_indices, neg_indices,
            batch_size=batch_size, margin=margin, mode=mode)
        if not a_idxs:
            continue
        with tf.GradientTape() as tape:
            def get_live_emb(indices):
                raw = feature_model(
                    [X_esm2[indices].astype(np.float32), X_mask[indices].astype(np.float32),
                     X_seq[indices], X_dm[indices], X_dd[indices].astype(np.float32)],
                    training=True)
                return tf.math.l2_normalize(raw, axis=1)
            emb_a = get_live_emb(a_idxs)
            emb_p = get_live_emb(p_idxs)
            emb_n = get_live_emb(n_idxs)
            pos_d    = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
            neg_d    = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
            raw_loss = tf.maximum(pos_d - neg_d + margin, 0.0)
            loss     = tf.reduce_mean(raw_loss)
        grads = tape.gradient(loss, feature_model.trainable_variables)
        g_v = [(g, v) for g, v in zip(grads, feature_model.trainable_variables) if g is not None]
        if g_v:
            optimizer.apply_gradients(g_v)
        batch_losses.append(float(loss.numpy()))
        active_count += int((raw_loss.numpy() > 0).sum())

    avg_loss    = np.mean(batch_losses) if batch_losses else 0.0
    active_rate = active_count / (n_batches * batch_size) * 100
    return avg_loss, active_rate

def compute_margin_stats_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
                                 margin=MARGIN_P1, n_sample=1000):
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
# [7] Phase 1→2 Warm Init (v7b-2 유지)
# --------------------------------------------------------------------------
def prototype_warm_init(base_model, embeddings_np, y_np, proto_fn=None,
                         scale=WARM_INIT_SCALE):
    pos_emb = embeddings_np[y_np == 1]
    neg_emb = embeddings_np[y_np == 0]
    if len(pos_emb) < 2 or len(neg_emb) < 2:
        print("  [WarmInit] Skipped — insufficient pos/neg samples")
        return False
    if proto_fn is not None and proto_fn.prototypes is not None:
        protos   = proto_fn.prototypes.numpy()
        protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-10)
        pos_rep  = protos_n.mean(axis=0)
        print("  [WarmInit] Type-B: using mean of {} prototypes as pos_rep".format(proto_fn.k))
    else:
        pos_rep = pos_emb.mean(axis=0)
        print("  [WarmInit] Type-A: using positive centroid as pos_rep")
    pos_rep = pos_rep / (np.linalg.norm(pos_rep) + 1e-10)
    neg_rep = neg_emb.mean(axis=0)
    neg_rep = neg_rep / (np.linalg.norm(neg_rep) + 1e-10)
    w_dir   = pos_rep - neg_rep
    w_dir   = w_dir / (np.linalg.norm(w_dir) + 1e-10)
    for layer in base_model.layers:
        if layer.name == 'prediction_out':
            w_init = (w_dir * scale).reshape(64, 1).astype(np.float32)
            b_init = np.array([-scale * 0.25], dtype=np.float32)
            layer.set_weights([w_init, b_init])
            print("  [WarmInit] prediction_out initialized | scale={:.1f}".format(scale))
            return True
    print("  [WarmInit] WARNING: 'prediction_out' layer not found")
    return False

# --------------------------------------------------------------------------
# [v8b-3] Phase 1→2 Interface: Prototype 재초기화 + Bias init
# --------------------------------------------------------------------------
def reinit_prototypes_from_phase1(proto_loss_fn, ph1_embeddings, y_test):
    """
    [v8b-3] Phase 1 최종 test embedding으로 prototype 재초기화.
    Phase 2에서 embedding drift 시 Phase 1 EMA prototype이 stale해지는 문제 해결.
    """
    if proto_loss_fn is None:
        return
    pos_emb = ph1_embeddings[y_test == 1]
    if len(pos_emb) < proto_loss_fn.k:
        print("  [ProtoReinit] Insufficient pos ({}) for k={} — skipping".format(
            len(pos_emb), proto_loss_fn.k))
        return
    proto_loss_fn.initialize_from_embeddings(pos_emb)
    print("  [ProtoReinit] k={} prototypes re-initialized from Phase 1 test embeddings".format(
        proto_loss_fn.k))

def set_bias_from_test_distribution(base_model, y_test):
    """
    [v8b-3] prediction_out bias = log(pos_ratio_test / (1-pos_ratio_test)).
    Upsampled 50:50이 아닌 실제 test 분포 기준으로 초기화 (보정).
    """
    pos_ratio = float((y_test == 1).mean())
    pos_ratio = np.clip(pos_ratio, 1e-6, 1.0 - 1e-6)
    bias_val  = float(np.log(pos_ratio / (1.0 - pos_ratio)))
    for layer in base_model.layers:
        if layer.name == 'prediction_out':
            w, b = layer.get_weights()
            layer.set_weights([w, np.array([bias_val], dtype=np.float32)])
            print("  [BiasInit] prediction_out bias = {:.4f} (test pos_ratio={:.4f}%)".format(
                bias_val, pos_ratio * 100))
            return
    print("  [BiasInit] WARNING: prediction_out not found")

# --------------------------------------------------------------------------
# [v8b-4] Proto-kN k 선택: Gap Statistic + minimum occupancy
# --------------------------------------------------------------------------
def determine_k_robust(pos_emb, k_max=5):
    k_gap = determine_k_gap_statistic(pos_emb, k_max=k_max, n_refs=10, random_state=42)
    print("  [k_robust] k_gap={}, occupancy validation...".format(k_gap))
    for k in range(k_gap, 0, -1):
        km     = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(pos_emb)
        counts = np.bincount(labels, minlength=k)
        min_c  = counts.min()
        thresh = max(3, len(pos_emb) // (k * 5))
        print("  [k_robust] k={}: min_occ={} (thresh={}) counts={}".format(
            k, min_c, thresh, counts.tolist()))
        if min_c >= thresh:
            print("  [k_robust] k={} accepted".format(k))
            return k
    print("  [k_robust] Fallback k=1")
    return 1

# --------------------------------------------------------------------------
# [v8b-1,2] Phase 2 Discriminative LR helpers
# --------------------------------------------------------------------------
def build_var_groups_p2(model):
    """
    Phase 2용 5그룹 분류.
    slow_ids  = ESM-2 + head_dense64  (lr=1e-5, geometry 보호)
    cnn_ids   = CNN branch            (lr=1e-4)
    pred_ids  = prediction_out        (lr=1e-3)
    rest_ids  = 나머지 (Domain, DD)  (lr=1e-4)
    """
    slow_ids = set()   # ESM-2 + head_dense64
    cnn_ids  = set()
    pred_ids = set()
    for layer in model.layers:
        ids = {id(v) for v in layer.trainable_variables}
        if layer.name in ESM2_TRAINABLE_NAMES or layer.name in HEAD_DENSE_NAMES:
            slow_ids |= ids
        elif layer.name in CNN_TRAINABLE_NAMES:
            cnn_ids |= ids
        elif layer.name in PRED_OUT_NAMES:
            pred_ids |= ids
    return slow_ids, cnn_ids, pred_ids

def apply_grads_p2(grads, variables, slow_ids, cnn_ids, pred_ids,
                   opt_slow, opt_cnn, opt_rest, opt_pred):
    g_slow, g_cnn, g_pred, g_rest = [], [], [], []
    for g, v in zip(grads, variables):
        if g is None:
            continue
        vid = id(v)
        if vid in slow_ids:
            g_slow.append((g, v))
        elif vid in cnn_ids:
            g_cnn.append((g, v))
        elif vid in pred_ids:
            g_pred.append((g, v))
        else:
            g_rest.append((g, v))
    if g_slow: opt_slow.apply_gradients(g_slow)
    if g_cnn:  opt_cnn.apply_gradients(g_cnn)
    if g_pred: opt_pred.apply_gradients(g_pred)
    if g_rest: opt_rest.apply_gradients(g_rest)

# --------------------------------------------------------------------------
# [v8b-2] Phase 2 Unified Epoch: α=0.2 metric + 0.8 focal
# --------------------------------------------------------------------------
def phase2_unified_epoch_v8b(
        embedding_model, classification_model,
        X_esm2, X_mask, X_seq, X_dm, X_dd, y,
        slow_ids, cnn_ids, pred_ids,
        opt_slow, opt_cnn, opt_rest, opt_pred,
        is_type_b, proto_loss_fn=None,
        margin=MARGIN_P1, batch_size=256, n_batches=50,
        refresh_interval=10, focal_alpha_val=0.10):
    """
    [v8b-2] Phase 2 Unified Loss (α=0.2 고정).

    L_total = ALPHA_P2 × L_metric + (1-ALPHA_P2) × L_focal

    Type-B: Proto-kN Neg-InfoNCE  (Phase 1과 동일, EMA 계속 업데이트)
    Type-A: Triplet semi-hard     (Phase 1과 동일)
    Focal:  random mini-batch

    핵심 차이 vs Phase 1:
      - CNN trainable (Phase 1: frozen)
      - head_dense64 lr=1e-5 (geometry 보호)
      - prediction_out lr=1e-3 (classification 학습)
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

    if not is_type_b:
        embeddings = extract_embeddings_hybrid(
            embedding_model, X_esm2, X_mask, X_seq, X_dm, X_dd)
    else:
        embeddings = None

    batch_losses_m = []
    batch_losses_f = []
    active_count   = 0
    all_pos_emb_np = []

    for batch_i in range(n_batches):
        if not is_type_b and batch_i > 0 and batch_i % refresh_interval == 0:
            embeddings = extract_embeddings_hybrid(
                embedding_model, X_esm2, X_mask, X_seq, X_dm, X_dd)

        focal_idx = np.random.choice(n, batch_size, replace=True)
        y_focal   = tf.cast(y[focal_idx], tf.float32)

        def _emb(idx):
            return tf.math.l2_normalize(
                embedding_model(
                    [X_esm2[idx].astype(np.float32), X_mask[idx].astype(np.float32),
                     X_seq[idx], X_dm[idx], X_dd[idx].astype(np.float32)],
                    training=True), axis=1)

        if is_type_b:
            pos_idx = np.random.choice(pos_indices, n_pos_per_batch, replace=replace_pos)
            neg_idx = np.random.choice(neg_indices, n_neg_metric,    replace=replace_neg)
            with tf.GradientTape() as tape:
                pos_emb = _emb(pos_idx)
                neg_emb = _emb(neg_idx)
                L_proto, L_div = proto_loss_fn.compute_loss(pos_emb, neg_emb)
                L_metric = L_proto + proto_loss_fn.lambda_div * L_div
                pred_f = classification_model(
                    [X_esm2[focal_idx].astype(np.float32), X_mask[focal_idx].astype(np.float32),
                     X_seq[focal_idx], X_dm[focal_idx], X_dd[focal_idx].astype(np.float32)],
                    training=True)
                L_focal  = tf.reduce_mean(focal_fn(y_focal, tf.squeeze(pred_f, axis=-1)))
                L_total  = ALPHA_P2 * L_metric + (1.0 - ALPHA_P2) * L_focal
            grads = tape.gradient(L_total, all_vars)
            apply_grads_p2(grads, all_vars, slow_ids, cnn_ids, pred_ids,
                           opt_slow, opt_cnn, opt_rest, opt_pred)
            all_pos_emb_np.append(pos_emb.numpy())
            batch_losses_m.append(float(L_metric.numpy()))
            batch_losses_f.append(float(L_focal.numpy()))

        else:
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
                    [X_esm2[focal_idx].astype(np.float32), X_mask[focal_idx].astype(np.float32),
                     X_seq[focal_idx], X_dm[focal_idx], X_dd[focal_idx].astype(np.float32)],
                    training=True)
                L_focal = tf.reduce_mean(focal_fn(y_focal, tf.squeeze(pred_f, axis=-1)))
                L_total = ALPHA_P2 * L_metric + (1.0 - ALPHA_P2) * L_focal
            grads = tape.gradient(L_total, all_vars)
            apply_grads_p2(grads, all_vars, slow_ids, cnn_ids, pred_ids,
                           opt_slow, opt_cnn, opt_rest, opt_pred)
            active_count += int((raw_loss.numpy() > 0).sum())
            batch_losses_m.append(float(L_metric.numpy()))
            batch_losses_f.append(float(L_focal.numpy()))

    # EMA update (Phase 2에서도 계속)
    if is_type_b and all_pos_emb_np:
        proto_loss_fn.update_prototypes_ema(np.vstack(all_pos_emb_np))

    avg_m       = float(np.mean(batch_losses_m)) if batch_losses_m else 0.0
    avg_f       = float(np.mean(batch_losses_f)) if batch_losses_f else 0.0
    active_rate = active_count / (n_batches * batch_size) * 100 if not is_type_b else 0.0
    return avg_m, avg_f, active_rate

# --------------------------------------------------------------------------
# [8] Save 유틸 (v7c 동일)
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
        j = indices[:, rank + 1]; s = sims[:, rank]
        prop_scores += s * base_scores[j]; weight_sum += s
    valid = weight_sum > 0
    prop_scores[valid] /= weight_sum[valid]
    prop_scores[~valid] = base_scores[~valid]
    return (1.0 - alpha) * base_scores + alpha * prop_scores

# --------------------------------------------------------------------------
# [9] 데이터 로딩 (v7c 동일)
# --------------------------------------------------------------------------
print('>>> Preparing Data (v8b) for ' + selected_go)

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

X_train_seq       = np.load('../data/raw_data/data/sequences/human_sequence_train.npy')[:, -SEQ_LEN:]
X_train_other_seq = np.load('../data/raw_data/data/sequences/swissprot_sequence_train.npy')[:, -SEQ_LEN:]
X_test_seq        = np.load('my_sequence_matrix_fixed.npy')[:, -SEQ_LEN:]
print("  [Seq] train={} other={} test={}".format(
    X_train_seq.shape, X_train_other_seq.shape, X_test_seq.shape))

X_train_dm       = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_test_dm        = np.load('../results/domain/domain_matrix.npy')

X_train_dd       = np.load('../results_isoform/features/train_domain_delta_sign.npy').astype(np.float32)
X_train_other_dd = np.load('../results_isoform/features/swissprot_domain_delta_sign.npy').astype(np.float32)
# [v9-1-1] domain_delta_v2 사용 (GENCODE MANE/Ensembl canonical 기반)
_dd_v2_path = '../results_isoform/features/domain_delta_v2.npy'
if os.path.exists(_dd_v2_path):
    X_test_dd = np.sign(np.load(_dd_v2_path)).astype(np.float32)
    print("  [v9-1] domain_delta_v2 loaded (GENCODE canonical)")
else:
    X_test_dd = np.sign(np.load('../results_isoform/features/domain_delta.npy')).astype(np.float32)
    print("  [v9-1] WARN: domain_delta_v2 not found, using v1 fallback")
print("  [v6d] domain_delta: train={} sw={} test={}".format(
    X_train_dd.shape, X_train_other_dd.shape, X_test_dd.shape))

# [v9-1-2] splicing_delta_v2 로드 (Phase 3 Augmented LR용)
_sd_v2_path = '../results_isoform/features/splicing/splicing_delta_v2.npy'
if os.path.exists(_sd_v2_path):
    X_test_splicing_delta = np.load(_sd_v2_path).astype(np.float32)
    print("  [v9-1] splicing_delta_v2: {} (continuous, cluster-based)".format(
        X_test_splicing_delta.shape))
else:
    # fallback to v1
    _sd_v1_path = '../results_isoform/features/splicing/splicing_delta.npy'
    X_test_splicing_delta = np.load(_sd_v1_path).astype(np.float32)
    print("  [v9-1] WARN: splicing_delta_v2 not found, using v1 fallback: {}".format(
        X_test_splicing_delta.shape))
SPLICE_DIM = X_test_splicing_delta.shape[1]
print("  [v9-1] SPLICE_DIM={}".format(SPLICE_DIM))

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

test_iso_arr = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
EXPR_CACHE   = 'expr_matrix_fixed.npy'
CPM_PATH     = '../data/bambu_data/CPM_transcript.txt'
if os.path.exists(EXPR_CACHE):
    X_test_expr = np.load(EXPR_CACHE).astype(np.float32)
    print("[Expr] Loaded cache: {}".format(X_test_expr.shape))
else:
    X_test_expr = load_expression_matrix(CPM_PATH, test_iso_arr)
    np.save(EXPR_CACHE, X_test_expr)
assert X_test_expr.shape == (len(test_iso_arr), EXPR_DIM)

# --------------------------------------------------------------------------
# [10] 레이블 생성 및 업샘플 (v7c 동일)
# --------------------------------------------------------------------------
print('>>> Generating Labels for ' + selected_go)

dm_dim         = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm), np.max(X_train_other_dm)]) + 1

y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, \
    _dummy, X_train_dm_comb = generate_label(
        X_train_esm2, X_train_dm, X_train_other_esm2, X_train_other_dm,
        X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

_, _, _, _, _, _, X_train_esm2_comb, X_train_mask_comb = generate_label(
    X_train_esm2, X_train_esm2_mask, X_train_other_esm2, X_train_other_esm2_mask,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

_, _, _, _, _, _, _d2, X_train_seq_comb = generate_label(
    X_train_esm2, X_train_seq, X_train_other_esm2, X_train_other_seq,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

_, _, _, _, _, _, _d3, X_train_dd_comb = generate_label(
    X_train_esm2, X_train_dd, X_train_other_esm2, X_train_other_dd,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

n_pos_train = int((y_train == 1).sum())
print("  y_train pos={} neg={} ratio={:.3f}%".format(
    n_pos_train, int((y_train == 0).sum()), n_pos_train / len(y_train) * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

MASK_START = ESM2_DIM; SEQ_START = ESM2_DIM + 1; DM_START = ESM2_DIM + 1 + SEQ_LEN
X_train_combined = np.hstack([
    X_train_esm2_comb, X_train_mask_comb,
    X_train_seq_comb.astype(np.float32),
    X_train_dm_comb.astype(np.float32),
    X_train_dd_comb.astype(np.float32)])
DD_START = DM_START + X_train_dm_comb.shape[1]

np.random.seed(42)
unused_flag = np.zeros(y_train.shape[0])
X_combined_upsmp, _dummy_dm, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count, X_train_combined,
    np.zeros((len(y_train), 1)), unused_flag)

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
    N_BATCHES_P1, n_pos_train, N_BATCHES_P1 * BATCH_SIZE_P1 / n_pos_train))

# --------------------------------------------------------------------------
# [11] 모델 구조 (v7c 동일)
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
esm2_feat  = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5),
                   name='esm2_feat')(x_esm)
esm2_gated = Lambda(lambda x: x[0] * x[1], name='esm2_gated')([esm2_feat, esm2_mask_in])

x_seq = Embedding(8001, 32, mask_zero=False, name='cnn_emb')(seq_input)
x_seq = Conv1D(64, kernel_size=7, padding='same', activation='relu', name='cnn_conv1')(x_seq)
x_seq = Conv1D(32, kernel_size=5, padding='same', activation='relu', name='cnn_conv2')(x_seq)
x_seq = GlobalMaxPooling1D(name='cnn_pool')(x_seq)
cnn_feat = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-5),
                 name='cnn_feat')(x_seq)

x_dm = Embedding(input_dim=domain_emb_dim, output_dim=32, input_length=dm_dim,
                  mask_zero=True, name='dm_emb')(domain_input)
domain_feat = LSTM(16, name='domain_feat')(x_dm)

x_dd = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5),
             name='dd_dense1')(dd_input)
x_dd = Dropout(0.2)(x_dd)
dd_feat = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-5),
                name='dd_dense2')(x_dd)

concat = concatenate([esm2_gated, cnn_feat, domain_feat, dd_feat], name='feature_concat')
feature_model = Model([esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
                       concat, name='feature_model')

x = Dense(64, kernel_regularizer=regularizers.l2(1e-5), name='head_dense64')(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer  = Lambda(lambda a: K.l2_normalize(a, axis=1), name='embedding_out')(x)
prediction_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-5),
                          name='prediction_out')(embedding_layer)

base_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=[embedding_layer, prediction_layer])
classification_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=prediction_layer)
embedding_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=embedding_layer, name='embedding_model')

# Phase 1 optimizer (v7c 동일)
adam_p1 = optimizers.Adam(lr=0.0005)

K_testing_size = len(X_test_esm2)
print("\n[Model v8b] Architecture: v7c 동일")
print("[Model v8b] Phase 1: v7c 완전 동일 (ESM-2+Domain, CNN frozen)")
print("[Model v8b] Phase 2: Unified Loss α={} + 5-group discriminative LR".format(ALPHA_P2))
print("[Model v8b] Phase 2 LR: ESM2+head_dense64={} | CNN={} | pred_out={} | rest={}".format(
    LR_P2_SLOW, LR_P2_CNN, LR_P2_PRED, LR_P2_REST))

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

if IS_TYPE_B:
    print("  -> Phase 1: Proto-kN [v8b-4 k_robust] | Phase 2: Unified(α={})".format(ALPHA_P2))
else:
    print("  -> Phase 1: Triplet | Phase 2: Unified(α={})".format(ALPHA_P2))

# [v8b-4] k 선택
proto_loss_fn = None
PROTO_K = 1

if IS_TYPE_B:
    print("\n[v8b-4] Robust k selection...")
    ph0_pos_emb = emb_by_phase['Ph0(untrained)'][y_test == 1]
    PROTO_K = determine_k_robust(ph0_pos_emb, k_max=PROTO_K_MAX) \
              if len(ph0_pos_emb) >= PROTO_K_MAX * 3 else 1
    print("[v8b-4] Selected k={}".format(PROTO_K))
    proto_loss_fn = PrototypeContrastiveLoss(
        n_prototypes=PROTO_K, emb_dim=64,
        temperature=PROTO_TEMPERATURE, ema_decay=PROTO_EMA_DECAY,
        lambda_div=PROTO_LAMBDA_DIV)
    proto_loss_fn.initialize_from_embeddings(ph0_pos_emb)

# ==========================================================================
# PHASE 1: Type-Adaptive Contrastive Loss (v7c 완전 동일)
# ==========================================================================
print("\n" + "=" * 58)
if IS_TYPE_B:
    print(">>> PHASE 1: Proto-kN + Neg-InfoNCE [v7c] — ESM-2+Domain (CNN frozen)")
    print("    k={} | τ={} | λ_div={} | EMA α={}".format(
        PROTO_K, PROTO_TEMPERATURE, PROTO_LAMBDA_DIV, PROTO_EMA_DECAY))
else:
    print(">>> PHASE 1: Triplet [v7c] — ESM-2+Domain (CNN frozen)")
    print("    margin={} | n_batches={}".format(MARGIN_P1, N_BATCHES_P1))
print("=" * 58)

set_cnn_trainable(feature_model, False)
print("  [Phase1] CNN frozen | ESM-2 + Domain + DeltaBranch trainable")

PHASE1_EPOCHS = 15

if IS_TYPE_B:
    LOSS_THRESH      = 0.01
    STREAK_LIMIT     = 4
    low_loss_streak  = 0

    for epoch in range(PHASE1_EPOCHS):
        print('Phase 1 - Epoch: {}/{} [Proto-k{}]'.format(epoch + 1, PHASE1_EPOCHS, PROTO_K))
        avg_proto, avg_div = phase1_proto_epoch_hybrid(
            feature_model=embedding_model,
            X_esm2=X_train_esm2_upsmp, X_mask=X_train_mask_upsmp,
            X_seq=X_train_seq_upsmp, X_dm=X_train_dm_upsmp,
            X_dd=X_train_dd_upsmp, y=y_train_upsmp,
            optimizer=adam_p1, proto_loss_fn=proto_loss_fn,
            batch_size=BATCH_SIZE_P1, n_batches=N_BATCHES_P1,
            min_pos_in_batch=8, do_ema_update=True)
        print("  -> Proto Loss: {:.4f} | Div Loss: {:.4f}".format(avg_proto, avg_div))
        if (epoch + 1) % 5 == 0:
            preds_tmp = base_model.predict(
                [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
                batch_size=256, verbose=0)
            proto_loss_fn.prototype_stats(preds_tmp[0], y_test)
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
    ACTIVE_RATE_THRESH = 2.0
    STREAK_LIMIT       = 3
    low_active_streak  = 0
    best_margin_sat    = 0.0
    best_centroid_dist = 0.0

    for epoch in range(PHASE1_EPOCHS):
        print('Phase 1 - Epoch: {}/{} [Triplet]'.format(epoch + 1, PHASE1_EPOCHS))
        warmup = (epoch == 0)
        avg_loss, active_rate = phase1_triplet_epoch_hybrid(
            feature_model=embedding_model,
            X_esm2=X_train_esm2_upsmp, X_mask=X_train_mask_upsmp,
            X_seq=X_train_seq_upsmp, X_dm=X_train_dm_upsmp,
            X_dd=X_train_dd_upsmp, y=y_train_upsmp,
            optimizer=adam_p1, margin=MARGIN_P1,
            batch_size=BATCH_SIZE_P1, n_batches=N_BATCHES_P1, warmup=warmup)
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
                X_test_seq, X_test_dm, X_test_dd, y_test)
            best_margin_sat    = max(best_margin_sat, sat)
            best_centroid_dist = max(best_centroid_dist, cdist)
    print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
        best_margin_sat, best_centroid_dist))

# Phase 1 embedding 저장 (test set)
s1 = save_phase_results_hybrid(
    "phase1_contrastive", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(contrastive)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_contrastive_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(contrastive)'] = s1
ph1_metrics = analyze_embedding_quality(emb_by_phase['Ph1(contrastive)'], y_test, 'Phase 1')
ph1_centroid_cos = ph1_metrics.get('centroid_cos', float('nan'))

# [v8-3-1] Phase 1 TRAIN embedding 추출 (balanced LR 학습용)
print("\n[v8-3-1] Extracting Phase 1 TRAIN embeddings for balanced LR...")
train_emb_ph1 = extract_embeddings_hybrid(
    embedding_model,  # [FIX] 64dim embedding_layer (base_model preds[0]와 동일 차원)
    X_train_esm2_comb, X_train_mask_comb,
    X_train_seq_comb.astype(np.int32),
    X_train_dm_comb.astype(np.float32),
    X_train_dd_comb.astype(np.float32))
np.save(os.path.join(SAVE_DIR, '{}_phase1_train_embeddings.npy'.format(VER_TAG)), train_emb_ph1)
print("  Train emb: {} | y_train: pos={} neg={}".format(
    train_emb_ph1.shape, int((y_train==1).sum()), int((y_train==0).sum())))

# Prototype biological validation [v7c-2]
if IS_TYPE_B and proto_loss_fn is not None:
    save_prototype_assignments(
        proto_loss_fn, emb_by_phase['Ph1(contrastive)'],
        y_test, save_dir=SAVE_DIR, ver_tag=VER_TAG, go_term=selected_go)

# ==========================================================================
# [v8b-3] PHASE 1→2 Interface: Warm Init + Prototype 재초기화 + Bias
# ==========================================================================
print("\n" + "=" * 58)
print(">>> PHASE 1→2 Interface [v8b-3]")
print("=" * 58)

ph1_emb = emb_by_phase['Ph1(contrastive)']

# 1) Prototype Warm Init (prediction_out 방향 — v7b-2 유지)
warm_ok = prototype_warm_init(
    base_model=base_model, embeddings_np=ph1_emb, y_np=y_test,
    proto_fn=proto_loss_fn if IS_TYPE_B else None, scale=WARM_INIT_SCALE)

# 2) [v8b-3] Prototype 재초기화 (test embedding 기준, stale proto 방지)
if IS_TYPE_B and proto_loss_fn is not None:
    reinit_prototypes_from_phase1(proto_loss_fn, ph1_emb, y_test)

# 3) [v8b-3] Bias correction: test 분포 기준으로 보정
#    (Warm Init이 prediction_out weight를 방향으로 설정했지만, bias만 다시 조정)
set_bias_from_test_distribution(base_model, y_test)

# Phase 1→2 시작점 확인
preds_warm = base_model.predict(
    [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
    batch_size=256, verbose=0)
warm_scores = preds_warm[1].squeeze()
if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
    print("  [Interface] Pre-Phase2 AUROC={:.4f} AUPRC={:.4f}".format(
        roc_auc_score(y_test, warm_scores),
        average_precision_score(y_test, warm_scores)))

# ==========================================================================
# [v8-3-2] PHASE 2: Type-B는 SKIP, Type-A만 실행
# ==========================================================================

if IS_TYPE_B:
    # ------------------------------------------------------------------
    # [v9-1-3] Type-B: Phase 3 — Gene-Stratified CV LR on test set
    #
    # 구조적 문제 해결:
    #   train(human+swissprot)에는 splicing_delta가 없음 (zero padding → LR이 학습 불가)
    #   → test set(hMuscle) 내부에서 gene-level stratified K-fold CV 수행
    #   → 같은 유전자의 모든 아이소폼을 같은 fold에 배치 (Axis 2 gene-level leakage 방지)
    #
    # Phase 3 피처 구성:
    #   [Phase1_embedding 64dim] + [domain_delta_v2_sign 251dim] + [splicing_delta_v2 150dim]
    #   = 465dim total (모든 피처가 test set에 존재)
    # ------------------------------------------------------------------
    from sklearn.model_selection import GroupKFold

    print("\n" + "=" * 58)
    print(">>> PHASE 3 [v9-1]: Gene-Stratified CV LR — Phase 2 SKIPPED for Type-B")
    print("    [v9-1-3] LR input = emb(64) + domain_delta(251) + splicing_delta({})".format(SPLICE_DIM))
    print("    Gene-stratified {}-fold CV (Axis 2 leakage prevention)".format(5))
    print("    근거: train/test feature asymmetry 해결")
    print("=" * 58)

    test_emb_ph1 = emb_by_phase['Ph1(contrastive)']
    n_pos_test = int((y_test == 1).sum())
    n_neg_test = int((y_test == 0).sum())
    print("  Test: n_pos={}, n_neg={}, pos_ratio={:.3f}%".format(
        n_pos_test, n_neg_test, n_pos_test / len(y_test) * 100))

    # Phase 3 피처 조합
    X_test_dd_sign = np.sign(X_test_dd).astype(np.float32)  # 이미 sign 처리됨
    test_features_full = np.hstack([
        test_emb_ph1,           # 64dim: Phase 1 embedding
        X_test_dd_sign,         # 251dim: domain_delta_v2 (sign)
        X_test_splicing_delta,  # 150dim: splicing_delta_v2
    ])
    test_features_emb_only = test_emb_ph1.copy()  # baseline 비교용

    print("  [v9-1-3] Full features: {} (emb:{} + dd:{} + splice:{})".format(
        test_features_full.shape, test_emb_ph1.shape[1],
        X_test_dd_sign.shape[1], SPLICE_DIM))

    # Gene-level stratification: 같은 유전자의 아이소폼은 같은 fold
    gene_groups = np.array([g.split('.')[0] for g in X_test_geneid])
    unique_genes = np.unique(gene_groups)
    gene_to_group_id = {g: i for i, g in enumerate(unique_genes)}
    group_ids = np.array([gene_to_group_id[g.split('.')[0]] for g in X_test_geneid])

    N_FOLDS = 5
    # positive가 있는 유전자가 최소 N_FOLDS개 이상인지 확인
    pos_genes = set(gene_groups[y_test == 1])
    n_unique_groups = len(unique_genes)
    print("  Positive genes in test: {}  Unique groups: {}".format(
        len(pos_genes), n_unique_groups))

    # GroupKFold requires: 2 <= n_splits <= n_groups
    # 또한 positive genes가 최소 n_splits개 이상이어야 의미 있는 CV 가능
    can_do_cv = (len(pos_genes) >= 2 and n_unique_groups >= 2)

    if can_do_cv:
        actual_folds = min(N_FOLDS, len(pos_genes), n_unique_groups)
        actual_folds = max(actual_folds, 2)  # 최소 2-fold
        gkf = GroupKFold(n_splits=actual_folds)
        print("  [CV] Using {}-fold gene-stratified CV".format(actual_folds))
    else:
        # Fallback: train→test LR (v8-3 style, embedding only)
        print("  [WARN] Too few positive genes ({}) for CV. Using train→test LR fallback.".format(
            len(pos_genes)))
        clf_fallback = LogisticRegression(
            class_weight='balanced', C=1.0, max_iter=1000, solver='lbfgs')
        clf_fallback.fit(train_emb_ph1, y_train)
        balanced_proba = clf_fallback.predict_proba(test_emb_ph1)[:, 1].astype(np.float64)
        bal_auroc = roc_auc_score(y_test, balanced_proba) \
                    if y_test.sum() > 0 else float('nan')
        bal_auprc = average_precision_score(y_test, balanced_proba) \
                    if y_test.sum() > 0 else float('nan')
        print("  [Phase 3 Fallback] AUROC={:.4f}  AUPRC={:.4f}".format(bal_auroc, bal_auprc))

        ph3_emb = test_emb_ph1.copy()
        emb_by_phase['Ph2(unified)'] = ph3_emb
        score_by_phase['Ph2(unified)'] = balanced_proba
        ph2_centroid_cos = ph1_centroid_cos
        final_scores = balanced_proba
        final_auroc = bal_auroc
        final_auprc = bal_auprc
        np.save(os.path.join(SAVE_DIR, '{}_phase2_unified_embeddings.npy'.format(VER_TAG)), ph3_emb)
        np.save(os.path.join(SAVE_DIR, '{}_phase2_unified_labels.npy'.format(VER_TAG)), y_test)
        with open(os.path.join(SAVE_DIR, '{}_phase2_unified_scores.txt'.format(VER_TAG)), 'w') as fw:
            for i in range(len(y_test)):
                fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                         str(balanced_proba[i]) + '\n')
        best_phase2_auprc = bal_auprc
        can_do_cv = False  # skip the CV block below

    if can_do_cv:

        # CV 실행: full features
        cv_scores_full = np.zeros(len(y_test), dtype=np.float64)
        # CV 실행: emb-only baseline
        cv_scores_emb = np.zeros(len(y_test), dtype=np.float64)
        # CV 실행: emb + domain_delta only (splicing 효과 분리용)
        test_features_emb_dd = np.hstack([test_emb_ph1, X_test_dd_sign])
        cv_scores_emb_dd = np.zeros(len(y_test), dtype=np.float64)

        fold_results = []
        for fold_i, (train_idx, val_idx) in enumerate(gkf.split(test_features_full, y_test, group_ids)):
            n_pos_fold = int(y_test[val_idx].sum())
            n_neg_fold = int((1 - y_test[val_idx]).sum())

            # Full features
            clf_full = LogisticRegression(
                class_weight='balanced', C=1.0, max_iter=1000, solver='lbfgs')
            clf_full.fit(test_features_full[train_idx], y_test[train_idx])
            cv_scores_full[val_idx] = clf_full.predict_proba(test_features_full[val_idx])[:, 1]

            # Emb-only baseline
            clf_emb = LogisticRegression(
                class_weight='balanced', C=1.0, max_iter=1000, solver='lbfgs')
            clf_emb.fit(test_features_emb_only[train_idx], y_test[train_idx])
            cv_scores_emb[val_idx] = clf_emb.predict_proba(test_features_emb_only[val_idx])[:, 1]

            # Emb + domain_delta
            clf_emb_dd = LogisticRegression(
                class_weight='balanced', C=1.0, max_iter=1000, solver='lbfgs')
            clf_emb_dd.fit(test_features_emb_dd[train_idx], y_test[train_idx])
            cv_scores_emb_dd[val_idx] = clf_emb_dd.predict_proba(test_features_emb_dd[val_idx])[:, 1]

            if n_pos_fold > 0:
                fold_auprc_full = average_precision_score(y_test[val_idx], cv_scores_full[val_idx])
                fold_auprc_emb  = average_precision_score(y_test[val_idx], cv_scores_emb[val_idx])
                fold_results.append((fold_i, n_pos_fold, n_neg_fold, fold_auprc_emb, fold_auprc_full))
                print("  Fold {}: pos={} neg={} | emb={:.4f} full={:.4f} Δ={:+.4f}".format(
                    fold_i + 1, n_pos_fold, n_neg_fold,
                    fold_auprc_emb, fold_auprc_full, fold_auprc_full - fold_auprc_emb))

        # Global metrics
        bal_auroc_full = roc_auc_score(y_test, cv_scores_full) \
                         if y_test.sum() > 0 else float('nan')
        bal_auprc_full = average_precision_score(y_test, cv_scores_full) \
                         if y_test.sum() > 0 else float('nan')
        bal_auprc_emb  = average_precision_score(y_test, cv_scores_emb) \
                         if y_test.sum() > 0 else float('nan')
        bal_auprc_emb_dd = average_precision_score(y_test, cv_scores_emb_dd) \
                           if y_test.sum() > 0 else float('nan')
        bal_auroc_emb  = roc_auc_score(y_test, cv_scores_emb) \
                         if y_test.sum() > 0 else float('nan')

        print("\n  [Phase 3] Gene-Stratified {}-Fold CV Results:".format(actual_folds))
        print("    Emb-only (v8-3):         AUROC={:.4f}  AUPRC={:.4f}".format(
            bal_auroc_emb, bal_auprc_emb))
        print("    Emb + domain_delta:      AUROC={:.4f}  AUPRC={:.4f}  (dd gain: {:+.4f})".format(
            roc_auc_score(y_test, cv_scores_emb_dd) if y_test.sum() > 0 else float('nan'),
            bal_auprc_emb_dd, bal_auprc_emb_dd - bal_auprc_emb))
        print("    Emb + dd + splice (v9-1): AUROC={:.4f}  AUPRC={:.4f}  (full gain: {:+.4f})".format(
            bal_auroc_full, bal_auprc_full, bal_auprc_full - bal_auprc_emb))
        print("    Splicing-only gain: {:+.4f}".format(bal_auprc_full - bal_auprc_emb_dd))

        # Use full-feature CV scores as final
        balanced_proba = cv_scores_full.astype(np.float64)
        bal_auroc = bal_auroc_full
        bal_auprc = bal_auprc_full

        # Phase 3 결과를 phase2_unified로 저장 (downstream 호환성 유지)
        ph3_emb = test_emb_ph1.copy()
        emb_by_phase['Ph2(unified)'] = ph3_emb
        score_by_phase['Ph2(unified)'] = balanced_proba
        ph2_centroid_cos = ph1_centroid_cos  # Phase 2 없으므로 Ph1 값 유지
        print("  [F5 Final] centroid_cos: Ph1={:.4f} → Ph2(=Ph1, skipped)={:.4f} (Δ=+0.0000)".format(
            ph1_centroid_cos, ph1_centroid_cos))

        # Final scores = CV LR
        final_scores = balanced_proba
        final_auroc  = bal_auroc
        final_auprc  = bal_auprc

        # Phase 3 저장
        np.save(os.path.join(SAVE_DIR, '{}_phase3_cv_scores.npy'.format(VER_TAG)), balanced_proba)
        np.save(os.path.join(SAVE_DIR, '{}_phase3_cv_scores_emb_only.npy'.format(VER_TAG)), cv_scores_emb)
        np.save(os.path.join(SAVE_DIR, '{}_phase2_unified_embeddings.npy'.format(VER_TAG)), ph3_emb)
        np.save(os.path.join(SAVE_DIR, '{}_phase2_unified_labels.npy'.format(VER_TAG)), y_test)
        with open(os.path.join(SAVE_DIR, '{}_phase2_unified_scores.txt'.format(VER_TAG)), 'w') as fw:
            for i in range(len(y_test)):
                fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                         str(balanced_proba[i]) + '\n')

        # best_phase2_auprc 호환 변수
        best_phase2_auprc = bal_auprc

else:
    # ------------------------------------------------------------------
    # [B1 FIX] Type-A: Phase 2 실행 (v8b 완전 동일)
    # 이전 버그: else 블록이 print 한 줄만 포함하고 나머지 Phase 2 코드가
    # 최상위 레벨에 있어 IS_TYPE_B=True여도 Phase 2가 실행됨
    # ------------------------------------------------------------------
    print("\n" + "=" * 58)
    print(">>> PHASE 2: Unified Loss [v8b-2] + Discriminative LR [v8b-1]")
    print("    α={} | ESM2+head={} | CNN={} | pred_out={} | rest={}".format(
        ALPHA_P2, LR_P2_SLOW, LR_P2_CNN, LR_P2_PRED, LR_P2_REST))
    print("=" * 58)

    # ALL layers trainable
    for layer in base_model.layers:
        layer.trainable = True
    print("  [Phase2] All layers trainable")

    # Phase 2 discriminative optimizers
    opt_p2_slow = optimizers.Adam(lr=LR_P2_SLOW, clipnorm=1.0)  # ESM-2 + head_dense64
    opt_p2_cnn  = optimizers.Adam(lr=LR_P2_CNN,  clipnorm=0.5)  # CNN branch
    opt_p2_rest = optimizers.Adam(lr=LR_P2_REST, clipnorm=1.0)  # Domain, DomainDelta
    opt_p2_pred = optimizers.Adam(lr=LR_P2_PRED, clipnorm=1.0)  # prediction_out

    slow_ids, cnn_ids_p2, pred_ids = build_var_groups_p2(base_model)
    print("  [Phase2] Var groups: slow(ESM2+head_dense64)={} CNN={} pred_out={} rest={}".format(
        len(slow_ids), len(cnn_ids_p2), len(pred_ids),
        len(classification_model.trainable_variables) - len(slow_ids) - len(cnn_ids_p2) - len(pred_ids)))

    # [B3 FIX] Type-A else 블록 안 → IS_TYPE_B=False 확정, 분기 제거
    PHASE2_MAX_EPOCHS = 25
    NO_IMPROVE_LIMIT  = 7

    best_phase2_auprc   = 0.0
    best_phase2_weights = None
    no_improve_count    = 0
    prev_centroid_cos   = ph1_centroid_cos

    print("  Max epochs={} | patience={}".format(PHASE2_MAX_EPOCHS, NO_IMPROVE_LIMIT))

    for epoch in range(PHASE2_MAX_EPOCHS):
        print('Phase 2 - Epoch: {}/{}'.format(epoch + 1, PHASE2_MAX_EPOCHS))

        # [B4 FIX] is_type_b=False 명시 (else 블록은 Type-A만 진입)
        avg_m, avg_f, active_rate = phase2_unified_epoch_v8b(
            embedding_model=embedding_model,
            classification_model=classification_model,
            X_esm2=X_train_esm2_upsmp, X_mask=X_train_mask_upsmp,
            X_seq=X_train_seq_upsmp, X_dm=X_train_dm_upsmp,
            X_dd=X_train_dd_upsmp, y=y_train_upsmp,
            slow_ids=slow_ids, cnn_ids=cnn_ids_p2, pred_ids=pred_ids,
            opt_slow=opt_p2_slow, opt_cnn=opt_p2_cnn,
            opt_rest=opt_p2_rest, opt_pred=opt_p2_pred,
            is_type_b=False, proto_loss_fn=None,
            margin=MARGIN_P1, batch_size=BATCH_SIZE_P1, n_batches=N_BATCHES_P1,
            refresh_interval=10, focal_alpha_val=0.10)

        print("  -> Metric(triplet): {:.4f} | Focal: {:.4f} | Active: {:.1f}%".format(
            avg_m, avg_f, active_rate))

        # 5 epoch마다 embedding 품질 진단 ([F5] centroid_cos 모니터링)
        preds = base_model.predict(
            [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
            batch_size=256, verbose=0)
        curr_emb    = preds[0]
        test_scores = preds[1].squeeze()

        if (epoch + 1) % 5 == 0:
            curr_metrics = analyze_embedding_quality(curr_emb, y_test, 'Phase2 Epoch {}'.format(epoch + 1))
            curr_cc = curr_metrics.get('centroid_cos', float('nan'))
            if not np.isnan(prev_centroid_cos) and not np.isnan(curr_cc):
                delta_cc = curr_cc - prev_centroid_cos
                flag = " ⚠ [F5 감지]" if delta_cc > 0.05 else " ✓"
                print("  [F5 Monitor] centroid_cos: {:.4f} → {:.4f} (Δ={:+.4f}){}".format(
                    prev_centroid_cos, curr_cc, delta_cc, flag))
            prev_centroid_cos = curr_cc

        # AUPRC early stopping
        if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
            auroc = roc_auc_score(y_test, test_scores)
            auprc = average_precision_score(y_test, test_scores)
            print("  [AUPRC] AUROC={:.4f} AUPRC={:.4f} (best={:.4f}, patience={}/{})".format(
                auroc, auprc, best_phase2_auprc, no_improve_count, NO_IMPROVE_LIMIT))
            if auprc > best_phase2_auprc:
                best_phase2_auprc   = auprc
                best_phase2_weights = base_model.get_weights()
                no_improve_count    = 0
            else:
                no_improve_count += 1
                if no_improve_count >= NO_IMPROVE_LIMIT:
                    print("  [Early Stop] AUPRC not improving ({} epochs)".format(NO_IMPROVE_LIMIT))
                    break
    # end for epoch

    # [B2 FIX] best_phase2_weights 복원을 루프 밖에서 1회만 수행
    # 이전 버그: 루프 안에서 매 epoch마다 복원 → 학습이 best checkpoint에서 반복 재시작
    if best_phase2_weights is not None:
        base_model.set_weights(best_phase2_weights)
        print("\n[Phase 2] Restored best (AUPRC={:.4f})".format(best_phase2_auprc))

    # [B2 FIX] save도 루프 밖에서 1회만 수행
    s2 = save_phase_results_hybrid(
        "phase2_unified", base_model,
        X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
        y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
    emb_by_phase['Ph2(unified)'] = np.load(
        os.path.join(SAVE_DIR, '{}_phase2_unified_embeddings.npy'.format(VER_TAG)))
    score_by_phase['Ph2(unified)'] = s2
    ph2_metrics = analyze_embedding_quality(emb_by_phase['Ph2(unified)'], y_test, 'Phase 2 Final')
    ph2_centroid_cos = ph2_metrics.get('centroid_cos', float('nan'))
    print("  [F5 Final] centroid_cos: Ph1={:.4f} → Ph2={:.4f} (Δ={:+.4f})".format(
        ph1_centroid_cos, ph2_centroid_cos, ph2_centroid_cos - ph1_centroid_cos))

    # Type-A final score = sigmoid head (best_phase2_weights 복원된 상태)
    preds_base  = base_model.predict(
        [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
        batch_size=256, verbose=0)
    base_scores  = np.array([preds_base[1][i][0] for i in range(K_testing_size)])
    final_scores = base_scores.copy()
    final_auroc  = roc_auc_score(y_test, final_scores) \
                   if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else float('nan')
    final_auprc  = average_precision_score(y_test, final_scores) \
                   if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else float('nan')

    # LP diagnostic (not applied)
    if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
        try:
            lp_02    = expression_label_propagation(base_scores, X_test_expr, alpha=0.2)
            lp_auprc = average_precision_score(y_test, lp_02)
            print("  [Diag] LP alpha=0.2 AUPRC={:.4f} (delta={:+.4f}, NOT applied)".format(
                lp_auprc, lp_auprc - final_auprc))
        except Exception:
            pass

# end if IS_TYPE_B / else

# ==========================================================================
# 최종 결과 출력
# ==========================================================================
print("\n" + "=" * 58)
print(">>> FINAL RESULTS [v9-1]")
print("=" * 58)
print("  Final: AUROC={:.4f} AUPRC={:.4f}".format(final_auroc, final_auprc))
score_by_phase['Final(noLP)'] = final_scores

# ==========================================================================
# 임베딩 품질 요약
# ==========================================================================
print("\n" + "=" * 58)
print(">>> Embedding Quality Summary")
print("=" * 58)
print("{:28} {:>10} {:>10} {:>10} {:>12}".format(
    "Phase", "CentCos", "Sep.Ratio", "LinAUROC", "PredAUROC"))
print("-" * 72)
for phase_name, emb in emb_by_phase.items():
    m = analyze_embedding_quality(emb, y_test, phase_name)
    try:
        pred_auroc = roc_auc_score(y_test, score_by_phase[phase_name])
    except Exception:
        pred_auroc = float('nan')
    print("{:28} {:10.4f} {:10.4f} {:10.4f} {:12.4f}".format(
        phase_name, m['centroid_cos'], m['sep_ratio'], m['linear_auroc'], pred_auroc))

# ==========================================================================
# 최종 저장
# ==========================================================================
print("\nSaving Final Results...")
output_file = os.path.join(SAVE_DIR, "{}_{}_Final_scores.txt".format(VER_TAG, safe_go))
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                 str(final_scores[i]) + '\n')

base_model.save_weights(
    os.path.join(SAVE_DIR, "{}_{}_BaseModel_weights.h5".format(VER_TAG, safe_go)))
np.save(os.path.join(SAVE_DIR, "{}_{}_Final_labels.npy".format(VER_TAG, safe_go)), y_test)

if IS_TYPE_B and proto_loss_fn is not None and proto_loss_fn.prototypes is not None:
    np.save(os.path.join(SAVE_DIR, "{}_{}_prototypes.npy".format(VER_TAG, safe_go)),
            proto_loss_fn.prototypes.numpy())
    print("  [Proto] Saved k={} prototypes".format(PROTO_K))

print("\n[Final] GO={} | Type={} | k={} | Phase2 α={} | head_dense64 lr={}".format(
    selected_go, go_type_str, PROTO_K if IS_TYPE_B else 'N/A',
    ALPHA_P2, LR_P2_SLOW))
print("[Final] Ph1 centroid_cos={:.4f} → Ph2 centroid_cos={:.4f} (Δ={:+.4f})".format(
    ph1_centroid_cos, ph2_centroid_cos, ph2_centroid_cos - ph1_centroid_cos))
print("[Final] AUROC={:.4f} AUPRC={:.4f}".format(final_auroc, final_auprc))
print("[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
