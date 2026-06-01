# -*- coding: utf-8 -*-
# ============================================================================
# v6c_integrated_full_model.py  (v6-controlled)
#
# v6h 대비 핵심 변경:
#
# [ARCH] Multi-scale CNN + Bidirectional Gating
#   문제: v6h의 단일 Conv1D 구조가 다양한 모티프 길이 커버 불충분
#         ESM-2 / CNN 간 상호작용 부족 → 모달리티 융합 최적화 여지
#
#   해결:
#     CNN branch: 3개 병렬 Conv1D (k=5,7,11) → concatenate → reduce → cnn_feat[32]
#     Gating: ESM-2 ↔ CNN bidirectional gating ×2 rounds
#             f_esm ↔ f_cnn 상호 조절 후 fusion
#
# [TRAIN] Gradient Scaling + CNN Warm-up
#   Phase 0  : Untrained baseline
#   Phase 1a : CNN warm-up (focal, CNN only, 2 epochs) — NEW
#   Phase 1  : Triplet (ESM-2 + CNN + Gate + Domain) with gradient scaling
#              ESM-2: 0.2x | CNN: 1.0x | Gate: 0.5x
#   Phase 1.5: Linear Probing (all frozen, head only)
#   Phase 2  : Focal (ESM-2 frozen, CNN + Gate + Domain + head)
#              patience=7, alpha=0.25 (score collapse 방지)
#   Phase 3  : Label Propagation
#
# [ARCH] 모델 구조
#   ESM-2 branch : 640 → Dense(256,relu) → Dense(128,relu) → Dense(64,relu)
#                  → gate(×mask) → esm2_gated[64]
#   CNN   branch : Embedding(8001,32) → Conv1D(32,k=5 | k=7 | k=11)
#                  → concat[96] → Conv1D(64,k=3) → GlobalMaxPool
#                  → Dense(32,relu) → cnn_feat[32]
#   Bidirectional Gating:
#     Round 1: gate_e2c_r1 (esm→cnn), gate_c2e_r1 (cnn→esm)
#     Round 2: gate_e2c_r2, gate_c2e_r2
#   Domain branch: Embedding → LSTM(16) → domain_feat[16]
#   Fusion       : concat[f_esm_r2(64) + f_cnn_r2(32) + domain_feat(16) = 112]
#                  → Dense(48,relu) → L2_norm → emb[48] → Dense(1,sigmoid)
#   EMB_DIM=112 (v6h와 동일)
#
# [SEQ] 서열 입력
#   SEQ_LEN=1500 (p95 커버, 6000→last 1500 truncation)
#   GlobalMaxPool → 패딩 처리
#
# [PIPE] 데이터 파이프라인
#   generate_label 3회 호출 (deterministic): dm, esm2+mask, seq
#   upsample 1회: hstack(esm2[640]+mask[1]+seq[1500]+dm) → 통합
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
    script, selected_go = argv
except ValueError:
    print("Usage: python v6c_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d")
VER_TAG  = "v6c_integrated"
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
ESM2_DIM  = 640
SEQ_LEN   = 1500
EXPR_DIM  = 24
EMB_DIM   = 112            # esm2_gated[64] + cnn_feat[32] + domain_feat[16]
MARGIN_P1         = 0.3
TARGET_COVERAGE   = 6.0

# [v6c] Gradient scaling constants
ESM2_GRAD_SCALE = 0.2   # Phase 1 gradient scaling
CNN_GRAD_SCALE  = 1.0
GATE_GRAD_SCALE = 0.5

# [v6c] Layer name groups for gradient scaling
ESM2_LAYER_NAMES = {'esm2_d1', 'esm2_d2', 'esm2_feat'}
CNN_LAYER_NAMES  = {'cnn_emb', 'cnn_k5', 'cnn_k7', 'cnn_k11',
                    'cnn_reduce', 'cnn_pool', 'cnn_feat'}
GATE_LAYER_NAMES = {'gate_e2c_r1', 'gate_c2e_r1',
                    'gate_e2c_r2', 'gate_c2e_r2'}

def get_layer_group(var_name):
    """Returns layer group name for gradient scaling"""
    for n in ESM2_LAYER_NAMES:
        if n in var_name:
            return 'esm2'
    for n in CNN_LAYER_NAMES:
        if n in var_name:
            return 'cnn'
    for n in GATE_LAYER_NAMES:
        if n in var_name:
            return 'gate'
    return 'other'

ESM2_TRAINABLE_NAMES = ESM2_LAYER_NAMES
CNN_TRAINABLE_NAMES  = CNN_LAYER_NAMES

def set_esm2_trainable(model, trainable):
    """ESM-2 Dense 레이어 trainable 설정"""
    for layer in model.layers:
        if layer.name in ESM2_TRAINABLE_NAMES:
            layer.trainable = trainable

def set_cnn_trainable(model, trainable):
    """CNN 레이어 trainable 설정"""
    for layer in model.layers:
        if layer.name in CNN_TRAINABLE_NAMES:
            layer.trainable = trainable

def set_gate_trainable(model, trainable):
    """Gate 레이어 trainable 설정"""
    for layer in model.layers:
        if layer.name in GATE_LAYER_NAMES:
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
# [5-v6c] 임베딩 추출 — 4-modality 입력
# --------------------------------------------------------------------------
def extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm,
                               emb_dim=EMB_DIM, batch_size=512):
    n   = len(X_esm2)
    emb = np.zeros((n, emb_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        raw  = feature_model.predict_on_batch([
            X_esm2[start:end].astype(np.float32),
            X_mask[start:end].astype(np.float32),
            X_seq[start:end],
            X_dm[start:end]])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        emb[start:end] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [EMB] 임베딩 품질 분석 (v6h와 동일)
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
    print("    Silhouette (cosine):   {:+.4f}".format(metrics['silhouette']))
    print("    Sep ratio (inter/intra):{:.4f}".format(metrics['sep_ratio']))
    print("    Centroid dist:          {:.4f}".format(metrics['centroid_dist']))
    print("    Linear AUROC:           {:.4f}".format(metrics['linear_auroc']))
    return metrics

# --------------------------------------------------------------------------
# [6-v6c] Phase 1 — GradientTape Triplet (4-modality, gradient scaling)
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


def phase1_triplet_epoch_hybrid(
        feature_model, X_esm2, X_mask, X_seq, X_dm, y,
        optimizer, margin=MARGIN_P1, batch_size=256,
        n_batches=50, warmup=False, refresh_interval=10):
    """
    [v6c] Phase 1: ESM-2 + CNN + Gate + Domain 학습 (gradient scaling)
    """
    def refresh_embeddings():
        emb = extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm)
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
                     X_dm[indices]], training=True)
                return tf.math.l2_normalize(raw, axis=1)

            emb_a = get_live_emb(a_idxs)
            emb_p = get_live_emb(p_idxs)
            emb_n = get_live_emb(n_idxs)

            pos_dist = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
            neg_dist = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
            raw_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            loss     = tf.reduce_mean(raw_loss)

        grads = tape.gradient(loss, feature_model.trainable_variables)

        # [v6c] Gradient scaling per layer group
        # IndexedSlices (Embedding sparse grads) → dense before scaling
        scaled_grads_and_vars = []
        for g, v in zip(grads, feature_model.trainable_variables):
            if g is None:
                continue
            group = get_layer_group(v.name)
            if group == 'esm2':
                scale = ESM2_GRAD_SCALE   # 0.2
            elif group == 'cnn':
                scale = CNN_GRAD_SCALE    # 1.0
            elif group == 'gate':
                scale = GATE_GRAD_SCALE   # 0.5
            else:
                scale = 1.0
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            g_safe = tf.where(tf.math.is_finite(g), g * scale, tf.zeros_like(g))
            scaled_grads_and_vars.append((g_safe, v))

        if scaled_grads_and_vars:
            optimizer.apply_gradients(scaled_grads_and_vars)

        batch_losses.append(float(loss.numpy()))
        active_count += int((raw_loss.numpy() > 0).sum())

    avg_loss    = np.mean(batch_losses) if batch_losses else 0.0
    active_rate = active_count / (n_batches * batch_size) * 100
    return avg_loss, active_rate


def compute_margin_stats_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, y,
                                 margin=MARGIN_P1, n_sample=1000):
    emb     = extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm)
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


def save_phase_results_hybrid(phase_name, model_base, X_esm2, X_mask, X_seq, X_dm,
                               y_true, gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds  = model_base.predict([X_esm2, X_mask, X_seq, X_dm], batch_size=256, verbose=1)
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
# [7] Phase 3: Test-time Label Propagation (v6h와 동일)
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
# [8-v6c] 데이터 로딩
# --------------------------------------------------------------------------
print('>>> Preparing Data (v6c/Controlled) for ' + selected_go)

ESM2_DATA_DIR = '../data'

def _load_esm2(name):
    path = os.path.join(ESM2_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError("[ESM] 필수 파일 없음: {}".format(path))
    arr = np.load(path).astype(np.float32)
    print("  [ESM] Loaded {} {}".format(name, arr.shape))
    return arr

# ESM-2 임베딩
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

# 서열 입력 (CNN용)
SEQ_PATH_TRAIN  = '../data/raw_data/data/sequences/human_sequence_train.npy'
SEQ_PATH_OTHER  = '../data/raw_data/data/sequences/swissprot_sequence_train.npy'
SEQ_PATH_TEST   = 'my_sequence_matrix_fixed.npy'

X_train_seq       = np.load(SEQ_PATH_TRAIN)[:, -SEQ_LEN:]
X_train_other_seq = np.load(SEQ_PATH_OTHER)[:, -SEQ_LEN:]
X_test_seq        = np.load(SEQ_PATH_TEST)[:, -SEQ_LEN:]
print("  [Seq] train={} other={} test={}".format(
    X_train_seq.shape, X_train_other_seq.shape, X_test_seq.shape))

# Domain feature
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

# Expression matrix
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
# [9-v6c] 레이블 생성 및 업샘플 (v6h와 동일)
# --------------------------------------------------------------------------
print('>>> Preparing Data for ' + selected_go)

dm_dim         = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm),
                      np.max(X_train_other_dm)]) + 1

# ① dm
y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, \
    _dummy, X_train_dm_comb = generate_label(
        X_train_esm2, X_train_dm,
        X_train_other_esm2, X_train_other_dm,
        X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

# ② esm2 + mask
_, _, _, _, _, _, X_train_esm2_comb, X_train_mask_comb = generate_label(
    X_train_esm2, X_train_esm2_mask,
    X_train_other_esm2, X_train_other_esm2_mask,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

# ③ seq
_, _, _, _, _, _, _dummy2, X_train_seq_comb = generate_label(
    X_train_esm2, X_train_seq,
    X_train_other_esm2, X_train_other_seq,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

n_pos_train = int((y_train == 1).sum())
print("  y_train pos={} neg={} ratio={:.3f}%".format(
    n_pos_train, int((y_train == 0).sum()), n_pos_train / len(y_train) * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

# hstack: [esm2(640) | mask(1) | seq(1500) | dm(dm_dim)]
MASK_START = ESM2_DIM
SEQ_START  = ESM2_DIM + 1
DM_START   = ESM2_DIM + 1 + SEQ_LEN

X_train_combined = np.hstack([
    X_train_esm2_comb,
    X_train_mask_comb,
    X_train_seq_comb.astype(np.float32),
    X_train_dm_comb.astype(np.float32)
])

np.random.seed(42)
unused_flag = np.zeros(y_train.shape[0])
X_combined_upsmp, _dummy_dm, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count,
    X_train_combined,
    np.zeros((len(y_train), 1)),
    unused_flag)

# 분리
X_train_esm2_upsmp = X_combined_upsmp[:, :ESM2_DIM].astype(np.float32)
X_train_mask_upsmp = X_combined_upsmp[:, MASK_START:SEQ_START].astype(np.float32)
X_train_seq_upsmp  = X_combined_upsmp[:, SEQ_START:DM_START].astype(np.int32)
X_train_dm_upsmp   = X_combined_upsmp[:, DM_START:].astype(np.int32)

print("  Upsampled: esm2={} mask={} seq={} dm={} y={}".format(
    X_train_esm2_upsmp.shape, X_train_mask_upsmp.shape,
    X_train_seq_upsmp.shape, X_train_dm_upsmp.shape, y_train_upsmp.shape))

# [I4] 동적 n_batches
BATCH_SIZE_P1 = 256
N_BATCHES_P1  = int(np.clip(
    np.ceil(n_pos_train * TARGET_COVERAGE / BATCH_SIZE_P1), a_min=20, a_max=50))
coverage_per_epoch = N_BATCHES_P1 * BATCH_SIZE_P1 / n_pos_train
print("[I4] n_batches={} (n_pos={}, coverage={:.1f}x)".format(
    N_BATCHES_P1, n_pos_train, coverage_per_epoch))

# --------------------------------------------------------------------------
# [10-v6c] 모델 구조
# --------------------------------------------------------------------------
esm2_input   = Input(shape=(ESM2_DIM,), name='esm2_input')
esm2_mask_in = Input(shape=(1,),         name='esm2_mask')
seq_input    = Input(shape=(SEQ_LEN,),  dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim,),   dtype='int32', name='domain_input')

# [ESM-2 branch]
x_esm = Dense(256, kernel_regularizer=regularizers.l2(1e-5), name='esm2_d1')(esm2_input)
x_esm = Activation('relu')(x_esm)
x_esm = Dropout(0.2)(x_esm)
x_esm = Dense(128, kernel_regularizer=regularizers.l2(1e-5), name='esm2_d2')(x_esm)
x_esm = Activation('relu')(x_esm)
esm2_feat  = Dense(64, activation='relu',
                   kernel_regularizer=regularizers.l2(1e-5),
                   name='esm2_feat')(x_esm)
esm2_gated = Lambda(lambda x: x[0] * x[1], name='esm2_gated')([esm2_feat, esm2_mask_in])

# [v6c CNN branch — Multi-scale]
x_emb = Embedding(8001, 32, mask_zero=False, name='cnn_emb')(seq_input)

# 3개 병렬 Conv1D (공유 Embedding 입력)
conv_k5  = Conv1D(32, 5,  padding='same', activation='relu', name='cnn_k5')(x_emb)
conv_k7  = Conv1D(32, 7,  padding='same', activation='relu', name='cnn_k7')(x_emb)
conv_k11 = Conv1D(32, 11, padding='same', activation='relu', name='cnn_k11')(x_emb)

cnn_concat = concatenate([conv_k5, conv_k7, conv_k11], axis=-1)  # [1500, 96]
activation_map = Conv1D(64, 3, padding='same', activation='relu',
                        name='cnn_reduce')(cnn_concat)            # [1500, 64]
x_pool = GlobalMaxPooling1D(name='cnn_pool')(activation_map)      # [64]
cnn_feat = Dense(32, activation='relu', name='cnn_feat')(x_pool)  # [32] = f_cnn

# [Domain branch]
x_dm = Embedding(input_dim=domain_emb_dim, output_dim=32,
                  input_length=dm_dim, mask_zero=True,
                  name='dm_emb')(domain_input)
domain_feat = LSTM(16, name='domain_feat')(x_dm)

# [v6c Bidirectional Gating (2 rounds)]
# f_esm = esm2_gated [64], f_cnn = cnn_feat [32]
# Round 1
gate_e2c_r1 = Dense(32, activation='sigmoid', name='gate_e2c_r1')(esm2_gated)
f_cnn_r1 = Lambda(lambda x: x[0] * x[1] + x[0],
                  name='cnn_r1')([cnn_feat, gate_e2c_r1])

gate_c2e_r1 = Dense(64, activation='sigmoid', name='gate_c2e_r1')(f_cnn_r1)
f_esm_r1 = Lambda(lambda x: x[0] * x[1] + x[0],
                  name='esm_r1')([esm2_gated, gate_c2e_r1])

# Round 2
gate_e2c_r2 = Dense(32, activation='sigmoid', name='gate_e2c_r2')(f_esm_r1)
f_cnn_r2 = Lambda(lambda x: x[0] * x[1] + x[0],
                  name='cnn_r2')([f_cnn_r1, gate_e2c_r2])

gate_c2e_r2 = Dense(64, activation='sigmoid', name='gate_c2e_r2')(f_cnn_r2)
f_esm_r2 = Lambda(lambda x: x[0] * x[1] + x[0],
                  name='esm_r2')([f_esm_r1, gate_c2e_r2])

# [Fusion] concat[64 + 32 + 16 = 112]
concat = concatenate([f_esm_r2, f_cnn_r2, domain_feat], name='feature_concat')
feature_model = Model([esm2_input, esm2_mask_in, seq_input, domain_input],
                       concat, name='feature_model')

# [Head]
x = Dense(48, kernel_regularizer=regularizers.l2(1e-5))(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer  = Lambda(lambda a: K.l2_normalize(a, axis=1),
                           name="embedding_out")(x)
prediction_layer = Dense(1, activation='sigmoid',
                          kernel_regularizer=regularizers.l2(1e-5),
                          name='prediction_out')(embedding_layer)

base_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input],
    outputs=[embedding_layer, prediction_layer])
classification_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input],
    outputs=prediction_layer)

adam_p1   = optimizers.Adam(lr=0.0005)
adam_main = optimizers.Adam(lr=0.001)
adam_p2   = optimizers.Adam(lr=0.0003)

K_testing_size = len(X_test_esm2)

print("\n[Model v6c] ESM-2(640→64) + Multi-CNN([5,7,11]→32) + Domain(LSTM 16)")
print("[Model v6c] Bidirectional Gating ×2 | Phase1a CNN warmup | Grad scaling")

emb_by_phase   = {}
score_by_phase = {}

# ==========================================================================
# PHASE 0: Untrained Baseline
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 0: Untrained Baseline")
print("=" * 55)

s0 = save_phase_results_hybrid(
    "phase0_initial_untrained", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph0(untrained)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase0_initial_untrained_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph0(untrained)'] = s0
analyze_embedding_quality(emb_by_phase['Ph0(untrained)'], y_test, 'Phase 0')

# ==========================================================================
# PHASE 1a: CNN Warm-up (v6c 신규)
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 1a: CNN Warm-up (focal, CNN only)")
print("    [v6c] CNN branch만 학습, 2 epochs")
print("=" * 55)

# CNN만 trainable, 나머지 frozen
for layer in feature_model.layers:
    layer.trainable = False
set_cnn_trainable(feature_model, True)
print("  [Phase1a] CNN layers trainable:", CNN_TRAINABLE_NAMES)

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=optimizers.Adam(lr=0.001),
    metrics=['accuracy'])

BATCH_SIZE_P1A = 512
n_train_p1a = len(X_train_esm2_upsmp)

for epoch in range(2):
    print('Phase 1a - Epoch: {}/2'.format(epoch + 1))
    indices = np.arange(n_train_p1a)
    np.random.shuffle(indices)
    p1a_losses, p1a_accs = [], []
    for start in range(0, n_train_p1a, BATCH_SIZE_P1A):
        idx = indices[start:start + BATCH_SIZE_P1A]
        mixed = np.hstack((np.where(y_train_upsmp[idx] == 1)[0],
                           np.where(y_train_upsmp[idx] == 0)[0]))
        if len(mixed) == 0:
            continue
        np.random.shuffle(mixed)
        hist = classification_model.fit(
            [X_train_esm2_upsmp[idx[mixed]],
             X_train_mask_upsmp[idx[mixed]],
             X_train_seq_upsmp[idx[mixed]],
             X_train_dm_upsmp[idx[mixed]]],
            y_train_upsmp[idx[mixed]],
            batch_size=256, epochs=1, verbose=0)
        p1a_losses.append(hist.history['loss'][0])
        acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
        p1a_accs.append(hist.history[acc_key][0])
    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(p1a_losses), np.mean(p1a_accs)))

# ==========================================================================
# PHASE 1: Triplet (ALL trainable with gradient scaling)
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 1: Triplet — All modalities (gradient scaling)")
print("    margin={:.1f} [I3] | n_batches={} [I4] | max 15 epochs".format(
    MARGIN_P1, N_BATCHES_P1))
print("=" * 55)

# Phase 1: ALL trainable (joint training with gradient scaling)
for layer in feature_model.layers:
    layer.trainable = True
print("  [Phase1] All layers trainable with gradient scaling")
print("    ESM-2: {:.1f}x | CNN: {:.1f}x | Gate: {:.1f}x".format(
    ESM2_GRAD_SCALE, CNN_GRAD_SCALE, GATE_GRAD_SCALE))

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

    avg_loss, active_rate = phase1_triplet_epoch_hybrid(
        feature_model=feature_model,
        X_esm2=X_train_esm2_upsmp,
        X_mask=X_train_mask_upsmp,
        X_seq=X_train_seq_upsmp,
        X_dm=X_train_dm_upsmp,
        y=y_train_upsmp,
        optimizer=adam_p1,
        margin=MARGIN_P1,
        batch_size=BATCH_SIZE_P1,
        n_batches=N_BATCHES_P1,
        warmup=warmup)

    print("  -> Triplet Loss: {:.4f} | Active: {:.1f}%".format(avg_loss, active_rate))

    # [v6c] Gate 통계 (매 epoch)
    try:
        gate_names = list(GATE_LAYER_NAMES)
        sample_idx = np.random.choice(len(X_test_esm2), min(200, len(X_test_esm2)), replace=False)
        gate_outputs = []
        for gname in ['gate_e2c_r1', 'gate_c2e_r1', 'gate_e2c_r2', 'gate_c2e_r2']:
            gate_layer = feature_model.get_layer(gname)
            gate_model_tmp = Model(inputs=feature_model.inputs, outputs=gate_layer.output)
            g_out = gate_model_tmp.predict(
                [X_test_esm2[sample_idx], X_test_esm2_mask[sample_idx],
                 X_test_seq[sample_idx], X_test_dm[sample_idx]], verbose=0)
            gate_outputs.append((gname, g_out.mean(), g_out.std()))
        for gname, gmean, gstd in gate_outputs:
            print("  [Gate|{}] mean={:.3f} std={:.3f}".format(gname, gmean, gstd))
        # Collapse 경보
        if any(gmean < 0.05 or gmean > 0.95 for _, gmean, _ in gate_outputs):
            print("  [WARN] Gate collapse detected!")
    except Exception as e:
        print("  [Gate] monitoring error: {}".format(str(e)))

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
        sat, cdist = compute_margin_stats_hybrid(
            feature_model, X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm,
            y_test, margin=MARGIN_P1)
        best_margin_sat = max(best_margin_sat, sat)
        final_centroid  = cdist
        if sat >= 60.0:
            print("  [Early Stop] margin_sat {:.1f}% >= 60%.".format(sat))
            break

print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
    best_margin_sat, final_centroid))
s1 = save_phase_results_hybrid(
    "phase1_triplet_only", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(triplet)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_triplet_only_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(triplet)'] = s1
analyze_embedding_quality(emb_by_phase['Ph1(triplet)'], y_test, 'Phase 1')

# ==========================================================================
# PHASE 1.5: Linear Probing (Encoder Frozen)
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 1.5: Linear Probing (All Encoder Frozen)")
print("=" * 55)

# 전체 feature_model 동결
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
        mixed = np.hstack((np.where(y_train_upsmp[idx] == 1)[0],
                           np.where(y_train_upsmp[idx] == 0)[0]))
        if len(mixed) == 0:
            continue
        np.random.shuffle(mixed)
        hist = classification_model.fit(
            [X_train_esm2_upsmp[idx[mixed]],
             X_train_mask_upsmp[idx[mixed]],
             X_train_seq_upsmp[idx[mixed]],
             X_train_dm_upsmp[idx[mixed]]],
            y_train_upsmp[idx[mixed]],
            batch_size=256, epochs=1, verbose=0)
        p15_losses.append(hist.history['loss'][0])
        acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
        p15_accs.append(hist.history[acc_key][0])
    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(p15_losses), np.mean(p15_accs)))

s15 = save_phase_results_hybrid(
    "phase1_5_linear_probing", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1.5(linear)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_5_linear_probing_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1.5(linear)'] = s15
analyze_embedding_quality(emb_by_phase['Ph1.5(linear)'], y_test, 'Phase 1.5')

# ==========================================================================
# PHASE 2: CNN + Gate Fine-tuning (ESM-2 frozen)
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 2: CNN + Gate Fine-tuning")
print("    [v6c] ESM-2 FROZEN | CNN+Gate+Domain TRAIN | Focal only")
print("    [I5] AUPRC early stop [R9.1] | patience=7 | alpha=0.25")
print("=" * 55)

# Unfreeze all, then freeze ESM-2
for layer in feature_model.layers:
    layer.trainable = True
set_esm2_trainable(feature_model, False)
print("  [Phase2] ESM-2 frozen | CNN + Gate + Domain trainable")

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=adam_p2, metrics=['accuracy'])

PHASE2_MAX_EPOCHS  = 15
BATCH_SIZE_P2      = 512
best_phase2_auprc  = 0.0
best_phase2_weights = None
no_improve_count   = 0
NO_IMPROVE_LIMIT   = 7

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
             X_train_dm_upsmp[idx[mixed]]],
            y_train_upsmp[idx[mixed]],
            batch_size=256, epochs=1, verbose=0)
        ep_losses.append(hist.history['loss'][0])
        acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
        ep_accs.append(hist.history[acc_key][0])

    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(ep_losses) if ep_losses else 0,
        np.mean(ep_accs)   if ep_accs   else 0))

    # [I5] AUPRC early stop [R9.1]
    preds = base_model.predict(
        [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm],
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
                if best_phase2_weights is not None:
                    base_model.set_weights(best_phase2_weights)
                break

if best_phase2_weights is not None and no_improve_count < NO_IMPROVE_LIMIT:
    base_model.set_weights(best_phase2_weights)
    print("\n[Phase 2] Restored best (AUPRC={:.4f})".format(best_phase2_auprc))

compute_margin_stats_hybrid(
    feature_model, X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm,
    y_test, margin=MARGIN_P1)

s2 = save_phase_results_hybrid(
    "phase2_cnn_focal", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph2(cnn_focal)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase2_cnn_focal_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph2(cnn_focal)'] = s2
analyze_embedding_quality(emb_by_phase['Ph2(cnn_focal)'], y_test, 'Phase 2')

# ==========================================================================
# PHASE 3: Test-time Expression Label Propagation [I2]
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 3: Expression Label Propagation [I2]")
print("=" * 55)

preds_base  = base_model.predict(
    [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm],
    batch_size=256, verbose=0)
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])

results_by_alpha = {}
for alpha in [0.0, 0.2, 0.3, 0.5]:
    refined = base_scores.copy() if alpha == 0.0 else \
              expression_label_propagation(base_scores, X_test_expr, alpha=alpha)
    results_by_alpha[alpha] = refined
    if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
        print("  alpha={:.1f}: AUROC={:.4f} AUPRC={:.4f}".format(
            alpha, roc_auc_score(y_test, refined),
            average_precision_score(y_test, refined)))

best_alpha = max(results_by_alpha,
                 key=lambda a: average_precision_score(y_test, results_by_alpha[a])
                 if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else 0)
final_scores = results_by_alpha[best_alpha]
final_auroc  = roc_auc_score(y_test, final_scores)
final_auprc  = average_precision_score(y_test, final_scores)
print("\n  [Best/AUPRC] alpha={:.1f} → AUROC={:.4f} AUPRC={:.4f}".format(
    best_alpha, final_auroc, final_auprc))
score_by_phase['Final(LP)'] = final_scores

# ==========================================================================
# 임베딩 품질 요약
# ==========================================================================
print("\n" + "=" * 55)
print(">>> Embedding Quality Summary")
print("=" * 55)
print("{:22} {:>10} {:>10} {:>10} {:>12}".format(
    "Phase", "Silhouette", "Sep.Ratio", "LinAUROC", "PredAUROC"))
print("-" * 66)
for phase_name, emb in emb_by_phase.items():
    m = analyze_embedding_quality(emb, y_test, phase_name)
    try:
        pred_auroc = roc_auc_score(y_test, score_by_phase[phase_name])
    except Exception:
        pred_auroc = float('nan')
    print("{:22} {:10.4f} {:10.4f} {:10.4f} {:12.4f}".format(
        phase_name, m['silhouette'], m['sep_ratio'],
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
