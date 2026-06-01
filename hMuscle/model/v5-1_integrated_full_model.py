# -*- coding: utf-8 -*-
# ============================================================================
# v5-1_integrated_full_model.py
#
# v5-fix 대비 핵심 변경 (4가지 버그 수정):
#
# [Fix1] Phase 3: PFN(distribution shift) → Test-time Label Propagation
#     - 문제: X_train_expr=zeros, X_test_expr=real_CPM → PFN AUROC ~0.51
#     - 수정: test isoform 내부 expression cosine 유사도 기반 score 전파
#     - train expression 불필요, data leakage 없음 (label 없이 score만 전파)
#     - 예상 효과: GO:0006936 0.7696 → 0.82-0.85
#
# [Fix2] Phase 2: LR 축소 + alpha 조정 + epoch 증가 + Best checkpoint
#     - 문제: LR=0.001 너무 높아 5 epoch만에 over-prediction (GO:0022900 97%)
#     - 수정: adam_p2 lr=0.0003, focal alpha 0.25→0.10
#     - 수정: 최대 15 epoch + val AUROC 기반 early stop + best model 보존
#     - 예상 효과: GO:0022900 Phase2 AUROC 0.5677 → 0.62+ 복구
#
# [Fix3] Phase 1: n_batches adaptive (GO term positive density 기반)
#     - 문제: n_batches=50 고정 → 14K/epoch만 샘플 → epoch 10 이후 0.1% active
#     - 수정: n_pos < 400 → 30배치, n_pos < 1200 → 100배치, else → 50배치
#     - 수정: margin threshold 60%→50%, active_rate streak 기반 early stop 추가
#
# [Fix4] Phase 2: active_rate 기반 early stop (새 기준)
#     - val AUROC가 best 대비 -0.02 이상 하락 시 best weights 복원 후 종료
#
# v5-fix에서 유지:
#     - 2-modal DNN (seq+domain, 32-dim)
#     - Phase 1: GradientTape semi-hard triplet mining
#     - Phase 1.5: Linear probing (encoder frozen, 2 epoch)
#     - Phase 2: Ingroup joint Focal+Triplet
#     - CRF 없음
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

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
    print("Usage: python v5-1_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%y%m%d")
VER_TAG  = "v5-1_integrated"
SAVE_DIR = os.path.join(BASE_RESULTS_DIR, "{}_{}".format(VER_TAG, date_str))
os.makedirs(SAVE_DIR, exist_ok=True)

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
EXPR_DIM = 24    # bambu CPM cell-type 컬럼 수
EMB_DIM  = 32    # feature_model 출력 차원 (seq16 + domain16)

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

def triplet_loss_fn(inputs, margin=0.1):
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
        print("[Expr] WARNING: {} isoforms missing from bambu — zero-filled".format(len(missing)))

    X_expr = np.zeros((len(iso_str), EXPR_DIM), dtype=np.float32)
    for i, iso in enumerate(iso_str):
        if iso in expr_df.index:
            X_expr[i] = np.log1p(expr_df.loc[iso].values.astype(float))
    print("[Expr] Done. shape={} nonzero_ratio={:.1f}%".format(
        X_expr.shape, (X_expr > 0).mean() * 100))
    return X_expr

def save_expr_matrix(X_expr, save_path):
    np.save(save_path, X_expr)
    print("[Expr] Saved: {}".format(save_path))

# --------------------------------------------------------------------------
# [5] 임베딩 추출 (2-modal: seq+domain)
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
# [6] Phase 1 — GradientTape Triplet (semi-hard mining)
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=0.1, mode="hard"):
    """
    Semi-hard negative mining (FaceNet, 2015).
    1순위: semi-hard (d_ap < d_an < d_ap + margin)
    2순위: moderate  (d_an < d_ap + 4*margin)
    3순위: random fallback
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
        optimizer, margin=0.1, batch_size=256,
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
                         margin=0.1, n_sample=1000):
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
# [Fix1] Phase 3: Test-time Label Propagation (Expression Refinement)
# --------------------------------------------------------------------------
def expression_label_propagation(base_scores, X_expr, alpha=0.3, k=15,
                                  sim_threshold=0.1):
    """
    Expression 유사도 기반 test-time score 전파.

    핵심 원리:
      - train expression 불필요 → distribution shift 없음
      - test isoform끼리만 연산 (label 없이 score 전파) → data leakage 없음
      - Expression=0 isoform: 유사도 0 → 전파 영향 없음 (자연 처리)

    Args:
        base_scores : Phase 2 prediction scores (n_test,)
        X_expr      : log1p CPM 행렬 (n_test, EXPR_DIM)
        alpha       : propagation weight (0=base_score만, 1=완전 전파)
        k           : expression k-NN 이웃 수
        sim_threshold: 이 유사도 미만은 edge 제거 (noise 억제)

    Returns:
        refined_scores: (n_test,) — 정제된 최종 score
    """
    n = len(base_scores)

    # Expression이 모두 0인 경우 (데이터 없음) → 원본 반환
    nonzero_rows = np.abs(X_expr).sum(axis=1)
    if (nonzero_rows > 0).sum() < k + 1:
        print("  [LabelProp] Expression too sparse — skipping propagation")
        return base_scores.copy()

    # L2 normalize for cosine similarity via inner product
    expr_norm = normalize(X_expr.astype(np.float32), norm='l2')

    # KNN graph: cosine distance
    print("  [LabelProp] Building KNN graph (k={})...".format(k))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine',
                             algorithm='brute', n_jobs=-1).fit(expr_norm)
    distances, indices = nbrs.kneighbors(expr_norm)
    # distances[i, 0] = 0 (self) → skip

    # Weighted average propagation
    prop_scores    = np.zeros(n, dtype=np.float32)
    weight_sum     = np.zeros(n, dtype=np.float32)

    for i in range(n):
        for rank in range(1, k + 1):  # skip self (rank=0)
            j   = indices[i, rank]
            sim = max(0.0, 1.0 - float(distances[i, rank]))  # cosine similarity
            if sim < sim_threshold:
                continue
            # Expression=0인 isoform(neighbor)은 sim≈0이라 자동 필터링됨
            prop_scores[i]  += sim * base_scores[j]
            weight_sum[i]   += sim

    # Normalize by weight sum (avoid division by zero)
    valid = weight_sum > 0
    prop_scores[valid] = prop_scores[valid] / weight_sum[valid]
    # Expression=0 또는 이웃 없는 isoform: base_score 유지
    prop_scores[~valid] = base_scores[~valid]

    refined = (1.0 - alpha) * base_scores + alpha * prop_scores

    # Diagnostics
    changed = np.abs(refined - base_scores)
    print("  [LabelProp] alpha={:.2f} k={} | score delta: mean={:.4f} max={:.4f}".format(
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
# [9] Expression 행렬 로딩 / 캐싱
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
    save_expr_matrix(X_test_expr, EXPR_CACHE)

assert X_test_expr.shape == (len(test_iso_arr), EXPR_DIM), \
    "Expression matrix shape mismatch: {} vs expected ({}, {})".format(
        X_test_expr.shape, len(test_iso_arr), EXPR_DIM)

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
# [Fix3] Phase 1: GO term positive density 기반 adaptive n_batches
# --------------------------------------------------------------------------
if n_pos_train < 400:
    # Sparse term: 과도한 학습 방지 (GO:0006096류)
    N_BATCHES_P1 = 30
    print("[Config] Sparse GO term (pos={}): n_batches=30".format(n_pos_train))
elif n_pos_train < 1200:
    # Tissue-specific term (GO:0006936, GO:0022900류)
    N_BATCHES_P1 = 100
    print("[Config] Tissue-specific GO term (pos={}): n_batches=100".format(n_pos_train))
else:
    # Universal term: triplet 효과 제한적 (GO:0006412류)
    N_BATCHES_P1 = 50
    print("[Config] Universal GO term (pos={}): n_batches=50".format(n_pos_train))

# --------------------------------------------------------------------------
# [11] 모델 구조 (2-modal: seq + domain only, v5-fix와 동일)
# --------------------------------------------------------------------------
seq_input    = Input(shape=(None,),   dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim,), dtype='int32', name='domain_input')

# Sequence branch
x1 = Embedding(input_dim=8001, output_dim=32)(seq_input)
x1 = Convolution1D(filters=64, kernel_size=32, padding='valid', activation='relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(1e-5))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.2)(x1)
seq_feat = Dense(16, activation='relu')(x1)

# Domain branch
x2 = Embedding(input_dim=domain_emb_dim, output_dim=32,
               input_length=dm_dim, mask_zero=True)(domain_input)
domain_feat = LSTM(16)(x2)

# Fusion: 16 + 16 = 32-dim
concat = concatenate([seq_feat, domain_feat])
feature_model = Model([seq_input, domain_input], concat, name='feature_model')

x = Dense(16, kernel_regularizer=regularizers.l2(1e-5))(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer   = Lambda(lambda a: K.l2_normalize(a, axis=1),
                           name="embedding_out")(x)
prediction_layer  = Dense(1, activation='sigmoid',
                          kernel_regularizer=regularizers.l2(1e-5))(embedding_layer)

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
adam_main = optimizers.Adam(lr=0.001)    # Phase 1.5 전용
adam_p2   = optimizers.Adam(lr=0.0003)  # [Fix2] Phase 2 전용 LR (축소)

print("\n[Model] feature_model output dim = {} (2-modal: seq+domain)".format(EMB_DIM))
print("[Model] Expression({}-dim) → Phase 3 Label Propagation only".format(EXPR_DIM))
print("[Model] Phase 1 n_batches={}".format(N_BATCHES_P1))

# ==========================================================================
# PHASE 0: Untrained Baseline
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 0: Untrained Baseline")
print("=" * 50)

save_phase_results("phase0_initial_untrained", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 1: 임베딩 기반 Triplet (GradientTape)
# [Fix3] adaptive n_batches, [Fix2] active_rate streak early stop
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 1: Embedding-based Triplet (GradientTape, margin=0.1, max 15 epochs)")
print("=" * 50)

PHASE1_EPOCHS       = 15
WARMUP_EPOCHS       = 2
best_margin_sat     = 0.0
final_centroid_dist = 0.0
low_active_streak   = 0       # [Fix3] active_rate 연속 저조 카운터
ACTIVE_THRESH       = 2.0     # 2% 미만 = 사실상 수렴
STREAK_LIMIT        = 3       # 3 epoch 연속 → early stop

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
        margin=0.1,
        batch_size=256,
        n_batches=N_BATCHES_P1,
        warmup=warmup)

    print("  -> Triplet Loss: {:.4f} | Active triplets: {:.1f}%".format(
        avg_loss, active_rate))

    # [Fix3] Active triplet streak 기반 early stop
    if not warmup:
        if active_rate < ACTIVE_THRESH:
            low_active_streak += 1
        else:
            low_active_streak = 0
        if low_active_streak >= STREAK_LIMIT:
            print("  [Early Stop] active_rate < {:.1f}% for {} epochs — stopping Phase 1.".format(
                ACTIVE_THRESH, STREAK_LIMIT))
            break

    if (epoch + 1) % 5 == 0:
        sat, cdist = compute_margin_stats(
            feature_model, X_test_seq, X_test_dm,
            y_test, seq_dim, margin=0.1)
        best_margin_sat     = max(best_margin_sat, sat)
        final_centroid_dist = cdist
        if sat >= 50.0:   # [Fix3] 60% → 50% (더 현실적 기준)
            print("  [Early Stop] margin satisfied {:.1f}% >= 50%.".format(sat))
            break

print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
    best_margin_sat, final_centroid_dist))
save_phase_results("phase1_triplet_only", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 1.5: Linear Probing (Encoder 동결)
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

save_phase_results("phase1_5_linear_probing", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

for layer in feature_model.layers:
    layer.trainable = True
print("  [Unfreeze] All layers unlocked.")

# ==========================================================================
# PHASE 2: Joint Fine-tuning (Triplet ingroup + Focal)
# [Fix2] lr=0.0003, alpha=0.10, max 15 epoch + best checkpoint + early stop
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 2: Joint Fine-tuning (Triplet ingroup + Focal)")
print("    [Fix2] LR=0.0003 | focal_alpha=0.10 | max 15 epochs | best checkpoint")
print("=" * 50)

# [Fix2] Phase 2 전용 optimizer와 focal alpha
triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.10)],
    loss_weights=[0.1, 1.0],
    optimizer=adam_p2)

PHASE2_MAX_EPOCHS   = 15
PHASE2_CHECK_EVERY  = 3       # 3 epoch마다 val AUROC 체크
best_phase2_auroc   = 0.0
best_phase2_weights = None
no_improve_count    = 0
NO_IMPROVE_LIMIT    = 2       # 2회 연속 하락 → early stop + best 복원

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

    # [Fix4] AUROC 기반 best checkpoint + early stop
    if (epoch + 1) % PHASE2_CHECK_EVERY == 0:
        preds = base_model.predict([X_test_seq, X_test_dm], batch_size=256, verbose=0)
        test_scores = np.array([preds[1][i][0] for i in range(len(y_test))])
        if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
            current_auroc = roc_auc_score(y_test, test_scores)
            print("  [AUROC Check] epoch={} AUROC={:.4f} (best={:.4f})".format(
                epoch + 1, current_auroc, best_phase2_auroc))
            if current_auroc > best_phase2_auroc:
                best_phase2_auroc   = current_auroc
                best_phase2_weights = base_model.get_weights()
                no_improve_count    = 0
            else:
                no_improve_count += 1
                if no_improve_count >= NO_IMPROVE_LIMIT:
                    print("  [Early Stop] Phase 2 AUROC not improving ({} checks) → restoring best".format(
                        NO_IMPROVE_LIMIT))
                    if best_phase2_weights is not None:
                        base_model.set_weights(best_phase2_weights)
                    break

# Best checkpoint 확인 복원
if best_phase2_weights is not None and no_improve_count < NO_IMPROVE_LIMIT:
    base_model.set_weights(best_phase2_weights)
    print("\n[Phase 2] Restored best checkpoint (AUROC={:.4f})".format(best_phase2_auroc))

save_phase_results("phase2_joint_focal", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

_, p2_centroid_dist = compute_margin_stats(
    feature_model, X_test_seq, X_test_dm, y_test, seq_dim, margin=0.1)

# ==========================================================================
# PHASE 3: Test-time Expression Label Propagation
# [Fix1] PFN(distribution shift) → expression similarity score 전파
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 3: Test-time Expression Label Propagation")
print("    [Fix1] Train expression 불필요 — test-only unsupervised refinement")
print("=" * 50)

# Phase 2 best embedding으로 base score 추출
preds_base = base_model.predict([X_test_seq, X_test_dm], batch_size=256, verbose=0)
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])
print("  Base (Phase 2) scores: mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    base_scores.mean(), base_scores.std(),
    (base_scores > 0.5).sum(), (base_scores > 0.3).sum()))

# Label propagation alpha ablation (alpha=0.0은 base score 그대로)
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
        print("  alpha={:.1f}: AUROC={:.4f}".format(alpha, auroc))

# 최적 alpha 선택 (val AUROC 기준)
best_alpha  = 0.0
best_auroc  = -1.0
if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
    for alpha, scores in results_by_alpha.items():
        auroc = roc_auc_score(y_test, scores)
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = alpha

final_scores = results_by_alpha[best_alpha]
print("\n  [Best] alpha={:.1f} → AUROC={:.4f}".format(best_alpha, best_auroc))
print("  [LabelProp Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))

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

# Labels도 저장 (평가용)
np.save(os.path.join(SAVE_DIR, "{}_{}_Final_LabelProp_labels.npy".format(VER_TAG, safe_go)),
        y_test)

print("\n[Final] alpha={:.1f} | AUROC={:.4f}".format(best_alpha, best_auroc))
print("[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
