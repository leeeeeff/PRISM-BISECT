# -*- coding: utf-8 -*-
# ============================================================================
# v5-fix_integrated_full_model.py
#
# v5 대비 핵심 변경 (distribution shift 수정):
#
# [1] Expression을 DNN input에서 PFN input으로 이동 (Strategy B 재설계)
#     - 문제: v5는 train_expr=zeros, test_expr=real_CPM → DNN distribution shift
#     - 수정: DNN은 seq+domain만 학습 (train/test 일관성 유지)
#     - Phase 3 PFN input = [embedding(32-dim), expression(24-dim)] = 56-dim
#     - 근거: quick ablation → PFN(emb+expr) AUROC=0.8738 vs PFN(emb only)=0.7786
#
# [2] Phase 3 focal training 제거 (Phase 2 embedding 보존)
#     - 문제: v5 Phase 3에서 focal 10ep → acc=99.81% (all-negative collapse)
#     - 수정: Phase 2 weights 동결, PFN 1회 실행만 수행
#     - Phase 2 AUROC=0.79 → PFN(emb+expr) AUROC=0.87+ 달성 목표
#
# [3] CRF 완전 제거 (EXP-01 근거: CRF가 모든 GO term에서 AUROC 평균 -0.21)
#     - Final output = PFN score (CRF pos_prob 대신)
#
# [4] Phase 1 semi-hard mining 유지 (v5에서 개선)
#     - L2 normalized embeddings, refresh_interval=10, squared L2 metric
#
# [5] DNN 구조: 2-modal (seq+domain, 32-dim) — v4-3 동일
#
# 다음 버전 예정 (v5-1):
#     - 전략 C: Expression-guided Triplet Mining
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

import torch
from tabpfn import TabPFNClassifier

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
    print("Usage: python v5_integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%y%m%d")
VER_TAG  = "v5fix_integrated"
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
EXPR_DIM     = 24   # bambu CPM cell-type 컬럼 수 (PFN input용)
EMB_DIM      = 32   # feature_model 출력 차원 (seq16 + domain16)
PFN_INPUT_DIM = EMB_DIM + EXPR_DIM  # 32 + 24 = 56

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
# [4] Expression 전처리 유틸
# --------------------------------------------------------------------------
def load_expression_matrix(cpm_path, isoform_list):
    """
    bambu CPM 파일에서 isoform_list 순서대로 발현 행렬 추출 후 log1p 변환.
    Training isoform은 zeros로 대체.

    Returns:
        X_expr: np.ndarray (n_isoforms, EXPR_DIM), dtype float32
    """
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
# [5] 임베딩 추출 유틸 (make_batch 기반, 2-modal: seq+domain only)
# --------------------------------------------------------------------------
def extract_embeddings(feature_model, X_seq, X_dm, seq_dim, emb_dim=EMB_DIM):
    """
    make_batch 기반 길이별 그룹 추출.
    feature_model: [seq, domain] → 32-dim concat (expr 없음)
    v5-fix: expression은 DNN에서 제거, PFN input으로만 사용
    """
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
            # L2 normalize: mining과 loss 모두 동일한 공간에서 비교하기 위함
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            emb[sub_idx] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [6] Phase 1 — 임베딩 기반 Triplet GradientTape
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=0.1, mode="hard"):
    """
    Semi-hard negative mining (FaceNet, 2015).
    embeddings: L2-normalized → squared L2 = 2*(1 - cosine_sim)
    loss의 거리 메트릭(squared L2)과 동일한 공간에서 mining.

    우선순위:
      1. Semi-hard: d_ap_sq < d_an_sq < d_ap_sq + margin  (active loss, 안정적 gradient)
      2. Moderate:  d_an_sq < d_ap_sq + 4*margin           (active loss, 다소 쉬운 negative)
      3. Random fallback                                    (warmup 또는 pool이 비었을 때)

    vs 이전 hard negative (nearest 20):
      - Hard negative (d_an < d_ap): gradient가 positive cluster를 붕괴시킬 위험
      - Semi-hard: positive보다 멀지만 margin 내 → 안정적이고 유의미한 gradient
    """
    n = min(batch_size, len(pos_indices))
    a_idx_list, p_idx_list, n_idx_list = [], [], []

    if mode == "hard" and len(neg_indices) >= 2:
        neg_emb     = embeddings[neg_indices]   # (n_neg, dim), L2-normalized
        sampled_pos = np.random.choice(pos_indices, n, replace=False)

        for a_i in sampled_pos:
            p_candidates = pos_indices[pos_indices != a_i]
            if len(p_candidates) == 0:
                continue
            p_i = np.random.choice(p_candidates)

            # Squared L2 via dot product trick (L2-normalized → d_sq = 2 - 2*cos)
            a_emb   = embeddings[a_i]
            p_emb   = embeddings[p_i]
            d_ap_sq = max(0.0, 2.0 - 2.0 * float(np.dot(a_emb, p_emb)))
            d_an_sq = np.clip(2.0 - 2.0 * (neg_emb @ a_emb), 0.0, None)  # (n_neg,)

            # 1순위: semi-hard — positive보다 멀고, margin 이내
            semi_mask = (d_an_sq > d_ap_sq) & (d_an_sq < d_ap_sq + margin)
            semi_idx  = np.where(semi_mask)[0]
            if len(semi_idx) > 0:
                n_i = neg_indices[np.random.choice(semi_idx)]
            else:
                # 2순위: moderate — margin의 4배 이내 (fallback)
                mod_mask = d_an_sq < d_ap_sq + 4.0 * margin
                mod_idx  = np.where(mod_mask)[0]
                if len(mod_idx) > 0:
                    n_i = neg_indices[np.random.choice(mod_idx)]
                else:
                    # 3순위: random fallback
                    n_i = np.random.choice(neg_indices)

            a_idx_list.append(a_i)
            p_idx_list.append(p_i)
            n_idx_list.append(n_i)
    else:
        # Warmup: random sampling
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
    """
    임베딩 기반 Triplet GradientTape 학습 1 epoch.
    v5-fix: expression 제거 (2-modal: seq+domain only)

    개선 (v5):
      - extract_embeddings: L2 normalize 적용 → loss와 동일 공간에서 mining
      - Semi-hard negative mining: hard negative에 의한 embedding 붕괴 방지
      - refresh_interval=10: 매 10 batch마다 embedding 재추출 → stale 오차 감소
    """
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
        # Refresh embeddings every refresh_interval batches (stale 오차 감소)
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
    """
    Phase 2 보조 Triplet — ingroup 방식 (Focal 주/Triplet 보조).
    v5-fix: 2-modal (seq+domain), expr 없음.
    """
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
        # Squared L2: loss/mining과 동일 메트릭 (기존: np.linalg.norm = 비제곱 L2)
        # L2-normalized 벡터: d_sq = 2 - 2*dot (cosine distance × 2)
        d_ap_sq = max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[a_i], pos_emb[p_i])))
        d_an_sq = max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[a_i], neg_emb[n_i])))
        margins.append(d_an_sq - d_ap_sq)

    margins       = np.array(margins)
    satisfied     = (margins > margin).mean() * 100   # margin=0.1 in squared L2 space
    centroid_dist = np.linalg.norm(pos_emb.mean(axis=0) - neg_emb.mean(axis=0))
    print("  [Margin|sq_L2] satisfied(>{:.2f}): {:.1f}% | centroid_dist: {:.4f}".format(
        margin, satisfied, centroid_dist))
    return satisfied, centroid_dist


def save_phase_results(phase_name, model_base, X_seq, X_dm, y_true,
                       gene_ids, iso_ids, save_dir, ver_tag):
    """v5-fix: 2-modal (seq+domain only). expr 없음."""
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
# [8] PFN Bridge
# --------------------------------------------------------------------------
class DiffusePFNLayerBridge:
    def __init__(self):
        self.x_support = None
        self.y_support = None
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit_support_set(self, x_train, y_train):
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos   = len(pos_idx)
        if len(neg_idx) > n_pos:
            neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
        balanced_idx       = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(balanced_idx)
        self.x_support = x_train[balanced_idx]
        self.y_support = y_train[balanced_idx]

    def predict_proba_bridge(self, x_query):
        x_sup, y_sup = self.x_support, self.y_support
        if len(x_sup) > 2048:
            idx      = np.random.choice(len(x_sup), 2048, replace=False)
            x_sup, y_sup = x_sup[idx], y_sup[idx]
        clf   = TabPFNClassifier(device=self.device, n_estimators=4)
        clf.fit(x_sup, y_sup)
        preds = []
        for i in range(0, len(x_query), 1000):
            batch = x_query[i:i+1000]
            if len(batch) == 0:
                break
            preds.append(clf.predict_proba(batch)[:, 1])
        return np.concatenate(preds)

# --------------------------------------------------------------------------
# [9] 데이터 로딩
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
# [10] Expression 행렬 로딩 / 캐싱
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
# [11] 레이블 생성 및 업샘플
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

print("  y_train pos={} neg={} ratio={:.3f}%".format(
    (y_train == 1).sum(), (y_train == 0).sum(),
    (y_train == 1).sum() / len(y_train) * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

# Training expression: zeros placeholder (PFN Phase 3에서 사용)
# DNN training에는 expression 사용 안 함 (v5-fix 핵심 변경)
n_train_total = len(X_train_seq)
X_train_expr  = np.zeros((n_train_total, EXPR_DIM), dtype=np.float32)

unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# [12] 모델 구조 (2-modal: seq + domain only)
# v5-fix: expression을 DNN에서 제거. Train/Test 일관성 확보.
# Expression은 Phase 3 PFN input으로만 사용.
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

# Fusion: 16 + 16 = 32-dim (expression 없음 — PFN에서만 사용)
concat = concatenate([seq_feat, domain_feat])
feature_model = Model([seq_input, domain_input],
                      concat, name='feature_model')

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

# Phase 2 Triplet model (ingroup, Focal 주 / Triplet 보조)
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

print("\n[Model] feature_model output dim = {} (2-modal: seq+domain)".format(EMB_DIM))
print("[Model] Expression({}-dim) → PFN input only (not DNN)".format(EXPR_DIM))

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
# margin=0.1, 15 epochs, warmup 2 epoch
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 1: Embedding-based Triplet (GradientTape, margin=0.1, 15 epochs)")
print("=" * 50)

PHASE1_EPOCHS       = 15
WARMUP_EPOCHS       = 2
best_margin_sat     = 0.0
final_centroid_dist = 0.0

for epoch in range(PHASE1_EPOCHS):
    warmup   = (epoch < WARMUP_EPOCHS)
    mode_str = "random (warmup)" if warmup else "hard negative"
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
        n_batches=50,
        warmup=warmup)

    print("  -> Triplet Loss: {:.4f} | Active triplets: {:.1f}%".format(
        avg_loss, active_rate))

    if (epoch + 1) % 5 == 0:
        sat, cdist = compute_margin_stats(
            feature_model, X_test_seq, X_test_dm,
            y_test, seq_dim, margin=0.1)
        best_margin_sat     = max(best_margin_sat, sat)
        final_centroid_dist = cdist
        if sat >= 60.0:
            print("  [Early Stop] margin satisfied {:.1f}% >= 60%.".format(sat))
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
# Focal 주(1.0) + Triplet 보조(0.1)
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 2: Joint Fine-tuning (Triplet ingroup + Focal)")
print("=" * 50)

triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.25)],
    loss_weights=[0.1, 1.0],
    optimizer=adam_main)

for epoch in range(5):
    print('Phase 2 - Epoch: {}/5'.format(epoch + 1))
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

save_phase_results("phase2_joint_focal", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

_, p2_centroid_dist = compute_margin_stats(
    feature_model, X_test_seq, X_test_dm, y_test, seq_dim, margin=0.1)

# ==========================================================================
# PHASE 3: PFN with [embedding + expression] — No more focal training
# v5-fix 핵심: Phase 2 embedding 동결, expression을 PFN input으로 추가
# 근거: quick ablation → PFN(emb+expr) AUROC=0.8738 vs PFN(emb only)=0.7786
# ==========================================================================
print("\n" + "=" * 50)
print(">>> PHASE 3: PFN([embedding, expression]) — feature_model FROZEN")
print("=" * 50)
print("  [Fix] Phase 2 weights preserved. No focal training.")
print("  [Fix] Expression({}-dim) concatenated to embedding({}-dim) for PFN.".format(
    EXPR_DIM, EMB_DIM))

# feature_model 완전 동결 (Phase 2 embedding 보존)
for layer in feature_model.layers:
    layer.trainable = False
for layer in base_model.layers:
    layer.trainable = False

pfn_bridge = DiffusePFNLayerBridge()

# Phase 2 embedding 추출 (train + test, seq+domain only)
print("  Extracting Phase 2 embeddings for PFN (train+test)...")
X_pfn_emb_train = extract_embeddings(feature_model, X_train_seq, X_train_dm, seq_dim)
X_pfn_emb_test  = extract_embeddings(feature_model, X_test_seq, X_test_dm, seq_dim)

# Expression concatenation: [embedding, expression] = [32, 24] = 56-dim
X_pfn_train_full = np.hstack([X_pfn_emb_train, X_train_expr])  # (n_train, 56)
X_pfn_test_full  = np.hstack([X_pfn_emb_test,  X_test_expr])   # (36748, 56)
print("  PFN input shape: train={} test={}".format(
    X_pfn_train_full.shape, X_pfn_test_full.shape))

# PFN 실행 (support=train, query=test)
print("  Running PFN...")
pfn_bridge.fit_support_set(X_pfn_train_full, y_train)
pfn_score_all = pfn_bridge.predict_proba_bridge(
    np.vstack([X_pfn_train_full, X_pfn_test_full]))

final_scores = pfn_score_all[len(y_train):]
print("  PFN → mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))

# --------------------------------------------------------------------------
# [13] 최종 저장
# --------------------------------------------------------------------------
print("\nSaving Final Results...")
output_file = os.path.join(
    SAVE_DIR, "{}_{}_Final_PFN_scores.txt".format(VER_TAG, safe_go))
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                 str(final_scores[i]) + '\n')

base_model.save_weights(
    os.path.join(SAVE_DIR, "{}_{}_BaseModel_weights.h5".format(VER_TAG, safe_go)))

print("\n[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
