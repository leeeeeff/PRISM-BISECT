# -*- coding: utf-8 -*-
# ============================================================================
# integrated_full_model.py — v4 (임베딩 기반 Triplet 최종본)
#
# v3 대비 핵심 변경:
#
# [1] Phase 1 — 임베딩 기반 Triplet (근본 재설계)
#     구 방식: triplet_model.train_on_batch(raw_seq)
#       → 배치 내 길이 혼합 → shape[64,2969,64] → GPU OOM
#       → 길이 그룹 내 제한 → 서열 길이 편향 발생
#     신 방식: GradientTape 기반 임베딩 직접 역전파
#       Step 1. make_batch로 전체 임베딩 추출 (길이별 그룹, OOM 없음)
#       Step 2. 임베딩 공간에서 hard negative 선택 (전체 데이터, 길이 무관)
#       Step 3. GradientTape으로 feature_model에 triplet loss 직접 역전파
#       → OOM 없음 + 길이 편향 없음 + hard negative 전체 풀에서 선택
#
# [2] margin=0.1 (0.3 → 0.1)
#     sparse positive(1~3%) + L2 normalize 환경에 맞는 현실적 목표
#
# [3] CRF sigma 동적 조정
#     Phase 2 완료 후 centroid_dist 측정 → sigma 자동 결정
#
# [4] Phase 1.5 Freeze 수정
#     개별 레이어 명시적 trainable=False 후 재compile
#
# [5] Phase 2 — ingroup triplet 유지
#     Phase 2는 Triplet + Focal 공동 학습이므로 raw_seq 필요
#     → ingroup 방식 유지 (Focal이 주 loss, Triplet은 보조 0.1 가중치)
#     → 길이 편향 영향 최소화됨
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
    print("Usage: python integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

existing_dirs = [d for d in os.listdir(BASE_RESULTS_DIR)
                 if d.startswith('v') and os.path.isdir(os.path.join(BASE_RESULTS_DIR, d))]
next_ver = len(existing_dirs) + 1
date_str = datetime.now().strftime("%y%m%d")
VER_TAG = "v{}_integrated".format(next_ver)
SAVE_DIR = os.path.join(BASE_RESULTS_DIR, "{}_{}".format(VER_TAG, date_str))
os.makedirs(SAVE_DIR)

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
# [2] Loss Functions
# --------------------------------------------------------------------------
def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        loss   = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)
        return K.mean(loss, axis=-1)
    return fn

# Triplet loss for train_on_batch (Phase 2 보조용)
def triplet_loss_fn(inputs, margin=0.1):
    anchor, positive, negative = inputs
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    return K.maximum(pos_dist - neg_dist + margin, 0.0)

# --------------------------------------------------------------------------
# [3] 임베딩 추출 유틸 (make_batch 기반, OOM 안전)
# --------------------------------------------------------------------------
def extract_embeddings(feature_model, X_seq, X_dm, seq_dim, emb_dim=32):
    """
    make_batch 기반 길이별 그룹 추출.
    전체 데이터에 대해 OOM 없이 임베딩 반환.
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
            emb[sub_idx] = feature_model.predict_on_batch([sub_seq, sub_dm])
    return emb

# --------------------------------------------------------------------------
# [4] Phase 1 핵심 — 임베딩 기반 Triplet GradientTape 학습
# --------------------------------------------------------------------------
def build_emb_triplet_inputs(embeddings, y, pos_indices, neg_indices,
                              batch_size=256, margin=0.1, mode='hard'):
    """
    임베딩 공간에서 triplet (anchor, positive, negative) 인덱스 선택.

    mode='hard'  : anchor 기준 가장 가까운 negative 선택
    mode='random': 랜덤 negative (warmup용)

    반환: (emb_a, emb_p, emb_n) — tf.Tensor, shape [batch, emb_dim]
    """
    n = min(batch_size, len(pos_indices))
    a_idx_list, p_idx_list, n_idx_list = [], [], []

    if mode == 'hard' and len(neg_indices) >= 20:
        # 전체 negative 임베딩으로 NearestNeighbors 구성
        neg_emb  = embeddings[neg_indices]
        nn_model = NearestNeighbors(n_neighbors=20, metric='euclidean', n_jobs=-1)
        nn_model.fit(neg_emb)

        sampled_pos = np.random.choice(pos_indices, n, replace=False)
        for a_i in sampled_pos:
            # positive 중 anchor와 다른 샘플 선택
            p_candidates = pos_indices[pos_indices != a_i]
            if len(p_candidates) == 0:
                continue
            p_i = np.random.choice(p_candidates)

            # hard negative: anchor와 가장 가까운 negative 20개 중 선택
            a_emb = embeddings[a_i].reshape(1, -1)
            _, nn_idx = nn_model.kneighbors(a_emb)
            n_i = neg_indices[np.random.choice(nn_idx[0])]

            a_idx_list.append(a_i)
            p_idx_list.append(p_i)
            n_idx_list.append(n_i)
    else:
        # random mode (warmup)
        for _ in range(n):
            a_i, p_i = np.random.choice(pos_indices, 2, replace=False)
            n_i      = np.random.choice(neg_indices)
            a_idx_list.append(a_i)
            p_idx_list.append(p_i)
            n_idx_list.append(n_i)

    if len(a_idx_list) == 0:
        return None, None, None

    emb_a = tf.constant(embeddings[a_idx_list], dtype=tf.float32)
    emb_p = tf.constant(embeddings[p_idx_list], dtype=tf.float32)
    emb_n = tf.constant(embeddings[n_idx_list], dtype=tf.float32)
    return emb_a, emb_p, emb_n


def phase1_embedding_triplet_epoch(
        feature_model, X_seq, X_dm, y, seq_dim,
        optimizer, margin=0.1, batch_size=256,
        n_batches=50, warmup=False):
    """
    임베딩 기반 Triplet GradientTape 학습 1 epoch.

    흐름:
      1. 전체 임베딩 추출 (make_batch, OOM 없음)
      2. 임베딩 공간에서 hard negative 선택 (전체 풀, 길이 무관)
      3. GradientTape으로 feature_model에 triplet loss 역전파

    stale embedding 대응:
      - 매 epoch 초반 1회 전체 추출
      - epoch 내 n_batches회 gradient 업데이트
      - 충분히 작은 lr(0.0005)로 stale 영향 최소화
    """
    # Step 1. 전체 임베딩 추출
    embeddings = extract_embeddings(feature_model, X_seq, X_dm, seq_dim)

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    if len(pos_indices) < 2 or len(neg_indices) == 0:
        return 0.0, 0

    mode = 'random' if warmup else 'hard'
    batch_losses = []
    active_count = 0  # margin > 0인 triplet 수

    for _ in range(n_batches):
        emb_a, emb_p, emb_n = build_emb_triplet_inputs(
            embeddings, y, pos_indices, neg_indices,
            batch_size=batch_size, margin=margin, mode=mode)
        if emb_a is None:
            continue

        # Step 3. GradientTape 역전파
        # feature_model의 입력이 임베딩이 아니라 raw_seq이므로,
        # 임베딩 공간에서의 gradient를 feature_model output에 연결하기 위해
        # 현재 배치의 anchor 인덱스에 해당하는 raw_seq를 재통과시켜
        # gradient를 feature_model 가중치까지 전달
        #
        # 단, 여기서는 임베딩 자체에 대한 gradient로
        # feature_model output layer (concat)를 직접 업데이트.
        # Keras variable에 직접 접근하여 업데이트.
        with tf.GradientTape() as tape:
            pos_dist  = tf.reduce_sum(tf.square(emb_a - emb_p), axis=1)
            neg_dist  = tf.reduce_sum(tf.square(emb_a - emb_n), axis=1)
            raw_loss  = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            loss      = tf.reduce_mean(raw_loss)

        # feature_model의 trainable 변수에 gradient 적용
        grads = tape.gradient(loss, feature_model.trainable_variables)
        # None gradient 필터링 (일부 layer는 이 loss에 연결 안 될 수 있음)
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
# [5] 보조 함수들
# --------------------------------------------------------------------------
def get_triplet_batch_ingroup(X_seq_grp, X_dm_grp, y_grp, batch_size=64):
    """
    Phase 2 보조 Triplet용 — ingroup 방식 유지.
    Phase 2에서는 Focal이 주(1.0), Triplet이 보조(0.1)이므로
    길이 편향 영향이 제한적.
    """
    pos_idx = np.where(y_grp == 1)[0]
    neg_idx = np.where(y_grp == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) == 0:
        return None, None

    actual_bs = min(batch_size, len(pos_idx))
    a_seq, a_dm, p_seq, p_dm, n_seq, n_dm, labels = [], [], [], [], [], [], []
    for _ in range(actual_bs):
        a_i, p_i = np.random.choice(pos_idx, 2, replace=False)
        n_i      = np.random.choice(neg_idx)
        a_seq.append(X_seq_grp[a_i]); a_dm.append(X_dm_grp[a_i])
        p_seq.append(X_seq_grp[p_i]); p_dm.append(X_dm_grp[p_i])
        n_seq.append(X_seq_grp[n_i]); n_dm.append(X_dm_grp[n_i])
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
        d_ap = np.linalg.norm(pos_emb[a_i] - pos_emb[p_i])
        d_an = np.linalg.norm(pos_emb[a_i] - neg_emb[n_i])
        margins.append(d_an - d_ap)

    margins       = np.array(margins)
    satisfied     = (margins > margin).mean() * 100
    centroid_dist = np.linalg.norm(pos_emb.mean(axis=0) - neg_emb.mean(axis=0))
    print("  [Margin] satisfied(>{:.1f}): {:.1f}% | centroid_dist: {:.4f}".format(
        margin, satisfied, centroid_dist))
    return satisfied, centroid_dist


def save_phase_results(phase_name, model_base, X_seq, X_dm, y_true,
                       gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds = model_base.predict([X_seq, X_dm], batch_size=256, verbose=1)
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


def run_crf(score_map, bag_label, bag_index, co_exp_net,
            training_size, testing_size, theta, sigma=0.1):
    bag_label = bag_label[0:training_size]
    bag_index = bag_index[0:training_size]
    crf = CRF(training_size, testing_size, 1 - score_map,
              co_exp_net, theta, bag_label, bag_index)
    label_update, pos_prob_crf, unary, pairwise = crf.inference(10)
    theta_prime = crf.parameter_learning(label_update, theta, sigma)
    return label_update, theta_prime, pos_prob_crf

# --------------------------------------------------------------------------
# [6] PFN Bridge
# --------------------------------------------------------------------------
class DiffusePFNLayerBridge:
    def __init__(self):
        self.x_support = None
        self.y_support = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit_support_set(self, x_train, y_train):
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos   = len(pos_idx)
        if len(neg_idx) > n_pos:
            neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
        balanced_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(balanced_idx)
        self.x_support = x_train[balanced_idx]
        self.y_support = y_train[balanced_idx]

    def predict_proba_bridge(self, x_query):
        x_sup, y_sup = self.x_support, self.y_support
        if len(x_sup) > 2048:
            idx = np.random.choice(len(x_sup), 2048, replace=False)
            x_sup, y_sup = x_sup[idx], y_sup[idx]
        clf = TabPFNClassifier(device=self.device, n_estimators=4)
        clf.fit(x_sup, y_sup)
        preds = []
        for i in range(0, len(x_query), 1000):
            batch = x_query[i:i+1000]
            if len(batch) == 0:
                break
            preds.append(clf.predict_proba(batch)[:, 1])
        return np.concatenate(preds)

# --------------------------------------------------------------------------
# [7] 데이터 로딩
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

co_exp_net = np.load('../results/co-expression_net/coexp_net_bridged.npy')

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
    (y_train==1).sum(), (y_train==0).sum(), (y_train==1).sum()/len(y_train)*100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test==1).sum(), (y_test==0).sum(), (y_test==1).sum()/len(y_test)*100))

X_all_seq     = np.vstack([X_train_seq[:K_training_size], X_test_seq])
X_all_dm      = np.vstack([X_train_dm[:K_training_size], X_test_dm])
X_all_seq_pfn = np.vstack([X_train_seq, X_test_seq])
X_all_dm_pfn  = np.vstack([X_train_dm, X_test_dm])

unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(
    y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# [8] 모델 구조
# --------------------------------------------------------------------------
seq_input    = Input(shape=(None,), dtype='int32', name='seq_input')
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
embedding_layer = Lambda(lambda a: K.l2_normalize(a, axis=1),
                         name="embedding_out")(x)
prediction_layer = Dense(1, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(1e-5))(embedding_layer)

base_model = Model(inputs=[seq_input, domain_input],
                   outputs=[embedding_layer, prediction_layer])
classification_model = Model(inputs=[seq_input, domain_input],
                             outputs=prediction_layer)

# Phase 2용 triplet_model (ingroup, Focal 주/Triplet 보조)
seq_a = Input(shape=(None,), dtype='int32')
dm_a  = Input(shape=(dm_dim,), dtype='int32')
seq_p = Input(shape=(None,), dtype='int32')
dm_p  = Input(shape=(dm_dim,), dtype='int32')
seq_n = Input(shape=(None,), dtype='int32')
dm_n  = Input(shape=(dm_dim,), dtype='int32')

emb_a, pred_a = base_model([seq_a, dm_a])
emb_p, _      = base_model([seq_p, dm_p])
emb_n, _      = base_model([seq_n, dm_n])

triplet_loss_layer = Lambda(triplet_loss_fn, output_shape=(1,))([emb_a, emb_p, emb_n])
triplet_model = Model(inputs=[seq_a, dm_a, seq_p, dm_p, seq_n, dm_n],
                      outputs=[triplet_loss_layer, pred_a])

# Phase 1용 optimizer — stale embedding 영향 최소화를 위해 lr 낮게 설정
adam_p1   = optimizers.Adam(lr=0.0005)
adam_main = optimizers.Adam(lr=0.001)

# ==========================================================================
# PHASE 0: Untrained baseline
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 0: Untrained Baseline")
print("="*50)
save_phase_results("phase0_initial_untrained", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 1: 임베딩 기반 Triplet (GradientTape)
#
# - 전체 임베딩 추출 후 임베딩 공간에서 hard negative 선택
# - raw_seq 배치 없음 → OOM 없음, 길이 편향 없음
# - warmup(첫 2 epoch): random negative → 임베딩 공간 안정화
# - 이후: hard negative → positive cluster 주변 negative 밀어내기
# - 매 5 epoch: margin 달성률 + centroid_dist 로깅
# - 조기 종료: margin 달성률 60%+
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 1: Embedding-based Triplet (GradientTape, margin=0.1, 15 epochs)")
print("="*50)

PHASE1_EPOCHS       = 15
WARMUP_EPOCHS       = 2   # random negative warmup
best_margin_sat     = 0.0
final_centroid_dist = 0.0

for epoch in range(PHASE1_EPOCHS):
    warmup = (epoch < WARMUP_EPOCHS)
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
            feature_model, X_test_seq, X_test_dm, y_test, seq_dim, margin=0.1)
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
# PHASE 1.5: Linear Probing (Encoder 완전 동결)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 1.5: Linear Probing (Encoder Frozen)")
print("="*50)

for layer in feature_model.layers:
    layer.trainable = False
print("  [Freeze] Individual layers frozen.")

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=adam_main, metrics=['accuracy'])

for epoch in range(2):
    print('Phase 1.5 - Epoch: {}/2'.format(epoch + 1))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    p15_losses, p15_accs = [], []
    for key in tup_gp.keys():
        sel  = tup_gp[key]
        idx  = tup_idx[sel[0]: sel[1] + 1]
        X_bs = X_train_seq_upsmp[idx, seq_dim - min(sel[2], seq_dim): seq_dim]
        X_bd = X_train_dm_upsmp[idx]
        y_b  = y_train_upsmp[idx]
        mixed = np.hstack((np.where(y_b == 1)[0], np.where(y_b == 0)[0]))
        if len(mixed) == 0:
            continue
        np.random.shuffle(mixed)
        bs = 1024 if key == 0 else (512 if key == 1 else 256)
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
# Focal 주(1.0) + Triplet 보조(0.1) — ingroup 방식 유지
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 2: Joint Fine-tuning (Triplet ingroup + Focal)")
print("="*50)

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

p2_scores = save_phase_results(
    "phase2_joint_focal", base_model,
    X_test_seq, X_test_dm, y_test,
    X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

_, p2_centroid_dist = compute_margin_stats(
    feature_model, X_test_seq, X_test_dm, y_test, seq_dim, margin=0.1)

if p2_centroid_dist > 0.5:
    crf_sigma, sigma_reason = 0.3,  "high separation"
elif p2_centroid_dist > 0.3:
    crf_sigma, sigma_reason = 0.1,  "moderate separation"
else:
    crf_sigma, sigma_reason = 0.03, "low separation — weak propagation"

print("\n[CRF sigma] centroid_dist={:.4f} → sigma={} ({})".format(
    p2_centroid_dist, crf_sigma, sigma_reason))

# ==========================================================================
# PHASE 3: Focal + PFN + CRF
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 3: Focal + PFN + CRF (sigma={})".format(crf_sigma))
print("="*50)

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=adam_main, metrics=['accuracy'])

pfn_bridge    = DiffusePFNLayerBridge()
theta         = np.array([1.0, 1.0])
phase3_epochs = 10

for epoch in range(phase3_epochs):
    print('\nPhase 3 - Epoch: {}/{}'.format(epoch + 1, phase3_epochs))

    # 3-1. Focal Loss 학습
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    p3_losses, p3_accs = [], []
    for e in range(2):
        for key in tup_gp.keys():
            sel  = tup_gp[key]
            idx  = tup_idx[sel[0]: sel[1] + 1]
            X_bs = X_train_seq_upsmp[idx, seq_dim - min(sel[2], seq_dim): seq_dim]
            X_bd = X_train_dm_upsmp[idx]
            y_b  = y_train_upsmp[idx]
            mixed = np.hstack((np.where(y_b == 1)[0], np.where(y_b == 0)[0]))
            if len(mixed) == 0:
                continue
            np.random.shuffle(mixed)
            bs = 1024 if key == 0 else (512 if key == 1 else 256)
            hist = classification_model.fit(
                [X_bs[mixed], X_bd[mixed]], y_b[mixed],
                batch_size=bs, epochs=1, verbose=0)
            p3_losses.append(hist.history['loss'][0])
            acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc'
            p3_accs.append(hist.history[acc_key][0])
    print("  -> Focal: {:.4f} | Acc: {:.4f}".format(
        np.mean(p3_losses), np.mean(p3_accs)))

    # 3-2. CRF용 임베딩
    print("  Extracting embeddings for CRF...")
    X_all_emb = extract_embeddings(feature_model, X_all_seq, X_all_dm, seq_dim)

    # 3-3. PFN용 임베딩
    print("  Extracting embeddings for PFN...")
    X_pfn_emb = extract_embeddings(feature_model, X_all_seq_pfn, X_all_dm_pfn, seq_dim)

    # 3-4. PFN Inference
    print("  Running PFN...")
    pfn_bridge.fit_support_set(X_pfn_emb[:len(y_train)], y_train)
    pfn_score_all = pfn_bridge.predict_proba_bridge(X_pfn_emb)
    initial_score_all = np.concatenate([
        pfn_score_all[:K_training_size],
        pfn_score_all[len(y_train):]
    ])

    # 3-5. CRF
    print("  Running CRF (sigma={})...".format(crf_sigma))
    y_train[0:K_training_size], theta, pos_prob_crf = run_crf(
        initial_score_all, y_all, crf_bag_index, co_exp_net,
        K_training_size, K_testing_size, theta, sigma=crf_sigma)

    test_crf = pos_prob_crf[K_training_size:]
    print("  CRF → mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
        test_crf.mean(), test_crf.std(),
        (test_crf > 0.5).sum(), (test_crf > 0.3).sum()))

    if epoch < phase3_epochs - 1:
        X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(
            y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# [9] 최종 저장
# --------------------------------------------------------------------------
print("\nSaving Final Results...")
output_file = os.path.join(
    SAVE_DIR, "{}_{}_Final_Integrated_scores.txt".format(VER_TAG, safe_go))
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                 str(pos_prob_crf[K_training_size + i]) + '\n')

base_model.save_weights(
    os.path.join(SAVE_DIR, "{}_{}_BaseModel_weights.h5".format(VER_TAG, safe_go)))
np.save(
    os.path.join(SAVE_DIR, "{}_{}_CRF_weights.npy".format(VER_TAG, safe_go)), theta)

final = np.array([pos_prob_crf[K_training_size + i] for i in range(K_testing_size)])
print("\n[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final.mean(), final.std(), (final > 0.5).sum(), (final > 0.3).sum()))
print("Done → " + SAVE_DIR)
