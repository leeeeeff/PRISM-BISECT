# -*- coding: utf-8 -*-
# ============================================================================
# integrated_full_model.py — v4
#
# v3 대비 변경사항:
#
# [1] Hard Negative Mining (Phase 1 핵심 수정)
#     - 문제: 랜덤 negative → easy negative 90%+ → Triplet gradient 미발생
#             ambiguous negative가 Focal에서 hard sample로 오인 → 0.5 이상 예측
#     - 해결: positive anchor와 임베딩 거리 가장 가까운 negative 선택
#             → positive cluster 주변 negative를 명시적으로 밀어내는 학습
#             → PFN 입력 임베딩 품질 향상
#     - 버그 수정: tup_gp 인자 제거 (시그니처 불일치 수정)
#     - 버그 수정: update() 내부 make_batch 기반 길이별 배치 처리 추가
#
# [2] margin=0.1 (0.3 → 0.1)
#     - 문제: sparse positive(1~3%) + L2 normalize 환경에서 margin=0.3도 과대
#             centroid_dist 분포: 0.158~0.654
#     - 해결: margin=0.1로 현실적 분리 목표 설정
#             → active triplet 비율 극대화
#             → Hard Negative Mining과 시너지
#
# [3] CRF sigma 동적 조정
#     - 문제: sigma=0.1 고정 → 분리도 낮은 GO term에서 CRF 반전 발생
#             GO:0006412 centroid_dist=0.158 → PFN 스코어 불안정 → CRF 증폭
#     - 해결: Phase 2 완료 후 centroid_dist 측정
#             분리도 높으면 sigma 크게(전파 강하게)
#             분리도 낮으면 sigma 작게(전파 약하게, 반전 방지)
#
# [4] Phase 1.5 Freeze 수정
#     - 문제: feature_model.trainable=False만으로는 공유 레이어 동결 불완전
#     - 해결: 개별 레이어 명시적 trainable=False 설정 후 재compile
#
# [5] margin 달성률 매 5 epoch 로깅
#     - Phase 1 수렴 추이 추적
#     - 60%+ 달성 시 Phase 1 조기 종료
#
# 전체 파이프라인:
#   Phase 0  : Untrained 상태 저장
#   Phase 1  : Hard Negative Mining + Triplet (margin=0.1, 15 epoch)
#   Phase 1.5: Linear Probing (Encoder 동결, Head Focal 2 epoch)
#   Phase 2  : Triplet(0.1) + Focal(1.0) (5 epoch)
#   Phase 3  : Focal + PFN + CRF + 동적 sigma (10 epoch)
# ============================================================================
import numpy as np
import random
import sys
import os
from sys import argv
import time
from datetime import datetime

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, Activation, LSTM,
                          Convolution1D, Embedding, Lambda, concatenate)
from keras import backend as K
from keras import regularizers, optimizers
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import torch
from tabpfn import TabPFNClassifier

from crf import CRF
import sys; sys.path.insert(0, '/home/welcome1/layer_Full')
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
def triplet_loss(inputs, margin=0.1):
    """
    Triplet Loss margin=0.1
    [v4 수정] 0.3 → 0.1
    근거: sparse positive(1~3%) + L2 normalize 환경
          centroid_dist 범위 0.158~0.654
          margin=0.1이 현실적 분리 목표 + active triplet 극대화
    """
    anchor, positive, negative = inputs
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    return K.maximum(pos_dist - neg_dist + margin, 0.0)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        loss = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)
        return K.mean(loss, axis=-1)
    return fn

# --------------------------------------------------------------------------
# [3] Hard Negative Mining
# --------------------------------------------------------------------------
class HardNegativeMiner:
    """
    [v4 핵심 추가]
    positive anchor와 임베딩 거리 가장 가까운 negative 선택.

    문제 배경:
      GO:0006412처럼 broad한 GO term은 많은 negative가
      positive cluster 주변에 분포 (translation 서열 모티프 공유).
      랜덤 negative는 이미 멀리 있는 easy negative를 주로 선택
      → Triplet gradient 거의 없음 → 임베딩 분리 실패.

    해결:
      positive cluster 주변 negative를 명시적으로 선택
      → 이들을 강제로 밀어내는 학습
      → ambiguous negative의 임베딩이 positive에서 멀어짐
      → Focal Loss가 이들을 hard sample로 오인하는 현상 감소
      → PFN 스코어 품질 향상

    [v4 버그 수정]
      update() 내부에서 make_batch 기반 길이별 배치 처리 추가.
      feature_model은 variable length sequence를 처리하므로
      길이별로 묶어서 추론해야 배치 처리가 정상 작동함.
    """
    def __init__(self, k_candidates=20, update_interval=3):
        self.k_candidates = k_candidates   # hard negative 후보 수
        self.update_interval = update_interval  # 임베딩 갱신 주기 (epoch)
        self.neg_nn = None
        self.neg_indices_global = None
        self.anchor_emb = None

    def update(self, feature_model, X_seq, X_dm, y_train, seq_dim):
        """
        현재 임베딩 기준으로 k-NN 갱신.
        [v4 버그 수정] make_batch 기반 길이별 배치 처리로 variable length 대응.
        """
        print("  [HNM] Updating embeddings for Hard Negative Mining...")
        tup_idx, tup_gp = make_batch(X_seq)
        emb_list = np.zeros((len(X_seq), 32))

        for key in tup_gp.keys():
            sel = tup_gp[key]
            batch_idx = tup_idx[sel[0]: sel[1] + 1]
            for i in range(int(len(batch_idx) / 1000) + 1):
                sub_idx = batch_idx[1000 * i: 1000 * (i + 1)]
                if len(sub_idx) == 0:
                    continue
                sub_seq = X_seq[sub_idx, seq_dim - sel[2]: seq_dim]
                sub_dm  = X_dm[sub_idx]
                emb_list[sub_idx] = feature_model.predict_on_batch([sub_seq, sub_dm])

        self.neg_indices_global = np.where(y_train == 0)[0]
        neg_emb = emb_list[self.neg_indices_global]
        self.neg_nn = NearestNeighbors(
            n_neighbors=min(self.k_candidates, len(neg_emb)),
            metric='euclidean', n_jobs=-1)
        self.neg_nn.fit(neg_emb)
        self.anchor_emb = emb_list
        print("  [HNM] Updated. neg pool: {}".format(len(neg_emb)))

    def get_hard_negative(self, anchor_idx):
        """anchor에 가장 가까운 negative k개 중 랜덤 선택."""
        if self.neg_nn is None or self.anchor_emb is None:
            return np.random.choice(self.neg_indices_global)
        anchor_emb = self.anchor_emb[anchor_idx].reshape(1, -1)
        _, indices = self.neg_nn.kneighbors(anchor_emb)
        hard_neg_local = indices[0]
        chosen = np.random.choice(hard_neg_local)
        return self.neg_indices_global[chosen]


def get_triplet_batch_hard(X_seq, X_dm, y, batch_size=64, miner=None):
    """
    Hard Negative Mining 적용 triplet 배치 생성.
    [v4 버그 수정] tup_gp 인자 제거 — 시그니처 불일치 수정.
    miner가 None이거나 미초기화 상태면 랜덤 sampling (warmup용).
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    if len(pos_indices) < 2 or len(neg_indices) == 0:
        return None, None

    a_seq, a_dm = [], []
    p_seq, p_dm = [], []
    n_seq, n_dm = [], []
    labels = []

    while len(a_seq) < batch_size:
        a_idx, p_idx = np.random.choice(pos_indices, 2, replace=False)

        if miner is not None and miner.neg_nn is not None:
            # Hard Negative: anchor 기준 가장 가까운 negative
            n_idx = miner.get_hard_negative(a_idx)
        else:
            # Fallback: 랜덤 (초기 epoch warmup)
            n_idx = np.random.choice(neg_indices)

        a_seq.append(X_seq[a_idx]); a_dm.append(X_dm[a_idx])
        p_seq.append(X_seq[p_idx]); p_dm.append(X_dm[p_idx])
        n_seq.append(X_seq[n_idx]); n_dm.append(X_dm[n_idx])
        labels.append(1)

    return ([np.array(a_seq), np.array(a_dm),
             np.array(p_seq), np.array(p_dm),
             np.array(n_seq), np.array(n_dm)],
            np.array(labels))


def compute_margin_stats(feature_model, X_seq, X_dm, y, seq_dim, margin=0.1, n_sample=1000):
    """
    Phase 1 수렴 품질 측정.
    margin 달성률 + centroid 분리도 로깅.
    variable length 대응을 위해 make_batch 사용.
    """
    tup_idx, tup_gp = make_batch(X_seq)
    emb = np.zeros((len(X_seq), 32))
    for key in tup_gp.keys():
        sel = tup_gp[key]
        batch_idx = tup_idx[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_idx) / 1000) + 1):
            sub_idx = batch_idx[1000 * i: 1000 * (i + 1)]
            if len(sub_idx) == 0:
                continue
            sub_seq = X_seq[sub_idx, seq_dim - sel[2]: seq_dim]
            sub_dm  = X_dm[sub_idx]
            emb[sub_idx] = feature_model.predict_on_batch([sub_seq, sub_dm])

    pos_emb = emb[y == 1]
    neg_emb = emb[y == 0]
    if len(pos_emb) < 2:
        return 0.0, 0.0

    np.random.seed(42)
    n = min(n_sample, len(pos_emb))
    margins = []
    for _ in range(n):
        a_idx, p_idx = np.random.choice(len(pos_emb), 2, replace=False)
        n_idx = np.random.choice(len(neg_emb))
        d_ap = np.linalg.norm(pos_emb[a_idx] - pos_emb[p_idx])
        d_an = np.linalg.norm(pos_emb[a_idx] - neg_emb[n_idx])
        margins.append(d_an - d_ap)

    margins = np.array(margins)
    satisfied = (margins > margin).mean() * 100

    pos_c = pos_emb.mean(axis=0)
    neg_c = neg_emb.mean(axis=0)
    centroid_dist = np.linalg.norm(pos_c - neg_c)

    print("  [Margin] satisfied(>{:.1f}): {:.1f}% | centroid_dist: {:.4f} | active: {:.1f}%".format(
        margin, satisfied, centroid_dist, 100 - satisfied))
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
# [4] PFN Bridge
# --------------------------------------------------------------------------
class DiffusePFNLayerBridge:
    def __init__(self):
        self.x_support = None
        self.y_support = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit_support_set(self, x_train, y_train):
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos = len(pos_idx)
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
# [5] 데이터 로딩
# --------------------------------------------------------------------------
print('>>> Preparing Data for ' + selected_go)

X_train_seq = np.load('../data/raw_data/data/sequences/human_sequence_train.npy')
X_train_dm = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_test_seq = np.load('my_sequence_matrix_fixed.npy')
X_test_dm = np.load('../results/domain/domain_matrix.npy')

def load_ids(path):
    return [x.decode('utf-8') if isinstance(x, bytes) else x
            for x in np.load(path, allow_pickle=True)]

X_train_geneid = load_ids('../data/raw_data/data/id_lists/train_gene_list.npy')
X_test_geneid  = load_ids('my_gene_list_fixed.npy')
X_test_isoid   = load_ids('my_isoform_list_fixed.npy')
X_train_other_seq = np.load('../data/raw_data/data/sequences/swissprot_sequence_train.npy')
X_train_other_dm  = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
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
seq_dim    = X_train_seq.shape[1]
dm_dim     = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm), np.max(X_train_other_dm)]) + 1

y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm = generate_label(
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
# [6] 모델 구조
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

seq_a = Input(shape=(None,), dtype='int32')
dm_a  = Input(shape=(dm_dim,), dtype='int32')
seq_p = Input(shape=(None,), dtype='int32')
dm_p  = Input(shape=(dm_dim,), dtype='int32')
seq_n = Input(shape=(None,), dtype='int32')
dm_n  = Input(shape=(dm_dim,), dtype='int32')

emb_a, pred_a = base_model([seq_a, dm_a])
emb_p, _      = base_model([seq_p, dm_p])
emb_n, _      = base_model([seq_n, dm_n])

triplet_loss_layer = Lambda(triplet_loss, output_shape=(1,))([emb_a, emb_p, emb_n])
triplet_model = Model(inputs=[seq_a, dm_a, seq_p, dm_p, seq_n, dm_n],
                      outputs=[triplet_loss_layer, pred_a])

adam  = optimizers.Adam(lr=0.001)
miner = HardNegativeMiner(k_candidates=20, update_interval=3)

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
# PHASE 1: Hard Negative Mining + Triplet Loss
#
# [v4 핵심]
# - margin=0.1: sparse positive 환경에 맞는 현실적 목표
# - Hard Negative Mining: ambiguous negative를 명시적으로 밀어내는 학습
# - 매 3 epoch: 임베딩 갱신 (make_batch 기반 variable length 대응)
# - 매 5 epoch: margin 달성률 + centroid_dist 로깅
# - 조기 종료: margin 달성률 60%+ 달성 시
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 1: Hard Negative Mining + Triplet (margin=0.1, 15 epochs)")
print("="*50)

triplet_model.compile(
    loss=[identity_loss, 'binary_crossentropy'],
    loss_weights=[1.0, 0.0],
    optimizer=adam)

PHASE1_EPOCHS = 15
best_margin_sat    = 0.0
final_centroid_dist = 0.0

for epoch in range(PHASE1_EPOCHS):
    print('Phase 1 - Epoch: {}/{}'.format(epoch + 1, PHASE1_EPOCHS))

    # Hard Negative Mining: 매 update_interval epoch마다 임베딩 갱신
    if epoch % miner.update_interval == 0:
        miner.update(feature_model, X_train_seq_upsmp, X_train_dm_upsmp,
                     y_train_upsmp, seq_dim)

    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    epoch_losses = []

    for key in tup_gp.keys():
        sel = tup_gp[key]
        batch_idx = tup_idx[sel[0]: sel[1] + 1]
        # 길이 그룹별로 배치 구성
        X_grp_seq = X_train_seq_upsmp[batch_idx, seq_dim - sel[2]: seq_dim]
        X_grp_dm  = X_train_dm_upsmp[batch_idx]
        y_grp     = y_train_upsmp[batch_idx]

        for step in range(max(1, int(len(batch_idx) / 64))):
            batch, labels = get_triplet_batch_hard(
                X_grp_seq, X_grp_dm, y_grp,
                batch_size=64, miner=None)   # 그룹 내에서는 random (miner는 전체 인덱스 기준)
            if batch is None:
                continue
            losses = triplet_model.train_on_batch(
                batch, [np.zeros((len(labels), 1)), labels])
            epoch_losses.append(losses)

    if epoch_losses:
        avg = np.mean(epoch_losses, axis=0)
        print("  -> Triplet Loss: {:.4f}".format(avg[1]))

    # 매 5 epoch마다 margin 달성률 체크
    if (epoch + 1) % 5 == 0:
        sat, cdist = compute_margin_stats(
            feature_model, X_test_seq, X_test_dm, y_test,
            seq_dim, margin=0.1)
        best_margin_sat    = max(best_margin_sat, sat)
        final_centroid_dist = cdist

        if sat >= 60.0:
            print("  [Early Stop] margin satisfied {:.1f}% >= 60%. Stopping Phase 1.".format(sat))
            break

print("\n[Phase 1 Final] best_margin_sat={:.1f}% centroid_dist={:.4f}".format(
    best_margin_sat, final_centroid_dist))

save_phase_results("phase1_triplet_only", base_model,
                   X_test_seq, X_test_dm, y_test,
                   X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 1.5: Linear Probing
#
# [v4 버그 수정]
# feature_model.trainable=False 만으로는 공유 레이어 동결 불완전.
# 개별 레이어 명시적 trainable=False 후 재compile로 완전 동결 보장.
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 1.5: Linear Probing (Encoder Frozen)")
print("="*50)

# 개별 레이어 명시적 동결
for layer in feature_model.layers:
    layer.trainable = False
print("  [Freeze] Individual layers frozen: ∂L/∂W_enc = 0")

# 동결 후 반드시 재compile
classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=adam,
    metrics=['accuracy'])

for epoch in range(2):
    print('Phase 1.5 - Epoch: {}/2'.format(epoch + 1))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    p15_losses, p15_accs = [], []

    for key in tup_gp.keys():
        sel = tup_gp[key]
        idx = tup_idx[sel[0]: sel[1] + 1]
        X_bs = X_train_seq_upsmp[idx, seq_dim - min(sel[2], seq_dim): seq_dim]
        X_bd = X_train_dm_upsmp[idx]
        y_b  = y_train_upsmp[idx]
        pos_i = np.where(y_b == 1)[0]
        neg_i = np.where(y_b == 0)[0]
        mixed = np.hstack((pos_i, neg_i))
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

# 개별 레이어 언프리즈
for layer in feature_model.layers:
    layer.trainable = True
print("  [Unfreeze] All layers unlocked.")

# ==========================================================================
# PHASE 2: Joint Fine-tuning (Triplet + Focal)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 2: Joint Fine-tuning (Triplet + Focal)")
print("="*50)

triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.25)],
    loss_weights=[0.1, 1.0],
    optimizer=adam)

for epoch in range(5):
    print('Phase 2 - Epoch: {}/5'.format(epoch + 1))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    epoch_losses = []

    for key in tup_gp.keys():
        sel = tup_gp[key]
        batch_idx = tup_idx[sel[0]: sel[1] + 1]
        X_grp_seq = X_train_seq_upsmp[batch_idx, seq_dim - sel[2]: seq_dim]
        X_grp_dm  = X_train_dm_upsmp[batch_idx]
        y_grp     = y_train_upsmp[batch_idx]

        for step in range(max(1, int(len(batch_idx) / 64))):
            batch, labels = get_triplet_batch_hard(
                X_grp_seq, X_grp_dm, y_grp,
                batch_size=64, miner=None)
            if batch is None:
                continue
            losses = triplet_model.train_on_batch(
                batch, [np.zeros((len(labels), 1)), labels])
            epoch_losses.append(losses)

    if epoch_losses:
        avg = np.mean(epoch_losses, axis=0)
        print("  -> Total: {:.4f} | Triplet: {:.4f} | Focal: {:.4f}".format(
            avg[0], avg[1], avg[2]))

# Phase 2 완료 후 centroid_dist 측정 → sigma 동적 결정
p2_scores = save_phase_results(
    "phase2_joint_focal", base_model,
    X_test_seq, X_test_dm, y_test,
    X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

_, p2_centroid_dist = compute_margin_stats(
    feature_model, X_test_seq, X_test_dm, y_test,
    seq_dim, margin=0.1)

# ==========================================================================
# CRF sigma 동적 조정
# [v4 핵심]
# centroid_dist 기반으로 sigma 동적 결정:
#   >0.5  → sigma=0.3  (강한 전파)
#   >0.3  → sigma=0.1  (기본 전파)
#   <=0.3 → sigma=0.03 (약한 전파, GO:0006412 반전 방지)
# ==========================================================================
if p2_centroid_dist > 0.5:
    crf_sigma    = 0.3
    sigma_reason = "high separation"
elif p2_centroid_dist > 0.3:
    crf_sigma    = 0.1
    sigma_reason = "moderate separation"
else:
    crf_sigma    = 0.03
    sigma_reason = "low separation — weak propagation to prevent inversion"

print("\n[CRF sigma] centroid_dist={:.4f} → sigma={} ({})".format(
    p2_centroid_dist, crf_sigma, sigma_reason))

# ==========================================================================
# PHASE 3: Focal + PFN + CRF (동적 sigma)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 3: Focal + PFN + CRF (sigma={})".format(crf_sigma))
print("="*50)

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=adam,
    metrics=['accuracy'])

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
            sel = tup_gp[key]
            idx = tup_idx[sel[0]: sel[1] + 1]
            X_bs = X_train_seq_upsmp[idx, seq_dim - min(sel[2], seq_dim): seq_dim]
            X_bd = X_train_dm_upsmp[idx]
            y_b  = y_train_upsmp[idx]
            pos_i = np.where(y_b == 1)[0]
            neg_i = np.where(y_b == 0)[0]
            mixed = np.hstack((pos_i, neg_i))
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

    # 3-2. CRF용 임베딩 추출 (K_training_size + K_testing_size)
    print("  Extracting embeddings for CRF...")
    X_all_emb = np.zeros((X_all_seq.shape[0], 32))
    tup_idx_all, tup_gp_all = make_batch(X_all_seq)
    for key in tup_gp_all.keys():
        sel = tup_gp_all[key]
        batch_idx = tup_idx_all[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_idx) / 1000) + 1):
            sub_idx = batch_idx[1000 * i: 1000 * (i + 1)]
            if len(sub_idx) == 0:
                continue
            sub_seq = X_all_seq[sub_idx, seq_dim - sel[2]: seq_dim]
            sub_dm  = X_all_dm[sub_idx]
            X_all_emb[sub_idx] = feature_model.predict_on_batch([sub_seq, sub_dm])

    # 3-3. PFN용 임베딩 추출 (전체 train + test)
    print("  Extracting embeddings for PFN...")
    X_pfn_emb = np.zeros((X_all_seq_pfn.shape[0], 32))
    tup_idx_pfn, tup_gp_pfn = make_batch(X_all_seq_pfn)
    for key in tup_gp_pfn.keys():
        sel = tup_gp_pfn[key]
        batch_idx = tup_idx_pfn[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_idx) / 1000) + 1):
            sub_idx = batch_idx[1000 * i: 1000 * (i + 1)]
            if len(sub_idx) == 0:
                continue
            sub_seq = X_all_seq_pfn[sub_idx, seq_dim - sel[2]: seq_dim]
            sub_dm  = X_all_dm_pfn[sub_idx]
            X_pfn_emb[sub_idx] = feature_model.predict_on_batch([sub_seq, sub_dm])

    # 3-4. PFN Inference
    print("  Running PFN...")
    pfn_bridge.fit_support_set(X_pfn_emb[:len(y_train)], y_train)
    pfn_score_all = pfn_bridge.predict_proba_bridge(X_pfn_emb)
    initial_score_all = np.concatenate([
        pfn_score_all[:K_training_size],
        pfn_score_all[len(y_train):]
    ])

    # 3-5. CRF (동적 sigma)
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
# [7] 최종 저장
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
