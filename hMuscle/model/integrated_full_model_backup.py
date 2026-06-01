# -*- coding: utf-8 -*-
import numpy as np
import random
import sys
import os
from sys import argv
import time

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, Convolution1D, Embedding, Lambda, Flatten, concatenate
from keras import backend as K
from keras import regularizers, losses, optimizers
from sklearn.metrics import roc_auc_score

# 사용자 라이브러리
from crf import CRF
from layer.PyramidPooling import PyramidPooling
from utils import generate_label, upsample, make_batch

# --------------------------------------------------------------------------
# [1] TensorFlow 설정 & Argument Parsing
# --------------------------------------------------------------------------
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

try:
    script, selected_go = argv
except ValueError:
    print("Usage: python integrated_full_model.py <GO_TERM>")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
SAVE_DIR = '../results/' + safe_go
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# --------------------------------------------------------------------------
# [2] Loss Functions & Helper Functions
# --------------------------------------------------------------------------
def triplet_loss(inputs, margin=1.0):
    anchor, positive, negative = inputs
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    return K.maximum(pos_dist - neg_dist + margin, 0.0)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow((1.0 - p_t), gamma)
        loss = -alpha_factor * modulating_factor * K.log(p_t)
        return K.mean(loss, axis=-1)
    return binary_focal_loss_fixed

def get_triplet_batch(X_seq, X_dm, y, tup_gp, batch_size=64):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    if len(pos_indices) < 2 or len(neg_indices) == 0: return None, None

    anchors_seq, anchors_dm, positives_seq, positives_dm, negatives_seq, negatives_dm, labels = [], [], [], [], [], [], []
    while len(anchors_seq) < batch_size:
        a_idx, p_idx = np.random.choice(pos_indices, 2, replace=False)
        n_idx = np.random.choice(neg_indices)
        anchors_seq.append(X_seq[a_idx]); anchors_dm.append(X_dm[a_idx])
        positives_seq.append(X_seq[p_idx]); positives_dm.append(X_dm[p_idx])
        negatives_seq.append(X_seq[n_idx]); negatives_dm.append(X_dm[n_idx])
        labels.append(1)

    return [np.array(anchors_seq), np.array(anchors_dm), np.array(positives_seq), np.array(positives_dm), 
            np.array(negatives_seq), np.array(negatives_dm)], np.array(labels)

def save_phase_results(phase_name, model_base, X_seq, X_dm, y_true, gene_ids, iso_ids, save_dir):
    print("\n[Save] Saving results for " + phase_name + "...")
    preds = model_base.predict([X_seq, X_dm], batch_size=256, verbose=1)
    np.save(os.path.join(save_dir, phase_name + '_embeddings.npy'), preds[0])
    np.save(os.path.join(save_dir, phase_name + '_labels.npy'), y_true)
    with open(os.path.join(save_dir, phase_name + '_scores.txt'), 'w') as fw:
        for i in range(len(y_true)): fw.write(gene_ids[i] + '\t' + iso_ids[i] + '\t' + str(preds[1][i][0]) + '\n')

def run_crf(score_map, bag_label, bag_index, co_exp_net, training_size, testing_size, theta, sigma = 10):
    bag_label = bag_label[0: training_size]
    bag_index = bag_index[0: training_size]
    crf = CRF(training_size, testing_size, 1 - score_map, co_exp_net, theta, bag_label, bag_index)
    label_update, pos_prob_crf, unary, pairwise = crf.inference(10)
    theta_prime = crf.parameter_learning(label_update, theta, sigma)
    return label_update, theta_prime, pos_prob_crf

# --------------------------------------------------------------------------
# [3] PFN Bridge Class
# --------------------------------------------------------------------------
class DiffusePFNLayerBridge:
    def __init__(self):
        self.x_support = None
        self.y_support = None

    def fit_support_set(self, x_train, y_train):
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos = len(pos_idx)
        if len(neg_idx) > n_pos: neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
        balanced_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(balanced_idx)
        self.x_support = x_train[balanced_idx]
        self.y_support = y_train[balanced_idx]

    def predict_proba_bridge(self, x_query):
        np.save('temp_x_support.npy', self.x_support)
        np.save('temp_y_support.npy', self.y_support)
        np.save('temp_x_query.npy', x_query)
        if os.system("/usr/bin/python3.8 pfn_worker.py") != 0:
            return np.zeros(len(x_query)) + 0.5
        if os.path.exists('temp_pred_result.npy'):
            return np.load('temp_pred_result.npy')
        return np.zeros(len(x_query))

# --------------------------------------------------------------------------
# [4] Data Loading & Preprocessing
# --------------------------------------------------------------------------
print('>>> Preparing Data for ' + selected_go)
# (간소화된 데이터 로딩 호출 - 기존 로직과 동일)
X_train_seq = np.load('../data/raw_data/data/sequences/human_sequence_train.npy')
X_train_dm = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_test_seq = np.load('my_sequence_matrix_fixed.npy')
X_test_dm = np.load('../results/domain/domain_matrix.npy')
X_train_geneid = list(np.load('../data/raw_data/data/id_lists/train_gene_list.npy'))
X_test_geneid = list(np.load('my_gene_list_fixed.npy'))
X_test_isoid = list(np.load('my_isoform_list_fixed.npy'))
X_train_other_seq = np.load('../data/raw_data/data/sequences/swissprot_sequence_train.npy')
X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_train_geneid_other = list(np.load('../data/raw_data/data/id_lists/train_swissprot_list.npy'))

positive_Gene = []
for file_name in ['human_annotations.txt', 'swissprot_annotations.txt']:
    with open('../data/raw_data/data/annotations/' + file_name, 'r') as fr:
        for line in fr:
            parts = line.strip().split('\t')
            if selected_go in parts[1:]: positive_Gene.append(parts[0])

co_exp_net = np.load('../results/co-expression_net/coexp_net_bridged.npy')

K_training_size = X_train_seq.shape[0]
K_testing_size = X_test_seq.shape[0]
X_all_seq = np.vstack([X_train_seq, X_test_seq])
X_all_dm = np.vstack([X_train_dm, X_test_dm])

seq_dim = X_train_seq.shape[1]
dm_dim = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm), np.max(X_train_other_dm)]) + 1

y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm = generate_label(
    X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene
)

unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# [5] Model Architecture Design
# --------------------------------------------------------------------------
seq_input = Input(shape=(None, ), dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim, ), dtype='int32', name='domain_input')

# Encoder
x1 = Embedding(input_dim=8001, output_dim=32)(seq_input)
x1 = Convolution1D(filters=64, kernel_size=32, strides=1, padding='valid', activation='relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(0.00001))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.2)(x1)
seq_feat = Dense(16, activation='relu')(x1)

x2 = Embedding(input_dim=domain_emb_dim, output_dim=32, input_length=dm_dim, mask_zero=True)(domain_input)
domain_feat = LSTM(16)(x2)

concat = concatenate([seq_feat, domain_feat]) # PFN용 임베딩 포인트
feature_model = Model([seq_input, domain_input], concat) # For PFN extraction

x = Dense(16, kernel_regularizer=regularizers.l2(0.00001))(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer = Lambda(lambda a: K.l2_normalize(a, axis=1), name='embedding')(x)
prediction_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001))(embedding_layer)

base_model = Model(inputs=[seq_input, domain_input], outputs=[embedding_layer, prediction_layer])
classification_model = Model(inputs=[seq_input, domain_input], outputs=prediction_layer) # For Phase 3

# Triplet Network
seq_a, dm_a = Input(shape=(None,), dtype='int32'), Input(shape=(dm_dim,), dtype='int32')
seq_p, dm_p = Input(shape=(None,), dtype='int32'), Input(shape=(dm_dim,), dtype='int32')
seq_n, dm_n = Input(shape=(None,), dtype='int32'), Input(shape=(dm_dim,), dtype='int32')

emb_a, pred_a = base_model([seq_a, dm_a])
emb_p, _      = base_model([seq_p, dm_p])
emb_n, _      = base_model([seq_n, dm_n])

triplet_loss_layer = Lambda(triplet_loss, output_shape=(1,))([emb_a, emb_p, emb_n])
triplet_model = Model(inputs=[seq_a, dm_a, seq_p, dm_p, seq_n, dm_n], outputs=[triplet_loss_layer, pred_a])

adam = optimizers.Adam(lr=0.001)

# ==========================================================================
# PHASE 1: Pure Metric Learning (Triplet 1.0)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 1: Metric Learning (Triplet Only)")
print("="*50)
triplet_model.compile(loss=[identity_loss, 'binary_crossentropy'], loss_weights=[1.0, 0.0], optimizer=adam)

for epoch in range(5): # Phase 1 에폭 축소 권장 (과적합 방지)
    print('Phase 1 - Epoch: {}/5'.format(epoch + 1))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    for step in range(int(len(X_train_seq_upsmp) / 64)):
        batch, labels = get_triplet_batch(X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, tup_gp)
        if batch: triplet_model.train_on_batch(batch, [np.zeros((len(labels), 1)), labels])

save_phase_results("phase1_triplet_only", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR)

# ==========================================================================
# PHASE 2: Joint Base Learning (Triplet 0.1 + BCE 1.0)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 2: Joint Base Learning (Triplet + BCE)")
print("="*50)
triplet_model.compile(loss=[identity_loss, 'binary_crossentropy'], loss_weights=[0.1, 1.0], optimizer=adam)

for epoch in range(5):
    print('Phase 2 - Epoch: {}/5'.format(epoch + 1))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    for step in range(int(len(X_train_seq_upsmp) / 64)):
        batch, labels = get_triplet_batch(X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, tup_gp)
        if batch: triplet_model.train_on_batch(batch, [np.zeros((len(labels), 1)), labels])

save_phase_results("phase2_joint_base", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR)

# ==========================================================================
# PHASE 3: Focal Loss + PFN Bridge + CRF Integration
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 3: Focal Loss Classification + PFN + CRF")
print("="*50)

# Focal Loss로 순수 분류기 재컴파일
classification_model.compile(loss=binary_focal_loss(gamma=2.0, alpha=0.25), optimizer=adam, metrics=['accuracy'])
pfn_bridge = DiffusePFNLayerBridge()
theta = np.array([1.0, 1.0])
phase3_epochs = 10

for epoch in range(phase3_epochs):
    print('\nPhase 3 (Focal/PFN/CRF) - Epoch: {}/{}'.format(epoch + 1, phase3_epochs))
    
    # 3-1. Train with Focal Loss (Standard Batches)
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    for e in range(2): # Inner epoch augmentation
        for key in tup_gp.keys():
            sel = tup_gp[key]
            st, ed, le = sel[0], sel[1], sel[2]
            idx = tup_idx[st: ed + 1]
            
            X_batch_seq = X_train_seq_upsmp[idx, seq_dim - min(le, seq_dim): seq_dim]
            X_batch_dm = X_train_dm_upsmp[idx]
            y_batch = y_train_upsmp[idx]

            pos_idx = np.where(y_batch == 1)[0]
            neg_idx = np.where(y_batch == 0)[0]
            mixed_idx = np.hstack((pos_idx, neg_idx))
            if len(mixed_idx) == 0: continue
            np.random.shuffle(mixed_idx)

            bs = 1024 if key == 0 else (512 if key == 1 else 256)
            classification_model.fit([X_batch_seq[mixed_idx], X_batch_dm[mixed_idx]], y_batch[mixed_idx], batch_size=bs, epochs=1, verbose=0)
            
    # 3-2. Extract Features for PFN
    print("  Extracting embeddings for PFN...")
    X_all_emb = np.zeros((X_all_seq.shape[0], 32))
    tup_idx_all, tup_gp_all = make_batch(X_all_seq)
    for key in tup_gp_all.keys():
        sel = tup_gp_all[key]
        batch_indices = tup_idx_all[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_indices) / 1000) + 1):
            sub_seq = X_all_seq[batch_indices[1000 * i: 1000 * (i + 1)], seq_dim - sel[2]: seq_dim]
            sub_dm = X_all_dm[batch_indices[1000 * i : 1000 * (i + 1)]]
            if len(sub_seq) > 0:
                X_all_emb[batch_indices[1000 * i : 1000 * (i + 1)]] = feature_model.predict_on_batch([sub_seq, sub_dm])

    # 3-3. PFN Inference
    print("  Running PFN Inference via Bridge...")
    pfn_bridge.fit_support_set(X_all_emb[:K_training_size], y_train)
    initial_score_all = pfn_bridge.predict_proba_bridge(X_all_emb)

    # 3-4. CRF Update & Upsample
    print("  Running CRF Update...")
    y_train[0: K_training_size], theta, pos_prob_crf = run_crf(
        initial_score_all, y_all, crf_bag_index, co_exp_net, K_training_size, K_testing_size, theta, sigma=0.1
    )
    if epoch < phase3_epochs - 1:
        X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(
            y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag
        )

# --------------------------------------------------------------------------
# [6] Final Output & Evaluation
# --------------------------------------------------------------------------
print("\nSaving Final Integrated Results...")
output_file = os.path.join(SAVE_DIR, safe_go + '_Final_Integrated_scores.txt')
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' + str(pos_prob_crf[K_training_size + i]) + '\n')

base_model.save_weights(os.path.join(SAVE_DIR, safe_go + '_Integrated_BaseModel_weights.h5'))
np.save(os.path.join(SAVE_DIR, safe_go + '_Integrated_CRF_weights.npy'), theta)
print("All Process Done! Results saved to " + SAVE_DIR)
