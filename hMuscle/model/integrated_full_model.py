# -*- coding: utf-8 -*-
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
from keras.layers import Input, Dense, Dropout, Activation, LSTM, Convolution1D, Embedding, Lambda, Flatten, concatenate
from keras import backend as K
from keras import regularizers, losses, optimizers
from sklearn.metrics import roc_auc_score

# TabPFN 직접 import
import torch
from tabpfn import TabPFNClassifier

# 사용자 라이브러리
from crf import CRF
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)
from PyramidPooling import PyramidPooling
from utils_Full import generate_label, upsample, make_batch

# --------------------------------------------------------------------------
# [1] TensorFlow 설정, Argument Parsing & 자동 디렉토리 생성
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
BASE_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_isoform', safe_go)

# --- 자동 버전 관리 및 폴더 생성 ---
if not os.path.exists(BASE_RESULTS_DIR): 
    os.makedirs(BASE_RESULTS_DIR)

existing_dirs = [d for d in os.listdir(BASE_RESULTS_DIR) if d.startswith('v') and os.path.isdir(os.path.join(BASE_RESULTS_DIR, d))]
next_ver = len(existing_dirs) + 1

date_str = datetime.now().strftime("%y%m%d")
VER_TAG = f"v{next_ver}_integrated"
DIR_NAME = f"{VER_TAG}_{date_str}"

SAVE_DIR = os.path.join(BASE_RESULTS_DIR, DIR_NAME)
os.makedirs(SAVE_DIR)

# --- [추가] 터미널 출력을 로그 파일로 자동 미러링 ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 즉시 저장
    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_file_path = os.path.join(SAVE_DIR, f"{VER_TAG}_training.log")
sys.stdout = Logger(log_file_path)

print(f"\n[Info] New experiment initialized. Data & Logs will be saved to: {SAVE_DIR}\n")

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
        # 🌟 정수형(int64) 라벨을 실수형(float32)으로 안전하게 캐스팅 (Type Error 해결)
        y_true = tf.cast(y_true, tf.float32)
        
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = (y_true * y_pred) + ((1.0 - y_true) * (1.0 - y_pred))
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
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

def save_phase_results(phase_name, model_base, X_seq, X_dm, y_true, gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] Saving results for " + phase_name + "...")
    preds = model_base.predict([X_seq, X_dm], batch_size=256, verbose=1)
    np.save(os.path.join(save_dir, f'{ver_tag}_{phase_name}_embeddings.npy'), preds[0])
    np.save(os.path.join(save_dir, f'{ver_tag}_{phase_name}_labels.npy'), y_true)
    with open(os.path.join(save_dir, f'{ver_tag}_{phase_name}_scores.txt'), 'w') as fw:
        for i in range(len(y_true)): fw.write(gene_ids[i] + '\t' + iso_ids[i] + '\t' + str(preds[1][i][0]) + '\n')

def run_crf(score_map, bag_label, bag_index, co_exp_net, training_size, testing_size, theta, sigma=10):
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        x_sup, y_sup = self.x_support, self.y_support
        if len(x_sup) > 2048:
            idx = np.random.choice(len(x_sup), 2048, replace=False)
            x_sup, y_sup = x_sup[idx], y_sup[idx]
        classifier = TabPFNClassifier(device=self.device, n_estimators=4)
        classifier.fit(x_sup, y_sup)
        preds = []
        for i in range(0, len(x_query), 1000):
            batch = x_query[i:i+1000]
            if len(batch) == 0: break
            preds.append(classifier.predict_proba(batch)[:, 1])
        return np.concatenate(preds)

# --------------------------------------------------------------------------
# [4] Data Loading & Preprocessing
# --------------------------------------------------------------------------
print('>>> Preparing Data for ' + selected_go)

X_train_seq = np.load('../data/raw_data/data/sequences/human_sequence_train.npy')
X_train_dm = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_test_seq = np.load('my_sequence_matrix_fixed.npy')
X_test_dm = np.load('../results/domain/domain_matrix.npy')
X_train_geneid = [x.decode('utf-8') if isinstance(x, bytes) else x for x in np.load('../data/raw_data/data/id_lists/train_gene_list.npy', allow_pickle=True)]
X_test_geneid = [x.decode('utf-8') if isinstance(x, bytes) else x for x in np.load('my_gene_list_fixed.npy', allow_pickle=True)]
X_test_isoid = [x.decode('utf-8') if isinstance(x, bytes) else x for x in np.load('my_isoform_list_fixed.npy', allow_pickle=True)]
X_train_other_seq = np.load('../data/raw_data/data/sequences/swissprot_sequence_train.npy')
X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_train_geneid_other = [x.decode('utf-8') if isinstance(x, bytes) else x for x in np.load('../data/raw_data/data/id_lists/train_swissprot_list.npy', allow_pickle=True)]

positive_Gene = []
for file_name in ['human_annotations.txt', 'swissprot_annotations.txt']:
    with open('../data/raw_data/data/annotations/' + file_name, 'r') as fr:
        for line in fr:
            parts = line.strip().split('\t')
            if selected_go in parts[1:]: positive_Gene.append(parts[0])

co_exp_net = np.load('../results/co-expression_net/coexp_net_bridged.npy')

K_training_size = X_train_seq.shape[0]
K_testing_size = X_test_seq.shape[0]

seq_dim = X_train_seq.shape[1]
dm_dim = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm), np.max(X_train_other_dm)]) + 1

y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm = generate_label(
    X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene
)

X_all_seq = np.vstack([X_train_seq[:K_training_size], X_test_seq])
X_all_dm = np.vstack([X_train_dm[:K_training_size], X_test_dm])

X_all_seq_pfn = np.vstack([X_train_seq, X_test_seq])
X_all_dm_pfn = np.vstack([X_train_dm, X_test_dm])

unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# [5] Model Architecture Design
# --------------------------------------------------------------------------
seq_input = Input(shape=(None, ), dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim, ), dtype='int32', name='domain_input')

x1 = Embedding(input_dim=8001, output_dim=32)(seq_input)
x1 = Convolution1D(filters=64, kernel_size=32, strides=1, padding='valid', activation='relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(0.00001))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.2)(x1)
seq_feat = Dense(16, activation='relu')(x1)

x2 = Embedding(input_dim=domain_emb_dim, output_dim=32, input_length=dm_dim, mask_zero=True)(domain_input)
domain_feat = LSTM(16)(x2)

concat = concatenate([seq_feat, domain_feat])
feature_model = Model([seq_input, domain_input], concat)

x = Dense(16, kernel_regularizer=regularizers.l2(0.00001))(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer = Lambda(lambda a: K.l2_normalize(a, axis=1), name="embedding_out")(x)
prediction_layer = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001))(embedding_layer)

base_model = Model(inputs=[seq_input, domain_input], outputs=[embedding_layer, prediction_layer])
classification_model = Model(inputs=[seq_input, domain_input], outputs=prediction_layer)

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
# PHASE 0: Initial Untrained State
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 0: Extracting Initial Embeddings (Untrained)")
print("="*50)
save_phase_results("phase0_initial_untrained", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 1: Pure Metric Learning (Triplet 1.0)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 1: Metric Learning (Triplet Only)")
print("="*50)
triplet_model.compile(loss=[identity_loss, 'binary_crossentropy'], loss_weights=[1.0, 0.0], optimizer=adam)

for epoch in range(5):
    print(f'Phase 1 - Epoch: {epoch + 1}/5')
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    epoch_losses = [] 
    for step in range(int(len(X_train_seq_upsmp) / 64)):
        batch, labels = get_triplet_batch(X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, tup_gp)
        if batch: 
            losses = triplet_model.train_on_batch(batch, [np.zeros((len(labels), 1)), labels])
            epoch_losses.append(losses)
    
    avg_loss = np.mean(epoch_losses, axis=0)
    print(f"  -> [Metric] Total Loss: {avg_loss[0]:.4f} | Triplet Loss: {avg_loss[1]:.4f}")

save_phase_results("phase1_triplet_only", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 2: Joint Base Learning (Triplet + Focal)
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 2: Joint Base Learning (Triplet + Focal)")
print("="*50)

triplet_model.compile(
    loss=[identity_loss, binary_focal_loss(gamma=2.0, alpha=0.25)], 
    loss_weights=[0.1, 1.0], 
    optimizer=adam
)

for epoch in range(5):
    print(f'Phase 2 - Epoch: {epoch + 1}/5')
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    epoch_losses = []
    for step in range(int(len(X_train_seq_upsmp) / 64)):
        batch, labels = get_triplet_batch(X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, tup_gp)
        if batch: 
            losses = triplet_model.train_on_batch(batch, [np.zeros((len(labels), 1)), labels])
            epoch_losses.append(losses)
            
    avg_loss = np.mean(epoch_losses, axis=0)
    print(f"  -> [Joint] Total: {avg_loss[0]:.4f} | Triplet: {avg_loss[1]:.4f} | Focal: {avg_loss[2]:.4f}")

save_phase_results("phase2_joint_base", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)

# ==========================================================================
# PHASE 3: Focal Loss + PFN Bridge + CRF Integration
# ==========================================================================
print("\n" + "="*50)
print(">>> PHASE 3: Focal Loss Classification + PFN + CRF")
print("="*50)

classification_model.compile(loss=binary_focal_loss(gamma=2.0, alpha=0.25), optimizer=adam, metrics=['accuracy'])
pfn_bridge = DiffusePFNLayerBridge()
theta = np.array([1.0, 1.0])
phase3_epochs = 10

for epoch in range(phase3_epochs):
    print(f'\nPhase 3 (Focal/PFN/CRF) - Epoch: {epoch + 1}/{phase3_epochs}')

    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    p3_losses = []
    p3_accs = []
    
    for e in range(2):
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
            
            hist = classification_model.fit([X_batch_seq[mixed_idx], X_batch_dm[mixed_idx]], y_batch[mixed_idx], batch_size=bs, epochs=1, verbose=0)
            
            p3_losses.append(hist.history['loss'][0])
            acc_key = 'accuracy' if 'accuracy' in hist.history else 'acc' 
            p3_accs.append(hist.history[acc_key][0])

    print(f"  -> [Focal Training] Avg Loss: {np.mean(p3_losses):.4f} | Avg Acc: {np.mean(p3_accs):.4f}")

    print("  Extracting embeddings for CRF...")
    X_all_emb = np.zeros((X_all_seq.shape[0], 32))
    tup_idx_all, tup_gp_all = make_batch(X_all_seq)
    for key in tup_gp_all.keys():
        sel = tup_gp_all[key]
        batch_indices = tup_idx_all[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_indices) / 1000) + 1):
            sub_seq = X_all_seq[batch_indices[1000 * i: 1000 * (i + 1)], seq_dim - sel[2]: seq_dim]
            sub_dm = X_all_dm[batch_indices[1000 * i: 1000 * (i + 1)]]
            if len(sub_seq) > 0:
                X_all_emb[batch_indices[1000 * i: 1000 * (i + 1)]] = feature_model.predict_on_batch([sub_seq, sub_dm])

    print("  Extracting embeddings for PFN...")
    X_pfn_emb = np.zeros((X_all_seq_pfn.shape[0], 32))
    tup_idx_pfn, tup_gp_pfn = make_batch(X_all_seq_pfn)
    for key in tup_gp_pfn.keys():
        sel = tup_gp_pfn[key]
        batch_indices = tup_idx_pfn[sel[0]: sel[1] + 1]
        for i in range(int(len(batch_indices) / 1000) + 1):
            sub_seq = X_all_seq_pfn[batch_indices[1000 * i: 1000 * (i + 1)], seq_dim - sel[2]: seq_dim]
            sub_dm = X_all_dm_pfn[batch_indices[1000 * i: 1000 * (i + 1)]]
            if len(sub_seq) > 0:
                X_pfn_emb[batch_indices[1000 * i: 1000 * (i + 1)]] = feature_model.predict_on_batch([sub_seq, sub_dm])

    print("  Running PFN Inference...")
    pfn_bridge.fit_support_set(X_pfn_emb[:len(y_train)], y_train)
    pfn_score_all = pfn_bridge.predict_proba_bridge(X_pfn_emb)

    initial_score_all = np.concatenate([
        pfn_score_all[:K_training_size],
        pfn_score_all[len(y_train):]
    ])

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
output_file = os.path.join(SAVE_DIR, f"{VER_TAG}_{safe_go}_Final_Integrated_scores.txt")
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' + str(pos_prob_crf[K_training_size + i]) + '\n')

base_model.save_weights(os.path.join(SAVE_DIR, f"{VER_TAG}_{safe_go}_Integrated_BaseModel_weights.h5"))
np.save(os.path.join(SAVE_DIR, f"{VER_TAG}_{safe_go}_Integrated_CRF_weights.npy"), theta)
print("All Process Done! Results saved to " + SAVE_DIR)
