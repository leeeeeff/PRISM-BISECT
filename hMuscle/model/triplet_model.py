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
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Convolution1D, Embedding, Lambda, Flatten, concatenate
from keras import backend as K
from keras import regularizers, losses, optimizers
from sklearn.metrics import roc_auc_score

# 사용자 라이브러리 (경로 확인 필요)
from crf import CRF
from layer.PyramidPooling import PyramidPooling
from utils import generate_label, upsample, make_batch

# --------------------------------------------------------------------------
# [Setup] Argument Parsing & Directory
# --------------------------------------------------------------------------
# [핵심] TensorFlow 세션 설정
config = tf.ConfigProto()

# 1. CPU 코어 개수 제한 (가장 중요!)
# 서버 전체 코어가 많더라도 이 프로세스는 4개만 쓰도록 강제합니다.
config.intra_op_parallelism_threads = 4 
config.inter_op_parallelism_threads = 4

# 2. GPU 메모리 관련 (GPU가 없더라도 에러 방지를 위해 설정)
config.gpu_options.allow_growth = True

# 3. 설정된 세션을 Keras에 적용
session = tf.Session(config=config)
K.set_session(session)

try:
    script, selected_go = argv
except ValueError:
    print("Usage: python triplet_model.py <GO_TERM>")
    sys.exit(1)

# 결과 저장 경로 설정
safe_go = selected_go.replace(':', '_')
SAVE_DIR = '../results/' + safe_go

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --------------------------------------------------------------------------
# [Helper Functions] Loss, Data Gen, Analysis
# --------------------------------------------------------------------------
def triplet_loss(inputs, margin=1.0):
    """
    Inputs: [anchor_embedding, positive_embedding, negative_embedding]
    L = max(d(A, P) - d(A, N) + margin, 0)
    """
    anchor, positive, negative = inputs
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = K.maximum(basic_loss, 0.0)
    return loss

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def get_triplet_batch(X_seq, X_dm, y, tup_gp, batch_size=64):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    if len(pos_indices) < 2 or len(neg_indices) == 0:
        return None, None # Return None tuple safely

    anchors_seq, anchors_dm = [], []
    positives_seq, positives_dm = [], []
    negatives_seq, negatives_dm = [], []
    labels = []

    while len(anchors_seq) < batch_size:
        a_idx = np.random.choice(pos_indices)
        p_idx = np.random.choice(pos_indices)
        while p_idx == a_idx:
            p_idx = np.random.choice(pos_indices)
        n_idx = np.random.choice(neg_indices)

        anchors_seq.append(X_seq[a_idx])
        anchors_dm.append(X_dm[a_idx])
        positives_seq.append(X_seq[p_idx])
        positives_dm.append(X_dm[p_idx])
        negatives_seq.append(X_seq[n_idx])
        negatives_dm.append(X_dm[n_idx])
        labels.append(1)

    return [np.array(anchors_seq), np.array(anchors_dm),
            np.array(positives_seq), np.array(positives_dm),
            np.array(negatives_seq), np.array(negatives_dm)], np.array(labels)

def save_phase_results(phase_name, model_base, X_seq, X_dm, y_true, gene_ids, iso_ids, save_dir):
    """
    Phase별 중간 결과(임베딩, 점수) 저장 함수
    """
    print("\n[Save] Saving results for " + phase_name + "...")

    # Predict: embedding(0), score(1)
    preds = model_base.predict([X_seq, X_dm], batch_size=256, verbose=1)
    embeddings = preds[0]
    scores = preds[1]

    # 1. Save Embeddings (.npy)
    np.save(os.path.join(save_dir, phase_name + '_embeddings.npy'), embeddings)
    np.save(os.path.join(save_dir, phase_name + '_labels.npy'), y_true)

    # 2. Save Scores (.txt)
    score_file = os.path.join(save_dir, phase_name + '_scores.txt')
    with open(score_file, 'w') as fw:
        for i in range(len(y_true)):
            fw.write(gene_ids[i] + '\t' + iso_ids[i] + '\t' + str(scores[i][0]) + '\n')

    print("[Save] Done. Results saved to " + save_dir)

def analyze_predictions(y_true, y_scores, gene_ids, iso_ids, threshold=0.9):
    """
    최종 결과 심층 분석 (True Positive & False Positive 확인)
    """
    print("\n" + "="*70)
    print("📊 [Analysis] High-Confidence Predictions (Threshold > %.2f)" % threshold)
    print("="*70)

    # Case 1: 정답(1)인데 모델도 잘 맞춘 경우
    print("\n✅ 1. Confirmed Positives (Label=1, Correctly Predicted):")
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_scores[i] > threshold:
            print("   - Gene: %s, Isoform: %s, Score: %.4f" % (gene_ids[i], iso_ids[i], y_scores[i]))
            count += 1
            if count >= 10: break

    if count == 0: print("   (None found.)")

    # Case 2: 정답(0)인데 모델은 있다고 우기는 경우 (잠재적 발견)
    print("\n🔍 2. Potential Novel Discoveries (Label=0, But High Score):")
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_scores[i] > threshold:
            print("   - Gene: %s, Isoform: %s, Score: %.4f" % (gene_ids[i], iso_ids[i], y_scores[i]))
            count += 1
            if count >= 10: break

    if count == 0: print("   (None found.)")
    print("="*70)

# --------------------------------------------------------------------------
# [Data Loading]
# --------------------------------------------------------------------------
def load_sequence_data():
    X_train_seq = np.load('../data/raw_data/data/sequences/human_sequence_train.npy')
    X_train_dm = np.load('../data/raw_data/data/domains/human_domain_train.npy')
    X_test_seq = np.load('my_sequence_matrix_fixed.npy')
    X_test_dm = np.load('../results/domain/domain_matrix.npy')
    X_train_geneid = np.load('../data/raw_data/data/id_lists/train_gene_list.npy')
    X_test_geneid = np.load('my_gene_list_fixed.npy')
    X_test_isoid = np.load('my_isoform_list_fixed.npy')
    X_train_other_seq = np.load('../data/raw_data/data/sequences/swissprot_sequence_train.npy')
    X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
    return X_train_seq, X_train_dm, X_test_seq, X_test_dm, X_train_geneid, X_test_geneid, X_test_isoid, X_train_other_seq, X_train_other_dm

def pos_gene_set(selected_go):
    positive_set = []
    try:
        fr = open('../data/raw_data/data/annotations/human_annotations.txt', 'r')
        while True:
            line = fr.readline()
            if not line: break
            line = line.strip().split('\t')
            if selected_go in line[1:]:
                positive_set.append(line[0])
        fr.close()
        
        fr = open('../data/raw_data/data/annotations/swissprot_annotations.txt', 'r')
        while True:
            line = fr.readline()
            if not line: break
            line = line.strip().split('\t')
            if selected_go in line[1:]:
                positive_set.append(line[0])
        fr.close()
    except IOError:
        print("Annotation file not found.")
    return positive_set

def run_crf(score_map, bag_label, bag_index, co_exp_net, training_size, testing_size, theta, sigma=10):
    bag_label = bag_label[0: training_size]
    bag_index = bag_index[0: training_size]
    positive_unary_energy = 1 - score_map
    crf = CRF(training_size, testing_size, positive_unary_energy, co_exp_net, theta, bag_label, bag_index)
    label_update, pos_prob_crf, unary_potential, pairwise_potential = crf.inference(10)
    theta_prime = crf.parameter_learning(label_update, theta, sigma)
    return label_update, theta_prime, pos_prob_crf, unary_potential, pairwise_potential

# --------------------------------------------------------------------------
# [Main Execution] Data Prep
# --------------------------------------------------------------------------
print 'Training Triplet Model for ' + selected_go

positive_Gene = pos_gene_set(selected_go)
co_exp_net = np.load('../results/co-expression_net/coexp_net_bridged.npy')
X_train_seq, X_train_dm, X_test_seq, X_test_dm, X_train_geneid, X_test_geneid, X_test_isoid, X_train_other_seq, X_train_other_dm = load_sequence_data()

K_training_size = X_train_seq.shape[0]
K_testing_size = X_test_seq.shape[0]
X_all_seq = np.vstack([X_train_seq, X_test_seq])
X_all_dm = np.vstack([X_train_dm, X_test_dm])

seq_dim = X_train_seq.shape[1]
dm_dim = X_train_dm.shape[1]
domain_emb_dim = max([np.max(X_train_dm), np.max(X_test_dm)]) + 1

print 'Generating initial label...'
y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm = generate_label(
    X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm,
    X_train_geneid, [], X_test_geneid, positive_Gene
)

# --------------------------------------------------------------------------
# [Model Architecture]
# --------------------------------------------------------------------------
# 1. Base Encoder & Predictor
seq_input = Input(shape=(None, ), dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim, ), dtype='int32', name='domain_input')

# CNN Part
x1 = Embedding(input_dim = 8001, output_dim = 32)(seq_input)
x1 = Convolution1D(filters = 64, kernel_size = 32, strides = 1, padding='valid', activation='relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(0.00001))(x1)
x1 = Activation('relu')(x1)
x1 = Dropout(0.2)(x1)
seq_feat = Dense(16)(x1)

# LSTM Part
x2 = Embedding(input_dim = domain_emb_dim, output_dim = 32, input_length = dm_dim, mask_zero = True)(domain_input)
domain_feat = LSTM(16)(x2)

# Concatenate -> Final Embedding
concat = concatenate([seq_feat, domain_feat])
dense_output = Dense(16, activation="relu")(concat)

# [L2 Normalize]
embedding_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='embedding')(dense_output)

# Prediction Head
prediction_layer = Dense(1, activation='sigmoid', name='prediction')(embedding_layer)

# Create Base Model (For Saving & Inference)
base_model = Model(inputs=[seq_input, domain_input], outputs=[embedding_layer, prediction_layer])

# 2. Triplet Network (For Training)
seq_a, dm_a = Input(shape=(None,), dtype='int32'), Input(shape=(dm_dim,), dtype='int32')
seq_p, dm_p = Input(shape=(None,), dtype='int32'), Input(shape=(dm_dim,), dtype='int32')
seq_n, dm_n = Input(shape=(None,), dtype='int32'), Input(shape=(dm_dim,), dtype='int32')

emb_a, pred_a = base_model([seq_a, dm_a])
emb_p, _      = base_model([seq_p, dm_p])
emb_n, _      = base_model([seq_n, dm_n])

triplet_loss_layer = Lambda(triplet_loss, output_shape=(1,), name='triplet_loss')([emb_a, emb_p, emb_n])

model = Model(inputs=[seq_a, dm_a, seq_p, dm_p, seq_n, dm_n], outputs=[triplet_loss_layer, pred_a])

inference_model = Model(inputs=[seq_input, domain_input], outputs=prediction_layer)

# --------------------------------------------------------------------------
# [Training Strategy] Setup
# --------------------------------------------------------------------------
adam = optimizers.Adam(lr = 0.001)
theta = np.array([1.0, 1.0])
unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# ==========================================================================
# PHASE 1: Pure Metric Learning (Triplet Only)
# ==========================================================================
print("\n" + "="*50)
print(">>> STARTING PHASE 1: Metric Learning (Triplet Only)")
print("="*50)

# Triplet=1.0, BCE=0.0
model.compile(loss=[identity_loss, 'binary_crossentropy'], 
              loss_weights=[1.0, 0.0], 
              optimizer=adam, 
              metrics=['accuracy'])

phase1_epochs = 10
for epoch in range(phase1_epochs):
    print('\nPhase 1 - Epoch: {}/{}'.format(epoch + 1, phase1_epochs))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    steps_per_epoch = len(X_train_seq_upsmp) / 64
    epoch_logs = []

    for step in range(int(steps_per_epoch)):
        # (사용자님이 작성하신 실시간 로그 코드 시작)
        triplet_batch, labels = get_triplet_batch(X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, tup_gp, batch_size=64)
        if triplet_batch is None: continue

        dummy_y = np.zeros((len(labels), 1))
        logs = model.train_on_batch(triplet_batch, [dummy_y, labels])
        
        if len(epoch_logs) == 0:
            epoch_logs = [0.0] * len(logs)
        for i in range(len(logs)):
            epoch_logs[i] += logs[i]

        if step % 10 == 0:
            log_str = "Step: %d/%d | " % (step, int(steps_per_epoch))
            for name, val in zip(model.metrics_names, logs):
                log_str += "%s: %.4f " % (name, val)
            print(log_str)
        # (사용자님이 작성하신 실시간 로그 코드 끝)

    print("\n" + "-"*30)
    final_avg = "  [Final Epoch Result] "
    for name, val in zip(model.metrics_names, epoch_logs):
        final_avg += "%s: %.4f | " % (name, val / steps_per_epoch)
    print(final_avg)
    
    import gc
    gc.collect()


# [SAVE PHASE 1]
save_phase_results("phase1_triplet_only", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR)

# ==========================================================================
# PHASE 2: Joint Learning (Triplet + BCE)
# ==========================================================================
print("\n" + "="*50)
print(">>> STARTING PHASE 2: Joint Learning (Triplet + BCE)")
print("="*50)

# Triplet=0.1, BCE=1.0
model.compile(loss=[identity_loss, 'binary_crossentropy'],
              loss_weights=[0.1, 1.0],
              optimizer=adam,
              metrics=['accuracy'])

phase2_epochs = 10

for epoch in range(phase2_epochs):
    print('\nPhase 2 - Epoch: {}/{}'.format(epoch + 1, phase2_epochs))
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    steps_per_epoch = len(X_train_seq_upsmp) / 64
    epoch_logs = []

    for step in range(int(steps_per_epoch)):
        # (사용자님이 작성하신 실시간 로그 코드 시작)
        triplet_batch, labels = get_triplet_batch(X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, tup_gp, batch_size=64)
        if triplet_batch is None: continue

        dummy_y = np.zeros((len(labels), 1))
        logs = model.train_on_batch(triplet_batch, [dummy_y, labels])

        if len(epoch_logs) == 0:
            epoch_logs = [0.0] * len(logs)
        for i in range(len(logs)):
            epoch_logs[i] += logs[i]

        if step % 10  == 0:
            log_str = "Step: %d/%d | " % (step, int(steps_per_epoch))
            for name, val in zip(model.metrics_names, logs):
                log_str += "%s: %.4f " % (name, val)
            print(log_str)
        # (사용자님이 작성하신 실시간 로그 코드 끝)

    print("\n" + "-"*30)
    final_avg = "  [Final Epoch Result] "
    for name, val in zip(model.metrics_names, epoch_logs):
        final_avg += "%s: %.4f | " % (name, val / steps_per_epoch)
    print(final_avg)
  

    # CRF Inference
    print("  Predicting for CRF...")
    initial_score_all = []
    tup_idx_all, tup_gp_all = make_batch(X_all_seq)
    all_preds = np.zeros(X_all_seq.shape[0])

    for key in tup_gp_all.keys():
        sel = tup_gp_all[key]
        st, ed, le = sel[0], sel[1], sel[2]
        batch_indices = tup_idx_all[st: ed + 1]
        X_batch_seq = X_all_seq[batch_indices]
        X_batch_dm = X_all_dm[batch_indices]

        for i in range(int(len(X_batch_seq) / 1000) + 1):
            sub_seq = X_batch_seq[1000 * i: 1000 * (i + 1), seq_dim - le: seq_dim]
            sub_dm = X_batch_dm[1000 * i : 1000 * (i + 1)]
            if len(sub_seq) == 0: continue

            p = inference_model.predict_on_batch([sub_seq, sub_dm])
            all_preds[batch_indices[1000*i : 1000*(i+1)]] = p.flatten()

    initial_score_all = all_preds

    # CRF Update
    y_train[0: K_training_size], theta, pos_prob_crf, _, _ = run_crf(initial_score_all, y_all, crf_bag_index, co_exp_net, K_training_size, K_testing_size, theta, sigma=0.1)

    # Upsample (Variable name fixed!)
    if epoch < phase2_epochs - 1:
        X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

    gc.collect()

# [SAVE PHASE 2]
save_phase_results("phase2_joint_final", base_model, X_test_seq, X_test_dm, y_test, X_test_geneid, X_test_isoid, SAVE_DIR)

# --------------------------------------------------------------------------
# [Final Results Saving & Analysis]
# --------------------------------------------------------------------------
# 1. Save Final CRF Scores
print("Saving Final CRF Refined Results...")
output_file = os.path.join(SAVE_DIR, safe_go + '_Triplet_prediction_scores.txt')

with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        final_score = str(pos_prob_crf[K_training_size + i])
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' + final_score + '\n')

print("Done. Saved to " + output_file)

# 2. Save Triplet Model Weights
model.save_weights('../saved_models/' + safe_go + '_Triplet_Final_weights.h5')

# 3. Final Analysis (Confirmed vs Novel)
final_test_scores = pos_prob_crf[K_training_size : K_training_size + K_testing_size]

analyze_predictions(
    y_true=y_test,
    y_scores=final_test_scores,
    gene_ids=X_test_geneid,
    iso_ids=X_test_isoid,
    threshold=0.85
)

print("All Process Done.")
