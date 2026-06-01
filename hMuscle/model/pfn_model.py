# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import subprocess
import sys
import time
from sys import argv

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Convolution1D, Embedding
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers, losses, optimizers
from sklearn.metrics import roc_auc_score

from crf import CRF
from layer.PyramidPooling import PyramidPooling
from utils import generate_label, upsample, make_batch

# --------------------------------------------------------------------------
# [Bridge Class] Python 2 -> Call Python 3 Script
# --------------------------------------------------------------------------
class DiffusePFNLayerBridge:
    def __init__(self):
        self.x_support = None
        self.y_support = None

    def fit_support_set(self, x_train, y_train):
        # Support Set Balancing (Positive:Negative = 1:1 or similar)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        
        n_pos = len(pos_idx)
        # Negative 샘플링 (너무 적으면 에러날 수 있으므로 체크)
        if len(neg_idx) > n_pos:
            neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
            
        balanced_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(balanced_idx)
        
        self.x_support = x_train[balanced_idx]
        self.y_support = y_train[balanced_idx]

    def predict_proba_bridge(self, x_query):
        """
        Python 3 Worker를 호출하여 예측 수행
        """
        print "[Bridge] Saving temp files for Python 3 worker..."
        np.save('temp_x_support.npy', self.x_support)
        np.save('temp_y_support.npy', self.y_support)
        np.save('temp_x_query.npy', x_query)
       
        print "[Bridge] Calling Python 3 PFN Worker..."
       
        python3_path = "/usr/bin/python3.8"

        cmd = "%s pfn_worker.py" % python3_path

        exit_code = os.system(cmd)

        if exit_code != 0:
            print "[Bridge Error] Python 3 worker failed! Check pfn_worker.py or logs."
            return np.zeros(len(x_query)) + 0.5
            
        if os.path.exists('temp_pred_result.npy'):
            preds = np.load('temp_pred_result.npy')
            # 임시 파일 삭제 (선택 사항)
            # os.remove('temp_x_support.npy')
            # os.remove('temp_y_support.npy')
            # os.remove('temp_x_query.npy')
            # os.remove('temp_pred_result.npy')
            return preds
        else:
            print "[Bridge Error] Result file not found!"
            return np.zeros(len(x_query))

# --------------------------------------------------------------------------
# Setup & Data Loading
# --------------------------------------------------------------------------
try:
    script, selected_go = argv
except ValueError:
    selected_go = "GO:0000000"
    print "Warning: No arguments provided."

def run_crf(score_map, bag_label, bag_index, co_exp_net, training_size, testing_size, theta, sigma = 10):
    bag_label = bag_label[0: training_size]
    bag_index = bag_index[0: training_size]
    positive_unary_energy = 1 - score_map

    crf = CRF(training_size, testing_size, positive_unary_energy, co_exp_net, theta, bag_label, bag_index)
    label_update, pos_prob_crf, unary_potential, pairwise_potential = crf.inference(10)
    theta_prime = crf.parameter_learning(label_update, theta, sigma)
    return label_update, theta_prime, pos_prob_crf, unary_potential, pairwise_potential

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
    fr = open('../data/raw_data/data/annotations/human_annotations.txt', 'r')
    while True:
        line = fr.readline()
        if not line: break
        line = line.strip().split('\t')
        if selected_go in line[1:]:
            positive_set.append(line[0])
    fr.close()
    
    fr = open('../data/raw_data/data/annotations/swissprot_annotations.txt')
    while True:
        line = fr.readline()
        if not line: break
        line = line.strip().split('\t')
        if selected_go in line[1:]:
            positive_set.append(line[0])
    fr.close()
    return positive_set

# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
print 'Training model for ' + selected_go

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
# generate_label Args: X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm, X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene
y_train, y_test, y_all, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm = generate_label(
    X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm, 
    X_train_geneid, [], X_test_geneid, positive_Gene
)

# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------
seq_input = Input(shape=(None, ), dtype='int32', name='seq_input')
x1 = Embedding(input_dim = 8001, output_dim = 32)(seq_input)
x1 = Convolution1D(filters = 64, kernel_size = 32, strides = 1, padding='valid', activation='relu')(x1)
x1 = PyramidPooling([1, 2, 4, 8])(x1)
x1 = Dense(32, kernel_regularizer=regularizers.l2(0.00001))(x1)
x1 = Activation('relu')(x1)
seq_output = Dense(16)(x1) # Feature extraction point

domain_input = Input(shape=(dm_dim, ), dtype='int32', name='domain_input')
x2 = Embedding(input_dim = domain_emb_dim, output_dim = 32, input_length = dm_dim, mask_zero = True)(domain_input)
domain_output = LSTM(16)(x2)

# Features for PFN
concat_features = keras.layers.concatenate([seq_output, domain_output])

# Backbone Classifier (Used ONLY for Representation Learning)
x = Dense(16)(concat_features)
x = Activation('relu')(x)
x = Dense(1)(x)
output = Activation('sigmoid')(x)

model = Model(inputs=[seq_input, domain_input], outputs=output)
feature_model = Model(inputs=[seq_input, domain_input], outputs=concat_features)

adam = optimizers.Adam(lr = 0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Initialize PFN Bridge
pfn_bridge = DiffusePFNLayerBridge()

nb_epoch = 5
theta = np.array([1.0, 1.0])
unused_flag = np.zeros(y_train.shape[0])
X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# Main Loop: Train -> Extract -> PFN -> CRF
# --------------------------------------------------------------------------
for epoch in range(nb_epoch):
    print 'epoch:', epoch
    
    # [Step 1] Train Backbone (Keras)
    # 기존 model_fixed.py의 배치 로직 복원
    tup_idx, tup_gp = make_batch(X_train_seq_upsmp)
    for e in range(3): # Epoch augmentation
        for key in tup_gp.keys():
            sel = tup_gp[key]
            st, ed, le = sel[0], sel[1], sel[2]
            
            # Batch slicing
            idx = tup_idx[st: ed + 1]
            X_batch_seq = X_train_seq_upsmp[idx, seq_dim - min(le, seq_dim): seq_dim]
            X_batch_dm = X_train_dm_upsmp[idx]
            y_batch = y_train_upsmp[idx]
            
            # Positive/Negative Balancing in batch
            pos_idx = np.where(y_batch == 1)[0]
            neg_idx = np.where(y_batch == 0)[0]
            mixed_idx = np.hstack((pos_idx, neg_idx))
            if len(mixed_idx) == 0: continue
            np.random.shuffle(mixed_idx)
            
            # Dynamic Batch Size
            bs = 1024 if key == 0 else (512 if key == 1 else 256)
            
            model.fit([X_batch_seq[mixed_idx], X_batch_dm[mixed_idx]], y_batch[mixed_idx], batch_size=bs, epochs=1, verbose=1)

    # [Step 2] Extract Embeddings (All Data)
    print "  Extracting embeddings for PFN..."
    
    # 순서를 보장하기 위해 미리 배열 할당 (Sequence + Domain feature dim = 16 + 16 = 32)
    # X_all_seq 순서대로 채워넣기 위해 make_batch 로직을 그대로 사용해야 함
    
    X_all_emb = np.zeros((X_all_seq.shape[0], 32)) 
    tup_idx_all, tup_gp_all = make_batch(X_all_seq)
    
    for key in tup_gp_all.keys():
        sel = tup_gp_all[key]
        st, ed, le = sel[0], sel[1], sel[2]
        
        batch_indices = tup_idx_all[st: ed + 1] # 실제 데이터의 위치 인덱스
        X_batch_seq = X_all_seq[batch_indices]
        X_batch_dm = X_all_dm[batch_indices]
        
        # 1000개씩 끊어서 예측 (메모리 보호)
        for i in range(int(len(X_batch_seq) / 1000) + 1):
            sub_seq = X_batch_seq[1000 * i: 1000 * (i + 1), seq_dim - le: seq_dim]
            sub_dm = X_batch_dm[1000 * i : 1000 * (i + 1)]
            
            if len(sub_seq) == 0: continue
            
            # Feature Model Prediction
            embs = feature_model.predict_on_batch([sub_seq, sub_dm])
            
            # 결과 저장 (원본 인덱스 위치에)
            current_indices = batch_indices[1000 * i : 1000 * (i + 1)]
            X_all_emb[current_indices] = embs

    # [Step 3] PFN Inference via Bridge
    print "  Running PFN Inference..."
    # Training Set 부분만 잘라서 Support Set으로 제공
    X_train_emb = X_all_emb[:K_training_size]
    
    pfn_bridge.fit_support_set(X_train_emb, y_train) # y_train is current iterative label
    initial_score_all = pfn_bridge.predict_proba_bridge(X_all_emb)
    
    # [Step 4] CRF Update
    print "  Running CRF Update..."
    y_train[0: K_training_size], theta, pos_prob_crf, _, _ = run_crf(initial_score_all, y_all, crf_bag_index, co_exp_net, K_training_size, K_testing_size, theta, sigma=0.1)
    
    # Upsampling for next epoch
    if epoch < nb_epoch - 1:
         X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag = upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag)

# --------------------------------------------------------------------------
# Final Output (Testing)
# --------------------------------------------------------------------------
print "Saving Results..."
safe_go = selected_go.replace(':', '_')
output_file = '../results/' + safe_go + '_PFN_scores.txt'

fw = open(output_file, 'w')
for i in range(K_testing_size):
    # GeneID \t IsoformID \t Probability
    fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' + str(pos_prob_crf[K_training_size + i]) + '\n')
fw.close()

print "Done. Saved to " + output_file
