# -*- coding: utf-8 -*-
# ============================================================================
# v7a_integrated_full_model.py  (v6g + Supervised Contrastive Loss)
#
# v6g 대비 핵심 변경:
# [v7a-1] Phase 1 손실: Triplet → Supervised Contrastive Loss (SupCon)
#   수식: L_i = log Σ_{a≠i} exp(z_i·z_a/τ) − (1/|P(i)|) Σ_{p∈P(i)} z_i·z_p/τ
#   τ = 0.1, P(i) = 배치 내 positive 샘플 (자신 제외)
#   근거: Triplet은 (a,p,n) 1쌍 vs SupCon은 배치 내 전체 positive 동시 학습
#         [R3.3]: Triplet active ratio < 5% → SupCon 교체 검토
#   ablation Exp 2a: v7a-SupCon vs v6g | acceptance: Δ Macro-AUPRC > +0.02
#
# v6g 변경사항 유지:
# [v6g-1] Phase 1.5 제거 (Phase 1 → Phase 2 직행)
#
# v6f 변경사항 유지:
# [v6f-1] Type-B Phase 2 LR: 0.0002 | [v6f-2] patience=10, max_epochs=25
# [v6f-3] TARGET_COVERAGE=6          | [v6f-4] NO_IMPROVE_LIMIT 변수 사용
#
# v6e 원본 변경사항 (유지):
# [v6e-1] Phase 0 GO term Type 자동 분류 (sep_ratio < 1.15 → Type-B)
# [v6e-2] Adaptive Phase 2 lr/clipnorm/patience
# [v6e-3] Label Propagation 제거 (alpha=0.0 고정)
# [v6e-4] coverage clip: min=10, max=80 (min=20→10 개선, max=50→80 개선)
#
# v6d 원본:
# [v6d] Domain Delta Branch (Alt 1)
#   domain_delta[i] = sign(domain_matrix[i] - domain_matrix[canonical_gene_i])
#   이소폼-특이적 Pfam 도메인 gain/loss 인코딩
#   Gene-level reference dominance 극복 [R2.1]
#   canonical: gene 내 Pfam domain count 최대 isoform
#   dd_feat[16] -> concat에 추가 -> EMB_DIM 112 -> 128
#
# v6h 원본 내용:
# [HYBRID] Modality-Role Separation
#   문제: v6 Phase 2 triplet이 ESM-2 임베딩 공간을 재조직 → Phase 1.5 calibration 파괴
#         → GO:0007204 AUPRC 0.508→0.300, GO:0030017 0.390→0.211 회귀
#   진단: Phase 2 Focal loss 소멸(0.003→0.000), Triplet만 작동 → score drift
#
#   해결 원칙: 모달리티별 학습 역할 분리
#     ESM-2:  전역 진화 맥락 → Phase 1 학습 완료 후 동결
#     CNN:    위치 특이적 로컬 모티프 → Phase 2에서만 학습
#     Domain: GO term 특이적 도메인 조합 → 전 구간 학습
#
# [ARCH] 모델 구조 (v6d)
#   ESM-2 branch : 640 → Dense(256,relu) → Dense(128,relu) → Dense(64,relu)
#                  → gate(×mask) → esm2_gated[64]
#   CNN   branch : Embedding(8001,32) → Conv1D(64,k=7) → Conv1D(32,k=5)
#                  → GlobalMaxPool → Dense(32,relu) → cnn_feat[32]
#   Domain branch: Embedding → LSTM(16) → domain_feat[16]
#   DomainDelta  : Input(251,sign) → Dense(64,relu) → Dense(16,relu) → dd_feat[16] [v6d NEW]
#   Fusion       : concat[64+32+16+16=128] → Dense(64,relu) → L2_norm → emb[64]
#                  → Dense(1,sigmoid)
#   EMB_DIM=128
#
# [TRAIN] 단계별 가중치 갱신 범위
#   Phase 1  (triplet, GradientTape): ESM-2 ✅ | CNN ❌(frozen) | Domain ✅ | DeltaBranch ✅
#   Phase 1.5(focal, encoder frozen): ESM-2 ❌ | CNN ❌         | Domain ❌  | DeltaBranch ❌
#   Phase 2  (focal only):            ESM-2 ❌(frozen) | CNN ✅ | Domain ✅  | DeltaBranch ✅
#   Phase 3  (label propagation):     학습 없음
#
# [SEQ] 서열 입력 처리
#   SEQ_LEN=1500 (p95 커버, 6000→last 1500 truncation)
#   make_batch 불필요 — GlobalMaxPool로 패딩 처리
#
# [PIPE] 데이터 파이프라인
#   generate_label 3회 호출 (deterministic):
#     ① dm용    ② esm2+mask용   ③ seq용
#   upsample 1회: hstack(esm2[640]+mask[1]+seq[1500]+dm) → 통합 처리
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

# [v7a-1] SupCon Phase 1 함수 임포트
from prototype_contrastive import (
    supervised_contrastive_loss,
    phase1_supcon_epoch_hybrid,
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
    if len(argv) == 3:
        script, selected_go, SUPCON_TAU_ARG = argv
        SUPCON_TAU_ARG = float(SUPCON_TAU_ARG)
    elif len(argv) == 2:
        script, selected_go = argv
        SUPCON_TAU_ARG = None   # 기본값은 아래 SUPCON_TAU에서 설정
    else:
        raise ValueError
except (ValueError, TypeError):
    print("Usage: python v7a_integrated_full_model.py <GO_TERM> [tau]")
    print("  tau: SupCon temperature (default 0.1, try 0.2 or 0.3)")
    sys.exit(1)

safe_go = selected_go.replace(':', '_')
BASE_RESULTS_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/' + safe_go

if not os.path.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)

date_str = datetime.now().strftime("%Y%m%d")
# [v7a] τ 파라미터화: argv[2]로 전달 가능, 없으면 기본값 0.1
_tau_default = 0.1
_tau_val = SUPCON_TAU_ARG if SUPCON_TAU_ARG is not None else _tau_default
_tau_str = "tau{:03d}".format(int(_tau_val * 100))   # e.g. tau010, tau020, tau030
VER_TAG  = "v7a_{}_integrated".format(_tau_str)       # e.g. v7a_tau010_integrated
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
SEQ_LEN   = 1500           # [HYBRID] p95 커버 (6000→last 1500 truncation)
EXPR_DIM  = 24
DD_DIM    = 251             # [v6d] domain_delta sign feature dimension
EMB_DIM   = 128            # esm2_gated[64] + cnn_feat[32] + domain_feat[16] + dd_feat[16]
MARGIN_P1         = 0.3    # [I3]
TARGET_COVERAGE   = 6.0    # [v6f-3] 6.0 원복 (v6e 4.0은 GO:0006096 coverage 50% 감소 유발)
SEP_RATIO_THRESHOLD = 1.15 # [v6e-1] Type A/B 분류 기준 (Phase 0 sep_ratio)

# ESM-2 / CNN / DomainDelta 레이어 이름 (단계별 freeze 제어용)
ESM2_TRAINABLE_NAMES  = {'esm2_d1', 'esm2_d2', 'esm2_feat'}
CNN_TRAINABLE_NAMES   = {'cnn_emb', 'cnn_conv1', 'cnn_conv2', 'cnn_feat'}
DELTA_TRAINABLE_NAMES = {'dd_dense1', 'dd_dense2'}  # [v6d]

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

def set_delta_trainable(model, trainable):
    """[v6d] DomainDelta branch 레이어 trainable 설정"""
    for layer in model.layers:
        if layer.name in DELTA_TRAINABLE_NAMES:
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
# [5-v6h] 임베딩 추출 — 4-modality 입력
# --------------------------------------------------------------------------
def extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd,
                               emb_dim=EMB_DIM, batch_size=512):
    n   = len(X_esm2)
    emb = np.zeros((n, emb_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        raw  = feature_model.predict_on_batch([
            X_esm2[start:end].astype(np.float32),
            X_mask[start:end].astype(np.float32),
            X_seq[start:end],
            X_dm[start:end],
            X_dd[start:end].astype(np.float32)])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        emb[start:end] = raw / np.clip(norms, 1e-8, None)
    return emb

# --------------------------------------------------------------------------
# [EMB] 임베딩 품질 분석 (변경 없음)
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
# [6-v7a] Phase 1 — Supervised Contrastive Loss (SupCon, Triplet 대체)
# --------------------------------------------------------------------------
# [v7a-1] Triplet (a,p,n) 3-way → SupCon 배치 전체 positive 쌍 동시 학습
# phase1_supcon_epoch_hybrid: prototype_contrastive.py에서 임포트
#
# 진단용: Triplet margin stats 제거, SupCon Sep-Ratio로 embedding 품질 추적

def compute_embedding_sep_ratio(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd, y,
                                 n_sample=500):
    """
    [v7a] Phase 1 진단: sep_ratio (inter/intra) 로 임베딩 분리도 추적.
    Triplet의 margin_stats 대체.
    """
    emb     = extract_embeddings_hybrid(feature_model, X_esm2, X_mask, X_seq, X_dm, X_dd)
    pos_emb = emb[y == 1]
    neg_emb = emb[y == 0]
    if len(pos_emb) < 2 or len(neg_emb) < 2:
        return float('nan'), float('nan')
    np.random.seed(42)
    n       = min(n_sample, min(len(pos_emb), len(neg_emb)))
    intra   = [max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[i], pos_emb[j])))
               for i, j in (np.random.choice(len(pos_emb), (n, 2), replace=True))]
    inter   = [max(0.0, 2.0 - 2.0 * float(np.dot(pos_emb[i], neg_emb[j])))
               for i, j in zip(np.random.choice(len(pos_emb), n),
                                np.random.choice(len(neg_emb), n))]
    intra_m = float(np.mean(intra))
    inter_m = float(np.mean(inter))
    sep     = inter_m / (intra_m + 1e-8)
    print("  [SepRatio] intra={:.4f} inter={:.4f} sep={:.4f}".format(intra_m, inter_m, sep))
    return sep, inter_m


def save_phase_results_hybrid(phase_name, model_base, X_esm2, X_mask, X_seq, X_dm, X_dd,
                               y_true, gene_ids, iso_ids, save_dir, ver_tag):
    print("\n[Save] {}...".format(phase_name))
    preds  = model_base.predict([X_esm2, X_mask, X_seq, X_dm, X_dd], batch_size=256, verbose=1)
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
# [7] Phase 3: Test-time Label Propagation (변경 없음)
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
# [8-v6h] 데이터 로딩
# --------------------------------------------------------------------------
print('>>> Preparing Data (v6e/DomainDelta) for ' + selected_go)

ESM2_DATA_DIR = '../data'

def _load_esm2(name):
    path = os.path.join(ESM2_DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError("[ESM] 필수 파일 없음: {}".format(path))
    arr = np.load(path).astype(np.float32)
    print("  [ESM] Loaded {} {}".format(name, arr.shape))
    return arr

# ESM-2 임베딩
X_train_esm2            = _load_esm2('esm2_train_human.npy')        # (31668, 640)
X_train_esm2_mask       = _load_esm2('esm2_train_human_mask.npy')   # (31668, 1)
X_train_other_esm2      = _load_esm2('esm2_train_swissprot.npy')    # (82703, 640)
X_train_other_esm2_mask = _load_esm2('esm2_train_swissprot_mask.npy')
X_test_esm2             = _load_esm2('esm2_embeddings_t30_150M.npy') # (36748, 640)
X_test_esm2_mask        = _load_esm2('esm2_mask.npy')                # (36748, 1)

train_human_cov = float(X_train_esm2_mask.sum()) / len(X_train_esm2_mask) * 100
test_cov        = float(X_test_esm2_mask.sum()) / len(X_test_esm2_mask) * 100
print("[ESM] Coverage — Human train: {:.1f}% | Test: {:.1f}%".format(
    train_human_cov, test_cov))

# 서열 입력 (CNN용) — last SEQ_LEN 위치 사용 (right-aligned, 0-padded on left)
SEQ_PATH_TRAIN  = '../data/raw_data/data/sequences/human_sequence_train.npy'
SEQ_PATH_OTHER  = '../data/raw_data/data/sequences/swissprot_sequence_train.npy'
SEQ_PATH_TEST   = 'my_sequence_matrix_fixed.npy'

X_train_seq       = np.load(SEQ_PATH_TRAIN)[:, -SEQ_LEN:]     # (31668, 1500) int32
X_train_other_seq = np.load(SEQ_PATH_OTHER)[:, -SEQ_LEN:]     # (82703, 1500) int32
X_test_seq        = np.load(SEQ_PATH_TEST)[:, -SEQ_LEN:]      # (36748, 1500) int32
print("  [Seq] train={} other={} test={}".format(
    X_train_seq.shape, X_train_other_seq.shape, X_test_seq.shape))

# Domain feature (LSTM용 정수 인코딩)
X_train_dm       = np.load('../data/raw_data/data/domains/human_domain_train.npy')
X_train_other_dm = np.load('../data/raw_data/data/domains/swissprot_domain_train.npy')
X_test_dm        = np.load('../results/domain/domain_matrix.npy')

# [v6d] Domain Delta (sign transform: {-1, 0, +1})
# train: 유전자 내 canonical 대비 도메인 gain/loss
# swissprot: all zeros (isoform 구조 없음)
# test: 동일 방식으로 계산됨
DD_TRAIN_PATH  = '../results_isoform/features/train_domain_delta_sign.npy'
DD_SW_PATH     = '../results_isoform/features/swissprot_domain_delta_sign.npy'
DD_TEST_PATH   = '../results_isoform/features/domain_delta.npy'
X_train_dd       = np.load(DD_TRAIN_PATH).astype(np.float32)       # (31668, 251)
X_train_other_dd = np.load(DD_SW_PATH).astype(np.float32)           # (82703, 251) zeros
X_test_dd        = np.sign(np.load(DD_TEST_PATH)).astype(np.float32) # (36748, 251) sign
print("  [v6d] domain_delta sign: train={} sw={} test={}".format(
    X_train_dd.shape, X_train_other_dd.shape, X_test_dd.shape))

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
# [9-v6h] 레이블 생성 및 업샘플
#
# generate_label 3회 호출 (모두 deterministic — 동일 geneid → 동일 add_index):
#   ① (esm2, dm)         → labels + X_train_dm_comb
#   ② (esm2, mask)       → X_train_esm2_comb, X_train_mask_comb
#   ③ (esm2, seq)        → _dummy, X_train_seq_comb
#
# upsample 1회: hstack(esm2[640] + mask[1] + seq[1500] + dm) → 통합
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

# ③ seq (esm2를 더미 첫 번째 인수로 사용 — add_index 동일 보장)
_, _, _, _, _, _, _dummy2, X_train_seq_comb = generate_label(
    X_train_esm2, X_train_seq,
    X_train_other_esm2, X_train_other_seq,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

# ④ [v6d] domain_delta (add_index 동일 보장)
_, _, _, _, _, _, _dummy3, X_train_dd_comb = generate_label(
    X_train_esm2, X_train_dd,
    X_train_other_esm2, X_train_other_dd,
    X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene)

n_pos_train = int((y_train == 1).sum())
print("  y_train pos={} neg={} ratio={:.3f}%".format(
    n_pos_train, int((y_train == 0).sum()), n_pos_train / len(y_train) * 100))
print("  y_test  pos={} neg={} ratio={:.3f}%".format(
    (y_test == 1).sum(), (y_test == 0).sum(),
    (y_test == 1).sum() / len(y_test) * 100))

# hstack: [esm2(640) | mask(1) | seq(1500) | dm(dm_dim) | dd(DD_DIM)]
MASK_START = ESM2_DIM                          # 640
SEQ_START  = ESM2_DIM + 1                      # 641
DM_START   = ESM2_DIM + 1 + SEQ_LEN           # 2141
DD_START   = ESM2_DIM + 1 + SEQ_LEN + 0       # placeholder — set after dm_dim known

X_train_combined = np.hstack([
    X_train_esm2_comb,                          # (N, 640)    float32
    X_train_mask_comb,                          # (N, 1)      float32
    X_train_seq_comb.astype(np.float32),        # (N, 1500)   int→float
    X_train_dm_comb.astype(np.float32),         # (N, dm_dim)
    X_train_dd_comb.astype(np.float32),         # (N, DD_DIM) [v6d]
])  # (N, 2141 + dm_dim + DD_DIM)

DD_START = DM_START + X_train_dm_comb.shape[1]  # 2141 + dm_dim

np.random.seed(42)  # [Fix-DA] 재현성 고정
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
X_train_dm_upsmp   = X_combined_upsmp[:, DM_START:DD_START].astype(np.int32)
X_train_dd_upsmp   = X_combined_upsmp[:, DD_START:].astype(np.float32)  # [v6d]

print("  Upsampled: esm2={} mask={} seq={} dm={} dd={} y={}".format(
    X_train_esm2_upsmp.shape, X_train_mask_upsmp.shape,
    X_train_seq_upsmp.shape, X_train_dm_upsmp.shape,
    X_train_dd_upsmp.shape, y_train_upsmp.shape))

# [I4] 동적 n_batches — [v6e-4] 균등 coverage (min=10, max=80)
BATCH_SIZE_P1 = 256
N_BATCHES_P1  = int(np.clip(
    np.ceil(n_pos_train * TARGET_COVERAGE / BATCH_SIZE_P1), a_min=10, a_max=80))
coverage_per_epoch = N_BATCHES_P1 * BATCH_SIZE_P1 / n_pos_train
print("[I4][v6e] n_batches={} (n_pos={}, coverage={:.1f}x, target={:.1f}x)".format(
    N_BATCHES_P1, n_pos_train, coverage_per_epoch, TARGET_COVERAGE))

# --------------------------------------------------------------------------
# [10-v6h] 모델 구조
# --------------------------------------------------------------------------
esm2_input   = Input(shape=(ESM2_DIM,), name='esm2_input')
esm2_mask_in = Input(shape=(1,),         name='esm2_mask')
seq_input    = Input(shape=(SEQ_LEN,),  dtype='int32', name='seq_input')
domain_input = Input(shape=(dm_dim,),   dtype='int32', name='domain_input')
dd_input     = Input(shape=(DD_DIM,),   name='dd_input')  # [v6d] domain_delta sign

# [ESM-2 branch] — Phase 1 학습, Phase 2 frozen
x_esm = Dense(256, kernel_regularizer=regularizers.l2(1e-5), name='esm2_d1')(esm2_input)
x_esm = Activation('relu')(x_esm)
x_esm = Dropout(0.2)(x_esm)
x_esm = Dense(128, kernel_regularizer=regularizers.l2(1e-5), name='esm2_d2')(x_esm)
x_esm = Activation('relu')(x_esm)
esm2_feat  = Dense(64, activation='relu',
                   kernel_regularizer=regularizers.l2(1e-5),
                   name='esm2_feat')(x_esm)
esm2_gated = Lambda(lambda x: x[0] * x[1], name='esm2_gated')([esm2_feat, esm2_mask_in])

# [CNN branch] — Phase 1 frozen, Phase 2 학습
# GlobalMaxPool → 가변 길이 서열 처리 가능, make_batch 불필요
x_seq = Embedding(8001, 32, mask_zero=False, name='cnn_emb')(seq_input)
x_seq = Conv1D(64, kernel_size=7, padding='same', activation='relu', name='cnn_conv1')(x_seq)
x_seq = Conv1D(32, kernel_size=5, padding='same', activation='relu', name='cnn_conv2')(x_seq)
x_seq = GlobalMaxPooling1D(name='cnn_pool')(x_seq)
cnn_feat = Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-5),
                 name='cnn_feat')(x_seq)

# [Domain branch] — 전 구간 학습
x_dm = Embedding(input_dim=domain_emb_dim, output_dim=32,
                  input_length=dm_dim, mask_zero=True,
                  name='dm_emb')(domain_input)
domain_feat = LSTM(16, name='domain_feat')(x_dm)

# [v6d] DomainDelta branch — sign{-1,0,+1} → 이소폼-특이적 도메인 gain/loss
# Phase 1: trainable (이소폼 분리 신호) | Phase 2: trainable (ESM-2 frozen)
x_dd = Dense(64, activation='relu',
             kernel_regularizer=regularizers.l2(1e-5),
             name='dd_dense1')(dd_input)
x_dd = Dropout(0.2)(x_dd)
dd_feat = Dense(16, activation='relu',
                kernel_regularizer=regularizers.l2(1e-5),
                name='dd_dense2')(x_dd)

# [Fusion] concat[64+32+16+16=128]
concat = concatenate([esm2_gated, cnn_feat, domain_feat, dd_feat], name='feature_concat')
feature_model = Model([esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
                       concat, name='feature_model')

# [Head]
x = Dense(64, kernel_regularizer=regularizers.l2(1e-5))(concat)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
embedding_layer  = Lambda(lambda a: K.l2_normalize(a, axis=1),
                           name="embedding_out")(x)
prediction_layer = Dense(1, activation='sigmoid',
                          kernel_regularizer=regularizers.l2(1e-5),
                          name='prediction_out')(embedding_layer)

base_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=[embedding_layer, prediction_layer])
classification_model = Model(
    inputs=[esm2_input, esm2_mask_in, seq_input, domain_input, dd_input],
    outputs=prediction_layer)

adam_p1   = optimizers.Adam(lr=0.0005)
adam_main = optimizers.Adam(lr=0.001)
adam_p2   = optimizers.Adam(lr=0.0003)

K_testing_size = len(X_test_esm2)

print("\n[Model v7a] ESM-2(640->64) + CNN(1500->32) + Domain(LSTM 16) + DomainDelta(251->16) -> concat[128]")
print("[Model v7a] Head: Dense(64) -> L2_norm -> emb[64] -> sigmoid")
print("[Model v7a] Phase 1: SupCon [v7a-1] τ=0.1 | Phase 2: ESM-2(frozen) CNN Domain DeltaBranch Focal")
print("[Model v7a] v6g base: Phase1.5 removed | v6f: TypeB lr=0.0002, patience=10, max_epochs=25")

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
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph0(untrained)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase0_initial_untrained_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph0(untrained)'] = s0
ph0_metrics = analyze_embedding_quality(emb_by_phase['Ph0(untrained)'], y_test, 'Phase 0')

# [v6e-1] Phase 0 sep_ratio 기반 GO term Type 자동 분류
# Type A (sep_ratio >= 1.15): positive 집합이 진화적으로 응집 — ESM-2가 이미 분리 가능
# Type B (sep_ratio < 1.15): positive 집합이 이질적 — 수렴 진화, Phase 2 CNN이 embedding 역전
ph0_sep_ratio = ph0_metrics.get('sep_ratio', float('nan'))
IS_TYPE_B = (ph0_sep_ratio < SEP_RATIO_THRESHOLD) if not np.isnan(ph0_sep_ratio) else False
go_type_str = 'Type-B (heterogeneous, sep={:.4f})'.format(ph0_sep_ratio) if IS_TYPE_B \
              else 'Type-A (coherent, sep={:.4f})'.format(ph0_sep_ratio)
print("\n[v6e-1] GO term classification: {}".format(go_type_str))
if IS_TYPE_B:
    print("  -> Phase 2 will use: lr=0.0002, clipnorm=0.5, patience=10, max_epochs=25")  # [v6f-1,2]
    print("  -> Biological note: positive set contains evolutionarily unrelated proteins")
    print("     sharing this GO term via convergent function (not sequence similarity)")
else:
    print("  -> Phase 2 will use: lr=0.0003, patience=7, max_epochs=15 (standard)")
    print("  -> Biological note: positive set shares evolutionary origin — ESM-2 coherent")

# ==========================================================================
# PHASE 1: Supervised Contrastive Loss [v7a-1] (CNN frozen)
# ==========================================================================
# [v7a-1] Triplet → SupCon:
#   수식: L_i = log Σ_{a≠i} exp(z_i·z_a/τ) − (1/|P(i)|) Σ_{p∈P(i)} z_i·z_p/τ
#   τ=0.1, batch: n_pos_per_batch=max(8, N_pos//4), n_neg=batch-n_pos
#   early stop: loss < 0.01 for 4 consecutive epochs
# ==========================================================================
SUPCON_TAU = _tau_val  # argv[2] 또는 기본값 0.1

print("\n" + "=" * 55)
print(">>> PHASE 1: SupCon [v7a-1] — ESM-2+Domain (CNN frozen)")
print("    tau={} | n_batches={} | max 15 epochs".format(SUPCON_TAU, N_BATCHES_P1))
print("=" * 55)

set_cnn_trainable(feature_model, False)
print("  [Phase1-v7a] CNN frozen | ESM-2+Domain+DeltaBranch trainable")

PHASE1_EPOCHS    = 15
LOSS_THRESH      = 0.01   # loss < 0.01 for STREAK_LIMIT → early stop
STREAK_LIMIT     = 4
low_loss_streak  = 0
best_sep_ratio   = 0.0

for epoch in range(PHASE1_EPOCHS):
    print('Phase 1 - Epoch: {}/{} [SupCon]'.format(epoch + 1, PHASE1_EPOCHS))

    avg_loss, avg_valid = phase1_supcon_epoch_hybrid(
        feature_model=feature_model,
        X_esm2=X_train_esm2_upsmp,
        X_mask=X_train_mask_upsmp,
        X_seq=X_train_seq_upsmp,
        X_dm=X_train_dm_upsmp,
        X_dd=X_train_dd_upsmp,
        y=y_train_upsmp,
        optimizer=adam_p1,
        temperature=SUPCON_TAU,
        batch_size=BATCH_SIZE_P1,
        n_batches=N_BATCHES_P1,
        min_pos_in_batch=8)

    print("  -> SupCon Loss: {:.4f} | Valid anchors/batch: {:.1f}".format(avg_loss, avg_valid))

    # Early stop: loss 수렴 판단
    if avg_loss < LOSS_THRESH:
        low_loss_streak += 1
    else:
        low_loss_streak = 0
    if low_loss_streak >= STREAK_LIMIT:
        print("  [Early Stop] SupCon loss < {:.3f} for {} epochs.".format(
            LOSS_THRESH, STREAK_LIMIT))
        break

    if (epoch + 1) % 5 == 0:
        sep, _ = compute_embedding_sep_ratio(
            feature_model, X_test_esm2, X_test_esm2_mask,
            X_test_seq, X_test_dm, X_test_dd, y_test)
        best_sep_ratio = max(best_sep_ratio, sep) if not np.isnan(sep) else best_sep_ratio

print("\n[Phase 1 Final] best_sep_ratio={:.4f}".format(best_sep_ratio))
s1 = save_phase_results_hybrid(
    "phase1_triplet_only", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph1(triplet)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase1_triplet_only_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph1(triplet)'] = s1
analyze_embedding_quality(emb_by_phase['Ph1(triplet)'], y_test, 'Phase 1')

# ==========================================================================
# [v6g-1] PHASE 1.5 제거 — Phase 1 → Phase 2 직행
# ==========================================================================
print("\n[v6g-1/v7a] Phase 1.5 SKIPPED — Phase 1 → Phase 2 direct")
print("  근거: Phase 1.5 embedding 실측 변화(max_diff=0.52) + v6f Macro-AUPRC 최저")
n_train = len(X_train_esm2_upsmp)

# ==========================================================================
# PHASE 2: CNN Fine-tuning (ESM-2 frozen, CNN + Domain + head 학습)
# [HYBRID] Triplet 제거 — focal only
# [HYBRID] ESM-2 frozen → Phase 1.5 calibration 보존
# [HYBRID] CNN이 ESM-2 mean-pool이 놓친 위치 특이적 모티프 학습
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 2: CNN + DeltaBranch Fine-tuning")
print("    [v6e] ESM-2 FROZEN | CNN+Domain+DeltaBranch TRAIN | Focal only")
print("    [v7a] Adaptive hyperparameter by GO term type")
print("=" * 55)

# Unfreeze: CNN + Domain + head (ESM-2는 계속 frozen)
for layer in feature_model.layers:
    layer.trainable = True          # 일단 전체 unfreeze
set_esm2_trainable(feature_model, False)  # ESM-2 재동결
print("  [Phase2-v6e] ESM-2 frozen | CNN + Domain + DeltaBranch trainable")

# [v6e-2] Type별 Phase 2 hyperparameter 설정
# Type B: 작은 LR + gradient clipping → Phase 1이 만든 embedding 구조 보존
# 근거: Type B에서 Phase 2 CNN이 이질적 positive들의 sub-cluster별 서열 패턴을
#       학습하면서 sep_ratio를 Phase 0 이하로 역전시킴 (v6d 3-run 관찰)
if IS_TYPE_B:
    adam_p2       = optimizers.Adam(lr=0.0002, clipnorm=0.5)  # [v6f-1] 0.0001→0.0002
    NO_IMPROVE_LIMIT = 10  # [v6f-2] 6→10 (lr=0.0002에서 ~20 epoch 수렴 예상)
    # 근거: v6e GO:0006941에서 lr=0.0001 + patience=6으로 AUPRC 0.197 (v6d 0.284 미달)
    #       epoch1 AUPRC=0.150 → 0.226 (15 epoch). lr=0.0002이면 수렴 가속 가능
    print("  [v6f-1] Type-B: lr=0.0002, clipnorm=0.5, patience={}".format(NO_IMPROVE_LIMIT))
    print("  [v6f-1] Reason: v6e lr=0.0001 too slow for GO:0006941 convergence")
else:
    adam_p2       = optimizers.Adam(lr=0.0003)  # v6d 동일
    NO_IMPROVE_LIMIT = 7   # v6d/v6e 동일
    print("  [v6g] Type-A: lr=0.0003, patience={} (standard)".format(NO_IMPROVE_LIMIT))

classification_model.compile(
    loss=binary_focal_loss(gamma=2.0, alpha=0.10),
    optimizer=adam_p2, metrics=['accuracy'])

PHASE2_MAX_EPOCHS  = 25  # [v7a] Type-A도 25 epoch — SupCon Phase 1 후 epoch=15에서 수렴 미완료 관찰
BATCH_SIZE_P2      = 512
best_phase2_auprc  = 0.0
best_phase2_weights = None
no_improve_count   = 0

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
             X_train_dm_upsmp[idx[mixed]],
             X_train_dd_upsmp[idx[mixed]]],
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
        [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
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
                break

if best_phase2_weights is not None:
    base_model.set_weights(best_phase2_weights)
    print("\n[Phase 2] Restored best (AUPRC={:.4f})".format(best_phase2_auprc))

compute_embedding_sep_ratio(
    feature_model, X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd, y_test)

s2 = save_phase_results_hybrid(
    "phase2_cnn_focal", base_model,
    X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd,
    y_test, X_test_geneid, X_test_isoid, SAVE_DIR, VER_TAG)
emb_by_phase['Ph2(cnn_focal)'] = np.load(
    os.path.join(SAVE_DIR, '{}_phase2_cnn_focal_embeddings.npy'.format(VER_TAG)))
score_by_phase['Ph2(cnn_focal)'] = s2
analyze_embedding_quality(emb_by_phase['Ph2(cnn_focal)'], y_test, 'Phase 2')

# ==========================================================================
# PHASE 3: Test-time Expression Label Propagation [I2]
# ==========================================================================
print("\n" + "=" * 55)
print(">>> PHASE 3: [v6e-3] Label Propagation REMOVED")
print("    근거: 5/5 GO term 중 4개에서 LP가 AUPRC 감소 또는 중립")
print("    GO:0006096: LP 적용 시 AUPRC -8.9% (v6d 3-run 측정)")
print("    alpha=0.0 고정 (LP 없음)")
print("=" * 55)

preds_base  = base_model.predict(
    [X_test_esm2, X_test_esm2_mask, X_test_seq, X_test_dm, X_test_dd],
    batch_size=256, verbose=0)
base_scores = np.array([preds_base[1][i][0] for i in range(K_testing_size)])

# [v6e-3] alpha=0.0 고정 — LP 제거
# 진단용으로 alpha=0.2 결과만 로그에 출력 (적용하지 않음)
final_scores = base_scores.copy()
final_auroc  = roc_auc_score(y_test, final_scores) \
               if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else float('nan')
final_auprc  = average_precision_score(y_test, final_scores) \
               if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0 else float('nan')
print("\n  [v6e-3] Final (no LP): AUROC={:.4f} AUPRC={:.4f}".format(
    final_auroc, final_auprc))

# 진단용 LP 비교 (적용 안 함, 로그만)
if (y_test == 1).sum() > 0 and (y_test == 0).sum() > 0:
    try:
        lp_02 = expression_label_propagation(base_scores, X_test_expr, alpha=0.2)
        lp_auprc = average_precision_score(y_test, lp_02)
        delta = lp_auprc - final_auprc
        print("  [Diag] LP alpha=0.2 AUPRC={:.4f} (delta={:+.4f}, NOT applied)".format(
            lp_auprc, delta))
    except Exception:
        pass

score_by_phase['Final(noLP)'] = final_scores

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
    SAVE_DIR, "{}_{}_Final_scores.txt".format(VER_TAG, safe_go))
with open(output_file, 'w') as fw:
    for i in range(K_testing_size):
        fw.write(X_test_geneid[i] + '\t' + X_test_isoid[i] + '\t' +
                 str(final_scores[i]) + '\n')

base_model.save_weights(
    os.path.join(SAVE_DIR, "{}_{}_BaseModel_weights.h5".format(VER_TAG, safe_go)))
np.save(os.path.join(SAVE_DIR, "{}_{}_Final_labels.npy".format(VER_TAG, safe_go)),
        y_test)

# [v6e] 최종 요약 — go_type, phase2 설정 포함
print("\n[Final] GO={} | Type={} | Phase2_LR={} | patience={}".format(
    selected_go, go_type_str,
    '0.0002' if IS_TYPE_B else '0.0003',  # [v6f-4] 0.0001→0.0002
    NO_IMPROVE_LIMIT))
print("[Final] AUROC={:.4f} AUPRC={:.4f}".format(final_auroc, final_auprc))
print("[Final] mean={:.4f} std={:.4f} >0.5={} >0.3={}".format(
    final_scores.mean(), final_scores.std(),
    (final_scores > 0.5).sum(), (final_scores > 0.3).sum()))
print("\n[Done] {} | {}".format(VER_TAG, SAVE_DIR))
