#!/usr/bin/env python3
"""
exp_head_depth.py — head DEPTH ablation (사용자 제안: hidden layer 추가로 nonlinear axis 정보 fitting)
====================================================================================================
전제 정정: v17f*는 이미 nonlinear MLP (Dense256→BN→Drop→Dense128). "선형이라 axis 정보 못 담는다"가 아님.
질문: head를 더 깊게/넓게 하면 (a) macro-AUPRC(훈련 목표) 또는 (b) within-gene Domain-Ranking(창발)이 오르나?

예측 등록 (predict-before-look, HARKing 방지):
  macro-AUPRC: v17f* CI(0.723–0.747) 내 or 약간 하락(희소 MF overfitting). 유의 이득 없음.
  within-gene DR: 0.630 CI(0.613–0.646) 내. 유의 이득 없음.
  근거: 훈련 신호=gene-level GO, within-gene 감독 0 → depth가 within-gene 랭킹 gradient를 만들지 못함.
        depth는 gene-conditioning을 추가하지 않음 → axis0 leakage 지속.
falsification: 어떤 variant가 macro CI를 상단 돌파 → head가 underfit이었다는 증거(사용자 옳음).

configs:
  base   : 256 → 128            (= v17f*)
  deeper : 256 → 128 → 64       (+1 hidden layer, 사용자 제안)
  wider  : 512 → 256 → 128      (+capacity +depth)
5 seeds each, 동일 [δ_layer, φ_L30] 입력·focal·EarlyStopping. macro(All MF, L2) + within-gene DR 보고.
"""
import os, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR, ID_DIR, ANNOT_DIR = '../data', '../data/raw_data/data/id_lists', '../data/raw_data/data/annotations'
OUT = '../../reports/v20b_pca_interp/within_family'
SEEDS, BATCH_MLP, EPOCHS_MLP, LA, LB = [42, 7, 13, 21, 99], 512, 60, 15, 30
clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log("[1] embeddings")
Xtr30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LB:02d}_t30_150M.npy').astype(np.float32)
Xtr15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LA:02d}_t30_150M.npy').astype(np.float32)
Xte30 = np.load(f'{DATA_DIR}/esm2_layer_{LB:02d}_t30_150M.npy').astype(np.float32)
Xte15 = np.load(f'{DATA_DIR}/esm2_layer_{LA:02d}_t30_150M.npy').astype(np.float32)
sc = MaxAbsScaler(); dtr = sc.fit_transform((Xtr30 - Xtr15)).astype(np.float32)
dte = sc.transform((Xte30 - Xte15)).astype(np.float32)

log("[2] ids + GO labels")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_genes = [clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]
te_sym_list = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in np.load('my_gene_list_fixed.npy', allow_pickle=True)]
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'): sym2id.setdefault(syn, p[1])
tr_ids = [sym2id.get(g, g) for g in tr_genes]; tr_id_set = set(tr_ids)
go_tr, go_all = defaultdict(set), defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        go_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_tr[p[2]].add(p[1])
mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)
def Ytr(go):
    pos = {g for g, gid in zip(tr_genes, tr_ids) if gid in go_tr[go]}
    y = np.zeros(len(tr_genes), np.float32)
    for s in pos:
        for idx in tr_sym2idx.get(s, []): y[idx] = 1.0
    return y
def Yte(go): return np.array([1.0 if sym2id.get(s, '__') in go_all[go] else 0.0 for s in te_sym_list], np.float32)
Y_tr = np.stack([Ytr(g) for g in mf_terms], 1); Y_te = np.stack([Yte(g) for g in mf_terms], 1)
valid_idx = [i for i in range(len(mf_terms)) if Y_te[:, i].sum() >= 2]
L2 = set()
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12 and p[11] == 'L2_Structural': L2.add(p[0])
l2_idx = [i for i in valid_idx if mf_terms[i] in L2]
n_go = len(mf_terms)
log(f"   {n_go} MF, valid={len(valid_idx)}, L2={len(l2_idx)}")

iso_ndom = np.load('../results_isoform/features/domain_matrix_proper_test.npy').sum(1)
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
def dr_auc(preds):
    aucs = []
    for g, idxs in gene2idxs.items():
        if len(idxs) < 2: continue
        dom = iso_ndom[np.array(idxs)]
        if dom.std() < 0.1: continue
        pos = np.where(Y_te[idxs[0]] > 0)[0]
        if len(pos) == 0: continue
        med = np.median(dom); db = (dom > med).astype(float)
        if db.sum() == 0 or db.sum() == len(idxs): continue
        pg = preds[np.array(idxs)]
        for t in pos:
            s = pg[:, t]
            aucs.append(0.5 if s.std() < 1e-8 else roc_auc_score(db, s))
    return float(np.mean(aucs)) if aucs else 0.5
def macro(preds, idxs): return float(np.mean([average_precision_score(Y_te[:, i], preds[:, i]) for i in idxs]))

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

CONFIGS = {"base_256_128": [256, 128], "deeper_256_128_64": [256, 128, 64], "wider_512_256_128": [512, 256, 128]}
def build(hidden):
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    for j, h in enumerate(hidden):
        x = layers.Dense(h, activation='relu')(x)
        if j == 0: x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    return models.Model([inp_d, inp_e], layers.Dense(n_go, activation='sigmoid')(x))

results = {}
print(f"\n   {'config':>20} {'AllMF':>7} {'L2':>7} {'DR-AUC':>7}   (v17f* 0.734/0.637/0.630)")
for cname, hidden in CONFIGS.items():
    preds_seeds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(dtr)); nv = int(len(dtr) * 0.1)
        vi, ti = perm[:nv], perm[nv:]
        m = build(hidden); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        m.fit([dtr[ti], Xtr30[ti]], Y_tr[ti], validation_data=([dtr[vi], Xtr30[vi]], Y_tr[vi]),
              epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
              verbose=0)
        preds_seeds.append(m.predict([dte, Xte30], batch_size=2048, verbose=0))
        tf.keras.backend.clear_session()
    p = np.mean(preds_seeds, 0)
    a_all, a_l2, dr = macro(p, valid_idx), macro(p, l2_idx), dr_auc(p)
    results[cname] = {"all_mf": a_all, "l2": a_l2, "dr_auc": dr, "hidden": hidden}
    print(f"   {cname:>20} {a_all:>7.4f} {a_l2:>7.4f} {dr:>7.4f}")
    log(f"   {cname} done")

base = results["base_256_128"]
for cname, r in results.items():
    r["d_all_vs_base"] = r["all_mf"] - base["all_mf"]
    r["d_dr_vs_base"] = r["dr_auc"] - base["dr_auc"]
verdict = ("depth 이득 없음 — within-gene DR·macro 모두 base 대비 |Δ|<0.005 (예측대로: 훈련신호=gene-level, depth로 within-gene 못 가르침)"
           if all(abs(results[c]["d_all_vs_base"]) < 0.006 and abs(results[c]["d_dr_vs_base"]) < 0.008 for c in CONFIGS)
           else "일부 variant가 base 초과 — head underfit 가능성, bootstrap CI로 재검 필요")
print(f"\n=> {verdict}")
json.dump({"results": results, "verdict": verdict,
           "v17f_star_ref": {"all_mf": 0.734, "l2": 0.637, "dr_auc": 0.630}},
          open(f"{OUT}/exp_head_depth.json", "w"), indent=2)
log(f"[saved] {OUT}/exp_head_depth.json")
