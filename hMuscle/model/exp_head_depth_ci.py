#!/usr/bin/env python3
"""
exp_head_depth_ci.py — wider(512→256→128) vs base(256→128) paired bootstrap CI
==============================================================================
exp_head_depth 단일-run: wider가 AllMF +0.013, L2 +0.021. seed noise인가 실 신호인가?
S4: term-level paired bootstrap(n=1000) — 두 앙상블의 per-term AUPRC 차이를 term 재표집.
또 seed-robustness: 각 config를 두 seed-set(A=[42,7,13,21,99], B=[1,2,3,4,5])으로 재훈련해
run-to-run 변동과 비교. macro 이득이 CI에서 0을 배제 & seed-set 재현되면 real.
"""
import os, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, ID_DIR, ANNOT_DIR = '../data', '../data/raw_data/data/id_lists', '../data/raw_data/data/annotations'
OUT = '../../reports/v20b_pca_interp/within_family'
BATCH, EPOCHS, LA, LB = 512, 60, 15, 30
clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

Xtr30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LB:02d}_t30_150M.npy').astype(np.float32)
Xtr15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LA:02d}_t30_150M.npy').astype(np.float32)
Xte30 = np.load(f'{DATA_DIR}/esm2_layer_{LB:02d}_t30_150M.npy').astype(np.float32)
Xte15 = np.load(f'{DATA_DIR}/esm2_layer_{LA:02d}_t30_150M.npy').astype(np.float32)
sc = MaxAbsScaler(); dtr = sc.fit_transform((Xtr30 - Xtr15)).astype(np.float32); dte = sc.transform((Xte30 - Xte15)).astype(np.float32)
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t');  ENSG2SYM.__setitem__(p[0], p[4]) if len(p) >= 5 else None
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

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
def build(hidden):
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    for j, h in enumerate(hidden):
        x = layers.Dense(h, activation='relu')(x)
        if j == 0: x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    return models.Model([inp_d, inp_e], layers.Dense(n_go, activation='sigmoid')(x))
def train_ensemble(hidden, seeds):
    ps = []
    for seed in seeds:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(dtr)); nv = int(len(dtr) * 0.1); vi, ti = perm[:nv], perm[nv:]
        m = build(hidden); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        m.fit([dtr[ti], Xtr30[ti]], Y_tr[ti], validation_data=([dtr[vi], Xtr30[vi]], Y_tr[vi]),
              epochs=EPOCHS, batch_size=BATCH,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
        ps.append(m.predict([dte, Xte30], batch_size=2048, verbose=0)); tf.keras.backend.clear_session()
    return np.mean(ps, 0)

SA, SB = [42, 7, 13, 21, 99], [1, 2, 3, 4, 5]
log("train base/wider × seedset A,B")
P = {"base_A": train_ensemble([256, 128], SA), "wider_A": train_ensemble([512, 256, 128], SA),
     "base_B": train_ensemble([256, 128], SB), "wider_B": train_ensemble([512, 256, 128], SB)}
log("trained")

def ap_vec(p, idxs): return np.array([average_precision_score(Y_te[:, i], p[:, i]) for i in idxs])
def summ(p): return float(ap_vec(p, valid_idx).mean()), float(ap_vec(p, l2_idx).mean())
for k in P: log(f"  {k}: AllMF={summ(P[k])[0]:.4f}  L2={summ(P[k])[1]:.4f}")

# paired term-level bootstrap on seed-set A (primary)
rng = np.random.default_rng(0)
def boot(pb, pw, idxs, n=1000):
    ab, aw = ap_vec(pb, idxs), ap_vec(pw, idxs); d = aw - ab; m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
res = {}
for tag, idxs in [("AllMF", valid_idx), ("L2", l2_idx)]:
    dA = boot(P["base_A"], P["wider_A"], idxs)
    res[tag] = {"delta_A": dA[0], "ci_lo": dA[1], "ci_hi": dA[2],
                "base_A": summ(P["base_A"])[0 if tag == "AllMF" else 1],
                "wider_A": summ(P["wider_A"])[0 if tag == "AllMF" else 1],
                "base_B": summ(P["base_B"])[0 if tag == "AllMF" else 1],
                "wider_B": summ(P["wider_B"])[0 if tag == "AllMF" else 1]}
    sig = "CI excludes 0" if dA[1] > 0 else "CI includes 0"
    print(f"\n[{tag}] wider−base Δ(seedA)={dA[0]:+.4f} 95%CI[{dA[1]:+.4f},{dA[2]:+.4f}] {sig}")
    print(f"   seedA base={res[tag]['base_A']:.4f} wider={res[tag]['wider_A']:.4f} | seedB base={res[tag]['base_B']:.4f} wider={res[tag]['wider_B']:.4f}")
    dB = res[tag]['wider_B'] - res[tag]['base_B']
    res[tag]["delta_B"] = dB
    print(f"   seed-set replication: Δ_A={dA[0]:+.4f}  Δ_B={dB:+.4f}  {'재현' if (dA[0]>0)==(dB>0) and dB>0 else '불일치/약함'}")
json.dump(res, open(f"{OUT}/exp_head_depth_ci.json", "w"), indent=2)
log(f"[saved] {OUT}/exp_head_depth_ci.json")
