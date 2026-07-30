#!/usr/bin/env python3
"""
exp_width_sweep.py — head WIDTH sweep + isoform-metric regression CI (옵션 A)
============================================================================
직전 발견: wider(512→256→128) head가 base(256→128) 대비 macro +0.025 AllMF/+0.042 L2 (CI-유의, seed재현),
within-gene DR 불변. 채택 전 게이트: (1) width 최적점, (2) isoform 지표(DR) 후퇴 없음 확인.

DEPTH를 고정(3-layer funnel W,W/2,W/4)하고 WIDTH만 스윕 → width를 depth와 분리.
  base_256_128 : v17f* published anchor (2-layer)
  w384/512/768/1024 : 3-layer funnel, front width만 변화
best-vs-base 에 대해:
  macro: paired term-level bootstrap CI (n=1000) [AllMF, L2]
  DR   : paired (gene,term)-level bootstrap CI (n=1000) — 후퇴(<0) 여부 = 순이득 게이트

예측 등록 (predict-before-look):
  macro: 512~768에서 최적, +0.02~0.03 유지 (CI>0). 1024는 overfit로 이득 감소 가능.
  DR   : 모든 width에서 published CI(0.613–0.646) 내 — width는 within-gene 감독을 추가 안 함 → 불변 예측.
  falsification: 어떤 width가 DR을 유의하게 낮추면(CI<0) → macro 이득이 isoform 해상도를 희생 = 순이득 아님.
"""
import os, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, ID_DIR, ANNOT_DIR = '../data', '../data/raw_data/data/id_lists', '../data/raw_data/data/annotations'
OUT = '../../reports/v20b_pca_interp/within_family'
SEEDS, BATCH, EPOCHS, LA, LB = [42, 7, 13, 21, 99], 512, 60, 15, 30
clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log("[1] embeddings")
Xtr30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LB:02d}_t30_150M.npy').astype(np.float32)
Xtr15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LA:02d}_t30_150M.npy').astype(np.float32)
Xte30 = np.load(f'{DATA_DIR}/esm2_layer_{LB:02d}_t30_150M.npy').astype(np.float32)
Xte15 = np.load(f'{DATA_DIR}/esm2_layer_{LA:02d}_t30_150M.npy').astype(np.float32)
sc = MaxAbsScaler(); dtr = sc.fit_transform((Xtr30 - Xtr15)).astype(np.float32); dte = sc.transform((Xte30 - Xte15)).astype(np.float32)

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
def dr_units(preds):
    """return dict {(gene,term): auc} aligned across models for paired bootstrap."""
    out = {}
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
            out[(g, int(t))] = 0.5 if s.std() < 1e-8 else roc_auc_score(db, s)
    return out
def ap_vec(p, idxs): return np.array([average_precision_score(Y_te[:, i], p[:, i]) for i in idxs])
def macro(p, idxs): return float(ap_vec(p, idxs).mean())

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
def build(hidden):
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    for j, h in enumerate(hidden):
        x = layers.Dense(h, activation='relu')(x)
        if j == 0: x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    return models.Model([inp_d, inp_e], layers.Dense(n_go, activation='sigmoid')(x))
def train_ensemble(hidden):
    ps = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(dtr)); nv = int(len(dtr) * 0.1); vi, ti = perm[:nv], perm[nv:]
        m = build(hidden); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        m.fit([dtr[ti], Xtr30[ti]], Y_tr[ti], validation_data=([dtr[vi], Xtr30[vi]], Y_tr[vi]),
              epochs=EPOCHS, batch_size=BATCH,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
        ps.append(m.predict([dte, Xte30], batch_size=2048, verbose=0)); tf.keras.backend.clear_session()
    return np.mean(ps, 0)

SWEEP = {"base_256_128": [256, 128], "w384": [384, 192, 96], "w512": [512, 256, 128],
         "w768": [768, 384, 192], "w1024": [1024, 512, 256]}
P, S = {}, {}
print(f"\n   {'config':>16} {'AllMF':>7} {'L2':>7} {'DR-AUC':>7}   (v17f* 0.734/0.637/0.630)")
for cname, hidden in SWEEP.items():
    p = train_ensemble(hidden); P[cname] = p
    du = dr_units(p); dr = float(np.mean(list(du.values())))
    S[cname] = {"all_mf": macro(p, valid_idx), "l2": macro(p, l2_idx), "dr_auc": dr, "hidden": hidden, "_dr": du}
    print(f"   {cname:>16} {S[cname]['all_mf']:>7.4f} {S[cname]['l2']:>7.4f} {dr:>7.4f}")
    log(f"   {cname} done")

# best width by AllMF among sweep (exclude anchor base)
sweep_only = [c for c in SWEEP if c != "base_256_128"]
best = max(sweep_only, key=lambda c: S[c]["all_mf"])
log(f"best width config by AllMF: {best}")

rng = np.random.default_rng(0)
def boot_macro(pb, pw, idxs, n=1000):
    ab, aw = ap_vec(pb, idxs), ap_vec(pw, idxs); d = aw - ab; m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def boot_dr(db_units, dw_units, n=1000):
    keys = sorted(set(db_units) & set(dw_units))
    d = np.array([dw_units[k] - db_units[k] for k in keys]); m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), m

res = {"sweep": {c: {k: S[c][k] for k in ("all_mf", "l2", "dr_auc", "hidden")} for c in SWEEP}, "best": best, "ci": {}}
for tag, idxs in [("AllMF", valid_idx), ("L2", l2_idx)]:
    d, lo, hi = boot_macro(P["base_256_128"], P[best], idxs)
    res["ci"][tag] = {"delta": d, "lo": lo, "hi": hi}
    print(f"\n[{tag}] {best}−base Δ={d:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] {'CI>0' if lo>0 else 'CI∋0'}")
dd, dlo, dhi, nk = boot_dr(S["base_256_128"]["_dr"], S[best]["_dr"])
res["ci"]["DR"] = {"delta": dd, "lo": dlo, "hi": dhi, "n_units": nk}
reg = "DR 후퇴 없음 (CI∋0 or >0)" if dhi >= 0 and dlo > -0.02 else "DR 유의 후퇴 — 순이득 아님"
print(f"[DR] {best}−base Δ={dd:+.4f} 95%CI[{dlo:+.4f},{dhi:+.4f}] (n={nk}) → {reg}")
res["regression_verdict"] = reg
json.dump(res, open(f"{OUT}/exp_width_sweep.json", "w"), indent=2)
log(f"[saved] {OUT}/exp_width_sweep.json")
