#!/usr/bin/env python3
"""
occlusion_axis_domain_ranking.py — 축별 causal occlusion (Q4 gold standard)
==========================================================================
질문: PRISM v17f*의 within-gene Domain-Ranking(AUC 0.630)에 8 PCA 축 중 어느 것이
      '인과적으로' 쓰이나? (attribution 상관 0.6~0.7은 overall 서열변화 공선으로 팽창 →
       개입(intervention)만이 인과를 준다.)

방법 (test-time occlusion):
  1. v17f* 원본 5-seed 앙상블을 unmodified train으로 1회 훈련.
  2. baseline Domain-Ranking AUC 측정 (published 0.630 재현 확인).
  3. 각 축 k: TEST φ_L15·φ_L30에서 축 k 방향 제거(per-layer z-space projection) →
     δ_te=φ_L30−φ_L15 재계산 + MaxAbsScaler.transform → 동일 앙상블로 재예측 →
     Domain-Ranking AUC. ΔAUC = baseline − occluded.
  4. 대조: (i) random 단위방향 제거(null floor) (ii) 8축 전부 제거.
  5. macro-AUPRC(gene-level)도 병기 — gene-level 성능은 어느 축이 견인?

사전 예측(predict-before-look, HARKing 방지):
  axis3(domain-architecture, length-독립) 제거 → Domain-Ranking AUC 최대 하락.
  axis5(length) 제거 → Domain-Ranking 미미 (PRISM은 length로 랭킹 안 함, length-AUC 0.560).
  → 이 패턴이 나오면 "예측이 length가 아니라 domain 축을 쓴다"가 인과 확정.
"""
import os, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR, ID_DIR, ANNOT_DIR = '../data', '../data/raw_data/data/id_lists', '../data/raw_data/data/annotations'
PCA = '../../reports/v20b_pca_interp'
OUT = '../../reports/v20b_pca_interp/within_family'
SEEDS, BATCH_MLP, EPOCHS_MLP, LAYER_A, LAYER_B = [42, 7, 13, 21, 99], 512, 60, 15, 30
clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ── axis occlusion machinery (per-layer z-space) ──
W = np.load(f"{PCA}/W_axes_8x640.npy").astype(np.float32)     # (8,640) orthonormal
MU = np.load(f"{PCA}/layer_stats_mu.npy").astype(np.float32)  # (30,640)
SD = np.load(f"{PCA}/layer_stats_sd.npy").astype(np.float32)
PMEAN = np.load(f"{PCA}/pca_mean_640.npy").astype(np.float32) # (640,) in z-space
def occlude(phi, L, dirs):
    """Remove given unit directions (list of 640-vec, orthonormal) from phi at layer L (1-based)."""
    z = (phi - MU[L-1]) / SD[L-1]
    zc = z - PMEAN
    for w in dirs:
        zc = zc - np.outer(zc @ w, w)
    return ((zc + PMEAN) * SD[L-1] + MU[L-1]).astype(np.float32)

# ── data ──
log("[1] embeddings")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
delta_tr = (X_tr_l30 - X_tr_l15).astype(np.float32)
scaler = MaxAbsScaler(); delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)

# ── IDs / labels (v17f logic) ──
log("[2] ids + GO labels")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
tr_genes = [clean(g) for g in np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)]
te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_genes_raw]
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    sym2id.setdefault(syn, p[1])
tr_ids = [sym2id.get(g, g) for g in tr_genes]; tr_id_set = set(tr_ids)
go_genes_tr, go_genes_all = defaultdict(set), defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606' or p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])
mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)
def build_Y_tr(go):
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in go_genes_tr[go]}
    y = np.zeros(len(tr_genes), np.float32)
    for s in pos_syms:
        for idx in tr_sym2idx.get(s, []): y[idx] = 1.0
    return y
def build_Y_te(go):
    return np.array([1.0 if sym2id.get(s, '__') in go_genes_all[go] else 0.0 for s in te_sym_list], np.float32)
Y_tr = np.stack([build_Y_tr(g) for g in mf_terms], 1)
Y_te = np.stack([build_Y_te(g) for g in mf_terms], 1)
valid_mask = Y_te.sum(0) >= 2
valid_idx = [i for i in range(len(mf_terms)) if valid_mask[i]]
n_go = len(mf_terms)
log(f"   {n_go} MF terms, valid={int(valid_mask.sum())}")

# ── domain ranking ──
iso_ndom = np.load('../results_isoform/features/domain_matrix_proper_test.npy').sum(1)
gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
def domain_ranking_auc(preds):
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
            if s.std() < 1e-8: aucs.append(0.5)
            else:
                try: aucs.append(roc_auc_score(db, s))
                except: pass
    return float(np.mean(aucs)) if aucs else 0.5, len(aucs)
def macro_auprc(preds):
    return float(np.mean([average_precision_score(Y_te[:, i], preds[:, i]) for i in valid_idx]))

# ── train ensemble ONCE ──
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
def build_mlp():
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model([inp_d, inp_e], layers.Dense(n_go, activation='sigmoid')(x))
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
# ── retrain-occlusion: remove axis from TRAIN+TEST, refit scaler, retrain ──
def occlude_all(l15,l30,dirs):
    return occlude(l15,LAYER_A,dirs), occlude(l30,LAYER_B,dirs)
def train_and_eval(l15tr,l30tr,l15te,l30te,tag):
    dtr=(l30tr-l15tr).astype(np.float32); sc=MaxAbsScaler(); dtr_s=sc.fit_transform(dtr).astype(np.float32)
    dte_s=sc.transform((l30te-l15te).astype(np.float32)).astype(np.float32)
    preds=[]
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm=np.random.permutation(len(dtr_s)); nv=int(len(dtr_s)*0.1); vi,ti=perm[:nv],perm[nv:]
        m=build_mlp(); m.compile(optimizer=optimizers.Adam(1e-3),loss=focal)
        m.fit([dtr_s[ti],l30tr[ti]],Y_tr[ti],validation_data=([dtr_s[vi],l30tr[vi]],Y_tr[vi]),
              epochs=EPOCHS_MLP,batch_size=BATCH_MLP,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)],verbose=0)
        preds.append(m.predict([dte_s,l30te],batch_size=2048,verbose=0))
    p=np.mean(preds,0); dr,n=domain_ranking_auc(p); ma=macro_auprc(p)
    log(f"   [{tag}] DR-AUC={dr:.4f} macro={ma:.4f}"); return dr,ma
rng=np.random.default_rng(42); vrand=rng.standard_normal(640).astype(np.float32); vrand/=np.linalg.norm(vrand)
conds={'baseline':None,'axis3':[W[3]],'axis5':[W[5]],'axis0':[W[0]],'random_dir':[vrand]}
log("[3] RETRAIN-occlusion (train+test removed, 5-seed each)")
res={}
for tag,dirs in conds.items():
    if dirs is None:
        dr,ma=train_and_eval(X_tr_l15,X_tr_l30,X_te_l15,X_te_l30,tag)
    else:
        l15tr,l30tr=occlude_all(X_tr_l15,X_tr_l30,dirs); l15te,l30te=occlude_all(X_te_l15,X_te_l30,dirs)
        dr,ma=train_and_eval(l15tr,l30tr,l15te,l30te,tag)
    res[tag]={'domain_ranking_auc':dr,'macro_auprc':ma}
b=res['baseline']
print(f"\n   {'cond':>12} {'DR-AUC':>8} {'dDR':>8} {'macro':>8} {'dMacro':>8}")
for tag,r in res.items():
    print(f"   {tag:>12} {r['domain_ranking_auc']:>8.4f} {b['domain_ranking_auc']-r['domain_ranking_auc']:>+8.4f} {r['macro_auprc']:>8.4f} {b['macro_auprc']-r['macro_auprc']:>+8.4f}")
    res[tag]['delta_dr']=b['domain_ranking_auc']-r['domain_ranking_auc']; res[tag]['delta_macro']=b['macro_auprc']-r['macro_auprc']
with open(f"{OUT}/occlusion_retrain.json","w") as f: json.dump(res,f,indent=2)
log(f"[done] saved -> {OUT}/occlusion_retrain.json")
print("\n판정: retrain 후에도 axis3 dDR>0(우회불가) & axis5 dDR≈0이면 인과 최종확정.")
