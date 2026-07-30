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
log("[3] train 5-seed ensemble (once)")
mlps = []
for seed in SEEDS:
    np.random.seed(seed); tf.random.set_seed(seed)
    perm = np.random.permutation(len(delta_tr_s)); nv = int(len(delta_tr_s) * 0.1)
    vi, ti = perm[:nv], perm[nv:]
    m = build_mlp(); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
    m.fit([delta_tr_s[ti], X_tr_l30[ti]], Y_tr[ti],
          validation_data=([delta_tr_s[vi], X_tr_l30[vi]], Y_tr[vi]),
          epochs=EPOCHS_MLP, batch_size=BATCH_MLP,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
          verbose=0)
    mlps.append(m); log(f"   seed {seed} done")

def predict_ensemble(l15, l30):
    d = scaler.transform((l30 - l15).astype(np.float32)).astype(np.float32)
    return np.mean([m.predict([d, l30], batch_size=2048, verbose=0) for m in mlps], 0)

# ── baseline ──
log("[4] baseline (no occlusion)")
p0 = predict_ensemble(X_te_l15, X_te_l30)
dr0, npair = domain_ranking_auc(p0); ap0 = macro_auprc(p0)
log(f"   BASELINE Domain-Ranking AUC={dr0:.4f} (N={npair}, published 0.630)  macro-AUPRC={ap0:.4f}")

# ── occlusion per axis ──
log("[5] per-axis occlusion")
rng = np.random.default_rng(42)
# 8 axes
axis_dr, axis_ma = {}, {}
log("[5] per-axis occlusion + 20-random null (firming)")
for k in range(8):
    l15o=occlude(X_te_l15,LAYER_A,[W[k]]); l30o=occlude(X_te_l30,LAYER_B,[W[k]])
    p=predict_ensemble(l15o,l30o); dr,_=domain_ranking_auc(p); ma=macro_auprc(p)
    axis_dr[k]=dr0-dr; axis_ma[k]=ap0-ma
# 20 random unit directions in 640-space (null floor for "remove an arbitrary direction")
null_dr, null_ma = [], []
for r in range(20):
    v=rng.standard_normal(640).astype(np.float32); v/=np.linalg.norm(v)
    l15o=occlude(X_te_l15,LAYER_A,[v]); l30o=occlude(X_te_l30,LAYER_B,[v])
    p=predict_ensemble(l15o,l30o); dr,_=domain_ranking_auc(p); ma=macro_auprc(p)
    null_dr.append(dr0-dr); null_ma.append(ap0-ma)
    log(f"   null {r+1}/20 dDR={dr0-dr:+.4f} dMacro={ap0-ma:+.4f}")
ndr_m,ndr_s=float(np.mean(null_dr)),float(np.std(null_dr)+1e-9)
nma_m,nma_s=float(np.mean(null_ma)),float(np.std(null_ma)+1e-9)
print(f"\n   {'axis':>5} {'dDR':>9} {'z_DR':>7} {'dMacro':>9} {'z_Macro':>8}")
results={'baseline':{'dr':dr0,'macro':ap0,'n_pairs':npair},
         'null':{'dDR_mean':ndr_m,'dDR_sd':ndr_s,'dMacro_mean':nma_m,'dMacro_sd':nma_s,'n':20}}
for k in range(8):
    zdr=(axis_dr[k]-ndr_m)/ndr_s; zma=(axis_ma[k]-nma_m)/nma_s
    results[f'axis{k}']={'dDR':axis_dr[k],'z_DR':zdr,'dMacro':axis_ma[k],'z_Macro':zma}
    print(f"   {k:>5} {axis_dr[k]:>+9.4f} {zdr:>+7.2f} {axis_ma[k]:>+9.4f} {zma:>+8.2f}")
with open(f"{OUT}/occlusion_firming.json","w") as f: json.dump(results,f,indent=2)
log(f"[done] saved -> {OUT}/occlusion_firming.json")
print("\n판정: axis3 z_DR가 크고(+) axis0 z_Macro가 크며 axis5 둘다 z≈0이면 이중해리 유의 확정.")
