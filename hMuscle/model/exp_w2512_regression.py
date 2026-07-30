#!/usr/bin/env python3
"""
exp_w2512_regression.py — w2_512 채택 게이트: isoform 지표 회귀검증 (옵션 A 잔여)
=================================================================================
finding(축소판): base[256,128]→w2_512[512,256] 공정 width 이득 AllMF+.013/L2+.023, DR 비열등 PASS.
남은 게이트: macro만 오르고 다른 isoform 지표가 후퇴하면 순이득 아님.
여기서 두 개 추가 회귀검증:
  (P1) muscle within-gene POS_BIAS (v17f* 2-입력 실아키텍처): positive multi-iso gene의 within-gene
       score std / global std = isoform 판별 진폭. base vs w2_512 per-term paired bootstrap.
  (P2) UniProt 45쌍 direction accuracy (검증된 벤치마크 고유 아키텍처=단일입력 [phi30,delta] 1280,
       dropout 0.3/0.2). base[256,128] vs wide[512,256] width만 바꿔 방향정확도 비교. 캐시 임베딩 사용(ESM 재실행 無).
(brain transfer는 multi-layer brain 임베딩+brain GO transfer 재구성 필요 → 별도, 여기 미포함.)

예측 등록 (predict-before-look):
  P1 pos_bias: base≈w2_512 (Δ CI∋0). width는 within-gene 감독 없음 → 진폭 불변 예측(DR 결과와 일관).
  P2 UniProt acc: base≈wide (둘 다 published 근처). width가 방향정확도 떨어뜨리면 FAIL.
  falsification: 어느 지표든 w2_512가 유의 후퇴(pos_bias Δ CI<0 or UniProt acc 하락) → 순이득 아님, 채택 철회.
"""
import os, csv, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
from scipy import stats
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, ID_DIR, ANNOT_DIR = '../data', '../data/raw_data/data/id_lists', '../data/raw_data/data/annotations'
OUT = '../../reports/v20b_pca_interp/within_family'
BENCH_CSV = '../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv'
UNIPROT_EMB = '../../reports/exp_h_uniprot_eval/v2/embeddings_v2.npy'
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
n_go = len(mf_terms); go_to_idx = {g: i for i, g in enumerate(mf_terms)}
log(f"   {n_go} MF, valid={len(valid_idx)}, L2={len(l2_idx)}")

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
iso_ndom = np.load('../results_isoform/features/domain_matrix_proper_test.npy').sum(1)

def dr_units(preds):
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

# per-term pos_bias: positive multi-iso gene 내 score std / global std (evaluation.py 정의)
multi_genes = {g: np.array(idxs) for g, idxs in gene2idxs.items() if len(idxs) >= 2}
def posbias_vec(preds):
    """return per-term pos_bias over valid_idx (nan-safe)."""
    out = {}
    for t in valid_idx:
        s_all = preds[:, t]; gstd = s_all.std()
        if gstd < 1e-9: out[t] = np.nan; continue
        within = []
        for g, idxs in multi_genes.items():
            if Y_te[idxs[0], t] > 0:  # positive gene for term t
                within.append(preds[idxs, t].std())
        out[t] = (np.nanmean(within) / gstd) if within else np.nan
    return out

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

# ===================== P1: muscle two-input (real v17f* arch) =====================
def build2(hidden):
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    for j, h in enumerate(hidden):
        x = layers.Dense(h, activation='relu')(x)
        if j == 0: x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    return models.Model([inp_d, inp_e], layers.Dense(n_go, activation='sigmoid')(x))
def ens2(hidden):
    ps = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(dtr)); nv = int(len(dtr) * 0.1); vi, ti = perm[:nv], perm[nv:]
        m = build2(hidden); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        m.fit([dtr[ti], Xtr30[ti]], Y_tr[ti], validation_data=([dtr[vi], Xtr30[vi]], Y_tr[vi]),
              epochs=EPOCHS, batch_size=BATCH,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
        ps.append(m.predict([dte, Xte30], batch_size=2048, verbose=0)); tf.keras.backend.clear_session()
    return np.mean(ps, 0)

log("[P1] muscle two-input base vs w2_512")
Pb = ens2([256, 128]); Pw = ens2([512, 256])
pbb, pbw = posbias_vec(Pb), posbias_vec(Pw)
drb, drw = dr_units(Pb), dr_units(Pw)
def mac(p, idxs): return float(np.mean([average_precision_score(Y_te[:, i], p[:, i]) for i in idxs]))
rng = np.random.default_rng(0)
tv = [t for t in valid_idx if not (np.isnan(pbb[t]) or np.isnan(pbw[t]))]
dpb = np.array([pbw[t] - pbb[t] for t in tv]); m = len(dpb)
bs = [dpb[rng.integers(0, m, m)].mean() for _ in range(1000)]
pb_lo, pb_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
# DR gene-cluster
keys = sorted(set(drb) & set(drw)); per_gene = defaultdict(list)
for (g, t) in keys: per_gene[g].append(drw[(g, t)] - drb[(g, t)])
genes = list(per_gene)
dg = [np.concatenate([per_gene[genes[i]] for i in rng.integers(0, len(genes), len(genes))]).mean() for _ in range(1000)]
dr_lo, dr_hi = float(np.percentile(dg, 2.5)), float(np.percentile(dg, 97.5))
P1 = {"base": {"all_mf": mac(Pb, valid_idx), "l2": mac(Pb, l2_idx),
               "pos_bias": float(np.nanmean([pbb[t] for t in tv])), "dr": float(np.mean(list(drb.values())))},
      "w2_512": {"all_mf": mac(Pw, valid_idx), "l2": mac(Pw, l2_idx),
                 "pos_bias": float(np.nanmean([pbw[t] for t in tv])), "dr": float(np.mean(list(drw.values())))},
      "pos_bias_delta": float(dpb.mean()), "pos_bias_ci": [pb_lo, pb_hi],
      "dr_delta_genecluster": float(np.concatenate(list(per_gene.values())).mean()), "dr_ci": [dr_lo, dr_hi]}
print(f"\n[P1 muscle] base pos_bias={P1['base']['pos_bias']:.4f} DR={P1['base']['dr']:.4f} | w2_512 pos_bias={P1['w2_512']['pos_bias']:.4f} DR={P1['w2_512']['dr']:.4f}")
print(f"  pos_bias Δ={P1['pos_bias_delta']:+.4f} CI[{pb_lo:+.4f},{pb_hi:+.4f}] {'후퇴' if pb_hi<0 else '유지/개선(CI∋0 or >0)'}")
print(f"  DR(gene-cluster) Δ={P1['dr_delta_genecluster']:+.4f} CI[{dr_lo:+.4f},{dr_hi:+.4f}] {'비열등 PASS' if dr_lo>-0.01 else '비열등 FAIL'}")

# ===================== P2: UniProt 45-pair (single-input benchmark arch) =====================
log("[P2] UniProt single-input width comparison")
X_tr1280 = np.concatenate([Xtr30, (Xtr30 - Xtr15)], 1).astype(np.float32)  # [phi30, delta] as exp_uniprot_v2
scU = MaxAbsScaler().fit(X_tr1280); Xtr_u = scU.transform(X_tr1280).astype(np.float32)
emb = np.load(UNIPROT_EMB, allow_pickle=True).item()
iso_order = sorted(emb.keys()); Xpred_u = scU.transform(np.stack([emb[i] for i in iso_order]).astype(np.float32)).astype(np.float32)
iso2row = {iso: k for k, iso in enumerate(iso_order)}
remap = {'GO:0004714':'GO:0004713','GO:0005007':'GO:0004713','GO:0004693':'GO:0004674','GO:0097553':'GO:0046982',
         'GO:0004197':'GO:0003824','GO:0006281':'GO:0003677','GO:0006977':'GO:0003700','GO:0008285':'GO:0019901',
         'GO:0005178':'GO:0048018','GO:0005158':'GO:0048018','GO:0000398':'GO:0003723','GO:0006357':'GO:0003700',
         'GO:0005200':'GO:0003779','GO:0007399':'GO:0005515','GO:0016079':'GO:0046982'}
bench = list(csv.DictReader(open(BENCH_CSV)))
def build1(hidden):
    inp = layers.Input(shape=(1280,)); x = inp
    for j, h in enumerate(hidden):
        x = layers.Dense(h, activation='relu')(x)
        if j == 0: x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
        else: x = layers.Dropout(0.2)(x)
    return models.Model(inp, layers.Dense(n_go, activation='sigmoid')(x))
def ens1(hidden):
    ps = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(Xtr_u)); nv = int(len(Xtr_u) * 0.1); vi, ti = perm[:nv], perm[nv:]
        m = build1(hidden); m.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        m.fit(Xtr_u[ti], Y_tr[ti], validation_data=(Xtr_u[vi], Y_tr[vi]), epochs=EPOCHS, batch_size=BATCH,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
        ps.append(m.predict(Xpred_u, batch_size=512, verbose=0)); tf.keras.backend.clear_session()
    return np.mean(ps, 0)
def uni_acc(pred):
    correct = tot = 0; gaps = []
    for r in bench:
        go = r['go_term'].replace('GO_', 'GO:'); go = go if go in go_to_idx else remap.get(go, go)
        if go not in go_to_idx or r['iso_a'] not in iso2row or r['iso_b'] not in iso2row: continue
        j = go_to_idx[go]; sa = pred[iso2row[r['iso_a']], j]; sb = pred[iso2row[r['iso_b']], j]
        d = r['direction']
        c = int(sa > sb) if d == 'A_only' else int(sb > sa) if d == 'B_only' else 1 if d == 'both' else None
        if c is None: continue
        correct += c; tot += 1; gaps.append(abs(sa - sb))
    return correct, tot, float(np.mean(gaps))
Ub = ens1([256, 128]); Uw = ens1([512, 256])
cb, tb, gb = uni_acc(Ub); cw, tw, gw = uni_acc(Uw)
P2 = {"base": {"correct": cb, "total": tb, "acc": cb / tb, "mean_gap": gb},
      "wide_512": {"correct": cw, "total": tw, "acc": cw / tw, "mean_gap": gw}}
print(f"\n[P2 UniProt] base {cb}/{tb}={cb/tb:.3f} (gap {gb:.3f}) | wide_512 {cw}/{tw}={cw/tw:.3f} (gap {gw:.3f})")
print(f"  direction accuracy Δ={cw/tw-cb/tb:+.3f} {'후퇴' if cw<cb else '유지/개선'}")

res = {"P1_muscle_posbias_dr": P1, "P2_uniprot_45pair": P2,
       "note": "brain transfer 미포함(별도 multi-layer 재구성 필요)"}
json.dump(res, open(f"{OUT}/exp_w2512_regression.json", "w"), indent=2)
log(f"[saved] {OUT}/exp_w2512_regression.json")
