#!/usr/bin/env python3
"""
exp_width_control.py — devils-advocate 판별 실험 (옵션 A)
========================================================
DA Attack1(치명): 기존 base=2-layer vs 스윕=3-layer → +0.032가 width+depth 혼재, iso-param 통제 無.
판별: DEPTH를 통제한 공정 width 비교. 전부 2-layer, width만 변화:
  base   [256,128]  (=v17f*)
  w2_512 [512,256]  ← base와 같은 depth, 첫 layer만 확장 = 공정한 width 격리
  w2_768 [768,384]
그리고 depth 기여 분리용 3-layer 짝:
  w3_512 [512,256,128]
  w3_768 [768,384,192]
비교:
  (Q1 공정 width) base vs w2_768 : 2-layer끼리, width 순효과. 여기서 이득이 나면 finding 생존(width 실재).
  (Q2 depth 기여) w2_768 vs w3_768 : 3번째 layer가 추가 이득 주나. ~0이면 depth 무관(오늘 deeper 결과와 일관).
추가 게이트:
  (Attack2) per-term 이득 분해: (w−base) AP delta의 L2 vs non-L2, 그리고 term의 domain-countability(ndom→term AUROC)와 상관.
  (Attack4b) DR을 gene-cluster bootstrap (gene resample, unit 아님)으로 CI 재계산.
  (Attack4a) DR 비열등 margin 사전명시 = −0.01 (base DR의 ~1.6%). CI_lo > −0.01 이면 비열등 PASS.

예측 등록 (predict-before-look):
  Q1: w2_768 > base +0.015~0.03 (width 실재) OR ~0 (base under-tuning artifact 확정 → finding 사망).
  Q2: |w3−w2| < 0.008 (depth 무이득, 오늘 deeper와 일관) — 그러면 이득은 순수 width.
  per-term: 이득이 L2/non-L2 고루 → 일반 개선; L2 편중 + domain-countability 상관>0.4 → 도메인-카운팅 overfit(서사 충돌).
  gene-cluster DR CI: unit CI[−.006,.013]보다 넓어짐. CI_lo > −0.01 이면 비열등 유지.
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
DR_MARGIN = -0.01  # 사전명시 비열등 margin
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

# term domain-countability: does ndom predict term membership across all test isoforms?
def term_domain_auroc(t):
    y = Y_te[:, t]
    if y.sum() < 2 or y.sum() == len(y): return np.nan
    try: return roc_auc_score(y, iso_ndom)
    except Exception: return np.nan
dom_auroc = {t: term_domain_auroc(t) for t in valid_idx}

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

CFG = {"base_256_128": [256, 128], "w2_512_256": [512, 256], "w2_768_384": [768, 384],
       "w3_512_256_128": [512, 256, 128], "w3_768_384_192": [768, 384, 192]}
P, S = {}, {}
print(f"\n   {'config':>16} {'depth':>5} {'AllMF':>7} {'L2':>7} {'DR-AUC':>7}   (v17f* 0.734/0.637/0.630)")
for c, h in CFG.items():
    p = train_ensemble(h); P[c] = p
    du = dr_units(p); dr = float(np.mean(list(du.values())))
    S[c] = {"all_mf": macro(p, valid_idx), "l2": macro(p, l2_idx), "dr_auc": dr, "hidden": h, "_dr": du}
    print(f"   {c:>16} {len(h):>5} {S[c]['all_mf']:>7.4f} {S[c]['l2']:>7.4f} {dr:>7.4f}")
    log(f"   {c} done")

rng = np.random.default_rng(0)
def boot_macro(pb, pw, idxs, n=1000):
    ab, aw = ap_vec(pb, idxs), ap_vec(pw, idxs); d = aw - ab; m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def boot_dr_gene(du_b, du_w, n=1000):
    """gene-cluster bootstrap: resample GENES, aggregate their (gene,term) unit deltas."""
    keys = sorted(set(du_b) & set(du_w))
    per_gene = defaultdict(list)
    for (g, t) in keys: per_gene[g].append(du_w[(g, t)] - du_b[(g, t)])
    genes = list(per_gene); flat = np.array([v for g in genes for v in per_gene[g]])
    point = float(flat.mean())
    bs = []
    for _ in range(n):
        gs = [genes[i] for i in rng.integers(0, len(genes), len(genes))]
        vals = np.concatenate([per_gene[g] for g in gs])
        bs.append(vals.mean())
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(genes), len(flat)

res = {"configs": {c: {k: S[c][k] for k in ("all_mf", "l2", "dr_auc", "hidden")} for c in CFG}, "dr_margin": DR_MARGIN, "cmp": {}}

def report_cmp(tag, base_c, w_c):
    print(f"\n=== {tag}: {w_c} − {base_c} ===")
    d = {}
    for mt, idxs in [("AllMF", valid_idx), ("L2", l2_idx)]:
        dd, lo, hi = boot_macro(P[base_c], P[w_c], idxs)
        d[mt] = {"delta": dd, "lo": lo, "hi": hi}
        print(f"  [{mt}] Δ={dd:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] {'CI>0' if lo>0 else 'CI∋0'}")
    dd, lo, hi, ng, nu = boot_dr_gene(S[base_c]["_dr"], S[w_c]["_dr"])
    ni = "비열등 PASS" if lo > DR_MARGIN else "비열등 FAIL"
    d["DR"] = {"delta": dd, "lo": lo, "hi": hi, "n_genes": ng, "n_units": nu, "noninf": ni}
    print(f"  [DR gene-cluster] Δ={dd:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] (genes={ng}, units={nu}) margin={DR_MARGIN} → {ni}")
    res["cmp"][tag] = d
    return d

q1 = report_cmp("Q1_fair_width(2L)", "base_256_128", "w2_768_384")     # 공정 width (depth 고정)
report_cmp("Q1b_fair_width_512(2L)", "base_256_128", "w2_512_256")
q2 = report_cmp("Q2_depth_768", "w2_768_384", "w3_768_384_192")        # depth 순효과
report_cmp("Q2b_depth_512", "w2_512_256", "w3_512_256_128")

# per-term 이득 분해 (Attack2): 공정 width w2_768 vs base
log("per-term gain decomposition (w2_768_384 vs base)")
gains = {t: average_precision_score(Y_te[:, t], P["w2_768_384"][:, t]) - average_precision_score(Y_te[:, t], P["base_256_128"][:, t]) for t in valid_idx}
l2_gain = np.mean([gains[t] for t in l2_idx]); non_l2_gain = np.mean([gains[t] for t in valid_idx if t not in l2_idx])
gv = np.array([gains[t] for t in valid_idx]); dv = np.array([dom_auroc[t] for t in valid_idx])
ok = ~np.isnan(dv)
from scipy.stats import spearmanr
rho, pval = spearmanr(gv[ok], dv[ok])
top = sorted(valid_idx, key=lambda t: -gains[t])[:8]
res["per_term"] = {"l2_mean_gain": float(l2_gain), "non_l2_mean_gain": float(non_l2_gain),
                   "gain_vs_domainAUROC_spearman": float(rho), "p": float(pval),
                   "top_gainers": [{"term": mf_terms[t], "gain": float(gains[t]), "L2": t in l2_idx, "dom_auroc": float(dom_auroc[t]) if not np.isnan(dom_auroc[t]) else None} for t in top]}
print(f"\n=== per-term (w2_768−base) ===")
print(f"  L2 mean gain {l2_gain:+.4f} | non-L2 mean gain {non_l2_gain:+.4f} | ratio {l2_gain/(non_l2_gain+1e-9):.2f}x")
print(f"  gain vs term-domain-AUROC Spearman ρ={rho:+.3f} (p={pval:.3f}) {'→ 도메인-카운팅 편향 의심' if rho>0.4 and pval<0.05 else '→ 도메인-카운팅 편향 근거 약함'}")
for t in top: print(f"    {mf_terms[t]:>12} gain {gains[t]:+.4f} L2={t in l2_idx} dom_auroc={dom_auroc[t]:.3f}")

# 판정
verdict = []
if q1["AllMF"]["lo"] > 0: verdict.append(f"Q1 공정 width: base→w2_768 AllMF Δ{q1['AllMF']['delta']:+.4f} CI>0 → WIDTH 실재(depth 통제됨)")
else: verdict.append(f"Q1 공정 width: CI∋0 → width 이득 소멸 = under-tuned base artifact 가능성")
if abs(q2["AllMF"]["delta"]) < 0.008 and q2["AllMF"]["lo"] <= 0: verdict.append(f"Q2 depth: w2→w3 Δ{q2['AllMF']['delta']:+.4f} ~0 → depth 무이득(이득은 순수 width)")
else: verdict.append(f"Q2 depth: w2→w3 Δ{q2['AllMF']['delta']:+.4f} → depth도 기여")
res["verdict"] = verdict
print("\n=> " + "\n   ".join(verdict))
json.dump(res, open(f"{OUT}/exp_width_control.json", "w"), indent=2)
log(f"[saved] {OUT}/exp_width_control.json")
