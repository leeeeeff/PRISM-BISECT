#!/usr/bin/env python3
"""
exp_within_supervision_surgical.py — surgical (domain-dependent-term-only) within-gene domain-rank
auxiliary loss, the explicitly-flagged untested variant from natcomm_v0.md S_dissection:
"(This falsifies these two specific supervisions, not every conceivable one: a surgical signal
weighted toward the domain-dependent term subset remains untested and is a natural target for
future work.)"

Identical to exp_within_supervision.py (the already-run, already-published BLANKET version) in every
respect EXCEPT one line: the margin-loss posmask is restricted to each gene's positive terms that are
ALSO in the domain-dependent L2_Structural subset (h2_layer_classification.tsv), instead of all of the
gene's positive MF terms. This directly targets the diagnosed failure mode of the blanket version
("a blanket 'more-domains-scores-higher' signal injects mis-directed gradient on the motif-dependent
terms — the 74.6% of within-gene specialisation events where domain count does not track function").

PREDICT-BEFORE-LOOK (S2, registered before running):
  The blanket version's failure was already causally bracketed as representation-bound (added
  capacity ALSO failed to move DR-AUC) -- two independent intervention families already point to
  "the ceiling is a property of the representation, not the supervision". The prior therefore leans
  toward this surgical variant ALSO failing to move DR-AUC decisively (H_representation extends),
  not toward a confident win. But the manuscript explicitly left this untested rather than assuming
  it would fail, because removing motif-term mis-targeting is a genuinely distinct mechanism from
  "add more capacity" or "add more (mis-targeted) supervision" -- so this is registered as a real,
  not foregone-conclusion, test:
    H_representation (leaning prior): DR-AUC stays flat (CI includes 0 vs base), extending the
      representation-bound conclusion to a third, better-targeted intervention.
    H_surgical-supervision (the untested possibility the manuscript flags): DR-AUC rises (CI excludes
      0, positive) AND domain-dependent-term macro AUPRC (L2 subset) does not degrade the way it did
      under the blanket loss (-0.086), because the mis-targeted gradient on motif-dependent terms is
      now removed by construction.
  Falsification of H_surgical-supervision: DR-AUC CI still includes 0, or L2 macro still degrades
  comparably to the blanket version.
"""
import os, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, ID_DIR, ANNOT_DIR = '../data', '../data/raw_data/data/id_lists', '../data/raw_data/data/annotations'
OUT = '../../reports/v20b_pca_interp/within_family'
SEEDS, BATCH, EPOCHS, LA, LB, MARGIN = [42, 7, 13, 21, 99], 512, 60, 15, 30, 0.1
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
L2_mask = np.array([1.0 if t in L2 else 0.0 for t in mf_terms], dtype=np.float32)  # <-- SURGICAL: the one new object
log(f"   {n_go} MF, valid={len(valid_idx)}, L2(domain-dependent)={len(l2_idx)}")

# ---- within-gene domain-diff training pairs (aux supervision) — SURGICAL: posmask restricted to L2 ----
log("[3] build within-gene domain-diff train pairs (surgical: domain-dependent terms only)")
tr_ndom = np.load('../results_isoform/features/domain_matrix_proper_train.npy').sum(1)
tr_g2i = defaultdict(list)
for i, g in enumerate(tr_genes): tr_g2i[g].append(i)
PI, PJ, PM = [], [], []
for g, idxs in tr_g2i.items():
    if len(idxs) < 2: continue
    idxs = np.array(idxs); d = tr_ndom[idxs]
    if d.std() < 0.1: continue
    posmask = (Y_tr[idxs[0]] > 0).astype(np.float32) * L2_mask  # SURGICAL CHANGE (only line differing from blanket version)
    if posmask.sum() == 0: continue
    for a in range(len(idxs)):
        for b in range(len(idxs)):
            if d[a] > d[b]:
                PI.append(idxs[a]); PJ.append(idxs[b]); PM.append(posmask)
PI = np.array(PI); PJ = np.array(PJ); PM = np.stack(PM).astype(np.float32)
log(f"   aux pairs: {len(PI)} (posmask mean terms/pair {PM.sum(1).mean():.2f})  [blanket version had 3097 pairs, mean terms/pair unrestricted]")

# ---- test DR metric (identical to blanket version) ----
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

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
tf.get_logger().setLevel('ERROR')
for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)
focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
def build():
    inp_d = layers.Input(shape=(640,)); inp_e = layers.Input(shape=(640,))
    x = layers.Concatenate()([inp_d, inp_e])
    x = layers.Dense(256, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model([inp_d, inp_e], layers.Dense(n_go, activation='sigmoid')(x))

def train_one(lam, seed):
    np.random.seed(seed); tf.random.set_seed(seed)
    perm = np.random.permutation(len(dtr)); nv = int(len(dtr) * 0.1); vi, ti = perm[:nv], perm[nv:]
    m = build(); opt = optimizers.Adam(1e-3)
    dtr_t, Xtr_t, Ytr_t = tf.constant(dtr), tf.constant(Xtr30), tf.constant(Y_tr)
    PI_t, PJ_t, PM_t = tf.constant(PI), tf.constant(PJ), tf.constant(PM)
    n_aux = len(PI); best_w, best_v, wait = None, 1e9, 0
    @tf.function
    def step(bd, be, by, pi, pj, pm):
        with tf.GradientTape() as tape:
            pr = m([bd, be], training=True)
            lf = focal(by, pr)
            la = 0.0
            if lam > 0:
                si = m([tf.gather(dtr_t, pi), tf.gather(Xtr_t, pi)], training=True)
                sj = m([tf.gather(dtr_t, pj), tf.gather(Xtr_t, pj)], training=True)
                mr = tf.nn.relu(MARGIN - (si - sj)) * pm
                la = tf.reduce_sum(mr) / (tf.reduce_sum(pm) + 1e-6)
            loss = lf + lam * la
        g = tape.gradient(loss, m.trainable_variables); opt.apply_gradients(zip(g, m.trainable_variables))
        return lf
    for ep in range(EPOCHS):
        order = np.random.permutation(ti)
        for s in range(0, len(order), BATCH):
            bi = order[s:s + BATCH]
            if lam > 0:
                pk = np.random.randint(0, n_aux, min(BATCH, n_aux))
                step(tf.gather(dtr_t, bi), tf.gather(Xtr_t, bi), tf.gather(Ytr_t, bi),
                     tf.gather(PI_t, pk), tf.gather(PJ_t, pk), tf.gather(PM_t, pk))
            else:
                step(tf.gather(dtr_t, bi), tf.gather(Xtr_t, bi), tf.gather(Ytr_t, bi),
                     PI_t[:1], PJ_t[:1], PM_t[:1])
        vpr = m([dtr[vi], Xtr30[vi]], training=False)
        vl = float(focal(Y_tr[vi], vpr))
        if vl < best_v - 1e-5: best_v, best_w, wait = vl, m.get_weights(), 0
        else:
            wait += 1
            if wait >= 10: break
    if best_w is not None: m.set_weights(best_w)
    pr = m.predict([dte, Xte30], batch_size=2048, verbose=0)
    tf.keras.backend.clear_session()
    return pr

LAMS = [0.0, 0.1, 0.3, 1.0]
P, S = {}, {}
print(f"\n   {'lambda':>8} {'AllMF':>7} {'L2':>7} {'DR-AUC':>7}   (base blanket-version comparison: 0.749/0.661/0.634)")
for lam in LAMS:
    ps = [train_one(lam, sd) for sd in SEEDS]
    p = np.mean(ps, 0); P[lam] = p
    du = dr_units(p); dr = float(np.mean(list(du.values())))
    S[lam] = {"all_mf": macro(p, valid_idx), "l2": macro(p, l2_idx), "dr_auc": dr, "_dr": du}
    print(f"   {lam:>8.2f} {S[lam]['all_mf']:>7.4f} {S[lam]['l2']:>7.4f} {dr:>7.4f}")
    log(f"   lambda={lam} done")

best = max([l for l in LAMS if l > 0], key=lambda l: S[l]["dr_auc"])
log(f"best lambda by DR: {best}")
rng = np.random.default_rng(0)
def boot_dr(du0, du1, n=1000):
    keys = sorted(set(du0) & set(du1))
    d = np.array([du1[k] - du0[k] for k in keys]); m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), m
def boot_macro(p0, p1, idxs, n=1000):
    a0, a1 = ap_vec(p0, idxs), ap_vec(p1, idxs); d = a1 - a0; m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

dd, dlo, dhi, nk = boot_dr(S[0.0]["_dr"], S[best]["_dr"])
print(f"\n[DR] λ={best}-base Δ={dd:+.4f} 95%CI[{dlo:+.4f},{dhi:+.4f}] (n={nk}) {'CI>0=surgical supervision WORKS' if dlo>0 else 'CI includes/below 0=representation-bound extends'}")
res = {"sweep": {str(l): {k: S[l][k] for k in ("all_mf", "l2", "dr_auc")} for l in LAMS},
       "best_lambda": best, "margin": MARGIN, "n_aux_pairs": int(len(PI)),
       "dr_ci": {"delta": dd, "lo": dlo, "hi": dhi, "n_units": nk}}
for tag, idxs in [("AllMF", valid_idx), ("L2", l2_idx)]:
    d, lo, hi = boot_macro(P[0.0], P[best], idxs)
    res.setdefault("macro_ci", {})[tag] = {"delta": d, "lo": lo, "hi": hi}
    print(f"[{tag}] λ={best}-base Δ={d:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] {'후퇴' if hi<0 else '유지/개선'}")
json.dump(res, open(f"{OUT}/exp_within_supervision_surgical.json", "w"), indent=2)
log(f"[saved] {OUT}/exp_within_supervision_surgical.json")
