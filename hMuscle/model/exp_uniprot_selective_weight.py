"""
exp_uniprot_selective_weight.py
================================
selective_weight 모델을 UniProt 42-pair 벤치마크에 적용.
v17f_selective_weight.py와 동일한 훈련 (selective weighting on 26 domain-dep MF terms)
+ UniProt isoform embeddings에 적용 후 42쌍 방향 정확도 평가.

비교:
  v17f*  baseline:     26/42 = 0.619
  confweight_a02:      23/42 = 0.548 (WORSE — suppressed real signal)
  selective_weight:    ???
"""

import numpy as np, os, time, csv, gzip, json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# GPU config BEFORE any TF ops
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError: pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ROOT      = Path("/home/welcome1/sw1686/DIFFUSE")
DATA_DIR  = str(ROOT / "hMuscle/data")
ID_DIR    = str(ROOT / "hMuscle/data/raw_data/data/id_lists")
ANNOT_DIR = str(ROOT / "hMuscle/data/raw_data/data/annotations")
FEAT_DIR  = ROOT / "hMuscle/results_isoform/features"
UNI_DIR   = ROOT / "reports/exp_h_uniprot_eval"
OUT_DIR   = ROOT / "reports/v17f_selective_weight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER_A, LAYER_B = 15, 30
N_SEED   = 5
ALPHA_W  = 0.2
N_CDD    = 512
NORM_RATIO_THR = 0.97

# ── 1. Train embeddings ───────────────────────────────────────────────────
print("[1] Train embeddings...")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
X_te_l30 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_te_l15 = np.load(f'{DATA_DIR}/esm2_layer_{LAYER_A:02d}_t30_150M.npy').astype(np.float32)
print(f"  tr: {X_tr_l30.shape}, te: {X_te_l30.shape}")

# ── 2. IDs ───────────────────────────────────────────────────────────────
print("[2] IDs...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes = [x.decode() if isinstance(x, bytes) else x for x in tr_genes_raw]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

def clean(g):
    s = str(g)
    for c in ["b'","'",'"',' ']: s = s.replace(c,'')
    return s

te_gene_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
               for g in te_gene_raw]

# ── 3. GO labels (MF + BP + CC for domain importance) ────────────────────
print("[3] GO labels (MF + BP + CC)...")
sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

tr_entrez = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_entrez)
tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

go_genes_tr_mf  = defaultdict(set)
go_genes_tr_bp  = defaultdict(set)
go_genes_tr_cc  = defaultdict(set)
go_genes_all_mf = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8 or p[0] != '9606': continue
        gid, go, cat = p[1], p[2], p[7]
        if cat == 'Function': go_genes_all_mf[go].add(gid)
        if gid not in tr_id_set: continue
        if cat == 'Function': go_genes_tr_mf[go].add(gid)
        elif cat == 'Process': go_genes_tr_bp[go].add(gid)
        elif cat == 'Component': go_genes_tr_cc[go].add(gid)

mf_terms = []
with open(str(ROOT / "reports/v_expanded_gomf/mf_domain_vs_prism.tsv")) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 1: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)

all_terms_info = {}
with open(str(ROOT / "reports/v_expanded_gomf/expanded_go_per_term.tsv")) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        all_terms_info[row['go_id']] = row
bp_terms = np.array([r['go_id'] for r in all_terms_info.values()
                     if r['cat']=='BP' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2])
cc_terms = np.array([r['go_id'] for r in all_terms_info.values()
                     if r['cat']=='CC' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2])
print(f"  MF: {len(mf_terms)}, BP: {len(bp_terms)}, CC: {len(cc_terms)}")

def build_Y_tr(go_id, go_dict):
    pos_ids  = go_dict[go_id]
    pos_syms = {g for g, eid in zip(tr_genes, tr_entrez) if eid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

print("  Building MF labels...")
Y_tr = np.stack([build_Y_tr(go, go_genes_tr_mf) for go in mf_terms], axis=1)
Y_te = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")
valid_te_terms = np.where(Y_te.sum(0) >= 2)[0]
print(f"  Y_tr: {Y_tr.shape} pos={Y_tr.mean():.4f}, Y_te: {Y_te.shape} valid={len(valid_te_terms)}")

print("  Building BP/CC labels for domain importance...")
Y_tr_bp = np.stack([build_Y_tr(go, go_genes_tr_bp) for go in bp_terms], axis=1)
Y_tr_cc = np.stack([build_Y_tr(go, go_genes_tr_cc) for go in cc_terms], axis=1)

# ── 4. CDD domain matrix ─────────────────────────────────────────────────
print("[4] CDD domain matrix...")
dm_tr = (np.load(str(FEAT_DIR / "domain_matrix_train_cdd.npy")) > 0).astype(np.float32)
if dm_tr.shape[1] > N_CDD: dm_tr = dm_tr[:, :N_CDD]
print(f"  {dm_tr.shape}, nonzero: {(dm_tr.sum(1)>0).sum()}")

# ── 5. Pfam importance (MF+BP+CC) ────────────────────────────────────────
print("[5] Pfam importance (MF+BP+CC)...")
eps = 1e-6
all_Y_tr = np.concatenate([Y_tr, Y_tr_bp, Y_tr_cc], axis=1)
n_all = all_Y_tr.shape[1]
pfam_imp_all = np.zeros((N_CDD, n_all), dtype=np.float32)
for j in range(n_all):
    pos_mask = all_Y_tr[:, j] > 0
    if pos_mask.sum() < 5: continue
    n_pos = pos_mask.sum(); n_neg = (~pos_mask).sum()
    pr = (dm_tr[pos_mask].sum(0) + eps) / (n_pos + eps)
    nr = (dm_tr[~pos_mask].sum(0) + eps) / (n_neg + eps)
    pfam_imp_all[:, j] = np.clip(np.log(pr / nr), -5, 5)

pfam_imp_mf    = pfam_imp_all[:, :len(mf_terms)]
joint_dom_imp  = pfam_imp_all.max(axis=1)

# ── 6. Norm-ratio classification ─────────────────────────────────────────
print("[6] Norm-ratio classification...")
norms_te = np.linalg.norm(X_te_l30, axis=1)
norm_ratio = np.ones(len(mf_terms))
for j in range(len(mf_terms)):
    pos_mask = Y_te[:, j] > 0
    if pos_mask.sum() < 5: continue
    norm_ratio[j] = norms_te[pos_mask].mean() / norms_te[~pos_mask].mean()
domain_dep_mask = norm_ratio < NORM_RATIO_THR
print(f"  Domain-dep: {domain_dep_mask.sum()}/82, Motif-dep: {(~domain_dep_mask).sum()}/82")

# ── 7. Confidence weights ─────────────────────────────────────────────────
print("[7] Confidence weights...")
tr_gene_arr = np.array(tr_genes)
canon_dm_tr = {}
for g in set(tr_genes):
    idxs = np.where(tr_gene_arr == g)[0]
    importance_scores = dm_tr[idxs] @ joint_dom_imp
    canon_dm_tr[g] = dm_tr[idxs[np.argmax(importance_scores)]]

label_conf_tr = np.ones_like(Y_tr)
eps2 = 1e-6
for j in range(len(mf_terms)):
    if not domain_dep_mask[j]: continue
    for i, g in enumerate(tr_genes):
        cdm = canon_dm_tr.get(g, np.zeros(N_CDD, dtype=np.float32))
        denom = float(np.dot(cdm, pfam_imp_mf[:, j]))
        if denom < eps2: label_conf_tr[i, j] = 1.0
        else: label_conf_tr[i, j] = float(np.dot(dm_tr[i], pfam_imp_mf[:, j])) / denom

conf_weight_tr = np.ones_like(Y_tr)
for j in range(len(mf_terms)):
    if domain_dep_mask[j]:
        conf_weight_tr[:, j] = ALPHA_W + (1.0 - ALPHA_W) * np.clip(label_conf_tr[:, j], 0, 1)
conf_weight_tf = tf.constant(conf_weight_tr, dtype=tf.float32)

# ── 8. UniProt embeddings ─────────────────────────────────────────────────
print("[8] UniProt embeddings...")
emb_dict = np.load(UNI_DIR / "uniprot_iso_embeddings.npy", allow_pickle=True).item()
iso_ids_uniprot = list(emb_dict.keys())
emb_mat = np.stack([emb_dict[k] for k in iso_ids_uniprot], axis=0).astype(np.float32)
L30_uni   = emb_mat[:, :640]
delta_uni = emb_mat[:, 640:]
print(f"  UniProt isoforms: {len(iso_ids_uniprot)}, emb: {emb_mat.shape}")

# ── 9. Delta embeddings + scaling ────────────────────────────────────────
print("[9] Delta + scaling...")
delta_tr = X_tr_l30 - X_tr_l15
scaler = MaxAbsScaler()
delta_tr_s  = scaler.fit_transform(delta_tr)
delta_uni_s = scaler.transform(delta_uni)

# ── 10. Model ──────────────────────────────────────────────────────────────
n_go = len(mf_terms)

def make_mlp():
    inp1 = keras.Input(shape=(640,))
    inp2 = keras.Input(shape=(640,))
    x = layers.Concatenate()([inp1, inp2])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(n_go, activation='sigmoid')(x)
    return keras.Model([inp1, inp2], x)

@tf.function
def selective_focal_loss(y_true, y_pred, conf_weight, gamma=2.0):
    eps = 1e-7
    p   = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt  = y_true * p + (1.0 - y_true) * (1.0 - p)
    fl  = -(1.0 - pt) ** gamma * tf.math.log(pt)
    w   = y_true * conf_weight + (1.0 - y_true) * 1.0
    return tf.reduce_mean(w * fl)

# ── 11. Train 5 seeds, accumulate UniProt predictions ────────────────────
print("[10] Training (5 seeds)...")
print(f"  GPU: {gpus[0].name if gpus else 'CPU'}")

X_d = delta_tr_s; X_e = X_tr_l30
n_tr = len(tr_genes)
all_uni_preds = []

for seed in range(N_SEED):
    t0 = time.time()
    tf.random.set_seed(seed)
    mlp = make_mlp()
    opt = keras.optimizers.Adam(1e-3)

    split_idx = int(n_tr * 0.85)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tr)
    tr_idx = perm[:split_idx]; val_idx = perm[split_idx:]

    batch_size = 256
    best_val = -1; best_weights = None; no_improve = 0; patience = 10

    for epoch in range(100):
        rng2 = np.random.default_rng(epoch + seed * 1000)
        bp = rng2.permutation(len(tr_idx))
        for b in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[bp[b:b+batch_size]]
            xd_b = tf.constant(X_d[bidx])
            xe_b = tf.constant(X_e[bidx])
            yt_b = tf.constant(Y_tr[bidx])
            cw_b = tf.gather(conf_weight_tf, bidx.tolist())
            with tf.GradientTape() as tape:
                pred = mlp([xd_b, xe_b], training=True)
                loss = selective_focal_loss(yt_b, pred, cw_b)
            grads = tape.gradient(loss, mlp.trainable_variables)
            opt.apply_gradients(zip(grads, mlp.trainable_variables))

        val_pred = mlp([X_d[val_idx], X_e[val_idx]], training=False).numpy()
        val_y = Y_tr[val_idx]
        valid_j = val_y.sum(0) >= 2
        if valid_j.sum() > 0:
            val_auprc = np.mean([average_precision_score(val_y[:,j], val_pred[:,j])
                                  for j in np.where(valid_j)[0]])
        else:
            val_auprc = 0.0
        if val_auprc > best_val:
            best_val = val_auprc; best_weights = mlp.get_weights(); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    mlp.set_weights(best_weights)
    uni_pred = mlp([delta_uni_s, L30_uni], training=False).numpy()
    all_uni_preds.append(uni_pred)
    print(f"  Seed {seed}: best_val={best_val:.4f}, epochs={epoch+1}, {time.time()-t0:.0f}s")

ens_uni = np.mean(all_uni_preds, axis=0)
iso_to_score = {iso: ens_uni[i] for i, iso in enumerate(iso_ids_uniprot)}
np.save(OUT_DIR / "uniprot_selective_weight_scores.npy", ens_uni)
np.save(OUT_DIR / "uniprot_selective_weight_iso_ids.npy", np.array(iso_ids_uniprot))

# ── 12. UniProt 42-pair evaluation ───────────────────────────────────────
print("\n[11] UniProt 42-pair evaluation...")
bench_csv = ROOT / "reports/exp_g_uniprot/uniprot_isoform_benchmark.csv"

remap = {
    'GO:0004714': 'GO:0004713', 'GO:0005007': 'GO:0004713',
    'GO:0004693': 'GO:0004674', 'GO:0097553': 'GO:0046982',
    'GO:0005244': 'GO:0046982', 'GO:0004197': 'GO:0003824',
    'GO:0006281': 'GO:0003677', 'GO:0006977': 'GO:0003700',
    'GO:0008285': 'GO:0019901', 'GO:0005178': 'GO:0048018',
    'GO:0005158': 'GO:0048018', 'GO:0008083': 'GO:0008083',
    'GO:0000398': 'GO:0003723', 'GO:0006357': 'GO:0003700',
    'GO:0005200': 'GO:0003779', 'GO:0007399': 'GO:0005515',
    'GO:0016079': 'GO:0046982', 'GO:0051015': 'GO:0003779',
    'GO:0019901': 'GO:0019901', 'GO:0008270': 'GO:0005515',
}
go_set = set(mf_terms)

# Load v17f* baseline for flip analysis
v17f_results = {}
v17f_path = ROOT / "reports/exp_h_uniprot_eval/pairwise_eval_v3_remapped.tsv"
if v17f_path.exists():
    with open(v17f_path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key = (row['gene'], row['iso_a'], row['iso_b'], row['go_term'])
            v17f_results[key] = row

results_sw = []
n_eval = 0; n_correct = 0
with open(bench_csv) as f:
    for row in csv.DictReader(f):
        iso_a = row['iso_a']; iso_b = row['iso_b']
        go_raw = row['go_term'].replace('GO_', 'GO:')
        go_norm = remap.get(go_raw, go_raw)
        direction = row['direction']

        if iso_a not in iso_to_score or iso_b not in iso_to_score:
            results_sw.append({'gene': row['gene'], 'iso_a': iso_a, 'iso_b': iso_b,
                                'go': go_norm, 'note': 'not embedded', 'correct': None})
            continue

        go_idx_arr = np.where(mf_terms == go_norm)[0]
        if len(go_idx_arr) == 0:
            results_sw.append({'gene': row['gene'], 'iso_a': iso_a, 'iso_b': iso_b,
                                'go': go_norm, 'note': f'GO not in model ({go_raw}→{go_norm})', 'correct': None})
            continue

        j = go_idx_arr[0]
        sa = float(iso_to_score[iso_a][j])
        sb = float(iso_to_score[iso_b][j])
        gap = abs(sa - sb)

        # Is this a domain-dep term?
        is_domain_dep = domain_dep_mask[j]

        if direction == 'A_only': pred_correct = sa > sb
        elif direction == 'B_only': pred_correct = sb > sa
        else: pred_correct = True

        n_eval += 1
        if pred_correct: n_correct += 1
        results_sw.append({
            'gene': row['gene'], 'iso_a': iso_a, 'iso_b': iso_b,
            'go': go_norm, 'direction': direction,
            'sa': sa, 'sb': sb, 'gap': gap,
            'correct': pred_correct, 'is_domain_dep': is_domain_dep,
            'note': 'OK'
        })

print(f"\n=== selective_weight UniProt Results ===")
print(f"  Evaluable: {n_eval}, Correct: {n_correct}, Accuracy: {n_correct/n_eval:.3f} ({n_correct}/{n_eval})")
print(f"  v17f* baseline:     26/42 = 0.619")
print(f"  confweight_a02:     23/42 = 0.548")

# High-gap subset
gaps = [r['gap'] for r in results_sw if r.get('gap') is not None]
med_gap = np.median(gaps) if gaps else 0
hi_gap = [r for r in results_sw if r.get('gap', 0) > med_gap and r.get('correct') is not None]
hi_n = len(hi_gap); hi_correct = sum(r['correct'] for r in hi_gap)
print(f"  High-gap (>{med_gap:.3f}): {hi_correct}/{hi_n} = {hi_correct/hi_n:.3f}")

# Domain-dep vs motif-dep breakdown
dd_r = [r for r in results_sw if r.get('correct') is not None and r.get('is_domain_dep')]
md_r = [r for r in results_sw if r.get('correct') is not None and not r.get('is_domain_dep', True)]
if dd_r:
    dd_correct = sum(r['correct'] for r in dd_r)
    print(f"  Domain-dep terms: {dd_correct}/{len(dd_r)} = {dd_correct/len(dd_r):.3f}")
if md_r:
    md_correct = sum(r['correct'] for r in md_r)
    print(f"  Motif-dep terms:  {md_correct}/{len(md_r)} = {md_correct/len(md_r):.3f}")

# Flip analysis vs v17f*
newly_right, newly_wrong = [], []
for r in results_sw:
    if r.get('correct') is None: continue
    key = (r['gene'], r['iso_a'], r['iso_b'], r['go'])
    v = v17f_results.get(key)
    if v is None: continue
    v_corr = v.get('correct', '').lower() == 'true'
    if not v_corr and r['correct']: newly_right.append(r)
    elif v_corr and not r['correct']: newly_wrong.append(r)

print(f"\n  Flip analysis vs v17f*:")
print(f"  Newly correct: {len(newly_right)}")
for r in newly_right:
    dep = "domain-dep" if r.get('is_domain_dep') else "motif-dep"
    print(f"    {r['gene']} {r['go']} {r['direction']} ({dep}): sa={r['sa']:.4f} sb={r['sb']:.4f} gap={r['gap']:.4f}")
print(f"  Newly wrong:   {len(newly_wrong)}")
for r in newly_wrong:
    dep = "domain-dep" if r.get('is_domain_dep') else "motif-dep"
    print(f"    {r['gene']} {r['go']} {r['direction']} ({dep}): sa={r['sa']:.4f} sb={r['sb']:.4f} gap={r['gap']:.4f}")

# Save
out_tsv = OUT_DIR / "pairwise_eval_selective_weight.tsv"
with open(out_tsv, 'w') as f:
    f.write("gene\tiso_a\tiso_b\tgo\tdirection\tsa\tsb\tgap\tcorrect\tis_domain_dep\tnote\n")
    for r in results_sw:
        f.write('\t'.join([str(r.get(k,'')) for k in
                           ['gene','iso_a','iso_b','go','direction','sa','sb','gap','correct','is_domain_dep','note']]) + '\n')

# Save result summary
result = {
    'evaluable': n_eval, 'correct': n_correct,
    'accuracy': float(n_correct/n_eval) if n_eval else 0,
    'v17f_star_baseline': '26/42=0.619',
    'confweight_a02': '23/42=0.548',
    'high_gap_correct': hi_correct, 'high_gap_n': hi_n,
    'domain_dep_correct': sum(r['correct'] for r in dd_r) if dd_r else 0,
    'domain_dep_n': len(dd_r),
    'motif_dep_correct': sum(r['correct'] for r in md_r) if md_r else 0,
    'motif_dep_n': len(md_r),
    'newly_right': len(newly_right),
    'newly_wrong': len(newly_wrong),
}
with open(OUT_DIR / "uniprot_eval_result.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: {out_tsv}")
print("Done.")
