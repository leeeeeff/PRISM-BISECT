"""
exp_uniprot_confweight.py
  confweight_a02 모델을 UniProt 42-pair 벤치마크에 적용.
  기존 uniprot_iso_embeddings.npy (1280-dim = [L30, delta]) 재사용.
  confweight_a02와 동일한 구조로 재훈련 → UniProt 예측.
"""
import numpy as np, os, time, csv, json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gzip as gz

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ROOT     = Path("/home/welcome1/sw1686/DIFFUSE")
DATA_DIR = str(ROOT / "hMuscle/data")
ID_DIR   = str(ROOT / "hMuscle/data/raw_data/data/id_lists")
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"
OUT_DIR  = ROOT / "reports/exp_h_uniprot_eval"

LAYER_A, LAYER_B = 15, 30
N_SEED, ALPHA_W  = 5, 0.2

# ── 1. Train embeddings ────────────────────────────────────────────────────
print("[1] Train embeddings...")
X_tr_l30 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_B:02d}_t30_150M.npy').astype(np.float32)
X_tr_l15 = np.load(f'{DATA_DIR}/esm2_train_human_layer{LAYER_A:02d}_t30_150M.npy').astype(np.float32)

# ── 2. IDs ────────────────────────────────────────────────────────────────
print("[2] IDs...")
tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes = [x.decode() if isinstance(x, bytes) else x for x in tr_genes_raw]

# ── 3. Labels (same as confweight_a02) ────────────────────────────────────
print("[3] Labels...")
ANNOT_DIR = f'{DATA_DIR}/raw_data/data/annotations'
sym2id = {}
with gz.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
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
go_genes_tr = defaultdict(set)
with gz.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 8: continue
        if p[0] != '9606' or p[7] != 'Function': continue
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_entrez) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
print(f"  {len(mf_terms)} MF | Y_tr pos={Y_tr.mean():.4f}")

# ── 4. CDD confidence weights (same as confweight_a02) ────────────────────
print("[4] CDD + label confidence...")
cdd_mat_path  = FEAT_DIR / "domain_matrix_train_cdd.npy"
cdd_vocab_path = FEAT_DIR / "domain_cdd_vocab.txt"
dm_tr = np.load(cdd_mat_path)
print(f"  CDD matrix: {dm_tr.shape}")

eps2 = 1e-8
n_cdd = dm_tr.shape[1]
imp_pos_tr = np.zeros((n_cdd, len(mf_terms)), dtype=np.float32)
for j in range(len(mf_terms)):
    pos_mask = Y_tr[:, j] > 0
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum(); n_neg = neg_mask.sum()
    if n_pos < 5: continue
    pr = (dm_tr[pos_mask].sum(0) + eps2) / (n_pos + eps2)
    nr = (dm_tr[neg_mask].sum(0) + eps2) / (n_neg + eps2)
    imp_pos_tr[:, j] = np.clip(np.log(pr / nr), -5, 5)

canon_dm_tr = {}
tr_gene_arr = np.array(tr_genes)
for g in set(tr_genes):
    mask = tr_gene_arr == g
    dc = dm_tr[mask].sum(1)
    best = np.argmax(dc)
    canon_dm_tr[g] = dm_tr[mask][best]

label_conf_tr = np.zeros_like(Y_tr)
for i, g in enumerate(tr_genes):
    cdm = canon_dm_tr.get(g, np.zeros(n_cdd, dtype=np.float32))
    for j in range(len(mf_terms)):
        denom = float(np.dot(cdm, imp_pos_tr[:, j]))
        if denom < eps2:
            label_conf_tr[i, j] = 1.0
        else:
            label_conf_tr[i, j] = float(np.dot(dm_tr[i], imp_pos_tr[:, j])) / denom

conf_weight_tr = ALPHA_W + (1.0 - ALPHA_W) * np.clip(label_conf_tr, 0, 1)
conf_weight_tf = tf.constant(conf_weight_tr, dtype=tf.float32)

# ── 5. Delta embeddings ────────────────────────────────────────────────────
print("[5] Delta embeddings + scaling...")
delta_tr = X_tr_l30 - X_tr_l15
scaler   = MaxAbsScaler()
delta_tr_s = scaler.fit_transform(delta_tr)
X_tr_in  = [delta_tr_s, X_tr_l30]

# ── 6. UniProt embeddings ─────────────────────────────────────────────────
print("[6] UniProt embeddings...")
emb_dict = np.load(OUT_DIR / "uniprot_iso_embeddings.npy", allow_pickle=True).item()
# Format: {uniprot_id: 1280-dim = [L30 (640), delta=L30-L15 (640)]}
iso_ids_uniprot = list(emb_dict.keys())
emb_mat = np.stack([emb_dict[k] for k in iso_ids_uniprot], axis=0).astype(np.float32)
L30_uni   = emb_mat[:, :640]              # L30
delta_uni = emb_mat[:, 640:]              # L30 - L15 (unscaled)
delta_uni_s = scaler.transform(delta_uni)
X_uni_in  = np.concatenate([delta_uni_s, L30_uni], axis=1)
print(f"  UniProt isoforms: {len(iso_ids_uniprot)}, embedding: {emb_mat.shape}")

# ── 7. Model definition ────────────────────────────────────────────────────
n_go = len(mf_terms)
n_feat = X_tr_in[0].shape[1] + X_tr_in[1].shape[1]

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
def conf_focal_loss(y_true, y_pred, conf_weight, gamma=2.0):
    eps = 1e-7
    p  = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pt = y_true * p + (1.0 - y_true) * (1.0 - p)
    fl = -(1.0 - pt) ** gamma * tf.math.log(pt)
    w  = y_true * conf_weight + (1.0 - y_true) * 1.0
    return tf.reduce_mean(w * fl)

# ── 8. Train 5 seeds, accumulate UniProt predictions ─────────────────────
print("[8] Training (5 seeds)...")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU: {gpus[0].name if gpus else 'CPU'}")

n_tr = len(tr_genes)
all_uni_preds = []

for seed in [42, 1, 2, 3, 4]:
    t0 = time.time()
    tf.random.set_seed(seed)
    mlp = make_mlp()
    opt = keras.optimizers.Adam(1e-3)

    split_idx = int(n_tr * 0.85)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_tr)
    tr_idx = perm[:split_idx]; val_idx = perm[split_idx:]

    X_d = delta_tr_s; X_e = X_tr_l30
    batch_size = 256
    best_val = -1; best_weights = None; no_improve = 0; patience = 10

    for epoch in range(100):
        rng2 = np.random.default_rng(epoch + seed * 1000)
        batch_perm = rng2.permutation(len(tr_idx))
        for b in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[batch_perm[b:b+batch_size]]
            xd_b = tf.constant(X_d[bidx])
            xe_b = tf.constant(X_e[bidx])
            yt_b = tf.constant(Y_tr[bidx])
            cw_b = tf.gather(conf_weight_tf, bidx.tolist())
            with tf.GradientTape() as tape:
                pred = mlp([xd_b, xe_b], training=True)
                loss = conf_focal_loss(yt_b, pred, cw_b)
            grads = tape.gradient(loss, mlp.trainable_variables)
            opt.apply_gradients(zip(grads, mlp.trainable_variables))

        val_pred = mlp([X_d[val_idx], X_e[val_idx]], training=False).numpy()
        val_y    = Y_tr[val_idx]
        valid_j  = val_y.sum(0) >= 2
        if valid_j.sum() > 0:
            val_auprc = np.mean([average_precision_score(val_y[:, j], val_pred[:, j])
                                  for j in np.where(valid_j)[0]])
        else:
            val_auprc = 0.0

        if val_auprc > best_val:
            best_val = val_auprc
            best_weights = mlp.get_weights()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    mlp.set_weights(best_weights)
    uni_pred = mlp([delta_uni_s, L30_uni], training=False).numpy()
    all_uni_preds.append(uni_pred)
    print(f"  Seed {seed}: best_val={best_val:.4f}, epochs={epoch+1}, {time.time()-t0:.0f}s")

ens_uni = np.mean(all_uni_preds, axis=0)  # (n_uniprot_isoforms, 82)
iso_to_score = {iso_id: ens_uni[i] for i, iso_id in enumerate(iso_ids_uniprot)}
# Save full score matrix for later analysis
np.save(OUT_DIR / "uniprot_confweight_a02_scores.npy", ens_uni)
np.save(OUT_DIR / "uniprot_confweight_a02_iso_ids.npy", np.array(iso_ids_uniprot))

# ── 9. Pairwise benchmark evaluation ─────────────────────────────────────
print("\n[9] Pairwise evaluation...")
bench_csv = ROOT / "reports/exp_g_uniprot/uniprot_isoform_benchmark.csv"
# Load v17f* remapped results to compare directly
v17f_results = {}
with open(OUT_DIR / "pairwise_eval_v3_remapped.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row['gene'], row['iso_a'], row['iso_b'], row['go_term'])
        v17f_results[key] = row

# GO term mapping with remap dict (same as original eval script)
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

results_cw = []
n_eval = 0; n_correct = 0
with open(bench_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        iso_a = row['iso_a']
        iso_b = row['iso_b']
        go_raw = row['go_term'].replace('GO_', 'GO:')
        go_norm = remap.get(go_raw, go_raw)  # apply remap
        direction = row['direction']

        if iso_a not in iso_to_score or iso_b not in iso_to_score:
            results_cw.append({'gene': row['gene'], 'iso_a': iso_a, 'iso_b': iso_b,
                                'go': go_norm, 'note': 'not embedded', 'correct': None})
            continue

        go_idx_arr = np.where(mf_terms == go_norm)[0]
        if len(go_idx_arr) == 0:
            results_cw.append({'gene': row['gene'], 'iso_a': iso_a, 'iso_b': iso_b,
                                'go': go_norm, 'note': f'GO not in model ({go_raw}→{go_norm})', 'correct': None})
            continue

        j = go_idx_arr[0]
        sa = float(iso_to_score[iso_a][j])
        sb = float(iso_to_score[iso_b][j])
        gap = abs(sa - sb)
        if direction == 'A_only':
            pred_correct = sa > sb
        elif direction == 'B_only':
            pred_correct = sb > sa
        else:  # 'both'
            pred_correct = True
        n_eval += 1
        if pred_correct: n_correct += 1
        results_cw.append({'gene': row['gene'], 'iso_a': iso_a, 'iso_b': iso_b,
                            'go': go_norm, 'direction': direction,
                            'sa': sa, 'sb': sb, 'gap': gap,
                            'correct': pred_correct, 'note': 'OK'})

print(f"\n=== confweight_a02 UniProt Results ===")
print(f"  Evaluable: {n_eval}, Correct: {n_correct}, Accuracy: {n_correct/n_eval:.3f} ({n_correct}/{n_eval})")
print(f"  v17f* baseline:   26/42 = 0.619")

# High-gap subset
gaps = [r['gap'] for r in results_cw if r.get('gap') is not None]
med_gap = np.median(gaps) if gaps else 0
hi_gap = [r for r in results_cw if r.get('gap', 0) > med_gap and r.get('correct') is not None]
hi_n = len(hi_gap)
hi_correct = sum(r['correct'] for r in hi_gap)
print(f"  High-gap subset: {hi_correct}/{hi_n} = {hi_correct/hi_n:.3f}")

# Flip analysis vs v17f*
print(f"\nFlip analysis vs v17f*:")
newly_right = []
newly_wrong = []
for r in results_cw:
    if r.get('correct') is None: continue
    key = (r['gene'], r['iso_a'], r['iso_b'], r['go'].replace('GO:', 'GO_'))
    v17f = v17f_results.get(key)
    if v17f is None: continue
    v17f_correct = v17f.get('correct', '').lower() == 'true'
    if not v17f_correct and r['correct']:
        newly_right.append(r)
    elif v17f_correct and not r['correct']:
        newly_wrong.append(r)

print(f"  Newly correct: {len(newly_right)}")
for r in newly_right:
    print(f"    {r['gene']} {r['go']} {r['direction']}: sa={r['sa']:.4f} sb={r['sb']:.4f}")
print(f"  Newly wrong:   {len(newly_wrong)}")
for r in newly_wrong:
    print(f"    {r['gene']} {r['go']} {r['direction']}: sa={r['sa']:.4f} sb={r['sb']:.4f}")

# Save
out_path = OUT_DIR / "pairwise_eval_confweight_a02.tsv"
with open(out_path, 'w') as f:
    f.write("gene\tiso_a\tiso_b\tgo\tdirection\tsa\tsb\tgap\tcorrect\tnote\n")
    for r in results_cw:
        f.write('\t'.join([str(r.get(k, '')) for k in
                           ['gene','iso_a','iso_b','go','direction','sa','sb','gap','correct','note']]) + '\n')
print(f"\nSaved: {out_path}")
