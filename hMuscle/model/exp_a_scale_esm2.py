#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a_scale_esm2.py
-------------------
ESM-2 scale generalization experiment.
Tests whether δ_layer principle holds for 8M (6-layer) and 35M (12-layer) models.

For each scale:
  δ = φ_L_final - φ_L_mid  (mid = 50% of total depth)
  8M:  δ = L6 - L3   (dim=320)
  35M: δ = L12 - L6  (dim=480)
  150M: δ = L30 - L15 (dim=640)  ← reference

Comparison:
  PRISM(scale) = plain ESM-2 MLP (no δ)
  v17f(scale)  = T_ψ + δ_layer + Stage 2
"""

import os, re, gzip, json, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/exp_a_scale'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS       = [42, 7, 13, 21, 99]
MARGIN      = 0.3
EPOCHS_T    = 50
EPOCHS_MLP  = 60
BATCH       = 512
EMBED_DIM_T = 64
MAX_TRIPS   = 30000
TRIPS_PER   = 300

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

# ── Protein sequence parsing ─────────────────────────────────────
TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}

def parse_pep_file(pep_path, max_len=1022):
    records = {}
    cur_id = cur_meta = None; cur_seq = []
    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None: return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq: return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank, score, length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq)
    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m_id   = re.match(r'>(\S+)', line)
                m_type = re.search(r'ORF type:(\S+)', line)
                m_sc   = re.search(r'score=([\d.]+)', line)
                m_len  = re.search(r'len:(\d+)', line)
                if not m_id: cur_id = None; continue
                raw_id = m_id.group(1)
                cur_id = re.sub(r'\.p\d+$', '', raw_id)
                orf_type = m_type.group(1) if m_type else 'internal'
                rank = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, float(m_sc.group(1)) if m_sc else 0.0,
                            int(m_len.group(1)) if m_len else 0)
            else:
                cur_seq.append(line)
    flush()
    return {k: v[3][:max_len] for k, v in records.items()}

def parse_fasta(fasta_path, max_len=1022):
    records = {}; cur_id = None; cur_seq = []
    def flush():
        if cur_id is None: return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if seq: records[cur_id] = seq[:max_len]
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m = re.match(r'>(\S+)', line)
                cur_id = m.group(1) if m else None
            else:
                cur_seq.append(line)
    flush()
    return records

# ── GO labels ───────────────────────────────────────────────────
def load_go_setup():
    import gzip as gz
    tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
    tr_genes     = [clean(g) for g in tr_genes_raw]
    ENSG2SYM = {}
    with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
    te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
    te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                    for g in te_genes_raw]

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
    tr_ids = [sym2id.get(g, g) for g in tr_genes]
    tr_id_set = set(tr_ids)
    go_genes_tr = defaultdict(set); go_genes_all = defaultdict(set)
    with gz.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] != '9606': continue
            if p[7] != 'Function': continue
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
    def build_Y_tr(go_id):
        pos_ids  = go_genes_tr[go_id]
        pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
        y = np.zeros(len(tr_genes), dtype=np.float32)
        for sym in pos_syms:
            for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
        return y
    def build_Y_te(go_id):
        pos_ids = go_genes_all[go_id]
        return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                         for s in te_sym_list], dtype=np.float32)
    Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
    Y_te = np.stack([build_Y_te(go)  for go in mf_terms], axis=1)
    valid_mask = Y_te.sum(0) >= 2

    H2_LAYERS = {}
    with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 12: H2_LAYERS[p[0]] = p[11]
    layer_groups = defaultdict(list)
    for i, go in enumerate(mf_terms):
        if valid_mask[i]: layer_groups[H2_LAYERS.get(go, 'Other')].append(i)

    gene2idxs_tr = defaultdict(list)
    for i, g in enumerate(tr_genes): gene2idxs_tr[g].append(i)
    gene2idxs_te = defaultdict(list)
    for i, g in enumerate(te_sym_list): gene2idxs_te[g].append(i)

    return (tr_genes, te_sym_list, Y_tr, Y_te, valid_mask,
            mf_terms, layer_groups, gene2idxs_tr, gene2idxs_te, sym2id)

# ── ESM-2 embedding computation ──────────────────────────────────
import torch

@torch.no_grad()
def compute_layer_embs(model, batch_converter, sequences, device, layers_needed):
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)
    results = model(tokens, repr_layers=layers_needed, return_contacts=False)
    out = {l: [] for l in layers_needed}
    for l in layers_needed:
        token_reps = results['representations'][l]  # (B, L+2, D)
        for i, (_, seq) in enumerate(sequences):
            seq_len = len(seq)
            emb = token_reps[i, 1:seq_len+1, :].mean(dim=0).cpu().float().numpy()
            out[l].append(emb)
    return {l: np.array(out[l]) for l in layers_needed}

def compute_scale_embeddings(model_tag, n_layers, mid_layer,
                              pep_path_te, fasta_path_tr,
                              iso_list_te, iso_list_tr, device, batch_size=64):
    import esm
    print(f"\n  Loading {model_tag}...")
    loader = getattr(esm.pretrained, model_tag)
    model, alphabet = loader()
    model = model.eval().to(device)
    bc    = alphabet.get_batch_converter()
    D = model.embed_dim
    print(f"  {model_tag}: {n_layers} layers, D={D}")

    layers_needed = [mid_layer, n_layers]  # L_mid and L_final

    out_cache = {}
    for split_name, iso_list, pep_or_fasta, is_pep in [
        ('test',  iso_list_te, pep_path_te,    True),
        ('train', iso_list_tr, fasta_path_tr,  False),
    ]:
        print(f"  Computing {split_name} embeddings ({len(iso_list)} isoforms)...")
        if is_pep:
            seqs = parse_pep_file(pep_or_fasta)
        else:
            seqs = parse_fasta(pep_or_fasta)
        N = len(iso_list)
        emb_final = np.zeros((N, D), dtype=np.float32)
        emb_mid   = np.zeros((N, D), dtype=np.float32)
        valid_idx = [(i, iso_list[i]) for i in range(N) if iso_list[i] in seqs]
        print(f"    Coverage: {len(valid_idx)}/{N}")
        t0 = time.time()
        for b_start in range(0, len(valid_idx), batch_size):
            batch = valid_idx[b_start:b_start+batch_size]
            batch_seqs = [(iid, seqs[iid]) for _, iid in batch]
            layer_embs = compute_layer_embs(model, bc, batch_seqs, device, layers_needed)
            for j, (i, _) in enumerate(batch):
                emb_final[i] = layer_embs[n_layers][j]
                emb_mid[i]   = layer_embs[mid_layer][j]
        print(f"    Done in {time.time()-t0:.0f}s")
        out_cache[f'{split_name}_final'] = emb_final
        out_cache[f'{split_name}_mid']   = emb_mid
    del model; torch.cuda.empty_cache()
    return out_cache, D

# ── Model building ───────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers as klayers, models as kmodels, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy

def build_Tpsi(delta_dim, embed_dim=EMBED_DIM_T):
    inp = klayers.Input(shape=(delta_dim,))
    x   = klayers.Dense(256, activation='relu')(inp)
    x   = klayers.BatchNormalization()(x)
    x   = klayers.Dropout(0.3)(x)
    x   = klayers.Dense(embed_dim)(x)
    x   = klayers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
    return kmodels.Model(inp, x, name='Tpsi')

def build_stage2_mlp(t_dim, esm_dim, n_labels):
    inp_t = klayers.Input(shape=(t_dim,))
    inp_e = klayers.Input(shape=(esm_dim,))
    x = klayers.Concatenate()([inp_t, inp_e])
    x = klayers.Dense(256, activation='relu')(x)
    x = klayers.BatchNormalization()(x)
    x = klayers.Dropout(0.3)(x)
    x = klayers.Dense(128, activation='relu')(x)
    x = klayers.Dropout(0.2)(x)
    out = klayers.Dense(n_labels, activation='sigmoid')(x)
    return kmodels.Model([inp_t, inp_e], out, name='Stage2')

def build_prism(esm_dim, n_labels):
    inp = klayers.Input(shape=(esm_dim,))
    x = klayers.Dense(256, activation='relu')(inp)
    x = klayers.BatchNormalization()(x)
    x = klayers.Dropout(0.3)(x)
    x = klayers.Dense(128, activation='relu')(x)
    x = klayers.Dropout(0.2)(x)
    out = klayers.Dense(n_labels, activation='sigmoid')(x)
    return kmodels.Model(inp, out, name='PRISM')

def triplet_loss_fn(anchor, positive, negative, margin=MARGIN):
    d_pos = 1 - tf.reduce_sum(anchor * positive, axis=-1)
    d_neg = 1 - tf.reduce_sum(anchor * negative, axis=-1)
    return tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

def build_triplets_go(gene_arr, Y_tr, max_trips=MAX_TRIPS, per_go=TRIPS_PER):
    rng = np.random.default_rng(42)
    trip_a, trip_p, trip_n = [], [], []
    for k in range(Y_tr.shape[1]):
        y_k      = Y_tr[:, k]
        pos_idxs = np.where(y_k == 1)[0]
        neg_idxs = np.where(y_k == 0)[0]
        if len(pos_idxs) < 5 or len(neg_idxs) < 10: continue
        if len(trip_a) >= max_trips: break
        n_anchor = min(per_go, len(pos_idxs))
        for a_idx in rng.choice(pos_idxs, n_anchor, replace=False):
            a_gene    = gene_arr[a_idx]
            cross_pos = pos_idxs[gene_arr[pos_idxs] != a_gene]
            cross_neg = neg_idxs[gene_arr[neg_idxs] != a_gene]
            if len(cross_pos) < 2 or len(cross_neg) < 2: continue
            trip_a.append(a_idx)
            trip_p.append(int(rng.choice(cross_pos)))
            trip_n.append(int(rng.choice(cross_neg)))
            if len(trip_a) >= max_trips: break
        if len(trip_a) >= max_trips: break
    return (np.array(trip_a, dtype=np.int32),
            np.array(trip_p, dtype=np.int32),
            np.array(trip_n, dtype=np.int32))

def run_experiment(scale_name, embs_tr_final, embs_tr_mid,
                   embs_te_final, embs_te_mid,
                   tr_genes, te_sym_list, Y_tr, Y_te, valid_mask,
                   mf_terms, layer_groups, gene2idxs_tr, gene2idxs_te, sym2id):
    print(f"\n{'='*65}")
    print(f"  Running {scale_name}: δ_layer + T_ψ + Stage 2")
    print(f"{'='*65}")
    t_total = time.time()

    D = embs_te_final.shape[1]
    n_labels = len(mf_terms)

    # δ_layer
    delta_tr = (embs_tr_final - embs_tr_mid).astype(np.float32)
    delta_te = (embs_te_final - embs_te_mid).astype(np.float32)
    scaler = MaxAbsScaler()
    delta_tr_s = scaler.fit_transform(delta_tr).astype(np.float32)
    delta_te_s = scaler.transform(delta_te).astype(np.float32)

    n_zero = (np.linalg.norm(delta_te_s, axis=1) < 1e-6).sum()
    print(f"  δ_layer norm mean: {np.linalg.norm(delta_te_s, axis=1).mean():.3f}  zero: {n_zero}/{len(delta_te_s)}")

    # Build gene array for triplet construction
    gene_arr_tr = np.array(tr_genes)

    # Triplets (GO-label cross-gene, same as v17f)
    print(f"  Building triplets...")
    A, P, N = build_triplets_go(gene_arr_tr, Y_tr, MAX_TRIPS, TRIPS_PER)
    print(f"  Triplets: {len(A)}")

    # PRISM baseline
    print(f"\n  [A] PRISM baseline (φ_L_final only)...")
    prism_preds = []
    focal = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(embs_tr_final))
        n_val = int(len(embs_tr_final) * 0.1)
        vi = perm[:n_val]; ti = perm[n_val:]
        mlp = build_prism(D, n_labels)
        mlp.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        mlp.fit(embs_tr_final[ti], Y_tr[ti],
                validation_data=(embs_tr_final[vi], Y_tr[vi]),
                epochs=EPOCHS_MLP, batch_size=BATCH,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0)
        prism_preds.append(mlp.predict(embs_te_final, batch_size=1024, verbose=0))
        print(f"    seed {seed} done")
    prism_ens = np.mean(prism_preds, axis=0)

    # v17f-style: T_ψ + Stage 2
    print(f"\n  [B] T_ψ Stage 1 training...")
    tpsi_model = build_Tpsi(D)
    tpsi_opt   = optimizers.Adam(1e-3)
    ds_tr_set  = tf.data.Dataset.from_tensor_slices(
        (delta_tr_s[A], delta_tr_s[P], delta_tr_s[N])
    ).shuffle(10000).batch(BATCH)
    for epoch in range(EPOCHS_T):
        losses = []
        for da, dp, dn in ds_tr_set:
            with tf.GradientTape() as tape:
                ea = tpsi_model(da, training=True)
                ep = tpsi_model(dp, training=True)
                en = tpsi_model(dn, training=True)
                loss = triplet_loss_fn(ea, ep, en)
            grads = tape.gradient(loss, tpsi_model.trainable_variables)
            tpsi_opt.apply_gradients(zip(grads, tpsi_model.trainable_variables))
            losses.append(float(loss))
        if epoch % 10 == 0 or epoch == EPOCHS_T - 1:
            # active ratio
            ea = tpsi_model(delta_tr_s[A[:1000]], training=False)
            ep = tpsi_model(delta_tr_s[P[:1000]], training=False)
            en = tpsi_model(delta_tr_s[N[:1000]], training=False)
            d_pos = 1 - tf.reduce_sum(ea*ep, axis=-1).numpy()
            d_neg = 1 - tf.reduce_sum(ea*en, axis=-1).numpy()
            active = (d_pos - d_neg + MARGIN > 0).mean()
            print(f"    epoch {epoch:3d} loss={np.mean(losses):.4f} active={active:.3f}")

    T_tr = tpsi_model.predict(delta_tr_s, batch_size=1024, verbose=0)
    T_te = tpsi_model.predict(delta_te_s, batch_size=1024, verbose=0)

    print(f"\n  [B] Stage 2 MLP training...")
    v17f_preds = []
    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        perm = np.random.permutation(len(T_tr))
        n_val = int(len(T_tr) * 0.1)
        vi = perm[:n_val]; ti = perm[n_val:]
        mlp2 = build_stage2_mlp(EMBED_DIM_T, D, n_labels)
        mlp2.compile(optimizer=optimizers.Adam(1e-3), loss=focal)
        mlp2.fit(
            [T_tr[ti], embs_tr_final[ti]], Y_tr[ti],
            validation_data=([T_tr[vi], embs_tr_final[vi]], Y_tr[vi]),
            epochs=EPOCHS_MLP, batch_size=BATCH,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0)
        v17f_preds.append(mlp2.predict([T_te, embs_te_final], batch_size=1024, verbose=0))
        print(f"    seed {seed} done")
    v17f_ens = np.mean(v17f_preds, axis=0)

    # Evaluation
    valid_idx = [i for i in range(n_labels) if valid_mask[i]]
    def eval_group(preds, label, idx_list):
        aps = [average_precision_score(Y_te[:,i], preds[:,i])
               for i in idx_list if Y_te[:,i].sum() >= 2]
        m = float(np.mean(aps)) if aps else float('nan')
        print(f"    {label:<30} {m:.4f}  (n={len(aps)})")
        return m

    print(f"\n  Results for {scale_name}:")
    print(f"  {'Layer':<30} {'PRISM':>7} {'v17f':>7} {'Δ':>7}")
    results = {'scale': scale_name, 'embed_dim': D, 'prism': {}, 'v17f': {}}
    for group_name, idxs in [
        ('All MF', valid_idx),
        *[(k, v) for k, v in layer_groups.items() if v]
    ]:
        p = eval_group(prism_ens, f'  PRISM {group_name}', idxs)
        f_val = eval_group(v17f_ens, f'  v17f  {group_name}', idxs)
        print(f"  {group_name:<30} {p:>7.4f} {f_val:>7.4f} {f_val-p:>+7.4f}")
        results['prism'][group_name] = p
        results['v17f'][group_name]  = f_val

    results['elapsed'] = time.time() - t_total
    print(f"\n  Total time: {results['elapsed']:.0f}s")
    return results


# ── MAIN ─────────────────────────────────────────────────────────
print("=" * 65)
print("  Experiment A: ESM-2 Scale Generalization")
print("=" * 65)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    device = torch.device('cuda:0')
    print(f"\n  GPU: {gpus[0].name}")
else:
    device = torch.device('cpu')
    print("\n  CPU mode")

# Load shared data
print("\n[1] Loading GO labels and gene info...")
(tr_genes, te_sym_list, Y_tr, Y_te, valid_mask,
 mf_terms, layer_groups, gene2idxs_tr, gene2idxs_te, sym2id) = load_go_setup()
print(f"  Train: {len(tr_genes)} | Test: {len(te_sym_list)} | Terms: {len(mf_terms)}")

# Isoform ID lists
iso_list_te_raw = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
iso_list_te = [clean(x) for x in iso_list_te_raw]
iso_list_tr_raw = np.load(f'{ID_DIR}/train_isoform_list.npy', allow_pickle=True)
iso_list_tr = [clean(x) for x in iso_list_tr_raw]

PEP_TE    = f'{DATA_DIR}/top30k_isoforms.pep'
FASTA_TR  = '../results_isoform/features/train_proteins.fasta'

scales = [
    ('ESM2_8M',  'esm2_t6_8M_UR50D',  6, 3),
    ('ESM2_35M', 'esm2_t12_35M_UR50D', 12, 6),
]

all_results = {}
for scale_name, model_tag, n_layers, mid_layer in scales:
    print(f"\n{'='*65}")
    print(f"  Scale: {scale_name}  mid_layer=L{mid_layer}")
    print(f"{'='*65}")
    emb_cache, D = compute_scale_embeddings(
        model_tag, n_layers, mid_layer,
        PEP_TE, FASTA_TR,
        iso_list_te, iso_list_tr, device
    )
    res = run_experiment(
        scale_name,
        emb_cache['train_final'], emb_cache['train_mid'],
        emb_cache['test_final'],  emb_cache['test_mid'],
        tr_genes, te_sym_list, Y_tr, Y_te, valid_mask,
        mf_terms, layer_groups, gene2idxs_tr, gene2idxs_te, sym2id
    )
    all_results[scale_name] = res

# Reference 150M numbers from existing results
all_results['ESM2_150M'] = {
    'scale': 'ESM2_150M', 'embed_dim': 640,
    'prism': {'All MF': 0.596, 'L2_Structural': 0.313, 'L4_CellState': 0.304,
              'L1_Generic_mid': 0.671, 'L1_Generic_high': 0.822},
    'v17f':  {'All MF': 0.717, 'L2_Structural': 0.622, 'L4_CellState': 0.576,
              'L1_Generic_mid': 0.787, 'L1_Generic_high': 0.859},
}

json.dump(all_results, open(f'{OUT_DIR}/scale_results.json', 'w'), indent=2)
print(f"\n[Saved] {OUT_DIR}/scale_results.json")

print("\n" + "=" * 65)
print("  Experiment A: COMPLETE")
print("=" * 65)
