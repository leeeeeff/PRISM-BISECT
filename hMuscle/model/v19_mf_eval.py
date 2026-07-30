"""
v19_mf_eval.py
==============
v19 curve_vec_norm evaluated on 82 MF GO terms — direct comparison to v17f*.

Input:  [L30(640) ∥ curve_vec_norm_scaled(240)] = 880-dim  (same as v19)
Labels: 82 MF GO terms via gene2go.gz (same as v17f_abl_no_tpsi.py)
Model:  Dense(256,ReLU)→BN→Dropout(0.2)→Dense(128,ReLU)→Dense(82,sigmoid)
Seeds:  5-seed ensemble (same as v17f*)

Baselines:
  v17f* (no T_ψ): All MF = 0.7325  L2_Structural = 0.6333
  Gene-mean:      ~0.803 (oracle ceiling)
"""
from __future__ import annotations

import os, json, gzip, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from collections import defaultdict
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.losses import BinaryFocalCrossentropy
import warnings; warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data"
MODEL    = ROOT / "model"
ID_DIR   = DATA / "raw_data/data/id_lists"
ANNOT_DIR= DATA / "raw_data/data/annotations"
MF_DIR   = ROOT.parent / "reports" / "v_expanded_gomf"
OUT_DIR  = ROOT.parent / "reports" / "v19_mf"
CACHE_DIR= ROOT.parent / "reports" / "v19_curve"   # reuse curve cache if available
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
K_PCA    = 8
SEED     = 42
N_SEEDS  = 5
SEEDS    = [42, 7, 13, 21, 99]
BATCH    = 512
EPOCHS   = 60

V17F_ALL = 0.7325   # v17f* confirmed: reports/v17f_abl_no_tpsi/results.json
V17F_L2  = 0.6333


# ── helpers ────────────────────────────────────────────────────────

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def build_traj_full(prefix):
    sample = np.load(DATA / (prefix % 1), mmap_mode="r")
    N = len(sample)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        arr = np.load(DATA / (prefix % L), mmap_mode="r")
        traj[:, L - 1, :] = arr.astype(np.float32)
        del arr
    return traj


def compute_curve_norm(traj, pca=None, layer_stats=None):
    N = traj.shape[0]
    traj_norm = np.empty_like(traj)
    fitted_stats = []
    for L in range(N_LAYERS):
        if layer_stats is None:
            mu = traj[:, L, :].mean(0)
            sd = traj[:, L, :].std(0) + 1e-6
            fitted_stats.append((mu, sd))
        else:
            mu, sd = layer_stats[L]
        traj_norm[:, L, :] = (traj[:, L, :] - mu) / sd
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    if pca is None:
        pca = PCA(n_components=K_PCA, random_state=SEED)
        reduced = pca.fit_transform(flat)
    else:
        reduced = pca.transform(flat)
    stats_out = fitted_stats if layer_stats is None else layer_stats
    return reduced.reshape(N, N_LAYERS * K_PCA).astype(np.float32), pca, stats_out


def build_model(input_dim, n_go):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_go, activation="sigmoid")(x)
    return models.Model(inp, out)


def macro_auprc(Y, preds, idxs):
    aps = [average_precision_score(Y[:, i], preds[:, i])
           for i in idxs if Y[:, i].sum() >= 2]
    return float(np.mean(aps)) if aps else float('nan')


# ── main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("[v19_mf] curve_vec_norm + L30 (880-dim) on 82 MF GO terms")

    # ── 1. IDs ──────────────────────────────────────────────────────
    print("[1] Loading IDs...")
    tr_genes_raw = load_ids(str(ID_DIR / "train_gene_list.npy"))
    te_genes_raw = load_ids(str(MODEL / "my_gene_list_fixed.npy"))

    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]

    tr_genes = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in tr_genes_raw]
    te_syms  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
    print(f"   Train: {len(tr_genes)}  Test: {len(te_syms)}")

    # ── 2. 82 MF labels ─────────────────────────────────────────────
    print("[2] Loading GO labels (82 MF via gene2go.gz)...")
    sym2id = {}
    with gzip.open(ANNOT_DIR / "Homo_sapiens.gene_info.gz", 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 2:
                sym2id[p[2]] = p[1]
                if len(p) > 4 and p[4] != '-':
                    for syn in p[4].split('|'):
                        if syn not in sym2id: sym2id[syn] = p[1]

    tr_ids   = [sym2id.get(g, g) for g in tr_genes]
    tr_id_set= set(tr_ids)
    go_genes_tr  = defaultdict(set)
    go_genes_all = defaultdict(set)
    with gzip.open(ANNOT_DIR / "gene2go.gz", 'rt') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] != '9606' or p[7] != 'Function': continue
            go_genes_all[p[2]].add(p[1])
            if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

    mf_terms = []
    with open(MF_DIR / "mf_domain_vs_prism.tsv") as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 6: mf_terms.append(p[0])
    n_go = len(mf_terms)
    print(f"   {n_go} MF terms loaded")

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
                         for s in te_syms], dtype=np.float32)

    print("   Building label matrices...")
    Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
    Y_te = np.stack([build_Y_te(go) for go in mf_terms], axis=1)
    valid_mask = Y_te.sum(0) >= 2

    L2_TERMS = set()
    with open(MF_DIR / "h2_layer_classification.tsv") as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 12 and p[11] == 'L2_Structural': L2_TERMS.add(p[0])

    valid_idx = [i for i in range(n_go) if valid_mask[i]]
    l2_valid  = [i for i in range(n_go) if mf_terms[i] in L2_TERMS and valid_mask[i]]
    print(f"   Valid: {len(valid_idx)}/{n_go} | L2_Structural: {len(l2_valid)}")

    # ── 3. curve_vec_norm ───────────────────────────────────────────
    # Check for cached vectors from v19 run
    cache_tr = CACHE_DIR / "curve_tr_sc.npy"
    cache_te = CACHE_DIR / "curve_te_sc.npy"

    if cache_tr.exists() and cache_te.exists():
        print(f"[3] Loading cached curve vectors from {CACHE_DIR}")
        curve_tr_sc = np.load(cache_tr).astype(np.float32)
        curve_te_sc = np.load(cache_te).astype(np.float32)
    else:
        print("[3] Computing curve_vec_norm from trajectories (no cache found)...")
        print(f"   [{time.time()-t0:.1f}s] Building TRAIN trajectory...")
        traj_tr = build_traj_full("esm2_train_human_layer%02d_t30_150M.npy")
        curve_tr, pca_fitted, layer_stats = compute_curve_norm(traj_tr)
        del traj_tr
        print(f"   [{time.time()-t0:.1f}s] Building TEST trajectory...")
        traj_te = build_traj_full("esm2_layer_%02d_t30_150M.npy")
        curve_te, _, _ = compute_curve_norm(traj_te, pca=pca_fitted, layer_stats=layer_stats)
        del traj_te
        sc = StandardScaler()
        curve_tr_sc = sc.fit_transform(curve_tr).astype(np.float32)
        curve_te_sc = sc.transform(curve_te).astype(np.float32)
        # Save cache
        np.save(cache_tr, curve_tr_sc)
        np.save(cache_te, curve_te_sc)
        print(f"   [{time.time()-t0:.1f}s] Curve vectors cached to {CACHE_DIR}")

    print(f"   curve shape: tr={curve_tr_sc.shape}  te={curve_te_sc.shape}")

    # ── 4. Concat with L30 ──────────────────────────────────────────
    print("[4] Loading L30 embeddings and concatenating...")
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)
    X_tr = np.concatenate([X_L30_tr, curve_tr_sc], axis=1)   # (N_tr, 880)
    X_te = np.concatenate([X_L30_te, curve_te_sc], axis=1)   # (N_te, 880)
    INPUT_DIM = X_tr.shape[1]
    print(f"   Input dim: {INPUT_DIM}  [{time.time()-t0:.1f}s]")

    # ── 5. GPU setup ────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"   GPU: {gpus[0].name}")

    focal_fn = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

    # ── 6. 5-seed ensemble ──────────────────────────────────────────
    print(f"[5] 5-seed ensemble (multi-output, {EPOCHS} epochs)...")
    all_preds = []
    for seed in SEEDS:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        perm  = np.random.permutation(len(X_tr))
        n_val = int(len(X_tr) * 0.1)
        val_idx_s = perm[:n_val]
        tr_idx_s  = perm[n_val:]

        m = build_model(INPUT_DIM, n_go)
        m.compile(optimizer=optimizers.Adam(1e-3), loss=focal_fn)
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
        ]
        m.fit(X_tr[tr_idx_s], Y_tr[tr_idx_s],
              validation_data=(X_tr[val_idx_s], Y_tr[val_idx_s]),
              epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=0)
        preds_i = m.predict(X_te, batch_size=1024, verbose=0)
        all_preds.append(preds_i)
        aps = [average_precision_score(Y_te[:, i], preds_i[:, i])
               for i in valid_idx if Y_te[:, i].sum() >= 2]
        print(f"  seed={seed}  AUPRC={np.mean(aps):.4f}  [{time.time()-t0:.0f}s]")

    preds = np.mean(all_preds, axis=0)

    # ── 7. Evaluate ─────────────────────────────────────────────────
    auprc_all = macro_auprc(Y_te, preds, valid_idx)
    auprc_l2  = macro_auprc(Y_te, preds, l2_valid)

    print(f"\n{'='*65}")
    print(f"  v19_mf results (curve_vec 880-dim, 82 MF GO terms):")
    print(f"  All MF AUPRC:    {auprc_all:.4f}  (v17f*={V17F_ALL}  Δ={auprc_all-V17F_ALL:+.4f})")
    print(f"  L2_Structural:   {auprc_l2:.4f}  (v17f*={V17F_L2}  Δ={auprc_l2-V17F_L2:+.4f})")
    print(f"{'='*65}")

    # ── 8. Save ─────────────────────────────────────────────────────
    np.save(OUT_DIR / "v19_mf_score_matrix.npy", preds)
    result = {
        "architecture": {
            "method": "v19_curve_mf",
            "input": "concat[L30(640), curve_vec_norm_scaled(240)] = 880-dim",
            "mlp": "880→256(ReLU)→BN→Drop(0.2)→128(ReLU)→82(sigmoid)",
            "loss": "BinaryFocalCrossentropy(gamma=2)",
            "seeds": SEEDS,
            "n_go": n_go,
        },
        "auprc_all_mf": auprc_all,
        "auprc_l2_structural": auprc_l2,
        "v17f_star_ref": {"all_mf": V17F_ALL, "l2": V17F_L2, "source": "reports/v17f_abl_no_tpsi/results.json"},
        "delta_vs_v17f_star": {
            "all_mf": round(auprc_all - V17F_ALL, 4),
            "l2":     round(auprc_l2  - V17F_L2,  4),
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "valid_go_count": len(valid_idx),
        "l2_go_count": len(l2_valid),
    }
    with open(OUT_DIR / "v19_mf_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[write] {OUT_DIR}/v19_mf_results.json")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
