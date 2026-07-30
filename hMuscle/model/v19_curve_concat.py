"""
v19_curve_concat.py
===================
PRISM v19: ESM-2 trajectory shape + L30 concatenated input.

Architecture change from v15d_bp_clean (PRISM baseline):
  v15d: input = L30 (640-dim)
  v19:  input = [L30 (640) ∥ curve_vec_norm (240)] = 880-dim

curve_vec_norm: per-layer z-score normalized, joint PCA K=8,
                30 layers × 8 axes = 240-dim trajectory shape vector.

PCA fit on TRAIN set → applied to TEST set (no leakage).

Everything else (focal loss, 5-seed ensemble, early stopping) identical
to v15d_bp_clean. Uses same 18 BP GO terms.

Motivation: LR diagnostic (exp_fluid_stage2f) showed consistent +3.7%
mean AUPRC gain from curve concat. v19 tests whether the deeper PRISM MLP
(vs LR) amplifies this gain. Critical test: within-gene Type-3 isoform
discrimination — does the trajectory shape help separate motif-dependent
isoforms that have the same Pfam domain count?
"""
from __future__ import annotations

import os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from collections import defaultdict
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data"
MODEL   = ROOT / "model"
FEAT_DIR= ROOT / "results_isoform" / "features"
ID_DIR  = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
OUT_DIR = ROOT.parent / "reports" / "v19_curve"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
K_PCA    = 8       # curve dim = 8 × 30 = 240
SEED     = 42
N_SEEDS  = 5

# Exact 18 GO terms from v15d_bp_clean.py (13 muscle + 5 neuro/cytoskeletal)
GO_18 = {
    "GO:0007204": "Ca2+ signaling",
    "GO:0045214": "Sarcomere organization",
    "GO:0006941": "Muscle contraction",
    "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",
    "GO:0007519": "Skeletal muscle dev",
    "GO:0042692": "Muscle cell diff",
    "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",
    "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",
    "GO:0030048": "Actin-based movement",
    "GO:0006096": "Glycolysis",
    "GO:0007268": "Synaptic transmission",
    "GO:0007018": "MT-based movement",
    "GO:0031175": "Neuron proj development",
    "GO:0030182": "Neuron diff",
    "GO:0000226": "MT cytoskeleton org",
}
N_GO = len(GO_18)

MID_TYPE_GOS = {"GO:0007204", "GO:0007018", "GO:0000226"}


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def build_traj(idx, prefix):
    """Load L1..L30 for given row indices. prefix: 'esm2_layer_%02d_t30_150M'
    or 'esm2_train_human_layer%02d_t30_150M'."""
    N = len(idx)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = DATA / (prefix % L)
        arr = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = arr[idx] if idx is not None else arr
        del arr
    return traj


def build_traj_full(prefix):
    """Load all rows (no index slicing)."""
    sample = np.load(DATA / (prefix % 1), mmap_mode="r")
    N = len(sample)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        arr = np.load(DATA / (prefix % L), mmap_mode="r")
        traj[:, L - 1, :] = arr.astype(np.float32)
        del arr
    return traj


def compute_curve_norm(traj: np.ndarray,
                       pca: PCA | None = None,
                       layer_stats: list | None = None):
    """
    per-layer z-score → joint PCA.
    layer_stats: list of (mu, sd) per layer from train set. If None, fit from traj.
    Returns (curve_vec, fitted_pca, layer_stats).
    Bug fix: test set must use TRAIN layer statistics (not its own) to stay
    in the same coordinate space as the train curve vectors.
    """
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


def build_model(input_dim: int):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def run_ensemble(X_tr, y_tr, X_te, input_dim):
    preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(X_tr))
        m = build_model(input_dim)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
        ]
        m.fit(X_tr[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        preds.append(m.predict(X_te, batch_size=1024, verbose=0).flatten())
    return np.mean(preds, axis=0)


def load_labels(go, sym_tr, sym_te):
    pos = set()
    with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return (np.array([1 if s in pos else 0 for s in sym_tr], dtype=np.float32),
            np.array([1 if s in pos else 0 for s in sym_te], dtype=np.float32))


def within_gene_spread(preds, symbols):
    """max-min predicted score per gene (≥2 isoforms)."""
    g2i = defaultdict(list)
    for i, g in enumerate(symbols):
        g2i[g].append(i)
    return {g: float(preds[idxs].max() - preds[idxs].min())
            for g, idxs in g2i.items() if len(idxs) >= 2}


def main():
    t0 = time.time()
    print("[v19] PRISM with curve_vec_norm + L30 concat (880-dim)")

    # ── IDs ─────────────────────────────────────────────────────────
    print("[1] Loading IDs...")
    tr_gene = load_ids(str(ID_DIR / "train_gene_list.npy"))
    te_iso  = load_ids(str(MODEL / "my_isoform_list_fixed.npy"))
    te_gene = load_ids(str(MODEL / "my_gene_list_fixed.npy"))

    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]

    sym_tr = [ENSG2SYM.get(g.split(".")[0], g.split(".")[0]) for g in tr_gene]
    sym_te = [ENSG2SYM.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    N_TR, N_TE = len(sym_tr), len(sym_te)

    # ── Type-3 flags for test set ───────────────────────────────────
    print("[2] Type-3 flags...")
    dm_path = FEAT_DIR / "domain_matrix_proper_test.npy"
    if dm_path.exists():
        dm = np.load(dm_path, mmap_mode="r")
        dc = np.array(dm.sum(1)).ravel()
        del dm
        te_gene_base = [g.split(".")[0] for g in te_gene]
        gene2dc = defaultdict(list)
        for i, g in enumerate(te_gene_base):
            gene2dc[g].append(dc[i])
        dc_range = {g: max(v) - min(v) for g, v in gene2dc.items()}
        type3_genes = {g for g, r in dc_range.items() if r == 0}
        is_type3_te = np.array([dc_range.get(g, 0) == 0 for g in te_gene_base])
        print(f"   Type-3 test isoforms: {is_type3_te.sum()}/{N_TE} ({is_type3_te.mean():.1%})")
    else:
        is_type3_te = np.zeros(N_TE, dtype=bool)
        type3_genes = set()

    # ── Build trajectory embeddings ─────────────────────────────────
    print("[3] Building TRAIN trajectory (31668 × 30 layers)...")
    traj_tr = build_traj_full("esm2_train_human_layer%02d_t30_150M.npy")
    print(f"   [{time.time()-t0:.1f}s] Computing train curve_vec_norm...")
    curve_tr, pca_fitted, layer_stats = compute_curve_norm(traj_tr)
    del traj_tr

    print(f"   [{time.time()-t0:.1f}s] Building TEST trajectory (36748 × 30 layers)...")
    traj_te = build_traj_full("esm2_layer_%02d_t30_150M.npy")
    print(f"   [{time.time()-t0:.1f}s] Computing test curve_vec_norm (train stats + train PCA)...")
    curve_te, _, _ = compute_curve_norm(traj_te, pca=pca_fitted, layer_stats=layer_stats)
    del traj_te

    # ── L30 ────────────────────────────────────────────────────────
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)

    # ── Concat ─────────────────────────────────────────────────────
    # Normalize curve before concat (important: scale to similar range as L30)
    sc = StandardScaler()
    curve_tr_sc = sc.fit_transform(curve_tr).astype(np.float32)
    curve_te_sc = sc.transform(curve_te).astype(np.float32)

    X_tr = np.concatenate([X_L30_tr, curve_tr_sc], axis=1)   # (31668, 880)
    X_te = np.concatenate([X_L30_te, curve_te_sc], axis=1)   # (36748, 880)
    INPUT_DIM = X_tr.shape[1]
    print(f"   Input dim: {INPUT_DIM}  [{time.time()-t0:.1f}s]")
    # Cache curve vectors for v19_mf_eval.py (avoid recomputation of trajectories)
    np.save(OUT_DIR / "curve_tr_sc.npy", curve_tr_sc)
    np.save(OUT_DIR / "curve_te_sc.npy", curve_te_sc)
    print(f"   [cache] Curve vectors saved to {OUT_DIR}")

    # ── Training per GO ─────────────────────────────────────────────
    print(f"[4] Training {N_GO} GO terms (5-seed ensemble each)...")
    score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
    auprc_list = []
    spread_results = []

    for gi, (go, go_name) in enumerate(GO_18.items()):  # noqa: E501
        t1 = time.time()
        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        preds = run_ensemble(X_tr, y_tr, X_te, INPUT_DIM)
        score_matrix[:, gi] = preds
        auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
        auprc_list.append(auprc)

        # Within-gene spread for Type3
        spread = within_gene_spread(preds, sym_te)
        t3_spread  = [v for g, v in spread.items() if g in type3_genes]
        t12_spread = [v for g, v in spread.items() if g not in type3_genes]
        spread_results.append({
            "go": go, "go_name": go_name,
            "is_mid": go in MID_TYPE_GOS,
            "auprc": round(auprc, 4),
            "T3_spread_mean":  round(float(np.mean(t3_spread)) if t3_spread else 0, 5),
            "T12_spread_mean": round(float(np.mean(t12_spread)) if t12_spread else 0, 5),
        })
        flag = "[MID]" if go in MID_TYPE_GOS else "     "
        print(f"  [{gi+1:2d}/{N_GO}] {flag} {go} {go_name[:20]:20s}  "
              f"AUPRC={auprc:.4f}  "
              f"T3_spread={np.mean(t3_spread) if t3_spread else 0:.4f}  "
              f"({time.time()-t1:.0f}s)")

    macro = float(np.mean(auprc_list))
    print(f"\n  === Macro AUPRC (18 GO) = {macro:.4f} ===")
    print(f"  v15d_bp_clean baseline: 0.7022 (muscle 18-GO)  Δ = {macro-0.7022:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    np.save(OUT_DIR / "v19_score_matrix.npy", score_matrix)
    import json
    result = {
        "macro_auprc": macro,
        "v15d_baseline": 0.7022,
        "delta_vs_v15d": round(macro - 0.7022, 4),
        "per_go": spread_results,
    }
    with open(OUT_DIR / "v19_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {OUT_DIR}/v19_results.json")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
