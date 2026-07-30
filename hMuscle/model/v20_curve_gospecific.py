"""
v20_curve_gospecific.py
=======================
PRISM v20: GO-specific Fisher-weighted curve + L30.

핵심 아이디어 (v19 positive-dilution 문제 해결):
  v19: [L30(640) ∥ curve_all(240)] = 880-dim  → 모든 layer curve 동시 입력
  v20: [L30(640) ∥ c_g(8)]        = 648-dim  → GO별 Fisher peak layer만 가중합

수식:
  Z_L^(i) = PCA(z-score(X_L^(i))) ∈ ℝ^8     (per layer, 8 PCA dims)
  α_{g,L} = softmax(F_{g,L} / T)              (Fisher AUPRC를 GO별 가중치로)
  c_g^(i) = Σ_L α_{g,L} · Z_L^(i) ∈ ℝ^8    (GO-specific 8-dim curve)
  input_g  = [X_30^(i) ∥ c_g^(i)] ∈ ℝ^648   (minimal overhead over v15d)

효과:
  - 불필요한 layer curve 제거 → positive score dilution 억제
  - GO-peak layer 집중 → Type-3 분리력 유지
  - 입력 640→648 (vs v19의 880): noise 최소화
"""
from __future__ import annotations

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from collections import defaultdict
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import warnings; warnings.filterwarnings('ignore')

ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data"
MODEL    = ROOT / "model"
FEAT_DIR = ROOT / "results_isoform" / "features"
ID_DIR   = DATA / "raw_data/data/id_lists"
ANNOT_DIR= DATA / "raw_data/data/annotations"
PROBE_DIR= ROOT.parent / "reports" / "layer_probe"
OUT_DIR  = ROOT.parent / "reports" / "v20_curve_gospecific"
CACHE_DIR= ROOT.parent / "reports" / "v20_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
K_PCA    = 8
SEED     = 42
N_SEEDS  = 5
TEMP     = 1.0   # Fisher softmax temperature; lower = sharper selection

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


# ── Fisher weights per GO ──────────────────────────────────────────

def load_fisher_weights(temp: float = 1.0) -> dict[str, np.ndarray]:
    """
    Per-layer LR AUPRC (30-dim) → softmax weights for each GO.
    Sources: 3 layer_probe result files covering all 18 GOs.
    """
    all_lr: dict[str, list] = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        d = json.load(open(PROBE_DIR / fname))
        all_lr.update(d["lr_auprc"])

    weights = {}
    for go in GO_18:
        raw = np.array(all_lr[go], dtype=np.float32)   # (30,)
        # softmax with temperature
        shifted = (raw - raw.max()) / temp
        w = np.exp(shifted) / np.exp(shifted).sum()
        weights[go] = w   # (30,)
        peak_L = int(np.argmax(raw)) + 1
    return weights


# ── Trajectory → per-layer PCA coords ─────────────────────────────

def build_traj_full(prefix: str) -> np.ndarray:
    sample = np.load(DATA / (prefix % 1), mmap_mode="r")
    N = len(sample)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        arr = np.load(DATA / (prefix % L), mmap_mode="r")
        traj[:, L - 1, :] = arr.astype(np.float32)
        del arr
    return traj


def compute_Z(traj: np.ndarray,
              pca: PCA | None = None,
              layer_stats: list | None = None):
    """
    per-layer z-score → joint PCA → Z[i, L, k] (N, 30, K_PCA).
    Returns (Z, fitted_pca, layer_stats).
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

    Z = reduced.reshape(N, N_LAYERS, K_PCA).astype(np.float32)
    stats_out = fitted_stats if layer_stats is None else layer_stats
    return Z, pca, stats_out


def fisher_weighted_curve(Z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Z: (N, 30, 8)   alpha: (30,)
    Returns c_g: (N, 8)  — GO-specific weighted sum over layers.
    """
    # einsum: N×L×K, L → N×K
    return np.einsum('nlk,l->nk', Z, alpha).astype(np.float32)


# ── Model ──────────────────────────────────────────────────────────

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
    g2i = defaultdict(list)
    for i, g in enumerate(symbols): g2i[g].append(i)
    return {g: float(preds[idxs].max() - preds[idxs].min())
            for g, idxs in g2i.items() if len(idxs) >= 2}


# ── Main ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("[v20] GO-specific Fisher-weighted curve (648-dim)")
    print(f"      Temperature T={TEMP} (lower=sharper layer selection)")

    # ── 1. Fisher weights ────────────────────────────────────────────
    print("[1] Loading Fisher weights per GO...")
    fisher_weights = load_fisher_weights(TEMP)
    for go, name in list(GO_18.items())[:3]:
        w = fisher_weights[go]
        peak_L = int(np.argmax(w)) + 1
        top3 = sorted(range(30), key=lambda l: w[l], reverse=True)[:3]
        print(f"   {go} {name[:20]:20s}: peak=L{peak_L:02d}  "
              f"top3_layers={[l+1 for l in top3]}  "
              f"w_peak={w.max():.3f}")

    # ── 2. IDs ───────────────────────────────────────────────────────
    print("[2] Loading IDs...")
    tr_gene = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    te_gene = np.load(MODEL / "my_gene_list_fixed.npy", allow_pickle=True)

    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

    def clean(raw):
        s = str(raw)
        for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
        return s

    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in tr_gene]
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
    N_TR, N_TE = len(sym_tr), len(sym_te)
    print(f"   Train: {N_TR}  Test: {N_TE}")

    # ── 3. Type-3 flags ──────────────────────────────────────────────
    print("[3] Type-3 flags...")
    dm_path = FEAT_DIR / "domain_matrix_proper_test.npy"
    te_gene_base = [clean(g).split('.')[0] for g in te_gene]
    if dm_path.exists():
        dm = np.load(dm_path, mmap_mode="r")
        dc = np.array(dm.sum(1)).ravel(); del dm
        gene2dc = defaultdict(list)
        for i, g in enumerate(te_gene_base): gene2dc[g].append(dc[i])
        dc_range = {g: max(v) - min(v) for g, v in gene2dc.items()}
        type3_genes = {g for g, r in dc_range.items() if r == 0}
        is_type3_te = np.array([dc_range.get(g, 0) == 0 for g in te_gene_base])
        print(f"   Type-3: {is_type3_te.sum()}/{N_TE} ({is_type3_te.mean():.1%})")
    else:
        type3_genes = set()
        is_type3_te = np.zeros(N_TE, dtype=bool)

    # ── 4. Z matrix (per-layer PCA) ──────────────────────────────────
    cache_Z_tr = CACHE_DIR / "Z_tr.npy"
    cache_Z_te = CACHE_DIR / "Z_te.npy"

    if cache_Z_tr.exists() and cache_Z_te.exists():
        print(f"[4] Loading cached Z matrices...")
        Z_tr = np.load(cache_Z_tr)   # (N_TR, 30, 8)
        Z_te = np.load(cache_Z_te)   # (N_TE, 30, 8)
    else:
        print(f"[4] Building TRAIN trajectory → Z_tr...")
        traj_tr = build_traj_full("esm2_train_human_layer%02d_t30_150M.npy")
        Z_tr, pca_fitted, layer_stats = compute_Z(traj_tr)
        del traj_tr
        print(f"   [{time.time()-t0:.1f}s] Building TEST trajectory → Z_te...")
        traj_te = build_traj_full("esm2_layer_%02d_t30_150M.npy")
        Z_te, _, _ = compute_Z(traj_te, pca=pca_fitted, layer_stats=layer_stats)
        del traj_te
        np.save(cache_Z_tr, Z_tr)
        np.save(cache_Z_te, Z_te)
        print(f"   [{time.time()-t0:.1f}s] Z cached: {cache_Z_tr}")

    print(f"   Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")

    # ── 5. L30 ───────────────────────────────────────────────────────
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)

    # ── 6. Per-GO training ───────────────────────────────────────────
    print(f"[5] Training {N_GO} GO terms (Fisher-weighted curve, 5-seed each)...")
    score_matrix = np.zeros((N_TE, N_GO), dtype=np.float32)
    auprc_list   = []
    spread_results = []

    for gi, (go, go_name) in enumerate(GO_18.items()):
        t1 = time.time()

        # GO-specific curve (8-dim Fisher weighted)
        alpha_g = fisher_weights[go]               # (30,)
        c_tr = fisher_weighted_curve(Z_tr, alpha_g)  # (N_TR, 8)
        c_te = fisher_weighted_curve(Z_te, alpha_g)  # (N_TE, 8)

        sc = StandardScaler()
        c_tr_sc = sc.fit_transform(c_tr).astype(np.float32)
        c_te_sc = sc.transform(c_te).astype(np.float32)

        X_tr = np.concatenate([X_L30_tr, c_tr_sc], axis=1)  # (N_TR, 648)
        X_te = np.concatenate([X_L30_te, c_te_sc], axis=1)  # (N_TE, 648)
        INPUT_DIM = X_tr.shape[1]

        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        preds = run_ensemble(X_tr, y_tr, X_te, INPUT_DIM)
        score_matrix[:, gi] = preds
        auprc = average_precision_score(y_te, preds) if y_te.sum() > 0 else 0.0
        auprc_list.append(auprc)

        spread = within_gene_spread(preds, sym_te)
        t3_spread  = [v for g, v in spread.items() if g in type3_genes]
        t12_spread = [v for g, v in spread.items() if g not in type3_genes]

        peak_L = int(np.argmax(alpha_g)) + 1
        spread_results.append({
            "go": go, "go_name": go_name,
            "is_mid": go in MID_TYPE_GOS,
            "peak_layer": peak_L,
            "auprc": round(auprc, 4),
            "T3_spread_mean":  round(float(np.mean(t3_spread))  if t3_spread  else 0, 5),
            "T12_spread_mean": round(float(np.mean(t12_spread)) if t12_spread else 0, 5),
            "T3_T12_ratio":    round(float(np.mean(t3_spread)/np.mean(t12_spread))
                                     if t3_spread and t12_spread else 0, 4),
        })
        flag = "[MID]" if go in MID_TYPE_GOS else "     "
        print(f"  [{gi+1:2d}/{N_GO}] {flag} {go} {go_name[:20]:20s}  "
              f"L{peak_L:02d}  AUPRC={auprc:.4f}  "
              f"T3_spread={np.mean(t3_spread) if t3_spread else 0:.4f}  "
              f"({time.time()-t1:.0f}s)")

    macro = float(np.mean(auprc_list))
    print(f"\n  === Macro AUPRC (18 GO) = {macro:.4f} ===")
    print(f"  v15d baseline:  0.7022  Δ = {macro-0.7022:+.4f}")
    print(f"  v19 (curve240): 0.6721  Δ = {macro-0.6721:+.4f}")

    # ── T3/T12 ratio summary ─────────────────────────────────────────
    all_ratios = [r["T3_T12_ratio"] for r in spread_results if r["T3_T12_ratio"] > 0]
    mid_ratios = [r["T3_T12_ratio"] for r in spread_results if r["is_mid"] and r["T3_T12_ratio"] > 0]
    print(f"\n  T3/T12 ratio: ALL={np.mean(all_ratios):.4f}  MID={np.mean(mid_ratios):.4f}")
    print(f"  (v15d: ALL=0.3377  MID=0.3432)")
    print(f"  (v19:  ALL=0.3843  MID=0.4493)")

    # ── Save ─────────────────────────────────────────────────────────
    np.save(OUT_DIR / "v20_score_matrix.npy", score_matrix)
    result = {
        "macro_auprc": macro,
        "v15d_baseline": 0.7022,
        "v19_curve240": 0.6721,
        "delta_vs_v15d": round(macro - 0.7022, 4),
        "delta_vs_v19":  round(macro - 0.6721, 4),
        "temperature": TEMP,
        "input_dim": 648,
        "t3_t12_ratio_all": round(float(np.mean(all_ratios)), 4),
        "t3_t12_ratio_mid": round(float(np.mean(mid_ratios)), 4),
        "per_go": spread_results,
    }
    with open(OUT_DIR / "v20_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {OUT_DIR}/v20_results.json")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
