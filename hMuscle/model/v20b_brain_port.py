"""
v20b_brain_port.py
==================
Brain zero-shot version of v20b_window_sweep.py.

Trains PRISM v20b (Strategy B: GO Fisher-peak window curve + L30 concat)
on MUSCLE training data and evaluates zero-shot on BRAIN isoforms.

Design decisions:
  - Fisher peaks: reuse MUSCLE peaks (from reports/layer_probe/layer_probe_v15d_terms_results.json)
    — this is the true "v20b applied to brain zero-shot" test.
  - Per-layer PCA(8): fit on MUSCLE train, apply to BRAIN (same feature basis).
  - Same MLP + focal loss + N_SEEDS=5 ensemble as muscle v20b.
  - GO labels for BRAIN: use human_annotations_unified_bp.txt with brain gene symbols.
  - T3/T12 spread: NOT computed on brain (no per-brain domain matrix); brain AUPRC is
    the primary comparison against v15d brain baseline (0.5998 zero-shot).

Resource-aware:
  - Uses ONE GPU only, with 8GB memory_limit (out of 24GB).
  - TF intra/inter-op threads capped at 4.
  - BLAS threads capped at 8.
  - nice +10.
"""
from __future__ import annotations

import os
# BLAS threads BEFORE numpy import
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Pick GPU with less memory used (server context). Default to GPU 1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

try:
    os.nice(10)
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Cap TF CPU threads
try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
except Exception:
    pass

# Cap GPU memory to 8GB
try:
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        tf.config.set_logical_device_configuration(
            g, [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
except Exception as e:
    print(f"[warn] GPU memory-limit setup failed: {e}")


ROOT      = Path("/home/welcome1/sw1686/DIFFUSE")
HMU       = ROOT / "hMuscle"
DATA      = HMU / "data"
BRAIN_DIR = DATA / "brain_isoquant_esm2/full"
MODEL_DIR = HMU / "model"
FEAT_DIR  = HMU / "results_isoform" / "features"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
PROBE_DIR = ROOT / "reports" / "layer_probe"
OUT_DIR   = ROOT / "reports" / "v20b_brain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
K_PCA    = 8
N_SEEDS  = 5
W_LIST   = [5, 7]
SEED     = 42

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
MID_GOs = {"GO:0007204", "GO:0007018", "GO:0000226"}


def load_muscle_fisher_peaks():
    """0-indexed peak layer per GO from muscle layer probe LR-AUPRC."""
    all_lr = {}
    for fname in ("layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"):
        p = PROBE_DIR / fname
        if p.exists():
            d = json.load(open(p))
            all_lr.update(d.get("lr_auprc", {}))
    peaks = {}
    for go in GO_18:
        arr = np.array(all_lr[go])
        peaks[go] = int(np.argmax(arr))
    return peaks


def build_traj(prefix_fmt, N):
    """Load all 30 layers into (N, 30, 640) float32."""
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = prefix_fmt(L)
        arr = np.load(p, mmap_mode='r')
        traj[:, L - 1, :] = arr.astype(np.float32)
        del arr
    return traj


def compute_Z_fit(traj):
    """Per-layer z-score + joint PCA(K) fit on this trajectory.
    Returns Z, layer_stats, pca."""
    N = traj.shape[0]
    traj_norm = np.empty_like(traj)
    layer_stats = []
    for L in range(N_LAYERS):
        mu = traj[:, L, :].mean(0)
        sd = traj[:, L, :].std(0) + 1e-6
        layer_stats.append((mu, sd))
        traj_norm[:, L, :] = (traj[:, L, :] - mu) / sd
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=K_PCA, random_state=SEED)
    reduced = pca.fit_transform(flat)
    Z = reduced.reshape(N, N_LAYERS, K_PCA).astype(np.float32)
    return Z, layer_stats, pca


def compute_Z_transform(traj, layer_stats, pca):
    """Apply pre-fitted stats + PCA."""
    N = traj.shape[0]
    traj_norm = np.empty_like(traj)
    for L in range(N_LAYERS):
        mu, sd = layer_stats[L]
        traj_norm[:, L, :] = (traj[:, L, :] - mu) / sd
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    reduced = pca.transform(flat)
    return reduced.reshape(N, N_LAYERS, K_PCA).astype(np.float32)


def window_curve(Z, peak_0idx, w):
    lo = max(0, peak_0idx - w)
    hi = min(N_LAYERS - 1, peak_0idx + w) + 1
    win_idx = list(range(lo, hi))
    return Z[:, win_idx, :].reshape(Z.shape[0], -1), win_idx


def build_model(input_dim):
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
        tf.keras.backend.clear_session()
    return np.mean(preds, axis=0)


def load_labels_for_syms(go, syms):
    pos = set()
    with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return np.array([1 if s in pos else 0 for s in syms], dtype=np.float32)


def run_w(w, peaks, Z_tr, Z_br, X_L30_tr, X_L30_br,
          sym_tr, sym_br, t0):
    print(f"\n{'='*70}")
    print(f"  Strategy B (BRAIN zero-shot) — Window w={w}  (±{w} around muscle peak)")
    print(f"  Max dim: 640 + {2*w+1}x{K_PCA} = {640 + (2*w+1)*K_PCA}")
    print(f"{'='*70}")

    auprc_list = []
    per_go = []
    score_matrix = np.zeros((Z_br.shape[0], len(GO_18)), dtype=np.float32)

    for gi, (go, go_name) in enumerate(GO_18.items()):
        t1 = time.time()
        peak = peaks[go]
        flag = "[MID]" if go in MID_GOs else "     "

        c_tr_raw, win_idx = window_curve(Z_tr, peak, w)
        c_br_raw, _       = window_curve(Z_br, peak, w)

        sc = StandardScaler()
        c_tr = sc.fit_transform(c_tr_raw).astype(np.float32)
        c_br = sc.transform(c_br_raw).astype(np.float32)

        X_tr = np.concatenate([X_L30_tr, c_tr], axis=1)
        X_br = np.concatenate([X_L30_br, c_br], axis=1)
        input_dim = X_tr.shape[1]

        y_tr = load_labels_for_syms(go, sym_tr)
        y_br = load_labels_for_syms(go, sym_br)
        if y_tr.sum() < 5 or y_br.sum() < 5:
            print(f"  [{gi+1:2d}/18] {flag} {go} {go_name[:22]:22s}  SKIP "
                  f"(tr_pos={int(y_tr.sum())}, br_pos={int(y_br.sum())})")
            continue

        preds = run_ensemble(X_tr, y_tr, X_br, input_dim)
        score_matrix[:, gi] = preds
        auprc = float(average_precision_score(y_br, preds))
        auprc_list.append(auprc)

        per_go.append({
            "go": go, "go_name": go_name,
            "is_mid": go in MID_GOs,
            "peak_layer_1idx": peak + 1,
            "win_layers": [l + 1 for l in win_idx],
            "n_layers_used": len(win_idx),
            "input_dim": input_dim,
            "n_pos_tr": int(y_tr.sum()),
            "n_pos_br": int(y_br.sum()),
            "brain_auprc": round(auprc, 4),
        })

        print(f"  [{gi+1:2d}/18] {flag} {go} {go_name[:22]:22s}  "
              f"L{peak+1:02d}±{w}[{len(win_idx)}L]  dim={input_dim}  "
              f"br_AUPRC={auprc:.4f}  (br_pos={int(y_br.sum())})  "
              f"({time.time()-t1:.0f}s)  [total {time.time()-t0:.0f}s]", flush=True)

    macro = float(np.mean(auprc_list)) if auprc_list else 0.0
    print(f"\n  === w={w} BRAIN Summary ===")
    print(f"  Brain macro AUPRC = {macro:.4f}")
    print(f"  Reference: v15d brain zero-shot = 0.5998, v17f* brain = 0.637")

    result = {
        "tissue": "brain",
        "w": w, "n_layers_max": 2*w+1,
        "brain_macro_auprc": round(macro, 4),
        "references": {
            "v15d_brain": 0.5998,
            "v17f_star_brain": 0.637,
            "v17f_brain": 0.622,
        },
        "n_seeds": N_SEEDS,
        "per_go": per_go,
    }
    out_file = OUT_DIR / f"w{w}_brain_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    np.save(OUT_DIR / f"w{w}_brain_score_matrix.npy", score_matrix)
    print(f"  [saved] {out_file}")
    return result


def main():
    t0 = time.time()
    print("[v20b-brain] Strategy B window sweep — brain zero-shot")
    print(f"  Config: N_SEEDS={N_SEEDS}, W_LIST={W_LIST}, GPU=CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")

    # ── 1. Muscle Fisher peaks (0-indexed) ────────────────────────────
    print("\n[1] Muscle Fisher peaks (LR-AUPRC)...")
    peaks = load_muscle_fisher_peaks()
    for go, name in list(GO_18.items())[:4]:
        print(f"   {go} {name[:20]:20s}  peak=L{peaks[go]+1:02d}")

    # ── 2. IDs ───────────────────────────────────────────────────────
    print("\n[2] IDs...")
    tr_gene = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]

    def clean(raw):
        s = str(raw)
        for c in ["b'", "'", '"', ' ']:
            s = s.replace(c, '')
        return s

    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in tr_gene]
    sym_br = [clean(s) for s in
              np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)]
    print(f"   sym_tr={len(sym_tr)}  sym_br={len(sym_br)}")

    # ── 3. Trajectories + Z (PCA) ─────────────────────────────────────
    print(f"\n[3] Muscle train trajectory (30 layers)... [{time.time()-t0:.0f}s]")
    traj_tr = build_traj(
        lambda L: DATA / f"esm2_train_human_layer{L:02d}_t30_150M.npy",
        len(sym_tr))
    print(f"   traj_tr {traj_tr.shape}, dtype={traj_tr.dtype}, "
          f"mem={traj_tr.nbytes/1e9:.2f}GB [{time.time()-t0:.0f}s]")

    print(f"[4] PCA(8) fit on muscle train...")
    Z_tr, layer_stats, pca = compute_Z_fit(traj_tr)
    del traj_tr
    print(f"   Z_tr {Z_tr.shape} [{time.time()-t0:.0f}s]")

    print(f"[5] Brain trajectory (30 layers)...")
    traj_br = build_traj(
        lambda L: BRAIN_DIR / f"brain_full_esm2_layer{L:02d}_t30_150M.npy",
        len(sym_br))
    print(f"   traj_br {traj_br.shape}, mem={traj_br.nbytes/1e9:.2f}GB "
          f"[{time.time()-t0:.0f}s]")

    print(f"[6] Apply muscle PCA/stats to brain...")
    Z_br = compute_Z_transform(traj_br, layer_stats, pca)
    del traj_br
    print(f"   Z_br {Z_br.shape} [{time.time()-t0:.0f}s]")

    # ── 7. L30 pooled embeddings ──────────────────────────────────────
    print(f"\n[7] L30 pooled embeddings...")
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_br = np.load(BRAIN_DIR / "brain_full_esm2_t30_150M.npy").astype(np.float32)
    print(f"   X_L30_tr={X_L30_tr.shape}  X_L30_br={X_L30_br.shape}")
    assert X_L30_tr.shape[0] == Z_tr.shape[0], "muscle train N mismatch"
    assert X_L30_br.shape[0] == Z_br.shape[0], "brain N mismatch"

    # ── 8. Run w=5 and w=7 ────────────────────────────────────────────
    all_results = {}
    for w in W_LIST:
        r = run_w(w, peaks, Z_tr, Z_br, X_L30_tr, X_L30_br,
                  sym_tr, sym_br, t0)
        all_results[f"w{w}"] = r

    # ── 9. Final Brain comparison ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BRAIN Zero-shot Comparison")
    print(f"{'='*70}")
    print(f"  {'Model':22s}  {'Brain AUPRC':>12}")
    print(f"  {'v15d (base) brain':22s}  {0.5998:>12.4f}")
    print(f"  {'v17f  brain':22s}  {0.622:>12.4f}")
    print(f"  {'v17f* brain BEST':22s}  {0.637:>12.4f}")
    for w in W_LIST:
        r = all_results[f"w{w}"]
        print(f"  {'v20b w='+str(w)+' brain':22s}  {r['brain_macro_auprc']:>12.4f}")

    with open(OUT_DIR / "all_brain_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] total elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
