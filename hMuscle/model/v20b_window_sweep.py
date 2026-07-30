"""
v20b_window_sweep.py
====================
PRISM v20b: GO-specific WINDOW curve selection + L30.
Strategy B (sweep 결과): contiguous layers around GO Fisher peak > top-K.

핵심 차이 (v20 대비):
  v20:  c_g = einsum('nlk,l->nk', Z, alpha)      → 8-dim (Fisher weighted avg)
  v20b: c_g = Z[:, win_idx, :].reshape(N, -1)    → n_layers*8-dim (window 선택)

W_LIST = [5, 7] 두 실험 순차 실행.
  w=5: peak ± 5  → 최대 11 layers × 8-dim = 88-dim  → 640+88  = 728-dim
  w=7: peak ± 7  → 최대 15 layers × 8-dim = 120-dim → 640+120 = 760-dim
  (경계 GO는 window가 clamp되어 dim 감소)

비교 기준:
  v15d: AUPRC=0.7022  T3/T12_ALL=0.3377  MID=0.3432
  v20:  AUPRC=0.7012  T3/T12_ALL=0.3480  MID=0.3140
  v19:  AUPRC=0.6721  T3/T12_ALL=0.3843  MID=0.4493
"""
from __future__ import annotations

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import warnings; warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data"
MODEL_DIR = ROOT / "model"
FEAT_DIR  = ROOT / "results_isoform" / "features"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
PROBE_DIR = ROOT.parent / "reports" / "layer_probe"
CACHE_DIR = ROOT.parent / "reports" / "v20_cache"
OUT_DIR   = ROOT.parent / "reports" / "v20b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
W_LIST  = [5, 7]   # half-window sizes to sweep

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


# ── Fisher peak per GO ─────────────────────────────────────────────

def load_fisher_peaks() -> dict[str, int]:
    """Returns 0-indexed peak layer for each GO."""
    all_lr: dict[str, list] = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        path = PROBE_DIR / fname
        if path.exists():
            d = json.load(open(path))
            all_lr.update(d["lr_auprc"])
    peaks = {}
    for go in GO_18:
        arr = np.array(all_lr[go])
        peaks[go] = int(np.argmax(arr))   # 0-indexed
    return peaks


# ── Window curve extraction ────────────────────────────────────────

def window_curve(Z: np.ndarray, peak_0idx: int, w: int):
    """
    Z: (N, 30, 8)
    peak_0idx: 0-indexed Fisher peak layer
    w: half-window

    Returns: (N, n_layers * 8) — n_layers = actual window size (≤ 2w+1)
    Also returns win_idx for logging.
    """
    lo = max(0, peak_0idx - w)
    hi = min(29, peak_0idx + w) + 1   # exclusive
    win_idx = list(range(lo, hi))
    curve = Z[:, win_idx, :].reshape(Z.shape[0], -1)
    return curve, win_idx


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


# ── Labels & spread ────────────────────────────────────────────────

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

def run_w(w, peaks, Z_tr, Z_te, X_L30_tr, X_L30_te,
          sym_tr, sym_te, type3_genes, t_global):
    """Train + eval for a single window size w."""
    print(f"\n{'='*70}")
    print(f"  Strategy B — Window w={w}  (±{w} layers around GO Fisher peak)")
    print(f"  Max dim: 640 + {2*w+1}×8 = {640 + (2*w+1)*8}")
    print(f"{'='*70}")

    auprc_list, spread_results = [], []
    score_matrix = np.zeros((Z_te.shape[0], len(GO_18)), dtype=np.float32)

    for gi, (go, go_name) in enumerate(GO_18.items()):
        t1 = time.time()
        peak = peaks[go]      # 0-indexed
        flag = "[MID]" if go in MID_GOs else "     "

        # Window curve
        c_tr_raw, win_idx = window_curve(Z_tr, peak, w)
        c_te_raw, _       = window_curve(Z_te, peak, w)

        sc = StandardScaler()
        c_tr = sc.fit_transform(c_tr_raw).astype(np.float32)
        c_te = sc.transform(c_te_raw).astype(np.float32)

        X_tr = np.concatenate([X_L30_tr, c_tr], axis=1)
        X_te = np.concatenate([X_L30_te, c_te], axis=1)
        input_dim = X_tr.shape[1]

        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        if y_te.sum() == 0:
            print(f"  [{gi+1:2d}/{len(GO_18)}] {flag} {go} {go_name[:20]:20s}  SKIP (no pos)")
            continue

        preds = run_ensemble(X_tr, y_tr, X_te, input_dim)
        score_matrix[:, gi] = preds
        auprc = float(average_precision_score(y_te, preds))
        auprc_list.append(auprc)

        spread = within_gene_spread(preds, sym_te)
        t3_sp  = [v for g, v in spread.items() if g in type3_genes]
        t12_sp = [v for g, v in spread.items() if g not in type3_genes]
        t3_t12 = float(np.mean(t3_sp)/np.mean(t12_sp)) if t3_sp and t12_sp else 0.0

        spread_results.append({
            "go": go, "go_name": go_name,
            "is_mid": go in MID_GOs,
            "peak_layer_1idx": peak + 1,
            "win_layers": [l+1 for l in win_idx],
            "n_layers_used": len(win_idx),
            "input_dim": input_dim,
            "auprc": round(auprc, 4),
            "T3_spread_mean":  round(float(np.mean(t3_sp))  if t3_sp  else 0, 5),
            "T12_spread_mean": round(float(np.mean(t12_sp)) if t12_sp else 0, 5),
            "T3_T12_ratio":    round(t3_t12, 4),
        })

        print(f"  [{gi+1:2d}/{len(GO_18)}] {flag} {go} {go_name[:20]:20s}  "
              f"L{peak+1:02d}±{w}[{len(win_idx)}L]  "
              f"dim={input_dim}  "
              f"AUPRC={auprc:.4f}  T3/T12={t3_t12:.4f}  ({time.time()-t1:.0f}s)")

    macro = float(np.mean(auprc_list))
    all_ratios = [r["T3_T12_ratio"] for r in spread_results if r["T3_T12_ratio"] > 0]
    mid_ratios = [r["T3_T12_ratio"] for r in spread_results if r["is_mid"] and r["T3_T12_ratio"] > 0]
    t3_all = float(np.mean(all_ratios)) if all_ratios else 0.0
    t3_mid = float(np.mean(mid_ratios)) if mid_ratios else 0.0

    print(f"\n  === w={w} Summary ===")
    print(f"  Macro AUPRC = {macro:.4f}  (v15d=0.7022 Δ={macro-0.7022:+.4f} | v20=0.7012 Δ={macro-0.7012:+.4f})")
    print(f"  T3/T12 ALL  = {t3_all:.4f}  (v15d=0.3377 Δ={t3_all-0.3377:+.4f} | v20=0.3480 Δ={t3_all-0.3480:+.4f})")
    print(f"  T3/T12 MID  = {t3_mid:.4f}  (v15d=0.3432 Δ={t3_mid-0.3432:+.4f} | v20=0.3140 Δ={t3_mid-0.3140:+.4f})")

    result = {
        "w": w, "n_layers_max": 2*w+1,
        "macro_auprc": round(macro, 4),
        "delta_vs_v15d": round(macro - 0.7022, 4),
        "delta_vs_v20":  round(macro - 0.7012, 4),
        "t3_t12_ratio_all": round(t3_all, 4),
        "t3_t12_ratio_mid": round(t3_mid, 4),
        "delta_t3_vs_v15d": round(t3_all - 0.3377, 4),
        "delta_t3_vs_v20":  round(t3_all - 0.3480, 4),
        "references": {
            "v15d": {"macro_auprc": 0.7022, "t3_t12_all": 0.3377, "t3_t12_mid": 0.3432},
            "v20_fisher_avg": {"macro_auprc": 0.7012, "t3_t12_all": 0.3480, "t3_t12_mid": 0.3140},
            "v19_curve240": {"macro_auprc": 0.6721, "t3_t12_all": 0.3843, "t3_t12_mid": 0.4493},
        },
        "per_go": spread_results,
    }

    out_file = OUT_DIR / f"w{w}_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    np.save(OUT_DIR / f"w{w}_score_matrix.npy", score_matrix)
    print(f"  [saved] {out_file}")
    return result


def main():
    t0 = time.time()
    print("[v20b] Strategy B window sweep — w=5 and w=7")
    print(f"       GO Fisher peaks → contiguous window selection")

    # ── Setup ───────────────────────────────────────────────────────
    print("\n[1] Fisher peaks...")
    peaks = load_fisher_peaks()
    for go, name in list(GO_18.items())[:4]:
        p = peaks[go]
        print(f"   {go} {name[:20]:20s}  peak=L{p+1:02d}")

    print("\n[2] IDs...")
    tr_gene = np.load(ID_DIR  / "train_gene_list.npy",    allow_pickle=True)
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)

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

    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in tr_gene]
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_gene]
    print(f"   Train={len(sym_tr)}  Test={len(sym_te)}")

    print("\n[3] Type-3 genes...")
    dm = np.load(FEAT_DIR / "domain_matrix_proper_test.npy", mmap_mode="r")
    dc = np.array(dm.sum(1)).ravel(); del dm
    te_gene_base = [clean(g).split('.')[0] for g in te_gene]
    gene2dc = defaultdict(list)
    for i, g in enumerate(te_gene_base): gene2dc[g].append(dc[i])
    dc_range = {g: max(v)-min(v) for g, v in gene2dc.items()}
    type3_genes = {g for g, r in dc_range.items() if r == 0}
    print(f"   Type-3: {len(type3_genes)} genes ({len(type3_genes)/len(dc_range):.1%} of multi-isoform genes)")

    print("\n[4] Z cache (per-layer PCA)...")
    Z_tr = np.load(CACHE_DIR / "Z_tr.npy")   # (31668, 30, 8)
    Z_te = np.load(CACHE_DIR / "Z_te.npy")   # (36748, 30, 8)
    print(f"   Z_tr={Z_tr.shape}  Z_te={Z_te.shape}")

    print("\n[5] L30 embeddings...")
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)
    print(f"   Setup complete [{time.time()-t0:.1f}s]")

    # ── Run w=5 and w=7 ─────────────────────────────────────────────
    all_results = {}
    for w in W_LIST:
        r = run_w(w, peaks, Z_tr, Z_te, X_L30_tr, X_L30_te,
                  sym_tr, sym_te, type3_genes, t0)
        all_results[f"w{w}"] = r

    # ── Final comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Final Comparison")
    print(f"{'='*70}")
    print(f"  {'Model':15s}  {'AUPRC':>8}  {'Δv15d':>7}  {'T3/T12_ALL':>11}  {'T3/T12_MID':>11}")
    print(f"  {'v15d (base)':15s}  {0.7022:>8.4f}  {'---':>7}  {0.3377:>11.4f}  {0.3432:>11.4f}")
    print(f"  {'v20(Fisher avg)':15s}  {0.7012:>8.4f}  {0.7012-0.7022:>+7.4f}  {0.3480:>11.4f}  {0.3140:>11.4f}")
    for w in W_LIST:
        r = all_results[f"w{w}"]
        print(f"  {'v20b w='+str(w):15s}  {r['macro_auprc']:>8.4f}  "
              f"{r['delta_vs_v15d']:>+7.4f}  {r['t3_t12_ratio_all']:>11.4f}  "
              f"{r['t3_t12_ratio_mid']:>11.4f}")
    print(f"  {'v19 (curve240)':15s}  {0.6721:>8.4f}  {0.6721-0.7022:>+7.4f}  {0.3843:>11.4f}  {0.4493:>11.4f}")

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] total elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
