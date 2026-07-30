"""
v17f_bp_delta.py
================
v17f*-style δ_layer = (L30 − L15)를 18 BP GO terms에 적용.

비교 목적:
  "단순 delta" vs "rectified flow GO-specific window (v20b)" 공정 비교.
  두 접근의 차이:
    v17f*-BP : δ = L30 − L15 (고정, GO 무관) → [L30 ∥ δ] = 1280-dim
    v20b w=7 : GO별 Fisher peak 주변 window 선택 → [L30 ∥ Z_win] = 760-dim

평가 기준 (18 BP GO, 동일 벤치마크):
  v15d:      AUPRC=0.7022  T3/T12_ALL=0.3377  MID=0.3432
  v20(avg):  AUPRC=0.7012  T3/T12_ALL=0.3480  MID=0.3140
  v19(all):  AUPRC=0.6721  T3/T12_ALL=0.3843  MID=0.4493
"""
from __future__ import annotations

import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 1 (GPU 0은 v20b 사용 중)

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
OUT_DIR   = ROOT.parent / "reports" / "v17f_bp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS  = 5
DELTA_L1 = 30   # L30 (final layer)
DELTA_L2 = 15   # L15 (mid layer) → δ = L30 - L15

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


def main():
    t0 = time.time()
    print("[v17f-BP] δ_layer = L30 − L15  (1280-dim) on 18 BP GO terms")
    print(f"          CUDA_VISIBLE_DEVICES=1 (GPU 1, parallel with v20b on GPU 0)")
    print(f"          Comparison target: v20b GO-specific window curve")

    # ── IDs ─────────────────────────────────────────────────────────
    print("\n[1] IDs...")
    tr_gene = np.load(ID_DIR  / "train_gene_list.npy",      allow_pickle=True)
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

    # ── Type-3 genes ─────────────────────────────────────────────────
    print("\n[2] Type-3 genes...")
    dm = np.load(FEAT_DIR / "domain_matrix_proper_test.npy", mmap_mode="r")
    dc = np.array(dm.sum(1)).ravel(); del dm
    te_gene_base = [clean(g).split('.')[0] for g in te_gene]
    gene2dc = defaultdict(list)
    for i, g in enumerate(te_gene_base): gene2dc[g].append(dc[i])
    dc_range = {g: max(v)-min(v) for g, v in gene2dc.items()}
    type3_genes = {g for g, r in dc_range.items() if r == 0}
    print(f"   Type-3: {len(type3_genes)} genes")

    # ── Load L30 and L15, compute δ ──────────────────────────────────
    print(f"\n[3] Loading L{DELTA_L1} and L{DELTA_L2} embeddings → δ = L{DELTA_L1} − L{DELTA_L2}...")

    # Train
    X_L30_tr = np.load(DATA / f"esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L15_tr = np.load(DATA / f"esm2_train_human_layer{DELTA_L2:02d}_t30_150M.npy").astype(np.float32)
    delta_tr  = X_L30_tr - X_L15_tr   # (N_TR, 640)
    print(f"   Train: L30={X_L30_tr.shape}  L15={X_L15_tr.shape}  δ={delta_tr.shape}")
    del X_L15_tr

    # Test
    X_L30_te = np.load(DATA / f"esm2_embeddings_t30_150M.npy").astype(np.float32)
    X_L15_te = np.load(DATA / f"esm2_layer_{DELTA_L2:02d}_t30_150M.npy").astype(np.float32)
    delta_te  = X_L30_te - X_L15_te   # (N_TE, 640)
    del X_L15_te

    # Standardize δ (fit on train)
    sc_delta = StandardScaler()
    delta_tr_sc = sc_delta.fit_transform(delta_tr).astype(np.float32)
    delta_te_sc = sc_delta.transform(delta_te).astype(np.float32)
    del delta_tr, delta_te

    # Final input: [L30 ∥ δ] = 1280-dim (same as v17f* on MF terms)
    X_tr_full = np.concatenate([X_L30_tr, delta_tr_sc], axis=1)   # (N_TR, 1280)
    X_te_full = np.concatenate([X_L30_te, delta_te_sc], axis=1)   # (N_TE, 1280)
    INPUT_DIM = X_tr_full.shape[1]
    print(f"   Full input: {X_tr_full.shape}  dim={INPUT_DIM}")
    print(f"   [{time.time()-t0:.1f}s] setup done")

    # ── Per-GO training ──────────────────────────────────────────────
    print(f"\n[4] Training {len(GO_18)} GO terms (δ_layer 1280-dim, {N_SEEDS} seeds each)...")
    auprc_list, spread_results = [], []
    score_matrix = np.zeros((len(sym_te), len(GO_18)), dtype=np.float32)

    for gi, (go, go_name) in enumerate(GO_18.items()):
        t1 = time.time()
        flag = "[MID]" if go in MID_GOs else "     "

        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        if y_te.sum() == 0:
            print(f"  [{gi+1:2d}/{len(GO_18)}] {flag} {go} {go_name[:20]:20s}  SKIP")
            continue

        preds = run_ensemble(X_tr_full, y_tr, X_te_full, INPUT_DIM)
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
            "auprc": round(auprc, 4),
            "T3_spread_mean":  round(float(np.mean(t3_sp))  if t3_sp  else 0, 5),
            "T12_spread_mean": round(float(np.mean(t12_sp)) if t12_sp else 0, 5),
            "T3_T12_ratio":    round(t3_t12, 4),
        })
        print(f"  [{gi+1:2d}/{len(GO_18)}] {flag} {go} {go_name[:20]:20s}  "
              f"AUPRC={auprc:.4f}  T3/T12={t3_t12:.4f}  ({time.time()-t1:.0f}s)")

    macro = float(np.mean(auprc_list))
    all_ratios = [r["T3_T12_ratio"] for r in spread_results if r["T3_T12_ratio"] > 0]
    mid_ratios = [r["T3_T12_ratio"] for r in spread_results if r["is_mid"] and r["T3_T12_ratio"] > 0]
    t3_all = float(np.mean(all_ratios)) if all_ratios else 0.0
    t3_mid = float(np.mean(mid_ratios)) if mid_ratios else 0.0

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  v17f*-BP  δ_layer(L30−L15) on 18 BP GO — Final Summary")
    print(f"{'='*70}")
    print(f"  Macro AUPRC = {macro:.4f}")
    print(f"  T3/T12 ALL  = {t3_all:.4f}")
    print(f"  T3/T12 MID  = {t3_mid:.4f}")
    print(f"\n  --- Full comparison (18 BP GO benchmark) ---")
    print(f"  {'Model':18s}  {'AUPRC':>8}  {'Δv15d':>7}  {'T3/T12_ALL':>11}  {'T3/T12_MID':>11}  Description")
    rows = [
        ("v15d (base)",    0.7022, 0.0000, 0.3377, 0.3432, "L30 only (640-dim)"),
        ("v20 Fisher avg", 0.7012, -0.0010, 0.3480, 0.3140, "L30 + Fisher-wtd 8-dim"),
        ("v17f*-BP(this)", macro,  macro-0.7022, t3_all, t3_mid, "L30 + δ(L30-L15) 1280-dim"),
    ]
    for name, auprc, da, t3a, t3m, desc in rows:
        print(f"  {name:18s}  {auprc:>8.4f}  {da:>+7.4f}  {t3a:>11.4f}  {t3m:>11.4f}  {desc}")
    print(f"  {'v19 curve240':18s}  {0.6721:>8.4f}  {0.6721-0.7022:>+7.4f}  {0.3843:>11.4f}  {0.4493:>11.4f}  L30 + 240-dim curve (all layers)")
    print(f"  {'v20b w=5 (TBD)':18s}  {'---':>8}  {'---':>7}  {'---':>11}  {'---':>11}  L30 + GO-win w=5 728-dim")
    print(f"  {'v20b w=7 (TBD)':18s}  {'---':>8}  {'---':>7}  {'---':>11}  {'---':>11}  L30 + GO-win w=7 760-dim")

    # ── Save ─────────────────────────────────────────────────────────
    result = {
        "model": "v17f_bp_delta",
        "description": "L30(640) + delta_L30_L15(640) = 1280-dim, 18 BP GO terms",
        "input_dim": INPUT_DIM,
        "delta_layers": f"L{DELTA_L1} - L{DELTA_L2}",
        "macro_auprc": round(macro, 4),
        "delta_vs_v15d": round(macro - 0.7022, 4),
        "t3_t12_ratio_all": round(t3_all, 4),
        "t3_t12_ratio_mid": round(t3_mid, 4),
        "references": {
            "v15d":  {"macro_auprc": 0.7022, "t3_t12_all": 0.3377, "t3_t12_mid": 0.3432},
            "v20":   {"macro_auprc": 0.7012, "t3_t12_all": 0.3480, "t3_t12_mid": 0.3140},
            "v19":   {"macro_auprc": 0.6721, "t3_t12_all": 0.3843, "t3_t12_mid": 0.4493},
            "v17f_mf": {"macro_auprc": 0.7325, "note": "82 MF GO terms (different benchmark)"},
        },
        "per_go": spread_results,
    }
    np.save(OUT_DIR / "v17f_bp_score_matrix.npy", score_matrix)
    with open(OUT_DIR / "v17f_bp_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {OUT_DIR}/v17f_bp_results.json")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
