"""
p1_p2_p3_occam_bp.py
=====================
Occam-null battery for the muscle-only BP δ_layer failure (v17f*-BP: 0.6588 vs v15d 0.7022, Δ=-0.0434).

devils-advocate 판정: "domain-intrinsic" 결론 이전에 더 단순한 대안부터 배제해야 함.

P1. δ-only 640-dim          — SNR dilution 테스트 (약한 δ signal이 L30 신호를 오염시키는가, 아니면 δ 자체가 무용한가)
P2. L30 zero-pad to 1280-dim — capacity/compression-ratio mismatch 테스트 (1280→256 압축비 자체가 문제인가)
P3. Dropout 0.15/0.1         — over-regularization 테스트 (640-dim에 튜닝된 dropout이 1280-dim에 과함)

전부 동일 프로토콜 유지: 18 BP GO, 5-seed ensemble, gene-disjoint muscle train/test, BinaryFocalCrossentropy(gamma=2.0).
"""
from __future__ import annotations

import os, json, time, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import warnings; warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data"
MODEL_DIR = ROOT / "model"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
OUT_DIR   = ROOT.parent / "reports" / "v17f_bp_occam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS  = 5
DELTA_L1 = 30
DELTA_L2 = 15

GO_18 = {
    "GO:0007204": "Ca2+ signaling", "GO:0045214": "Sarcomere organization",
    "GO:0006941": "Muscle contraction", "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS", "GO:0007519": "Skeletal muscle dev",
    "GO:0042692": "Muscle cell diff", "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org", "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling", "GO:0030048": "Actin-based movement",
    "GO:0006096": "Glycolysis", "GO:0007268": "Synaptic transmission",
    "GO:0007018": "MT-based movement", "GO:0031175": "Neuron proj development",
    "GO:0030182": "Neuron diff", "GO:0000226": "MT cytoskeleton org",
}


def build_model(input_dim: int, dropout1=0.3, dropout2=0.2):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout2)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def run_ensemble(X_tr, y_tr, X_te, input_dim, dropout1=0.3, dropout2=0.2):
    preds = []
    for s in range(N_SEEDS):
        tf.random.set_seed(s * 137 + 42)
        np.random.seed(s * 137 + 42)
        idx = np.random.permutation(len(X_tr))
        m = build_model(input_dim, dropout1, dropout2)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0))
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
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


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_ids():
    tr_gene = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in tr_gene]
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_gene]
    return sym_tr, sym_te


def run_arm(arm_name, X_tr_full, X_te_full, sym_tr, sym_te, dropout1=0.3, dropout2=0.2):
    input_dim = X_tr_full.shape[1]
    print(f"\n[{arm_name}] input_dim={input_dim}  dropout=({dropout1},{dropout2})")
    auprc_list, per_go = [], []
    for gi, (go, name) in enumerate(GO_18.items()):
        t1 = time.time()
        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        if y_te.sum() == 0:
            continue
        preds = run_ensemble(X_tr_full, y_tr, X_te_full, input_dim, dropout1, dropout2)
        auprc = float(average_precision_score(y_te, preds))
        auprc_list.append(auprc)
        per_go.append({"go": go, "name": name, "auprc": round(auprc, 4)})
        print(f"  [{gi+1:2d}/18] {go} {name[:25]:25s} AUPRC={auprc:.4f} ({time.time()-t1:.0f}s)")
    macro = float(np.mean(auprc_list))
    print(f"  {arm_name} macro AUPRC = {macro:.4f}")
    return macro, per_go


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Occam-null battery: P1 (δ-only) / P2 (L30 zero-pad) / P3 (low dropout)")
    print("=" * 70)

    sym_tr, sym_te = load_ids()
    print(f"Train={len(sym_tr)}  Test={len(sym_te)}")

    print("\n[Loading embeddings]")
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L15_tr = np.load(DATA / f"esm2_train_human_layer{DELTA_L2:02d}_t30_150M.npy").astype(np.float32)
    delta_tr = X_L30_tr - X_L15_tr

    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)
    X_L15_te = np.load(DATA / f"esm2_layer_{DELTA_L2:02d}_t30_150M.npy").astype(np.float32)
    delta_te = X_L30_te - X_L15_te
    del X_L15_tr, X_L15_te

    sc_delta = StandardScaler()
    delta_tr_sc = sc_delta.fit_transform(delta_tr).astype(np.float32)
    delta_te_sc = sc_delta.transform(delta_te).astype(np.float32)
    del delta_tr, delta_te
    print(f"  L30 train={X_L30_tr.shape}  delta train={delta_tr_sc.shape}")

    results = {}

    # ===== P1: δ-only 640-dim (SNR dilution test) =====
    print("\n" + "="*70)
    print("P1: δ-only 640-dim — if delta alone is worse than L30-only (0.7022),")
    print("    delta signal itself is weak/uninformative for these BP terms.")
    print("    if delta alone beats v17f*-BP (0.6588), dilution from concat confirmed.")
    print("="*70)
    macro_p1, per_go_p1 = run_arm("P1_delta_only", delta_tr_sc, delta_te_sc, sym_tr, sym_te)
    results['P1_delta_only'] = {'macro_auprc': macro_p1, 'per_go': per_go_p1}

    # ===== P2: L30 zero-padded to 1280-dim (capacity mismatch test) =====
    print("\n" + "="*70)
    print("P2: L30 zero-padded to 1280-dim — same input_dim/bottleneck ratio as v17f*-BP")
    print("    but zero informative delta signal. If this recovers ~0.7022, the 1280-dim")
    print("    capacity/compression ratio itself is NOT the problem (rules out capacity).")
    print("    If it stays near v17f*-BP's 0.6588, capacity/compression mismatch confirmed.")
    print("="*70)
    zeros_tr = np.zeros_like(X_L30_tr)
    zeros_te = np.zeros_like(X_L30_te)
    X_tr_p2 = np.concatenate([X_L30_tr, zeros_tr], axis=1)
    X_te_p2 = np.concatenate([X_L30_te, zeros_te], axis=1)
    del zeros_tr, zeros_te
    macro_p2, per_go_p2 = run_arm("P2_L30_zeropad", X_tr_p2, X_te_p2, sym_tr, sym_te)
    results['P2_L30_zeropad'] = {'macro_auprc': macro_p2, 'per_go': per_go_p2}
    del X_tr_p2, X_te_p2

    # ===== P3: v17f*-BP with lower dropout (over-regularization test) =====
    print("\n" + "="*70)
    print("P3: [L30 || δ] 1280-dim with dropout (0.15, 0.1) instead of (0.3, 0.2).")
    print("    If this recovers toward 0.7022, dropout tuned for 640-dim was")
    print("    over-suppressing the (weaker) delta signal at 1280-dim.")
    print("="*70)
    X_tr_p3 = np.concatenate([X_L30_tr, delta_tr_sc], axis=1)
    X_te_p3 = np.concatenate([X_L30_te, delta_te_sc], axis=1)
    macro_p3, per_go_p3 = run_arm("P3_low_dropout", X_tr_p3, X_te_p3, sym_tr, sym_te,
                                    dropout1=0.15, dropout2=0.1)
    results['P3_low_dropout'] = {'macro_auprc': macro_p3, 'per_go': per_go_p3}

    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  v15d (L30 only, baseline)        = 0.7022")
    print(f"  v17f*-BP (L30||δ, dropout .3/.2)  = 0.6588  (Δ=-0.0434, prior result)")
    print(f"  P1  δ-only (640-dim)              = {macro_p1:.4f}  (Δ vs v15d = {macro_p1-0.7022:+.4f})")
    print(f"  P2  L30 zero-pad (1280-dim)       = {macro_p2:.4f}  (Δ vs v15d = {macro_p2-0.7022:+.4f})")
    print(f"  P3  L30||δ low-dropout (1280-dim) = {macro_p3:.4f}  (Δ vs v17f*-BP = {macro_p3-0.6588:+.4f})")

    results['references'] = {
        'v15d_L30_only': 0.7022,
        'v17f_bp_original': 0.6588,
    }
    results['elapsed_sec'] = time.time() - t0

    with open(OUT_DIR / "occam_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {OUT_DIR}/occam_results.json")
    print(f"[done] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
