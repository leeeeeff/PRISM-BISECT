"""
train_expanded41_truebrain.py
==============================
Head-to-head architecture ablation for the 41-term BP panel (18 existing_18 +
23 new_23 brain-relevant terms), trained on muscle, evaluated true-brain zero-shot
(63,994 isoforms) -- the actual deployment target of this model family.

Two architectures, identical GO-term set, identical train/test split, identical
focal-loss/5-seed protocol:
  (A) v15d      : L30 mean-pooled only (640-dim)                  -- current deployed arch
  (B) v17f*-BP  : [L30 || delta(L30-L15)] concat (1280-dim)        -- paper's SOTA arch (for MF)

Why this run exists: the existing v17f_bp_delta.py ablation (18 BP terms, MUSCLE
held-out only) already showed delta_layer HURTS BP performance (0.6588 vs v15d's
0.7022, Delta=-0.0434). This script extends that exact comparison to (i) the full
41-term deployment panel and (ii) true-brain zero-shot -- the actual use case --
to settle whether that muscle-only 18-term result generalizes, rather than relying
on a proxy. This also produces the first committed, reproducible training script
for the 41-term v15d baseline (the original June 2026 app-only run's script was
never found in the repo).

Ground truth: human_annotations_unified_bp.txt (UniProt/QuickGO unified), the
SAME source verified against the canonical 18-term brain eval
(reports/v15d_brain_eval/brain_eval_20260519_2125.json, macro_auprc_all18=0.5998
matching the paper's cited 0.600) -- NOT gene2go.gz, which the separate 279-term
panel uses and which undercounts positives for broad BP terms by ~2-3x.

Output: reports/expanded41_truebrain/{v15d,v17fstar}_score_matrix.npy + meta.json
"""
from __future__ import annotations
import os, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 1: confirmed free at launch time

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import warnings; warnings.filterwarnings('ignore')

ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data"
BRAIN_DIR = DATA / "brain_isoquant_esm2" / "full"
ID_DIR    = DATA / "raw_data" / "data" / "id_lists"
ANNOT_DIR = DATA / "raw_data" / "data" / "annotations"
APP_DIR   = ROOT.parent / "prism_app" / "data" / "demo"
OUT_DIR   = ROOT.parent / "reports" / "expanded41_truebrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
DELTA_L1, DELTA_L2 = 30, 15

meta41 = json.load(open(APP_DIR / "brain_full_expanded_41_meta.json"))
GO_41  = {go: meta41['go_names'][go] for go in meta41['go_ids']}
GO_SOURCE = meta41['go_source']
print(f"[setup] {len(GO_41)} GO terms (existing_18={sum(1 for v in GO_SOURCE.values() if v=='existing_18')}, "
      f"new_23={sum(1 for v in GO_SOURCE.values() if v=='new_23')})")


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_labels(go, sym_tr, sym_te):
    pos = set()
    with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in sym_tr], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in sym_te], dtype=np.float32)
    return y_tr, y_te


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
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
        ]
        m.fit(X_tr[idx], y_tr[idx], epochs=80, batch_size=512,
              validation_split=0.1, callbacks=cb, verbose=0)
        preds.append(m.predict(X_te, batch_size=2048, verbose=0).flatten())
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  41-term BP panel: v15d vs v17f*-BP head-to-head, true-brain zero-shot")
    print("=" * 70)

    # ── IDs (muscle train, true brain test) ──────────────────────────
    print("\n[1] IDs...")
    tr_gene_raw = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in tr_gene_raw]

    te_gene_raw = np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)
    sym_te = [clean(g) for g in te_gene_raw]   # true brain: already symbols
    print(f"   Train={len(sym_tr)}  TrueBrain Test={len(sym_te)}")

    # ── Embeddings: L30 (mean-pool baseline) + L15 (for delta) ───────
    print("\n[2] Loading embeddings...")
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(BRAIN_DIR / "brain_full_esm2_layer30_t30_150M.npy").astype(np.float32)
    print(f"   L30 train={X_L30_tr.shape}  test={X_L30_te.shape}")

    X_L15_tr = np.load(DATA / f"esm2_train_human_layer{DELTA_L2:02d}_t30_150M.npy").astype(np.float32)
    X_L15_te = np.load(BRAIN_DIR / f"brain_full_esm2_layer{DELTA_L2:02d}_t30_150M.npy").astype(np.float32)
    delta_tr = X_L30_tr - X_L15_tr
    delta_te = X_L30_te - X_L15_te
    del X_L15_tr, X_L15_te

    sc = StandardScaler()
    delta_tr_sc = sc.fit_transform(delta_tr).astype(np.float32)
    delta_te_sc = sc.transform(delta_te).astype(np.float32)
    del delta_tr, delta_te

    X_tr_v15d = X_L30_tr
    X_te_v15d = X_L30_te
    X_tr_v17f = np.concatenate([X_L30_tr, delta_tr_sc], axis=1)
    X_te_v17f = np.concatenate([X_L30_te, delta_te_sc], axis=1)
    print(f"   v15d input dim={X_tr_v15d.shape[1]}  v17f*-BP input dim={X_tr_v17f.shape[1]}")
    print(f"   [{time.time()-t0:.1f}s] setup done")

    go_list = list(GO_41.items())
    for arch_name, X_tr_full, X_te_full, in_dim in [
        ("v15d",     X_tr_v15d, X_te_v15d, X_tr_v15d.shape[1]),
        ("v17fstar", X_tr_v17f, X_te_v17f, X_tr_v17f.shape[1]),
    ]:
        print(f"\n[3] Training {arch_name} ({len(go_list)} terms x {N_SEEDS} seeds)...")
        score_matrix = np.zeros((len(sym_te), len(go_list)), dtype=np.float32)
        auprc_list, per_go = [], []
        for gi, (go, name) in enumerate(go_list):
            t1 = time.time()
            y_tr, y_te = load_labels(go, sym_tr, sym_te)
            if y_te.sum() < 2:
                print(f"  [{gi+1:2d}/{len(go_list)}] {go} {name[:30]:30s} SKIP (n_pos_te<2)")
                continue
            preds = run_ensemble(X_tr_full, y_tr, X_te_full, in_dim)
            score_matrix[:, gi] = preds
            auprc = float(average_precision_score(y_te, preds))
            auprc_list.append(auprc)
            per_go.append({'go': go, 'name': name, 'source': GO_SOURCE[go],
                            'n_pos_tr': int(y_tr.sum()), 'n_pos_te': int(y_te.sum()), 'auprc': round(auprc, 4)})
            print(f"  [{gi+1:2d}/{len(go_list)}] {go} {name[:30]:30s} AUPRC={auprc:.4f} ({time.time()-t1:.0f}s)")

        macro = float(np.mean(auprc_list))
        print(f"\n  {arch_name} macro AUPRC (true brain, {len(go_list)} terms) = {macro:.4f}")
        np.save(OUT_DIR / f"{arch_name}_score_matrix.npy", score_matrix)
        with open(OUT_DIR / f"{arch_name}_meta.json", "w") as f:
            json.dump({
                'model': f'{arch_name}_expanded41_truebrain', 'n_go': len(go_list),
                'input_dim': in_dim, 'macro_auprc_truebrain': round(macro, 4),
                'go_ids': [g for g, _ in go_list], 'go_names': GO_41, 'go_source': GO_SOURCE,
                'per_go': per_go,
            }, f, indent=2)
        print(f"  [saved] {OUT_DIR}/{arch_name}_score_matrix.npy + _meta.json")

    # ids/gene_ids for downstream DR-AUC/pos_bias validation (same order as score matrices)
    np.save(OUT_DIR / "ids.npy", np.array(sym_te, dtype=object))  # gene symbols, brain isoform order
    print(f"\n[done] total elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
