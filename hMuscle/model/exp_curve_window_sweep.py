"""
exp_curve_window_sweep.py
=========================
GO-specific layer 선택 폭 sweep 실험.

핵심 질문: curve를 몇 개 layer로 잘라야 노이즈 최소화 + T3 분리 최대화 균형점?

두 가지 선택 전략:
  A. Top-K:    GO별 Fisher score 상위 K개 layer 선택 (비연속 가능)
  B. Window:   GO별 peak ± w 범위의 연속 layer 선택

K 또는 w 별 평가:
  - Macro AUPRC (LR proxy, 빠름)
  - T3/T12 within-gene spread ratio
  - Positive mean score (dilution 추적)

입력 구조:
  K layer 선택 시: [L30(640) | Z_{sel}(K×8)] = (640 + K×8) dim
  K=1  → 648-dim
  K=5  → 680-dim
  K=10 → 720-dim
  K=30 → 880-dim (= v19)

비교 기준:
  v15d: AUPRC=0.7022, T3/T12 ratio=0.338
  v19:  AUPRC=0.6721, T3/T12 ratio=0.384 (K=30 전체, 가중합 아님)
"""
from __future__ import annotations

import json, time
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data"
MODEL    = ROOT / "model"
ID_DIR   = DATA / "raw_data/data/id_lists"
ANNOT_DIR= DATA / "raw_data/data/annotations"
FEAT_DIR = ROOT / "results_isoform" / "features"
PROBE_DIR= ROOT.parent / "reports" / "layer_probe"
CACHE_DIR= ROOT.parent / "reports" / "v20_cache"
OUT_DIR  = ROOT.parent / "reports" / "curve_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Sweep configs
TOP_K_LIST   = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
WINDOW_SIZES = [0, 1, 2, 3, 4, 5, 7, 10, 14]   # ±w → 2w+1 layers


def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s


def load_fisher_scores() -> dict[str, np.ndarray]:
    """Raw Fisher (LR AUPRC) per layer per GO — NOT softmax weighted."""
    all_lr: dict[str, list] = {}
    for fname in ["layer_probe_v15d_terms_results.json",
                  "layer_probe_expanded_results.json",
                  "layer_probe_results.json"]:
        d = json.load(open(PROBE_DIR / fname))
        all_lr.update(d["lr_auprc"])
    return {go: np.array(all_lr[go], dtype=np.float32) for go in GO_18}


def load_labels(go: str, sym_tr, sym_te):
    pos = set()
    with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return (np.array([1 if s in pos else 0 for s in sym_tr], np.float32),
            np.array([1 if s in pos else 0 for s in sym_te], np.float32))


def within_gene_spread_ratio(preds, sym_te, type3_genes):
    g2i = defaultdict(list)
    for i, g in enumerate(sym_te): g2i[g].append(i)
    t3, t12 = [], []
    for g, idxs in g2i.items():
        if len(idxs) < 2: continue
        sp = float(preds[idxs].max() - preds[idxs].min())
        (t3 if g in type3_genes else t12).append(sp)
    if not t3 or not t12: return 0.0, 0.0, 0.0
    return float(np.mean(t3)), float(np.mean(t12)), float(np.mean(t3) / np.mean(t12))


def eval_config(layer_indices: list[int],
                Z_tr, Z_te, X_L30_tr, X_L30_te,
                sym_tr, sym_te, type3_genes,
                fisher_scores: dict) -> dict:
    """
    layer_indices: list of 0-indexed layers to include.
    Returns macro AUPRC, T3/T12 ratio, mean pos score across 18 GOs.
    """
    n_layers = len(layer_indices)
    auprc_list, ratio_list = [], []
    pos_mean_list = []

    for go in GO_18:
        y_tr, y_te = load_labels(go, sym_tr, sym_te)
        if y_te.sum() == 0:
            continue

        if n_layers == 0:
            # baseline: L30 only
            X_tr_full = X_L30_tr
            X_te_full = X_L30_te
        else:
            # GO-specific top-K or window selection
            # Use the per-GO fisher ordering if strategy is "topk"
            # If strategy is "window", layer_indices is already GO-specific (passed in)
            c_tr = Z_tr[:, layer_indices, :].reshape(Z_tr.shape[0], -1)
            c_te = Z_te[:, layer_indices, :].reshape(Z_te.shape[0], -1)
            sc = StandardScaler()
            c_tr = sc.fit_transform(c_tr).astype(np.float32)
            c_te = sc.transform(c_te).astype(np.float32)
            X_tr_full = np.concatenate([X_L30_tr, c_tr], axis=1)
            X_te_full = np.concatenate([X_L30_te, c_te], axis=1)

        lr = LogisticRegression(max_iter=200, C=1.0, solver='liblinear',
                                class_weight='balanced', random_state=42)
        lr.fit(X_tr_full, y_tr)
        preds = lr.predict_proba(X_te_full)[:, 1]

        auprc = average_precision_score(y_te, preds)
        t3_sp, t12_sp, ratio = within_gene_spread_ratio(preds, sym_te, type3_genes)
        auprc_list.append(auprc)
        ratio_list.append(ratio)
        if y_te.sum() > 0:
            pos_mask = y_te == 1
            pos_mean_list.append(float(preds[pos_mask].mean()))

    return {
        "macro_auprc":    float(np.mean(auprc_list)),
        "t3_t12_ratio":   float(np.mean(ratio_list)),
        "pos_mean":       float(np.mean(pos_mean_list)) if pos_mean_list else 0.0,
        "n_layers":       n_layers,
        "input_dim":      640 + n_layers * 8,
    }


def main():
    t0 = time.time()
    print("[sweep] GO-specific layer selection window sweep")

    # ── Setup ──────────────────────────────────────────────────────
    te_gene = np.load(MODEL / "my_gene_list_fixed.npy", allow_pickle=True)
    tr_gene = np.load(ID_DIR  / "train_gene_list.npy",  allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

    sym_tr = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in tr_gene]
    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0]) for g in te_gene]
    te_gene_base = [clean(g).split('.')[0] for g in te_gene]

    dm = np.load(FEAT_DIR / "domain_matrix_proper_test.npy", mmap_mode="r")
    dc = np.array(dm.sum(1)).ravel(); del dm
    gene2dc = defaultdict(list)
    for i, g in enumerate(te_gene_base): gene2dc[g].append(dc[i])
    dc_range = {g: max(v)-min(v) for g,v in gene2dc.items()}
    type3_genes = {g for g,r in dc_range.items() if r==0}

    Z_tr = np.load(CACHE_DIR / "Z_tr.npy")   # (31668, 30, 8)
    Z_te = np.load(CACHE_DIR / "Z_te.npy")   # (36748, 30, 8)
    X_L30_tr = np.load(DATA / "esm2_train_human_t30_150M.npy").astype(np.float32)
    X_L30_te = np.load(DATA / "esm2_embeddings_t30_150M.npy").astype(np.float32)

    fisher_scores = load_fisher_scores()
    print(f"  Setup done: [{time.time()-t0:.1f}s]")

    # ── Strategy A: Top-K sweep (GO-specific, non-contiguous) ──────
    print("\n" + "="*80)
    print("Strategy A: Top-K Fisher layers (GO-specific, best K non-contiguous layers)")
    print("="*80)
    print(f"{'K':>4} {'dim':>5} {'AUPRC':>8} {'ΔAUPRC':>8} {'T3/T12':>8} {'Δratio':>8} {'pos_mean':>9}")

    # Baseline: K=0 (L30 only, approximates v15d)
    baseline = eval_config([], Z_tr, Z_te, X_L30_tr, X_L30_te,
                           sym_tr, sym_te, type3_genes, fisher_scores)
    print(f"{'0(base)':>7} {640:>5} {baseline['macro_auprc']:>8.4f} {'---':>8} "
          f"{baseline['t3_t12_ratio']:>8.4f} {'---':>8} {baseline['pos_mean']:>9.4f}")

    topk_results = []
    for K in TOP_K_LIST:
        t1 = time.time()
        # Per-GO: select top-K layers by Fisher score
        # For the eval, we compute per-GO differently
        # Modified: evaluate per-GO with its own layer selection
        auprc_list, ratio_list, pos_list = [], [], []
        for go in GO_18:
            y_tr, y_te = load_labels(go, sym_tr, sym_te)
            if y_te.sum() == 0: continue

            # GO-specific top-K
            top_k_idx = np.argsort(fisher_scores[go])[::-1][:K]   # 0-indexed
            c_tr = Z_tr[:, top_k_idx, :].reshape(len(sym_tr), -1)
            c_te = Z_te[:, top_k_idx, :].reshape(len(sym_te), -1)
            sc = StandardScaler()
            c_tr = sc.fit_transform(c_tr).astype(np.float32)
            c_te = sc.transform(c_te).astype(np.float32)
            X_tr = np.concatenate([X_L30_tr, c_tr], axis=1)
            X_te = np.concatenate([X_L30_te, c_te], axis=1)

            lr = LogisticRegression(max_iter=200, C=1.0, solver='liblinear',
                                    class_weight='balanced', random_state=42)
            lr.fit(X_tr, y_tr)
            preds = lr.predict_proba(X_te)[:, 1]

            auprc = average_precision_score(y_te, preds)
            _, _, ratio = within_gene_spread_ratio(preds, sym_te, type3_genes)
            auprc_list.append(auprc)
            ratio_list.append(ratio)
            pos_mask = y_te == 1
            pos_list.append(float(preds[pos_mask].mean()))

        r = {
            "K": K, "dim": 640 + K * 8,
            "macro_auprc": float(np.mean(auprc_list)),
            "t3_t12_ratio": float(np.mean(ratio_list)),
            "pos_mean": float(np.mean(pos_list)),
        }
        topk_results.append(r)
        da = r['macro_auprc'] - baseline['macro_auprc']
        dr = r['t3_t12_ratio'] - baseline['t3_t12_ratio']
        marker = " ★" if (da > -0.005 and dr > 0.01) else ""
        print(f"{K:>4} {r['dim']:>5} {r['macro_auprc']:>8.4f} {da:>+8.4f} "
              f"{r['t3_t12_ratio']:>8.4f} {dr:>+8.4f} {r['pos_mean']:>9.4f} "
              f"  [{time.time()-t1:.0f}s]{marker}")

    # ── Strategy B: Window sweep (contiguous, centered on peak) ────
    print("\n" + "="*80)
    print("Strategy B: Contiguous window ±w around GO-specific peak layer")
    print("="*80)
    print(f"{'±w':>4} {'n_L':>4} {'dim':>5} {'AUPRC':>8} {'ΔAUPRC':>8} {'T3/T12':>8} {'Δratio':>8}")

    window_results = []
    for w in WINDOW_SIZES:
        t1 = time.time()
        auprc_list, ratio_list = [], []
        for go in GO_18:
            y_tr, y_te = load_labels(go, sym_tr, sym_te)
            if y_te.sum() == 0: continue

            peak = int(np.argmax(fisher_scores[go]))   # 0-indexed
            lo = max(0, peak - w)
            hi = min(29, peak + w)
            win_idx = list(range(lo, hi + 1))          # contiguous

            c_tr = Z_tr[:, win_idx, :].reshape(len(sym_tr), -1)
            c_te = Z_te[:, win_idx, :].reshape(len(sym_te), -1)
            sc = StandardScaler()
            c_tr = sc.fit_transform(c_tr).astype(np.float32)
            c_te = sc.transform(c_te).astype(np.float32)
            X_tr = np.concatenate([X_L30_tr, c_tr], axis=1)
            X_te = np.concatenate([X_L30_te, c_te], axis=1)

            lr = LogisticRegression(max_iter=200, C=1.0, solver='liblinear',
                                    class_weight='balanced', random_state=42)
            lr.fit(X_tr, y_tr)
            preds = lr.predict_proba(X_te)[:, 1]

            auprc = average_precision_score(y_te, preds)
            _, _, ratio = within_gene_spread_ratio(preds, sym_te, type3_genes)
            auprc_list.append(auprc)
            ratio_list.append(ratio)

        n_L_avg = 2 * w + 1
        r = {
            "w": w, "n_layers_avg": n_L_avg, "dim_approx": 640 + n_L_avg * 8,
            "macro_auprc": float(np.mean(auprc_list)),
            "t3_t12_ratio": float(np.mean(ratio_list)),
        }
        window_results.append(r)
        da = r['macro_auprc'] - baseline['macro_auprc']
        dr = r['t3_t12_ratio'] - baseline['t3_t12_ratio']
        marker = " ★" if (da > -0.005 and dr > 0.01) else ""
        print(f"{w:>4} {n_L_avg:>4} {r['dim_approx']:>5} {r['macro_auprc']:>8.4f} "
              f"{da:>+8.4f} {r['t3_t12_ratio']:>8.4f} {dr:>+8.4f}  "
              f"[{time.time()-t1:.0f}s]{marker}")

    # ── Per-GO peak layers (reference) ────────────────────────────
    print("\n== GO별 Fisher peak layer & top-5 selected layers (참고) ==")
    print(f"{'GO':12s} {'peak_L':>7} {'top5_layers (0-idx)':>30}  profile")
    for go, name in GO_18.items():
        fs = fisher_scores[go]
        peak = int(np.argmax(fs)) + 1
        top5 = sorted(range(30), key=lambda l: fs[l], reverse=True)[:5]
        norm = (fs - fs.min()) / (fs.max() - fs.min() + 1e-6)
        bar = ''.join('█' if v>0.7 else ('▓' if v>0.5 else ('▒' if v>0.3 else '░')) for v in norm)
        mid = "[MID]" if go in MID_GOs else "     "
        print(f"{go:12s} {mid} L{peak:02d}  top5={[l+1 for l in top5]}   {bar}")

    # ── Save ───────────────────────────────────────────────────────
    result = {
        "baseline_L30_only":  baseline,
        "v15d_mlp_ref":       {"macro_auprc": 0.7022, "t3_t12_ratio": 0.3377},
        "v19_curve240_ref":   {"macro_auprc": 0.6721, "t3_t12_ratio": 0.3843},
        "strategy_A_topk":    topk_results,
        "strategy_B_window":  window_results,
        "elapsed_sec":        round(time.time()-t0, 1),
    }
    with open(OUT_DIR / "sweep_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[write] {OUT_DIR}/sweep_results.json")
    print(f"[done]  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
