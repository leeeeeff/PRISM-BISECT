"""
interp_v20b_pca_axes.py
=======================
v20b가 사용하는 Joint PCA 8축 W(8x640)의 해석 — STEP 1: variance decomposition.

핵심 질문(사용자 통찰 승격):
  8개 축의 생물학적 의미를 해석하기 전에, 먼저 "어느 축이 within-gene(isoform 구별) 분산을
  담는가"를 확정한다. gene-family(between-gene) 분산만 담는 축은 같은 유전자의 isoform을 구별할
  수 없으므로, 그 축의 생물학 해석은 isoform 기능 예측 관점에서 무의미하다.
  (이것은 macro AUPRC 97.7%=gene-level 문제가 표현(representation) 수준에서 재등장하는지 검사.)

모델 표현 재현 (v20b_brain_port.compute_Z_fit와 동일):
  per-layer z-score → flatten (N*30, 640) → joint PCA(8) on MUSCLE train → W(8x640).
  brain에 동일 stats+PCA 적용 → Z_br (N,30,8).

분산 분해 (per axis k, per layer L, multi-isoform isoform 집단 위에서):
  SS_total = SS_between_gene + SS_within_gene   (one-way ANOVA)
  within_frac = SS_within / SS_total
  → within_frac이 큰 축 = isoform 구별 가능 축 (생물학 해석 가치 있음).
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
HMU = ROOT / "hMuscle"
DATA = HMU / "data"
ID_DIR = DATA / "raw_data/data/id_lists"
BRAIN_DIR = DATA / "brain_isoquant_esm2/full"
OUT_DIR = ROOT / "reports" / "v20b_pca_interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS, EMB_DIM, K_PCA, SEED = 30, 640, 8, 42
t0 = time.time()


def build_traj(prefix_fmt, N):
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        arr = np.load(prefix_fmt(L), mmap_mode="r")
        traj[:, L - 1, :] = arr.astype(np.float32)
        del arr
    return traj


def compute_Z_fit(traj):
    N = traj.shape[0]
    tn = np.empty_like(traj)
    stats = []
    for L in range(N_LAYERS):
        mu = traj[:, L, :].mean(0); sd = traj[:, L, :].std(0) + 1e-6
        stats.append((mu, sd)); tn[:, L, :] = (traj[:, L, :] - mu) / sd
    flat = tn.reshape(N * N_LAYERS, EMB_DIM)
    pca = PCA(n_components=K_PCA, random_state=SEED)
    Z = pca.fit_transform(flat).reshape(N, N_LAYERS, K_PCA).astype(np.float32)
    return Z, stats, pca


def compute_Z_transform(traj, stats, pca):
    N = traj.shape[0]
    tn = np.empty_like(traj)
    for L in range(N_LAYERS):
        mu, sd = stats[L]; tn[:, L, :] = (traj[:, L, :] - mu) / sd
    return pca.transform(tn.reshape(N * N_LAYERS, EMB_DIM)).reshape(N, N_LAYERS, K_PCA).astype(np.float32)


def variance_decomp(Zk_layer, gene_of, multi_mask):
    """Zk_layer: (N,) values of axis k at one layer. Decompose over MULTI-iso isoforms.
    Returns within_frac, between_frac."""
    x = Zk_layer[multi_mask]
    g = gene_of[multi_mask]
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    # group by gene
    idx_by_g = defaultdict(list)
    for i, gg in enumerate(g):
        idx_by_g[gg].append(i)
    ss_within = 0.0
    ss_between = 0.0
    for gg, idx in idx_by_g.items():
        xi = x[idx]
        gm = xi.mean()
        ss_within += float(((xi - gm) ** 2).sum())
        ss_between += float(len(idx) * (gm - grand) ** 2)
    within_frac = ss_within / ss_total if ss_total > 0 else float("nan")
    return within_frac, ss_within, ss_between, ss_total


def main():
    print("[1] muscle train IDs / genes")
    tr_gene = np.load(ID_DIR / "train_gene_list.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                ENSG2SYM[p[0]] = p[4]
    clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
    sym_tr = [ENSG2SYM.get(clean(g).split(".")[0], clean(g).split(".")[0]) for g in tr_gene]
    N_tr = len(sym_tr)

    print(f"[2] muscle train trajectory (N={N_tr}) ... [{time.time()-t0:.0f}s]")
    traj_tr = build_traj(lambda L: DATA / f"esm2_train_human_layer{L:02d}_t30_150M.npy", N_tr)
    print(f"[3] PCA(8) fit on muscle train ... [{time.time()-t0:.0f}s]")
    Z_tr, stats, pca = compute_Z_fit(traj_tr)
    del traj_tr
    W = pca.components_.astype(np.float32)               # (8, 640)
    evr = pca.explained_variance_ratio_.astype(np.float32)
    np.save(OUT_DIR / "W_axes_8x640.npy", W)
    np.save(OUT_DIR / "pca_mean_640.npy", pca.mean_.astype(np.float32))
    np.save(OUT_DIR / "layer_stats_mu.npy", np.stack([s[0] for s in stats]))
    np.save(OUT_DIR / "layer_stats_sd.npy", np.stack([s[1] for s in stats]))
    print(f"   W saved (8x640). explained_var_ratio per axis: "
          f"{[round(float(v),4) for v in evr]}  sum={float(evr.sum()):.4f}")

    print(f"[4] brain trajectory + transform ... [{time.time()-t0:.0f}s]")
    sym_br = np.array([clean(s) for s in np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)])
    traj_br = build_traj(lambda L: BRAIN_DIR / f"brain_full_esm2_layer{L:02d}_t30_150M.npy", len(sym_br))
    Z_br = compute_Z_transform(traj_br, stats, pca)
    del traj_br
    np.save(OUT_DIR / "Z_brain_Nx30x8.npy", Z_br)
    print(f"   Z_br {Z_br.shape} saved [{time.time()-t0:.0f}s]")

    # multi-isoform mask (brain)
    g2c = defaultdict(int)
    for g in sym_br:
        g2c[g] += 1
    multi_mask = np.array([g2c[g] >= 2 for g in sym_br])
    print(f"[5] variance decomposition (brain, multi-iso isoforms n={int(multi_mask.sum())})")

    # per-axis: within_frac averaged over layers, and at each layer
    result = {"explained_var_ratio": [float(v) for v in evr], "per_axis": {}}
    print(f"\n{'axis':>4} {'evr':>7} {'within_frac(mean±sd over L)':>28} {'min_L':>6} {'max_L':>6}")
    for k in range(K_PCA):
        wf_by_layer = []
        for L in range(N_LAYERS):
            wf, _, _, _ = variance_decomp(Z_br[:, L, k], sym_br, multi_mask)
            wf_by_layer.append(wf)
        wf_arr = np.array(wf_by_layer)
        result["per_axis"][k] = {
            "explained_var_ratio": float(evr[k]),
            "within_frac_by_layer": [float(v) for v in wf_arr],
            "within_frac_mean": float(np.nanmean(wf_arr)),
            "within_frac_std": float(np.nanstd(wf_arr)),
        }
        print(f"{k:>4} {evr[k]:>7.4f} {np.nanmean(wf_arr):>18.4f} ± {np.nanstd(wf_arr):.4f}"
              f"   {np.nanmin(wf_arr):>6.3f} {np.nanmax(wf_arr):>6.3f}")

    with open(OUT_DIR / "variance_decomp.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {OUT_DIR/'variance_decomp.json'}")
    print("\n해석: within_frac이 클수록 그 축은 같은 유전자 isoform을 구별하는 '지역' 분산을 담는다.")
    print("       within_frac이 0에 가까우면 그 축은 gene-family(between-gene) 축 → isoform 해석 무의미.")


if __name__ == "__main__":
    main()
