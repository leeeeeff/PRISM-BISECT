#!/usr/bin/env python3
"""
Tier C — 순수 within-gene 640-dim PCA (gene-mean 잔차 basis)
==========================================================
joint PCA는 total variance 최대화 → between-gene(68.6%)에 지배되어 within-gene domain 신호가
저분산 축(axis3, ndom|len +0.274)으로 밀림. within-gene PCA는 gene-mean 잔차의 분산을 직접
최대화 → top 축이 곧 within-gene 구조. 질문: 이 basis가 joint axis3보다 강한 domain 축을 주나?

방법: brain traj(per-layer z, muscle stat) → 각 layer gene-mean 제거(in place) → pooled PCA(8)
      = W_wg. joint W와 principal angle 비교(낮을 것—다른 basis). 각 W_wg 축의 within-gene-resid
      biology + ndom|len partial → within-gene domain 축 식별.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "6"); os.environ.setdefault("MKL_NUM_THREADS", "6")
import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE"); OUT = ROOT / "reports/v20b_pca_interp"; WF = OUT / "within_family"
K, NL, EMB = 8, 30, 640; FEAT = M.FEAT_NAMES
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log("[1] brain traj (muscle z-score)")
traj, sym = M.load_brain_traj_normalized(); N = len(sym)
_, gidx = np.unique(sym, return_inverse=True); G = gidx.max() + 1
gcount = np.bincount(gidx, minlength=G); multi = gcount[gidx] >= 2
log(f"    N={N} genes={G} multi-iso={int(multi.sum())}")

log("[2] gene-center each layer (within-gene residual) in place")
M.family_center_inplace(traj, gidx, G)     # subtract gene mean per layer

log("[3] pooled within-gene PCA(8)")
W_wg, evr_wg, Z_wg = M.fit_pca(traj.reshape(N * NL, EMB)); del traj
log(f"    within-gene evr={[round(float(v),4) for v in evr_wg]} sum={float(evr_wg.sum()):.4f}")

log("[4] principal angle vs joint / within-family")
W_joint = np.load(OUT / "W_axes_8x640.npy"); W_wfam = np.load(WF / "W_within_8x640.npy")
pa = lambda A, B: np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0, 1)
paj, paw = pa(W_wg, W_joint), pa(W_wg, W_wfam)
log(f"    pa(withingene~joint)  = {[round(float(v),3) for v in paj]} mean={paj.mean():.3f}  (낮을수록 joint와 다른 basis)")
log(f"    pa(withingene~wfamily)= {[round(float(v),3) for v in paw]} mean={paw.mean():.3f}")

log("[5] within-gene biology + domain(ndom|len) partial per axis")
feat = np.load(OUT / "feat_matrix_brain.npy")
axis = Z_wg.mean(1)                                   # (N,8)
def gmean(v):
    s = np.bincount(gidx, np.nan_to_num(v), G); c = np.bincount(gidx, (~np.isnan(v)).astype(float), G)
    return (s / np.maximum(c, 1))[gidx]
wr = lambda v: v - gmean(v)
def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def spart(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab, rac, rbc = np.corrcoef(ra, rb)[0,1], np.corrcoef(ra, rc)[0,1], np.corrcoef(rb, rc)[0,1]
    return float((rab - rac*rbc)/np.sqrt((1-rac**2)*(1-rbc**2)+1e-12))
lenr, ndomr = wr(feat[:, FEAT.index("length")]), wr(feat[:, FEAT.index("ndom")])
mm = multi & ~np.isnan(lenr) & ~np.isnan(ndomr)
res = {"evr": [float(v) for v in evr_wg], "pa_joint": [float(v) for v in paj],
       "pa_wfamily": [float(v) for v in paw], "axes": {}}
print(f"\n {'axis':>4} {'evr':>7} {'top within-feat':>26} {'r(ndom)':>8} {'r(ndom|len)':>12} {'r(len|ndom)':>12}")
for k in range(K):
    a_r = wr(axis[:, k]); mk = mm & ~np.isnan(a_r)
    cors = []
    for j, fn in enumerate(FEAT):
        f_r = wr(feat[:, j]); m2 = mk & ~np.isnan(f_r)
        if m2.sum() < 50: continue
        cors.append((fn, sp(a_r[m2], f_r[m2])))
    cors.sort(key=lambda x: -abs(x[1]))
    r_nd = sp(a_r[mk], ndomr[mk]); r_nd_l = spart(a_r[mk], ndomr[mk], lenr[mk]); r_l_nd = spart(a_r[mk], lenr[mk], ndomr[mk])
    top = ", ".join(f"{fn}({r:+.2f})" for fn, r in cors[:2])
    res["axes"][k] = {"evr": float(evr_wg[k]), "top": cors[:3], "r_ndom": r_nd,
                      "r_ndom_given_len": r_nd_l, "r_len_given_ndom": r_l_nd}
    print(f" {k:>4} {evr_wg[k]:>7.4f} {top:>26} {r_nd:>+8.2f} {r_nd_l:>+12.3f} {r_l_nd:>+12.3f}")

np.save(WF / "W_withingene_8x640.npy", W_wg)
np.save(WF / "Z_withingene_Nx30x8.npy", Z_wg)
json.dump(res, open(WF / "tierC_within_gene.json", "w"), indent=2, default=float)
log(f"[done] saved -> {WF/'tierC_within_gene.json'}")
print("\n판정: within-gene PCA top축의 r(ndom|len)이 joint axis3(+0.274)보다 크면 → 순수 within basis가 domain 축을 더 선명히 포착.")
