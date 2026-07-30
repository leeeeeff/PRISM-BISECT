#!/usr/bin/env python3
"""
Option B — proper gene-family (diamond 서열 homology 클러스터) 로 within-family PCA 재실행
=========================================================================================
사용자 지적 2 완전대응: Pfam-dominant-family는 isoform 51%를 무도메인 singleton으로 떨궈
family-control이 gene-centering으로 퇴화. → 모든 gene을 서열 homology로 클러스터링(diamond,
approx-id 30 / cover 50)해 proper gene-family 지정 후, geometry 보존(cos)·within_frac 상승이
family 정의에 robust한지 검증.

비교: seqfam-centered W vs (i) brain-fit joint (ii) 원본 muscle W (iii) Pfam-family W_within.
within_frac: all-multiiso + genuine multigene subset (+ analytic null).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "6"); os.environ.setdefault("MKL_NUM_THREADS", "6")
import json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT = ROOT / "reports/v20b_pca_interp"; WF = OUT / "within_family"
K, N_LAYERS, EMB = 8, 30, 640
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def seqfam_mapping(sym):
    """gene -> family from diamond clusters; no-seq gene -> NOSEQ::gene singleton."""
    g2fam = {}
    for line in open(WF / "gene_clusters.tsv"):
        c, m = line.rstrip("\n").split("\t"); g2fam[m] = c
    return np.array([g2fam.get(g, f"NOSEQ::{g}") for g in sym])


def within_frac_masked(v, gidx, mask):
    x = v[mask]; g = gidx[mask]; grand = x.mean()
    ss_tot = float(((x - grand) ** 2).sum()); idx = defaultdict(list)
    for i, gg in enumerate(g): idx[gg].append(i)
    ss_w = sum(float(((x[np.array(ii)] - x[np.array(ii)].mean()) ** 2).sum()) for ii in idx.values())
    Gs, Ns = len(idx), len(x)
    return (ss_w / ss_tot if ss_tot > 0 else np.nan), ((Ns - Gs) / (Ns - 1) if Ns > 1 else np.nan)


def pa(Wa, Wb): return np.clip(np.linalg.svd(Wa @ Wb.T, compute_uv=False), 0, 1)


def main():
    log("[1] brain traj (muscle z-score)")
    traj, sym = M.load_brain_traj_normalized(); N = len(sym)
    fam = seqfam_mapping(sym)
    fams, fam_idx = np.unique(fam, return_inverse=True); F = len(fams)
    _, gidx = np.unique(sym, return_inverse=True); G = gidx.max() + 1
    gcount = np.bincount(gidx, minlength=G); multi = gcount[gidx] >= 2
    fam_genes = defaultdict(set)
    for i in range(N): fam_genes[fam[i]].add(sym[i])
    is_mgf = np.array([len(fam_genes[fam[i]]) >= 2 for i in range(N)])
    n_iso_mgf = int(is_mgf.sum())
    log(f"    N={N} genes={G} seq-families={F} | isoforms in multigene-fam={n_iso_mgf} ({100*n_iso_mgf/N:.1f}%) "
        f"(cf Pfam 49.0%) | multi-iso={int(multi.sum())}")

    log("[2] CONTROL brain-fit joint PCA(8)")
    W_joint, _, Z_joint = M.fit_pca(traj.reshape(N * N_LAYERS, EMB))

    log("[3] TREATMENT seqfam-center -> PCA(8)")
    M.family_center_inplace(traj, fam_idx, F)
    W_sf, evr_sf, Z_sf = M.fit_pca(traj.reshape(N * N_LAYERS, EMB)); del traj

    log("[4] geometry vs joint / original / Pfam-family")
    W_orig = np.load(OUT / "W_axes_8x640.npy"); W_pfam = np.load(WF / "W_within_8x640.npy")
    pa_j, pa_o, pa_p = pa(W_sf, W_joint), pa(W_sf, W_orig), pa(W_sf, W_pfam)
    log(f"    pa(seqfam~joint) = {[round(float(v),3) for v in pa_j]} mean={pa_j.mean():.3f}")
    log(f"    pa(seqfam~orig ) = {[round(float(v),3) for v in pa_o]} mean={pa_o.mean():.3f}")
    log(f"    pa(seqfam~Pfam ) = {[round(float(v),3) for v in pa_p]} mean={pa_p.mean():.3f}  (두 family 정의 일치도)")

    log("[5] within_frac (seqfam axes): all-multiiso | multigene-fam | singleton | analytic-null(mg)")
    ax_j = Z_joint.mean(1); ax_s = Z_sf.mean(1)
    sub = multi & is_mgf; sgl = multi & ~is_mgf
    res_wf = {}
    log(f"    {'axis':>4} {'joint(all)':>11} {'seqfam(all)':>12} {'seqfam(mg)':>11} {'seqfam(sgl)':>12} {'null(mg)':>9}")
    for k in range(K):
        wj, _ = within_frac_masked(ax_j[:, k], gidx, multi)
        wa, _ = within_frac_masked(ax_s[:, k], gidx, multi)
        wm, nullm = within_frac_masked(ax_s[:, k], gidx, sub)
        ws, _ = within_frac_masked(ax_s[:, k], gidx, sgl)
        res_wf[k] = dict(joint_all=wj, seqfam_all=wa, seqfam_mg=wm, seqfam_sgl=ws, null_mg=nullm)
        log(f"    {k:>4} {wj:>11.3f} {wa:>12.3f} {wm:>11.3f} {ws:>12.3f} {nullm:>9.3f}")

    np.save(WF / "W_seqfam_8x640.npy", W_sf)
    with open(WF / "optB_seqfam.json", "w") as f:
        json.dump({"n_iso_multigene_fam": n_iso_mgf, "frac": n_iso_mgf / N, "n_seq_families": int(F),
                   "pa_seqfam_joint": [float(v) for v in pa_j],
                   "pa_seqfam_orig": [float(v) for v in pa_o],
                   "pa_seqfam_pfam": [float(v) for v in pa_p], "within_frac": res_wf}, f, indent=2)
    log(f"[done] saved -> {WF/'optB_seqfam.json'}")
    log("해석: pa(seqfam~joint)≈0.95면 geometry 보존이 family 정의에 robust(Pfam과 무관하게 같은 축).")
    log("      seqfam(mg) within_frac이 joint(all)보다 높고 null(mg)보다 낮으면 → 상승 재현 + between-paralog 잔존.")


if __name__ == "__main__":
    main()
