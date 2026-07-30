#!/usr/bin/env python3
"""
interp_within_family_followup.py  —  Option A (isoform-level biology) + B (multi-gene restrict)
==============================================================================================
사용자 비판 2건 반영:
 [1] 레벨 혼동: within-family 축의 biology를 gene-mean으로 잰 것은 "β-sheet=between-family"를
     증명하지 못한다(22feat 자체가 gene-level이라 family 제거 시 동반 붕괴 가능). 정확한 검정은
     **isoform-level(within-gene 잔차)** 상관 — β-sheet/size가 within-gene에서도 변하는가?
 [2] Pfam-family 커버리지 함정: isoform 51%가 무도메인 singleton(=gene-centering으로 퇴화).
     genuine multi-gene family만 restrict해 singleton-inflate 제거 후 within_frac이 null을 넘나?

저장된 Z_within(reports/v20b_pca_interp/within_family/Z_within_Nx30x8.npy)로 재계산 — 무거운 PCA 재실행 없음.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
import interp_within_family_pca as M  # reuse gene_to_family()

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT = ROOT / "reports/v20b_pca_interp"
WF = OUT / "within_family"
K = 8
FEAT = M.FEAT_NAMES

Z = np.load(WF / "Z_within_Nx30x8.npy")          # (N,30,8) family-centered axis scores
axis = Z.mean(1)                                  # (N,8) layer-mean per isoform
feat = np.load(OUT / "feat_matrix_brain.npy")     # (N,22) per isoform
sym, fam = M.gene_to_family()
N = len(sym)

# vectorized gene-mean via factorize
gene_lab, gidx = np.unique(sym, return_inverse=True); G = len(gene_lab)
gcount = np.bincount(gidx, minlength=G)
def gene_mean(v):
    s = np.bincount(gidx, weights=np.nan_to_num(v), minlength=G)
    c = np.bincount(gidx, weights=(~np.isnan(v)).astype(float), minlength=G)
    return (s / np.maximum(c, 1))[gidx]
def within_gene_resid(v):
    return v - gene_mean(v)

multi_iso = gcount[gidx] >= 2      # isoform belongs to a >=2-isoform gene

# family gene-composition -> multi-gene families
fam_genes = defaultdict(set)
for i in range(N): fam_genes[fam[i]].add(sym[i])
is_multigene_fam = np.array([len(fam_genes[fam[i]]) >= 2 for i in range(N)])

print("=" * 78)
print(" OPTION A — isoform-level (within-gene residual) biology of within-family axes")
print("=" * 78)
print(" 각 축: gene-mean ρ (between-gene, family 포함) vs within-gene-resid ρ (순수 within-gene)")
print(" within-gene 잔차 상관은 multi-iso isoform에서만 정의됨\n")

resA = {}
mi = multi_iso
for k in range(K):
    a_gm = gene_mean(axis[:, k])
    a_wr = within_gene_resid(axis[:, k])
    row_gm, row_wr = [], []
    for j, fn in enumerate(FEAT):
        f_gm = gene_mean(feat[:, j]); f_wr = within_gene_resid(feat[:, j])
        m1 = ~np.isnan(a_gm) & ~np.isnan(f_gm)
        r_gm = stats.spearmanr(a_gm[m1], f_gm[m1])[0]
        m2 = mi & ~np.isnan(a_wr) & ~np.isnan(f_wr)
        r_wr = stats.spearmanr(a_wr[m2], f_wr[m2])[0]
        row_gm.append((fn, float(r_gm))); row_wr.append((fn, float(r_wr)))
    row_gm.sort(key=lambda x: -abs(x[1])); row_wr.sort(key=lambda x: -abs(x[1]))
    resA[k] = {"gene_mean_top3": row_gm[:3], "within_resid_top3": row_wr[:3],
               "max_abs_gene_mean": abs(row_gm[0][1]), "max_abs_within_resid": abs(row_wr[0][1])}
    print(f" axis {k}:")
    print(f"    gene-mean  : " + ", ".join(f"{fn}({r:+.2f})" for fn, r in row_gm[:3]))
    print(f"    within-gene: " + ", ".join(f"{fn}({r:+.2f})" for fn, r in row_wr[:3]))

print("\n 요약: max|ρ| gene-mean vs within-gene-resid (축별)")
for k in range(K):
    print(f"    axis {k}: gene-mean {resA[k]['max_abs_gene_mean']:.2f}  |  within-gene {resA[k]['max_abs_within_resid']:.2f}")

print("\n" + "=" * 78)
print(" OPTION B — multi-gene-family restricted within_frac (singleton-inflate 제거)")
print("=" * 78)

def within_frac_masked(v, mask):
    x = v[mask]; g = gidx[mask]
    grand = x.mean(); ss_tot = float(((x - grand) ** 2).sum())
    idx = defaultdict(list)
    for i, gg in enumerate(g): idx[gg].append(i)
    ss_w = sum(float(((x[np.array(ii)] - x[np.array(ii)].mean()) ** 2).sum()) for ii in idx.values())
    G_sub = len(idx); N_sub = len(x)
    null = (N_sub - G_sub) / (N_sub - 1) if N_sub > 1 else np.nan
    return (ss_w / ss_tot if ss_tot > 0 else np.nan), null, N_sub, G_sub

subset = multi_iso & is_multigene_fam         # genuine family control AND >=2 iso
singleton = multi_iso & ~is_multigene_fam     # self-centered (gene-centering degenerate)
print(f" subsets: all-multiiso n={int(multi_iso.sum())}  |  multigene-family n={int(subset.sum())}  "
      f"|  singleton-family n={int(singleton.sum())}")
print(f"\n {'axis':>4} {'wf_all':>8} {'wf_multigene':>13} {'wf_singleton':>13} {'analytic_null(mg)':>18}")
resB = {}
for k in range(K):
    wf_all, null_all, _, _ = within_frac_masked(axis[:, k], multi_iso)
    wf_mg, null_mg, nmg, gmg = within_frac_masked(axis[:, k], subset)
    wf_sg, _, _, _ = within_frac_masked(axis[:, k], singleton)
    resB[k] = {"wf_all": wf_all, "wf_multigene": wf_mg, "wf_singleton": wf_sg,
               "analytic_null_multigene": null_mg, "n_multigene": nmg, "g_multigene": gmg}
    print(f" {k:>4} {wf_all:>8.3f} {wf_mg:>13.3f} {wf_sg:>13.3f} {null_mg:>18.3f}")
print("\n 해석: wf_singleton은 gene-centering 퇴화로 1.0에 근접(inflate 확인).")
print("       wf_multigene가 analytic_null(mg)보다 '낮으면' family 제거 후에도 between-paralog 구조가 남음")
print("       (=within-gene만이 아님). null과 같으면 paralog 구조 소멸(순수 within).")

with open(WF / "followup_A_B.json", "w") as f:
    json.dump({"optionA_biology": resA, "optionB_within_frac": resB,
               "n_multiiso": int(multi_iso.sum()), "n_multigene_fam_iso": int(subset.sum()),
               "n_singleton_fam_iso": int(singleton.sum())}, f, indent=2)
print(f"\n[saved] {WF/'followup_A_B.json'}")
