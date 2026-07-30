#!/usr/bin/env python3
"""
Option A — within-gene axis: length vs domain-architecture 분리
==============================================================
within-family 축의 within-gene-residual 신호(axis5 length−.56 & ndom−.45 등)가 순수 length
confound인지, length 통제 후에도 domain-architecture(ndom) 신호가 남는지 분리한다.

방법 (전부 within-gene 잔차 = isoform − gene-mean, multi-iso isoform):
  1) Spearman partial:  r(axis, ndom | length)  및  r(axis, length | ndom)
  2) length-matched binning: within-gene length-잔차 decile 안에서 r(axis, ndom) 평균
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT = ROOT / "reports/v20b_pca_interp"; WF = OUT / "within_family"
K = 8; FEAT = M.FEAT_NAMES
LEN_J = FEAT.index("length"); NDOM_J = FEAT.index("ndom")

Z = np.load(WF / "Z_within_Nx30x8.npy"); axis = Z.mean(1)      # (N,8)
feat = np.load(OUT / "feat_matrix_brain.npy")                  # (N,22)
sym, _ = M.gene_to_family(); N = len(sym)
_, gidx = np.unique(sym, return_inverse=True); G = gidx.max() + 1
gcount = np.bincount(gidx, minlength=G)
def gene_mean(v):
    s = np.bincount(gidx, weights=np.nan_to_num(v), minlength=G)
    c = np.bincount(gidx, weights=(~np.isnan(v)).astype(float), minlength=G)
    return (s / np.maximum(c, 1))[gidx]
wr = lambda v: v - gene_mean(v)                                # within-gene residual
multi = gcount[gidx] >= 2

def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def spartial(a, b, c):   # r(a,b | c), spearman-based
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab = np.corrcoef(ra, rb)[0, 1]; rac = np.corrcoef(ra, rc)[0, 1]; rbc = np.corrcoef(rb, rc)[0, 1]
    return float((rab - rac * rbc) / np.sqrt((1 - rac**2) * (1 - rbc**2) + 1e-12))

length_r = wr(feat[:, LEN_J]); ndom_r = wr(feat[:, NDOM_J])
m = multi & ~np.isnan(length_r) & ~np.isnan(ndom_r)
print(f"multi-iso isoforms n={int(m.sum())}")
print(f"within-gene corr(length_resid, ndom_resid) = {sp(length_r[m], ndom_r[m]):+.3f} (공선성 정도)\n")

print(f"{'axis':>4} {'r(ax,len)':>10} {'r(ax,ndom)':>11} {'r(ax,ndom|len)':>15} {'r(ax,len|ndom)':>15} {'lenmatched r(ax,ndom)':>22}")
res = {}
for k in range(K):
    a_r = wr(axis[:, k]); mm = m & ~np.isnan(a_r)
    A, Ln, Nd = a_r[mm], length_r[mm], ndom_r[mm]
    r_al = sp(A, Ln); r_an = sp(A, Nd)
    r_an_l = spartial(A, Nd, Ln)   # domain signal controlling length
    r_al_n = spartial(A, Ln, Nd)   # length signal controlling domain
    # length-matched: bin by length residual decile, corr(axis, ndom) within bin
    dec = np.digitize(Ln, np.quantile(Ln, np.linspace(0.1, 0.9, 9)))
    rs = []
    for d in range(10):
        idx = dec == d
        if idx.sum() > 30 and np.std(Nd[idx]) > 0 and np.std(A[idx]) > 0:
            rs.append(sp(A[idx], Nd[idx]))
    lm = float(np.mean(rs)) if rs else np.nan
    res[k] = dict(r_ax_len=r_al, r_ax_ndom=r_an, r_ax_ndom_given_len=r_an_l,
                  r_ax_len_given_ndom=r_al_n, lenmatched_r_ax_ndom=lm)
    print(f"{k:>4} {r_al:>10.3f} {r_an:>11.3f} {r_an_l:>15.3f} {r_al_n:>15.3f} {lm:>22.3f}")

with open(WF / "optA_length_confound.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"\n[saved] {WF/'optA_length_confound.json'}")
print("\n해석: r(ax,ndom|len)이 0에 붕괴하면 domain 신호=length confound. 유지되면 length-독립 도메인 신호.")
print("      length-matched r(ax,ndom)이 raw r(ax,ndom)와 비슷하게 유지되면 confound 아님(교차검증).")
