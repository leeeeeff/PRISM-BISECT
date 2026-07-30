#!/usr/bin/env python3
"""
decompose_variance_hierarchy.py — 계층적 분산 분해 (taxonomy 실증)
================================================================
사용자 질문: family→gene→isoform 이 분류할 수 있는 최대의 합리적 분류의 끝인가?
             within-gene 변산 중 family/gene와 '함께 움직이는' 것 vs '완전 독립' 것을 가른다.

계층 nested ANOVA (joint PCA 8축 각각, layer-mean 축점수):
  SS_total = SS_between_family + SS_between_gene|family + SS_within_gene   (family⊃gene⊃isoform)
  → 이 3분할이 nesting 상 완전분해. "between-gene 무관 family" = between_family+between_gene|family.

within-gene 공동이동 vs 독립 (8-dim 다변량):
  between-gene 주방향 V_k = gene-mean 벡터 공분산의 top-k 고유벡터.
  within-gene 잔차 r_i = z_i − gene_mean. co-moving = V_k 부분공간에 실린 within-gene 분산,
  independent = 직교여공간. → within-gene이 gene 구조와 같은 축이면 co-moving(중복 정체성),
  직교면 independent(순수 isoform 특이).
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
import interp_within_family_pca as M

P = Path("/home/welcome1/sw1686/DIFFUSE/reports/v20b_pca_interp")
Zj = np.load(P / "Z_brain_Nx30x8.npy").mean(1)        # (N,8) joint layer-mean
sym, fam = M.gene_to_family(); N = len(sym); K = 8
gl, gidx = np.unique(sym, return_inverse=True); G = len(gl)
fl, fidx = np.unique(fam, return_inverse=True); F = len(fl)
# gene -> family map
gene_fam = np.zeros(G, int)
for i in range(N): gene_fam[gidx[i]] = fidx[i]

print("=" * 76)
print(" NESTED ANOVA per joint axis: family ⊃ gene ⊃ isoform")
print("=" * 76)
print(f" N={N} isoforms, G={G} genes, F={F} families")
print(f"\n {'axis':>4} {'betw_family':>11} {'betw_gene|fam':>13} {'within_gene':>12} {'(bg_total)':>11}")
res = {}
for k in range(K):
    x = Zj[:, k]; grand = x.mean()
    ss_tot = float(((x - grand) ** 2).sum())
    # gene means / counts
    gsum = np.bincount(gidx, x, G); gcnt = np.bincount(gidx, None, G).astype(float)
    gmean = gsum / np.maximum(gcnt, 1)
    # family means / counts
    fsum = np.bincount(fidx, x, F); fcnt = np.bincount(fidx, None, F).astype(float)
    fmean = fsum / np.maximum(fcnt, 1)
    ss_bf = float((fcnt * (fmean - grand) ** 2).sum())
    ss_bg_wf = float((gcnt * (gmean - fmean[gene_fam]) ** 2).sum())
    ss_wg = float(((x - gmean[gidx]) ** 2).sum())
    fbf, fbg, fwg = ss_bf / ss_tot, ss_bg_wf / ss_tot, ss_wg / ss_tot
    res[k] = dict(betw_family=fbf, betw_gene_within_family=fbg, within_gene=fwg,
                  between_gene_total=fbf + fbg)
    print(f" {k:>4} {fbf:>11.3f} {fbg:>13.3f} {fwg:>12.3f} {fbf+fbg:>11.3f}")
# pooled (sum over axes weighted by axis variance)
tot_bf = np.mean([res[k]['betw_family'] for k in range(K)])
tot_bg = np.mean([res[k]['betw_gene_within_family'] for k in range(K)])
tot_wg = np.mean([res[k]['within_gene'] for k in range(K)])
print(f"\n mean over 8 axes: between-family={tot_bf:.3f}  between-gene|family={tot_bg:.3f}  within-gene={tot_wg:.3f}")
print(" → family⊃gene⊃isoform 3분할이 nesting 상 완전분해(합=1.0). '순수 within-gene'=within_gene 열.")

print("\n" + "=" * 76)
print(" WITHIN-GENE: co-moving (gene 구조와 같은 축) vs independent (직교) — 8-dim")
print("=" * 76)
# between-gene covariance (gene-mean vectors, weighted by gene size)
Gm = np.zeros((G, K))
for k in range(K):
    Gm[:, k] = np.bincount(gidx, Zj[:, k], G) / np.maximum(np.bincount(gidx, None, G), 1)
w = np.bincount(gidx, None, G).astype(float)
Gc = Gm - np.average(Gm, 0, weights=w)
C_bg = (Gc * w[:, None]).T @ Gc / w.sum()
# within-gene residual covariance
R = Zj - Gm[gidx]
C_wg = R.T @ R / len(R)
# eigen of between-gene cov
evals, evecs = np.linalg.eigh(C_bg); order = np.argsort(-evals); evecs = evecs[:, order]; evals = evals[order]
tr_wg = float(np.trace(C_wg))
print(f" total within-gene variance (8-dim trace) = {tr_wg:.4f}")
print(f"\n {'top-k BG subspace':>18} {'co-moving frac':>15} {'independent frac':>17}")
comov = {}
for kk in range(1, K):
    Vk = evecs[:, :kk]
    cov = float(np.trace(Vk.T @ C_wg @ Vk))
    comov[kk] = cov / tr_wg
    print(f" {kk:>18} {cov/tr_wg:>15.3f} {1-cov/tr_wg:>17.3f}")
print("\n 해석: top-1 BG방향에 within-gene의 X%가 실림=그만큼 within이 gene차이 축과 '같은 방향'(co-moving,")
print("       주로 length/domain gradation). 나머지=순수 isoform-특이(independent, gene구조와 직교).")

import json
json.dump({"nested_anova": res, "mean_between_family": tot_bf, "mean_between_gene_within_family": tot_bg,
           "mean_within_gene": tot_wg, "within_gene_comoving_frac": comov,
           "bg_eigenvalues": [float(v) for v in evals]},
          open(P / "within_family/variance_hierarchy.json", "w"), indent=2)
print(f"\n[saved] {P/'within_family/variance_hierarchy.json'}")
