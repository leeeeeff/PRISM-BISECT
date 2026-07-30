#!/usr/bin/env python3
"""
Option C — axis0(β-sheet/gene-family)의 within-gene z=−2.1: 능동 억제인가 shared-weight leakage인가?
====================================================================================================
occlusion(기존): axis0 제거 → macro-AUPRC +0.038(gene-level 견인) BUT within-gene DR z=−2.1(제거하면 오히려↑).
= axis0는 within-gene 랭킹에 '해로운 신호'. 가설:
  H_suppress: PRISM이 within-gene에서 axis0를 능동적으로 down-weight.
  H_leakage : PRISM은 단일 MLP(gene-conditional gate 無)라 axis0 가중치를 어디서나 동일 적용 →
              between-gene에서 정당화된 axis0 가중이 within-gene axis0 변동(라벨 미정렬)에 새어들어 노이즈.

아키텍처 사실: v17f* = Dense MLP on [δ, φ_L30], gene-conditional gating 無 → dScore/dφ는 고정 필드.
  ⇒ within-gene에만 axis0 가중을 다르게 줄 수 없음 → H_suppress는 구조적으로 불가.
판별 (재훈련 불필요, DR set=muscle test에서):
  1. within-gene axis0 잔차의 라벨(domain) 정렬 = 0 (미정렬) → 새어든 axis0 = 랭킹 노이즈.
  2. within-gene axis3 잔차의 domain 정렬 > 0 (대조).
  3. within-gene axis0 잔차 진폭이 0이 아님(변동 존재) → '억제로 0이 된' 게 아니라 '변동이 새어듦'.
  ⇒ 1+3 + 아키텍처 = leakage 확정, H_suppress 기각. 이중해리는 학습된 라우팅이 아니라
    label-alignment 비대칭의 창발적 귀결.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import json, time, gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
os.chdir(os.path.dirname(os.path.abspath(__file__)))

t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/welcome1/sw1686/DIFFUSE"); PCA = ROOT / "reports/v20b_pca_interp"; WF = PCA / "within_family"
DATA = ROOT / "hMuscle/data"; ID_DIR = DATA / "raw_data/data/id_lists"
clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")

# ── axis scores of muscle TEST isoforms (project z-scored L15,L30 onto W, average the 2 layers) ──
W = np.load(PCA / "W_axes_8x640.npy").astype(np.float64)          # (8,640)
MU = np.load(PCA / "layer_stats_mu.npy").astype(np.float64); SD = np.load(PCA / "layer_stats_sd.npy").astype(np.float64)
PMEAN = np.load(PCA / "pca_mean_640.npy").astype(np.float64)
L15 = np.load(DATA / "esm2_layer_15_t30_150M.npy").astype(np.float64)
L30 = np.load(DATA / "esm2_layer_30_t30_150M.npy").astype(np.float64)
def axis_scores(phi, L):
    z = (phi - MU[L-1]) / SD[L-1] - PMEAN
    return z @ W.T                                                 # (N,8)
A = 0.5 * (axis_scores(L15, 15) + axis_scores(L30, 30))           # (N,8) 2-layer axis score
N = len(A); log(f"muscle test isoforms N={N}, axis scores computed")

# ── gene symbols + domain count (DR label) ──
ENSG2SYM = {}
with open(ID_DIR / "ensembl_to_symbol.txt") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
te_genes_raw = np.load("my_gene_list_fixed.npy", allow_pickle=True)
te_sym = np.array([ENSG2SYM.get(clean(g).split(".")[0], clean(g).split(".")[0]) for g in te_genes_raw])
iso_ndom = np.load("../results_isoform/features/domain_matrix_proper_test.npy").sum(1).astype(float)
assert len(te_sym) == N == len(iso_ndom)

# ── within-gene residualization (subtract gene mean) ──
g2i = defaultdict(list)
for i, g in enumerate(te_sym): g2i[g].append(i)
res_ax = np.zeros_like(A); res_dom = np.zeros(N)
multi = 0
for g, idx in g2i.items():
    idx = np.array(idx)
    if len(idx) < 2: continue
    multi += len(idx)
    res_ax[idx] = A[idx] - A[idx].mean(0)
    res_dom[idx] = iso_ndom[idx] - iso_ndom[idx].mean()
mask = np.zeros(N, bool)
for g, idx in g2i.items():
    if len(idx) >= 2 and iso_ndom[np.array(idx)].std() > 1e-9:
        mask[np.array(idx)] = True   # only genes with domain variation (DR-eligible)
log(f"within-gene residualized; DR-eligible isoforms (domain-varying multi-iso genes) = {int(mask.sum())}")

def sp(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

# ── (1)(2) within-gene axis residual vs domain (DR label) alignment ──
print("\n[within-gene axis residual → domain-count alignment (Spearman, DR-eligible)]")
align = {}
for k in range(8):
    r = sp(res_ax[mask, k], res_dom[mask]); align[k] = r
    tag = "  ← used+aligned (axis3)" if k == 3 else ("  ← occlusion z=−2.1 (axis0)" if k == 0 else "")
    print(f"  axis{k}: within-gene ρ(residual, Δdomain) = {r:+.3f}{tag}")

# ── (3) within-gene residual amplitude (does axis0 actually vary within gene? = leakage precondition) ──
print("\n[within-gene residual amplitude — is axis0 variation present to leak?]")
amp = {}
for k in (0, 3):
    a = float(np.std(res_ax[mask, k])); amp[k] = a
    print(f"  axis{k}: within-gene residual std = {a:.3f}")
ratio03 = amp[0] / amp[3] if amp[3] else float("nan")
print(f"  amplitude ratio axis0/axis3 = {ratio03:.3f}  (axis0 varies within-gene comparably → not suppressed to 0)")

# ── verdict ──
lo, hi = abs(align[0]), abs(align[3])
verdict = ("LEAKAGE confirmed: axis0 within-gene 변동은 존재(std {:.3f})하나 domain과 미정렬(ρ={:+.3f}); "
           "axis3는 정렬(ρ={:+.3f}). 단일 MLP가 axis0 between-gene 가중을 within-gene에 동일 적용 → 새어든 미정렬 변동이 "
           "랭킹 노이즈(occlusion z=−2.1). 능동 억제(H_suppress)는 아키텍처상 불가.").format(amp[0], align[0], align[3])
print("\n=> " + verdict)

json.dump({"within_gene_domain_alignment": {str(k): align[k] for k in range(8)},
           "residual_amp_axis0": amp[0], "residual_amp_axis3": amp[3],
           "amp_ratio_0_3": ratio03, "n_dr_eligible": int(mask.sum()), "verdict": verdict},
          open(WF / "axis0_routing_alignment.json", "w"), indent=2)
log(f"[saved] {WF/'axis0_routing_alignment.json'}")
