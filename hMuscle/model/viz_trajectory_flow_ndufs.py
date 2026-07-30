#!/usr/bin/env python3
"""
Option B — trajectory-flow 시각화 (isoform identity descriptor)
=============================================================
encoded-but-not-predictively-used 축과 used 축의 L1~L30 궤적을 isoform별로 그려,
같은 유전자 isoform이 네트워크 어디서 갈라지는가를 기술(description, 예측 아님).

패널: gene(NDUFS2 domain-split / NDUFS7 1-domain / NDUFS4 no-domain 음성대조) × 축(axis3 domain-used, axis5 length-encoded-only).
- domain-bearing(PF) vs domain-lost isoform 색 구분.
- axis3에서 domain+/− 가 갈라지고 그 분기 layer를 보이면 = 표현이 도메인차를 네트워크 중간에서 인코딩.
- axis5(length)는 갈라지나 예측엔 안 씀 → identity 기술 자산.
"""
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
OUT = ROOT / "reports/v20b_pca_interp/within_family"
Z = np.load(ROOT / "reports/v20b_pca_interp/Z_brain_Nx30x8.npy")  # (N,30,8) joint axes
sym, _ = M.gene_to_family()
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])

# domain sets
name2enst = {}
for line in open(GTF):
    if "\ttranscript\t" not in line: continue
    em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
    if em and nm: name2enst[nm.group(1)] = em.group(1).split(".")[0]
enst2dom = defaultdict(set)
for line in open(HMM):
    if line.startswith("#") or not line.strip(): continue
    p = line.split()
    if float(p[12]) > 1e-5: continue
    enst2dom[p[3].split(".p")[0].split(".")[0]].add(p[1].split(".")[0])
def ndom(i):
    b = bids[i]
    e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else name2enst.get(b, ""))
    return len(enst2dom.get(e, set()))

OK_BLUE, OK_ORANGE, OK_GREY = "#0072B2", "#D55E00", "#999999"
GENES = ["NDUFS2", "NDUFS7", "NDUFS4"]
AXES = [(3, "axis 3 — domain-architecture (prediction-USED)"),
        (5, "axis 5 — length (encoded-ONLY)")]
L = np.arange(1, 31)
fig, axg = plt.subplots(len(GENES), len(AXES), figsize=(11, 10), sharex=True)
for gi, g in enumerate(GENES):
    idx = [i for i in range(len(sym)) if sym[i] == g]
    doms = np.array([ndom(i) for i in idx])
    for ai, (k, title) in enumerate(AXES):
        ax = axg[gi, ai]
        for i, d in zip(idx, doms):
            col = OK_ORANGE if d >= 1 else OK_BLUE
            ax.plot(L, Z[i, :, k], color=col, alpha=0.55, lw=1.1)
        # mean trajectories by domain status
        dm = doms >= 1
        if dm.sum() > 0:
            ax.plot(L, Z[np.array(idx)[dm]][:, :, k].mean(0), color=OK_ORANGE, lw=2.8, label=f"domain+ (n={int(dm.sum())})")
        if (~dm).sum() > 0:
            ax.plot(L, Z[np.array(idx)[~dm]][:, :, k].mean(0), color=OK_BLUE, lw=2.8, label=f"domain− (n={int((~dm).sum())})")
        ax.axvline(17, color=OK_GREY, ls=":", lw=1, alpha=0.7)  # mid-layer divergence peak
        if gi == 0: ax.set_title(title, fontsize=10)
        if ai == 0: ax.set_ylabel(f"{g}\naxis score", fontsize=10)
        if gi == len(GENES) - 1: ax.set_xlabel("ESM-2 layer (L1→L30)", fontsize=9)
        ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.2)
fig.suptitle("Per-isoform axis trajectories (L1–L30): where same-gene isoforms diverge\n"
             "orange=domain-bearing, blue=domain-lost; dotted=mid-layer(L17) divergence peak", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outpng = OUT / "fig_trajectory_flow_ndufs.png"
fig.savefig(outpng, dpi=140); fig.savefig(OUT / "fig_trajectory_flow_ndufs.pdf")
print(f"[saved] {outpng}")

# quantify divergence: |mean(domain+) - mean(domain-)| per layer, per gene, axis3 vs axis5
print("\n per-layer domain+/− separation (axis3 vs axis5), peak layer:")
for g in GENES:
    idx = np.array([i for i in range(len(sym)) if sym[i] == g])
    dm = np.array([ndom(i) for i in idx]) >= 1
    if dm.sum() == 0 or (~dm).sum() == 0:
        print(f"  {g}: no domain variation (all domain−) — negative control"); continue
    for k in (3, 5):
        sep = np.abs(Z[idx[dm]][:, :, k].mean(0) - Z[idx[~dm]][:, :, k].mean(0))
        print(f"  {g} axis{k}: peak sep={sep.max():.3f} at L{int(sep.argmax()+1)}  (L30 sep={sep[-1]:.3f})")
