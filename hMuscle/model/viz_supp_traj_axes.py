#!/usr/bin/env python3
"""
Supplementary figures — S_traj (per-isoform axis trajectories) + S_axes (encoding-vs-usage)
===========================================================================================
S_traj : NDUFS2 / NDUFS7 / NDUFS4 / NDUFAF5 의 L1~L30 axis3(domain, USED) & axis5(length-like,
         encoded-only) 궤적. 같은 유전자 isoform이 네트워크 어디서 갈라지는가(description).
S_axes : 8축 encoding-vs-usage 2D 지도. x=within-gene encoding(within_frac),
         y=within-gene predictive usage(z_DR, Domain-Ranking occlusion), 색=gene-level usage(ΔMacro).
         → axis3=encoded+used(우상단), axis0=gene-level-used but within-gene-unused, 나머지 encoded-only.
figures_v2 규약(Okabe-Ito, PDF+PNG). 참조: reports/natcomm_v0.md §4c.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
P = ROOT / "reports/v20b_pca_interp"
OUTV2 = ROOT / "reports/figures_v2"
Z = np.load(P / "Z_brain_Nx30x8.npy")            # (N,30,8) joint axes
sym, _ = M.gene_to_family()
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])
dossier = json.load(open(P / "within_family/axis_dossier.json"))

# Okabe-Ito
OK_BLUE, OK_ORANGE, OK_GREY, OK_GREEN, OK_VERM = "#0072B2", "#E69F00", "#999999", "#009E73", "#D55E00"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# ---- domain sets ----
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
    b = bids[i]; e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else name2enst.get(b, ""))
    return len(enst2dom.get(e, set()))

# =================================================================== S_traj
GENES = ["NDUFS2", "NDUFS7", "NDUFS4", "NDUFAF5"]
AXES = [(3, "axis 3 — domain architecture (prediction-USED)"),
        (5, "axis 5 — Pro-turn/length-like (encoded-ONLY)")]
L = np.arange(1, 31)
fig, axg = plt.subplots(len(GENES), len(AXES), figsize=(10, 12), sharex=True)
for gi, g in enumerate(GENES):
    idx = np.array([i for i in range(len(sym)) if sym[i] == g])
    doms = np.array([ndom(i) for i in idx])
    dm = doms >= 1
    for ai, (k, title) in enumerate(AXES):
        ax = axg[gi, ai]
        for i, d in zip(idx, doms):
            ax.plot(L, Z[i, :, k], color=(OK_ORANGE if d >= 1 else OK_BLUE), alpha=0.5, lw=1.0)
        if dm.sum() > 0:
            ax.plot(L, Z[idx[dm]][:, :, k].mean(0), color=OK_ORANGE, lw=2.8)
        if (~dm).sum() > 0:
            ax.plot(L, Z[idx[~dm]][:, :, k].mean(0), color=OK_BLUE, lw=2.8)
        ax.axvline(17, color=OK_GREY, ls=":", lw=1, alpha=0.7)
        if gi == 0: ax.set_title(title, fontsize=9)
        if ai == 0:
            ax.set_ylabel(f"{g}  (n={len(idx)})\naxis score", fontsize=9)
        if gi == len(GENES) - 1: ax.set_xlabel("ESM-2 layer (L1 → L30)", fontsize=9)
        ax.grid(alpha=0.15)
leg = [Line2D([0], [0], color=OK_ORANGE, lw=2.8, label="domain-bearing isoform mean"),
       Line2D([0], [0], color=OK_BLUE, lw=2.8, label="domain-lost isoform mean"),
       Line2D([0], [0], color=OK_GREY, ls=":", lw=1, label="mid-layer (L17) divergence peak")]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Supplementary Fig. S_traj — per-isoform ESM-2 axis trajectories\n"
             "same-gene isoforms diverge on the domain axis (axis 3) at mid-network; NDUFS4/NDUFAF5 = no-domain negative controls",
             fontsize=10)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUTV2 / "S_traj.png", dpi=160); fig.savefig(OUTV2 / "S_traj.pdf")
print(f"[saved] {OUTV2/'S_traj.pdf'}")
plt.close(fig)

# =================================================================== S_axes
names = [dossier[str(k)]["name"] for k in range(8)]
wf = np.array([dossier[str(k)]["within_frac"] for k in range(8)])      # encoding (within-gene variance frac)
zdr = np.array([dossier[str(k)]["z_DR"] for k in range(8)])            # within-gene usage
dmac = np.array([dossier[str(k)]["dMacro"] for k in range(8)])         # gene-level usage
fig2, ax = plt.subplots(figsize=(8.2, 6.2))
sc = ax.scatter(wf, zdr, c=dmac, s=260, cmap="viridis", edgecolor="k", linewidth=0.8, zorder=3)
ax.axhline(0, color=OK_GREY, lw=0.8, ls="--")
ax.axhline(2, color=OK_VERM, lw=0.8, ls=":", alpha=0.7)
ax.text(wf.min(), 2.15, "usage significance (|z|≈2)", color=OK_VERM, fontsize=7)
# per-axis label offsets (dx, dy, ha) to avoid overlaps / edge & colorbar clipping
LOFF = {0: (0.001, -0.55, "left"), 1: (0.001, 0.40, "left"), 2: (0.001, 0.42, "left"),
        3: (-0.001, -0.55, "right"), 4: (0.001, 0.40, "left"), 5: (-0.001, 0.42, "right"),
        6: (-0.001, -0.55, "right"), 7: (0.001, 0.40, "left")}
for k in range(8):
    dx, dy, ha = LOFF[k]
    ax.annotate(f"ax{k}: {names[k]}", (wf[k], zdr[k]), (wf[k]+dx, zdr[k]+dy),
                fontsize=7.5, ha=ha, zorder=4)
ax.set_xlim(wf.min()-0.012, wf.max()+0.012)
ax.set_ylim(zdr.min()-1.2, zdr.max()+1.2)
ax.set_xlabel("within-gene ENCODING  (isoform variance fraction, within_frac)")
ax.set_ylabel("within-gene USAGE  (Domain-Ranking occlusion z)")
cb = fig2.colorbar(sc, ax=ax); cb.set_label("gene-level usage  (ΔMacro-AUPRC when occluded)", fontsize=8)
ax.set_title("Supplementary Fig. S_axes — encoding vs prediction-usage of the 8 ESM-2 axes\n"
             "axis 3 (domain) = encoded AND used within-gene; axis 0 (β-sheet) = gene-level driver, within-gene unused",
             fontsize=9)
# quadrant guides
ax.text(0.60, 0.97, "encoded + used", transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color=OK_GREEN, weight="bold")
ax.text(0.99, 0.03, "encoded-only / anti-useful", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=OK_GREY, weight="bold")
fig2.tight_layout()
fig2.savefig(OUTV2 / "S_axes.png", dpi=160); fig2.savefig(OUTV2 / "S_axes.pdf")
print(f"[saved] {OUTV2/'S_axes.pdf'}")

# quantify S_traj divergence for caption
print("\nS_traj per-gene axis3 vs axis5 domain+/- separation peak:")
for g in GENES:
    idx = np.array([i for i in range(len(sym)) if sym[i] == g])
    dm = np.array([ndom(i) for i in idx]) >= 1
    if dm.sum() == 0 or (~dm).sum() == 0:
        print(f"  {g}: all same domain-status (negative/uniform control)"); continue
    for k in (3, 5):
        sep = np.abs(Z[idx[dm]][:, :, k].mean(0) - Z[idx[~dm]][:, :, k].mean(0))
        print(f"  {g} axis{k}: peak sep={sep.max():.3f} @ L{int(sep.argmax()+1)}")
