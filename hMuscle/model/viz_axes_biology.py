#!/usr/bin/env python3
"""
viz_axes_biology.py
===================
(1) axis1 refinement: what does the weak axis-1 uniquely encode beyond axis-3?
(2) Visualise each PCA axis vs its biological substance:
    Fig A  axis x feature Spearman heatmap (gene-level + isoform-residual level)
    Fig B  per-axis binned-mean relationship panels (axis score vs top feature)
Caches the brain feature matrix to avoid recomputing ProtParam.
"""
from __future__ import annotations
import re, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
PEP = ROOT / "hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
OUT = ROOT / "reports/v20b_pca_interp"
Z = np.load(OUT / "Z_brain_Nx30x8.npy")
_bad = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]')

FEATS = ["length", "gravy", "aromatic", "instab", "pI", "ndom",
         "helix", "turn", "sheet", "net_charge", "pos_frac", "neg_frac",
         "charged_frac", "cys", "pro", "gly", "his", "trp",
         "aliphatic", "aromatic2", "max_aa", "entropy"]
AXIS_NAME = {0: "β-sheet / hydrophobic", 1: "length (size, weak)",
             2: "Pro-turn order / stability", 3: "size + domain + acidic",
             4: "helix-charge vs Cys", 5: "Pro-turn disorder",
             6: "inverse domain / low-complexity", 7: "acidic α-helical"}

clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
sym = np.array([clean(s) for s in np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)])
bids = np.array([str(x) for x in np.load(BRAIN_DIR / "brain_full_ids.npy", allow_pickle=True)])
N = len(bids)

cache = OUT / "feat_matrix_brain.npy"
if cache.exists():
    X = np.load(cache); have = ~np.isnan(X[:, 0])
    print(f"[cache] feature matrix {X.shape}, have={have.sum()}")
else:
    name2enst = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line: continue
        em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm: name2enst[nm.group(1)] = em.group(1).split(".")[0]
    pep, cur, buf = {}, None, []
    for line in open(PEP):
        if line.startswith(">"):
            if cur and buf:
                s = "".join(buf)
                if len(s) > len(pep.get(cur, "")): pep[cur] = s
            cur = line[1:].split()[0].split(".p")[0].split(".")[0]; buf = []
        else: buf.append(line.strip())
    if cur and buf:
        s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)
    enst2ndom = defaultdict(set)
    for line in open(HMM):
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        if float(p[12]) > 1e-5: continue
        enst2ndom[p[3].split(".p")[0].split(".")[0]].add(p[1].split(".")[0])
    def entropy(seq):
        from collections import Counter
        c = Counter(seq); n = len(seq)
        return -sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0.0
    X = np.full((N, len(FEATS)), np.nan, dtype=np.float32)
    for i, bid in enumerate(bids):
        enst = bid.split(".")[0] if bid.startswith("ENST") else (bid if bid.startswith("transcript") else name2enst.get(bid, ""))
        seq = pep.get(enst, "")
        if not seq: continue
        seq = _bad.sub("", seq.upper().replace("*", ""))
        L = len(seq)
        if L < 10: continue
        pa = ProteinAnalysis(seq); hel, trn, sht = pa.secondary_structure_fraction()
        cnt = {a: seq.count(a) for a in "ACDEFGHIKLMNPQRSTVWY"}; f = lambda *aa: sum(cnt[a] for a in aa)/L
        X[i] = [L, pa.gravy(), pa.aromaticity(), pa.instability_index(), pa.isoelectric_point(),
                len(enst2ndom.get(enst, set())), hel, trn, sht,
                (cnt['K']+cnt['R']-cnt['D']-cnt['E'])/L, f('K','R'), f('D','E'), f('K','R','D','E'),
                f('C'), f('P'), f('G'), f('H'), f('W'), f('I','L','V'), f('F','Y','W'),
                max(cnt.values())/L, entropy(seq)]
    np.save(cache, X); have = ~np.isnan(X[:, 0])
    print(f"[computed] feature matrix {X.shape}, have={have.sum()}, cached")

axis = Z.mean(1)  # (N,8)
gene2idx = defaultdict(list)
for i in range(N):
    if have[i]: gene2idx[sym[i]].append(i)
def gmean(v):
    out = np.full(N, np.nan)
    for gg, ix in gene2idx.items(): out[np.array(ix)] = np.nanmean(v[ix])
    return out

# ── correlation matrices ──────────────────────────────────────────────
Rg = np.zeros((8, len(FEATS))); Ri = np.zeros((8, len(FEATS)))
for k in range(8):
    ak = axis[:, k]; ag = gmean(ak)
    for j in range(len(FEATS)):
        xj = X[:, j]; xg = gmean(xj); m = have & ~np.isnan(xj)
        Rg[k, j] = stats.spearmanr(ag[m], xg[m])[0]
        Ri[k, j] = stats.spearmanr((ak-ag)[m], (xj-xg)[m])[0]

# ── (1) axis-1 refinement: partial corr controlling axis-3 ────────────
a1 = gmean(axis[:, 1]); a3 = gmean(axis[:, 3])
def presid(x, c, m):
    rx, rc = stats.rankdata(x[m]), stats.rankdata(c[m]); b = np.polyfit(rc, rx, 1); return rx-(b[0]*rc+b[1])
print("\n[axis-1 refinement] partial Spearman(axis1, feat | axis3), gene-level:")
rows = []
for j, fn in enumerate(FEATS):
    xg = gmean(X[:, j]); m = have & ~np.isnan(X[:, j])
    r_raw = stats.spearmanr(a1[m], xg[m])[0]
    r_par = stats.spearmanr(presid(a1, a3, m), presid(xg, a3, m))[0]
    rows.append((fn, r_raw, r_par))
rows.sort(key=lambda x: -abs(x[2]))
for fn, rr, rp in rows[:6]:
    print(f"    {fn:>12}: raw={rr:+.3f}  partial|axis3={rp:+.3f}")

# ── Fig A: heatmap ────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
for ax, R, ttl in [(axs[0], Rg, "Gene-level"), (axs[1], Ri, "Isoform-residual level")]:
    im = ax.imshow(R, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90, fontsize=8)
    ax.set_yticks(range(8)); ax.set_yticklabels([f"ax{k}: {AXIS_NAME[k]}" for k in range(8)], fontsize=8)
    ax.set_title(f"Axis × biology Spearman ρ ({ttl})", fontsize=11)
    for k in range(8):
        for j in range(len(FEATS)):
            if abs(R[k, j]) >= 0.30:
                ax.text(j, k, f"{R[k,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(R[k, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025)
plt.tight_layout(); plt.savefig(OUT / "fig_axes_heatmap.png", dpi=140, bbox_inches="tight"); plt.close()
print(f"\nSaved: {OUT/'fig_axes_heatmap.png'}")

# ── Fig B: per-axis binned relationship (gene-level, representative feature) ─
# manual representative feature per axis (distinct + bounded for clarity)
REP = {0: "sheet", 1: "length", 2: "instab", 3: "ndom",
       4: "helix", 5: "pro", 6: "ndom", 7: "neg_frac"}
LOG_Y = {"length"}
fig, axs = plt.subplots(2, 4, figsize=(20, 9))
for k in range(8):
    ax = axs[k // 4, k % 4]
    fn = REP[k]; j = FEATS.index(fn)
    ag = gmean(axis[:, k]); xg = gmean(X[:, j]); m = have & ~np.isnan(X[:, j])
    xs, ys = ag[m], xg[m]
    yplot = np.log10(ys + 1) if fn in LOG_Y else ys
    ylo, yhi = np.percentile(yplot, [1, 99])
    vis = (yplot >= ylo) & (yplot <= yhi)
    ax.hexbin(xs[vis], yplot[vis], gridsize=40, cmap="Greys", mincnt=1, alpha=0.55)
    q = np.quantile(xs, np.linspace(0, 1, 11)); cen, mu, se = [], [], []
    for b in range(10):
        bm = (xs >= q[b]) & (xs <= q[b+1])
        if bm.sum() > 5:
            cen.append(xs[bm].mean()); mu.append(np.median(yplot[bm]))
            se.append(yplot[bm].std()/np.sqrt(bm.sum()))
    ax.errorbar(cen, mu, yerr=se, fmt="o-", color="crimson", ms=5, lw=1.8, capsize=2)
    ax.set_ylim(ylo, yhi)
    ylab = f"log10({fn}+1)" if fn in LOG_Y else fn
    ax.set_title(f"axis {k}: {AXIS_NAME[k]}\nvs {fn}  (ρ={Rg[k,j]:+.2f})", fontsize=9.5)
    ax.set_xlabel(f"axis {k} score (gene-mean)", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)
plt.suptitle("PCA axis vs its top biological feature (gene-level; hexbin density + decile means±SE)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(OUT / "fig_axes_relationships.png", dpi=140, bbox_inches="tight"); plt.close()
print(f"Saved: {OUT/'fig_axes_relationships.png'}")
