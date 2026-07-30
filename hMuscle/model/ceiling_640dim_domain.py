#!/usr/bin/env python3
"""
Option A (ceiling) — 8축 압축 손실 vs ESM-2 640-dim 상한
========================================================
optB에서 8축 length-matched AUROC(domain-different binary)=0.71을 얻었다.
질문: 이 0.71은 (a) 8축 PCA 압축이 domain 정보를 버린 탓인가, (b) ESM-2 표현 자체의 상한인가?
판별: 동일 표현(z-scored 640-dim layer-mean; 8축은 이 위에서 W_axes로 투영된 것)에서
      전 640-dim을 써 같은 length-matched 프로토콜로 domain-different AUROC를 측정.
      640-dim AUROC ≈ 8축 AUROC  → 8축이 domain 정보를 거의 다 담음(압축 손실 미미).
      640-dim AUROC ≫ 8축 AUROC  → 8축 압축이 손실, ESM-2 상한은 더 높음.

메모리: 30 layer를 한 장씩 누적해 z-scored layer-mean(N,640)만 만든다(전체 (N,30,640) 미적재).
프로토콜: optB와 동일 — within-gene pair, domain-different(Hamming>0) binary,
          length-difference decile 내 length-matched, |Δrep| feature LR, 5-fold-ish half-split AUROC.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import re, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import interp_within_family_pca as M

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

ROOT = Path("/home/welcome1/sw1686/DIFFUSE"); P = ROOT / "reports/v20b_pca_interp"; WF = P / "within_family"
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
rng = np.random.default_rng(42)

# ---- 8-axis layer-mean (baseline reproduce) ----
Z8 = np.load(P / "Z_brain_Nx30x8.npy").mean(1)          # (N,8)
sym, _ = M.gene_to_family(); N = len(sym)
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])
feat = np.load(P / "feat_matrix_brain.npy"); length = feat[:, M.FEAT_NAMES.index("length")]
log(f"N={N}")

# ---- 640-dim z-scored layer-mean, accumulated one layer at a time ----
mu = np.load(P / "layer_stats_mu.npy"); sd = np.load(P / "layer_stats_sd.npy")   # (30,640)
X640 = np.zeros((N, 640), dtype=np.float64)
for L in range(30):
    arr = np.load(BRAIN / f"brain_full_esm2_layer{L+1:02d}_t30_150M.npy", mmap_mode="r")
    X640 += (arr.astype(np.float64) - mu[L]) / sd[L]
    del arr
X640 /= 30.0
log("built z-scored 640-dim layer-mean")

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
def dom_of(i):
    b = bids[i]; e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else name2enst.get(b, ""))
    return enst2dom.get(e, frozenset())
domset = [frozenset(dom_of(i)) for i in range(N)]

# ---- within-gene pairs (identical protocol to optB) ----
g2i = defaultdict(list)
for i in range(N): g2i[sym[i]].append(i)
pairs = []
for g, idx in g2i.items():
    idx = [i for i in idx if not np.isnan(length[i])]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs.append((idx[a], idx[b]))
pairs = np.array(pairs)
if len(pairs) > 60000:
    pairs = pairs[rng.choice(len(pairs), 60000, replace=False)]
i1, i2 = pairs[:, 0], pairs[:, 1]
dham = np.array([len(domset[a] ^ domset[b]) for a, b in pairs])
dlen = np.abs(length[i1] - length[i2])
domdiff = (dham > 0).astype(int)
vl = ~np.isnan(dlen)
log(f"pairs n={len(pairs):,}  domain-different={int(domdiff.sum()):,}")

def lenmatched_auroc(X, y, ln, seed=0):
    """length-difference decile 내 half-split LR AUROC 평균."""
    r = np.random.default_rng(seed)
    dec = np.digitize(ln, np.quantile(ln, np.linspace(0.1, 0.9, 9)))
    aucs = []
    for d in range(10):
        m = dec == d
        if m.sum() < 100 or y[m].sum() < 20 or y[m].sum() == m.sum(): continue
        Xd, yd = X[m], y[m]
        # standardize features within bin (helps 640-dim LR conditioning)
        mu_ = Xd.mean(0); sdv = Xd.std(0) + 1e-8
        Xd = (Xd - mu_) / sdv
        n = len(Xd); perm = r.permutation(n); cut = n // 2
        tr, te = perm[:cut], perm[cut:]
        C = 0.05 if Xd.shape[1] > 50 else 1.0     # stronger L2 for 640-dim
        lr = LogisticRegression(max_iter=1000, C=C).fit(Xd[tr], yd[tr])
        aucs.append(roc_auc_score(yd[te], lr.predict_proba(Xd[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)

X8 = np.abs(Z8[i1] - Z8[i2])[vl]
Xf = np.abs(X640[i1] - X640[i2])[vl]
y = domdiff[vl]; ln = dlen[vl]

a8, s8, n8 = lenmatched_auroc(X8, y, ln, seed=1)
af, sf, nf = lenmatched_auroc(Xf, y, ln, seed=1)
log(f"[8-axis ] length-matched AUROC = {a8:.3f} ± {s8:.3f}  (n_bins={n8})")
log(f"[640-dim] length-matched AUROC = {af:.3f} ± {sf:.3f}  (n_bins={nf})")
log(f"compression gap (640-dim − 8-axis) = {af-a8:+.3f}")
verdict = ("8축이 domain 정보 대부분 보존(압축 손실 미미)" if af - a8 < 0.03
           else "8축 압축이 domain 정보 손실 — ESM-2 상한 더 높음")
log(f"=> {verdict}")

json.dump({"auroc_8axis": a8, "std_8axis": s8, "auroc_640dim": af, "std_640dim": sf,
           "compression_gap": af - a8, "n_pairs": int(len(pairs)),
           "n_domain_diff": int(domdiff.sum()), "verdict": verdict},
          open(WF / "ceiling_640dim_domain.json", "w"), indent=2)
log(f"[saved] {WF/'ceiling_640dim_domain.json'}")
