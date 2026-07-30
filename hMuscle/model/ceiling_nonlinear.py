#!/usr/bin/env python3
"""
Option A — 640-dim 0.84의 나머지 0.16이 ESM-2 표현 한계인가 decoder(LR) 한계인가
================================================================================
ceiling_640dim: length-matched LINEAR(LR) → 8축 0.715, 640-dim 0.838.
질문: 0.84 위 잔여 0.16은 (a) ESM-2가 domain-change를 더 못 담아서(표현 한계)인가,
      (b) 선형 분류기가 |Δ| 표현을 덜 짜내서(decoder 한계)인가?
판별: 동일 length-matched 프로토콜에서 NONLINEAR(HistGradientBoosting) 재측정.
      nonlinear ≫ linear → 잔여는 decoder 한계(표현엔 더 있음).
      nonlinear ≈ linear → 표현 한계에 근접(선형으로 이미 다 짜냄).
캐시: X640(z-scored layer-mean)을 scratchpad에 저장/재사용(빌드 170s 회피).
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import re, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import interp_within_family_pca as M

t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/welcome1/sw1686/DIFFUSE"); P = ROOT / "reports/v20b_pca_interp"; WF = P / "within_family"
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
CACHE = Path("/tmp/claude-1811/-home-welcome1-sw1686-DIFFUSE/010f9706-8801-4761-a27e-2255c2663dd1/scratchpad/X640_zmean.npy")
rng = np.random.default_rng(42)

Z8 = np.load(P / "Z_brain_Nx30x8.npy").mean(1)
sym, _ = M.gene_to_family(); N = len(sym)
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])
feat = np.load(P / "feat_matrix_brain.npy"); length = feat[:, M.FEAT_NAMES.index("length")]

if CACHE.exists():
    X640 = np.load(CACHE); log(f"loaded cached X640 {X640.shape}")
else:
    mu = np.load(P / "layer_stats_mu.npy"); sd = np.load(P / "layer_stats_sd.npy")
    X640 = np.zeros((N, 640), dtype=np.float64)
    for L in range(30):
        arr = np.load(BRAIN / f"brain_full_esm2_layer{L+1:02d}_t30_150M.npy", mmap_mode="r")
        X640 += (arr.astype(np.float64) - mu[L]) / sd[L]; del arr
    X640 /= 30.0
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.save(CACHE, X640); log("built+cached X640")

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
def dom_of(i):
    b = bids[i]; e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else name2enst.get(b, ""))
    return enst2dom.get(e, frozenset())
domset = [frozenset(dom_of(i)) for i in range(N)]

g2i = defaultdict(list)
for i in range(N): g2i[sym[i]].append(i)
pairs = []
for g, idx in g2i.items():
    idx = [i for i in idx if not np.isnan(length[i])]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)): pairs.append((idx[a], idx[b]))
pairs = np.array(pairs)
if len(pairs) > 60000: pairs = pairs[rng.choice(len(pairs), 60000, replace=False)]
i1, i2 = pairs[:, 0], pairs[:, 1]
dham = np.array([len(domset[a] ^ domset[b]) for a, b in pairs])
dlen = np.abs(length[i1] - length[i2]); domdiff = (dham > 0).astype(int); vl = ~np.isnan(dlen)
log(f"pairs n={len(pairs):,}  domain-different={int(domdiff.sum()):,}")

def lenmatched(X, y, ln, clf_fn, seed=1):
    r = np.random.default_rng(seed)
    dec = np.digitize(ln, np.quantile(ln, np.linspace(0.1, 0.9, 9)))
    aucs = []
    for d in range(10):
        m = dec == d
        if m.sum() < 100 or y[m].sum() < 20 or y[m].sum() == m.sum(): continue
        Xd, yd = X[m], y[m]; mu_ = Xd.mean(0); sdv = Xd.std(0) + 1e-8; Xd = (Xd - mu_) / sdv
        n = len(Xd); perm = r.permutation(n); cut = n // 2; tr, te = perm[:cut], perm[cut:]
        clf = clf_fn(Xd.shape[1]).fit(Xd[tr], yd[tr])
        aucs.append(roc_auc_score(yd[te], clf.predict_proba(Xd[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), len(aucs)

def lr_fn(dim): return LogisticRegression(max_iter=1000, C=(0.05 if dim > 50 else 1.0))
def hgb_fn(dim): return HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                        learning_rate=0.08, l2_regularization=1.0, random_state=0, early_stopping=True)

X8 = np.abs(Z8[i1] - Z8[i2])[vl]; Xf = np.abs(X640[i1] - X640[i2])[vl]
y = domdiff[vl]; ln = dlen[vl]

res = {}
for tag, X in [("8axis", X8), ("640dim", Xf)]:
    la, ls, ln_ = lenmatched(X, y, ln, lr_fn); ha, hs, hn = lenmatched(X, y, ln, hgb_fn)
    res[tag] = {"linear_auroc": la, "linear_std": ls, "nonlinear_auroc": ha, "nonlinear_std": hs,
                "nonlinear_gain": ha - la}
    log(f"[{tag:>7}] linear LR={la:.3f}±{ls:.3f}  nonlinear HGB={ha:.3f}±{hs:.3f}  gain={ha-la:+.3f}")

gap640 = res["640dim"]["linear_auroc"]
nl_gain = res["640dim"]["nonlinear_gain"]
verdict = ("잔여는 DECODER 한계 — 표현엔 domain 정보 더 있음(nonlinear로 상승)" if nl_gain > 0.02
           else "표현 한계에 근접 — 선형으로 이미 대부분 짜냄(nonlinear 이득 미미)")
log(f"=> 640-dim linear {gap640:.3f}, nonlinear gain {nl_gain:+.3f} → {verdict}")
res["verdict"] = verdict
json.dump(res, open(WF / "ceiling_nonlinear.json", "w"), indent=2)
log(f"[saved] {WF/'ceiling_nonlinear.json'}")
