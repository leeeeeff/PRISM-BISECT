#!/usr/bin/env python3
"""
Option B — ρ~0.27 within-gene domain ceiling: feature 한계인가 ESM-2 한계인가
==========================================================================
joint axis3 within-gene r(ndom|len)=+0.264. 이 상한이:
 (a) domain-COUNT feature의 조악함? → richer domain descriptor(identity Hamming)로 오르나?
 (b) 8축 압축 탓? → 8축 다변량(multiple R)이 단일축보다 큰가?
 (c) ESM-2 자체 한계? → (검증은 640-dim 필요, 여기선 8축 상한만 — 다음단계 표시)

방법 (within-gene isoform pair, layer-mean 8축):
 1. |Δaxis3| vs Δndom(count) vs domain-set Hamming(identity) Spearman 비교.
 2. 8축 다변량 → domain-different(binary) 예측 AUROC (length-matched) = 8축이 담은 domain 상한.
 3. |Δ(8축벡터)| vs Hamming(identity) Spearman + length 통제 partial.
"""
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE"); P = ROOT / "reports/v20b_pca_interp"; WF = P / "within_family"
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
Z = np.load(P / "Z_brain_Nx30x8.npy").mean(1)     # (N,8) joint layer-mean
sym, _ = M.gene_to_family(); N = len(sym)
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])
feat = np.load(P / "feat_matrix_brain.npy"); length = feat[:, M.FEAT_NAMES.index("length")]
rng = np.random.default_rng(42)

# domain sets + protein length
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
ndom = np.array([len(d) for d in domset])

# enumerate within-gene pairs, at least one domain-bearing, both length valid
g2i = defaultdict(list)
for i in range(N): g2i[sym[i]].append(i)
pairs = []
for g, idx in g2i.items():
    idx = [i for i in idx if not np.isnan(length[i]) and (len(domset[i]) > 0 or True)]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs.append((idx[a], idx[b]))
pairs = np.array(pairs)
if len(pairs) > 60000:
    pairs = pairs[rng.choice(len(pairs), 60000, replace=False)]
i1, i2 = pairs[:, 0], pairs[:, 1]
dax3 = np.abs(Z[i1, 3] - Z[i2, 3])
dvec = np.linalg.norm(Z[i1] - Z[i2], axis=1)                 # |Δ 8-axis vector|
dcnt = np.abs(ndom[i1] - ndom[i2])                          # count diff
dham = np.array([len(domset[a] ^ domset[b]) for a, b in pairs])  # identity Hamming
dlen = np.abs(length[i1] - length[i2])
domdiff = (dham > 0).astype(int)
vl = ~np.isnan(dlen)
print(f"within-gene pairs n={len(pairs):,}  domain-different={int(domdiff.sum()):,}")

def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def spart(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab, rac, rbc = np.corrcoef(ra, rb)[0,1], np.corrcoef(ra, rc)[0,1], np.corrcoef(rb, rc)[0,1]
    return float((rab - rac*rbc)/np.sqrt((1-rac**2)*(1-rbc**2)+1e-12))

print("\n[1] |Δaxis3| vs domain COUNT vs IDENTITY(Hamming)  (length 통제)")
print(f"  Spearman(|Δaxis3|, Δcount)          = {sp(dax3[vl],dcnt[vl]):+.3f}  | partial|len = {spart(dax3[vl],dcnt[vl],dlen[vl]):+.3f}")
print(f"  Spearman(|Δaxis3|, Hamming identity) = {sp(dax3[vl],dham[vl]):+.3f}  | partial|len = {spart(dax3[vl],dham[vl],dlen[vl]):+.3f}")
print("\n[2] |Δ(8-axis vector)| vs domain (전 8축 다변량 거리)")
print(f"  Spearman(|Δ8axis|, Δcount)          = {sp(dvec[vl],dcnt[vl]):+.3f}  | partial|len = {spart(dvec[vl],dcnt[vl],dlen[vl]):+.3f}")
print(f"  Spearman(|Δ8axis|, Hamming identity) = {sp(dvec[vl],dham[vl]):+.3f}  | partial|len = {spart(dvec[vl],dham[vl],dlen[vl]):+.3f}")

print("\n[3] 8축이 domain-different(binary)를 예측하는 AUROC (length-matched 5-fold)")
# length-matched: within length-diff deciles, train LR on 8-axis |Δ| to predict domain-diff
X = np.abs(Z[i1] - Z[i2])[vl]; y = domdiff[vl]; ln = dlen[vl]
dec = np.digitize(ln, np.quantile(ln, np.linspace(0.1, 0.9, 9)))
aucs = []
for d in range(10):
    m = dec == d
    if m.sum() < 100 or y[m].sum() < 20 or y[m].sum() == m.sum(): continue
    Xd, yd = X[m], y[m]; n = len(Xd); perm = rng.permutation(n); cut = n // 2
    tr, te = perm[:cut], perm[cut:]
    lr = LogisticRegression(max_iter=500).fit(Xd[tr], yd[tr])
    aucs.append(roc_auc_score(yd[te], lr.predict_proba(Xd[te])[:, 1]))
print(f"  length-matched AUROC (8-axis → domain-different) = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (n_bins={len(aucs)})")
print("  → AUROC가 0.5 훨씬 초과면 8축이 domain identity를 length-독립으로 담음(0.27은 count의 선형상관 한계).")
print("  ⚠️ ESM-2 자체 상한 검증은 640-dim ridge 필요(다음단계).")

import json
json.dump({"r_ax3_count":sp(dax3[vl],dcnt[vl]),"r_ax3_count_partial":spart(dax3[vl],dcnt[vl],dlen[vl]),
           "r_ax3_hamming":sp(dax3[vl],dham[vl]),"r_ax3_hamming_partial":spart(dax3[vl],dham[vl],dlen[vl]),
           "r_8axis_hamming":sp(dvec[vl],dham[vl]),"r_8axis_hamming_partial":spart(dvec[vl],dham[vl],dlen[vl]),
           "auroc_8axis_domaindiff_lenmatched":float(np.mean(aucs))},
          open(WF/"optB_domain_ceiling.json","w"), indent=2)
print(f"\n[saved] {WF/'optB_domain_ceiling.json'}")
