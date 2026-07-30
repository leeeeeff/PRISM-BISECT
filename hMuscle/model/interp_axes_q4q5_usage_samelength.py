#!/usr/bin/env python3
"""
Q4 (축이 예측에 쓰이나) + Q5 (동일-길이 isoform에서 축 차이가 유효한가)
=====================================================================
공유: within-gene isoform pair 열거 + 축 차이 + PRISM score 차이 + 도메인 차이 + 길이 차이.

Q5 (same-length validity): 길이-매칭 strata / strict same-length(|Δlen|≤10)에서, 각 축 k가
   domain-different pair를 domain-same pair보다 크게 벌리는가? → 길이-독립으로 유효한 축 식별.
Q4 (prediction usage): ΔPRISM(=두 isoform GO score L2 차이)를 각 |Δaxis_k|가 예측하는가?
   length 통제 partial. + domain 차이 ↔ ΔPRISM. → PRISM 예측이 실제로 어느 축에 민감한가(정렬).
   ⚠️ 이것은 attribution(정렬)이지 causal occlusion 아님 — occlusion은 gold standard next step.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from scipy.stats import rankdata
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT = ROOT / "reports/v20b_pca_interp"; WF = OUT / "within_family"
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
PEP = ROOT / "hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
K, SEED = 8, 42
rng = np.random.default_rng(SEED)

axes = np.load(WF / "Z_within_Nx30x8.npy").mean(1)             # (N,8) layer-mean axis scores
prism = np.load(ROOT / "reports/brain_full_672_scores.npy")   # (N,672) PRISM GO scores
sym, _ = M.gene_to_family(); N = len(sym)
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])

# ---- brain domain sets + protein length per isoform ----
_bad = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]')
name2enst = {}
for line in open(GTF):
    if "\ttranscript\t" not in line: continue
    em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
    if em and nm: name2enst[nm.group(1)] = em.group(1).split(".")[0]
def enst_of(b):
    if b.startswith("ENST"): return b.split(".")[0]
    if b.startswith("transcript"): return b
    return name2enst.get(b, "")
enst2dom = defaultdict(set)
for line in open(HMM):
    if line.startswith("#") or not line.strip(): continue
    p = line.split()
    if float(p[12]) > 1e-5: continue
    enst2dom[p[3].split(".p")[0].split(".")[0]].add(p[1].split(".")[0])
pep, cur, buf = {}, None, []
for line in open(PEP):
    if line.startswith(">"):
        if cur and buf:
            s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)
        cur = line[1:].split()[0].split(".p")[0].split(".")[0]; buf = []
    else: buf.append(line.strip())
if cur and buf:
    s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)
domset = [frozenset(enst2dom.get(enst_of(b), ())) for b in bids]
length = np.full(N, np.nan)
for i, b in enumerate(bids):
    s = pep.get(enst_of(b), "")
    if s: length[i] = len(_bad.sub("", s.upper().replace("*", "")))
featL = np.load(OUT / "feat_matrix_brain.npy")[:, M.FEAT_NAMES.index("length")]
mm = np.isnan(length) & ~np.isnan(featL); length[mm] = featL[mm]

# ---- enumerate within-gene pairs (both must have length; at least one domain-bearing) ----
gene2idx = defaultdict(list)
for i in range(N): gene2idx[sym[i]].append(i)
pairs = []
for g, idx in gene2idx.items():
    idx = [i for i in idx if not np.isnan(length[i])]
    if len(idx) < 2: continue
    allp = [(idx[a], idx[b]) for a in range(len(idx)) for b in range(a + 1, len(idx))]
    if len(allp) > 45:
        allp = [allp[t] for t in rng.choice(len(allp), 45, replace=False)]
    pairs.extend(allp)
pairs = np.array(pairs)
i1, i2 = pairs[:, 0], pairs[:, 1]
dax = np.abs(axes[i1] - axes[i2])                              # (P,8)
dprism = np.linalg.norm(prism[i1] - prism[i2], axis=1)         # (P,)
dlen = np.abs(length[i1] - length[i2])                         # (P,)
ddom = np.array([len(domset[a] ^ domset[b]) for a, b in pairs])  # symmetric-diff size (Hamming)
domdiff = ddom > 0
# domain-same restricted to both domain-bearing (informative)
both_dom = np.array([len(domset[a]) > 0 and len(domset[b]) > 0 for a, b in pairs])
domsame = (~domdiff) & both_dom
print(f"within-gene pairs: {len(pairs):,}  domain-diff={int(domdiff.sum()):,}  domain-same(both≥1)={int(domsame.sum()):,}\n")

# =============================== Q5 ===============================
print("=" * 74)
print(" Q5 — same-length validity: 각 축 |Δaxis| domain-diff vs domain-same (길이-매칭)")
print("=" * 74)
def ratio_in_mask(k, mask):
    dd = mask & domdiff; ds = mask & domsame
    if dd.sum() < 20 or ds.sum() < 20: return np.nan, np.nan, int(dd.sum()), int(ds.sum())
    md, ms = dax[dd, k].mean(), dax[ds, k].mean()
    p = stats.mannwhitneyu(dax[dd, k], dax[ds, k], alternative="greater").pvalue
    return md / ms if ms > 0 else np.nan, p, int(dd.sum()), int(ds.sum())

# length-difference deciles (matched strata)
q5 = {}
lbins = np.digitize(dlen, np.quantile(dlen[~np.isnan(dlen)], np.linspace(0.1, 0.9, 9)))
strict = dlen <= 10   # near-same-length
print(f"{'axis':>4} {'ratio_all':>10} {'p_all':>10} {'ratio_lenmatch(mean)':>20} {'ratio_sameLen(≤10aa)':>20} {'p_sameLen':>10}")
for k in range(K):
    r_all, p_all, _, _ = ratio_in_mask(k, np.ones(len(pairs), bool))
    r_bins = [ratio_in_mask(k, lbins == d)[0] for d in range(10)]
    r_bins = [r for r in r_bins if not np.isnan(r)]
    r_lm = float(np.mean(r_bins)) if r_bins else np.nan
    r_sl, p_sl, nd_sl, ns_sl = ratio_in_mask(k, strict)
    q5[k] = dict(ratio_all=r_all, p_all=p_all, ratio_lenmatched=r_lm,
                 ratio_samelen=r_sl, p_samelen=p_sl, n_dd_samelen=nd_sl, n_ds_samelen=ns_sl)
    print(f"{k:>4} {r_all:>10.3f} {p_all:>10.2g} {r_lm:>20.3f} {r_sl:>20.3f} {p_sl:>10.2g}")
print(f"\n samelen subset(|Δlen|≤10aa): domain-diff n={int((strict&domdiff).sum())}, domain-same n={int((strict&domsame).sum())}")
print(" 해석: ratio>1 & p<0.05가 same-length에서 유지되는 축 = 길이-독립으로 도메인차를 담는 '유효' 축.")

# =============================== Q4 ===============================
print("\n" + "=" * 74)
print(" Q4 — prediction usage: ΔPRISM(GO score L2) ~ |Δaxis_k| (length 통제 partial)")
print("=" * 74)
def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def spartial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rab, rac, rbc = np.corrcoef(ra, rb)[0,1], np.corrcoef(ra, rc)[0,1], np.corrcoef(rb, rc)[0,1]
    return float((rab - rac*rbc) / np.sqrt((1-rac**2)*(1-rbc**2) + 1e-12))
vl = ~np.isnan(dlen)
print(f"{'axis':>4} {'r(Δax,ΔPRISM)':>15} {'r(Δax,ΔPRISM|Δlen)':>20} {'r(Δax,Δlen)':>13}")
q4 = {}
for k in range(K):
    r = sp(dax[vl, k], dprism[vl])
    rp = spartial(dax[vl, k], dprism[vl], dlen[vl])
    rl = sp(dax[vl, k], dlen[vl])
    q4[k] = dict(r_prism=r, r_prism_given_len=rp, r_axis_len=rl)
    print(f"{k:>4} {r:>15.3f} {rp:>20.3f} {rl:>13.3f}")
r_len_prism = sp(dlen[vl], dprism[vl])
r_dom_prism = sp(ddom[vl], dprism[vl])
r_dom_prism_len = spartial(ddom[vl].astype(float), dprism[vl], dlen[vl])
print(f"\n reference: r(Δlength, ΔPRISM)={r_len_prism:+.3f}  r(Δdomain, ΔPRISM)={r_dom_prism:+.3f}  "
      f"r(Δdomain,ΔPRISM|Δlen)={r_dom_prism_len:+.3f}")
print(" 해석: |Δaxis_k|가 ΔPRISM을 length 통제 후에도 예측하면 그 축이 예측에 정렬(사용). "
      "length 자체 r(Δlen,ΔPRISM)이 낮으면 예측은 length-비의존.")

with open(WF / "q4q5_usage_samelength.json", "w") as f:
    json.dump({"n_pairs": int(len(pairs)), "n_domdiff": int(domdiff.sum()), "n_domsame": int(domsame.sum()),
               "Q5": q5, "Q4": q4, "r_len_prism": r_len_prism, "r_dom_prism": r_dom_prism,
               "r_dom_prism_given_len": r_dom_prism_len}, f, indent=2, default=float)
print(f"\n[saved] {WF/'q4q5_usage_samelength.json'}")
