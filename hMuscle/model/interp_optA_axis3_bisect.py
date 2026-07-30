#!/usr/bin/env python3
"""
Option A(2) — axis3(length-독립 domain-architecture 축)를 실제 BISECT 도메인-스위치 케이스에 투영
==============================================================================================
표현축 → downstream 생물학 다리: within-gene domain 축(axis3)이 실제 isoform 간 Pfam 도메인
loss/gain 방향과 정렬하는가?

핵심 성질: 같은 gene 두 isoform은 family-centering이 동일 family 평균을 빼므로 Δaxis3(A−B)는
centering에 **불변** → within-gene domain 신호의 순수 검정.

가설: axis3는 ndom과 +상관(r=+0.289) → 도메인 net gain(N_gained−N_lost>0) isoform이 axis3 ↑.
검정: 84 domain-change BISECT 케이스에서 sign(Δaxis3)==sign(Δdomain) concordance + Spearman.
      Δlength 대비 partial로 length-독립성 재확인. Null: binomial sign test.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
from pathlib import Path
import numpy as np
from scipy import stats
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT = ROOT / "reports/v20b_pca_interp"; WF = OUT / "within_family"
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
TAB = ROOT / "Final_analysis/pipeline_bioanalysis/outputs/supplementary_table_S_bisect_121cases.tsv"

AX = 3  # length-독립 domain-architecture 축
Z = np.load(WF / "Z_within_Nx30x8.npy"); axis3 = Z.mean(1)[:, AX]      # (N,)
sym, _ = M.gene_to_family()
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])

# protein length per isoform from pep (feat_matrix length은 절반 NaN이라 직접 계산)
import re
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
PEP = ROOT / "hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep"
_bad = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]')
name2enst = {}
for line in open(GTF):
    if "\ttranscript\t" not in line: continue
    em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
    if em and nm: name2enst[nm.group(1)] = em.group(1).split(".")[0]
pep, cur, buf = {}, None, []
for line in open(PEP):
    if line.startswith(">"):
        if cur and buf:
            s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)
        cur = line[1:].split()[0].split(".p")[0].split(".")[0]; buf = []
    else: buf.append(line.strip())
if cur and buf:
    s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)
def enst_of(bid):
    if bid.startswith("ENST"): return bid.split(".")[0]
    if bid.startswith("transcript"): return bid
    return name2enst.get(bid, "")
length = np.full(len(bids), np.nan)
for i, b in enumerate(bids):
    s = pep.get(enst_of(b), "")
    if s: length[i] = len(_bad.sub("", s.upper().replace("*", "")))
# fallback: feat_matrix length column where pep missing
featL = np.load(OUT / "feat_matrix_brain.npy")[:, M.FEAT_NAMES.index("length")]
miss = np.isnan(length) & ~np.isnan(featL)
length[miss] = featL[miss]
print(f"length coverage: pep+feat = {int((~np.isnan(length)).sum())}/{len(bids)}")
id2idx = {b: i for i, b in enumerate(bids)}
# also strip version for ENST
for i, b in enumerate(bids):
    id2idx.setdefault(b.split(".")[0], i)

def lookup(tid):
    tid = str(tid)
    for key in (tid, tid.split(".")[0]):
        if key in id2idx: return id2idx[key]
    return None

# parse domain-change cases
rows = []
with open(TAB) as f:
    hdr = f.readline().rstrip("\n").split("\t")
    ci = {h: k for k, h in enumerate(hdr)}
    for line in f:
        p = line.rstrip("\n").split("\t")
        nlost = int(float(p[ci["N_lost"]] or 0)); ngain = int(float(p[ci["N_gained"]] or 0))
        if nlost == 0 and ngain == 0: continue
        rows.append((p[ci["Gene"]], p[ci["CT_transcript"]], p[ci["AD_transcript"]], nlost, ngain))

d_ax, d_len, d_dom, names = [], [], [], []
for gene, ct, ad, nlost, ngain in rows:
    ic, ia = lookup(ct), lookup(ad)
    if ic is None or ia is None: continue
    if sym[ic] != sym[ia]:  # must be same gene (within-gene)
        pass  # keep anyway (BISECT pairs are same gene by design; symbol mismatch possible for novel)
    net_dom = ngain - nlost
    if net_dom == 0: continue  # need directional domain change
    d_ax.append(axis3[ia] - axis3[ic]); d_len.append(length[ia] - length[ic]); d_dom.append(net_dom)
    names.append(f"{gene}({'+' if net_dom>0 else ''}{net_dom}dom)")

d_ax, d_len, d_dom = np.array(d_ax), np.array(d_len), np.array(d_dom)
n = len(d_ax)
print(f"domain-change cases: {len(rows)} total, {n} mapped to brain with net domain change\n")

# sign concordance (all mapped)
conc = np.mean(np.sign(d_ax) == np.sign(d_dom))
k_conc = int(np.sum(np.sign(d_ax) == np.sign(d_dom)))
p_binom = stats.binomtest(k_conc, n, 0.5, alternative="greater").pvalue
rho_dom = stats.spearmanr(d_ax, d_dom)[0]
print(f"sign concordance sign(Δaxis3)==sign(Δdomain): {conc:.3f} ({k_conc}/{n})  binomial p(>0.5)={p_binom:.4g}")
print(f"Spearman(Δaxis3, Δdomain net)            : {rho_dom:+.3f}")

# length-controlled subset (both isoforms have valid protein length)
from scipy.stats import rankdata
vl = ~np.isnan(d_len); nv = int(vl.sum())
rho_len = stats.spearmanr(d_ax[vl], d_len[vl])[0]
ra, rd, rl = rankdata(d_ax[vl]), rankdata(d_dom[vl]), rankdata(d_len[vl])
rad, ral, rdl = np.corrcoef(ra, rd)[0,1], np.corrcoef(ra, rl)[0,1], np.corrcoef(rd, rl)[0,1]
partial = (rad - ral*rdl) / np.sqrt((1-ral**2)*(1-rdl**2) + 1e-12)
print(f"\n[length-controlled subset n={nv}]")
print(f"Spearman(Δaxis3, Δlength)                : {rho_len:+.3f}  (Δdomain↔Δlength collinearity rho={rdl:+.3f})")
print(f"partial(Δaxis3, Δdomain | Δlength)       : {partial:+.3f}  ← length 통제 후에도 도메인 정렬?")
print(f"partial(Δaxis3, Δlength | Δdomain)       : "
      f"{(ral-rad*rdl)/np.sqrt((1-rad**2)*(1-rdl**2)+1e-12):+.3f}")
# concordance among length-DECREASE-matched: cases where |Δlength| small but domain changes
small_len = vl & (np.abs(d_len) <= np.nanmedian(np.abs(d_len[vl])))
if small_len.sum() > 5:
    c2 = np.mean(np.sign(d_ax[small_len]) == np.sign(d_dom[small_len]))
    print(f"concordance in SMALL-Δlength half (n={int(small_len.sum())}): {c2:.3f}  (길이변화 작은데도 도메인 정렬하나)")

print("\n케이스별 (Δaxis3 부호 vs net domain):")
for nm, da, dd in sorted(zip(names, d_ax, d_dom), key=lambda x: x[2]):
    mark = "OK" if np.sign(da) == np.sign(dd) else "x "
    print(f"  {mark} {nm:<28} Δaxis3={da:+.3f}")

with open(WF / "optA_axis3_bisect.json", "w") as f:
    json.dump({"n_mapped": n, "concordance": float(conc), "k": k_conc, "binom_p": float(p_binom),
               "spearman_domain": float(rho_dom), "n_lengthctrl": nv, "spearman_length": float(rho_len),
               "partial_domain_given_length": float(partial)}, f, indent=2)
print(f"\n[saved] {WF/'optA_axis3_bisect.json'}")
print("해석: concordance>0.5(p유의) + partial(도메인|길이)이 유의하면 axis3가 실제 도메인-스위치를 length-독립으로 포착.")
