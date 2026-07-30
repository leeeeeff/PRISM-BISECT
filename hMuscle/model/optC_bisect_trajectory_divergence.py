#!/usr/bin/env python3
"""
Option C — BISECT 케이스 전체 trajectory 분기 layer 분포 + NDUFS7 permutation
==========================================================================
Q1: domain-스위치 isoform 쌍이 axis3(domain축) 궤적에서 어느 layer에서 최대로 갈라지나?
    → 분기 peak layer 분포가 mid-network(L~17)에 몰리면 "표현이 도메인차를 중간층에서 인코딩" 입증.
Q2: NDUFS7 단일 domain+ isoform(n=1)의 axis3 이탈이 domain−(n=10) 대비 유의한가? (permutation)
"""
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import interp_within_family_pca as M

ROOT = Path("/home/welcome1/sw1686/DIFFUSE"); P = ROOT / "reports/v20b_pca_interp"; WF = P / "within_family"
BRAIN = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
TAB = ROOT / "Final_analysis/pipeline_bioanalysis/outputs/supplementary_table_S_bisect_121cases.tsv"
AX = 3
Z = np.load(P / "Z_brain_Nx30x8.npy")            # (N,30,8) joint
sym, _ = M.gene_to_family()
bids = np.array([str(x) for x in np.load(BRAIN / "brain_full_ids.npy", allow_pickle=True)])
id2idx = {}
for i, b in enumerate(bids):
    id2idx[b] = i; id2idx.setdefault(b.split(".")[0], i)
def lk(t):
    t = str(t)
    for key in (t, t.split(".")[0]):
        if key in id2idx: return id2idx[key]
    return None

# --- Q1: BISECT domain-change cases, per-layer |Δaxis3| peak ---
rows = []
with open(TAB) as f:
    hdr = f.readline().rstrip("\n").split("\t"); ci = {h: k for k, h in enumerate(hdr)}
    for line in f:
        p = line.rstrip("\n").split("\t")
        nl = int(float(p[ci["N_lost"]] or 0)); ng = int(float(p[ci["N_gained"]] or 0))
        if nl == 0 and ng == 0: continue
        rows.append((p[ci["Gene"]], p[ci["CT_transcript"]], p[ci["AD_transcript"]], ng - nl))

peak_layers, peak_seps = [], []
for gene, ct, ad, netd in rows:
    if netd == 0: continue
    ic, ia = lk(ct), lk(ad)
    if ic is None or ia is None: continue
    dtraj = np.abs(Z[ia, :, AX] - Z[ic, :, AX])          # per-layer |Δaxis3|
    peak_layers.append(int(dtraj.argmax() + 1)); peak_seps.append(float(dtraj.max()))
peak_layers = np.array(peak_layers)
print(f"[Q1] BISECT domain-change cases mapped: n={len(peak_layers)}")
print(f"     axis3 divergence PEAK LAYER distribution: median L{int(np.median(peak_layers))}, "
      f"mean L{peak_layers.mean():.1f}, IQR [L{int(np.percentile(peak_layers,25))}, L{int(np.percentile(peak_layers,75))}]")
mid = ((peak_layers >= 10) & (peak_layers <= 22)).mean()
early = (peak_layers < 10).mean(); late = (peak_layers > 22).mean()
print(f"     fraction peaking early(<L10)={early:.2f}  mid(L10-22)={mid:.2f}  late(>L22)={late:.2f}")
# null: random within-gene pairs (any gene, 2 iso) peak-layer distribution
rng = np.random.default_rng(42)
g2i = defaultdict(list)
for i in range(len(sym)): g2i[sym[i]].append(i)
multi = [v for v in g2i.values() if len(v) >= 2]
null_pl = []
for _ in range(len(peak_layers) * 20):
    grp = multi[rng.integers(len(multi))]; a, b = rng.choice(grp, 2, replace=False)
    null_pl.append(int(np.abs(Z[a, :, AX] - Z[b, :, AX]).argmax() + 1))
null_pl = np.array(null_pl)
print(f"     NULL random within-gene pairs peak-layer: median L{int(np.median(null_pl))}, mid-frac={((null_pl>=10)&(null_pl<=22)).mean():.2f}")
print(f"     Mann-Whitney BISECT vs null peak-layer: p={stats.mannwhitneyu(peak_layers,null_pl).pvalue:.3g}")

# --- Q2: NDUFS7 single domain+ vs domain- permutation on axis3 (layer-mean) ---
def ndom_of(i):
    b = bids[i]; e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else b)
    return None
# build domain sets
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
def nd(i):
    b = bids[i]; e = b.split(".")[0] if b.startswith("ENST") else (b if b.startswith("transcript") else name2enst.get(b, ""))
    return len(enst2dom.get(e, set()))
print("\n[Q2] NDUFS7 single domain+ vs domain- (axis3 layer-mean), permutation")
idx = np.array([i for i in range(len(sym)) if sym[i] == "NDUFS7"])
dm = np.array([nd(i) >= 1 for i in idx])
score = Z[idx][:, :, AX].mean(1)                     # layer-mean axis3 per isoform
obs = score[dm].mean() - score[~dm].mean()
perm = []
for _ in range(10000):
    pl = rng.permutation(len(idx)); m = pl[:dm.sum()]
    perm.append(score[m].mean() - score[np.setdiff1d(np.arange(len(idx)), m)].mean())
perm = np.array(perm); pval = (np.abs(perm) >= abs(obs)).mean()
print(f"     domain+ (n={int(dm.sum())}) axis3={score[dm].mean():.3f}  domain- (n={int((~dm).sum())}) axis3={score[~dm].mean():.3f}")
print(f"     observed Δ={obs:+.3f}  permutation p={pval:.3f}  (n=1 vs 10, low power expected)")

import json
json.dump({"peak_layer_median": float(np.median(peak_layers)), "n_cases": len(peak_layers),
           "mid_frac": float(mid), "null_mid_frac": float(((null_pl>=10)&(null_pl<=22)).mean()),
           "mw_p": float(stats.mannwhitneyu(peak_layers,null_pl).pvalue),
           "ndufs7_delta": float(obs), "ndufs7_perm_p": float(pval)},
          open(WF/"optC_bisect_trajectory.json","w"), indent=2)
print(f"\n[saved] {WF/'optC_bisect_trajectory.json'}")
