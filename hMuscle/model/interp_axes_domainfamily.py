#!/usr/bin/env python3
"""
interp_axes_domainfamily.py
===========================
Concrete biological identity of each PCA axis via Pfam domain-family enrichment
(complements the compositional-feature correlations). Also targets the two weak
axes (1, 6) with positional features (N-term hydrophobicity = signal-peptide-like,
transmembrane-segment count) not in the 22-feature panel.

For each axis: top vs bottom tercile by gene-mean axis score; log-odds enrichment
of each Pfam family (>=30 isoforms) among high vs low. Reports strongest +/-.
"""
from __future__ import annotations
import re, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
PEP = ROOT / "hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
OUT = ROOT / "reports/v20b_pca_interp"
Z = np.load(OUT / "Z_brain_Nx30x8.npy")
_bad = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]')

clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
sym = np.array([clean(s) for s in np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)])
bids = np.array([str(x) for x in np.load(BRAIN_DIR / "brain_full_ids.npy", allow_pickle=True)])
N = len(bids)
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
# sequences for positional features
pep, cur, buf = {}, None, []
for line in open(PEP):
    if line.startswith(">"):
        if cur and buf:
            s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)
        cur = line[1:].split()[0].split(".p")[0].split(".")[0]; buf = []
    else: buf.append(line.strip())
if cur and buf:
    s = "".join(buf); pep[cur] = s if len(s) > len(pep.get(cur, "")) else pep.get(cur, s)

KD = dict(zip("AILMFVPGWYSTCNQHKRED", [1.8,4.5,3.8,1.9,2.8,4.2,-1.6,-0.4,-0.9,-1.3,-0.8,-0.7,2.5,-3.5,-3.5,-3.2,-3.9,-4.5,-3.5,-3.5]))
def tm_count(seq, win=18, thr=1.6):
    kd = np.array([KD.get(a, 0) for a in seq]); n = 0; i = 0
    while i <= len(kd)-win:
        if kd[i:i+win].mean() >= thr: n += 1; i += win
        else: i += 1
    return n
def nterm_gravy(seq, k=30):
    s = seq[:k]; return np.mean([KD.get(a, 0) for a in s]) if s else 0.0

axis = Z.mean(1)
enst_of = {}
for i, bid in enumerate(bids):
    enst_of[i] = bid.split(".")[0] if bid.startswith("ENST") else (bid if bid.startswith("transcript") else name2enst.get(bid, ""))
gene2idx = defaultdict(list)
for i in range(N): gene2idx[sym[i]].append(i)
def gmean(v):
    out = np.full(N, np.nan)
    for gg, ix in gene2idx.items(): out[np.array(ix)] = np.nanmean(v[ix])
    return out

# domain family universe
domcount = Counter()
for i in range(N):
    for d in enst2dom.get(enst_of[i], ()): domcount[d] += 1
FAMS = [d for d, c in domcount.items() if c >= 30]
print(f"Pfam families with >=30 isoforms: {len(FAMS)}")

print("\n" + "=" * 74)
print("  DOMAIN-FAMILY identity per axis (top tercile vs bottom tercile, log-odds)")
print("=" * 74)
for k in range(8):
    ag = gmean(axis[:, k])
    valid = ~np.isnan(ag)
    q1, q2 = np.nanpercentile(ag, [33, 67])
    hi = valid & (ag >= q2); lo = valid & (ag <= q1)
    enr = []
    for d in FAMS:
        members = np.array([d in enst2dom.get(enst_of[i], ()) for i in range(N)])
        a = (members & hi).sum(); b = (members & lo).sum()
        na, nb = hi.sum(), lo.sum()
        lodds = math.log(((a+0.5)/(na-a+0.5)) / ((b+0.5)/(nb-b+0.5)))
        enr.append((d, lodds, a, b))
    enr.sort(key=lambda x: -x[1])
    pos = [f"{d}(+{lo_:.1f})" for d, lo_, a, b in enr[:4] if lo_ > 0.5]
    neg = [f"{d}({lo_:.1f})" for d, lo_, a, b in enr[-4:] if lo_ < -0.5]
    print(f"\n  axis {k}:")
    print(f"    HIGH-enriched: {', '.join(pos) if pos else '(none strong)'}")
    print(f"    LOW-enriched:  {', '.join(neg) if neg else '(none strong)'}")

# targeted positional features for weak axes 1 & 6
print("\n" + "=" * 74)
print("  WEAK-AXIS targeted features (N-term hydrophobicity, TM count)")
print("=" * 74)
tmc = np.full(N, np.nan); ntg = np.full(N, np.nan)
for i in range(N):
    seq = pep.get(enst_of[i], "")
    if not seq: continue
    seq = _bad.sub("", seq.upper().replace("*", ""))
    if len(seq) < 30: continue
    tmc[i] = tm_count(seq); ntg[i] = nterm_gravy(seq)
for k in (1, 6, 0, 4):
    ag = gmean(axis[:, k])
    for fn, fv in [("TM_count", gmean(tmc)), ("Nterm_gravy", gmean(ntg))]:
        m = ~np.isnan(ag) & ~np.isnan(fv)
        r = stats.spearmanr(ag[m], fv[m])[0]
        print(f"  axis {k} vs {fn}: rho={r:+.3f}")
