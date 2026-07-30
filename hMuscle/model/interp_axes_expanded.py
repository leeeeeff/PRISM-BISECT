#!/usr/bin/env python3
"""
interp_axes_expanded.py
=======================
Completeness check for the 8 Joint-PCA axes. The 17-feature panel left axes
1,2,4,5 unidentified (max |rho| ~0.25-0.34). Expand the panel with structural
/ compositional descriptors to (a) reveal unidentified axes and (b) resolve
identified axes (0=hydropathy, 3=size) more finely.

Added features (all GO/MLP-independent, sequence-intrinsic):
  secondary structure: helix_frac, turn_frac, sheet_frac (ProtParam)
  charge: net_charge_per_res, pos_frac(K,R), neg_frac(D,E), charged_frac
  residue classes: cys_frac, pro_frac, gly_frac, his_frac, trp_frac,
                   aliphatic_frac(ILV), aromatic2_frac(FYW)
  composition bias: max_aa_frac (low-complexity), shannon_entropy
Axis scalar = mean over 30 layers of Z[:,:,k]. Correlate at gene-level and
isoform-level (within-gene residual).
"""
from __future__ import annotations
import re, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
PEP = ROOT / "hMuscle/data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
Z = np.load(ROOT / "reports/v20b_pca_interp/Z_brain_Nx30x8.npy")
OUT = ROOT / "reports/v20b_pca_interp"
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
pep = {}
cur, buf = None, []
for line in open(PEP):
    if line.startswith(">"):
        if cur and buf:
            s = "".join(buf)
            if len(s) > len(pep.get(cur, "")): pep[cur] = s
        cur = line[1:].split()[0].split(".p")[0].split(".")[0]; buf = []
    else: buf.append(line.strip())
if cur and buf:
    s = "".join(buf)
    if len(s) > len(pep.get(cur, "")): pep[cur] = s
enst2ndom = defaultdict(set)
for line in open(HMM):
    if line.startswith("#") or not line.strip(): continue
    p = line.split()
    if float(p[12]) > 1e-5: continue
    enst2ndom[p[3].split(".p")[0].split(".")[0]].add(p[1].split(".")[0])

def entropy(seq):
    from collections import Counter
    c = Counter(seq); n = len(seq)
    return -sum((v/n) * math.log2(v/n) for v in c.values()) if n else 0.0

FEATS = ["length", "gravy", "aromatic", "instab", "pI", "ndom",
         "helix", "turn", "sheet",
         "net_charge", "pos_frac", "neg_frac", "charged_frac",
         "cys", "pro", "gly", "his", "trp",
         "aliphatic", "aromatic2", "max_aa", "entropy"]
X = np.full((N, len(FEATS)), np.nan, dtype=np.float32)
have = np.zeros(N, bool)
for i, bid in enumerate(bids):
    enst = bid.split(".")[0] if bid.startswith("ENST") else (bid if bid.startswith("transcript") else name2enst.get(bid, ""))
    seq = pep.get(enst, "")
    if not seq: continue
    seq = _bad.sub("", seq.upper().replace("*", ""))
    L = len(seq)
    if L < 10: continue
    have[i] = True
    pa = ProteinAnalysis(seq)
    hel, trn, sht = pa.secondary_structure_fraction()
    cnt = {a: seq.count(a) for a in "ACDEFGHIKLMNPQRSTVWY"}
    f = lambda *aa: sum(cnt[a] for a in aa) / L
    net = (cnt['K'] + cnt['R'] - cnt['D'] - cnt['E']) / L
    X[i] = [L, pa.gravy(), pa.aromaticity(), pa.instability_index(), pa.isoelectric_point(),
            len(enst2ndom.get(enst, set())),
            hel, trn, sht,
            net, f('K', 'R'), f('D', 'E'), f('K', 'R', 'D', 'E'),
            f('C'), f('P'), f('G'), f('H'), f('W'),
            f('I', 'L', 'V'), f('F', 'Y', 'W'), max(cnt.values()) / L, entropy(seq)]

axis = Z.mean(1)  # (N,8) mean over layers
g = sym
gene2idx = defaultdict(list)
for i in range(N):
    if have[i]: gene2idx[g[i]].append(i)

def gene_mean(v, idxs):
    out = np.full(N, np.nan)
    for gg, ix in idxs.items():
        out[np.array(ix)] = np.nanmean(v[ix])
    return out

print("=" * 74)
print("  EXPANDED axis-vs-biology (22 features). |rho| flagged: >0.30 strong, >0.20 moderate")
print("=" * 74)
res = {"features": FEATS, "gene_level": {}, "isoform_level": {}}
for k in range(8):
    ak = axis[:, k]
    print(f"\n--- AXIS {k} ---")
    for level in ("gene", "iso"):
        rhos = []
        for j, fn in enumerate(FEATS):
            xj = X[:, j]
            m = have & ~np.isnan(xj)
            if level == "gene":
                a = gene_mean(ak, gene2idx); b = gene_mean(xj, gene2idx)
                mm = m & ~np.isnan(a) & ~np.isnan(b)
                rho, _ = stats.spearmanr(a[mm], b[mm])
            else:  # isoform residual
                a = ak - gene_mean(ak, gene2idx); b = xj - gene_mean(xj, gene2idx)
                mm = m & ~np.isnan(a) & ~np.isnan(b)
                rho, _ = stats.spearmanr(a[mm], b[mm])
            rhos.append((fn, float(rho) if rho == rho else 0.0))
        rhos.sort(key=lambda x: -abs(x[1]))
        lvlkey = "gene_level" if level == "gene" else "isoform_level"
        res[lvlkey][str(k)] = {fn: r for fn, r in rhos}
        top = ", ".join(f"{fn}={r:+.2f}" for fn, r in rhos[:4])
        print(f"  {level:>4}: {top}")

with open(OUT / "axes_vs_biology_expanded.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"\nSaved: {OUT/'axes_vs_biology_expanded.json'}")
