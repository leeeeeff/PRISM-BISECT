"""
interp_axes_vs_biology.py
=========================
재프레임 결정 실험: v20b Joint-PCA 8축이 GO·MLP와 독립인 실제 생물학 feature를
인코딩하는가 — 두 수준(gene-level flow / isoform-level residual flow)에서 검증.

참이면: "축 k = NLS 축, 축 m = phospho 축"처럼 모델이 무엇을 구별하는지 증명 → interpretability 논문.
거짓이면(diffuse/entangled): 재프레임은 서사로만 남음.

feature (전부 GO·MLP 독립):
  - SLiM 10종 (ELM 정규식 count): NLS/NES/PxxP/phospho_CK2/PKA/CDK/KFERQ/RGD/CAAX/DEG
  - biophysical: length, GRAVY(소수성), MW, aromaticity, instability, pI
  - domain count (hmmscan Pfam)
축 scalar = mean over 30 layers of Z[:,:,k].
gene-level: gene-mean(axis) vs gene-mean(feat). isoform-level: within-gene residual vs residual.
"""
from __future__ import annotations
import re, json
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
Z = np.load(ROOT / "reports/v20b_pca_interp/Z_brain_Nx30x8.npy")  # (N,30,8)
OUT = ROOT / "reports/v20b_pca_interp"

SLIMS = {
    'NLS': r'[KR]{3,}|[KR].{1,2}[KR]{2,}', 'NES': r'L.{2,3}[LIVMF].{2,3}L.{2,3}L',
    'PXXP': r'P.{2}P', 'RGD': r'RGD', 'KFERQ': r'[KQRE].{1,3}[KQRE].*F',
    'CK2': r'[ST].{2}[ED]', 'PKA': r'[RK].{2}[ST]', 'CDK': r'[ST]P[KR]',
    'CAAX': r'C[AC].{1}[LIVMF]$', 'DEG': r'[RK].{2}L',
}
SLIM_RE = {k: re.compile(v) for k, v in SLIMS.items()}
_bad = re.compile(r'[^ACDEFGHIKLMNPQRSTVWY]')


def main():
    clean = lambda r: str(r).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")
    sym = np.array([clean(s) for s in np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)])
    bids = np.array([str(x) for x in np.load(BRAIN_DIR / "brain_full_ids.npy", allow_pickle=True)])
    N = len(bids)

    # id -> ENST
    name2enst = {}
    for line in open(GTF):
        if "\ttranscript\t" not in line: continue
        em = re.search(r'transcript_id "([^"]+)"', line); nm = re.search(r'transcript_name "([^"]+)"', line)
        if em and nm: name2enst[nm.group(1)] = em.group(1).split(".")[0]

    # ENST -> pep (longest ORF)
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

    # domains
    enst2ndom = defaultdict(set)
    for line in open(HMM):
        if line.startswith("#") or not line.strip(): continue
        p = line.split()
        if float(p[12]) > 1e-5: continue
        enst2ndom[p[3].split(".p")[0].split(".")[0]].add(p[1].split(".")[0])

    FEATS = list(SLIMS) + ["length", "gravy", "mw", "aromatic", "instab", "pI", "ndom"]
    X = np.full((N, len(FEATS)), np.nan, dtype=np.float32)
    have = np.zeros(N, bool)
    for i, bid in enumerate(bids):
        enst = bid.split(".")[0] if bid.startswith("ENST") else (bid if bid.startswith("transcript") else name2enst.get(bid, ""))
        seq = pep.get(enst, "")
        if not seq: continue
        seq = seq.upper().replace("*", "")
        have[i] = True
        row = [len(SLIM_RE[k].findall(seq)) for k in SLIMS]
        L = len(seq)
        clean_seq = _bad.sub("", seq)
        try:
            pa = ProteinAnalysis(clean_seq) if clean_seq else None
            gravy = pa.gravy() if pa else np.nan
            mw = pa.molecular_weight() if pa else np.nan
            arom = pa.aromaticity() if pa else np.nan
            instab = pa.instability_index() if pa else np.nan
            pI = pa.isoelectric_point() if pa else np.nan
        except Exception:
            gravy = mw = arom = instab = pI = np.nan
        row += [L, gravy, mw, arom, instab, pI, len(enst2ndom.get(enst, set()))]
        X[i] = row
    print(f"isoforms with pep seq: {have.sum()} / {N}")

    axis_scalar = Z.mean(1)  # (N,8) mean over layers

    # gene grouping
    g2idx = defaultdict(list)
    for i, g in enumerate(sym):
        if have[i]: g2idx[g].append(i)
    valid = np.array([i for i in range(N) if have[i]])
    multi = {g: idx for g, idx in g2idx.items() if len(idx) >= 2}
    multi_idx = np.array([i for g in multi for i in multi[g]])
    print(f"valid={len(valid)}, multi-iso valid={len(multi_idx)}")

    def gene_mean_vec(vals, idxset):
        gm = np.full_like(vals, np.nan, dtype=np.float64)
        gmap = defaultdict(list)
        for i in idxset: gmap[sym[i]].append(i)
        for g, idx in gmap.items():
            gm[idx] = np.nanmean(vals[idx])
        return gm

    print("\n=== GENE-LEVEL flow: 각 축이 gene 수준에서 어떤 feature를 인코딩 ===")
    print(f"{'':>8}" + "".join(f"{f[:6]:>8}" for f in FEATS))
    gene_tab = {}
    for k in range(8):
        a = axis_scalar[valid, k]
        rr = []
        for fi, f in enumerate(FEATS):
            fv = X[valid, fi]
            m = ~np.isnan(fv)
            rho = stats.spearmanr(a[m], fv[m]).correlation if m.sum() > 100 else np.nan
            rr.append(rho)
        gene_tab[k] = rr
        print(f"axis{k:>2}  " + "".join(f"{(('%+.2f'%r) if not np.isnan(r) else '  .  '):>8}" for r in rr))

    print("\n=== ISOFORM-LEVEL flow (gene-mean 제거 residual): 진짜 재프레임 판정 ===")
    print(f"{'':>8}" + "".join(f"{f[:6]:>8}" for f in FEATS))
    iso_tab = {}
    # precompute residual features on multi_idx
    for k in range(8):
        a = axis_scalar[:, k].astype(np.float64)
        a_res = a - gene_mean_vec(a, multi_idx)
        rr = []
        for fi, f in enumerate(FEATS):
            fv = X[:, fi].astype(np.float64)
            fv_res = fv - gene_mean_vec(fv, multi_idx)
            m = np.isfinite(a_res) & np.isfinite(fv_res)
            m &= np.isin(np.arange(N), multi_idx)
            rho = stats.spearmanr(a_res[m], fv_res[m]).correlation if m.sum() > 100 else np.nan
            rr.append(rho)
        iso_tab[k] = rr
        print(f"axis{k:>2}  " + "".join(f"{(('%+.2f'%r) if not np.isnan(r) else '  .  '):>8}" for r in rr))

    # strongest isoform-level axis<->feature mappings
    print("\n=== isoform-level 최강 축↔feature (|rho|>=0.15) ===")
    hits = []
    for k in range(8):
        for fi, f in enumerate(FEATS):
            r = iso_tab[k][fi]
            if not np.isnan(r) and abs(r) >= 0.15:
                hits.append((abs(r), k, f, r))
    for _, k, f, r in sorted(hits, reverse=True):
        print(f"  axis{k} <-> {f:10s} rho={r:+.3f}")
    if not hits:
        print("  (없음 — isoform-level에서 축↔feature 클린 매핑 실패 = 신호 entangled)")

    json.dump({"features": FEATS, "gene_level": {k: gene_tab[k] for k in gene_tab},
               "isoform_level": {k: iso_tab[k] for k in iso_tab}},
              open(OUT / "axes_vs_biology.json", "w"), indent=2, default=str)
    print(f"\n[saved] {OUT/'axes_vs_biology.json'}")


if __name__ == "__main__":
    main()
