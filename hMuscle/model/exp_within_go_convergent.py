"""
exp_within_go_convergent.py  — Option A (사용자 Point 3: 수렴진화)
================================================================
질문: 동일 GO의 positive isoform들이 구조적으로 다른 핵심 클러스터로 나뉘는가
      (수렴진화: actin/myosin처럼 같은 기능, 다른 구조)? 그리고 모델이 그 구조 분할을 보는가?

독립 ground truth = Pfam domain architecture (hmmscan, GO·모델과 무관).
Step1: GO별 positive 중 서로 다른 도메인 아키텍처 클러스터가 ≥2개 존재하는가 (수렴 존재성).
Step2: 존재하면, L30 임베딩이 그 두 아키텍처 그룹을 선형 분리하는가 (모델이 구조를 봄).
       + axis0(소수성)/axis3(크기)가 두 그룹 간 다른가.
"""
from __future__ import annotations
import re, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings; warnings.filterwarnings("ignore")

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BD = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMM = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"

GO_18 = ["GO:0007204","GO:0045214","GO:0006941","GO:0006914","GO:0043161","GO:0007519",
         "GO:0042692","GO:0055074","GO:0007005","GO:0007517","GO:0032006","GO:0030048",
         "GO:0006096","GO:0007268","GO:0007018","GO:0031175","GO:0030182","GO:0000226"]
GO_NAME = {"GO:0007204":"Ca signaling","GO:0045214":"Sarcomere","GO:0006941":"Muscle contr",
           "GO:0006914":"Autophagy","GO:0043161":"Proteasome","GO:0007519":"Skel musc dev",
           "GO:0042692":"Musc cell diff","GO:0055074":"Ca homeostasis","GO:0007005":"Mito org",
           "GO:0007517":"Musc organ dev","GO:0032006":"TOR sig","GO:0030048":"Actin move",
           "GO:0006096":"Glycolysis","GO:0007268":"Synapse","GO:0007018":"Microtubule mot",
           "GO:0031175":"Neuron proj","GO:0030182":"Neuron diff","GO:0000226":"MT cytoskel"}
MIN = 10


def main():
    clean = lambda r: str(r).replace("b'","").replace("'","").replace('"',"").replace(" ","")
    bids = np.array([str(x) for x in np.load(BD/"brain_full_ids.npy", allow_pickle=True)])
    Y = np.load(BD/"brain_full_labels.npy")  # (N,18)
    L30 = np.load(BD/"brain_full_esm2_t30_150M.npy").astype(np.float32)
    Z = np.load(ROOT/"reports/v20b_pca_interp/Z_brain_Nx30x8.npy")
    axis = Z.mean(1)  # (N,8)
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
    row_dom = [None]*N
    for i, bid in enumerate(bids):
        e = bid.split(".")[0] if bid.startswith("ENST") else (bid if bid.startswith("transcript") else name2enst.get(bid,""))
        if e and e in enst2dom: row_dom[i] = frozenset(enst2dom[e])

    print(f"{'GO':14s} {'name':16s} {'nPos+dom':>8} {'nArchCls':>8} {'top2 sizes':>12} "
          f"{'L30 sep AUC':>11} {'hydropathy Δ':>12} {'size Δ':>8}")
    results = {}
    for k, go in enumerate(GO_18):
        pos = [i for i in range(N) if Y[i,k]==1 and row_dom[i] is not None]
        if len(pos) < 2*MIN:
            print(f"{go:14s} {GO_NAME[go][:16]:16s} {len(pos):>8} {'—':>8}"); continue
        # cluster by domain architecture signature
        sig2idx = defaultdict(list)
        for i in pos: sig2idx[row_dom[i]].append(i)
        big = sorted([(len(v),s) for s,v in sig2idx.items() if len(v)>=MIN], reverse=True)
        n_arch = len(big)
        if n_arch < 2:
            print(f"{go:14s} {GO_NAME[go][:16]:16s} {len(pos):>8} {n_arch:>8}  (단일 아키텍처)");
            results[go]={"n_pos":len(pos),"n_arch":n_arch}; continue
        # top-2 architecture groups -> L30 separability
        g1 = sig2idx[big[0][1]]; g2 = sig2idx[big[1][1]]
        Xsep = L30[g1+g2]; ysep = np.array([0]*len(g1)+[1]*len(g2))
        try:
            auc = cross_val_score(LogisticRegression(max_iter=500,C=0.1), Xsep, ysep,
                                  cv=min(5,len(g1),len(g2)), scoring="roc_auc").mean()
        except Exception:
            auc = float("nan")
        hyd_d = float(axis[g1,0].mean()-axis[g2,0].mean())
        siz_d = float(axis[g1,3].mean()-axis[g2,3].mean())
        print(f"{go:14s} {GO_NAME[go][:16]:16s} {len(pos):>8} {n_arch:>8} "
              f"{str((big[0][0],big[1][0])):>12} {auc:>11.3f} {hyd_d:>+12.3f} {siz_d:>+8.3f}")
        results[go] = {"n_pos":len(pos),"n_arch":n_arch,"top2":(big[0][0],big[1][0]),
                       "L30_sep_auc":auc,"hydropathy_delta":hyd_d,"size_delta":siz_d}
    json.dump(results, open(ROOT/"reports/v20b_pca_interp/within_go_convergent.json","w"), indent=2, default=str)
    print("\n해석: nArchCls≥2 = 수렴진화 존재(같은 GO, 다른 도메인 구조). "
          "L30 sep AUC 높음 = 모델이 그 구조 분할을 봄.")


if __name__ == "__main__":
    main()
