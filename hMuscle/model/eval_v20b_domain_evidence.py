"""
eval_v20b_domain_evidence.py
=============================
Label Granularity Ceiling(발견: within-gene GO ranking AUC 항상 n_valid_gos=0,
gene-level 라벨 구조상 원천 불가)을 우회하는 독립 검증.

GO 라벨이 아니라 Pfam 도메인 구성(hmmscan, GO와 완전 무관한 서열-구조적 증거)을
"이 isoform 쌍이 실제로 달라야 하는가"의 ground truth로 사용한다.

질문: v20b(trajectory/window curve concat)가 v15d(flat mean-pool)보다
      동일 유전자 내 isoform 쌍의 실제 도메인 구성 차이를 점수 차이로 더 잘 반영하는가?

방법:
  1. GTF transcript_name -> ENST 매핑 (extract_brain_aux_features.py와 동일 레시피)
  2. hmmscan_brain.domtblout -> ENST별 Pfam accession set
  3. 같은 유전자, 도메인 정보 있는 isoform 쌍에 대해:
       domain_jaccard_dist = 1 - |A∩B|/|A∪B|
       score_diff_v15d, score_diff_v20b = mean_k |pred_i,k - pred_j,k| (18 GO 평균)
  4. Spearman(domain_dist, score_diff) : v15d vs v20b 비교
  5. Case study: domain_dist가 큰데(완전 다른 도메인 구성) v15d는 거의 구별 못하고
     v20b는 뚜렷이 구별하는 쌍 상위 N개 추출.
"""
from __future__ import annotations
import re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
BRAIN_DIR = ROOT / "hMuscle/data/brain_isoquant_esm2/full"
BRAIN_GTF = ROOT / "hMuscle/data/brain_esm2/brain_only.gtf"
HMMSCAN = ROOT / "hMuscle/results_isoform/features/hmmscan_brain.domtblout"
OUT_DIR = ROOT / "reports" / "v20b_domain_evidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v15d": ROOT / "reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy",
    "v20b_w5": ROOT / "reports/v20b_brain/w5_brain_score_matrix.npy",
    "v20b_w7": ROOT / "reports/v20b_brain/w7_brain_score_matrix.npy",
}
N_GO = 18


def build_name2enst() -> dict:
    name2enst = {}
    with open(BRAIN_GTF) as f:
        for line in f:
            if "\ttranscript\t" not in line:
                continue
            em = re.search(r'transcript_id "([^"]+)"', line)
            nm = re.search(r'transcript_name "([^"]+)"', line)
            if em and nm:
                name2enst[nm.group(1)] = em.group(1).split(".")[0]
    return name2enst


def parse_hmmscan_domains() -> dict:
    """ENST(no version) -> set of Pfam accessions (deduped, independent E-value filter)."""
    dom = defaultdict(set)
    with open(HMMSCAN) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pfam_acc = parts[1].split(".")[0]   # accession, e.g. PF00664
            query = parts[3]                    # e.g. ENST00000003084.p1
            i_eval = float(parts[12])            # independent E-value
            if i_eval > 1e-5:
                continue
            enst = query.split(".p")[0].split(".")[0]
            dom[enst].add(pfam_acc)
    return dom


def main():
    print("[1] GTF transcript_name -> ENST 매핑")
    name2enst = build_name2enst()
    print(f"   mappings: {len(name2enst):,}")

    print("[2] hmmscan_brain.domtblout 파싱 (Pfam domain sets, i-Evalue<=1e-5)")
    enst2dom = parse_hmmscan_domains()
    print(f"   ENST with >=1 domain: {len(enst2dom):,}")

    brain_ids = np.load(BRAIN_DIR / "brain_full_ids.npy", allow_pickle=True)
    brain_ids = np.array([str(x) for x in brain_ids])
    gene_names = np.load(BRAIN_DIR / "brain_full_gene_names.npy", allow_pickle=True)
    gene_names = np.array([str(x) for x in gene_names])
    N = len(brain_ids)
    print(f"[3] brain isoforms: {N:,}")

    # map each row -> domain set (None if unmappable/no hits)
    row_dom = [None] * N
    n_mapped = 0
    for i, bid in enumerate(brain_ids):
        if bid.startswith("ENST"):
            enst = bid.split(".")[0]
        elif bid.startswith("transcript"):
            enst = bid
        else:
            enst = name2enst.get(bid, "")
        if enst and enst in enst2dom:
            row_dom[i] = enst2dom[enst]
            n_mapped += 1
    print(f"   rows with domain evidence: {n_mapped:,} ({100*n_mapped/N:.1f}%)")

    preds = {name: np.load(path)[:, :N_GO] for name, path in MODELS.items()}
    for name, p in preds.items():
        assert p.shape[0] == N, f"{name} row mismatch"

    print("[4] 유전자별 isoform pair 구성 (도메인 정보 있는 것만)")
    g2idx = defaultdict(list)
    for i, g in enumerate(gene_names):
        if row_dom[i] is not None:
            g2idx[g].append(i)
    multi = {g: idx for g, idx in g2idx.items() if len(idx) >= 2}
    print(f"   multi-isoform genes w/ domain evidence: {len(multi):,}")

    rows = []
    for g, idx in multi.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                da, db = row_dom[i], row_dom[j]
                union = da | db
                if not union:
                    continue
                jacc_dist = 1.0 - len(da & db) / len(union)
                rec = {
                    "gene": g, "i": int(i), "j": int(j),
                    "id_i": brain_ids[i], "id_j": brain_ids[j],
                    "n_dom_i": len(da), "n_dom_j": len(db),
                    "domain_jaccard_dist": jacc_dist,
                }
                for name, p in preds.items():
                    rec[f"score_diff_{name}"] = float(np.mean(np.abs(p[i] - p[j])))
                rows.append(rec)
    print(f"   isoform pairs (union domain nonempty): {len(rows):,}")

    dd = np.array([r["domain_jaccard_dist"] for r in rows])
    print("\n[5] Spearman(domain_jaccard_dist, score_diff) — 도메인 증거가 점수 차이를 예측하는가")
    corr_results = {}
    for name in MODELS:
        sd = np.array([r[f"score_diff_{name}"] for r in rows])
        rho, p = stats.spearmanr(dd, sd)
        corr_results[name] = {"rho": float(rho), "p": float(p), "n": len(rows)}
        print(f"   {name:10s}: rho={rho:+.4f}  p={p:.3e}  (n={len(rows)})")

    print("\n[6] Case study — domain 완전히 다른데(jaccard_dist==1.0) v15d는 구별 못하고 v20b는 구별하는 쌍")
    strict = [r for r in rows if r["domain_jaccard_dist"] == 1.0 and r["n_dom_i"] >= 1 and r["n_dom_j"] >= 1]
    print(f"   완전 비중첩 도메인 쌍(둘다 도메인 보유, 공통 0개): {len(strict)}")
    cases = sorted(
        strict,
        key=lambda r: r["score_diff_v20b_w5"] - r["score_diff_v15d"],
        reverse=True,
    )[:10]
    for c in cases:
        print(f"   {c['gene']:10s} {c['id_i']:>14s} vs {c['id_j']:<14s} "
              f"domain_i={c['n_dom_i']} domain_j={c['n_dom_j']}  "
              f"v15d_diff={c['score_diff_v15d']:.4f}  v20b_w5_diff={c['score_diff_v20b_w5']:.4f}  "
              f"gain={c['score_diff_v20b_w5']-c['score_diff_v15d']:+.4f}")

    out = {
        "n_pairs_total": len(rows),
        "n_pairs_full_domain_mismatch": len(strict),
        "spearman": corr_results,
        "top_cases": cases,
    }
    with open(OUT_DIR / "domain_evidence_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[saved] {OUT_DIR/'domain_evidence_results.json'}")


if __name__ == "__main__":
    main()
