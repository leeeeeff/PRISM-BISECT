"""
exp_fluid_stage1b_rep_isoforms.py
==================================
Representative isoform nomination for the 4 Stage-1 winner clusters
(c1 / c3 / c10 / c11 in the 3-GO pilot).

Steps
-----
1.  Load cached curve_cluster npz (subset_idx, cluster_id, curve_vec_240).
2.  Reconstruct te_iso, te_sym for the pilot subset from
    my_isoform_list_fixed.npy / my_gene_list_fixed.npy + ensembl_to_symbol.
3.  For each cluster of interest, compute cluster centroid in the
    240-D curve space, rank members by Euclidean distance to centroid.
4.  Emit top-15 nearest isoforms per cluster + gene symbol +
    BISECT-hit flag.
5.  Also emit gene-symbol frequency table per cluster (top-25 genes).

BISECT known hits (from Session 2026-06-23 / 2026-06-25 updates):
  NDUFS4, NDUFS7, NDUFS8, DOCK11, DLG1, LRPPRC, ZNF736, ERCC6L2, NDUFAF5
"""

import os, json, glob
import numpy as np
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "../data"
ID_DIR   = "../data/raw_data/data/id_lists"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
CACHE_DIR = "../../reports/fluid_stage1"
OUT_DIR   = CACHE_DIR

WINNER_CLUSTERS = {
    1:  ("Glycolysis winner (weakest)",  "GO:0006096"),
    3:  ("Muscle contraction winner",    "GO:0006941"),
    10: ("Glycolysis winner (mid)",      "GO:0006096"),
    11: ("Glycolysis winner (strongest)","GO:0006096"),
}
INTERESTING_CLUSTERS = {
    5:  ("Mitochondrion sub-mode",       "GO:0007005"),
    6:  ("Muscle-only sub-mode",         "GO:0006941"),
    8:  ("Muscle contraction sub-mode",  "GO:0006941"),
}

BISECT_HITS = {
    "NDUFS4", "NDUFS7", "NDUFS8", "DOCK11", "DLG1",
    "LRPPRC", "ZNF736", "ERCC6L2", "NDUFAF5", "KIF21B",
}
TARGET_GENES = {
    "TPM1", "TPM2", "TPM3", "KIF1B", "SEH1L", "GABARAPL1", "DMD", "OBSCN",
}


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def load_ensg_to_symbol():
    m = {}
    with open(f"{ID_DIR}/ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                m[p[0]] = p[4]
    return m


def load_go_pos(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return pos


def main():
    # ── load cache
    npz_path = sorted(glob.glob(f"{CACHE_DIR}/curve_cluster_*.npz"))[-1]
    d = np.load(npz_path)
    subset_idx = d["subset_idx"]
    cid        = d["cluster_id"]
    cv         = d["curve_vec_240"]     # (N, 240)
    print(f"Loaded {npz_path}")
    print(f"  N={len(subset_idx)}  clusters={len(set(cid))}  cv_dim={cv.shape[1]}")

    # ── reconstruct symbols
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s     = load_ensg_to_symbol()
    te_sym  = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    sub_iso  = [te_iso[i]  for i in subset_idx]
    sub_sym  = [te_sym[i]  for i in subset_idx]
    sub_gene = [te_gene[i] for i in subset_idx]

    # ── GO membership lookup
    go_pos = {}
    for _, go in {**WINNER_CLUSTERS, **INTERESTING_CLUSTERS}.values():
        if go not in go_pos:
            go_pos[go] = load_go_pos(go)

    # ── rep nomination
    report = {"clusters": {}, "meta": {"cache": npz_path,
                                       "bisect_hits": sorted(BISECT_HITS),
                                       "target_genes": sorted(TARGET_GENES)}}

    for cluster_dict, tag in [(WINNER_CLUSTERS, "winner"),
                              (INTERESTING_CLUSTERS, "interesting")]:
        for c, (label, go) in cluster_dict.items():
            mask = (cid == c)
            n_c = int(mask.sum())
            centroid = cv[mask].mean(axis=0)
            dists = np.linalg.norm(cv[mask] - centroid, axis=1)
            order = np.argsort(dists)

            member_idx = np.where(mask)[0]
            top = []
            for k in range(min(15, n_c)):
                i = member_idx[order[k]]
                sym  = sub_sym[i]
                iso  = sub_iso[i]
                gene = sub_gene[i]
                top.append(dict(
                    rank=k+1, dist=float(dists[order[k]]),
                    isoform=iso, gene=gene, symbol=sym,
                    is_go_pos=(sym in go_pos[go]),
                    is_bisect=(sym in BISECT_HITS),
                    is_target=(sym in TARGET_GENES),
                ))

            gene_counts = Counter([sub_sym[i] for i in member_idx])
            top_genes = gene_counts.most_common(25)

            n_bisect = sum(1 for s in [sub_sym[i] for i in member_idx]
                           if s in BISECT_HITS)
            n_target = sum(1 for s in [sub_sym[i] for i in member_idx]
                           if s in TARGET_GENES)

            report["clusters"][str(c)] = dict(
                tag=tag, label=label, target_go=go,
                n_members=n_c, n_bisect=n_bisect, n_target=n_target,
                bisect_symbols=sorted({s for s in
                                       [sub_sym[i] for i in member_idx]
                                       if s in BISECT_HITS}),
                target_symbols=sorted({s for s in
                                       [sub_sym[i] for i in member_idx]
                                       if s in TARGET_GENES}),
                top15_nearest=top,
                top25_genes=[[g, n] for g, n in top_genes],
            )

            print(f"\n── cluster c{c} ({tag}: {label}, target {go}) "
                  f"n={n_c} bisect={n_bisect} target={n_target}")
            print(f"   top15 nearest (rank | iso | gene | GO+ | BISECT | TARGET):")
            for t in top:
                flags = "".join(["G" if t["is_go_pos"]  else "-",
                                 "B" if t["is_bisect"]  else "-",
                                 "T" if t["is_target"]  else "-"])
                print(f"      {t['rank']:2d}  {t['isoform']:20s}"
                      f"  {t['symbol']:12s} [{flags}]  d={t['dist']:.3f}")
            print(f"   top10 genes in cluster:")
            for g, n in top_genes[:10]:
                mk = "B" if g in BISECT_HITS else ("T" if g in TARGET_GENES else " ")
                print(f"      [{mk}] {g:12s}  n={n}")

    with open(f"{OUT_DIR}/rep_isoforms.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/rep_isoforms.json")


if __name__ == "__main__":
    main()
