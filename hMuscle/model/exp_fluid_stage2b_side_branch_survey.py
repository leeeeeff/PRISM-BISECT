"""
exp_fluid_stage2b_side_branch_survey.py
========================================
All-bundle side-branch survey + biological validation.

Approach
--------
1.  Reconstruct Stage 2 pilot (34 specific BP terms, N ~ 15000) with
    layer-normalized joint PCA.
2.  Fixed KMeans (K=16, seed=42) on cv_nm.
3.  For each GO (34) find its best-ratio cluster (>= 2x baseline).
4.  For each bundle, compute:
      - bundle mean trajectory C(L) in top-3 PC axes
      - per-member per-layer deviation d(i, L) = ||x[i,L] - C(L)||
      - side-branch score sb(i) = max_L d(i, L) - d(i, L=30)
      - peak layer of divergence
5.  Take top-20 side-branch isoforms per bundle -> ~560 candidates.
6.  Bio-validation via cross-reference:
      - UniProt curated isoform-pair benchmark
        (reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv, 51 pairs)
      - BISECT hits (hard-coded)
      - TARGET_GENES (hard-coded)
7.  Enrichment: side-branch set gene overlap vs pilot-background gene
    frequency (Fisher exact per database).
8.  Report:
      - all_side_branches.csv (~ 20 * 34 rows, all bundles)
      - bio_validation_summary.json
      - top-50 hits + novel candidates
"""

import os, json, time, gc, csv
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_FILE = "../data/raw_data/data/annotations/human_annotations_unified_bp.txt"
ID_DIR     = "../data/raw_data/data/id_lists"
UNIPROT_CSV = "../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv"
OUT_DIR    = "../../reports/fluid_stage2"
os.makedirs(OUT_DIR, exist_ok=True)

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 8
K_CLUS   = 16
TOP_N    = 20            # side-branch per bundle

BISECT_HITS = {
    "NDUFS4", "NDUFS7", "NDUFS8", "DOCK11", "DLG1",
    "LRPPRC", "ZNF736", "ERCC6L2", "NDUFAF5", "NDUFAF6",
    "KIF21B", "PRSS12", "RBFOX1",
}
TARGET_GENES = {
    "TPM1", "TPM2", "TPM3", "KIF1B", "SEH1L", "GABARAPL1", "DMD", "OBSCN",
}

GO_TERMS = {
    "GO:0006974": "DNA damage response",
    "GO:0035556": "Intracellular signal",
    "GO:0006508": "Proteolysis",
    "GO:0043161": "Proteasome-UPS",
    "GO:0006281": "DNA repair",
    "GO:0000226": "MT cytoskeleton org",
    "GO:0005975": "Carbohydrate metabolism",
    "GO:0055074": "Ca2+ homeostasis",
    "GO:0000165": "MAPK cascade",
    "GO:0000398": "mRNA splicing",
    "GO:0006417": "Regulation of translation",
    "GO:0007015": "Actin filament org",
    "GO:0007204": "Ca2+ signaling",
    "GO:0007059": "Chromosome segregation",
    "GO:0007265": "Ras signaling",
    "GO:0007018": "MT-based movement",
    "GO:0006816": "Ca2+ transport",
    "GO:0006888": "ER-Golgi transport",
    "GO:0006402": "mRNA catabolism",
    "GO:0006486": "Protein glycosylation",
    "GO:0006914": "Autophagy",
    "GO:0006470": "Dephosphorylation",
    "GO:0006836": "Neurotransmitter transp",
    "GO:0006414": "Translational elongation",
    "GO:0030048": "Actin-based movement",
    "GO:0032465": "Cytokinesis",
    "GO:0006906": "Vesicle fusion",
    "GO:0006418": "tRNA aminoacylation",
    "GO:0006754": "ATP biosynthesis",
    "GO:0006635": "FA beta-oxidation",
    "GO:0006120": "Complex I NADH ox",
    "GO:0045214": "Sarcomere organization",
    "GO:0006096": "Glycolysis",
    "GO:0006099": "TCA cycle",
}


def load_ids(p):
    a = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def load_e2s():
    m = {}
    with open(f"{ID_DIR}/ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                m[p[0]] = p[4]
    return m


def load_go_pos(go):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return pos


def build_trajectory(idx_subset):
    N = len(idx_subset)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = f"{DATA_DIR}/esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = np.asarray(arr[idx_subset], dtype=np.float32)
        del arr
    return traj


def load_uniprot_genes():
    genes = set()
    if not os.path.exists(UNIPROT_CSV):
        return genes
    with open(UNIPROT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get("gene", "").strip()
            if g:
                genes.add(g)
    return genes


def fisher_enrichment(hit_syms_side, hit_syms_bg, ref_set):
    """
    2x2 table: (in ref, side) vs (in ref, bg-only) vs (not-in-ref, side) vs (not-in-ref, bg-only)
    Returns odds ratio + p-value.
    """
    A = sum(1 for s in hit_syms_side if s in ref_set)      # in ref, in side
    B = len(hit_syms_side) - A                             # in side, not in ref
    C = sum(1 for s in hit_syms_bg if s in ref_set) - A    # in ref, in bg only
    D = len(hit_syms_bg) - len(hit_syms_side) - C          # not-in-ref, bg only
    if C < 0: C = 0
    if D < 0: D = 0
    if A + B == 0 or C + D == 0:
        return float("nan"), float("nan"), A, len(hit_syms_side)
    or_, p = fisher_exact([[A, B], [C, D]], alternative="greater")
    return float(or_), float(p), A, len(hit_syms_side)


def main():
    t0 = time.time()

    # ---- IDs and labels
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    go_pos = {go: load_go_pos(go) for go in GO_TERMS}
    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool), 15000 - len(pos_idx))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_iso  = [te_iso[i]  for i in subset_idx]
    sub_sym  = [te_sym[i]  for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot N={N}")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=np.int32) for go in GO_TERMS}
    baseline = {go: float(y[go].mean()) for go in GO_TERMS}

    # ---- trajectory
    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    print(f"[{time.time()-t0:5.1f}s] traj {traj.shape}")

    # ---- layer normalization + PCA
    layer_mean = traj.mean(axis=0)
    layer_std  = traj.std(axis=0) + 1e-6
    traj_norm  = (traj - layer_mean) / layer_std
    del traj
    gc.collect()
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    K_MAX = 16
    pca = PCA(n_components=K_MAX, random_state=SEED, svd_solver="randomized")
    reduced = pca.fit_transform(flat).reshape(N, N_LAYERS, K_MAX)
    del flat
    gc.collect()
    cv_nm = reduced[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    print(f"[{time.time()-t0:5.1f}s] cv_nm ready, expl_var(K={K_MAX})="
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # ---- KMeans reference clustering
    km = KMeans(n_clusters=K_CLUS, n_init=5, random_state=SEED)
    cid = km.fit_predict(cv_nm)
    print(f"[{time.time()-t0:5.1f}s] KMeans k={K_CLUS} done")

    # ---- per-GO best cluster
    bundle_map = {}
    for go in GO_TERMS:
        K_pos = int(y[go].sum())
        if K_pos == 0:
            continue
        best_c = None; best_ratio = 0; best_frac = 0
        for c in range(K_CLUS):
            m = (cid == c)
            n_c = int(m.sum())
            if n_c < 10:
                continue
            k_p = int(y[go][m].sum())
            if k_p == 0:
                continue
            frac = k_p / n_c
            ratio = frac / baseline[go] if baseline[go] > 0 else 0
            if ratio > best_ratio and k_p >= 5:
                best_ratio = ratio
                best_c = c
                best_frac = frac
        if best_c is not None and best_ratio >= 2.0:
            bundle_map[go] = dict(cluster=best_c, ratio=float(best_ratio),
                                  frac=float(best_frac),
                                  n_members=int((cid == best_c).sum()),
                                  K_pos=K_pos)

    print(f"[{time.time()-t0:5.1f}s] winner bundles (ratio >= 2x): "
          f"{len(bundle_map)}/{len(GO_TERMS)}")

    # ---- side-branch scoring per bundle
    all_hits = []
    for go, b in bundle_map.items():
        c = b["cluster"]
        member_idx = np.where(cid == c)[0]
        # bundle mean in top-3 PC axes
        bundle_mean = reduced[member_idx][:, :, :3].mean(axis=0)   # (30, 3)
        # per-iso per-layer deviation
        dev = np.linalg.norm(
            reduced[member_idx][:, :, :3] - bundle_mean[None, :, :],
            axis=2)                                                  # (n_c, 30)
        sb = dev.max(axis=1) - dev[:, -1]
        peak_L = dev.argmax(axis=1) + 1
        order = np.argsort(-sb)
        for k in range(min(TOP_N, len(order))):
            i_sub = member_idx[order[k]]
            all_hits.append(dict(
                go=go, go_name=GO_TERMS[go],
                cluster=int(c),
                rank=k + 1,
                iso=sub_iso[i_sub],
                gene=sub_sym[i_sub],
                sb_score=float(sb[order[k]]),
                peak_layer=int(peak_L[order[k]]),
                is_go_pos=bool(y[go][i_sub]),
            ))

    print(f"[{time.time()-t0:5.1f}s] side-branch candidates: {len(all_hits)}")

    # ---- bio-validation
    uniprot_genes = load_uniprot_genes()
    print(f"[{time.time()-t0:5.1f}s] UniProt curated genes: {len(uniprot_genes)}")

    hit_syms = list({h["gene"] for h in all_hits})
    bg_syms  = sub_sym
    all_bg_unique = list(set(bg_syms))

    def enrich(ref, tag):
        or_, p, n_hit, n_side = fisher_enrichment(hit_syms, all_bg_unique, ref)
        n_ref_in_pilot = sum(1 for s in all_bg_unique if s in ref)
        print(f"  {tag:15s}  hit={n_hit:>3d}/{n_side:>3d}  "
              f"ref_in_pilot={n_ref_in_pilot:>3d}  OR={or_:.2f}  p={p:.3e}")
        return dict(n_hit=int(n_hit), n_side=int(n_side),
                    n_ref_in_pilot=int(n_ref_in_pilot),
                    odds_ratio=float(or_), p=float(p))

    print(f"\n== Bio-validation enrichment ==")
    print(f"unique side-branch genes: {len(hit_syms)}")
    enr = {}
    enr["uniprot"] = enrich(uniprot_genes, "UniProt curated")
    enr["bisect"]  = enrich(BISECT_HITS,   "BISECT hits")
    enr["target"]  = enrich(TARGET_GENES,  "TARGET genes")
    combined = uniprot_genes | BISECT_HITS | TARGET_GENES
    enr["combined"] = enrich(combined,      "any known")

    # ---- per-type breakdown (attach type from earlier run if available)
    prev_json = sorted(
        [f for f in os.listdir(OUT_DIR) if f.startswith("typed_flow_")])
    go_type_map = {}
    if prev_json:
        prev = json.load(open(f"{OUT_DIR}/{prev_json[-1]}"))
        go_type_map = prev.get("go_type", {})
    per_type = defaultdict(lambda: {"hits": [], "any_known": 0})
    for h in all_hits:
        tp = go_type_map.get(h["go"], "unknown")
        h["type"] = tp
        per_type[tp]["hits"].append(h)
        if h["gene"] in combined:
            per_type[tp]["any_known"] += 1

    print(f"\n== per-type side-branch hit-rate ==")
    for tp, r in per_type.items():
        n_h = len(r["hits"])
        rate = r["any_known"] / n_h if n_h else 0
        print(f"  {tp:8s}  n_candidates={n_h:>4d}  "
              f"any_known={r['any_known']:>3d}  rate={rate:.3f}")

    # ---- top-50 hits table + novel-only table
    all_hits.sort(key=lambda x: -x["sb_score"])
    ts = time.strftime("%Y%m%d_%H%M")

    with open(f"{OUT_DIR}/all_side_branches_{ts}.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["go", "go_name", "type", "cluster", "rank",
                    "iso", "gene", "sb_score", "peak_layer",
                    "is_go_pos", "in_uniprot", "in_bisect", "in_target"])
        for h in all_hits:
            w.writerow([h["go"], h["go_name"], h.get("type", "-"),
                        h["cluster"], h["rank"],
                        h["iso"], h["gene"],
                        f"{h['sb_score']:.3f}",
                        h["peak_layer"], int(h["is_go_pos"]),
                        int(h["gene"] in uniprot_genes),
                        int(h["gene"] in BISECT_HITS),
                        int(h["gene"] in TARGET_GENES)])

    print(f"\n== TOP-25 SIDE-BRANCH HITS (by sb_score) ==")
    print(f"  {'rank':>4}  {'gene':<12}  {'sb':>7}  {'peakL':>5}  "
          f"{'GO':<11}  {'type':<5}  {'uni':>3}  {'BIS':>3}  {'TGT':>3}  "
          f"{'GO+':>3}  {'iso'}")
    for k, h in enumerate(all_hits[:25]):
        marks = "".join([
            "U" if h["gene"] in uniprot_genes else "-",
            "B" if h["gene"] in BISECT_HITS else "-",
            "T" if h["gene"] in TARGET_GENES else "-",
        ])
        g = "+" if h["is_go_pos"] else "-"
        print(f"  {k+1:>4d}  {h['gene']:<12}  "
              f"{h['sb_score']:>7.2f}  {h['peak_layer']:>5d}  "
              f"{h['go']:<11}  {h.get('type','?')[:5]:<5}  "
              f"{marks[0]:>3}  {marks[1]:>3}  {marks[2]:>3}  "
              f"{g:>3}  {h['iso']}")

    # save summary
    with open(f"{OUT_DIR}/side_branch_survey_{ts}.json", "w") as f:
        json.dump(dict(
            pilot_N=int(N),
            n_bundles=len(bundle_map),
            n_candidates=len(all_hits),
            unique_side_branch_genes=len(hit_syms),
            enrichment=enr,
            per_type=dict((tp, dict(n=len(r["hits"]),
                                    any_known=r["any_known"]))
                          for tp, r in per_type.items()),
            top50=all_hits[:50],
            bundle_map={go: b for go, b in bundle_map.items()},
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nsaved: {OUT_DIR}/all_side_branches_{ts}.csv")
    print(f"saved: {OUT_DIR}/side_branch_survey_{ts}.json")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
