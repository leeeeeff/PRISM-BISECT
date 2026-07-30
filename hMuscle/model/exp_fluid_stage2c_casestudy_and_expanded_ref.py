"""
exp_fluid_stage2c_casestudy_and_expanded_ref.py
================================================
Combined case-study deep dive (Option A) + expanded reference-set
bio-validation (Option B).

Part A. Case study on top-5 GO- side-branch candidates
------------------------------------------------------
Genes: MYO1D, JPH2, ZCCHC10, ANKRD12, PRSS23
For each gene:
    - list ALL its isoforms in the pilot subset (not just the top-1)
    - fetch ORF length from top30k_isoforms.pep
    - compute per-isoform side-branch score in the winner bundle
    - report per-isoform layer trajectory, peak-L, GO annotation status
    - side-by-side table so the reader can see which isoform is the
      outlier vs which is the "normal" gene member

Part B. Expanded reference-set enrichment
-----------------------------------------
Original reference: UniProt curated 51 pairs (very small).
Expand with:
    (1) SwissProt gene-level GO annotations
        (../data/raw_data/data/annotations/swissprot_annotations.txt)
    (2) NCBI gene-level BP annotations
        (../data/raw_data/data/annotations/human_annotations_ncbi_bp.txt)
    (3) genes with >= 3 isoforms in the pilot (potential alternative
        splicing signal from Ensembl / Bambu itself)
Rerun Fisher enrichment with the expanded reference sets to check
whether the fluid framework's discoveries overlap ANY annotated
alternative-splicing biology at scale.
"""

import os, json, csv, time, gc
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ANNOT_DIR  = "../data/raw_data/data/annotations"
ID_DIR     = "../data/raw_data/data/id_lists"
UNIPROT_CSV = "../../reports/exp_g_uniprot/uniprot_isoform_benchmark_v2.csv"
SIDE_CSV   = "../../reports/fluid_stage2/all_side_branches_20260706_1957.csv"
PEP_FILE   = f"{DATA_DIR}/top30k_isoforms.pep"
OUT_DIR    = "../../reports/fluid_stage2"

N_LAYERS = 30
EMB_DIM  = 640
SEED     = 42
K_PCA    = 8
K_CLUS   = 16

CASE_GENES = ["MYO1D", "JPH2", "ZCCHC10", "ANKRD12", "PRSS23"]

# 34 GO catalog (identical to stage 2b)
GO_TERMS = {
    "GO:0006974": "DNA damage response", "GO:0035556": "Intracellular signal",
    "GO:0006508": "Proteolysis", "GO:0043161": "Proteasome-UPS",
    "GO:0006281": "DNA repair", "GO:0000226": "MT cytoskeleton org",
    "GO:0005975": "Carbohydrate metabolism", "GO:0055074": "Ca2+ homeostasis",
    "GO:0000165": "MAPK cascade", "GO:0000398": "mRNA splicing",
    "GO:0006417": "Regulation of translation", "GO:0007015": "Actin filament org",
    "GO:0007204": "Ca2+ signaling", "GO:0007059": "Chromosome segregation",
    "GO:0007265": "Ras signaling", "GO:0007018": "MT-based movement",
    "GO:0006816": "Ca2+ transport", "GO:0006888": "ER-Golgi transport",
    "GO:0006402": "mRNA catabolism", "GO:0006486": "Protein glycosylation",
    "GO:0006914": "Autophagy", "GO:0006470": "Dephosphorylation",
    "GO:0006836": "Neurotransmitter transp", "GO:0006414": "Translational elongation",
    "GO:0030048": "Actin-based movement", "GO:0032465": "Cytokinesis",
    "GO:0006906": "Vesicle fusion", "GO:0006418": "tRNA aminoacylation",
    "GO:0006754": "ATP biosynthesis", "GO:0006635": "FA beta-oxidation",
    "GO:0006120": "Complex I NADH ox", "GO:0045214": "Sarcomere organization",
    "GO:0006096": "Glycolysis", "GO:0006099": "TCA cycle",
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


def load_annot_genes(path):
    """Return set of gene symbols (first column) that appear in annot file."""
    genes = set()
    if not os.path.exists(path):
        return genes
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            g = parts[0].strip()
            # swissprot format uses "GENE_SPECIES" (e.g. SMYD2_MOUSE)
            # keep only "GENE_HUMAN" or human-derived entries
            if "_HUMAN" in g:
                g = g.split("_")[0]
                genes.add(g)
            elif "_" not in g:
                genes.add(g)
    return genes


def parse_pep_lengths():
    import re
    len_re = re.compile(r"len:(\d+)")
    lengths = {}
    cur = None; cur_len = None; cur_seq = 0
    with open(PEP_FILE) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    lengths[cur] = cur_len if cur_len is not None else cur_seq
                head = line[1:].split()[0]
                base = head.split(".p")[0]
                m = len_re.search(line)
                cur = base
                cur_len = int(m.group(1)) if m else None
                cur_seq = 0
            else:
                cur_seq += len(line.strip())
        if cur:
            lengths[cur] = cur_len if cur_len is not None else cur_seq
    return lengths


def build_trajectory(idx_subset):
    N = len(idx_subset)
    traj = np.empty((N, N_LAYERS, EMB_DIM), dtype=np.float32)
    for L in range(1, N_LAYERS + 1):
        p = f"{DATA_DIR}/esm2_layer_{L:02d}_t30_150M.npy"
        arr = np.load(p, mmap_mode="r")
        traj[:, L - 1, :] = np.asarray(arr[idx_subset], dtype=np.float32)
        del arr
    return traj


def load_go_pos(go):
    pos = set()
    with open(f"{ANNOT_DIR}/human_annotations_unified_bp.txt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and go in parts[1:]:
                pos.add(parts[0])
    return pos


def load_uniprot_genes():
    genes = set()
    if not os.path.exists(UNIPROT_CSV):
        return genes
    with open(UNIPROT_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            g = row.get("gene", "").strip()
            if g:
                genes.add(g)
    return genes


def main():
    t0 = time.time()
    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s = load_e2s()
    te_sym = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]
    print(f"[{time.time()-t0:5.1f}s] N_TE={len(te_iso)}")

    # ---- rebuild pilot (same as stage2b)
    go_pos = {go: load_go_pos(go) for go in GO_TERMS}
    all_pos = set().union(*go_pos.values())
    pos_idx = sorted({i for i, s in enumerate(te_sym) if s in all_pos})
    neg_pool = [i for i, s in enumerate(te_sym) if s not in all_pos]
    n_neg = min(len(pos_idx), len(neg_pool), 15000 - len(pos_idx))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(neg_pool, size=n_neg, replace=False).tolist()
    subset_idx = np.array(sorted(pos_idx + list(neg_idx)))
    N = len(subset_idx)
    sub_iso = [te_iso[i] for i in subset_idx]
    sub_sym = [te_sym[i] for i in subset_idx]
    print(f"[{time.time()-t0:5.1f}s] pilot N={N}")

    y = {go: np.array([1 if s in go_pos[go] else 0 for s in sub_sym],
                      dtype=np.int32) for go in GO_TERMS}
    baseline = {go: float(y[go].mean()) for go in GO_TERMS}

    # ---- rebuild traj + layer-norm PCA
    print(f"[{time.time()-t0:5.1f}s] building trajectory ...")
    traj = build_trajectory(subset_idx)
    layer_mean = traj.mean(axis=0)
    layer_std  = traj.std(axis=0) + 1e-6
    traj_norm = (traj - layer_mean) / layer_std
    del traj
    gc.collect()
    flat = traj_norm.reshape(N * N_LAYERS, EMB_DIM)
    K_MAX = 16
    pca = PCA(n_components=K_MAX, random_state=SEED, svd_solver="randomized")
    reduced = pca.fit_transform(flat).reshape(N, N_LAYERS, K_MAX)
    del flat
    gc.collect()
    cv_nm = reduced[:, :, :K_PCA].reshape(N, N_LAYERS * K_PCA)
    print(f"[{time.time()-t0:5.1f}s] cv_nm ready")

    km = KMeans(n_clusters=K_CLUS, n_init=5, random_state=SEED)
    cid = km.fit_predict(cv_nm)

    # ---- Part A: CASE STUDY
    print(f"\n[{time.time()-t0:5.1f}s] Part A: case study on top-5 GO- candidates")
    pep_len = parse_pep_lengths()
    def lkp_len(iso):
        return pep_len.get(iso, pep_len.get(iso.split(".")[0], None))

    # for each case gene, find all its isoforms in pilot + per-iso side-branch
    case_report = {}
    for gene in CASE_GENES:
        member_idx_all = [i for i, s in enumerate(sub_sym) if s == gene]
        if not member_idx_all:
            print(f"  {gene:12s}  NOT in pilot -- skipping")
            continue

        # for each isoform of this gene, compute per-bundle sb + peak_L
        iso_records = []
        for i_sub in member_idx_all:
            iso = sub_iso[i_sub]
            gene_i = te_gene[subset_idx[i_sub]]
            length = lkp_len(iso)
            cluster = int(cid[i_sub])
            # compute side-branch score in this cluster
            cluster_mask = (cid == cluster)
            bundle_mean = reduced[cluster_mask][:, :, :3].mean(axis=0)
            dev = np.linalg.norm(
                reduced[i_sub, :, :3] - bundle_mean, axis=1)
            sb   = float(dev.max() - dev[-1])
            peakL = int(dev.argmax()) + 1
            # which GOs this cluster wins
            wins = []
            for go in GO_TERMS:
                K_pos = int(y[go].sum())
                if K_pos == 0:
                    continue
                n_c = int(cluster_mask.sum())
                k_p = int(y[go][cluster_mask].sum())
                if k_p < 5:
                    continue
                frac = k_p / n_c
                ratio = frac / baseline[go] if baseline[go] > 0 else 0
                if ratio >= 2.0:
                    wins.append((go, ratio, k_p))
            # this iso's GO annotations across it
            iso_go_flags = {go: bool(y[go][i_sub]) for go in GO_TERMS}
            iso_records.append(dict(
                iso=iso, gene_ens=gene_i, length=length,
                cluster=cluster, sb=sb, peak_L=peakL,
                bundle_wins=[dict(go=g, ratio=r, k_p=k) for g, r, k in wins],
                go_annotated=[g for g in GO_TERMS if iso_go_flags[g]],
            ))
        # sort by sb descending
        iso_records.sort(key=lambda r: -r["sb"])
        case_report[gene] = dict(n_isoforms=len(iso_records),
                                 isoforms=iso_records)

        print(f"\n  == {gene} ==  n_iso_in_pilot={len(iso_records)}")
        print(f"     {'iso':<25} {'len':>5} {'clu':>4} {'sb':>7} "
              f"{'peakL':>5}  bundle_wins")
        for r in iso_records[:10]:
            wins_str = ",".join([f"{w['go']}(x{w['ratio']:.1f})"
                                 for w in r['bundle_wins'][:3]])
            print(f"     {r['iso']:<25} {str(r['length']):>5} "
                  f"{r['cluster']:>4} {r['sb']:>7.2f} "
                  f"{r['peak_L']:>5}  {wins_str}")

    # ---- Part B: EXPANDED REFERENCE SET
    print(f"\n[{time.time()-t0:5.1f}s] Part B: expanded reference-set validation")

    # load side-branch survey
    side_rows = []
    with open(SIDE_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            side_rows.append(row)
    side_gene_set = list({r["gene"] for r in side_rows})
    print(f"  side-branch unique genes: {len(side_gene_set)}")

    # load reference sets
    uniprot = load_uniprot_genes()
    swissprot = load_annot_genes(f"{ANNOT_DIR}/swissprot_annotations.txt")
    ncbi = load_annot_genes(f"{ANNOT_DIR}/human_annotations_ncbi_bp.txt")
    unified = load_annot_genes(f"{ANNOT_DIR}/human_annotations_unified_bp.txt")

    # multi-isoform genes: any gene with >= 3 isoforms in pilot
    sym_iso_count = Counter(sub_sym)
    multi_iso3 = {s for s, c in sym_iso_count.items() if c >= 3}
    multi_iso5 = {s for s, c in sym_iso_count.items() if c >= 5}

    # Ensembl gene-level annotation only counts entries in te_gene
    unified_test = {s for s in unified if s in sub_sym}

    # background = all pilot gene symbols
    bg_syms = list(set(sub_sym))

    # enrichment tester
    def enr(ref, tag):
        A = sum(1 for s in side_gene_set if s in ref)
        B = len(side_gene_set) - A
        C = sum(1 for s in bg_syms if s in ref) - A
        D = len(bg_syms) - len(side_gene_set) - C
        if C < 0: C = 0
        if D < 0: D = 0
        if (A + B) == 0 or (C + D) == 0:
            return None
        or_, p = fisher_exact([[A, B], [C, D]], alternative="greater")
        n_ref_in_pilot = sum(1 for s in bg_syms if s in ref)
        print(f"  {tag:35s}  hit={A:>4d}/{len(side_gene_set):>4d}  "
              f"ref_in_pilot={n_ref_in_pilot:>5d}  "
              f"OR={or_:.2f}  p={p:.3e}")
        return dict(hit=A, n_side=len(side_gene_set),
                    n_ref_in_pilot=int(n_ref_in_pilot),
                    odds_ratio=float(or_), p=float(p))

    print(f"  == expanded reference enrichment ==")
    print(f"  Reference sizes:  UniProt curated={len(uniprot)},  "
          f"SwissProt={len(swissprot)},  NCBI-BP={len(ncbi)},  "
          f"unified-BP={len(unified)}")
    print(f"  multi-iso (>=3) genes in pilot: {len(multi_iso3)},  "
          f"multi-iso (>=5): {len(multi_iso5)}")

    results = {}
    results["uniprot_curated"] = enr(uniprot, "UniProt curated (51 pairs)")
    results["swissprot"]       = enr(swissprot, "SwissProt annotated")
    results["ncbi_bp"]         = enr(ncbi, "NCBI BP annotated")
    results["unified_bp"]      = enr(unified_test, "unified-BP annotated (pilot restricted)")
    results["multi_iso3"]      = enr(multi_iso3, "Multi-iso >=3 in pilot")
    results["multi_iso5"]      = enr(multi_iso5, "Multi-iso >=5 in pilot")

    # combined (any known curated)
    combined = uniprot | swissprot
    results["curated_union"] = enr(combined, "UniProt ∪ SwissProt (curated union)")

    # ---- SAVE
    ts = time.strftime("%Y%m%d_%H%M")
    with open(f"{OUT_DIR}/casestudy_expanded_{ts}.json", "w") as f:
        json.dump(dict(
            case_report=case_report,
            expanded_enrichment=results,
            reference_sizes=dict(
                uniprot=len(uniprot), swissprot=len(swissprot),
                ncbi_bp=len(ncbi), unified_bp=len(unified),
                multi_iso3=len(multi_iso3), multi_iso5=len(multi_iso5),
                curated_union=len(combined),
            ),
        ), f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else o)
    print(f"\nsaved: {OUT_DIR}/casestudy_expanded_{ts}.json")
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
