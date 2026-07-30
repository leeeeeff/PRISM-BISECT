"""
exp_fluid_stage1d_length_confound.py
====================================
Investigate whether cluster c6 (TPM1 + TTN + ZNF proteins) is driven
primarily by sequence length rather than functional similarity.

Steps
-----
1.  Load protein sequences from top30k_isoforms.pep, extract per-isoform
    length (from FASTA header "len:NNN" or from actual sequence chars).
2.  Load cached curve_cluster npz (2926 pilot isoforms, 12 clusters,
    curve_vec_240).
3.  Length distribution per cluster + median/IQR + KS-test against
    global length distribution.
4.  Correlation of length with each of the 240 curve_vec dimensions +
    top-10 most length-correlated dims.
5.  Length residualization: fit linear model dim_i = a + b * length,
    subtract fit, re-KMeans on residualized 240-D vectors, compare c6
    composition (TPM1 / TTN / ZNF membership before vs after).
6.  Repeat check for winner clusters (c1, c3, c10, c11) to see if length
    also confounds them.
7.  Save: cluster length stats, correlation table, residualized
    cluster assignments, before/after diagnostic PNG.
"""

import os, json, glob, re
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = "../data"
ID_DIR     = "../data/raw_data/data/id_lists"
CACHE_DIR  = "../../reports/fluid_stage1"
OUT_DIR    = CACHE_DIR

PEP_FILE = f"{DATA_DIR}/top30k_isoforms.pep"


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


def parse_pep_lengths():
    """
    Parse top30k_isoforms.pep -> dict(iso_base_id -> length).
    Header form:  >BambuTx10.p1 GENE.BambuTx10~~BambuTx10.p1  ORF ...  len:688
    We store the isoform ID stripped of '.p1' variant suffix.
    """
    lengths = {}
    len_re = re.compile(r"len:(\d+)")
    cur_id = None
    cur_len = None
    cur_seq = 0
    with open(PEP_FILE) as f:
        for line in f:
            if line.startswith(">"):
                if cur_id and cur_len is None:
                    lengths[cur_id] = cur_seq
                elif cur_id:
                    lengths[cur_id] = cur_len
                # parse header
                head = line[1:].split()[0]      # e.g. BambuTx10.p1 or ENST00000...p1
                base = head.split(".p")[0]      # BambuTx10 / ENST00000...
                m = len_re.search(line)
                cur_id = base
                cur_len = int(m.group(1)) if m else None
                cur_seq = 0
            else:
                cur_seq += len(line.strip())
        if cur_id:
            lengths[cur_id] = cur_len if cur_len is not None else cur_seq
    return lengths


def main():
    # ---- cached pilot
    npz = sorted(glob.glob(f"{CACHE_DIR}/curve_cluster_*.npz"))[-1]
    d = np.load(npz)
    subset_idx = d["subset_idx"]
    cid        = d["cluster_id"]
    cv         = d["curve_vec_240"]
    print(f"cache: {npz}")
    print(f"  N={len(subset_idx)}, dim={cv.shape[1]}, n_clusters={len(set(cid))}")

    te_iso  = load_ids("my_isoform_list_fixed.npy")
    te_gene = load_ids("my_gene_list_fixed.npy")
    e2s     = load_ensg_to_symbol()
    te_sym  = [e2s.get(g.split(".")[0], g.split(".")[0]) for g in te_gene]

    sub_iso  = [te_iso[i]  for i in subset_idx]
    sub_sym  = [te_sym[i]  for i in subset_idx]

    # ---- length lookup
    print("parsing pep file for lengths ...")
    len_map = parse_pep_lengths()
    print(f"  parsed {len(len_map)} sequences")

    # pep keys are the ID *with* Ensembl version (ENST00000691057.1) or
    # bare BambuTxN — match te_iso IDs directly with a fallback that
    # strips the Ensembl version if not found.
    def lookup_len(x):
        if x in len_map:
            return len_map[x]
        base = x.split(".")[0]
        return len_map.get(base, np.nan)

    lengths = np.array([lookup_len(x) for x in sub_iso], dtype=np.float64)
    valid = ~np.isnan(lengths)
    n_valid = int(valid.sum())
    print(f"  matched {n_valid}/{len(sub_iso)} pilot isoforms to length")

    lengths_v = lengths[valid]
    cv_v      = cv[valid]
    cid_v     = cid[valid]
    sub_sym_v = [sub_sym[i] for i in np.where(valid)[0]]

    # ---- 1. Length distribution per cluster
    print("\n== per-cluster length distribution ==")
    per_cluster = {}
    for c in sorted(set(cid_v)):
        m = (cid_v == c)
        L = lengths_v[m]
        stat, pval = ks_2samp(L, lengths_v)
        per_cluster[int(c)] = dict(
            n=int(len(L)),
            median=float(np.median(L)),
            iqr=[float(np.percentile(L, 25)), float(np.percentile(L, 75))],
            mean=float(np.mean(L)),
            std=float(np.std(L)),
            ks_stat=float(stat),
            ks_p=float(pval),
        )
        print(f"  c{c:2d}  n={len(L):4d}  med={np.median(L):6.0f}  "
              f"IQR=[{np.percentile(L,25):5.0f},{np.percentile(L,75):5.0f}]"
              f"  mean={np.mean(L):6.0f}+-{np.std(L):5.0f}  "
              f"KS_p={pval:.2e}")

    global_stat = dict(
        median=float(np.median(lengths_v)),
        iqr=[float(np.percentile(lengths_v, 25)),
             float(np.percentile(lengths_v, 75))],
        mean=float(np.mean(lengths_v)),
        std=float(np.std(lengths_v)),
    )
    print(f"  GLOBAL  n={len(lengths_v)}  med={global_stat['median']:6.0f}"
          f"  IQR=[{global_stat['iqr'][0]:5.0f},{global_stat['iqr'][1]:5.0f}]"
          f"  mean={global_stat['mean']:6.0f}+-{global_stat['std']:5.0f}")

    # ---- 2. Length correlation per curve_vec dim
    print("\n== top length-correlated curve_vec dims (Spearman) ==")
    rhos = np.zeros(cv_v.shape[1])
    ps   = np.zeros(cv_v.shape[1])
    log_len = np.log10(lengths_v + 1)
    for j in range(cv_v.shape[1]):
        r, p = spearmanr(cv_v[:, j], log_len)
        rhos[j] = r
        ps[j]  = p
    order = np.argsort(-np.abs(rhos))
    for k in range(10):
        j = order[k]
        print(f"  dim {j:3d}  rho={rhos[j]:+.3f}  p={ps[j]:.2e}")

    max_abs_rho = float(np.abs(rhos).max())
    print(f"  max |rho| = {max_abs_rho:.3f}")

    # ---- 3. Length-explained variance in cluster ID (ANOVA-style R^2)
    from sklearn.metrics import r2_score
    # length -> cluster mean length prediction (categorical)
    cluster_mean_len = np.array([per_cluster[int(c)]["mean"] for c in cid_v])
    r2_len_by_cluster = r2_score(lengths_v, cluster_mean_len)
    print(f"\n  R^2 (cluster mean length predicting length) = "
          f"{r2_len_by_cluster:.3f}")

    # ---- 4. Length residualization
    print("\n== residualizing length from curve_vec + re-clustering ==")
    lr = LinearRegression()
    lr.fit(log_len.reshape(-1, 1), cv_v)
    cv_pred    = lr.predict(log_len.reshape(-1, 1))
    cv_residual = cv_v - cv_pred
    print(f"  fitted per-dim linear model (coef_norm={np.linalg.norm(lr.coef_):.3f})")

    km2 = KMeans(n_clusters=12, n_init=10, random_state=42)
    cid_res = km2.fit_predict(cv_residual)

    # ---- Cluster c6 tracking
    C6_TARGETS = {"TPM1", "TTN"}
    C6_ZNFS = {s for s in sub_sym_v if s.startswith("ZNF") or s.startswith("ZKSCAN")}
    print(f"\n== c6 composition before/after residualization ==")

    def dump_membership(cid_arr, label):
        # find which cluster contains TPM1 majority in this labeling
        cnts_by_cluster = {c: Counter() for c in range(12)}
        for c, s in zip(cid_arr, sub_sym_v):
            cnts_by_cluster[c][s] += 1
        # locate TPM1 dominant cluster
        tpm1_by_cluster = {c: cnts_by_cluster[c]["TPM1"] for c in range(12)}
        tpm1_top = max(tpm1_by_cluster, key=tpm1_by_cluster.get)
        ttn_by_cluster  = {c: cnts_by_cluster[c]["TTN"]  for c in range(12)}
        ttn_top  = max(ttn_by_cluster, key=ttn_by_cluster.get)
        print(f"  [{label}] TPM1-dominant cluster: c{tpm1_top} "
              f"({tpm1_by_cluster[tpm1_top]} TPM1 iso)")
        print(f"  [{label}] TTN-dominant cluster:  c{ttn_top} "
              f"({ttn_by_cluster[ttn_top]} TTN iso)")
        # ZNF fraction in TPM1-dominant cluster
        target_cluster_syms = [s for c, s in zip(cid_arr, sub_sym_v)
                               if c == tpm1_top]
        znf_frac = sum(1 for s in target_cluster_syms
                       if s.startswith("ZNF") or s.startswith("ZKSCAN")) \
                   / max(len(target_cluster_syms), 1)
        print(f"  [{label}] c{tpm1_top} n={len(target_cluster_syms)}  "
              f"ZNF/ZKSCAN fraction={znf_frac:.3f}  "
              f"median_length={np.median([lengths_v[i] for i in range(len(cid_arr)) if cid_arr[i] == tpm1_top]):.0f}")
        return dict(tpm1_cluster=int(tpm1_top),
                    ttn_cluster=int(ttn_top),
                    znf_frac=float(znf_frac),
                    n_members=int(len(target_cluster_syms)))

    before = dump_membership(cid_v, "before")
    after  = dump_membership(cid_res, "after")

    # ---- 5. Winner clusters length audit
    print("\n== winner-cluster length audit (before) ==")
    winner_audit = {}
    for wc in [1, 3, 10, 11]:
        m = (cid_v == wc)
        if m.sum() == 0:
            continue
        L = lengths_v[m]
        stat, p = ks_2samp(L, lengths_v)
        z = (np.median(L) - np.median(lengths_v)) / np.std(lengths_v)
        winner_audit[wc] = dict(
            n=int(m.sum()), median=float(np.median(L)),
            global_median=float(np.median(lengths_v)),
            z_median=float(z), ks_p=float(p),
        )
        print(f"  c{wc:2d}  n={m.sum():4d}  med={np.median(L):6.0f}  "
              f"z_med={z:+.2f}  KS_p={p:.2e}")

    # ---- 6. Diagnostic plot
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    # panel A: length dist per cluster
    labels = sorted(per_cluster.keys())
    box_data = [lengths_v[cid_v == c] for c in labels]
    axs[0].boxplot(box_data, labels=[f"c{c}" for c in labels], showfliers=False)
    axs[0].set_yscale("log")
    axs[0].axhline(np.median(lengths_v), color="red", lw=1, ls="--",
                   label=f"global median={np.median(lengths_v):.0f}")
    axs[0].set_ylabel("protein length (log)")
    axs[0].set_title("length distribution per cluster (before residualization)")
    axs[0].legend()
    # panel B: sorted |rho|
    axs[1].plot(np.sort(np.abs(rhos))[::-1], marker=".")
    axs[1].axhline(0.2, color="orange", ls="--", label="|rho|=0.2")
    axs[1].axhline(0.4, color="red", ls="--", label="|rho|=0.4")
    axs[1].set_xlabel("dim rank")
    axs[1].set_ylabel("|Spearman rho|")
    axs[1].set_title(f"length correlation per dim  (max={max_abs_rho:.3f})")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/length_confound_diag.png", dpi=140)
    plt.close(fig)

    # ---- save report
    report = dict(
        pilot_N=int(len(cid_v)),
        n_valid=n_valid,
        global_length=global_stat,
        per_cluster=per_cluster,
        winner_audit=winner_audit,
        length_curve_corr=dict(
            max_abs_rho=max_abs_rho,
            top10_dims=[dict(dim=int(j), rho=float(rhos[j]),
                             p=float(ps[j])) for j in order[:10]],
        ),
        cluster_R2_from_length=float(r2_len_by_cluster),
        before=before,
        after=after,
    )
    with open(f"{OUT_DIR}/length_confound.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {OUT_DIR}/length_confound.json")
    print(f"saved: {OUT_DIR}/length_confound_diag.png")


if __name__ == "__main__":
    main()
