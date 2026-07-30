"""
analyze_convdiv_v2.py
=====================
v2: divergent case를 재정의.

[CONV] 다른 gene + 같은 GO
       L1(다른 서열) 큼 → L30(같은 기능 subspace) 작음.

[DIV]  다른 gene + 다른 GO  (NEW)
       L1(비슷한 서열) 작음 → L30(다른 기능 subspace) 큼.
       할당 GO 교집합=∅ 을 요구하여 "실제로 다른 GO로 발산"함을 보장.

v1은 same-gene isoform을 div로 정의했으나 gene-level GO annotation 특성상
할당 GO가 반드시 동일해져 "divergent → different GO" 의미가 훼손됨.
"""
from __future__ import annotations

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from itertools import combinations

SEED = 42
rng  = np.random.default_rng(SEED)

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
DATA      = ROOT / "data"
ID_DIR    = DATA / "raw_data/data/id_lists"
ANNOT_DIR = DATA / "raw_data/data/annotations"
CACHE_DIR = ROOT.parent / "reports" / "v20_cache"
OUT_DIR   = ROOT.parent / "reports" / "curve_sweep"

GO_18 = {
    "GO:0007204": "Ca2+ signaling",       "GO:0045214": "Sarcomere org",
    "GO:0006941": "Muscle contract",      "GO:0006914": "Autophagy",
    "GO:0043161": "Proteasome-UPS",       "GO:0007519": "Skeletal musc dev",
    "GO:0042692": "Muscle cell diff",     "GO:0055074": "Ca2+ homeostasis",
    "GO:0007005": "Mitochondrion org",    "GO:0007517": "Muscle organ dev",
    "GO:0032006": "TOR signaling",        "GO:0030048": "Actin-based mov",
    "GO:0006096": "Glycolysis",           "GO:0007268": "Synaptic transm",
    "GO:0007018": "MT-based mov",         "GO:0031175": "Neuron proj dev",
    "GO:0030182": "Neuron diff",          "GO:0000226": "MT cytosk org",
}
MID_GOs = {"GO:0007204", "GO:0007018", "GO:0000226"}

MAX_PAIRS_KEEP  = 6


def load_all():
    print("[load] IDs, Z, labels...")
    te_gene = np.load(MODEL_DIR / "my_gene_list_fixed.npy", allow_pickle=True)
    te_iso  = np.load(MODEL_DIR / "my_isoform_list_fixed.npy", allow_pickle=True)
    ENSG2SYM = {}
    with open(ID_DIR / "ensembl_to_symbol.txt") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

    def clean(raw):
        s = str(raw)
        for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
        return s

    sym_te = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
              for g in te_gene]
    iso_te = [clean(i) for i in te_iso]

    Z = np.load(CACHE_DIR / "Z_te.npy")   # (N, 30, 8)
    print(f"  Z: {Z.shape}, N_iso: {len(iso_te)}")

    go_labels = {}
    for go in GO_18:
        pos = set()
        with open(ANNOT_DIR / "human_annotations_unified_bp.txt") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1 and go in parts[1:]:
                    pos.add(parts[0])
        go_labels[go] = np.array([s in pos for s in sym_te], dtype=bool)
    return Z, sym_te, iso_te, go_labels


def pairwise_dist_at_layer(Z_subset, L):
    X = Z_subset[:, L, :]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return D


def convergence_layer(d_seq, threshold_ratio=0.5):
    if d_seq[0] < 1e-6: return -1
    target = d_seq[0] * threshold_ratio
    for L in range(1, 30):
        if d_seq[L] < target: return L + 1
    return -1


def divergence_layer(d_seq, threshold_ratio=2.0):
    if d_seq[0] < 1e-6: return -1
    target = d_seq[0] * threshold_ratio
    for L in range(1, 30):
        if d_seq[L] > target: return L + 1
    return -1


def iso_gos(i, go_labels):
    return set(go for go, m in go_labels.items() if m[i])


# ── CONV: different genes, same GO ─────────────────────────────────

def find_convergent_pairs(Z, sym_te, iso_te, go_labels):
    print("\n[A] Searching convergent pairs (diff-gene, same-GO)...")
    candidates = []
    SAMPLE_PER_GO = 40

    for go, name in GO_18.items():
        mask = go_labels[go]
        pos_idx = np.where(mask)[0]
        if len(pos_idx) < 10: continue

        gene_to_isos = defaultdict(list)
        for i in pos_idx:
            gene_to_isos[sym_te[i]].append(i)
        distinct_gene_isos = [ivs[0] for ivs in gene_to_isos.values()]
        if len(distinct_gene_isos) < 10: continue

        n_samp = min(SAMPLE_PER_GO, len(distinct_gene_isos))
        samp_idx = rng.choice(distinct_gene_isos, size=n_samp, replace=False)
        Z_sub = Z[samp_idx]

        D_L1  = pairwise_dist_at_layer(Z_sub, 0)
        D_L30 = pairwise_dist_at_layer(Z_sub, 29)

        for a, b in combinations(range(n_samp), 2):
            d1 = D_L1[a, b]; d30 = D_L30[a, b]
            if d1 < 8.0 or d30 > 6.0: continue
            score = d1 - d30
            i_a, i_b = samp_idx[a], samp_idx[b]
            d_seq = np.sqrt(((Z[i_a] - Z[i_b]) ** 2).sum(-1))
            conv_L = convergence_layer(d_seq, 0.5)

            candidates.append({
                "go": go, "go_name": name,
                "is_mid": go in MID_GOs,
                "i_a": int(i_a), "i_b": int(i_b),
                "gene_a": sym_te[i_a], "gene_b": sym_te[i_b],
                "iso_a": iso_te[i_a], "iso_b": iso_te[i_b],
                "d_L1": float(d1), "d_L30": float(d30),
                "score": float(score),
                "conv_layer": int(conv_L),
                "d_seq": d_seq.tolist(),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen = set(); top = []
    for c in candidates:
        key = tuple(sorted([c["gene_a"], c["gene_b"]]))
        if key in seen: continue
        seen.add(key); top.append(c)
        if len(top) >= MAX_PAIRS_KEEP: break
    print(f"  Found {len(candidates)}; top {len(top)} kept")
    return top


# ── DIV (v2): different genes, DIFFERENT GOs ───────────────────────

def find_divergent_pairs_v2(Z, sym_te, iso_te, go_labels):
    """
    Redefinition of divergent case:
      - Pair from DIFFERENT genes
      - L1 embedding SMALL (similar starting embedding) — d_L1 < 5.0
      - L30 embedding LARGE (functional divergence) — d_L30 > 15.0
      - Their assigned GO sets DO NOT OVERLAP within GO_18 (different GO)

    Rank by (d_L30 - d_L1). Dedup by primary GO pair to get diverse examples.
    """
    print("\n[B] Searching divergent pairs (diff-gene, diff-GO)...")
    # Build isoform → GO set within 18-GO panel
    iso_gosets = [iso_gos(i, go_labels) for i in range(Z.shape[0])]

    # Sample candidate isoforms with at least one 18-GO label,
    # from as many distinct genes as possible
    with_any_go = [i for i in range(Z.shape[0]) if iso_gosets[i]]
    print(f"  isoforms with any 18-GO label: {len(with_any_go)}")

    # Restrict to one representative per gene (first occurrence)
    seen_gene = set(); rep_idxs = []
    for i in with_any_go:
        g = sym_te[i]
        if g in seen_gene: continue
        seen_gene.add(g); rep_idxs.append(i)
    print(f"  distinct-gene representatives: {len(rep_idxs)}")

    # Compute pairwise L1/L30 for all reps (subsample to keep this tractable)
    N_SUB = min(2500, len(rep_idxs))
    if len(rep_idxs) > N_SUB:
        rep_idxs = list(rng.choice(rep_idxs, size=N_SUB, replace=False))
    Z_sub = Z[rep_idxs]

    D_L1  = pairwise_dist_at_layer(Z_sub, 0)
    D_L30 = pairwise_dist_at_layer(Z_sub, 29)

    candidates = []
    n = len(rep_idxs)
    for a in range(n):
        for b in range(a + 1, n):
            d1  = D_L1[a, b]
            d30 = D_L30[a, b]
            if d1 > 5.0 or d30 < 15.0: continue
            i_a, i_b = rep_idxs[a], rep_idxs[b]
            # Enforce different GO annotation within 18-GO panel
            ga, gb = iso_gosets[i_a], iso_gosets[i_b]
            if not ga or not gb: continue
            if ga & gb: continue     # any overlap → skip

            score = d30 - d1
            d_seq = np.sqrt(((Z[i_a] - Z[i_b]) ** 2).sum(-1))
            div_L = divergence_layer(d_seq, 2.0)

            candidates.append({
                "i_a": int(i_a), "i_b": int(i_b),
                "gene_a": sym_te[i_a], "gene_b": sym_te[i_b],
                "iso_a": iso_te[i_a], "iso_b": iso_te[i_b],
                "gos_a": sorted(ga),
                "gos_b": sorted(gb),
                "d_L1": float(d1), "d_L30": float(d30),
                "score": float(score),
                "div_layer": int(div_L),
                "d_seq": d_seq.tolist(),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Dedup by (gene_a, gene_b) pair and primary-GO pair to get diverse examples
    seen_gene_pair = set()
    seen_go_pair   = set()
    top = []
    for c in candidates:
        gpair = tuple(sorted([c["gene_a"], c["gene_b"]]))
        if gpair in seen_gene_pair: continue
        go_pair = tuple(sorted([c["gos_a"][0], c["gos_b"][0]]))
        if go_pair in seen_go_pair: continue
        seen_gene_pair.add(gpair); seen_go_pair.add(go_pair)
        top.append(c)
        if len(top) >= MAX_PAIRS_KEEP: break

    print(f"  Found {len(candidates)}; top {len(top)} kept")
    return top


def main():
    Z, sym_te, iso_te, go_labels = load_all()

    conv_pairs = find_convergent_pairs(Z, sym_te, iso_te, go_labels)
    div_pairs  = find_divergent_pairs_v2(Z, sym_te, iso_te, go_labels)

    summary = {
        "convergent_top": [{k: v for k, v in c.items() if k != "d_seq"}
                           for c in conv_pairs],
        "divergent_top":  [{k: v for k, v in c.items() if k != "d_seq"}
                           for c in div_pairs],
    }
    with open(OUT_DIR / "convergence_divergence.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {OUT_DIR}/convergence_divergence.json  "
          f"({len(conv_pairs)} conv + {len(div_pairs)} div)")

    print("\nCONV top:")
    for i, c in enumerate(conv_pairs, 1):
        print(f"  #{i} GO={c['go_name']:20s} {c['gene_a']:>10s} vs {c['gene_b']:<10s}  "
              f"d(L1)={c['d_L1']:.2f}→d(L30)={c['d_L30']:.2f}  conv@L{c['conv_layer']}")

    print("\nDIV top:")
    for i, d in enumerate(div_pairs, 1):
        ga_name = GO_18[d['gos_a'][0]]; gb_name = GO_18[d['gos_b'][0]]
        print(f"  #{i} {d['gene_a']:>10s} ({ga_name:20s}) vs {d['gene_b']:<10s} ({gb_name})  "
              f"d(L1)={d['d_L1']:.2f}→d(L30)={d['d_L30']:.2f}  div@L{d['div_layer']}")


if __name__ == "__main__":
    main()
