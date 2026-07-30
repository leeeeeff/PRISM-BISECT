#!/usr/bin/env python3
"""
exp_brain_muscle_variance_structure.py
======================================
User claim (2026-07-15): brain vs muscle differ in representation geometry —
 (1) gene-family separation is CLEARER in brain, and
 (2) within-gene isoform variance is LARGER in brain.

These are DIFFERENT quantities (separability = supervised discriminability;
within-gene variance = unsupervised variance fraction). Both can hold at once.

This script measures the tissue-comparable part cleanly: two-level nested variance
decomposition (between-gene vs within-gene) on the SAME representation (raw
delta_layer = L30 - L15), for muscle and brain, controlling for the singleton-gene
composition confound (brain full has many novel single-isoform genes).

within-gene fraction:  SS_within / SS_total   pooled over 640 dims.
Reported on: (a) ALL genes, (b) MULTI-ISOFORM genes only (the honest apples-to-apples).
Also reports the between-gene variance magnitude and a simple separability proxy
(between-gene / within-gene variance ratio among multi-iso genes = F-like statistic).
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
ID_DIR = DATA / 'raw_data/data/id_lists'
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
OUT.mkdir(parents=True, exist_ok=True)


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def load_tissue(name, l15_path, l30_path, gene_path):
    x15 = np.load(l15_path).astype(np.float32)
    x30 = np.load(l30_path).astype(np.float32)
    delta = x30 - x15                      # raw delta_layer
    genes = np.array([clean(g) for g in np.load(gene_path, allow_pickle=True)])
    assert len(genes) == delta.shape[0], f"{name}: {len(genes)} genes vs {delta.shape[0]} rows"
    return delta, genes


def decompose(X, genes, multi_only=False, exact_n=None):
    """two-level nested variance: between-gene vs within-gene, pooled over dims.
    z-score per-dim first so all dims contribute comparably.
    exact_n: if set, restrict to genes with EXACTLY exact_n isoforms (cardinality control)."""
    gl, gidx = np.unique(genes, return_inverse=True)
    G = len(gl)
    gcnt = np.bincount(gidx, None, G).astype(float)
    if exact_n is not None:
        mask = gcnt[gidx] == exact_n
        X = X[mask]; genes_m = genes[mask]
        gl, gidx = np.unique(genes_m, return_inverse=True)
        G = len(gl); gcnt = np.bincount(gidx, None, G).astype(float)
    elif multi_only:
        multi_mask = gcnt[gidx] >= 2          # rows in multi-iso genes
        X = X[multi_mask]
        genes_m = genes[multi_mask]
        gl, gidx = np.unique(genes_m, return_inverse=True)
        G = len(gl)
        gcnt = np.bincount(gidx, None, G).astype(float)
    # z-score per dim (comparable scale)
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Z = (X - mu) / sd
    N, D = Z.shape
    grand = Z.mean(0)
    ss_tot = float(((Z - grand) ** 2).sum())
    # gene means
    gmean = np.zeros((G, D), np.float64)
    np.add.at(gmean, gidx, Z)
    gmean /= gcnt[:, None]
    ss_within = float(((Z - gmean[gidx]) ** 2).sum())
    ss_between = ss_tot - ss_within
    within_frac = ss_within / ss_tot
    between_frac = ss_between / ss_tot
    # F-like separability: MS_between / MS_within (df: between=G-1, within=N-G)
    df_b = max(G - 1, 1); df_w = max(N - G, 1)
    ms_b = ss_between / df_b; ms_w = ss_within / df_w
    f_like = ms_b / ms_w
    n_multi = int((gcnt >= 2).sum())
    return dict(N=int(N), G=int(G), n_multi_genes=n_multi,
                within_frac=within_frac, between_frac=between_frac,
                f_like_separability=f_like,
                mean_iso_per_gene=float(N / G))


def main():
    print("Loading muscle (esm2_train_human) ...")
    m_delta, m_genes = load_tissue(
        'muscle',
        DATA / 'esm2_train_human_layer15_t30_150M.npy',
        DATA / 'esm2_train_human_layer30_t30_150M.npy',
        ID_DIR / 'train_gene_list.npy')
    print(f"  muscle: {m_delta.shape}, {len(np.unique(m_genes))} genes")

    print("Loading brain (brain_full) ...")
    b_delta, b_genes = load_tissue(
        'brain',
        BRAIN / 'brain_full_esm2_layer15_t30_150M.npy',
        BRAIN / 'brain_full_esm2_layer30_t30_150M.npy',
        BRAIN / 'brain_full_gene_names.npy')
    print(f"  brain: {b_delta.shape}, {len(np.unique(b_genes))} genes")

    res = {}
    for tissue, X, genes in [('muscle', m_delta, m_genes), ('brain', b_delta, b_genes)]:
        res[tissue] = {
            'all_genes': decompose(X, genes, multi_only=False),
            'multi_iso_only': decompose(X, genes, multi_only=True),
            'exact_2iso': decompose(X, genes, exact_n=2),   # cardinality-controlled
            'exact_3iso': decompose(X, genes, exact_n=3),
        }
        print(f"\n=== {tissue.upper()} (raw delta_layer L30-L15, z-scored) ===")
        for k, d in res[tissue].items():
            print(f"  [{k}] N={d['N']} G={d['G']} multi={d['n_multi_genes']} "
                  f"iso/gene={d['mean_iso_per_gene']:.2f}")
            print(f"      within_frac={d['within_frac']:.4f}  between_frac={d['between_frac']:.4f}  "
                  f"F-like(sep)={d['f_like_separability']:.3f}")

    # head-to-head on the honest comparison (multi-iso only)
    mm = res['muscle']['multi_iso_only']; bb = res['brain']['multi_iso_only']
    res['_meta'] = {
        'claim2_within_gene_variance': {
            'muscle_within_frac': mm['within_frac'],
            'brain_within_frac': bb['within_frac'],
            'brain_minus_muscle': bb['within_frac'] - mm['within_frac'],
            'supports_claim2_brain_larger': bool(bb['within_frac'] > mm['within_frac']),
        },
        'claim1_gene_separability_proxy': {
            'muscle_F_like': mm['f_like_separability'],
            'brain_F_like': bb['f_like_separability'],
            'brain_minus_muscle': bb['f_like_separability'] - mm['f_like_separability'],
            'supports_claim1_brain_clearer': bool(bb['f_like_separability'] > mm['f_like_separability']),
            'note': 'F-like = MS_between/MS_within = gene-identity separability in raw delta space '
                    '(NOT family-level; family map not applied here). Also see gene-mean oracle AUPRC '
                    'brain 0.811 vs muscle 0.803 from prior runs.',
        },
        'caveat': 'raw delta_layer, z-scored per dim; multi_iso_only controls singleton composition. '
                  'F-like conflates cardinality (G,N differ) — read within_frac as primary, F-like as directional.',
    }
    (OUT / 'variance_structure.json').write_text(json.dumps(res, indent=2))
    print(f"\nSaved -> {OUT / 'variance_structure.json'}")
    print("\n--- HEAD TO HEAD (multi-iso only) ---")
    print(f" within_frac  muscle={mm['within_frac']:.4f}  brain={bb['within_frac']:.4f}  "
          f"(brain-muscle {bb['within_frac']-mm['within_frac']:+.4f})")
    print(f" F-like sep   muscle={mm['f_like_separability']:.3f}  brain={bb['f_like_separability']:.3f}")


if __name__ == '__main__':
    main()
