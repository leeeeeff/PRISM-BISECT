#!/usr/bin/env python3
"""
exp_brain_muscle_supervised_sep.py
==================================
Option B: supervised gene-separability, brain vs muscle, on the SAME representation
(raw delta_layer L30-L15, z-scored), controlling cardinality / class-count / density.

User claim (1): gene-family separation is CLEARER in brain than muscle.
The variance decomposition (finding-brain-within-gene-variance) could not test this
independently (between = 1 - within). Here we test it with a genuinely supervised,
cardinality-matched retrieval measure.

Design (fully matched):
  - exactly-2-isoform genes only (matched within-gene cardinality)
  - subsample to EQUAL number of genes G* in both tissues (matched class count & density
    -> matched chance level 1/(2G*-1) for nearest-neighbor)
  - z-score per-dim (fit on the subsample)
  - Top-1 sibling retrieval: is each isoform's nearest neighbor (cosine & euclid) its
    sibling? higher = genes more separable = claim(1) direction.
  - d_between/d_within ratio: median over genes (silhouette-like separability).
  - 5 subsample seeds -> mean +/- sd.

PREDICTION (pre-registered): variance decomp showed brain within_frac >> muscle, so
siblings are farther apart in brain. If d_within grows faster than d_between,
MUSCLE > BRAIN on retrieval => claim(1) refuted at supervised level too. If BRAIN > MUSCLE,
brain's between-gene structure compensates => claim(1) rescued.
"""
import numpy as np
from pathlib import Path
import json

DATA = Path('/home/welcome1/sw1686/DIFFUSE/hMuscle/data')
BRAIN = DATA / 'brain_isoquant_esm2/full'
ID_DIR = DATA / 'raw_data/data/id_lists'
OUT = Path('/home/welcome1/sw1686/DIFFUSE/reports/truebrain_rerun_20260714/exp_variance_structure')
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7, 13, 21, 99]


def clean(g):
    return str(g).replace("b'", "").replace("'", "").replace('"', "").replace(" ", "")


def load_delta(l15, l30, gpath):
    d = (np.load(l30).astype(np.float32) - np.load(l15).astype(np.float32))
    genes = np.array([clean(g) for g in np.load(gpath, allow_pickle=True)])
    return d, genes


def dedup_global(X, genes):
    """remove exact-duplicate embedding rows (identical protein sequence -> identical ESM-2).
    keeps first occurrence in original order. Brain long-read has ~28% such redundancy."""
    _, idx = np.unique(X, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx], genes[idx]


def two_iso_genes(genes):
    gl, gidx = np.unique(genes, return_inverse=True)
    cnt = np.bincount(gidx, None, len(gl))
    keep = gl[cnt == 2]
    return keep


def eval_tissue(X, genes, two_genes, G_star, rng):
    sel = rng.choice(two_genes, size=G_star, replace=False)
    sel_set = set(sel.tolist())
    mask = np.array([g in sel_set for g in genes])
    Xs = X[mask]; gs = genes[mask]
    # order so siblings are adjacent
    order = np.argsort(gs, kind='stable')
    Xs = Xs[order]; gs = gs[order]
    # z-score per dim (fit on subsample)
    mu = Xs.mean(0); sd = Xs.std(0) + 1e-8
    Z = (Xs - mu) / sd
    N = Z.shape[0]
    # sibling index: pairs are consecutive (each gene exactly 2 rows after sort)
    # build sibling map
    gl2, ginv = np.unique(gs, return_inverse=True)
    sib = np.full(N, -1, int)
    for gi in range(len(gl2)):
        idx = np.where(ginv == gi)[0]
        sib[idx[0]] = idx[1]; sib[idx[1]] = idx[0]
    # cosine NN
    Zc = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    S = Zc @ Zc.T
    np.fill_diagonal(S, -np.inf)
    nn_cos = S.argmax(1)
    acc_cos = float((nn_cos == sib).mean())
    # euclidean NN (on z-scored)
    # ||a-b||^2 = |a|^2+|b|^2-2 a.b ; use dot on Z
    sq = (Z ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * (Z @ Z.T)
    np.fill_diagonal(D, np.inf)
    nn_euc = D.argmin(1)
    acc_euc = float((nn_euc == sib).mean())
    # d_between/d_within (euclid, z-scored): d_within = dist to sibling;
    # d_between = dist to nearest NON-sibling
    d_sib = np.sqrt(np.maximum(D[np.arange(N), sib], 0))
    Dnb = D.copy()
    Dnb[np.arange(N), sib] = np.inf   # exclude sibling
    d_nn_other = np.sqrt(np.maximum(Dnb.min(1), 0))
    ratio = np.median(d_nn_other / (d_sib + 1e-8))
    return dict(acc_cos_sibling=acc_cos, acc_euc_sibling=acc_euc,
                between_over_within=float(ratio),
                chance=1.0 / (N - 1), N=int(N))


def main():
    print("loading...")
    m_d, m_g = load_delta(DATA / 'esm2_train_human_layer15_t30_150M.npy',
                          DATA / 'esm2_train_human_layer30_t30_150M.npy',
                          ID_DIR / 'train_gene_list.npy')
    b_d, b_g = load_delta(BRAIN / 'brain_full_esm2_layer15_t30_150M.npy',
                          BRAIN / 'brain_full_esm2_layer30_t30_150M.npy',
                          BRAIN / 'brain_full_gene_names.npy')
    # CRITICAL: dedup identical-sequence isoforms (brain ~28% redundant rows) before
    # measuring within-gene distinctness -> otherwise identical siblings (d_within=0)
    # fabricate a 0.5/1.0 artifact. Applied to BOTH tissues identically.
    n_m0, n_b0 = len(m_d), len(b_d)
    m_d, m_g = dedup_global(m_d, m_g)
    b_d, b_g = dedup_global(b_d, b_g)
    print(f"dedup: muscle {n_m0}->{len(m_d)}  brain {n_b0}->{len(b_d)}")
    m_two = two_iso_genes(m_g); b_two = two_iso_genes(b_g)
    G_star = min(len(m_two), len(b_two))
    print(f"distinct 2-iso genes  muscle={len(m_two)}  brain={len(b_two)}  -> G*={G_star}")

    res = {'G_star': int(G_star), 'muscle': {}, 'brain': {}}
    for tissue, X, genes, two in [('muscle', m_d, m_g, m_two), ('brain', b_d, b_g, b_two)]:
        runs = [eval_tissue(X, genes, two, G_star, np.random.default_rng(s)) for s in SEEDS]
        agg = {}
        for k in ['acc_cos_sibling', 'acc_euc_sibling', 'between_over_within', 'chance']:
            vals = np.array([r[k] for r in runs])
            agg[k] = {'mean': float(vals.mean()), 'sd': float(vals.std())}
        agg['N'] = runs[0]['N']
        res[tissue] = agg
        print(f"\n=== {tissue.upper()} (2-iso, G*={G_star}, N={agg['N']}) ===")
        print(f"  Top-1 sibling acc  cosine={agg['acc_cos_sibling']['mean']:.4f}"
              f"±{agg['acc_cos_sibling']['sd']:.4f}   euclid={agg['acc_euc_sibling']['mean']:.4f}"
              f"±{agg['acc_euc_sibling']['sd']:.4f}  (chance={agg['chance']['mean']:.2e})")
        print(f"  d_between/d_within (median) = {agg['between_over_within']['mean']:.4f}"
              f"±{agg['between_over_within']['sd']:.4f}")

    mc = res['muscle']['acc_cos_sibling']['mean']; bc = res['brain']['acc_cos_sibling']['mean']
    mr = res['muscle']['between_over_within']['mean']; br = res['brain']['between_over_within']['mean']
    res['_meta'] = {
        'claim1_supervised_gene_separability': {
            'muscle_sibling_acc_cos': mc, 'brain_sibling_acc_cos': bc,
            'brain_minus_muscle': bc - mc,
            'muscle_between_over_within': mr, 'brain_between_over_within': br,
            'supports_claim1_brain_clearer': bool(bc > mc),
            'verdict': ('CLAIM1 SUPPORTED: brain genes more separable (supervised)'
                        if bc > mc else
                        'CLAIM1 REFUTED at supervised level: muscle genes more separable'),
        },
        'note': 'exactly-2-iso, class-count & density matched (G*), z-scored raw delta. '
                'sibling retrieval = supervised gene-separability independent of the '
                'between=1-within complementarity that blocked the variance-decomp test.',
    }
    (OUT / 'supervised_separability.json').write_text(json.dumps(res, indent=2))
    print(f"\n>>> {res['_meta']['claim1_supervised_gene_separability']['verdict']}")
    print(f"    sibling-acc(cos): muscle={mc:.4f} vs brain={bc:.4f} ({bc-mc:+.4f})")
    print(f"Saved -> {OUT / 'supervised_separability.json'}")


if __name__ == '__main__':
    main()
