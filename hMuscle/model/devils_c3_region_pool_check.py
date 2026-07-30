#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""devils_c3_region_pool_check.py

Applies the gene-permutation logic to the ONE surviving region-pool finding
(brain domain_binary rescue, from analyze_severity_region_covariates.py's
Part B size-quintile-matched paired bootstrap: region beats scram beats
structmatch).

IMPORTANT DISTINCTION from the invalidated orient=+1 tests: Part B's
discrimination test uses domain_diff SIGN (an EXTERNAL label: domain-count(long)
- domain-count(short)) to define "aligned" vs "decoupled", NOT a raw
"long-minus-short, always +1" self-consistency convention. Structurally this
is closer to candidates 1-4 (validated: covariate-based split, checked
against a random-relabeling null) than to the internal-edit self-consistency
tests (invalidated: gene-ID-invariant). So the claim "region-pool > scram-pool
for domain discrimination" was never a "genes share a mechanism" claim in the
first place.

Still, since severity_score's own coherence-DIRECTION training step (mean of
train-fold D, gene-disjoint CV) was shown to be gene-permutation-invariant,
this script directly verifies: does the region>scram GAP survive when the
FIXED coherence direction is instead trained on a gene-ID-PERMUTED fold split
(only the direction-training step is sabotaged; D_region/D_scram vectors and
domain_diff labels stay attached to their real pairs)? If yes -> the region>
scram comparison doesn't depend on genuine gene-specific direction-training,
consistent with it never being a "gene-sharing" claim -- the finding stands
on its own terms. If the gap collapses -> a real problem, needs escalation.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
SEV = ROOT / 'reports/severity_pairs'
DATA = ROOT / 'hMuscle/data'
BRAIN = DATA / 'brain_isoquant_esm2/full'
N_FOLDS = 5
N_BOOT = 500
N_BINS = 5
SEED = 42
rng = np.random.default_rng(456)

DOMAIN_MAT = {'muscle': 'domain_matrix_proper_test.npy', 'brain': 'domain_matrix_brain_full.npy'}
MEANPOOL_L15 = {'muscle': 'esm2_layer_15_t30_150M.npy',
                'brain': 'brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy'}
MEANPOOL_L30 = {'muscle': 'esm2_layer_30_t30_150M.npy',
                'brain': 'brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy'}


def gene_disjoint_folds(genes, seed):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % N_FOLDS for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def train_directions(D_mean_full, fold):
    directions = {}
    for k in range(N_FOLDS):
        tr = fold != k
        d = D_mean_full[tr].mean(0)
        n = np.linalg.norm(d)
        directions[k] = d / n if n > 0 else d
    return directions


def score(D, fold, directions):
    out = np.zeros(len(D))
    for k, d in directions.items():
        m = fold == k
        out[m] = D[m] @ d
    return out


def domain_size_matched_gap(df, score_col_a, score_col_b, dom_count):
    """Paired bootstrap of (size-matched rate A - size-matched rate B), gene-cluster resample."""
    sub = df[df['domain_binary'] == 1].copy()
    sub['domain_diff'] = dom_count[sub['long_idx'].to_numpy()] - dom_count[sub['short_idx'].to_numpy()]
    pos = sub[sub['domain_diff'] > 0].copy()
    neg = sub[sub['domain_diff'] < 0].copy()
    if len(neg) < 10:
        return None
    bin_edges = np.quantile(neg['size'], np.linspace(0, 1, N_BINS + 1))
    bin_edges[0] = -np.inf; bin_edges[-1] = np.inf
    neg['bin'] = pd.cut(neg['size'], bin_edges, labels=False, include_lowest=True)
    pos['bin'] = pd.cut(pos['size'], bin_edges, labels=False, include_lowest=True)
    bin_weights = neg['bin'].value_counts(normalize=True).sort_index()
    bin_weights = bin_weights.reindex(range(N_BINS), fill_value=0.0).to_numpy()
    pos_valid = pos.dropna(subset=['bin'])
    pos_valid = pos_valid[pos_valid['bin'].isin(range(N_BINS))].reset_index(drop=True)

    def weighted_rate(score_col, rows):
        r_al = (pos_valid[score_col].to_numpy() > 0).astype(float)[rows]
        r_bin = pos_valid['bin'].to_numpy()[rows]
        w, tw = 0.0, 0.0
        for k in range(N_BINS):
            mask = r_bin == k
            if mask.sum() == 0:
                continue
            w += bin_weights[k] * r_al[mask].mean(); tw += bin_weights[k]
        return w / tw if tw > 0 else np.nan

    point_a = weighted_rate(score_col_a, np.arange(len(pos_valid)))
    point_b = weighted_rate(score_col_b, np.arange(len(pos_valid)))
    genes = pos_valid['gene'].to_numpy()
    uniq_genes = np.unique(genes)
    gene_to_rows = {g: np.where(genes == g)[0] for g in uniq_genes}
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sampled = rng.choice(uniq_genes, size=len(uniq_genes), replace=True)
        rows = np.concatenate([gene_to_rows[g] for g in sampled])
        diffs[b] = weighted_rate(score_col_a, rows) - weighted_rate(score_col_b, rows)
    ci = np.percentile(diffs, [2.5, 97.5])
    return point_a, point_b, ci


def analyze(tissue):
    print(f"\n{'='*70}\n[{tissue}] region-pool domain_binary rescue: gene-permuted-direction check\n{'='*70}")
    df = pd.read_csv(SEV / f'{tissue}_severity_pairs_region.tsv', sep='\t')
    cache = np.load(SEV / f'{tissue}_region_embeddings.npz')
    D_region, D_scram = cache['D_region'], cache['D_scram']

    L15 = np.load(ROOT / 'hMuscle/data' / MEANPOOL_L15[tissue]).astype(np.float32)
    L30 = np.load(ROOT / 'hMuscle/data' / MEANPOOL_L30[tissue]).astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    D_mean = emb[df['long_idx'].to_numpy()] - emb[df['short_idx'].to_numpy()]

    dom = np.load(ROOT / 'hMuscle/results_isoform/features' / DOMAIN_MAT[tissue])
    dom_count = dom.sum(axis=1).astype(np.int32)

    # REAL fold: MUST be trained on the FULL original population's gene set
    # (fold shuffle order depends on which genes are present) -- reproducing
    # on the region-pool-valid SUBSET's own gene list gives a different,
    # wrong fold assignment (same lesson hit repeatedly earlier this session).
    df_full = pd.read_csv(SEV / f'{tissue}_severity_pairs_scored.tsv', sep='\t')
    df_full = df_full[df_full['tissue'] == tissue].reset_index(drop=True)
    L15f = np.load(ROOT / 'hMuscle/data' / MEANPOOL_L15[tissue]).astype(np.float32)
    L30f = np.load(ROOT / 'hMuscle/data' / MEANPOOL_L30[tissue]).astype(np.float32)
    emb_full = np.concatenate([L15f, L30f], axis=1)
    D_mean_full = emb_full[df_full['long_idx'].to_numpy()] - emb_full[df_full['short_idx'].to_numpy()]
    fold_full = gene_disjoint_folds(df_full['gene'].to_numpy(), seed=SEED)
    directions_real = train_directions(D_mean_full, fold_full)
    check_full = score(D_mean_full, fold_full, directions_real)
    assert np.allclose(check_full, df_full['severity_score'].to_numpy(), atol=1e-3), \
        "real-fold reproduction failed on FULL population"

    # row-level fold permutation (reassigns which fold each ROW lands in,
    # regardless of its real gene -- simpler and unambiguous than trying to
    # route through a permuted gene-name dictionary, which is ill-defined
    # when a gene appears in multiple rows)
    fold_perm_full = np.random.default_rng(99).permutation(fold_full)
    directions_perm = train_directions(D_mean_full, fold_perm_full)

    key_cols = ['gene', 'canonical_idx', 'other_idx', 'long_idx', 'short_idx']
    df_full = df_full.reset_index().rename(columns={'index': '_full_idx'})
    df_full['fold_real'] = fold_full
    df_full['fold_perm'] = fold_perm_full
    merged = df.merge(df_full[key_cols + ['fold_real', 'fold_perm']], on=key_cols, how='left')
    assert merged['fold_real'].notna().all(), "merge failed to match all region-pool subset rows"
    fold_real = merged['fold_real'].to_numpy().astype(int)
    fold_perm = merged['fold_perm'].to_numpy().astype(int)

    check = score(D_mean, fold_real, directions_real)
    assert np.allclose(check, df['severity_score'].to_numpy(), atol=1e-3), \
        "real-fold reproduction failed on region-pool subset"

    region_real = df['severity_score_region'].to_numpy()  # already scored w/ real-fold direction
    scram_real = df['severity_score_scram'].to_numpy()
    df['sr_real'] = region_real; df['ss_real'] = scram_real
    out_real = domain_size_matched_gap(df, 'sr_real', 'ss_real', dom_count)
    print(f"  [REAL direction] region_rate={out_real[0]:.3f}  scram_rate={out_real[1]:.3f}  "
          f"gap={out_real[0]-out_real[1]:+.3f}  CI=[{out_real[2][0]:+.3f},{out_real[2][1]:+.3f}]")

    # PERMUTED-gene (row-level fold shuffle) direction training
    region_perm = score(D_region, fold_perm, directions_perm)
    scram_perm = score(D_scram, fold_perm, directions_perm)
    df['sr_perm'] = region_perm; df['ss_perm'] = scram_perm
    out_perm = domain_size_matched_gap(df, 'sr_perm', 'ss_perm', dom_count)
    print(f"  [PERMUTED-gene direction] region_rate={out_perm[0]:.3f}  scram_rate={out_perm[1]:.3f}  "
          f"gap={out_perm[0]-out_perm[1]:+.3f}  CI=[{out_perm[2][0]:+.3f},{out_perm[2][1]:+.3f}]")

    print(f"\n  => real gap {'survives with permuted-gene direction (finding does NOT depend on real gene structure, as expected for a non-gene-sharing claim)' if out_perm[2][0]*out_perm[2][1] > 0 else 'COLLAPSES under permuted-gene direction (needs escalation)'}")


def main():
    for tissue in ['muscle', 'brain']:
        analyze(tissue)


if __name__ == '__main__':
    main()
