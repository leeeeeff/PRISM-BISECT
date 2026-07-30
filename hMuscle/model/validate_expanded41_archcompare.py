#!/usr/bin/env python3
"""
validate_expanded41_archcompare.py
-----------------------------------
Rigor validation (gene-mean oracle + Domain-Ranking AUC + pos_bias) for BOTH
arms of the v15d-vs-v17f*-BP head-to-head (train_expanded41_truebrain.py),
41-term BP panel, true-brain zero-shot (63,994 isoforms). Mirrors the metric
definitions in validate_expanded41.py (applied there to the original
prism_v15d_expanded41 app score matrix) so the two are directly comparable.

Inputs: reports/expanded41_truebrain/{v15d,v17fstar}_score_matrix.npy + _meta.json,
        reports/expanded41_truebrain/ids.npy (gene symbols, brain isoform order)
Output: reports/expanded41_truebrain/rigor_{arch}.json
"""
import os, json
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RUN_DIR = '../../reports/expanded41_truebrain'
DOMAIN_MAT = '../results_isoform/features/domain_matrix_brain_full.npy'

gene_sym = [str(x) for x in np.load(f'{RUN_DIR}/ids.npy', allow_pickle=True)]
gene2idxs = defaultdict(list)
for i, g in enumerate(gene_sym): gene2idxs[g].append(i)

domain_mat = np.load(DOMAIN_MAT)
iso_n_domains = domain_mat.sum(axis=1).astype(np.int32)
assert domain_mat.shape[0] == len(gene_sym)

rng = np.random.default_rng(42)
B_POSBIAS = 1000
rng2 = np.random.default_rng(7)


def pos_bias_for_term(scores_col, y, gene2idxs_local):
    global_std = scores_col.std()
    if global_std < 1e-8:
        return None
    pos_genes = [g for g in gene2idxs_local if len(gene2idxs_local[g]) >= 2 and y[gene2idxs_local[g][0]] > 0]
    if len(pos_genes) < 3:
        return None
    all_multi_genes = [g for g in gene2idxs_local if len(gene2idxs_local[g]) >= 2]

    def mean_within_gene_std(gene_list):
        return float(np.mean([scores_col[gene2idxs_local[g]].std() for g in gene_list]))

    obs = mean_within_gene_std(pos_genes) / global_std
    boots = np.array([mean_within_gene_std(list(rng.choice(all_multi_genes, size=len(pos_genes), replace=True)))
                       / global_std for _ in range(B_POSBIAS)])
    p = float((boots >= obs).mean())
    return obs, p


def compute_dr_auc(preds_mat, gene2idxs_local, Y_te_local):
    aucs = []
    for g, idxs in gene2idxs_local.items():
        if len(idxs) < 2: continue
        domains = iso_n_domains[idxs]
        if domains.std() < 0.1: continue
        pos_terms = np.where(Y_te_local[idxs[0]] > 0)[0]
        if len(pos_terms) == 0: continue
        med = np.median(domains)
        domain_binary = (domains > med).astype(float)
        if domain_binary.sum() == 0 or domain_binary.sum() == len(idxs): continue
        p_g = preds_mat[idxs]
        for t in pos_terms:
            sc = p_g[:, t]
            if sc.std() < 1e-8:
                aucs.append(0.5); continue
            try: aucs.append(roc_auc_score(domain_binary, sc))
            except Exception: pass
    return (float(np.mean(aucs)) if aucs else 0.5), len(aucs)


def bootstrap_dr(preds_mat, Y_te_local, qual_genes, n_boot=500, seed=42):
    rb = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rb.choice(qual_genes, size=len(qual_genes), replace=True)
        auc, _ = compute_dr_auc(preds_mat, {g: gene2idxs[g] for g in samp}, Y_te_local)
        boots.append(auc)
    boots = np.array(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


for arch in ['v15d', 'v17fstar']:
    print("=" * 70)
    print(f"  Rigor validation: {arch}  (41-term BP, true-brain 63,994)")
    print("=" * 70)
    scores = np.load(f'{RUN_DIR}/{arch}_score_matrix.npy').astype(np.float32)
    meta = json.load(open(f'{RUN_DIR}/{arch}_meta.json'))
    go_ids, go_names, go_source = meta['go_ids'], meta['go_names'], meta['go_source']
    N_GO = len(go_ids)

    go2pos_syms = defaultdict(set)
    with open('../data/raw_data/data/annotations/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                for go in parts[1:]: go2pos_syms[go].add(parts[0])
    Y_te = np.stack([np.array([1.0 if s in go2pos_syms[go] else 0.0 for s in gene_sym], dtype=np.float32)
                      for go in go_ids], axis=1)
    n_pos_te = Y_te.sum(0)

    gene_mean_preds = np.zeros_like(scores)
    for g, idxs in gene2idxs.items():
        gene_mean_preds[idxs] = scores[idxs].mean(0)

    auprc_prism = np.array([average_precision_score(Y_te[:, k], scores[:, k]) if n_pos_te[k] >= 1 else np.nan
                             for k in range(N_GO)])
    auprc_oracle = np.array([average_precision_score(Y_te[:, k], gene_mean_preds[:, k]) if n_pos_te[k] >= 1 else np.nan
                              for k in range(N_GO)])
    print(f"  macro AUPRC: {arch}={np.nanmean(auprc_prism):.4f}  gene_mean_oracle={np.nanmean(auprc_oracle):.4f}  "
          f"gap={np.nanmean(auprc_oracle)-np.nanmean(auprc_prism):+.4f}")

    pos_bias_results = []
    for k in range(N_GO):
        r = pos_bias_for_term(scores[:, k], Y_te[:, k], gene2idxs)
        if r is not None:
            obs, p = r
            pos_bias_results.append({'go_id': go_ids[k], 'name': go_names[go_ids[k]],
                                      'source': go_source[go_ids[k]], 'obs_pos_bias': obs, 'p_vs_shuffled': p})
    pvals = np.array([r['p_vs_shuffled'] for r in pos_bias_results])
    order = np.argsort(pvals); m = len(pvals); q = np.empty(m); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        rr = m - rank
        prev = min(prev, pvals[idx] * m / rr)
        q[idx] = prev
    for r, qv in zip(pos_bias_results, q): r['q_BH'] = float(qv)
    n_sig = sum(1 for r in pos_bias_results if r['q_BH'] < 0.05)
    print(f"  pos_bias significant (q<0.05): {n_sig}/{len(pos_bias_results)}")

    random_preds = rng2.random((len(gene_sym), N_GO)).astype(np.float32)
    dr_results = {}
    for name, preds in [(arch, scores), ('gene_mean_oracle', gene_mean_preds), ('random', random_preds)]:
        auc, n = compute_dr_auc(preds, gene2idxs, Y_te)
        dr_results[name] = {'auc': auc, 'n_pairs': n}
        print(f"  DR-AUC {name:<20}: {auc:.4f}  (N={n:,})")

    qual_genes = [g for g, idxs in gene2idxs.items()
                  if len(idxs) >= 2 and iso_n_domains[idxs].std() >= 0.1 and Y_te[idxs[0]].sum() > 0]
    for name, preds in [(arch, scores), ('gene_mean_oracle', gene_mean_preds)]:
        mean, lo, hi = bootstrap_dr(preds, Y_te, qual_genes, n_boot=500)
        dr_results[name]['bootstrap_mean'] = mean
        dr_results[name]['bootstrap_ci_lo'] = lo
        dr_results[name]['bootstrap_ci_hi'] = hi
        print(f"  DR-AUC {name:<20}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    out = {
        'arch': arch, 'n_go': N_GO,
        'macro_auprc': float(np.nanmean(auprc_prism)),
        'macro_auprc_gene_mean_oracle': float(np.nanmean(auprc_oracle)),
        'pos_bias_n_sig_q05': n_sig, 'pos_bias_n_evaluable': len(pos_bias_results),
        'domain_ranking_auc': dr_results,
    }
    with open(f'{RUN_DIR}/rigor_{arch}.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"  Saved -> {RUN_DIR}/rigor_{arch}.json\n")
