#!/usr/bin/env python3
"""
validate_expanded41.py
-----------------------
Rigor-upgrade for the 41-term Brain-expanded BP panel (prism_v15d_expanded41,
git 20cedaf, 2026-06-07): that experiment predates this paper's gene-mean-oracle /
within-gene Domain-Ranking AUC / pos_bias framework (established ~2026-07-01) and
was never checked against it. Its headline macro AUPRC 0.6724 could be entirely a
gene-family-identification artifact -- exactly the confound this paper's methodology
was built to detect and exclude.

This script computes, on the EXISTING saved 41-term score matrix (no retraining):
  1. Per-term macro AUPRC (recomputed with the paper's standard average_precision_score)
  2. Gene-mean oracle macro AUPRC (isoform score := within-gene mean of the model's
     OWN predictions) -- must collapse within-gene AUC/DR to 0.5 by construction;
     quantifies how much of (1) is gene-identity vs isoform-level signal.
  3. pos_bias per term (within-gene positive-class score SD / global SD), B=1000
     gene-level bootstrap CI, label-shuffled null floor, Benjamini-Hochberg FDR.
  4. Within-gene Domain-Ranking AUC (domain-count median split per gene, using the
     TRUE-brain Pfam matrix), B=500 gene-level bootstrap CI, vs Gene-mean oracle
     (must be 0.5) and Random.

Inputs (all pre-existing, none retrained):
  prism_app/data/demo/brain_full_expanded_41_{scores,ids,gene_ids}.npy
  prism_app/data/demo/brain_full_expanded_41_meta.json  (go_ids, go_names, go_source)
  hMuscle/results_isoform/features/domain_matrix_brain_full.npy  (63994x512 Pfam, TRUE BRAIN)
  hMuscle/data/raw_data/data/annotations/{gene2go.gz,Homo_sapiens.gene_info.gz}

Output: reports/expanded41_validation/results.json
"""
import os, json, gzip
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

APP_DIR  = '../../prism_app/data/demo'
ANNOT_DIR = '../data/raw_data/data/annotations'
DOMAIN_MAT = '../results_isoform/features/domain_matrix_brain_full.npy'
OUT_DIR = '../../reports/expanded41_validation'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("  Rigor validation: 41-term Brain-expanded BP panel")
print("  (gene-mean oracle + Domain-Ranking AUC + pos_bias, true brain 63,994)")
print("=" * 70)

# ── 1. Load existing predictions (no retraining) ────────────────────────────
scores  = np.load(f'{APP_DIR}/brain_full_expanded_41_scores.npy').astype(np.float32)  # (63994, 41)
iso_ids = [str(x) for x in np.load(f'{APP_DIR}/brain_full_expanded_41_ids.npy', allow_pickle=True)]
gene_sym = [str(x) for x in np.load(f'{APP_DIR}/brain_full_expanded_41_gene_ids.npy', allow_pickle=True)]
meta = json.load(open(f'{APP_DIR}/brain_full_expanded_41_meta.json'))
go_ids = meta['go_ids']
go_names = meta['go_names']
go_source = meta['go_source']
n_iso, N_GO = scores.shape
print(f"\n[1] Loaded predictions: {scores.shape}  ({n_iso} isoforms, {N_GO} GO terms)")
assert N_GO == 41 and len(go_ids) == 41

gene2idxs = defaultdict(list)
for i, g in enumerate(gene_sym): gene2idxs[g].append(i)
print(f"  {len(gene2idxs)} genes, multi-isoform (>=2): {sum(1 for v in gene2idxs.values() if len(v)>=2)}")

# ── 2. Ground truth Y_te (same source as the canonical 18-term brain eval:
#      human_annotations_unified_bp.txt, UniProt/QuickGO unified w/ IEA fallback
#      -- NOT gene2go.gz, which is what the separate 279-term panel uses. Verified
#      against reports/v15d_brain_eval/brain_eval_20260519_2125.json (macro_auprc_all18
#      =0.5998, matching the paper's cited "0.600"): gene2go.gz undercounts positives
#      for broad terms by ~2-3x relative to this file (e.g. GO:0007204 551 vs 194,
#      GO:0006941 593 vs 283), while narrow/already-saturated terms like GO:0045214
#      Sarcomere organization match exactly (108=108) -- confirming unified_bp.txt,
#      not gene2go.gz, is this model family's correct ground truth. ────────────────
print("\n[2] Building ground-truth Y_te from human_annotations_unified_bp.txt...")
go2pos_syms = defaultdict(set)
with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            sym = parts[0]
            for go in parts[1:]:
                go2pos_syms[go].add(sym)

Y_te = np.stack([
    np.array([1.0 if s in go2pos_syms[go] else 0.0 for s in gene_sym], dtype=np.float32)
    for go in go_ids
], axis=1)
n_pos_te = Y_te.sum(0)
print(f"  n_pos_te per term: min={n_pos_te.min():.0f} median={np.median(n_pos_te):.0f} max={n_pos_te.max():.0f}")
print(f"  Terms with <2 positive test isoforms: {(n_pos_te < 2).sum()}/41")

# ── 3. Macro AUPRC (recomputed, standard metric) + Gene-mean oracle ─────────
print("\n[3] Macro AUPRC: PRISM_expanded41 vs Gene-mean oracle...")
gene_mean_preds = np.zeros_like(scores)
for g, idxs in gene2idxs.items():
    gene_mean_preds[idxs] = scores[idxs].mean(0)

auprc_prism = np.array([average_precision_score(Y_te[:, k], scores[:, k]) if n_pos_te[k] >= 1 else np.nan
                         for k in range(N_GO)])
auprc_oracle = np.array([average_precision_score(Y_te[:, k], gene_mean_preds[:, k]) if n_pos_te[k] >= 1 else np.nan
                          for k in range(N_GO)])
print(f"  PRISM_expanded41 macro AUPRC:  {np.nanmean(auprc_prism):.4f}")
print(f"  Gene-mean oracle macro AUPRC: {np.nanmean(auprc_oracle):.4f}")
print(f"  Gap (oracle - PRISM):          {np.nanmean(auprc_oracle) - np.nanmean(auprc_prism):+.4f}")
print("  (large positive gap = macro AUPRC dominated by gene-identity, as elsewhere in this paper)")

# ── 4. pos_bias per term, B=1000 gene-level bootstrap, BH-FDR ───────────────
print("\n[4] pos_bias per term (B=1000 bootstrap, label-shuffled null, BH-FDR)...")
rng = np.random.default_rng(42)
B = 1000

def pos_bias_for_term(k, gene_mean_pred_ok=True):
    y = Y_te[:, k]
    pos_genes = [g for g, idxs in gene2idxs.items() if len(idxs) >= 2 and y[idxs[0]] > 0
                 and len(set(y[i] for i in idxs)) == 1]
    # positive-class genes with >=2 isoforms (gene-level label, so uniform within gene)
    pos_genes = [g for g in gene2idxs if len(gene2idxs[g]) >= 2 and y[gene2idxs[g][0]] > 0]
    if len(pos_genes) < 3:
        return None
    global_std = scores[:, k].std()
    if global_std < 1e-8:
        return None

    def mean_within_gene_std(gene_list):
        stds = [scores[gene2idxs[g], k].std() for g in gene_list]
        return float(np.mean(stds))

    obs = mean_within_gene_std(pos_genes) / global_std

    all_multi_genes = [g for g in gene2idxs if len(gene2idxs[g]) >= 2]
    boots = []
    for _ in range(B):
        samp = rng.choice(all_multi_genes, size=len(pos_genes), replace=True)
        boots.append(mean_within_gene_std(list(samp)) / global_std)
    boots = np.array(boots)
    p = float((boots >= obs).mean())
    return {
        'go_id': go_ids[k], 'name': go_names[go_ids[k]], 'source': go_source[go_ids[k]],
        'n_pos_genes': len(pos_genes), 'obs_pos_bias': obs,
        'shuf_floor': float(boots.mean()),
        'boot_ci_lo': float(np.percentile(boots, 2.5)), 'boot_ci_hi': float(np.percentile(boots, 97.5)),
        'p_vs_shuffled': p,
    }

pos_bias_results = []
for k in range(N_GO):
    r = pos_bias_for_term(k)
    if r is not None:
        pos_bias_results.append(r)

pvals = np.array([r['p_vs_shuffled'] for r in pos_bias_results])
order = np.argsort(pvals)
m = len(pvals)
q = np.empty(m)
prev = 1.0
for rank, idx in enumerate(order[::-1]):
    r = m - rank
    prev = min(prev, pvals[idx] * m / r)
    q[idx] = prev
for r, qv in zip(pos_bias_results, q):
    r['q_BH'] = float(qv)

n_sig = sum(1 for r in pos_bias_results if r['q_BH'] < 0.05 and r['obs_pos_bias'] > r['boot_ci_lo'])
n_sig_strict = sum(1 for r in pos_bias_results if r['q_BH'] < 0.05)
print(f"  {len(pos_bias_results)}/{N_GO} terms evaluable (>=3 positive multi-isoform genes)")
print(f"  Significant at q<0.05: {n_sig_strict}/{len(pos_bias_results)}")
top5 = sorted(pos_bias_results, key=lambda r: -r['obs_pos_bias'])[:5]
for r in top5:
    print(f"    {r['name']:<45} pb={r['obs_pos_bias']:.3f}  q={r['q_BH']:.4f}  source={r['source']}")

# ── 5. Domain-Ranking AUC (within-gene, Pfam domain-count median split) ─────
print("\n[5] Within-gene Domain-Ranking AUC (true-brain Pfam matrix)...")
domain_mat = np.load(DOMAIN_MAT)
assert domain_mat.shape[0] == n_iso, f"domain matrix rows {domain_mat.shape[0]} != {n_iso}"
iso_n_domains = domain_mat.sum(axis=1).astype(np.int32)

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
            try:
                aucs.append(roc_auc_score(domain_binary, sc))
            except Exception:
                pass
    return (float(np.mean(aucs)) if aucs else 0.5), len(aucs)

rng2 = np.random.default_rng(7)
random_preds = rng2.random((n_iso, N_GO)).astype(np.float32)

dr_results = {}
for name, preds in [('PRISM_expanded41', scores), ('Gene_mean_oracle', gene_mean_preds), ('Random', random_preds)]:
    auc, n = compute_dr_auc(preds, gene2idxs, Y_te)
    dr_results[name] = {'auc': auc, 'n_pairs': n}
    print(f"  {name:<20}: Domain-Ranking AUC = {auc:.4f}  (N={n:,} gene-term pairs)")

print("\n[6] Bootstrap CI for Domain-Ranking AUC (B=500, gene-level)...")
qual_genes = [g for g, idxs in gene2idxs.items()
              if len(idxs) >= 2 and iso_n_domains[idxs].std() >= 0.1 and Y_te[idxs[0]].sum() > 0]
print(f"  Qualifying genes for DR-AUC bootstrap: {len(qual_genes)}")

def bootstrap_dr(preds_mat, n_boot=500, seed=42):
    rb = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rb.choice(qual_genes, size=len(qual_genes), replace=True)
        auc, _ = compute_dr_auc(preds_mat, {g: gene2idxs[g] for g in samp}, Y_te)
        boots.append(auc)
    boots = np.array(boots)
    return float(boots.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

for name in ['PRISM_expanded41', 'Gene_mean_oracle']:
    preds = scores if name == 'PRISM_expanded41' else gene_mean_preds
    mean, lo, hi = bootstrap_dr(preds, n_boot=500)
    dr_results[name]['bootstrap_mean'] = mean
    dr_results[name]['bootstrap_ci_lo'] = lo
    dr_results[name]['bootstrap_ci_hi'] = hi
    print(f"  {name:<20}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

# ── 7. Save + final verdict ──────────────────────────────────────────────
summary = {
    'model': 'prism_v15d_expanded41',
    'n_go': N_GO, 'n_brain_isoforms': n_iso,
    'macro_auprc_prism': float(np.nanmean(auprc_prism)),
    'macro_auprc_gene_mean_oracle': float(np.nanmean(auprc_oracle)),
    'per_term_auprc': {go_ids[k]: {'name': go_names[go_ids[k]], 'source': go_source[go_ids[k]],
                                    'auprc_prism': float(auprc_prism[k]), 'auprc_oracle': float(auprc_oracle[k]),
                                    'n_pos_te': int(n_pos_te[k])} for k in range(N_GO)},
    'pos_bias_per_term': pos_bias_results,
    'domain_ranking_auc': dr_results,
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f"\nSaved -> {OUT_DIR}/results.json")

print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)
prism_dr = dr_results['PRISM_expanded41']['bootstrap_mean']
oracle_dr = dr_results['Gene_mean_oracle']['bootstrap_mean']
print(f"  Macro AUPRC:        PRISM {np.nanmean(auprc_prism):.4f} vs Gene-mean oracle {np.nanmean(auprc_oracle):.4f}")
print(f"  Domain-Ranking AUC: PRISM {prism_dr:.4f} vs Gene-mean oracle {oracle_dr:.4f} (null=0.5)")
print(f"  pos_bias significant (q<0.05): {n_sig_strict}/{len(pos_bias_results)}")
if prism_dr > oracle_dr + 0.01:
    print("  -> within-gene isoform-level signal SURVIVES gene-mean-oracle control.")
else:
    print("  -> within-gene isoform-level signal DOES NOT clearly exceed gene-mean oracle -- inspect further.")
print("=" * 70)
