#!/usr/bin/env python3
"""
within_gene_metrics_all_domains.py
====================================
v17f* within-gene 핵심 지표를 MF/BP/CC 세 도메인에 대해 계산.

pos_bias(k) 정의 (evaluation.py compute_bias_score 기반):
  obs_pos_bias(k) = mean_std_within_gene(positive_genes_k) / global_std_k
  shuf_floor:     = obs_pos_bias when gene labels are randomly shuffled
  p-value:        fraction of shuffled bootstrap samples exceeding observed

Representational reversal (v15d 기준과 동일):
  within_var  = mean squared deviation of each isoform from its gene mean
  between_var = variance of gene-mean predictions
  ratio = within_var / between_var; reversal if ratio > 1.0

출력: reports/within_gene_metrics_all_domains/results.json
"""

import os, sys, gzip, json, csv, time
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/within_gene_metrics_all_domains'
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT   = 1000
RNG_SEED = 42

print("=" * 70)
print("  Within-Gene Metrics: MF / BP / CC")
print("=" * 70)
sys.stdout.flush()

# ── 1. Gene / isoform identity ────────────────────────────────────────
print("\n[1] Loading test set identities...")
sys.stdout.flush()

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_arr   = np.array([ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                          for g in te_genes_raw])
n_iso        = len(te_sym_arr)

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_arr): gene2idxs[g].append(i)

# Multi-isoform genes only (for pos_bias)
multi_gene_list = [g for g, v in gene2idxs.items() if len(v) >= 2]
multi_gene_idxs = [np.array(gene2idxs[g]) for g in multi_gene_list]
n_multi = len(multi_gene_list)

print(f"  {n_iso} isoforms, {len(gene2idxs)} genes, {n_multi} multi-isoform genes")
sys.stdout.flush()

# ── 2. GO labels ──────────────────────────────────────────────────────
print("\n[2] Loading GO labels...")
sys.stdout.flush()

sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]

go_genes = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        go_genes[p[2]].add(p[1])

all_terms_info = {}
with open('../../reports/v_expanded_gomf/expanded_go_per_term.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        all_terms_info[row['go_id']] = row

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

bp_terms = [r['go_id'] for r in all_terms_info.values()
            if r['cat']=='BP' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2]
cc_terms = [r['go_id'] for r in all_terms_info.values()
            if r['cat']=='CC' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2]
print(f"  MF: {len(mf_terms)}  BP: {len(bp_terms)}  CC: {len(cc_terms)}")
sys.stdout.flush()

# Gene-level label matrix: (n_multi_genes, n_terms) for each domain
# gene is positive if sym2id(gene_symbol) in go_genes[go_id]
te_gids = np.array([sym2id.get(g, '__') for g in te_sym_arr])

def build_gene_label_matrix(terms):
    """Returns (n_multi_genes, n_terms) bool matrix."""
    K = len(terms)
    mat = np.zeros((n_multi, K), dtype=np.float32)
    for k, go_id in enumerate(terms):
        pos_ids = go_genes[go_id]
        for gi, g in enumerate(multi_gene_list):
            gid = sym2id.get(g, '__')
            if gid in pos_ids:
                mat[gi, k] = 1.0
    return mat

print("  Building gene label matrices...")
sys.stdout.flush()
L_mf = build_gene_label_matrix(mf_terms)  # (n_multi, 82)
L_bp = build_gene_label_matrix(bp_terms)  # (n_multi, 103)
L_cc = build_gene_label_matrix(cc_terms)  # (n_multi, 93)
print(f"  L_mf: {L_mf.shape}  L_bp: {L_bp.shape}  L_cc: {L_cc.shape}")
sys.stdout.flush()

# ── 3. Predictions ─────────────────────────────────────────────────────
print("\n[3] Loading predictions...")
sys.stdout.flush()
P_mf = np.load('../../reports/v17f_star_bootstrap/v17f_star_preds.npy').astype(np.float32)
P_bp = np.load('../../reports/v17f_bp_cc_eval/BP_preds.npy').astype(np.float32)
P_cc = np.load('../../reports/v17f_bp_cc_eval/CC_preds.npy').astype(np.float32)
print(f"  P_mf:{P_mf.shape}  P_bp:{P_bp.shape}  P_cc:{P_cc.shape}")
sys.stdout.flush()

# Per-gene, per-term: within-gene score std (n_multi, n_terms)
def compute_within_std_matrix(preds):
    """For each multi-isoform gene, compute within-gene std per term."""
    n_genes = len(multi_gene_idxs)
    K = preds.shape[1]
    mat = np.zeros((n_genes, K), dtype=np.float32)
    for gi, idxs in enumerate(multi_gene_idxs):
        sc = preds[idxs]  # (n_iso, K)
        mat[gi] = sc.std(axis=0, ddof=0)
    return mat

print("\n[4] Computing within-gene std matrices...")
sys.stdout.flush()
t0 = time.time()
WS_mf = compute_within_std_matrix(P_mf)   # (n_multi, 82)
WS_bp = compute_within_std_matrix(P_bp)   # (n_multi, 103)
WS_cc = compute_within_std_matrix(P_cc)   # (n_multi, 93)
print(f"  Done: {time.time()-t0:.1f}s")
sys.stdout.flush()

# Global std per term (all isoforms)
G_mf = P_mf.std(axis=0, ddof=0)  # (82,)
G_bp = P_bp.std(axis=0, ddof=0)  # (103,)
G_cc = P_cc.std(axis=0, ddof=0)  # (93,)

# ── 4. pos_bias computation ───────────────────────────────────────────
def compute_pos_bias(within_std_mat, global_std, label_mat, terms, domain_label, rng_seed):
    """
    pos_bias(k) = mean_std_within_gene(positive_genes_k) / global_std(k)

    Null model: shuffle gene-level labels to estimate shuf_floor.
    p-value: fraction of shuffled bootstrap samples where shuf_pb >= obs_pb.
    Bootstrap CI: gene-block bootstrap on observed pos_bias.
    """
    rng = np.random.default_rng(rng_seed)
    K = len(terms)
    n_genes = within_std_mat.shape[0]
    eps = 1e-12

    t0 = time.time()
    print(f"\n  [{domain_label}] {K} terms, B={N_BOOT}...")
    sys.stdout.flush()

    # Observed pos_bias per term
    obs_pb = np.zeros(K)
    n_pos  = np.zeros(K, dtype=int)
    for k in range(K):
        pos_mask = label_mat[:, k] > 0
        n_pos[k] = pos_mask.sum()
        if n_pos[k] < 3 or global_std[k] < eps:
            obs_pb[k] = np.nan
            continue
        obs_pb[k] = float(within_std_mat[pos_mask, k].mean()) / (global_std[k] + eps)

    print(f"    Point estimates done: {time.time()-t0:.1f}s")
    sys.stdout.flush()

    # Bootstrap on observed (gene-block)
    boot_obs = np.full((N_BOOT, K), np.nan)
    gene_idx_range = np.arange(n_genes)
    for b in range(N_BOOT):
        boot_g = rng.integers(0, n_genes, n_genes)
        bws  = within_std_mat[boot_g]  # (n_genes, K)
        blbl = label_mat[boot_g]       # (n_genes, K)
        for k in range(K):
            pm = blbl[:, k] > 0
            if pm.sum() < 3 or global_std[k] < eps: continue
            boot_obs[b, k] = float(bws[pm, k].mean()) / (global_std[k] + eps)
        if (b+1) % 250 == 0:
            print(f"    obs boot {b+1}/{N_BOOT}: {time.time()-t0:.1f}s")
            sys.stdout.flush()

    # Shuffled null bootstrap (shuffle gene labels each time)
    boot_shuf = np.full((N_BOOT, K), np.nan)
    for b in range(N_BOOT):
        # Resample genes first, then shuffle labels
        boot_g = rng.integers(0, n_genes, n_genes)
        bws    = within_std_mat[boot_g]
        blbl   = label_mat[boot_g].copy()
        # Shuffle label assignment across genes independently per term
        for k in range(K):
            rng.shuffle(blbl[:, k])
        for k in range(K):
            pm = blbl[:, k] > 0
            n_pm = pm.sum()
            if n_pm < 3 or global_std[k] < eps: continue
            boot_shuf[b, k] = float(bws[pm, k].mean()) / (global_std[k] + eps)
        if (b+1) % 250 == 0:
            print(f"    shuf boot {b+1}/{N_BOOT}: {time.time()-t0:.1f}s")
            sys.stdout.flush()

    # Statistics per term
    results = []
    pvals = []
    valid_k = []
    for k in range(K):
        if np.isnan(obs_pb[k]):
            results.append({
                'go_id': terms[k],
                'name': all_terms_info.get(terms[k], {}).get('name', ''),
                'n_pos_genes': int(n_pos[k]),
                'obs_pos_bias': float('nan'), 'shuf_floor': float('nan'),
                'boot_ci_lo': float('nan'), 'boot_ci_hi': float('nan'),
                'p_vs_shuffled': float('nan'), 'q_BH': float('nan'),
            })
            continue

        bk_obs  = boot_obs[:, k][~np.isnan(boot_obs[:, k])]
        bk_shuf = boot_shuf[:, k][~np.isnan(boot_shuf[:, k])]
        shuf_floor = float(bk_shuf.mean()) if len(bk_shuf) > 0 else float('nan')

        # p-value: fraction of shuffled bootstrap samples exceeding observed
        p_val = float((bk_shuf >= obs_pb[k]).mean()) if len(bk_shuf) > 0 else 1.0
        pvals.append(p_val)
        valid_k.append(k)

        results.append({
            'go_id': terms[k],
            'name':  all_terms_info.get(terms[k], {}).get('name', ''),
            'n_pos_genes': int(n_pos[k]),
            'obs_pos_bias':  float(obs_pb[k]),
            'shuf_floor':    shuf_floor,
            'boot_mean_obs': float(bk_obs.mean()) if len(bk_obs) > 0 else float('nan'),
            'boot_ci_lo':    float(np.percentile(bk_obs, 2.5)) if len(bk_obs) > 0 else float('nan'),
            'boot_ci_hi':    float(np.percentile(bk_obs, 97.5)) if len(bk_obs) > 0 else float('nan'),
            'p_vs_shuffled': float(p_val),
            'q_BH':          float('nan'),  # filled below
        })

    # BH correction
    n_valid = len(pvals)
    ranked  = sorted(range(n_valid), key=lambda i: pvals[i])
    qvals   = [1.0] * n_valid
    prev_q  = 1.0
    for rank, idx in enumerate(ranked[::-1], 1):
        q = min(prev_q, pvals[idx] * n_valid / (n_valid - rank + 1))
        qvals[idx] = q; prev_q = q
    for ii, k in enumerate(valid_k):
        results[k]['q_BH'] = float(qvals[ii])

    sig = sum(1 for r in results if not np.isnan(r.get('q_BH', float('nan')))
              and r['q_BH'] < 0.05)
    all_pb = [r['obs_pos_bias'] for r in results
              if not np.isnan(r.get('obs_pos_bias', float('nan')))]
    shuf_mean = float(np.nanmean([r['shuf_floor'] for r in results
                                   if 'shuf_floor' in r]))

    print(f"    Significant BH<0.05: {sig}/{n_valid}")
    print(f"    Mean obs pos_bias: {np.mean(all_pb):.3f}  Shuf floor: {shuf_mean:.3f}")
    print(f"    Total time: {time.time()-t0:.1f}s")
    sys.stdout.flush()

    return {
        'domain': domain_label,
        'n_terms': K,
        'n_valid': n_valid,
        'n_sig_BH05': sig,
        'mean_pos_bias': float(np.mean(all_pb)),
        'shuf_floor':    shuf_mean,
        'per_term':      results,
    }

mf_pb = compute_pos_bias(WS_mf, G_mf, L_mf, mf_terms, 'MF', RNG_SEED)
bp_pb = compute_pos_bias(WS_bp, G_bp, L_bp, bp_terms, 'BP', RNG_SEED+1)
cc_pb = compute_pos_bias(WS_cc, G_cc, L_cc, cc_terms, 'CC', RNG_SEED+2)

# ── 5. Representational reversal ──────────────────────────────────────
print("\n[5] Representational reversal...")
sys.stdout.flush()

def representational_reversal(preds, domain_label):
    """
    within_var  = mean over all isoforms of: mean squared deviation from gene mean
    between_var = variance of gene-mean predictions (across genes)
    ratio = within_var / between_var → reversal if > 1.0
    """
    # Build gene-mean matrix
    gene_means = np.vstack([preds[idxs].mean(0) for idxs in multi_gene_idxs])
    global_mean = gene_means.mean(0)

    # Within-gene: mean over all isoforms of ||p(i) - p_gene||^2
    within_sq = []
    for gi, idxs in enumerate(multi_gene_idxs):
        diff = preds[idxs] - gene_means[gi]  # (n_iso, K)
        within_sq.append((diff ** 2).mean())
    within_var = float(np.mean(within_sq))

    # Between-gene: variance of gene means
    between_var = float(((gene_means - global_mean) ** 2).mean())

    ratio = within_var / between_var if between_var > 1e-12 else 0.0
    print(f"  {domain_label}: within={within_var:.6f}  between={between_var:.6f}  "
          f"ratio={ratio:.4f}  {'REVERSAL' if ratio > 1.0 else 'no reversal'}")
    sys.stdout.flush()
    return {'within_var': within_var, 'between_var': between_var,
            'ratio': ratio, 'reversal': ratio > 1.0}

rr_mf = representational_reversal(P_mf, 'MF')
rr_bp = representational_reversal(P_bp, 'BP')
rr_cc = representational_reversal(P_cc, 'CC')

# Reference: v15d muscle (from manuscript)
print("  v15d (muscle, 18 BP, from manuscript): within=0.00126  between=0.00070  ratio=1.80  REVERSAL")
sys.stdout.flush()

# ── 6. Intra-gene CV ──────────────────────────────────────────────────
print("\n[6] Intra-gene CV...")
sys.stdout.flush()

def intra_gene_cv(preds, domain_label):
    cvs = []
    for idxs in multi_gene_idxs:
        sc = preds[idxs].mean(1)  # mean across terms → scalar per isoform
        m = sc.mean()
        if m > 1e-8: cvs.append(sc.std() / m)
    cv_mean = float(np.mean(cvs))
    print(f"  {domain_label}: mean intra-gene CV = {cv_mean:.4f}  (N={len(cvs)} genes)")
    sys.stdout.flush()
    return {'mean_cv': cv_mean, 'n_genes': len(cvs)}

cv_mf = intra_gene_cv(P_mf, 'MF')
cv_bp = intra_gene_cv(P_bp, 'BP')
cv_cc = intra_gene_cv(P_cc, 'CC')

# ── 7. Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
domain_auprc = {'MF': 0.734, 'BP': 0.664, 'CC': 0.617}

for dom, pb, rr, cv_ in [('MF', mf_pb, rr_mf, cv_mf),
                          ('BP', bp_pb, rr_bp, cv_bp),
                          ('CC', cc_pb, rr_cc, cv_cc)]:
    print(f"  {dom}: AUPRC={domain_auprc[dom]:.3f}  "
          f"pos_bias_sig={pb['n_sig_BH05']}/{pb['n_valid']}  "
          f"mean_pb={pb['mean_pos_bias']:.3f}  "
          f"shuf_floor={pb['shuf_floor']:.3f}  "
          f"reversal={rr['reversal']}(ratio={rr['ratio']:.3f})")
sys.stdout.flush()

# Top pos_bias per domain
for dom, pb in [('MF', mf_pb), ('BP', bp_pb), ('CC', cc_pb)]:
    valid = [r for r in pb['per_term']
             if not np.isnan(r.get('obs_pos_bias', float('nan')))]
    top5 = sorted(valid, key=lambda r: r['obs_pos_bias'], reverse=True)[:5]
    print(f"\n  Top-5 {dom} pos_bias (obs vs shuf_floor):")
    for r in top5:
        sig = "***" if r.get('q_BH', 1) < 0.05 else "ns"
        print(f"    {r['go_id']}  {r.get('name','')[:30]:<30}  "
              f"pb={r['obs_pos_bias']:.3f} shuf={r['shuf_floor']:.3f}  {sig}")
sys.stdout.flush()

# ── 8. Save ────────────────────────────────────────────────────────────
results = {
    'metadata': {'n_isoforms': n_iso, 'n_multi_isoform_genes': n_multi, 'n_boot': N_BOOT},
    'auprc': domain_auprc,
    'pos_bias_summary': {
        'MF': {k: v for k, v in mf_pb.items() if k != 'per_term'},
        'BP': {k: v for k, v in bp_pb.items() if k != 'per_term'},
        'CC': {k: v for k, v in cc_pb.items() if k != 'per_term'},
    },
    'pos_bias_per_term': {
        'MF': mf_pb['per_term'],
        'BP': bp_pb['per_term'],
        'CC': cc_pb['per_term'],
    },
    'representational_reversal': {'MF': rr_mf, 'BP': rr_bp, 'CC': rr_cc,
                                   'v15d_muscle_ref': {'within_var': 0.00126,
                                                        'between_var': 0.00070,
                                                        'ratio': 1.80, 'reversal': True}},
    'intra_gene_cv': {'MF': cv_mf, 'BP': cv_bp, 'CC': cv_cc},
}

out_path = f'{OUT_DIR}/results.json'
with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n[Saved] {out_path}")
sys.stdout.flush()
