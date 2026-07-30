#!/usr/bin/env python3
"""
within_gene_metrics_all_domains_v2_truebrain.py
=================================================
TISSUE-MISLABELING BUGFIX RERUN (2026-07-18, line-310 decision, step 3/3).

Original within_gene_metrics_all_domains.py computes pos_bias/reversal/intra-gene-CV
for MF/BP/CC and its docstring / manuscript citation (natcomm_v0.md §153) calls this
"v17f* evaluated zero-shot on the full 277-term expanded GO ontology" (i.e. a BRAIN
zero-shot claim), but it loads `my_gene_list_fixed.npy` (MUSCLE held-out, 36,748
isoforms) as its test gene/isoform identity, and consumes v17f_star_bootstrap's
OLD (pre-2026-07-14-fix) MF predictions plus v17f_bp_cc_eval's MUSCLE-only BP/CC
predictions. All of it is muscle, not brain.

This rerun re-points every input at the TRUE brain isoform set (63,994 isoforms)
and the TRUE-brain predictions produced by:
  - v17f_star_bootstrap_ci_v2_truebrain.py  (MF, 2026-07-14)
  - v17f_bp_cc_eval_v2_truebrain.py         (BP/CC, this session)

pos_bias(k) definition, bootstrap, and BH-FDR procedure are UNCHANGED from the
original script -- only the tissue identity of predictions/labels is corrected.
"""

import os, sys, gzip, json, csv, time
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/truebrain_rerun_20260714/within_gene_metrics_all_domains'
os.makedirs(OUT_DIR, exist_ok=True)
TERMS_SRC_DIR    = '../../reports/v_expanded_gomf'  # term IDs tissue-independent, reused
MF_PREDS_DIR     = '../../reports/truebrain_rerun_20260714/v17f_star_bootstrap'
BP_CC_PREDS_DIR  = '../../reports/truebrain_rerun_20260714/v17f_bp_cc_eval'

N_BOOT   = 1000
RNG_SEED = 42

print("=" * 70)
print("  Within-Gene Metrics: MF / BP / CC -- TRUE BRAIN")
print("=" * 70)
sys.stdout.flush()

# ── 1. Gene / isoform identity (TRUE BRAIN) ────────────────────────────
print("\n[1] Loading TRUE BRAIN test set identities...")
sys.stdout.flush()

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

# TRUE BRAIN: gene names already plain symbols (e.g. 'A1BG'), no ENSG2SYM mapping needed.
te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_sym_arr   = np.array([clean(g) for g in te_genes_raw])
n_iso        = len(te_sym_arr)

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_arr): gene2idxs[g].append(i)

multi_gene_list = [g for g, v in gene2idxs.items() if len(v) >= 2]
multi_gene_idxs = [np.array(gene2idxs[g]) for g in multi_gene_list]
n_multi = len(multi_gene_list)

print(f"  {n_iso} isoforms, {len(gene2idxs)} genes, {n_multi} multi-isoform genes (TRUE BRAIN)")
sys.stdout.flush()

# ── 2. GO labels (term identities reused, tissue-independent) ─────────
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
with open(f'{TERMS_SRC_DIR}/expanded_go_per_term.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        all_terms_info[row['go_id']] = row

mf_terms = []
with open(f'{TERMS_SRC_DIR}/mf_domain_vs_prism.tsv') as f:
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

te_gids = np.array([sym2id.get(g, '__') for g in te_sym_arr])

def build_gene_label_matrix(terms):
    K = len(terms)
    mat = np.zeros((n_multi, K), dtype=np.float32)
    for k, go_id in enumerate(terms):
        pos_ids = go_genes[go_id]
        for gi, g in enumerate(multi_gene_list):
            gid = sym2id.get(g, '__')
            if gid in pos_ids:
                mat[gi, k] = 1.0
    return mat

print("  Building gene label matrices (TRUE BRAIN genes)...")
sys.stdout.flush()
L_mf = build_gene_label_matrix(mf_terms)
L_bp = build_gene_label_matrix(bp_terms)
L_cc = build_gene_label_matrix(cc_terms)
print(f"  L_mf: {L_mf.shape}  L_bp: {L_bp.shape}  L_cc: {L_cc.shape}")
sys.stdout.flush()

# ── 3. Predictions (TRUE BRAIN) ─────────────────────────────────────────
print("\n[3] Loading TRUE BRAIN predictions...")
sys.stdout.flush()
P_mf = np.load(f'{MF_PREDS_DIR}/v17f_star_preds.npy').astype(np.float32)
P_bp = np.load(f'{BP_CC_PREDS_DIR}/BP_preds_truebrain.npy').astype(np.float32)
P_cc = np.load(f'{BP_CC_PREDS_DIR}/CC_preds_truebrain.npy').astype(np.float32)
print(f"  P_mf:{P_mf.shape}  P_bp:{P_bp.shape}  P_cc:{P_cc.shape}")
assert P_mf.shape[0] == n_iso, f"MF preds isoform count {P_mf.shape[0]} != TRUE BRAIN {n_iso}"
assert P_bp.shape[0] == n_iso, f"BP preds isoform count {P_bp.shape[0]} != TRUE BRAIN {n_iso}"
assert P_cc.shape[0] == n_iso, f"CC preds isoform count {P_cc.shape[0]} != TRUE BRAIN {n_iso}"
sys.stdout.flush()

def compute_within_std_matrix(preds):
    n_genes = len(multi_gene_idxs)
    K = preds.shape[1]
    mat = np.zeros((n_genes, K), dtype=np.float32)
    for gi, idxs in enumerate(multi_gene_idxs):
        sc = preds[idxs]
        mat[gi] = sc.std(axis=0, ddof=0)
    return mat

print("\n[4] Computing within-gene std matrices...")
sys.stdout.flush()
t0 = time.time()
WS_mf = compute_within_std_matrix(P_mf)
WS_bp = compute_within_std_matrix(P_bp)
WS_cc = compute_within_std_matrix(P_cc)
print(f"  Done: {time.time()-t0:.1f}s")
sys.stdout.flush()

G_mf = P_mf.std(axis=0, ddof=0)
G_bp = P_bp.std(axis=0, ddof=0)
G_cc = P_cc.std(axis=0, ddof=0)

# ── 4. pos_bias computation (UNCHANGED procedure) ──────────────────────
def compute_pos_bias(within_std_mat, global_std, label_mat, terms, domain_label, rng_seed):
    rng = np.random.default_rng(rng_seed)
    K = len(terms)
    n_genes = within_std_mat.shape[0]
    eps = 1e-12

    t0 = time.time()
    print(f"\n  [{domain_label}] {K} terms, B={N_BOOT}...")
    sys.stdout.flush()

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

    boot_obs = np.full((N_BOOT, K), np.nan)
    for b in range(N_BOOT):
        boot_g = rng.integers(0, n_genes, n_genes)
        bws  = within_std_mat[boot_g]
        blbl = label_mat[boot_g]
        for k in range(K):
            pm = blbl[:, k] > 0
            if pm.sum() < 3 or global_std[k] < eps: continue
            boot_obs[b, k] = float(bws[pm, k].mean()) / (global_std[k] + eps)
        if (b+1) % 250 == 0:
            print(f"    obs boot {b+1}/{N_BOOT}: {time.time()-t0:.1f}s")
            sys.stdout.flush()

    boot_shuf = np.full((N_BOOT, K), np.nan)
    for b in range(N_BOOT):
        boot_g = rng.integers(0, n_genes, n_genes)
        bws    = within_std_mat[boot_g]
        blbl   = label_mat[boot_g].copy()
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
            'q_BH':          float('nan'),
        })

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
print("\n[5] Representational reversal (TRUE BRAIN)...")
sys.stdout.flush()

def representational_reversal(preds, domain_label):
    gene_means = np.vstack([preds[idxs].mean(0) for idxs in multi_gene_idxs])
    global_mean = gene_means.mean(0)

    within_sq = []
    for gi, idxs in enumerate(multi_gene_idxs):
        diff = preds[idxs] - gene_means[gi]
        within_sq.append((diff ** 2).mean())
    within_var = float(np.mean(within_sq))

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

# ── 6. Intra-gene CV ──────────────────────────────────────────────────
print("\n[6] Intra-gene CV (TRUE BRAIN)...")
sys.stdout.flush()

def intra_gene_cv(preds, domain_label):
    cvs = []
    for idxs in multi_gene_idxs:
        sc = preds[idxs].mean(1)
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
print("  FINAL SUMMARY (TRUE BRAIN)")
print("=" * 70)

# Load actual true-brain AUPRC point estimates rather than hardcoding muscle values.
bp_cc_results_path = f'{BP_CC_PREDS_DIR}/results_truebrain.json'
domain_auprc = {'MF': 0.647, 'BP': None, 'CC': None}
if os.path.exists(bp_cc_results_path):
    with open(bp_cc_results_path) as fh:
        bp_cc_r = json.load(fh)
    domain_auprc['BP'] = bp_cc_r['BP']['point']
    domain_auprc['CC'] = bp_cc_r['CC']['point']

for dom, pb, rr, cv_ in [('MF', mf_pb, rr_mf, cv_mf),
                          ('BP', bp_pb, rr_bp, cv_bp),
                          ('CC', cc_pb, rr_cc, cv_cc)]:
    auprc_disp = f"{domain_auprc[dom]:.3f}" if domain_auprc[dom] is not None else "N/A"
    print(f"  {dom}: AUPRC(TRUE BRAIN)={auprc_disp}  "
          f"pos_bias_sig={pb['n_sig_BH05']}/{pb['n_valid']}  "
          f"mean_pb={pb['mean_pos_bias']:.3f}  "
          f"shuf_floor={pb['shuf_floor']:.3f}  "
          f"reversal={rr['reversal']}(ratio={rr['ratio']:.3f})")
sys.stdout.flush()

for dom, pb in [('MF', mf_pb), ('BP', bp_pb), ('CC', cc_pb)]:
    valid = [r for r in pb['per_term']
             if not np.isnan(r.get('obs_pos_bias', float('nan')))]
    top5 = sorted(valid, key=lambda r: r['obs_pos_bias'], reverse=True)[:5]
    print(f"\n  Top-5 {dom} pos_bias TRUE BRAIN (obs vs shuf_floor):")
    for r in top5:
        sig = "***" if r.get('q_BH', 1) < 0.05 else "ns"
        print(f"    {r['go_id']}  {r.get('name','')[:30]:<30}  "
              f"pb={r['obs_pos_bias']:.3f} shuf={r['shuf_floor']:.3f}  {sig}")
sys.stdout.flush()

# ── 8. Save ────────────────────────────────────────────────────────────
results = {
    'metadata': {'test_set': 'TRUE_BRAIN_63994', 'n_isoforms': n_iso,
                 'n_multi_isoform_genes': n_multi, 'n_boot': N_BOOT},
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

out_path = f'{OUT_DIR}/results_truebrain.json'
with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n[Saved] {out_path}")
sys.stdout.flush()
