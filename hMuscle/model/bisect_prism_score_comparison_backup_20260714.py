#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bisect_prism_score_comparison.py  (Option B)
---------------------------------------------
Uses saved v17f* predictions to check isoform-level discrimination
for BISECT-validated genes.

For each BISECT target gene:
  1. Find all isoforms in the brain test set
  2. Get v17f* predicted scores for GO terms relevant to that gene
  3. Compute within-gene score variance (proxy for isoform discrimination)
  4. Compare to between-gene score variance (reference)

If within-gene variance >> 0:
  → model assigns different scores to different isoforms of same gene
  → isoform-level discrimination signal exists (even without ground-truth isoform labels)

Additionally compute:
  - Discrimination ratio: max_score / mean_score within gene
  - Normalized spread: std(within-gene) / std(all-gene)
"""

import os, json, gzip
import numpy as np
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
PRED_DIR  = '../../reports/v17f_star_bootstrap'
OUT_DIR   = '../../reports/bisect_prism_scores'
os.makedirs(OUT_DIR, exist_ok=True)

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

print("=" * 65)
print("  BISECT × PRISM: isoform discrimination score analysis")
print("=" * 65)

# ── 1. Load saved v17f* predictions ──────────────────────────────
print("\n[1] Loading saved v17f* predictions...")
preds = np.load(f'{PRED_DIR}/v17f_star_preds.npy')   # (36748, 82)
Y_te  = np.load(f'{PRED_DIR}/Y_te.npy')               # (36748, 82)
print(f"  preds: {preds.shape}  Y_te: {Y_te.shape}")

# ── 2. Load gene IDs + GO terms ───────────────────────────────────
print("\n[2] Loading gene IDs and GO terms...")
ENSG2SYM = {}
SYM2ENSG = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ENSG2SYM[p[0]] = p[4]
            SYM2ENSG[p[4]] = p[0]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]
te_ensg_list = [clean(g).split('.')[0] for g in te_genes_raw]

gene2idxs = defaultdict(list)
for i, g in enumerate(te_sym_list): gene2idxs[g].append(i)
ensg2idxs = defaultdict(list)
for i, g in enumerate(te_ensg_list): ensg2idxs[g].append(i)

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

# GO term name lookup
go_names = {}
try:
    with gzip.open(f'{ANNOT_DIR}/../go-basic.obo.gz', 'rt') as f:
        cur_id, cur_name = None, None
        for line in f:
            if line.startswith('id: GO:'): cur_id = line.strip().split(' ')[1]
            elif line.startswith('name: '): go_names[cur_id] = line.strip()[6:] if cur_id else None
except:
    pass  # optional

print(f"  {len(te_sym_list)} test isoforms, {len(gene2idxs)} unique genes")
print(f"  {len(mf_terms)} MF GO terms")

# ── 3. BISECT target genes + relevant GO terms ────────────────────
# GO terms relevant to each gene (subset of 82 MF terms)
# Complex I: oxidoreductase, electron transport, NADH dehydrogenase
# DOCK11: GEF activity, GTPase regulation
# KIF21B: motor activity, ATPase

BISECT_TARGETS = {
    'NDUFS8':  {
        'ensg': 'ENSG00000110717',
        'function': 'Complex I subunit — NADH:ubiquinone oxidoreductase, MTS targeting',
        'go_keywords': ['oxidoreductase', 'electron', 'NADH', 'ubiquinone', 'dehydrogenase',
                        'mitochondr', 'catalytic', 'transferase'],
        'bisect_note': 'NIC isoform has disrupted MTS; should lose mitochondrial/oxidoreductase scores',
        'expected': 'isoform_with_lower_oxidoreductase_score = NIC (AD-enriched)',
    },
    'DOCK11':  {
        'ensg': 'ENSG00000147251',
        'function': 'Rho GEF — GTPase exchange factor activity',
        'go_keywords': ['GTPase', 'nucleotide', 'exchange', 'signal', 'binding', 'transferase'],
        'bisect_note': 'DOCK11-201 isoform adds GEF-activating domain in AD',
        'expected': 'isoform with higher GEF activity score = DOCK11-201',
    },
    'NDUFAF5': {
        'ensg': 'ENSG00000101247',
        'function': 'Complex I assembly factor — methyltransferase',
        'go_keywords': ['methyltransferase', 'transferase', 'oxidoreductase', 'catalytic'],
        'bisect_note': 'Multiple isoforms; AD-specific alternative usage of assembly factor isoforms',
        'expected': 'score variation reveals assembly factor isoform preferences',
    },
    'NDUFS4':  {
        'ensg': 'ENSG00000164258',
        'function': 'Complex I subunit — electron transfer',
        'go_keywords': ['oxidoreductase', 'electron', 'NADH', 'catalytic'],
        'bisect_note': 'NDUFS4-201 AD-enriched in Excitatory + Inhibitory neurons',
        'expected': 'AD-enriched isoform may show different oxidoreductase scores',
    },
    'NDUFS7':  {
        'ensg': 'ENSG00000115286',
        'function': 'Complex I subunit — iron-sulfur protein',
        'go_keywords': ['oxidoreductase', 'electron', 'iron', 'NADH', 'catalytic'],
        'bisect_note': 'NDUFS7-202 AD-enriched in Excitatory neurons',
        'expected': 'AD-enriched isoform may differ in iron-sulfur / oxidoreductase activity',
    },
}

# ── 4. Find relevant GO indices for each gene ─────────────────────
def find_relevant_go_idxs(keywords, mf_terms):
    """Find GO term indices matching any keyword (case-insensitive)."""
    hits = []
    for i, go in enumerate(mf_terms):
        go_name = go_names.get(go, go).lower()
        if any(kw.lower() in go_name for kw in keywords):
            hits.append(i)
    return hits

# ── 5. Compute within-gene score dispersion ───────────────────────
def within_gene_dispersion(preds, idxs, go_idxs):
    """
    For each relevant GO term, compute the spread of predictions
    across the isoforms of this gene.

    Returns:
      mean_std:  average std of predictions across GO terms
      max_range: max (max_score - min_score) across GO terms
      scores:    (n_isoforms, n_go) score matrix
    """
    if len(idxs) < 2 or not go_idxs:
        return None
    scores = preds[np.array(idxs)][:, go_idxs]   # (n_iso, n_go)
    stds   = scores.std(axis=0)                    # per GO term
    ranges = scores.max(axis=0) - scores.min(axis=0)
    return {
        'n_isoforms': len(idxs),
        'n_go_terms': len(go_idxs),
        'mean_std':   float(stds.mean()),
        'max_range':  float(ranges.max()),
        'mean_range': float(ranges.mean()),
        'scores':     scores.tolist(),
        'go_idxs':    go_idxs,
    }

# Population-level reference: between-gene std
all_gene_means = np.array([preds[idxs].mean(axis=0) for idxs in gene2idxs.values()])
between_gene_std = all_gene_means.std(axis=0).mean()
print(f"\n  Between-gene std (reference): {between_gene_std:.4f}")

# ── 6. Analyze each BISECT gene ───────────────────────────────────
print("\n[3] Analyzing BISECT target genes...")
results = {}

for gene, info in BISECT_TARGETS.items():
    ensg    = info['ensg']
    idxs    = ensg2idxs.get(ensg, [])
    go_idxs = find_relevant_go_idxs(info['go_keywords'], mf_terms)
    relevant_go_names = [go_names.get(mf_terms[i], mf_terms[i]) for i in go_idxs]

    print(f"\n  {gene} ({ensg}): {len(idxs)} isoforms, {len(go_idxs)} relevant GO terms")
    print(f"    Function: {info['function']}")

    if not idxs:
        print(f"    NOT FOUND in test set")
        results[gene] = {'found': False}
        continue

    disp = within_gene_dispersion(preds, idxs, go_idxs)
    if disp is None:
        print(f"    No valid GO terms or <2 isoforms")
        results[gene] = {'found': True, 'error': 'no valid terms'}
        continue

    # Normalized dispersion: within-gene / between-gene
    norm_std   = disp['mean_std'] / between_gene_std if between_gene_std > 0 else 0
    norm_range = disp['max_range'] / between_gene_std if between_gene_std > 0 else 0

    print(f"    Relevant GO terms ({len(go_idxs)}): {relevant_go_names[:4]}{'...' if len(go_idxs)>4 else ''}")
    print(f"    Within-gene score std:   {disp['mean_std']:.4f}  (norm: {norm_std:.3f} × between-gene)")
    print(f"    Max score range:          {disp['max_range']:.4f}  (norm: {norm_range:.3f} × between-gene)")

    # Per-isoform scores
    scores = np.array(disp['scores'])
    for iso_i, iso_idx in enumerate(idxs):
        mean_score = scores[iso_i, :].mean()
        max_score  = scores[iso_i, :].max()
        go_name_best = relevant_go_names[scores[iso_i, :].argmax()] if go_idxs else 'n/a'
        print(f"    Isoform[{iso_i+1}] test_idx={iso_idx}: "
              f"mean={mean_score:.4f}  max={max_score:.4f} ({go_name_best})")

    if norm_std > 0.1:
        disc_level = "STRONG"
    elif norm_std > 0.03:
        disc_level = "MODERATE"
    else:
        disc_level = "WEAK"

    print(f"    Discrimination: {disc_level} (norm_std={norm_std:.3f})")
    print(f"    BISECT note: {info['bisect_note']}")

    results[gene] = {
        'found': True,
        'n_isoforms': len(idxs),
        'n_go_terms': len(go_idxs),
        'mean_std': disp['mean_std'],
        'max_range': disp['max_range'],
        'norm_std_vs_between_gene': norm_std,
        'norm_range_vs_between_gene': norm_range,
        'discrimination_level': disc_level,
        'relevant_go_names': relevant_go_names,
        'per_isoform_mean_scores': [float(scores[i,:].mean()) for i in range(len(idxs))],
    }

# ── 7. Summary ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  BISECT × PRISM Isoform Discrimination Summary")
print(f"{'='*65}")
print(f"  Between-gene std (reference): {between_gene_std:.4f}")
print(f"")
print(f"  {'Gene':<10} {'n_iso':>6} {'norm_std':>10} {'norm_range':>11} {'Level':<12}")
print(f"  {'-'*55}")
for gene, r in results.items():
    if not r.get('found', False): continue
    if 'error' in r: continue
    print(f"  {gene:<10} {r['n_isoforms']:>6} {r['norm_std_vs_between_gene']:>10.3f} "
          f"{r['norm_range_vs_between_gene']:>11.3f} {r['discrimination_level']:<12}")

print(f"\n  Interpretation:")
print(f"    norm_std > 0.1  → STRONG: model clearly differentiates isoforms")
print(f"    norm_std > 0.03 → MODERATE: some isoform-level signal present")
print(f"    norm_std < 0.03 → WEAK: model treats isoforms near-identically")

all_norm_stds = [r['norm_std_vs_between_gene'] for r in results.values()
                 if r.get('found') and 'norm_std_vs_between_gene' in r]
if all_norm_stds:
    overall = np.mean(all_norm_stds)
    print(f"\n  Mean norm_std across BISECT genes: {overall:.3f}")
    if overall > 0.1:
        print(f"  → v17f* DOES discriminate isoforms of BISECT genes (publishable evidence)")
    elif overall > 0.03:
        print(f"  → v17f* shows partial isoform discrimination (moderate evidence)")
    else:
        print(f"  → v17f* treats BISECT isoforms near-identically (gene-level classifier)")

results['_meta'] = {
    'between_gene_std': float(between_gene_std),
    'model': 'v17f_star_no_tpsi',
    'n_test_isoforms': len(te_sym_list),
}
with open(f'{OUT_DIR}/results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n[Saved] {OUT_DIR}/results.json")
print(f"{'='*65}")
