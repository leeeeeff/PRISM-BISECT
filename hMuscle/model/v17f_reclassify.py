#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17f_reclassify.py
-------------------
Per-term v17f* AUPRC computation and H2 reclassification.
Uses saved preds from:
  - MF (82 terms): B2_preds.npy (proxy, Δ=0.006 vs v17f*)
  - BP (103 terms): BP_preds.npy
  - CC (93 terms): CC_preds.npy
  - L3 (23 terms): C0_seq_preds.npy
  - L4 (112 terms): L4_preds.npy (run after v17f_l4_cellstate.py)
Output: v17f_per_term_auprc.tsv + new classification table
"""

import os, json, gzip
import numpy as np
import csv
from collections import defaultdict
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
OUT_DIR   = '../../reports/v17f_reclassify'
os.makedirs(OUT_DIR, exist_ok=True)

PREDS_MF = '../../reports/v17f_b2_bootstrap/B2_preds.npy'   # proxy
Y_TE_MF  = '../../reports/v17f_b2_bootstrap/Y_te.npy'
PREDS_BP = '../../reports/v17f_bp_cc_eval/BP_preds.npy'
PREDS_CC = '../../reports/v17f_bp_cc_eval/CC_preds.npy'
PREDS_L3 = '../../reports/v17f_l3_recovery/C0_seq_preds.npy'
PREDS_L4 = '../../reports/v17f_l4_cellstate/L4_preds.npy'
Y_TE_L4  = '../../reports/v17f_l4_cellstate/Y_te.npy'

print("=" * 65)
print("  v17f* per-term reclassification")
print("=" * 65)

# ── 1. Shared gene/GO setup ──────────────────────────────────────
print("\n[1] Loading gene IDs & GO labels...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]

te_genes_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
te_sym_list  = [ENSG2SYM.get(clean(g).split('.')[0], clean(g).split('.')[0])
                for g in te_genes_raw]

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

tr_ids    = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)
go_tr  = defaultdict(set)
go_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        go_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_tr[p[2]].add(p[1])

def build_Y_te(terms):
    return np.stack([
        np.array([1.0 if sym2id.get(s, '__') in go_all[go_id] else 0.0
                  for s in te_sym_list], dtype=np.float32)
        for go_id in terms
    ], axis=1)

# ── 2. Load term lists ───────────────────────────────────────────
all_terms = {}
with open('../../reports/v_expanded_gomf/expanded_go_per_term.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        all_terms[row['go_id']] = row

mf_terms = list(open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv').readlines())
mf_ids   = [l.split('\t')[0] for l in mf_terms[1:]]  # skip header

bp_ids = [r['go_id'] for r in all_terms.values()
          if r['cat']=='BP' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2]
cc_ids = [r['go_id'] for r in all_terms.values()
          if r['cat']=='CC' and int(r['n_pos_te'])>=2 and int(r['n_pos_tr'])>=2]

h2_info = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        h2_info[row['go_id']] = row

l3_ids = [go_id for go_id, r in h2_info.items() if r['layer'] == 'L3_CellType']
l4_ids = [go_id for go_id, r in h2_info.items() if r['layer'] == 'L4_CellState']

print(f"  MF:{len(mf_ids)} BP:{len(bp_ids)} CC:{len(cc_ids)} L3:{len(l3_ids)} L4:{len(l4_ids)}")

# ── 3. Per-term AUPRC computation ────────────────────────────────
def per_term_auprc(preds, Y_te, term_ids):
    result = {}
    for i, go_id in enumerate(term_ids):
        if Y_te[:, i].sum() >= 2:
            result[go_id] = float(average_precision_score(Y_te[:, i], preds[:, i]))
        else:
            result[go_id] = float('nan')
    return result

print("\n[2] Computing per-term AUPRC from saved preds...")

# MF — use B2 preds as proxy
preds_mf = np.load(PREDS_MF)
Y_mf     = np.load(Y_TE_MF)
auprc_mf = per_term_auprc(preds_mf, Y_mf, mf_ids)
print(f"  MF: mean={np.nanmean(list(auprc_mf.values())):.4f}  (B2 proxy, v17f*=0.734)")

# BP
preds_bp = np.load(PREDS_BP)
Y_bp     = build_Y_te(bp_ids)
auprc_bp = per_term_auprc(preds_bp, Y_bp, bp_ids)
print(f"  BP: mean={np.nanmean(list(auprc_bp.values())):.4f}")

# CC
preds_cc = np.load(PREDS_CC)
Y_cc     = build_Y_te(cc_ids)
auprc_cc = per_term_auprc(preds_cc, Y_cc, cc_ids)
print(f"  CC: mean={np.nanmean(list(auprc_cc.values())):.4f}")

# L3
preds_l3 = np.load(PREDS_L3)
Y_l3     = build_Y_te(l3_ids)
auprc_l3 = per_term_auprc(preds_l3, Y_l3, l3_ids)
print(f"  L3: mean={np.nanmean(list(auprc_l3.values())):.4f}")

# L4
auprc_l4 = {}
if os.path.exists(PREDS_L4):
    preds_l4 = np.load(PREDS_L4)
    Y_l4     = np.load(Y_TE_L4) if os.path.exists(Y_TE_L4) else build_Y_te(l4_ids)
    auprc_l4 = per_term_auprc(preds_l4, Y_l4, l4_ids)
    print(f"  L4: mean={np.nanmean(list(auprc_l4.values())):.4f}")
else:
    print(f"  L4: preds not yet available — run v17f_l4_cellstate.py first")

# ── 4. Merge all per-term AUPRC ──────────────────────────────────
all_auprc = {}
all_auprc.update(auprc_mf)
all_auprc.update(auprc_bp)
all_auprc.update(auprc_cc)
all_auprc.update(auprc_l3)
all_auprc.update(auprc_l4)

# ── 5. New classification ─────────────────────────────────────────
print("\n[3] New classification based on v17f* per-term AUPRC...")

def classify_v17f(auprc):
    if np.isnan(auprc): return 'Insufficient_data'
    if auprc >= 0.65:   return 'T1_Sequence_Strong'   # seq model handles well
    if auprc >= 0.50:   return 'T2_Sequence_Partial'  # recoverable
    if auprc >= 0.35:   return 'T3_Sequence_Limited'  # hard for seq
    return                     'T4_Sequence_Floor'    # genuine limit

rows_out = []
for go_id, info in all_terms.items():
    if go_id not in all_auprc: continue
    v17f_auprc = all_auprc[go_id]
    prism_brain = float(info['prism_brain'])
    old_layer = h2_info.get(go_id, {}).get('layer', 'non_H2')
    new_tier  = classify_v17f(v17f_auprc)
    rows_out.append({
        'go_id':       go_id,
        'name':        info['name'],
        'cat':         info['cat'],
        'n_pos_te':    info['n_pos_te'],
        'prism_brain': f'{prism_brain:.4f}',
        'v17f_auprc':  f'{v17f_auprc:.4f}' if not np.isnan(v17f_auprc) else 'nan',
        'delta':       f'{v17f_auprc - prism_brain:.4f}' if not np.isnan(v17f_auprc) else 'nan',
        'old_h2_layer': old_layer,
        'new_tier':    new_tier,
    })

# Write output
with open(f'{OUT_DIR}/v17f_per_term_auprc.tsv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows_out)

# ── 6. Summary statistics ─────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RECLASSIFICATION SUMMARY (v17f* per-term AUPRC)")
print(f"{'='*65}")

tiers = ['T1_Sequence_Strong', 'T2_Sequence_Partial', 'T3_Sequence_Limited', 'T4_Sequence_Floor']
all_values = [(r['new_tier'], float(r['v17f_auprc']), r['old_h2_layer'])
              for r in rows_out if r['v17f_auprc'] != 'nan']
for tier in tiers:
    vals = [v for t, v, _ in all_values if t == tier]
    h2   = sum(1 for t, v, h in all_values if t == tier and h != 'non_H2')
    if vals:
        print(f"  {tier}: {len(vals):3d} terms  mean={np.mean(vals):.4f}  (H2 original: {h2})")

print(f"\n  Originally H2 (168 terms) — new distribution:")
h2_vals = [(t, v) for t, v, h in all_values if h != 'non_H2']
for tier in tiers:
    n = sum(1 for t, v in h2_vals if t == tier)
    pct = 100 * n / len(h2_vals) if h2_vals else 0
    print(f"    {tier}: {n:3d} ({pct:.0f}%)")

# Per-domain breakdown
print(f"\n  Per original H2 layer → v17f* tier distribution:")
for old_layer in ['L2_Structural', 'L3_CellType', 'L4_CellState']:
    layer_rows = [(t, v) for t, v, h in all_values if h == old_layer]
    if not layer_rows: continue
    vals = [v for _, v in layer_rows]
    tier_counts = {tier: sum(1 for t, _ in layer_rows if t == tier) for tier in tiers}
    print(f"    {old_layer}: mean={np.mean(vals):.4f}  " +
          "  ".join(f"{t.split('_')[0]}:{n}" for t, n in tier_counts.items() if n > 0))

# Specific: how many originally L4 are now T1/T2?
if auprc_l4:
    l4_t1t2 = sum(1 for go_id in l4_ids if all_auprc.get(go_id, 0) >= 0.50)
    print(f"\n  L4→T1/T2 (now recoverable): {l4_t1t2}/{len(l4_ids)}")

print(f"\n[Saved] {OUT_DIR}/v17f_per_term_auprc.tsv")

# Also save summary JSON
summary = {
    'tier_counts': {tier: sum(1 for t, v, _ in all_values if t == tier) for tier in tiers},
    'h2_tier_counts': {tier: sum(1 for t, v, h in all_values if t == tier and h != 'non_H2') for tier in tiers},
    'domain_means': {
        'MF': float(np.nanmean(list(auprc_mf.values()))),
        'BP': float(np.nanmean(list(auprc_bp.values()))),
        'CC': float(np.nanmean(list(auprc_cc.values()))),
        'L3': float(np.nanmean(list(auprc_l3.values()))),
        'L4': float(np.nanmean(list(auprc_l4.values()))) if auprc_l4 else None,
    },
    'note': 'MF uses B2_preds as proxy (v17f* Δ=0.006). L4 requires v17f_l4_cellstate.py.',
}
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[Saved] {OUT_DIR}/summary.json")
