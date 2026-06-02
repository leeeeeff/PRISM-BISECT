#!/usr/bin/env python3
"""
assign_isoform_modules.py
==========================
각 brain isoform을 기능 모듈에 배정하고 isoform type별 분포 분석.

입력:
  - brain_full_672_scores.npy       (63994 × 672)
  - brain_go_modules_672.json       (go_id → module_id)
  - brain_full_extended_ids.npy     (63994 isoform IDs)
  - brain_full_extended_types.npy   (63994 types: Known/NIC/NNIC)
  - brain_full_extended_gene_ids.npy (63994 gene names)

출력:
  - brain_isoform_modules.tsv  : isoform_id, gene, type, primary_module, module_scores
  - brain_module_type_crosstab.json : Known/NIC/NNIC × module 분포
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, entropy

BASE    = Path('/home/welcome1/sw1686/DIFFUSE')
REPORTS = BASE / 'reports'
DEMO    = BASE / 'prism_app/data/demo'

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading score matrix and module assignments...")
S = np.load(REPORTS / 'brain_full_672_scores.npy')   # (63994, 672)

with open(REPORTS / 'brain_go_modules_672.json') as f:
    mod_data = json.load(f)

with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    go_meta = json.load(f)
go_ids = go_meta['go_ids']  # 672개, 순서 고정

n_iso, n_go = S.shape
n_modules = mod_data['n_modules']
modules   = mod_data['modules']   # str(mod_id) → {go_ids, size, label}
go_module = mod_data['go_module_map']  # go_id → int(mod_id)

print(f"  Isoforms: {n_iso}, GO terms: {n_go}, Modules: {n_modules}")

# GO index → module_id array
go_mod_arr = np.array([go_module[g] for g in go_ids], dtype=np.int32)  # (672,)

# ── Compute per-module mean score for each isoform ────────────────────────────
print("Computing isoform × module score matrix...")
# mod_scores[i, m] = mean score of isoform i for GO terms in module m
mod_ids_unique = sorted(modules.keys(), key=int)
n_mod = len(mod_ids_unique)
mod_id_to_idx = {mid: idx for idx, mid in enumerate(mod_ids_unique)}

mod_scores = np.zeros((n_iso, n_mod), dtype=np.float32)
for idx, mid in enumerate(mod_ids_unique):
    go_list = modules[mid]['go_ids']
    go_indices = [go_ids.index(g) for g in go_list if g in go_ids]
    if go_indices:
        mod_scores[:, idx] = S[:, go_indices].mean(axis=1)

# Primary module: argmax across module scores
primary_mod_idx = mod_scores.argmax(axis=1)    # 0-indexed
primary_mod_id  = np.array([int(mod_ids_unique[i]) for i in primary_mod_idx])

# Module specificity: inverse entropy of module score distribution
# High entropy = spread across many modules (generalist)
# Low entropy = concentrated in few modules (specialist)
eps = 1e-8
mod_probs = mod_scores / (mod_scores.sum(axis=1, keepdims=True) + eps)
specificity = 1.0 - (entropy(mod_probs.T + eps) / np.log(n_mod))

# ── Load isoform metadata ─────────────────────────────────────────────────────
print("Loading isoform metadata...")
iso_ids   = np.load(DEMO / 'brain_full_extended_ids.npy',      allow_pickle=True)
iso_types = np.load(DEMO / 'brain_full_extended_types.npy',    allow_pickle=True)
gene_ids  = np.load(DEMO / 'brain_full_extended_gene_ids.npy', allow_pickle=True)

iso_ids   = np.array([str(x) for x in iso_ids])
iso_types = np.array([str(x) for x in iso_types])
gene_ids  = np.array([str(x) for x in gene_ids])

# ── Build result DataFrame ────────────────────────────────────────────────────
print("Building result DataFrame...")
records = []
for i in range(n_iso):
    mid = int(primary_mod_id[i])
    mid_str = str(mid)
    mod_label = modules[mid_str]['label'] if mid_str in modules else f'Module_{mid}'
    records.append({
        'isoform_id':     iso_ids[i],
        'gene':           gene_ids[i],
        'type':           iso_types[i],
        'primary_module': mid,
        'module_label':   mod_label,
        'module_score':   float(mod_scores[i, primary_mod_idx[i]]),
        'specificity':    float(specificity[i]),
    })
df = pd.DataFrame(records)
print(f"  DataFrame: {df.shape}")
print(f"\n  Type distribution:")
print(df['type'].value_counts().to_string())
print(f"\n  Primary module distribution (top 10):")
print(df['primary_module'].value_counts().head(10).to_string())

# ── Cross-tabulation: type × module ──────────────────────────────────────────
print("\nCross-tabulation: isoform type × module...")
crosstab = pd.crosstab(df['type'], df['primary_module'])

# Chi-squared test
chi2, p_val, dof, expected = chi2_contingency(crosstab.values)
print(f"  Chi-squared: {chi2:.2f}, p={p_val:.2e}, dof={dof}")
print(f"  Interpretation: {'type distribution DIFFERS across modules (p<0.05)' if p_val < 0.05 else 'no significant difference'}")

# Per-type module preference
print("\nPer-type: top 3 preferred modules:")
for t in ['Known', 'NIC', 'NNIC']:
    sub = df[df['type'] == t]
    if len(sub) == 0:
        continue
    top_mods = sub['primary_module'].value_counts().head(3)
    top_str = ', '.join([f"M{mid}({n})" for mid, n in top_mods.items()])
    print(f"  {t:6s} (n={len(sub):6d}): {top_str}")

# ── Save ──────────────────────────────────────────────────────────────────────
tsv_path = REPORTS / 'brain_isoform_modules.tsv'
df.to_csv(tsv_path, sep='\t', index=False)
print(f"\nSaved: {tsv_path}")

# Module type distribution JSON
mod_type_dist = {}
for mid_str in mod_ids_unique:
    mid_int = int(mid_str)
    sub = df[df['primary_module'] == mid_int]
    dist = sub['type'].value_counts().to_dict()
    mod_type_dist[mid_str] = {
        'size': len(sub),
        'type_dist': dist,
        'pct_known': round(100 * dist.get('Known', 0) / max(len(sub), 1), 1),
        'pct_nic':   round(100 * dist.get('NIC',   0) / max(len(sub), 1), 1),
        'pct_nnic':  round(100 * dist.get('NNIC',  0) / max(len(sub), 1), 1),
        'label': modules[mid_str]['label'],
    }

summary = {
    'n_isoforms': int(n_iso),
    'n_modules': n_modules,
    'chi2': round(float(chi2), 2),
    'p_value': float(p_val),
    'dof': int(dof),
    'significant': bool(p_val < 0.05),
    'module_type_dist': mod_type_dist,
}
json_path = REPORTS / 'brain_module_type_crosstab.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {json_path}")

# ── Specificity by type ───────────────────────────────────────────────────────
print("\nModule specificity by isoform type (mean ± std):")
for t in ['Known', 'NIC', 'NNIC']:
    sub = df[df['type'] == t]['specificity']
    if len(sub):
        print(f"  {t:6s}: {sub.mean():.4f} ± {sub.std():.4f}  (n={len(sub)})")

print("\nAll done.")
