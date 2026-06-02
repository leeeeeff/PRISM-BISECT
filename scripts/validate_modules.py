#!/usr/bin/env python3
"""
validate_modules.py
====================
GO-GO 모듈의 생물학적 타당성 검증 (PRISM score 독립).

[1] Module semantic coherence
    - 각 모듈 내 GO-GO 의미적 유사도 (Pearson r of GO-GO annotation co-occurrence)
    - Within-module vs random-shuffled-module 비교
    - Note: true semantic similarity (Lin/Resnik) requires goatools+GO hierarchy
            → 여기서는 GO annotation co-occurrence Jaccard similarity 사용 (lighter)

[2] Score-threshold filtered assignment
    - max score > 0.3 이상인 isoform만 신뢰 배정
    - Module 3 (low-score dump) 제거 후 재분석

[3] BISECT case module check
    - 84 PASS cases: S1/S3 isoforms의 module 배정 확인
    - known vs novel isoform이 다른 모듈에 배정되는지 확인
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr
from scipy.spatial.distance import jaccard

BASE    = Path('/home/welcome1/sw1686/DIFFUSE')
REPORTS = BASE / 'reports'
DEMO    = BASE / 'prism_app/data/demo'

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
S = np.load(REPORTS / 'brain_full_672_scores.npy')  # (63994, 672)
df = pd.read_csv(REPORTS / 'brain_isoform_modules.tsv', sep='\t')

with open(REPORTS / 'brain_go_modules_672.json') as f:
    mod_data = json.load(f)
with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    go_meta = json.load(f)

go_ids   = go_meta['go_ids']
modules  = mod_data['modules']
go_mod   = mod_data['go_module_map']  # go_id → mod_id
n_go     = len(go_ids)
go_idx   = {g: i for i, g in enumerate(go_ids)}

# ── [1] Module semantic coherence via GO annotation co-occurrence ──────────────
print("\n=== [1] Module Semantic Coherence ===")
print("Building GO annotation co-occurrence matrix from gene annotations...")

gene_go = {}
with open(BASE / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        gene_go[parts[0]] = set(parts[1:])

# Binary annotation matrix: gene × GO
annot_genes = sorted(gene_go.keys())
n_gene = len(annot_genes)
A = np.zeros((n_gene, n_go), dtype=np.float32)
for i, gene in enumerate(annot_genes):
    for go in gene_go[gene]:
        if go in go_idx:
            A[i, go_idx[go]] = 1.0

# GO-GO Jaccard similarity (annotation-based, independent of PRISM scores)
print(f"  Computing GO-GO Jaccard from annotations ({n_go}×{n_go})...")
# Jaccard(GO_i, GO_j) = |genes with both| / |genes with either|
# Use dot product trick: intersect = A.T @ A, union = col_sums + col_sums.T - intersect
col_sums = A.sum(axis=0)  # (n_go,)
intersect = A.T @ A       # (n_go, n_go)
union = col_sums[:, None] + col_sums[None, :] - intersect
union = np.maximum(union, 1e-8)
Jac = intersect / union   # GO-GO Jaccard (annotation-based)
np.fill_diagonal(Jac, 0.0)

# Within-module vs cross-module Jaccard
within_sims = []
cross_sims  = []
mod_ids = sorted(modules.keys(), key=int)

for mid_str in mod_ids:
    go_list = modules[mid_str]['go_ids']
    indices = [go_idx[g] for g in go_list if g in go_idx]
    if len(indices) < 2:
        continue
    # Within-module pairs
    for ii in range(len(indices)):
        for jj in range(ii+1, len(indices)):
            within_sims.append(Jac[indices[ii], indices[jj]])

# Random sample of cross-module pairs (same count as within)
rng = np.random.default_rng(42)
n_sample = min(len(within_sims) * 2, 50000)
all_indices = list(range(n_go))
for _ in range(n_sample):
    i, j = rng.choice(all_indices, size=2, replace=False)
    if go_mod.get(go_ids[i]) != go_mod.get(go_ids[j]):
        cross_sims.append(Jac[i, j])

within_arr = np.array(within_sims)
cross_arr  = np.array(cross_sims)
stat, p = mannwhitneyu(within_arr, cross_arr, alternative='greater')

print(f"  Within-module Jaccard: {within_arr.mean():.4f} ± {within_arr.std():.4f}  (n={len(within_arr)})")
print(f"  Cross-module Jaccard:  {cross_arr.mean():.4f} ± {cross_arr.std():.4f}  (n={len(cross_arr)})")
print(f"  Mann-Whitney U: stat={stat:.0f}, p={p:.2e}")
print(f"  → {'COHERENT (within > cross, p<0.05)' if p < 0.05 else 'Not significant'}")

# Per-module coherence
print("\n  Per-module coherence (top 10 by mean within-Jaccard):")
mod_coh = {}
for mid_str in mod_ids:
    go_list = modules[mid_str]['go_ids']
    indices = [go_idx[g] for g in go_list if g in go_idx]
    if len(indices) < 2:
        mod_coh[mid_str] = 0.0
        continue
    pairs = []
    for ii in range(len(indices)):
        for jj in range(ii+1, len(indices)):
            pairs.append(Jac[indices[ii], indices[jj]])
    mod_coh[mid_str] = float(np.mean(pairs))

for mid_str, coh in sorted(mod_coh.items(), key=lambda x: -x[1])[:10]:
    label = modules[mid_str]['label'][:55]
    print(f"    Module {mid_str:>3s} (coh={coh:.4f}, n={modules[mid_str]['size']:2d}): {label}")

# ── [2] Score-threshold filtered analysis ─────────────────────────────────────
print("\n=== [2] Score-threshold filtered (max score > 0.3) ===")
max_scores = S.max(axis=1)
df['max_score'] = max_scores
df_hi = df[df['max_score'] > 0.3].copy()
print(f"  High-confidence isoforms: {len(df_hi)}/{len(df)} ({100*len(df_hi)/len(df):.1f}%)")

print("\n  Per-type top 3 modules (high-confidence only):")
for t in ['known', 'nic', 'nnic']:
    sub = df_hi[df_hi['type'] == t]
    top = sub['primary_module'].value_counts().head(3)
    print(f"\n  {t.upper()} (n={len(sub)}):")
    for mid, cnt in top.items():
        label = modules.get(str(mid), {}).get('label', '?')[:55]
        pct = 100*cnt/len(sub)
        print(f"    Module {mid:2d} ({pct:4.1f}%): {label}")

# Novel isoform enrichment in brain-specific modules (high-conf only)
print("\n  Brain-specific module NIC+NNIC enrichment (high-conf):")
bg_novel_hi = len(df_hi[df_hi['type'].isin(['nic','nnic'])]) / len(df_hi)
print(f"  Background NIC+NNIC: {100*bg_novel_hi:.1f}%")
brain_mods = [37, 36, 13, 14, 11, 35, 12]
for mid in brain_mods:
    mid_str = str(mid)
    if mid_str not in modules: continue
    sub = df_hi[df_hi['primary_module'] == mid]
    if len(sub) < 10: continue
    nf = len(sub[sub['type'].isin(['nic','nnic'])]) / len(sub)
    label = modules[mid_str]['label'][:50]
    flag = '↓ underrepresented' if nf < bg_novel_hi * 0.85 else ('↑ enriched' if nf > bg_novel_hi * 1.15 else '~')
    print(f"    M{mid:2d} (n={len(sub):4d}, NIC+NNIC={100*nf:.1f}%) {flag}: {label}")

# ── [3] BISECT case check ─────────────────────────────────────────────────────
print("\n=== [3] BISECT PASS case module check ===")
bisect_path = BASE / 'Final_analysis/pipeline_bioanalysis/results/bisect_results.json'
if not bisect_path.exists():
    # Try alternative paths
    for p in BASE.rglob('bisect_results*.json'):
        bisect_path = p
        break

if bisect_path.exists():
    with open(bisect_path) as f:
        bisect = json.load(f)
    pass_cases = [c for c in bisect if c.get('stage1_pass') or c.get('pass')]
    print(f"  BISECT PASS cases: {len(pass_cases)}")

    # Check isoform module assignments for BISECT cases
    iso_id_to_row = {row['isoform_id']: row for _, row in df.iterrows()}
    found = 0
    module_diverge = 0
    for case in pass_cases[:20]:
        novel_iso = case.get('isoform_id') or case.get('novel_isoform')
        canon_iso = case.get('canonical_isoform') or case.get('reference_isoform')
        if not novel_iso: continue
        r = iso_id_to_row.get(novel_iso)
        if r is None: continue
        found += 1
        novel_mod = r['primary_module']
        print(f"    {novel_iso}: M{novel_mod} ({modules.get(str(novel_mod),{}).get('label','?')[:40]})")
    print(f"  Found in df: {found}/20 cases checked")
else:
    print(f"  BISECT results not found at {bisect_path}")
    print("  → Checking alternative: gene-level module divergence for known BISECT genes")
    bisect_genes = ['NDUFS4', 'NDUFS7', 'NDUFS8', 'DLG1', 'KIF21B', 'PTPRF']
    for gene in bisect_genes:
        sub = df[df['gene'] == gene]
        if len(sub) == 0: continue
        mods = sub['primary_module'].unique()
        types = sub['type'].unique()
        print(f"    {gene} (n={len(sub)} isoforms): modules={sorted(mods)}, types={sorted(types)}")

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    'validation': {
        'within_module_jaccard_mean': round(float(within_arr.mean()), 4),
        'cross_module_jaccard_mean':  round(float(cross_arr.mean()), 4),
        'mannwhitney_p': float(p),
        'significant': bool(p < 0.05),
    },
    'high_conf_isoforms': int(len(df_hi)),
    'high_conf_pct': round(100*len(df_hi)/len(df), 1),
    'background_novel_rate_hi': round(100*bg_novel_hi, 1),
    'module_coherence': {k: round(v, 4) for k, v in mod_coh.items()},
}
with open(REPORTS / 'module_validation_672.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: module_validation_672.json")
print("\nAll done.")
