#!/usr/bin/env python3
"""
v10_within_gene_validate.py  [Option B]
=========================================
counts_transcript.txt N 샘플 기반 within-gene expression ratio 검증:
  - Gene당 isoform 수 분포
  - Within-gene ratio variance (label discriminativity)
  - Feature-ratio 상관 (domain/splice가 ratio와 관련 있는가)
  - Coverage 확인
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict
import os, json, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FEAT_DIR  = '../results_isoform/features'
DATA_DIR  = '../data'
COUNTS    = '../data/counts_transcript.txt'
OUT_DIR   = '../../reports/within_gene'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print(" Within-Gene Expression Ratio Validation")
print("=" * 65)

# ── 1. Counts 로드 (N 샘플만) ─────────────────────────────────────
print("\n[1] Loading N-sample counts ...")
df = pd.read_csv(COUNTS, sep='\t')
n_cols = [c for c in df.columns if c.startswith('N')]
d_cols = [c for c in df.columns if c.startswith('D')]
print(f"  N (normal) samples: {n_cols}")
print(f"  D (disease) samples: {d_cols}")

df['mean_N'] = df[n_cols].mean(axis=1)
df['gene_base'] = df['GENEID'].str.split('.').str[0]
print(f"  Total transcripts in file: {len(df)}")

# ── 2. Isoform list 로드 ──────────────────────────────────────────
print("\n[2] Matching to our isoform list ...")
iso_arr = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_arr]
gene_arr = np.load('my_gene_list_fixed.npy', allow_pickle=True)
gene_list = [s.decode() if isinstance(s, bytes) else s for s in gene_arr]
iso_to_idx = {iso: i for i, iso in enumerate(iso_list)}
iso_to_gene = {iso: gene_list[i] for i, iso in enumerate(iso_list)}

# counts에서 우리 isoform만 필터
df_ours = df[df['TXNAME'].isin(iso_to_idx)].copy()
print(f"  Matched: {len(df_ours)} / {len(iso_list)} ({len(df_ours)/len(iso_list)*100:.1f}%)")

# ── 3. Within-gene ratio 계산 ─────────────────────────────────────
print("\n[3] Computing within-gene expression ratios ...")
gene_totals = df_ours.groupby('gene_base')['mean_N'].sum()
df_ours['gene_total'] = df_ours['gene_base'].map(gene_totals)
df_ours['ratio'] = df_ours['mean_N'] / (df_ours['gene_total'] + 1e-8)

# ── 4. 통계 분석 ─────────────────────────────────────────────────
print("\n[4] Within-gene ratio statistics ...")

# Gene당 isoform 수
gene_iso_count = df_ours.groupby('gene_base')['TXNAME'].count()
multi_iso_genes = (gene_iso_count > 1).sum()
print(f"  Total genes (in our list): {len(gene_iso_count)}")
print(f"  Genes with >1 isoform: {multi_iso_genes} ({multi_iso_genes/len(gene_iso_count)*100:.1f}%)")
print(f"  Median isoforms per gene: {gene_iso_count.median():.0f}")
print(f"  Max isoforms per gene: {gene_iso_count.max()}")

# Within-gene ratio variance
gene_ratio_var = df_ours.groupby('gene_base')['ratio'].std()
genes_with_variance = (gene_ratio_var > 0.05).sum()
print(f"\n  Genes with ratio std > 0.05 (discriminable): {genes_with_variance} ({genes_with_variance/len(gene_ratio_var)*100:.1f}%)")
print(f"  Median within-gene ratio std: {gene_ratio_var.median():.3f}")
print(f"  Ratio distribution (among multi-iso genes):")
multi_ratios = df_ours[df_ours['gene_base'].isin(gene_iso_count[gene_iso_count>1].index)]['ratio']
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"    {int(q*100)}th pct: {multi_ratios.quantile(q):.3f}")

# ── 5. Feature 로드 & ratio 상관 계산 ────────────────────────────
print("\n[5] Feature-ratio Spearman correlation ...")
# isoform index 매핑
df_ours['iso_idx'] = df_ours['TXNAME'].map(iso_to_idx)
valid = df_ours.dropna(subset=['iso_idx'])
idxs = valid['iso_idx'].astype(int).values
ratios = valid['ratio'].values

# ESM-2
esm2 = np.load(f'{DATA_DIR}/esm2_embeddings_t30_150M.npy').astype(np.float32)[idxs]
esm2_norm = np.linalg.norm(esm2, axis=1)
r_esm, _ = spearmanr(esm2_norm, ratios)
print(f"  Spearman(|ESM2|, ratio) = {r_esm:.4f}")

# domain delta
dd = np.load(f'{FEAT_DIR}/domain_delta_proper_test_v2.npy').astype(np.float32)[idxs]
dd_norm = np.abs(dd).sum(axis=1)
r_dd, _ = spearmanr(dd_norm, ratios)
print(f"  Spearman(|domain_delta|, ratio) = {r_dd:.4f}")

# splice delta
sd = np.load(f'{FEAT_DIR}/splicing/splicing_delta_v2.npy').astype(np.float32)[idxs]
sd_norm = np.abs(sd).sum(axis=1)
r_sd, _ = spearmanr(sd_norm, ratios)
print(f"  Spearman(|splice_delta|, ratio) = {r_sd:.4f}")

# ── 6. 기준 isoform (ratio 최고 = canonical과 일치?) ─────────────
print("\n[6] Do expression-dominant isoforms match MANE canonical?")
canon_ref = {}
with open(f'{FEAT_DIR}/canonical_reference.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4:
            try:
                canon_ref[p[0]] = int(p[3])
            except: pass

idx_to_iso = {i: iso for iso, i in iso_to_idx.items()}
gene_base_list = [g.split('.')[0] for g in gene_list]

# 각 gene에서 ratio 최고 isoform이 canonical인지 확인
n_match = n_total = 0
for gene_base, grp in df_ours.groupby('gene_base'):
    if len(grp) < 2: continue
    top_iso = grp.loc[grp['ratio'].idxmax(), 'TXNAME']
    top_idx = iso_to_idx.get(top_iso)
    if top_idx is None: continue
    gb = gene_base_list[top_idx]
    canon_idx = canon_ref.get(gb)
    if canon_idx is not None:
        n_total += 1
        if top_idx == canon_idx:
            n_match += 1

if n_total > 0:
    print(f"  Dominant isoform == MANE canonical: {n_match}/{n_total} ({n_match/n_total*100:.1f}%)")
    print(f"  → {'High' if n_match/n_total > 0.6 else 'Low'} concordance with MANE")

# ── 7. BambuTx 발현 상태 ─────────────────────────────────────────
print("\n[7] BambuTx isoform expression status ...")
bambu_in_counts = df_ours[df_ours['TXNAME'].str.contains('BambuTx', na=False)]
bambu_expressed = (bambu_in_counts['mean_N'] > 0).sum()
print(f"  BambuTx in counts & our list: {len(bambu_in_counts)}")
print(f"  BambuTx with N-sample expression > 0: {bambu_expressed} ({bambu_expressed/max(len(bambu_in_counts),1)*100:.1f}%)")

# ── 저장 ──────────────────────────────────────────────────────────
# ratio 배열 저장 (isoform index 기준)
ratio_arr = np.full(len(iso_list), np.nan, dtype=np.float32)
for _, row in valid.iterrows():
    ratio_arr[int(row['iso_idx'])] = row['ratio']

np.save(f'{FEAT_DIR}/within_gene_ratio.npy', ratio_arr)

results = {
    'n_matched_isoforms': int(len(df_ours)),
    'n_genes': int(len(gene_iso_count)),
    'n_multi_iso_genes': int(multi_iso_genes),
    'median_iso_per_gene': float(gene_iso_count.median()),
    'genes_with_variance': int(genes_with_variance),
    'median_ratio_std': float(gene_ratio_var.median()),
    'spearman_esm2_ratio': float(r_esm),
    'spearman_dd_ratio': float(r_dd),
    'spearman_sd_ratio': float(r_sd),
    'mane_concordance': float(n_match/n_total) if n_total > 0 else None,
    'bambu_expressed': int(bambu_expressed),
    'timestamp': time.strftime('%Y%m%d_%H%M'),
}

with open(f'{OUT_DIR}/validation_{time.strftime("%Y%m%d_%H%M")}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Saved] within_gene_ratio.npy & validation JSON")
print("\n" + "=" * 65)
print(" VALIDATION SUMMARY")
print("=" * 65)
print(f"  Within-gene label coverage: {len(df_ours)}/{len(iso_list)} ({len(df_ours)/len(iso_list)*100:.1f}%)")
print(f"  Discriminable genes (std>0.05): {genes_with_variance}")
print(f"  MANE concordance: {n_match/n_total*100:.1f}%" if n_total > 0 else "  MANE concordance: N/A")
print(f"  Spearman(domain_delta, ratio): {r_dd:.4f}")
print(f"  Spearman(splice_delta, ratio): {r_sd:.4f}")
feas = "FEASIBLE" if genes_with_variance > 1000 and (abs(r_dd) > 0.05 or abs(r_sd) > 0.05) else "MARGINAL"
print(f"\n  Feasibility: {feas}")
print("Done.")
