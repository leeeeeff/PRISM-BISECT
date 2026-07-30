#!/usr/bin/env python3
"""
exp_true_motif_level.py
========================
[A] True Motif-Level Resolution Analysis

Type3 same-domain pairs에서 length diff < 50aa인 "진짜 같은 서열" 쌍만 추려
실제 motif-level PRISM sensitivity를 정량화.

Questions:
  A1. len_diff < 50aa 제한 후 gap 분포 — 진짜 motif level gap이 존재하는가?
  A2. gap > 0.05인 353개 케이스에서 어떤 서열 특성이 다른가?
  A3. 알려진 SLiM (phospho-site, NLS, NES, PIP box 등) regex 탐색
  A4. ESM-2 cosine distance: len-filtered에서도 gap과 상관하는가?
  A5. 어떤 GO term, 어떤 gene에서 진짜 motif-level 감지가 일어나는가?

Outputs: reports/true_motif_level/
"""
import os, re, csv
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score

ROOT = '/home/welcome1/sw1686/DIFFUSE'
OUT  = os.path.join(ROOT, 'reports/true_motif_level')
os.makedirs(OUT, exist_ok=True)

# ─── 0. 데이터 로드 ────────────────────────────────────────────────────────
print("[0] Loading data...", flush=True)

pair_df = pd.read_csv(os.path.join(ROOT, 'reports/go_acquisition_type3/type3_pair_analysis.tsv'), sep='\t')
pair_df['len_diff'] = (pair_df['len_a'] - pair_df['len_b']).abs()

emb   = np.load(os.path.join(ROOT, 'reports/exp_d_finetune/ft_D1_test_l30.npy'))
preds = np.load(os.path.join(ROOT, 'reports/v17f_star_bootstrap/v17f_star_preds.npy'))

go_names = []
with open(os.path.join(ROOT, 'reports/v_expanded_gomf/mf_domain_vs_prism.tsv')) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            go_names.append((p[0], p[1]))

# Protein sequences
pep_seqs = {}
current_id, current_seq = None, []
with open(os.path.join(ROOT, 'hMuscle/data/transcripts.fasta.transdecoder.pep')) as f:
    for line in f:
        if line.startswith('>'):
            if current_id and current_seq:
                seq = ''.join(current_seq)
                base = current_id.split('.p')[0]
                if len(seq) > len(pep_seqs.get(base, '')):
                    pep_seqs[base] = seq
            current_id = line.strip()[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip().rstrip('*'))
    if current_id and current_seq:
        seq = ''.join(current_seq)
        base = current_id.split('.p')[0]
        if len(seq) > len(pep_seqs.get(base, '')):
            pep_seqs[base] = seq

print(f"  Pairs loaded: {len(pair_df):,}")
print(f"  ESM-2 emb: {emb.shape}")

# ─── A1. len_diff < 50aa 필터 후 gap 분포 ─────────────────────────────────
print("\n[A1] True same-length Type3 pairs (len_diff < 50aa)")

SHORT_THR = 50
true_same = pair_df[pair_df['len_diff'] < SHORT_THR].copy()
print(f"  True same-length pairs: {len(true_same):,} ({len(true_same)/len(pair_df):.1%} of all same-domain)")

for thr, label in [(0.05, 'high-gap'), (0.02, 'moderate'), (0.01, 'low-gap')]:
    sub = true_same[true_same['gap'] > thr]
    print(f"  gap > {thr}: {len(sub):,} ({len(sub)/len(true_same):.2%})")

# Gap distribution statistics
print(f"\n  Gap distribution (len_diff<50):")
print(f"    mean={true_same['gap'].mean():.5f}")
print(f"    median={true_same['gap'].median():.5f}")
print(f"    p95={true_same['gap'].quantile(0.95):.5f}")
print(f"    p99={true_same['gap'].quantile(0.99):.5f}")
print(f"    max={true_same['gap'].max():.5f}")

# Compare to all Type3 pairs
print(f"\n  Comparison: full Type3 (len_diff any) vs len<50:")
print(f"    Full: mean={pair_df['gap'].mean():.5f}, >0.05: {(pair_df['gap']>0.05).mean():.2%}")
print(f"    Len<50: mean={true_same['gap'].mean():.5f}, >0.05: {(true_same['gap']>0.05).mean():.2%}")

# ─── A2. High-gap cases in true same-length group ─────────────────────────
print("\n[A2] High-gap (>0.05) cases in true same-length Type3 pairs")

hg_true = true_same[true_same['gap'] > 0.05].copy()
lg_true = true_same[true_same['gap'] < 0.005].copy()
print(f"  High-gap (>0.05): n={len(hg_true)}")
print(f"  Low-gap  (<0.005): n={len(lg_true)}")

if len(hg_true) > 0:
    print("\n  Top high-gap true-motif pairs:")
    for _, r in hg_true.sort_values('gap', ascending=False).head(20).iterrows():
        go_idx = int(r.top_go_idx)
        goid, gname = go_names[go_idx]
        print(f"    gene={r.gene} | {r.iso_a}({r.len_a}aa) vs {r.iso_b}({r.len_b}aa) | "
              f"len_diff={r.len_diff:.0f}aa | gap={r.gap:.4f} | "
              f"top_GO: {goid}({gname[:35]}) gap={r.top_go_gap:.4f}")

# ─── A3. ESM-2 cosine distance in true same-length group ──────────────────
print("\n[A3] ESM-2 cosine distance in true same-length pairs")

N = min(2000, len(hg_true), len(lg_true))
np.random.seed(42)
hg_sample = hg_true.sample(min(N, len(hg_true)), random_state=42) if len(hg_true) > 0 else hg_true
lg_sample = lg_true.sample(min(N, len(lg_true)), random_state=42)

if len(hg_sample) > 0:
    hi_cos = [cosine(emb[int(r.idx_a)], emb[int(r.idx_b)]) for _, r in hg_sample.iterrows()]
    lo_cos = [cosine(emb[int(r.idx_a)], emb[int(r.idx_b)]) for _, r in lg_sample.iterrows()]
    hi_cos = np.array(hi_cos)
    lo_cos = np.array(lo_cos)
    print(f"  High-gap cosine dist: mean={hi_cos.mean():.5f} std={hi_cos.std():.5f}")
    print(f"  Low-gap  cosine dist: mean={lo_cos.mean():.5f} std={lo_cos.std():.5f}")
    print(f"  Ratio: {hi_cos.mean()/max(lo_cos.mean(),1e-6):.2f}×")
    if len(hi_cos) > 1 and len(lo_cos) > 1:
        stat, p = mannwhitneyu(hi_cos, lo_cos, alternative='greater')
        print(f"  MWU p={p:.4e}")

    # Within true same-length: spearman gap ~ cosine
    all_sample = true_same.sample(min(3000, len(true_same)), random_state=42)
    all_cos = [cosine(emb[int(r.idx_a)], emb[int(r.idx_b)]) for _, r in all_sample.iterrows()]
    rho, rp = spearmanr(all_sample['gap'].values, all_cos)
    print(f"  Spearman ρ(cosine, gap) within len<50: ρ={rho:.4f}, p={rp:.4e}")

# ─── A4. SLiM / Regulatory Motif Detection ────────────────────────────────
print("\n[A4] SLiM/Regulatory Motif Scanning in high-gap true-motif pairs")

# Known regulatory SLiMs (regex)
SLIMS = {
    'NLS_basic_cluster': r'[KR]{3,}|[KR].{1,2}[KR]{2,}',  # Classical NLS
    'NES_leucine_rich':  r'L.{2,3}[LIVMF].{2,3}L.{2,3}L',  # Nuclear Export Signal
    'PXXP_SH3_binding':  r'P.{2}P',                          # SH3 domain ligand
    'RGD_integrin':      r'RGD',                              # Integrin binding
    'KFERQ_autophagy':   r'[KQRE].{1,3}[KQRE].*F',          # CMA targeting
    'phospho_CK2':       r'[ST].{2}[ED]',                    # CK2 phospho-site
    'phospho_PKA':       r'[RK].{2}[ST]',                    # PKA site
    'phospho_CDK':       r'[ST]P[KR]',                       # CDK minimal
    'CAAX_prenyl':       r'C[AC].{1}[LIVMF]$',              # Prenylation C-term
    'DEG_box':           r'[RK]XXL',                          # Degron box
}

slim_diff_results = []
if len(hg_true) > 0:
    for _, r in hg_true.iterrows():
        seq_a = pep_seqs.get(r.iso_a, '').upper()
        seq_b = pep_seqs.get(r.iso_b, '').upper()
        if not seq_a or not seq_b: continue

        for slim_name, pattern in SLIMS.items():
            try:
                hits_a = len(re.findall(pattern, seq_a))
                hits_b = len(re.findall(pattern, seq_b))
                if hits_a != hits_b:
                    slim_diff_results.append({
                        'gene': r.gene,
                        'iso_a': r.iso_a, 'iso_b': r.iso_b,
                        'gap': r.gap,
                        'slim': slim_name,
                        'hits_a': hits_a,
                        'hits_b': hits_b,
                        'diff': hits_a - hits_b,
                        'top_go': go_names[int(r.top_go_idx)][0],
                        'top_go_name': go_names[int(r.top_go_idx)][1][:40],
                    })
            except re.error:
                continue

slim_df = pd.DataFrame(slim_diff_results)
if len(slim_df) > 0:
    print(f"  Pairs with ≥1 SLiM difference: {slim_df['gene'].nunique()} genes, {len(slim_df)} SLiM-pair combinations")
    print("\n  SLiM types differing between high-gap true-motif isoforms:")
    for slim, grp in slim_df.groupby('slim'):
        n_pairs = grp['gene'].nunique()
        mean_diff = grp['diff'].abs().mean()
        print(f"    {slim}: {n_pairs} gene-pairs, mean |diff|={mean_diff:.2f}")

    # Top cases with SLiM differences
    print("\n  Top cases (gap + SLiM difference):")
    shown = set()
    for _, r in slim_df.sort_values('gap', ascending=False).iterrows():
        key = (r.gene, r.iso_a, r.iso_b)
        if key not in shown:
            print(f"    gene={r.gene} gap={r.gap:.4f} SLiM={r.slim} "
                  f"(iso_a:{r.hits_a} vs iso_b:{r.hits_b}) "
                  f"GO={r.top_go_name}")
            shown.add(key)
            if len(shown) >= 15: break

    slim_df.to_csv(os.path.join(OUT, 'slim_differences_true_motif.tsv'), sep='\t', index=False)
else:
    print("  No SLiM differences found in high-gap true-motif pairs")

# ─── A5. GO term specificity in true same-length high-gap pairs ───────────
print("\n[A5] GO term breakdown in true same-length high-gap pairs")
if len(hg_true) > 0:
    go_cnts = Counter(hg_true['top_go_idx'].values)
    print("  Top GO terms (true motif-level high-gap):")
    for go_idx, cnt in go_cnts.most_common(15):
        goid, gname = go_names[go_idx]
        mean_g = hg_true[hg_true['top_go_idx']==go_idx]['top_go_gap'].mean()
        print(f"    [{go_idx:02d}] {goid} ({gname[:45]}): {cnt} pairs, mean_gap={mean_g:.4f}")

    # Amino acid composition difference in high-gap pairs
    print("\n  Amino acid composition difference (per position comparison):")
    aa_diff_counts = Counter()
    for _, r in hg_true.sample(min(100, len(hg_true)), random_state=42).iterrows():
        sa = pep_seqs.get(r.iso_a, '').upper()
        sb = pep_seqs.get(r.iso_b, '').upper()
        if not sa or not sb: continue
        min_len = min(len(sa), len(sb))
        # C-term alignment (where splicing differences typically end)
        sa_trim = sa[-min_len:]
        sb_trim = sb[-min_len:]
        for a_aa, b_aa in zip(sa_trim, sb_trim):
            if a_aa != b_aa and a_aa.isalpha() and b_aa.isalpha():
                aa_diff_counts[f'{a_aa}→{b_aa}'] += 1

    print("  Most frequent substitutions (C-terminal alignment, high-gap pairs):")
    for sub, cnt in aa_diff_counts.most_common(10):
        print(f"    {sub}: {cnt}")

# ─── A6. Gene name resolution + GO term 조합 ──────────────────────────────
print("\n[A6] True motif-level high-gap case summary (with gene context)")
if len(hg_true) > 0:
    # Group by gene
    gene_summary = []
    for gene, grp in hg_true.groupby('gene'):
        max_gap = grp['gap'].max()
        top_go_idx = int(grp.sort_values('gap', ascending=False).iloc[0]['top_go_idx'])
        goid, gname = go_names[top_go_idx]
        len_diffs = (grp['len_a'] - grp['len_b']).abs()
        gene_summary.append({
            'gene': gene,
            'n_high_gap_pairs': len(grp),
            'max_gap': round(max_gap, 4),
            'mean_gap': round(grp['gap'].mean(), 4),
            'top_GO': goid,
            'top_GO_name': gname[:50],
            'mean_len_diff': round(len_diffs.mean(), 1),
        })
    gene_sum_df = pd.DataFrame(gene_summary).sort_values('max_gap', ascending=False)
    print(f"  {len(gene_sum_df)} genes with true motif-level high-gap pairs")
    print("  Top genes:")
    for _, r in gene_sum_df.head(20).iterrows():
        print(f"    {r.gene}: max_gap={r.max_gap:.4f}, n_pairs={r.n_high_gap_pairs}, "
              f"len_diff={r.mean_len_diff:.0f}aa, top_GO={r.top_GO_name}")
    gene_sum_df.to_csv(os.path.join(OUT, 'true_motif_gene_summary.tsv'), sep='\t', index=False)

# ─── Summary ──────────────────────────────────────────────────────────────
import json
summary = {
    'true_same_length': {
        'n_pairs': len(true_same),
        'frac_of_all_type3': round(len(true_same)/len(pair_df), 4),
        'mean_gap': round(float(true_same['gap'].mean()), 6),
        'median_gap': round(float(true_same['gap'].median()), 6),
        'p95_gap': round(float(true_same['gap'].quantile(0.95)), 6),
        'high_gap_n': len(hg_true),
        'high_gap_frac': round(len(hg_true)/len(true_same), 4),
    },
    'esm2_cosine_within_len50': {
        'high_gap_mean': round(float(hi_cos.mean()), 6) if len(hg_sample) > 0 else None,
        'low_gap_mean': round(float(lo_cos.mean()), 6) if len(hg_sample) > 0 else None,
        'spearman_rho': round(float(rho), 4),
        'spearman_p': float(rp),
    },
    'slim_differences': {
        'n_genes_with_slim_diff': int(slim_df['gene'].nunique()) if len(slim_df) > 0 else 0,
        'by_slim': {
            slim: {
                'n_pairs': int(len(grp)),
                'mean_abs_diff': round(float(grp['diff'].abs().mean()), 2)
            }
            for slim, grp in slim_df.groupby('slim')
        } if len(slim_df) > 0 else {}
    }
}
with open(os.path.join(OUT, 'true_motif_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[Done] Outputs: {OUT}")
print("  true_motif_gene_summary.tsv")
print("  slim_differences_true_motif.tsv")
print("  true_motif_summary.json")
