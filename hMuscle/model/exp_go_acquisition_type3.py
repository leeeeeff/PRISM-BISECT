#!/usr/bin/env python3
"""
exp_go_acquisition_type3.py
============================
[B] GO Acquisition Case Listing
    - 동일 유전자 내에서 gene-mean 대비 특정 isoform이 높은 GO 점수를 받은 케이스
    - 원래 gene에 없는 GO term을 PRISM이 높게 예측한 novel GO 케이스
    - Feature type별 GO acquisition 패턴

[A] Type3 Motif-Level Resolution Analysis
    - Same-domain pair 중 |ΔPRISMmean| > 0.05인 2.8% 케이스의 sequence 특성
    - ESM-2 embedding 거리, sequence 길이 차이, PCA 분해
    - 어떤 GO term, 어떤 Pfam domain이 motif-level 감지와 연관되는가

Outputs: reports/go_acquisition_type3/
"""
import os, sys, csv, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
FEAT    = os.path.join(ROOT, 'hMuscle/results_isoform/features')
BOOT    = os.path.join(ROOT, 'reports/v17f_star_bootstrap')
OUT     = os.path.join(ROOT, 'reports/go_acquisition_type3')
os.makedirs(OUT, exist_ok=True)

def clean(x):
    s = str(x)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

# ─── 0. 데이터 로드 ────────────────────────────────────────────────────────
print("[0] Loading data...", flush=True)
iso_raw  = np.load('my_isoform_list_fixed.npy', allow_pickle=True)
gene_raw = np.load('my_gene_list_fixed.npy', allow_pickle=True)
iso_ids  = [clean(x) for x in iso_raw]
gene_ids = [clean(x).split('.')[0] for x in gene_raw]

preds = np.load(os.path.join(BOOT, 'v17f_star_preds.npy'))   # (36748, 82)
Y     = np.load(os.path.join(BOOT, 'Y_te.npy'))               # (36748, 82)
dm    = np.load(os.path.join(FEAT, 'domain_matrix_proper_test_v3.npy'))  # (36748, 512)
emb   = np.load(os.path.join(ROOT, 'reports/exp_d_finetune/ft_D1_test_l30.npy'))  # (36748, 640)

n_iso, n_go = preds.shape
print(f"  isoforms={n_iso}, GO terms={n_go}")

# GO term names
go_names = []
with open(os.path.join(ROOT, 'reports/v_expanded_gomf/mf_domain_vs_prism.tsv')) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2:
            go_names.append((p[0], p[1]))
go_names = go_names[:n_go]
print(f"  GO terms loaded: {len(go_names)}")

# Pfam vocab
pfam_vocab = []
with open(os.path.join(FEAT, 'domain_pfam_vocab_v3.txt')) as f:
    pfam_vocab = [l.strip() for l in f]
print(f"  Pfam vocab: {len(pfam_vocab)}")

# Feature type per isoform
ft_arr = []
with open(os.path.join(ROOT, 'reports/isoform_resolution_full/full_isoform_feature_types.tsv')) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
ft_arr = np.array(ft_arr[:n_iso])

# Protein sequence lengths from FASTA
print("  Loading FASTA lengths...", flush=True)
pep_lengths = defaultdict(int)
pep_seqs = {}
current_id, current_seq = None, []
with open(os.path.join(ROOT, 'hMuscle/data/transcripts.fasta.transdecoder.pep')) as f:
    for line in f:
        if line.startswith('>'):
            if current_id and current_seq:
                seq = ''.join(current_seq)
                base = current_id.split('.p')[0]
                if len(seq) > pep_lengths[base]:
                    pep_lengths[base] = len(seq)
                    pep_seqs[base] = seq
            current_id = line.strip()[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip().rstrip('*'))
    if current_id and current_seq:
        seq = ''.join(current_seq)
        base = current_id.split('.p')[0]
        if len(seq) > pep_lengths[base]:
            pep_lengths[base] = len(seq)
            pep_seqs[base] = seq

iso_lengths = np.array([pep_lengths.get(iso_ids[i], 0) for i in range(n_iso)])
print(f"  Lengths coverage: {(iso_lengths > 0).mean():.1%}")

# Gene → isoform index map
gene2iso = defaultdict(list)
for i, g in enumerate(gene_ids):
    gene2iso[g].append(i)
multi_genes = {g: np.array(idxs) for g, idxs in gene2iso.items() if len(idxs) >= 2}
print(f"  Multi-isoform genes: {len(multi_genes)}")

domain_counts = dm.sum(1).astype(int)

# ─── [B] GO ACQUISITION ANALYSIS ──────────────────────────────────────────
print("\n[B] GO Acquisition Analysis", flush=True)

# [B1] 유전자 내 GO specialization: gene-mean 대비 특정 isoform이 높은 점수
# "GO acquisition" = pred[i,j] - gene_mean[gene,j] > DELTA_THR, Y[i,j]=1 (gene has GO)
# "Novel GO"       = pred[i,j] > NOVEL_THR, Y[i,j]=0 (gene does NOT have GO)
DELTA_THR = 0.05   # excess above gene mean threshold
NOVEL_THR = 0.40   # threshold for predicting GO gene doesn't have

acquisition_rows = []
novel_rows = []

gene_mean_preds = np.zeros((n_iso, n_go), dtype=np.float32)
for g, idxs in gene2iso.items():
    gm = preds[idxs].mean(0)  # (n_go,)
    gene_mean_preds[idxs] = gm

# per-isoform excess above gene mean
excess = preds - gene_mean_preds  # (n_iso, n_go)

print("  Computing GO acquisition cases...", flush=True)
for g, idxs in multi_genes.items():
    gm   = gene_mean_preds[idxs[0]]  # gene-mean prediction
    y_g  = Y[idxs[0]]                # gene-level GO labels (same for all isoforms in gene)
    dc_g = domain_counts[idxs]

    for idx_pos, i in enumerate(idxs):
        for j in range(n_go):
            ex = excess[i, j]
            pr = float(preds[i, j])
            lbl = int(y_g[j])

            # [B1] Gene HAS GO annotation: isoform specialization
            if lbl == 1 and ex > DELTA_THR:
                acquisition_rows.append({
                    'gene': g,
                    'isoform': iso_ids[i],
                    'isoform_idx': i,
                    'go_idx': j,
                    'go_id': go_names[j][0],
                    'go_name': go_names[j][1],
                    'pred_score': round(pr, 4),
                    'gene_mean': round(float(gm[j]), 4),
                    'excess': round(float(ex), 4),
                    'feature_type': ft_arr[i],
                    'domain_count': int(dc_g[idx_pos]),
                    'seq_length': int(iso_lengths[i]),
                    'n_gene_isoforms': len(idxs),
                    'type': 'specialization',
                })

            # [B2] Gene does NOT have GO: novel prediction
            if lbl == 0 and pr > NOVEL_THR:
                novel_rows.append({
                    'gene': g,
                    'isoform': iso_ids[i],
                    'isoform_idx': i,
                    'go_idx': j,
                    'go_id': go_names[j][0],
                    'go_name': go_names[j][1],
                    'pred_score': round(pr, 4),
                    'gene_mean': round(float(gm[j]), 4),
                    'excess': round(float(ex), 4),
                    'feature_type': ft_arr[i],
                    'domain_count': int(dc_g[idx_pos]),
                    'seq_length': int(iso_lengths[i]),
                    'n_gene_isoforms': len(idxs),
                    'type': 'novel_GO',
                })

acq_df = pd.DataFrame(acquisition_rows)
nov_df = pd.DataFrame(novel_rows)
print(f"  GO specialization cases: {len(acq_df):,} (gene HAS GO, isoform excess >{DELTA_THR})")
print(f"  Novel GO cases:          {len(nov_df):,} (gene NO GO, pred >{NOVEL_THR})")

# Feature type breakdown
print("\n  [B1] Specialization by feature type:")
if len(acq_df) > 0:
    for ft, grp in acq_df.groupby('feature_type'):
        print(f"    {ft}: {len(grp):,} cases, mean excess={grp['excess'].mean():.4f}, "
              f"mean pred={grp['pred_score'].mean():.4f}")

print("\n  [B2] Novel GO by feature type:")
if len(nov_df) > 0:
    for ft, grp in nov_df.groupby('feature_type'):
        print(f"    {ft}: {len(grp):,} cases, mean pred={grp['pred_score'].mean():.4f}")

# Top GO terms being acquired
print("\n  [B3] Top GO terms acquired (specialization):")
if len(acq_df) > 0:
    go_counts = acq_df.groupby(['go_id','go_name']).size().sort_values(ascending=False)
    for (goid, gname), cnt in go_counts.head(15).items():
        mean_exc = acq_df[acq_df['go_id']==goid]['excess'].mean()
        print(f"    {goid} ({gname[:40]}): {cnt} isoforms, mean excess={mean_exc:.4f}")

# Scale analysis: excess score magnitude by feature type
print("\n  [B4] Score scale (excess above gene mean) by feature type:")
if len(acq_df) > 0:
    for ft in ['Type0_NoDomain','Type1_DomainLoss','Type2_PartialTrunc','Type3_SameDomain']:
        sub = acq_df[acq_df['feature_type'] == ft]['excess']
        if len(sub) > 0:
            print(f"    {ft}: n={len(sub):,} mean={sub.mean():.4f} "
                  f"median={sub.median():.4f} max={sub.max():.4f} "
                  f"[{sub.quantile(.25):.3f},{sub.quantile(.75):.3f}]")

# Per-gene specialization: how many genes have ≥1 isoform with acquisition
if len(acq_df) > 0:
    n_acq_genes = acq_df['gene'].nunique()
    print(f"\n  GO specialization present in {n_acq_genes:,} genes "
          f"({n_acq_genes/len(multi_genes):.1%} of multi-isoform genes)")

# Novel GO: compare with gene GO annotation scope
if len(nov_df) > 0:
    print(f"\n  Novel GO: {nov_df['gene'].nunique():,} genes predict GO not in annotation")
    print("  Top novel GO terms:")
    novel_go_cnt = nov_df.groupby(['go_id','go_name']).size().sort_values(ascending=False)
    for (goid, gname), cnt in novel_go_cnt.head(10).items():
        mp = nov_df[nov_df['go_id']==goid]['pred_score'].mean()
        print(f"    {goid} ({gname[:40]}): {cnt} isoforms, mean pred={mp:.4f}")

# Save
acq_df.to_csv(os.path.join(OUT, 'go_specialization_cases.tsv'), sep='\t', index=False)
nov_df.to_csv(os.path.join(OUT, 'novel_go_cases.tsv'), sep='\t', index=False)
print(f"\n  Saved: go_specialization_cases.tsv, novel_go_cases.tsv")

# ─── [A] TYPE3 MOTIF-LEVEL RESOLUTION ─────────────────────────────────────
print("\n[A] Type3 Motif-Level Resolution Analysis", flush=True)

# Type3: same domain count pairs within gene
# Collect all within-gene pairs where both isoforms have same domain_count
HIGH_GAP = 0.05
LOW_GAP  = 0.01

pair_rows = []
print("  Building same-domain pairs...", flush=True)
for g, idxs in multi_genes.items():
    if len(idxs) > 20: continue  # skip extremely large gene clusters
    dc = domain_counts[idxs]
    pred_mean = preds[idxs].mean(1)  # per-isoform mean score across all GO terms

    for a_pos in range(len(idxs)):
        for b_pos in range(a_pos + 1, len(idxs)):
            i, j = idxs[a_pos], idxs[b_pos]
            # Same domain count = Type3 criterion
            if dc[a_pos] != dc[b_pos]: continue

            gap = abs(float(pred_mean[a_pos] - pred_mean[b_pos]))
            gap_per_go = np.abs(preds[i] - preds[j])  # (n_go,)

            pair_rows.append({
                'gene': g,
                'iso_a': iso_ids[i],
                'iso_b': iso_ids[j],
                'idx_a': i,
                'idx_b': j,
                'domain_count': int(dc[a_pos]),
                'gap': round(gap, 5),
                'ft_a': ft_arr[i],
                'ft_b': ft_arr[j],
                'len_a': int(iso_lengths[i]),
                'len_b': int(iso_lengths[j]),
                'top_go_idx': int(gap_per_go.argmax()),
                'top_go_gap': round(float(gap_per_go.max()), 5),
            })

pair_df = pd.DataFrame(pair_rows)
print(f"  Same-domain pairs: {len(pair_df):,}")

high_gap = pair_df[pair_df['gap'] > HIGH_GAP]
low_gap  = pair_df[pair_df['gap'] < LOW_GAP]
print(f"  High-gap (>{HIGH_GAP}): {len(high_gap):,} ({len(high_gap)/len(pair_df):.1%})")
print(f"  Low-gap  (<{LOW_GAP}): {len(low_gap):,} ({len(low_gap)/len(pair_df):.1%})")

# [A1] Sequence length difference analysis
pair_df['len_diff'] = np.abs(pair_df['len_a'] - pair_df['len_b'])
pair_df['len_ratio'] = pair_df[['len_a','len_b']].min(1) / (pair_df[['len_a','len_b']].max(1) + 1)

print("\n  [A1] Sequence length difference: high-gap vs low-gap")
for label, sub in [('high-gap', high_gap), ('low-gap', low_gap)]:
    ld = np.abs(sub['len_a'].values - sub['len_b'].values)
    lratio = sub[['len_a','len_b']].min(1) / (sub[['len_a','len_b']].max(1) + 1)
    print(f"    {label}: mean_len_diff={ld.mean():.1f} aa, "
          f"mean_len_ratio(shorter/longer)={lratio.mean():.3f}, "
          f">10aa_diff: {(ld>10).mean():.1%}")

# MWU test: len_diff high vs low
if len(high_gap) > 0 and len(low_gap) > 0:
    ld_hi = np.abs(high_gap['len_a'].values - high_gap['len_b'].values)
    ld_lo = np.abs(low_gap['len_a'].values  - low_gap['len_b'].values)
    stat, p = mannwhitneyu(ld_hi, ld_lo, alternative='greater')
    rho, rp = spearmanr(pair_df['len_diff'].values, pair_df['gap'].values)
    print(f"    MWU len_diff high>low: p={p:.4e}")
    print(f"    Spearman ρ(len_diff, gap): {rho:.4f}, p={rp:.4e}")

# [A2] ESM-2 embedding cosine distance
print("\n  [A2] ESM-2 cosine distance: high-gap vs low-gap pairs", flush=True)
N_SAMPLE = min(3000, len(high_gap), len(low_gap))
np.random.seed(42)
hi_sample = high_gap.sample(min(N_SAMPLE, len(high_gap)))
lo_sample = low_gap.sample(min(N_SAMPLE, len(low_gap)))

hi_cos = [cosine(emb[r.idx_a], emb[r.idx_b]) for _, r in hi_sample.iterrows()]
lo_cos = [cosine(emb[r.idx_a], emb[r.idx_b]) for _, r in lo_sample.iterrows()]
hi_cos = np.array(hi_cos)
lo_cos = np.array(lo_cos)
print(f"    high-gap cosine dist: mean={hi_cos.mean():.5f}, std={hi_cos.std():.5f}")
print(f"    low-gap  cosine dist: mean={lo_cos.mean():.5f}, std={lo_cos.std():.5f}")
stat, p = mannwhitneyu(hi_cos, lo_cos, alternative='greater')
rho2, rp2 = spearmanr(
    pair_df.sample(min(5000, len(pair_df)), random_state=42)['gap'].values,
    [cosine(emb[r.idx_a], emb[r.idx_b])
     for _, r in pair_df.sample(min(5000, len(pair_df)), random_state=42).iterrows()]
)
print(f"    MWU cosine high>low: p={p:.4e}, ratio={hi_cos.mean()/lo_cos.mean():.3f}×")
print(f"    Spearman ρ(cosine_dist, gap): {rho2:.4f}, p={rp2:.4e}")

# [A3] PCA: which embedding dimensions drive the gap?
print("\n  [A3] ESM-2 PCA: dimensions distinguishing high-gap vs low-gap", flush=True)
# δ_emb = emb_a - emb_b for each pair
hi_idx = hi_sample[['idx_a','idx_b']].values
lo_idx = lo_sample[['idx_a','idx_b']].values

hi_delta = emb[hi_idx[:,0]] - emb[hi_idx[:,1]]  # (N, 640)
lo_delta = emb[lo_idx[:,0]] - emb[lo_idx[:,1]]  # (N, 640)

# Pool and compute PCA on delta embeddings
all_delta = np.vstack([hi_delta, lo_delta])
labels_pca = np.array([1]*len(hi_delta) + [0]*len(lo_delta))

scaler = StandardScaler()
all_delta_z = scaler.fit_transform(all_delta)

pca = PCA(n_components=20, random_state=42)
Z = pca.fit_transform(all_delta_z)

# For each PC, test discrimination of high vs low gap
print("    PC discrimination (high-gap vs low-gap, AUROC):")
pc_aurocs = []
for k in range(20):
    try:
        auc = roc_auc_score(labels_pca, Z[:, k])
        auc = max(auc, 1-auc)  # make directional
        pc_aurocs.append((k, auc, pca.explained_variance_ratio_[k]))
    except:
        pc_aurocs.append((k, 0.5, pca.explained_variance_ratio_[k]))

pc_aurocs.sort(key=lambda x: -x[1])
for k, auc, var in pc_aurocs[:8]:
    print(f"    PC{k+1:02d}: AUROC={auc:.4f}, explained_var={var:.3f}")

# Top discriminating PC — find which raw embedding dimensions load heavily
top_pc = pc_aurocs[0][0]
loadings = pca.components_[top_pc]
top_dims = np.argsort(np.abs(loadings))[::-1][:20]
print(f"\n    Top PC{top_pc+1} loading dimensions (ESM-2 feature indices):")
for dim in top_dims[:10]:
    print(f"      dim {dim:03d}: loading={loadings[dim]:.4f}")

# [A4] GO term specificity: which GO terms drive high-gap in Type3 pairs?
print("\n  [A4] GO terms driving high-gap Type3 pairs:")
if len(high_gap) > 0:
    top_go_counts = Counter(high_gap['top_go_idx'].values)
    print("    Top GO terms (by frequency as top-discriminating term):")
    for go_idx, cnt in top_go_counts.most_common(15):
        goid, gname = go_names[go_idx]
        mean_gap_this = high_gap[high_gap['top_go_idx']==go_idx]['top_go_gap'].mean()
        print(f"    [{go_idx:02d}] {goid} ({gname[:45]}): {cnt} pairs, mean_gap={mean_gap_this:.4f}")

# [A5] Pfam domain context: which domains are present in high-gap Type3 pairs?
print("\n  [A5] Pfam domains in high-gap Type3 pairs:")
if len(high_gap) > 0 and len(low_gap) > 0:
    hi_dm = dm[high_gap['idx_a'].values] + dm[high_gap['idx_b'].values]  # union
    lo_dm = dm[low_gap['idx_a'].values]  + dm[low_gap['idx_b'].values]
    hi_dm_freq = (hi_dm > 0).mean(0)  # (512,)
    lo_dm_freq = (lo_dm > 0).mean(0)

    # Enriched domains in high-gap
    enrichment = hi_dm_freq - lo_dm_freq
    top_enrich_idx = np.argsort(enrichment)[::-1][:15]
    print("    Domains enriched in high-gap pairs (high-gap freq - low-gap freq):")
    for d_idx in top_enrich_idx:
        if d_idx < len(pfam_vocab) and hi_dm_freq[d_idx] > 0.01:
            print(f"      {pfam_vocab[d_idx]}: hi={hi_dm_freq[d_idx]:.3f} "
                  f"lo={lo_dm_freq[d_idx]:.3f} delta={enrichment[d_idx]:.3f}")

# [A6] SLiM proxy: sequence property differences in high-gap pairs
# (charge, disorder proxy via low-complexity, PTM site density)
print("\n  [A6] Sequence property analysis for high-gap Type3 pairs:", flush=True)

def seq_props(seq):
    if not seq or len(seq) == 0:
        return {'charge': 0, 'disorder': 0, 'length': 0, 'polar_frac': 0, 'hydro_frac': 0}
    seq = seq.upper().replace('*','')
    L = max(len(seq), 1)
    charge = (seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')) / L
    # Low-complexity proxy: S, T, G repeats (disorder indicator)
    disorder = (seq.count('S') + seq.count('T') + seq.count('G') + seq.count('A')) / L
    polar = (seq.count('S') + seq.count('T') + seq.count('N') + seq.count('Q')) / L
    hydro = (seq.count('L') + seq.count('I') + seq.count('V') + seq.count('F') + seq.count('W')) / L
    return {'charge': charge, 'disorder': disorder, 'length': L,
            'polar_frac': polar, 'hydro_frac': hydro}

SAMPLE_N = min(1000, len(high_gap), len(low_gap))
prop_diffs_hi, prop_diffs_lo = [], []

for _, r in high_gap.sample(SAMPLE_N, random_state=42).iterrows():
    sa = pep_seqs.get(r.iso_a, '')
    sb = pep_seqs.get(r.iso_b, '')
    pa, pb = seq_props(sa), seq_props(sb)
    prop_diffs_hi.append({k: abs(pa[k] - pb[k]) for k in pa})

for _, r in low_gap.sample(SAMPLE_N, random_state=42).iterrows():
    sa = pep_seqs.get(r.iso_a, '')
    sb = pep_seqs.get(r.iso_b, '')
    pa, pb = seq_props(sa), seq_props(sb)
    prop_diffs_lo.append({k: abs(pa[k] - pb[k]) for k in pa})

hi_props = pd.DataFrame(prop_diffs_hi)
lo_props = pd.DataFrame(prop_diffs_lo)
print("    Property differences (|prop_a - prop_b|): high-gap vs low-gap")
for col in hi_props.columns:
    hv = hi_props[col].mean()
    lv = lo_props[col].mean()
    stat, p = mannwhitneyu(hi_props[col], lo_props[col], alternative='greater')
    print(f"    Δ{col}: hi={hv:.4f} lo={lv:.4f} ratio={hv/(lv+1e-6):.2f}× MWU_p={p:.3e}")

# [A7] Case examples: top high-gap Type3 pairs with gene names
print("\n  [A7] Top high-gap Type3 case examples:")
if len(high_gap) > 0:
    top_cases = high_gap.sort_values('gap', ascending=False).head(20)
    for _, r in top_cases.iterrows():
        go_idx = int(r.top_go_idx)
        goid, gname = go_names[go_idx]
        print(f"    gene={r.gene} | {r.iso_a} vs {r.iso_b} | "
              f"gap={r.gap:.4f} | len={r.len_a}/{r.len_b}aa | "
              f"top_GO: {goid}({gname[:30]}) gap={r.top_go_gap:.4f}")

# ─── Save results ─────────────────────────────────────────────────────────
pair_df.to_csv(os.path.join(OUT, 'type3_pair_analysis.tsv'), sep='\t', index=False)
high_gap.to_csv(os.path.join(OUT, 'type3_high_gap_pairs.tsv'), sep='\t', index=False)

# Summary JSON
import json
summary = {
    # [B] GO acquisition
    'go_specialization': {
        'total_cases': len(acq_df),
        'n_genes': int(acq_df['gene'].nunique()) if len(acq_df) > 0 else 0,
        'by_feature_type': {
            ft: {
                'n': int(len(grp)),
                'mean_excess': round(float(grp['excess'].mean()), 5),
                'mean_pred': round(float(grp['pred_score'].mean()), 5),
            }
            for ft, grp in acq_df.groupby('feature_type')
        } if len(acq_df) > 0 else {},
        'top_go_terms': [
            {'go_id': goid, 'go_name': gname[:60], 'n_isoforms': int(cnt)}
            for (goid, gname), cnt in (acq_df.groupby(['go_id','go_name']).size()
                                        .sort_values(ascending=False).head(10).items())
        ] if len(acq_df) > 0 else [],
    },
    'novel_go': {
        'total_cases': len(nov_df),
        'n_genes': int(nov_df['gene'].nunique()) if len(nov_df) > 0 else 0,
    },
    # [A] Type3
    'type3_pairs': {
        'total_same_domain_pairs': len(pair_df),
        'high_gap_n': len(high_gap),
        'high_gap_frac': round(len(high_gap)/max(len(pair_df),1), 4),
        'low_gap_n': len(low_gap),
        'esm2_cosine': {
            'high_gap_mean': round(float(hi_cos.mean()), 6),
            'low_gap_mean': round(float(lo_cos.mean()), 6),
            'ratio': round(float(hi_cos.mean()/lo_cos.mean()), 4),
            'mwu_p': float(p),
        },
        'len_diff_spearman_rho': round(float(rho), 4),
        'len_diff_spearman_p': float(rp),
        'top_discriminating_pc': {
            'pc_num': top_pc + 1,
            'auroc': round(float(pc_aurocs[0][1]), 4),
        },
        'top_pc_aurocs': [(k+1, round(a,4)) for k,a,_ in pc_aurocs[:5]],
    }
}

with open(os.path.join(OUT, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[Done] All outputs saved to {OUT}")
print("  go_specialization_cases.tsv")
print("  novel_go_cases.tsv")
print("  type3_pair_analysis.tsv")
print("  type3_high_gap_pairs.tsv")
print("  analysis_summary.json")
