#!/usr/bin/env python3
"""
exp_novel_go_validation.py
==========================
[C] Novel GO Case Biological Validation

PRISM이 높게 예측하지만 gene-level annotation에 없는 GO term 케이스들을:
  1. 신뢰도 필터링 (pred > 0.5, gene-mean 대비 excess)
  2. UniProt 기존 isoform annotation과 대조
  3. BISECT 케이스 유전자와 교차 (이미 검증된 스위칭 유전자)
  4. GO term 계통 분류 (기능 범주별)
  5. Feature type별 패턴 (어떤 isoform 유형이 novel GO 획득)
  6. 상위 케이스 목록 (실험 검증 우선순위)

Outputs: reports/novel_go_validation/
"""
import os, json, csv
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import mannwhitneyu

ROOT = '/home/welcome1/sw1686/DIFFUSE'
OUT  = os.path.join(ROOT, 'reports/novel_go_validation')
os.makedirs(OUT, exist_ok=True)

# ─── 0. 데이터 로드 ────────────────────────────────────────────────────────
print("[0] Loading data...", flush=True)

nov = pd.read_csv(os.path.join(ROOT, 'reports/go_acquisition_type3/novel_go_cases.tsv'), sep='\t')
acq = pd.read_csv(os.path.join(ROOT, 'reports/go_acquisition_type3/go_specialization_cases.tsv'), sep='\t')

# GO term names
go_names = []
go_id_to_name = {}
with open(os.path.join(ROOT, 'reports/v_expanded_gomf/mf_domain_vs_prism.tsv')) as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 3:
            goid, gname = p[0], p[1]
            go_names.append((goid, gname))
            go_id_to_name[goid] = gname

# PRISM predictions + labels
preds = np.load(os.path.join(ROOT, 'reports/v17f_star_bootstrap/v17f_star_preds.npy'))
Y     = np.load(os.path.join(ROOT, 'reports/v17f_star_bootstrap/Y_te.npy'))

iso_raw  = np.load(os.path.join(ROOT, 'hMuscle/model/my_isoform_list_fixed.npy'), allow_pickle=True)
gene_raw = np.load(os.path.join(ROOT, 'hMuscle/model/my_gene_list_fixed.npy'), allow_pickle=True)
def clean(x):
    s = str(x)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s
iso_ids  = [clean(x) for x in iso_raw]
gene_ids = [clean(x).split('.')[0] for x in gene_raw]

gene2iso = defaultdict(list)
for i, g in enumerate(gene_ids):
    gene2iso[g].append(i)

ft_arr = []
with open(os.path.join(ROOT, 'reports/isoform_resolution_full/full_isoform_feature_types.tsv')) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
ft_arr = np.array(ft_arr)

print(f"  Novel GO cases: {len(nov):,}")

# ─── C1. UniProt 이미 검증된 유전자 목록과 교차 ───────────────────────────
print("\n[C1] Cross-reference with UniProt eval genes", flush=True)

# UniProt eval pairwise v2
uniprot_df = pd.read_csv(
    os.path.join(ROOT, 'reports/exp_h_uniprot_eval/pairwise_eval_v2.tsv'), sep='\t')
print(f"  UniProt eval pairs: {len(uniprot_df)}")
print(f"  Columns: {list(uniprot_df.columns)}")

# UniProt에 등록된 유전자들을 ENSG ID로 매핑 시도
# gene_id 컬럼이 있으면 직접 매칭, 없으면 isoform ID로 역추적
if 'gene' in uniprot_df.columns:
    uniprot_genes = set(uniprot_df['gene'].dropna().unique())
elif 'gene_id' in uniprot_df.columns:
    uniprot_genes = set(uniprot_df['gene_id'].dropna().unique())
else:
    # isoform transcript ID에서 역추적
    uniprot_isos = set()
    for col in uniprot_df.columns:
        if 'iso' in col.lower() or 'transcript' in col.lower():
            uniprot_isos.update(uniprot_df[col].dropna().unique())
    iso2gene = {iso_ids[i]: gene_ids[i] for i in range(len(iso_ids))}
    uniprot_genes = {iso2gene.get(iso, '') for iso in uniprot_isos}
    uniprot_genes.discard('')
print(f"  UniProt eval genes: {len(uniprot_genes)}")
print(f"  Sample: {list(uniprot_genes)[:5]}")

# Novel GO 케이스 중 UniProt 유전자와 교차
nov_in_uniprot = nov[nov['gene'].isin(uniprot_genes)]
print(f"  Novel GO cases in UniProt eval genes: {len(nov_in_uniprot)}")

# ─── C2. BISECT 케이스 유전자와 교차 ─────────────────────────────────────
print("\n[C2] Cross-reference with BISECT confirmed cases", flush=True)

bisect_df = None
for path in [
    os.path.join(ROOT, 'reports/supplementary_table_S_bisect_83cases.tsv'),
    os.path.join(ROOT, 'Final_analysis/pipeline_bioanalysis/outputs/bisect_results_all.tsv'),
]:
    if os.path.exists(path):
        bisect_df = pd.read_csv(path, sep='\t')
        print(f"  BISECT file: {path}, shape={bisect_df.shape}")
        print(f"  BISECT columns: {list(bisect_df.columns)[:8]}")
        break

if bisect_df is not None:
    # gene_id 또는 gene 컬럼에서 ENSG ID 추출
    bisect_gene_col = None
    for col in ['gene', 'gene_id', 'gene_name', 'ensg']:
        if col in bisect_df.columns:
            bisect_gene_col = col
            break
    if bisect_gene_col:
        bisect_genes = set(bisect_df[bisect_gene_col].dropna().astype(str).str.split('.').str[0].unique())
        print(f"  BISECT confirmed genes: {len(bisect_genes)}")
        nov_in_bisect = nov[nov['gene'].isin(bisect_genes)]
        acq_in_bisect = acq[acq['gene'].isin(bisect_genes)]
        print(f"  Novel GO in BISECT genes: {len(nov_in_bisect)}")
        print(f"  GO specialization in BISECT genes: {len(acq_in_bisect)}")

        if len(nov_in_bisect) > 0:
            print("\n  Novel GO cases in BISECT genes (high confidence):")
            hc_bisect = nov_in_bisect[nov_in_bisect['pred_score'] > 0.5].sort_values('pred_score', ascending=False)
            for _, r in hc_bisect.head(20).iterrows():
                print(f"    gene={r.gene} iso={r.isoform} GO={r.go_id} "
                      f"({r.go_name[:35]}) pred={r.pred_score:.4f} "
                      f"ft={r.feature_type}")

# ─── C3. 신뢰도 필터링: 고신뢰 novel GO 케이스 ────────────────────────────
print("\n[C3] High-confidence Novel GO filtering", flush=True)

# pred > 0.5 AND excess (above gene mean) > 0.1
hc_nov = nov[nov['pred_score'] > 0.5].copy()
print(f"  pred > 0.5: {len(hc_nov):,} cases, {hc_nov['gene'].nunique():,} genes")
hc_nov2 = nov[(nov['pred_score'] > 0.5) & (nov['excess'] > 0.1)].copy()
print(f"  pred > 0.5 AND excess > 0.1: {len(hc_nov2):,} cases, {hc_nov2['gene'].nunique():,} genes")
hc_nov3 = nov[nov['pred_score'] > 0.6].copy()
print(f"  pred > 0.6: {len(hc_nov3):,} cases, {hc_nov3['gene'].nunique():,} genes")

# ─── C4. Feature type 별 novel GO 패턴 ────────────────────────────────────
print("\n[C4] Feature type analysis of novel GO", flush=True)

for pred_thr, label in [(0.5, 'pred>0.5'), (0.6, 'pred>0.6')]:
    sub = nov[nov['pred_score'] > pred_thr]
    print(f"\n  {label} ({len(sub)} cases):")
    for ft, grp in sub.groupby('feature_type'):
        print(f"    {ft}: {len(grp):,} cases "
              f"mean_pred={grp['pred_score'].mean():.4f} "
              f"mean_excess={grp['excess'].mean():.4f}")

# ─── C5. GO term 계통 분류 — 어떤 기능이 novel로 예측되는가 ──────────────
print("\n[C5] GO term functional categories in novel predictions", flush=True)

# GO term functional annotation (simplified)
GO_CATEGORIES = {
    'binding': ['GO:0005515', 'GO:0042802', 'GO:0003677', 'GO:0003723', 'GO:0042803',
                'GO:0003676', 'GO:0019901', 'GO:0003682', 'GO:0008017', 'GO:0005080'],
    'kinase': ['GO:0004672', 'GO:0004674', 'GO:0106310', 'GO:0004713'],
    'transcription': ['GO:0000978', 'GO:0000977', 'GO:0000981', 'GO:0000976', 'GO:0003700',
                      'GO:0140297', 'GO:0001228', 'GO:0001227'],
    'catalytic': ['GO:0003824', 'GO:0016491', 'GO:0016301', 'GO:0016787',
                  'GO:0003924', 'GO:0000287'],
    'structural': ['GO:0005198', 'GO:0008307'],
    'zinc': ['GO:0008270', 'GO:0046914'],
    'ATP_nucleotide': ['GO:0005524', 'GO:0005525', 'GO:0051377'],
    'RNA': ['GO:0003723', 'GO:0003729', 'GO:0000049'],
}

# Reverse map
go_to_cat = {}
for cat, terms in GO_CATEGORIES.items():
    for t in terms:
        go_to_cat[t] = cat

hc_nov3['go_cat'] = hc_nov3['go_id'].map(go_to_cat).fillna('other')
print(f"  GO category distribution (pred > 0.6):")
cat_cnt = hc_nov3.groupby('go_cat').agg(
    n=('gene', 'count'),
    n_genes=('gene', 'nunique'),
    mean_pred=('pred_score', 'mean')
).sort_values('n', ascending=False)
print(cat_cnt.to_string())

# ─── C6. GO term-specific novelty rate ────────────────────────────────────
print("\n[C6] Per-GO term: how often is it predicted novel (gene doesn't have it)", flush=True)

go_novel_stats = []
for j in range(preds.shape[1]):
    goid, gname = go_names[j]
    # All isoforms where Y[:,j] = 0 (gene doesn't have GO)
    no_annot = Y[:, j] == 0
    high_pred = preds[:, j] > 0.5
    n_novel = int((no_annot & high_pred).sum())
    n_no_annot = int(no_annot.sum())
    rate = n_novel / max(n_no_annot, 1)
    go_novel_stats.append({
        'go_idx': j,
        'go_id': goid,
        'go_name': gname[:60],
        'n_with_annotation': int(Y[:, j].sum()),
        'n_without_annotation': n_no_annot,
        'n_novel_high_pred': n_novel,
        'novel_rate': round(rate, 5),
    })
go_novel_df = pd.DataFrame(go_novel_stats).sort_values('novel_rate', ascending=False)

print("  Top GO terms by novel prediction rate (pred>0.5 in unannotated isoforms):")
for _, r in go_novel_df.head(15).iterrows():
    print(f"    [{r.go_idx:02d}] {r.go_id} ({r.go_name[:40]}): "
          f"{r.n_novel_high_pred}/{r.n_without_annotation} "
          f"= {r.novel_rate:.3%}")

go_novel_df.to_csv(os.path.join(OUT, 'go_term_novelty_rates.tsv'), sep='\t', index=False)

# ─── C7. 우선순위 검증 목록 생성 ──────────────────────────────────────────
print("\n[C7] Priority validation list", flush=True)

# 기준: pred > 0.6, feature type = Type1/Type2 (structural change = interpretable)
# 또는 BISECT gene
priority_rules = [
    ('BISECT gene + pred>0.5', lambda df: df[(df['gene'].isin(bisect_genes if bisect_df is not None else set())) & (df['pred_score'] > 0.5)]),
    ('Type1_DomainLoss + pred>0.6', lambda df: df[(df['feature_type']=='Type1_DomainLoss') & (df['pred_score'] > 0.6)]),
    ('Type2_PartialTrunc + pred>0.6', lambda df: df[(df['feature_type']=='Type2_PartialTrunc') & (df['pred_score'] > 0.6)]),
    ('Type3_SameDomain + pred>0.7', lambda df: df[(df['feature_type']=='Type3_SameDomain') & (df['pred_score'] > 0.7)]),
]

all_priority = []
for rule_name, rule_fn in priority_rules:
    sub = rule_fn(nov)
    print(f"\n  Rule: {rule_name}")
    print(f"    {len(sub)} cases, {sub['gene'].nunique()} genes")
    top = sub.sort_values('pred_score', ascending=False).drop_duplicates(['gene','go_id']).head(20)
    for _, r in top.head(10).iterrows():
        print(f"      gene={r.gene} iso={r.isoform} GO={r.go_id} "
              f"({r.go_name[:35]}) pred={r.pred_score:.4f} ft={r.feature_type}")
    top['priority_rule'] = rule_name
    all_priority.append(top)

priority_df = pd.concat(all_priority, ignore_index=True).drop_duplicates(['gene','go_id'])
priority_df = priority_df.sort_values('pred_score', ascending=False)
priority_df.to_csv(os.path.join(OUT, 'priority_validation_list.tsv'), sep='\t', index=False)
print(f"\n  Priority list: {len(priority_df)} unique (gene, GO) pairs")

# ─── C8. 검증 가능 케이스 상세: Specialization이 있고 Novel이 있는 유전자 ──
print("\n[C8] Genes with BOTH specialization AND novel GO prediction", flush=True)

acq_genes = set(acq['gene'].unique())
nov_genes  = set(nov[nov['pred_score'] > 0.5]['gene'].unique())
both_genes = acq_genes & nov_genes
print(f"  Genes with specialization: {len(acq_genes):,}")
print(f"  Genes with novel GO (pred>0.5): {len(nov_genes):,}")
print(f"  Both: {len(both_genes):,}")

# 이런 유전자는: 기존 GO를 특정 isoform이 전담 + 다른 GO를 novel로 획득
both_nov = nov[(nov['gene'].isin(both_genes)) & (nov['pred_score'] > 0.5)]
both_acq = acq[acq['gene'].isin(both_genes)]
print("\n  Top dual-function genes (specialization + novel, sorted by novel pred):")
top_both = both_nov.sort_values('pred_score', ascending=False).drop_duplicates('gene').head(20)
for _, r in top_both.iterrows():
    n_acq = len(both_acq[both_acq['gene']==r.gene])
    print(f"    gene={r.gene} novel_GO={r.go_name[:35]} pred={r.pred_score:.4f} "
          f"acq_cases={n_acq} ft={r.feature_type}")

# ─── Summary ──────────────────────────────────────────────────────────────
summary = {
    'novel_go_total': len(nov),
    'novel_go_pred_gt05': len(hc_nov),
    'novel_go_pred_gt05_genes': int(hc_nov['gene'].nunique()),
    'novel_go_pred_gt06': len(hc_nov3),
    'bisect_overlap': len(nov_in_bisect) if bisect_df is not None else 'n/a',
    'both_spec_and_novel': len(both_genes),
    'priority_list_size': len(priority_df),
    'top_novel_go_terms': go_novel_df.head(5)[['go_id','go_name','novel_rate']].to_dict('records'),
}
with open(os.path.join(OUT, 'novel_go_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[Done] Outputs: {OUT}")
print("  go_term_novelty_rates.tsv")
print("  priority_validation_list.tsv")
print("  novel_go_summary.json")
