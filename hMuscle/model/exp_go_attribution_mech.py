"""
exp_go_attribution_mech.py — S2 유전자 기전 심층 분석
=====================================================
GO term별 다른 아이소폼이 top인 S2 유전자(42.4%)에서
어떤 구조적/기능적 메커니즘이 GO 분화를 drive하는가.

분석 항목:
  A. S2 유전자 내 isoform type composition
     - S1 vs S2 유전자에서 Type0/1/2/3 비율 차이
     - "GO를 dominate하는 아이소폼"의 type이 GO별로 다른가?

  B. GO-term 분화 패턴
     - 어떤 GO term 쌍이 가장 자주 서로 다른 아이소폼에 분화되는가
     - 특정 GO term 클러스터링 (kinase 활성 vs 전사 활성 등)

  C. Within-gene score gap과 S2의 관계
     - S2 유전자에서 top/non-top 아이소폼 간 score gap이 더 큰가

  D. Isoform 수와 S2 비율
     - n_isoforms 증가에 따른 S2 비율 상승 곡선

  E. Type1 (domain-loss) 아이소폼의 역할
     - S2 유전자에서 Type1 아이소폼이 특정 GO term에 특화되는가
     - 혹은 Type1은 낮은 점수로 GO를 받지 못하는가

  F. S2 대표 케이스 상세 분석 (GO term명 포함)
"""

import numpy as np
from pathlib import Path
from scipy import stats
import csv, json
from collections import defaultdict, Counter

ROOT     = Path("/home/welcome1/sw1686/DIFFUSE")
OUT_DIR  = ROOT / "reports/isoform_resolution_full"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"

# ── 데이터 로드 ──────────────────────────────────────────────────────────
print("[0] 데이터 로드...")
preds = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")   # (36748, 82)
Y     = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")               # (36748, 82)

gene_raw = np.load(ROOT / "hMuscle/model/my_gene_list_fixed.npy", allow_pickle=True)
iso_raw  = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_raw]
iso_list  = [x.decode() if isinstance(x, bytes) else x for x in iso_raw]

dm = np.load(FEAT_DIR / "domain_matrix_proper_test_v3.npy")  # (36748, 512)

# feature type per isoform
ft_arr = []
iso_to_ft = {}
with open(OUT_DIR / "full_isoform_feature_types.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ft_arr.append(row['feature_type'])
        iso_to_ft[row['isoform_id']] = row['feature_type']
ft_arr = np.array(ft_arr)

# GO term names (index → GO ID)
mf_terms = []
with open(ROOT / "reports/v_expanded_gomf/mf_domain_vs_prism.tsv") as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)

# GO attribution results (gene-level)
go_attr = {}
with open(OUT_DIR / "go_attribution_per_gene.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        go_attr[row['gene']] = row

# gene → isoform index mapping
gene_to_idx = defaultdict(list)
for i, g in enumerate(gene_list):
    gene_to_idx[g].append(i)

n_iso_total, n_go = preds.shape
print(f"  isoforms={n_iso_total}, GO terms={n_go}")

# ── A. S1 vs S2 유전자에서 isoform type composition ──────────────────────
print("\n[A] S1 vs S2 isoform type composition...")
type_names = ['Type0_NoDomain', 'Type1_DomainLoss', 'Type2_PartialTrunc', 'Type3_SameDomain']

s1_type_counts = Counter()
s2_type_counts = Counter()
mixed_type_counts = Counter()
s1_genes = []; s2_genes = []; mixed_genes = []

for gene, row in go_attr.items():
    sc = row['scenario']
    idxs = gene_to_idx[gene]
    types_g = [ft_arr[i] for i in idxs]
    if sc == 'S1':
        for t in types_g: s1_type_counts[t] += 1
        s1_genes.append(gene)
    elif sc == 'S2':
        for t in types_g: s2_type_counts[t] += 1
        s2_genes.append(gene)
    else:
        for t in types_g: mixed_type_counts[t] += 1
        mixed_genes.append(gene)

s1_total = sum(s1_type_counts.values())
s2_total = sum(s2_type_counts.values())

print(f"\n  {'Type':<25} {'S1 %':>8} {'S2 %':>8} {'Diff':>8}")
print(f"  {'-'*50}")
for t in type_names:
    s1_pct = s1_type_counts[t] / s1_total * 100 if s1_total else 0
    s2_pct = s2_type_counts[t] / s2_total * 100 if s2_total else 0
    print(f"  {t:<25} {s1_pct:>7.1f}% {s2_pct:>7.1f}% {s2_pct-s1_pct:>+7.1f}%")

# ── B. "Top 아이소폼"의 type이 GO별로 다른가? ─────────────────────────────
print("\n[B] S2 유전자: top isoform type per GO term...")
# For each S2 gene: what type is the top isoform for each GO term?
top_type_pairs = Counter()   # (type_of_go_a_top, type_of_go_b_top) for GO pairs that differ

go_type_matrix = defaultdict(list)  # GO index → list of top-isoform types across S2 genes
s2_top_types_same = 0   # top types are all same
s2_top_types_diff = 0   # top types differ across GO terms

for gene in s2_genes:
    idxs = np.array(gene_to_idx[gene])
    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    if len(pos_go) < 2: continue

    top_iso_per_go = scores_g[:, pos_go].argmax(0)  # local index within idxs
    top_types_per_go = [ft_arr[idxs[t]] for t in top_iso_per_go]

    unique_top_types = set(top_types_per_go)
    if len(unique_top_types) == 1:
        s2_top_types_same += 1
    else:
        s2_top_types_diff += 1
        # Count type pairs that diverge
        for a in range(len(pos_go)):
            for b in range(a+1, len(pos_go)):
                if top_iso_per_go[a] != top_iso_per_go[b]:
                    t1, t2 = sorted([top_types_per_go[a], top_types_per_go[b]])
                    top_type_pairs[(t1, t2)] += 1

    for j, go_idx in enumerate(pos_go):
        go_type_matrix[go_idx].append(top_types_per_go[j])

print(f"\n  S2 genes with top type SAME across GO terms: {s2_top_types_same} ({s2_top_types_same/len(s2_genes)*100:.1f}%)")
print(f"  S2 genes with top type DIFFERENT across GO terms: {s2_top_types_diff} ({s2_top_types_diff/len(s2_genes)*100:.1f}%)")
print(f"\n  Top diverging type pairs (different GO terms → different types):")
for (t1, t2), cnt in top_type_pairs.most_common(10):
    t1s = t1.split('_')[0] + '_' + t1.split('_')[1][:3]
    t2s = t2.split('_')[0] + '_' + t2.split('_')[1][:3]
    print(f"    {t1s} ↔ {t2s}: {cnt}")

# ── C. Score gap: top vs 2nd best isoform ────────────────────────────────
print("\n[C] Score gap: top vs 2nd isoform, by scenario...")
def compute_score_gap(gene):
    idxs = np.array(gene_to_idx[gene])
    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    if len(pos_go) < 1 or len(idxs) < 2: return float('nan')
    gaps = []
    for j in pos_go:
        col = scores_g[:, j]
        sorted_scores = np.sort(col)[::-1]
        gaps.append(sorted_scores[0] - sorted_scores[1])
    return np.mean(gaps)

s1_gaps = [g for gene in s1_genes if not np.isnan(g := compute_score_gap(gene))]
s2_gaps = [g for gene in s2_genes if not np.isnan(g := compute_score_gap(gene))]
mixed_gaps = [g for gene in mixed_genes if not np.isnan(g := compute_score_gap(gene))]

print(f"  Mean score gap (top - 2nd):")
print(f"    S1: {np.mean(s1_gaps):.4f} ± {np.std(s1_gaps):.4f}")
print(f"    S2: {np.mean(s2_gaps):.4f} ± {np.std(s2_gaps):.4f}")
print(f"    Mixed: {np.mean(mixed_gaps):.4f} ± {np.std(mixed_gaps):.4f}")
mwu_s, mwu_p = stats.mannwhitneyu(s2_gaps, s1_gaps, alternative='two-sided')
print(f"  MWU S2 vs S1: p = {mwu_p:.4e}")

# ── D. n_isoforms vs S2 비율 ─────────────────────────────────────────────
print("\n[D] n_isoforms vs Scenario 2 fraction...")
n_iso_to_scenario = defaultdict(lambda: {'S1': 0, 'S2': 0, 'Mixed': 0})
for gene, row in go_attr.items():
    n = int(row['n_isoforms'])
    n_iso_to_scenario[n][row['scenario']] += 1

print(f"  {'n_iso':>6} {'n_genes':>8} {'S1%':>6} {'S2%':>6}")
for n in sorted(n_iso_to_scenario):
    d = n_iso_to_scenario[n]
    total = sum(d.values())
    if total < 5: continue
    s1_p = d['S1'] / total * 100
    s2_p = d['S2'] / total * 100
    print(f"  {n:>6} {total:>8} {s1_p:>5.1f}% {s2_p:>5.1f}%")

# ── E. Type1 아이소폼의 역할 in S2 유전자 ─────────────────────────────────
print("\n[E] Type1 isoform role in S2 genes...")
type1_is_top_count = 0    # Type1 아이소폼이 어떤 GO term에서 top
type1_is_top_go_list = []  # which GO terms Type1 tops
type1_never_top = 0        # Type1이 한 번도 top 아닌 유전자

s2_has_type1 = 0
for gene in s2_genes:
    idxs = np.array(gene_to_idx[gene])
    types_g = np.array([ft_arr[i] for i in idxs])
    if 'Type1_DomainLoss' not in types_g: continue
    s2_has_type1 += 1
    type1_local_idxs = np.where(types_g == 'Type1_DomainLoss')[0]

    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    top_iso_per_go = scores_g[:, pos_go].argmax(0)

    type1_tops = sum(1 for t in top_iso_per_go if t in type1_local_idxs)
    if type1_tops > 0:
        type1_is_top_count += 1
        type1_is_top_go_list.extend([pos_go[k] for k, t in enumerate(top_iso_per_go)
                                      if t in type1_local_idxs])
    else:
        type1_never_top += 1

print(f"  S2 genes with Type1 isoforms: {s2_has_type1} / {len(s2_genes)} ({s2_has_type1/len(s2_genes)*100:.1f}%)")
if s2_has_type1 > 0:
    print(f"  Among those, Type1 is top for ≥1 GO term: {type1_is_top_count} ({type1_is_top_count/s2_has_type1*100:.1f}%)")
    print(f"  Type1 never top (suppressed by model):      {type1_never_top} ({type1_never_top/s2_has_type1*100:.1f}%)")

# Which GO terms does Type1 dominate?
if type1_is_top_go_list:
    top_go_cnt = Counter(type1_is_top_go_list)
    print(f"\n  GO terms most often topped by Type1 isoforms:")
    for go_idx, cnt in top_go_cnt.most_common(8):
        print(f"    GO term [{go_idx}] {mf_terms[go_idx]}: {cnt} times")

# ── F. S2 대표 케이스: 어떤 GO terms가 어떤 type의 아이소폼으로 가는가 ────
print("\n[F] S2 대표 케이스 상세 (GO term + isoform type)...")

# Sort S2 genes by mean_go_corr (most negative = strongest S2)
s2_details = [(gene, float(go_attr[gene]['mean_go_corr'])) for gene in s2_genes]
s2_details.sort(key=lambda x: x[1])

n_shown = 0
for gene, corr in s2_details:
    if n_shown >= 15: break
    idxs = np.array(gene_to_idx[gene])
    if len(idxs) < 3: continue  # only show ≥3 isoform genes for cleaner display
    types_g = np.array([ft_arr[i] for i in idxs])
    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    top_iso_per_go = scores_g[:, pos_go].argmax(0)
    top_types_per_go = [types_g[t].replace('Type', 'T').replace('_DomainLoss','DL')
                         .replace('_PartialTrunc','PT').replace('_NoDomain','ND')
                         .replace('_SameDomain','SD') for t in top_iso_per_go]
    go_names = [mf_terms[j] for j in pos_go[:5]]
    gene_short = gene.split('.')[0]
    print(f"  {gene_short} (corr={corr:.3f}, {len(idxs)}iso):")
    for gname, ttype in zip(go_names, top_types_per_go[:5]):
        print(f"    → {gname}  top_type={ttype}")
    n_shown += 1

# ── G. GO term 분화 빈도: 어떤 GO term들이 자주 분리되는가 ───────────────
print("\n[G] GO terms most frequently splitting across isoforms in S2...")
go_split_count = Counter()  # GO term index → how often it has a DIFFERENT top isoform from the mode

for gene in s2_genes:
    idxs = np.array(gene_to_idx[gene])
    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    if len(pos_go) < 2: continue
    top_iso_per_go = scores_g[:, pos_go].argmax(0)
    mode_top = int(stats.mode(top_iso_per_go, keepdims=True).mode[0])
    for k, go_idx in enumerate(pos_go):
        if top_iso_per_go[k] != mode_top:
            go_split_count[go_idx] += 1

print(f"  Top GO terms that most often go to a 'minority' isoform:")
for go_idx, cnt in go_split_count.most_common(12):
    print(f"    [{go_idx:2d}] {mf_terms[go_idx]}: splits in {cnt} S2 genes")

# ── H. 요약 저장 ─────────────────────────────────────────────────────────
print("\n[H] 요약 저장...")
summary = {
    'n_s1': len(s1_genes),
    'n_s2': len(s2_genes),
    'n_mixed': len(mixed_genes),
    's1_type_fractions': {t: s1_type_counts[t]/s1_total for t in type_names},
    's2_type_fractions': {t: s2_type_counts[t]/s2_total for t in type_names},
    's2_top_type_same_frac': s2_top_types_same / len(s2_genes),
    's2_top_type_diff_frac': s2_top_types_diff / len(s2_genes),
    'mean_score_gap_s1': float(np.mean(s1_gaps)),
    'mean_score_gap_s2': float(np.mean(s2_gaps)),
    'mwu_gap_s2_vs_s1_p': float(mwu_p),
    's2_with_type1_frac': s2_has_type1 / len(s2_genes) if s2_genes else 0,
    'type1_tops_go_in_s2_frac': type1_is_top_count / s2_has_type1 if s2_has_type1 else 0,
    'top_go_split_terms': [(int(k), int(v)) for k, v in go_split_count.most_common(10)],
}
with open(OUT_DIR / "go_attribution_mech_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {OUT_DIR}/go_attribution_mech_summary.json")
