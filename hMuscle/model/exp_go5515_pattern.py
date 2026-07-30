"""
exp_go5515_pattern.py — GO:0005515 (protein binding) 분화 패턴 분석
=================================================================
S2 유전자에서 GO:0005515가 "비주류 아이소폼"으로 분화되는 현상의
서열/구조적 결정 인자를 분석.

분석:
  1. 도메인 카운트: GO:0005515-winner vs 다른 GO-winner
     → 도메인 많을수록 / 적을수록 protein binding 획득?

  2. ESM-2 L30 임베딩 거리
     → GO:0005515-winner가 canonical(mode top)에서 얼마나 멀리 떨어져 있나
     → 더 먼 아이소폼이 protein binding을 "전담"하는가

  3. Canonical vs non-canonical
     → GO:0005515를 top하는 아이소폼이 canonical인가 아닌가

  4. Prediction score for GO:0005515: 분화 유전자 vs 비분화 유전자
     → 모델이 GO:0005515를 얼마나 자신있게 예측하는가

  5. ESM-2 embedding norm 차이
     → L30 norm이 큰 아이소폼이 protein binding에 특화?
"""

import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import csv

ROOT     = Path("/home/welcome1/sw1686/DIFFUSE")
OUT_DIR  = ROOT / "reports/isoform_resolution_full"
FEAT_DIR = ROOT / "hMuscle/results_isoform/features"
DATA_DIR = ROOT / "hMuscle/data"

print("[0] 데이터 로드...")
preds    = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")  # (36748, 82)
Y        = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")              # (36748, 82)
X_l30    = np.load(f'{DATA_DIR}/esm2_layer_30_t30_150M.npy').astype(np.float32)  # (36748, 640)

gene_raw = np.load(ROOT / "hMuscle/model/my_gene_list_fixed.npy", allow_pickle=True)
iso_raw  = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_raw]
iso_list  = [x.decode() if isinstance(x, bytes) else x for x in iso_raw]

# feature type + domain count per isoform
ft_arr = []
dc_arr = []  # domain count
iso_ft = {}
iso_dc = {}
with open(OUT_DIR / "full_isoform_feature_types.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, row in enumerate(reader):
        ft_arr.append(row['feature_type'])
        dc = int(row['domain_count'])
        dc_arr.append(dc)
        iso_ft[row['isoform_id']] = row['feature_type']
        iso_dc[row['isoform_id']] = dc
ft_arr = np.array(ft_arr)
dc_arr = np.array(dc_arr)

# canonical isoform per gene (longest CDS)
canonical_iso = {}  # gene_base → canonical iso_id
with open(FEAT_DIR / "canonical_map_v2.txt") as f:
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 3:
            canonical_iso[p[0]] = p[2]  # gene → canonical iso

# GO terms
mf_terms = []
with open(ROOT / "reports/v_expanded_gomf/mf_domain_vs_prism.tsv") as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])
mf_terms = np.array(mf_terms)
go5515_idx = int(np.where(mf_terms == 'GO:0005515')[0][0])
print(f"  GO:0005515 index: {go5515_idx}")

gene_to_idx = defaultdict(list)
for i, g in enumerate(gene_list):
    gene_to_idx[g].append(i)

# Load go attribution scenario classification
go_attr = {}
with open(OUT_DIR / "go_attribution_per_gene.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        go_attr[row['gene']] = row

# Identify S2 genes where GO:0005515 splits
# = S2 gene AND GO:0005515 is among positive GO terms
# AND the top isoform for GO:0005515 is DIFFERENT from the mode top for other GO terms
s2_5515_split = []   # genes where 5515 goes to minority isoform
s2_5515_nosplit = [] # genes where 5515 goes to same (mode) isoform
s1_5515 = []         # S1 genes with 5515

for gene, row in go_attr.items():
    idxs = np.array(gene_to_idx[gene])
    if len(idxs) < 3: continue
    y_g = Y[idxs]; scores_g = preds[idxs]
    pos_go = np.where(y_g.any(0))[0]
    if go5515_idx not in pos_go: continue  # skip if gene doesn't have GO:0005515

    top_iso_per_go = scores_g[:, pos_go].argmax(0)  # local index
    mode_top = int(stats.mode(top_iso_per_go, keepdims=True).mode[0])
    go5515_local_pos = np.where(pos_go == go5515_idx)[0][0]
    go5515_top_local = top_iso_per_go[go5515_local_pos]

    if row['scenario'] == 'S2':
        if go5515_top_local != mode_top:
            s2_5515_split.append((gene, idxs, pos_go, mode_top, go5515_top_local))
        else:
            s2_5515_nosplit.append((gene, idxs, pos_go, mode_top, go5515_top_local))
    elif row['scenario'] == 'S1':
        s1_5515.append((gene, idxs, pos_go, mode_top, go5515_top_local))

print(f"\n  S2 genes with 5515 splitting to minority: {len(s2_5515_split)}")
print(f"  S2 genes where 5515 goes to mode top:    {len(s2_5515_nosplit)}")
print(f"  S1 genes with 5515 present:              {len(s1_5515)}")

# ── 1. Domain count: 5515-winner vs mode-top ─────────────────────────────
print("\n[1] Domain count: GO:0005515-top vs mode-top isoform...")
dc_5515_winner  = []  # domain count of isoform that tops 5515
dc_mode_top     = []  # domain count of mode-top isoform (tops other GO terms)
dc_diff_5515    = []  # diff = dc(5515_winner) - dc(mode_top)

for gene, idxs, pos_go, mode_top, go5515_top in s2_5515_split:
    dc_5515   = dc_arr[idxs[go5515_top]]
    dc_mode   = dc_arr[idxs[mode_top]]
    dc_5515_winner.append(dc_5515)
    dc_mode_top.append(dc_mode)
    dc_diff_5515.append(dc_5515 - dc_mode)

dc_5515_winner = np.array(dc_5515_winner)
dc_mode_top    = np.array(dc_mode_top)
dc_diff_5515   = np.array(dc_diff_5515)

print(f"  5515-winner domain count:   {dc_5515_winner.mean():.2f} ± {dc_5515_winner.std():.2f}")
print(f"  mode-top domain count:      {dc_mode_top.mean():.2f} ± {dc_mode_top.std():.2f}")
print(f"  diff (5515 - mode):         {dc_diff_5515.mean():.3f} ± {dc_diff_5515.std():.3f}")
_, p_dc = stats.wilcoxon(dc_diff_5515[dc_diff_5515 != 0]) if (dc_diff_5515 != 0).sum() > 0 else (0, 1.0)
print(f"  Wilcoxon signed-rank p:     {p_dc:.4e}")
print(f"  5515-winner has FEWER domains: {(dc_diff_5515 < 0).sum()} / {len(dc_diff_5515)} ({(dc_diff_5515<0).mean()*100:.1f}%)")
print(f"  5515-winner has MORE domains:  {(dc_diff_5515 > 0).sum()} / {len(dc_diff_5515)} ({(dc_diff_5515>0).mean()*100:.1f}%)")
print(f"  5515-winner == mode domain count: {(dc_diff_5515 == 0).sum()} / {len(dc_diff_5515)}")

# ── 2. ESM-2 embedding distance ───────────────────────────────────────────
print("\n[2] ESM-2 L30 embedding distance: 5515-winner vs mode-top...")

def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return float('nan')
    return 1.0 - np.dot(a, b) / (na * nb)

dist_5515_to_mode = []   # cosine dist between 5515-winner and mode-top
norm_5515_winner  = []   # L2 norm of 5515-winner embedding
norm_mode_top_arr = []   # L2 norm of mode-top embedding

for gene, idxs, pos_go, mode_top, go5515_top in s2_5515_split:
    emb_5515 = X_l30[idxs[go5515_top]]
    emb_mode = X_l30[idxs[mode_top]]
    d = cosine_dist(emb_5515, emb_mode)
    dist_5515_to_mode.append(d)
    norm_5515_winner.append(np.linalg.norm(emb_5515))
    norm_mode_top_arr.append(np.linalg.norm(emb_mode))

dist_5515_to_mode = np.array([d for d in dist_5515_to_mode if not np.isnan(d)])
norm_5515 = np.array(norm_5515_winner)
norm_mode = np.array(norm_mode_top_arr)

# Compare with S1 genes (5515 goes to canonical = mode top, should be dist=0)
# Instead compare within-gene average pairwise distance for S1 vs S2
s1_avg_dist = []
for gene, idxs, pos_go, mode_top, go5515_top in s1_5515[:200]:  # sample
    dists = []
    for j in range(len(idxs)):
        for k in range(j+1, len(idxs)):
            d = cosine_dist(X_l30[idxs[j]], X_l30[idxs[k]])
            if not np.isnan(d): dists.append(d)
    if dists: s1_avg_dist.append(np.mean(dists))

print(f"  5515-split genes — cosine dist (5515-winner to mode-top):")
print(f"    mean = {dist_5515_to_mode.mean():.4f} ± {dist_5515_to_mode.std():.4f}")
print(f"    median = {np.median(dist_5515_to_mode):.4f}")
print(f"\n  ESM-2 L30 norm:")
print(f"    5515-winner: {norm_5515.mean():.3f} ± {norm_5515.std():.3f}")
print(f"    mode-top:    {norm_mode.mean():.3f} ± {norm_mode.std():.3f}")
_, p_norm = stats.wilcoxon(norm_5515 - norm_mode) if len(norm_5515) > 5 else (0, 1.0)
print(f"    diff Wilcoxon p: {p_norm:.4e}")
print(f"    5515-winner has HIGHER norm: {(norm_5515 > norm_mode).sum()} / {len(norm_5515)} ({(norm_5515>norm_mode).mean()*100:.1f}%)")

# ── 3. Canonical vs non-canonical ─────────────────────────────────────────
print("\n[3] Canonical vs non-canonical: who tops GO:0005515?...")
n_5515_is_canonical   = 0
n_5515_is_longer      = 0  # 5515-winner has more domains than mode-top
n_5515_is_type0       = 0  # 5515-winner is Type0 (no domain)
n_5515_is_type1       = 0  # 5515-winner is Type1 (domain loss)
n_5515_is_type3       = 0  # 5515-winner is Type3 (same domain)

type_pairs = defaultdict(int)

for gene, idxs, pos_go, mode_top, go5515_top in s2_5515_split:
    gene_base = gene.split('.')[0]
    canon_iso = canonical_iso.get(gene_base, None)
    top_iso_id = iso_list[idxs[go5515_top]]
    if canon_iso and top_iso_id == canon_iso:
        n_5515_is_canonical += 1

    ft_5515 = ft_arr[idxs[go5515_top]]
    ft_mode  = ft_arr[idxs[mode_top]]
    type_pairs[(ft_5515[:5], ft_mode[:5])] += 1

    if 'Type0' in ft_5515: n_5515_is_type0 += 1
    elif 'Type1' in ft_5515: n_5515_is_type1 += 1
    elif 'Type3' in ft_5515: n_5515_is_type3 += 1

N = len(s2_5515_split)
print(f"  5515-winner is canonical:  {n_5515_is_canonical} / {N} ({n_5515_is_canonical/N*100:.1f}%)")
print(f"  5515-winner is Type0 (NoDomain): {n_5515_is_type0} / {N} ({n_5515_is_type0/N*100:.1f}%)")
print(f"  5515-winner is Type1 (DomLoss):  {n_5515_is_type1} / {N} ({n_5515_is_type1/N*100:.1f}%)")
print(f"  5515-winner is Type3 (SameDom):  {n_5515_is_type3} / {N} ({n_5515_is_type3/N*100:.1f}%)")

print(f"\n  (5515_type, mode_type) pairs:")
for (t5, tm), cnt in sorted(type_pairs.items(), key=lambda x: -x[1])[:10]:
    print(f"    5515={t5} ← mode={tm}: {cnt}")

# ── 4. Prediction score distribution ──────────────────────────────────────
print("\n[4] GO:0005515 prediction scores: 5515-split vs S1 genes...")
score_5515_winner_list = []
score_mode_for5515_list = []

for gene, idxs, pos_go, mode_top, go5515_top in s2_5515_split:
    score_5515_winner_list.append(preds[idxs[go5515_top], go5515_idx])
    score_mode_for5515_list.append(preds[idxs[mode_top], go5515_idx])

score_5515w = np.array(score_5515_winner_list)
score_5515m = np.array(score_mode_for5515_list)

print(f"  5515-winner GO:0005515 score:   {score_5515w.mean():.4f} ± {score_5515w.std():.4f}")
print(f"  mode-top GO:0005515 score:       {score_5515m.mean():.4f} ± {score_5515m.std():.4f}")
print(f"  gap (winner - mode):             {(score_5515w - score_5515m).mean():.4f}")

# S1 genes: both isoforms → mode-top tops everything including 5515
score_s1_5515 = []
for gene, idxs, pos_go, mode_top, go5515_top in s1_5515:
    score_s1_5515.append(preds[idxs[mode_top], go5515_idx])
score_s1_5515 = np.array(score_s1_5515)
print(f"\n  S1 mode-top GO:0005515 score:   {score_s1_5515.mean():.4f} ± {score_s1_5515.std():.4f}")
_, p_s1_s2 = stats.mannwhitneyu(score_5515w, score_s1_5515, alternative='two-sided')
print(f"  MWU (S2-5515-winner vs S1-mode): p = {p_s1_s2:.4e}")

# ── 5. What other GO terms does the mode-top isoform carry? ───────────────
print("\n[5] Functional profile: what GO terms does the mode-top carry vs 5515-winner?...")
mode_top_go_types = Counter2 = defaultdict(int)  # GO index → count in mode-top
winner_go_types   = defaultdict(int)

for gene, idxs, pos_go, mode_top, go5515_top in s2_5515_split:
    top_iso_per_go = preds[idxs][:, pos_go].argmax(0)
    for k, go_idx in enumerate(pos_go):
        if top_iso_per_go[k] == mode_top:
            mode_top_go_types[go_idx] += 1
        elif top_iso_per_go[k] == go5515_top:
            winner_go_types[go_idx] += 1

print(f"\n  GO terms most often attributed to mode-top isoform (the 'specific function' carrier):")
for go_idx, cnt in sorted(mode_top_go_types.items(), key=lambda x: -x[1])[:10]:
    print(f"    [{go_idx:2d}] {mf_terms[go_idx]}: {cnt}")

print(f"\n  GO terms most often attributed to 5515-winner isoform (the 'binding' carrier):")
for go_idx, cnt in sorted(winner_go_types.items(), key=lambda x: -x[1])[:10]:
    print(f"    [{go_idx:2d}] {mf_terms[go_idx]}: {cnt}")

# ── 6. IDR proxy: L30 embedding norm as disorder proxy ────────────────────
print("\n[6] ESM-2 norm as IDR proxy...")
# High ESM-2 norm → more structured / more uniform context
# Low ESM-2 norm → potentially more disordered/variable

# Among all isoforms: norm distribution by type
for tname in ['Type0_NoDomain', 'Type1_DomainLoss', 'Type2_PartialTrunc', 'Type3_SameDomain']:
    mask = ft_arr == tname
    norms = np.linalg.norm(X_l30[mask], axis=1)
    print(f"  {tname}: mean norm = {norms.mean():.3f} ± {norms.std():.3f} (n={mask.sum()})")

# For 5515-split genes: norm comparison
print(f"\n  In GO:0005515-splitting S2 genes:")
print(f"    5515-winner mean norm: {norm_5515.mean():.3f}")
print(f"    mode-top mean norm:    {norm_mode.mean():.3f}")
diff_sign = "HIGHER" if norm_5515.mean() > norm_mode.mean() else "LOWER"
print(f"    → 5515-winner has {diff_sign} ESM-2 norm")
