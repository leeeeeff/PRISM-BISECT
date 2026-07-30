"""
PRISM Feature Type 전수조사 — 전체 36,748 아이소폼 통계 확장
=============================================================================
기존 42쌍 UniProt + 101 BISECT → 전체 brain test set으로 확장

Feature Type 자동 분류 (domain_matrix 기반):
  Type 1  DOMAIN_COMPLETE_LOSS   : canonical domains >0, isoform domains ==0
  Type 2  DOMAIN_PARTIAL_TRUNC   : 0 < iso_domains < canonical_domains
  Type 3  SAME_DOMAIN_RETAINED   : iso_domains == canonical_domains > 0
  Type 0  NO_DOMAIN              : canonical domains == 0 (uncharacterized)
  (Type 6 Localization: BISECT LOC 케이스 — 별도 표기)

출력: reports/isoform_resolution_full/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import average_precision_score
from scipy import stats
from pathlib import Path

ROOT  = Path("/home/welcome1/sw1686/DIFFUSE")
OUT   = ROOT / "reports/isoform_resolution_full"
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════
print("=== 데이터 로드 ===")

# PRISM v17f* predictions & labels (model 36,748 isoforms)
preds    = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")   # (36748, 82)
Y_te     = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")               # (36748, 82)
model_iso = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
model_iso = np.array([x.decode() if isinstance(x,bytes) else x for x in model_iso])

# domain matrix (aligned with model isoform list) — proper binary Pfam (512 terms)
dm = np.load(ROOT / "hMuscle/results_isoform/features/domain_matrix_proper_test.npy")  # (36748,512) binary
domain_count = dm.sum(axis=1).astype(int)   # 0–7 per isoform

# gene mapping: test_set (36,776) → subset to model (36,748)
ts_iso  = np.load(ROOT / "hMuscle/data/test_set/isoform_list.npy", allow_pickle=True)
ts_gene = np.load(ROOT / "hMuscle/data/test_set/gene_list.npy",    allow_pickle=True)
ts_iso  = np.array([x.decode() if isinstance(x,bytes) else x for x in ts_iso])
ts_gene = np.array([x.decode() if isinstance(x,bytes) else x for x in ts_gene])

iso2gene = dict(zip(ts_iso, ts_gene))
gene_arr = np.array([iso2gene.get(i, "UNKNOWN") for i in model_iso])

print(f"Model isoforms: {len(model_iso)}")
print(f"PRISM preds: {preds.shape}, Y_te: {Y_te.shape}")
print(f"Domain count range: {domain_count.min()}–{domain_count.max()}, "
      f"mean={domain_count.mean():.2f}")
print(f"Genes: {len(set(gene_arr))} unique (excl. UNKNOWN)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature Type 자동 분류
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Feature Type 분류 ===")

# 유전자별 canonical domain count = 유전자 내 최대
gene_max_domain = {}
for gene in set(gene_arr):
    mask = (gene_arr == gene)
    gene_max_domain[gene] = int(domain_count[mask].max())

canonical_dc = np.array([gene_max_domain.get(g, 0) for g in gene_arr])

# 분류
feature_type = np.empty(len(model_iso), dtype=object)

for i, (iso_dc, can_dc) in enumerate(zip(domain_count, canonical_dc)):
    if can_dc == 0:
        feature_type[i] = "Type0_NoDomain"
    elif iso_dc == 0:
        feature_type[i] = "Type1_DomainLoss"
    elif iso_dc < can_dc:
        feature_type[i] = "Type2_PartialTrunc"
    else:                          # iso_dc == can_dc  (same domain count)
        feature_type[i] = "Type3_SameDomain"

# BISECT Type 6 Localization 케이스 오버레이
# (consequence_string에 loc_change 있는 BISECT PASS 케이스)
BISECT_TSV = ROOT / "Final_analysis/pipeline_bioanalysis/outputs/bisect_138_feature_classified.tsv"
if BISECT_TSV.exists():
    bdf = pd.read_csv(BISECT_TSV, sep="\t")
    loc_genes = set(bdf[(bdf["stage2_pass"]=="YES") &
                        (bdf["consequence_string"].str.contains("loc_change", na=False))]["gene"])
    # gene name match (gene_arr는 ENSG, BISECT는 gene symbol → 별도 표시)
    print(f"  BISECT LOC genes (by symbol): {loc_genes}")

counts_full = pd.Series(feature_type).value_counts()
print("\n전체 분포:")
print(counts_full)
total = len(model_iso)
for k, v in counts_full.items():
    print(f"  {k}: {v:,} / {total:,} = {v/total*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Feature Type별 PRISM AUPRC 계산
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Feature Type별 PRISM AUPRC ===")

TYPE_ORDER = ["Type1_DomainLoss", "Type2_PartialTrunc", "Type3_SameDomain", "Type0_NoDomain"]
TYPE_LABEL = {
    "Type1_DomainLoss"  : "Type 1\nComplete Domain Loss",
    "Type2_PartialTrunc": "Type 2\nPartial Truncation",
    "Type3_SameDomain"  : "Type 3\nSame Domain\n(Subtle/Quantitative)",
    "Type0_NoDomain"    : "Type 0\nNo Domain\n(Uncharacterized)",
}
TYPE_COLOR = {
    "Type1_DomainLoss"  : "#2E75B6",
    "Type2_PartialTrunc": "#ED7D31",
    "Type3_SameDomain"  : "#A9A9A9",
    "Type0_NoDomain"    : "#D9D9D9",
}

def macro_auprc(y_true, y_pred, min_pos=3):
    scores = []
    for j in range(y_true.shape[1]):
        if y_true[:,j].sum() >= min_pos:
            scores.append(average_precision_score(y_true[:,j], y_pred[:,j]))
    return np.mean(scores) if scores else np.nan, len(scores)

# Bootstrap CI
def bootstrap_auprc(y_true, y_pred, n_boot=500, min_pos=3, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s, _ = macro_auprc(y_true[idx], y_pred[idx], min_pos)
        if not np.isnan(s):
            boot_scores.append(s)
    return np.percentile(boot_scores, [2.5, 97.5]) if boot_scores else [np.nan, np.nan]

type_stats = []
for t in TYPE_ORDER:
    mask = (feature_type == t)
    n = mask.sum()
    if n < 10:
        print(f"  {t}: n={n} (too few)")
        continue
    yt = Y_te[mask]
    yp = preds[mask]
    auprc, n_terms = macro_auprc(yt, yp)
    ci = bootstrap_auprc(yt, yp, n_boot=300)
    print(f"  {t}: n={n:,}, terms={n_terms}, AUPRC={auprc:.4f} [{ci[0]:.4f},{ci[1]:.4f}]")
    type_stats.append({
        "type": t, "n": int(n), "n_terms": n_terms,
        "auprc": auprc, "ci_lo": ci[0], "ci_hi": ci[1],
    })

type_df = pd.DataFrame(type_stats)

# ── 전체 vs Type별 비교 ──────────────────────────────────────────────────
overall_auprc, n_terms_all = macro_auprc(Y_te, preds)
overall_ci = bootstrap_auprc(Y_te, preds, n_boot=300)
print(f"\n  Overall: n={len(Y_te):,}, AUPRC={overall_auprc:.4f} "
      f"[{overall_ci[0]:.4f},{overall_ci[1]:.4f}]")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Domain Count 분포 분석
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Domain Count 분포 ===")

for t in TYPE_ORDER:
    mask = (feature_type == t)
    dc_sub = domain_count[mask]
    if mask.sum() == 0: continue
    print(f"  {t}: mean={dc_sub.mean():.2f}, median={np.median(dc_sub):.0f}, "
          f"max={dc_sub.max()}, 0-domain_frac={( dc_sub==0).mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. 유전자 내 이소폼 해상도 분석 (within-gene 분포)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Within-Gene 분포 (PRISM score variance) ===")

gene_var_records = []
unique_genes = [g for g in set(gene_arr) if g != "UNKNOWN"]

for gene in unique_genes:
    mask = (gene_arr == gene)
    if mask.sum() < 2:
        continue  # 단일 이소폼 유전자 제외
    yp_gene  = preds[mask]          # (n_iso, 82)
    ft_gene  = feature_type[mask]
    dc_gene  = domain_count[mask]
    can_dc   = gene_max_domain[gene]

    n_iso = mask.sum()
    n_type1 = (ft_gene == "Type1_DomainLoss").sum()
    n_type2 = (ft_gene == "Type2_PartialTrunc").sum()

    # within-gene score variance (mean over GO terms)
    var_within = yp_gene.var(axis=0).mean()

    # max gap = canonical score - min score
    max_pred = yp_gene.max(axis=0)
    min_pred = yp_gene.min(axis=0)
    mean_gap = (max_pred - min_pred).mean()

    gene_var_records.append({
        "gene": gene, "n_iso": int(n_iso),
        "n_type1": int(n_type1), "n_type2": int(n_type2),
        "canonical_dc": int(can_dc),
        "var_within": float(var_within),
        "mean_gap": float(mean_gap),
        "has_domain_variation": int(n_type1 + n_type2 > 0),
    })

gv_df = pd.DataFrame(gene_var_records)
print(f"  Multi-isoform genes: {len(gv_df):,}")
print(f"  Genes with domain variation (Type1 or Type2): "
      f"{gv_df['has_domain_variation'].sum():,} "
      f"({gv_df['has_domain_variation'].mean()*100:.1f}%)")

# 도메인 변이 있는 vs 없는 유전자의 within-gene gap 비교
dom_var  = gv_df[gv_df["has_domain_variation"]==1]["mean_gap"]
no_dom   = gv_df[gv_df["has_domain_variation"]==0]["mean_gap"]
u_stat, p_val = stats.mannwhitneyu(dom_var, no_dom, alternative="greater")
print(f"\n  With domain variation: mean_gap={dom_var.mean():.4f}, n={len(dom_var)}")
print(f"  No domain variation  : mean_gap={no_dom.mean():.4f}, n={len(no_dom)}")
print(f"  MWU test (dom_var > no_dom): U={u_stat:.0f}, p={p_val:.4e}")
print(f"  Effect size (r): {u_stat / (len(dom_var)*len(no_dom)):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Fisher's Exact: gap≥0.10 케이스와 feature type 연관성
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Fisher's Exact: High-Gap와 Feature Type 연관성 ===")

# 각 이소폼의 mean PRISM score (canonical과의 within-gene max-min gap)
iso_gap = np.zeros(len(model_iso))
for gene in unique_genes:
    mask = (gene_arr == gene)
    if mask.sum() < 2: continue
    yp_gene = preds[mask]
    max_s = yp_gene.max(axis=0)
    min_s = yp_gene.min(axis=0)
    gap_v = (max_s - min_s).mean()
    iso_gap[mask] = gap_v  # 동일 유전자 내 모든 이소폼에 같은 gap 부여

high_gap_thr = 0.10
is_high_gap = (iso_gap >= high_gap_thr)

print(f"  High-gap isoforms (gene mean_gap≥{high_gap_thr}): {is_high_gap.sum():,} / {len(model_iso):,}")

for t in ["Type1_DomainLoss", "Type2_PartialTrunc"]:
    is_type = (feature_type == t)
    a = (is_type & is_high_gap).sum()    # type + high gap
    b = (is_type & ~is_high_gap).sum()   # type + low gap
    c = (~is_type & is_high_gap).sum()   # not type + high gap
    d = (~is_type & ~is_high_gap).sum()  # not type + low gap
    oddr, p = stats.fisher_exact([[a,b],[c,d]], alternative="greater")
    print(f"  {t}: OR={oddr:.2f}, p={p:.4e} "
          f"(high_gap: {a}/{a+b}={a/(a+b)*100:.1f}% in type, "
          f"{c}/{c+d}={c/(c+d)*100:.1f}% outside)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. 전수조사 결과 저장
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 결과 저장 ===")

# per-isoform feature type 저장
full_df = pd.DataFrame({
    "isoform_id": model_iso,
    "gene_id": gene_arr,
    "domain_count": domain_count,
    "canonical_domain_count": canonical_dc,
    "feature_type": feature_type,
    "within_gene_gap": iso_gap,
})
full_df.to_csv(OUT / "full_isoform_feature_types.tsv", sep="\t", index=False)
print(f"  Saved: {OUT}/full_isoform_feature_types.tsv ({len(full_df):,} rows)")

# per-type AUPRC 저장
type_df.to_csv(OUT / "type_auprc_stats.tsv", sep="\t", index=False)
print(f"  Saved: {OUT}/type_auprc_stats.tsv")

gv_df.to_csv(OUT / "gene_within_variance.tsv", sep="\t", index=False)
print(f"  Saved: {OUT}/gene_within_variance.tsv")

# ═══════════════════════════════════════════════════════════════════════════
# 8. 시각화
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 시각화 ===")

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.38)

# ── 8a. Feature Type 분포 (전체 36,748) ────────────────────────────────
ax = axes[0]
type_labels = [TYPE_LABEL.get(t, t).replace("\n", " ") for t in TYPE_ORDER]
type_ns     = [counts_full.get(t, 0) for t in TYPE_ORDER]
type_cols   = [TYPE_COLOR.get(t, "#888") for t in TYPE_ORDER]

bars = ax.bar(range(len(TYPE_ORDER)), type_ns, color=type_cols,
              edgecolor="#333", linewidth=0.7, alpha=0.85)
for b, n in zip(bars, type_ns):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+50,
            f"{n:,}\n({n/total*100:.0f}%)", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(TYPE_ORDER)))
ax.set_xticklabels([TYPE_LABEL.get(t,t) for t in TYPE_ORDER], fontsize=7.5)
ax.set_ylabel("Isoform Count", fontsize=10)
ax.set_title("① Feature Type 분포\n(전체 36,748 아이소폼)", fontsize=10, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

# ── 8b. Feature Type별 AUPRC ───────────────────────────────────────────
ax = axes[1]
if len(type_df) > 0:
    xt = type_df["type"].values
    auprcs = type_df["auprc"].values
    ci_lo  = type_df["ci_lo"].values
    ci_hi  = type_df["ci_hi"].values
    cols   = [TYPE_COLOR.get(t, "#888") for t in xt]

    bars2 = ax.bar(range(len(xt)), auprcs, color=cols,
                   edgecolor="#333", linewidth=0.7, alpha=0.85)
    ax.errorbar(range(len(xt)), auprcs,
                yerr=[auprcs-ci_lo, ci_hi-auprcs],
                fmt="none", color="#333", capsize=4, linewidth=1.5)
    ax.axhline(overall_auprc, color="#C00000", lw=1.5, ls="--",
               label=f"Overall={overall_auprc:.3f}")

    for i, (b, v, n) in enumerate(zip(bars2, auprcs, type_df["n"].values)):
        ax.text(b.get_x()+b.get_width()/2, ci_hi[i]+0.005,
                f"{v:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels([TYPE_LABEL.get(t,t) for t in xt], fontsize=7.5)
    ax.set_ylabel("Macro AUPRC", fontsize=10)
    ax.set_title("② Feature Type별 PRISM AUPRC\n(95% CI, 300 bootstrap)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.spines[["top","right"]].set_visible(False)

# ── 8c. Within-gene gap: domain variation vs none ──────────────────────
ax = axes[2]
bins = np.linspace(0, 0.6, 30)
ax.hist(no_dom.clip(0, 0.6), bins=bins, color="#A9A9A9", alpha=0.65,
        label=f"No domain var (n={len(no_dom):,})", density=True)
ax.hist(dom_var.clip(0, 0.6), bins=bins, color="#2E75B6", alpha=0.65,
        label=f"Domain variation (n={len(dom_var):,})", density=True)
ax.axvline(0.10, color="#C00000", lw=1.5, ls="--", label="gap=0.10")
ax.set_xlabel("Within-gene Max PRISM Gap", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"③ Domain 변이 있는 유전자의\nwithin-gene gap 분포\n"
             f"MWU p={p_val:.2e}", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# ── 8d. Domain count vs PRISM max-gap scatter (per gene) ────────────────
ax = axes[3]
# domain count 변이 크기 (canonical - min) vs mean_gap
gv_plot = gv_df[gv_df["canonical_dc"] > 0].copy()
gv_plot["dc_diff"] = gv_plot["canonical_dc"] - (gv_plot["canonical_dc"] - gv_plot["n_type1"] - gv_plot["n_type2"])
# color by has_domain_variation
colors_scatter = ["#2E75B6" if v else "#A9A9A9" for v in gv_plot["has_domain_variation"]]
ax.scatter(gv_plot["canonical_dc"], gv_plot["mean_gap"],
           c=colors_scatter, s=8, alpha=0.35, linewidths=0)

# trend line
x_arr = gv_plot["canonical_dc"].values
y_arr = gv_plot["mean_gap"].values
res_spearman = stats.spearmanr(x_arr, y_arr)
slope, p_corr = res_spearman.statistic, res_spearman.pvalue
ax.set_xlabel("Gene Canonical Domain Count", fontsize=10)
ax.set_ylabel("Within-gene Max PRISM Gap", fontsize=10)
ax.set_title(f"④ Canonical Domain Count vs\nWithin-gene PRISM Gap\n"
             f"Spearman ρ={slope:.3f}, p={p_corr:.2e}", fontsize=10, fontweight="bold")
patches = [mpatches.Patch(color="#2E75B6", label="Has Type1/2"),
           mpatches.Patch(color="#A9A9A9", label="Type3/0 only")]
ax.legend(handles=patches, fontsize=8)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("PRISM Feature Type 전수조사 — 전체 36,748 아이소폼 통계",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "figure_full_feature_stats.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Figure saved: {OUT}/figure_full_feature_stats.png")

# ═══════════════════════════════════════════════════════════════════════════
# 9. 요약 출력
# ═══════════════════════════════════════════════════════════════════════════
summary_lines = [
    "PRISM Feature Type 전수조사 (36,748 아이소폼)",
    "=" * 60,
    "",
    "[ Feature Type 분포 ]",
]
for t in TYPE_ORDER:
    n = counts_full.get(t, 0)
    summary_lines.append(f"  {t}: {n:,} ({n/total*100:.1f}%)")

summary_lines += [
    "",
    "[ Feature Type별 Macro AUPRC ]",
]
for _, row in type_df.iterrows():
    summary_lines.append(
        f"  {row['type']}: {row['auprc']:.4f} [{row['ci_lo']:.4f},{row['ci_hi']:.4f}] "
        f"n={row['n']:,}"
    )
summary_lines.append(f"  Overall: {overall_auprc:.4f} [{overall_ci[0]:.4f},{overall_ci[1]:.4f}]")

summary_lines += [
    "",
    "[ Within-Gene Gap 분석 ]",
    f"  Multi-isoform genes: {len(gv_df):,}",
    f"  With domain variation: {gv_df['has_domain_variation'].sum():,} ({gv_df['has_domain_variation'].mean()*100:.1f}%)",
    f"  Gap (domain var):  mean={dom_var.mean():.4f}",
    f"  Gap (no dom var):  mean={no_dom.mean():.4f}",
    f"  MWU p={p_val:.4e}",
    "",
    "[ Fisher's Exact: High-Gap와 Feature Type ]",
]

summary = "\n".join(summary_lines)
print("\n" + summary)
with open(OUT / "full_summary.txt", "w") as f:
    f.write(summary)
print(f"\n  Summary saved: {OUT}/full_summary.txt")
