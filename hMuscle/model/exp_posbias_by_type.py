"""
pos_bias_by_type.py — Feature Type별 gene-label confusion 정량화
=============================================================================
핵심 질문: Type1/2 AUPRC가 Type3보다 낮은 이유가 gene-label confusion인가?

pos_bias = mean_std(scores within positive-labeled isoforms) / global_std
         → >1이면 isoform-level 분산이 gene-level 평균보다 크다
         → gene-label confusion: 도메인 소실 이소폼에 gene-level GO label 할당
           → positive-labeled isoform 집합에 domain-loss 케이스 혼입

추가: canonical-confusion score
  각 유전자에서 canonical(domain_count 최대) vs alt(domain_count < max) 이소폼의
  PRISM 점수 역전 비율 → gene-label이 alt에도 그대로 붙으면 점수 차이가 사라짐

출력: reports/isoform_resolution_full/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT  = ROOT / "reports/isoform_resolution_full"
OUT.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ──────────────────────────────────────────────────────────
print("=== 데이터 로드 ===")
preds     = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")   # (36748,82)
Y_te      = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")               # (36748,82)
dm        = np.load(ROOT / "hMuscle/results_isoform/features/domain_matrix_proper_test.npy")
domain_ct = dm.sum(axis=1).astype(int)

model_iso = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
model_iso = np.array([x.decode() if isinstance(x,bytes) else x for x in model_iso])

ts_iso    = np.load(ROOT / "hMuscle/data/test_set/isoform_list.npy", allow_pickle=True)
ts_gene   = np.load(ROOT / "hMuscle/data/test_set/gene_list.npy",    allow_pickle=True)
ts_iso    = np.array([x.decode() if isinstance(x,bytes) else x for x in ts_iso])
ts_gene   = np.array([x.decode() if isinstance(x,bytes) else x for x in ts_gene])
iso2gene  = dict(zip(ts_iso, ts_gene))
gene_arr  = np.array([iso2gene.get(i, "UNKNOWN") for i in model_iso])

# feature_type 재계산
gene_max_dc = {}
for g in set(gene_arr):
    gene_max_dc[g] = int(domain_ct[gene_arr==g].max())

canonical_dc = np.array([gene_max_dc.get(g,0) for g in gene_arr])
feature_type = np.empty(len(model_iso), dtype=object)
for i,(idc, cdc) in enumerate(zip(domain_ct, canonical_dc)):
    if cdc == 0:           feature_type[i] = "Type0_NoDomain"
    elif idc == 0:         feature_type[i] = "Type1_DomainLoss"
    elif idc < cdc:        feature_type[i] = "Type2_PartialTrunc"
    else:                  feature_type[i] = "Type3_SameDomain"

TYPE_ORDER = ["Type1_DomainLoss","Type2_PartialTrunc","Type3_SameDomain","Type0_NoDomain"]
TYPE_COLOR = {"Type1_DomainLoss":"#2E75B6","Type2_PartialTrunc":"#ED7D31",
              "Type3_SameDomain":"#A9A9A9","Type0_NoDomain":"#D9D9D9"}
n = len(model_iso)
print(f"Loaded {n} isoforms, 82 GO terms")

# ═══════════════════════════════════════════════════════════════════════════
# 1. pos_bias per feature type
#    pos_bias = std(preds[positive_isoforms]) / std(preds[all_isoforms])
#    null: label shuffle → expected ~1.0
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== pos_bias per Feature Type ===")

def compute_posbias(y_true, y_pred, n_shuffle=200, rng_seed=42):
    """
    Returns: (observed_posbias, shuffle_mean, shuffle_std, z_score)
    y_true: (n,) binary, y_pred: (n,) float
    """
    rng = np.random.RandomState(rng_seed)
    obs_pos = y_true.astype(bool)
    if obs_pos.sum() < 2 or (~obs_pos).sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    global_std = y_pred.std()
    if global_std < 1e-9:
        return np.nan, np.nan, np.nan, np.nan
    observed = y_pred[obs_pos].std() / global_std

    shuffle_vals = []
    for _ in range(n_shuffle):
        shuf = rng.permutation(y_true)
        pos_shuf = shuf.astype(bool)
        if pos_shuf.sum() >= 2:
            shuffle_vals.append(y_pred[pos_shuf].std() / global_std)
    if not shuffle_vals:
        return observed, np.nan, np.nan, np.nan
    sm, ss = np.mean(shuffle_vals), np.std(shuffle_vals)
    z = (observed - sm) / ss if ss > 0 else np.nan
    return observed, sm, ss, z

results = []
for t in TYPE_ORDER:
    mask_t = (feature_type == t)
    n_t = mask_t.sum()
    if n_t < 20:
        continue

    term_obs, term_z = [], []
    for j in range(Y_te.shape[1]):
        y_j  = Y_te[mask_t, j]
        yp_j = preds[mask_t, j]
        if y_j.sum() < 2:
            continue
        obs, sm, ss, z = compute_posbias(y_j, yp_j, n_shuffle=100)
        if not np.isnan(obs):
            term_obs.append(obs)
            if not np.isnan(z):
                term_z.append(z)

    if not term_obs:
        continue

    mean_pb = np.mean(term_obs)
    med_pb  = np.median(term_obs)
    n_sig   = sum(z > 1.96 for z in term_z)  # ~95th percentile
    print(f"  {t}: n_iso={n_t:,}, n_terms={len(term_obs)}, "
          f"mean_pos_bias={mean_pb:.4f}, median={med_pb:.4f}, "
          f"n_sig(z>1.96)={n_sig}/{len(term_z)}")
    results.append({
        "type": t, "n_iso": n_t, "n_terms": len(term_obs),
        "mean_pos_bias": mean_pb, "median_pos_bias": med_pb,
        "n_sig": n_sig, "n_total_terms": len(term_z),
    })

pb_df = pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Canonical-Confusion Score
#    유전자 내 canonical(max_dc) vs alt(dc < max) 이소폼 점수 비교
#    Reversal rate = alt > canonical 비율 (label confusion의 직접 증거)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Canonical-Confusion Score ===")

reversal_by_type = {"Type1_DomainLoss":[], "Type2_PartialTrunc":[], "Type3_SameDomain":[]}
gene_list = [g for g in set(gene_arr) if g != "UNKNOWN"]

for gene in gene_list:
    mask = (gene_arr == gene)
    if mask.sum() < 2:
        continue
    dc_g  = domain_ct[mask]
    ft_g  = feature_type[mask]
    yp_g  = preds[mask]        # (n_iso, 82)
    can_dc = gene_max_dc[gene]

    if can_dc == 0:
        continue  # Type0 gene

    can_mask = (dc_g == can_dc)
    if can_mask.sum() == 0:
        continue

    can_score = yp_g[can_mask].mean(axis=0)  # (82,) canonical mean

    for i_local, (dc_i, ft_i) in enumerate(zip(dc_g, ft_g)):
        if ft_i not in reversal_by_type:
            continue
        alt_score = yp_g[i_local]   # (82,)
        # reversal: alt > canonical (per GO term)
        reversal = (alt_score > can_score).mean()
        reversal_by_type[ft_i].append(float(reversal))

print(f"  (Reversal = alt isoform score > canonical score)")
for t, vals in reversal_by_type.items():
    if not vals:
        continue
    print(f"  {t}: n={len(vals)}, mean_reversal={np.mean(vals):.4f} "
          f"(expected~0.5 if confused, <0.5 if canonical dominates)")

# Type 1 vs Type 3 reversal rate 비교
t1 = np.array(reversal_by_type.get("Type1_DomainLoss",[]))
t2 = np.array(reversal_by_type.get("Type2_PartialTrunc",[]))
t3 = np.array(reversal_by_type.get("Type3_SameDomain",[]))

if len(t1) > 0 and len(t3) > 0:
    u1, p1 = stats.mannwhitneyu(t1, t3, alternative="greater")
    print(f"\n  Type1 vs Type3 reversal (Type1 > Type3?): MWU p={p1:.4e}")
if len(t2) > 0 and len(t3) > 0:
    u2, p2 = stats.mannwhitneyu(t2, t3, alternative="greater")
    print(f"  Type2 vs Type3 reversal (Type2 > Type3?): MWU p={p2:.4e}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Gene-Label Confusion 직접 측정
#    Type1 이소폼에 얼마나 많은 GO label이 부모 유전자에서 상속됐나?
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Gene-Label Confusion (Type1 상속 label 분석) ===")

type1_mask = (feature_type == "Type1_DomainLoss")
type3_mask = (feature_type == "Type3_SameDomain")

# Type1 이소폼의 positive GO label 수
t1_pos_per_iso  = Y_te[type1_mask].sum(axis=1)  # 각 이소폼의 positive term 수
t3_pos_per_iso  = Y_te[type3_mask].sum(axis=1)

print(f"  Type1 (domain_loss) positive GO labels per isoform: "
      f"mean={t1_pos_per_iso.mean():.2f}, median={np.median(t1_pos_per_iso):.0f}, "
      f"max={t1_pos_per_iso.max()}")
print(f"  Type3 (same_domain) positive GO labels per isoform: "
      f"mean={t3_pos_per_iso.mean():.2f}, median={np.median(t3_pos_per_iso):.0f}, "
      f"max={t3_pos_per_iso.max()}")

u_lab, p_lab = stats.mannwhitneyu(t1_pos_per_iso, t3_pos_per_iso, alternative="less")
print(f"  Type1 < Type3 labels? MWU p={p_lab:.4e}")

# Type1 이소폼 중 GO label이 1개 이상인 비율 (→ 부모 유전자에서 상속된 confusion)
t1_has_label = (t1_pos_per_iso > 0).mean()
t3_has_label = (t3_pos_per_iso > 0).mean()
print(f"  Type1 with ≥1 GO label: {t1_has_label*100:.1f}%  (confusion = gene-label inherited)")
print(f"  Type3 with ≥1 GO label: {t3_has_label*100:.1f}%  (expected = isoform has domain)")

# ═══════════════════════════════════════════════════════════════════════════
# 4. 시각화
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.38)

# ── 4a. pos_bias per type ────────────────────────────────────────────────
ax = axes[0]
if len(pb_df) > 0:
    xt = pb_df["type"].values
    vals = pb_df["mean_pos_bias"].values
    cols = [TYPE_COLOR.get(t,"#888") for t in xt]
    bars = ax.bar(range(len(xt)), vals, color=cols, edgecolor="#333",
                  linewidth=0.7, alpha=0.85, width=0.6)
    for b, v, n_s, n_tot in zip(bars, vals, pb_df["n_sig"].values, pb_df["n_total_terms"].values):
        ax.text(b.get_x()+b.get_width()/2, v+0.01,
                f"{v:.3f}\n({n_s}/{n_tot} sig)", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="#888", lw=1.5, ls="--", label="null (shuffle mean)")
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels([t.replace("_","\n") for t in xt], fontsize=8)
    ax.set_ylabel("Mean pos_bias (obs/null)", fontsize=10)
    ax.set_title("① pos_bias per Feature Type\n(>1 = positive isoforms cluster together)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(vals)*1.3 if len(vals) else 2)
    ax.spines[["top","right"]].set_visible(False)

# ── 4b. Reversal rate: Type1/2 vs Type3 (boxplot) ───────────────────────
ax = axes[1]
data_box = [t1, t2, t3]
labels_b = ["Type1\nDomain Loss", "Type2\nPartial Trunc", "Type3\nSame Domain"]
colors_b = ["#2E75B6","#ED7D31","#A9A9A9"]
bp = ax.boxplot([d for d in data_box if len(d)>0],
                labels=[l for l,d in zip(labels_b,data_box) if len(d)>0],
                patch_artist=True, medianprops={"color":"black","lw":2},
                showfliers=False)
for patch, col in zip(bp["boxes"], [c for c,d in zip(colors_b,data_box) if len(d)>0]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.axhline(0.5, color="#C00000", lw=1.5, ls="--", label="50% = confusion baseline")
ax.set_ylabel("Reversal Rate\n(alt > canonical per GO term)", fontsize=10)
ax.set_title("② Canonical-Confusion Score\n(50% = completely confused, <50% = canonical dominates)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# ── 4c. GO label count per type (Type1 vs Type3) ────────────────────────
ax = axes[2]
bins = np.arange(0, 30, 1)
ax.hist(t1_pos_per_iso.clip(0,28), bins=bins, color="#2E75B6", alpha=0.65,
        density=True, label=f"Type1 Domain Loss (n={type1_mask.sum():,})")
ax.hist(t3_pos_per_iso.clip(0,28), bins=bins, color="#A9A9A9", alpha=0.65,
        density=True, label=f"Type3 Same Domain (n={type3_mask.sum():,})")
ax.axvline(t1_pos_per_iso.mean(), color="#2E75B6", lw=2, ls="--",
           label=f"T1 mean={t1_pos_per_iso.mean():.1f}")
ax.axvline(t3_pos_per_iso.mean(), color="#555", lw=2, ls="--",
           label=f"T3 mean={t3_pos_per_iso.mean():.1f}")
ax.set_xlabel("# Positive GO Labels per Isoform", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"③ Gene-Label Confusion\nType1 has ≥1 label: {t1_has_label*100:.0f}%\n"
             f"(domain-loss isoform inheriting gene GO label)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Feature Type별 gene-label confusion 정량화",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "figure_posbias_by_type.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {OUT}/figure_posbias_by_type.png")

# ── 요약 저장 ────────────────────────────────────────────────────────────
summary = f"""
pos_bias & Gene-Label Confusion Analysis (Feature Type별)
==========================================================

[pos_bias per Feature Type]
{pb_df.to_string(index=False) if len(pb_df)>0 else 'N/A'}

[Canonical-Confusion (Reversal Rate)]
Type1_DomainLoss   : mean={np.mean(t1):.4f} (n={len(t1)})
Type2_PartialTrunc : mean={np.mean(t2):.4f} (n={len(t2)})
Type3_SameDomain   : mean={np.mean(t3):.4f} (n={len(t3)})
Type1 > Type3? MWU p={p1:.4e}
Type2 > Type3? MWU p={p2:.4e}

[Gene-Label Confusion (Inherited GO Labels)]
Type1 with >=1 GO label : {t1_has_label*100:.1f}% (domain_loss isoform gets gene-level label)
Type3 with >=1 GO label : {t3_has_label*100:.1f}%
Type1 mean labels/isoform: {t1_pos_per_iso.mean():.2f}
Type3 mean labels/isoform: {t3_pos_per_iso.mean():.2f}
Type1 < Type3 labels? MWU p={p_lab:.4e}
"""
print(summary)
with open(OUT / "posbias_confusion_summary.txt", "w") as f:
    f.write(summary)
print(f"Summary saved: {OUT}/posbias_confusion_summary.txt")
