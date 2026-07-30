"""
exp_go_domain_attention.py — GO term별 도메인 중요도 Attention 계산
=============================================================================
아이디어 (사용자 제안):
  GO term T에 positive인 이소폼들의 도메인 벡터를 모아
  어떤 도메인이 T와 강하게 연결돼 있는지 중요도 계산 (Attention weight)
  → 각 이소폼의 label confidence = 보유 도메인의 중요도 가중합

구현:
  1. GO term별 domain importance 계산
     imp(d, j) = P(domain_d | pos_j) - P(domain_d | neg_j)  [PMI-like]
     → PMI / cosine / log-odds 모두 계산, 가장 해석 쉬운 log-odds 채택

  2. 각 이소폼별 label confidence
     conf(i, j) = Σ_d [imp(d,j)⁺ × dm[i,d]] / Σ_d [imp(d,j)⁺ × dm_canonical[i,d]]
     → 1.0 = canonical 수준, 0.0 = 도메인 완전 소실

  3. v3 domain matrix 사용 (버전 불일치 수정 후)

출력: reports/isoform_resolution_full/
  - go_domain_attention.tsv  (GO×512 importance matrix)
  - label_confidence.npy     (36748×82 continuous label weights)
  - figure_domain_attention.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/welcome1/sw1686/DIFFUSE")
OUT  = ROOT / "reports/isoform_resolution_full"
OUT.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ───────────────────────────────────────────────────────────
print("=== 데이터 로드 ===")

# domain matrix v3 사용 (수정 버전); 없으면 v1 fallback
dm_v3_path = ROOT / "hMuscle/results_isoform/features/domain_matrix_proper_test_v3.npy"
dm_v1_path = ROOT / "hMuscle/results_isoform/features/domain_matrix_proper_test.npy"
if dm_v3_path.exists():
    dm = np.load(dm_v3_path)
    print(f"  Using v3 domain matrix: {dm.shape}, nonzero={( dm.sum(1)>0).sum()}")
else:
    dm = np.load(dm_v1_path)
    print(f"  Fallback to v1: {dm.shape}, nonzero={(dm.sum(1)>0).sum()}")

Y_te   = np.load(ROOT / "reports/v17f_star_bootstrap/Y_te.npy")    # (36748, 82)
preds  = np.load(ROOT / "reports/v17f_star_bootstrap/v17f_star_preds.npy")  # (36748, 82)
n_iso, n_go = Y_te.shape
n_dom = dm.shape[1]
print(f"  isoforms={n_iso}, GO terms={n_go}, Pfam domains={n_dom}")

# gene grouping
ts_iso  = np.load(ROOT / "hMuscle/data/test_set/isoform_list.npy", allow_pickle=True)
ts_gene = np.load(ROOT / "hMuscle/data/test_set/gene_list.npy", allow_pickle=True)
ts_iso  = [x.decode() if isinstance(x,bytes) else x for x in ts_iso]
ts_gene = [x.decode() if isinstance(x,bytes) else x for x in ts_gene]
model_iso = np.load(ROOT / "hMuscle/model/my_isoform_list_fixed.npy", allow_pickle=True)
model_iso = [x.decode() if isinstance(x,bytes) else x for x in model_iso]
iso2gene  = dict(zip(ts_iso, ts_gene))
gene_arr  = np.array([iso2gene.get(i, "UNKNOWN") for i in model_iso])

domain_ct = dm.sum(axis=1).astype(int)
gene_max_dc = {}
for g in set(gene_arr):
    gene_max_dc[g] = int(domain_ct[gene_arr==g].max())
canonical_dc = np.array([gene_max_dc.get(g, 0) for g in gene_arr])

# feature_type
feature_type = np.empty(n_iso, dtype=object)
for i, (idc, cdc) in enumerate(zip(domain_ct, canonical_dc)):
    if cdc == 0:           feature_type[i] = "Type0"
    elif idc == 0:         feature_type[i] = "Type1"
    elif idc < cdc:        feature_type[i] = "Type2"
    else:                  feature_type[i] = "Type3"

print(f"  Feature types: {dict(zip(*np.unique(feature_type, return_counts=True)))}")

# canonical domain vector per isoform (gene내 max domain count 이소폼의 도메인)
gene_canonical_dm = {}
for g in set(gene_arr):
    mask = (gene_arr == g)
    dc_g = domain_ct[mask]
    if dc_g.max() == 0:
        gene_canonical_dm[g] = np.zeros(n_dom, dtype=np.float32)
    else:
        # 도메인 가장 많은 이소폼들의 union
        max_dc = dc_g.max()
        canonical_mask = mask.copy()
        canonical_mask[mask] = (dc_g == max_dc)
        gene_canonical_dm[g] = dm[canonical_mask].max(axis=0)

canonical_dm = np.stack([gene_canonical_dm.get(g, np.zeros(n_dom, dtype=np.float32))
                          for g in gene_arr])  # (36748, 512)

# ═══════════════════════════════════════════════════════════════════════════
# 1. GO term별 Domain Importance (log-odds)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== GO-Domain Importance (log-odds) ===")

EPS = 1e-4
importance = np.zeros((n_go, n_dom), dtype=np.float32)  # (82, 512)

valid_go = []
for j in range(n_go):
    pos_mask = Y_te[:, j] == 1
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos < 3:
        continue
    valid_go.append(j)

    # P(domain d | positive) vs P(domain d | negative)
    p_pos = (dm[pos_mask].sum(axis=0) + EPS) / (n_pos + EPS)  # (512,)
    p_neg = (dm[neg_mask].sum(axis=0) + EPS) / (n_neg + EPS)  # (512,)

    log_odds = np.log(p_pos / p_neg)  # positive = domain enriched in pos class
    importance[j] = log_odds

print(f"  Valid GO terms: {len(valid_go)}/{n_go}")
print(f"  log-odds range: [{importance[valid_go].min():.3f}, {importance[valid_go].max():.3f}]")

# Top 도메인-GO 연결 (가장 강한 양성 연결)
print("\n  Top 10 domain-GO associations (highest log-odds):")
with open(ROOT / "hMuscle/results_isoform/features/domain_pfam_vocab_v2.txt") as f:
    vocab = {}
    for line in f:
        parts = line.strip().split('\t')
        vocab[int(parts[0])] = parts[1]

top_idx = np.unravel_index(
    np.argsort(importance[valid_go].flatten())[-10:][::-1],
    importance[valid_go].shape)
for gi_local, di in zip(top_idx[0], top_idx[1]):
    gi = valid_go[gi_local]
    print(f"  GO[{gi}] — Domain {vocab.get(di,'?')} (col={di}): log-odds={importance[gi,di]:.3f}, "
          f"n_pos={Y_te[:,gi].sum():.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. 이소폼별 Label Confidence 계산
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Label Confidence 계산 ===")

# imp_pos[j,d] = max(log_odds[j,d], 0) → 도메인 d가 GO j와 양성으로 연결된 정도
imp_pos = np.maximum(importance, 0)  # (82, 512) only positive associations

# confidence(i, j) = dot(dm[i], imp_pos[j]) / (dot(canonical_dm[i], imp_pos[j]) + eps)
# → 1.0: 이소폼이 canonical과 동일한 관련 도메인 보유
# → 0.0: 관련 도메인 전부 소실
# → >1.0: canonical보다 더 많은 관련 도메인 (드물지만 possible in gene family)

label_conf = np.zeros((n_iso, n_go), dtype=np.float32)

for j in valid_go:
    imp_j = imp_pos[j]  # (512,)
    total_importance = imp_j.sum()
    if total_importance < EPS:
        # 이 GO term에 연결된 도메인 없음 → 도메인으로 판단 불가
        label_conf[:, j] = 1.0  # label 그대로 유지
        continue

    iso_score = dm @ imp_j          # (n_iso,)  dot product
    can_score = canonical_dm @ imp_j  # (n_iso,)

    # confidence = iso_score / can_score (nan 처리)
    with np.errstate(divide='ignore', invalid='ignore'):
        conf = np.where(can_score > EPS, iso_score / can_score, 1.0)
    conf = np.clip(conf, 0.0, 1.5)  # 1.5 이상은 클리핑
    label_conf[:, j] = conf.astype(np.float32)

print(f"  label_conf shape: {label_conf.shape}")
print(f"  Overall: mean={label_conf.mean():.4f}, median={np.median(label_conf):.4f}")

# Feature Type별 label confidence 분포
for t in ["Type1", "Type2", "Type3"]:
    mask = (feature_type == t)
    conf_t = label_conf[mask]
    print(f"  {t}: mean_conf={conf_t.mean():.4f}, "
          f"<0.5={( conf_t<0.5).mean()*100:.1f}%, "
          f"<0.1={( conf_t<0.1).mean()*100:.1f}%")

np.save(OUT / "label_confidence.npy", label_conf)
print(f"  Saved: {OUT}/label_confidence.npy")

# ═══════════════════════════════════════════════════════════════════════════
# 3. AUPRC with confidence-weighted labels vs original
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== AUPRC 비교: original vs confidence-masked ===")
from sklearn.metrics import average_precision_score

def macro_auprc(y_true, y_pred, min_pos=3):
    scores = []
    for j in range(y_true.shape[1]):
        if y_true[:,j].sum() >= min_pos:
            scores.append(average_precision_score(y_true[:,j], y_pred[:,j]))
    return np.mean(scores) if scores else np.nan

# Type1 이소폼만 분석
for t, mask_t in [("Type1", feature_type=="Type1"),
                   ("Type2", feature_type=="Type2"),
                   ("Type3", feature_type=="Type3")]:
    if mask_t.sum() < 20:
        continue

    yt_orig = Y_te[mask_t]
    yp = preds[mask_t]
    conf_t = label_conf[mask_t]

    # 원래 AUPRC
    auprc_orig = macro_auprc(yt_orig, yp)

    # Confidence threshold masking: conf < 0.3 → label을 0으로 처리
    yt_masked = yt_orig.copy().astype(float)
    yt_masked[conf_t < 0.3] = 0.0
    # 단, 양성이 최소 3개 남아있는 term만 평가
    auprc_masked = macro_auprc(yt_masked, yp)

    # Confidence-weighted: conf<0.3인 positive label을 unknown(제외)으로 처리
    # → label 유지하되 낮은 비율 확인
    conf_of_pos = conf_t[yt_orig == 1]
    low_conf_rate = (conf_of_pos < 0.3).mean()

    print(f"  {t}: n={mask_t.sum():,}")
    print(f"    orig AUPRC:         {auprc_orig:.4f}")
    print(f"    masked(≥0.3) AUPRC: {auprc_masked:.4f}  "
          f"(low-conf pos removed: {low_conf_rate*100:.1f}%)")
    print(f"    → low-conf rate interpretation: {low_conf_rate*100:.1f}% of positive labels "
          f"are domain-inconsistent (gene-label confusion)")

# ═══════════════════════════════════════════════════════════════════════════
# 4. 시각화
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.patch.set_facecolor("white")
plt.subplots_adjust(wspace=0.35)

# ── 4a. GO-Domain importance heatmap (top 20 domains × 12 GO terms) ─────
ax = axes[0]
top20_dom = np.argsort(imp_pos[valid_go].max(axis=0))[-20:][::-1]
top12_go  = valid_go[:12]  # first 12 valid GO terms
heat = importance[np.ix_(top12_go, top20_dom)]
im = ax.imshow(heat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(20))
ax.set_xticklabels([vocab.get(d,"?") for d in top20_dom], fontsize=6, rotation=90)
ax.set_yticks(range(12))
ax.set_yticklabels([f"GO[{j}]" for j in top12_go], fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("① GO-Domain log-odds\n(red=enriched in positive)", fontsize=10, fontweight="bold")

# ── 4b. Label Confidence 분포 by Feature Type ───────────────────────────
ax = axes[1]
colors_t = {"Type1":"#2E75B6","Type2":"#ED7D31","Type3":"#A9A9A9"}
for t, c in colors_t.items():
    mask_t = (feature_type == t)
    conf_vals = label_conf[mask_t].flatten()
    conf_vals = conf_vals[Y_te[mask_t].flatten() == 1]  # positive label인 경우만
    if len(conf_vals) == 0:
        continue
    bins = np.linspace(0, 1.5, 40)
    ax.hist(conf_vals, bins=bins, color=c, alpha=0.6, density=True,
            label=f"{t} (n pos={len(conf_vals):,})")
ax.axvline(0.3, color="#C00000", lw=1.5, ls="--", label="conf=0.3 threshold")
ax.axvline(1.0, color="#333", lw=1.2, ls=":", label="canonical level")
ax.set_xlabel("Label Confidence", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("② Label Confidence 분포\n(positive-labeled isoforms만)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# ── 4c. Low-confidence label 비율 by feature type ───────────────────────
ax = axes[2]
thresholds = [0.1, 0.2, 0.3, 0.5]
x = np.arange(len(thresholds))
width = 0.25
for i, (t, c) in enumerate(colors_t.items()):
    mask_t = (feature_type == t)
    conf_pos = label_conf[mask_t][Y_te[mask_t] == 1]
    frac = [(conf_pos < thr).mean() for thr in thresholds]
    ax.bar(x + i*width, frac, width=width, color=c, alpha=0.85,
           edgecolor="#333", linewidth=0.7, label=t)

ax.set_xticks(x + width)
ax.set_xticklabels([f"conf<{t}" for t in thresholds], fontsize=9)
ax.set_ylabel("Fraction of positive labels", fontsize=10)
ax.set_title("③ Low-confidence positive label 비율\n(= gene-label confusion 정도)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle("GO-Domain Attention: Label Confidence Quantification",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "figure_domain_attention.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {OUT}/figure_domain_attention.png")

# ── importance matrix 저장 ─────────────────────────────────────────────
imp_df = pd.DataFrame(importance, columns=[vocab.get(i,f"PF{i}") for i in range(n_dom)])
imp_df.index = [f"GO_{j}" for j in range(n_go)]
imp_df.to_csv(OUT / "go_domain_importance.tsv", sep="\t")
print(f"Importance matrix saved: {OUT}/go_domain_importance.tsv")

print("\n=== 완료 ===")
