# -*- coding: utf-8 -*-
"""
compare_baselines.py
====================
DIFFUSE 원본 및 각 버전 이소폼 기능 예측 성능 비교 분석.

실행:
    cd /home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform
    python compare_baselines.py

출력:
    - 콘솔: 전체 성능 비교 테이블 (AUROC / AUPRC / Best-F1)
    - plots/comparison_auroc.png
    - plots/comparison_auprc.png
    - plots/comparison_prcurve.png  (PR curve 오버레이, GO_0006936 기준)
    - comparison_results.csv
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve, f1_score)

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------
BASE_ISO  = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform'
BASE_GENE = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results'
PLOTS_DIR = os.path.join(BASE_ISO, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

GO_TERMS = ['GO_0006096', 'GO_0006412', 'GO_0006936', 'GO_0022900']
GO_NAMES = {
    'GO_0006096': 'Glycolysis\n(pos=76)',
    'GO_0006412': 'Translation\n(pos=701)',
    'GO_0006936': 'Muscle contraction\n(pos=597)',
    'GO_0022900': 'Electron transport\n(pos=291)',
}

# --------------------------------------------------------------------------
# 비교 모델 정의
# --------------------------------------------------------------------------
# 각 항목: (display_name, score_path_template, label_path_template, category)
# {go} = GO term (e.g. GO_0006096)

MODELS = [
    # 1. 원본 DIFFUSE (gene-level 모델, isoform-level 평가)
    ("DIFFUSE (orig)",
     BASE_GENE + "/{go}/{go}_epoch_50_scores.txt",
     BASE_ISO  + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_labels.npy",
     "Baseline"),

    # 2. DIFFUSE + Triplet (이전 실험)
    ("DIFFUSE+Triplet",
     BASE_GENE + "/{go}/{go}_Triplet_prediction_scores.txt",
     BASE_ISO  + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_labels.npy",
     "Baseline"),

    # 3. DIFFUSE + Focal
    ("DIFFUSE+Focal",
     BASE_GENE + "/{go}/{go}_focal_fitted_scores.txt",
     BASE_ISO  + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_labels.npy",
     "Baseline"),

    # 4. DIFFUSE + Ensemble
    ("DIFFUSE+Ensemble",
     BASE_GENE + "/{go}/{go}_ensenbl_fitted_scores.txt",
     BASE_ISO  + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_labels.npy",
     "Baseline"),

    # 5. PFN (원본 PFN bridge)
    ("DIFFUSE+PFN",
     BASE_GENE + "/{go}/{go}_PFN_scores.txt",
     BASE_ISO  + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_labels.npy",
     "Baseline"),

    # 6. v4-3 Phase 2 (CRF 제거, 첫 유효 버전)
    ("v4-3 (Phase2)",
     BASE_ISO + "/{go}/v4-3_integrated_260325/v4-3_integrated_phase2_joint_focal_scores.txt",
     BASE_ISO + "/{go}/v4-3_integrated_260325/v4-3_integrated_phase2_joint_focal_labels.npy",
     "Ours"),

    # 7. v5fix Phase 2
    ("v5fix (Phase2)",
     BASE_ISO + "/{go}/v5fix_integrated_260403/v5fix_integrated_phase2_joint_focal_scores.txt",
     BASE_ISO + "/{go}/v5fix_integrated_260403/v5fix_integrated_phase2_joint_focal_labels.npy",
     "Ours"),

    # 8. v5-1 Phase 2
    ("v5-1 (Phase2)",
     BASE_ISO + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_scores.txt",
     BASE_ISO + "/{go}/v5-1_integrated_260403/v5-1_integrated_phase2_joint_focal_labels.npy",
     "Ours"),

    # 9. v5-1 Final (LabelProp, AUROC criterion)
    ("v5-1 Final",
     BASE_ISO + "/{go}/v5-1_integrated_260403/v5-1_integrated_{go}_Final_LabelProp_scores.txt",
     BASE_ISO + "/{go}/v5-1_integrated_260403/v5-1_integrated_{go}_Final_LabelProp_labels.npy",
     "Ours"),

    # 10. v5-2 Phase 2
    ("v5-2 (Phase2)",
     BASE_ISO + "/{go}/v5-2_integrated_260407/v5-2_integrated_phase2_joint_focal_scores.txt",
     BASE_ISO + "/{go}/v5-2_integrated_260407/v5-2_integrated_phase2_joint_focal_labels.npy",
     "Ours"),

    # 11. v5-2 Final (LabelProp, AUPRC criterion)
    ("v5-2 Final [I2]",
     BASE_ISO + "/{go}/v5-2_integrated_260407/v5-2_integrated_{go}_Final_LabelProp_scores.txt",
     BASE_ISO + "/{go}/v5-2_integrated_260407/v5-2_integrated_{go}_Final_LabelProp_labels.npy",
     "Ours"),

    # 12. v5-3 Phase 2 (margin=0.1 복귀 + [I2][I4 개선][I5])
    ("v5-3 (Phase2)",
     BASE_ISO + "/{go}/v5-3_integrated_20260407/v5-3_integrated_phase2_joint_focal_scores.txt",
     BASE_ISO + "/{go}/v5-3_integrated_20260407/v5-3_integrated_phase2_joint_focal_labels.npy",
     "Ours"),

    # 13. v5-3 Final (LabelProp, AUPRC criterion)
    ("v5-3 Final [I2]",
     BASE_ISO + "/{go}/v5-3_integrated_20260407/v5-3_integrated_{go}_Final_LabelProp_scores.txt",
     BASE_ISO + "/{go}/v5-3_integrated_20260407/v5-3_integrated_{go}_Final_LabelProp_labels.npy",
     "Ours"),
]

# --------------------------------------------------------------------------
# 헬퍼
# --------------------------------------------------------------------------
def load_scores_labels(score_tmpl, label_tmpl, go):
    sf = score_tmpl.format(go=go)
    lf = label_tmpl.format(go=go)
    if not os.path.exists(sf) or not os.path.exists(lf):
        return None, None
    try:
        data   = np.genfromtxt(sf, dtype=str)
        scores = data[:, 2].astype(float) if data.ndim > 1 else None
        labels = np.load(lf)
        if scores is None or len(scores) != len(labels):
            return None, None
        return scores, labels
    except Exception:
        return None, None


def compute_metrics(scores, labels):
    """AUROC, AUPRC, Best-F1 계산."""
    if scores is None or labels is None:
        return dict(auroc=None, auprc=None, best_f1=None)
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return dict(auroc=None, auprc=None, best_f1=None)
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1s  = 2 * prec * rec / (prec + rec + 1e-8)
    return dict(auroc=auroc, auprc=auprc, best_f1=f1s.max())


# --------------------------------------------------------------------------
# 계산
# --------------------------------------------------------------------------
print("\n" + "=" * 90)
print("Isoform Function Prediction — Baseline Comparison")
print("=" * 90)

all_results = {}   # all_results[model_name][go] = metrics dict

for model_name, score_tmpl, label_tmpl, category in MODELS:
    all_results[model_name] = {}
    for go in GO_TERMS:
        scores, labels = load_scores_labels(score_tmpl, label_tmpl, go)
        all_results[model_name][go] = compute_metrics(scores, labels)

# --------------------------------------------------------------------------
# AUROC 테이블 출력
# --------------------------------------------------------------------------
def fmt(v, bold=False):
    if v is None:
        return "{:>8}".format("—")
    s = "{:.4f}".format(v)
    return "{:>8}".format(s)


print("\n--- AUROC ---")
header = "{:22} {:>8} {:>8} {:>8} {:>8}  {:>8}".format(
    "Model", *[g.replace('GO_00','GO:00') for g in GO_TERMS], "Mean")
print(header)
print("-" * 68)
best_auroc = {go: 0.0 for go in GO_TERMS}
for model_name, _, _, _ in MODELS:
    r = all_results[model_name]
    vals = [r[go]['auroc'] for go in GO_TERMS]
    for go, v in zip(GO_TERMS, vals):
        if v is not None and v > best_auroc[go]:
            best_auroc[go] = v
    mean_val = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
    print("{:22}".format(model_name[:22]) + "".join(fmt(v) for v in vals) +
          ("  " + fmt(mean_val)))

print("\n--- AUPRC ---")
print(header)
print("-" * 68)
best_auprc = {go: 0.0 for go in GO_TERMS}
for model_name, _, _, _ in MODELS:
    r = all_results[model_name]
    vals = [r[go]['auprc'] for go in GO_TERMS]
    for go, v in zip(GO_TERMS, vals):
        if v is not None and v > best_auprc[go]:
            best_auprc[go] = v
    mean_val = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
    print("{:22}".format(model_name[:22]) + "".join(fmt(v) for v in vals) +
          ("  " + fmt(mean_val)))

print("\n--- Best-F1 ---")
print(header)
print("-" * 68)
for model_name, _, _, _ in MODELS:
    r = all_results[model_name]
    vals = [r[go]['best_f1'] for go in GO_TERMS]
    mean_val = np.mean([v for v in vals if v is not None]) if any(v is not None for v in vals) else None
    print("{:22}".format(model_name[:22]) + "".join(fmt(v) for v in vals) +
          ("  " + fmt(mean_val)))

# --------------------------------------------------------------------------
# CSV 저장
# --------------------------------------------------------------------------
import csv
csv_path = os.path.join(BASE_ISO, 'comparison_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header_row = ['model', 'category'] + [
        '{}_{}'.format(go, m) for go in GO_TERMS for m in ['auroc','auprc','f1']
    ]
    writer.writerow(header_row)
    for model_name, _, _, category in MODELS:
        row = [model_name, category]
        for go in GO_TERMS:
            r = all_results[model_name][go]
            row += [
                "{:.4f}".format(r['auroc'])   if r['auroc']   is not None else '',
                "{:.4f}".format(r['auprc'])   if r['auprc']   is not None else '',
                "{:.4f}".format(r['best_f1']) if r['best_f1'] is not None else '',
            ]
        writer.writerow(row)
print("\n[Saved] {}".format(csv_path))

# --------------------------------------------------------------------------
# 시각화
# --------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── 1. AUROC / AUPRC bar chart (4 GO terms × selected models) ──────────
    selected_models = [
        ("DIFFUSE (orig)",   "gray",    "--"),
        ("DIFFUSE+Triplet",  "#95a5a6", "--"),
        ("v4-3 (Phase2)",    "#3498db", "-"),
        ("v5-1 (Phase2)",    "#2ecc71", "-"),
        ("v5-1 Final",       "#27ae60", "-"),
        ("v5-2 (Phase2)",    "#e67e22", "-"),
        ("v5-2 Final [I2]",  "#e74c3c", "-"),
    ]
    # filter to available
    selected_models = [(n, c, ls) for n, c, ls in selected_models
                       if any(all_results[n][go]['auroc'] is not None for go in GO_TERMS)]

    x = np.arange(len(GO_TERMS))
    bar_w = 0.8 / max(len(selected_models), 1)

    for metric_key, metric_label, y_min in [
            ('auroc', 'AUROC', 0.4),
            ('auprc', 'AUPRC', 0.0)]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (mname, color, _) in enumerate(selected_models):
            vals = [all_results[mname][go][metric_key] for go in GO_TERMS]
            vals_plot = [v if v is not None else 0.0 for v in vals]
            offset = (i - len(selected_models) / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals_plot, bar_w * 0.9,
                          label=mname, color=color, alpha=0.85,
                          edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([GO_NAMES[g] for g in GO_TERMS], fontsize=9)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_ylim(y_min, 1.02)
        ax.set_title("Isoform Function Prediction — {}".format(metric_label), fontsize=12)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.8, ncol=2)
        ax.axhline(0.5, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, 'comparison_{}.png'.format(metric_key))
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print("[Plot] Saved: {}".format(path))

    # ── 2. PR Curve overlay (GO_0006936, selected models) ─────────────────
    focus_go = 'GO_0006936'
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    pr_models = [
        ("DIFFUSE (orig)",   "gray",    "--",  2.0),
        ("v4-3 (Phase2)",    "#3498db", "-",   2.0),
        ("v5-1 Final",       "#27ae60", "-",   2.0),
        ("v5-2 Final [I2]",  "#e74c3c", "-",   2.5),
    ]
    pr_models = [(n, c, ls, lw) for n, c, ls, lw in pr_models
                 if all_results[n][focus_go]['auroc'] is not None]

    for mname, _, _, _ in MODELS:
        score_tmpl = [s for name, s, _, _ in MODELS if name == mname][0]
        label_tmpl = [l for name, _, l, _ in MODELS if name == mname][0]

    for mname, color, ls, lw in pr_models:
        score_tmpl_item = [s for name, s, _, _ in MODELS if name == mname][0]
        label_tmpl_item = [l for name, _, l, _ in MODELS if name == mname][0]
        scores, labels = load_scores_labels(score_tmpl_item, label_tmpl_item, focus_go)
        if scores is None:
            continue
        prec, rec, _ = precision_recall_curve(labels, scores)
        auprc = average_precision_score(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(rec, prec, color=color, linestyle=ls, linewidth=lw,
                label="{} (AUPRC={:.3f}, AUROC={:.3f})".format(mname, auprc, auroc))

    # random baseline
    pos_ratio = (labels == 1).mean() if labels is not None else 0.016
    ax.axhline(pos_ratio, color='black', linestyle=':', linewidth=1.0,
               label="Random (AUPRC={:.4f})".format(pos_ratio))

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("{} ({}) — Precision-Recall Curve".format(
        focus_go, "Muscle contraction"), fontsize=11)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'comparison_prcurve_{}.png'.format(focus_go))
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[Plot] Saved: {}".format(path))

    # ── 3. Version evolution line plot (AUROC + AUPRC 추이) ────────────────
    evo_models = ["DIFFUSE (orig)", "v4-3 (Phase2)", "v5fix (Phase2)",
                  "v5-1 (Phase2)", "v5-1 Final", "v5-2 (Phase2)", "v5-2 Final [I2]"]
    evo_models = [m for m in evo_models if m in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for go_idx, go in enumerate(GO_TERMS):
        color = colors[go_idx]
        for ax_i, metric in enumerate(['auroc', 'auprc']):
            vals = [all_results[m][go][metric] for m in evo_models]
            x_plot = range(len(evo_models))
            valid_x = [i for i, v in enumerate(vals) if v is not None]
            valid_v = [vals[i] for i in valid_x]
            if valid_v:
                axes[ax_i].plot(valid_x, valid_v, 'o-', color=color,
                                label=go.replace('GO_00','GO:00'), linewidth=1.8,
                                markersize=5, alpha=0.85)

    for ax_i, (metric, ylabel) in enumerate([('auroc', 'AUROC'), ('auprc', 'AUPRC')]):
        axes[ax_i].set_xticks(range(len(evo_models)))
        axes[ax_i].set_xticklabels(evo_models, rotation=25, ha='right', fontsize=8)
        axes[ax_i].set_ylabel(ylabel, fontsize=10)
        axes[ax_i].set_title("Version Evolution — {}".format(ylabel), fontsize=10)
        axes[ax_i].legend(fontsize=8, framealpha=0.8)
        axes[ax_i].grid(alpha=0.3)
        axes[ax_i].set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'comparison_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[Plot] Saved: {}".format(path))

except ImportError as e:
    print("[Plot] matplotlib not available: {}".format(e))

print("\n[Done] Comparison complete. Plots in: {}".format(PLOTS_DIR))
