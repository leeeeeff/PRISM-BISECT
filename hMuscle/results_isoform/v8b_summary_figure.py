# -*- coding: utf-8 -*-
# v8b_summary_figure.py
#
# v8b 임베딩 분석 종합 Figure 생성
#
# Figure 1: Phase 진행 t-SNE (5 GO term × Phase 0/1/2) + 지표 annotation
# Figure 2: Prototype Group 분석 (Type-B 2개 GO term)
# Figure 3: 성능 요약 bar chart (v7c vs v8b)
#
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ── 설정 ──────────────────────────────────────────────────────────────────────
VER_TAG  = 'v8b_integrated'
BASE_DIR = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform'
OUT_DIR  = os.path.join(BASE_DIR, 'v8b_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# 결과 디렉토리 매핑
SAVE_DIRS = {
    'GO:0007204': os.path.join(BASE_DIR, 'GO_0007204', 'v8b_integrated_20260430_1459'),
    'GO:0030017': os.path.join(BASE_DIR, 'GO_0030017', 'v8b_integrated_20260430_1543'),
    'GO:0006941': os.path.join(BASE_DIR, 'GO_0006941', 'v8b_integrated_20260430_1618'),
    'GO:0003774': os.path.join(BASE_DIR, 'GO_0003774', 'v8b_integrated_20260430_1459'),
    'GO:0006096': os.path.join(BASE_DIR, 'GO_0006096', 'v8b_integrated_20260430_1539'),
}

# v7c baseline (from memory)
V7C_AUPRC = {
    'GO:0007204': 0.1220, 'GO:0030017': 0.1580,
    'GO:0006941': 0.0850, 'GO:0003774': 0.4200, 'GO:0006096': 0.7900,
}
V8B_AUPRC = {
    'GO:0007204': 0.1462, 'GO:0030017': 0.1570,
    'GO:0006941': 0.1177, 'GO:0003774': 0.5686, 'GO:0006096': 0.7945,
}
GO_TYPE = {
    'GO:0007204': 'B', 'GO:0030017': 'B', 'GO:0006941': 'B',
    'GO:0003774': 'A', 'GO:0006096': 'A',
}
GO_NAME = {
    'GO:0007204': 'Calcium\nsignaling',
    'GO:0030017': 'Sarcomere\norganization',
    'GO:0006941': 'Striated muscle\ncontraction',
    'GO:0003774': 'Myosin motor\nactivity',
    'GO:0006096': 'Glycolytic\nprocess',
}

# Phase별 지표 (분석 결과 하드코딩)
PHASE_METRICS = {
    'GO:0003774': {
        'ph0': {'cc': 0.9956, 'sep': 1.4139, 'sil': -0.1289, 'lr': 0.9141},
        'ph1': {'cc': 0.5592, 'sep': 1.4800, 'sil':  0.3308, 'lr': 0.9784},
        'ph2': {'cc': 0.6237, 'sep': 1.8080, 'sil':  0.2757, 'lr': 0.9844},
        'delta_cc': +0.0645, 'f5': True,
    },
    'GO:0006096': {
        'ph0': {'cc': 0.9814, 'sep': 1.8250, 'sil':  0.0542, 'lr': 0.9911},
        'ph1': {'cc': 0.1473, 'sep': 2.4437, 'sil':  0.6967, 'lr': 0.9993},
        'ph2': {'cc': 0.1200, 'sep': 2.6303, 'sil':  0.7791, 'lr': 0.9993},
        'delta_cc': -0.0273, 'f5': False,
    },
    'GO:0006941': {
        'ph0': {'cc': 0.9982, 'sep': 1.0551, 'sil':  0.0386, 'lr': 0.8459},
        'ph1': {'cc': 0.7541, 'sep': 1.1385, 'sil':  0.2170, 'lr': 0.9438},
        'ph2': {'cc': 0.7454, 'sep': None,   'sil':  None,   'lr': None  },
        'delta_cc': -0.0087, 'f5': False,
    },
    'GO:0007204': {
        'ph0': {'cc': 0.9905, 'sep': 1.0602, 'sil':  0.0661, 'lr': 0.8274},
        'ph1': {'cc': 0.8072, 'sep': 1.0250, 'sil':  0.2873, 'lr': 0.9136},
        'ph2': {'cc': 0.8030, 'sep': 1.0399, 'sil':  0.3282, 'lr': 0.9205},
        'delta_cc': -0.0041, 'f5': False,
    },
    'GO:0030017': {
        'ph0': {'cc': 0.9970, 'sep': 1.0620, 'sil':  0.0101, 'lr': 0.8011},
        'ph1': {'cc': 0.8075, 'sep': 1.0818, 'sil':  0.1943, 'lr': 0.9136},
        'ph2': {'cc': 0.8100, 'sep': 1.0962, 'sil':  0.2370, 'lr': 0.9212},
        'delta_cc': +0.0025, 'f5': False,
    },
}

PROTO_DATA = {
    'GO:0007204': {
        'k': 5,
        'proto_cos': [[1.000,0.541,0.248,0.180,0.249],
                      [0.541,1.000,0.603,0.255,0.422],
                      [0.248,0.603,1.000,0.455,0.424],
                      [0.180,0.255,0.455,1.000,0.388],
                      [0.249,0.422,0.424,0.388,1.000]],
        'pos_dist': [4, 134, 80, 23, 69],
        'neg_dist': [93-4, 30188-134, 191-80, 621-23, 5655-69],
        'per_sep':  [3.714, 0.650, 3.280, 1.463, 1.089],
        'neg_cos':  [0.643, 0.347, 0.110, 0.050, 0.086],
        'lr_acc': 0.923,
    },
    'GO:0030017': {
        'k': 5,
        'proto_cos': [[1.000,0.290,0.618,0.614,0.277],
                      [0.290,1.000,0.405,0.423,0.471],
                      [0.618,0.405,1.000,0.319,0.118],
                      [0.614,0.423,0.319,1.000,0.284],
                      [0.277,0.471,0.118,0.284,1.000]],
        'pos_dist': [180, 12, 219, 9, 32],
        'neg_dist': [1729-180, 330-12, 33162-219, 519-9, 1008-32],
        'per_sep':  [2.380, 0.820, 0.691, 1.467, 1.203],
        'neg_cos':  [0.082, 0.484, 0.067, 0.135, 0.672],
        'lr_acc': 0.920,
    },
}

# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────
POS_C = '#d62728'; NEG_C = '#aec7e8'
PROTO_COLORS = ['#2ca02c','#ff7f0e','#9467bd','#8c564b','#e377c2']
PHASE_COLORS = {'ph0': '#cccccc', 'ph1': '#4878d0', 'ph2': '#ee854a'}

def load_emb(save_dir, phase_name):
    p = os.path.join(save_dir, '{}_{}_embeddings.npy'.format(VER_TAG, phase_name))
    if not os.path.exists(p):
        return None
    e = np.load(p).astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(n, 1e-8, None)

def load_labels(save_dir, go_safe):
    p = os.path.join(save_dir, '{}_{}_Final_labels.npy'.format(VER_TAG, go_safe))
    return np.load(p).flatten().astype(int) if os.path.exists(p) else None

def subsample(emb, labels, max_neg=1500, seed=42):
    np.random.seed(seed)
    pi = np.where(labels == 1)[0]
    ni = np.where(labels == 0)[0]
    if len(ni) > max_neg:
        ni = np.random.choice(ni, max_neg, replace=False)
    idx = np.sort(np.concatenate([pi, ni]))
    return emb[idx], labels[idx], idx

def run_tsne(emb, perp=25, seed=42):
    p = min(perp, len(emb)//4, 50); p = max(p, 5)
    return TSNE(n_components=2, perplexity=p, random_state=seed,
                n_iter=500, method='barnes_hut').fit_transform(emb.astype(np.float32))

def annotate_metrics(ax, m, phase_key, fontsize=7.5):
    """ax 우상단에 지표 텍스트 박스"""
    cc  = m.get('cc');  sep = m.get('sep')
    sil = m.get('sil'); lr  = m.get('lr')
    lines = []
    if cc  is not None: lines.append('cc={:+.3f}'.format(cc))
    if sep is not None: lines.append('sep={:.3f}'.format(sep))
    if sil is not None: lines.append('sil={:+.3f}'.format(sil))
    if lr  is not None: lines.append('LR={:.3f}'.format(lr))
    txt = '\n'.join(lines)
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            fontsize=fontsize, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75, ec='gray', lw=0.5))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Phase 진행 t-SNE (5 GO × 3 Phase)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Figure 1: Phase Progression t-SNE ===")

GO_ORDER = ['GO:0007204','GO:0030017','GO:0006941','GO:0003774','GO:0006096']
PHASES   = [
    ('phase0_initial_untrained', 'Phase 0\n(Untrained)',    'ph0'),
    ('phase1_contrastive',       'Phase 1\n(Contrastive)', 'ph1'),
    ('phase2_unified',           'Phase 2\n(Unified)',     'ph2'),
]

fig1 = plt.figure(figsize=(18, 16))
fig1.suptitle('v8b — Embedding Space Progression (t-SNE, neg subsampled n=1500)',
              fontsize=14, fontweight='bold', y=0.995)

outer = gridspec.GridSpec(len(GO_ORDER), 1, figure=fig1,
                          hspace=0.45, left=0.06, right=0.98, top=0.97, bottom=0.04)

for row_i, go_id in enumerate(GO_ORDER):
    save_dir = SAVE_DIRS[go_id]
    go_safe  = go_id.replace(':', '_')
    labels   = load_labels(save_dir, go_safe)
    if labels is None:
        continue
    n_pos = int((labels == 1).sum())
    m_all = PHASE_METRICS[go_id]

    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[row_i], wspace=0.08)

    # 좌측 GO term 정보 라벨
    ax_label = fig1.add_subplot(outer[row_i])
    ax_label.set_visible(False)

    for col_i, (pname, ptitle, pkey) in enumerate(PHASES):
        ax = fig1.add_subplot(inner[col_i])
        emb = load_emb(save_dir, pname)
        if emb is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        emb_s, lab_s, _ = subsample(emb, labels)
        t2d = run_tsne(emb_s)
        pm  = (lab_s == 1); nm = (lab_s == 0)

        ax.scatter(t2d[nm,0], t2d[nm,1], c=NEG_C, alpha=0.3, s=6, lw=0, rasterized=True)
        ax.scatter(t2d[pm,0], t2d[pm,1], c=POS_C, alpha=0.9, s=18, lw=0, zorder=3)

        # 지표 annotation
        m = m_all[pkey]
        annotate_metrics(ax, m, pkey)

        # [F5] 표시 (Phase 2만)
        if col_i == 2:
            dcc   = m_all['delta_cc']
            f5_ok = not m_all['f5']
            cc_str = 'Δcc={:+.3f} {}'.format(dcc, '✓' if f5_ok else '⚠[F5]')
            color  = '#2ca02c' if f5_ok else '#d62728'
            ax.text(0.03, 0.97, cc_str, transform=ax.transAxes,
                    fontsize=7.5, va='top', ha='left', color=color, fontweight='bold')

        # 제목 (첫 행만 컬럼 레이블)
        if row_i == 0:
            ax.set_title(ptitle, fontsize=10, fontweight='bold', pad=4)

        # GO term 라벨 (첫 컬럼만)
        if col_i == 0:
            type_str = 'Type-{}'.format(GO_TYPE[go_id])
            go_label = '{}\n{}\n{}\nn_pos={}'.format(
                go_id, GO_NAME[go_id], type_str, n_pos)
            ax.set_ylabel(go_label, fontsize=7.5, labelpad=6,
                          rotation=0, ha='right', va='center',
                          multialignment='center')

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

# 공통 legend
leg_handles = [
    Patch(color=POS_C, label='Positive (functional)'),
    Patch(color=NEG_C, label='Negative (subsampled 1500)'),
    Line2D([0],[0], marker='', color='none', label='—— Annotation box: cc=centroid_cos, sep=sep_ratio,'),
    Line2D([0],[0], marker='', color='none', label='    sil=silhouette, LR=logistic regression AUROC'),
    Line2D([0],[0], marker='', color='#2ca02c', lw=2, label='Δcc ✓: Phase2 centroid_cos stable (|Δ|<0.05)'),
    Line2D([0],[0], marker='', color='#d62728', lw=2, label='Δcc ⚠[F5]: centroid_cos worsened (Δ>0.05)'),
]
fig1.legend(handles=leg_handles, loc='lower center', ncol=3,
            fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.001))

out1 = os.path.join(OUT_DIR, 'fig1_phase_tsne.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  Saved: {}".format(out1))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Prototype Group 분석 (Type-B: GO:0007204, GO:0030017)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Figure 2: Prototype Group Analysis ===")

TYPE_B_GOS = ['GO:0007204', 'GO:0030017']

fig2 = plt.figure(figsize=(20, 12))
fig2.suptitle('v8b — Prototype Group Analysis (Type-B GO terms)',
              fontsize=13, fontweight='bold', y=0.99)

outer2 = gridspec.GridSpec(2, 1, figure=fig2, hspace=0.45,
                           left=0.05, right=0.98, top=0.95, bottom=0.07)

for row_i, go_id in enumerate(TYPE_B_GOS):
    save_dir = SAVE_DIRS[go_id]
    go_safe  = go_id.replace(':', '_')
    labels   = load_labels(save_dir, go_safe)
    pd_info  = PROTO_DATA[go_id]
    k        = pd_info['k']

    emb_p2  = load_emb(save_dir, 'phase2_unified')
    assign  = np.load(os.path.join(save_dir,
              '{}_{}_proto_assignments.npy'.format(VER_TAG, go_safe))).astype(int)

    emb_s, lab_s, idx_s = subsample(emb_p2, labels)
    assign_s = assign[idx_s]
    t2d = run_tsne(emb_s)

    inner2 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer2[row_i], wspace=0.25,
        width_ratios=[1.4, 1.4, 0.9, 0.9])

    # ── (A) pos/neg + prototype positions ──────────────────────────────────
    ax = fig2.add_subplot(inner2[0])
    pm = (lab_s == 1); nm = (lab_s == 0)
    ax.scatter(t2d[nm,0], t2d[nm,1], c=NEG_C, alpha=0.25, s=5, lw=0, rasterized=True)
    ax.scatter(t2d[pm,0], t2d[pm,1], c=POS_C, alpha=0.9,  s=18, lw=0, zorder=3)

    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    proto_np = np.load(os.path.join(save_dir,
               '{}_{}_proto_embeddings.npy'.format(VER_TAG, go_safe))).astype(np.float32)
    sim = cos_sim(proto_np, emb_s)
    for pi in range(k):
        ni_nn = sim[pi].argmax()
        sep_v = pd_info['per_sep'][pi]
        color = PROTO_COLORS[pi % len(PROTO_COLORS)]
        ec    = 'red' if sep_v < 1.0 else ('orange' if sep_v < 1.5 else 'black')
        ax.scatter(t2d[ni_nn,0], t2d[ni_nn,1], c=color, s=200, marker='*',
                   edgecolors=ec, linewidths=1.5, zorder=6,
                   label='P{} sep={:.2f}'.format(pi, sep_v))

    m_p2 = PHASE_METRICS[go_id]['ph2']
    annotate_metrics(ax, m_p2, 'ph2', fontsize=7)
    ax.set_title('(A) Pos/Neg + Prototype positions\n{} {} | AUPRC={:.4f}'.format(
        go_id, GO_NAME[go_id].replace('\n',' '), V8B_AUPRC[go_id]), fontsize=8.5)
    ax.legend(fontsize=6.5, markerscale=0.8, loc='lower left',
              title='★ = proto center\n(border: red=sep<1, orange=sep<1.5)', title_fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])

    # ── (B) Positive prototype group coloring ──────────────────────────────
    ax2 = fig2.add_subplot(inner2[1])
    ax2.scatter(t2d[nm,0], t2d[nm,1], c='#e0e0e0', alpha=0.2, s=5, lw=0, rasterized=True)
    pos_sub_idx = np.where(lab_s == 1)[0]
    pos_assign_s = assign_s[lab_s == 1]
    for gi in range(k):
        grp = (pos_assign_s == gi)
        if grp.sum() == 0: continue
        n_g   = int(grp.sum())
        sep_v = pd_info['per_sep'][gi]
        neg_c_v = pd_info['neg_cos'][gi]
        color = PROTO_COLORS[gi % len(PROTO_COLORS)]
        ax2.scatter(t2d[pos_sub_idx[grp],0], t2d[pos_sub_idx[grp],1],
                    c=color, alpha=0.9, s=20, lw=0, zorder=3,
                    label='G{}: n={} sep={:.2f} neg_cos={:.2f}'.format(
                        gi, n_g, sep_v, neg_c_v))
    ax2.set_title('(B) Positive sub-groups by prototype\nLR group acc={:.3f}'.format(
        pd_info['lr_acc']), fontsize=8.5)
    ax2.legend(fontsize=6.5, loc='lower left', markerscale=1.2,
               title='Group: n=count sep=sep_ratio neg_cos=proto↔neg', title_fontsize=5.5)
    ax2.set_xticks([]); ax2.set_yticks([])

    # ── (C) Proto-Proto cosine heatmap ────────────────────────────────────
    ax3 = fig2.add_subplot(inner2[2])
    cos_mat = np.array(pd_info['proto_cos'])
    im = ax3.imshow(cos_mat, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    for i in range(k):
        for j in range(k):
            v = cos_mat[i,j]
            color = 'white' if v > 0.6 else 'black'
            ax3.text(j, i, '{:.2f}'.format(v), ha='center', va='center',
                     fontsize=7.5, color=color, fontweight='bold' if i==j else 'normal')
    ax3.set_xticks(range(k)); ax3.set_yticks(range(k))
    ax3.set_xticklabels(['P{}'.format(i) for i in range(k)], fontsize=7)
    ax3.set_yticklabels(['P{}'.format(i) for i in range(k)], fontsize=7)
    ax3.set_title('(C) Proto-Proto\ncosine similarity', fontsize=8.5)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='cosine')

    # ── (D) Per-group sep_ratio + neg_cos bar ─────────────────────────────
    ax4 = fig2.add_subplot(inner2[3])
    x = np.arange(k)
    w = 0.35
    sep_vals = pd_info['per_sep']
    neg_vals = pd_info['neg_cos']
    pos_n    = pd_info['pos_dist']

    bars1 = ax4.bar(x - w/2, sep_vals, w, color=PROTO_COLORS[:k], alpha=0.85,
                    label='sep_ratio (inter/intra)', edgecolor='black', linewidth=0.5)
    bars2 = ax4.bar(x + w/2, neg_vals, w, color=PROTO_COLORS[:k], alpha=0.4,
                    label='proto↔neg cos', edgecolor='black', linewidth=0.5, hatch='//')

    ax4.axhline(1.0, color='red',    lw=1.2, ls='--', label='sep=1.0 (threshold)')
    ax4.axhline(0.5, color='orange', lw=1.0, ls=':',  label='neg_cos=0.5 (caution)')

    for bi, (bar, n) in enumerate(zip(bars1, pos_n)):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 'n={}'.format(n), ha='center', va='bottom', fontsize=6.5)

    ax4.set_xticks(x)
    ax4.set_xticklabels(['G{}'.format(i) for i in range(k)], fontsize=8)
    ax4.set_ylabel('Score', fontsize=8)
    ax4.set_title('(D) Per-group separation\n& neg proximity', fontsize=8.5)
    ax4.legend(fontsize=6, loc='upper right')
    ax4.set_ylim(0, max(max(sep_vals), max(neg_vals)) * 1.25)
    ax4.grid(axis='y', alpha=0.3)

out2 = os.path.join(OUT_DIR, 'fig2_prototype_analysis.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved: {}".format(out2))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: 성능 + 임베딩 품질 종합 요약
# ══════════════════════════════════════════════════════════════════════════════
print("=== Figure 3: Performance & Embedding Summary ===")

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('v8b vs v7c — Performance & Embedding Quality Summary',
              fontsize=13, fontweight='bold')

GO_LIST = ['GO:0007204','GO:0030017','GO:0006941','GO:0003774','GO:0006096']
x = np.arange(len(GO_LIST))
short_labels = ['{}\n(Type-{})'.format(g.replace('GO:',''), GO_TYPE[g]) for g in GO_LIST]

# ── (3-1) AUPRC 비교 bar ──────────────────────────────────────────────────
ax = axes3[0,0]
w  = 0.35
v7c_vals = [V7C_AUPRC[g] for g in GO_LIST]
v8b_vals = [V8B_AUPRC[g] for g in GO_LIST]
b1 = ax.bar(x - w/2, v7c_vals, w, label='v7c (baseline)', color='#4878d0', alpha=0.8,
            edgecolor='navy', linewidth=0.5)
b2 = ax.bar(x + w/2, v8b_vals, w, label='v8b (ours)',     color='#ee854a', alpha=0.8,
            edgecolor='darkorange', linewidth=0.5)
for bi, (v7, v8) in enumerate(zip(v7c_vals, v8b_vals)):
    delta = (v8 - v7) / v7 * 100
    color = '#2ca02c' if delta >= 0 else '#d62728'
    ax.text(bi, max(v7, v8) + 0.01, '{:+.1f}%'.format(delta),
            ha='center', fontsize=8, color=color, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
ax.set_ylabel('AUPRC'); ax.set_title('(1) AUPRC: v7c vs v8b', fontweight='bold')
ax.legend(fontsize=8)
ax.axhline(np.mean(v7c_vals), color='#4878d0', ls='--', lw=1,
           label='v7c macro={:.3f}'.format(np.mean(v7c_vals)))
ax.axhline(np.mean(v8b_vals), color='#ee854a', ls='--', lw=1,
           label='v8b macro={:.3f}'.format(np.mean(v8b_vals)))
ax.set_ylim(0, 0.72)
ax.grid(axis='y', alpha=0.3)
# Macro 텍스트
ax.text(0.97, 0.92,
        'Macro-AUPRC\nv7c: {:.4f}\nv8b: {:.4f}\nΔ: {:+.4f}'.format(
            np.mean(v7c_vals), np.mean(v8b_vals),
            np.mean(v8b_vals)-np.mean(v7c_vals)),
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', fc='#fffbe6', ec='orange', alpha=0.9))

# ── (3-2) centroid_cos Phase 진행 ──────────────────────────────────────────
ax = axes3[0,1]
phases_label = ['Ph0\n(Untrained)', 'Ph1\n(Contrastive)', 'Ph2\n(Unified)']
for go_id in GO_LIST:
    m = PHASE_METRICS[go_id]
    vals = [m['ph0']['cc'], m['ph1']['cc'], m['ph2']['cc'] if m['ph2']['cc'] else np.nan]
    style = '-o' if GO_TYPE[go_id]=='B' else '--s'
    color = '#d62728' if GO_TYPE[go_id]=='B' else '#1f77b4'
    ax.plot([0,1,2], vals, style, color=color, alpha=0.75, lw=1.5, ms=6,
            label='{} ({})'.format(go_id, 'B' if GO_TYPE[go_id]=='B' else 'A'))
ax.axhline(0.05, color='gray', ls=':', lw=0.8, label='ideal threshold')
ax.set_xticks([0,1,2]); ax.set_xticklabels(phases_label)
ax.set_ylabel('centroid_cos ↓ better')
ax.set_title('(2) centroid_cos Progression', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax.text(0.02, 0.06, '↓ Lower = more pos/neg separation', transform=ax.transAxes,
        fontsize=7, style='italic', color='gray')

# ── (3-3) sep_ratio Phase 진행 ─────────────────────────────────────────────
ax = axes3[0,2]
for go_id in GO_LIST:
    m = PHASE_METRICS[go_id]
    vals = [m['ph0']['sep'], m['ph1']['sep'],
            m['ph2']['sep'] if m['ph2']['sep'] else np.nan]
    style = '-o' if GO_TYPE[go_id]=='B' else '--s'
    color = '#d62728' if GO_TYPE[go_id]=='B' else '#1f77b4'
    ax.plot([0,1,2], vals, style, color=color, alpha=0.75, lw=1.5, ms=6,
            label='{} ({})'.format(go_id, GO_TYPE[go_id]))
ax.axhline(1.15, color='orange', ls='--', lw=1, label='Type-A/B threshold (1.15)')
ax.axhline(1.0,  color='red',    ls=':',  lw=1, label='sep=1.0')
ax.set_xticks([0,1,2]); ax.set_xticklabels(phases_label)
ax.set_ylabel('sep_ratio ↑ better')
ax.set_title('(3) sep_ratio Progression', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# ── (3-4) silhouette Phase 진행 ────────────────────────────────────────────
ax = axes3[1,0]
for go_id in GO_LIST:
    m = PHASE_METRICS[go_id]
    sil_ph2 = m['ph2']['sil'] if m['ph2']['sil'] is not None else np.nan
    vals = [m['ph0']['sil'], m['ph1']['sil'], sil_ph2]
    style = '-o' if GO_TYPE[go_id]=='B' else '--s'
    color = '#d62728' if GO_TYPE[go_id]=='B' else '#1f77b4'
    ax.plot([0,1,2], vals, style, color=color, alpha=0.75, lw=1.5, ms=6,
            label='{} ({})'.format(go_id, GO_TYPE[go_id]))
ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_xticks([0,1,2]); ax.set_xticklabels(phases_label)
ax.set_ylabel('Silhouette ↑ better')
ax.set_title('(4) Silhouette Progression', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# ── (3-5) [F5] 요약 + kNN AUROC ────────────────────────────────────────────
ax = axes3[1,1]
knn_auroc = {
    'GO:0007204': 0.809, 'GO:0030017': 0.815, 'GO:0006941': None,
    'GO:0003774': 0.872, 'GO:0006096': 0.954,
}
delta_cc = {g: PHASE_METRICS[g]['delta_cc'] for g in GO_LIST}
f5_flag  = {g: PHASE_METRICS[g]['f5'] for g in GO_LIST}

colors_bar = ['#d62728' if f5_flag[g] else '#2ca02c' for g in GO_LIST]
bars = ax.bar(x, [delta_cc[g] for g in GO_LIST], color=colors_bar, alpha=0.8,
              edgecolor='black', linewidth=0.5)
ax.axhline(0.05,  color='red',    ls='--', lw=1.2, label='⚠ F5 threshold (+0.05)')
ax.axhline(-0.05, color='gray',   ls=':',  lw=0.8)
ax.axhline(0,     color='black',  ls='-',  lw=0.5)
for bi, g in enumerate(GO_LIST):
    knn = knn_auroc[g]
    if knn:
        ax.text(bi, delta_cc[g] + (0.003 if delta_cc[g]>=0 else -0.005),
                'kNN={:.3f}'.format(knn), ha='center', fontsize=7,
                va='bottom' if delta_cc[g]>=0 else 'top')
ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
ax.set_ylabel('Δ centroid_cos (Ph1→Ph2)')
ax.set_title('(5) [F5] Monitor: centroid_cos Change\n(+kNN AUROC Phase2)',
             fontweight='bold')
leg_handles = [Patch(color='#2ca02c', label='✓ [F5] controlled (Δ<0.05)'),
               Patch(color='#d62728', label='⚠ [F5] warning (Δ≥0.05)')]
ax.legend(handles=leg_handles, fontsize=8)
ax.grid(axis='y', alpha=0.3)

# ── (3-6) LR-AUROC Phase 비교 ──────────────────────────────────────────────
ax = axes3[1,2]
for go_id in GO_LIST:
    m = PHASE_METRICS[go_id]
    lr_ph2 = m['ph2']['lr'] if m['ph2']['lr'] is not None else np.nan
    vals = [m['ph0']['lr'], m['ph1']['lr'], lr_ph2]
    style = '-o' if GO_TYPE[go_id]=='B' else '--s'
    color = '#d62728' if GO_TYPE[go_id]=='B' else '#1f77b4'
    ax.plot([0,1,2], vals, style, color=color, alpha=0.75, lw=1.5, ms=6,
            label='{} ({})'.format(go_id, GO_TYPE[go_id]))
ax.set_xticks([0,1,2]); ax.set_xticklabels(phases_label)
ax.set_ylabel('Linear AUROC ↑ better')
ax.set_title('(6) Linear Separability (LR-AUROC)', fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(0.75, 1.02)
ax.grid(alpha=0.3)
ax.text(0.02, 0.04, '─ Type-B (●)   ── Type-A (■)',
        transform=ax.transAxes, fontsize=7.5, style='italic', color='gray')

# 전체 legend 설명
fig3.text(0.5, 0.01,
    'Type-B (red ●): GO terms with low initial sep_ratio (<1.15) — heterogeneous positives, harder to separate\n'
    'Type-A (blue ■): GO terms with high initial sep_ratio (≥1.15) — homogeneous positives, Phase2 benefits most',
    ha='center', fontsize=8.5, style='italic',
    bbox=dict(boxstyle='round', fc='#f0f4ff', ec='#aaaaaa', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
out3 = os.path.join(OUT_DIR, 'fig3_performance_summary.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  Saved: {}".format(out3))

print("\n[Done] All figures saved to: {}".format(OUT_DIR))
print("  fig1_phase_tsne.png         — 5 GO × 3 Phase t-SNE progression")
print("  fig2_prototype_analysis.png — Type-B prototype group detail")
print("  fig3_performance_summary.png— v7c vs v8b performance + metrics")
