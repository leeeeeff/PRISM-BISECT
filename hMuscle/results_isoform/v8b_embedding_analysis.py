# -*- coding: utf-8 -*-
# v8b_embedding_analysis.py
#
# v8b 임베딩 공간 세밀 분석
# 1. Phase 0→1→2 t-SNE 진행 비교 (pos/neg)
# 2. Prototype group t-SNE (Type-B GO term)
# 3. Positive/Negative 분리 정확도 (LR, kNN)
# 4. Prototype group 간 분류 정확도
# 5. Phase별 정량 지표 테이블
#
# Usage:
#   cd hMuscle/results_isoform
#   conda run -n isoform_env python v8b_embedding_analysis.py
#
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, silhouette_score,
                              confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
VER_TAG   = 'v8b_integrated'
BASE_DIR  = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform'

# v8b 결과 디렉토리 자동 탐색
V8B_DIRS = {}
for go_dir in sorted(glob.glob(os.path.join(BASE_DIR, 'GO_*'))):
    go_id = os.path.basename(go_dir)
    candidates = sorted(glob.glob(os.path.join(go_dir, 'v8b_integrated_*')))
    if candidates:
        V8B_DIRS[go_id] = candidates[-1]  # 가장 최근 결과

print("[Info] Found v8b results:")
for k, v in V8B_DIRS.items():
    print("  {} → {}".format(k, os.path.basename(v)))

# ─────────────────────────────────────────────
# Helper: 임베딩 로드
# ─────────────────────────────────────────────
def load_phase_data(save_dir, go_safe, phase_name):
    """phase_name: 'phase0_initial_untrained', 'phase1_contrastive', 'phase2_unified'"""
    emb_path = os.path.join(save_dir, '{}_{}_{}_embeddings.npy'.format(VER_TAG, phase_name, go_safe))
    lab_path = os.path.join(save_dir, '{}_{}_{}_labels.npy'.format(VER_TAG, phase_name, go_safe))
    # labels는 phase2_unified만 go_safe 포함, phase0/1은 go_safe 없음
    if not os.path.exists(emb_path):
        emb_path = os.path.join(save_dir, '{}_{}_embeddings.npy'.format(VER_TAG, phase_name))
        lab_path = os.path.join(save_dir, '{}_{}_labels.npy'.format(VER_TAG, phase_name))
    if not os.path.exists(emb_path):
        return None, None
    emb = np.load(emb_path).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, None)
    lab = np.load(lab_path).flatten().astype(int) if os.path.exists(lab_path) else None
    return emb, lab

# ─────────────────────────────────────────────
# Helper: 임베딩 품질 지표
# ─────────────────────────────────────────────
def compute_embedding_metrics(emb, labels, n_pairs=1000, seed=42):
    np.random.seed(seed)
    pos_emb = emb[labels == 1]
    neg_emb = emb[labels == 0]
    m = {}

    # centroid_cos
    if len(pos_emb) > 0 and len(neg_emb) > 0:
        pc = normalize(pos_emb.mean(axis=0, keepdims=True))[0]
        nc = normalize(neg_emb.mean(axis=0, keepdims=True))[0]
        m['centroid_cos']  = float(np.dot(pc, nc))
        m['centroid_dist'] = float(np.linalg.norm(pos_emb.mean(0) - neg_emb.mean(0)))
    else:
        m['centroid_cos'] = np.nan; m['centroid_dist'] = np.nan

    # sep_ratio (intra/inter cosine distance)
    if len(pos_emb) >= 2 and len(neg_emb) >= 2:
        n = min(n_pairs, len(pos_emb))
        intra, inter = [], []
        for _ in range(n):
            i, j = np.random.choice(len(pos_emb), 2, replace=False)
            intra.append(max(0, 2 - 2 * float(np.dot(pos_emb[i], pos_emb[j]))))
        for _ in range(n):
            i = np.random.choice(len(pos_emb))
            j = np.random.choice(len(neg_emb))
            inter.append(max(0, 2 - 2 * float(np.dot(pos_emb[i], neg_emb[j]))))
        m['intra_dist'] = float(np.mean(intra))
        m['inter_dist'] = float(np.mean(inter))
        m['sep_ratio']  = m['inter_dist'] / (m['intra_dist'] + 1e-8)

        # frac_pos_near_neg (cosine > 0.9)
        n_check = min(500, len(pos_emb))
        sample  = pos_emb[np.random.choice(len(pos_emb), n_check, replace=False)]
        near    = sum(1 for pe in sample if (neg_emb @ pe > 0.9).any())
        m['frac_pos_near_neg'] = float(near / n_check)
    else:
        m.update({'intra_dist': np.nan, 'inter_dist': np.nan,
                  'sep_ratio': np.nan, 'frac_pos_near_neg': np.nan})

    # silhouette
    if len(pos_emb) >= 2 and len(neg_emb) >= 2 and len(np.unique(labels)) > 1:
        n_sub = min(4000, len(emb))
        idx   = np.random.choice(len(emb), n_sub, replace=False)
        try:
            m['silhouette'] = float(silhouette_score(
                emb[idx], labels[idx], metric='cosine',
                sample_size=min(2000, n_sub), random_state=seed))
        except Exception:
            m['silhouette'] = np.nan
    else:
        m['silhouette'] = np.nan

    # LR linear separability
    if len(pos_emb) >= 5 and len(neg_emb) >= 5 and len(np.unique(labels)) > 1:
        try:
            sc  = StandardScaler()
            X_s = sc.fit_transform(emb)
            lr  = LogisticRegression(max_iter=300, C=1.0, class_weight='balanced', random_state=seed)
            cv  = cross_val_score(lr, X_s, labels, cv=min(5, len(pos_emb)), scoring='roc_auc')
            m['lr_auroc'] = float(cv.mean())
            lr.fit(X_s, labels)
            m['lr_auroc_full'] = float(roc_auc_score(labels, lr.predict_proba(X_s)[:, 1]))
        except Exception:
            m['lr_auroc'] = np.nan; m['lr_auroc_full'] = np.nan
    else:
        m['lr_auroc'] = np.nan; m['lr_auroc_full'] = np.nan

    return m

# ─────────────────────────────────────────────
# Helper: t-SNE (서브샘플링)
# ─────────────────────────────────────────────
TSNE_MAX_NEG = 2000   # neg 최대 샘플 수 (속도)
TSNE_N_ITER  = 500    # 반복 횟수 (기본 1000 → 500)

def subsample_for_tsne(emb, labels, max_neg=TSNE_MAX_NEG, seed=42):
    """pos 전체 + neg 서브샘플 → t-SNE 입력"""
    np.random.seed(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(neg_idx) > max_neg:
        neg_idx = np.random.choice(neg_idx, max_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    idx_sorted = np.sort(idx)
    return emb[idx_sorted], labels[idx_sorted], idx_sorted

def run_tsne(emb, perplexity=30, seed=42):
    n = len(emb)
    perp = min(perplexity, n // 3, 50)
    perp = max(perp, 5)
    return TSNE(n_components=2, perplexity=perp, random_state=seed,
                n_iter=TSNE_N_ITER, method='barnes_hut').fit_transform(emb.astype(np.float32))

# ─────────────────────────────────────────────
# Helper: Prototype group 분석
# ─────────────────────────────────────────────
def analyze_prototype_groups(emb, labels, proto_np, proto_assign, go_id):
    """
    Type-B GO term: prototype 그룹 간 분류 정확도 + 분리 정도
    proto_assign: shape (N_test,) — 모든 test 샘플의 nearest prototype index
    """
    pos_emb = emb[labels == 1]
    neg_emb = emb[labels == 0]
    k = len(proto_np)

    print("\n  [Proto Analysis] k={} | n_pos={} | n_neg={}".format(
        k, len(pos_emb), len(neg_emb)))

    # 1) Prototype 간 cosine similarity 행렬
    proto_norm = proto_np / (np.linalg.norm(proto_np, axis=1, keepdims=True) + 1e-10)
    cos_mat = proto_norm @ proto_norm.T
    print("  [Proto-Proto Cosine Matrix]")
    for i in range(k):
        row = '  '.join('{:+.3f}'.format(cos_mat[i, j]) for j in range(k))
        print("    proto_{}: {}".format(i, row))

    # 2) Prototype 점유율
    if proto_assign is not None and len(proto_assign) > 0:
        counts = np.bincount(proto_assign.astype(int), minlength=k)
        print("  [Proto Occupancy] {}".format(
            ' | '.join('k{}: {}'.format(i, c) for i, c in enumerate(counts))))
        dead = (counts == 0).sum()
        if dead > 0:
            print("  [!] Dead prototypes: {}".format(dead))

    # 3) 그룹 간 분류 (LR on positive embeddings only)
    if proto_assign is not None:
        pos_assign = proto_assign[labels == 1]  # positives only
        unique_g   = np.unique(pos_assign)
        print("  [Pos Prototype Distribution] {}".format(
            ' | '.join('k{}: {}'.format(gi, int((pos_assign==gi).sum()))
                       for gi in range(k))))
        if len(unique_g) > 1 and len(pos_emb) >= k * 2:
            try:
                sc  = StandardScaler()
                X_s = sc.fit_transform(pos_emb)
                lr  = LogisticRegression(max_iter=300, C=1.0, random_state=42,
                                         multi_class='auto')
                cv_acc = cross_val_score(lr, X_s, pos_assign,
                                         cv=min(3, len(unique_g)), scoring='accuracy')
                lr.fit(X_s, pos_assign)
                train_acc = accuracy_score(pos_assign, lr.predict(X_s))
                print("  [Proto LR CV acc (pos only)]: {:.3f} ± {:.3f} | Train acc: {:.3f}".format(
                    cv_acc.mean(), cv_acc.std(), train_acc))
            except Exception as e:
                print("  [Proto LR] Error: {}".format(e))

    # 4) Prototype-to-negative 거리
    if len(neg_emb) > 0:
        for i, p in enumerate(proto_norm):
            neg_cos = neg_emb @ p
            print("  [Proto_{} ↔ neg] cos: mean={:.3f} std={:.3f} max={:.3f}".format(
                i, neg_cos.mean(), neg_cos.std(), neg_cos.max()))

    # 5) 각 Prototype 그룹의 pos/neg 분리도
    if proto_assign is not None:
        pos_assign_local = proto_assign[labels == 1]   # positives only
        print("\n  [Per-Group Sep Ratio]")
        for gi in range(k):
            grp_mask_local = (pos_assign_local == gi)
            if grp_mask_local.sum() < 2:
                print("    Group {}: too few samples ({})".format(gi, grp_mask_local.sum()))
                continue
            grp_emb  = pos_emb[grp_mask_local]
            grp_c    = grp_emb.mean(axis=0)
            neg_dists = np.array([max(0, 2 - 2 * float(np.dot(grp_c / (np.linalg.norm(grp_c)+1e-10),
                                      ne))) for ne in neg_emb[:500]])
            intra_d  = np.mean([max(0, 2 - 2 * float(np.dot(grp_emb[i], grp_emb[j])))
                                 for i in range(len(grp_emb))
                                 for j in range(i+1, min(len(grp_emb), i+10))])
            inter_d  = float(np.mean(neg_dists))
            sep      = inter_d / (intra_d + 1e-8)
            print("    Group {}: n={} | intra={:.3f} | inter={:.3f} | sep={:.3f}".format(
                gi, int(grp_mask_local.sum()), intra_d, inter_d, sep))

    return cos_mat

# ─────────────────────────────────────────────
# 분석 메인
# ─────────────────────────────────────────────
PHASE_NAMES = [
    ('phase0_initial_untrained', 'Phase 0\n(Untrained)'),
    ('phase1_contrastive',       'Phase 1\n(Contrastive)'),
    ('phase2_unified',           'Phase 2\n(Unified)'),
]
POS_COLOR = '#d62728'   # 빨강
NEG_COLOR = '#1f77b4'   # 파랑
PROTO_COLORS = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

all_summary = []

for go_id, save_dir in sorted(V8B_DIRS.items()):
    print("\n" + "=" * 65)
    print(">>> [{}]  {}".format(go_id, os.path.basename(save_dir)))
    print("=" * 65)

    go_safe = go_id.replace(':', '_')
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────
    phases_data = {}
    for pkey, _ in PHASE_NAMES:
        emb, lab = load_phase_data(save_dir, go_safe, pkey)
        if emb is not None:
            phases_data[pkey] = (emb, lab)

    if not phases_data:
        print("  No embedding files found — skip")
        continue

    # labels는 phase2가 확실, 나머지 phase에도 같은 labels
    ref_lab = None
    for pkey in ['phase2_unified', 'phase1_contrastive', 'phase0_initial_untrained']:
        if pkey in phases_data and phases_data[pkey][1] is not None:
            ref_lab = phases_data[pkey][1]
            break

    if ref_lab is None:
        lab_path = os.path.join(save_dir, '{}_{}_Final_labels.npy'.format(VER_TAG, go_safe))
        if os.path.exists(lab_path):
            ref_lab = np.load(lab_path).flatten().astype(int)

    # Phase별 labels 통일
    for pkey in list(phases_data.keys()):
        emb, lab = phases_data[pkey]
        if lab is None and ref_lab is not None:
            phases_data[pkey] = (emb, ref_lab)

    y = ref_lab if ref_lab is not None else np.zeros(len(list(phases_data.values())[0][0]))
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    print("  Labels: pos={} neg={} ratio={:.2f}%".format(n_pos, n_neg,
          n_pos / max(1, len(y)) * 100))

    # ── Prototype 데이터 로드 (Type-B) ──────────────
    proto_np     = None
    proto_assign = None
    is_type_b    = False

    proto_path  = os.path.join(save_dir, '{}_{}_prototypes.npy'.format(VER_TAG, go_safe))
    assign_path = os.path.join(save_dir, '{}_{}_proto_assignments.npy'.format(VER_TAG, go_safe))
    proemb_path = os.path.join(save_dir, '{}_{}_proto_embeddings.npy'.format(VER_TAG, go_safe))

    if os.path.exists(proto_path):
        proto_np  = np.load(proto_path).astype(np.float32)
        is_type_b = True
        if os.path.exists(assign_path):
            # shape: (N_test,) — all test samples, not just positives
            proto_assign = np.load(assign_path).astype(int)
        print("  Type-B | k={} prototypes loaded".format(len(proto_np)))
    else:
        print("  Type-A | no prototypes")

    # ── Phase별 정량 지표 ────────────────────────────
    print("\n  [Phase Metrics]")
    fmt = "{:28} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}"
    print(fmt.format("Phase", "centCos", "sepRatio", "silhouet", "lrAUROC", "frac_pnn", "intraD"))
    print("  " + "-" * 75)

    phase_metrics = {}
    for pkey, plabel in PHASE_NAMES:
        if pkey not in phases_data:
            continue
        emb, lab = phases_data[pkey]
        if lab is None:
            lab = y
        m = compute_embedding_metrics(emb, lab)
        phase_metrics[pkey] = m
        label_short = plabel.replace('\n', ' ')
        print(("  " + fmt).format(
            label_short,
            '{:+.4f}'.format(m['centroid_cos']),
            '{:.4f}'.format(m['sep_ratio']),
            '{:+.4f}'.format(m['silhouette']),
            '{:.4f}'.format(m['lr_auroc_full']),
            '{:.4f}'.format(m['frac_pos_near_neg']),
            '{:.4f}'.format(m['intra_dist'])))

    # centroid_cos Phase 1→2 변화 강조
    if 'phase1_contrastive' in phase_metrics and 'phase2_unified' in phase_metrics:
        cc1 = phase_metrics['phase1_contrastive']['centroid_cos']
        cc2 = phase_metrics['phase2_unified']['centroid_cos']
        delta = cc2 - cc1
        flag = " ⚠ [F5]" if delta > 0.05 else " ✓"
        print("\n  [F5] centroid_cos Ph1→Ph2: {:+.4f} → {:+.4f}  Δ={:+.4f}{}".format(
            cc1, cc2, delta, flag))

    # ── Prototype group 분석 (Type-B) ────────────────
    if is_type_b and 'phase2_unified' in phases_data:
        emb_p2, lab_p2 = phases_data['phase2_unified']
        if lab_p2 is None: lab_p2 = y
        cos_mat = analyze_prototype_groups(emb_p2, lab_p2, proto_np, proto_assign, go_id)

    # ── Positive/Negative 분리 정확도 (kNN) ─────────
    print("\n  [kNN Separation] Phase 2:")
    if 'phase2_unified' in phases_data:
        emb_p2, lab_p2 = phases_data['phase2_unified']
        if lab_p2 is None: lab_p2 = y
        if len(np.unique(lab_p2)) > 1 and n_pos >= 3:
            try:
                sc  = StandardScaler()
                X_s = sc.fit_transform(emb_p2)
                for k_nn in [3, 5, 11]:
                    knn = KNeighborsClassifier(n_neighbors=min(k_nn, n_pos, n_neg),
                                               metric='cosine', weights='distance')
                    cv_k = cross_val_score(knn, X_s, lab_p2,
                                           cv=min(5, n_pos), scoring='roc_auc')
                    print("    kNN(k={:2d}) CV-AUROC: {:.4f} ± {:.4f}".format(
                        k_nn, cv_k.mean(), cv_k.std()))
            except Exception as e:
                print("    kNN error: {}".format(e))

    # ─────────────────────────────────────────────────
    # t-SNE 시각화
    # ─────────────────────────────────────────────────
    print("\n  [t-SNE] Running...")

    # 1) Phase 진행 비교 (pos/neg coloring)
    n_cols  = len(PHASE_NAMES)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle('{} — Phase Progression (t-SNE)'.format(go_id), fontsize=13, y=1.01)

    for ax, (pkey, plabel) in zip(axes, PHASE_NAMES):
        if pkey not in phases_data:
            ax.set_visible(False); continue
        emb, lab = phases_data[pkey]
        if lab is None: lab = y
        emb_sub, lab_sub, _ = subsample_for_tsne(emb, lab)
        tsne_2d = run_tsne(emb_sub)
        lab = lab_sub
        pos_m = (lab == 1); neg_m = (lab == 0)
        ax.scatter(tsne_2d[neg_m, 0], tsne_2d[neg_m, 1],
                   c=NEG_COLOR, alpha=0.3, s=8, label='Neg (n={})'.format(neg_m.sum()))
        ax.scatter(tsne_2d[pos_m, 0], tsne_2d[pos_m, 1],
                   c=POS_COLOR, alpha=0.8, s=20, label='Pos (n={})'.format(pos_m.sum()))
        m = phase_metrics.get(pkey, {})
        cc   = m.get('centroid_cos', np.nan)
        sep  = m.get('sep_ratio', np.nan)
        lrau = m.get('lr_auroc_full', np.nan)
        ax.set_title('{}\ncc={:.3f} sep={:.3f} lr={:.3f}'.format(
            plabel, cc, sep, lrau), fontsize=9)
        ax.legend(fontsize=7, markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out1 = os.path.join(plots_dir, '{}_v8b_phase_tsne.png'.format(go_safe))
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: {}".format(os.path.basename(out1)))

    # 2) Phase 2 + Prototype coloring (Type-B only)
    if is_type_b and 'phase2_unified' in phases_data:
        emb_p2, lab_p2 = phases_data['phase2_unified']
        if lab_p2 is None: lab_p2 = y
        k = len(proto_np)
        emb_p2_sub, lab_p2_sub, idx_p2_sub = subsample_for_tsne(emb_p2, lab_p2)
        tsne_p2 = run_tsne(emb_p2_sub)
        # proto_assign도 서브샘플에 맞게 조정
        proto_assign_sub = proto_assign[idx_p2_sub] if proto_assign is not None else None

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('{} — Phase 2 Embedding (Type-B, k={})'.format(go_id, k),
                     fontsize=12, y=1.01)

        # 왼쪽: pos/neg
        ax = axes[0]
        ax.scatter(tsne_p2[lab_p2_sub==0, 0], tsne_p2[lab_p2_sub==0, 1],
                   c=NEG_COLOR, alpha=0.25, s=8, label='Neg')
        ax.scatter(tsne_p2[lab_p2_sub==1, 0], tsne_p2[lab_p2_sub==1, 1],
                   c=POS_COLOR, alpha=0.9, s=22, label='Pos', zorder=3)
        # prototypes: nearest neighbor in subsampled space
        proto_norm_f = proto_np / (np.linalg.norm(proto_np, axis=1, keepdims=True) + 1e-10)
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(proto_norm_f, emb_p2_sub)
        for pi in range(k):
            nn_idx = sim[pi].argmax()
            ax.scatter(tsne_p2[nn_idx, 0], tsne_p2[nn_idx, 1],
                       c=PROTO_COLORS[pi % len(PROTO_COLORS)],
                       s=150, marker='*', zorder=5,
                       label='Proto_{}'.format(pi), edgecolors='black', linewidths=0.5)
        ax.set_title('Pos / Neg + Prototypes', fontsize=10)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        # 오른쪽: prototype group assignment (positives only, in subsampled space)
        ax = axes[1]
        ax.scatter(tsne_p2[lab_p2_sub==0, 0], tsne_p2[lab_p2_sub==0, 1],
                   c='lightgray', alpha=0.2, s=8, label='Neg')
        if proto_assign_sub is not None:
            pos_idx_sub = np.where(lab_p2_sub == 1)[0]
            pos_assign_s = proto_assign_sub[lab_p2_sub == 1]
            for gi in range(k):
                grp = (pos_assign_s == gi)
                if grp.sum() == 0: continue
                ax.scatter(tsne_p2[pos_idx_sub[grp], 0], tsne_p2[pos_idx_sub[grp], 1],
                           c=PROTO_COLORS[gi % len(PROTO_COLORS)],
                           alpha=0.9, s=22, label='Group {} (n={})'.format(gi, int(grp.sum())),
                           zorder=3)
        else:
            sim_pos = cosine_similarity(emb_p2_sub[lab_p2_sub==1], proto_norm_f)
            assign_auto = sim_pos.argmax(axis=1)
            pos_idx_sub = np.where(lab_p2_sub == 1)[0]
            for gi in range(k):
                grp = (assign_auto == gi)
                if grp.sum() == 0: continue
                ax.scatter(tsne_p2[pos_idx_sub[grp], 0], tsne_p2[pos_idx_sub[grp], 1],
                           c=PROTO_COLORS[gi % len(PROTO_COLORS)],
                           alpha=0.9, s=22, label='Group {} (n={})'.format(gi, grp.sum()),
                           zorder=3)
        ax.set_title('Positive Prototype Groups', fontsize=10)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        out2 = os.path.join(plots_dir, '{}_v8b_proto_tsne.png'.format(go_safe))
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(os.path.basename(out2)))

    # 3) Phase 1 vs Phase 2: 같은 TSNE 공간 비교 (공유 임베딩 시 유의미)
    if 'phase1_contrastive' in phases_data and 'phase2_unified' in phases_data:
        e1, l1 = phases_data['phase1_contrastive']
        e2, l2 = phases_data['phase2_unified']
        if l1 is None: l1 = y
        if l2 is None: l2 = y
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('{} — Phase 1 vs Phase 2 Detail'.format(go_id), fontsize=12)
        for ax, (emb_i, lab_i, title) in zip(axes, [
                (e1, l1, 'Phase 1 (Contrastive)'),
                (e2, l2, 'Phase 2 (Unified)')]):
            emb_s, lab_s, _ = subsample_for_tsne(emb_i, lab_i)
            t2d = run_tsne(emb_s)
            emb_i, lab_i = emb_s, lab_s
            m   = compute_embedding_metrics(emb_i, lab_i)
            pos_m = (lab_i == 1); neg_m = (lab_i == 0)
            ax.scatter(t2d[neg_m, 0], t2d[neg_m, 1], c=NEG_COLOR, alpha=0.25, s=8)
            ax.scatter(t2d[pos_m, 0], t2d[pos_m, 1], c=POS_COLOR, alpha=0.85, s=20)
            legend_el = [Patch(color=POS_COLOR, label='Pos (n={})'.format(pos_m.sum())),
                         Patch(color=NEG_COLOR, label='Neg (n={})'.format(neg_m.sum()))]
            ax.legend(handles=legend_el, fontsize=8)
            ax.set_title('{}\ncc={:.4f}  sep={:.4f}  sil={:.4f}'.format(
                title, m['centroid_cos'], m['sep_ratio'], m['silhouette']), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        out3 = os.path.join(plots_dir, '{}_v8b_ph1_vs_ph2.png'.format(go_safe))
        plt.savefig(out3, dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(os.path.basename(out3)))

    # ── 요약 수집 ────────────────────────────────────
    row = {'GO_ID': go_id, 'n_pos': n_pos, 'n_neg': n_neg, 'is_type_b': is_type_b}
    phase_key_map = {
        'phase0_initial_untrained': 'phase0',
        'phase1_contrastive':       'phase1',
        'phase2_unified':           'phase2',
    }
    for pkey, _ in PHASE_NAMES:
        if pkey in phase_metrics:
            prefix = phase_key_map[pkey]
            for mk, mv in phase_metrics[pkey].items():
                row['{}_{}'.format(prefix, mk)] = mv
    all_summary.append(row)

# ─────────────────────────────────────────────
# 전체 요약 출력
# ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print(">>> SUMMARY: v8b Embedding Analysis")
print("=" * 70)

hdr = "{:12} {:5} {:6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
    "GO_ID", "Type", "n_pos",
    "Ph0_sepR", "Ph1_sepR", "Ph2_sepR",
    "Ph1_cc", "Ph2_cc", "Ph2_lrAU")
print(hdr)
print("-" * 85)

for r in all_summary:
    t = 'B' if r.get('is_type_b') else 'A'
    def gv(key, default=np.nan): return r.get(key, default)
    print("{:12} {:5} {:6} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}".format(
        r['GO_ID'], t, int(r['n_pos']),
        gv('phase0_sep_ratio'), gv('phase1_sep_ratio'), gv('phase2_sep_ratio'),
        gv('phase1_centroid_cos'), gv('phase2_centroid_cos'),
        gv('phase2_lr_auroc_full')))

import pandas as pd
if all_summary:
    df = pd.DataFrame(all_summary)
    out_csv = os.path.join(BASE_DIR, 'v8b_embedding_summary.csv')
    df.to_csv(out_csv, index=False)
    print("\nSaved summary: {}".format(out_csv))

print("\n[Done] v8b_embedding_analysis.py")
