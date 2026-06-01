"""
Diagnostic B: Embedding Space Analysis
GO:0007204 (Type-B, Ca2+ signaling) vs GO:0003774 (Type-A, motor activity)

질문: Linear AUROC=0.999인데 AUPRC=0.122인 이유는 무엇인가?
     Positive들이 embedding space에서 어떻게 분포하는가?
     가능성 1: outlier positive가 negative zone에 있음 (embedding 불완전)
     가능성 2: positive들이 여러 섬으로 분리 (multi-cluster, sigmoid 한계)
     가능성 3: annotation noise / positive set 내부 이질성

분석 대상:
- Phase 0, 1, 2 embeddings (각 GO term별)
- GO:0007204 (Type-B, AUPRC=0.122) vs GO:0003774 (Type-A, AUPRC=0.420)
- GO:0006941 (Type-B, AUPRC=0.085) 비교

출력:
- embedding_analysis_GO_XXXXXX.png: t-SNE 시각화 (Phase 0/1/2)
- 정량 지표: silhouette, intra-positive dist, inter-cluster variance
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─── 설정 ────────────────────────────────────────────────────────────────────

RESULT_ROOT = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform'
SAVE_DIR    = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/diagnostics'
os.makedirs(SAVE_DIR, exist_ok=True)

GO_CONFIGS = {
    'GO_0007204': {
        'dir'  : 'GO_0007204/v7c_integrated_20260429_0706',
        'tag'  : 'v7c_integrated',
        'type' : 'Type-B',
        'auprc': 0.122,
        'has_proto': True,
    },
    'GO_0006941': {
        'dir'  : 'GO_0006941/v7c_integrated_20260429_0651',
        'tag'  : 'v7c_integrated',
        'type' : 'Type-B',
        'auprc': 0.085,
        'has_proto': False,
    },
    'GO_0003774': {
        'dir'  : 'GO_0003774/v7c_integrated_20260429_0651',
        'tag'  : 'v7c_integrated',
        'type' : 'Type-A',
        'auprc': 0.420,
        'has_proto': False,
    },
}

# ─── 유틸 ────────────────────────────────────────────────────────────────────

def load_embeddings(go_dir, tag):
    base = os.path.join(RESULT_ROOT, go_dir)
    out = {}
    for phase in ['phase0_initial_untrained', 'phase1_contrastive', 'phase2_cnn_focal']:
        ep = os.path.join(base, f'{tag}_{phase}_embeddings.npy')
        lp = os.path.join(base, f'{tag}_{phase}_labels.npy')
        if os.path.exists(ep):
            out[phase] = {'emb': np.load(ep), 'lab': np.load(lp)}
    # prototype
    pp = os.path.join(base, f'{tag}_GO_*_prototypes.npy')
    import glob
    proto_files = glob.glob(pp)
    if proto_files:
        out['prototypes'] = np.load(proto_files[0])
    assn_files = glob.glob(os.path.join(base, f'{tag}_GO_*_proto_assignments.npy'))
    if assn_files:
        out['assignments'] = np.load(assn_files[0])
    return out


def cosine_sim_matrix(a, b):
    """a: (N, D), b: (M, D) → (N, M)"""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def compute_metrics(emb, lab):
    pos_emb = emb[lab == 1]
    neg_emb = emb[lab == 0]
    n_pos = len(pos_emb)

    if n_pos < 2:
        return {}

    # 1. Centroid distance
    pos_cent = pos_emb.mean(axis=0)
    neg_cent = neg_emb.mean(axis=0)
    pos_cent_n = pos_cent / (np.linalg.norm(pos_cent) + 1e-8)
    neg_cent_n = neg_cent / (np.linalg.norm(neg_cent) + 1e-8)
    centroid_cos = float(pos_cent_n @ neg_cent_n)

    # 2. Intra-positive cosine distance (are positives spread out?)
    if n_pos <= 200:
        sim_pp = cosine_sim_matrix(pos_emb, pos_emb)
        np.fill_diagonal(sim_pp, np.nan)
        intra_pos_sim = float(np.nanmean(sim_pp))
    else:
        idx = np.random.choice(n_pos, 200, replace=False)
        sim_pp = cosine_sim_matrix(pos_emb[idx], pos_emb[idx])
        np.fill_diagonal(sim_pp, np.nan)
        intra_pos_sim = float(np.nanmean(sim_pp))

    # 3. Nearest negative distance for each positive
    #    (positive가 negative zone에 있는지 체크)
    neg_sample = neg_emb[np.random.choice(len(neg_emb), min(2000, len(neg_emb)), replace=False)]
    sim_pn = cosine_sim_matrix(pos_emb, neg_sample)
    nearest_neg_sim = sim_pn.max(axis=1)  # each positive → closest negative
    frac_pos_in_neg = float((nearest_neg_sim > 0.9).mean())  # threshold: very close to a negative

    # 4. Silhouette-like: mean(nearest_neg - nearest_pos) per sample
    if n_pos <= 500:
        sim_pp2 = cosine_sim_matrix(pos_emb, pos_emb)
        np.fill_diagonal(sim_pp2, -np.inf)
        nearest_pos_sim = sim_pp2.max(axis=1)
        margin_per_pos = nearest_pos_sim - nearest_neg_sim
        mean_margin = float(margin_per_pos.mean())
        frac_margin_pos = float((margin_per_pos > 0).mean())
    else:
        mean_margin = float('nan')
        frac_margin_pos = float('nan')

    # 5. Linear separability (AUROC via logistic regression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    emb_sc = StandardScaler().fit_transform(emb)
    try:
        lr = LogisticRegression(max_iter=300, C=1.0, class_weight='balanced', random_state=42)
        lr.fit(emb_sc, lab.astype(int))
        scores = lr.predict_proba(emb_sc)[:, 1]
        lin_auroc = float(roc_auc_score(lab, scores))
        lin_auprc = float(average_precision_score(lab, scores))
    except Exception:
        lin_auroc = float('nan')
        lin_auprc = float('nan')

    # 6. Positive sub-cluster analysis (k-means k=2..6)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_sil = -1
    best_k = 1
    sil_scores = {}
    for k in range(2, min(7, n_pos)):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_km = km.fit_predict(pos_emb)
            if len(np.unique(labels_km)) > 1:
                s = silhouette_score(pos_emb, labels_km)
                sil_scores[k] = float(s)
                if s > best_sil:
                    best_sil = s
                    best_k = k
        except Exception:
            pass

    return {
        'n_pos': n_pos,
        'n_neg': len(neg_emb),
        'centroid_cos': centroid_cos,
        'intra_pos_sim': intra_pos_sim,
        'frac_pos_near_neg': frac_pos_in_neg,
        'mean_pos_margin': mean_margin,
        'frac_margin_pos': frac_margin_pos,
        'lin_auroc': lin_auroc,
        'lin_auprc': lin_auprc,
        'best_sub_k': best_k,
        'best_sub_sil': best_sil,
        'sil_by_k': sil_scores,
    }


def analyze_prototype_assignments(emb, lab, assignments, prototypes):
    """Positive들이 prototype별로 어떻게 분배되는가?"""
    pos_mask = lab == 1
    pos_assignments = assignments[pos_mask]
    neg_assignments = assignments[~pos_mask]

    k = len(prototypes)
    print(f"  [Proto] k={k} prototypes")
    print(f"  [Proto] Positive assignment distribution:")
    for ki in range(k):
        n_pos_k = (pos_assignments == ki).sum()
        n_neg_k = (neg_assignments == ki).sum()
        pct_pos = n_pos_k / pos_mask.sum() * 100
        pct_neg = n_neg_k / (~pos_mask).sum() * 100
        print(f"    proto{ki}: pos={n_pos_k} ({pct_pos:.1f}%) neg={n_neg_k} ({pct_neg:.1f}%)")

    # proto간 거리
    proto_sims = cosine_sim_matrix(prototypes, prototypes)
    np.fill_diagonal(proto_sims, np.nan)
    print(f"  [Proto] Inter-prototype similarity: mean={np.nanmean(proto_sims):.3f} min={np.nanmin(proto_sims):.3f}")

    # 각 prototype에서 positive가 몇 개나 negative보다 가까운가?
    for ki in range(k):
        proto = prototypes[ki:ki+1]
        sim_pos = cosine_sim_matrix(emb[pos_mask], proto).flatten()
        sim_neg = cosine_sim_matrix(emb[~pos_mask], proto).flatten()
        threshold = np.percentile(sim_neg, 95)
        n_pos_above = (sim_pos > threshold).sum()
        print(f"    proto{ki}: pos above neg-P95 threshold = {n_pos_above}/{pos_mask.sum()}")


def run_tsne(emb, lab, title, ax, assignments=None):
    """t-SNE 시각화"""
    from sklearn.manifold import TSNE
    import matplotlib.patches as mpatches

    n_total = len(emb)
    # 전체가 많으면 negative subsample
    n_neg = (lab == 0).sum()
    n_pos = (lab == 1).sum()

    if n_neg > 3000:
        neg_idx = np.where(lab == 0)[0]
        neg_sub = np.random.choice(neg_idx, 3000, replace=False)
        pos_idx = np.where(lab == 1)[0]
        sub_idx = np.concatenate([pos_idx, neg_sub])
        sub_emb = emb[sub_idx]
        sub_lab = lab[sub_idx]
        sub_assn = assignments[sub_idx] if assignments is not None else None
    else:
        sub_emb = emb
        sub_lab = lab
        sub_assn = assignments

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    z = tsne.fit_transform(sub_emb)

    neg_mask = sub_lab == 0
    pos_mask = sub_lab == 1

    ax.scatter(z[neg_mask, 0], z[neg_mask, 1], c='#cccccc', s=3, alpha=0.3, label='neg')

    if sub_assn is not None and pos_mask.sum() > 0:
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        pos_z = z[pos_mask]
        pos_a = sub_assn[pos_mask]
        for ki in range(int(pos_a.max()) + 1):
            mask_k = pos_a == ki
            ax.scatter(pos_z[mask_k, 0], pos_z[mask_k, 1],
                      c=colors[ki % len(colors)], s=40, alpha=0.9,
                      label=f'pos proto{ki} (n={mask_k.sum()})')
    else:
        ax.scatter(z[pos_mask, 0], z[pos_mask, 1],
                  c='#e41a1c', s=40, alpha=0.9, label=f'pos (n={pos_mask.sum()})')

    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])


# ─── 메인 ────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("Diagnostic B: Embedding Space Analysis")
print("=" * 70)

results_all = {}

for go_id, cfg in GO_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"GO term: {go_id} ({cfg['type']}, v7c AUPRC={cfg['auprc']})")
    print('='*60)

    data = load_embeddings(cfg['dir'], cfg['tag'])
    phases = ['phase0_initial_untrained', 'phase1_contrastive', 'phase2_cnn_focal']
    phase_labels = ['Phase 0 (untrained)', 'Phase 1 (contrastive)', 'Phase 2 (focal)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{go_id} {cfg['type']} — v7c Embedding Space\n"
                 f"Final AUPRC={cfg['auprc']}", fontsize=11)

    go_results = {}
    assignments_phase1 = data.get('assignments', None)

    for pi, (phase, plabel) in enumerate(zip(phases, phase_labels)):
        if phase not in data:
            print(f"  [{phase}] NOT FOUND — skipping")
            continue

        emb = data[phase]['emb']
        lab = data[phase]['lab']

        print(f"\n[{plabel}] n={len(emb)}, pos={int((lab==1).sum())}, neg={int((lab==0).sum())}")

        metrics = compute_metrics(emb, lab)
        go_results[phase] = metrics

        print(f"  Linear AUROC = {metrics.get('lin_auroc', float('nan')):.4f} | "
              f"AUPRC = {metrics.get('lin_auprc', float('nan')):.4f}")
        print(f"  Centroid cos sim = {metrics.get('centroid_cos', float('nan')):.4f} "
              f"(lower = better separation)")
        print(f"  Intra-positive sim = {metrics.get('intra_pos_sim', float('nan')):.4f} "
              f"(higher = more coherent positives)")
        print(f"  Frac pos near neg (cos>0.9) = {metrics.get('frac_pos_near_neg', float('nan')):.3f}")
        print(f"  Frac positives with margin>0 = {metrics.get('frac_margin_pos', float('nan')):.3f}")
        print(f"  Best sub-cluster k = {metrics.get('best_sub_k', '?')} "
              f"(sil={metrics.get('best_sub_sil', float('nan')):.3f})")
        print(f"  Silhouette by k: {metrics.get('sil_by_k', {})}")

        # t-SNE
        assn = assignments_phase1 if phase == 'phase1_contrastive' else None
        try:
            run_tsne(emb, lab, f"{plabel}\nAUROC={metrics.get('lin_auroc',0):.3f} AUPRC={metrics.get('lin_auprc',0):.3f}",
                     axes[pi], assignments=assn)
        except Exception as e:
            print(f"  [t-SNE] FAILED: {e}")
            axes[pi].set_title(f"{plabel}\n[t-SNE failed]")

    # Prototype analysis
    if 'prototypes' in data and 'assignments' in data:
        print(f"\n[Prototype Analysis]")
        ph1_data = data.get('phase1_contrastive', None)
        if ph1_data:
            analyze_prototype_assignments(
                ph1_data['emb'], ph1_data['lab'],
                data['assignments'], data['prototypes']
            )

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'embedding_analysis_{go_id}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  [Saved] {save_path}")
    results_all[go_id] = go_results


# ─── 요약 테이블 ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY TABLE — Phase 1 metrics")
print("="*70)
print(f"{'GO term':<14} {'Type':<8} {'AUPRC':<8} {'LinAUROC':<10} {'LinAUPRC':<10} "
      f"{'IntraPosSim':<13} {'FracPosNearNeg':<16} {'BestSubK':<10} {'SubSil':<8}")
print("-"*100)

for go_id, cfg in GO_CONFIGS.items():
    ph1 = results_all.get(go_id, {}).get('phase1_contrastive', {})
    print(f"{go_id:<14} {cfg['type']:<8} {cfg['auprc']:<8.3f} "
          f"{ph1.get('lin_auroc', float('nan')):<10.4f} "
          f"{ph1.get('lin_auprc', float('nan')):<10.4f} "
          f"{ph1.get('intra_pos_sim', float('nan')):<13.4f} "
          f"{ph1.get('frac_pos_near_neg', float('nan')):<16.4f} "
          f"{ph1.get('best_sub_k', '?'):<10} "
          f"{ph1.get('best_sub_sil', float('nan')):<8.4f}")

print("\n[Key Question] GO:0007204:")
ph1_0007204 = results_all.get('GO_0007204', {}).get('phase1_contrastive', {})
frac_near = ph1_0007204.get('frac_pos_near_neg', float('nan'))
intra_sim = ph1_0007204.get('intra_pos_sim', float('nan'))
best_k    = ph1_0007204.get('best_sub_k', '?')

print(f"  frac_pos_near_neg = {frac_near:.3f}")
print(f"  → If > 0.3: Possibility 1 (outliers in negative zone) is likely")
print(f"  intra_pos_sim = {intra_sim:.4f}")
print(f"  → If < 0.2: Positives are highly dispersed (Possibility 2: multi-cluster)")
print(f"  best_sub_k = {best_k}")
print(f"  → If > 2 with high silhouette: multi-cluster structure confirmed")

print("\nDone. See diagnostics/ folder for t-SNE plots.")
