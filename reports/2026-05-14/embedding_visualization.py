"""
Embedding Space Visualization — 2026-05-14
==========================================
UMAP projection of phase2 unified embeddings for Type-A and Type-B GO terms.
Highlights: positive cluster cores, representative isoforms, sub-cluster structure.

Generates:
  fig_embedding_typeA_GO0006096.png   (Glycolysis, D256)
  fig_embedding_typeA_GO0003774.png   (Motor activity, D256)
  fig_embedding_typeB_GO0007204.png   (Ca²⁺ signaling, P3-512)
  fig_embedding_typeB_GO0030017.png   (Sarcomere, P3-512)
  fig_embedding_overview.png          (4-panel overview)
  fig_cluster_core_table.png          (Representative isoforms table)
"""

import os, json, collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
import umap

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = '/home/welcome1/sw1686/DIFFUSE'
RES_DIR   = f'{ROOT}/hMuscle/results_isoform'
MODEL_DIR = f'{ROOT}/hMuscle/model'
DATA_DIR  = f'{ROOT}/hMuscle/data'
OUT_DIR   = f'{ROOT}/reports/2026-05-14'

# ─── Case definitions ─────────────────────────────────────────────────────────
CASES = [
    {
        'go': 'GO:0006096', 'go_tag': 'GO_0006096',
        'model': 'D256', 'type': 'A',
        'dir': 'v8b-D256_integrated_20260514_0353',
        'prefix': 'v8b-D256_integrated',
        'label': 'Glycolysis (GO:0006096)',
        'color_pos': '#d62728', 'color_neg': '#aec7e8',
    },
    {
        'go': 'GO:0003774', 'go_tag': 'GO_0003774',
        'model': 'D256', 'type': 'A',
        'dir': 'v8b-D256_integrated_20260514_0315',
        'prefix': 'v8b-D256_integrated',
        'label': 'Motor Activity (GO:0003774)',
        'color_pos': '#ff7f0e', 'color_neg': '#aec7e8',
    },
    {
        'go': 'GO:0007204', 'go_tag': 'GO_0007204',
        'model': 'P3-512', 'type': 'B',
        'dir': 'v8b-P3-512_integrated_20260514_1114',
        'prefix': 'v8b-P3-512_integrated',
        'label': 'Ca²⁺ Signaling (GO:0007204)',
        'color_pos': '#1f77b4', 'color_neg': '#d3d3d3',
    },
    {
        'go': 'GO:0030017', 'go_tag': 'GO_0030017',
        'model': 'P3-512', 'type': 'B',
        'dir': 'v8b-P3-512_integrated_20260514_1137',
        'prefix': 'v8b-P3-512_integrated',
        'label': 'Sarcomere (GO:0030017)',
        'color_pos': '#9467bd', 'color_neg': '#d3d3d3',
    },
]

# ─── Gene symbol map ──────────────────────────────────────────────────────────
ensg_to_sym = {}
with open(f'{DATA_DIR}/raw_data/data/id_lists/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5:
            ensg_to_sym[p[1]] = p[4]
            ensg_to_sym[p[0]] = p[4]

iso_list  = np.load(f'{MODEL_DIR}/my_isoform_list_fixed.npy', allow_pickle=True)
gene_list = np.load(f'{MODEL_DIR}/my_gene_list_fixed.npy',    allow_pickle=True)
iso_list  = [x.decode() if isinstance(x, bytes) else x for x in iso_list]
gene_list = [x.decode() if isinstance(x, bytes) else x for x in gene_list]

def gene_sym(ensg):
    return ensg_to_sym.get(ensg) or ensg_to_sym.get(ensg.split('.')[0]) or ensg

# ─── Literature-validated cluster representatives ─────────────────────────────
# Filled after literature search; used for annotation in figures
CORE_META = {
    # GO:0006096 — glycolytic enzymes; PFKM is the muscle-dominant isoform,
    # PFKP/PFKL reflect training corpus breadth (partial tissue mixing)
    'GO:0006096': [
        ('PFKP', 'Phosphofructokinase\n(Platelet isoform)',
         'Rate-limiting glycolytic enzyme;\nPlatelet/brain isoform — PFKM is\nmuscle-dominant; partial tissue mixing',
         'Platelet/brain\n(non-muscle)'),
        ('PFKL', 'Phosphofructokinase\n(Liver isoform)',
         'Hepatic PFK isoform forming L4\nhomotetramers; liver/kidney primary;\nPFKM dominant in adult muscle',
         'Liver/kidney\n(non-muscle)'),
        ('PKM', 'Pyruvate Kinase M\n(PKM2 splice isoform)',
         'PKM2 = fetal/proliferating isoform;\nPKM1 is adult skeletal muscle isoform;\nAlt-splicing exon 9↔10 switch',
         'Proliferating/fetal\n(PKM1 = muscle)'),
    ],
    # GO:0007204 — centroid dominated by non-muscle GPCRs: direct evidence
    # of Tissue Context Mixing (SwissProt Ca²⁺ annotations ≠ muscle Ca²⁺)
    'GO:0007204': [
        ('F2R', 'PAR1 (Thrombin\nReceptor / GPCR)',
         'Gαq→PLCβ→Ca²⁺; canonical in\nplatelets & vascular endothelium;\nno established muscle role',
         'Platelet/vascular ⚠\n[PMID:35025229]'),
        ('LPAR6', 'LPA Receptor 6\n(GPCR)',
         'Gα12/13 + Gαq GPCR;\nhair follicle morphogenesis;\ntissue: adipose/skin/lymphoid',
         'Lymphoid/skin ⚠\n[OMIM:609239]'),
        ('CX3CR1', 'Fractalkine\nReceptor (GPCR)',
         'Chemokine GPCR; Ca²⁺/PKC in\nmonocytes, NK, microglia;\nHPA: absent from skeletal muscle',
         'Immune/CNS ⚠\n[PMC:5833124]'),
    ],
    # GO:0003774 — mixed: KIF26A/KIF21A neural, MYO10 genuinely muscle-relevant
    'GO:0003774': [
        ('KIF26A', 'Kinesin-26A\n(Non-motor kinesin)',
         'Constitutive microtubule binder;\nlacks ATPase motility residues;\nGDNF-Ret signaling, enteric neurons',
         'Neural/renal\n[OMIM:613231]'),
        ('KIF21A', 'Kinesin-21A\n(CFEOM1 locus)',
         'Plus-end kinesin; autoinhib.\ndysregulation → CFEOM1;\nno intrinsic skeletal muscle role',
         'Oculomotor neurons\n[PMC:srep30668]'),
        ('MYO10', 'Myosin-X\n(Unconventional)',
         'Filopodia motor delivering\nMyomaker/Myomixer at myoblast\nfusion sites; DMD regeneration marker',
         'Myoblast/satellite ✓\n[PMC:8500716]'),
    ],
    # GO:0030017 — sarcomere; ACTN2 & PPP1R12B muscle-specific; ANK3 mixed
    'GO:0030017': [
        ('ANK3', 'Ankyrin-G\n(AnkG107 isoform)',
         'Dominant isoforms: axon initial\nsegment / nodes of Ranvier;\nAnkG107 (76-aa insert) = costamere',
         'CNS dominant;\nmuscle isoform: costamere\n[PMID:15953600]'),
        ('ACTN2', 'α-Actinin-2\n(Z-disc anchor)',
         'Cross-links actin & titin at Z-disc;\nexpression: cardiac + skeletal muscle\nonly; HCM/DCM mutations',
         'Cardiac/skeletal ✓\n[OMIM:102573]'),
        ('PPP1R12B', 'MYPT2\n(MLCP Reg. Subunit)',
         'Striated muscle MLCP targeting\nsubunit; dephosphorylates MyRLC;\ncardiac hypertrophy protection',
         'Striated muscle ✓\n[PMID:38224947]'),
    ],
}

# ─── Load data and run UMAP ───────────────────────────────────────────────────
print("=" * 60)
print(" Embedding Visualization")
print("=" * 60)

case_data = {}
for c in CASES:
    print(f"\n[Loading] {c['label']} ...")
    dpath = f"{RES_DIR}/{c['go_tag']}/{c['dir']}"

    emb_f   = [f for f in os.listdir(dpath) if 'phase2_unified_embeddings' in f][0]
    label_f = [f for f in os.listdir(dpath) if 'phase2_unified_labels' in f][0]
    score_f = [f for f in os.listdir(dpath) if 'phase2_unified_scores' in f][0]

    emb    = np.load(f'{dpath}/{emb_f}')
    labels = np.load(f'{dpath}/{label_f}')
    df_s   = pd.read_csv(f'{dpath}/{score_f}', sep='\t', header=None,
                         names=['GeneID', 'IsoID', 'Score'])
    scores = df_s['Score'].values

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_emb = emb[pos_idx]

    # Centroid + core detection
    centroid = pos_emb.mean(axis=0)
    dists    = np.linalg.norm(pos_emb - centroid, axis=1)
    n_core   = min(10, len(pos_idx))
    core_local  = np.argsort(dists)[:n_core]       # indices within pos_idx
    core_global = pos_idx[core_local]               # global indices in 36748

    # Sub-positive: 10–40% most central positives
    sub_local  = np.argsort(dists)[n_core:min(40, len(pos_idx))]
    sub_global = pos_idx[sub_local]

    # Gene info for core
    core_info = []
    seen_sym = set()
    for li, gi in zip(core_local, core_global):
        sym = gene_sym(gene_list[gi])
        if sym not in seen_sym:
            seen_sym.add(sym)
            core_info.append({
                'iso': iso_list[gi], 'gene': gene_list[gi],
                'sym': sym, 'score': float(scores[gi]),
                'dist': float(dists[li]),
            })

    # UMAP (PCA first for speed)
    print(f"  Running PCA 50d → UMAP 2d on {len(emb)} isoforms ...")
    rng_seed = 42
    # Subsample negatives for speed (keep all positives)
    n_neg_sample = min(5000, len(neg_idx))
    neg_sample   = np.random.RandomState(rng_seed).choice(neg_idx, n_neg_sample, replace=False)
    all_sample   = np.concatenate([pos_idx, sub_global, neg_sample])
    all_sample   = np.unique(all_sample)

    emb_sub = emb[all_sample]
    pca     = PCA(n_components=min(50, emb_sub.shape[1]), random_state=rng_seed)
    emb_pca = pca.fit_transform(emb_sub)

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric='cosine', random_state=rng_seed, verbose=False)
    emb_2d  = reducer.fit_transform(emb_pca)

    # Build lookup: sample_idx → umap coords
    umap_coords = {idx: emb_2d[i] for i, idx in enumerate(all_sample)}

    case_data[c['go']] = {
        **c,
        'emb': emb, 'labels': labels, 'scores': scores,
        'pos_idx': pos_idx, 'neg_idx': neg_idx,
        'centroid': centroid, 'dists': dists,
        'core_local': core_local, 'core_global': core_global,
        'sub_global': sub_global,
        'core_info': core_info,
        'all_sample': all_sample, 'umap_coords': umap_coords,
        'emb_2d': emb_2d,
    }
    auprc = average_precision_score(labels, scores)
    print(f"  Done — AUPRC={auprc:.4f}  pos={len(pos_idx)}  core={len(core_info)}")
    print(f"  Core genes: {[ci['sym'] for ci in core_info[:5]]}")

# ─── Helper: plot single GO term embedding ───────────────────────────────────
def plot_embedding(ax, cd, show_legend=True, title_extra=''):
    all_sample  = cd['all_sample']
    umap_coords = cd['umap_coords']
    pos_idx     = cd['pos_idx']
    sub_global  = cd['sub_global']
    core_global = cd['core_global']
    labels      = cd['labels']
    scores      = cd['scores']
    core_info   = cd['core_info']

    pos_set  = set(pos_idx.tolist())
    sub_set  = set(sub_global.tolist())
    core_set = set(core_global.tolist())

    # Collect coordinates
    neg_xy   = [umap_coords[i] for i in all_sample if i not in pos_set and i in umap_coords]
    sub_xy   = [umap_coords[i] for i in all_sample if i in sub_set and i not in core_set]
    core_xy  = [umap_coords[i] for i in all_sample if i in core_set]
    # remaining positives not in sub or core
    outer_xy = [umap_coords[i] for i in all_sample if i in pos_set and i not in sub_set and i not in core_set]

    if neg_xy:
        neg_arr = np.array(neg_xy)
        ax.scatter(neg_arr[:,0], neg_arr[:,1], s=4, c='#d3d3d3',
                   alpha=0.3, rasterized=True, label=f'Negative (n≈{len(neg_arr):,})')

    if outer_xy:
        outer_arr = np.array(outer_xy)
        ax.scatter(outer_arr[:,0], outer_arr[:,1], s=25,
                   c=cd['color_pos'], alpha=0.5, edgecolors='none',
                   label=f'Positive (peripheral)')

    if sub_xy:
        sub_arr = np.array(sub_xy)
        ax.scatter(sub_arr[:,0], sub_arr[:,1], s=45,
                   c=cd['color_pos'], alpha=0.8, edgecolors='white', linewidths=0.5,
                   label=f'Positive (inner cluster)')

    if core_xy:
        core_arr = np.array(core_xy)
        ax.scatter(core_arr[:,0], core_arr[:,1], s=120,
                   c=cd['color_pos'], edgecolors='black', linewidths=1.2,
                   zorder=6, label=f'Cluster core (top-10)')

    # Annotate top-3 core genes
    lit_meta = CORE_META.get(cd['go'], [])
    annotated = set()
    ann_count = 0
    for ci in core_info[:5]:
        gi = None
        for idx in all_sample:
            if idx in core_set and gene_sym(gene_list[idx]) == ci['sym'] and idx not in annotated:
                gi = idx
                break
        if gi is None or gi not in umap_coords:
            continue
        xy  = umap_coords[gi]
        sym = ci['sym']
        # tissue label from literature metadata
        tissue = ''
        for lm in lit_meta:
            if lm[0] == sym:
                tissue = lm[3]
                break
        label_str = f'{sym}\n({tissue})' if tissue else sym
        # warn marker for non-muscle Ca²⁺
        warn = '⚠' if cd['go'] == 'GO:0007204' and '⚠' in tissue else ''
        txt = ax.annotate(
            f'{warn}{sym}',
            xy=xy, xytext=(xy[0] + 0.8, xy[1] + 0.5),
            fontsize=7.5, fontweight='bold', color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.7),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85,
                      edgecolor=cd['color_pos'], linewidth=0.8),
            zorder=7,
        )
        annotated.add(gi)
        ann_count += 1
        if ann_count >= 3:
            break

    auprc = average_precision_score(labels, scores)
    n_pos = int(labels.sum())
    title = f"{cd['label']}\n{cd['model']} model | AUPRC={auprc:.4f} | n_pos={n_pos}{title_extra}"
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=9)
    ax.set_ylabel('UMAP-2', fontsize=9)
    ax.tick_params(labelsize=7)

    if cd['go'] == 'GO:0007204':
        # Add tissue mixing annotation
        ax.text(0.02, 0.04,
                '⚠ Cluster core: non-muscle GPCRs\n'
                '   (vascular/immune context)\n'
                '   → SP injects tissue noise',
                transform=ax.transAxes, fontsize=7.5,
                color='#1f77b4', style='italic',
                bbox=dict(boxstyle='round', facecolor='#e8f4fd', alpha=0.85, linewidth=0.5))

    if show_legend:
        ax.legend(fontsize=7, loc='upper right', markerscale=0.8,
                  framealpha=0.85, edgecolor='gray')

# ─── Figure: 4-panel overview ─────────────────────────────────────────────────
print("\n[Fig] Generating 4-panel overview ...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
ax_map = {
    'GO:0006096': axes[0, 0],
    'GO:0003774': axes[0, 1],
    'GO:0007204': axes[1, 0],
    'GO:0030017': axes[1, 1],
}

for go, cd in case_data.items():
    ax = ax_map[go]
    tp_label = f'\n[Type-{"A" if cd["type"]=="A" else "B"}: SP {"Required" if cd["type"]=="A" else "Harmful"}]'
    plot_embedding(ax, cd, show_legend=True, title_extra=tp_label)

# Type labels
for (r, c), label, color in [
    ((0,0), 'TYPE-A GO Terms\n(SP Required)', '#d62728'),
    ((1,0), 'TYPE-B GO Terms\n(SP Harmful)', '#1f77b4'),
]:
    axes[r, c].text(-0.18, 0.5, label,
                    transform=axes[r, c].transAxes,
                    rotation=90, va='center', ha='center',
                    fontsize=11, fontweight='bold', color=color)

plt.suptitle(
    'Embedding Space Distribution — Phase 2 Unified Embeddings\n'
    'UMAP projection (cosine metric) | Gene-block bootstrap validated\n'
    'Cluster core = top-10 isoforms nearest to positive centroid',
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
out_path = f'{OUT_DIR}/fig_embedding_overview.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# ─── Figure: individual GO terms (high-res) ───────────────────────────────────
for go, cd in case_data.items():
    print(f"\n[Fig] Individual: {cd['label']} ...")
    fig, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: main UMAP
    plot_embedding(axes2[0], cd, show_legend=True)

    # Right: Score distribution (pos vs neg)
    ax2 = axes2[1]
    pos_scores = cd['scores'][cd['pos_idx']]
    neg_scores = cd['scores'][cd['neg_idx']]

    # Violin-style with individual points
    parts = ax2.violinplot([neg_scores, pos_scores], positions=[0, 1],
                           showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.6)
    parts['bodies'][0].set_facecolor('#d3d3d3')
    parts['bodies'][1].set_facecolor(cd['color_pos'])
    parts['cmedians'].set_color('black')

    # Jitter overlay
    jitter_neg = np.random.RandomState(42).uniform(-0.08, 0.08, size=min(500, len(neg_scores)))
    jitter_pos = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(pos_scores))
    ax2.scatter(jitter_neg, neg_scores[:len(jitter_neg)], s=4, c='#888', alpha=0.3, rasterized=True)
    ax2.scatter(jitter_pos + 1, pos_scores, s=30, c=cd['color_pos'], alpha=0.7,
                edgecolors='black', linewidths=0.5, zorder=5)

    # Annotate core isoforms on violin
    core_scores = cd['scores'][cd['core_global']]
    seen_sym2 = set()
    for ci in cd['core_info'][:3]:
        sym = ci['sym']
        if sym in seen_sym2:
            continue
        seen_sym2.add(sym)
        sc = ci['score']
        ax2.annotate(sym, xy=(1, sc), xytext=(1.15, sc),
                     fontsize=7.5, fontweight='bold', va='center',
                     arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               alpha=0.85, edgecolor=cd['color_pos'], lw=0.7))

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f'Negative\n(n={len(neg_scores):,})',
                          f'Positive\n(n={len(pos_scores)})'], fontsize=10)
    ax2.set_ylabel('Prediction Score (Phase 2 Unified)', fontsize=10)
    auprc = average_precision_score(cd['labels'], cd['scores'])
    ax2.set_title(f'Score Distribution\nAUPRC = {auprc:.4f}', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    go_str = go.replace(':', '_')
    plt.suptitle(f"{cd['label']} | {cd['model']} model | Type-{cd['type']}",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = f'{OUT_DIR}/fig_embedding_{go_str}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# ─── Figure: Cluster core representative table ────────────────────────────────
print("\n[Fig] Cluster core representative table ...")

go_order = ['GO:0006096', 'GO:0003774', 'GO:0007204', 'GO:0030017']
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

for panel_idx, go in enumerate(go_order):
    cd       = case_data[go]
    lit_meta = CORE_META.get(go, [])
    ax       = fig.add_subplot(gs[panel_idx // 2, panel_idx % 2])
    ax.axis('off')

    col_labels = ['Gene', 'Isoform', 'd_centroid', 'Score', 'Function', 'Tissue']
    rows = []
    seen = set()
    for ci in cd['core_info']:
        if ci['sym'] in seen or len(rows) >= 5:
            continue
        seen.add(ci['sym'])
        func_str = tissue_str = '—'
        for lm in lit_meta:
            if lm[0] == ci['sym']:
                func_str   = lm[2][:45] + '…' if len(lm[2]) > 45 else lm[2]
                tissue_str = lm[3]
                break
        rows.append([
            ci['sym'],
            ci['iso'][:22],
            f"{ci['dist']:.3f}",
            f"{ci['score']:.4f}",
            func_str,
            tissue_str,
        ])

    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        loc='center', cellLoc='left',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 2.2)

    hdr_color = '#2c3e50'
    body_a    = '#f8f9fa'
    body_b    = 'white'
    warn_color = '#fff3cd'

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_facecolor(hdr_color)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor(body_a if row % 2 == 1 else body_b)
            # warn non-muscle tissue for GO:0007204
            if go == 'GO:0007204' and col == 5 and row > 0:
                cell.set_facecolor(warn_color)

    auprc = average_precision_score(cd['labels'], cd['scores'])
    type_str  = 'Type-A (SP Required)' if cd['type'] == 'A' else 'Type-B (SP Harmful)'
    warn_str  = '\n⚠ Core = non-muscle GPCRs → Tissue Context Mixing' if go == 'GO:0007204' else ''
    ax.set_title(
        f"{cd['label']} | {cd['model']} | {type_str}\n"
        f"AUPRC={auprc:.4f} | Cluster core: top-5 nearest positive isoforms{warn_str}",
        fontsize=9, fontweight='bold', pad=18,
        color='#d62728' if go == 'GO:0007204' else 'black',
    )

plt.suptitle(
    'Cluster Core Representative Isoforms per GO Term\n'
    'd_centroid = L2 distance from positive cluster centroid '
    '(smaller = more central)',
    fontsize=12, fontweight='bold', y=1.01,
)
out_path = f'{OUT_DIR}/fig_cluster_core_table.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# ─── Figure: Centroid distance distribution (core vs periphery) ───────────────
print("\n[Fig] Centroid distance violin ...")
fig, axes3 = plt.subplots(1, 4, figsize=(16, 5), sharey=False)

for ai, (go, cd) in enumerate(case_data.items()):
    ax = axes3[ai]
    pos_dists = cd['dists']
    n_core    = min(10, len(pos_dists))

    core_d  = np.sort(pos_dists)[:n_core]
    sub_d   = np.sort(pos_dists)[n_core:min(40, len(pos_dists))]
    outer_d = np.sort(pos_dists)[min(40, len(pos_dists)):]

    groups  = [core_d, sub_d, outer_d] if len(sub_d) > 0 and len(outer_d) > 0 else [core_d, outer_d]
    labels3 = (['Core\n(top-10)', 'Sub-cluster\n(11–40)', 'Peripheral']
               if len(groups) == 3 else ['Core\n(top-10)', 'Peripheral'])
    colors3 = ['#d62728', '#ff7f0e', '#aec7e8'][:len(groups)]

    pos_arr = list(range(len(groups)))
    parts3  = ax.violinplot(groups, positions=pos_arr,
                            showmedians=True, showextrema=False)
    for j, (pc, col) in enumerate(zip(parts3['bodies'], colors3)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts3['cmedians'].set_color('black')
    parts3['cmedians'].set_linewidth(1.5)

    for j, (grp, col) in enumerate(zip(groups, colors3)):
        jitter = np.random.RandomState(42).uniform(-0.07, 0.07, size=len(grp))
        ax.scatter(j + jitter, grp, s=15, c=col, alpha=0.6, edgecolors='none', zorder=5)

    ax.set_xticks(pos_arr)
    ax.set_xticklabels(labels3, fontsize=8)
    ax.set_ylabel('L2 distance from centroid', fontsize=9)
    ax.set_title(f"{cd['label']}\n[Type-{cd['type']}]", fontsize=9, fontweight='bold',
                 color='#d62728' if cd['type'] == 'A' else '#1f77b4')
    ax.grid(axis='y', alpha=0.3)

    # Annotate gap
    if len(groups) >= 2:
        gap = groups[1].mean() - core_d.mean() if len(groups) > 1 else 0
        ax.text(0.5, 0.93, f'Core-Sub gap: {gap:.3f}',
                transform=ax.transAxes, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(
    'Positive Isoform Cluster Structure: Core / Sub-cluster / Peripheral\n'
    'Smaller centroid distance = more central (representative) in embedding space',
    fontsize=11, fontweight='bold', y=1.01,
)
plt.tight_layout()
out_path = f'{OUT_DIR}/fig_cluster_structure.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# ─── Save core isoform JSON ───────────────────────────────────────────────────
core_export = {}
for go, cd in case_data.items():
    lit_meta = CORE_META.get(go, [])
    entries  = []
    seen     = set()
    for ci in cd['core_info']:
        if ci['sym'] in seen:
            continue
        seen.add(ci['sym'])
        entry = dict(ci)
        for lm in lit_meta:
            if lm[0] == ci['sym']:
                entry['lit_name']   = lm[1].replace('\n', ' ')
                entry['lit_func']   = lm[2].replace('\n', ' ')
                entry['lit_tissue'] = lm[3].replace('\n', ' ')
                break
        entries.append(entry)
    core_export[go] = {
        'model': cd['model'], 'type': cd['type'],
        'auprc': float(average_precision_score(cd['labels'], cd['scores'])),
        'n_positives': int((cd['labels'] == 1).sum()),
        'core': entries,
    }

json_path = f'{OUT_DIR}/cluster_core_isoforms.json'
with open(json_path, 'w') as f:
    json.dump(core_export, f, indent=2, ensure_ascii=False)
print(f"\n[Saved] {json_path}")

print("\n" + "=" * 60)
print(" All embedding figures generated")
print("=" * 60)
files = [
    'fig_embedding_overview.png',
    'fig_embedding_GO_0006096.png',
    'fig_embedding_GO_0003774.png',
    'fig_embedding_GO_0007204.png',
    'fig_embedding_GO_0030017.png',
    'fig_cluster_core_table.png',
    'fig_cluster_structure.png',
    'cluster_core_isoforms.json',
]
for fname in files:
    p = f'{OUT_DIR}/{fname}'
    if os.path.exists(p):
        print(f"  {fname}  ({os.path.getsize(p)//1024} KB)")
