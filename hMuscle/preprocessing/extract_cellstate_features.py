#!/usr/bin/env python3
"""
extract_cellstate_features.py
==============================
브레인 long-read scRNA-seq AnnData에서 두 종류의 cellular context 특징 추출:

  1. Cell-type expression vector (Layer 3 — 세포 유형 정체성)
     각 transcript의 8개 세포 유형별 평균 정규화 발현량 벡터
     출력: cell_type_expression_vectors.npy  (63994 × 8)

  2. Cell-state association vector (Layer 4 aggregated — 세포 상태)
     9개 세포 상태의 marker gene 점수와 각 transcript 발현의 상관관계
     출력: cell_state_association_vectors.npy  (63994 × 9)

두 특징 모두 각 isoform의 절대적(absolute) 특징 — gene-level relative 특징이 아님
→ v16 gate collapse와 동일한 gradient 충돌 없음

사용법:
  cd /home/welcome1/sw1686/DIFFUSE/hMuscle/preprocessing
  conda activate isoform_env
  python extract_cellstate_features.py
"""

import os, warnings, json
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from collections import defaultdict

warnings.filterwarnings('ignore')

ADATA_PATH = '/home/dhkim1674/Project_AD_with_refTSS_novel/03_AnnData/adata_transcript_loose_filtering_for_bulk_analysis.h5ad'
OUT_DIR    = '/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 세포 상태 마커 정의 (Layer 4 → aggregated static) ─────────────────────────
# 각 상태는 문헌에서 검증된 marker gene 집합으로 정의
CELL_STATE_MARKERS = {
    'neuronal_activity': [      # 활동전위 의존적 즉각 반응 유전자 (IEGs)
        'ARC', 'FOS', 'FOSB', 'NPAS4', 'EGR1', 'NR4A1', 'JUNB', 'BDNF',
    ],
    'DAM_microglia': [          # Disease-associated microglia (Keren-Shaul 2017)
        'TREM2', 'TYROBP', 'AXL', 'SPP1', 'LPL', 'CST7', 'CD9', 'LGALS3BP',
    ],
    'homeostatic_microglia': [  # 정상 상주 마이크로글리아 (Butovsky 2014)
        'P2RY12', 'TMEM119', 'CX3CR1', 'SIGLECH', 'SLC2A5',
    ],
    'reactive_astrocyte': [     # A1/A2 반응성 아스트로사이트 (Liddelow 2017)
        'GFAP', 'VIM', 'LCN2', 'SERPINA3', 'GBP2',
    ],
    'oxphos_active': [          # 산화적 인산화 / 미토콘드리아 활성 상태
        'MT-CO1', 'MT-CO2', 'MT-CO3', 'MT-ND1', 'UQCRC1', 'ATP5F1A', 'VDAC1',
    ],
    'cell_cycle_active': [      # 세포 분열 활성 상태 (Ki67+)
        'MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1', 'AURKB',
    ],
    'er_stress': [              # ER 스트레스 / UPR 활성
        'HSPA5', 'DDIT3', 'ATF4', 'XBP1', 'PPP1R15A',
    ],
    'mature_oligodendrocyte': [ # 성숙 올리고덴드로사이트 (MBP+) vs OPC
        'MBP', 'MOG', 'MAG', 'PLP1',
    ],
    'wnt_active': [             # Wnt 신호 활성 상태 (β-catenin target genes)
        'AXIN2', 'LEF1', 'MYC', 'WNT5A', 'DKK1',
    ],
}

CELL_TYPES = ['Astrocyte', 'Excitatory neuron', 'Inhibitory neuron',
              'Lymphocyte', 'Microglia', 'OPC', 'Oligodendrocyte', 'Vascular cell']
N_CELLTYPES = len(CELL_TYPES)
N_STATES    = len(CELL_STATE_MARKERS)
STATE_NAMES = list(CELL_STATE_MARKERS.keys())

print("=" * 70)
print("  Cellular Context Feature Extraction")
print("  Layer 3 (cell type) + Layer 4 aggregated (cell state)")
print("=" * 70)

# ─── 1. Load AnnData via h5py ──────────────────────────────────────────────────
import h5py
print("\n[1] Loading AnnData...")
with h5py.File(ADATA_PATH, 'r') as f:
    # obs metadata
    ct_cats  = [c.decode() if isinstance(c, bytes) else c
                for c in f['obs']['cell_type']['categories'][:]]
    ct_codes = f['obs']['cell_type']['codes'][:].astype(int)   # (n_cells,)
    n_cells  = len(ct_codes)

    # var metadata
    enst_ids    = [v.decode() if isinstance(v, bytes) else v
                   for v in f['var']['ENST_ID'][:]]
    gene_names  = []
    gn_cats     = [g.decode() if isinstance(g, bytes) else g
                   for g in f['var']['gene_name']['categories'][:]]
    gn_codes    = f['var']['gene_name']['codes'][:].astype(int)
    gene_names  = [gn_cats[c] for c in gn_codes]
    n_transcripts = len(enst_ids)

    # sparse X matrix (cells × transcripts, CSR)
    data    = f['X']['data'][:]
    indices = f['X']['indices'][:]
    indptr  = f['X']['indptr'][:]

X = sp.csr_matrix((data, indices, indptr),
                  shape=(n_cells, n_transcripts), dtype=np.float32)

print(f"  Cells: {n_cells:,}  |  Transcripts: {n_transcripts:,}")
print(f"  Cell types: {ct_cats}")
print(f"  Sparsity: {X.nnz/(n_cells*n_transcripts)*100:.3f}% non-zero")

# ─── 2. Normalize per cell (log1p CPM) ────────────────────────────────────────
print("\n[2] Normalizing (log1p CPM)...")
cell_totals = np.asarray(X.sum(axis=1)).squeeze()
cell_totals[cell_totals == 0] = 1
scale = 1e4 / cell_totals           # CPM scaling factor per cell
# Scale rows of sparse matrix
X_norm = X.copy()
X_norm = X_norm.astype(np.float32)
# Efficient row-wise scaling using diagonal multiplication
from scipy.sparse import diags
X_norm = diags(scale.astype(np.float32)) @ X_norm
# log1p
X_norm.data = np.log1p(X_norm.data)
print(f"  Done. Mean non-zero value: {X_norm.data.mean():.3f}")

# ─── 3. Cell-type expression vectors (Layer 3) ─────────────────────────────────
print("\n[3] Computing cell-type expression vectors...")
celltype_vecs = np.zeros((n_transcripts, N_CELLTYPES), dtype=np.float32)
ct_counts = np.zeros(N_CELLTYPES, dtype=int)

for ct_idx, ct_name in enumerate(ct_cats):
    if ct_name not in CELL_TYPES:
        continue
    col_idx = CELL_TYPES.index(ct_name)
    mask = (ct_codes == ct_idx)
    ct_counts[col_idx] = mask.sum()
    if mask.sum() == 0:
        continue
    X_ct = X_norm[mask]
    # mean expression per transcript in this cell type
    celltype_vecs[:, col_idx] = np.asarray(X_ct.mean(axis=0)).squeeze()
    print(f"    {ct_name}: {mask.sum():,} cells")

print(f"  Cell-type expression matrix: {celltype_vecs.shape}")

# ─── 4. Gene-level aggregation for state markers ──────────────────────────────
print("\n[4] Building gene → transcript index for marker computation...")
gene2transcripts = defaultdict(list)
for t_idx, gname in enumerate(gene_names):
    gene2transcripts[gname].append(t_idx)

def get_gene_expression(gene, X_mat):
    """Return per-cell mean expression of all transcripts of a gene."""
    t_idxs = gene2transcripts.get(gene, [])
    if not t_idxs:
        return None
    return np.asarray(X_mat[:, t_idxs].mean(axis=1)).squeeze()  # (n_cells,)

# ─── 5. Cell-state association vectors (Layer 4 aggregated) ──────────────────
print("\n[5] Computing cell-state association vectors...")
print("    Strategy: for each state, compute cell-level score → correlate with isoform expression")

state_scores = np.zeros((n_cells, N_STATES), dtype=np.float32)
state_coverage = {}

for s_idx, (state_name, markers) in enumerate(CELL_STATE_MARKERS.items()):
    found_markers = []
    score = np.zeros(n_cells, dtype=np.float32)
    for marker in markers:
        expr = get_gene_expression(marker, X_norm)
        if expr is not None:
            score += expr
            found_markers.append(marker)
    if found_markers:
        score /= len(found_markers)
    state_scores[:, s_idx] = score
    state_coverage[state_name] = f"{len(found_markers)}/{len(markers)}"
    print(f"    {state_name}: {len(found_markers)}/{len(markers)} markers found")

print(f"  State score matrix: {state_scores.shape}")

# ─── 6. Compute isoform-state association (fold-enrichment, not Spearman) ─────
# For efficiency, use fold-enrichment in top-25% vs bottom-25% state cells
# This is faster than full Spearman on 95k cells and avoids OOM
print("\n[6] Computing isoform-state associations (fold-enrichment in high vs low state cells)...")

state_assoc = np.zeros((n_transcripts, N_STATES), dtype=np.float32)

# Convert X_norm to CSC for efficient column (transcript) access
print("    Converting to CSC for column access...")
X_norm_csc = X_norm.tocsc()

for s_idx, state_name in enumerate(STATE_NAMES):
    score = state_scores[:, s_idx]
    if score.max() == 0:
        print(f"    {state_name}: no marker expression found, skipping")
        continue

    # Top and bottom 25% cells by state score
    q75 = np.percentile(score[score > 0], 75) if (score > 0).sum() > 100 else score.max()
    q25 = np.percentile(score[score > 0], 25) if (score > 0).sum() > 100 else 0
    high_mask = score >= q75
    low_mask  = score <= q25
    n_high = high_mask.sum()
    n_low  = low_mask.sum()

    if n_high < 50 or n_low < 50:
        print(f"    {state_name}: insufficient cells (high={n_high}, low={n_low}), skipping")
        continue

    # Mean expression in high vs low state cells per transcript
    X_high = X_norm_csc[high_mask, :]
    X_low  = X_norm_csc[low_mask, :]
    mean_high = np.asarray(X_high.mean(axis=0)).squeeze()
    mean_low  = np.asarray(X_low.mean(axis=0)).squeeze()

    # Log fold enrichment: log2(mean_high / mean_low), clipped
    denominator = mean_low + 1e-6
    lfc = np.log2((mean_high + 1e-6) / denominator)
    state_assoc[:, s_idx] = np.clip(lfc, -5, 5).astype(np.float32)
    print(f"    {state_name}: high={n_high:,} / low={n_low:,} cells "
          f"| mean |LFC|={np.abs(lfc).mean():.3f}")

print(f"  State association matrix: {state_assoc.shape}")

# ─── 7. Save outputs ──────────────────────────────────────────────────────────
print("\n[7] Saving outputs...")

np.save(f'{OUT_DIR}/cell_type_expression_vectors.npy', celltype_vecs)
np.save(f'{OUT_DIR}/cell_state_association_vectors.npy', state_assoc)

# Save metadata JSON
meta = {
    'n_transcripts': n_transcripts,
    'cell_types': CELL_TYPES,
    'cell_type_counts': {CELL_TYPES[i]: int(ct_counts[i]) for i in range(N_CELLTYPES)},
    'state_names': STATE_NAMES,
    'state_marker_coverage': state_coverage,
    'cell_type_vec_shape': list(celltype_vecs.shape),
    'state_assoc_shape': list(state_assoc.shape),
    'enst_ids_sample': enst_ids[:5],
    'description': {
        'cell_type_expression_vectors': 'Layer 3: per-transcript mean log1p-CPM per cell type (63994 x 8)',
        'cell_state_association_vectors': 'Layer 4 aggregated: log2 fold-enrichment in high vs low state cells (63994 x 9)',
        'usage': 'absolute per-isoform features, no gene-level relative computation -> no gradient conflict'
    }
}
with open(f'{OUT_DIR}/cellular_context_meta.json', 'w') as fh:
    import json; json.dump(meta, fh, indent=2)

# Save transcript-to-isoform index file for downstream alignment
import pandas as pd
id_df = pd.DataFrame({
    'enst_id': enst_ids,
    'gene_name': gene_names,
})
id_df.to_csv(f'{OUT_DIR}/brain_transcript_index.tsv', sep='\t', index=False)

print(f"\n  Saved:")
print(f"    {OUT_DIR}/cell_type_expression_vectors.npy    {celltype_vecs.shape}")
print(f"    {OUT_DIR}/cell_state_association_vectors.npy  {state_assoc.shape}")
print(f"    {OUT_DIR}/cellular_context_meta.json")
print(f"    {OUT_DIR}/brain_transcript_index.tsv")

# ─── 8. Summary statistics ───────────────────────────────────────────────────
print("\n[8] Summary statistics:")
print(f"\n  Cell-type expression (Layer 3):")
for i, ct in enumerate(CELL_TYPES):
    nz = (celltype_vecs[:, i] > 0).sum()
    print(f"    {ct:25s}: {nz:6,}/{n_transcripts} transcripts expressed "
          f"(mean={celltype_vecs[:,i][celltype_vecs[:,i]>0].mean():.3f})")

print(f"\n  Cell-state associations (Layer 4 aggregated):")
for i, s in enumerate(STATE_NAMES):
    high = (state_assoc[:, i] > 0.5).sum()
    low  = (state_assoc[:, i] < -0.5).sum()
    print(f"    {s:30s}: {high:5,} enriched / {low:5,} depleted isoforms (|LFC|>0.5)")

print(f"\n{'='*70}  DONE  {'='*70}")
