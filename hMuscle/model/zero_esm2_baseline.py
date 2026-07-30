# -*- coding: utf-8 -*-
"""
zero_esm2_baseline.py
======================
v15d 기존 score matrix로 zero-ESM2 subset AUPRC 측정.

질문: "ESM-2가 within-gene isoform을 구분하지 못하는 1,741개 유전자에서
      v15d의 현재 성능은 얼마나 나쁜가?"

이것이 v15d_splice가 개선해야 하는 baseline 수치.

실행:
  cd hMuscle/model
  conda activate isoform_env
  python zero_esm2_baseline.py
"""

import os, json
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 경로 ────────────────────────────────────────────────────────────────────
SCORE_MATRIX_PATH = '../../reports/v15_bp_clean/score_matrix_18go_20260519_1914.npy'
ESM2_PATH         = '../data/esm2_embeddings_t30_150M.npy'
ISO_LIST_PATH     = 'my_isoform_list_fixed.npy'
GENE_LIST_PATH    = 'my_gene_list_fixed.npy'
ANNOT_FILE        = '../data/raw_data/data/annotations/human_annotations_unified_bp.txt'
ID_DIR            = '../data/raw_data/data/id_lists'
SD_PATH           = '../results_isoform/features/splicing/splicing_delta_v2.npy'
OUT_JSON          = '../../reports/zero_esm2_baseline_v15d.json'

GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
GO_KEYS = list(GO_TERMS.keys())


def load_ids(p):
    arr = np.load(p, allow_pickle=True)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


print("=" * 65)
print("  Zero-ESM2 Subset Baseline — v15d")
print("=" * 65)

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
print("\n[1] Loading data...")
score_matrix = np.load(SCORE_MATRIX_PATH).astype(np.float32)
X_te         = np.load(ESM2_PATH).astype(np.float32)
te_iso       = load_ids(ISO_LIST_PATH)
te_gene      = load_ids(GENE_LIST_PATH)
sd           = np.load(SD_PATH).astype(np.float32)

print(f"  score_matrix: {score_matrix.shape}")
print(f"  ESM-2:        {X_te.shape}")
print(f"  splice_delta: {sd.shape}")
print(f"  n_isoforms:   {len(te_iso)}")

# ── 2. gene → isoform 인덱스 맵 ──────────────────────────────────────────
g2idx = defaultdict(list)
for i, g in enumerate(te_gene):
    g2idx[g].append(i)

multi_genes = {g: idxs for g, idxs in g2idx.items() if len(idxs) >= 2}
print(f"\n[2] Multi-isoform genes: {len(multi_genes)}")

# ── 3. Zero-ESM2 gene 분류 ────────────────────────────────────────────────
print("\n[3] Classifying zero-ESM2 genes...")
from sklearn.metrics.pairwise import cosine_similarity

zero_esm2_diff_splice   = []   # DIFF_SPLICE_SAME_PROT (핵심 관심 그룹)
zero_esm2_pure_dup      = []   # PURE_DUPLICATE
zero_esm2_identical_all = []   # IDENTICAL_ALL
nonzero_esm2_genes      = []   # ESM-2 정상 구분

for g, idxs in multi_genes.items():
    embs = X_te[idxs]   # (n_iso, 640)
    deltas = sd[idxs]   # (n_iso, 150)

    # pairwise cosine max
    if len(idxs) == 1:
        continue
    sims = cosine_similarity(embs)
    np.fill_diagonal(sims, 0)
    max_dist = 1 - sims.max()   # max cosine distance

    iso_ids_unique = len(set(te_iso[i] for i in idxs))

    if max_dist < 0.001:  # zero ESM-2 separation
        delta_nnz = (np.abs(deltas).sum(axis=1) > 0)
        any_delta_diff = (delta_nnz.sum() > 0 and
                          not np.allclose(deltas, deltas[0]))

        if iso_ids_unique < len(idxs):
            zero_esm2_pure_dup.append(g)
        elif any_delta_diff:
            zero_esm2_diff_splice.append(g)
        else:
            zero_esm2_identical_all.append(g)
    else:
        nonzero_esm2_genes.append(g)

print(f"  DIFF_SPLICE_SAME_PROT (관심): {len(zero_esm2_diff_splice)}")
print(f"  PURE_DUPLICATE (제외):         {len(zero_esm2_pure_dup)}")
print(f"  IDENTICAL_ALL  (제외):         {len(zero_esm2_identical_all)}")
print(f"  Normal ESM-2:                  {len(nonzero_esm2_genes)}")

# ── 4. 서브셋 isoform 인덱스 ─────────────────────────────────────────────
def gene_list_to_idx(gene_list):
    idxs = []
    for g in gene_list:
        idxs.extend(g2idx[g])
    return sorted(set(idxs))

diff_splice_idx  = gene_list_to_idx(zero_esm2_diff_splice)
normal_idx       = gene_list_to_idx(nonzero_esm2_genes)
all_multi_idx    = gene_list_to_idx(list(multi_genes.keys()))

print(f"\n  DIFF_SPLICE isoform count: {len(diff_splice_idx)}")
print(f"  Normal isoform count:      {len(normal_idx)}")
print(f"  All multi isoform count:   {len(all_multi_idx)}")

# ── 5. 라벨 로드 ──────────────────────────────────────────────────────────
print("\n[4] Loading labels...")
ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]
te_sym = [ENSG2SYM.get(g.split('.')[0], g.split('.')[0]) for g in te_gene]

def load_labels_te(go_term):
    pos = set()
    with open(ANNOT_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    return np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)

# ── 6. AUPRC per subset ────────────────────────────────────────────────────
print("\n[5] Computing AUPRC by subset...")
print(f"\n{'GO term':<15} {'name':<25} {'ALL':>8} {'DIFF_SP':>8} {'NORMAL':>8} {'Δ(DS-N)':>8}")
print('-' * 80)

results_per_go = []
for j, (go, name) in enumerate(GO_TERMS.items()):
    y_te_all = load_labels_te(go)
    preds    = score_matrix[:, j]

    # ALL
    auprc_all = average_precision_score(y_te_all, preds) if y_te_all.sum() > 0 else None

    # DIFF_SPLICE subset
    y_ds  = y_te_all[diff_splice_idx]
    p_ds  = preds[diff_splice_idx]
    auprc_ds = average_precision_score(y_ds, p_ds) if y_ds.sum() > 0 else None

    # NORMAL subset
    y_nm  = y_te_all[normal_idx]
    p_nm  = preds[normal_idx]
    auprc_nm = average_precision_score(y_nm, p_nm) if y_nm.sum() > 0 else None

    delta = (auprc_ds - auprc_nm) if (auprc_ds is not None and auprc_nm is not None) else None

    a_str  = f"{auprc_all:.4f}" if auprc_all is not None else "  N/A "
    ds_str = f"{auprc_ds:.4f}"  if auprc_ds  is not None else "  N/A "
    nm_str = f"{auprc_nm:.4f}"  if auprc_nm  is not None else "  N/A "
    d_str  = f"{delta:+.4f}"    if delta      is not None else "   N/A"

    print(f"{go:<15} {name:<25} {a_str:>8} {ds_str:>8} {nm_str:>8} {d_str:>8}")
    results_per_go.append({
        'go': go, 'name': name,
        'auprc_all':   auprc_all,
        'auprc_diff_splice': auprc_ds,
        'auprc_normal':     auprc_nm,
        'delta':            delta,
        'n_pos_ds': int(y_ds.sum()),
        'n_pos_nm': int(y_nm.sum()),
    })

# 매크로 평균
valid_all = [r['auprc_all']         for r in results_per_go if r['auprc_all']  is not None]
valid_ds  = [r['auprc_diff_splice'] for r in results_per_go if r['auprc_diff_splice'] is not None]
valid_nm  = [r['auprc_normal']      for r in results_per_go if r['auprc_normal'] is not None]

macro_all = np.mean(valid_all) if valid_all else 0
macro_ds  = np.mean(valid_ds)  if valid_ds  else 0
macro_nm  = np.mean(valid_nm)  if valid_nm  else 0

print('-' * 80)
print(f"{'Macro AUPRC':<15} {'':>25} {macro_all:>8.4f} {macro_ds:>8.4f} {macro_nm:>8.4f} {macro_ds-macro_nm:>+8.4f}")

print(f"\n{'='*65}")
print(f"  KEY RESULT:")
print(f"  DIFF_SPLICE_SAME_PROT (zero-ESM2) macro AUPRC: {macro_ds:.4f}")
print(f"  Normal ESM-2          macro AUPRC:              {macro_nm:.4f}")
print(f"  Gap (DS - Normal):                              {macro_ds-macro_nm:+.4f}")
print(f"{'='*65}")
print(f"\n  해석: 이 gap이 v15d_splice가 메워야 하는 목표.")

# ── 7. splice_delta 활용 가능성 확인 ─────────────────────────────────────
print("\n[6] splice_delta coverage in DIFF_SPLICE genes...")
sd_ds = sd[diff_splice_idx]
sd_nm = sd[normal_idx]
nonzero_ds = (np.abs(sd_ds).sum(axis=1) > 0).mean()
nonzero_nm = (np.abs(sd_nm).sum(axis=1) > 0).mean()
active_dims_ds = (sd_ds != 0).sum(axis=1)
active_dims_nm = (sd_nm != 0).sum(axis=1)
print(f"  DIFF_SPLICE: nonzero splice_delta = {100*nonzero_ds:.1f}%, mean active dims = {active_dims_ds.mean():.1f}")
print(f"  Normal:      nonzero splice_delta = {100*nonzero_nm:.1f}%, mean active dims = {active_dims_nm.mean():.1f}")

# ── 8. 저장 ──────────────────────────────────────────────────────────────
out = {
    'model': 'v15d_bp_clean (score_matrix_18go_20260519_1914.npy)',
    'macro_auprc_all':         macro_all,
    'macro_auprc_diff_splice': macro_ds,
    'macro_auprc_normal':      macro_nm,
    'gap_diff_vs_normal':      macro_ds - macro_nm,
    'n_diff_splice_genes':     len(zero_esm2_diff_splice),
    'n_normal_genes':          len(nonzero_esm2_genes),
    'n_diff_splice_isoforms':  len(diff_splice_idx),
    'n_normal_isoforms':       len(normal_idx),
    'splice_delta_coverage_diff_splice': float(nonzero_ds),
    'splice_delta_coverage_normal':      float(nonzero_nm),
    'per_go': results_per_go,
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n[7] Results → {OUT_JSON}")
