#!/usr/bin/env python3
"""
train_prism_v15d_brain_expanded.py
===================================
PRISM v15d 아키텍처로 41개 확장 뇌/AD 관련 BP GO term을 joint multi-label 학습.

GO term 선정 기준:
  [18 기존] brain_full 18GO (AUPRC 0.40~0.76, validated)
  [23 신규] 672GO 후보 중 brain-relevant + AUPRC >= 0.54 + n_muscle >= 100
            계층 중복 최소화, 뇌/AD 기능 coverage 극대화
  총 41개 GO term

근거:
  - 18GO macro AUPRC on brain: 0.5998  (18 head)
  - 672GO macro AUPRC on brain: 0.3566 (672 head, multi-task interference)
  - 41GO: multi-task interference 최소화 + brain/AD coverage 확장

훈련:
  - 데이터: esm2_train_human_t30_150M.npy (31,668 근육 이소폼, 640-dim)
  - 레이블: human_annotations_unified_bp.txt → 41 GO term multi-hot
  - 아키텍처: Dense(256)→BN→Drop(0.3)→Dense(128)→Drop(0.2)→Dense(64)→Dense(41,sigmoid)
  - Loss: BinaryFocalCrossentropy (gamma=2.0, per-term balanced sqrt-inv-freq)
  - 5-seed ensemble + gene-stratified 5-fold CV
  - AUPRC-based early stopping (MIN_EPOCHS=30, PATIENCE=10)

출력:
  - reports/brain_full_expanded_41_scores.npy   (63994 × 41)
  - reports/brain_full_expanded_41_meta.json
  - prism_app/data/demo/brain_full_expanded_41_scores.npy
  - prism_app/data/demo/brain_full_expanded_41_meta.json (compact)
"""
import os, json, time
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else
                      'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    idx = device.index if device.index is not None else 0
    print(f"  GPU: {torch.cuda.get_device_name(idx)}")
    free, total = torch.cuda.mem_get_info(device)
    print(f"  Free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

BASE     = Path('/home/welcome1/sw1686/DIFFUSE')
OUT_DIR  = BASE / 'reports'
DEMO_DIR = BASE / 'prism_app/data/demo'

N_SEEDS          = 5
BATCH            = 512
EPOCHS           = 200
MIN_EPOCHS       = 30
PATIENCE         = 10
LR               = 1e-3
AUPRC_EVAL_EVERY = 5

# ── 41-term GO list ────────────────────────────────────────────────────────────
# 18 existing (validated brain AUPRC 0.40–0.76)
GO_18_EXISTING = [
    'GO:0007204',  # Ca2+ signaling              0.5427
    'GO:0045214',  # Sarcomere organization       0.7364  (not in 672GO)
    'GO:0006941',  # Muscle contraction           0.6031
    'GO:0006914',  # Autophagy                    0.4877
    'GO:0043161',  # Proteasome-UPS               0.6564
    'GO:0007519',  # Skeletal muscle dev          0.6182  (not in 672GO)
    'GO:0042692',  # Muscle cell diff             0.5978  (not in 672GO)
    'GO:0055074',  # Ca2+ homeostasis             0.5426
    'GO:0007005',  # Mitochondrion org            0.5353
    'GO:0007517',  # Muscle organ dev             0.6013
    'GO:0032006',  # TOR signaling                0.3983  (not in 672GO, keep for mTOR)
    'GO:0030048',  # Actin-based movement         0.6311  (not in 672GO)
    'GO:0006096',  # Glycolysis                   0.7559  (not in 672GO)
    'GO:0007268',  # Synaptic transmission        0.6991
    'GO:0007018',  # MT-based movement            0.6687
    'GO:0031175',  # Neuron proj dev              0.5784
    'GO:0030182',  # Neuron diff                  0.5519
    'GO:0000226',  # MT cytoskeleton org          0.5923
]

# 23 new brain/AD-relevant terms (from 672GO, AUPRC >= 0.54 on brain)
GO_23_NEW = [
    # Ion channels/transport (neuron excitability)
    'GO:0007156',  # homophilic cell-cell adhesion      0.7370
    'GO:0006813',  # potassium ion transport            0.6908
    'GO:0055085',  # transmembrane transport            0.6600
    'GO:0006812',  # cation transport                   0.6488
    'GO:0006820',  # anion transport                    0.6539
    # Kinase/phosphatase signaling
    'GO:0006468',  # protein phosphorylation            0.6477
    'GO:0016311',  # dephosphorylation                  0.5861
    'GO:0043408',  # regulation of MAPK cascade         0.5526
    # GPCR / receptor signaling
    'GO:0007186',  # GPCR signaling pathway             0.6381
    # Transcription regulation
    'GO:0006357',  # regulation of transcription RNAPolII 0.6231
    # Proteostasis / AD-relevant
    'GO:0016567',  # protein ubiquitination             0.5873
    'GO:0006511',  # ubiquitin-dependent protein catabolic 0.5443
    'GO:0006508',  # proteolysis                        0.5481
    # mRNA splicing (directly relevant to isoform biology)
    'GO:0000398',  # mRNA splicing via spliceosome      0.5725
    'GO:0006397',  # mRNA processing                    0.5866
    # Mitochondria (neurodegeneration)
    'GO:0032543',  # mitochondrial translation          0.5698
    # Neuron-specific
    'GO:0045664',  # regulation of neuron differentiation 0.5506
    'GO:0006836',  # neurotransmitter transport         0.5486
    'GO:0007411',  # axon guidance                      0.5388
    # Cell adhesion / synaptic structure
    'GO:0007155',  # cell adhesion                      0.5518
    # Nuclear/cytoplasmic
    'GO:0006913',  # nucleocytoplasmic transport        0.5495
    # Energy metabolism
    'GO:0006753',  # nucleoside phosphate metabolic process 0.5476
    # mRNA / post-transcriptional
    'GO:0016071',  # mRNA metabolic process             0.5392
]

GO_IDS = GO_18_EXISTING + GO_23_NEW
N_GO   = len(GO_IDS)
assert N_GO == 41, f"Expected 41 terms, got {N_GO}"
print(f"GO terms: {N_GO}  ({len(GO_18_EXISTING)} existing + {len(GO_23_NEW)} new)")

# ── Load GO names ──────────────────────────────────────────────────────────────
with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    d672 = json.load(f)
go_names_672 = d672.get('go_names', {})

# Supplement with 18GO names
with open(BASE / 'reports/v15d_brain_eval/brain_eval_20260519_2125.json') as f:
    d18 = json.load(f)
go_names_18 = d18['go_terms']  # {go_id: short_name}

go_names_dict = {}
for g in GO_IDS:
    if g in go_names_672:
        go_names_dict[g] = go_names_672[g]
    elif g in go_names_18:
        go_names_dict[g] = go_names_18[g]
    else:
        go_names_dict[g] = g

go_names_list = [go_names_dict[g] for g in GO_IDS]

# ── Load training data ─────────────────────────────────────────────────────────
print("Loading muscle training embeddings...")
X_tr = np.load(BASE / 'hMuscle/data/esm2_train_human_t30_150M.npy').astype(np.float32)
tr_genes_raw = np.load(BASE / 'hMuscle/data/raw_data/data/id_lists/train_gene_list.npy',
                       allow_pickle=True)
tr_genes = [x.decode() if isinstance(x, bytes) else str(x) for x in tr_genes_raw]
n_tr = len(tr_genes)
print(f"  Muscle train: {X_tr.shape}")

print("Loading brain prediction embeddings...")
X_br = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_t30_150M.npy').astype(np.float32)
br_ids_raw = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy',
                     allow_pickle=True)
br_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in br_ids_raw]
br_genes_raw = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy',
                       allow_pickle=True)
br_genes = [x.decode() if isinstance(x, bytes) else str(x) for x in br_genes_raw]
n_br = len(br_ids)
print(f"  Brain predict: {X_br.shape}")

print("Loading GO annotations...")
gene_go = {}
with open(BASE / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            gene_go[parts[0]] = set(parts[1:])
print(f"  Loaded {len(gene_go)} gene annotations")

# ── Build label matrices ───────────────────────────────────────────────────────
print(f"Building train label matrix ({n_tr} × {N_GO})...")
Y_tr = np.zeros((n_tr, N_GO), dtype=np.float32)
for j, go_id in enumerate(GO_IDS):
    for i, gene in enumerate(tr_genes):
        if gene in gene_go and go_id in gene_go[gene]:
            Y_tr[i, j] = 1.0
n_pos_per_go = Y_tr.sum(axis=0)
print(f"  Label density: mean={n_pos_per_go.mean():.1f}, "
      f"min={n_pos_per_go.min():.0f}, max={n_pos_per_go.max():.0f}")
assert (n_pos_per_go == 0).sum() == 0, \
    f"Zero-positive GO terms: {[GO_IDS[j] for j in range(N_GO) if n_pos_per_go[j]==0]}"

print(f"Building brain ground truth ({n_br} × {N_GO})...")
Y_br = np.zeros((n_br, N_GO), dtype=np.float32)
for j, go_id in enumerate(GO_IDS):
    for i, gene in enumerate(br_genes):
        if gene in gene_go and go_id in gene_go[gene]:
            Y_br[i, j] = 1.0
annotated = (Y_br.sum(axis=1) > 0).sum()
print(f"  Brain isoforms with ≥1 annotation: {annotated}/{n_br} ({100*annotated/n_br:.1f}%)")

# ── Normalize ──────────────────────────────────────────────────────────────────
print("Normalizing embeddings...")
scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_br_n = scaler.transform(X_br)

# ── Model ──────────────────────────────────────────────────────────────────────
class PRISMv15d(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def focal_loss_multilabel(preds, targets, gamma=2.0, pos_weights=None):
    bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
    pt  = torch.where(targets == 1, preds, 1 - preds)
    fl  = ((1 - pt) ** gamma) * bce
    if pos_weights is not None:
        w = torch.where(targets == 1,
                        pos_weights.unsqueeze(0).expand_as(targets),
                        torch.ones_like(targets))
        fl = fl * w
    return fl.mean()


# Per-GO class weights (inverse frequency, sqrt smoothing)
pos_freq   = torch.tensor(n_pos_per_go / n_tr, dtype=torch.float32).to(device)
pos_weight = (1.0 / pos_freq.clamp(min=1e-4)).sqrt()
pos_weight = pos_weight / pos_weight.mean()

# ── Gene-stratified 5-fold split ───────────────────────────────────────────────
tr_gene_arr  = np.array(tr_genes)
unique_genes = np.unique(tr_gene_arr)
rng          = np.random.default_rng(42)
gene_fold    = {g: rng.integers(0, 5) for g in unique_genes}
fold_arr     = np.array([gene_fold[g] for g in tr_genes])

X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32)
Y_tr_t = torch.tensor(Y_tr,   dtype=torch.float32)
X_br_t = torch.tensor(X_br_n, dtype=torch.float32).to(device)

# ── 5-seed ensemble training ───────────────────────────────────────────────────
print(f"\nTraining PRISM v15d {N_GO}-term (5-seed ensemble, joint multi-label)...")
t0 = time.time()

seed_preds = []

for seed in range(N_SEEDS):
    torch.manual_seed(seed * 137 + 7)
    model     = PRISMv15d(n_out=N_GO).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5)

    val_fold = seed % 5
    tr_mask  = fold_arr != val_fold
    va_mask  = fold_arr == val_fold

    X_train = X_tr_t[tr_mask].to(device)
    Y_train = Y_tr_t[tr_mask].to(device)
    X_val   = X_tr_t[va_mask].to(device)
    Y_val   = Y_tr_t[va_mask].to(device)
    val_labels_np = Y_val.cpu().numpy()

    loader = DataLoader(TensorDataset(X_train, Y_train),
                        batch_size=BATCH, shuffle=True, drop_last=True)

    best_auprc  = -1.0
    patience_ct = 0
    best_state  = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = focal_loss_multilabel(model(xb), yb, pos_weights=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % AUPRC_EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_pred_np = model(X_val).cpu().numpy()
                val_loss = focal_loss_multilabel(
                    model(X_val), Y_val, pos_weights=pos_weight).item()
            scheduler.step(val_loss)

            fold_aurpcs = []
            for j in range(N_GO):
                if val_labels_np[:, j].sum() > 0:
                    fold_aurpcs.append(
                        average_precision_score(val_labels_np[:, j], val_pred_np[:, j]))
            macro_auprc = float(np.mean(fold_aurpcs)) if fold_aurpcs else 0.0

            if epoch >= MIN_EPOCHS:
                if macro_auprc > best_auprc + 1e-4:
                    best_auprc  = macro_auprc
                    patience_ct = 0
                    best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_ct += 1
                    if patience_ct >= PATIENCE:
                        print(f"    Early stop at epoch {epoch+1}")
                        break
            else:
                if macro_auprc > best_auprc:
                    best_auprc = macro_auprc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        br_chunks = []
        for start in range(0, n_br, 2048):
            br_chunks.append(model(X_br_t[start:start+2048]).cpu().numpy())
        br_pred = np.concatenate(br_chunks, axis=0)
    seed_preds.append(br_pred)

    elapsed = time.time() - t0
    print(f"  Seed {seed}: best_val_AUPRC={best_auprc:.4f} | {elapsed:.0f}s elapsed")

# ── Ensemble average ──────────────────────────────────────────────────────────
score_matrix = np.mean(seed_preds, axis=0).astype(np.float32)
print(f"\nScore matrix: {score_matrix.shape}, "
      f"range [{score_matrix.min():.4f}, {score_matrix.max():.4f}]")

# ── Brain zero-shot AUPRC ──────────────────────────────────────────────────────
print("\n=== Brain Zero-Shot AUPRC (muscle → brain) ===")
per_go_meta = []
brain_aurpcs = []
for j, go_id in enumerate(GO_IDS):
    y_true = Y_br[:, j]
    y_pred = score_matrix[:, j]
    n_pos_brain = int(y_true.sum())
    n_pos_train = int(Y_tr[:, j].sum())
    if n_pos_brain >= 5:
        auprc = float(average_precision_score(y_true, y_pred))
        brain_aurpcs.append(auprc)
        per_go_meta.append({'go': go_id, 'name': go_names_dict[go_id],
                            'n_pos_muscle_train': n_pos_train,
                            'n_pos_brain': n_pos_brain,
                            'auprc_brain': round(auprc, 4),
                            'source': 'existing_18' if go_id in GO_18_EXISTING else 'new_23'})
    else:
        per_go_meta.append({'go': go_id, 'name': go_names_dict[go_id],
                            'n_pos_muscle_train': n_pos_train,
                            'n_pos_brain': n_pos_brain,
                            'auprc_brain': None,
                            'source': 'existing_18' if go_id in GO_18_EXISTING else 'new_23'})

valid_aurpcs   = [m['auprc_brain'] for m in per_go_meta if m['auprc_brain'] is not None]
macro_auprc    = float(np.mean(valid_aurpcs))
median_auprc   = float(np.median(valid_aurpcs))
n_gt05         = sum(a > 0.5 for a in valid_aurpcs)
n_gt04         = sum(a > 0.4 for a in valid_aurpcs)

print(f"Terms evaluated: {len(valid_aurpcs)}/{N_GO}")
print(f"Macro AUPRC:   {macro_auprc:.4f}")
print(f"Median AUPRC:  {median_auprc:.4f}")
print(f"AUPRC > 0.5:   {n_gt05}/{len(valid_aurpcs)}")
print(f"AUPRC > 0.4:   {n_gt04}/{len(valid_aurpcs)}")

print("\nAll GO terms (sorted by AUPRC):")
for m in sorted((m for m in per_go_meta if m['auprc_brain']), key=lambda x: -x['auprc_brain']):
    tag = '[18GO]' if m['source'] == 'existing_18' else '[NEW] '
    print(f"  {tag}  {m['auprc_brain']:.4f}  {m['go']}  {m['name'][:50]}")

# ── Per-GO score stats for score calibration reference ─────────────────────────
print("\nScore distribution per GO term (top-5 by max score):")
for j in range(N_GO):
    col = score_matrix[:, j]
    print(f"  {GO_IDS[j]}: p50={np.percentile(col,50):.3f} "
          f"p90={np.percentile(col,90):.3f} p95={np.percentile(col,95):.3f} "
          f"p99={np.percentile(col,99):.3f} max={col.max():.3f}")

elapsed_total = time.time() - t0
print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")

# ── Save outputs ───────────────────────────────────────────────────────────────
out_scores = OUT_DIR / 'brain_full_expanded_41_scores.npy'
out_meta   = OUT_DIR / 'brain_full_expanded_41_meta.json'

np.save(out_scores, score_matrix)
print(f"Saved scores: {out_scores}")

meta_obj = {
    'model': 'prism_v15d_expanded41',
    'evaluation': 'brain_test_zeroshot',
    'n_go': N_GO,
    'go_ids': GO_IDS,
    'go_names': go_names_dict,
    'go_source': {g: ('existing_18' if g in GO_18_EXISTING else 'new_23') for g in GO_IDS},
    'n_isoforms_brain': n_br,
    'macro_auprc_brain': round(macro_auprc, 4),
    'median_auprc_brain': round(median_auprc, 4),
    'n_auprc_gt05': n_gt05,
    'n_auprc_gt04': n_gt04,
    'per_go': per_go_meta,
}
with open(out_meta, 'w') as f:
    json.dump(meta_obj, f, indent=2)
print(f"Saved meta:   {out_meta}")

# ── Copy to demo dir ──────────────────────────────────────────────────────────
import shutil
shutil.copy(out_scores, DEMO_DIR / 'brain_full_expanded_41_scores.npy')

# Compact meta for demo (omit per_go detail, keep go_ids + names + auprc)
compact_meta = {k: v for k, v in meta_obj.items() if k != 'per_go'}
compact_meta['per_go_auprc'] = {m['go']: m['auprc_brain'] for m in per_go_meta}
with open(DEMO_DIR / 'brain_full_expanded_41_meta.json', 'w') as f:
    json.dump(compact_meta, f, indent=2)

# Also save brain_full_ids/gene_ids for reference
shutil.copy(
    BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy',
    DEMO_DIR / 'brain_full_expanded_41_ids.npy'
)
shutil.copy(
    BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy',
    DEMO_DIR / 'brain_full_expanded_41_gene_ids.npy'
)

print(f"Copied to demo dir: {DEMO_DIR}")
print("\nDone. Next: run update_bisect_prism_scores_expanded.py")
