#!/usr/bin/env python3
"""
train_prism_v15d_brain672.py
============================
PRISM v15d_bp_clean 아키텍처로 672개 뇌 BP GO term을 joint multi-label 학습.

선정 기준:
  - brain-annotated genes >= 100 (14,398 isoforms 중 충분한 양성)
  - muscle train-annotated genes >= 50 (학습 신호 충분)
  - GO BP name 존재 (gene2go.gz 기반)
  총 672개 GO term

훈련:
  - 데이터: esm2_train_human_t30_150M.npy (31,668 근육 이소폼, 640-dim)
  - 레이블: human_annotations_unified_bp.txt → 672 GO term multi-hot
  - 아키텍처: Dense(256)→BN→Drop(0.3)→Dense(128)→Drop(0.2)→Dense(64)→Dense(672,sigmoid)
  - Loss: BinaryFocalCrossentropy (gamma=2.0, per-term balanced)
  - 5-seed ensemble + gene-stratified 5-fold CV
  - AUPRC-based early stopping (MIN_EPOCHS=30, PATIENCE=10)

평가:
  - Brain test set zero-shot AUPRC (muscle → brain cross-tissue)
  - 기준: 672 GO term 각각 brain isoform 기반 AUPRC
"""
import os, json, time, gzip
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

# ── GPU 설정 ──────────────────────────────────────────────────────────────────
device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else
                      'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    idx = device.index if device.index is not None else 0
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    free, total = torch.cuda.mem_get_info(device)
    print(f"  Free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

BASE     = Path('/home/welcome1/sw1686/DIFFUSE')
OUT_DIR  = BASE / 'reports'
DEMO_DIR = BASE / 'prism_app/data/demo'
OUT_DIR.mkdir(exist_ok=True)

N_SEEDS          = 5
BATCH            = 512
EPOCHS           = 200
MIN_EPOCHS       = 30
PATIENCE         = 10
LR               = 1e-3
AUPRC_EVAL_EVERY = 5

# ── 672 GO term 로드 (brain>=100, train>=50) ──────────────────────────────────
print("Loading 672 GO terms from brain672_go_terms.json...")
with open(BASE / 'hMuscle/data/brain672_go_terms.json') as f:
    go_meta = json.load(f)
go_ids   = go_meta['go_ids']      # 672개, brain n_pos 내림차순
go_names_d = go_meta['go_names']  # dict
go_names = [go_names_d[g] for g in go_ids]
N_GO     = len(go_ids)
print(f"  GO terms: {N_GO}  (brain range: "
      f"{go_meta['brain_counts'][go_ids[-1]]}~{go_meta['brain_counts'][go_ids[0]]})")

# ── Load training embeddings ──────────────────────────────────────────────────
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

# ── Build multi-label matrix ──────────────────────────────────────────────────
print(f"Building label matrix ({n_tr} × {N_GO})...")
Y_tr = np.zeros((n_tr, N_GO), dtype=np.float32)
for j, go_id in enumerate(go_ids):
    for i, gene in enumerate(tr_genes):
        if gene in gene_go and go_id in gene_go[gene]:
            Y_tr[i, j] = 1.0
n_pos_per_go = Y_tr.sum(axis=0)
print(f"  Label density: mean={n_pos_per_go.mean():.1f}, "
      f"min={n_pos_per_go.min():.0f}, max={n_pos_per_go.max():.0f}")
print(f"  Terms with 0 positives: {(n_pos_per_go == 0).sum()}")

# ── Brain ground truth (for zero-shot eval) ───────────────────────────────────
print(f"Building brain ground truth ({n_br} × {N_GO})...")
Y_br = np.zeros((n_br, N_GO), dtype=np.float32)
for j, go_id in enumerate(go_ids):
    for i, gene in enumerate(br_genes):
        if gene in gene_go and go_id in gene_go[gene]:
            Y_br[i, j] = 1.0
annotated_brain = (Y_br.sum(axis=1) > 0).sum()
print(f"  Brain isoforms with ≥1 annotation: {annotated_brain}/{n_br} ({100*annotated_brain/n_br:.1f}%)")

# ── Normalize ─────────────────────────────────────────────────────────────────
print("Normalizing...")
scaler   = StandardScaler()
X_tr_n   = scaler.fit_transform(X_tr)
X_br_n   = scaler.transform(X_br)

# ── PRISM v15d: 672-term output ───────────────────────────────────────────────
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

# ── Gene-stratified 5-fold split ─────────────────────────────────────────────
tr_gene_arr  = np.array(tr_genes)
unique_genes = np.unique(tr_gene_arr)
rng          = np.random.default_rng(42)
gene_fold    = {g: rng.integers(0, 5) for g in unique_genes}
fold_arr     = np.array([gene_fold[g] for g in tr_genes])

X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32)
Y_tr_t = torch.tensor(Y_tr,   dtype=torch.float32)
X_br_t = torch.tensor(X_br_n, dtype=torch.float32).to(device)

# ── 5-seed ensemble training ─────────────────────────────────────────────────
print(f"\nTraining PRISM v15d {N_GO}-term (5-seed ensemble, joint multi-label)...")
t0 = time.time()

seed_preds = []
oof_preds  = np.zeros((n_tr, N_GO), dtype=np.float32)

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

            # Macro AUPRC on validation fold (only terms with positives)
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
        oof_val = model(X_val).cpu().numpy()
    oof_preds[va_mask] += oof_val

    with torch.no_grad():
        # Predict in batches to avoid OOM with 672-dim output × 63994 isoforms
        br_chunks = []
        for start in range(0, n_br, 2048):
            br_chunks.append(model(X_br_t[start:start+2048]).cpu().numpy())
        br_pred = np.concatenate(br_chunks, axis=0)
    seed_preds.append(br_pred)

    elapsed = time.time() - t0
    fold_final = [average_precision_score(val_labels_np[:, j], oof_val[:, j])
                  for j in range(N_GO) if val_labels_np[:, j].sum() > 0]
    print(f"  Seed {seed}: best_val_AUPRC={best_auprc:.4f}, "
          f"final_fold_AUPRC={np.mean(fold_final):.4f} | {elapsed:.0f}s elapsed")

# ── Ensemble average ─────────────────────────────────────────────────────────
score_matrix = np.mean(seed_preds, axis=0).astype(np.float32)
print(f"\nScore matrix: {score_matrix.shape}, "
      f"range [{score_matrix.min():.4f}, {score_matrix.max():.4f}]")

# ── Brain zero-shot AUPRC ─────────────────────────────────────────────────────
print("\n=== Brain Test Set AUPRC (zero-shot, muscle→brain) ===")
brain_aurpcs = []
meta = []
for j, (go_id, go_name) in enumerate(zip(go_ids, go_names)):
    y_true = Y_br[:, j]
    y_pred = score_matrix[:, j]
    n_pos_brain = int(y_true.sum())
    n_pos_train = int(Y_tr[:, j].sum())
    if n_pos_brain >= 5:
        auprc = float(average_precision_score(y_true, y_pred))
        brain_aurpcs.append(auprc)
        meta.append({'go': go_id, 'name': go_name,
                     'n_pos_muscle_train': n_pos_train,
                     'n_pos_brain': n_pos_brain,
                     'auprc_brain': round(auprc, 4)})
    else:
        meta.append({'go': go_id, 'name': go_name,
                     'n_pos_muscle_train': n_pos_train,
                     'n_pos_brain': n_pos_brain,
                     'auprc_brain': None})

valid_aurpcs = [m['auprc_brain'] for m in meta if m['auprc_brain'] is not None]
print(f"Terms evaluated: {len(valid_aurpcs)}/{N_GO}")
print(f"Macro AUPRC (brain zero-shot): {np.mean(valid_aurpcs):.4f}")
print(f"Median AUPRC:                  {np.median(valid_aurpcs):.4f}")
print(f"AUPRC > 0.5:  {sum(a>0.5 for a in valid_aurpcs)}/{len(valid_aurpcs)}")
print(f"AUPRC > 0.4:  {sum(a>0.4 for a in valid_aurpcs)}/{len(valid_aurpcs)}")

print("\nTop 10:")
for m in sorted((m for m in meta if m['auprc_brain']), key=lambda x: x['auprc_brain'], reverse=True)[:10]:
    print(f"  {m['go']} ({m['name'][:40]}): {m['auprc_brain']:.4f}  n_pos={m['n_pos_brain']}")
print("\nBottom 5:")
for m in sorted((m for m in meta if m['auprc_brain']), key=lambda x: x['auprc_brain'])[:5]:
    print(f"  {m['go']} ({m['name'][:40]}): {m['auprc_brain']:.4f}  n_pos={m['n_pos_brain']}")

elapsed_total = time.time() - t0
print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")

# ── Save outputs ──────────────────────────────────────────────────────────────
out_npy  = OUT_DIR / 'brain_full_672_scores.npy'
out_meta = OUT_DIR / 'brain_full_672_meta.json'

np.save(out_npy, score_matrix)
print(f"Saved scores: {out_npy}")

with open(out_meta, 'w') as f:
    json.dump({
        'model': 'prism_v15d_joint672',
        'evaluation': 'brain_test_zeroshot',
        'criteria': 'brain>=100_train>=50_BP',
        'go_ids': go_ids,
        'go_names': go_names_d,
        'n_isoforms_brain': n_br,
        'n_go': N_GO,
        'macro_auprc_brain': round(float(np.mean(valid_aurpcs)), 4),
        'median_auprc_brain': round(float(np.median(valid_aurpcs)), 4),
        'n_auprc_gt05': int(sum(a > 0.5 for a in valid_aurpcs)),
        'n_auprc_gt04': int(sum(a > 0.4 for a in valid_aurpcs)),
        'per_go': meta,
    }, f, indent=2)
print(f"Saved meta:   {out_meta}")

# ── Copy score matrix to demo dir ────────────────────────────────────────────
import shutil
demo_scores = DEMO_DIR / 'brain_full_672_scores.npy'
shutil.copy(out_npy, demo_scores)
print(f"Copied to demo: {demo_scores}")
print("\nAll done.")
