#!/usr/bin/env python3
"""
train_prism_v15d_brain73.py
============================
PRISM v15d_bp_clean 아키텍처로 73 brain GO term을 동시에 multi-label 학습.

차이점 (vs generate_brain_full_extended_scores_gpu.py):
  - 이전: GO term별 독립 MLP 73개 (per-term probe)
  - 이번: 73-dim sigmoid 출력으로 joint multi-label 학습
           → shared representation, multi-task regularization

훈련:
  - 데이터: esm2_train_human_t30_150M.npy (31,668 근육 이소폼, 640-dim)
  - 레이블: human_annotations_unified_bp.txt → 73 brain GO term 전파
  - 아키텍처: Dense(256)→BN→Drop(0.3)→Dense(128)→Drop(0.2)→Dense(64)→Dense(73,sigmoid)
  - Loss: BinaryFocalCrossentropy (gamma=2.0, per-term balanced)
  - 5-seed ensemble + gene-stratified 5-fold CV

예측:
  - 대상: brain_full_esm2_t30_150M.npy (63,994 뇌 이소폼)
  - 출력: brain_full_extended_scores_v15d.npy (63994×73)
"""
import os, json, time
import numpy as np
from pathlib import Path
from collections import Counter

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

N_SEEDS   = 5
BATCH     = 512
EPOCHS    = 200
MIN_EPOCHS = 30      # AUPRC-based early stopping starts only after this
PATIENCE  = 10       # patience in AUPRC (not loss)
LR        = 1e-3
AUPRC_EVAL_EVERY = 5  # evaluate macro AUPRC every N epochs

# ── Brain GO terms (73개) ─────────────────────────────────────────────────────
BRAIN_EXTENDED = {
    'GO:0007186': 'G protein-coupled receptor signaling pathway',
    'GO:0030182': 'Neuron differentiation',
    'GO:0007167': 'Enzyme-linked receptor protein signaling pathway',
    'GO:0048666': 'Neuron development',
    'GO:0050767': 'Regulation of neurogenesis',
    'GO:0007169': 'Cell surface receptor protein tyrosine kinase signaling',
    'GO:0010469': 'Regulation of signaling receptor activity',
    'GO:0031175': 'Neuron projection development',
    'GO:0002768': 'Immune response-regulating cell surface receptor signaling',
    'GO:0007420': 'Brain development',
    'GO:0002429': 'Immune response-activating cell surface receptor signaling',
    'GO:0045664': 'Regulation of neuron differentiation',
    'GO:0007268': 'Chemical synaptic transmission',
    'GO:0007166': 'Cell surface receptor signaling pathway',
    'GO:0055074': 'Calcium ion homeostasis',
    'GO:0006874': 'Intracellular calcium ion homeostasis',
    'GO:0048812': 'Neuron projection morphogenesis',
    'GO:0042391': 'Regulation of membrane potential',
    'GO:0010975': 'Regulation of neuron projection development',
    'GO:0050804': 'Modulation of chemical synaptic transmission',
    'GO:0061564': 'Axon development',
    'GO:0007409': 'Axonogenesis',
    'GO:0050769': 'Positive regulation of neurogenesis',
    'GO:0051480': 'Regulation of cytosolic calcium ion concentration',
    'GO:0007204': 'Positive regulation of cytosolic calcium ion concentration',
    'GO:0006898': 'Receptor-mediated endocytosis',
    'GO:0006816': 'Calcium ion transport',
    'GO:0007411': 'Axon guidance',
    'GO:0050851': 'Antigen receptor-mediated signaling pathway',
    'GO:0045666': 'Positive regulation of neuron differentiation',
    'GO:0007187': 'GPCR signaling, coupled to cyclic nucleotide 2nd messenger',
    'GO:0010976': 'Positive regulation of neuron projection development',
    'GO:0007189': 'Adenylate cyclase-activating GPCR signaling',
    'GO:0007188': 'Adenylate cyclase-modulating GPCR signaling',
    'GO:0070588': 'Calcium ion transmembrane transport',
    'GO:0008037': 'Cell recognition',
    'GO:0038093': 'Fc receptor signaling pathway',
    'GO:0019722': 'Calcium-mediated signaling',
    'GO:0051924': 'Regulation of calcium ion transport',
    'GO:0050808': 'Synapse organization',
    'GO:0071805': 'Potassium ion transmembrane transport',
    'GO:0097485': 'Neuron projection guidance',
    'GO:0006813': 'Potassium ion transport',
    'GO:0007178': 'Cell surface receptor protein serine/threonine kinase signaling',
    'GO:0030522': 'Intracellular receptor signaling pathway',
    'GO:0050768': 'Negative regulation of neurogenesis',
    'GO:0007200': 'Phospholipase C-activating GPCR signaling',
    'GO:0050890': 'Cognition',
    'GO:0038094': 'Fc-gamma receptor signaling pathway',
    'GO:0006836': 'Neurotransmitter transport',
    'GO:0038096': 'Fc-gamma receptor signaling in phagocytosis',
    'GO:0030512': 'Negative regulation of TGF-beta receptor signaling',
    'GO:0002431': 'Fc receptor mediated stimulatory signaling',
    'GO:0050852': 'T cell receptor signaling pathway',
    'GO:0045665': 'Negative regulation of neuron differentiation',
    'GO:0008277': 'Regulation of GPCR signaling',
    'GO:0051592': 'Response to calcium ion',
    'GO:0050853': 'B cell receptor signaling pathway',
    'GO:0048167': 'Regulation of synaptic plasticity',
    'GO:0002221': 'Pattern recognition receptor signaling',
    'GO:0007611': 'Learning or memory',
    'GO:0038095': 'Fc-epsilon receptor signaling pathway',
    'GO:0050807': 'Regulation of synapse organization',
    'GO:0043524': 'Negative regulation of neuron apoptotic process',
    'GO:0010977': 'Negative regulation of neuron projection development',
    'GO:0046425': 'Regulation of receptor signaling via JAK-STAT',
    'GO:0050770': 'Regulation of axonogenesis',
    'GO:0007179': 'TGF-beta receptor signaling pathway',
    'GO:0007218': 'Neuropeptide signaling pathway',
    'GO:0007193': 'Adenylate cyclase-inhibiting GPCR signaling',
    'GO:1903169': 'Regulation of calcium ion transmembrane transport',
    'GO:0043523': 'Regulation of neuron apoptotic process',
    'GO:0001508': 'Action potential',
}
go_ids   = list(BRAIN_EXTENDED.keys())
go_names = list(BRAIN_EXTENDED.values())
N_GO     = len(go_ids)
print(f"GO terms: {N_GO}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading muscle training embeddings...")
X_tr = np.load(BASE / 'hMuscle/data/esm2_train_human_t30_150M.npy').astype(np.float32)
tr_genes = np.load(BASE / 'hMuscle/data/raw_data/data/id_lists/train_gene_list.npy',
                   allow_pickle=True)
tr_genes = [x.decode() if isinstance(x, bytes) else str(x) for x in tr_genes]
n_tr = len(tr_genes)
print(f"  Muscle train: {X_tr.shape}")

print("Loading brain prediction embeddings...")
X_br = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_t30_150M.npy').astype(np.float32)
br_ids = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
br_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in br_ids]
br_genes = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy', allow_pickle=True)
br_genes = [x.decode() if isinstance(x, bytes) else str(x) for x in br_genes]
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
print("Building label matrix (31668 × 73)...")
Y_tr = np.zeros((n_tr, N_GO), dtype=np.float32)
for j, go_id in enumerate(go_ids):
    for i, gene in enumerate(tr_genes):
        if gene in gene_go and go_id in gene_go[gene]:
            Y_tr[i, j] = 1.0
n_pos_per_go = Y_tr.sum(axis=0)
print(f"  Label density: mean={n_pos_per_go.mean():.1f}, "
      f"min={n_pos_per_go.min():.0f}, max={n_pos_per_go.max():.0f}")

# ── Normalize ─────────────────────────────────────────────────────────────────
print("Normalizing...")
scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_br_n = scaler.transform(X_br)

# ── PRISM v15d_bp_clean: multi-label 73-term ──────────────────────────────────
class PRISMv15d(nn.Module):
    def __init__(self, n_out=73):
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
    """Binary focal loss, averaged over all (isoform, GO) pairs."""
    bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
    pt  = torch.where(targets == 1, preds, 1 - preds)
    fl  = ((1 - pt) ** gamma) * bce
    if pos_weights is not None:
        # Up-weight positives: shape (n_go,)
        w = torch.where(targets == 1,
                        pos_weights.unsqueeze(0).expand_as(targets),
                        torch.ones_like(targets))
        fl = fl * w
    return fl.mean()


# Per-GO class weights (inverse frequency)
pos_freq   = torch.tensor(n_pos_per_go / n_tr, dtype=torch.float32).to(device)
pos_weight = (1.0 / pos_freq.clamp(min=1e-4)).sqrt()   # sqrt smoothing
pos_weight = pos_weight / pos_weight.mean()              # normalise

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
print(f"\nTraining PRISM v15d 73-term (5-seed ensemble, joint multi-label)...")
t0 = time.time()

seed_preds = []    # list of (n_br, 73) arrays
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

    loader = DataLoader(TensorDataset(X_train, Y_train),
                        batch_size=BATCH, shuffle=True)

    best_auprc  = -1.0
    patience_ct = 0
    best_state  = None
    val_labels_np = Y_val.cpu().numpy()

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = focal_loss_multilabel(model(xb), yb, pos_weights=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # AUPRC-based early stopping (every AUPRC_EVAL_EVERY epochs, after MIN_EPOCHS)
        if (epoch + 1) % AUPRC_EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_pred_np = model(X_val).cpu().numpy()
            val_loss = focal_loss_multilabel(model(X_val), Y_val,
                                             pos_weights=pos_weight).item()
            scheduler.step(val_loss)

            # Macro AUPRC on validation fold
            fold_aurpcs = []
            for j in range(N_GO):
                if val_labels_np[:, j].sum() > 0:
                    fold_aurpcs.append(
                        average_precision_score(val_labels_np[:, j], val_pred_np[:, j]))
            macro_auprc = float(np.mean(fold_aurpcs)) if fold_aurpcs else 0.0

            if epoch >= MIN_EPOCHS and macro_auprc > best_auprc + 1e-4:
                best_auprc  = macro_auprc
                patience_ct = 0
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif epoch >= MIN_EPOCHS:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    break
            else:
                # Before MIN_EPOCHS: always save best
                if macro_auprc > best_auprc:
                    best_auprc = macro_auprc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # OOF predictions for final AUPRC calculation
    model.eval()
    with torch.no_grad():
        oof_val = model(X_val).cpu().numpy()
    oof_preds[va_mask] += oof_val

    # Brain predictions
    with torch.no_grad():
        br_pred = model(X_br_t).cpu().numpy()
    seed_preds.append(br_pred)

    elapsed = time.time() - t0
    # Report final fold AUPRC
    val_labels_np2 = Y_val.cpu().numpy()
    fold_aurpcs = [average_precision_score(val_labels_np2[:, j], oof_val[:, j])
                   for j in range(N_GO) if val_labels_np2[:, j].sum() > 0]
    print(f"  Seed {seed}: best_val_AUPRC={best_auprc:.4f}, "
          f"final_fold_AUPRC={np.mean(fold_aurpcs):.4f} | {elapsed:.0f}s")

# ── Ensemble average ─────────────────────────────────────────────────────────
score_matrix = np.mean(seed_preds, axis=0).astype(np.float32)
print(f"\nScore matrix: {score_matrix.shape}, range [{score_matrix.min():.4f}, {score_matrix.max():.4f}]")

# ── Brain Test Set AUPRC (zero-shot cross-tissue evaluation) ─────────────────
print("\n=== Brain Test Set AUPRC (zero-shot, gene-stratified) ===")
print("Building brain ground truth labels from gene annotations...")

# Load brain gene names
br_gene_path = BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy'
br_gene_names_raw = np.load(br_gene_path, allow_pickle=True)
br_gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in br_gene_names_raw]
n_br = len(br_gene_names)

# Build ground truth for brain isoforms
Y_br = np.zeros((n_br, N_GO), dtype=np.float32)
for j, go_id in enumerate(go_ids):
    for i, gene in enumerate(br_gene_names):
        if gene in gene_go and go_id in gene_go[gene]:
            Y_br[i, j] = 1.0

print(f"  Brain isoforms with ≥1 GO annotation: "
      f"{(Y_br.sum(axis=1)>0).sum()} / {n_br} "
      f"({100*(Y_br.sum(axis=1)>0).mean():.1f}%)")

# Evaluate brain AUPRC using ensemble score_matrix
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
                     'auprc_brain': round(auprc, 4),
                     'method': 'prism_v15d_joint73_brain_zeroshot'})
    else:
        meta.append({'go': go_id, 'name': go_name,
                     'n_pos_muscle_train': n_pos_train,
                     'n_pos_brain': n_pos_brain,
                     'auprc_brain': None,
                     'method': 'prism_v15d_joint73_brain_zeroshot'})

valid_meta = [m for m in meta if m['auprc_brain'] is not None]
aurpcs_valid = [m['auprc_brain'] for m in valid_meta]
print(f"Terms with AUPRC: {len(valid_meta)}/{N_GO}")
print(f"Macro AUPRC (brain, zero-shot): {np.mean(aurpcs_valid):.4f}")
print(f"Median AUPRC:                   {np.median(aurpcs_valid):.4f}")
print(f"AUPRC > 0.5:                    {sum(a>0.5 for a in aurpcs_valid)}/{len(aurpcs_valid)}")
print(f"AUPRC > 0.4:                    {sum(a>0.4 for a in aurpcs_valid)}/{len(aurpcs_valid)}")
print(f"\nTop 5:")
for m in sorted(valid_meta, key=lambda x: x['auprc_brain'], reverse=True)[:5]:
    print(f"  {m['go']} ({m['name'][:40]}): {m['auprc_brain']}")
print(f"\nBottom 5:")
for m in sorted(valid_meta, key=lambda x: x['auprc_brain'])[:5]:
    print(f"  {m['go']} ({m['name'][:40]}): {m['auprc_brain']}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)")

# ── Save outputs ──────────────────────────────────────────────────────────────
out_npy = OUT_DIR / 'brain_full_scores_extended_v15d.npy'
np.save(out_npy, score_matrix)
print(f"Saved: {out_npy}")

out_meta = OUT_DIR / 'brain_full_extended_v15d_meta.json'
with open(out_meta, 'w') as f:
    json.dump({'model': 'prism_v15d_joint73',
               'evaluation': 'brain_test_zeroshot',
               'go_terms': go_ids, 'go_names': go_names,
               'n_isoforms_brain': n_br, 'n_go': N_GO,
               'macro_auprc_brain': round(float(np.mean(aurpcs_valid)), 4),
               'median_auprc_brain': round(float(np.median(aurpcs_valid)), 4),
               'n_auprc_gt05': int(sum(a > 0.5 for a in aurpcs_valid)),
               'n_auprc_gt04': int(sum(a > 0.4 for a in aurpcs_valid)),
               'per_go': meta}, f, indent=2)
print(f"Saved meta: {out_meta}")

# ── Copy to demo dir (overwrite v15d version) ─────────────────────────────────
import shutil
demo_out = DEMO_DIR / 'brain_full_extended_scores.npy'
shutil.copy(out_npy, demo_out)
print(f"Copied to demo: {demo_out}")
print("\nAll done.")
