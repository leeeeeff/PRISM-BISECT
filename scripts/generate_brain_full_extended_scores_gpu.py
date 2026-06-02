#!/usr/bin/env python3
"""
generate_brain_full_extended_scores_gpu.py
==========================================
전체 63,994 brain isoform × 73 brain GO term 스코어 행렬 생성 (GPU 버전).

v15d_bp_clean 아키텍처 (PyTorch):
  ESM-2 640-dim → Linear(256,ReLU) → BN → Dropout(0.3)
                → Linear(128,ReLU) → Dropout(0.2)
                → Linear(64,ReLU)
                → Linear(73,sigmoid)
  Loss: Binary Focal Cross-Entropy (gamma=2)
  GPU: GPU 1 (23GB free)
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

# ── GPU 설정: GPU 1 사용 ──────────────────────────────────────────────────────
device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Free memory: {torch.cuda.mem_get_info(device)[0] / 1e9:.1f} GB")

BASE     = Path('/home/welcome1/sw1686/DIFFUSE')
OUT_DIR  = BASE / 'reports'
DEMO_DIR = BASE / 'prism_app/data/demo'
OUT_DIR.mkdir(exist_ok=True)

N_SEEDS = 5

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
N_GO = len(go_ids)
print(f"GO terms: {N_GO}")

# ── Load ESM-2 embeddings (full brain, 63994 × 640) ─────────────────────────
print("Loading ESM-2 embeddings...")
X = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_t30_150M.npy').astype(np.float32)
ids = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
gene_names = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy', allow_pickle=True)
gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in gene_names]
n_total = len(ids)
print(f"  ESM-2: {X.shape}, IDs: {n_total}")

# ── Load muscle ESM-2 embeddings for training ────────────────────────────────
print("Loading muscle ESM-2 embeddings (training)...")
X_muscle = np.load(BASE / 'hMuscle/data/esm2_train_human_t30_150M.npy').astype(np.float32)
muscle_gene_names = np.load(BASE / 'hMuscle/data/raw_data/data/id_lists/train_gene_list.npy', allow_pickle=True)
muscle_gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in muscle_gene_names]
n_muscle = len(muscle_gene_names)
print(f"  Muscle ESM-2: {X_muscle.shape}")

# ── Load gene GO annotations ─────────────────────────────────────────────────
print("Loading gene GO annotations...")
gene_go = {}
annot_file = BASE / 'hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'
with open(annot_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            gene_go[parts[0]] = set(parts[1:])
print(f"  Loaded {len(gene_go)} gene annotations")

# ── Normalize features ────────────────────────────────────────────────────────
print("Normalizing features...")
scaler = StandardScaler()
# Fit on brain (full), transform both
X_n = scaler.fit_transform(X)
X_muscle_n = scaler.transform(X_muscle)

# ── PyTorch model: v15d_bp_clean architecture ─────────────────────────────────
class PRISMModel(nn.Module):
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


def focal_loss(preds, targets, gamma=2.0):
    bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
    pt = torch.where(targets == 1, preds, 1 - preds)
    fl = ((1 - pt) ** gamma) * bce
    return fl.mean()


def build_labels(gene_list, go_term):
    labels = np.zeros(len(gene_list), dtype=np.float32)
    for i, gene in enumerate(gene_list):
        if gene in gene_go and go_term in gene_go[gene]:
            labels[i] = 1.0
    return labels


# ── Gene-stratified 5-fold split on MUSCLE training data ─────────────────────
muscle_gene_arr = np.array(muscle_gene_names)
unique_genes = np.unique(muscle_gene_arr)
rng = np.random.default_rng(42)
gene_fold = {g: rng.integers(0, 5) for g in unique_genes}
fold_arr = np.array([gene_fold[g] for g in muscle_gene_names])

# ── Training ──────────────────────────────────────────────────────────────────
print(f"Training {N_GO} PRISM probes (v15d_bp_clean, 5-seed ensemble) on {n_muscle} muscle isoforms...")
print(f"  Predicting on {n_total} brain isoforms")
t0 = time.time()

score_matrix = np.zeros((n_total, N_GO), dtype=np.float32)
meta = []

X_tr_tensor = torch.tensor(X_muscle_n, dtype=torch.float32)
X_pred_tensor = torch.tensor(X_n, dtype=torch.float32).to(device)

for j, (go_id, go_name) in enumerate(zip(go_ids, go_names)):
    y_muscle = build_labels(muscle_gene_names, go_id)
    n_pos = int(y_muscle.sum())

    if n_pos < 5:
        score_matrix[:, j] = n_pos / n_muscle
        meta.append({'go': go_id, 'name': go_name, 'n_pos': n_pos,
                     'auprc': None, 'method': 'constant'})
        continue

    # 5-seed ensemble: train on ALL muscle data, predict on brain
    seed_preds = []
    auprc_folds = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 137)
        model = PRISMModel(n_out=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # Validation fold for AUPRC (fold == seed % 5)
        val_fold = seed % 5
        tr_mask = fold_arr != val_fold
        va_mask = fold_arr == val_fold

        y_tensor_full = torch.tensor(y_muscle, dtype=torch.float32).unsqueeze(1)
        y_tr = y_tensor_full[tr_mask].to(device)
        X_tr = X_tr_tensor[tr_mask].to(device)
        y_va = y_tensor_full[va_mask].to(device)
        X_va = X_tr_tensor[va_mask].to(device)

        dataset = TensorDataset(X_tr, y_tr)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)

        best_val_loss = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(100):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = focal_loss(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = focal_loss(model(X_va), y_va).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= 5:
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Validation AUPRC
        model.eval()
        with torch.no_grad():
            va_preds = model(X_va).cpu().numpy().squeeze()
        va_labels = y_va.cpu().numpy().squeeze()
        if va_labels.sum() > 0:
            auprc_folds.append(average_precision_score(va_labels, va_preds))

        # Predict on all brain isoforms
        with torch.no_grad():
            brain_preds = model(X_pred_tensor).cpu().numpy().squeeze()
        seed_preds.append(brain_preds)

    # Ensemble average
    score_matrix[:, j] = np.mean(seed_preds, axis=0).astype(np.float32)
    macro_auprc = float(np.mean(auprc_folds)) if auprc_folds else None

    meta.append({'go': go_id, 'name': go_name, 'n_pos': n_pos,
                 'auprc': round(macro_auprc, 4) if macro_auprc else None,
                 'method': 'prism_v15d'})

    if (j + 1) % 5 == 0 or j == 0:
        elapsed = time.time() - t0
        auprc_str = f"{macro_auprc:.3f}" if macro_auprc else "N/A"
        print(f"  [{j+1}/{N_GO}] {go_id} ({go_name[:28]}): n_pos={n_pos}, "
              f"AUPRC={auprc_str} | {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Score matrix: {score_matrix.shape}, range [{score_matrix.min():.4f}, {score_matrix.max():.4f}]")

# ── Save outputs ──────────────────────────────────────────────────────────────
out_npy = OUT_DIR / 'brain_full_scores_extended.npy'
np.save(out_npy, score_matrix)
print(f"Saved: {out_npy}")

out_meta = OUT_DIR / 'brain_full_extended_meta.json'
with open(out_meta, 'w') as f:
    json.dump({'go_terms': go_ids, 'go_names': go_names,
               'n_isoforms': n_total, 'n_go': N_GO,
               'per_go': meta}, f, indent=2)
print(f"Saved: {out_meta}")

# ── Copy to demo dir ───────────────────────────────────────────────────────────
import shutil
demo_out = DEMO_DIR / 'brain_full_extended_scores.npy'
shutil.copy(out_npy, demo_out)
print(f"Copied to demo: {demo_out}")

# ── Save gene IDs for brain_extended ─────────────────────────────────────────
gene_ids_out = DEMO_DIR / 'brain_full_extended_gene_ids.npy'
np.save(gene_ids_out, np.array(gene_names, dtype=str))
print(f"Saved gene IDs: {gene_ids_out}")

# ── Save full isoform IDs ─────────────────────────────────────────────────────
ids_out = DEMO_DIR / 'brain_full_extended_ids.npy'
np.save(ids_out, np.array(ids, dtype=str))
print(f"Saved IDs: {ids_out}")

# ── Derive types from IDs ─────────────────────────────────────────────────────
def id_to_type(isoform_id):
    s = str(isoform_id).lower()
    if s.endswith('.nnic'): return 'nnic'
    if s.endswith('.nic'):  return 'nic'
    return 'known'

types = np.array([id_to_type(i) for i in ids], dtype=str)
print("Type distribution:", dict(Counter(types)))
types_out = DEMO_DIR / 'brain_full_extended_types.npy'
np.save(types_out, types)
print(f"Saved types: {types_out}")

# ── Summary ───────────────────────────────────────────────────────────────────
valid_meta = [m for m in meta if m['auprc'] is not None]
if valid_meta:
    aurpcs = [m['auprc'] for m in valid_meta]
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"GO terms with AUPRC: {len(valid_meta)}/{N_GO}")
    print(f"Macro AUPRC: {np.mean(aurpcs):.4f}")
    print(f"Median AUPRC: {np.median(aurpcs):.4f}")
    print(f"Top 5 GO terms by AUPRC:")
    for m in sorted(valid_meta, key=lambda x: x['auprc'], reverse=True)[:5]:
        print(f"  {m['go']} ({m['name'][:40]}): AUPRC={m['auprc']}")

print("\nAll done.")
