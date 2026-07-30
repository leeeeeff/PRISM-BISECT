#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_d_finetune_comparison_v2_truebrain.py
-------------------------------------------
TISSUE-MISLABELING BUGFIX RERUN (2026-07-14).
Original exp_d_finetune_comparison.py loaded my_gene_list_fixed.npy / my_isoform_list_fixed.npy
/ transcripts.fasta.transdecoder.pep as its "test" set -- these are MUSCLE data (BambuTx IDs,
36748 isoforms), despite being described in the manuscript as "brain zero-shot". This rerun
re-points the TEST side only at the TRUE brain isoform set: brain_full_gene_names.npy /
brain_full_ids.npy (63994 isoforms / 18514 unique genes, IsoQuant IDs like A1BG-204) and the
TRUE brain protein FASTA built in Step 1 of this rerun
(reports/truebrain_rerun_20260714/data/brain_full_proteins.fa, 53826/63994 = 84.1%
protein-coding coverage, matching brain_full_mask.npy exactly). Training side
(train_gene_list.npy, train_proteins.fasta) is UNCHANGED.

Original (mislabeled-as-brain, actually muscle): exp_d_finetune_comparison.py, backed up
at exp_d_finetune_comparison_backup_20260714.py before this rerun was created.

exp_d_finetune_comparison.py
----------------------------
Fine-tuning vs frozen ESM-2 comparison for brain zero-shot evaluation.

Question: Does fine-tuning ESM-2 on muscle 82 MF GO labels improve or degrade
brain zero-shot generalization vs frozen ESM-2 + δ_layer approach?

Experiments:
  D0 [reference]: Frozen ESM-2 → pre-computed L30 → PRISM MLP   (AUPRC ~0.596)
  D1: Fine-tune last 5 ESM-2 layers → extract L30 → PRISM MLP
  D2: Fine-tune all 30 ESM-2 layers → extract L30 → PRISM MLP
  [ref] v17f:  Frozen ESM-2 + δ_layer + T_ψ                      (AUPRC 0.717)

Key argument: If fine-tuning hurts zero-shot transfer, frozen + δ_layer is the
right approach for cross-tissue generalization.
"""

import os, re, gzip, json, time
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
import warnings; warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR  = '../data'
BRAIN_DIR = '../data/brain_isoquant_esm2/full'
ID_DIR    = '../data/raw_data/data/id_lists'
ANNOT_DIR = '../data/raw_data/data/annotations'
SEQ_FASTA = '../results_isoform/features/train_proteins.fasta'
SEQ_PEP   = '../../reports/truebrain_rerun_20260714/data/brain_full_proteins.fa'  # TRUE BRAIN, 53826/63994=84.1% coding coverage
OUT_DIR   = '../../reports/truebrain_rerun_20260714/exp_d_finetune'
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = [42, 7, 13, 21, 99]
EPOCHS_FT  = 5    # fine-tuning epochs for ESM-2
EPOCHS_MLP = 60   # MLP training epochs
BATCH_EMB  = 32   # batch size for embedding extraction
BATCH_FT   = 4    # batch for ESM-2 fine-tuning (memory-efficient)
GRAD_ACCUM = 8    # effective batch = 4 × 8 = 32
LR_ESM     = 5e-6 # low LR for ESM-2 to prevent catastrophic forgetting
LR_MLP_FT  = 1e-3 # LR for MLP head during fine-tuning

print("=" * 65)
print("  Experiment D: Fine-tuning vs Frozen ESM-2 Comparison")
print("  Brain zero-shot 82 MF GO terms")
print("=" * 65)

# ── Sequence parsing ─────────────────────────────────────────────
def clean(raw):
    s = str(raw)
    for c in ["b'", "'", '"', ' ']: s = s.replace(c, '')
    return s

def parse_fasta(path, max_len=1022):
    records = {}; cur_id = None; cur_seq = []
    def flush():
        if cur_id is None: return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if seq: records[cur_id] = seq[:max_len]
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m = re.match(r'>(\S+)', line)
                cur_id = m.group(1) if m else None
            else:
                cur_seq.append(line)
    flush()
    return records

TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}

def parse_pep_file(pep_path, max_len=1022):
    records = {}; cur_id = cur_meta = None; cur_seq = []
    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None: return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq: return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank, score, length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq)
    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m_id   = re.match(r'>(\S+)', line)
                m_type = re.search(r'ORF type:(\S+)', line)
                m_sc   = re.search(r'score=([\d.]+)', line)
                m_len  = re.search(r'len:(\d+)', line)
                if not m_id: cur_id = None; continue
                raw_id = m_id.group(1)
                cur_id = re.sub(r'\.p\d+$', '', raw_id)
                orf_type = m_type.group(1) if m_type else 'internal'
                rank = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, float(m_sc.group(1)) if m_sc else 0.0,
                            int(m_len.group(1)) if m_len else 0)
            else:
                cur_seq.append(line)
    flush()
    return {k: v[3][:max_len] for k, v in records.items()}

# ── Data loading ─────────────────────────────────────────────────
print("\n[1] Loading GO labels and isoform lists...")

tr_genes_raw = np.load(f'{ID_DIR}/train_gene_list.npy', allow_pickle=True)
tr_genes     = [clean(g) for g in tr_genes_raw]
tr_iso_raw   = np.load(f'{ID_DIR}/train_isoform_list.npy', allow_pickle=True)
tr_isos      = [clean(x) for x in tr_iso_raw]

ENSG2SYM = {}
with open(f'{ID_DIR}/ensembl_to_symbol.txt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 5: ENSG2SYM[p[0]] = p[4]

# TRUE BRAIN: gene names already symbols (e.g. 'A1BG'), no ENSG2SYM mapping needed.
te_genes_raw = np.load(f'{BRAIN_DIR}/brain_full_gene_names.npy', allow_pickle=True)
te_iso_raw   = np.load(f'{BRAIN_DIR}/brain_full_ids.npy', allow_pickle=True)
te_sym_list  = [clean(g) for g in te_genes_raw]
te_isos      = [clean(x) for x in te_iso_raw]

sym2id = {}
with gzip.open(f'{ANNOT_DIR}/Homo_sapiens.gene_info.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 2:
            sym2id[p[2]] = p[1]
            if len(p) > 4 and p[4] != '-':
                for syn in p[4].split('|'):
                    if syn not in sym2id: sym2id[syn] = p[1]
tr_ids = [sym2id.get(g, g) for g in tr_genes]
tr_id_set = set(tr_ids)

go_genes_tr = defaultdict(set); go_genes_all = defaultdict(set)
with gzip.open(f'{ANNOT_DIR}/gene2go.gz', 'rt') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] != '9606': continue
        if p[7] != 'Function': continue
        go_genes_all[p[2]].add(p[1])
        if p[1] in tr_id_set: go_genes_tr[p[2]].add(p[1])

mf_terms = []
with open('../../reports/v_expanded_gomf/mf_domain_vs_prism.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 6: mf_terms.append(p[0])

tr_sym2idx = defaultdict(list)
for i, g in enumerate(tr_genes): tr_sym2idx[g].append(i)

def build_Y_tr(go_id):
    pos_ids  = go_genes_tr[go_id]
    pos_syms = {g for g, gid in zip(tr_genes, tr_ids) if gid in pos_ids}
    y = np.zeros(len(tr_genes), dtype=np.float32)
    for sym in pos_syms:
        for idx in tr_sym2idx.get(sym, []): y[idx] = 1.0
    return y

def build_Y_te(go_id):
    pos_ids = go_genes_all[go_id]
    return np.array([1.0 if sym2id.get(s, '__') in pos_ids else 0.0
                     for s in te_sym_list], dtype=np.float32)

Y_tr = np.stack([build_Y_tr(go) for go in mf_terms], axis=1)
Y_te = np.stack([build_Y_te(go)  for go in mf_terms], axis=1)
valid_mask = Y_te.sum(0) >= 2
print(f"  {len(mf_terms)} MF terms  |  train={len(tr_genes)}  test={len(te_sym_list)}")
print(f"  valid MF terms (test≥2): {valid_mask.sum()}")

# ── H2 layer groups (for reporting) ──────────────────────────────
H2_LAYERS = {}
with open('../../reports/v_expanded_gomf/h2_layer_classification.tsv') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 12: H2_LAYERS[p[0]] = p[11]

layer_groups = defaultdict(list)
for i, go in enumerate(mf_terms):
    if valid_mask[i]: layer_groups[H2_LAYERS.get(go, 'Other')].append(i)
L2_idxs   = layer_groups.get('L2_Structural', [])
L4_idxs   = layer_groups.get('L4_CellState', [])
all_valid  = [i for i in range(len(mf_terms)) if valid_mask[i]]

print(f"  L2_Structural: {len(L2_idxs)}  L4_CellState: {len(L4_idxs)}  All valid: {len(all_valid)}")

# ── Load protein sequences ────────────────────────────────────────
print("\n[2] Parsing protein sequences...")
t0 = time.time()
train_seqs_raw = parse_fasta(SEQ_FASTA)
test_seqs_raw  = parse_pep_file(SEQ_PEP)
print(f"  Training sequences in FASTA: {len(train_seqs_raw)}")
print(f"  Test sequences in PEP:       {len(test_seqs_raw)}")

# Map to array order: training isoforms → NM_ IDs
tr_seq_list = []
tr_found = 0
for iso in tr_isos:
    # iso is NM_XXXXXX.X → try exact, then base
    base_id = iso.split('.')[0]
    seq = train_seqs_raw.get(iso) or train_seqs_raw.get(base_id)
    if seq:
        tr_seq_list.append(seq)
        tr_found += 1
    else:
        tr_seq_list.append('M')  # placeholder for missing

# Map test isoforms (BambuTx IDs and ENST) → sequences
te_seq_list = []
te_found = 0
for iso in te_isos:
    base_id = re.sub(r'\.\d+$', '', iso)
    seq = test_seqs_raw.get(iso) or test_seqs_raw.get(base_id)
    if seq:
        te_seq_list.append(seq)
        te_found += 1
    else:
        te_seq_list.append('M')  # placeholder

print(f"  Train sequences found: {tr_found}/{len(tr_isos)} ({100*tr_found/len(tr_isos):.1f}%)")
print(f"  Test sequences found:  {te_found}/{len(te_isos)} ({100*te_found/len(te_isos):.1f}%)")
print(f"  Sequence parsing: {time.time()-t0:.1f}s")

# ── PyTorch ESM-2 utilities ───────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import esm as esm_lib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\n  Using device: {device}")

def extract_l30_embeddings(model, batch_converter, seq_list, batch_size=BATCH_EMB):
    """Extract L30 mean-pooled embeddings without grad."""
    model.eval()
    all_embs = []
    for start in range(0, len(seq_list), batch_size):
        batch = [(f'seq{start+i}', s) for i, s in enumerate(seq_list[start:start+batch_size])]
        with torch.no_grad():
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[30], return_contacts=False)
        reps = out['representations'][30]  # (B, L+2, D)
        for i, (_, s) in enumerate(batch):
            seq_len = min(len(s), tokens.shape[1]-2)
            emb = reps[i, 1:seq_len+1, :].mean(dim=0).cpu().float().numpy()
            all_embs.append(emb)
    return np.array(all_embs, dtype=np.float32)

# ── Reference: load pre-computed L30 (PRISM baseline) ────────────
print("\n[3] Loading pre-computed reference embeddings (D0 = PRISM baseline)...")
X_l30_tr_ref = np.load(f'{DATA_DIR}/esm2_train_human_layer30_t30_150M.npy').astype(np.float32)
X_l30_te_ref = np.load(f'{BRAIN_DIR}/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
print(f"  train: {X_l30_tr_ref.shape}  test: {X_l30_te_ref.shape}")

# ── TensorFlow PRISM MLP (consistent with rest of codebase) ──────
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Memory growth: TF allocates GPU memory on demand only (not pre-allocate all),
# allowing PyTorch to share the same GPU for ESM-2 fine-tuning.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def build_prism_mlp(input_dim, n_terms):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(256, activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(128, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_terms, activation='sigmoid')(x)
    return tf.keras.Model(inp, out)

def train_and_eval_prism(X_tr, Y_tr_in, X_te, Y_te_in, seeds=SEEDS):
    """Train PRISM MLP (5-seed ensemble), return per-term AUPRC for test."""
    scaler = MaxAbsScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    preds_te = []
    for seed in seeds:
        tf.random.set_seed(seed); np.random.seed(seed)
        model = build_prism_mlp(X_tr_s.shape[1], Y_tr_in.shape[1])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
        )
        model.fit(X_tr_s, Y_tr_in, epochs=EPOCHS_MLP, batch_size=512,
                  verbose=0, validation_split=0.0)
        preds_te.append(model.predict(X_te_s, verbose=0))
    pred_avg = np.mean(preds_te, axis=0)
    aps = []
    for k in range(Y_te_in.shape[1]):
        if Y_te_in[:, k].sum() >= 2:
            aps.append(average_precision_score(Y_te_in[:, k], pred_avg[:, k]))
    return np.array(aps), pred_avg

def report_auprc(aps_all, label):
    """Report All MF, L2_Structural, L4_CellState AUPRC."""
    valid_idx = [i for i in all_valid]
    l2_aps    = [aps_all[i] for i in L2_idxs  if i < len(aps_all)]
    l4_aps    = [aps_all[i] for i in L4_idxs  if i < len(aps_all)]
    all_aps   = [aps_all[i] for i in all_valid if i < len(aps_all)]
    print(f"  {label:<40}  All={np.mean(all_aps):.4f}  L2={np.mean(l2_aps) if l2_aps else float('nan'):.4f}  L4={np.mean(l4_aps) if l4_aps else float('nan'):.4f}")
    return np.mean(all_aps), np.mean(l2_aps) if l2_aps else float('nan')

# ── D0: PRISM baseline (pre-computed, already known) ─────────────
print("\n[D0] PRISM baseline (pre-computed L30, frozen ESM-2)...")
t0 = time.time()
aps_d0, _ = train_and_eval_prism(X_l30_tr_ref, Y_tr, X_l30_te_ref, Y_te)
all_d0, l2_d0 = report_auprc(aps_d0, 'D0: Frozen ESM-2 L30 (PRISM)')
print(f"  [D0 done in {time.time()-t0:.0f}s]")

results = {'D0_frozen_prism': {'all_mf': float(all_d0), 'l2_structural': float(l2_d0)}}

# ── ESM-2 fine-tuning ─────────────────────────────────────────────
print("\n[4] Loading ESM-2 150M model...")
t0 = time.time()
esm_model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
esm_model = esm_model.to(device)
batch_converter = alphabet.get_batch_converter()
print(f"  ESM-2 loaded in {time.time()-t0:.1f}s  params={sum(p.numel() for p in esm_model.parameters())/1e6:.1f}M")

def run_finetune_experiment(n_unfrozen_layers, exp_label):
    """
    Fine-tune last n_unfrozen_layers of ESM-2 on muscle 82 MF labels.
    Extract L30 → train PRISM MLP → evaluate brain zero-shot.
    n_unfrozen_layers=0 → frozen (D0); n_unfrozen_layers=30 → full fine-tuning (D2)
    """
    print(f"\n[{exp_label}] Fine-tuning last {n_unfrozen_layers} ESM-2 layers...")
    t0 = time.time()

    # Reload fresh ESM-2 to avoid contamination between experiments
    ft_model, ft_alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    ft_model = ft_model.to(device)
    ft_batch_converter = ft_alphabet.get_batch_converter()

    # Freeze / unfreeze
    for name, param in ft_model.named_parameters():
        param.requires_grad = False

    if n_unfrozen_layers > 0:
        # Identify last n_unfrozen_layers transformer blocks
        # ESM-2 layers named: layers.0, layers.1, ..., layers.29
        for i in range(30 - n_unfrozen_layers, 30):
            for param in ft_model.layers[i].parameters():
                param.requires_grad = True
        # Also unfreeze emb_layer_norm_after (final layer norm)
        for param in ft_model.emb_layer_norm_after.parameters():
            param.requires_grad = True

    n_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f}M / {sum(p.numel() for p in ft_model.parameters())/1e6:.1f}M")

    # MLP head for fine-tuning (not used at eval time — just for supervised signal)
    ft_head = nn.Sequential(
        nn.Linear(640, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, len(mf_terms)), nn.Sigmoid()
    ).to(device)

    params_to_opt = list(ft_model.parameters() if n_unfrozen_layers == 30
                         else [p for p in ft_model.parameters() if p.requires_grad])
    params_to_opt += list(ft_head.parameters())
    optimizer = optim.Adam([
        {'params': [p for p in ft_model.parameters() if p.requires_grad], 'lr': LR_ESM},
        {'params': ft_head.parameters(), 'lr': LR_MLP_FT}
    ])

    # Focal loss (γ=2) in PyTorch
    def focal_loss(pred, target, gamma=2.0):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt  = torch.where(target == 1, pred, 1 - pred)
        fl  = ((1 - pt) ** gamma) * bce
        return fl.mean()

    # Build training batches from sequence list
    # Filter to isoforms with known sequences
    valid_tr_idx = [i for i, s in enumerate(tr_seq_list) if len(s) > 1]
    print(f"  Training isoforms with sequences: {len(valid_tr_idx)}/{len(tr_seq_list)}")

    Y_tr_pt = torch.tensor(Y_tr[valid_tr_idx], dtype=torch.float32).to(device)

    # Fine-tuning loop
    ft_model.train()
    ft_head.train()
    print(f"  Fine-tuning for {EPOCHS_FT} epochs (batch={BATCH_FT}, grad_accum={GRAD_ACCUM})...")

    for epoch in range(EPOCHS_FT):
        perm = np.random.permutation(len(valid_tr_idx))
        epoch_loss = 0.0; n_batches = 0
        optimizer.zero_grad()

        for step_start in range(0, len(perm), BATCH_FT):
            batch_idx = perm[step_start:step_start + BATCH_FT]
            real_idx  = [valid_tr_idx[i] for i in batch_idx]
            seqs_batch = [(f's{j}', tr_seq_list[j]) for j in real_idx]

            try:
                _, _, tokens = ft_batch_converter(seqs_batch)
                tokens = tokens.to(device)
                out = ft_model(tokens, repr_layers=[30], return_contacts=False)
                reps = out['representations'][30]

                # Mean pool
                embs = []
                for i, (_, s) in enumerate(seqs_batch):
                    seq_len = min(len(s), tokens.shape[1]-2)
                    embs.append(reps[i, 1:seq_len+1, :].mean(dim=0))
                emb_batch = torch.stack(embs)  # (B, 640)

                y_batch = Y_tr_pt[batch_idx]
                pred = ft_head(emb_batch)
                loss = focal_loss(pred, y_batch) / GRAD_ACCUM
                loss.backward()
                epoch_loss += loss.item() * GRAD_ACCUM
                n_batches += 1

                # Gradient accumulation
                accum_step = (step_start // BATCH_FT + 1)
                if accum_step % GRAD_ACCUM == 0 or step_start + BATCH_FT >= len(perm):
                    nn.utils.clip_grad_norm_(
                        [p for p in ft_model.parameters() if p.requires_grad] +
                        list(ft_head.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise e

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        print(f"    Epoch {epoch+1}/{EPOCHS_FT}  loss={avg_loss:.4f}  [{elapsed:.0f}s]")

    # Extract L30 embeddings from fine-tuned model
    print(f"  Extracting fine-tuned L30 for training set ({len(tr_seq_list)} isoforms)...")
    X_tr_ft = extract_l30_embeddings(ft_model, ft_batch_converter, tr_seq_list)
    print(f"  Extracting fine-tuned L30 for brain test set ({len(te_seq_list)} isoforms)...")
    X_te_ft = extract_l30_embeddings(ft_model, ft_batch_converter, te_seq_list)
    print(f"  Extraction done: train={X_tr_ft.shape}  test={X_te_ft.shape}")

    # Save embeddings
    np.save(f'{OUT_DIR}/ft_{exp_label}_train_l30.npy', X_tr_ft)
    np.save(f'{OUT_DIR}/ft_{exp_label}_test_l30.npy',  X_te_ft)

    # Train PRISM MLP on fine-tuned embeddings, evaluate brain zero-shot
    print(f"  Training PRISM MLP on fine-tuned embeddings...")
    aps, _ = train_and_eval_prism(X_tr_ft, Y_tr, X_te_ft, Y_te)
    all_auprc, l2_auprc = report_auprc(aps, f'{exp_label}: Fine-tune {n_unfrozen_layers} layers')

    # Clean up GPU memory
    del ft_model, ft_head
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    print(f"  [{exp_label} done in {total_time:.0f}s = {total_time/60:.1f}min]")
    return float(all_auprc), float(l2_auprc)

# Run experiments
print("\n" + "=" * 65)
print("  Running fine-tuning experiments (GPU0)")
print("=" * 65)

# D1: Fine-tune last 5 layers (partial fine-tuning)
all_d1, l2_d1 = run_finetune_experiment(5, 'D1')
results['D1_finetune_last5'] = {'all_mf': all_d1, 'l2_structural': l2_d1}

# D2: Full fine-tuning (all 30 layers)
all_d2, l2_d2 = run_finetune_experiment(30, 'D2')
results['D2_finetune_full'] = {'all_mf': all_d2, 'l2_structural': l2_d2}

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Experiment D: RESULTS SUMMARY")
print("=" * 65)
print(f"  {'Method':<45} {'All MF':>8} {'L2_Struct':>10}")
print(f"  {'-'*65}")
print(f"  {'D0: Frozen ESM-2 L30 (PRISM)':<45} {all_d0:>8.4f} {l2_d0:>10.4f}")
print(f"  {'D1: Fine-tune last 5 ESM-2 layers':<45} {all_d1:>8.4f} {l2_d1:>10.4f}")
print(f"  {'D2: Full ESM-2 fine-tuning (all 30 layers)':<45} {all_d2:>8.4f} {l2_d2:>10.4f}")
print(f"  {'[ref] v17f: Frozen ESM-2 + δ_layer + T_ψ':<45} {'0.717':>8} {'0.622':>10}")
print(f"  {'-'*65}")

# Compute deltas from D0 baseline
print(f"\n  Δ from PRISM baseline (D0 = {all_d0:.4f}):")
print(f"    D1 (last-5 fine-tune): {all_d1 - all_d0:+.4f}")
print(f"    D2 (full fine-tune):   {all_d2 - all_d0:+.4f}")
print(f"    v17f (δ_layer):        +{0.717 - all_d0:.4f}")

# Interpretation
print("\n  INTERPRETATION:")
if all_d1 < all_d0 and all_d2 < all_d0:
    print("  [CONFIRMS HYPOTHESIS] Both fine-tuning configurations DECREASE brain zero-shot AUPRC.")
    print("  Fine-tuning ESM-2 on muscle GO labels distorts general representations,")
    print("  reducing cross-tissue generalizability. Frozen + δ_layer is the correct approach.")
elif all_d1 < all_d0 or all_d2 < all_d0:
    print("  [PARTIAL] Deeper fine-tuning shows degradation. Shallow fine-tuning may help slightly.")
else:
    print("  [UNEXPECTED] Fine-tuning helps. Review results carefully before reporting.")

results['v17f_reference'] = {'all_mf': 0.717, 'l2_structural': 0.622}
results['delta_from_prism'] = {
    'D1': all_d1 - all_d0,
    'D2': all_d2 - all_d0,
    'v17f': 0.717 - all_d0
}

json.dump(results, open(f'{OUT_DIR}/finetune_results.json', 'w'), indent=2)
print(f"\n[Saved] {OUT_DIR}/finetune_results.json")
print("\n" + "=" * 65)
print("  Experiment D: COMPLETE")
print("=" * 65)
