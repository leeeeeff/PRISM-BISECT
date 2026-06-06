#!/usr/bin/env python3
"""
generate_brain_full_extended_scores.py
=======================================
Generates PRISM brain extended scores: a 63,994 × 73 GO-term score matrix
covering all brain isoforms (known + novel) in the IsoQuant long-read dataset.

Purpose
-------
Produces the full brain extended score matrix used by the PRISM app and BISECT
downstream analysis. Scores represent per-isoform probability of each of 73
brain-relevant GO biological process terms, learned via logistic regression
probes trained on ESM-2 L30 embeddings.

Method
------
- ESM-2 L30 embeddings (640-dim, pre-computed) are loaded for all 63,994 isoforms.
- One logistic regression probe is trained per GO term using known isoforms
  (gene-level GO annotation propagated to isoform level).
- Trained probes are applied to all 63,994 isoforms (known + novel).

Input files
-----------
hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_t30_150M.npy  -- ESM-2 embeddings (63994 × 640)
hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy             -- isoform IDs (63994,)
hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy      -- gene names (63994,)
prism_app/data/demo/brain_full_scores.npy                            -- existing score matrix (used for GO label lookup)
prism_app/data/demo/brain_full_meta.json                             -- existing meta (73 GO term definitions)

Output files
------------
reports/brain_full_scores_extended.npy   -- full score matrix (63994 × 73, float32)
reports/brain_full_extended_meta.json    -- GO term list and per-term AUPRC validation

Usage
-----
    python scripts/generate_brain_full_extended_scores.py

Notes
-----
- Requires: numpy, scikit-learn, pathlib (stdlib). No GPU needed.
- Runtime: ~5–10 minutes on CPU for 73 probes × 63,994 isoforms.
- Output .npy files are gitignored; re-run to regenerate after new embedding batches.

(이전 한국어 설명: 전체 63,994 brain isoform에 대해 73 brain GO term 스코어 행렬 생성.
방법: ESM-2 L30 임베딩 → GO term별 logistic regression probe → 전체 isoform 예측.
출력: brain_full_scores_extended.npy (63994×73), brain_full_extended_meta.json)
"""
import numpy as np
import json
import time
from pathlib import Path

BASE = Path('/home/welcome1/sw1686/DIFFUSE')
OUT_DIR = BASE / 'reports'
DEMO_DIR = BASE / 'prism_app/data/demo'
OUT_DIR.mkdir(exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import warnings; warnings.filterwarnings('ignore')

# ── Load ESM-2 embeddings (full brain, 63994 × 640) ─────────────────────────
print("Loading ESM-2 embeddings...")
X = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_t30_150M.npy').astype(np.float32)
ids = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
gene_names = np.load(BASE / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy', allow_pickle=True)
gene_names = [x.decode() if isinstance(x, bytes) else str(x) for x in gene_names]
n_total = len(ids)
print(f"  ESM-2: {X.shape}, IDs: {n_total}")

# ── 73 brain GO terms ────────────────────────────────────────────────────────
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

# ── Build isoform-level labels ────────────────────────────────────────────────
# gene_names array contains gene symbol for each isoform (e.g. 'KIF21B')
gene_arr = np.array(gene_names)

def build_labels(go_term):
    labels = np.zeros(n_total, dtype=np.int8)
    for i, gene in enumerate(gene_names):
        if gene in gene_go and go_term in gene_go[gene]:
            labels[i] = 1
    return labels

# ── Normalize features ────────────────────────────────────────────────────────
print("Normalizing features...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_n = scaler.fit_transform(X)

# ── Train LR probes and predict for ALL isoforms ──────────────────────────────
print(f"Training {N_GO} logistic regression probes on {n_total} isoforms...")
t0 = time.time()

score_matrix = np.zeros((n_total, N_GO), dtype=np.float32)
meta = []

# Gene-stratified 5-fold split to avoid gene leakage in validation
unique_genes = np.unique(gene_arr)
rng = np.random.default_rng(42)
gene_fold = {g: rng.integers(0, 5) for g in unique_genes}
fold_arr = np.array([gene_fold[g] for g in gene_names])

for j, (go_id, go_name) in enumerate(zip(go_ids, go_names)):
    y = build_labels(go_id)
    n_pos = int(y.sum())

    if n_pos < 5:
        # Too few positives: use mean positive rate as constant prediction
        score_matrix[:, j] = n_pos / n_total
        meta.append({'go': go_id, 'name': go_name, 'n_pos': n_pos,
                     'auprc': None, 'method': 'constant'})
        continue

    # Gene-stratified 5-fold cross-validated scores (validation only)
    oof_preds = np.zeros(n_total, dtype=np.float32)
    auprc_folds = []

    for fold in range(5):
        tr_mask = fold_arr != fold
        va_mask = fold_arr == fold
        if y[tr_mask].sum() == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs',
                                class_weight='balanced', n_jobs=1)
        lr.fit(X_n[tr_mask], y[tr_mask])
        oof_preds[va_mask] = lr.predict_proba(X_n[va_mask])[:, 1]
        if y[va_mask].sum() > 0:
            auprc_folds.append(average_precision_score(y[va_mask], oof_preds[va_mask]))

    # Final model trained on ALL data for deployment predictions
    lr_full = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs',
                                  class_weight='balanced', n_jobs=1)
    lr_full.fit(X_n, y)
    score_matrix[:, j] = lr_full.predict_proba(X_n)[:, 1].astype(np.float32)

    macro_auprc = float(np.mean(auprc_folds)) if auprc_folds else None
    meta.append({'go': go_id, 'name': go_name, 'n_pos': n_pos,
                 'auprc': round(macro_auprc, 4) if macro_auprc else None,
                 'method': 'lr_probe'})

    if (j + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{j+1}/{N_GO}] {go_id} ({go_name[:30]}): n_pos={n_pos}, "
              f"AUPRC={macro_auprc:.3f if macro_auprc else 'N/A'} | {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
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

# ── Also save gene IDs for brain_extended ─────────────────────────────────────
gene_ids_out = DEMO_DIR / 'brain_full_extended_gene_ids.npy'
np.save(gene_ids_out, np.array(gene_names, dtype=str))
print(f"Saved gene IDs: {gene_ids_out}")

# ── Save full isoform IDs for brain_full_extended ────────────────────────────
ids_out = DEMO_DIR / 'brain_full_extended_ids.npy'
np.save(ids_out, np.array(ids, dtype=str))
print(f"Saved IDs: {ids_out}")

# ── Derive types from IDs ────────────────────────────────────────────────────
def id_to_type(isoform_id):
    s = str(isoform_id).lower()
    if s.endswith('.nnic'): return 'nnic'
    if s.endswith('.nic'):  return 'nic'
    return 'known'

types = np.array([id_to_type(i) for i in ids], dtype=str)
from collections import Counter
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
