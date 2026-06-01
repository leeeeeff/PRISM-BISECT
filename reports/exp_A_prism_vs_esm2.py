"""
Experiment A: PRISM representation (18-dim) vs raw ESM-2 (640-dim)
for brain 73 GO term linear probes.

If PRISM trained representation outperforms raw ESM-2 → PRISM training adds value.
If not → ESM-2 attribution problem confirmed.

Representations compared:
  A: ESM-2 L27 (640-dim) — already done (AUPRC=0.610), used as baseline
  B: PRISM 18-dim output (sigmoid scores, brain isoforms)
  C: PRISM 18-dim + ESM-2 640-dim concatenated (658-dim)
"""
import numpy as np, gzip, csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

BASE = '/home/welcome1/sw1686/DIFFUSE'

print("Loading data...")
# ESM-2 L27 (640-dim)
X_esm = np.load(f'{BASE}/hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer27_t30_150M.npy').astype(np.float32)
# PRISM 18-dim output (brain isoforms scored by v15d_bp_clean)
X_prism = np.load(f'{BASE}/reports/v15d_brain_eval/brain_full_score_matrix_20260519_2125.npy').astype(np.float32)
# Brain isoform IDs
ids = np.load(f'{BASE}/hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]
n_total = len(ids)
print(f"ESM-2: {X_esm.shape}, PRISM-18: {X_prism.shape}")

# Brain GO term labels (from brain_go_extension_results)
# Re-load GO terms used in expansion
go_csv = f'{BASE}/reports/brain_go_extension_results.csv'
go_terms = []
with open(go_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        go_terms.append((row['go'], row['name']))
go_ids = [g for g, n in go_terms]
go_names = [n for g, n in go_terms]
N_GO = len(go_ids)
print(f"Brain GO terms: {N_GO}")

# Load gene-level GO annotations
gene_go = {}
annot_file = f'{BASE}/hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'
with open(annot_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            gene = parts[0]
            gos = set(parts[1:])
            gene_go[gene] = gos

# Map isoform IDs to gene symbols
def id_to_gene(isoform_id):
    # Known: "GENENAME-NNN" → extract GENENAME
    if '-' in isoform_id and not isoform_id.startswith('transcript'):
        return isoform_id.rsplit('-', 1)[0]
    return None

gene_to_idxs = {}
for i, iso_id in enumerate(ids):
    gene = id_to_gene(iso_id)
    if gene:
        gene_to_idxs.setdefault(gene, []).append(i)

def build_labels(go_term):
    labels = np.zeros(n_total, dtype=int)
    for gene, gos in gene_go.items():
        if go_term in gos and gene in gene_to_idxs:
            for idx in gene_to_idxs[gene]:
                labels[idx] = 1
    return labels

# Normalize representations
scaler_esm = StandardScaler()
X_esm_n = scaler_esm.fit_transform(X_esm)
scaler_prism = StandardScaler()
X_prism_n = scaler_prism.fit_transform(X_prism)
X_concat = np.hstack([X_esm_n, X_prism_n])  # 640+18=658

print("Running linear probes (sample 20 GO terms for speed)...")
# Sample GO terms that cover different AUPRC ranges
sample_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42,  # top and mid
                  45, 50, 55, 60, 65, 68, 70, 1, 3, 7]
sample_indices = [i for i in sample_indices if i < N_GO]

results = []
for gi in sample_indices:
    go_id = go_ids[gi]
    go_name = go_names[gi]
    y = build_labels(go_id)
    if y.sum() < 20:
        continue

    auprc_vals = {}
    for name, X in [('esm2_640', X_esm_n), ('prism_18', X_prism_n), ('concat_658', X_concat)]:
        try:
            lr = LogisticRegression(C=1.0, max_iter=300, solver='lbfgs', n_jobs=-1)
            # 5-fold cross-val
            n = len(y)
            fold_size = n // 5
            preds = np.zeros(n)
            for fold in range(5):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < 4 else n
                tr_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
                va_idx = np.arange(val_start, val_end)
                if y[tr_idx].sum() == 0: continue
                lr.fit(X[tr_idx], y[tr_idx])
                preds[va_idx] = lr.predict_proba(X[va_idx])[:, 1]
            auprc = average_precision_score(y, preds) if y.sum() > 0 else 0.0
        except:
            auprc = 0.0
        auprc_vals[name] = round(auprc, 4)

    results.append({
        'go': go_id,
        'name': go_name[:40],
        'n_pos': int(y.sum()),
        **auprc_vals
    })
    delta_prism = auprc_vals.get('prism_18', 0) - auprc_vals.get('esm2_640', 0)
    delta_concat = auprc_vals.get('concat_658', 0) - auprc_vals.get('esm2_640', 0)
    print(f"  {go_id} ({go_name[:30]}): ESM={auprc_vals.get('esm2_640','?')} | PRISM={auprc_vals.get('prism_18','?')} (Δ{delta_prism:+.3f}) | Concat={auprc_vals.get('concat_658','?')} (Δ{delta_concat:+.3f})")

print("\n=== SUMMARY ===")
if results:
    esm_vals = [r['esm2_640'] for r in results]
    prism_vals = [r['prism_18'] for r in results]
    concat_vals = [r['concat_658'] for r in results]
    print(f"Mean AUPRC — ESM-2(640): {np.mean(esm_vals):.4f}")
    print(f"Mean AUPRC — PRISM(18):  {np.mean(prism_vals):.4f}  (Δ{np.mean(prism_vals)-np.mean(esm_vals):+.4f})")
    print(f"Mean AUPRC — Concat(658):{np.mean(concat_vals):.4f}  (Δ{np.mean(concat_vals)-np.mean(esm_vals):+.4f})")
    better = sum(1 for r in results if r['prism_18'] > r['esm2_640'])
    print(f"PRISM-18 > ESM-2 in {better}/{len(results)} GO terms")
    better_c = sum(1 for r in results if r['concat_658'] > r['esm2_640'])
    print(f"Concat > ESM-2 in {better_c}/{len(results)} GO terms")

import json
out = f'{BASE}/reports/exp_A_results.json'
with open(out, 'w') as f:
    json.dump({'results': results}, f, indent=2)
print(f"\nSaved: {out}")
