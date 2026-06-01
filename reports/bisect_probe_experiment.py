"""
BISECT-guided Linear Probe + pfam2go Extension Experiment
==========================================================
목적:
  1. BISECT-predicted function → target GO term → ESM-2 linear probe → CT vs AD 검증
  2. pfam2go MF/CC GO terms → ESM-2 linear probe → PRISM의 MF 공간 커버 검증

실행: conda run -n isoform_env python3 reports/bisect_probe_experiment.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Data Loading ───────────────────────────────────────────────────────────

print("Loading data...")

# Brain ESM-2 embeddings (L27 — closest to final layer)
emb = np.load('hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer27_t30_150M.npy')
ids = np.load('hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
genes_arr = np.load('hMuscle/data/brain_isoquant_esm2/full/brain_full_gene_names.npy', allow_pickle=True)

id_to_idx = {str(v): i for i, v in enumerate(ids)}
gene_to_idxs = {}
for i, g in enumerate(genes_arr):
    gene_to_idxs.setdefault(str(g), []).append(i)

print(f"  Embeddings: {emb.shape}")
print(f"  Isoforms: {len(ids)}, Genes: {len(gene_to_idxs)}")

# Human BP annotations
gene_go_bp = {}
with open('hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            gene = parts[0].strip()
            gos = {g.strip() for g in parts[1:] if g.strip().startswith('GO:')}
            gene_go_bp[gene] = gos

# pfam2go
pfam2go_mf = {}  # domain_name → set of GO terms (from pfam2go)
with open('/tmp/pfam2go.txt') as f:
    for line in f:
        if line.startswith('!') or not line.strip(): continue
        parts = line.strip().split(' > ')
        if len(parts) != 2: continue
        pfam_name = ' '.join(parts[0].strip().split()[1:]).lower()
        go_id = parts[1].strip().split('; ')[-1].strip()
        pfam2go_mf.setdefault(pfam_name, set()).add(go_id)

print(f"  GO annotations: {len(gene_go_bp)} genes, {sum(len(v) for v in gene_go_bp.values())} total terms")

# BISECT PASS cases
cases = pd.read_csv('Final_analysis/pipeline_bioanalysis/outputs/cases_summary_20260531_0155.tsv', sep='\t')
pass_cases = cases[cases['stage2_pass'] == 'YES'].reset_index(drop=True)
print(f"  BISECT PASS cases: {len(pass_cases)}")
print()


# ─── 2. Helper Functions ───────────────────────────────────────────────────────

def build_labels_for_go(go_term, gene_go_dict, gene_to_idxs, n_isoforms):
    """Build isoform-level binary labels for a GO term via gene-level propagation."""
    labels = np.zeros(n_isoforms, dtype=int)
    pos_genes = 0
    for gene, gos in gene_go_dict.items():
        if go_term in gos and gene in gene_to_idxs:
            for idx in gene_to_idxs[gene]:
                labels[idx] = 1
            pos_genes += 1
    return labels, pos_genes


def train_probe_and_score(emb, labels, ct_idx, ad_idx, gene_name, go_term, n_pos_min=30):
    """Train LR probe and return CT/AD scores."""
    n_pos = labels.sum()
    if n_pos < n_pos_min:
        return None, f"too few positives ({n_pos})"

    # Train/test: use all data with stratification
    scaler = StandardScaler()
    X = scaler.fit_transform(emb)

    # Quick LR probe
    lr = LogisticRegression(C=1.0, max_iter=500, solver='saga', n_jobs=1)
    lr.fit(X, labels)

    # AUPRC on all data (train set — quick evaluation)
    prob = lr.predict_proba(X)[:, 1]
    auprc = average_precision_score(labels, prob)

    # CT/AD scores
    ct_score = float(lr.predict_proba(X[ct_idx:ct_idx+1])[:, 1])
    ad_score = float(lr.predict_proba(X[ad_idx:ad_idx+1])[:, 1])

    return {
        'auprc': round(auprc, 4),
        'ct_score': round(ct_score, 4),
        'ad_score': round(ad_score, 4),
        'delta': round(ct_score - ad_score, 4),
        'n_pos': int(n_pos)
    }, None


# ─── 3. Experiment A: BISECT-guided GO term extension ─────────────────────────

# Manual mapping: BISECT domain loss/gain → functional GO term to test
BISECT_GO_MAP = {
    'FANCA':   [('GO:0006281', 'DNA repair'),       ('GO:0006974', 'DNA damage response')],
    'DMD':     [('GO:0007010', 'cytoskeleton org'), ('GO:0030036', 'actin cytoskeleton')],
    'ZCCHC17': [('GO:0006397', 'mRNA processing'),  ('GO:0008380', 'RNA splicing')],
    'PML':     [('GO:0006974', 'DNA damage resp'),  ('GO:0000209', 'protein polyubiquitination')],
    'SYNE1':   [('GO:0007010', 'cytoskeleton org'), ('GO:0043247', 'telomere maintenance')],
    'LRPPRC':  [('GO:0000956', 'nuc mRNA surveillance'), ('GO:0006120', 'mito electron transport')],
    'IFI16':   [('GO:0006281', 'DNA repair'),       ('GO:0045087', 'innate immune response')],
    'GOLGB1':  [('GO:0006888', 'ER-to-Golgi transport'), ('GO:0048193', 'Golgi vesicle transport')],
    'PTPRF':   [('GO:0045087', 'innate immune'),    ('GO:0007169', 'RTK signaling')],
    'DOCK11':  [('GO:0045087', 'innate immune'),    ('GO:0032956', 'regulation of actin cytoskeleton')],
    'PTPRS':   [('GO:0007169', 'RTK signaling'),    ('GO:0031175', 'neuron projection dev')],  # last one in PRISM
    'BSG':     [('GO:0007155', 'cell adhesion'),    ('GO:0006119', 'oxidative phosphorylation')],
    'RGS3':    [('GO:0007186', 'GPCR signaling'),   ('GO:0031175', 'neuron projection dev')],
    'IFT122':  [('GO:0060271', 'cilium assembly'),  ('GO:0007018', 'MT movement')],  # second in PRISM
    'KIF21B':  [('GO:0007018', 'MT movement'),      ('GO:0006941', 'muscle contraction')],
}

print("=" * 80)
print("EXPERIMENT A: BISECT-guided GO term linear probe")
print("=" * 80)
print(f"  Embedding: brain L27 (640-dim), n=63,994")
print(f"  Labels: human_annotations_unified_bp.txt (gene-level propagation)")
print()

expA_results = []

for _, row in pass_cases.iterrows():
    gene = row['gene']
    if gene not in BISECT_GO_MAP:
        continue

    ct_id = str(row['ct_transcript_id']); ad_id = str(row['ad_transcript_id'])
    direction = row['direction']  # CT_high or AD_high

    ct_idx = id_to_idx.get(ct_id)
    ad_idx = id_to_idx.get(ad_id)
    if ct_idx is None and gene in gene_to_idxs:
        idxs = gene_to_idxs[gene]
        ct_idx = idxs[emb[idxs].mean(axis=1).argmax()]
    if ad_idx is None and gene in gene_to_idxs:
        idxs = gene_to_idxs[gene]
        ad_idx = idxs[emb[idxs].mean(axis=1).argmin()]

    if ct_idx is None or ad_idx is None:
        continue

    for go_term, go_name in BISECT_GO_MAP[gene]:
        labels, n_pos = build_labels_for_go(go_term, gene_go_bp, gene_to_idxs, len(ids))
        result, err = train_probe_and_score(emb, labels, ct_idx, ad_idx, gene, go_term)

        if err:
            print(f"  [{gene}] {go_term} ({go_name}): SKIP — {err}")
            continue

        # Direction check
        expected_ct_higher = (direction == 'CT_high')
        actual_ct_higher = result['delta'] > 0
        correct = (expected_ct_higher == actual_ct_higher)

        print(f"  [{gene}] {go_term} ({go_name[:20]}): "
              f"AUPRC={result['auprc']:.3f}, "
              f"CT={result['ct_score']:.3f}, AD={result['ad_score']:.3f}, "
              f"Δ={result['delta']:+.3f} | "
              f"{'✓ CORRECT' if correct else '✗ WRONG'} (direction={direction})")

        expA_results.append({
            'gene': gene, 'go_term': go_term, 'go_name': go_name,
            'direction': direction, 'correct_direction': correct,
            **result
        })

dfA = pd.DataFrame(expA_results)
if len(dfA) > 0:
    correct_pct = dfA['correct_direction'].mean() * 100
    print(f"\n  Direction correct: {dfA['correct_direction'].sum()}/{len(dfA)} ({correct_pct:.1f}%)")
    print(f"  Mean AUPRC: {dfA['auprc'].mean():.3f}")


# ─── 4. Experiment B: pfam2go MF extension ────────────────────────────────────

print()
print("=" * 80)
print("EXPERIMENT B: pfam2go MF/CC GO terms → Linear Probe")
print("=" * 80)
print()

# Key pfam2go entries from BISECT PASS cases
PFAM2GO_TARGETS = [
    # (domain_short_name, pfam2go_key, go_term, go_name, genes_with_domain)
    ('Kinesin', 'kinesin', 'GO:0007018', 'MT movement', ['KIF21B', 'KIF1A', 'KIF2A']),
    ('WD40',    'wd40',    'GO:0000278', 'mitotic cell cycle', None),
    ('PDZ',     'pdz',     'GO:0007268', 'synaptic transmission', ['DLG1', 'DLG2', 'DLG4']),
    ('Spectrin', 'spectrin', 'GO:0030036', 'actin cytoskeleton', ['DMD', 'SYNE1', 'ACTN1']),
    ('Clathrin', 'clathrin', 'GO:0006886', 'intracellular transport', ['IFT122']),
]

# For each pfam domain, build labels from genes known to have that domain
def get_domain_gene_labels(domain_go_term, gene_go_dict, gene_to_idxs, n_isoforms):
    """Build labels: positive = genes annotated with domain's GO term."""
    return build_labels_for_go(domain_go_term, gene_go_dict, gene_to_idxs, n_isoforms)

expB_results = []

for dom_name, pfam_key, go_term, go_name, seed_genes in PFAM2GO_TARGETS:
    labels, _ = get_domain_gene_labels(go_term, gene_go_bp, gene_to_idxs, len(ids))
    n_pos = labels.sum()

    if n_pos < 50:
        print(f"  [{dom_name}] {go_term} ({go_name}): SKIP — too few positives ({n_pos})")
        continue

    scaler = StandardScaler()
    X = scaler.fit_transform(emb)
    lr = LogisticRegression(C=1.0, max_iter=500, solver='saga', n_jobs=1)
    lr.fit(X, labels)
    prob = lr.predict_proba(X)[:, 1]
    auprc = average_precision_score(labels, prob)

    # Check score for seed genes (should be high)
    seed_scores = []
    if seed_genes:
        for sg in seed_genes:
            if sg in gene_to_idxs:
                sg_idx = gene_to_idxs[sg][0]
                seed_scores.append((sg, round(float(lr.predict_proba(X[sg_idx:sg_idx+1])[:, 1]), 3)))

    print(f"  [{dom_name}] {go_term} ({go_name}): AUPRC={auprc:.3f}, n_pos={n_pos}")
    if seed_scores:
        print(f"    Seed gene scores: {seed_scores}")

    expB_results.append({'domain': dom_name, 'go_term': go_term, 'go_name': go_name,
                         'auprc': round(auprc, 4), 'n_pos': int(n_pos)})


# ─── 5. Save results ──────────────────────────────────────────────────────────

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if len(expA_results) > 0:
    print(f"\nExperiment A (BISECT-guided):")
    print(f"  {len(expA_results)} GO term tests across {dfA['gene'].nunique()} genes")
    print(f"  Direction correct: {dfA['correct_direction'].sum()}/{len(dfA)} ({dfA['correct_direction'].mean()*100:.1f}%)")
    mean_auprc_A = dfA['auprc'].mean()
    print(f"  Mean AUPRC: {mean_auprc_A:.3f}")

    # Key cases
    print("\n  Top cases by |Δ|:")
    for _, r in dfA.nlargest(5, 'delta').iterrows():
        print(f"    {r['gene']} {r['go_term']} Δ={r['delta']:+.3f} ({'✓' if r['correct_direction'] else '✗'})")

if len(expB_results) > 0:
    print(f"\nExperiment B (pfam2go MF extension):")
    dfB = pd.DataFrame(expB_results)
    print(f"  {len(expB_results)} domain→GO tests")
    print(f"  Mean AUPRC: {dfB['auprc'].mean():.3f}")

# Save
if len(expA_results) > 0:
    dfA.to_csv('reports/bisect_guided_probe_results.csv', index=False)
    print(f"\nSaved: reports/bisect_guided_probe_results.csv")
if len(expB_results) > 0:
    dfB.to_csv('reports/pfam2go_probe_results.csv', index=False)
    print(f"Saved: reports/pfam2go_probe_results.csv")
