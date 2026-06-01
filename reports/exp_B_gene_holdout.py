"""
Experiment B: Gene-level holdout test for 541 novel isoforms.

For each novel isoform with any brain GO score > 0.5:
1. Map to gene symbol via IsoQuant GTF
2. Check if gene is in PRISM training annotations (human_annotations_unified_bp.txt)
3. Classify:
   - KNOWN_GENE: gene in training set → gene-family generalization
   - NOVEL_GENE: gene NOT in training set → true novel gene prediction
   - UNMAPPED: transcript ID cannot be mapped to a gene

Honest reclassification of the 541 novel isoforms claim.
"""
import numpy as np, gzip, re, csv, os
from collections import defaultdict

BASE = '/home/welcome1/sw1686/DIFFUSE'

print("Loading data...")
ids = np.load(f'{BASE}/hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]

# Novel isoform mask
novel_mask = np.array(['nic' in x.lower() or 'nnic' in x.lower() for x in ids])
novel_ids = [x for x, m in zip(ids, novel_mask) if m]
print(f"Novel isoforms: {len(novel_ids)}")

# Load novel scores (7903 × 73)
novel_scores = np.load(f'{BASE}/reports/brain_novel_scores_extended.npy')
# Which are high-scorers (>0.5)?
high_mask = novel_scores.max(axis=1) > 0.5
high_novel_ids = [x for x, h in zip(novel_ids, high_mask) if h]
high_novel_scores = novel_scores[high_mask]
print(f"High-score novel isoforms (>0.5): {len(high_novel_ids)}")

# Load GO term names
go_csv = f'{BASE}/reports/brain_go_extension_results.csv'
go_names = []
with open(go_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        go_names.append(row['name'])

# Load training gene set
training_genes = set()
annot_file = f'{BASE}/hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt'
with open(annot_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            training_genes.add(parts[0])
print(f"Training genes: {len(training_genes)}")

# Map transcript IDs to gene symbols using IsoQuant GTF
GTF = '/home/dhkim1674/Project_AD_with_refTSS_novel/02_Isoquant_Output/extended_annotation_including_refTSS_umi10_donor3_supported_novel_tx.gtf'
transcript_to_gene = {}

print("Parsing GTF for transcript→gene mapping...")
try:
    with open(GTF) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            if parts[2] != 'transcript': continue
            attr = parts[8]
            # Extract transcript_id and gene_name
            tid_m = re.search(r'transcript_id "([^"]+)"', attr)
            gname_m = re.search(r'gene_name "([^"]+)"', attr)
            gsym_m = re.search(r'gene_id "([^"]+)"', attr)
            if tid_m:
                tid = tid_m.group(1)
                gene = (gname_m.group(1) if gname_m else
                        gsym_m.group(1) if gsym_m else None)
                if gene:
                    transcript_to_gene[tid] = gene
    print(f"GTF transcript→gene mappings: {len(transcript_to_gene)}")
except Exception as e:
    print(f"GTF parse error: {e}")

# Classify high-score novel isoforms
results = []
counts = {'KNOWN_GENE': 0, 'NOVEL_GENE': 0, 'UNMAPPED': 0}

for iso_id, scores in zip(high_novel_ids, high_novel_scores):
    gene = transcript_to_gene.get(iso_id, None)
    if gene is None:
        # Try without suffix (e.g. 'transcript12345.chr1.nic' → 'transcript12345')
        base = iso_id.split('.')[0]
        gene = transcript_to_gene.get(base, None)

    if gene is None:
        status = 'UNMAPPED'
    elif gene in training_genes:
        status = 'KNOWN_GENE'
    else:
        status = 'NOVEL_GENE'

    counts[status] += 1
    top_go_idx = scores.argmax()
    results.append({
        'iso_id': iso_id,
        'gene': gene or 'UNMAPPED',
        'status': status,
        'max_score': round(float(scores.max()), 4),
        'top_go': go_names[top_go_idx] if top_go_idx < len(go_names) else '?',
        'n_terms_above_0_5': int((scores > 0.5).sum()),
    })

print("\n=== GENE-LEVEL HOLDOUT RESULTS ===")
total = len(results)
print(f"Total high-score novel isoforms: {total}")
print(f"  KNOWN_GENE (gene in training set):  {counts['KNOWN_GENE']} ({100*counts['KNOWN_GENE']/total:.1f}%)")
print(f"  NOVEL_GENE (gene NOT in training):  {counts['NOVEL_GENE']} ({100*counts['NOVEL_GENE']/total:.1f}%)")
print(f"  UNMAPPED   (no GTF gene found):     {counts['UNMAPPED']} ({100*counts['UNMAPPED']/total:.1f}%)")

# Show top NOVEL_GENE cases
novel_gene_cases = [r for r in results if r['status'] == 'NOVEL_GENE']
if novel_gene_cases:
    novel_gene_cases.sort(key=lambda x: -x['max_score'])
    print(f"\nTop TRUE NOVEL GENE cases (gene not in training):")
    for r in novel_gene_cases[:20]:
        print(f"  {r['iso_id']:<40} gene={r['gene']:<20} score={r['max_score']:.3f} GO={r['top_go'][:30]}")

known_cases = [r for r in results if r['status'] == 'KNOWN_GENE']
if known_cases:
    known_cases.sort(key=lambda x: -x['max_score'])
    print(f"\nTop KNOWN_GENE (gene-family generalization) cases:")
    for r in known_cases[:10]:
        print(f"  {r['iso_id']:<40} gene={r['gene']:<20} score={r['max_score']:.3f} GO={r['top_go'][:30]}")

# Save
import json
out = f'{BASE}/reports/exp_B_gene_holdout_results.json'
with open(out, 'w') as f:
    json.dump({'counts': counts, 'total': total, 'results': results}, f, indent=2)
print(f"\nSaved: {out}")
