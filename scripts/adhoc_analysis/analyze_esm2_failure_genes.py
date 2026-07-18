#!/usr/bin/env python3
"""
Deep dive into genes where ESM-2 fails but splice_delta succeeds
"""

import numpy as np
import json

# Load data
splice_delta = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/results_isoform/features/splicing/splicing_delta_v2.npy')
esm2_emb = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/data/esm2_embeddings_t30_150M.npy')
isoform_ids = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/model/my_isoform_list_fixed.npy', allow_pickle=True)
gene_ids = np.load('/home/welcome1/sw1686/DIFFUSE/hMuscle/model/my_gene_list_fixed.npy', allow_pickle=True)

# Top 10 genes from previous analysis
top_genes = [
    b'ENSG00000197102.14',
    b'ENSG00000102763.18',
    b'ENSG00000176208.9',
    b'ENSG00000215252.12',
    b'ENSG00000183722.9',
    b'ENSG00000171467.16',
    b'ENSG00000088812.18',
    b'ENSG00000130940.15',
    b'ENSG00000106399.11',
    b'ENSG00000151320.11'
]

print("="*80)
print("DETAILED ANALYSIS: ESM-2 FAILURE CASES")
print("="*80)

for gene_id in top_genes:
    gene_mask = gene_ids == gene_id
    gene_indices = np.where(gene_mask)[0]
    
    if len(gene_indices) < 2:
        continue
    
    print(f"\n{'='*80}")
    print(f"Gene: {gene_id.decode()}")
    print(f"Number of isoforms: {len(gene_indices)}")
    print(f"{'='*80}")
    
    # Get isoform IDs
    iso_ids = [isoform_ids[i].decode() for i in gene_indices]
    print("\nIsoforms:")
    for i, iso_id in enumerate(iso_ids):
        print(f"  [{i}] {iso_id}")
    
    # ESM-2 analysis
    esm2_vecs = esm2_emb[gene_indices]
    print(f"\nESM-2 embedding norms:")
    for i, norm in enumerate(np.linalg.norm(esm2_vecs, axis=1)):
        print(f"  [{i}] {norm:.4f}")
    
    # Pairwise ESM-2 cosine distances
    from scipy.spatial.distance import cosine
    print(f"\nESM-2 pairwise cosine distances:")
    for i in range(len(gene_indices)):
        for j in range(i+1, len(gene_indices)):
            dist = cosine(esm2_vecs[i], esm2_vecs[j])
            print(f"  [{i}]-[{j}]: {dist:.6f}")
    
    # splice_delta analysis
    splice_vecs = splice_delta[gene_indices]
    splice_abs_sums = np.abs(splice_vecs).sum(axis=1)
    print(f"\nsplice_delta |Δ|.sum():")
    for i, s in enumerate(splice_abs_sums):
        print(f"  [{i}] {s:.2f}")
    
    # Pairwise splice_delta L1 distances
    print(f"\nsplice_delta pairwise L1 distances:")
    for i in range(len(gene_indices)):
        for j in range(i+1, len(gene_indices)):
            dist = np.abs(splice_vecs[i] - splice_vecs[j]).sum()
            print(f"  [{i}]-[{j}]: {dist:.2f}")
    
    # Check where splice_delta differs most
    splice_diff = splice_vecs[1] - splice_vecs[0] if len(gene_indices) >= 2 else None
    if splice_diff is not None:
        top_diff_positions = np.argsort(np.abs(splice_diff))[-5:][::-1]
        print(f"\nTop 5 splice_delta difference positions (isoform[1] - isoform[0]):")
        for pos in top_diff_positions:
            print(f"  Position {pos}: Δ={splice_diff[pos]:.2f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThese genes exhibit IDENTICAL ESM-2 embeddings for different isoforms,")
print("but LARGE splice_delta differences.")
print("\nPossible explanations:")
print("  1. ESM-2 (protein LLM) only sees final translated sequence")
print("  2. Alternative splicing changes non-coding regions (UTRs)")
print("  3. Frame-preserving in-frame insertions/deletions (IDR regions)")
print("  4. Alternative promoters creating identical protein sequences")
print("\nsplice_delta captures structural gene-level information that ESM-2 misses.")
