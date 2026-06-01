#!/usr/bin/env python3
"""
decode_train_to_fasta.py
=========================
human_sequence_train.npy (3-gram encoded) → protein FASTA for hmmscan

출력: data/domain/train_proteins.fasta
"""
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

NUM_TO_AA = {
    1:'F', 2:'L', 3:'I', 4:'M', 5:'V', 6:'S', 7:'P', 8:'T', 9:'A',
    10:'Y', 11:'H', 12:'Q', 13:'N', 14:'K', 15:'D', 16:'E', 17:'C',
    18:'W', 19:'R', 20:'G'
}

def decode_3gram(row):
    nz = row[row > 0]
    if len(nz) == 0:
        return None
    aa_seq = []
    for ng in nz:
        ng = int(ng)
        n1 = (ng - 1) // 400 + 1
        aa_seq.append(NUM_TO_AA.get(n1, 'X'))
    # 마지막 3-gram의 나머지 2 AA
    ng = int(nz[-1])
    n2 = ((ng - 1) // 20) % 20 + 1
    n3 = (ng - 1) % 20 + 1
    aa_seq.append(NUM_TO_AA.get(n2, 'X'))
    aa_seq.append(NUM_TO_AA.get(n3, 'X'))
    return ''.join(aa_seq)

print("Loading train sequences and isoform IDs ...")
seq_mat = np.load('data/raw_data/data/sequences/human_sequence_train.npy', allow_pickle=True)
train_ids = np.load('data/raw_data/data/id_lists/train_isoform_list.npy', allow_pickle=True)
train_ids = [s.decode() if isinstance(s, bytes) else s for s in train_ids]

print(f"  Sequences: {seq_mat.shape}")

out_path = 'results_isoform/features/train_proteins.fasta'
n_written = n_short = 0
with open(out_path, 'w') as fout:
    for i, (nm_id, row) in enumerate(zip(train_ids, seq_mat)):
        protein = decode_3gram(row)
        if protein is None or len(protein) < 10:
            n_short += 1
            continue
        # FASTA ID: comma-separated NM_ IDs → use first one
        fasta_id = nm_id.split(',')[0].strip()
        fout.write(f'>{fasta_id}\n')
        # 60 AA per line
        for j in range(0, len(protein), 60):
            fout.write(protein[j:j+60] + '\n')
        n_written += 1

print(f"  Written: {n_written}, Skipped (too short): {n_short}")
print(f"  Saved: {out_path}")
print("Done.")
