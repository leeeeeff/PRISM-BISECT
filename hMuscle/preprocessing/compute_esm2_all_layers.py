#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_esm2_all_layers.py
--------------------------
ESM-2의 모든 30 layer에서 mean-pooled embedding을 동시 추출.
단일 forward pass로 repr_layers=list(range(1,31))를 요청하여 효율 극대화.

출력:
  ../data/esm2_layer_NN_t30_150M.npy  (NN=01..30)  shape: (36748, 640)  float32

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=0 python compute_esm2_all_layers.py [--batch_size 32] [--gpu 0]
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import torch
from datetime import datetime

DATA_DIR = '../data'
ISO_LIST = '../model/my_isoform_list_fixed.npy'
PEP_FILE = '../data/top30k_isoforms.pep'
MODEL_TAG = 'esm2_t30_150M_UR50D'
N_LAYERS  = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=32,
                   help='Batch size (reduce to 16 if OOM)')
    p.add_argument('--max_len',    type=int, default=1022)
    p.add_argument('--gpu',        type=int, default=0)
    p.add_argument('--out_dir',    default=DATA_DIR)
    return p.parse_args()


TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}


def parse_pep_file(pep_path, max_len=1022):
    records = {}
    cur_id = cur_meta = None
    cur_seq = []

    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None:
            return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq:
            return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank, score, length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq)

    with open(pep_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                cur_seq = []
                m_id    = re.match(r'>(\S+)', line)
                m_type  = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len   = re.search(r'len:(\d+)', line)
                if not m_id:
                    cur_id = None; continue
                raw_id   = m_id.group(1)
                cur_id   = re.sub(r'\.p\d+$', '', raw_id)
                orf_type = m_type.group(1) if m_type else 'internal'
                score    = float(m_score.group(1)) if m_score else 0.0
                length   = int(m_len.group(1))     if m_len   else 0
                rank     = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score, length)
            else:
                cur_seq.append(line)
    flush()

    return {k: v[3][:max_len] for k, v in records.items()}


@torch.no_grad()
def compute_all_layers_batch(model, batch_converter, sequences, device, n_layers):
    """
    Returns: dict {layer_idx: np.ndarray (B, 640)}
    """
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    all_repr_layers = list(range(1, n_layers + 1))
    results = model(tokens, repr_layers=all_repr_layers, return_contacts=False)

    layer_embs = {}
    for l in all_repr_layers:
        token_reps = results['representations'][l]  # (B, L+2, D)
        embs = []
        for i, (_, seq) in enumerate(sequences):
            seq_len = len(seq)
            rep = token_reps[i, 1:seq_len + 1, :]
            embs.append(rep.mean(dim=0).cpu().float().numpy())
        layer_embs[l] = np.array(embs)  # (B, D)

    return layer_embs


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now():%H:%M:%S}] Device: {device}")

    # Check which layers already computed
    def out_path(l):
        return os.path.join(args.out_dir, f'esm2_layer_{l:02d}_t30_150M.npy')

    done = [l for l in range(1, N_LAYERS + 1) if os.path.exists(out_path(l))]
    todo = [l for l in range(1, N_LAYERS + 1) if l not in done]
    if done:
        print(f"  Already done: layers {done}")
    if not todo:
        print("  All 30 layers already extracted. Exiting.")
        return
    print(f"  To extract: {len(todo)} layers ({todo[0]}..{todo[-1]})")

    # Load isoform list
    iso_arr  = np.load(ISO_LIST, allow_pickle=True)
    iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_arr]
    N = len(iso_list)
    print(f"  Isoforms: {N}")

    # Parse protein sequences
    print(f"[{datetime.now():%H:%M:%S}] Parsing {PEP_FILE}")
    pep_seqs = parse_pep_file(PEP_FILE, max_len=args.max_len)
    seqs_ordered = [pep_seqs.get(iso_id, None) for iso_id in iso_list]
    valid_indices = [(i, iso_list[i], seqs_ordered[i])
                     for i in range(N) if seqs_ordered[i] is not None]
    n_missing = N - len(valid_indices)
    print(f"  Coverage: {len(valid_indices)}/{N}  missing={n_missing}")

    # Load ESM-2
    print(f"[{datetime.now():%H:%M:%S}] Loading {MODEL_TAG}")
    import esm
    loader = getattr(esm.pretrained, MODEL_TAG)
    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    assert model.num_layers == N_LAYERS, f"Expected {N_LAYERS} layers, got {model.num_layers}"
    print(f"  num_layers={model.num_layers}  embed_dim={model.embed_dim}")
    print(f"  GPU memory after model load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    # Pre-allocate result arrays for all needed layers
    D = model.embed_dim  # 640
    layer_matrices = {l: np.zeros((N, D), dtype=np.float32) for l in todo}

    # Batch inference
    bs = args.batch_size
    n_valid = len(valid_indices)
    n_batches = (n_valid + bs - 1) // bs
    t0 = time.time()

    print(f"\n[{datetime.now():%H:%M:%S}] Starting inference ({n_batches} batches, bs={bs})")
    for b_idx in range(n_batches):
        b_start = b_idx * bs
        b_end   = min(b_start + bs, n_valid)
        batch   = valid_indices[b_start:b_end]

        indices  = [x[0] for x in batch]
        labels   = [x[1] for x in batch]
        aa_seqs  = [x[2] for x in batch]
        seqs     = list(zip(labels, aa_seqs))

        try:
            layer_embs = compute_all_layers_batch(model, batch_converter, seqs, device, N_LAYERS)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n  [OOM] Batch {b_idx}: splitting to half-batches")
                half = max(1, bs // 2)
                layer_embs = {l: np.zeros((len(batch), D), dtype=np.float32) for l in range(1, N_LAYERS+1)}
                for sub_start in range(0, len(batch), half):
                    sub = batch[sub_start:sub_start + half]
                    sub_seqs = list(zip([x[1] for x in sub], [x[2] for x in sub]))
                    sub_embs = compute_all_layers_batch(model, batch_converter, sub_seqs, device, N_LAYERS)
                    for l in range(1, N_LAYERS + 1):
                        layer_embs[l][sub_start:sub_start + len(sub)] = sub_embs[l]
            else:
                raise

        for l in todo:
            for k, idx in enumerate(indices):
                layer_matrices[l][idx] = layer_embs[l][k]

        if (b_idx + 1) % 20 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (b_end) * (n_valid - b_end) if b_end < n_valid else 0
            print(f"  [{b_idx+1:4d}/{n_batches}] {b_end}/{n_valid}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"GPU={torch.cuda.memory_allocated(device)/1024**3:.2f}GB")

    # Save
    print(f"\n[{datetime.now():%H:%M:%S}] Saving {len(todo)} layer files...")
    os.makedirs(args.out_dir, exist_ok=True)
    for l in todo:
        path = out_path(l)
        np.save(path, layer_matrices[l])
        norms = np.linalg.norm(layer_matrices[l], axis=1)
        print(f"  Layer {l:02d}: {path}  norm mean={norms.mean():.3f}")

    total = time.time() - t0
    print(f"\n[{datetime.now():%H:%M:%S}] Done. {total:.0f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main()
