#!/usr/bin/env python3
"""
compute_esm2_train_all_layers.py
----------------------------------
훈련 단백질 Human isoform (31668개)에서 ESM-2 L1~L30 전체 추출.
이미 존재하는 레이어 파일은 건너뜀.

출력:
  ../data/esm2_train_human_layer{LL}_t30_150M.npy  (31668, 640)  for LL in 01..30

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u compute_esm2_train_all_layers.py \
      > ../../logs_isoform/train_all_layers_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import os, re, sys, time
import numpy as np
import torch
from datetime import datetime

ALL_LAYERS = list(range(1, 31))
DATA_DIR   = '../data'
RAW_SEQ    = os.path.join(DATA_DIR, 'raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt')
ISO_LIST   = os.path.join(DATA_DIR, 'raw_data/data/id_lists/train_isoform_list.npy')
BATCH_SIZE = 64
MAX_LEN    = 1022

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()

def parse_human_seq(path, max_len=MAX_LEN):
    records = {}
    cur_key, cur_seq = None, []
    def flush():
        nonlocal cur_key, cur_seq
        if cur_key and cur_seq:
            seq = ''.join(cur_seq).replace('*', '').strip()
            if seq:
                records[cur_key] = seq[:max_len]
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m = re.match(r'>(\S+)\|(\S+)', line)
                cur_key = m.group(2) if m else None
            else:
                cur_seq.append(line)
    flush()
    return records

@torch.no_grad()
def compute_batch_multilayer(model, batch_converter, sequences, device, layers_to_extract):
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)
    results = model(tokens, repr_layers=layers_to_extract, return_contacts=False)
    out = {}
    for L in layers_to_extract:
        token_reps = results['representations'][L]
        embs = []
        for i, (_, seq) in enumerate(sequences):
            embs.append(token_reps[i, 1:len(seq)+1, :].mean(0).cpu().float().numpy())
        out[L] = np.array(embs)
    return out

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 이미 존재하는 레이어 건너뜀
    needed = []
    for L in ALL_LAYERS:
        out_path = os.path.join(DATA_DIR, f'esm2_train_human_layer{L:02d}_t30_150M.npy')
        if os.path.exists(out_path):
            print(f"  [SKIP] L{L:02d} already exists: {out_path}")
        else:
            needed.append(L)

    if not needed:
        print("All 30 training layers already extracted. Nothing to do.")
        return

    print(f"\n[{datetime.now():%H:%M:%S}] Device: {device}")
    print(f"  Layers to extract: {needed}  ({len(needed)} remaining)")

    import esm
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    D = model.embed_dim
    print(f"  embed_dim={D}  num_layers={model.num_layers}")

    print(f"\n[{datetime.now():%H:%M:%S}] Parsing sequences...", flush=True)
    human_seqs = parse_human_seq(RAW_SEQ)
    print(f"  Parsed {len(human_seqs)} unique NM_IDs")

    iso_arr = np.load(ISO_LIST, allow_pickle=True)
    iso_list = [x.decode() if isinstance(x, bytes) else x for x in iso_arr]
    N = len(iso_list)

    valid = [(i, iso_list[i], human_seqs[iso_list[i]])
             for i in range(N) if iso_list[i] in human_seqs]
    print(f"  id_list size: {N}  Valid: {len(valid)}  Missing: {N - len(valid)}")

    # all needed layers in memory — 31668 × 640 × 4B × n_layers
    mats = {L: np.zeros((N, D), dtype=np.float32) for L in needed}
    n_valid   = len(valid)
    n_batches = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    print(f"\n  Processing {n_valid} seqs in {n_batches} batches "
          f"(extracting {len(needed)} layers)...\n", flush=True)

    for b_idx in range(n_batches):
        b_start = b_idx * BATCH_SIZE
        b_end   = min(b_start + BATCH_SIZE, n_valid)
        batch   = valid[b_start:b_end]
        indices = [x[0] for x in batch]
        seqs    = list(zip([x[1] for x in batch], [x[2] for x in batch]))

        try:
            emb_dict = compute_batch_multilayer(model, batch_converter, seqs, device, needed)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                half = BATCH_SIZE // 2
                emb_dict = {L: np.zeros((len(batch), D), dtype=np.float32) for L in needed}
                for sub_s in range(0, len(batch), half):
                    sub = batch[sub_s:sub_s+half]
                    sub_seqs = list(zip([x[1] for x in sub], [x[2] for x in sub]))
                    sub_emb  = compute_batch_multilayer(model, batch_converter, sub_seqs, device, needed)
                    for L in needed:
                        emb_dict[L][sub_s:sub_s+len(sub)] = sub_emb[L]
            else:
                raise

        for L in needed:
            for k, idx in enumerate(indices):
                mats[L][idx] = emb_dict[L][k]

        if (b_idx + 1) % 40 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t0
            done    = b_end
            eta     = elapsed / done * (n_valid - done) if done < n_valid else 0
            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
            print(f"  [{b_idx+1:4d}/{n_batches}] {done}/{n_valid} seqs | "
                  f"{elapsed:.0f}s | ETA {eta:.0f}s | GPU {gpu_mem:.2f}GB", flush=True)

    print("\nSaving...", flush=True)
    for L in needed:
        out_path = os.path.join(DATA_DIR, f'esm2_train_human_layer{L:02d}_t30_150M.npy')
        np.save(out_path, mats[L])
        norms = np.linalg.norm(mats[L], axis=1)
        print(f"  Saved: {out_path}  shape={mats[L].shape}  "
              f"norm_mean={norms.mean():.4f}  norm_std={norms.std():.4f}", flush=True)

    total = time.time() - t0
    print(f"\n[{datetime.now():%H:%M:%S}] DONE. Total: {total:.0f}s ({total/60:.1f}min)", flush=True)

if __name__ == '__main__':
    main()
