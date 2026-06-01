#!/usr/bin/env python3
"""
compute_esm2_train_layers.py
-----------------------------
훈련 단백질 Human isoform (31668개)에서 ESM-2 중간 레이어 (L7, L18, L27) 추출.
v15e layer fusion 모델 훈련용 전처리.

출력:
  ../data/esm2_train_human_layer07_t30_150M.npy  (31668, 640)
  ../data/esm2_train_human_layer18_t30_150M.npy  (31668, 640)
  ../data/esm2_train_human_layer27_t30_150M.npy  (31668, 640)

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u compute_esm2_train_layers.py \
      > ../../logs_isoform/train_layers_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import os, re, sys, time, argparse
import numpy as np
import torch
from datetime import datetime

EXTRACT_LAYERS = [7, 18, 27]
DATA_DIR = '../data'
RAW_SEQ  = os.path.join(DATA_DIR, 'raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt')
ISO_LIST = os.path.join(DATA_DIR, 'raw_data/data/id_lists/train_isoform_list.npy')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_len', type=int, default=1022)
    return p.parse_args()

def parse_human_seq(path, max_len=1022):
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
                flush()
                cur_seq = []
                m = re.match(r'>(\S+)\|(\S+)', line)
                cur_key = m.group(2) if m else None
            else:
                cur_seq.append(line)
    flush()
    return records

@torch.no_grad()
def compute_batch_multilayer(model, batch_converter, sequences, device, extract_layers):
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)
    results = model(tokens, repr_layers=extract_layers, return_contacts=False)
    out = {}
    for L in extract_layers:
        token_reps = results['representations'][L]
        embs = []
        for i, (_, seq) in enumerate(sequences):
            seq_len = len(seq)
            embs.append(token_reps[i, 1:seq_len+1, :].mean(0).cpu().float().numpy())
        out[L] = np.array(embs)
    return out

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now():%H:%M:%S}] Device: {device}  Extracting layers: {EXTRACT_LAYERS}", flush=True)

    import esm
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    print(f"  embed_dim={model.embed_dim}  num_layers={model.num_layers}", flush=True)

    print(f"\n[{datetime.now():%H:%M:%S}] Parsing sequences...", flush=True)
    human_seqs = parse_human_seq(RAW_SEQ, max_len=args.max_len)
    print(f"  Parsed {len(human_seqs)} unique NM_IDs", flush=True)

    iso_arr = np.load(ISO_LIST, allow_pickle=True)
    iso_list = [x.decode() if isinstance(x, bytes) else x for x in iso_arr]
    N = len(iso_list)
    D = model.embed_dim

    valid = [(i, iso_list[i], human_seqs.get(iso_list[i]))
             for i in range(N) if human_seqs.get(iso_list[i]) is not None]
    print(f"  id_list size: {N}  Valid: {len(valid)}  Missing: {N - len(valid)}", flush=True)

    mats = {L: np.zeros((N, D), dtype=np.float32) for L in EXTRACT_LAYERS}
    n_valid = len(valid)
    n_batches = (n_valid + args.batch_size - 1) // args.batch_size
    t0 = time.time()

    for b_idx in range(n_batches):
        b_start = b_idx * args.batch_size
        b_end   = min(b_start + args.batch_size, n_valid)
        batch   = valid[b_start:b_end]
        indices = [x[0] for x in batch]
        seqs    = list(zip([x[1] for x in batch], [x[2] for x in batch]))

        try:
            emb_dict = compute_batch_multilayer(model, batch_converter, seqs, device, EXTRACT_LAYERS)
            for L in EXTRACT_LAYERS:
                for k, idx in enumerate(indices):
                    mats[L][idx] = emb_dict[L][k]
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                half = args.batch_size // 2
                for sub_s in range(0, len(batch), half):
                    sub = batch[sub_s:sub_s+half]
                    sub_idx = [x[0] for x in sub]
                    sub_seqs = list(zip([x[1] for x in sub], [x[2] for x in sub]))
                    sub_emb = compute_batch_multilayer(model, batch_converter, sub_seqs, device, EXTRACT_LAYERS)
                    for L in EXTRACT_LAYERS:
                        for k, idx in enumerate(sub_idx):
                            mats[L][idx] = sub_emb[L][k]
            else:
                raise

        if (b_idx+1) % 20 == 0 or b_idx == n_batches-1:
            elapsed = time.time() - t0
            done = b_end
            eta = elapsed/done*(n_valid-done) if done < n_valid else 0
            gpu_mem = torch.cuda.memory_allocated(device)/1024**3 if torch.cuda.is_available() else 0
            print(f"  [{b_idx+1:4d}/{n_batches}] {done}/{n_valid} seqs | "
                  f"{elapsed:.0f}s | ETA {eta:.0f}s | GPU {gpu_mem:.2f}GB", flush=True)

    for L in EXTRACT_LAYERS:
        out_path = os.path.join(DATA_DIR, f'esm2_train_human_layer{L:02d}_t30_150M.npy')
        np.save(out_path, mats[L])
        norms = np.linalg.norm(mats[L], axis=1)
        print(f"  Saved: {out_path}  shape={mats[L].shape}  "
              f"norm_mean={norms.mean():.4f}  norm_std={norms.std():.4f}", flush=True)

    total = time.time() - t0
    print(f"\n[{datetime.now():%H:%M:%S}] DONE. Total: {total:.0f}s ({total/60:.1f}min)", flush=True)

if __name__ == '__main__':
    main()
