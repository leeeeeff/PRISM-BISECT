#!/usr/bin/env python3
"""
compute_brain_isoquant_layers.py
---------------------------------
뇌 조직 isoform (63994개) ESM-2 중간 레이어 (L7, L18, L27) 추출.
v15e brain 평가용 전처리.

출력:
  data/brain_isoquant_esm2/full/brain_full_esm2_layer07_t30_150M.npy  (63994, 640)
  data/brain_isoquant_esm2/full/brain_full_esm2_layer18_t30_150M.npy  (63994, 640)
  data/brain_isoquant_esm2/full/brain_full_esm2_layer27_t30_150M.npy  (63994, 640)

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u compute_brain_isoquant_layers.py \
      > ../../logs_isoform/brain_layers_$(date +%Y%m%d_%H%M).log 2>&1 &
"""

import os, sys, re, time, argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

PROJECT_DIR = '/home/dhkim1674/Project_AD_with_refTSS_novel'
CLF_CSV     = f'{PROJECT_DIR}/02_Isoquant_Output/SQANTI3_output/isoforms_classification_with_tx_name_and_gene_name.csv'
COUNT_CSV   = f'{PROJECT_DIR}/04_Counts/Long_Read/Cell_Type/counts_by_cell_type/tx_counts_by_cell_type.csv'
OUT_DIR     = 'data/brain_isoquant_esm2/full'
EXTRACT_LAYERS = [7, 18, 27]
BATCH_SIZE  = 64
MAX_LEN     = 1022

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()

def load_target_isoforms():
    cnt = pd.read_csv(COUNT_CSV, index_col=0)
    all_ids = list(cnt.index)
    gene_col = cnt['gene_name'] if 'gene_name' in cnt.columns else pd.Series('', index=cnt.index)
    target_ids   = all_ids
    target_genes = [str(gene_col.iloc[i]) if not pd.isna(gene_col.iloc[i]) else ''
                    for i in range(len(all_ids))]
    is_novel = [str(tid).startswith('transcript') for tid in all_ids]
    return target_ids, target_genes, is_novel

def build_orf_map(target_ids, is_novel_flags):
    print("  Loading classification CSV...", flush=True)
    clf = pd.read_csv(CLF_CSV,
                      usecols=['isoform', 'transcript_name', 'gene_name', 'coding', 'ORF_seq'],
                      low_memory=False)
    novel_map = dict(zip(clf['isoform'], clf['ORF_seq']))
    known_map = dict(zip(clf['transcript_name'], clf['ORF_seq']))

    orf_seqs = []
    coding_mask = []
    for tid, is_nov in zip(target_ids, is_novel_flags):
        seq = novel_map.get(tid) if is_nov else known_map.get(tid)
        if pd.notna(seq) and isinstance(seq, str) and len(seq) > 0:
            clean = seq.replace('*', '').strip()
            orf_seqs.append(clean[:MAX_LEN])
            coding_mask.append(1)
        else:
            orf_seqs.append(None)
            coding_mask.append(0)

    n_coding = sum(coding_mask)
    print(f"  Target: {len(target_ids)}  coding: {n_coding}  non-coding: {len(target_ids)-n_coding}", flush=True)
    return orf_seqs, coding_mask

def compute_esm2_layers(orf_seqs, coding_mask, gpu_id):
    import esm
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    coding_indices = [i for i, m in enumerate(coding_mask) if m == 1]
    coding_seqs    = [(str(i), orf_seqs[i]) for i in coding_indices]

    N = len(orf_seqs)
    EMB_DIM = 640
    mats = {L: np.zeros((N, EMB_DIM), dtype=np.float32) for L in EXTRACT_LAYERS}

    t0 = time.time()
    n_batches = (len(coding_seqs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Computing {len(coding_seqs)} embeddings × {len(EXTRACT_LAYERS)} layers "
          f"in {n_batches} batches...", flush=True)

    with torch.no_grad():
        for bi in range(0, len(coding_seqs), BATCH_SIZE):
            batch = coding_seqs[bi:bi+BATCH_SIZE]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=EXTRACT_LAYERS, return_contacts=False)

            for j, (orig_i_str, seq) in enumerate(batch):
                orig_i = int(orig_i_str)
                seq_len = min(len(seq), MAX_LEN)
                for L in EXTRACT_LAYERS:
                    reps = out['representations'][L]
                    emb = reps[j, 1:seq_len+1, :].mean(0).cpu().numpy()
                    mats[L][orig_i] = emb

            batch_idx = bi // BATCH_SIZE
            if (batch_idx + 1) % 20 == 0:
                done = bi + len(batch)
                elapsed = time.time() - t0
                eta = elapsed / done * (len(coding_seqs) - done) if done < len(coding_seqs) else 0
                print(f"    [{done}/{len(coding_seqs)}] {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return mats

def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"  Brain IsoQuant Layer Extraction  Layers={EXTRACT_LAYERS}  GPU={args.gpu}", flush=True)
    print("=" * 60, flush=True)

    print(f"\n[1/3] Loading target isoforms...", flush=True)
    target_ids, target_genes, is_novel_flags = load_target_isoforms()
    print(f"  Target isoforms: {len(target_ids)}", flush=True)

    print(f"\n[2/3] Building ORF map from SQANTI3 classification...", flush=True)
    orf_seqs, coding_mask = build_orf_map(target_ids, is_novel_flags)

    print(f"\n[3/3] Computing ESM-2 layer embeddings...", flush=True)
    mats = compute_esm2_layers(orf_seqs, coding_mask, args.gpu)

    print("\nSaving results...", flush=True)
    for L in EXTRACT_LAYERS:
        out_path = f'{OUT_DIR}/brain_full_esm2_layer{L:02d}_t30_150M.npy'
        np.save(out_path, mats[L])
        norms = np.linalg.norm(mats[L], axis=1)
        print(f"  Saved: {out_path}  shape={mats[L].shape}  "
              f"norm_mean={norms[norms>0].mean():.4f}", flush=True)

    print("\nDONE", flush=True)

if __name__ == '__main__':
    main()
