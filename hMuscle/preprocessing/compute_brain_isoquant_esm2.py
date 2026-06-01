#!/usr/bin/env python3
"""
compute_brain_isoquant_esm2.py
================================
Samsung Alzheimer scLR-seq (Project_AD_with_refTSS_novel) 뇌 조직 isoform의
ESM-2 t30_150M 임베딩 계산.

입력:
  /home/dhkim1674/Project_AD_with_refTSS_novel/
    02_Isoquant_Output/SQANTI3_output/
      isoforms_classification_with_tx_name_and_gene_name.csv  (ORF_seq 포함)
    04_Counts/Long_Read/Cell_Type/counts_by_cell_type/
      tx_counts_by_cell_type.csv  (target isoform list)

출력 (DIFFUSE/hMuscle/data/brain_isoquant_esm2/):
  novel/
    brain_novel_esm2_t30_150M.npy   (7899, 640)  float32
    brain_novel_ids.npy              (7899,)       str  [transcript*.nnic]
    brain_novel_gene_names.npy       (7899,)       str
    brain_novel_mask.npy             (7899,)       int8  1=coding, 0=non-coding

  full/ (--mode full 옵션 시)
    brain_full_esm2_t30_150M.npy    (63994, 640)
    brain_full_ids.npy               (63994,)
    brain_full_gene_names.npy        (63994,)
    brain_full_mask.npy              (63994,)

실행:
  # Novel only (~15분)
  cd hMuscle/preprocessing/
  conda activate isoform_env
  python compute_brain_isoquant_esm2.py --mode novel

  # Full including known (~2.5시간)
  python compute_brain_isoquant_esm2.py --mode full
"""

import os, sys, re, time, argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = '/home/dhkim1674/Project_AD_with_refTSS_novel'
CLF_CSV      = f'{PROJECT_DIR}/02_Isoquant_Output/SQANTI3_output/isoforms_classification_with_tx_name_and_gene_name.csv'
COUNT_CSV    = f'{PROJECT_DIR}/04_Counts/Long_Read/Cell_Type/counts_by_cell_type/tx_counts_by_cell_type.csv'
OUT_BASE     = 'data/brain_isoquant_esm2'
MODEL_NAME   = 'esm2_t30_150M_UR50D'
BATCH_SIZE   = 64
MAX_LEN      = 1022
GPU_ID       = 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['novel', 'full'], default='novel',
                   help='novel: novel isoforms only (~15min) | full: all 63994 (~2.5h)')
    p.add_argument('--gpu', type=int, default=GPU_ID)
    return p.parse_args()

def load_target_isoforms(mode):
    """Count matrix에서 대상 isoform ID 및 gene name 로드."""
    cnt = pd.read_csv(COUNT_CSV, index_col=0)
    all_ids = list(cnt.index)
    gene_col = cnt['gene_name'] if 'gene_name' in cnt.columns else pd.Series('', index=cnt.index)

    is_novel = pd.Series(all_ids).str.startswith('transcript').values

    if mode == 'novel':
        mask_select = is_novel
    else:
        mask_select = np.ones(len(all_ids), dtype=bool)

    target_ids   = [all_ids[i] for i in range(len(all_ids)) if mask_select[i]]
    target_genes = [str(gene_col.iloc[i]) if not pd.isna(gene_col.iloc[i]) else ''
                    for i in range(len(all_ids)) if mask_select[i]]
    is_nov_flag  = [is_novel[i] for i in range(len(all_ids)) if mask_select[i]]
    return target_ids, target_genes, is_nov_flag

def build_orf_map(target_ids, is_novel_flags):
    """Classification CSV에서 {id: orf_seq} 딕셔너리 구축."""
    print("  Loading classification CSV...")
    clf = pd.read_csv(CLF_CSV,
                      usecols=['isoform', 'transcript_name', 'gene_name', 'coding', 'ORF_seq'],
                      low_memory=False)

    # novel: isoform 칼럼 (transcript*.nnic)
    # known: transcript_name 칼럼 (GENE-201)
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
    print(f"  Target: {len(target_ids)}  coding: {n_coding}  non-coding: {len(target_ids)-n_coding}")
    return orf_seqs, coding_mask

def compute_esm2(orf_seqs, coding_mask, gpu_id):
    """ESM-2 t30_150M 임베딩 계산. non-coding은 zero vector."""
    import esm

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    coding_indices = [i for i, m in enumerate(coding_mask) if m == 1]
    coding_seqs    = [(str(i), orf_seqs[i]) for i in coding_indices]

    EMB_DIM = 640
    embeddings = np.zeros((len(orf_seqs), EMB_DIM), dtype=np.float32)

    n_batches = (len(coding_seqs) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    print(f"  Computing {len(coding_seqs)} ESM-2 embeddings in {n_batches} batches...")

    with torch.no_grad():
        for bi in range(0, len(coding_seqs), BATCH_SIZE):
            batch = coding_seqs[bi:bi+BATCH_SIZE]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[30], return_contacts=False)
            reps = out['representations'][30]  # (B, L+2, 640)
            # mean pooling (excluding BOS/EOS)
            for j, (orig_i_str, seq) in enumerate(batch):
                orig_i = int(orig_i_str)
                seq_len = min(len(seq), MAX_LEN)
                emb = reps[j, 1:seq_len+1, :].mean(0).cpu().numpy()
                embeddings[coding_indices[bi//BATCH_SIZE*BATCH_SIZE + j]] = emb

            if (bi // BATCH_SIZE + 1) % 20 == 0:
                done = bi + len(batch)
                elapsed = time.time() - t0
                eta = elapsed / done * (len(coding_seqs) - done)
                print(f"    [{done}/{len(coding_seqs)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    print(f"  Done in {time.time()-t0:.0f}s")
    return embeddings

def main():
    args = parse_args()
    out_dir = os.path.join(OUT_BASE, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Brain IsoQuant ESM-2 [{args.mode.upper()} mode]")
    print(f"  Model: {MODEL_NAME}  GPU: {args.gpu}")
    print("=" * 60)

    # 1. 대상 isoform 로드
    print("\n[1/3] Loading target isoforms...")
    target_ids, target_genes, is_novel_flags = load_target_isoforms(args.mode)
    print(f"  Target isoforms: {len(target_ids)}")

    # 2. ORF 서열 매핑
    print("\n[2/3] Building ORF map from SQANTI3 classification...")
    orf_seqs, coding_mask = build_orf_map(target_ids, is_novel_flags)

    # 3. ESM-2 계산
    print("\n[3/3] Computing ESM-2 embeddings...")
    embeddings = compute_esm2(orf_seqs, coding_mask, args.gpu)

    # 4. 저장
    prefix = f'brain_{args.mode}'
    np.save(f'{out_dir}/{prefix}_esm2_t30_150M.npy', embeddings)
    np.save(f'{out_dir}/{prefix}_ids.npy',            np.array(target_ids))
    np.save(f'{out_dir}/{prefix}_gene_names.npy',     np.array(target_genes))
    np.save(f'{out_dir}/{prefix}_mask.npy',           np.array(coding_mask, dtype=np.int8))

    print(f"\n  Saved to {out_dir}/")
    print(f"  {prefix}_esm2_t30_150M.npy : {embeddings.shape}")
    print(f"  {prefix}_ids.npy           : {len(target_ids)}")
    print(f"  coding mask sum (valid emb): {sum(coding_mask)}")
    print("\nDONE")

if __name__ == '__main__':
    main()
