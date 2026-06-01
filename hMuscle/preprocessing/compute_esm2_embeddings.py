#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_esm2_embeddings.py
--------------------------
목적: my_isoform_list_fixed.npy의 36,748개 BambuTx 이소폼에 대해
     ESM-2 mean-pooled protein embeddings를 사전 계산하여 저장.

입력:
  ../data/top30k_isoforms.pep   ← TransDecoder 단백질 서열 (all BambuTx 커버)
  ../model/my_isoform_list_fixed.npy ← 36,748개 이소폼 ID (고정)

출력:
  ../data/esm2_embeddings_t30_150M.npy  shape: (36748, 640)  float32
  ../data/esm2_mask.npy                 shape: (36748, 1)    float32  (유효=1, 0벡터=0)

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=0 python compute_esm2_embeddings.py [--model MODEL] [--batch_size N]

옵션:
  --model      esm2_t12_35M_UR50D | esm2_t30_150M_UR50D | esm2_t33_650M_UR50D
               (default: esm2_t30_150M_UR50D)
  --batch_size 처리 배치 크기 (default: 64, OOM 시 32로 줄이기)
  --max_len    최대 아미노산 길이, 초과 시 N-terminal truncation (default: 1022)
  --pep_file   단백질 서열 파일 경로 (default: ../data/top30k_isoforms.pep)
  --iso_list   이소폼 ID 파일 경로 (default: ../model/my_isoform_list_fixed.npy)
  --output     출력 파일 경로 (default: ../data/esm2_embeddings_{model_tag}.npy)
  --gpu        사용할 GPU (default: 0)
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import torch
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 인수 파싱
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Compute ESM-2 embeddings for isoforms')
    p.add_argument('--model',      default='esm2_t30_150M_UR50D',
                   choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
                            'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_len',    type=int, default=1022)
    p.add_argument('--pep_file',   default='../data/top30k_isoforms.pep')
    p.add_argument('--iso_list',   default='../model/my_isoform_list_fixed.npy')
    p.add_argument('--output',     default=None,
                   help='Output .npy path (auto-named if not set)')
    p.add_argument('--gpu',        type=int, default=0)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# PEP 파일 파싱 — 이소폼당 최고 ORF 1개 선택
# ─────────────────────────────────────────────────────────────
def parse_pep_file(pep_path, max_len=1022):
    """
    Returns: dict {isoform_id: aa_sequence (str, max max_len aa)}

    ORF 선택 우선순위:
      1) ORF type: complete > 5prime_partial > 3prime_partial > internal
      2) 동점 시 TransDecoder score 최대값
      3) 동점 시 len 최대값
    """
    TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}

    records  = {}   # isoform_id → (rank, score, len, seq)
    cur_id   = None
    cur_meta = None
    cur_seq  = []

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
                # >BambuTx10.p1 GENE.BambuTx10~~BambuTx10.p1  ORF type:complete (+),score=218.69 len:688 ...
                m_id    = re.match(r'>(\S+)', line)
                m_type  = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len   = re.search(r'len:(\d+)', line)

                if not m_id:
                    cur_id = None; continue

                raw_id  = m_id.group(1)
                cur_id  = re.sub(r'\.p\d+$', '', raw_id)  # BambuTx10.p1 → BambuTx10

                orf_type = m_type.group(1)  if m_type  else 'internal'
                score    = float(m_score.group(1)) if m_score else 0.0
                length   = int(m_len.group(1))     if m_len   else 0
                rank     = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score, length)
            else:
                cur_seq.append(line)

    flush()

    # N-terminal truncation for sequences > max_len
    result = {}
    for iso_id, (_, _, _, seq) in records.items():
        result[iso_id] = seq[:max_len]

    return result

# ─────────────────────────────────────────────────────────────
# ESM-2 배치 추론 — mean pool
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_esm2_batch(model, batch_converter, sequences, device, repr_layer):
    """
    sequences: list of (label, aa_seq) tuples
    returns: np.ndarray shape (len(sequences), embed_dim)
    """
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    # forward
    results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    token_reps = results['representations'][repr_layer]  # (B, L+2, D)

    # mean pool over residue positions (exclude BOS/EOS tokens)
    embs = []
    for i, (_, seq) in enumerate(sequences):
        seq_len = len(seq)
        rep = token_reps[i, 1:seq_len + 1, :]  # (seq_len, D)
        embs.append(rep.mean(dim=0).cpu().float().numpy())

    return np.array(embs)  # (B, D)

# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # GPU 설정
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now():%H:%M:%S}] Device: {device}")

    # 출력 경로
    model_tag = args.model.replace('esm2_', '').replace('_UR50D', '')
    if args.output is None:
        out_dir = os.path.dirname(args.iso_list).replace('model', 'data')
        args.output = os.path.join(out_dir, f'esm2_embeddings_{model_tag}.npy')
    mask_output = args.output.replace('.npy', '_mask.npy')

    # 이소폼 목록 로딩
    print(f"[{datetime.now():%H:%M:%S}] Loading isoform list: {args.iso_list}")
    iso_arr = np.load(args.iso_list, allow_pickle=True)
    iso_list = [s.decode() if isinstance(s, bytes) else s for s in iso_arr]
    N = len(iso_list)
    print(f"  Total isoforms: {N}")

    # PEP 파싱
    print(f"[{datetime.now():%H:%M:%S}] Parsing pep file: {args.pep_file}")
    pep_seqs = parse_pep_file(args.pep_file, max_len=args.max_len)
    print(f"  Parsed {len(pep_seqs)} unique isoforms with protein seqs")

    # 이소폼 → 서열 매핑 (없으면 None)
    seqs_ordered = []
    n_missing = 0
    for iso_id in iso_list:
        seq = pep_seqs.get(iso_id, None)
        if seq is None:
            n_missing += 1
        seqs_ordered.append(seq)
    print(f"  Coverage: {N - n_missing}/{N} ({(N - n_missing)/N*100:.1f}%)")
    if n_missing > 0:
        print(f"  Missing (→ zero vector): {n_missing}")

    # ESM-2 모델 로딩
    print(f"[{datetime.now():%H:%M:%S}] Loading ESM-2 model: {args.model}")
    import esm
    loader = getattr(esm.pretrained, args.model)
    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers          # last layer
    embed_dim  = model.embed_dim
    print(f"  embed_dim={embed_dim}, repr_layer={repr_layer}")
    gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
    print(f"  GPU memory (model): {gpu_mem:.2f} GB")

    # 결과 배열 초기화 (0-vector for missing)
    emb_matrix = np.zeros((N, embed_dim), dtype=np.float32)
    mask_matrix = np.zeros((N, 1), dtype=np.float32)

    # 유효 이소폼 인덱스 수집
    valid_indices = [(i, iso_list[i], seqs_ordered[i])
                     for i in range(N) if seqs_ordered[i] is not None]

    # 배치 처리
    batch_size = args.batch_size
    n_valid = len(valid_indices)
    n_batches = (n_valid + batch_size - 1) // batch_size
    t_start = time.time()

    print(f"\n[{datetime.now():%H:%M:%S}] Starting ESM-2 inference")
    print(f"  Valid sequences: {n_valid} | Batch size: {batch_size} | Batches: {n_batches}")

    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_end   = min(b_start + batch_size, n_valid)
        batch   = valid_indices[b_start:b_end]

        indices  = [x[0] for x in batch]
        labels   = [x[1] for x in batch]
        aa_seqs  = [x[2] for x in batch]
        sequences = list(zip(labels, aa_seqs))

        try:
            embs = compute_esm2_batch(model, batch_converter, sequences, device, repr_layer)
            for k, idx in enumerate(indices):
                emb_matrix[idx]  = embs[k]
                mask_matrix[idx] = 1.0
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n  [OOM] Batch {b_idx}: batch_size {batch_size} → try reducing --batch_size")
                torch.cuda.empty_cache()
                # 절반 크기로 재시도
                half = batch_size // 2
                for sub_start in range(0, len(batch), half):
                    sub = batch[sub_start:sub_start + half]
                    sub_idx  = [x[0] for x in sub]
                    sub_seqs = list(zip([x[1] for x in sub], [x[2] for x in sub]))
                    sub_embs = compute_esm2_batch(model, batch_converter, sub_seqs, device, repr_layer)
                    for k, idx in enumerate(sub_idx):
                        emb_matrix[idx]  = sub_embs[k]
                        mask_matrix[idx] = 1.0
            else:
                raise

        # 진행 출력 (10배치마다)
        if (b_idx + 1) % 10 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t_start
            done    = b_end
            eta     = elapsed / done * (n_valid - done) if done < n_valid else 0
            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
            print(f"  [{b_idx+1:4d}/{n_batches}] {done}/{n_valid} seqs "
                  f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s | GPU {gpu_mem:.2f}GB")

    # 저장
    print(f"\n[{datetime.now():%H:%M:%S}] Saving embeddings...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, emb_matrix)
    np.save(mask_output, mask_matrix)
    print(f"  Saved: {args.output}  shape={emb_matrix.shape}")
    print(f"  Saved: {mask_output}  valid={int(mask_matrix.sum())}/{N}")

    # 검증
    print(f"\n[검증]")
    assert not np.isnan(emb_matrix).any(), "NaN detected!"
    assert not np.isinf(emb_matrix).any(), "Inf detected!"
    norms = np.linalg.norm(emb_matrix[mask_matrix[:,0] == 1], axis=1)
    print(f"  Valid embedding L2 norm — mean={norms.mean():.4f}  std={norms.std():.4f}  min={norms.min():.4f}")
    print(f"  Zero rows: {(emb_matrix == 0).all(axis=1).sum()} (= missing isoforms)")

    total_time = time.time() - t_start
    print(f"\n[{datetime.now():%H:%M:%S}] Done. Total time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
