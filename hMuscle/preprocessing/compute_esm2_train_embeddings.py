#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_esm2_train_embeddings.py
---------------------------------
목적: 훈련 단백질(human isoforms + SwissProt)의 ESM-2 임베딩 사전 계산.

입력:
  (human)
  ../data/raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt
      포맷: >GENE|NM_ID\\nSEQUENCE
  ../data/raw_data/data/id_lists/train_isoform_list.npy
      형식: ['NM_000015', 'NM_001322051', ...]  (31668개)

  (swissprot)
  ../data/raw_data/data/raw_data/sequence_data/swissprot_protein_sequences.txt
      포맷: >ACCESSION|ENTRY_NAME\\nSEQUENCE
  ../data/raw_data/data/id_lists/train_swissprot_list.npy
      형식: ['1433G_CHICK', '1433G_SHEEP', ...]  (82703개)

출력:
  ../data/esm2_train_human.npy          shape: (31668, 640)  float32
  ../data/esm2_train_human_mask.npy     shape: (31668, 1)    float32
  ../data/esm2_train_swissprot.npy      shape: (82703, 640)  float32
  ../data/esm2_train_swissprot_mask.npy shape: (82703, 1)    float32

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=1 nohup python compute_esm2_train_embeddings.py \\
      > ../logs_isoform/esm2_train_20260408.log 2>&1 &

옵션:
  --model      esm2_t6_8M_UR50D | esm2_t12_35M_UR50D | esm2_t30_150M_UR50D
               (default: esm2_t30_150M_UR50D)
  --batch_size 처리 배치 크기 (default: 64, OOM 시 32로 줄이기)
  --max_len    최대 아미노산 길이 (default: 1022)
  --gpu        사용할 GPU index (default: 0)
  --skip       'human' 또는 'swissprot' 중 하나만 건너뛰기
"""

import os
import re
import sys
import time
import argparse
import numpy as np
import torch
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# 인수 파싱
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Compute ESM-2 embeddings for training proteins')
    p.add_argument('--model',      default='esm2_t30_150M_UR50D',
                   choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D',
                            'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_len',    type=int, default=1022)
    p.add_argument('--gpu',        type=int, default=0)
    p.add_argument('--skip',       default=None, choices=['human', 'swissprot'],
                   help='Skip one of the two datasets')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# FASTA 파싱 — human isoform (>GENE|NM_ID 포맷)
#   key: NM_ID  (예: 'NM_001322051')
# ─────────────────────────────────────────────────────────────
def parse_human_seq(path, max_len=1022):
    """
    Returns: dict {nm_id: aa_seq}
    One entry per header line; key = NM_ part of '>GENE|NM_ID'.
    """
    records = {}
    cur_key = None
    cur_seq = []

    def flush():
        nonlocal cur_key, cur_seq
        if cur_key and cur_seq:
            seq = ''.join(cur_seq).replace('*', '').strip()
            if seq:
                records[cur_key] = seq[:max_len]

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                cur_seq = []
                # >GENE|NM_ID  형식 파싱
                m = re.match(r'>(\S+)\|(\S+)', line)
                cur_key = m.group(2) if m else None
            else:
                cur_seq.append(line)
    flush()
    return records


# ─────────────────────────────────────────────────────────────
# FASTA 파싱 — SwissProt (>ACCESSION|ENTRY_NAME 포맷)
#   key: ENTRY_NAME  (예: '1433G_CHICK')
# ─────────────────────────────────────────────────────────────
def parse_swissprot_seq(path, max_len=1022):
    """
    Returns: dict {entry_name: aa_seq}
    Key = second field after | in '>ACCESSION|ENTRY_NAME'.
    For entries without |, use the whole id.
    """
    records = {}
    cur_key = None
    cur_seq = []

    def flush():
        nonlocal cur_key, cur_seq
        if cur_key and cur_seq:
            seq = ''.join(cur_seq).replace('*', '').strip()
            if seq:
                records[cur_key] = seq[:max_len]

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                cur_seq = []
                m = re.match(r'>(\S+)\|(\S+)', line)
                if m:
                    cur_key = m.group(2)   # ENTRY_NAME
                else:
                    m2 = re.match(r'>(\S+)', line)
                    cur_key = m2.group(1) if m2 else None
            else:
                cur_seq.append(line)
    flush()
    return records


# ─────────────────────────────────────────────────────────────
# ESM-2 배치 추론 — mean pool (재사용)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_esm2_batch(model, batch_converter, sequences, device, repr_layer):
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)
    results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    token_reps = results['representations'][repr_layer]  # (B, L+2, D)

    embs = []
    for i, (_, seq) in enumerate(sequences):
        seq_len = len(seq)
        rep = token_reps[i, 1:seq_len + 1, :]
        embs.append(rep.mean(dim=0).cpu().float().numpy())
    return np.array(embs)


# ─────────────────────────────────────────────────────────────
# 단일 데이터셋 임베딩 계산
# ─────────────────────────────────────────────────────────────
def compute_embeddings(model, batch_converter, device, repr_layer,
                       id_list, seq_dict, batch_size, label):
    """
    id_list : list[str] — N개 ID (순서대로)
    seq_dict: dict{id: seq}
    Returns : (emb_matrix: (N,D) float32, mask: (N,1) float32)
    """
    N = len(id_list)
    embed_dim = model.embed_dim

    emb_matrix  = np.zeros((N, embed_dim), dtype=np.float32)
    mask_matrix = np.zeros((N, 1),         dtype=np.float32)

    # 유효 인덱스 수집
    valid = [(i, id_list[i], seq_dict.get(id_list[i]))
             for i in range(N) if seq_dict.get(id_list[i]) is not None]
    missing = N - len(valid)
    print("  [{}] Total={} | Valid={} | Missing(→0)={}".format(
        label, N, len(valid), missing))

    n_valid   = len(valid)
    n_batches = (n_valid + batch_size - 1) // batch_size
    t_start   = time.time()

    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_end   = min(b_start + batch_size, n_valid)
        batch   = valid[b_start:b_end]

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
                print("  [OOM] batch {}: retrying at half size".format(b_idx))
                torch.cuda.empty_cache()
                half = batch_size // 2
                for sub_start in range(0, len(batch), half):
                    sub = batch[sub_start:sub_start + half]
                    sub_idx  = [x[0] for x in sub]
                    sub_seqs = list(zip([x[1] for x in sub], [x[2] for x in sub]))
                    sub_embs = compute_esm2_batch(
                        model, batch_converter, sub_seqs, device, repr_layer)
                    for k, idx in enumerate(sub_idx):
                        emb_matrix[idx]  = sub_embs[k]
                        mask_matrix[idx] = 1.0
            else:
                raise

        if (b_idx + 1) % 20 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t_start
            done    = b_end
            eta     = elapsed / done * (n_valid - done) if done < n_valid else 0
            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
            print("  [{}] [{:4d}/{}] {}/{} seqs | {:.0f}s | ETA {:.0f}s | GPU {:.2f}GB".format(
                label, b_idx + 1, n_batches, done, n_valid, elapsed, eta, gpu_mem))

    return emb_matrix, mask_matrix


# ─────────────────────────────────────────────────────────────
# 저장 + 검증
# ─────────────────────────────────────────────────────────────
def save_and_verify(emb_matrix, mask_matrix, out_emb, out_mask):
    os.makedirs(os.path.dirname(out_emb), exist_ok=True)
    np.save(out_emb,  emb_matrix)
    np.save(out_mask, mask_matrix)
    print("  Saved: {}  shape={}".format(out_emb,  emb_matrix.shape))
    print("  Saved: {}  valid={}/{}".format(out_mask, int(mask_matrix.sum()), len(mask_matrix)))

    assert not np.isnan(emb_matrix).any(), "NaN detected!"
    assert not np.isinf(emb_matrix).any(), "Inf detected!"
    valid_mask = mask_matrix[:, 0] == 1
    if valid_mask.sum() > 0:
        norms = np.linalg.norm(emb_matrix[valid_mask], axis=1)
        print("  Valid L2 norm — mean={:.4f}  std={:.4f}  min={:.4f}".format(
            norms.mean(), norms.std(), norms.min()))
    zero_rows = (emb_matrix == 0).all(axis=1).sum()
    print("  Zero rows (missing): {}".format(zero_rows))


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print("[{}] Device: {}".format(datetime.now().strftime('%H:%M:%S'), device))

    # 경로 설정 (preprocessing/ 에서 실행 기준)
    DATA_DIR = '../data'
    RAW_SEQ  = os.path.join(DATA_DIR,
        'raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt')
    SSP_SEQ  = os.path.join(DATA_DIR,
        'raw_data/data/raw_data/sequence_data/swissprot_protein_sequences.txt')
    ISO_LIST = os.path.join(DATA_DIR, 'raw_data/data/id_lists/train_isoform_list.npy')
    SSP_LIST = os.path.join(DATA_DIR, 'raw_data/data/id_lists/train_swissprot_list.npy')

    model_tag = args.model.replace('esm2_', '').replace('_UR50D', '')

    OUT_HUMAN_EMB  = os.path.join(DATA_DIR, 'esm2_train_human_{}.npy'.format(model_tag))
    OUT_HUMAN_MASK = os.path.join(DATA_DIR, 'esm2_train_human_{}_mask.npy'.format(model_tag))
    OUT_SSP_EMB    = os.path.join(DATA_DIR, 'esm2_train_swissprot_{}.npy'.format(model_tag))
    OUT_SSP_MASK   = os.path.join(DATA_DIR, 'esm2_train_swissprot_{}_mask.npy'.format(model_tag))

    # 표준 이름 심볼릭 링크용 — t30_150M 이면 short alias도 저장
    if model_tag == 't30_150M':
        OUT_HUMAN_EMB_SHORT  = os.path.join(DATA_DIR, 'esm2_train_human.npy')
        OUT_HUMAN_MASK_SHORT = os.path.join(DATA_DIR, 'esm2_train_human_mask.npy')
        OUT_SSP_EMB_SHORT    = os.path.join(DATA_DIR, 'esm2_train_swissprot.npy')
        OUT_SSP_MASK_SHORT   = os.path.join(DATA_DIR, 'esm2_train_swissprot_mask.npy')
    else:
        OUT_HUMAN_EMB_SHORT = OUT_HUMAN_MASK_SHORT = None
        OUT_SSP_EMB_SHORT   = OUT_SSP_MASK_SHORT   = None

    # ── ESM-2 모델 로딩 ──────────────────────────────────────
    print("[{}] Loading ESM-2: {}".format(datetime.now().strftime('%H:%M:%S'), args.model))
    import esm
    loader = getattr(esm.pretrained, args.model)
    esm_model, alphabet = loader()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = esm_model.num_layers
    print("  embed_dim={} repr_layer={}".format(esm_model.embed_dim, repr_layer))

    t_total = time.time()

    # ── 1. Human isoforms ────────────────────────────────────
    if args.skip != 'human':
        print("\n[{}] === Human isoforms ===".format(datetime.now().strftime('%H:%M:%S')))

        print("  Parsing: {}".format(RAW_SEQ))
        human_seqs = parse_human_seq(RAW_SEQ, max_len=args.max_len)
        print("  Parsed {} unique NM_IDs".format(len(human_seqs)))

        iso_arr  = np.load(ISO_LIST, allow_pickle=True)
        iso_list = [x.decode() if isinstance(x, bytes) else x for x in iso_arr]
        print("  id_list size: {}".format(len(iso_list)))

        emb_h, mask_h = compute_embeddings(
            esm_model, batch_converter, device, repr_layer,
            iso_list, human_seqs, args.batch_size, label='Human')

        save_and_verify(emb_h, mask_h, OUT_HUMAN_EMB, OUT_HUMAN_MASK)
        # 표준 이름 저장 (t30_150M 기본)
        if OUT_HUMAN_EMB_SHORT:
            np.save(OUT_HUMAN_EMB_SHORT,  emb_h)
            np.save(OUT_HUMAN_MASK_SHORT, mask_h)
            print("  Also saved: {} / {}".format(OUT_HUMAN_EMB_SHORT, OUT_HUMAN_MASK_SHORT))

    # ── 2. SwissProt ─────────────────────────────────────────
    if args.skip != 'swissprot':
        print("\n[{}] === SwissProt ===".format(datetime.now().strftime('%H:%M:%S')))

        print("  Parsing: {}".format(SSP_SEQ))
        ssp_seqs = parse_swissprot_seq(SSP_SEQ, max_len=args.max_len)
        print("  Parsed {} unique ENTRY_NAMEs".format(len(ssp_seqs)))

        ssp_arr  = np.load(SSP_LIST, allow_pickle=True)
        ssp_list = [x.decode() if isinstance(x, bytes) else x for x in ssp_arr]
        print("  id_list size: {}".format(len(ssp_list)))

        emb_s, mask_s = compute_embeddings(
            esm_model, batch_converter, device, repr_layer,
            ssp_list, ssp_seqs, args.batch_size, label='SwissProt')

        save_and_verify(emb_s, mask_s, OUT_SSP_EMB, OUT_SSP_MASK)
        if OUT_SSP_EMB_SHORT:
            np.save(OUT_SSP_EMB_SHORT,  emb_s)
            np.save(OUT_SSP_MASK_SHORT, mask_s)
            print("  Also saved: {} / {}".format(OUT_SSP_EMB_SHORT, OUT_SSP_MASK_SHORT))

    total_time = time.time() - t_total
    print("\n[{}] All done. Total: {:.0f}s ({:.1f}min)".format(
        datetime.now().strftime('%H:%M:%S'), total_time, total_time / 60))


if __name__ == '__main__':
    main()
