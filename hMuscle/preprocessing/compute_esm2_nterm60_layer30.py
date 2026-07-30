#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_esm2_nterm60_layer30.py
--------------------------------
A2 candidate feature: N-terminal-window (first 60 residues) mean-pool at layer 30, for the FULL
train+test corpus. This is the standalone-isoform analog of the offline "edit-core" signal that
showed beyond-composition predictive power for N-terminal degron-type positional SLiMs
(b_elm_beyond_comp.py: DEG_Nend_Nbox +0.239, UBRbox +0.236, LIG_BIR_II +0.197) -- that signal was
defined via paired isoform comparison (long vs short edit region), which has no direct analog for
PRISM's real per-isoform (unpaired) scoring pipeline. This N-term-60aa window mean-pool is the most
faithful standalone-isoform translation (NTERM_WIN=60 convention already used elsewhere in the
project, e.g. explore_internal_edit_covariates.py).

Same ESM-2 forward pass cost as the original full-sequence mean-pool extraction (attention needs the
whole sequence for correct context) -- only layer 30 requested (not all 30) and only ONE additional
(N,640) array per split, not per-residue storage, to keep this a modest incremental job.

출력:
  hMuscle/data/esm2_nterm60_layer30_t30_150M.npy        (36748, 640)  test
  hMuscle/data/esm2_train_nterm60_layer30_t30_150M.npy  (31668, 640)  train

실행 (nice, single GPU):
  cd hMuscle/preprocessing/
  nice -n 10 python3 -u compute_esm2_nterm60_layer30.py --gpu 1 \
      > ../logs_isoform/nterm60_layer30_$(date +%Y%m%d_%H%M).log 2>&1
"""
import os
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '4'
import re
import time
import argparse
import numpy as np
import torch
from datetime import datetime

DATA_DIR = '../data'
NTERM_WIN = 60
MAX_LEN = 1022
MODEL_TAG = 'esm2_t30_150M_UR50D'
LAYER = 30

TEST_ISO_LIST = '../model/my_isoform_list_fixed.npy'
TEST_PEP_FILE = '../data/top30k_isoforms.pep'
TRAIN_ISO_LIST = '../data/raw_data/data/id_lists/train_isoform_list.npy'
TRAIN_RAW_SEQ = '../data/raw_data/data/raw_data/sequence_data/isoform_cds_sequences.txt'

TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}


def parse_test_pep(pep_path, max_len=MAX_LEN):
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

    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m_id = re.match(r'>(\S+)', line)
                m_type = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len = re.search(r'len:(\d+)', line)
                if not m_id:
                    cur_id = None; continue
                raw_id = m_id.group(1)
                cur_id = re.sub(r'\.p\d+$', '', raw_id)
                orf_type = m_type.group(1) if m_type else 'internal'
                score = float(m_score.group(1)) if m_score else 0.0
                length = int(m_len.group(1)) if m_len else 0
                rank = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score, length)
            else:
                cur_seq.append(line)
    flush()
    return {k: v[3][:max_len] for k, v in records.items()}


def parse_train_seq(path, max_len=MAX_LEN):
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
def compute_batch_nterm(model, batch_converter, sequences, device):
    """sequences: list of (label, seq). Returns (B,640) mean over first NTERM_WIN residues."""
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)
    results = model(tokens, repr_layers=[LAYER], return_contacts=False)
    token_reps = results['representations'][LAYER]
    embs = []
    for i, (_, seq) in enumerate(sequences):
        n = min(len(seq), NTERM_WIN)
        embs.append(token_reps[i, 1:1 + n, :].mean(0).cpu().float().numpy())
    return np.array(embs)


def run_split(name, iso_list_path, seqs_dict, out_path, model, batch_converter, device,
              batch_size=32, log_every=50):
    if os.path.exists(out_path):
        print(f'  [SKIP] {name}: {out_path} already exists')
        return
    iso_arr = np.load(iso_list_path, allow_pickle=True)
    iso_list = [s.decode() if isinstance(s, bytes) else str(s) for s in iso_arr]
    N = len(iso_list)
    out = np.zeros((N, 640), dtype=np.float32)
    valid = [(i, iso_list[i]) for i in range(N) if iso_list[i] in seqs_dict]
    print(f'  {name}: {len(valid)}/{N} sequences resolved')

    t0 = time.time()
    for bstart in range(0, len(valid), batch_size):
        batch = valid[bstart:bstart + batch_size]
        seq_batch = [(iso_id, seqs_dict[iso_id]) for i, iso_id in batch]
        embs = compute_batch_nterm(model, batch_converter, seq_batch, device)
        for (idx, _), e in zip(batch, embs):
            out[idx] = e
        if (bstart // batch_size) % log_every == 0:
            elapsed = time.time() - t0
            print(f'    [{name}] {bstart+len(batch)}/{len(valid)}  ({elapsed:.0f}s)', flush=True)

    np.save(out_path, out)
    print(f'  [SAVE] {name} -> {out_path}  {out.shape}  ({time.time()-t0:.0f}s total)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[{datetime.now():%H:%M:%S}] Device: {device}  NTERM_WIN={NTERM_WIN}  layer={LAYER}')

    import esm
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    print(f'[{datetime.now():%H:%M:%S}] Model loaded. embed_dim={model.embed_dim}')

    print(f'[{datetime.now():%H:%M:%S}] Parsing test sequences...')
    test_seqs = parse_test_pep(TEST_PEP_FILE)
    print(f'  {len(test_seqs)} test sequences parsed')
    run_split('test', TEST_ISO_LIST, test_seqs,
               os.path.join(DATA_DIR, 'esm2_nterm60_layer30_t30_150M.npy'),
               model, batch_converter, device, args.batch_size)

    print(f'[{datetime.now():%H:%M:%S}] Parsing train sequences...')
    train_seqs = parse_train_seq(TRAIN_RAW_SEQ)
    print(f'  {len(train_seqs)} train sequences parsed')
    run_split('train', TRAIN_ISO_LIST, train_seqs,
               os.path.join(DATA_DIR, 'esm2_train_nterm60_layer30_t30_150M.npy'),
               model, batch_converter, device, args.batch_size)

    print(f'[{datetime.now():%H:%M:%S}] [done]')


if __name__ == '__main__':
    main()
