#!/usr/bin/env python3
"""
compute_brain_esm2.py
=====================
목적: 39,728개 뇌 고유 ENST 이소폼의 ESM-2 t30_150M 임베딩 계산
     → brain_only_esm2_t30_150M.npy (39728, 640)

이후 20,437개 근육 overlap 임베딩과 합산 →
     brain_esm2_60165.npy (60165, 640) — 전체 뇌 전사체 커버

입력:
  data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep
  /tmp/brain_only_enst.txt   (39,728 ENST IDs, sorted)

출력:
  data/brain_esm2/brain_only_esm2_t30_150M.npy   (39728, 640)
  data/brain_esm2/brain_only_esm2_mask.npy        (39728, 1)

실행:
  cd hMuscle/preprocessing/
  CUDA_VISIBLE_DEVICES=1 python compute_brain_esm2.py
"""

import os, sys, re, time, argparse
import numpy as np
import torch
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

PEP_FILE  = 'data/brain_esm2/brain_only_transcripts.fa.transdecoder.pep'
ISO_LIST  = '/tmp/brain_only_enst.txt'
OUT_NPY   = 'data/brain_esm2/brain_only_esm2_t30_150M.npy'
MASK_NPY  = 'data/brain_esm2/brain_only_esm2_mask.npy'
MODEL_NAME = 'esm2_t30_150M_UR50D'
BATCH_SIZE = 64
MAX_LEN    = 1022
GPU        = 1

TYPE_RANK = {'complete': 4, '5prime_partial': 3, '3prime_partial': 2, 'internal': 1}


def parse_pep_file(pep_path, max_len=1022):
    """
    TransDecoder .pep → {enst_base: aa_sequence}
    헤더 예: >ENST00000000233.3.p1 GENE.ENST...  ORF type:complete (+),score=...
    ENST ID: version 제거 + .p1 제거
    """
    records = {}
    cur_id = cur_meta = None
    cur_seq = []

    def flush():
        nonlocal cur_id, cur_meta, cur_seq
        if cur_id is None: return
        seq = ''.join(cur_seq).replace('*', '').strip()
        if not seq: return
        rank, score, length = cur_meta
        prev = records.get(cur_id)
        if prev is None or (rank, score, length) > prev[:3]:
            records[cur_id] = (rank, score, length, seq)

    with open(pep_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush(); cur_seq = []
                m_id    = re.match(r'>(\S+)', line)
                m_type  = re.search(r'ORF type:(\S+)', line)
                m_score = re.search(r'score=([\d.]+)', line)
                m_len   = re.search(r'len:(\d+)', line)
                if not m_id: cur_id = None; continue
                raw_id = m_id.group(1)
                # ENST00000000233.3.p1 → ENST00000000233
                cur_id = re.sub(r'\.\d+$', '', re.sub(r'\.p\d+$', '', raw_id))
                orf_type = m_type.group(1) if m_type else 'internal'
                score_val = float(m_score.group(1)) if m_score else 0.0
                length_val = int(m_len.group(1)) if m_len else 0
                rank = TYPE_RANK.get(orf_type.split('(')[0], 1)
                cur_meta = (rank, score_val, length_val)
            else:
                cur_seq.append(line)
    flush()

    return {k: v[3][:max_len] for k, v in records.items()}


def compute_esm2_batch(model, batch_converter, sequences, device, repr_layer):
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    token_reps = results["representations"][repr_layer]
    embs = []
    for i, (_, seq) in enumerate(sequences):
        L = len(seq)
        embs.append(token_reps[i, 1:L+1].mean(0).cpu().numpy())
    return embs


def main():
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now():%H:%M:%S}] Device: {device}")

    # Load ENST ID list (39,728 brain-only, no version)
    with open(ISO_LIST) as f:
        iso_list = [line.strip() for line in f if line.strip()]
    N = len(iso_list)
    print(f"  Brain-only ENSTs: {N}")

    # Parse protein sequences
    print(f"[{datetime.now():%H:%M:%S}] Parsing pep: {PEP_FILE}")
    pep_seqs = parse_pep_file(PEP_FILE, max_len=MAX_LEN)
    print(f"  Parsed {len(pep_seqs)} unique proteins")

    seqs_ordered = [pep_seqs.get(iso_id) for iso_id in iso_list]
    n_missing = sum(1 for s in seqs_ordered if s is None)
    print(f"  Coverage: {N - n_missing}/{N} ({(N - n_missing)/N*100:.1f}%)")
    if n_missing > 0:
        print(f"  Missing (→ zero vector): {n_missing}")

    # Load ESM-2
    print(f"[{datetime.now():%H:%M:%S}] Loading {MODEL_NAME}")
    import esm
    loader = getattr(esm.pretrained, MODEL_NAME)
    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    embed_dim  = model.embed_dim
    print(f"  embed_dim={embed_dim}, repr_layer={repr_layer}")

    emb_matrix  = np.zeros((N, embed_dim), dtype=np.float32)
    mask_matrix = np.zeros((N, 1), dtype=np.float32)

    valid_indices = [(i, iso_list[i], seqs_ordered[i])
                     for i in range(N) if seqs_ordered[i] is not None]
    n_valid   = len(valid_indices)
    n_batches = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE
    t_start   = time.time()

    print(f"\n[{datetime.now():%H:%M:%S}] ESM-2 inference: {n_valid} seqs, {n_batches} batches")

    for b_idx in range(n_batches):
        b_start = b_idx * BATCH_SIZE
        b_end   = min(b_start + BATCH_SIZE, n_valid)
        batch   = valid_indices[b_start:b_end]

        indices = [x[0] for x in batch]
        seqs    = list(zip([x[1] for x in batch], [x[2] for x in batch]))

        try:
            embs = compute_esm2_batch(model, batch_converter, seqs, device, repr_layer)
            for k, idx in enumerate(indices):
                emb_matrix[idx]  = embs[k]
                mask_matrix[idx] = 1.0
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                half = BATCH_SIZE // 2
                for ss in range(0, len(batch), half):
                    sub = batch[ss:ss+half]
                    sub_idx  = [x[0] for x in sub]
                    sub_seqs = list(zip([x[1] for x in sub], [x[2] for x in sub]))
                    sub_embs = compute_esm2_batch(model, batch_converter, sub_seqs, device, repr_layer)
                    for k, idx in enumerate(sub_idx):
                        emb_matrix[idx]  = sub_embs[k]
                        mask_matrix[idx] = 1.0
            else:
                raise

        if (b_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (b_idx + 1) * BATCH_SIZE / elapsed
            eta  = (n_valid - (b_idx + 1) * BATCH_SIZE) / max(rate, 1)
            pct = (b_idx + 1) / n_batches * 100
            print(f"  [{pct:5.1f}%] batch {b_idx+1}/{n_batches} "
                  f"| {rate:.0f} seq/s | ETA {eta/60:.1f}min")

    elapsed = time.time() - t_start
    print(f"\n[{datetime.now():%H:%M:%S}] Done: {elapsed/60:.1f} min")
    print(f"  Coverage rate: {mask_matrix.sum()/N*100:.1f}%")

    np.save(OUT_NPY, emb_matrix)
    np.save(MASK_NPY, mask_matrix)
    print(f"  Saved: {OUT_NPY}  shape={emb_matrix.shape}")
    print(f"  Saved: {MASK_NPY}")

    # Build combined 60,165 brain embedding
    print(f"\n[Building full brain embedding: 60,165 × 640]")
    build_combined_brain_embeddings(emb_matrix, mask_matrix, iso_list)


def build_combined_brain_embeddings(brain_only_emb, brain_only_mask, brain_only_list):
    """
    Combine:
      - 39,728 new brain-only ESM-2 embeddings
      - 20,437 existing muscle embeddings (reuse: same sequence = same embedding)
    → brain_esm2_60165.npy ordered by brain h5ad var index
    """
    import h5py
    from collections import defaultdict

    # Load muscle test embeddings
    X_te = np.load('data/esm2_embeddings_t30_150M.npy').astype(np.float32)
    te_isos = np.load('model/my_isoform_list_fixed.npy', allow_pickle=True)
    te_isos = [x.decode() if isinstance(x, bytes) else str(x) for x in te_isos]
    muscle_base_to_idx = defaultdict(list)
    for i, iso in enumerate(te_isos):
        base = iso.split('.')[0]
        if base.startswith('ENST'):
            muscle_base_to_idx[base].append(i)

    # Brain ENST order from h5ad
    f = h5py.File('/home/dhkim1674/Samsung_Alzheimer/03_AnnData/adata_transcript_for_UMAP.h5ad', 'r')
    brain_enst_arr = [x.decode() for x in f['var']['ENST_ID'][()]]
    f.close()
    n_brain = len(brain_enst_arr)  # 60,165

    brain_only_id_to_idx = {enst: i for i, enst in enumerate(brain_only_list)}

    emb_full = np.zeros((n_brain, 640), dtype=np.float32)
    mask_full = np.zeros(n_brain, dtype=np.float32)

    for brain_var_idx, enst in enumerate(brain_enst_arr):
        if enst in brain_only_id_to_idx:
            idx = brain_only_id_to_idx[enst]
            if brain_only_mask[idx, 0] > 0:
                emb_full[brain_var_idx] = brain_only_emb[idx]
                mask_full[brain_var_idx] = 1.0
        elif enst in muscle_base_to_idx:
            midx = muscle_base_to_idx[enst][0]
            emb_full[brain_var_idx] = X_te[midx]
            mask_full[brain_var_idx] = 1.0

    n_covered = int(mask_full.sum())
    print(f"  Combined: {n_covered}/{n_brain} ({n_covered/n_brain*100:.1f}%) covered")
    np.save('data/brain_esm2/brain_esm2_60165.npy', emb_full)
    np.save('data/brain_esm2/brain_esm2_60165_mask.npy', mask_full)
    print(f"  Saved: data/brain_esm2/brain_esm2_60165.npy  shape={emb_full.shape}")


if __name__ == '__main__':
    main()
