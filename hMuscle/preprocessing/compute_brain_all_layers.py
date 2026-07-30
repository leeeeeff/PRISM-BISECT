#!/usr/bin/env python3
"""
compute_brain_all_layers.py
---------------------------
뇌 조직 isoform (63994개) ESM-2 전체 30 layer 임베딩 추출.
기존 L07/L18/L27 재사용, 나머지 27개 layer 계산.

Checkpoint: layer 단위로 저장 후 다음으로 진행.
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

PROJECT_DIR = '/home/dhkim1674/Project_AD_with_refTSS_novel'
CLF_CSV     = f'{PROJECT_DIR}/02_Isoquant_Output/SQANTI3_output/isoforms_classification_with_tx_name_and_gene_name.csv'
COUNT_CSV   = f'{PROJECT_DIR}/04_Counts/Long_Read/Cell_Type/counts_by_cell_type/tx_counts_by_cell_type.csv'
OUT_DIR     = 'data/brain_isoquant_esm2/full'
BATCH_SIZE  = 16       # 30 layers memory: ~1.3 GB per batch
MAX_LEN     = 1022
# When CUDA_VISIBLE_DEVICES is set, the visible GPU is always index 0 to torch.
GPU_ID      = 0

# Layers to compute: all 30 minus what we already have
ALL_LAYERS       = list(range(1, 31))
SKIP_LAYERS      = [7, 18, 27]   # already exist
LAYERS_TO_COMPUTE = [L for L in ALL_LAYERS if L not in SKIP_LAYERS]

# Group layers to keep GPU memory manageable
# Process 5 layers at a time per forward pass
GROUP_SIZE = 5


def load_target_isoforms():
    cnt = pd.read_csv(COUNT_CSV, index_col=0)
    all_ids = list(cnt.index)
    gene_col = cnt['gene_name'] if 'gene_name' in cnt.columns else pd.Series('', index=cnt.index)
    target_genes = [str(gene_col.iloc[i]) if not pd.isna(gene_col.iloc[i]) else ''
                    for i in range(len(all_ids))]
    is_novel = [str(tid).startswith('transcript') for tid in all_ids]
    return all_ids, target_genes, is_novel


def build_orf_map(target_ids, is_novel_flags):
    print("  Loading classification CSV...", flush=True)
    clf = pd.read_csv(CLF_CSV,
                      usecols=['isoform', 'transcript_name', 'gene_name',
                               'coding', 'ORF_seq'],
                      low_memory=False)
    novel_map = dict(zip(clf['isoform'], clf['ORF_seq']))
    known_map = dict(zip(clf['transcript_name'], clf['ORF_seq']))

    orf_seqs, coding_mask = [], []
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
    print(f"  Target: {len(target_ids)}  coding: {n_coding}  "
          f"non-coding: {len(target_ids)-n_coding}", flush=True)
    return orf_seqs, coding_mask


def compute_layer_group(orf_seqs, coding_mask, layer_group, gpu_id):
    """Compute ESM-2 mean-pooled embeddings for a group of layers in a single forward pass."""
    import esm
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}  layers={layer_group}", flush=True)

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    coding_indices = [i for i, m in enumerate(coding_mask) if m == 1]
    coding_seqs = [(str(i), orf_seqs[i]) for i in coding_indices]

    N = len(orf_seqs)
    EMB_DIM = 640
    mats = {L: np.zeros((N, EMB_DIM), dtype=np.float32) for L in layer_group}

    t0 = time.time()
    n_batches = (len(coding_seqs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Computing {len(coding_seqs)} × {len(layer_group)} layers in {n_batches} batches "
          f"(batch_size={BATCH_SIZE})", flush=True)

    with torch.no_grad():
        for bi in range(0, len(coding_seqs), BATCH_SIZE):
            batch = coding_seqs[bi:bi + BATCH_SIZE]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=layer_group, return_contacts=False)

            for j, (orig_i_str, seq) in enumerate(batch):
                orig_i = int(orig_i_str)
                seq_len = min(len(seq), MAX_LEN)
                for L in layer_group:
                    reps = out['representations'][L]
                    emb = reps[j, 1:seq_len + 1, :].mean(0).cpu().numpy()
                    mats[L][orig_i] = emb

            batch_idx = bi // BATCH_SIZE
            if (batch_idx + 1) % 40 == 0:
                done = bi + len(batch)
                elapsed = time.time() - t0
                eta = elapsed / done * (len(coding_seqs) - done) if done < len(coding_seqs) else 0
                print(f"    [{done}/{len(coding_seqs)}] {elapsed:.0f}s  ETA {eta:.0f}s",
                      flush=True)

    print(f"  Group {layer_group} done in {time.time()-t0:.0f}s", flush=True)

    for L in layer_group:
        out_path = f'{OUT_DIR}/brain_full_esm2_layer{L:02d}_t30_150M.npy'
        np.save(out_path, mats[L])
        norms = np.linalg.norm(mats[L], axis=1)
        print(f"    Saved: layer{L:02d}  norm_mean={norms[norms>0].mean():.4f}",
              flush=True)

    del model
    torch.cuda.empty_cache()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60, flush=True)
    print(f"  Brain ALL-layer ESM-2 extraction  GPU={GPU_ID}", flush=True)
    print(f"  Layers to compute: {LAYERS_TO_COMPUTE}", flush=True)
    print(f"  Group size: {GROUP_SIZE}", flush=True)
    print("=" * 60, flush=True)

    # Filter already-computed
    remaining = []
    for L in LAYERS_TO_COMPUTE:
        p = f'{OUT_DIR}/brain_full_esm2_layer{L:02d}_t30_150M.npy'
        if os.path.exists(p):
            print(f"  Skip L{L:02d}: already exists", flush=True)
        else:
            remaining.append(L)
    print(f"  Remaining: {remaining}", flush=True)

    if not remaining:
        print("All layers already computed. Nothing to do.", flush=True)
        return

    print(f"\n[1/3] Loading target isoforms...", flush=True)
    target_ids, target_genes, is_novel_flags = load_target_isoforms()
    print(f"  Target isoforms: {len(target_ids)}", flush=True)

    print(f"\n[2/3] Building ORF map...", flush=True)
    orf_seqs, coding_mask = build_orf_map(target_ids, is_novel_flags)

    print(f"\n[3/3] Computing ESM-2 layer embeddings in groups...", flush=True)
    groups = [remaining[i:i + GROUP_SIZE] for i in range(0, len(remaining), GROUP_SIZE)]
    print(f"  Total groups: {len(groups)}", flush=True)

    t0_all = time.time()
    for gi, group in enumerate(groups, 1):
        print(f"\n--- Group {gi}/{len(groups)}: layers {group} ---", flush=True)
        compute_layer_group(orf_seqs, coding_mask, group, GPU_ID)
        elapsed = time.time() - t0_all
        remaining_groups = len(groups) - gi
        eta = elapsed / gi * remaining_groups if gi > 0 else 0
        print(f"  === Cumulative: {elapsed/60:.1f} min  |  ETA {eta/60:.1f} min ===",
              flush=True)

    print(f"\nDONE. Total time: {(time.time()-t0_all)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
