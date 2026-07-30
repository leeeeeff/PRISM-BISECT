#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_structmatch_null.py

M3 fix (devils-advocate MODERATE attack, confirmed live by
devils_m3_position_check.py -- 65.5%/56.2% of real long-side changed intervals
in muscle/brain fall in the first 10% of the protein, vs 10% expected under
uniform; the existing scrambled-window null draws a UNIFORM random start, so
it is NOT a fair null for "is region-pooling doing something beyond pooling
over an N-terminal window" -- a structure-matched null is needed).

Design: for each pair/side, instead of a uniform-random contiguous window
(the original scram), draw the window's relative START POSITION from the
EMPIRICAL distribution of REAL (non-fallback) interval start positions on
that tissue+side (precomputed once, all ~15-33k real positions pooled), then
place a length-matched (same L0 as the real interval) window there on the
CURRENT pair's sequence. This directly tests whether region-pooling's effect
survives against a null that pools "a window from roughly the same part of
the protein real edits usually occur in" rather than "a window from anywhere."

Reuses cached per-tissue sequences/intervals (same opcode_intervals recipe as
build_severity_region_embeddings_{muscle,brain}.py) -- requires ONE new ESM-2
forward pass (per-residue tensors were discarded after the original run, so
cannot be recovered without rerunning; same cost order as before, ~10-15 min).
Produces severity_score_structmatch alongside the existing severity_score,
severity_score_region, severity_score_scram columns (loaded from the region TSV).
"""
import os, time, importlib.util
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
SEV = ROOT / 'reports/severity_pairs'
TISSUE = os.environ.get('SM_TISSUE', 'muscle')
MAXLEN = 1022
N_LAYERS_KEEP = (15, 30)
GPU = int(os.environ.get('EMB_GPU', '0'))
MEM_FRAC = float(os.environ.get('EMB_MEM_FRAC', '0.35'))
BATCH = 8
N_FOLDS = 5
SEED = 42
NTERM_FALLBACK = 60
rng = np.random.default_rng(909)

spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
bsp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsp)


def opcode_intervals(long_s, short_s):
    sm = SequenceMatcher(None, long_s, short_s, autojunk=False)
    livs, sivs = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if i2 > i1:
            livs.append((i1, i2))
        if j2 > j1:
            sivs.append((j1, j2))
    return livs, sivs


def load_pairs_and_seqs():
    df = pd.read_csv(SEV / f'{TISSUE}_severity_pairs_region.tsv', sep='\t')
    if TISSUE == 'muscle':
        iso = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    else:
        iso = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
        iso = [s.decode() if isinstance(s, bytes) else str(s) for s in iso]
        seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    print(f"[load] {len(df)} {TISSUE} region-pool-valid pairs, {len(iso)} isoform ids", flush=True)
    return df, iso, seqs


def build_occurrences_and_position_pools(df, iso, seqs):
    """Rerun the exact opcode_intervals recipe to recover (a) per-occurrence
    real intervals with fallback handling, and (b) the empirical pool of
    REAL (non-fallback) relative start positions per side, for donor sampling."""
    occ = {}
    ivs_size = {}
    pos_pool = {'long': [], 'short': []}
    for row_idx, r in df.iterrows():
        long_id = iso[int(r['long_idx'])]
        short_id = iso[int(r['short_idx'])]
        long_s = seqs[long_id][:MAXLEN]
        short_s = seqs[short_id][:MAXLEN]
        livs, sivs = opcode_intervals(long_s, short_s)
        occ.setdefault(long_id, []).append((row_idx, 'long', livs, len(long_s)))
        occ.setdefault(short_id, []).append((row_idx, 'short', sivs, len(short_s)))
        ivs_size[(row_idx, 'long')] = sum(v - u for u, v in livs) if livs else min(NTERM_FALLBACK, len(long_s))
        ivs_size[(row_idx, 'short')] = sum(v - u for u, v in sivs) if sivs else min(NTERM_FALLBACK, len(short_s))
        if livs:
            pos_pool['long'].append(min(u for u, v in livs) / len(long_s))
        if sivs:
            pos_pool['short'].append(min(u for u, v in sivs) / len(short_s))
    for side in ['long', 'short']:
        pos_pool[side] = np.array(pos_pool[side])
        print(f"[pos_pool] {TISSUE} {side}-side: n={len(pos_pool[side])} "
              f"empirical real-interval start positions pooled for donor sampling", flush=True)
    iso_seq = {k: seqs[k][:MAXLEN] for k in occ}
    return occ, ivs_size, pos_pool, iso_seq


def run_esm2(iso_seq, occ, pos_pool):
    import esm
    dev = f'cuda:{GPU}'
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC, device=GPU)
    model, alph = esm.pretrained.esm2_t30_150M_UR50D()
    bc = alph.get_batch_converter()
    model.eval().to(dev)
    ids = sorted(iso_seq.keys(), key=lambda i: len(iso_seq[i]))
    n = len(ids)
    region15 = {}; region30 = {}; sm15 = {}; sm30 = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, BATCH):
            ch = ids[s:s + BATCH]
            _, _, toks = bc([(i, iso_seq[i]) for i in ch])
            toks = toks.to(dev)
            rep = model(toks, repr_layers=list(N_LAYERS_KEEP))['representations']
            r15 = rep[15].cpu().numpy(); r30 = rep[30].cpu().numpy()
            for bi, iso_id in enumerate(ch):
                Ln = len(iso_seq[iso_id])
                a15 = r15[bi, 1:Ln + 1, :]
                a30 = r30[bi, 1:Ln + 1, :]
                for (row_idx, side, ivs, seq_len) in occ[iso_id]:
                    cidx = [k for (u, v) in ivs for k in range(u, v) if k < Ln]
                    if not cidx:
                        cidx = list(range(min(NTERM_FALLBACK, Ln)))
                    L0 = len(cidx)
                    # structure-matched null: donor relative position from the
                    # EMPIRICAL real-interval-position pool (same side, same tissue)
                    donor_rel = pos_pool[side][rng.integers(0, len(pos_pool[side]))]
                    start = int(round(donor_rel * Ln))
                    start = max(0, min(start, max(0, Ln - L0)))
                    smidx = list(range(start, min(start + L0, Ln)))
                    if not smidx:
                        smidx = list(range(min(L0, Ln)))
                    key = (row_idx, side)
                    region15[key] = a15[cidx].mean(0); region30[key] = a30[cidx].mean(0)
                    sm15[key] = a15[smidx].mean(0); sm30[key] = a30[smidx].mean(0)
            if (s // BATCH) % 50 == 0:
                print(f"  [esm2] {s + len(ch)}/{n} isoforms, {time.time() - t0:.0f}s elapsed", flush=True)
    print(f"[esm2] done, {n} isoforms in {time.time() - t0:.0f}s", flush=True)
    return region15, region30, sm15, sm30


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def mean_pool_fold_directions(df_full):
    if TISSUE == 'muscle':
        L15 = np.load(ROOT / 'hMuscle/data/esm2_layer_15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/esm2_layer_30_t30_150M.npy').astype(np.float32)
    else:
        L15 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer15_t30_150M.npy').astype(np.float32)
        L30 = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_esm2_layer30_t30_150M.npy').astype(np.float32)
    emb = np.concatenate([L15, L30], axis=1)
    long_idx = df_full['long_idx'].to_numpy()
    short_idx = df_full['short_idx'].to_numpy()
    D = emb[long_idx] - emb[short_idx]
    fold = gene_disjoint_folds(df_full['gene'].to_numpy())
    directions = {}
    mean_scores = np.zeros(len(df_full))
    for k in range(N_FOLDS):
        train_mask = fold != k
        direction = D[train_mask].mean(axis=0)
        norm = np.linalg.norm(direction)
        directions[k] = direction / norm if norm > 0 else direction
        mean_scores[fold == k] = D[fold == k] @ directions[k]
    return fold, directions, mean_scores


def score_with_fixed_directions(D, fold, directions):
    scores = np.zeros(len(D))
    for k, direction in directions.items():
        mask = fold == k
        scores[mask] = D[mask] @ direction
    return scores


def main():
    df_full = pd.read_csv(SEV / f'{TISSUE}_severity_pairs_scored.tsv', sep='\t')
    df_full = df_full[df_full['tissue'] == TISSUE].reset_index(drop=True)
    fold_full, directions, mean_scores_full = mean_pool_fold_directions(df_full)
    assert np.allclose(mean_scores_full, df_full['severity_score'].to_numpy(), atol=1e-3), \
        f"[{TISSUE}] mean-pool severity_score reproduction failed"
    print(f"[check] {TISSUE} mean-pool severity_score reproduced on full population", flush=True)
    gene_to_fold = dict(zip(df_full['gene'].to_numpy(), fold_full))

    df, iso, seqs = load_pairs_and_seqs()

    smoke_limit = os.environ.get('SM_SMOKE_LIMIT')
    if smoke_limit:
        df = df.iloc[:int(smoke_limit)].reset_index(drop=True)
        print(f"[smoke] limited to {len(df)} pairs", flush=True)

    fold = df['gene'].map(gene_to_fold).to_numpy()

    occ, ivs_size, pos_pool, iso_seq = build_occurrences_and_position_pools(df, iso, seqs)
    region15, region30, sm15, sm30 = run_esm2(iso_seq, occ, pos_pool)

    n = len(df)
    D_structmatch = np.zeros((n, 1280), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)
    for pos, row_idx in enumerate(df.index):
        klong = (row_idx, 'long'); kshort = (row_idx, 'short')
        if klong not in sm15 or kshort not in sm15:
            continue
        sl = np.concatenate([sm15[klong], sm30[klong]])
        ss = np.concatenate([sm15[kshort], sm30[kshort]])
        D_structmatch[pos] = sl - ss
        valid_mask[pos] = True

    df = df.loc[valid_mask].reset_index(drop=True)
    fold = fold[valid_mask]
    D_structmatch = D_structmatch[valid_mask]
    df['severity_score_structmatch'] = score_with_fixed_directions(D_structmatch, fold, directions)

    out_tsv = SEV / f'{TISSUE}_severity_pairs_structmatch.tsv'
    out_npz = SEV / f'{TISSUE}_structmatch_embeddings.npz'
    df.to_csv(out_tsv, sep='\t', index=False)
    np.savez(out_npz, D_structmatch=D_structmatch)
    print(f"[done] wrote {out_tsv} ({len(df)} pairs) and {out_npz}", flush=True)
    print(df[['severity_score', 'severity_score_region', 'severity_score_scram',
               'severity_score_structmatch']].describe(), flush=True)


if __name__ == '__main__':
    main()
