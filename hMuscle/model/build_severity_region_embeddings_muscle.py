#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_severity_region_embeddings_muscle.py

Muscle-only pilot (user-selected Option C, 2026-07-21): scale region-pool ESM-2
embedding computation from the 2-iso decisive subset (18% of multi-isoform genes,
[[finding-region-pool-trajectory-pca]]) to the FULL canonical-anchored pair
expansion for muscle only (15,885 pairs / 6,001 genes / 21,886 unique isoforms,
reports/severity_pairs/muscle_severity_pairs_scored.tsv). Rationale for muscle
first: muscle splice edits are more fragmented (n_intervals mediation dominant,
89.2% of the size x tissue interaction, [[approach-severity-covariate-surrogate-chain]])
than brain, so a more diverse population of edit geometries is expected here --
brain scale-up is deferred pending this pilot's results (S3 parsimony).

Design -- reuses established, devils-advocate-vetted guards, does not invent new
methodology:
  - Interval definition: difflib.SequenceMatcher opcodes on (long_s, short_s)
    AFTER MAXLEN truncation, returning BOTH long-side and short-side coordinate
    intervals (opcode_intervals, identical to exp_domdiff_region_pool.py). This
    is NOT the same as build_severity_pairs.changed_intervals, which only
    records long-side coordinates and is insufficient for pooling the short
    isoform's own residues.
  - C1 (mandatory null): length-matched scrambled contiguous window per side,
    identical recipe to exp_traj_region_projection.py (random contiguous start,
    same length as the real changed region on that side).
  - C2 (mandatory, avoids the coherence-retrain / variance-confound pitfall in
    [[approach-axis-encoding-vs-usage-occlusion]]): the coherence direction used
    to score region-pool and scrambled-pool pairs is the SAME mean-pool-trained,
    gene-disjoint 5-fold direction as build_severity_score.py -- retrained here
    identically on mean-pool embeddings, never on region/scram embeddings.
  - Sequence source: hMuscle/data/top30k_isoforms.pep via
    build_severity_pairs.parse_pep_sequences (the CORRECT full source; the
    truncated muscle_2iso.fa was a prior, already-fixed bug -- see
    [[finding-severity-regression-canonical-anchored]]).
  - Layers: L15 + L30 only (matches severity_score's coherence-projection basis;
    NOT the 30-layer trajectory-PCA basis of the earlier decisive subset, since
    this run targets covariate discrimination, not axis-biology mapping).
  - Cost control: ESM-2 forward pass ONCE per unique isoform sequence (21,886),
    per-residue representations pooled per pair-occurrence then discarded
    immediately (no persistent per-residue cache -- would be too large).

Output: reports/severity_pairs/muscle_severity_pairs_region.tsv -- the scored
pair table (all original columns) restricted to pairs where region-pooling was
possible after truncation, with new columns:
  severity_score_region, severity_score_scram, size_long_region, size_short_region
(interval lengths, for downstream size-quintile-matched re-analysis).
"""
import os, sys, time, importlib.util
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import torch

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
MODEL = ROOT / 'hMuscle/model'
OUT_DIR = ROOT / 'reports/severity_pairs'
IN_TSV = OUT_DIR / 'muscle_severity_pairs_scored.tsv'
OUT_TSV = OUT_DIR / 'muscle_severity_pairs_region.tsv'
EMB_CACHE = OUT_DIR / 'muscle_region_embeddings.npz'
MAXLEN = 1022
N_LAYERS_KEEP = (15, 30)
GPU = int(os.environ.get('EMB_GPU', '0'))
MEM_FRAC = float(os.environ.get('EMB_MEM_FRAC', '0.35'))
BATCH = 8
N_FOLDS = 5
SEED = 42
rng = np.random.default_rng(707)

spec = importlib.util.spec_from_file_location('bsp', MODEL / 'build_severity_pairs.py')
bsp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsp)


def opcode_intervals(long_s, short_s):
    """Both-side coordinate intervals -- identical to exp_domdiff_region_pool.opcode_intervals."""
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
    df = pd.read_csv(IN_TSV, sep='\t')
    df = df[df['tissue'] == 'muscle'].reset_index(drop=True)
    iso_m = np.load(MODEL / 'my_isoform_list_fixed.npy', allow_pickle=True)
    iso_m = [s.decode() if isinstance(s, bytes) else str(s) for s in iso_m]
    seqs = bsp.parse_pep_sequences(ROOT / 'hMuscle/data/top30k_isoforms.pep')
    print(f"[load] {len(df)} muscle pairs, {len(iso_m)} isoform ids, {len(seqs)} sequences", flush=True)
    return df, iso_m, seqs


NTERM_FALLBACK = 60


def build_occurrences(df, iso_m, seqs):
    """For each pair row, truncate both sequences to MAXLEN, compute both-side
    changed intervals on the TRUNCATED sequences (matches prior scripts' order
    of operations). A side with an EMPTY interval list (e.g. a pure deletion --
    the short isoform literally has no extra/substituted residues to point to)
    is NOT dropped: it falls back to the first NTERM_FALLBACK residues, exactly
    matching exp_traj_region_projection.compute_traj's
    `if not cidx: cidx = list(range(min(60, Ln)))`. Dropping instead of
    falling back was an earlier bug in this script -- it silently removed 42%
    of pairs (pure-deletion edits), which would have biased the muscle sample
    AWAY FROM the very edit-diversity this pilot is meant to capture (muscle's
    edits are more fragmented than brain's, [[approach-severity-covariate-surrogate-chain]]).
    Only genuinely degenerate pairs (identical after MAXLEN truncation) are
    dropped. Returns: valid row indices, per-isoform occurrence list
    (isoform_id -> seq, plus list of (row_idx, side, ivs))."""
    keep_rows = []
    iso_seq = {}
    occ = {}  # isoform_id -> list of (row_idx, side, ivs)
    ivs_size = {}  # (row_idx, side) -> total changed-residue count on that side (0 = fallback used)
    n_missing_seq = 0
    n_degenerate_after_trunc = 0
    n_fallback_long = 0
    n_fallback_short = 0
    for row_idx, r in df.iterrows():
        long_id = iso_m[int(r['long_idx'])]
        short_id = iso_m[int(r['short_idx'])]
        if long_id not in seqs or short_id not in seqs:
            n_missing_seq += 1
            continue
        long_s = seqs[long_id][:MAXLEN]
        short_s = seqs[short_id][:MAXLEN]
        if long_s == short_s:
            n_degenerate_after_trunc += 1
            continue
        livs, sivs = opcode_intervals(long_s, short_s)
        if not livs:
            n_fallback_long += 1
        if not sivs:
            n_fallback_short += 1
        keep_rows.append(row_idx)
        iso_seq[long_id] = long_s
        iso_seq[short_id] = short_s
        occ.setdefault(long_id, []).append((row_idx, 'long', livs))
        occ.setdefault(short_id, []).append((row_idx, 'short', sivs))
        ivs_size[(row_idx, 'long')] = sum(v - u for u, v in livs) if livs else min(NTERM_FALLBACK, len(long_s))
        ivs_size[(row_idx, 'short')] = sum(v - u for u, v in sivs) if sivs else min(NTERM_FALLBACK, len(short_s))
    print(f"[occ] valid pairs: {len(keep_rows)} / {len(df)} "
          f"(missing-seq: {n_missing_seq}, degenerate-after-trunc: {n_degenerate_after_trunc}, "
          f"fallback-used long-side: {n_fallback_long}, short-side: {n_fallback_short})", flush=True)
    print(f"[occ] unique isoform sequences needing ESM-2 forward pass: {len(iso_seq)}", flush=True)
    return keep_rows, iso_seq, occ, ivs_size


def run_esm2(iso_seq, occ):
    import esm
    dev = f'cuda:{GPU}'
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC, device=GPU)
    model, alph = esm.pretrained.esm2_t30_150M_UR50D()
    bc = alph.get_batch_converter()
    model.eval().to(dev)
    ids = sorted(iso_seq.keys(), key=lambda i: len(iso_seq[i]))
    n = len(ids)
    region15 = {}; region30 = {}; scram15 = {}; scram30 = {}
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
                a15 = r15[bi, 1:Ln + 1, :]  # (Ln,640)
                a30 = r30[bi, 1:Ln + 1, :]
                for (row_idx, side, ivs) in occ[iso_id]:
                    cidx = [k for (u, v) in ivs for k in range(u, v) if k < Ln]
                    if not cidx:
                        cidx = list(range(min(NTERM_FALLBACK, Ln)))
                    L0 = len(cidx)
                    if L0 >= Ln:
                        sidx = list(range(Ln))
                    else:
                        start = int(rng.integers(0, Ln - L0 + 1))
                        sidx = list(range(start, start + L0))
                    key = (row_idx, side)
                    region15[key] = a15[cidx].mean(0); region30[key] = a30[cidx].mean(0)
                    scram15[key] = a15[sidx].mean(0); scram30[key] = a30[sidx].mean(0)
            if (s // BATCH) % 50 == 0:
                print(f"  [esm2] {s + len(ch)}/{n} isoforms, {time.time() - t0:.0f}s elapsed", flush=True)
    print(f"[esm2] done, {n} isoforms in {time.time() - t0:.0f}s", flush=True)
    return region15, region30, scram15, scram30


def gene_disjoint_folds(genes, n_folds=N_FOLDS, seed=SEED):
    uniq = np.array(sorted(set(genes)))
    r = np.random.default_rng(seed)
    r.shuffle(uniq)
    fold_of_gene = {g: i % n_folds for i, g in enumerate(uniq)}
    return np.array([fold_of_gene[g] for g in genes])


def mean_pool_fold_directions(df_full):
    """Retrain the mean-pool coherence direction per fold IDENTICALLY to
    build_severity_score.py, on the FULL 15,885-pair population (NOT the
    region-pool-valid subset) -- C2: this is the FIXED direction region/scram
    get projected onto, and must match the original severity_score column
    exactly (fold assignment is shuffle-order-dependent on the gene set, so
    training on a subset would silently produce a DIFFERENT fold/direction and
    break reproducibility of the mean-pool column)."""
    L15 = np.load(ROOT / 'hMuscle/data/esm2_layer_15_t30_150M.npy').astype(np.float32)
    L30 = np.load(ROOT / 'hMuscle/data/esm2_layer_30_t30_150M.npy').astype(np.float32)
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
        direction = direction / norm if norm > 0 else direction
        directions[k] = direction
        mean_scores[fold == k] = D[fold == k] @ direction
    return fold, directions, mean_scores


def score_with_fixed_directions(D, fold, directions):
    scores = np.zeros(len(D))
    for k, direction in directions.items():
        mask = fold == k
        scores[mask] = D[mask] @ direction
    return scores


def main():
    df, iso_m, seqs = load_pairs_and_seqs()

    fold_full, directions, mean_scores_full = mean_pool_fold_directions(df)
    assert np.allclose(mean_scores_full, df['severity_score'].to_numpy(), atol=1e-3), \
        "recomputed mean-pool severity_score (full population) does not match stored column"
    print("[check] mean-pool severity_score reproduced exactly on full population -- fold/direction OK", flush=True)

    keep_rows, iso_seq, occ, ivs_size = build_occurrences(df, iso_m, seqs)

    smoke_limit = os.environ.get('EMB_SMOKE_LIMIT')
    if smoke_limit:
        smoke_limit = int(smoke_limit)
        keep_rows = keep_rows[:smoke_limit]
        keep_ids = {iso_m[int(df.loc[r]['long_idx'])] for r in keep_rows} | \
                   {iso_m[int(df.loc[r]['short_idx'])] for r in keep_rows}
        iso_seq = {k: v for k, v in iso_seq.items() if k in keep_ids}
        occ = {k: [o for o in v if o[0] in set(keep_rows)] for k, v in occ.items() if k in keep_ids}
        print(f"[smoke] limited to {len(keep_rows)} pairs / {len(iso_seq)} isoforms for smoke test", flush=True)

    df_valid = df.loc[keep_rows].reset_index(drop=True)
    fold_valid = fold_full[keep_rows]
    row_pos = {orig: i for i, orig in enumerate(keep_rows)}

    region15, region30, scram15, scram30 = run_esm2(iso_seq, occ)

    n = len(df_valid)
    D_region = np.zeros((n, 1280), dtype=np.float32)
    D_scram = np.zeros((n, 1280), dtype=np.float32)
    size_long_region = np.zeros(n, dtype=np.int32)
    size_short_region = np.zeros(n, dtype=np.int32)
    valid_mask = np.zeros(n, dtype=bool)
    for orig_idx, pos in row_pos.items():
        klong = (orig_idx, 'long'); kshort = (orig_idx, 'short')
        if klong not in region15 or kshort not in region15:
            continue
        rl = np.concatenate([region15[klong], region30[klong]])
        rs = np.concatenate([region15[kshort], region30[kshort]])
        sl = np.concatenate([scram15[klong], scram30[klong]])
        ss = np.concatenate([scram15[kshort], scram30[kshort]])
        D_region[pos] = rl - rs
        D_scram[pos] = sl - ss
        size_long_region[pos] = ivs_size[klong]
        size_short_region[pos] = ivs_size[kshort]
        valid_mask[pos] = True

    df_valid = df_valid.loc[valid_mask].reset_index(drop=True)
    fold_valid = fold_valid[valid_mask]
    D_region = D_region[valid_mask]
    D_scram = D_scram[valid_mask]
    size_long_region = size_long_region[valid_mask]
    size_short_region = size_short_region[valid_mask]
    print(f"[score] {len(df_valid)} pairs with complete region+scram embeddings on both sides", flush=True)

    df_valid['severity_score_region'] = score_with_fixed_directions(D_region, fold_valid, directions)
    df_valid['severity_score_scram'] = score_with_fixed_directions(D_scram, fold_valid, directions)
    df_valid['size_long_region'] = size_long_region
    df_valid['size_short_region'] = size_short_region

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(OUT_TSV, sep='\t', index=False)
    np.savez(EMB_CACHE, D_region=D_region, D_scram=D_scram)
    print(f"[done] wrote {OUT_TSV} ({len(df_valid)} pairs) and {EMB_CACHE}", flush=True)
    print(df_valid[['severity_score', 'severity_score_region', 'severity_score_scram']].describe(), flush=True)


if __name__ == '__main__':
    main()
