"""
b_idr_boundary_pilot.py

B5 label candidate #5 (after GO / is_alt_functional / ELM-per-class / ELM-pooled-binary
all failed — see reports/model_interpretability_map/FEATURE_CASCADE_INVENTORY.md §5).

Cheap-check pilot (same discipline as prior B5 attempts, run BEFORE committing to a full
editcore-vs-pooled supervision experiment): does an order<->disorder transition at the edit
region (metapredict, context-aware BRNN predictor, NOT the existing TOP-IDP composition-lookup
disorder_frac) give (a) enough within-manifest sample size and (b) information beyond AA
composition?

Unlike the ELM pooled-binary label (devils-advocate FATAL: label defined by edit-region
membership itself = circular with editcore's own definition), this label is computed from
sequence *context* around each residue via an independently-trained disorder predictor —
no dependency on PRISM's architecture or on any external curated-annotation database (so no
gene-inherited / selection-bias risk either).
"""
import sys
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = Path('/home/welcome1/sw1686/DIFFUSE')
sys.path.insert(0, str(ROOT / 'hMuscle/model'))
import build_severity_pairs as bsp  # noqa: E402
import metapredict as meta  # noqa: E402

MANIFEST = ROOT / 'reports/model_interpretability_map/b_manifest_pairs.tsv'
OUT_DIR = ROOT / 'reports/model_interpretability_map/assets'


def load_sequences():
    iso_ids = np.load(ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_ids.npy', allow_pickle=True)
    iso_ids = [str(x) for x in iso_ids]
    seqs = bsp.parse_fasta_sequences(ROOT / 'reports/truebrain_rerun_20260714/data/brain_full_proteins.fa')
    return iso_ids, seqs


def score_disorder(manifest, iso_ids, seqs):
    uniq_ids = set()
    for _, r in manifest.iterrows():
        uniq_ids.add(iso_ids[int(r['long_idx'])])
        uniq_ids.add(iso_ids[int(r['short_idx'])])
    cache = {}
    for sid in uniq_ids:
        cache[sid] = meta.predict_disorder(seqs[sid])
    return cache


def edit_region_transition(manifest, iso_ids, seqs, disorder_cache):
    TOPIDP = bsp.TOPIDP
    rows = []
    for _, r in manifest.iterrows():
        lid, sid = iso_ids[int(r['long_idx'])], iso_ids[int(r['short_idx'])]
        long_s, short_s = seqs[lid], seqs[sid]
        dl, ds = disorder_cache[lid], disorder_cache[sid]

        ops = SequenceMatcher(None, long_s, short_s, autojunk=False).get_opcodes()
        long_res, short_res, long_ti, short_ti = [], [], [], []
        for tag, i1, i2, j1, j2 in ops:
            if tag == 'equal':
                continue
            if i2 > i1:
                long_res.extend(dl[i1:i2])
                long_ti.extend(TOPIDP.get(long_s[k], 0.0) for k in range(i1, i2))
            if j2 > j1:
                short_res.extend(ds[j1:j2])
                short_ti.extend(TOPIDP.get(short_s[k], 0.0) for k in range(j1, j2))

        if not long_res and not short_res:
            continue
        rows.append({
            'cls': r['cls'], 'gene': r['gene'],
            'meta_long': float(np.mean(long_res)) if long_res else None,
            'meta_short': float(np.mean(short_res)) if short_res else None,
            'topidp_long': float(np.mean(long_ti)) if long_ti else None,
            'topidp_short': float(np.mean(short_ti)) if short_ti else None,
        })
    return pd.DataFrame(rows)


def main():
    manifest = pd.read_csv(MANIFEST, sep='\t')
    iso_ids, seqs = load_sequences()
    disorder_cache = score_disorder(manifest, iso_ids, seqs)
    df = edit_region_transition(manifest, iso_ids, seqs, disorder_cache)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'idr_boundary_pilot_raw.tsv', sep='\t', index=False)

    both = df.dropna(subset=['meta_long', 'meta_short']).copy()
    both['meta_delta'] = both['meta_long'] - both['meta_short']
    both['topidp_delta'] = both['topidp_long'] - both['topidp_short']
    both['boundary_cross'] = ((both['meta_long'] > 0.5).astype(int)
                               != (both['meta_short'] > 0.5).astype(int)).astype(int)
    both.to_csv(OUT_DIR / 'idr_boundary_pilot_both.tsv', sep='\t', index=False)

    print(f'n pairs (>=1 side has edit-region residues): {len(df)}')
    print(f'n pairs (BOTH sides have residues, substitution-type edits): {len(both)}')
    print('\nboundary_cross label balance:')
    print(both['boundary_cross'].value_counts())
    print('\ncorrelation meta_delta vs topidp_delta (novelty-beyond-composition check):')
    print(both[['meta_delta', 'topidp_delta']].corr())


if __name__ == '__main__':
    main()
