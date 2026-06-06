"""
update_bisect_prism_scores.py
==============================
Updates each BISECT case in bisect_cases.json with PRISM functional-score
vectors derived from the brain 672-GO-term score matrix.

Purpose
-------
Enriches BISECT case records with isoform-level PRISM predictions so that the
PRISM app can display top GO terms, functional gain/loss, and delta scores for
every CT↔AD isoform pair without re-running the model at query time.

Input files
-----------
prism_app/data/demo/bisect_cases.json          -- BISECT case list (modified in-place)
prism_app/data/demo/brain_full_672_ids.npy     -- isoform IDs (N,)
prism_app/data/demo/brain_full_672_scores.npy  -- PRISM score matrix (N × 672, float32)
prism_app/data/demo/brain_full_672_gene_ids.npy -- gene ID per isoform (N,)
hMuscle/data/brain672_go_terms.json            -- GO term ID list and name map (672 terms)

Output files
------------
prism_app/data/demo/bisect_cases.json  -- same file, with added fields per case:
    prism_ct_top_go, prism_ad_top_go, prism_gain_go, prism_loss_go,
    prism_ct_max_score, prism_ct_max_go, prism_ad_max_score, prism_ad_max_go,
    prism_delta_max, prism_delta_min, prism_match_ct, prism_match_ad

Matching strategy
-----------------
1. Exact transcript-ID match in score matrix (preferred).
2. Gene-level median across all isoforms of that gene (fallback).
3. Zero vector if gene not found (prism_match = 'no_match').

Usage
-----
    python scripts/update_bisect_prism_scores.py

Notes
-----
- Run after regenerating brain_full_672_scores.npy or after adding new BISECT cases.
- Requires: numpy, pathlib (stdlib). No GPU needed.
- TOP_N and DELTA_MIN thresholds can be adjusted at the top of the file.
"""
import json
import numpy as np
from pathlib import Path

DEMO_DIR   = Path(__file__).parents[1] / 'prism_app' / 'data' / 'demo'
REPORT_DIR = Path(__file__).parents[1] / 'hMuscle' / 'data'

BISECT_JSON  = DEMO_DIR / 'bisect_cases.json'
IDS_NPY      = DEMO_DIR / 'brain_full_672_ids.npy'
SCORES_NPY   = DEMO_DIR / 'brain_full_672_scores.npy'
GENE_IDS_NPY = DEMO_DIR / 'brain_full_672_gene_ids.npy'
GO_TERMS_JSON = REPORT_DIR / 'brain672_go_terms.json'

TOP_N      = 5   # top-N GO terms to report per isoform
DELTA_MIN  = 0.05  # min |score delta| to report as GAIN/LOSS


def load_data():
    ids      = np.load(IDS_NPY, allow_pickle=True).astype(str)
    scores   = np.load(SCORES_NPY, allow_pickle=True).astype(np.float32)
    gene_ids = np.load(GENE_IDS_NPY, allow_pickle=True).astype(str)
    with open(GO_TERMS_JSON) as f:
        go_data = json.load(f)
    go_ids   = go_data['go_ids']          # list[str], length 672
    go_name_map = go_data['go_names']     # dict: GO_ID → name
    go_names = [go_name_map.get(gid, gid) for gid in go_ids]  # list[str]
    return ids, scores, gene_ids, go_ids, go_names


def build_lookups(ids, scores, gene_ids):
    id_to_idx  = {iid: i for i, iid in enumerate(ids)}
    gene_to_idxs = {}
    for i, g in enumerate(gene_ids):
        gene_to_idxs.setdefault(g, []).append(i)
    # Pre-compute gene-level medians (fallback)
    gene_median = {}
    for gene, idxs in gene_to_idxs.items():
        gene_median[gene] = np.median(scores[idxs], axis=0)
    return id_to_idx, gene_to_idxs, gene_median


def get_score_vector(transcript_id, gene, id_to_idx, gene_to_idxs, gene_median, scores):
    """Return (score_vector, match_method)."""
    if transcript_id in id_to_idx:
        return scores[id_to_idx[transcript_id]], 'exact'
    if gene in gene_median:
        return gene_median[gene], 'gene_median'
    return None, 'no_match'


def top_go(score_vec, go_ids, go_names, n=TOP_N, threshold=0.0):
    """Return top-N GO terms by score as list of dicts."""
    idxs = np.argsort(score_vec)[::-1][:n]
    return [
        {'go_id': go_ids[i], 'go_name': go_names[i], 'score': round(float(score_vec[i]), 4)}
        for i in idxs if score_vec[i] > threshold
    ]


def gain_loss_go(ct_vec, ad_vec, go_ids, go_names, delta_min=DELTA_MIN, n=TOP_N):
    """Compute GAIN (AD > CT) and LOSS (CT > AD) GO terms."""
    delta = ad_vec - ct_vec
    # GAIN: AD score significantly higher → disease acquires function
    gain_idxs = np.where(delta > delta_min)[0]
    gain_idxs = gain_idxs[np.argsort(delta[gain_idxs])[::-1]][:n]
    gain = [
        {'go_id': go_ids[i], 'go_name': go_names[i],
         'ct_score': round(float(ct_vec[i]), 4),
         'ad_score': round(float(ad_vec[i]), 4),
         'delta': round(float(delta[i]), 4)}
        for i in gain_idxs
    ]
    # LOSS: CT score significantly higher → disease loses function
    loss_idxs = np.where(delta < -delta_min)[0]
    loss_idxs = loss_idxs[np.argsort(delta[loss_idxs])][:n]
    loss = [
        {'go_id': go_ids[i], 'go_name': go_names[i],
         'ct_score': round(float(ct_vec[i]), 4),
         'ad_score': round(float(ad_vec[i]), 4),
         'delta': round(float(delta[i]), 4)}
        for i in loss_idxs
    ]
    return gain, loss


def enrich_case(case, id_to_idx, gene_to_idxs, gene_median, scores, go_ids, go_names):
    gene    = case['gene']
    ct_id   = case['ct_transcript_id']
    ad_id   = case['ad_transcript_id']

    ct_vec, ct_method = get_score_vector(ct_id, gene, id_to_idx, gene_to_idxs, gene_median, scores)
    ad_vec, ad_method = get_score_vector(ad_id, gene, id_to_idx, gene_to_idxs, gene_median, scores)

    if ct_vec is None and ad_vec is None:
        case['prism_match'] = 'no_match'
        return case

    # If one side is missing, use the other as proxy for direction context
    if ct_vec is None:
        ct_vec = gene_median.get(gene, np.zeros(len(go_ids)))
        ct_method = 'gene_median'
    if ad_vec is None:
        ad_vec = gene_median.get(gene, np.zeros(len(go_ids)))
        ad_method = 'gene_median'

    gain, loss = gain_loss_go(ct_vec, ad_vec, go_ids, go_names)

    case['prism_ct_top_go']     = top_go(ct_vec, go_ids, go_names)
    case['prism_ad_top_go']     = top_go(ad_vec, go_ids, go_names)
    case['prism_gain_go']       = gain   # AD > CT (function gained in disease)
    case['prism_loss_go']       = loss   # CT > AD (function lost in disease)
    case['prism_ct_max_score']  = round(float(ct_vec.max()), 4)
    case['prism_ct_max_go']     = go_names[int(ct_vec.argmax())]
    case['prism_ad_max_score']  = round(float(ad_vec.max()), 4)
    case['prism_ad_max_go']     = go_names[int(ad_vec.argmax())]
    case['prism_delta_max']     = round(float((ad_vec - ct_vec).max()), 4)
    case['prism_delta_min']     = round(float((ad_vec - ct_vec).min()), 4)
    case['prism_match_ct']      = ct_method
    case['prism_match_ad']      = ad_method
    return case


def main():
    print("Loading brain_672 data...")
    ids, scores, gene_ids, go_ids, go_names = load_data()
    print(f"  isoforms: {len(ids)}, GO terms: {len(go_ids)}")

    id_to_idx, gene_to_idxs, gene_median = build_lookups(ids, scores, gene_ids)

    with open(BISECT_JSON) as f:
        cases = json.load(f)
    print(f"BISECT cases: {len(cases)}")

    exact_ct = exact_ad = fallback_ct = fallback_ad = 0
    for case in cases:
        ct_id = case['ct_transcript_id']
        ad_id = case['ad_transcript_id']
        ct_m  = 'exact' if ct_id in id_to_idx else 'gene_median'
        ad_m  = 'exact' if ad_id in id_to_idx else 'gene_median'
        if ct_m == 'exact': exact_ct += 1
        else:               fallback_ct += 1
        if ad_m == 'exact': exact_ad += 1
        else:               fallback_ad += 1

        enrich_case(case, id_to_idx, gene_to_idxs, gene_median, scores, go_ids, go_names)

    print(f"CT match  — exact: {exact_ct}, gene_median: {fallback_ct}")
    print(f"AD match  — exact: {exact_ad}, gene_median: {fallback_ad}")

    with open(BISECT_JSON, 'w') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Saved → {BISECT_JSON}")

    # Spot-check
    sample = cases[0]
    print(f"\nSpot-check [{sample['gene']}]:")
    print(f"  CT top-{TOP_N} GO: {[x['go_name'] for x in sample.get('prism_ct_top_go', [])]}")
    print(f"  AD top-{TOP_N} GO: {[x['go_name'] for x in sample.get('prism_ad_top_go', [])]}")
    print(f"  GAIN GO: {[x['go_name'] for x in sample.get('prism_gain_go', [])]}")
    print(f"  LOSS GO: {[x['go_name'] for x in sample.get('prism_loss_go', [])]}")


if __name__ == '__main__':
    main()
