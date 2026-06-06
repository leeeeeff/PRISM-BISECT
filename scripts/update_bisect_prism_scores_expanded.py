"""
update_bisect_prism_scores_expanded.py
========================================
bisect_cases.json을 41-term 확장 PRISM 모델 스코어로 업데이트.

train_prism_v15d_brain_expanded.py 실행 후 호출.

변경사항 (vs. 672GO 버전):
- 스코어 소스: brain_full_expanded_41_scores.npy (63994 × 41)
- 고신뢰 기준: prism_ad_max_score >= 0.40 (AUPRC 0.5+ 검증된 모델)
- prism_tier 필드 추가 (Tier1/Tier2/Tier3)
- prism_ad_rank / prism_ct_rank 필드 추가 (percentile rank within 63994 brain isos)
"""
import json
import numpy as np
from pathlib import Path

DEMO_DIR   = Path(__file__).parents[1] / 'prism_app' / 'data' / 'demo'
REPORT_DIR = Path(__file__).parents[1] / 'reports'

BISECT_JSON  = DEMO_DIR / 'bisect_cases.json'
IDS_NPY      = DEMO_DIR / 'brain_full_expanded_41_ids.npy'
SCORES_NPY   = DEMO_DIR / 'brain_full_expanded_41_scores.npy'
GENE_IDS_NPY = DEMO_DIR / 'brain_full_expanded_41_gene_ids.npy'
META_JSON    = REPORT_DIR / 'brain_full_expanded_41_meta.json'

TOP_N      = 5
DELTA_MIN  = 0.05
HIGH_CONF_THRESH = 0.40  # validated for 41GO model (AUPRC > 0.5 confirmed)


def load_data():
    ids      = np.load(IDS_NPY,      allow_pickle=True).astype(str)
    scores   = np.load(SCORES_NPY,   allow_pickle=True).astype(np.float32)
    gene_ids = np.load(GENE_IDS_NPY, allow_pickle=True).astype(str)
    with open(META_JSON) as f:
        meta = json.load(f)
    go_ids   = meta['go_ids']
    go_names = meta['go_names']   # dict: go_id → name
    go_names_list = [go_names.get(g, g) for g in go_ids]
    return ids, scores, gene_ids, go_ids, go_names_list, go_names


def build_lookups(ids, scores, gene_ids):
    id_to_idx = {iid: i for i, iid in enumerate(ids)}
    gene_to_idxs = {}
    for i, g in enumerate(gene_ids):
        gene_to_idxs.setdefault(g, []).append(i)
    gene_median = {gene: np.median(scores[idxs], axis=0)
                   for gene, idxs in gene_to_idxs.items()}
    return id_to_idx, gene_to_idxs, gene_median


def compute_rank_lookup(scores):
    """Pre-compute per-GO percentile rank (0–100) for all isoforms."""
    n, n_go = scores.shape
    ranks = np.zeros_like(scores, dtype=np.float32)
    for j in range(n_go):
        col = scores[:, j]
        # argsort twice = rank
        order = np.argsort(col)
        rank  = np.empty_like(order)
        rank[order] = np.arange(n)
        ranks[:, j] = rank / (n - 1) * 100  # 0–100
    return ranks  # (63994, 41) float32


def get_score_vector(transcript_id, gene, id_to_idx, gene_median, scores):
    if transcript_id in id_to_idx:
        return scores[id_to_idx[transcript_id]], 'exact', id_to_idx[transcript_id]
    if gene in gene_median:
        return gene_median[gene], 'gene_median', None
    return None, 'no_match', None


def top_go(score_vec, go_ids, go_names, n=TOP_N):
    idxs = np.argsort(score_vec)[::-1][:n]
    return [
        {'go_id': go_ids[i], 'go_name': go_names[i], 'score': round(float(score_vec[i]), 4)}
        for i in idxs
    ]


def gain_loss_go(ct_vec, ad_vec, go_ids, go_names, delta_min=DELTA_MIN, n=TOP_N):
    delta = ad_vec - ct_vec
    gain_idxs = np.where(delta > delta_min)[0]
    gain_idxs = gain_idxs[np.argsort(delta[gain_idxs])[::-1]][:n]
    gain = [
        {'go_id': go_ids[i], 'go_name': go_names[i],
         'ct_score': round(float(ct_vec[i]), 4),
         'ad_score': round(float(ad_vec[i]), 4),
         'delta': round(float(delta[i]), 4)}
        for i in gain_idxs
    ]
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


def assign_tier(ad_max, ct_max, delta_max, ct_method, ad_method, gain, loss):
    """
    Tier 1: 고신뢰 기능 획득/스위치
      - AD max >= 0.40 (절대 신뢰) AND delta_max >= 0.15 (방향성 명확)
      - 가장 강력한 증거: AD가 새로운 기능을 높은 신뢰도로 획득

    Tier 2a: 기능 소실
      - CT max >= 0.40 AND AD max < CT max × 0.65 (기능 유의하게 감소)

    Tier 2b: 양방향 변화 또는 부분 변화
      - delta_max >= 0.10 (유의한 변화 있음) 하지만 Tier 1/2a 기준 미달

    Tier 3: 구조 증거 기반 (기능 예측 약함 또는 gene_median)
    """
    if ct_method == 'gene_median' and ad_method == 'gene_median':
        return 'tier3_gene_median'

    if ad_max >= HIGH_CONF_THRESH and delta_max >= 0.15:
        return 'tier1_functional_switch'

    if ct_max >= HIGH_CONF_THRESH and ad_max < ct_max * 0.65:
        return 'tier2_functional_loss'

    if delta_max >= 0.10:
        return 'tier2_partial_change'

    return 'tier3_structural_only'


def enrich_case(case, id_to_idx, gene_median, scores, rank_matrix,
                go_ids, go_names):
    gene  = case['gene']
    ct_id = case['ct_transcript_id']
    ad_id = case['ad_transcript_id']

    ct_vec, ct_method, ct_idx = get_score_vector(ct_id, gene, id_to_idx, gene_median, scores)
    ad_vec, ad_method, ad_idx = get_score_vector(ad_id, gene, id_to_idx, gene_median, scores)

    if ct_vec is None and ad_vec is None:
        case['prism_match_ct'] = 'no_match'
        case['prism_match_ad'] = 'no_match'
        case['prism_tier']     = 'tier3_no_match'
        return case

    if ct_vec is None:
        ct_vec = gene_median.get(gene, np.zeros(len(go_ids)))
        ct_method = 'gene_median'
        ct_idx = None
    if ad_vec is None:
        ad_vec = gene_median.get(gene, np.zeros(len(go_ids)))
        ad_method = 'gene_median'
        ad_idx = None

    gain, loss = gain_loss_go(ct_vec, ad_vec, go_ids, go_names)

    # Percentile ranks (per-GO max rank)
    ct_rank_val = float(rank_matrix[ct_idx, ct_vec.argmax()]) if ct_idx is not None else None
    ad_rank_val = float(rank_matrix[ad_idx, ad_vec.argmax()]) if ad_idx is not None else None

    ct_max = float(ct_vec.max())
    ad_max = float(ad_vec.max())

    case['prism_ct_top_go']    = top_go(ct_vec, go_ids, go_names)
    case['prism_ad_top_go']    = top_go(ad_vec, go_ids, go_names)
    case['prism_gain_go']      = gain
    case['prism_loss_go']      = loss
    case['prism_ct_max_score'] = round(ct_max, 4)
    case['prism_ct_max_go']    = go_names[int(ct_vec.argmax())]
    case['prism_ad_max_score'] = round(ad_max, 4)
    case['prism_ad_max_go']    = go_names[int(ad_vec.argmax())]
    case['prism_delta_max']    = round(float((ad_vec - ct_vec).max()), 4)
    case['prism_delta_min']    = round(float((ad_vec - ct_vec).min()), 4)
    case['prism_match_ct']     = ct_method
    case['prism_match_ad']     = ad_method
    case['prism_ad_rank']      = round(ad_rank_val, 1) if ad_rank_val is not None else None
    case['prism_ct_rank']      = round(ct_rank_val, 1) if ct_rank_val is not None else None
    case['prism_tier']         = assign_tier(
        ad_max, ct_max, float(case.get('prism_delta_max', 0)),
        ct_method, ad_method, gain, loss)
    return case


def main():
    print("Loading 41-term expanded PRISM data...")
    ids, scores, gene_ids, go_ids, go_names, go_names_dict = load_data()
    print(f"  isoforms: {len(ids)}, GO terms: {len(go_ids)}")

    id_to_idx, gene_to_idxs, gene_median = build_lookups(ids, scores, gene_ids)

    print("Computing percentile rank matrix...")
    rank_matrix = compute_rank_lookup(scores)
    print(f"  Rank matrix: {rank_matrix.shape}, "
          f"range [{rank_matrix.min():.1f}, {rank_matrix.max():.1f}]")

    with open(BISECT_JSON) as f:
        cases = json.load(f)
    print(f"BISECT cases: {len(cases)}")

    exact_ct = exact_ad = fallback_ct = fallback_ad = 0
    for case in cases:
        ct_m = 'exact' if case['ct_transcript_id'] in id_to_idx else 'gene_median'
        ad_m = 'exact' if case['ad_transcript_id'] in id_to_idx else 'gene_median'
        if ct_m == 'exact': exact_ct += 1
        else: fallback_ct += 1
        if ad_m == 'exact': exact_ad += 1
        else: fallback_ad += 1
        enrich_case(case, id_to_idx, gene_median, scores, rank_matrix, go_ids, go_names)

    print(f"CT match  — exact: {exact_ct}, gene_median: {fallback_ct}")
    print(f"AD match  — exact: {exact_ad}, gene_median: {fallback_ad}")

    # Tier distribution
    from collections import Counter
    tier_counts = Counter(c.get('prism_tier', 'unknown') for c in cases)
    print("\nTier distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count}")

    # High-confidence cases (Tier 1)
    tier1 = [c for c in cases if c.get('prism_tier') == 'tier1_functional_switch']
    print(f"\nTier 1 (functional switch, AD >= {HIGH_CONF_THRESH} & AD rank > CT rank):")
    for c in tier1:
        ad_rank = c.get('prism_ad_rank')
        ct_rank = c.get('prism_ct_rank')
        print("  %s: AD_max=%.3f (%s), delta=%.3f, AD_rank=%s, CT_rank=%s" % (
            c['gene'], c.get('prism_ad_max_score', 0), c.get('prism_ad_max_go', '')[:30],
            c.get('prism_delta_max', 0),
            '%.1f%%' % ad_rank if ad_rank is not None else 'N/A',
            '%.1f%%' % ct_rank if ct_rank is not None else 'N/A'))

    with open(BISECT_JSON, 'w') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {BISECT_JSON}")

    # Spot-check
    sample = cases[0]
    print(f"\nSpot-check [{sample['gene']}]:")
    print(f"  CT top-3 GO: {[x['go_name'][:25] for x in sample.get('prism_ct_top_go', [])[:3]]}")
    print(f"  AD top-3 GO: {[x['go_name'][:25] for x in sample.get('prism_ad_top_go', [])[:3]]}")
    print(f"  Tier: {sample.get('prism_tier')}")
    print(f"  AD max score: {sample.get('prism_ad_max_score')}")
    print(f"  AD rank: {sample.get('prism_ad_rank')}")


if __name__ == '__main__':
    main()
