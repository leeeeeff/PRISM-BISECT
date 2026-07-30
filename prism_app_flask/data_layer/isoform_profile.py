"""개별 유전자/아이소폼 프로파일 조립 (flagship 데이터 엔진).

precompute 인덱스 + mmap 원본에서 패널 A~D 데이터를 만든다:
  A. GO 예측 점수 top-k          scores[row]
  C. 8축 위치 + percentile + 축별 30-layer 궤적    axis_pos / axis_pos_pct / Z[row]
  D. trajectory flow: 레이어별 정보량 곡선 + peak + population 참조밴드   traj_mag[row]
  (B. covariate 는 fixed-pop(muscle)에만 있어 brain universe 에선 secondary — 후속)

모든 함수는 순수(부작용 없음) → Flask blueprint 에서 그대로 JSON 직렬화.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from prism_app_flask import config
from prism_app_flask.data_layer import loaders

# 각 축의 plain-language 의미 (high 방향이 무엇을 뜻하나) — R4.
# 근거: severity-covariate 특성화 + axes_vs_biology feature 상관. axis3/5/0 이 tissue-안정.
AXIS_MEANING = {
    0: 'β-sheet / transmembrane character (gene-level component)',
    1: 'LRR / Ig repeat character (weak axis)',
    2: 'Pro-turn order character (weak axis)',
    3: 'domain content — higher = richer / more complete Pfam domains',
    4: 'helix-charge character (weak axis)',
    5: 'length — higher = longer isoform',
    6: 'KRAB-ZNF / inverse-domain character (weak axis)',
    7: 'acidic-helical character (weak axis)',
}


def _axis_interpretation(axis: int, pct: float) -> str:
    band = 'upper population' if pct >= 66 else ('lower population' if pct <= 33 else 'mid population')
    meaning = AXIS_MEANING.get(axis, '')
    return f'{band} (p{pct:.0f}) · high = {meaning}'


def _fingerprint_summary(axes: list[dict]) -> str:
    """8-axis radar 해석 문단 (I4 R4) — 기존 hi/lo 경계(66/33, _axis_interpretation과 동일)를 그대로
    재사용해 'extreme'/'notable' 판정: extreme = |percentile-50|>=34 (p>=84 또는 p<=16),
    notable = 16<=|percentile-50|<34 (기존 hi/lo 경계 p>=66/p<=33과 정확히 겹침, 그중 extreme
    문턱을 못 넘은 나머지). 새 임계값을 즉흥적으로 도입하지 않고 코드베이스 관례를 통일한다.
    가장 극단적인 축부터 최대 3개만 서술(전부 나열하면 8줄 요약이 8줄 원본과 다를 바 없어짐)."""
    ranked = sorted(axes, key=lambda a: abs(a['percentile'] - 50), reverse=True)
    extreme = [a for a in ranked if abs(a['percentile'] - 50) >= 34][:3]
    notable = [a for a in ranked if 16 <= abs(a['percentile'] - 50) < 34]
    if not extreme and not notable:
        return 'No axis is far from the population median — this isoform sits close to typical on all 8 axes.'
    parts = []
    if extreme:
        # a['label'] already reads "axisN · description" (build_isoform_index.AXIS_LABEL) — no
        # need to prefix it with "aN" again.
        items = '; '.join(f"{a['label']} ({_axis_interpretation(a['axis'], a['percentile'])})"
                           for a in extreme)
        parts.append(f"Extreme on {len(extreme)} of 8 axes: {items}.")
    if notable:
        items = ', '.join(f"{a['label']} (p{a['percentile']:.0f})" for a in notable)
        parts.append(f"Notable (not extreme): {items}.")
    return ' '.join(parts)


@lru_cache(maxsize=4)
def _population_refs(universe: str) -> dict:
    """패널 참조용 population 통계(1회 계산, 캐시)."""
    idx = loaders.load_index(universe)
    tm = idx['traj_mag']                    # (N, 30)
    return {
        'traj_median': np.median(tm, axis=0).tolist(),
        'traj_q25': np.percentile(tm, 25, axis=0).tolist(),
        'traj_q75': np.percentile(tm, 75, axis=0).tolist(),
        'axis_mean': idx['axis_pos'].mean(axis=0).tolist(),
        'axis_std': idx['axis_pos'].std(axis=0).tolist(),
    }


def resolve_query(q: str, universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """유전자 심볼 또는 아이소폼 id → 매칭 종류 판정."""
    idx = loaders.load_index(universe)
    q = q.strip()
    if q in idx['gene_to_rows']:
        return {'kind': 'gene', 'key': q}
    if q in idx['id_to_row']:
        return {'kind': 'isoform', 'key': q}
    # 대소문자 무시 유전자 매칭
    up = q.upper()
    for g in idx['gene_to_rows']:
        if g.upper() == up:
            return {'kind': 'gene', 'key': g}
    return {'kind': 'none', 'key': q}


def go_cohort_trajectories(go_id: str, exclude: str = '', n: int = 60,
                           universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """선택 GO 로 예측된 top-N 아이소폼의 PC1-3 3D 궤적 (3D cohort overlay, I6).

    현재 분석 아이소폼(exclude)은 제외 → 클라이언트가 별도 색으로 겹친다.
    """
    idx = loaders.load_index(universe)
    go_ids = idx['meta']['go_ids']
    if go_id not in go_ids:
        return {'error': f'{go_id} not in panel'}
    gi = go_ids.index(go_id)
    scores = np.asarray(idx['scores'][:, gi], dtype=np.float32)
    ex_row = idx['id_to_row'].get(exclude)
    top = np.argsort(scores)[::-1]
    cohort = []
    for r in top:
        if len(cohort) >= n:
            break
        if r == ex_row or scores[r] <= 0.3:
            continue
        pc = np.asarray(idx['Z'][r][:, :3], dtype=np.float32)   # (30,3) PC1-3
        cohort.append({'iso': str(idx['isoform_ids'][r]), 'score': round(float(scores[r]), 3),
                       'traj': [[round(float(x), 2) for x in row] for row in pc]})
    return {
        'go_id': go_id, 'go_name': idx['meta']['go_names'].get(go_id, go_id),
        'n_above_thr': int((scores > 0.3).sum()), 'cohort': cohort,
    }


def go_fisher_curve(go_id: str, universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """선택 GO term 의 per-layer Fisher discriminability 곡선."""
    import numpy as np
    glf = loaders.go_layer_fisher(universe)
    idx = loaders.load_index(universe)
    if glf is None or go_id not in glf['go_row']:
        return {'error': f'no fisher for {go_id}'}
    row = glf['mat'][glf['go_row'][go_id]]
    if np.all(np.isnan(row)):
        return {'error': f'fisher undefined for {go_id}'}
    scores = [None if np.isnan(v) else round(float(v), 4) for v in row]
    valid = [v for v in scores if v is not None]
    peak = int(np.nanargmax(row)) + 1 if valid else None
    return {
        'go_id': go_id,
        'go_name': idx['meta']['go_names'].get(go_id, go_id),
        'scores': scores,
        'peak_layer': peak,
        'label': 'per-layer Fisher discriminability (AUROC, gene-level labels)',
    }


COV_LABELS = {
    'disorder_frac': 'disorder fraction (metapredict)',
    'helix_nterm': 'N-term α-helix propensity (Chou-Fasman)',
    'sheet_nterm': 'N-term β-sheet propensity (Chou-Fasman)',
    'hydro_nterm': 'N-term hydropathy (Kyte-Doolittle)',
    'charge_nterm': 'N-term net charge (D/E −, K/R +, H ±)',
    'aromatic_nterm': 'N-term aromatic fraction (F/W/Y)',
}

# short plain-language interpretation of each covariate (E5 — Panel B legibility)
COV_META = {
    'disorder_frac': {'unit': 'frac', 'lo': 'structured', 'hi': 'intrinsically disordered',
                      'note': 'fraction of residues predicted disordered across the whole protein'},
    'helix_nterm':  {'unit': 'P_α', 'lo': 'non-helical N-term', 'hi': 'helix-prone N-term',
                     'note': 'first 60 aa · Chou-Fasman α-helix propensity'},
    'sheet_nterm':  {'unit': 'P_β', 'lo': 'non-sheet N-term', 'hi': 'sheet-prone N-term',
                     'note': 'first 60 aa · Chou-Fasman β-sheet propensity'},
    'hydro_nterm':  {'unit': 'KD', 'lo': 'hydrophilic N-term', 'hi': 'hydrophobic N-term',
                     'note': 'first 60 aa · Kyte-Doolittle mean hydropathy'},
    'charge_nterm': {'unit': 'net q', 'lo': 'acidic N-term', 'hi': 'basic N-term',
                     'note': 'first 60 aa · net charge (targeting/localization signal)'},
    'aromatic_nterm': {'unit': 'frac', 'lo': 'few aromatics', 'hi': 'aromatic-rich N-term',
                       'note': 'first 60 aa · F/W/Y fraction (π-interactions)'},
}


def _covariate_block(row: int, universe: str) -> dict:
    """실제 서열 covariate (없으면 available=False)."""
    cov = loaders.brain_covariates(universe)
    if cov is None:
        return {'available': False, 'reason': 'covariate index not built'}
    import numpy as np
    if not bool(cov['mask'][row]):
        return {'available': False, 'reason': 'no protein sequence (non-coding / novel transcript)'}
    items = []
    for j, nm in enumerate(cov['names']):
        m = COV_META.get(nm, {})
        items.append({'name': nm, 'label': COV_LABELS.get(nm, nm),
                      'value': round(float(cov['values'][row, j]), 3),
                      'percentile': round(float(cov['pct'][row, j]), 1),
                      'unit': m.get('unit', ''), 'lo': m.get('lo', 'low'),
                      'hi': m.get('hi', 'high'), 'note': m.get('note', '')})
    return {'available': True, 'covariates': items}


def _cov_pct_row(row: int, universe: str):
    cov = loaders.brain_covariates(universe)
    if cov is None or not bool(cov['mask'][row]):
        return None
    return [round(float(v), 1) for v in cov['pct'][row]]


def _top_go(row: int, idx: dict, k: int = 8) -> list[dict]:
    s = np.asarray(idx['scores'][row], dtype=np.float32)
    go_ids = idx['meta']['go_ids']
    go_names = idx['meta']['go_names']
    top = s.argsort()[::-1][:k]
    return [{'go_id': go_ids[t], 'name': go_names.get(go_ids[t], go_ids[t]),
             'score': round(float(s[t]), 4)} for t in top]


def isoform_profile(iso_id: str, universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """단일 아이소폼의 패널 A/C/D 데이터."""
    idx = loaders.load_index(universe)
    row = idx['id_to_row'].get(iso_id)
    if row is None:
        return {'error': f'isoform {iso_id!r} not found'}
    refs = _population_refs(universe)
    meta = idx['meta']

    Zrow = np.asarray(idx['Z'][row], dtype=np.float32)          # (30, 8)
    axis_pos = idx['axis_pos'][row].tolist()
    axis_pct = idx['axis_pos_pct'][row].tolist()
    traj = idx['traj_mag'][row].tolist()
    peak = int(idx['peak_layer'][row])

    # 레이어별 population 정규화(원본 normalize_layers): (Z - mean_L) / std_L
    lmean = idx['layer_axis_mean']    # (30, 8)
    lstd = idx['layer_axis_std']      # (30, 8)
    Znorm = (Zrow - lmean) / lstd     # (30, 8)

    axes = []
    for a in range(meta['n_axes']):
        axes.append({
            'axis': a,
            'label': meta['axis_labels'][str(a)],
            'position': round(float(axis_pos[a]), 3),
            'percentile': round(float(axis_pct[a]), 1),
            'z_score': round((axis_pos[a] - refs['axis_mean'][a]) /
                             (refs['axis_std'][a] + 1e-9), 2),
            'meaning': _axis_interpretation(a, float(axis_pct[a])),   # R4 plain-language
            'trajectory': [round(float(v), 3) for v in Zrow[:, a]],       # raw 30-layer
            'trajectory_norm': [round(float(v), 3) for v in Znorm[:, a]], # per-layer pop-normalized
        })

    coding_by_id = loaders.coding_status_by_id()
    return {
        'universe': universe,
        'isoform_id': iso_id,
        'gene': str(idx['gene_ids'][row]),
        'row': int(row),
        'is_noncoding': bool(coding_by_id is not None and not coding_by_id.get(iso_id, True)),
        'panel_A_go': _top_go(row, idx),
        'panel_C_axes': axes,
        'panel_C_summary': _fingerprint_summary(axes),
        'panel_D_trajectory': {
            'magnitude': [round(float(v), 3) for v in traj],
            'peak_layer': peak,
            'peak_value': round(float(traj[peak - 1]), 3),
            'ref_median': [round(v, 3) for v in refs['traj_median']],
            'ref_q25': [round(v, 3) for v in refs['traj_q25']],
            'ref_q75': [round(v, 3) for v in refs['traj_q75']],
            'n_layers': meta['n_layers'],
        },
        'layer_fisher': {
            'scores': meta.get('layer_fisher'),
            'label': meta.get('layer_fisher_label', 'per-layer discriminability'),
            'peak_layer': (int(np.argmax(meta['layer_fisher'])) + 1) if meta.get('layer_fisher') else None,
        },
        'covariates': _covariate_block(row, universe),   # 실제 서열 covariate (옵션 B)
    }


def gene_profile(gene: str, universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """유전자의 모든 아이소폼 요약 + 대표(최고 max-score) 아이소폼 상세."""
    idx = loaders.load_index(universe)
    rows = idx['gene_to_rows'].get(gene)
    if not rows:
        return {'error': f'gene {gene!r} not found'}

    coding_by_id = loaders.coding_status_by_id()   # None if unavailable (defensive: don't flag)
    iso_rows = []
    best_row, best_score = rows[0], -1.0
    for r in rows:
        s = np.asarray(idx['scores'][r], dtype=np.float32)
        mx = float(s.max())
        if mx > best_score:
            best_score, best_row = mx, r
        iso_id = str(idx['isoform_ids'][r])
        iso_rows.append({
            'isoform_id': iso_id,
            'peak_layer': int(idx['peak_layer'][r]),
            'axis3_domain': round(float(idx['axis_pos'][r, 3]), 2),
            'axis3_pct': round(float(idx['axis_pos_pct'][r, 3]), 0),
            'axis5_length': round(float(idx['axis_pos'][r, 5]), 2),
            'axis_pos': [round(float(v), 3) for v in idx['axis_pos'][r]],       # 8 axes (I4)
            'axis_pct': [round(float(v), 1) for v in idx['axis_pos_pct'][r]],
            'cov_pct': _cov_pct_row(r, universe),                               # 4 covariate pct (or None)
            'max_go_score': round(mx, 3),
            'top_go': _top_go(r, idx, k=1)[0],
            # non-coding transcript = ESM-2 zero-embedding by design → sigmoid(bias)-only
            # predictions, no real functional signal ([[finding-brain672-noncoding-zero-embedding-flag-needed]]).
            'is_noncoding': bool(coding_by_id is not None and not coding_by_id.get(iso_id, True)),
        })
    iso_rows.sort(key=lambda d: d['max_go_score'], reverse=True)

    return {
        'universe': universe,
        'gene': gene,
        'n_isoforms': len(rows),
        'isoforms': iso_rows,
        'representative': str(idx['isoform_ids'][best_row]),
        'heatmap': _gene_heatmap(rows, idx, top_k=15),
        'detail': isoform_profile(str(idx['isoform_ids'][best_row]), universe),
    }


def _gene_heatmap(rows: list[int], idx: dict, top_k: int = 15) -> dict:
    """transcripts × GO 히트맵 (R3): 이 유전자 아이소폼들이 어떤 기능에 예측되는지.

    GO term 선택 = 아이소폼 전체에서 max score 가 큰 top_k → 기능 모듈 구조가 드러남.
    """
    sub = np.asarray(idx['scores'][rows], dtype=np.float32)      # (n_iso, n_go)
    go_ids = idx['meta']['go_ids']
    go_names = idx['meta']['go_names']
    col_max = sub.max(axis=0)
    top_cols = np.argsort(col_max)[::-1][:top_k]
    submat = sub[:, top_cols]                                   # (n_iso, k)

    # 기능 모듈 = top GO 열을 아이소폼 프로파일로 클러스터 (I3)
    k = submat.shape[1]
    n_mod = int(min(3, k, max(1, len(rows))))
    if n_mod >= 2 and len(rows) >= 2:
        from sklearn.cluster import KMeans
        # 열(GO) 벡터 = 아이소폼별 score → 표준화 후 클러스터
        colvec = submat.T
        cv = (colvec - colvec.mean(0)) / (colvec.std(0) + 1e-6)
        mod_of_col = KMeans(n_clusters=n_mod, n_init=5, random_state=0).fit_predict(cv)
    else:
        mod_of_col = np.zeros(k, dtype=int)

    # 모듈별로 GO 를 묶어 정렬(같은 모듈이 인접) + 모듈 내 아이소폼-평균 내림차순
    col_mean = submat.mean(axis=0)
    order_local = sorted(range(k), key=lambda j: (mod_of_col[j], -col_mean[j]))
    order = top_cols[order_local]
    mods_ordered = [int(mod_of_col[j]) for j in order_local]
    iso_ids = [str(idx['isoform_ids'][r]) for r in rows]
    names_ordered = [go_names.get(go_ids[j], go_ids[j]) for j in order]

    # 각 모듈의 기능명 = 그 모듈에서 아이소폼-평균 최고 GO (I1)
    mod_labels = {}
    for m in set(mods_ordered):
        best_j = max((jj for jj in range(k) if mod_of_col[jj] == m), key=lambda jj: col_mean[jj])
        mod_labels[int(m)] = go_names.get(go_ids[top_cols[best_j]], go_ids[top_cols[best_j]])
    return {
        'isoform_ids': iso_ids,
        'go_ids': [go_ids[j] for j in order],
        'go_names': names_ordered,
        'modules': mods_ordered,                                # 열별 모듈 id (I3)
        'module_labels': mod_labels,                            # 모듈 id → 대표 기능명 (I1)
        'matrix': [[round(float(sub[i, j]), 3) for j in order] for i in range(len(rows))],
    }


def domain_architecture(iso_id: str, universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """이 아이소폼의 도메인 아키텍처 + 유전자 reference(최다 도메인 span) 대비 loss/gain/truncation.

    reference = 같은 유전자 아이소폼 중 총 도메인 span 이 가장 큰 것(≈ 가장 완전한 형태).
    """
    idx = loaders.load_index(universe)
    doms = loaders.brain_domains(universe)
    row = idx['id_to_row'].get(iso_id)
    if row is None:
        return {'error': f'{iso_id} not found'}
    gene = str(idx['gene_ids'][row])
    sib_ids = [str(idx['isoform_ids'][r]) for r in idx['gene_to_rows'].get(gene, [])]

    def span(dl):
        return sum(d['end'] - d['start'] for d in dl)

    # reference = 도메인 span 최대 아이소폼
    ref_id, ref_dom = None, []
    for sid in sib_ids:
        dl = doms.get(sid, [])
        if span(dl) > span(ref_dom):
            ref_id, ref_dom = sid, dl
    this_dom = doms.get(iso_id, [])
    if not this_dom and not ref_dom:
        return {'available': False, 'reason': 'no domain annotation (non-coding / novel / unmapped)'}

    # 도메인 family 별 비교 (reference 기준)
    def fam_span(dl):
        m = {}
        for d in dl:
            m.setdefault(d['name'], 0)
            m[d['name']] += d['end'] - d['start']
        return m
    ref_fam, this_fam = fam_span(ref_dom), fam_span(this_dom)
    changes = []
    for fam in sorted(set(ref_fam) | set(this_fam)):
        rs, ts = ref_fam.get(fam, 0), this_fam.get(fam, 0)
        if ts == 0 and rs > 0:
            changes.append({'domain': fam, 'status': 'lost', 'ref_len': rs, 'this_len': 0, 'retained': 0})
        elif rs == 0 and ts > 0:
            changes.append({'domain': fam, 'status': 'gained', 'ref_len': 0, 'this_len': ts, 'retained': 100})
        elif ts < rs * 0.9:
            changes.append({'domain': fam, 'status': 'truncated', 'ref_len': rs, 'this_len': ts,
                            'retained': round(100 * ts / rs)})
        else:
            changes.append({'domain': fam, 'status': 'intact', 'ref_len': rs, 'this_len': ts, 'retained': 100})

    def track(dl, iid):
        L = max([d['end'] for d in dl] + [1]) if dl else 1
        # 서열 길이(있으면 covariate mask 없으니 도메인 최대끝 사용)
        return {'iso': iid, 'len': int(L),
                'domains': [{'name': d['name'], 'start': d['start'], 'end': d['end']} for d in dl]}
    return {
        'available': True, 'gene': gene,
        'this': track(this_dom, iso_id),
        'ref': track(ref_dom, ref_id) if ref_id else None,
        'ref_id': ref_id, 'is_ref': (ref_id == iso_id),
        'changes': changes,
    }


def iso_trajectory(iso_id: str, universe: str = config.DEFAULT_UNIVERSE) -> dict:
    """단일 아이소폼의 PC1-3 3D 궤적 (형제 다중선택 overlay, I4/3D)."""
    idx = loaders.load_index(universe)
    row = idx['id_to_row'].get(iso_id)
    if row is None:
        return {'error': f'{iso_id} not found'}
    pc = np.asarray(idx['Z'][row][:, :3], dtype=np.float32)
    return {'iso': iso_id, 'traj': [[round(float(x), 2) for x in r] for r in pc]}
