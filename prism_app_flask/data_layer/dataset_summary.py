"""내 데이터 분석 — tissue 데이터셋의 핵심 지표를 한 페이지로 요약.

원칙:
  - QC/coverage/landscape 는 mmap score matrix 위 **벡터화 numpy**(빠름, 요청마다 가능).
  - 4-시나리오 분류는 canonical `prism_app.core.classifier.classify_isoforms` **재사용**
    (streamlit 앱과 동일 수치) — 단 63,994행 Python 루프라 lru_cache 로 1회만.
  - compute_functional_consequence 는 672-GO 패널에서 O(n_dtu×n_go)로 느림 → eager 실행 안 함;
    DTU 는 요약 통계만(streamlit 도 ≥100 GO 에서 apply-time skip).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from prism_app_flask import config
from prism_app_flask.data_layer import loaders as idx_loaders

# 재사용: framework-agnostic 로직
from prism_app.core.classifier import classify_isoforms
from prism_app.reports.coverage import generate_coverage_report

DEMO = config.DEMO_DIR
ANNOT = config.ANNOT_DIR

# "고신뢰" 구조타입별 분포 파이의 고정 임계값 — streamlit `app/components/sidebar.py` 의
# score_threshold 슬라이더 기본값(0.4)과 동일. "논문 S1×BISECT cross-link 수(32 genes)를
# 재현하는 기본값" 이라는 코멘트가 있는 앱 전역 고신뢰 기준 — 이 대시보드의 가변 QC
# threshold(기본 0.5, 사용자가 tissue 별로 바꿀 수 있음)와는 별개로 항상 0.4 로 고정한다.
TYPE_HIGH_CONF_THR = 0.4

# 지원 tissue → 파일 4종 (streamlit sidebar._load_demo_data 와 동일 매핑)
TISSUES = {
    'brain_672': ('brain_full_672_scores.npy', 'brain_full_672_ids.npy',
                  'brain_full_672_types.npy', 'brain_full_672_gene_ids.npy', 'brain_dtu.tsv'),
    'muscle':    ('muscle_scores.npy', 'muscle_ids.npy',
                  'muscle_types.npy', 'muscle_gene_ids.npy', None),
}

# Okabe-Ito 색각이상-안전 팔레트(instrument.js OKABE_ITO_SAFE 와 동일 값, 프런트-백엔드 통일).
# 이전엔 --signal/--amber 토큰의 구버전 neon 값(#FF5C1A/#FFB020, main.css 주석에 "was neon" 으로
# 남아있음)을 독립적으로 하드코딩해 토큰이 muted 로 바뀐 뒤에도 여기만 안 바뀐 채 남아있었다.
SCENARIO_META = {
    1: ('Functional Switch', 'DTU+ / novel GO+', '#D55E00'),   # vermillion — headline scenario
    2: ('Expression Switch', 'DTU+ / novel GO−', '#E69F00'),   # orange
    3: ('Constitutive Novel', 'DTU− / novel GO+', '#0072B2'),  # blue
    4: ('Background', 'DTU− / novel GO−', '#8A94A8'),          # neutral grey
}

# non-coding transcript(ORF 없음)는 ESM-2 임베딩이 설계상 zero-vector 처리되어
# (hMuscle/preprocessing/compute_brain_isoquant_esm2.py:111) 모든 GO term에서 sigmoid(bias)만
# 동일하게 출력한다 — 실제 기능 신호 없음(perfect partition으로 확인됨,
# [[finding-brain672-noncoding-zero-embedding-flag-needed]]). 이 마스크로 UMAP 클러스터링·
# GO-dist 하이라이트에서 제외한다. muscle 은 대응 마스크 파일이 없어 brain_672 전용.
_CODING_MASK_PATHS = {
    'brain_672': config.ROOT / 'hMuscle/data/brain_isoquant_esm2/full/brain_full_mask.npy',
}


@lru_cache(maxsize=2)
def _coding_mask(tissue: str):
    """이소폼별 coding(True)/non-coding(False) 마스크, 없으면 None. `load_tissue()`의
    ids/genes/sm과 같은 row 순서로 정렬돼 있음(실측 확인됨)."""
    p = _CODING_MASK_PATHS.get(tissue)
    if p is None or not p.exists():
        return None
    return np.load(p).astype(bool)


@lru_cache(maxsize=2)
def _existing_annotations() -> dict:
    """gene symbol → [GO:...] (novel GO 판정용). brain gene_ids 는 이미 symbol."""
    ann: dict[str, list[str]] = {}
    p = ANNOT / 'human_annotations_unified_bp.txt'
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split('\t')
            if len(parts) >= 2:
                ann[parts[0]] = parts[1:]
    return ann


@lru_cache(maxsize=2)
def load_tissue(tissue: str) -> dict:
    if tissue not in TISSUES:
        raise KeyError(tissue)
    sf, idf, tf, gf, dtuf = TISSUES[tissue]
    sm = np.load(DEMO / sf, mmap_mode='r')
    ids = np.load(DEMO / idf, allow_pickle=True)
    types = np.load(DEMO / tf, allow_pickle=True) if (DEMO / tf).exists() else None
    genes = np.load(DEMO / gf, allow_pickle=True) if (DEMO / gf).exists() else None
    dtu = pd.read_csv(DEMO / dtuf, sep='\t') if (dtuf and (DEMO / dtuf).exists()) else None
    # GO term id/name (brain_672 전용 소스; 그 외는 컬럼 인덱스)
    go_ids, go_names = _go_terms(tissue, sm.shape[1])
    coding_mask = _coding_mask(tissue)
    if coding_mask is not None and len(coding_mask) != len(ids):
        coding_mask = None  # 길이 불일치 시 안전하게 무시(잘못된 필터링보다 안 켜는 게 낫다)
    return {'sm': sm, 'ids': ids, 'types': types, 'genes': genes, 'dtu': dtu,
            'go_ids': go_ids, 'go_names': go_names, 'coding_mask': coding_mask}


@lru_cache(maxsize=4)
def _classify_cached(tissue: str, threshold: float = 0.5) -> pd.DataFrame:
    """classify_isoforms 결과 캐시 — `_compute_summary` / gene quick report / UMAP 색칠이 공유.
    63,994행 Python 루프라 (tissue, threshold) 조합당 1회만 계산."""
    d = load_tissue(tissue)
    sm = np.asarray(d['sm'])
    return classify_isoforms(
        sm, d['ids'], gene_ids=d['genes'], go_terms=d['go_ids'],
        existing_annotations=_existing_annotations(),
        dtu_df=d['dtu'], score_threshold=threshold)


def _go_terms(tissue: str, n_go: int):
    import json
    if tissue.startswith('brain'):
        p = config.ROOT / 'hMuscle/data/brain672_go_terms.json'
        if p.exists():
            gn = json.loads(p.read_text()).get('go_names', {})
            ids = list(gn.keys())[:n_go]
            return ids, gn
    # 그 외 tissue: canonical preset 에서 id/name 순서 그대로 재사용
    try:
        from prism_app.core.go_utils import TISSUE_PRESETS
        preset = TISSUE_PRESETS.get(tissue) or TISSUE_PRESETS.get('muscle')
        if preset and len(preset) >= n_go:
            ids = list(preset.keys())[:n_go]
            return ids, dict(preset)
    except Exception:  # noqa: BLE001
        pass
    ids = [f'GO_{i}' for i in range(n_go)]
    return ids, {g: g for g in ids}


def _cache_path(tissue: str, threshold: float) -> Path:
    return config.INDEX_DIR / f'summary_{tissue}_thr{threshold:.2f}_v2.json'


# ── Novel-GO Venn: existing-annotation space vs PRISM-predicted space ────────
# §4-scenario 의 "novel GO+"(DTU 무관)를 한 겹 더 쪼갠다. 애초 가설(기존 툴 주석이 MF라서
# 새 BP를 novel로 잡는다)은 기각됨 — panel 672-term 중 660/672, existing-annotation lookup
# 파일(human_annotations_unified_bp.txt) 중 11,260/11,594 가 전부 Biological Process 도메인
# (gene2go.gz 로 직접 검증, 나머지는 MF/CC 가 아니라 이 덤프에 없는 구식/병합된 GO id).
# 즉 도메인 불일치가 아니라 "얼마나 진짜 새로운가"의 문제.
#
# 이전 버전(Case A/B/C, isoform 단위)은 첫 사용자 피드백에서 "네이밍이 안 직관적" + "Case B
# gap histogram 이 핵심 정보인지 모르겠다"로 재검토 요청받아, (gene, GO term) **쌍** 단위의
# 2-집합 벤다이어그램으로 재구성했다 — "기존 annotation 공간"과 "PRISM 예측 공간"의 교집합/
# 차집합이라는 사용자의 프레이밍을 그대로 반영:
#   overlap      = 이미 주석된 GO term을 PRISM도 예측(threshold 이상) — 기존 annotation 재현.
#   known_only   = 이미 주석됐지만 PRISM이 놓침(threshold 미만) — 모델의 재현 실패(누락).
#   new(predicted_only) = PRISM만 예측(threshold 이상), 기존 주석엔 없음 — 새 예측 전부
#     (이전 버전의 Case A+B+C를 합친 것과 동치: 완전 무주석 gene의 모든 예측도 여기 포함).
# new 안에서 두 개의 독립적 하위집합(색 on/off 토글, Venn 안엔 겹쳐서 표시):
#   hiconf       = new 중 score > hiconf_threshold(고신뢰, 기본 0.7).
#   beats        = new 중, gene에 기존 주석이 있는 경우에 한해(완전 무주석 gene은 비교 대상이
#                  없으므로 제외) score가 그 gene의 기존-주석 term 중 최고점수를 넘는 것
#                  (이전 버전의 Case B와 동일 정의, gene-pair 단위로 재계산).
def _gene_symbol_overlap(genes: np.ndarray, ann: dict) -> float:
    uniq = set(str(g) for g in np.unique(genes))
    if not uniq:
        return 0.0
    return len(uniq & set(ann.keys())) / len(uniq)


@lru_cache(maxsize=4)
def _novel_go_venn(tissue: str, threshold: float = 0.5, hiconf_threshold: float = 0.7):
    """None 이면 이 tissue 는 gene id 형식이 annotation lookup(symbol 기준)과 안 맞아 계산 불가
    (예: muscle 은 ENSG*.* — symbol 매핑 없이 계산하면 전부 predicted-only 로 쏠리는 artifact
    발생, [[research-method]] S0). brain_672 는 이미 symbol 이라 정상 동작."""
    d = load_tissue(tissue)
    genes = d['genes']
    if genes is None:
        return None
    genes = np.asarray(genes, dtype=str)
    ann = _existing_annotations()
    if _gene_symbol_overlap(genes, ann) < 0.05:
        return None

    sm = np.asarray(d['sm'])
    N, G = sm.shape
    go_ids = d['go_ids']
    go_idx = {g: j for j, g in enumerate(go_ids)}
    ids = d['ids']
    coding_mask = d['coding_mask']
    if coding_mask is not None:
        sm = np.where(coding_mask[:, None], sm, 0.0)   # non-coding isoform은 신호 없는 예측이므로
        # 사실상 "예측 안 됨"과 동일하게 취급 — gene의 다른(coding) isoform이 살아있으면 그걸로 판정.

    gene_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        gene_to_rows.setdefault(g, []).append(i)

    n_overlap = n_known_only = n_new = n_new_hiconf = n_new_beats = 0
    genes_overlap, genes_known_only, genes_new, genes_new_hiconf, genes_new_beats = (set() for _ in range(5))
    n_genes_gap = n_genes_annotated = 0
    beats_examples = []   # (gene, isoform_id, go, score, own_best)

    for g, rows in gene_to_rows.items():
        rows_arr = np.asarray(rows)
        sub = sm[rows_arr]                                  # (n_i, G)
        predicted_mask = (sub > threshold).any(axis=0)       # (G,)
        known_panel_idx = [go_idx[t] for t in ann.get(g, []) if t in go_idx]

        if not known_panel_idx:
            n_genes_gap += 1
            new_idx = np.where(predicted_mask)[0]
            if len(new_idx):
                genes_new.add(g)
                n_new += len(new_idx)
                max_scores = sub[:, new_idx].max(axis=0)
                hiconf_idx = new_idx[max_scores > hiconf_threshold]
                if len(hiconf_idx):
                    genes_new_hiconf.add(g); n_new_hiconf += len(hiconf_idx)
            continue

        n_genes_annotated += 1
        known_mask = np.zeros(G, dtype=bool)
        known_mask[known_panel_idx] = True
        overlap_mask = known_mask & predicted_mask
        known_only_mask = known_mask & ~predicted_mask
        new_mask = predicted_mask & ~known_mask

        if overlap_mask.any():
            genes_overlap.add(g); n_overlap += int(overlap_mask.sum())
        if known_only_mask.any():
            genes_known_only.add(g); n_known_only += int(known_only_mask.sum())
        new_idx = np.where(new_mask)[0]
        if len(new_idx):
            genes_new.add(g); n_new += len(new_idx)
            own_best = float(sub[:, known_panel_idx].max())
            max_scores = sub[:, new_idx].max(axis=0)
            hiconf_sel = max_scores > hiconf_threshold
            beats_sel = max_scores > own_best
            if hiconf_sel.any():
                genes_new_hiconf.add(g); n_new_hiconf += int(hiconf_sel.sum())
            if beats_sel.any():
                genes_new_beats.add(g); n_new_beats += int(beats_sel.sum())
                for j in new_idx[beats_sel]:
                    # concrete isoform example — which isoform of g actually carries this score
                    col = sub[:, j]
                    best_row = rows_arr[int(np.argmax(col))]
                    beats_examples.append((g, str(ids[best_row]), go_ids[j],
                                            float(col.max()), own_best))

    beats_examples.sort(key=lambda x: -(x[3] - x[4]))

    return {
        'threshold': threshold, 'hiconf_threshold': hiconf_threshold,
        'n_genes_annotation_gap': n_genes_gap, 'n_genes_annotated_on_panel': n_genes_annotated,
        'counts': {'overlap': n_overlap, 'known_only': n_known_only, 'new': n_new,
                   'new_hiconf': n_new_hiconf, 'new_beats': n_new_beats},
        'genes_affected': {'overlap': len(genes_overlap), 'known_only': len(genes_known_only),
                           'new': len(genes_new), 'new_hiconf': len(genes_new_hiconf),
                           'new_beats': len(genes_new_beats)},
        'top_beats_examples': [
            {'gene': g, 'isoform_id': iso, 'novel_go': go, 'go_name': d['go_names'].get(go, go),
             'novel_score': round(sc, 4), 'own_best_score': round(ob, 4)}
            for g, iso, go, sc, ob in beats_examples[:12]
        ],
    }


NOVEL_GO_REGIONS = ('known_only', 'overlap', 'new', 'new_hiconf', 'new_beats')


@lru_cache(maxsize=16)
def novel_go_region_isoforms(tissue: str, region: str, threshold: float = 0.5,
                              hiconf_threshold: float = 0.7, cap: int = 500) -> dict:
    """Venn 영역 클릭 → 그 영역에 속한 (gene, GO term) pair 의 대표 isoform 목록.

    _novel_go_venn() 과 동일한 (gene, GO) pair 분류 기준을 재사용(집계 대신 개별 pair 수집) —
    각 pair 는 그 유전자의 isoform 중 해당 GO term 점수가 최고인 것으로 대표시킨다(beats_examples
    와 동일 관례). known_only 는 정의상 threshold 미달이라 대표 isoform 의 score 도 threshold 이하로
    나올 수 있음(annotated 됐지만 PRISM 이 놓친 케이스이므로 오히려 그게 정보) — 에러 아님.
    """
    if region not in NOVEL_GO_REGIONS:
        return {'error': f'unknown region {region!r}'}
    d = load_tissue(tissue)
    genes = d['genes']
    if genes is None:
        return {'error': f'{tissue} has no gene-symbol mapping for annotation lookup'}
    genes = np.asarray(genes, dtype=str)
    ann = _existing_annotations()
    if _gene_symbol_overlap(genes, ann) < 0.05:
        return {'error': f'{tissue} gene ids do not match the symbol-keyed annotation panel'}

    sm = np.asarray(d['sm'])
    N, G = sm.shape
    go_ids = d['go_ids']
    go_idx = {g: j for j, g in enumerate(go_ids)}
    ids = d['ids']
    coding_mask = d['coding_mask']
    if coding_mask is not None:
        sm = np.where(coding_mask[:, None], sm, 0.0)

    gene_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        gene_to_rows.setdefault(g, []).append(i)

    rows_out = []
    total = 0

    def emit(gene, go_j, score, row):
        nonlocal total
        total += 1
        if len(rows_out) < cap:
            rows_out.append({'gene': gene, 'isoform_id': str(ids[row]), 'go_id': go_ids[go_j],
                              'go_name': d['go_names'].get(go_ids[go_j], go_ids[go_j]),
                              'score': round(float(score), 4)})

    for g, rows in gene_to_rows.items():
        rows_arr = np.asarray(rows)
        sub = sm[rows_arr]
        predicted_mask = (sub > threshold).any(axis=0)
        known_panel_idx = [go_idx[t] for t in ann.get(g, []) if t in go_idx]

        if not known_panel_idx:
            if region not in ('new', 'new_hiconf'):
                continue
            new_idx = np.where(predicted_mask)[0]
            if not len(new_idx):
                continue
            max_scores = sub[:, new_idx].max(axis=0)
            sel = new_idx if region == 'new' else new_idx[max_scores > hiconf_threshold]
            for j in sel:
                col = sub[:, j]
                best_row = rows_arr[int(np.argmax(col))]
                emit(g, j, col.max(), best_row)
            continue

        known_mask = np.zeros(G, dtype=bool)
        known_mask[known_panel_idx] = True
        overlap_mask = known_mask & predicted_mask
        known_only_mask = known_mask & ~predicted_mask
        new_mask = predicted_mask & ~known_mask

        if region == 'known_only':
            for j in np.where(known_only_mask)[0]:
                col = sub[:, j]
                best_row = rows_arr[int(np.argmax(col))]
                emit(g, j, col.max(), best_row)
        elif region == 'overlap':
            for j in np.where(overlap_mask)[0]:
                col = sub[:, j]
                best_row = rows_arr[int(np.argmax(col))]
                emit(g, j, col.max(), best_row)
        else:   # new / new_hiconf / new_beats
            new_idx = np.where(new_mask)[0]
            if not len(new_idx):
                continue
            max_scores = sub[:, new_idx].max(axis=0)
            if region == 'new':
                sel = new_idx
            elif region == 'new_hiconf':
                sel = new_idx[max_scores > hiconf_threshold]
            else:   # new_beats
                own_best = float(sub[:, known_panel_idx].max())
                sel = new_idx[max_scores > own_best]
            for j in sel:
                col = sub[:, j]
                best_row = rows_arr[int(np.argmax(col))]
                emit(g, j, col.max(), best_row)

    rows_out.sort(key=lambda r: -r['score'])
    return {'region': region, 'threshold': threshold, 'hiconf_threshold': hiconf_threshold,
            'total': total, 'isoforms': rows_out, 'truncated': total > len(rows_out)}


# ── Triage ranked list — supersedes the S1-S4 (DTU × novel-GO) donut ─────────
# Spec: prism_app_flask/docs/mydata_triage_design.md §3. Every isoform that differs from its
# gene's canonical (MANE/Ensembl/APPRIS/longest_CDS), ranked by VALIDATED-consequence evidence
# first (domain-family identity, ORF change) — never by edit magnitude alone (§3.2, Attack 5).
# No hard tier walls: every alt isoform is shown, each row just carries whichever evidence kinds
# it has (§3.3) — nothing is buried into a "not worth looking" bucket.
#
# canonical.json / domains.json / covariates.npz live under the isoform_index/<universe> tree
# (built by precompute/build_canonical_map.py etc.) and are row-aligned to THAT universe's
# isoform_ids.npy — not necessarily this module's own `load_tissue()` row order. Only 'brain_672'
# has been verified to share row order 1:1 with isoform_index universe 'brain' (63,994 isoforms,
# identical id sequence, checked at implementation time) — 'muscle' has no isoform_index built, so
# it degrades gracefully to novel-GO + structural-type evidence only (no domain/ORF/descriptor).
TRIAGE_TISSUE_TO_UNIVERSE = {'brain_672': 'brain'}

# disorder_frac Δ below this magnitude is noise, not a reportable descriptor (spec §3.3 guardrail).
_TRIAGE_DISORDER_NOISE = 0.1

# evidence kind → priority for the default 'consequence' sort (§3.2: named-domain/ORF > IDR/
# compositional > novel-GO-only > nothing detected). Never includes edit magnitude as a factor.
_TRIAGE_KIND_RANK = {'domain': 4, 'orf': 3, 'idr': 2, 'novel_go': 1, 'none': 0}

TRIAGE_CONSEQUENCE_TYPES = ('domain', 'orf', 'idr', 'novel_go', 'none')
TRIAGE_SORTS = ('consequence', 'score', 'confidence')


def _domain_fam_span(domain_list: list[dict]) -> dict:
    m: dict[str, int] = {}
    for d in domain_list:
        m[d['name']] = m.get(d['name'], 0) + (d['end'] - d['start'])
    return m


def _domain_changes(this_dom: list[dict], ref_dom: list[dict]) -> list[dict]:
    """이 아이소폼의 도메인 family span 을 canonical 대비 비교 — lost/gained/truncated 만 보고
    (intact 은 '변화'가 아니므로 생략, isoform_profile.domain_architecture 와 동일 규칙 재사용)."""
    ref_fam, this_fam = _domain_fam_span(ref_dom), _domain_fam_span(this_dom)
    changes = []
    for fam in sorted(set(ref_fam) | set(this_fam)):
        rs, ts = ref_fam.get(fam, 0), this_fam.get(fam, 0)
        if ts == 0 and rs > 0:
            changes.append({'domain': fam, 'status': 'lost', 'retained': 0})
        elif rs == 0 and ts > 0:
            changes.append({'domain': fam, 'status': 'gained', 'retained': 100})
        elif ts < rs * 0.9:
            changes.append({'domain': fam, 'status': 'truncated', 'retained': round(100 * ts / rs)})
    return changes


@lru_cache(maxsize=4)
def _novel_go_owner_map(tissue: str, threshold: float = 0.5, hiconf_threshold: float = 0.7) -> dict:
    """isoform_id -> 그 아이소폼이 실제로 '들고 있는'(자기 유전자 내에서 그 GO 컬럼 최고점수)
    novel-GO(new_mask) 예측 중 최고점 1개. `_novel_go_venn` 과 동일한 (gene,GO) 분류 기준을
    isoform 단위로 재키(re-key)한 것 — 집계 대신 개별 소유자를 남긴다."""
    d = load_tissue(tissue)
    genes = d['genes']
    if genes is None:
        return {}
    genes = np.asarray(genes, dtype=str)
    ann = _existing_annotations()
    if _gene_symbol_overlap(genes, ann) < 0.05:
        return {}

    sm = np.asarray(d['sm'])
    N, G = sm.shape
    go_ids = d['go_ids']
    go_idx = {g: j for j, g in enumerate(go_ids)}
    ids = d['ids']
    coding_mask = d['coding_mask']
    if coding_mask is not None:
        sm = np.where(coding_mask[:, None], sm, 0.0)

    gene_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        gene_to_rows.setdefault(g, []).append(i)

    out: dict[str, dict] = {}
    for g, rows in gene_to_rows.items():
        rows_arr = np.asarray(rows)
        sub = sm[rows_arr]
        predicted_mask = (sub > threshold).any(axis=0)
        known_panel_idx = [go_idx[t] for t in ann.get(g, []) if t in go_idx]
        if not known_panel_idx:
            new_idx = np.where(predicted_mask)[0]
        else:
            known_mask = np.zeros(G, dtype=bool)
            known_mask[known_panel_idx] = True
            new_idx = np.where(predicted_mask & ~known_mask)[0]
        if not len(new_idx):
            continue
        for j in new_idx:
            col = sub[:, j]
            best_local = int(np.argmax(col))
            score = float(col[best_local])
            iso_id = str(ids[int(rows_arr[best_local])])
            prev = out.get(iso_id)
            if prev is None or score > prev['score']:
                out[iso_id] = {'go_id': go_ids[j], 'go_name': d['go_names'].get(go_ids[j], go_ids[j]),
                               'score': round(score, 4), 'hiconf': score > hiconf_threshold}
    return out


@lru_cache(maxsize=4)
def _triage_rows_raw(tissue: str, threshold: float = 0.5, hiconf_threshold: float = 0.7) -> tuple[dict, ...]:
    """전 alt-isoform(자기 유전자의 canonical 과 다른, 2+ isoform 유전자에 속한 row) 의 원시
    evidence row — 정렬/필터 없음(캡도 없음). (tissue, threshold, hiconf_threshold) 조합당 1회만
    계산(gene 루프, `_novel_go_venn` 과 동일 비용대). 필터/정렬/cap 은 `triage_ranked()` 가 이
    캐시된 리스트 위에서 매 요청마다 수행(파이썬 정렬 4-5만 row 는 충분히 빠름)."""
    d = load_tissue(tissue)
    genes = d['genes']
    if genes is None:
        return ()
    genes = np.asarray(genes, dtype=str)
    ids = np.asarray(d['ids'], dtype=str)
    types = np.asarray(d['types'], dtype=str) if d['types'] is not None else None
    sm = np.asarray(d['sm'])

    universe = TRIAGE_TISSUE_TO_UNIVERSE.get(tissue)
    if universe:
        canon = idx_loaders.canonical_map(universe)
        doms = idx_loaders.brain_domains(universe)
        cov = idx_loaders.brain_covariates(universe)
        coding = idx_loaders.coding_status_by_id()
    else:
        canon, doms, cov, coding = {}, {}, None, None
    disorder_j = (cov['names'].index('disorder_frac')
                  if (cov is not None and 'disorder_frac' in cov['names']) else None)
    novel = _novel_go_owner_map(tissue, threshold, hiconf_threshold)

    gene_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        gene_to_rows.setdefault(g, []).append(i)

    rows_out = []
    for g, rows in gene_to_rows.items():
        if len(rows) < 2:
            continue   # lone isoform — nothing to compare against, not an "alt" by definition
        c = canon.get(g)
        canon_row, canon_id = (c['row'], c['id']) if c else (None, None)
        for r in rows:
            if canon_row is not None and r == canon_row:
                continue   # canonical itself is the reference, not a triage candidate
            iso_id = ids[r]
            entry = {
                'gene': g, 'isoform_id': iso_id,
                'structural_type': types[r] if types is not None else None,
                'max_score': round(float(sm[r].max()), 3),
                'no_canonical_ref': canon_row is None,
                'canonical_id': canon_id,
                'domain_changes': [], 'orf_change': None,
                'descriptor': None, 'novel_go': novel.get(iso_id),
            }
            if canon_row is not None:
                this_dom, ref_dom = doms.get(iso_id, []), doms.get(canon_id, [])
                if this_dom or ref_dom:
                    entry['domain_changes'] = _domain_changes(this_dom, ref_dom)
                if coding is not None:
                    tc, rc = coding.get(iso_id), coding.get(canon_id)
                    if tc is not None and rc is not None and tc != rc:
                        entry['orf_change'] = 'lost' if (rc and not tc) else 'gained'
                if cov is not None and disorder_j is not None \
                        and bool(cov['mask'][r]) and bool(cov['mask'][canon_row]):
                    dv = float(cov['values'][r, disorder_j] - cov['values'][canon_row, disorder_j])
                    if abs(dv) >= _TRIAGE_DISORDER_NOISE:
                        entry['descriptor'] = {'disorder_delta': round(dv, 3)}
            if entry['domain_changes']:
                kind = 'domain'
            elif entry['orf_change']:
                kind = 'orf'
            elif entry['descriptor']:
                kind = 'idr'
            elif entry['novel_go']:
                kind = 'novel_go'
            else:
                kind = 'none'
            entry['kind'] = kind
            entry['kind_rank'] = _TRIAGE_KIND_RANK[kind]
            rows_out.append(entry)
    return tuple(rows_out)


def triage_ranked(tissue: str, threshold: float = 0.5, hiconf_threshold: float = 0.7,
                   consequence: str = '', structural_type: str = '', sort: str = 'consequence',
                   cap: int = 500) -> dict:
    """§3.1 단일 ranked list API — 필터/정렬은 요청마다 캐시된 원시 row 위에서 적용."""
    rows = list(_triage_rows_raw(tissue, threshold, hiconf_threshold))
    if consequence in TRIAGE_CONSEQUENCE_TYPES:
        rows = [r for r in rows if r['kind'] == consequence]
    if structural_type in ('known', 'nic', 'nnic'):
        rows = [r for r in rows if r['structural_type'] == structural_type]

    if sort == 'score':
        rows.sort(key=lambda r: -r['max_score'])
    elif sort == 'confidence':
        # lit-checked novel-GO first (hiconf badge), then validated-discriminator kind, tie by score.
        rows.sort(key=lambda r: (0 if (r['novel_go'] and r['novel_go']['hiconf']) else 1,
                                  -r['kind_rank'], -r['max_score']))
    else:   # 'consequence' (default) — §3.2: validated-consequence evidence first, magnitude never
        rows.sort(key=lambda r: (-r['kind_rank'], -r['max_score']))

    total = len(rows)
    kind_counts = {k: 0 for k in TRIAGE_CONSEQUENCE_TYPES}
    for r in rows:
        kind_counts[r['kind']] += 1
    return {
        'tissue': tissue, 'threshold': threshold, 'hiconf_threshold': hiconf_threshold,
        'consequence_filter': consequence or None, 'structural_type_filter': structural_type or None,
        'sort': sort, 'total': total, 'kind_counts': kind_counts,
        'rows': rows[:cap], 'truncated': total > cap,
    }


# ── GO term score distribution, per isoform, gene-highlightable (replaces the max-per-isoform
# histogram) ── 이유: "max GO score per isoform" 히스토그램은 QC용 sanity check이지, 연구자가
# 실제로 궁금해하는 "내가 학습시킨 이 GO term에 대해, 내 데이터의 유전자들이 어떤 점수 분포를
# 보이는가"에 답하지 못한다. GO term 하나를 고르면 그 컬럼의 raw score 를 아이소폼 단위 그대로
# 반환 — per-gene KDE smoothing 은 하지 않는다(어느 값이 어느 gene 인지는 hover/검색으로 직접
# 확인, 곡선 모양으로 추정하지 않음). population 전체에 대한 KDE 참조선만 별도로 계산.
def go_term_list(tissue: str) -> list[dict]:
    """드롭다운용 — 이 tissue 로 학습된 MLP 가 실제로 예측하는 GO term 전체 목록."""
    d = load_tissue(tissue)
    go_ids, gn = d['go_ids'], d['go_names']
    return [{'go_id': g, 'name': gn.get(g, g)} for g in go_ids]


# population-wide 참조선(aggregate) 1개에만 쓰는 고정 격자·bandwidth.
_DIST_GRID = np.linspace(0.0, 1.0, 24, dtype=np.float32)
_DIST_BANDWIDTH = 0.05


def _gaussian_kde_1d(values: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    diffs = (grid[None, :] - values[:, None]) / bandwidth
    return np.exp(-0.5 * diffs ** 2).sum(axis=0)


@lru_cache(maxsize=20)
def go_term_distribution(tissue: str, go_id: str) -> dict:
    """선택한 GO term 하나 → 아이소폼별 raw score(spike 하나 = 아이소폼 하나, 값 그대로 — smoothing
    없음)와 그 gene 라벨을 반환. 프론트(mydata.js)가 각 spike 를 hover 가능한 마커로 그리고,
    hover/검색 시 같은 gene 소속 spike 전부를 하이라이트한다 — "이 값이 어느 gene 소속인가"를
    KDE 곡선 모양으로 추정하지 않고 실제 데이터 포인트로 직접 보여주기 위함(Occam: per-gene KDE
    보다 단순하고, y축에 "relative density"라는 해석이 필요한 통계량이 없다).

    `aggregate` 는 전체 population(모든 아이소폼 pooled) 위의 단일 KDE 참조선 — gene 무관하게
    "이 GO term 점수가 전체적으로 어디에 분포하는가"를 보여주는 통계 요약이며, 프론트에서
    항상 다른 trace보다 앞에 그려진다(scattergl 위 scatter 는 레이어가 항상 위)."""
    d = load_tissue(tissue)
    go_ids = d['go_ids']
    if go_id not in go_ids:
        return {'error': f'unknown go_id {go_id!r} for tissue {tissue!r}'}
    col = go_ids.index(go_id)
    sm = np.asarray(d['sm'])
    scores = sm[:, col].astype(np.float32)
    ids = np.asarray(d['ids'], dtype=str)
    genes = np.asarray(d['genes'], dtype=str) if d['genes'] is not None else None
    gn = d['go_names']
    grid = _DIST_GRID
    grid_list = grid.round(4).tolist()

    agg = _gaussian_kde_1d(scores, grid, _DIST_BANDWIDTH)
    agg = agg / (agg.max() or 1)

    # non-coding transcript는 ESM-2 임베딩이 설계상 zero-vector라 모든 GO term에서 동일한
    # sigmoid(bias)만 출력 — 실제 신호 없음([[finding-brain672-noncoding-zero-embedding-flag-needed]]).
    # 프론트가 이 이소폼들을 구분 표시(회색·기본 숨김)할 수 있도록 함께 반환.
    coding_mask = d['coding_mask']
    is_noncoding = (~coding_mask).tolist() if coding_mask is not None else [False] * len(scores)

    return {
        'go_id': go_id, 'go_name': gn.get(go_id, go_id), 'grid': grid_list,
        'aggregate': agg.round(4).tolist(),
        'isoform_scores': scores.round(4).tolist(),
        'isoform_ids': ids.tolist(),
        'isoform_genes': genes.tolist() if genes is not None else [],
        'n_isoforms_total': int(len(scores)),
        'is_noncoding': is_noncoding, 'n_noncoding': int(sum(is_noncoding)),
    }


# 단일 아이소폼 유전자의 "고득점" 판정 임계값 — 앱 전역 고신뢰 기준(TYPE_HIGH_CONF_THR)과
# 동일하게 고정, QC threshold(가변)와는 독립.
GENE_HIGHLIGHT_SINGLE_ISO_THR = TYPE_HIGH_CONF_THR
# 단일-아이소폼-고득점 리스트는 상한 없이 임계값 이상 전부 반환하되, 응답 크기 방어용 하드캡.
_SINGLE_ISO_HIGH_CAP = 500


@lru_cache(maxsize=20)
def go_term_gene_highlights(tissue: str, go_id: str) -> dict:
    """GO term score distribution·by gene 위의 4개 하이라이트 criterion:
      - top_score: 이 GO term에서 가장 높은 점수를 받은 단일 아이소폼이 속한 유전자 (1개)
      - top_variance: 유전자 내 아이소폼 간 점수 분산이 가장 큰 유전자 top-10 (n_isoforms>=2 대상)
      - single_iso_high: 아이소폼이 1개뿐이면서 그 점수가 GENE_HIGHLIGHT_SINGLE_ISO_THR 이상인
        유전자 전부(점수 내림차순, 방어적 상한 _SINGLE_ISO_HIGH_CAP)
      - top_mean: 유전자의 아이소폼 평균 점수가 가장 큰 유전자 top-10
    각 항목은 프론트에서 이미 보유한 `_goDistGeneRows`(gene→row indices)로 바로 하이라이트
    가능하도록 gene 이름만 있으면 충분 — 대표 isoform_id/통계값은 리스트 표시용으로 곁들인다."""
    d = load_tissue(tissue)
    go_ids = d['go_ids']
    if go_id not in go_ids:
        return {'error': f'unknown go_id {go_id!r} for tissue {tissue!r}'}
    if d['genes'] is None:
        return {'error': 'no gene mapping for this tissue'}
    col = go_ids.index(go_id)
    sm = np.asarray(d['sm'])
    scores = sm[:, col].astype(np.float32)
    ids = np.asarray(d['ids'], dtype=str)
    genes = np.asarray(d['genes'], dtype=str)
    df = pd.DataFrame({'gene': genes, 'isoform_id': ids, 'score': scores})

    # non-coding transcript는 zero-embedding→sigmoid(bias)뿐인 신호 없는 예측이므로, "흥미로운
    # 유전자"를 찾는 4개 criterion 전부에서 제외한다 — 안 그러면 희귀 GO term에서 이 baseline
    # 값이 우연히 top_score/top_mean으로 뽑힐 위험이 있다([[finding-brain672-noncoding-zero-embedding-flag-needed]]).
    coding_mask = d['coding_mask']
    n_noncoding_excluded = 0
    if coding_mask is not None:
        n_noncoding_excluded = int((~coding_mask).sum())
        df = df[coding_mask].reset_index(drop=True)
        if df.empty:
            return {'error': 'all isoforms non-coding for this tissue'}

    top_row = df.loc[df['score'].idxmax()]
    top_score = {'gene': str(top_row['gene']), 'isoform_id': str(top_row['isoform_id']),
                 'score': round(float(top_row['score']), 4)}

    grouped = df.groupby('gene')['score']
    # ddof=0: 이 GO term에서 이 유전자의 아이소폼 전부가 population(샘플 아님) — Bessel 보정 불필요.
    stats = grouped.agg(n_isoforms='count', mean_score='mean',
                         var_score=lambda s: s.var(ddof=0), max_score='max').reset_index()
    stats['var_score'] = stats['var_score'].fillna(0.0)

    tv = stats[stats['n_isoforms'] >= 2].sort_values('var_score', ascending=False).head(10)
    top_variance = [{'gene': str(r['gene']), 'variance': round(float(r['var_score']), 5),
                     'n_isoforms': int(r['n_isoforms']), 'max_score': round(float(r['max_score']), 4)}
                    for _, r in tv.iterrows()]

    single_all = (stats[(stats['n_isoforms'] == 1) & (stats['max_score'] >= GENE_HIGHLIGHT_SINGLE_ISO_THR)]
                  .sort_values('max_score', ascending=False))
    single = single_all.head(_SINGLE_ISO_HIGH_CAP)
    single_id_by_gene = dict(zip(df['gene'], df['isoform_id']))
    single_iso_high = [{'gene': str(r['gene']), 'isoform_id': str(single_id_by_gene.get(r['gene'], '')),
                        'score': round(float(r['max_score']), 4)} for _, r in single.iterrows()]

    tm = stats.sort_values('mean_score', ascending=False).head(10)
    top_mean = [{'gene': str(r['gene']), 'mean_score': round(float(r['mean_score']), 4),
                'n_isoforms': int(r['n_isoforms'])} for _, r in tm.iterrows()]

    return {
        'go_id': go_id, 'go_name': d['go_names'].get(go_id, go_id),
        'top_score': top_score, 'top_variance': top_variance,
        'single_iso_high': single_iso_high, 'single_iso_thr': GENE_HIGHLIGHT_SINGLE_ISO_THR,
        'single_iso_high_truncated': bool(len(single_all) > _SINGLE_ISO_HIGH_CAP),
        'top_mean': top_mean, 'n_noncoding_excluded': n_noncoding_excluded,
    }


@lru_cache(maxsize=4)
def summarize(tissue: str, threshold: float = 0.5) -> dict:
    """디스크 캐시 우선(classify_isoforms 63,994행 루프는 cold ~12s) → 있으면 즉시 로드."""
    import json
    cp = _cache_path(tissue, threshold)
    if cp.exists():
        return json.loads(cp.read_text())
    result = _compute_summary(tissue, threshold)
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(result, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass
    return result


def _compute_summary(tissue: str, threshold: float) -> dict:
    d = load_tissue(tissue)
    sm = np.asarray(d['sm'])                      # (N, G) — RAM 로드(요약 계산용)
    N, G = sm.shape
    genes = np.asarray(d['genes'], dtype=str) if d['genes'] is not None else None

    # ── 벡터화 QC ──────────────────────────────────────────────
    max_per_iso = sm.max(axis=1)
    high_mask = sm > threshold
    n_high_per_iso = high_mask.sum(axis=1)
    cov_per_go = high_mask.sum(axis=0)            # (G,) 각 GO term 커버리지
    n_highconf = int((max_per_iso > threshold).sum())

    # 점수 분포 히스토그램(max-per-iso)
    hist, edges = np.histogram(max_per_iso, bins=30, range=(0, 1))

    # 타입 분해
    types_breakdown = {}
    if d['types'] is not None:
        vals, counts = np.unique(np.asarray(d['types'], dtype=str), return_counts=True)
        types_breakdown = {str(v): int(c) for v, c in sorted(
            zip(vals, counts), key=lambda x: -x[1])}

    # 모듈 landscape: 커버리지 top-15 GO term
    order = np.argsort(cov_per_go)[::-1][:15]
    landscape = [{'go_id': d['go_ids'][j], 'name': d['go_names'].get(d['go_ids'][j], d['go_ids'][j]),
                  'coverage': int(cov_per_go[j])} for j in order]

    # ── 고신뢰(score>0.4) 아이소폼의 구조타입별 분포 — canonical coverage 리포트 재사용
    #    (streamlit 01_qc.py "Isoforms with score>thr by structural type" 파이와 동일 정의).
    #    대시보드 가변 threshold 와 무관하게 앱 전역 고신뢰 기준(0.4) 로 고정.
    type_high_conf = None
    if d['types'] is not None:
        cov04 = generate_coverage_report(sm, d['ids'], d['types'], d['go_ids'], d['go_names'],
                                          score_threshold=TYPE_HIGH_CONF_THR)
        type_high_conf = {
            'threshold': TYPE_HIGH_CONF_THR,
            'known': cov04.n_known_with_high, 'nic': cov04.n_nic_with_high, 'nnic': cov04.n_nnic_with_high,
            'total_known': cov04.n_known, 'total_nic': cov04.n_nic, 'total_nnic': cov04.n_nnic,
        }

    # ── novel-GO Venn (existing-annotation space vs PRISM-predicted space, brain_672 전용) ──
    novel_go_breakdown = None
    try:
        novel_go_breakdown = _novel_go_venn(tissue, threshold)
    except Exception as e:  # noqa: BLE001
        novel_go_breakdown = {'error': str(e)}

    # ── DTU 요약 (통계만) ──────────────────────────────────────
    dtu_summary = None
    if d['dtu'] is not None:
        dd = d['dtu']
        sig = dd[dd['pvalue'] < 0.05] if 'pvalue' in dd else dd
        dtu_summary = {'n_events': int(len(dd)), 'n_sig': int(len(sig)),
                       'n_genes': int(dd['gene_id'].nunique()) if 'gene_id' in dd else None,
                       'conditions': sorted(dd['condition'].astype(str).unique().tolist())
                                     if 'condition' in dd else []}

    return {
        'tissue': tissue, 'threshold': threshold,
        'n_isoforms': int(N), 'n_go': int(G),
        'n_genes': int(len(np.unique(genes))) if genes is not None else None,
        'n_highconf': n_highconf, 'highconf_pct': round(100 * n_highconf / N, 1),
        'median_max_score': round(float(np.median(max_per_iso)), 3),
        'mean_n_high': round(float(n_high_per_iso.mean()), 2),
        'hist': {'counts': hist.tolist(), 'edges': [round(e, 3) for e in edges.tolist()]},
        'types': types_breakdown,
        'type_high_conf': type_high_conf,
        'landscape': landscape,
        'novel_go_breakdown': novel_go_breakdown,
        'dtu': dtu_summary,
    }


# ── GO-Score space UMAP (brain_672 전용 — 사전계산 20K 샘플 좌표 재사용) ──────
# streamlit 03_patterns.py 는 매 세션 UMAP fit(느림, umap-learn/t-SNE 폴백) 이지만,
# Flask 는 고정 universe(brain_672) 이므로 target_hub 가 이미 쓰는 사전계산 좌표를
# 그대로 재사용 — fit 없이 즉시 응답.
#
# 두 embedding space 를 사용자가 고를 수 있다(전환형, color-by 와 별개 축):
#   'go_score' (기본) — 672-dim MLP 예측 확률 벡터 위 UMAP. "기능적으로 비슷한 아이소폼".
#                        streamlit 03_patterns.py 원본과 동일 공간(탭 이름의 유래).
#   'axis8'           — Individual Analysis 의 8-axis(L30, ESM-2 30-layer PCA 압축) 위 UMAP.
#                        "시퀀스 표현 자체가 비슷한 아이소폼" — 예측 head 이전 공간.
# 두 UMAP 모두 같은 20K sample_idx 위에서 계산되어 있어 1:1 비교 가능
# (precompute/build_umap_axis8.py, README 참조 — go_score 쪽은 원본 target_hub 산출물).
_UMAP_SAMPLE_PATH = DEMO / 'umap_sample_idx.npy'
_UMAP_COORDS_PATHS = {
    'go_score': DEMO / 'umap_coords.npy',
    'axis8': DEMO / 'umap_coords_axis8.npy',
}
UMAP_SPACES = tuple(_UMAP_COORDS_PATHS.keys())

# 클러스터 수 — streamlit 03_patterns.py 의 실제 공식 min(8, max(3, len(go)//3)) 그대로 사용
# (brain_672 처럼 큰 GO 패널에서는 8 로 클램프됨). 이전엔 "top-15 GO" 관례에 맞춰 15 로
# 고정했었으나, 실측 결과 k 가 커질수록 라벨 중복(작은 파편 클러스터가 배경성 GO term으로
# 수렴)이 심해져 streamlit 이 애초에 8로 캡을 둔 이유였음을 확인 — 아래 [[label enrichment]]
# 주석 참조.
N_UMAP_CLUSTERS = 8

# 2단계(module→GO term) drill-down을 열어줄 최소 클러스터 크기 — 너무 작은 파편 클러스터는
# 하위 breakdown이 통계적으로 무의미하므로, 이 이상인 큰 클러스터에만 노출.
SUBCLUSTER_MIN_COUNT = 500

# drill-down에서 개별 색으로 보여줄 상위 subterm 개수 — 5였을 때는 "이 모듈에 속한 이소폼인데
# 자기 진짜 최고점수 GO term이 top-5 밖이면 어떻게 되나"라는 질문에 답이 좋지 않았다: 이전 구현은
# 멤버의 진짜 전역 argmax가 아니라 "5개 subterm 컬럼 중에서만" argmax를 다시 계산해 강제로
# 5개 중 하나에 배정했다(포인트가 사라지진 않지만, 자기 진짜 1등 GO가 아닌 걸로 오분류됨).
# 672-term panel 특성상(사용자 지적) 30으로 늘리고, 그래도 안 걸리는 멤버는 별도 'etc GO'
# 버킷(모듈 소속은 맞지만 top-30 밖)으로 명시 — "other clusters"(모듈 자체가 다름)와는 다른 개념.
N_SUBTERMS = 30

# count만으로는 부족하다 — mode-collapse된 클러스터(멤버 전부가 byte-identical한 GO score
# 벡터를 가짐, [[finding-brain672-goscore-prediction-collapse-15pct]])는 count>=500을 통과해도
# top-5 subterm 배정이 100%/0%/0%/0%/0%로 완전히 퇴화해 "가짜 다양성"으로 보인다(devils-advocate
# C1, 2026-07-28). Shannon entropy(bits)로 실제 breakdown이 있는지 확인 — 0.5 bits는 대략
# "90/10 두 항목 분할"보다 더 치우친 경우를 배제하는 낮은 문턱(완전 동질=0, 5항목 균등=log2(5)≈2.32).
SUBTERM_ENTROPY_MIN_BITS = 0.5


@lru_cache(maxsize=len(UMAP_SPACES))
def _umap_base(space: str = 'go_score'):
    coords_path = _UMAP_COORDS_PATHS.get(space)
    if coords_path is None or not (coords_path.exists() and _UMAP_SAMPLE_PATH.exists()):
        return None
    coords = np.load(coords_path)
    sample_idx = np.load(_UMAP_SAMPLE_PATH)
    return coords, sample_idx


@lru_cache(maxsize=len(UMAP_SPACES))
def _umap_clusters(space: str = 'go_score'):
    """precomputed UMAP 2D 좌표(embedding space 는 `space` 로 선택) 위에서 KMeans(k=8) →
    클러스터별 대표 GO 라벨.

    클러스터링 자체는 `space`(go_score 672-dim 또는 axis8 8-dim)에서 하지만, 라벨은 항상
    GO score matrix 기준 differential enrichment — "이 아이소폼 묶음이 시퀀스 표현으로
    뭉쳤을 때 어떤 기능이 우세한가"를 axis8 모드에서도 그대로 답할 수 있게 한다.

    [[label enrichment]] 라벨은 클러스터 평균 GO score의 단순 argmax가 아니라
    (클러스터 평균 − 전체 population 평균)의 argmax — 즉 "이 클러스터에서만 상대적으로
    튀는" GO term을 고른다. 실측 검증: 단순 argmax 방식(streamlit umap_plot.py 원본 그대로)은
    brain_672(672-dim, 상호 상관된 GO term多)에서 "reproductive process"처럼 그냥 전역
    평균이 높은 배경성 term을 15개 중 8개 클러스터가 동일하게 고르는 현상을 보였다
    (population 전체 대비 그 클러스터만의 특이성을 전혀 반영 못 함 — differential-expression
    marker 선택에서 raw mean 대신 fold-change/z-score를 쓰는 것과 동일한 이유로 필요한 보정).

    [[z-score 정규화, 2026-07-28]] 위 보정만으로는 부족했다 — raw (mean_diff)의 argmax는
    여전히 그 GO term 컬럼 자체의 전역 분산(변동폭) 크기에 좌우된다. differential-expression
    marker 선택의 표준 관행대로 (mean_diff / 전역 std)로 z-score화하여 "이 컬럼 자체의 변동폭
    대비 이 클러스터가 얼마나 튀는가"로 바꾼다 — global_std가 0에 가까운(거의 상수인) GO term이
    분모 폭주로 아무 데서나 이기는 것을 막기 위해 eps로 하한을 둔다.

    [[non-coding 제외, 2026-07-29]] z-score 정규화 후에도 GO:0006629(lipid metabolic process)가
    8개 클러스터 중 5개를 독점하는 문제가 남아있었다 — 원인은 라벨링이 아니라 데이터: 이
    5개 클러스터의 멤버 전부가 non-coding transcript였고, ESM-2 임베딩이 설계상 zero-vector라
    모든 GO term에서 sigmoid(bias)만 동일하게 출력한다(실제 신호 없음 —
    [[finding-brain672-noncoding-zero-embedding-flag-needed]]). KMeans가 이 단일 동질 집단을
    2D 좌표 노이즈만으로 5조각 임의 분할한 것뿐이었다. **이제 KMeans 적합·enrichment 통계
    (global_mean/std) 전부 coding 이소폼만으로 계산**하고, non-coding은 별도의 고정
    pseudo-cluster(`id=-1`, 항상 non-drillable)로 좌표만 유지해 프런트에서 회색으로 구분
    표시한다 — "8개 기능 클러스터" 파티션 자체가 이제 실제 기능 다양성만 반영한다.
    좌표·score matrix 모두 고정(brain_672)이므로 (space, k) 조합당 1회만 계산해 캐시."""
    base = _umap_base(space)
    if base is None:
        return None
    coords, sample_idx = base
    d = load_tissue('brain_672')
    sm = np.asarray(d['sm'])[sample_idx]
    go_ids, gn = d['go_ids'], d['go_names']
    coding_mask_full = d['coding_mask']
    coding_mask = coding_mask_full[sample_idx] if coding_mask_full is not None else np.ones(len(sample_idx), dtype=bool)

    sm_coding = sm[coding_mask]
    global_mean = sm_coding.mean(axis=0) if len(sm_coding) else None
    global_std = sm_coding.std(axis=0) if len(sm_coding) else None
    _EPS = 1e-3  # brain_672 GO term std 실측 범위 0.049~0.112(전부 이보다 훨씬 큼, 2026-07-28
                 # 확인) — 이 floor는 "정말 거의 상수인" 컬럼에서만 작동하는 안전장치, 정상
                 # 범위 GO term엔 영향 없음.
    global_std_floored = np.maximum(global_std, _EPS) if global_std is not None else None

    from sklearn.cluster import KMeans
    labels = np.full(len(coords), -1, dtype=int)  # -1 = non-coding(클러스터링 제외), 기본값
    if coding_mask.sum() >= N_UMAP_CLUSTERS:
        km = KMeans(n_clusters=N_UMAP_CLUSTERS, random_state=42, n_init='auto')
        labels[coding_mask] = km.fit_predict(coords[coding_mask])

    clusters = []
    for c in range(N_UMAP_CLUSTERS):
        mask = labels == c
        count = int(mask.sum())
        cx = float(coords[mask, 0].mean()) if count else 0.0
        cy = float(coords[mask, 1].mean()) if count else 0.0
        go_id, label, subterms = None, f'Cluster {c + 1}', []
        subterm_entropy, drillable = 0.0, False
        if count and go_ids:
            z = (sm[mask].mean(axis=0) - global_mean) / global_std_floored
            # 상위 N_SUBTERMS개(중복 없는 GO term) — rank1이 곧 클러스터 대표 라벨(=`go_id`/`label`,
            # max_go 색칠에서 쓰는 값), 나머지는 2단계 module→GO-term drill-down용
            # (count>=500인 큰 클러스터에서만 프런트가 노출).
            topN_idx = np.argsort(z)[::-1][:N_SUBTERMS]
            go_id = go_ids[int(topN_idx[0])]
            label = gn.get(go_id, go_id)
            subterms = [{'go_id': go_ids[int(i)], 'label': gn.get(go_ids[int(i)], go_ids[int(i)]),
                         'z': round(float(z[int(i)]), 3)} for i in topN_idx]
            if count >= SUBCLUSTER_MIN_COUNT:
                # 실제 breakdown이 있는지 확인 — 멤버 자신의 *진짜* 전역 argmax(672개 전체 컬럼
                # 기준, 이 클러스터 최고점수 GO term)를 top-N 집합과 대조한다(N개 컬럼끼리만
                # 다시 argmax하던 이전 방식은 진짜 1등이 top-N 밖이어도 강제로 N개 중 하나에
                # 배정해버리는 오분류가 있었다 — 아래 'etc' 버킷으로 분리해 해결).
                topN_set = set(int(i) for i in topN_idx)
                true_argmax = sm[mask].argmax(axis=1)
                cat = np.array([(int(a) if int(a) in topN_set else -1) for a in true_argmax])
                p = np.array([(cat == int(i)).sum() for i in topN_idx] + [(cat == -1).sum()],
                             dtype=np.float64)
                p /= p.sum()
                nz = p[p > 0]
                subterm_entropy = float(-(nz * np.log2(nz)).sum())
                drillable = subterm_entropy >= SUBTERM_ENTROPY_MIN_BITS
        clusters.append({'id': c, 'x': round(cx, 3), 'y': round(cy, 3),
                          'go_id': go_id, 'label': label, 'count': count, 'subterms': subterms,
                          'subterm_entropy': round(subterm_entropy, 3), 'drillable': drillable})

    # non-coding pseudo-cluster — 항상 id=-1, 항상 non-drillable(기능 breakdown이 아니라 데이터
    # 결측이므로). count=0이면(마스크 없는 tissue 등) 프런트가 알아서 숨김.
    nc_mask = labels == -1
    nc_count = int(nc_mask.sum())
    clusters.append({
        'id': -1, 'x': round(float(coords[nc_mask, 0].mean()), 3) if nc_count else 0.0,
        'y': round(float(coords[nc_mask, 1].mean()), 3) if nc_count else 0.0,
        'go_id': None, 'label': 'non-coding (no ESM-2 signal)', 'count': nc_count,
        'subterms': [], 'subterm_entropy': 0.0, 'drillable': False,
    })
    return {'labels': labels, 'clusters': clusters}


def umap_points(color_by: str = 'max_go', space: str = 'go_score',
                 threshold: float = TYPE_HIGH_CONF_THR, cluster_id: int | None = None) -> dict:
    """20K 샘플(브레인 63,994 중) 사전계산 UMAP 좌표 + 색칠 옵션.

    space:    'go_score'(기본 — 672-dim 예측 확률 UMAP) | 'axis8'(8-axis L30 ESM-2 표현 UMAP)
    color_by: 'max_go'(기본 — GO 클러스터) | 'isoform_type'(Known/NIC/NNIC) |
              'scenario'(1-4) | 'max_score'(연속값) | 'max_go_sub'(cluster_id 필수 — 1단계
              클러스터 하나를 골라 그 안의 top-5 subterm으로 2단계 drill-down)

    'max_go' 는 point 별 원본 max_go(최대 수백 개의 고유값)를 그대로 이산 색상으로 쓰면
    범례가 무의미해지므로, 대신 그 점이 속한 KMeans 클러스터(대표 GO 로 명명됨)로 칠한다 —
    "GO 클러스터 라벨링" 자체가 이 모드의 정의. space 를 바꿔도 라벨링 방식은 동일(위 참조).
    """
    if space not in UMAP_SPACES:
        space = 'go_score'
    base = _umap_base(space)
    if base is None:
        return {'error': f'no precomputed umap coords for space={space!r}'}
    coords, sample_idx = base
    d = load_tissue('brain_672')
    ids = np.asarray(d['ids'], dtype=str)[sample_idx]
    cl = _umap_clusters(space)
    # max_score is always included (not just when color_by=='max_score') so the frontend can
    # render the continuous-gradient view as an overlay on top of ANY categorical fetch (module/
    # subterm membership) without a second request — needed for the score-gradient "module
    # outline" mode, which needs both the raw score AND the cluster/subterm membership at once.
    sm_sample = np.asarray(d['sm'])[sample_idx]
    out = {
        'x': coords[:, 0].round(3).tolist(), 'y': coords[:, 1].round(3).tolist(),
        'isoform_id': ids.tolist(), 'color_by': color_by, 'space': space,
        'cluster_id': (cl['labels'].tolist() if cl else []),
        'clusters': (cl['clusters'] if cl else []),
        'max_score': sm_sample.max(axis=1).round(3).tolist(),
    }

    if color_by == 'max_go_sub':
        if cl is None:
            return {'error': 'clusters unavailable'}
        by_id = {c['id']: c for c in cl['clusters']}
        parent = by_id.get(cluster_id) if cluster_id is not None else None
        if parent is None:
            return {'error': f'invalid cluster_id {cluster_id!r}'}
        if parent['count'] < SUBCLUSTER_MIN_COUNT:
            return {'error': f"cluster #{cluster_id+1} too small for drill-down (n={parent['count']} < {SUBCLUSTER_MIN_COUNT})"}
        if not parent['drillable']:
            # count>=500이어도 멤버 전부가 (거의) 동일한 GO score 벡터인 mode-collapsed 클러스터는
            # subterm 배정이 100%/0%/0%/0%/0%로 퇴화 — "가짜 다양성"을 보여주지 않도록 거부.
            # [[finding-brain672-goscore-prediction-collapse-15pct]] 참조(devils-advocate C1).
            return {'error': f"cluster #{cluster_id+1} has no real subcategory structure "
                              f"(entropy={parent['subterm_entropy']:.2f} bits < {SUBTERM_ENTROPY_MIN_BITS}) — "
                              f"its members share (near-)identical GO score predictions, likely a "
                              f"prediction-collapse artifact rather than genuine functional diversity"}
        subterms = parent['subterms']
        go_ids = d['go_ids']
        sub_go_set = {st['go_id'] for st in subterms}
        sub_go_by_idx = {go_ids.index(st['go_id']): st['go_id'] for st in subterms}
        sm_full = np.asarray(d['sm'])[sample_idx]                     # (n, G) — 전체 패널
        member_mask = np.asarray(cl['labels']) == cluster_id
        # 멤버 자신의 *진짜* 전역 argmax(672개 전체 컬럼) — top-N 밖이면 'etc_go'로 분리한다
        # (top-N 컬럼끼리만 다시 argmax하던 이전 방식은 진짜 1등이 top-N 밖이어도 강제로 N개
        # 중 하나에 잘못 배정했다 — 사용자 지적으로 발견/수정).
        true_argmax = sm_full.argmax(axis=1)
        color = np.full(len(ids), 'other', dtype=object)
        for i in np.where(member_mask)[0]:
            color[i] = sub_go_by_idx.get(int(true_argmax[i]), 'etc_go')
        out['color'] = color.tolist()
        counts = {st['go_id']: int((color == st['go_id']).sum()) for st in subterms}
        out['color_meta'] = {st['go_id']: {'label': st['label'], 'count': counts[st['go_id']], 'z': st['z']}
                              for st in subterms}
        n_etc = int((color == 'etc_go').sum())
        out['color_meta']['etc_go'] = {
            'label': f'etc GO (this module, rank>{N_SUBTERMS})', 'count': n_etc}
        # 'other'는 이 클러스터 바깥의 나머지 7개 클러스터 전부를 뭉친 이질적 집합이지, 하나의
        # 카테고리가 아니다 — 라벨에 그 사실을 명시(devils-advocate: 무경고 export 시 오독 위험).
        out['color_meta']['other'] = {'label': 'other clusters (mixed, not one category)',
                                       'count': int((~member_mask).sum())}
        out['parent_cluster'] = {'id': parent['id'], 'label': parent['label'], 'count': parent['count'],
                                  'subterm_entropy': parent['subterm_entropy'], 'n_subterms': len(subterms)}
        return out

    if color_by == 'scenario':
        clf = _classify_cached('brain_672', threshold)
        scen_by_id = dict(zip(clf['isoform_id'], clf['scenario']))
        out['color'] = [int(scen_by_id.get(i, 0)) for i in ids]
        out['color_meta'] = {str(k): {'label': v[0], 'color': v[2]} for k, v in SCENARIO_META.items()}
    elif color_by == 'max_score':
        out['color'] = out['max_score']   # already computed above (now unconditional)
    elif color_by == 'isoform_type':
        types = np.asarray(d['types'], dtype=str)[sample_idx] if d['types'] is not None else np.full(len(ids), 'known')
        out['color'] = types.tolist()
        # Okabe-Ito, must match mydata.js ISOTYPE_PAL exactly (type-pie + this UMAP legend agree).
        out['color_meta'] = {'known': '#0072B2', 'nic': '#009E73', 'nnic': '#D55E00'}
    else:  # max_go (default) — colour = cluster id, legend label = cluster's dominant GO name
        out['color'] = out['cluster_id']
        out['color_meta'] = {str(c['id']): {'label': c['label'], 'count': c['count']}
                              for c in out['clusters']}
    return out


def umap_isoform_list(color_by: str, values: list[str], space: str = 'go_score',
                       threshold: float = TYPE_HIGH_CONF_THR, cluster_id: int | None = None) -> dict:
    """UMAP 필터 선택에 해당하는 아이소폼 목록 — "Selected isoform list" 버튼을 눌렀을 때, UMAP
    범례에서 현재 켜져 있는(legend click 으로 껐다 켰다 하는 "color on") 카테고리들 전부의
    합집합(OR)을 돌려준다. max_go/isoform_type/scenario 는 여러 값을 동시에 받을 수 있고
    (예: 켜진 클러스터 3개), max_score 는 연속값이라 단일 임계값(values[0]) 만 사용.
    space 는 color_by='max_go' 일 때만 의미(어느 embedding 의 클러스터인지) — 나머지 color_by
    는 embedding space 와 무관(같은 20K 샘플의 카테고리 값이므로).
    항상 차트에 그려진 것과 같은 20K 사전계산 샘플 범위로 스코프(화면에 보이는 것 = 목록)."""
    if space not in UMAP_SPACES:
        space = 'go_score'
    base = _umap_base(space)
    if base is None:
        return {'error': 'no precomputed umap coords'}
    if not values:
        return {'error': 'no selection'}
    coords, sample_idx = base
    d = load_tissue('brain_672')
    clf = _classify_cached('brain_672', threshold)
    sub = clf.iloc[sample_idx].reset_index(drop=True).copy()
    gn = d['go_names']
    types = (np.asarray(d['types'], dtype=str)[sample_idx] if d['types'] is not None
             else np.full(len(sub), 'known'))
    sub['isoform_type'] = types
    sub['max_go_name'] = sub['max_go'].map(lambda x: gn.get(x, x))

    if color_by == 'max_go_sub':
        cl = _umap_clusters(space)
        if cl is None:
            return {'error': 'clusters unavailable'}
        by_id = {c['id']: c for c in cl['clusters']}
        parent = by_id.get(cluster_id) if cluster_id is not None else None
        if parent is None:
            return {'error': f'invalid cluster_id {cluster_id!r}'}
        if not parent['drillable']:
            return {'error': f"cluster #{cluster_id+1} has no real subcategory structure "
                              f"(entropy={parent['subterm_entropy']:.2f} bits < {SUBTERM_ENTROPY_MIN_BITS}) — "
                              f"its members share (near-)identical GO score predictions"}
        subterms = parent['subterms']
        valid_goids = {st['go_id'] for st in subterms}
        goids = [v for v in values if v in valid_goids]
        want_etc = 'etc_go' in values
        want_other = 'other' in values
        if not goids and not want_etc and not want_other:
            return {'error': 'invalid subterm selection'}
        go_ids = d['go_ids']
        sub_go_by_idx = {go_ids.index(st['go_id']): st['go_id'] for st in subterms}
        sm_full = np.asarray(d['sm'])[sample_idx]
        member_mask = np.asarray(cl['labels']) == cluster_id
        # true argmax across the full panel (same fix as umap_points: top-N-column-only argmax
        # mis-sorts a member's real #1 GO term into one of the N when it isn't actually one of them).
        true_argmax = sm_full.argmax(axis=1)
        winner_goid = np.full(len(sub), '', dtype=object)
        for i in np.where(member_mask)[0]:
            winner_goid[i] = sub_go_by_idx.get(int(true_argmax[i]), 'etc_go')
        mask = member_mask & np.isin(winner_goid, goids) if goids else np.zeros(len(sub), dtype=bool)
        by_goid = {st['go_id']: st['label'] for st in subterms}
        label_parts = [f"#{cluster_id + 1} {parent['label']} → " + ' · '.join(by_goid[g] for g in goids)] if goids else []
        if want_etc:
            mask = mask | (member_mask & (winner_goid == 'etc_go'))
            label_parts.append(f'etc GO (this module, rank>{N_SUBTERMS})')
        if want_other:
            # 'other'는 이 클러스터 바깥 7개 클러스터를 뭉친 이질적 집합 — 선택은 허용하되
            # 라벨에 그 사실을 명시해 "하나의 카테고리"로 오독하지 않도록 한다.
            mask = mask | ~member_mask
            label_parts.append('other clusters (mixed, not one category)')
        label = ' + '.join(label_parts)
    elif color_by == 'max_go':
        cl = _umap_clusters(space)
        if cl is None:
            return {'error': 'clusters unavailable'}
        try:
            # -1 = non-coding pseudo-cluster(항상 존재), 0..N-1 = 실제 KMeans 클러스터
            cids = [int(v) for v in values if int(v) == -1 or 0 <= int(v) < N_UMAP_CLUSTERS]
        except (TypeError, ValueError):
            return {'error': 'invalid cluster id'}
        if not cids:
            return {'error': 'invalid cluster id'}
        mask = np.isin(cl['labels'], cids)
        by_id = {c['id']: c for c in cl['clusters']}
        label = ' · '.join((by_id[cid]['label'] if cid == -1 else f"#{cid + 1} {by_id[cid]['label']}")
                            for cid in cids if cid in by_id)
    elif color_by == 'isoform_type':
        vals = [v for v in values if v in ('known', 'nic', 'nnic')]
        if not vals:
            return {'error': 'invalid isoform_type'}
        mask = sub['isoform_type'].isin(vals).values
        label = 'isoform_type = ' + ', '.join(vals)
    elif color_by == 'scenario':
        vals = [v for v in values if v in ('1', '2', '3', '4')]
        if not vals:
            return {'error': 'invalid scenario'}
        mask = sub['scenario'].astype(str).isin(vals).values
        label = 'Scenario ' + ', '.join(f"S{v}" for v in vals)
    elif color_by == 'max_score':
        try:
            thr_v = float(values[0])
        except (TypeError, ValueError, IndexError):
            return {'error': 'invalid threshold'}
        mask = sub['max_score'].values >= thr_v
        label = f'max_score ≥ {thr_v:.2f}'
    else:
        return {'error': f'unknown color_by {color_by!r}'}

    rows = (sub.loc[mask, ['isoform_id', 'gene_id', 'max_go_name', 'max_score', 'isoform_type', 'scenario']]
            .sort_values('max_score', ascending=False))
    return {
        'label': label, 'n': int(len(rows)),
        'isoforms': rows.round({'max_score': 3}).to_dict('records'),
    }


# ── Gene Quick Report candidates (brain_672 전용 — Individual tab search aid) ─
# 4 axes, revised from the original streamlit target_hub port after an empirical check
# found the original axes NEVER overlapped with BISECT's own 81 externally-validated
# genes (domain/AlphaFold/PPI/conservation cross-checked) — see chat discussion.
#   S1 (Functional Switch)   — kept: DTU+ / novel-GO+, PRISM's own headline scenario.
#   S3 (Novel Constitutive)  — kept: DTU− / novel-GO+, a genuinely distinct lens.
#   BISECT Validated         — NEW, replaces the old unfiltered "DTU top" axis: cross-links
#     to bisect_cases.py tier A-DR/A-BP (statistically + multi-evidence corroborated), so at
#     least one axis surfaces cases validated beyond a single score-matrix ranking.
#   Diversity                — kept, but now gated: the top isoform must clear the app's
#     high-confidence threshold, so a large score_range can't be driven by two low-confidence
#     (i.e. noisy) calls.
@lru_cache(maxsize=2)
def quick_report_candidates(tissue: str = 'brain_672', threshold: float = TYPE_HIGH_CONF_THR) -> dict:
    d = load_tissue(tissue)
    clf = _classify_cached(tissue, threshold)
    gn = d['go_names']

    def _top(df, n=8):
        return [{'gene': r['gene_id'], 'isoform_id': r['isoform_id'],
                  'max_score': round(float(r['max_score']), 3),
                  'max_go': r['max_go'], 'max_go_name': gn.get(r['max_go'], r['max_go'])}
                for _, r in df.head(n).iterrows()]

    s1 = clf[clf['scenario'] == 1].sort_values('max_score', ascending=False)
    s3 = clf[clf['scenario'] == 3].sort_values('max_score', ascending=False)

    div_cands = []
    if 'gene_id' in clf.columns:
        def _div_stats(g):
            if len(g) < 2:
                return None
            top_i = g['max_score'].idxmax()
            top_score = float(g.loc[top_i, 'max_score'])
            if top_score < TYPE_HIGH_CONF_THR:   # confidence floor — anchor the spread on a real call
                return None
            return pd.Series({
                'gene_id': g.name, 'isoform_id': g.loc[top_i, 'isoform_id'],
                'n_isoforms': len(g),
                'score_range': round(float(g['max_score'].max() - g['max_score'].min()), 4),
                'max_score': top_score, 'max_go': g.loc[top_i, 'max_go'],
            })
        dv = (clf.groupby('gene_id', sort=False).apply(_div_stats, include_groups=False)
              .dropna().sort_values('score_range', ascending=False).head(8))
        div_cands = [{'gene': r['gene_id'], 'isoform_id': r['isoform_id'],
                      'max_score': round(float(r['max_score']), 3),
                      'max_go': r['max_go'], 'max_go_name': gn.get(r['max_go'], r['max_go']),
                      'n_isoforms': int(r['n_isoforms']), 'score_range': float(r['score_range'])}
                     for _, r in dv.iterrows()]

    bisect_cands = []
    if tissue == 'brain_672':
        from prism_app_flask.data_layer import bisect_cases as bc
        tier_rank = {'A-DR': 0, 'A-BP': 1}
        cases = sorted(
            (c for c in bc.all_cases() if c.get('bisect_tier') in tier_rank),
            key=lambda c: (tier_rank[c['bisect_tier']], -abs(float(c.get('delta') or 0))))
        seen_genes = set()
        for c in cases:
            g = c.get('gene')
            if g in seen_genes:
                continue
            seen_genes.add(g)
            bisect_cands.append({
                'gene': g, 'isoform_id': c.get('ad_transcript_id') or c.get('ct_transcript_id') or '',
                'max_go_name': c.get('prism_ad_max_go') or c.get('prism_ct_max_go') or '—',
                'bisect_tier': c['bisect_tier'], 'delta': round(float(c.get('delta') or 0), 3),
                'mechanism': c.get('m16_mechanism') or '—',
            })
            if len(bisect_cands) >= 8:
                break

    return {
        's1': _top(s1), 's3': _top(s3), 'bisect': bisect_cands, 'diversity': div_cands,
        'meta': {
            's1': {'label': 'S1 · Functional Switch', 'color': SCENARIO_META[1][2],
                   'desc': 'DTU+ / novel GO+ — highest-priority candidates'},
            's3': {'label': 'S3 · Novel Constitutive', 'color': SCENARIO_META[3][2],
                   'desc': 'DTU− / novel GO+ — constitutively expressed novel function'},
            'bisect': {'label': 'BISECT Validated', 'color': '#4FB4D0',
                       'desc': 'Tier A-DR/A-BP — cross-checked by domain, AlphaFold, PPI & conservation evidence'},
            'diversity': {'label': 'High Divergence', 'color': '#C99A52',
                          'desc': f'largest within-gene max-score spread (top isoform score>{TYPE_HIGH_CONF_THR})'},
        },
    }


# ── Gene Quick Report detail (per-card content for the carousel panels) ──────
# 개별분석 flagship 의 gene_profile(heatmap+isoform 목록)을 재사용하고,
# module_landscape 의 type/module 라벨 + DTU 유의 이벤트 수만 얹는다 — 중복 계산 없음.
def quick_case_gene_detail(gene: str, tissue: str = 'brain_672') -> dict:
    from prism_app_flask.data_layer import isoform_profile as ip
    from prism_app_flask.data_layer import module_landscape as ml

    prof = ip.gene_profile(gene)
    if 'error' in prof:
        return prof

    refs = ml._load_refs()
    mod_lookup = {}
    if refs is not None:
        sub = refs['df_iso'][refs['df_iso']['gene'] == gene]
        mod_lookup = {r['isoform_id']: {'type': r['type'], 'module_label': r['module_label']}
                      for _, r in sub.iterrows()}

    n_novel = 0
    n_high_conf = 0
    for iso in prof['isoforms']:
        m = mod_lookup.get(iso['isoform_id'], {})
        iso['type'] = m.get('type', '—')
        iso['module_label'] = (m.get('module_label') or '—').split('/')[0].strip()[:30]
        if str(iso['type']).lower() in ('nic', 'nnic'):
            n_novel += 1
        if iso['max_go_score'] > TYPE_HIGH_CONF_THR:
            n_high_conf += 1

    d = load_tissue(tissue)
    n_dtu_sig = 0
    if d['dtu'] is not None and 'gene_id' in d['dtu'].columns:
        gd = d['dtu'][d['dtu']['gene_id'] == gene]
        n_dtu_sig = int((gd['pvalue'] < 0.05).sum()) if 'pvalue' in gd.columns else int(len(gd))

    prof['n_high_conf'] = n_high_conf
    prof['n_novel'] = n_novel
    prof['n_dtu_sig'] = n_dtu_sig
    prof['high_conf_threshold'] = TYPE_HIGH_CONF_THR
    return prof
