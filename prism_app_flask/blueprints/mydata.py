"""내 데이터 분석 — tissue 프리셋 → 핵심 지표 한 페이지 요약."""
import csv
import io

from flask import Blueprint, Response, jsonify, render_template, request

from prism_app_flask.data_layer import dataset_summary as ds
from prism_app_flask.data_layer import module_landscape as ml

bp = Blueprint('mydata', __name__)

TISSUE_OPTIONS = [
    {'key': 'brain_672', 'label': 'Brain · 672 GO',
     'desc': '63,994 transcripts (genome-wide GENCODE) · brain 8 cell-type DTU'},
    {'key': 'muscle', 'label': 'Muscle · 18 GO', 'desc': '36,748 transcripts · muscle preset'},
]

# Clarify that the universe is a transcript-annotation set, not a tissue-expressed subset.
UNIVERSE_NOTE = ('Note: "brain·672" refers to evaluation under brain DTU conditions; the isoform set '
                 'itself is the genome-wide GENCODE transcriptome (63,994) — not a tissue-restricted subset.')


@bp.route('/mydata')
def home():
    tissue = request.args.get('tissue', 'brain_672')
    return render_template('mydata.html', tissues=TISSUE_OPTIONS, active=tissue,
                           universe_note=UNIVERSE_NOTE)


@bp.route('/api/summary/<tissue>')
def api_summary(tissue: str):
    try:
        thr = float(request.args.get('threshold', 0.5))
    except ValueError:
        thr = 0.5
    try:
        return jsonify(ds.summarize(tissue, thr))
    except KeyError:
        return jsonify({'error': f'unknown tissue {tissue!r}'}), 404


# ── GO-Score space explorer (brain_672 전용 참조 그림 3종) ───────────────────
_UMAP_COLOR_BY = ('max_go', 'isoform_type', 'scenario', 'max_score', 'max_go_sub')


def _parse_cluster_id():
    raw = request.args.get('cluster_id')
    if raw is None or raw == '':
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@bp.route('/api/umap')
def api_umap():
    color_by = request.args.get('color_by', 'max_go')
    if color_by not in _UMAP_COLOR_BY:
        color_by = 'max_go'
    space = request.args.get('space', 'go_score')
    if space not in ds.UMAP_SPACES:
        space = 'go_score'
    return jsonify(ds.umap_points(color_by, space, cluster_id=_parse_cluster_id()))


# "Selected isoform list" 버튼용 — UMAP 범례에서 현재 켜져 있는("color on") 카테고리 전부(콤마
# 구분, 클러스터 여러 개 동시 선택 가능)에 대한 아이소폼 목록. format=csv 로 다운로드.
@bp.route('/api/umap/isoforms')
def api_umap_isoforms():
    color_by = request.args.get('color_by', 'max_go')
    if color_by not in _UMAP_COLOR_BY:
        color_by = 'max_go'
    space = request.args.get('space', 'go_score')
    if space not in ds.UMAP_SPACES:
        space = 'go_score'
    values = [v for v in request.args.get('values', '').split(',') if v != '']
    result = ds.umap_isoform_list(color_by, values, space, cluster_id=_parse_cluster_id())
    if 'error' in result:
        return jsonify(result), 404
    if request.args.get('format') == 'csv':
        rows = result['isoforms']
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        fname = f"umap_{color_by}_{'-'.join(values)}".replace(' ', '_').replace('/', '-') + '.csv'
        return Response(buf.getvalue(), mimetype='text/csv',
                         headers={'Content-Disposition': f'attachment; filename="{fname}"'})
    return jsonify(result)


# GO term score distribution, overlaid per gene — dropdown lists every GO term the MLP was
# trained on for this tissue; picking one shows each gene's (isoform>=2) score distribution
# for that column, all overlaid on one plot.
@bp.route('/api/summary/<tissue>/go_terms')
def api_go_terms(tissue: str):
    try:
        return jsonify(ds.go_term_list(tissue))
    except KeyError:
        return jsonify({'error': f'unknown tissue {tissue!r}'}), 404


@bp.route('/api/summary/<tissue>/go_distribution')
def api_go_distribution(tissue: str):
    go_id = request.args.get('go_id', '')
    if not go_id:
        return jsonify({'error': 'go_id required'}), 400
    try:
        result = ds.go_term_distribution(tissue, go_id)
    except KeyError:
        return jsonify({'error': f'unknown tissue {tissue!r}'}), 404
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


# GO term score distribution·by gene 위의 4개 하이라이트 criterion(top-score isoform's gene /
# top-variance gene / single-isoform high-score genes / top-mean genes) — color on/off 토글용.
@bp.route('/api/summary/<tissue>/go_distribution/highlights')
def api_go_distribution_highlights(tissue: str):
    go_id = request.args.get('go_id', '')
    if not go_id:
        return jsonify({'error': 'go_id required'}), 400
    try:
        result = ds.go_term_gene_highlights(tissue, go_id)
    except KeyError:
        return jsonify({'error': f'unknown tissue {tissue!r}'}), 404
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


# Existing-annotation-vs-PRISM Venn — region click → its (gene, GO term) pairs, isoform-attributed.
# Lazy on-click fetch (same pattern as quick_case_detail/domain_architecture) — the aggregate counts
# already come back with /api/summary, this only runs when a region is actually opened.
@bp.route('/api/summary/<tissue>/novel_go_region')
def api_novel_go_region(tissue: str):
    region = request.args.get('region', '')
    if region not in ds.NOVEL_GO_REGIONS:
        return jsonify({'error': f'region must be one of {ds.NOVEL_GO_REGIONS}'}), 400
    try:
        thr = float(request.args.get('threshold', 0.5))
    except ValueError:
        thr = 0.5
    try:
        hiconf = float(request.args.get('hiconf_threshold', 0.7))
    except ValueError:
        hiconf = 0.7
    result = ds.novel_go_region_isoforms(tissue, region, thr, hiconf)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


# /mydata triage ranked-list — supersedes the S1-S4 donut (docs/mydata_triage_design.md §3, §7).
@bp.route('/api/summary/<tissue>/triage_ranked')
def api_triage_ranked(tissue: str):
    try:
        thr = float(request.args.get('threshold', 0.5))
    except ValueError:
        thr = 0.5
    try:
        hiconf = float(request.args.get('hiconf_threshold', 0.7))
    except ValueError:
        hiconf = 0.7
    consequence = request.args.get('consequence', '')
    structural_type = request.args.get('structural_type', '')
    sort = request.args.get('sort', 'consequence')
    if sort not in ds.TRIAGE_SORTS:
        sort = 'consequence'
    try:
        cap = int(request.args.get('cap', 500))
    except ValueError:
        cap = 500
    try:
        result = ds.triage_ranked(tissue, thr, hiconf, consequence, structural_type, sort, cap)
    except KeyError:
        return jsonify({'error': f'unknown tissue {tissue!r}'}), 404
    return jsonify(result)


@bp.route('/api/module_landscape/bubble')
def api_module_bubble():
    return jsonify({'modules': ml.bubble_data(), 'meta': ml.bubble_meta()})


@bp.route('/api/module_landscape/corr')
def api_module_corr():
    d = ml.corr_matrix()
    if d is None:
        return jsonify({'error': 'correlation matrix unavailable'}), 404
    return jsonify(d)
