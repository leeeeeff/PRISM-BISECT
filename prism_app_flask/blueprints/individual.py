"""개별 유전자·아이소폼 분석 (flagship).

라우트:
    /gene                     검색 랜딩(입력 폼)
    /gene/<query>             유전자 심볼 또는 아이소폼 ID → 결과 페이지(서버 렌더 + JS 차트)
    /api/resolve?q=           검색어 종류 판정(gene/isoform/none)
    /api/gene/<gene>          유전자 프로파일 JSON
    /api/isoform/<iso_id>     아이소폼 프로파일 JSON (패널 A/C/D)
    /api/suggest?q=           자동완성(유전자 심볼 prefix)
"""
from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from prism_app_flask.data_layer import isoform_profile as ip
from prism_app_flask.data_layer import loaders
from prism_app_flask.data_layer import dataset_summary as ds

bp = Blueprint('individual', __name__)


@bp.route('/gene')
def search_landing():
    return render_template('individual.html', query=None, resolved=None)


@bp.route('/gene/<path:query>')
def result(query: str):
    resolved = ip.resolve_query(query)
    return render_template('individual.html', query=query, resolved=resolved)


# ── JSON API ────────────────────────────────────────────────────────────────
@bp.route('/api/resolve')
def api_resolve():
    q = request.args.get('q', '')
    return jsonify(ip.resolve_query(q))


@bp.route('/api/gene/<path:gene>')
def api_gene(gene: str):
    prof = ip.gene_profile(gene)
    return jsonify(prof), (404 if 'error' in prof else 200)


@bp.route('/api/isoform/<path:iso_id>')
def api_isoform(iso_id: str):
    prof = ip.isoform_profile(iso_id)
    return jsonify(prof), (404 if 'error' in prof else 200)


@bp.route('/api/go_fisher/<path:go_id>')
def api_go_fisher(go_id: str):
    d = ip.go_fisher_curve(go_id)
    return jsonify(d), (404 if 'error' in d else 200)


@bp.route('/api/domains/<path:iso_id>')
def api_domains(iso_id: str):
    d = ip.domain_architecture(iso_id)
    return jsonify(d), (404 if 'error' in d else 200)


@bp.route('/api/iso_traj/<path:iso_id>')
def api_iso_traj(iso_id: str):
    d = ip.iso_trajectory(iso_id)
    return jsonify(d), (404 if 'error' in d else 200)


@bp.route('/api/go_cohort/<path:go_id>')
def api_go_cohort(go_id: str):
    exclude = request.args.get('exclude', '')
    try:
        n = min(int(request.args.get('n', 60)), 120)
    except ValueError:
        n = 60
    d = ip.go_cohort_trajectories(go_id, exclude, n)
    return jsonify(d), (404 if 'error' in d else 200)


@bp.route('/api/suggest')
def api_suggest():
    q = request.args.get('q', '').strip().upper()
    if len(q) < 1:
        return jsonify([])
    hits = [g for g in loaders.gene_symbols() if g.upper().startswith(q)][:15]
    return jsonify(hits)


@bp.route('/api/quick_cases')
def api_quick_cases():
    """검색 보조용 'Gene Quick Report' 후보 4개 분석축 (brain·672 전용)."""
    return jsonify(ds.quick_report_candidates('brain_672'))


@bp.route('/api/quick_case_detail/<path:gene>')
def api_quick_case_detail(gene: str):
    """Quick Report 캐러셀 카드 1개 분량 상세(heatmap+isoform 표+지표)."""
    d = ds.quick_case_gene_detail(gene)
    return jsonify(d), (404 if 'error' in d else 200)
