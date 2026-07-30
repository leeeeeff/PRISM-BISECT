"""BISECT 분석 — 아이소폼 스위치 케이스 탐색기 + tier/mechanism."""
from flask import Blueprint, jsonify, render_template, request

from prism_app_flask.data_layer import bisect_cases as bc

bp = Blueprint('bisect', __name__)


@bp.route('/bisect')
def home():
    return render_template('bisect.html', summary=bc.summary())


@bp.route('/api/bisect/cases')
def api_cases():
    return jsonify({'summary': bc.summary(), 'cases': bc.table()})


@bp.route('/api/bisect/case/<gene>')
def api_case(gene: str):
    d = bc.detail(gene, request.args.get('cell_type'))
    return jsonify(d), (404 if 'error' in d else 200)
