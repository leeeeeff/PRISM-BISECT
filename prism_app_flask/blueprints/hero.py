"""Hero 스크롤 랜딩 — 이 툴이 무엇을 하는지 모식도 + 4줄 요약."""
from flask import Blueprint, render_template

bp = Blueprint('hero', __name__)


@bp.route('/')
def index():
    capabilities = [
        {'icon': 'dna', 'title': 'Individual Gene · Isoform',
         'desc': 'GO prediction, 8-axis representation-map position, and per-layer information trajectory at a glance',
         'href': '/analysis#individual'},
        {'icon': 'bar-chart-3', 'title': 'My Data Analysis',
         'desc': 'QC through functional patterns and condition analysis — the essentials on one page',
         'href': '/mydata'},
        {'icon': 'flask-conical', 'title': 'BISECT Analysis',
         'desc': 'Characterize the functional consequences of isoform switching, case by case',
         'href': '/bisect'},
        {'icon': 'book-open', 'title': 'Data Generation · Reproduction Tutorial',
         'desc': 'Input generation, key figures, statistics, and principles — step by step',
         'href': '/tutorial'},
    ]
    return render_template('hero.html', capabilities=capabilities)
