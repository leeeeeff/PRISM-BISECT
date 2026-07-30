"""분석 허브 — gene/isoform 카트에서 push된 케이스를 메모와 함께 정리하는 개인 대시보드.
(예전엔 4개 모드 바로가기 랜딩이었으나, topbar nav와 중복되어 제거 — 카트/메모로 대체.)"""
from flask import Blueprint, render_template

bp = Blueprint('analysis_home', __name__)


@bp.route('/analysis')
def home():
    return render_template('analysis_home.html')
