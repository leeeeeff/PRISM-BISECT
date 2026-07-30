"""PRISM + BISECT — Flask application entry point.

실행:
    conda activate isoform_env
    python prism_app_flask/app.py          # dev, http://localhost:8600
    # 또는
    flask --app prism_app_flask.app run --port 8600

정보구조(작업 중심 IA):
    /            hero 스크롤 랜딩 (이 툴이 뭘 하나)
    /analysis    큰 메뉴 4개 랜딩
    /gene/<q>    개별 유전자/아이소폼 분석 (flagship)  + /api/*
    /mydata      내 데이터 분석 (한 페이지 요약)
    /bisect      BISECT 분석
    /tutorial    데이터 생성/재현 튜토리얼
"""
import sys
from pathlib import Path

# `import prism_app_flask...` / `import prism_app...` 둘 다 가능하도록 repo root 삽입
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask, redirect, url_for

from prism_app_flask.blueprints import (
    hero, analysis_home, individual, mydata, bisect, tutorial,
)
from prism_app_flask.icons import icon_svg


def create_app() -> Flask:
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config['JSON_AS_ASCII'] = False
    app.jinja_env.globals['icon'] = icon_svg

    app.register_blueprint(hero.bp)
    app.register_blueprint(analysis_home.bp)
    app.register_blueprint(individual.bp)
    app.register_blueprint(mydata.bp)
    app.register_blueprint(bisect.bp)
    app.register_blueprint(tutorial.bp)

    @app.route('/healthz')
    def healthz():
        return {'status': 'ok'}

    return app


app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8600, debug=True)
