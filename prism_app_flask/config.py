"""PRISM Flask — central paths & config.

모든 경로는 여기서 단일 관리. 대용량 원본(Z, GO score matrix)은 복사하지 않고
precompute 인덱스가 기록한 절대경로를 런타임에 mmap 참조한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # /home/welcome1/sw1686/DIFFUSE
FLASK_ROOT = Path(__file__).resolve().parent            # prism_app_flask
INDEX_DIR = FLASK_ROOT / 'data' / 'isoform_index'
DEMO_DIR = ROOT / 'prism_app' / 'data' / 'demo'
ANNOT_DIR = ROOT / 'prism_app' / 'data' / 'annotations'

# 개별 분석 flagship 의 기본 universe (통일 좌표계: GO score + Z + gene symbol 모두 정렬).
DEFAULT_UNIVERSE = 'brain'

# 기존 Streamlit 앱의 framework-agnostic 로직을 import 하기 위한 경로.
PRISM_APP_IMPORTABLE = ROOT  # `import prism_app.core...` 가능하게

# GitHub (튜토리얼 링크용)
GITHUB_URL = 'https://github.com/leeeeeff/PRISM-BISECT'

# 캐시
MAX_UMAP = 15_000
