# PRISM + BISECT — Flask 앱

작업(task) 중심 IA + 계측기(Data-Instrument) 미학으로 재구성한 웹앱.
기존 Streamlit 앱(`prism_app/`)은 그대로 보존되며, 이 앱은 별도 디렉토리에서 독립 실행된다.

## 정보 구조
| 경로 | 내용 |
|---|---|
| `/` | hero — cascade(B0→B5) 모식도 + capability 4 |
| `/analysis` | 분석 허브(4 모드) |
| `/gene/<q>` | **개별 유전자·아이소폼 계측**(flagship): GO 예측 · 8축 표현 지도 · trajectory flow |
| `/mydata` | 내 데이터 분석 — tissue 요약 한 페이지 |
| `/bisect` | BISECT 케이스 탐색기 |
| `/tutorial` | 데이터 생성·재현 튜토리얼 |

## 기동
```bash
conda activate isoform_env
pip install -r prism_app_flask/requirements.txt      # flask, gunicorn (최초 1회)

./prism_app_flask/run.sh dev      # 개발(reloader)   http://localhost:8600
./prism_app_flask/run.sh prod     # gunicorn(2 worker, preload)
./prism_app_flask/run.sh stop     # 정지
```

### 상시 서비스(부팅 자동기동·리버스프록시) — 관리자 권한 필요
- `deploy/prism-flask.service` — systemd unit 템플릿(자동기동·크래시 재시작). 헤더 주석의 설치 절차 참조.
- `deploy/nginx-prism.conf` — nginx 리버스 프록시(표준 포트·정적 캐시·gzip·TLS). 도메인/경로 교체 후 사용.

## 데이터 인덱스 (최초 1회 precompute)
```bash
python prism_app_flask/precompute/build_isoform_index.py   # 개별분석 인덱스(브레인)
# mydata 요약은 첫 요청 시 계산 후 디스크 캐시(data/isoform_index/summary_*.json)
```

## 아키텍처
- **서버 렌더(Jinja) + Plotly.js**. 차트/UI 토큰은 `static/js/instrument.js` ↔ `static/css/main.css :root` 단일 소스.
- 데이터 계층 `data_layer/`:
  - `loaders.py` — 개별분석 인덱스(mmap 원본 참조)
  - `isoform_profile.py` — 패널 A/C/D 조립
  - `dataset_summary.py` — tissue 요약(canonical `classify_isoforms` 재사용 + 디스크 캐시)
  - `bisect_cases.py` — BISECT 케이스
- **통일 좌표계 = 브레인(63,994)**: GO score(`brain_full_672_scores.npy`) + 8축·30레이어(`Z_brain_Nx30x8.npy`)
  + gene symbol 이 모두 같은 순서. 자세한 근거는 memory `finding-flask-app-brain-unified-coordinate`.

## 알려진 한계 (defer)
- 개별분석 **패널 B(covariate)**: fixed-pop(ENST/BambuTx) 좌표계에만 있어 브레인 A1BG-204 universe 엔
  미연결. 브레인 서열 소싱 + metapredict/hmmscan 필요.
- canonical/APPRIS 플래그: 매핑 소스 부재.
