#!/usr/bin/env bash
# PRISM Flask 앱 기동 스크립트.
#   ./run.sh dev     → Flask dev server (reloader, debug)   :8600
#   ./run.sh prod    → gunicorn WSGI (2 worker, preload)     :8600
#   ./run.sh stop    → 실행 중인 인스턴스 정지
set -euo pipefail

PORT="${PORT:-8600}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/logs_isoform/prism_flask.log"
MODE="${1:-dev}"

# conda env
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate isoform_env 2>/dev/null || true
cd "$ROOT"

case "$MODE" in
  dev)
    echo "▸ dev server  http://0.0.0.0:${PORT}"
    OMP_NUM_THREADS=4 python prism_app_flask/app.py
    ;;
  prod)
    # --preload: 마스터가 인덱스를 1회 로드 → copy-on-write 로 worker 간 공유(메모리 절약)
    echo "▸ gunicorn  http://0.0.0.0:${PORT}  (log: ${LOG})"
    OMP_NUM_THREADS=4 nice -n 10 nohup gunicorn \
      --workers 2 --preload --timeout 120 \
      --bind "0.0.0.0:${PORT}" \
      prism_app_flask.app:app > "${LOG}" 2>&1 &
    echo "  pid $! · tail -f ${LOG}"
    ;;
  stop)
    pkill -f "prism_app_flask.app:app" 2>/dev/null && echo "▸ gunicorn 정지" || true
    pkill -f "prism_app_flask/app.py"  2>/dev/null && echo "▸ dev server 정지" || true
    ;;
  *)
    echo "usage: ./run.sh [dev|prod|stop]"; exit 1;;
esac
