#!/bin/bash
# PRISM+BISECT 웹 앱 로컬 실행 스크립트
#
# 일반 실행:   ./run_app.sh
# 포트 지정:   ./run_app.sh --server.port 8502
#
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 을 먼저 설정해야
# umap-learn이 TensorFlow/protobuf 버전 충돌 없이 동작한다.

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "$(dirname "$0")"

exec streamlit run prism_app/app/main.py \
    --server.headless false \
    "$@"
