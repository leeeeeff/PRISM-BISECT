#!/bin/bash
# wait_and_run_P3.sh
# GPU 0,1 각각 8GB 이상 여유 메모리 확인 후 P3 → P3-256 순차 실행

MIN_FREE_MB=8000
CHECK_INTERVAL=60
MODEL_DIR=/home/welcome1/sw1686/DIFFUSE/hMuscle/model
LOG_DIR=/home/welcome1/sw1686/DIFFUSE/logs_isoform

echo "[$(date)] P3 대기 시작 — GPU 0,1 각각 ${MIN_FREE_MB}MB 이상 여유 대기 중"

while true; do
    FREE0=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)
    FREE1=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
    echo "[$(date)] GPU 0 free=${FREE0}MB  GPU 1 free=${FREE1}MB"

    if [ "$FREE0" -ge "$MIN_FREE_MB" ] && [ "$FREE1" -ge "$MIN_FREE_MB" ]; then
        echo "[$(date)] GPU 여유 확인 — P3 실행 시작"
        break
    fi

    sleep $CHECK_INTERVAL
done

source /home/welcome1/miniconda3/etc/profile.d/conda.sh
conda activate isoform_env
cd "$MODEL_DIR"

# ── Experiment 1: v8b-P3 (SwissProt 제거만) ──────────────────────────
LOG_P3="${LOG_DIR}/run_v8b-P3_$(date +%Y%m%d_%H%M).log"
echo "[$(date)] [1/2] v8b-P3 실행 → $LOG_P3"
python run_GPU_v8-P3.py > "$LOG_P3" 2>&1
echo "[$(date)] [1/2] v8b-P3 완료"

# ── Experiment 2: v8b-P3-256 (SwissProt 제거 + dim 256) ─────────────
LOG_P3256="${LOG_DIR}/run_v8b-P3-256_$(date +%Y%m%d_%H%M).log"
echo "[$(date)] [2/2] v8b-P3-256 실행 → $LOG_P3256"
python run_GPU_v8-P3-256.py > "$LOG_P3256" 2>&1
echo "[$(date)] [2/2] v8b-P3-256 완료"

echo "[$(date)] 전체 완료. P3 log: $LOG_P3 | P3-256 log: $LOG_P3256"
