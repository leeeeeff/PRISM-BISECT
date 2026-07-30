#!/bin/bash
# 650M + ProtT5 임베딩 완료 감지 → 자동 scan 실행
cd /home/welcome1/sw1686/DIFFUSE/hMuscle/model
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isoform_env

DATA="../data"
# Train-only files (brain skipped due to BambuTx ID mismatch with SQANTI3 CSV)
F650M_TRAIN="${DATA}/esm2_train_human_layer33_t33_650M.npy"
FPROT_TRAIN="${DATA}/esm2_train_human_layer24_prot_t5_xl.npy"
LOG="../../logs_isoform/exp_f_scan_$(date +%Y%m%d_%H%M).log"

echo "[monitor] Waiting for embedding files (train-only; brain skipped)..." | tee -a "$LOG"
echo "[monitor] Checking every 60s" | tee -a "$LOG"

# 650M train 완료 확인 (이미 완료됨)
if [ -f "$F650M_TRAIN" ]; then
  echo "[$(date +%H:%M)] 650M train READY" | tee -a "$LOG"
else
  until [ -f "$F650M_TRAIN" ]; do
    echo "[$(date +%H:%M)] 650M train not ready" | tee -a "$LOG"
    sleep 60
  done
  echo "[$(date +%H:%M)] 650M train READY" | tee -a "$LOG"
fi

# ProtT5 train 완료 대기 (최대 2시간)
WAIT=0
until [ -f "$FPROT_TRAIN" ]; do
  WAIT=$((WAIT + 60))
  if [ "$WAIT" -ge 7200 ]; then
    echo "[$(date +%H:%M)] ProtT5 timeout (2h) — running scan with available models" | tee -a "$LOG"
    break
  fi
  echo "[$(date +%H:%M)] ProtT5 not ready ($((WAIT/60))min elapsed)" | tee -a "$LOG"
  sleep 60
done

if [ -f "$FPROT_TRAIN" ]; then
  echo "[$(date +%H:%M)] ProtT5 train READY" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "[$(date +%H:%M)] Running exp_f_plm_scale_scan.py..." | tee -a "$LOG"
python3 -u exp_f_plm_scale_scan.py 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M)] Scan complete." | tee -a "$LOG"
