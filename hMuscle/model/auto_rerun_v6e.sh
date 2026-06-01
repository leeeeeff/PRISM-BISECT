#!/bin/bash
# v6e run1 완료 후 자동 재실행 (patience=6 fix 적용됨)
LOG_DIR="/home/welcome1/sw1686/DIFFUSE/hMuscle/logs_isoform"
MODEL_DIR="/home/welcome1/sw1686/DIFFUSE/hMuscle/model"
OUT="/tmp/v6e_rerun_status.txt"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto_rerun_v6e.sh started" > $OUT

wait_all_done() {
    while true; do
        cnt=0
        for go in GO_0006941 GO_0007204 GO_0030017 GO_0003774 GO_0006096; do
            log=$(ls $LOG_DIR/$go/v6e_*.log 2>/dev/null | sort | tail -1)
            [ -n "$log" ] && grep -q "\[Done\]" "$log" 2>/dev/null && cnt=$((cnt+1))
        done
        echo "[$(date '+%H:%M:%S')] run1 done: $cnt/5" >> $OUT
        [ $cnt -ge 5 ] && return 0
        sleep 60
    done
}

wait_all_done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 5 GO terms done." >> $OUT

# run1 결과 수집
echo "=== run1 results ===" >> $OUT
for go in GO_0006941 GO_0007204 GO_0030017 GO_0003774 GO_0006096; do
    log=$(ls $LOG_DIR/$go/v6e_*.log 2>/dev/null | sort | tail -1)
    type=$(grep "GO term class" "$log" 2>/dev/null | grep -oP "Type-[AB]" | head -1)
    auprc=$(grep "\[Final\] AUROC" "$log" 2>/dev/null | grep -oP "AUPRC=[0-9.]+" | tail -1)
    estop=$(grep "Early Stop" "$log" 2>/dev/null | tail -1 | xargs)
    echo "  $go | $type | $auprc | $estop" | tee -a $OUT
done

echo "" >> $OUT
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting re-run (patience=6)..." >> $OUT
cd $MODEL_DIR
source /home/welcome1/miniconda3/etc/profile.d/conda.sh
conda activate isoform_env
python run_GPU_v6e.py >> $OUT 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Re-run complete." >> $OUT
