#!/bin/bash
# BISECT v2.0 re-run: 51 v1.x cases -> --force migration
cd /home/welcome1/sw1686/DIFFUSE/Final_analysis/pipeline_bioanalysis
LOG="/home/welcome1/sw1686/DIFFUSE/logs_bisect/rerun_v2_$(date +%Y%m%d_%H%M).log"
echo "Starting BISECT v2.0 re-run at $(date)" | tee "$LOG"
echo "Cases CSV: cases_v1x_rerun.csv" | tee -a "$LOG"
conda run -n isoform_env python orchestrate.py \
  --cases cases_v1x_rerun.csv \
  --mode deep \
  --force \
  --workers 1 2>&1 | tee -a "$LOG"
echo "Done at $(date)" | tee -a "$LOG"
