#!/bin/bash
# 파일명: compare_all.sh

GO_TERM="GO:0022900" # 원하는 GO Term으로 변경

# 파일명 패턴은 귀하가 저장한 방식에 맞춰 수정하세요
FILE_DIFFUSE="../results/${GO_TERM}/${GO_TERM}_DNN_prediction_scores.txt"
FILE_FOCAL="../results/${GO_TERM}/${GO_TERM}_Focal_prediction_scores.txt"
FILE_TRIPLET="../results/${GO_TERM}/${GO_TERM}_Triplet_prediction_scores.txt"
FILE_PFN="../results/${GO_TERM}/${GO_TERM}_PFN_scores.txt"
# FILE_FULL="..."

echo ">>> Comparing Models for ${GO_TERM} <<<"

echo "1. Original DIFFUSE"
python calc_perf.py $GO_TERM $FILE_DIFFUSE

echo "2. DIFFUSE + Focal"
python calc_perf.py $GO_TERM $FILE_FOCAL

echo "3. DIFFUSE + Triplet"
python calc_perf.py $GO_TERM $FILE_TRIPLET

echo "4. DIFFUSE + PFN"
python calc_perf.py $GO_TERM $FILE_PFN
