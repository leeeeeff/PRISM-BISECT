#!/bin/bash
# muscle 2-iso Pfam scan, apples-to-apples with brain (--cut_ga). Resource-disciplined:
# 4 chunks in flight, --cpu 2 each (=8 cores), nice+10.
HMM_DB=/home/welcome1/sw1686/DIFFUSE/hMuscle/data/pfam/Pfam-A.hmm
CDIR=/home/welcome1/sw1686/DIFFUSE/reports/muscle_labelgap/chunks
ODIR=/home/welcome1/sw1686/DIFFUSE/reports/muscle_labelgap/hmmscan_out
source /home/welcome1/miniconda3/etc/profile.d/conda.sh
conda activate isoform_env
MAXJOBS=4
for f in $CDIR/chunk_*.fa; do
  base=$(basename $f .fa)
  if [ -s "$ODIR/${base}.domtbl" ]; then echo "[skip] $base"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[start] $base $(date)"
  nice -n 10 hmmscan --cut_ga --cpu 2 --domtblout $ODIR/${base}.domtbl $HMM_DB $f > $ODIR/${base}.log 2>&1 &
done
wait
echo "[ALL DONE] $(date)"
