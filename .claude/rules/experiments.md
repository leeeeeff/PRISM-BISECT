# Experiment Rules

---
paths:
  - "hMuscle/model/**"
  - "hMuscle/results_isoform/**"
---

## Run
nohup python run_GPU_Full.py > ../logs_isoform/run_$(date +%Y%m%d_%H%M).log 2>&1 &

## Evaluation
- GO terms: GO_0006096, GO_0006412, GO_0006936, GO_0022900
- Metrics: Precision, Recall, F1, AUROC — 4개 항상 함께
- Sparse (positive < 50): AUPRC primary [R9.1]
- Bootstrap CI 필수 [R9.4]

## Ablation Naming
no_triplet / no_focal / no_ppi / no_esm / no_cellloc / no_isoform_specific
