# Data Pipeline Rules

---
paths:
  - "hMuscle/data/**"
  - "hMuscle/preprocessing/**"
  - "**.npy"
---

## 보호 파일 (절대 덮어쓰기 금지)
- my_isoform_list_fixed.npy
- my_gene_list_fixed.npy
- my_sequence_matrix_fixed.npy

## Rules
- _fixed.npy: 읽기 전용, 새 버전은 날짜 suffix
- Support/query split: isoform-level stratification (gene-level 금지)
- Synthetic data: SpliceAI score > 0.5, 유효 exon boundary 필수 [R5.1]
