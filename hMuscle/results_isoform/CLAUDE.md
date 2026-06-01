# Results & Evaluation

## Entry Points
- evaluation.py     → 메인 성능 평가
- triplet_analysis.py → embedding space 분석
- compare_all.sh    → baseline 비교

## GO Terms
| Dir | GO ID | Note |
|-----|-------|------|
| GO_0006096 | Glycolysis | sparse — 주의 |
| GO_0006412 | Translation | |
| GO_0006936 | Muscle contraction | hMuscle 핵심 |
| GO_0022900 | Electron transport | |

## Evaluation Protocol
1. 4지표 항상 함께: Precision, Recall, F1, AUROC
2. Sparse (positive < 50): AUPRC primary [R9.1]
3. Baseline delta 필수
4. 순서: evaluation → triplet_analysis → compare_all → /paper-check
