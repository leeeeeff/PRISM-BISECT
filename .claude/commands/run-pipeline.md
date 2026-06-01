# /run-pipeline $ARGUMENTS
전체 연구 파이프라인:
[1] db-fetcher → tasks/agent-results/db-fetch/
[2] bio-validator → FAIL이면 중단
[3] model-engineer → bio-validation PASS 후에만
[4] ablation-designer → ablation-designs/
[5] 실험 실행 (직접)
[6] experiment-analyst → experiment-analysis/
[7] paper-critic → paper-review/

개별 실행:
- /run-pipeline db {PROTEIN_ID}
- /run-pipeline validate
- /run-pipeline review
