# Model Directory

## File Status
| File | Status |
|------|--------|
| integrated_full_model.py | PRODUCTION |
| v4-3_integrated_full_model.py | LATEST EXP |
| pfn_model.py | CORE (수정 시 ablation 필수) |
| model_fixed.py | REFERENCE (수정 금지) |

## Versioning
- 새 버전: v5_, v5-1_ 순으로
- 수정 전 반드시: _backup_{DATE}.py 생성
- 변경 주석: # v{N} [이유]

## DO NOT
- _fixed.npy 덮어쓰기 금지
- pfn_worker.py 무단 수정 금지
- PFN forward pass 시그니처 변경 금지
