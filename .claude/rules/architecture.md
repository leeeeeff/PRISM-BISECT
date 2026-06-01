# Architecture Rules

## Core
- PFN backbone — ablation 없이 교체 금지
- Multimodal inputs 필수: ESM + cell localization + PPI
- 새 모듈: PFN forward pass 시그니처 변경 없이 플러그인

## Anti-Gene-Bias
- isoform_emb를 gene_emb보다 먼저 계산
- gene context는 반드시 attention/gating으로 처리
- 직접 concatenation 금지 [R2.1]

## Versioning
- Production: integrated_full_model.py
- Latest: v4-3_integrated_full_model.py
- New: v5_, v5-1_ ...
- 수정 전 _backup_{DATE}.py 생성
