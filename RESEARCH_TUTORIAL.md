# DIFFUSE Research System — 실사용 튜토리얼
# 대상: 모델 성능 안정화 단계 (현재) → 후보 아이소폼 분석 단계 (이후)
# 업데이트: 2026-04-03

════════════════════════════════════════════════════════════
0. 시작 전 체크
════════════════════════════════════════════════════════════

## 매 세션 시작 루틴
```bash
cd ~/sw1686/DIFFUSE
conda activate isoform_env
nvidia-smi                    # GPU 상태 확인
claude                        # Claude Code 실행
```

## Claude Code 실행 후 첫 번째로 할 것
```
/memory
```
→ 이전 세션에서 Claude가 자동 저장한 메모 확인
→ 이어서 작업할 내용이 있으면 바로 파악 가능

════════════════════════════════════════════════════════════
1. 지금 당장 써야 하는 것 (성능 안정화 단계)
════════════════════════════════════════════════════════════

## 현재 단계에서 실제로 유용한 에이전트: 2개
  - experiment-analyst  → 로그/결과 분석
  - model-engineer      → 코드 수정 제안

## 나머지는 이후 단계
  - db-fetcher, bio-validator → 후보 아이소폼 확정 후
  - ablation-designer         → 성능 안정화 후
  - paper-critic              → 논문 작성 단계

════════════════════════════════════════════════════════════
2. 실험 실행
════════════════════════════════════════════════════════════

## 기본 실행
```bash
# 터미널에서 (Claude Code 밖)
cd ~/sw1686/DIFFUSE/hMuscle/model
conda activate isoform_env
nohup python run_GPU_Full.py \
  > ../logs_isoform/run_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "PID: $!"   # 백그라운드 실행 확인
```

## 실시간 로그 모니터링
```bash
# 가장 최근 로그 실시간 확인
tail -f ../logs_isoform/$(ls -t ../logs_isoform/ | head -1)

# GPU 사용률 모니터링
watch -n 5 nvidia-smi
```

## 새 버전 모델 실행 (Claude Code 안에서)
```
/new-version "gene-level gating 레이어 추가"
```
→ 현재 최신 버전 자동 확인 (v4-3)
→ v5_integrated_full_model.py 생성
→ _backup 파일 자동 생성
→ 변경 부분에 주석 추가
→ run_GPU_Full.py import 경로 업데이트 방법 안내

════════════════════════════════════════════════════════════
3. 실험 결과 분석
════════════════════════════════════════════════════════════

## 기본 사용 (가장 자주 쓸 명령)
```
Use the experiment-analyst agent to analyze the latest logs in hMuscle/logs_isoform/
```

## 특정 버전 결과 분석
```
Use the experiment-analyst agent to analyze v4-3 results in
hMuscle/results_isoform/total_performance_summary.csv
and compare with baseline
```

## 특정 문제에 집중한 분석
```
Use the experiment-analyst agent to check if there is
mode collapse in the latest experiment results.
Focus on GO_0006096 which is a sparse class.
```

```
Use the experiment-analyst agent to diagnose gene-level bias
in the latest model. Check if predictions are too similar
across isoforms of the same gene.
```

## 분석 결과 해석 요청
```
The experiment-analyst just finished. Based on the results in
tasks/agent-results/experiment-analysis/,
what is the most critical issue to fix right now?
```

## 분석 후 자동으로 나오는 정보
- 4개 GO term × 4개 지표 (Precision, Recall, F1, AUROC)
- Baseline 대비 delta
- Collapse score (< 0.3이면 mode collapse)
- Gene-bias score (< 0.3이면 gene-level 편향)
- Triplet active ratio (< 5%이면 easy triplet 지배)
- 어느 Axis가 문제인지 자동 진단

════════════════════════════════════════════════════════════
4. 모델 코드 수정
════════════════════════════════════════════════════════════

## 기본 패턴: 분석 결과 → 수정 제안

```
Use the experiment-analyst agent to analyze the latest results.
Then use the model-engineer agent to propose code changes
based on the analysis.
```

## 특정 문제 해결 요청

### Mode Collapse 해결
```
The collapse score is 0.18 which indicates mode collapse.
Use the model-engineer agent to propose a fix.
The fix should address Axis 1 (Data Sparsity).
Refer to R1.1 Focal Loss gamma adjustment or R1.4 LDAM.
```

### Gene-level Bias 해결
```
The gene-bias score is 0.21. Isoforms from the same gene
are getting identical predictions.
Use the model-engineer agent to propose a solution using
gradient reversal layer [R2.4] or attention-based gene gating [R4.2].
```

### Triplet Loss 개선
```
Triplet active ratio is 3%. Most triplets are easy.
Use the model-engineer agent to implement hard negative mining [R3.2].
Ensure negatives are cross-gene only, never intra-gene.
```

### Embedding Space 개선
```
Use the model-engineer agent to check the current embedding quality
using IsoScore [R6.2] and suggest improvements.
The goal is IsoScore > 0.3 and effective rank > d/4.
```

## 수정 제안 검토 후 적용
```
The model-engineer proposed changes in
tasks/agent-results/model-proposals/.
Please show me the diff and explain the mathematical justification
for each change before I approve.
```

## 수정 적용
```
/new-version "hard negative mining 적용, cross-gene 강제화 [R3.2]"
```

════════════════════════════════════════════════════════════
5. 수학적 검증 요청
════════════════════════════════════════════════════════════

## 새 아이디어의 수학적 타당성 확인
```
I want to add a gradient reversal layer to reduce gene-level bias.
Before writing any code, validate this approach mathematically:
1. Which Axis does this address?
2. What is the mathematical justification? Reference [Rx.y]
3. What are the known failure modes?
4. How should we verify it worked?
```

## 현재 loss 설정 점검
```
Current settings: gamma=2, margin=0.3, lambda_focal=1.0, lambda_triplet=0.5
Given the latest experiment results, are these settings
mathematically appropriate? Check against Axis 1 and Axis 3 rules.
```

## 새 방법론 비교 분석
```
Compare these two approaches for fixing gene-level bias:
A) Gradient Reversal Layer [R2.4]
B) Invariant Risk Minimization [R2.2]
Which is more appropriate for our current codebase and data situation?
Consider implementation complexity and theoretical guarantees.
```

════════════════════════════════════════════════════════════
6. 어블레이션 설계 (성능 안정화 후)
════════════════════════════════════════════════════════════

## 기본 어블레이션 설계
```
/ablation triplet_loss
```

## 새 컴포넌트 검증
```
Use the ablation-designer agent to design ablation experiments
for the gene gating layer added in v5.
We need to prove this component is necessary for Nature Methods.
```

## 어블레이션 결과 분석
```
Use the experiment-analyst agent to analyze ablation results
in logs_isoform/ablation_*.log
Compare no_triplet vs no_focal vs full model.
Which component contributes most to isoform-specific prediction?
```

════════════════════════════════════════════════════════════
7. DB 조회 (후보 아이소폼 확정 후)
════════════════════════════════════════════════════════════

## 특정 단백질 조회
```
/db-query MYH7
```
→ UniProt에서 MYH7 아이소폼 목록 + GO term
→ AlphaFold pLDDT score + IDR 영역
→ Novel case 후보 플래그

## 자연어로 조회
```
Use the db-fetcher agent to query MYH7 (Myosin-7).
Focus on isoforms with different GO terms from the canonical.
Flag any isoforms that might be novel cases for our model.
```

## 여러 단백질 순차 조회
```
Use the db-fetcher agent to query these muscle-related proteins
one by one: MYH7, TNNT2, TPM1.
For each, identify isoforms that have different localization
or GO terms from their canonical form.
Save results to tasks/agent-results/db-fetch/
```

════════════════════════════════════════════════════════════
8. 전체 파이프라인 (후보 아이소폼 확정 후)
════════════════════════════════════════════════════════════

## 전체 실행
```
/run-pipeline MYH7
```

## 단계별 실행
```
/run-pipeline db MYH7        # db-fetcher만
/run-pipeline validate       # bio-validator만
/run-pipeline review         # paper-critic만
```

## 파이프라인 상태 확인
```
Show me the current pipeline status from
tasks/agent-results/pipeline-status.md
and summarize what has been completed and what is pending.
```

════════════════════════════════════════════════════════════
9. 논문 기여도 검토 (성능 안정화 후)
════════════════════════════════════════════════════════════

## 기본 검토
```
/paper-check v5
```

## 특정 결과에 대한 검토
```
Use the paper-critic agent to review the latest experiment results.
Act as a Nature Methods reviewer who has already read the DIFFUSE paper.
Be critical and identify what additional experiments are needed.
```

## 리뷰어 반박 준비
```
Use the paper-critic agent to anticipate reviewer criticisms
for our gene-bias reduction approach.
Suggest rebuttal arguments based on our ablation results.
```

════════════════════════════════════════════════════════════
10. 작업 기록 관리
════════════════════════════════════════════════════════════

## 세션 중 발견한 규칙 즉시 저장
```
# Claude Code 안에서 # 으로 시작하면 CLAUDE.md에 자동 추가
# GO_0006096은 positive < 10이라 AUPRC만 신뢰 가능
# triplet negative를 gene family 기준으로 분리하면 active ratio 개선됨
```

## 실험 결과 기록
```
Update tasks/todo.md with today's results:
- v4-3 GO_0006936 F1: 0.71 (baseline: 0.65)
- Collapse score: 0.28 (still concerning)
- Next: try gamma=2.5 per R1.1
```

## 실수/발견 패턴 기록
```
Add to tasks/lessons.md:
Date: 2026-04-03
Problem: intra-gene negative mining이 gene-bias를 강화했음
Cause: negative sample이 같은 gene의 다른 isoform이었음
Fix: cross-gene 필터링 추가
Rule: negative는 항상 gene_id 기준으로 필터링할 것
```

════════════════════════════════════════════════════════════
11. 단계별 사용 로드맵
════════════════════════════════════════════════════════════

## 지금 (성능 안정화)
```
실험 실행
→ experiment-analyst로 진단
→ model-engineer로 코드 수정 제안
→ 검토 후 /new-version 적용
→ 반복
```
쓰는 것: experiment-analyst, model-engineer, /new-version, /memory, #메모

## 성능 안정화 후
```
ablation-designer로 어블레이션 설계
→ 실험 실행
→ experiment-analyst로 분석
→ /paper-check로 기여도 확인
```
추가되는 것: /ablation, ablation-designer, paper-critic, /paper-check

## 후보 아이소폼 확정 후
```
후보 단백질 리스트업
→ /db-query로 각 단백질 조회
→ /run-pipeline으로 전체 분석
→ bio-validator PASS된 것만 분석 대상
```
추가되는 것: /db-query, /run-pipeline, db-fetcher, bio-validator

════════════════════════════════════════════════════════════
12. 자주 하는 실수 & 주의사항
════════════════════════════════════════════════════════════

## 절대 하면 안 되는 것
- _fixed.npy 파일 수정 (my_isoform_list_fixed.npy 등)
- pfn_worker.py 무단 수정
- integrated_full_model.py 직접 수정 (항상 새 버전 파일로)
- intra-gene negative 사용 [R3.1 위반]
- gene context 직접 concatenation [R2.1 위반]

## 수치 해석 주의사항
- F1만 보지 말 것 → AUROC + Precision + Recall 함께
- Sparse class (GO_0006096 등): AUPRC primary [R9.1]
- delta F1 < 0.05: 통계적 유의성 의심 → bootstrap CI 확인 [R9.4]
- Overall 성능: Macro-F1 (Micro-F1은 majority class 편향) [R9.5]

## 에이전트 호출 순서 주의
- model-engineer는 bio-validator PASS 후에만
- paper-critic은 experiment-analyst 결과가 있어야 의미 있음
- 성능 안정화 전에 /run-pipeline 전체 돌리지 말 것

## 파일 관리
- 실험 결과는 날짜 suffix로 구분
- agent-results/는 실제 연구 타겟 결과만
- 테스트 결과는 agent-results/_test/로 분리

════════════════════════════════════════════════════════════
13. 빠른 참조 카드
════════════════════════════════════════════════════════════

## 매일 쓰는 것
| 목적 | 명령 |
|------|------|
| 로그 분석 | Use the experiment-analyst agent to analyze the latest logs |
| 코드 수정 제안 | Use the model-engineer agent to propose fixes based on latest analysis |
| 새 버전 생성 | /new-version "변경 내용" |
| 세션 메모 저장 | # 내용 |
| 이전 메모 확인 | /memory |

## 가끔 쓰는 것
| 목적 | 명령 |
|------|------|
| 어블레이션 설계 | /ablation {컴포넌트} |
| 논문 기여도 확인 | /paper-check {버전} |
| DB 조회 | /db-query {단백질ID} |
| 전체 파이프라인 | /run-pipeline {타겟} |
| 파이프라인 상태 | Show pipeline-status.md |

## 문제 발생 시
| 증상 | 확인할 것 | 참조 |
|------|---------|------|
| 특정 GO term으로 예측 쏠림 | collapse_score < 0.3 | Axis 1, R1.1 |
| 같은 gene isoform 동일 예측 | bias_score < 0.3 | Axis 2, R2.4 |
| loss가 안 떨어짐 | active_ratio < 5% | Axis 3, R3.2 |
| 특정 modality만 학습 | gradient norm 불균형 | Axis 4, R4.4 |
| novel isoform 성능 급락 | OOD score 확인 | Axis 7, R7.4 |

## 수식 참조가 필요할 때
tasks/reference-knowledge-base.md
→ Axis 1~9 × 수식 + 레퍼런스 + 주의사항 전체 수록
