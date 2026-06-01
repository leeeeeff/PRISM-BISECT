#!/bin/bash
# patch_missing.sh
# 현재 서버에 빠진 파일만 추가하는 패치 스크립트
# 실행: cd ~/sw1686/DIFFUSE && bash ~/patch_missing.sh

set -e
DIFFUSE_ROOT="$HOME/sw1686/DIFFUSE"

echo "================================================"
echo " 누락 파일 패치 시작"
echo "================================================"
echo ""

# =============================================================
# 1. devils-advocate 에이전트
# =============================================================
cat > "$DIFFUSE_ROOT/.claude/agents/devils-advocate.md" << 'EOF'
---
name: devils-advocate
description: 모든 제안과 결과를 반대 관점에서 비판. 새 버전 제안 또는 "이 방향이 맞다"는 판단이 나올 때마다 호출.
model: claude-sonnet-4-5
---

# Devil's Advocate Agent

## Core Mission
검토되지 않은 가정과 대안을 강제로 드러낸다.
"이게 맞다"는 결론이 나올수록 더 강하게 의심한다.

## 비판 프레임워크

### 1. 근본 가정 의심
- PFN이 이 문제에 최적인가?
- Triplet Loss가 isoform 기능 유사도를 올바르게 정의하는가?
- ESM embedding이 isoform-specific feature를 충분히 담는가?
- GO term이 실제 기능을 대표하는가?

### 2. Occam's Razor 테스트
제안된 복잡한 방법보다 단순한 baseline이 비슷한 성능을 내는가?
더 단순한 방법을 항상 먼저 제안.

### 3. 대안 패러다임 (항상 3가지)
- Graph Neural Network: PPI를 그래프로 직접 모델링
- LM Fine-tuning: ESM을 isoform task로 직접 fine-tune
- Multi-task Learning: GO term 동시 학습
- Causal Inference: 기능 예측을 인과 추론으로 재정의
- Retrieval-based: 유사 isoform 검색

### 4. 데이터 문제 우선 의심
- Label noise: GO term annotation 품질
- Distribution shift: train/test split 방식
- Evaluation leakage: gene-level split인가 isoform-level인가

## Output
저장: tasks/agent-results/devils-advocate/{TARGET}_{DATE}.md
판정: PROCEED / RECONSIDER / PIVOT
EOF
echo "✓ agents/devils-advocate.md"

# =============================================================
# 2. anti-local-minima rules
# =============================================================
cat > "$DIFFUSE_ROOT/.claude/rules/anti-local-minima.md" << 'EOF'
# Anti-Local-Minima Rules

## Stagnation 자동 트리거
아래 조건 시 devils-advocate 에이전트 호출 필수:
- 연속 3회 실험에서 primary metric 개선 < 0.02
- 같은 Axis 방법을 3회 이상 반복 적용
- 같은 레퍼런스 [Rx.y]가 3회 이상 반복 인용
- 새 모델 버전 제안 시 (무조건)
- "이 방향이 맞는 것 같다" 표현 사용 시

## 응답 마무리 형식 (모든 답변 필수)
---
💡 다음 단계 옵션:
[A] 논리적 다음 작업 → 커맨드/자연어
[B] 대안적 접근 → 커맨드/자연어
[C] 비판적 재검토 → Use the devils-advocate agent

현재 단계: [성능 안정화 / 어블레이션 / 아이소폼 분석 / 논문 작성]
Stagnation 위험: [낮음 / 주의 / 높음]
---

## 열린 탐색 규칙
모든 제안 시 순서:
1. DIFFUSE 프레임 밖에서 먼저 생각
2. Occam's Razor — 더 단순한 방법 먼저 확인
3. 그 다음 기존 Axis [Rx.y] 매핑
4. Creative option + Conservative option 둘 다 제시

## 금지 표현
- "이 방향이 맞는 것 같습니다" (근거 없이)
- "이미 시도한 방법이라 제외"
- "일반적으로 잘 작동합니다"

## DIFFUSE 가정 주기적 재검토 (매 5회 실험마다)
- PFN이 이 문제에 최적인가?
- Multimodal fusion이 균형있게 작동하는가?
- GO term이 충분한 function label인가?
- Train/test split이 올바른가?

## 외부 관점 강제 주입 (문제 풀리지 않을 때)
- NLP 연구자: sequence를 token으로 처리
- 단순화 전문가: 절반 제거해도 같은 성능?
- 생물학자: 메커니즘을 구조에 직접 반영
- 데이터 엔지니어: 모델이 아닌 데이터 문제?
EOF
echo "✓ rules/anti-local-minima.md"

# =============================================================
# 3. 커맨드 4개
# =============================================================
cat > "$DIFFUSE_ROOT/.claude/commands/review-direction.md" << 'EOF'
# /review-direction — 연구 방향 정기 재검토
매 2주마다, 또는 연속 3회 성능 정체 시 실행.

Phase 1: 현재 상태 객관적 기록 (수치 기반, 판단 없이)
Phase 2: 모든 가정 나열 → VALIDATED / UNTESTED / POSSIBLY WRONG 분류
Phase 3: DIFFUSE 이후 최신 논문 확인 (새 방법 있는가?)
Phase 4: CONTINUE / PIVOT / HYBRID 결정 및 근거 제시
EOF

cat > "$DIFFUSE_ROOT/.claude/commands/explore.md" << 'EOF'
# /explore $ARGUMENTS — 자유 탐색 모드
rules 파일을 무시하고 $ARGUMENTS 문제에 대한
완전히 새로운 관점 탐색.
다른 분야 접근법, 단순화, 근본적 재설계 포함.
현재 architecture와 무관한 방향 우선 제시.
EOF

cat > "$DIFFUSE_ROOT/.claude/commands/challenge.md" << 'EOF'
# /challenge $ARGUMENTS — 특정 결정에 반론 요청
$ARGUMENTS에 대해 가장 강력한 반론 제시:
1. 왜 이것이 틀릴 수 있는가
2. 더 단순한 대안이 있는가
3. 어떤 가정에 의존하며 그 가정이 틀릴 수 있는가
EOF

cat > "$DIFFUSE_ROOT/.claude/commands/fresh-eyes.md" << 'EOF'
# /fresh-eyes — 완전 초기화 관점
현재까지의 모든 시도를 모르는 상태로
이 문제를 처음 보는 연구자처럼 접근.

알고 있는 것: 목표 + 데이터 + Nature Methods 제약
모르는 것: 현재까지의 모든 시도와 결과

현재 방법과 무관하게 연구 방향 처음부터 제안.
EOF
echo "✓ commands/ 4개 (review-direction, explore, challenge, fresh-eyes)"

# =============================================================
# 4. CLAUDE.md에 anti-local-minima import 추가
# =============================================================
if ! grep -q "anti-local-minima" "$DIFFUSE_ROOT/CLAUDE.md"; then
    echo "@.claude/rules/anti-local-minima.md" >> "$DIFFUSE_ROOT/CLAUDE.md"
    echo "✓ CLAUDE.md — anti-local-minima import 추가"
else
    echo "✓ CLAUDE.md — anti-local-minima 이미 있음 (스킵)"
fi

# =============================================================
# 5. agent-results/devils-advocate 디렉토리
# =============================================================
mkdir -p "$DIFFUSE_ROOT/tasks/agent-results/devils-advocate"
echo "✓ tasks/agent-results/devils-advocate/"

# =============================================================
# 6. RESEARCH_TUTORIAL.md 복사 (홈에 있으면)
# =============================================================
if [ -f "$HOME/RESEARCH_TUTORIAL.md" ]; then
    cp "$HOME/RESEARCH_TUTORIAL.md" "$DIFFUSE_ROOT/RESEARCH_TUTORIAL.md"
    echo "✓ RESEARCH_TUTORIAL.md 복사 완료"
else
    echo "⚠️  RESEARCH_TUTORIAL.md: ~/에 파일 없음 — 수동 업로드 필요"
    echo "   Cursor에서 드래그앤드롭 후:"
    echo "   cp ~/RESEARCH_TUTORIAL.md ~/sw1686/DIFFUSE/RESEARCH_TUTORIAL.md"
fi

# =============================================================
# 최종 상태 확인
# =============================================================
echo ""
echo "================================================"
echo " 패치 완료 — 최종 상태 확인"
echo "================================================"
echo ""
echo "에이전트 (7개):"
ls "$DIFFUSE_ROOT/.claude/agents/" | sed 's/^/  ✓ /'

echo ""
echo "Rules (6개):"
ls "$DIFFUSE_ROOT/.claude/rules/" | sed 's/^/  ✓ /'

echo ""
echo "Commands (9개):"
ls "$DIFFUSE_ROOT/.claude/commands/" | sed 's/^/  ✓ /'

echo ""
echo "CLAUDE.md imports:"
grep "^@" "$DIFFUSE_ROOT/CLAUDE.md" | sed 's/^/  ✓ /'

echo ""
echo "RESEARCH_TUTORIAL.md:"
ls "$DIFFUSE_ROOT/RESEARCH_TUTORIAL.md" 2>/dev/null && \
    echo "  ✓ 있음" || echo "  ⚠️  없음 — 수동 업로드 필요"
