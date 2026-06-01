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
