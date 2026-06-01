# /db-query $ARGUMENTS
UniProt + AlphaFold 조회:
1. 아이소폼 목록 + GO term + 기능 도메인
2. pLDDT score, IDR 영역 (< 70)
3. Gene-level vs Isoform-level GO term 차이
4. Novel case 후보 플래그
5. Checklist 항목 1, 4 체크
→ 저장: tasks/agent-results/db-fetch/{ID}_{DATE}.md
