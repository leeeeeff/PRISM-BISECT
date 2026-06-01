# DIFFUSE 연구 레포트 인덱스
**최종 업데이트:** 2026-04-29  
**현재 best 모델:** v6g (Macro-AUPRC 0.465)  
**현재 실행 중:** v7a-SupCon (Exp 2a)

---

## 레포트 목록

| 번호 | 파일 | 내용 | 언제 참고 |
|------|------|------|----------|
| 01 | `01_experiment_results.md` | 전체 버전별 AUPRC 결과, GO term별 특이사항 | 실험 결과 비교 시 |
| 02 | `02_architecture_decisions.md` | 아키텍처 설계 근거, 훈련 단계 설계 | 모델 수정 전 |
| 03 | `03_go_term_analysis.md` | GO term별 특성, 난이도, 생물학적 배경 | GO term 분석 시 |
| 04 | `04_phase15_removal_analysis.md` | Phase 1.5 제거 결정 근거 (실측 데이터) | Phase 관련 논의 시 |
| 05 | `05_v7a_design_rationale.md` | SupCon + Prototype Loss 설계 근거 | v7 구현 시 |
| 06 | `06_ablation_plan.md` | 남은 실험 계획, 결정 트리, GPU 예산 | 다음 실험 계획 시 |
| 07 | `07_known_issues_and_rules.md` | 과거 실수, 규칙, 미검증 가정 | 실험 전 체크리스트 |

---

## 빠른 참조

### 현재 best 성능
```
v6g Macro-AUPRC: 0.465
GO:0006096: 0.823 | GO:0003774: 0.591 | GO:0007204: 0.591*
GO:0030017: 0.201 | GO:0006941: 0.121
* GO:0007204 재현성 주의 (lucky run)
```

### 다음 결정 포인트
- v7a 결과 → Δ AUPRC > +0.02 확인 → Exp 2b 또는 Triplet 회귀

### 중요 파일 위치
```
모델:      hMuscle/model/v7a_integrated_full_model.py
모듈:      hMuscle/model/prototype_contrastive.py
실행:      hMuscle/model/run_GPU_v7a.py
로그:      hMuscle/logs_isoform/GO_*/v7a_*.log
결과:      hMuscle/results_isoform/GO_*/v7a_integrated_*/
평가:      hMuscle/results_isoform/evaluation.py
```
