# 레포트 07: 알려진 문제 및 규칙 (Anti-Patterns)
**작성일:** 2026-04-29  
**목적:** 과거 실수 재발 방지, 미래 작업 시 반드시 확인

---

## 1. 반복하면 안 되는 실수

### [FAIL-1] Phase 1.5 frozen 선언 신뢰 오류
- **상황:** v6e/v6f에서 Phase 1.5 feature_model을 frozen 선언
- **실제:** `max_diff=0.52-0.63` — 임베딩이 실측 변화
- **교훈:** GradientTape(수동)과 model.compile+fit(자동) 혼용 시 freeze 효과 불완전
- **규칙:** Phase 간 frozen 선언 시 반드시 임베딩 전후 max_diff 측정

### [FAIL-2] Gene-level bias score 엔트로피 공식 오류
- **상황:** 초기 `compute_bias_score()` 엔트로피 기반 공식
  - `bias_score = 1 - H(y|isoform_id) / H(y|gene_id)` 시도
- **실제:** 모든 GO annotation이 gene-level → H_iso=0 trivially → bias_score=1 항상
- **교훈:** 데이터 구조 먼저 확인 후 공식 설계
- **수정:** score-variance 방식으로 교체
  - `bias_score = mean(within_gene_score_std) / global_score_std`

### [FAIL-3] TARGET_COVERAGE=4.0 과도한 하향
- **상황:** v6e에서 upsampling coverage를 6→4로 낮춤
- **실제:** GO:0006096 positive coverage 50% 감소 → AUPRC 0.793→0.783
- **교훈:** coverage 변경은 sparse GO term에서 큰 영향
- **규칙:** coverage 변경 시 모든 5 GO term에서 coverage 수치 출력 확인

### [FAIL-4] run_GPU script sed 치환 문법 오류
- **상황:** v6g 스크립트 생성 시 `sed` 치환으로 포맷 문자열 파괴
  ```python
  # 오류: cmd = "...".format(gpu_id, MODEL_SCRIPT = "v6g_...")
  ```
- **교훈:** script 생성은 반드시 `ast.parse()` 문법 검사 후 실행
- **규칙:** 모든 신규 Python 파일 생성 후 `python -c "import ast; ast.parse(open(f).read())"` 실행

### [FAIL-5] model-engineer agent의 잘못된 D1.1 진단
- **상황:** model-engineer가 Phase 1.5 freeze를 "PASS"로 판정
- **실제:** 코드상 올바르게 구현되었으나 실측 데이터에서 임베딩 변화 확인
- **교훈:** 코드 정적 분석만으로 동적 동작(freeze) 검증 불가
- **규칙:** freeze 관련 결정은 반드시 실험 log의 embedding diff 수치 기반

### [FAIL-6] compute_bias_score NaN 처리 누락
- **상황:** 초기 버전에서 multi_iso_genes가 비어있을 때 division by zero
- **수정:** `if n_multi == 0: return {'bias_score': np.nan, ...}`

---

## 2. 반드시 지켜야 할 규칙

### 데이터 보호
- `my_isoform_list_fixed.npy`, `my_gene_list_fixed.npy`, `my_sequence_matrix_fixed.npy` 절대 덮어쓰기 금지
- 새 데이터 파일: 날짜 suffix 추가 (예: `domain_delta_20260428.npy`)

### 버전 관리
- 새 모델 수정 전: `_backup_{DATE}.py` 생성 필수
- v6g → v7a → v7b 순서 유지

### 평가 유효성 [R9.1]
- Primary metric: **AUPRC** (sparse positive class)
- Secondary: AUROC (참고용)
- 개선 주장: Bootstrap CI n=1000 필수 [R9.4]
- "개선"의 최소 기준: Δ AUPRC > 0.02

### GO:0007204 재현성 경고
- v6g GO:0007204 AUPRC=0.591은 비정상적 lucky run (sep_ratio=1.450, 보통 ~1.07)
- v7 비교 기준: v6f 0.309 또는 v6e 0.388 사용
- 3-run 평균으로 실제 성능 측정 필요

### Phase 1 early stop 조건 (v7a SupCon)
- Triplet: active_rate < 2% × 4 epoch
- SupCon: loss < 0.01 × 4 epoch (동등한 기준 필요, 향후 조정 가능)

---

## 3. 미검증 가정 (UNTESTED)

### UNTESTED-1: sep_ratio ≥ 1.15 threshold
- Type A/B 분류 기준 1.15는 경험적 값 (v6e에서 설정)
- GO:0006096이 1.145-1.148에서 Type-B 오분류 가능
- 검증 방법: threshold=1.10으로 낮추거나 multi-sample averaging

### UNTESTED-2: SupCon τ=0.1 최적성
- 논문 기본값이나 이 데이터셋에서 최적인지 미검증
- τ∈[0.05, 0.1, 0.2] grid search 가능

### UNTESTED-3: Cross-gene negative 필요성
- Triplet에서 intra-gene negative 금지 [R2.1]
- SupCon에서는 이 제약 미적용 (배치 내 모든 negative 사용)
- intra-gene negative가 SupCon에서도 해로울 수 있음 → 검토 필요

### UNTESTED-4: DomainDelta 251d 설계
- sign transform이 최적인지 (절대값, log ratio 대안 미검증)
- DomainDelta 기여도 ablation 미실시 (no_domain_delta 조건)

### UNTESTED-5: Phase 2 focal loss α=0.10 최적성
- positive 비율 0.69%(GO:0006941)에 α=0.10이 최적인지 미검증
- [R1.2] class-balanced α 공식 미적용

---

## 4. 로그에서 반드시 확인해야 할 항목

v7a 결과 확인 시 체크리스트:
- [ ] Phase 1 SupCon loss 수렴 곡선 (감소하는가?)
- [ ] Phase 1 valid anchors/batch (≥ 5 이상인가?)
- [ ] Phase 1 후 sep_ratio (v6g Phase 1과 비교)
- [ ] Phase 2 AUPRC 수렴 곡선 (collapse 없는가?)
- [ ] Final AUPRC vs v6g 비교 (GO term별)
- [ ] GO:0006941: Phase 2 epoch 2에서 Acc jump 여전한가?
- [ ] GO:0007204: Phase 0 sep_ratio 값 (Type A/B 분류 확인)

---

## 5. 코드 품질 체크

실험 전 확인:
```bash
# 문법 검사
python -c "import ast; ast.parse(open('v7a_integrated_full_model.py').read())"

# 임포트 경로 확인
python -c "import sys; sys.path.insert(0, '.'); from prototype_contrastive import supervised_contrastive_loss; print('OK')"
```

실험 후 확인:
```bash
# 결과 파일 존재 확인
ls results_isoform/GO_*/v7a_integrated_*/

# Final AUPRC 추출
grep "\[Final\] AUROC" logs_isoform/GO_*/v7a_*.log
```
