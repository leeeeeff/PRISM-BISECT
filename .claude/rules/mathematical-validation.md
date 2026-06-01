# Mathematical Validation Rules

## 새 알고리즘 제안 전 필수 진단

### Collapse Score
collapse_score = pred_dist.std() / pred_dist.mean()
→ < 0.3: mode collapse → Axis 1 (Focal Loss 재검토)

### Gene-Bias Score
bias_score = 1 - (H(y|isoform_id) / H(y|gene_id))
→ < 0.3: gene-level 편향 → Axis 2 (Adversarial/IRM)

### Triplet Active Ratio
active_ratio = (triplet_loss > 0).float().mean()
→ < 0.05: easy triplet 지배 → Axis 3 (Hard negative mining)

### Modality Gradient Norm
각 modality의 gradient norm 비교
→ 10배 이상 차이: → Axis 4 (Gradient Modulation)

---

## 제안 시 필수 형식
변경 내용: [무엇을]
수학적 근거: [Rx.y] {핵심 수식}
해결 문제: Axis {N}
⚠️ 주의사항: {알려진 함정}
검증 방법: {어떤 지표로}

---

## Evaluation Validity [Axis 9]
- Sparse class: AUPRC primary, AUROC secondary [R9.1]
- F1 비교: 동일 threshold 조건 명시 [R9.2]
- 개선 주장: bootstrap CI (n=1000) 필수 [R9.4]
- Overall: Macro-F1 [R9.5]

## 전체 레퍼런스
tasks/reference-knowledge-base.md (9개 Axis × 수식/레퍼런스)
