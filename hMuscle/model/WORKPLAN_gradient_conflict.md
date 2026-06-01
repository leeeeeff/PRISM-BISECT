# Gradient Conflict Resolution — 실행 계획

## 진단 요약 (2026-05-18 기준)

| 실험 | Macro AUPRC | Mean bias | 판정 |
|------|------------|-----------|------|
| v10-B (기준선) | 0.6935 | 0.0938 | GENE-LEVEL |
| v11-Slim (λ=0.1, 동시) | 0.4231 | 0.1675 | ISOFORM-SPEC |
| v12-Sequential | ~0.25 | ~0.19 | 순서만 바꾸면 더 나빠짐 |

**확정된 근본 원인**: GO label(gene-level) ↔ Canonical label(isoform-level) label space 불일치.  
동일 backbone을 두 반대 방향으로 당기는 gradient가 충돌함.

---

## 전체 로드맵

```
[Phase 0] PCGrad 진단 (1일)          ← 즉시 시작
     │
     ├─ AUPRC > 0.65 → PCGrad 효과 있음
     │       └─ [Phase 2A] Splice-aware auxiliary (1~2주)
     │
     └─ AUPRC < 0.50 → Label space 문제 확정
             └─ [Phase 2B] MIL 재구성 (2~3일)
                     └─ [Phase 3] Splice auxiliary + NeXtProt (병렬)

[Phase 1B] NeXtProt 커버리지 확인 (병렬, 0.5일)
[Phase 4] 뇌조직 검증 + 논문 업데이트 (Phase 2 완료 후)
```

---

## Phase 0: PCGrad TF 구현 (즉시, ~1일)

### 목적
Gradient conflict 가설 직접 검증.  
PCGrad가 효과 없으면 → label space 문제가 주원인 확정 → MIL 필연.  
PCGrad가 효과 있으면 → gradient 충돌이 주원인 → 구조적으로 해결 가능.

### 파일명
`v13_pcgrad.py`

### 아키텍처 (v11-Slim과 동일)
```
ESM-2 (640d)
    ↓
shared_backbone: Dense(256→BN→Drop0.3→Dense128)
    ├─ go_head: Drop0.2 → Dense(64) → Dense(1, sigmoid)
    └─ can_head: Dense(32) → Dense(1, sigmoid)
```

### PCGrad 구현 핵심 (TensorFlow GradientTape)
```python
@tf.function
def pcgrad_train_step(model, x, y_go, y_can, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        go_pred, can_pred = model(x, training=True)
        L_go  = focal_loss(y_go, go_pred)
        L_can = bce(y_can, can_pred)

    shared_vars = model.backbone.trainable_variables
    g_go  = tape.gradient(L_go,  shared_vars)
    g_can = tape.gradient(L_can, shared_vars)
    del tape

    # PCGrad projection: g_go → normal plane of g_can (if conflicting)
    g_go_proj = []
    for g1, g2 in zip(g_go, g_can):
        g1_flat = tf.reshape(g1, [-1])
        g2_flat = tf.reshape(g2, [-1])
        dot = tf.reduce_sum(g1_flat * g2_flat)
        if dot < 0:
            g2_sq = tf.reduce_sum(g2_flat * g2_flat) + 1e-12
            g1 = g1 - tf.reshape((dot / g2_sq) * g2_flat, tf.shape(g1))
        g_go_proj.append(g1)

    # 대칭 투영: g_can도 같은 방식으로 g_go 방향 성분 제거
    g_can_proj = []
    for g1, g2 in zip(g_can, g_go):
        g1_flat = tf.reshape(g1, [-1])
        g2_flat = tf.reshape(g2, [-1])
        dot = tf.reduce_sum(g1_flat * g2_flat)
        if dot < 0:
            g2_sq = tf.reduce_sum(g2_flat * g2_flat) + 1e-12
            g1 = g1 - tf.reshape((dot / g2_sq) * g2_flat, tf.shape(g1))
        g_can_proj.append(g1)

    # 평균 projected gradient를 backbone에 적용
    g_shared = [(a + b) / 2.0 for a, b in zip(g_go_proj, g_can_proj)]

    # head gradient는 원래 그대로
    go_head_vars  = model.go_head.trainable_variables
    can_head_vars = model.can_head.trainable_variables

    # 실제로는 한 번의 tape에서 모두 계산해야 함 — 아래는 pseudo-code
    # optimizer.apply_gradients(zip(g_shared + g_go_heads + g_can_heads, all_vars))
```

### 진단 지표 (추가 로깅)
```python
# 학습 중 매 epoch gradient conflict 비율 측정
conflict_ratio = tf.reduce_mean(
    tf.cast(tf.reduce_sum(g_go_flat * g_can_flat, axis=-1) < 0, tf.float32)
)
# conflict_ratio > 0.5 → 진단 확정
```

### 평가
- 13 GO term × 5 seeds (v10-B와 동일 조건)
- 기준: Type-B macro AUPRC 0.6935 (v10-B)
- JSON: `reports/v13_pcgrad/v13_pcgrad_{timestamp}.json`

### 결과 해석
| AUPRC | bias_score | 판정 | 다음 단계 |
|-------|-----------|------|-----------|
| > 0.65 | > 0.12 | PCGrad 효과 있음 | Phase 2A (Splice auxiliary) |
| 0.50~0.65 | > 0.12 | 부분 효과 | MIL + PCGrad 결합 |
| < 0.50 | any | Label space 문제 확정 | Phase 2B (MIL) |

---

## Phase 1B: NeXtProt 커버리지 확인 (병렬, 0.5일)

### 목적
이미 isoform-level GO annotation이 존재하는지 확인.  
GO가 아닌 다른 label system(NeXtProt functional annotation)을 쓸 경우  
label space 문제 자체가 사라짐.

### 확인 사항
```bash
# NeXtProt isoform-specific annotation API
curl "https://api.nextprot.org/entry/NX_P15090/isoform-function" | jq '.annotationList | length'

# 우리 13 GO term 해당 단백질 중 NeXtProt isoform-level annotation 보유 비율
# > 30%면 Phase 2에서 대안 label source로 검토 가능
```

### 결과에 따른 행동
- isoform-level annotation 충분 (> 30%): 논문에 "annotation 한계" 섹션 추가 + NeXtProt 활용 방향 제시
- annotation 부족 (< 10%): MIL이 유일한 현실적 해결책 → Phase 2B 강화

---

## Phase 2A: Splice-aware Self-supervised Auxiliary (PCGrad 성공 시, 1~2주)

### 목적
GO gradient와 직교(orthogonal)하는 isoform-specific supervision signal 도입.  
Long-read sequencing 데이터를 직접 활용하는 첫 구조.

### 아이디어
```
ESM-2 (640d per isoform)
    ↓
shared_backbone
    ├─ GO head (focal loss, gene-level label)
    └─ Splice prediction head: f(x_i) - f(x_j) ≈ splice_delta(i,j)
           [isoform pair (i,j)의 exon inclusion 차이 예측]
```

### splice_delta 생성 (Phase 1 선행 필요)
```
GENCODE v44 GTF → exon cluster 정의 (150 clusters)
RefSeq NM_ → ENST 매핑 (biomart)
각 isoform: 150-dim binary exon inclusion vector
splice_delta(i,j) = exon_vec_i - exon_vec_j  (연속값 또는 이진)
```

### 핵심 특성
- Splice delta = isoform 고유 exon 조합 → GO label과 직교
- Self-supervised: label이 annotation 아닌 sequencing data에서 자동 생성
- 논문 contribution: "Long-read sequencing → isoform function 직접 학습" 스토리

---

## Phase 2B: MIL 재구성 (PCGrad 실패 시, 2~3일)

### 목적
GO label의 gene-level 특성을 모델 구조에 명시적으로 반영.  
"gene G의 isoform 중 최소 1개가 GO:X를 수행한다" → max-pooling MIL.

### 수식
```
# 현재 (v10-B):  L_GO = BCE(f(x_i), y_gene_i)  [모든 isoform에 동일 label]
# MIL:           L_GO = BCE(max_{i∈G} f(x_i), y_G)  [gene G에 대해 1번만]
# 또는:          L_GO = BCE(attention_pool({f(x_i)}_{i∈G}), y_G)
```

### 구현 파일명
`v13_mil.py`

### 핵심 변경사항
```python
# Gene grouping (train step 내)
gene_to_idxs = {}
for i, g in enumerate(gene_ids_batch):
    gene_to_idxs.setdefault(g.numpy(), []).append(i)

# MIL max-pooling
L_go_total = 0.0
for gene_id, idxs in gene_to_idxs.items():
    isoform_preds = go_preds[idxs]             # (n_iso, 1)
    gene_pred = tf.reduce_max(isoform_preds)   # max over isoforms
    y_gene = y_go[idxs[0]]                     # gene-level label
    L_go_total += focal_loss(y_gene, gene_pred)
L_go = L_go_total / len(gene_to_idxs)

# Canonical loss는 isoform-level 그대로 유지
L_can = bce(y_can, can_preds)
```

### 생물학적 근거
- "Gene G가 GO:X에 관여" = "G의 isoform 중 적어도 하나가 수행"
- GO annotation 자체의 gene-level 특성을 수식에 정확히 반영
- within-gene gradient가 이제 반대 방향이 아님:
  - 기존: 모든 isoform에 y=1 → 같은 gene isoform을 같은 방향으로 밀어붙임
  - MIL: max 취하는 canonical isoform만 y=1에 맞춤, 나머지는 free

### 평가 기준
| 결과 | 해석 |
|------|------|
| AUPRC > 0.65 AND bias > 0.12 | MIL 성공 — 핵심 기여 |
| AUPRC > 0.65 AND bias ≈ 0.09 | AUPRC 회복만, bias 미개선 → 추가 필요 |
| AUPRC < 0.50 | label 문제 이외 다른 원인 존재 → 진단 필요 |

---

## Phase 3: 뇌조직 검증 + 논문 업데이트 (Phase 2 완료 후)

### 뇌조직 독립 검증
```bash
# 데이터 경로 확인 후
conda run -n isoform_env python v_brain_eval.py --model best_v13
# 근육: GO:0006941, GO:0006936, ...
# 뇌: GO:0007268, GO:0045202, ...
```

### 논문 업데이트
- Table 1: XGBoost < LR < v10-B < v13-MIL (or v13-PCGrad)
- Figure 2: bias_score 개선 trajectory
- Methods: MIL formulation 수식 포함
- Contribution: "isoform-resolution function prediction with biologically-grounded MIL supervision"

---

## 현재 작업 체크리스트

### ✅ 완료
- [x] v10-B 기준선 확립 (Macro AUPRC 0.6935, bias 0.0938)
- [x] Domain matrix 실패 원인 확정 (λ=0 진단)
- [x] v11-Slim: Canonical aux TRADE-OFF 확인 (AUPRC -0.30, bias +0.07)
- [x] v12-Sequential: Sequential 방식이 더 나쁨 확인 (GO gradient = regularizer)
- [x] 근본 원인 확정: GO(gene-level) ↔ Canonical(isoform-level) label space 충돌
- [x] 7가지 auxiliary task 실패 분석 완료
- [x] 작업 계획 수립

### 🔲 진행 예정
- [ ] **[즉시]** v13_pcgrad.py 구현 및 13-term 실험 (1일)
- [ ] **[병렬]** NeXtProt isoform-level annotation 커버리지 확인 (0.5일)
- [ ] **[조건부 A]** v13_splice_aux.py (PCGrad 성공 시, 1~2주)
- [ ] **[조건부 B]** v13_mil.py (PCGrad 실패 시, 2~3일)
- [ ] Phase 1: splice_delta_v3 생성 (RefSeq→GENCODE 매핑)
- [ ] 뇌조직 독립 검증
- [ ] 논문 Table 1, Methods 업데이트
- [ ] Bootstrap CI (n=500, gene-block) 전체 모델 비교

---

## 핵심 제약 (절대 위반 금지)

- `my_isoform_list_fixed.npy`, `my_gene_list_fixed.npy`: 읽기 전용
- AUPRC bootstrap CI: gene-block resampling [R9.4]
- Framework: TensorFlow/Keras (PCGrad = GradientTape, PyTorch 라이브러리 사용 불가)
- Gene context: attention/gating만 허용, 직접 concatenation 금지 [R2.1]
