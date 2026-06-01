# v6c 성능 이슈 분석 레포트
**작성일**: 2026-04-13  
**목적**: v6h 대비 v6c 성능 대폭 하락 원인 분석 및 대응 방향

---

## 핵심 발견

GO:0006941 기준:

| 모델 | AUPRC | Linear AUROC (Phase 0) | 최종 AUROC |
|------|-------|----------------------|-----------|
| v5-5 | 0.0947 | - | 0.6452 |
| v6h  | 0.2854 | - | 0.8968 |
| v6c  | **0.0585** | 0.7343 | **0.5224** |

**Phase 0 Linear AUROC=0.7343 → Phase 2 최종 AUROC=0.5224**: 학습 후 오히려 저하.

---

## Phase별 진단

### Phase 1 (Triplet, gradient scaling)

```
[Phase 1 Final] best_margin_sat=0.4%  centroid_dist=0.0706
  Silhouette (Phase 1): +0.8922   ← 클러스터링은 되었으나
  Linear AUROC:          0.8000   ← 분류 성능 자체는 유지
```

- `margin_sat=0.4%`: 트리플렛 마진 만족이 매우 낮음 → embedding이 분리되지 않음
- `active_rate < 2%` 4 epoch → early stop 발동
- **Silhouette +0.8922**: 클러스터 구조는 형성됐지만 GO term 기준 분리가 아닐 가능성

### Phase 1.5 (Linear Probing)

```
Silhouette:   +0.9804   ← 높음 (클러스터 존재)
Linear AUROC:  0.7690   ← Head 재조정 후도 제한적
```

### Phase 2 (Focal, ESM-2 frozen)

```
Epoch 1: AUPRC=0.0557 (best)
Epoch 2: AUPRC=0.0545
Epoch 3: AUPRC=0.0544
...
Epoch 8: AUPRC=0.0542  → Early stop (patience=7)
```

- **첫 epoch이 최고**: 학습이 전혀 개선되지 않음
- **AUPRC≈0.055**: 랜덤에 가까운 수준 (양성 비율 0.69%)

---

## 원인 가설

### 가설 1: ESM-2 gradient scale 0.2 → Phase 1에서 ESM-2 표현 파괴

Phase 1에서 ESM-2 gradient를 0.2×로 제한했음에도 15 epoch × 20 배치 = 300 업데이트.  
Silhouette +0.8922로 보아 어느 방향으로는 수렴했으나, 그 방향이 GO term 분류와 무관할 수 있음.

```
예상: ESM-2 표현이 triplet objective로 과최적화 → Phase 2 focal에서 복구 불가
```

**테스트**: Phase 1에서 ESM-2 완전 frozen, CNN+Gate만 학습

### 가설 2: Multi-scale CNN concat이 Phase 1에서 noise 증폭

3개 Conv1D(k=5,7,11) → concat[96] → reduce[64] → Dense(32) 구조에서  
Phase 1a 2 epoch만으로는 reduce 레이어가 수렴 부족 → random feature가 triplet signal 희석.

**테스트**: Phase 1a epoch 5으로 증가, 또는 CNN→Gate 경로만 Phase 1 학습

### 가설 3: Bidirectional gating이 정보 차단

Round 1, 2에서 gate mean≈0.5, std≈0.03~0.08 (낮음).  
`gate × feature + feature` 잔차 구조에서 gate≈0.5이면:  
`output ≈ 0.5 × feature + feature = 1.5 × feature` → 단순 스케일링, 실제 gating 없음.

**테스트**: gate entropy 모니터링, gate ×1이면 single round로 단순화

### 가설 4: SEQ_LEN=1500 + Embedding(8001,32) 메모리/배치 불균형

Phase 1에서 배치당 sequence tensor가 커서 실제 effective batch size 감소 → triplet 신호 약화.

**테스트**: SEQ_LEN=1000으로 축소, 또는 CNN 출력 캐싱

---

## 즉시 실행 가능한 수정 방향

### Option A: Phase 1에서 ESM-2 완전 Frozen (단순화)
```python
# Phase 1: CNN + Gate만 triplet 학습, ESM-2는 feature extractor로만 사용
set_esm2_trainable(feature_model, False)   # ESM-2 frozen
set_cnn_trainable(feature_model, True)
set_gate_trainable(feature_model, True)
# gradient scaling 제거 (frozen이므로 불필요)
```
→ IndexedSlices 문제 재발 없음, ESM-2 표현 보존 확실

### Option B: Phase 1a 확장 + CNN-Gate 분리 학습
```python
# Phase 1a: CNN 5 epoch (현재 2)
# Phase 1b: Gate만 3 epoch (CNN frozen, ESM-2 frozen)
# Phase 1: 전체 triplet (ESM-2 scale=0.5 → 현재 0.2보다 완화)
```

### Option C: v6h 구조 + Multi-scale CNN만 추가 (Gating 제거)
```python
# v6h의 안정적 phase 구조 유지
# CNN branch만 k=3 → multi-scale(k=5,7,11)로 교체
# Gating 제거 → 직접 concat 대신 MLP 융합
concat = concatenate([esm2_feat, cnn_feat, domain_feat])  # 기존 방식
```
→ 최소 변경으로 multi-scale 효과 검증

---

## 권장 결정 기준

v6c 5개 GO term 결과 완료 후:
- 5개 중 3개 이상 v6h 대비 하락 → **Option A** (ESM-2 frozen Phase 1)
- 2개 이하 하락, 나머지 개선 → 현재 구조 유지, 하이퍼파라미터 조정
- 전 GO term 하락 → **Option C** (Gating 제거, multi-scale CNN만)

---

*분석 작성: 2026-04-13 | 결과 업데이트 예정: v6c 완료 후*
