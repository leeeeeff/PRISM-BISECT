# 레포트 06: 남은 Ablation 계획
**작성일:** 2026-04-29  
**현재 상태:** Exp 2a (v7a-SupCon) 실행 중  
**기준 모델:** v6g (Macro-AUPRC 0.465)

---

## 1. 전체 실험 흐름

```
[현재] Exp 2a: v7a-SupCon (실행 중)
          ↓
   Δ AUPRC > +0.02?
   ↓ YES              ↓ NO
Exp 2b: Proto-k1   Triplet 유지 + hard negative 강화
          ↓
   k>1 이 Type-B에 유리?
   ↓ YES              ↓ NO
Exp 2c: Proto-kN   k=1 고정으로 진행
          ↓
   [전처리 필요] acorde network 구축
          ↓
Exp 3a: v7b-LP (α 탐색)
          ↓
Exp 3b: v7b-LP (ci_threshold 탐색)
          ↓
Exp 4: v6g vs v7b 최종 비교 (10-run Bootstrap CI)
```

---

## 2. 실험별 상세

### Exp 2a: v7a-SupCon ← 현재 진행 중
| 항목 | 내용 |
|------|------|
| **가설** | SupCon이 Triplet보다 배치 내 positive 쌍을 더 효율적으로 학습 |
| **변경** | Phase 1: Triplet → SupCon (τ=0.1) |
| **기준** | v6g Macro-AUPRC 0.465 |
| **acceptance** | Δ Macro-AUPRC > +0.02 (≥ 0.485) |
| **falsification** | Δ Macro-AUPRC < 0 → SupCon 포기 |
| **실행 수** | 1 (5 GO terms 동시) → 안정 확인 후 3-run |
| **상태** | 🔄 GPU 0+1 실행 중 |

**주의:** GO:0007204 비교 기준은 v6g 0.591이 아닌 v6f 0.309 사용

---

### Exp 2b: v7a-Proto-k1 (SupCon PASS 후)
| 항목 | 내용 |
|------|------|
| **가설** | k=1 prototype이 positive centroid를 명시적으로 유지해 희소 GO term에 유리 |
| **변경** | Phase 1: SupCon → ProtoConLoss (k=1 강제) |
| **코드** | `prototype_contrastive.py` PrototypeContrastiveLoss(n_prototypes=1) 사용 |
| **acceptance** | Type-A: AUPRC ≥ v7a-SupCon - 0.01 (손실 없음) |
|  | Type-B: 유의미한 방향 (Exp 2c 필요성 판단용) |
| **실행 수** | 3-run |
| **의존성** | Exp 2a PASS |

---

### Exp 2c: v7a-Proto-kN (Exp 2b 후)
| 항목 | 내용 |
|------|------|
| **가설** | Gap Statistic이 Type-B GO term의 서브그룹 수 k를 올바르게 선택 |
| **변경** | Phase 1: ProtoConLoss, k = determine_k_gap_statistic() |
| **k_max** | 5 |
| **acceptance** | Type-B: Proto-kN > Proto-k1 by ≥ 0.02 (≥ 2/3 Type-B terms) |
|  | Gap Statistic이 Type-B에서 k > 1 선택 |
|  | L_diversity epoch15 < epoch1 (prototypes 분리됨) |
| **falsification** | kN이 k1보다 나쁘면 → k=1 고정, Gap Statistic 폐기 |
| **실행 수** | 5-run (k 선택 안정성 확인) |
| **의존성** | Exp 2b |

---

### [전처리] acorde Network 구축 (Exp 3 전)
| 항목 | 내용 |
|------|------|
| **목적** | hMuscle 24 샘플 기반 co-expression network (bootstrap CI 포함) |
| **방법** | acorde (NatComm 2022): Percentile correlation + bootstrap |
| **파일** | `hMuscle/preprocessing/build_acorde_network.py` (미작성) |
| **출력** | `hMuscle/data/acorde_network_{ci_threshold}.npz` |
| **소요 시간** | ~4시간 (CPU) |
| **의존성** | Exp 2c 완료 전 병렬 시작 가능 |

---

### Exp 3a: v7b-LP α 탐색 (acorde 완료 후)
| 항목 | 내용 |
|------|------|
| **가설** | acorde network가 기존 LP (alpha=0.0)보다 beneficial LP 활성화 가능 |
| **변경** | 기존 LP 제거된 자리에 acorde network + α ∈ [0.1, 0.2, 0.3, 0.5] 시험 |
| **ci_threshold** | 0.3 (기본값) |
| **acceptance** | LP가 ≥ 3/5 GO term에서 AUPRC 개선 (≥ 0.01 each) |
| **falsification** | < 3/5 GO term 개선 → α=0.0 고정, LP 불가 선언 |
| **실행 수** | 4α × 5GO = 20 sub-runs |
| **의존성** | acorde network, Exp 2c |

**근거:** 기존 LP 실패 이유는 Pearson correlation의 희소 데이터 부적합성
acorde의 Percentile correlation + bootstrap CI는 24 샘플에서도 robust한 네트워크 구축 가능

---

### Exp 3b: v7b-LP ci_threshold 탐색
| 항목 | 내용 |
|------|------|
| **가설** | ci_threshold=0.3이 edge 신뢰도 vs 밀도 트레이드오프 최적점 |
| **변경** | ci_threshold ∈ [0.2, 0.3, 0.4, 0.5] (best α 고정) |
| **acceptance** | ci_threshold=0.3이 ≥ 3/5 GO term에서 최고 |
| **falsification** | 모두 유사 (Δ < 0.01) → ci_threshold=0.3 기본값 유지 |
| **실행 수** | 4 × 5GO = 20 sub-runs |
| **의존성** | Exp 3a |

---

### Exp 4: 최종 비교 v6g vs v7b (모든 ablation 완료 후)
| 항목 | 내용 |
|------|------|
| **비교** | v6g (현재 best) vs v7b (best hyperparameters from Exp 2-3) |
| **통계** | Bootstrap CI n=1000, Δ AUPRC lower bound > 0.01 |
| **acceptance** | Macro-AUPRC > +0.03, ≥ 4/5 GO term 개선, 악화 GO term < -0.05 없음 |
| **falsification** | v7b Δ < 0.02 → v6g로 롤백 |
| **실행 수** | 10-run (5 v6g + 5 v7b) |
| **의존성** | 모든 이전 실험 |

---

## 3. GPU 시간 예산

| 실험 | GPU 시간 | 상태 |
|------|----------|------|
| Exp 2a (v7a-SupCon) | ~2h | 🔄 진행 중 |
| Exp 2b (Proto-k1) | ~6h | 대기 |
| Exp 2c (Proto-kN) | ~10h | 대기 |
| Exp 3a (LP α) | ~10h | 대기 |
| Exp 3b (LP ci) | ~8h | 대기 |
| Exp 4 (최종) | ~20h | 대기 |
| **합계** | **~56h** | (~2.5일, 2 GPU) |

---

## 4. 결정 트리 (결과에 따른 분기)

```
Exp 2a SupCon PASS?
  YES → Exp 2b → 2c → 3a → 3b → 4
  NO  → Triplet 유지 + 아래 대안 탐색:
        [Alt A] Hard negative mining 강화 (margin 감소, k-NN 기반 mining)
        [Alt B] SupConLoss margin 변형 (supervised soft triplet)
        [Alt C] Axis 변경 → IRM [R2.2] (domain invariance)

Exp 3a LP PASS?
  YES → best α로 Exp 3b
  NO  → α=0.0 고정, v7b = v7a-Proto (LP 없음)
        → "24 샘플로 LP 불충분" 결론 명시

Exp 4 PASS?
  YES → v7b 배포, 논문 방법론 섹션 작성
  NO  → v6g 유지, 다른 Axis 탐색:
        [Alt D] Spectral Norm [R7.3]
        [Alt E] Gradient Reversal Layer [R2.4]
```

---

## 5. 아직 미작성 코드

| 파일 | 내용 | 우선순위 |
|------|------|----------|
| `preprocessing/build_acorde_network.py` | acorde co-expression network 구축 | P1 (Exp 3 전) |
| `v7a_proto_k1_integrated_full_model.py` | Exp 2b용 모델 | P2 (Exp 2a 후) |
| `v7a_proto_kN_integrated_full_model.py` | Exp 2c용 모델 | P3 |
| `v7b_integrated_full_model.py` | v7a + acorde LP | P3 |
| `results_isoform/ablation_results.csv` | 전체 실험 결과 통합 CSV | 필요 시 |

---

## 6. 논문 기여도 관점에서의 ablation 의미

각 실험이 논문 (Nature Methods / NMI)에서 기여하는 바:

| 실험 | 논문 주장 |
|------|----------|
| Exp 2a (SupCon) | "Triplet보다 SupCon이 희소 isoform GO prediction에 효과적" |
| Exp 2b/c (Proto) | "이질적 GO term에서 다중 prototype이 subgroup 분리에 필요" |
| Exp 3 (acorde LP) | "Bootstrap CI 기반 co-expression network가 희소 데이터에서 LP 가능하게 함" |
| Exp 4 (전체) | "v7b는 v6g 대비 Macro-AUPRC +X.XX (95% CI [a, b])" |

**주의:** Exp 4 falsification (Δ < 0.02) 시 v7 전체를 포기하고 Axis 변경 필요.
논문 deadline이 있다면 Exp 2a 결과로 SupCon 기여만 주장 가능 (simpler claim).
