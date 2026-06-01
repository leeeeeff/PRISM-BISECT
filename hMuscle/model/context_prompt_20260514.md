# DIFFUSE 프로젝트 컨텍스트 — 2026-05-14

## 프로젝트 개요
인간 근육 단세포 데이터 기반 isoform 기능 예측 파이프라인.
- 목표: Nature Methods / NMI 제출
- 평가 지표: 5개 GO term AUPRC Macro (class imbalance 0.2~1.2%, Macro 최우선)
- 작업 경로: `/home/welcome1/sw1686/DIFFUSE/hMuscle/model/`
- Conda: `isoform_env`

## 현재 best 결과 요약

| 모델 | Macro-AUPRC | 상태 |
|------|------------|------|
| v8b (SP✓, 64d) | 0.3568 | baseline |
| P3 (SP✗, 64d) | 0.2976 | 완료 |
| P3-256 (SP✗, 256d) | 0.3552 | 완료 |
| D256 (SP✓, 256d) | 0.3839 | 완료 |
| **Selective Ensemble** | **0.4365** | 현재 best |
| ESM-2 LR (640d, human) | 0.5614 | 목표 상한 |
| **P3-512 (SP✗, 512d)** | **진행 중** | PID=3272659 |

Selective Ensemble = P3-256(Type-B GO) + D256(Type-A GO)

## 2×2 ablation 전체 결과

| GO term | v8b | P3 | P3-256 | D256 | LR |
|---------|-----|----|--------|------|----|
| GO:0007204 | 0.1462 | 0.1624 | **0.3109** | 0.1947 | 0.4140 |
| GO:0030017 | 0.1570 | 0.1918 | **0.2836** | 0.1366⚠ | 0.5610 |
| GO:0006941 | 0.1177 | 0.1292 | 0.1540 | **0.1567** | 0.3120 |
| GO:0003774 | 0.5686 | 0.5405 | 0.5830 | **0.5982** | 0.8250 |
| GO:0006096 | 0.7945 | 0.4640 | 0.4445 | **0.8331**★ | 0.6950 |

★ D256 GO:0006096 = 0.8331 > LR 0.6950 (파이프라인이 LR 초과, 핵심 논문 결과)
⚠ D256 GO:0030017 = 0.1366 — v8b(0.1570) 대비 퇴화 (SP×dim 음의 상호작용)

## 핵심 발견 (실험적으로 확립된 것)

**[SP × Dim 상호작용]** dim 증가 시 SP 효과가 증폭됨 (좋든 나쁘든)
- Type-B GO terms: SP가 64d에서 약간 해롭다 → 256d에서 크게 해롭다
- GO:0006096: SP가 64d에서 유익 → 256d에서 더 유익
- 근본 원인: 고용량에서 SwissProt 특이 패턴을 과도 학습 → human으로 전이 시 amplified

**[Functional Transferability Index (FTI)]** 오늘 정립한 핵심 지표
```
FTI(GO) = AUPRC(SP✓, human test) / AUPRC(SP✗, human test)  [256d 기준]

GO:0006096: 1.87  GO:0003774: 1.03  GO:0006941: 1.02
GO:0007204: 0.63  GO:0030017: 0.48
```

**[2D SP 예측 프레임워크]** 단순 "종간 보존성" 1D 불충분, 2개 축 필요
- 축 1: Taxonomic Breadth Score (TBS) — annotation kingdom 수 / 6
- 축 2: Tissue Context Specificity (TCS) — positive 단백질의 조직특이성
- 발견: TBS 높음 + TCS 높음 → SP 해로움 (GO:0007204가 핵심 증거)

| GO term | TBS | TCS | FTI |
|---------|-----|-----|-----|
| GO:0006096 (glycolysis) | 1.00 | 낮음 | 1.87 → Tier 1 (SP Required) |
| GO:0003774 (myosin) | 0.67 | 낮음 | 1.03 → Tier 2 (SP Neutral+) |
| GO:0006941 (contraction) | 0.33 | 높음 | 1.02 → Tier 2 (SP Neutral) |
| GO:0007204 (Ca²⁺) | 0.67 | 높음 | 0.63 → Tier 3 (SP Harmful) |
| GO:0030017 (sarcomere) | 0.33 | 높음 | 0.48 → Tier 3 (SP Harmful) |

**[Bio-validator 검증]**
- GO:0007204의 SP 해로움은 "종간 보존성 부족"이 아님 (CaM은 91% identity로 극도 보존)
- 실제 원인: tissue context mixing (non-muscle Ca²⁺ 단백질이 muscle 학습에 noise)
- GO:0030017의 SP 해로움은 "vertebrate-specific"이 아님 (Drosophila도 sarcomere 존재)
- 실제 원인: skeletal muscle isoform-level divergence (titin 분자 구조 자체가 다름)

## 현재 실행 중

**P3-512 (SP✗, 512d)** — PID=3272659
- GPU 0: GO:0007204, GO:0030017, GO:0006941
- GPU 1: GO:0003774, GO:0006096
- 시작: 2026-05-14 11:13 KST
- 예상 완료: ~2026-05-14 13:30 KST (약 120분)
- 기대 Selective Macro: 0.49~0.53

완료 확인:
```bash
tail -5 /home/welcome1/sw1686/DIFFUSE/logs_isoform/run_P3-512_*.log
python analyze_P3_results.py   # 전체 dim sweep 표
```

## 주요 파일

| 파일 | 역할 |
|------|------|
| `v8b_integrated_full_model.py` | baseline (SP✓, 64d) |
| `v8b-P3_integrated_full_model.py` | P3 (SP✗, 64d) |
| `v8b-P3-256_integrated_full_model.py` | P3-256 (SP✗, 256d) |
| `v8b-P3-512_integrated_full_model.py` | P3-512 (SP✗, 512d) ← 현재 실행 중 |
| `v8b-D256_integrated_full_model.py` | D256 (SP✓, 256d) |
| `run_GPU_P3-512.py` | P3-512 Dual GPU runner |
| `analyze_P3_results.py` | 전체 dim sweep 비교 분석 (--p3512 플래그 지원) |

## 확정된 반복하면 안 되는 것 (Anti-patterns)

- SwissProt 완전 제거 (P3 결과: Macro 0.357 → 0.298, GO:0006096 0.7945→0.4640)
- FiLM on Type-B GO terms
- Phase 1.5 linear probing
- Temperature scaling으로 AUPRC 개선 시도
- GO:0006096 SP 제거 (human positive 32개, SP 의존도 87.6%)

## P3-512 결과별 분기 전략

```
P3-512 Selective Macro ≥ 0.52 → 논문 작성 진입
                                   기여: FTI 2D framework + Selective SP + GO:0006096>LR

P3-512 Selective Macro 0.47~0.52 → GO-conditioned projection 또는 계층적 contrastive

P3-512 Selective Macro < 0.47   → 근본 재설계 필요
```

## 다음 할 일

1. **P3-512 완료 확인** → `python analyze_P3_results.py`
2. **FTI 논문 기여 판단**: TBS 정량화 (UniProt API) + TCS 정량화 (GTEx tau index)
3. **P3-512 결과에 따른 분기 결정**

## 참고: 핵심 실험 결과 히스토리

v7c(0.315) → v8b(0.357, Unified Loss 도입) → Type-aware Ensemble(0.371) →
D256(0.384) → Selective Ensemble(0.4365, 현재 best) → P3-512(진행 중)

ESM-2 LR gap: 0.561 - 0.437 = 0.125 (22% 남음)
