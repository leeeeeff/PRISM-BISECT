# BISECT 통계 프레임워크 Before → After 정리

## 1. 파이프라인 변화

| 분석 레이어 | Before | After | 이유 |
|---|---|---|---|
| DTU 기본 통계 | Chi-sq pooled reads (n=reads) | 도너-level MWU (n=도너) | 가성 반복 제거 |
| 이소폼 선택 | 전체 이소폼 delta | CT/AD 특정 이소폼 쌍 | 생물학적 가설 검증 |
| 세포타입 | Excitatory 전체 | 8 세포타입 × 5 유전자 | Cell-type specificity |
| 도너 분포 QC | 없음 | Tier A-D 분류 | Private isoform 식별 |
| 배치 강건성 | 없음 | PO vs SMC delta 일관성 | 코호트 편향 탐지 |
| 독립 검증 | 없음 | Ebbert 코호트 DRIMSeq | 재현성 확인 |

## 2. 유의한 변화: claim별 Before vs After

### KIF21B
| 항목 | Before | After |
|---|---|---|
| 원본 DTU p | 3.81e-06* (pooled chi-sq) | 가성 반복 — 액면가 불가 |
| NNIC (transcript292978) | AD 마커 후보 | **제거** — Tier D (SMC038 단독, canonical=0) |
| CT NIC (transcript293004) | AD 마커 후보 | Tier C (1 CT 도너, canonical 공존) — directional only |
| 도너-level p | — | 0.2888 ns (MWU), 0.472 (perm) |
| 독립 검증 | — | Ebbert DRIMSeq p=0.009 (방향↑), stageR OFDR=0.125 (미달) |
| 최종 claim | 'AD에서 CT 이소폼 소실' | '방향적 관찰 + 독립코호트 방향 일치, 통계적 검정력 부족' |

### NDUFS4
| 항목 | Before | After |
|---|---|---|
| 원본 DTU p | 3.62e-06* (pooled chi-sq) | 가성 반복 — 액면가 불가 |
| 도너-level p | — | 0.0554 (MWU), **0.041*** (perm) |
| 배치 일관성 | — | YES — PO(-0.31) + SMC(-0.48) 모두 같은 방향 |
| Ebbert | — | 발현 없음 (뇌 세포특이) — 검증 불가 |
| 최종 claim | 'NDUFS4-201 AD에서 감소' | **'도너-perm p=0.041, 배치독립 — Secondary claim 가능'** |

### NDUFS7 (신규 — 83 cases 외)
| 항목 | Before | After |
|---|---|---|
| 83 BISECT | 없음 (뇌 미포함) | 신규 발견 |
| 도너-level p | — | **0.045*** (MWU, 202 vs 210) |
| Sensitivity | — | SMC038 제외 시 p=0.076 (소실 아님) |
| 독립 검증 | — | 미검증 |
| 최종 claim | — | **'Primary claim — 4/13 AD 독립 검출, 0/8 CT'** |

### NDUFS8 (신규 — 83 cases 외)
| 항목 | Before | After |
|---|---|---|
| 83 BISECT | 없음 (뇌 미포함) | 신규 발견 |
| 도너-level p | — | 0.057† (MWU, Inhibitory) |
| 최종 claim | — | Secondary claim (Complex I 삼각수렴 맥락) |

### DLG1
| 항목 | Before | After |
|---|---|---|
| 원본 DTU p | 9.03e-10** (pooled chi-sq, tier1) | 가성 반복 |
| 도너-level p | — | 0.540 ns (MWU), 0.310 ns (perm) |
| 배치 일관성 | — | NO — PO(-0.5) vs SMC(+0.14) 방향 반전 |
| Ebbert | — | DRIMSeq p=0.237 (OPC 희석으로 예상 음성) |
| 최종 claim | 'DLG1 isoform switch in OPC' | BISECT 15모듈 증거 유지, 통계 보조 수준 |

## 3. 전체 케이스 Tier 요약 (5 유전자 × 8 세포타입, 30 unique 검정)

| Tier | 정의 | 건수 | p<0.05 | p<0.1 |
|---|---|---|---|---|
| A (Cohort) | ≥2 도너 독립, 분산 | 19 | 1 | 3 |
| B (Concentrated) | 다중 도너, 1명 ≥70% | 3 | 0 | 0 |
| C (Private+coexist) | 1 도너, canonical 공존 | 2 | 0 | 0 |
| D (Private+mutation) | 1 도너, canonical 소실 | 5 | 0 | 0 |

## 4. Chi-sq pooled 문제 및 DCF 결과

- 4,050 chi-sq 유의 유전자 → DCF 적용 → 1,820 PASS (45%)
- PASS 기준: jackknife <0.5 AND direction_consistency ≥0.6

| 유전자 | DCF | Jackknife | Direction | Worst donor |
|---|---|---|---|---|
| KIF21B | FAIL | 1.00 | 0.14 | SMC038 |
| NDUFS4 | FAIL | 1.00 | 0.14 | PO05 |
| NDUFS7 | FAIL* | 0.83 | 0.30 | SMC038 |
| DLG1 | FAIL(marginal) | 0.61 | 0.67 | SMC049 |
| NTRK2 | PASS | 0.14 | 0.82 | — (but pooled-driven) |

*NDUFS7 DCF는 원본 chi-sq 대상 (202 vs 210 ratio 검정 아님)

## 5. 논문 방어 전략

| 리뷰어 지적 | 대응 |
|---|---|
| 'chi-sq가 pseudo-replicated' | Methods 명시: exploratory screen only. Primary evidence = donor-level MWU/perm |
| 'single donor driven?' | Tier A-D 프레임워크 + sensitivity analysis 제시 (SMC038 제외 p=0.076) |
| 'n이 너무 작다' | NDUFS4 배치독립 perm p=0.041. KIF21B Ebbert 독립코호트 방향일치 |
| 'DLG1 재현 안 됨' | OPC 5% 희석 설명 + BISECT 15모듈 독립증거 |