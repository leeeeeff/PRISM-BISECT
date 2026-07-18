# DLG1 OPC 배치 방향 반전 원인 분석

**날짜**: 2026-06-23  
**결론**: 배치 방향 반전은 real biological batch effect가 아닌 **극단적 read sparsity + single-donor dominance artifact**

---

## 원본 데이터 (donor-level, OPC, transcript319500.chr3.nnic 기준)

| Donor | Cond | Batch | Total reads | CT-iso reads | CT-iso frac |
|-------|------|-------|-------------|--------------|-------------|
| PO05  | AD   | PO    | 2           | 0            | 0.000       |
| PO11  | AD   | PO    | 16          | 0            | 0.000       |
| PO13  | CT   | PO    | **34**      | **34**       | **1.000**   |
| PO15  | CT   | PO    | 0           | 0            | nan         |
| PO20  | CT   | PO    | 0           | 0            | nan         |
| PO23  | CT   | PO    | 1           | 0            | 0.000       |
| PO28  | AD   | PO    | 2           | 0            | 0.000       |
| PO41  | AD   | PO    | 3           | 0            | 0.000       |
| PO42  | AD   | PO    | 4           | 0            | 0.000       |
| SMC027 | CT  | SMC  | 1           | 0            | 0.000       |
| SMC029 | AD  | SMC  | **9**       | **5**        | **0.556**   |
| SMC030 | CT  | SMC  | 6           | 0            | 0.000       |
| SMC032 | AD  | SMC  | 3           | 0            | 0.000       |
| SMC033 | CT  | SMC  | 0           | 0            | nan         |
| SMC036 | AD  | SMC  | 2           | 0            | 0.000       |
| SMC038–049 | AD | SMC | 0        | 0            | nan         |
| SMC052 | CT  | SMC  | 0           | 0            | nan         |

---

## 배치 delta 계산 세부 과정

### PO batch delta = −0.500
- PO AD donors (reads>0): PO05/PO11/PO28/PO41/PO42 → CT-iso frac = [0.0, 0.0, 0.0, 0.0, 0.0] → mean = **0.000**
- PO CT donors (reads>0): PO13/PO23 → CT-iso frac = [1.0, 0.0] → mean = **0.500**
- Delta = 0.000 − 0.500 = **−0.500**

**PO13 단일 도너가 PO CT mean 전체를 결정**: 34 reads가 모두 transcript319500에 매핑됨 (1.0 fraction). PO23은 1 read도 매핑 안 됨.

### SMC batch delta = +0.139
- SMC AD donors (reads>0): SMC029/SMC032/SMC036/SMC041 → CT-iso frac = [0.556, 0.0, 0.0, 0.0] → mean = **0.139**
- SMC CT donors (reads>0): SMC027/SMC030 → CT-iso frac = [0.0, 0.0] → mean = **0.000**
- Delta = 0.139 − 0.000 = **+0.139**

**SMC029 단일 AD 도너가 SMC AD mean 전체를 결정**: 9 reads 중 5개가 transcript319500에 매핑됨.

---

## 결론

### "배치 방향 반전"의 실제 원인

1. **극단적 read sparsity**: OPC에서 DLG1 유전자는 도너당 최대 34 reads (대부분 0-6)
2. **Single-donor dominance**: PO batch 전체 CT signal = PO13 한 명 / SMC batch AD signal = SMC029 한 명
3. **Random sampling noise**: 34 reads에서 모두 동일 transcript로 매핑될 확률은 이 transcript가 dominant일 때 높음 (PO13 CTL, 우연히 transcript319500 dominant)
4. **이건 batch effect가 아님**: PO13 CTL과 SMC029 AD가 우연히 반대 방향으로 dominant transcript를 가진 것

### AD isoform (DLG1-201) 검출 실패

**DLG1-201은 모든 도너에서 reads = 0**

pooled chi-sq p=9.0×10⁻¹⁰은 transcript319500의 AD pool 내 감소를 감지한 것이지, DLG1-201의 증가를 감지한 게 아님. 즉:
- chi-sq가 발견한 것: pooled AD에서 transcript319500의 상대적 비율 감소
- 실제로 없는 것: AD에서 DLG1-201 증가

### DLG1 OPC DTU 신호의 진단

| 검사 | 결과 |
|------|------|
| 도너당 최대 reads | 34 (PO13) |
| AD isoform reads | 0 (모든 도너) |
| 실질적 n (reads>5) | PO13(CT), SMC029(AD), SMC030(CT) = 3명 |
| 배치 방향 일관성 | PO −0.5 vs SMC +0.14 (반전) |
| 실제 원인 | read noise at n<5 per donor |

---

## 권고사항

**DLG1은 Tier A에서 제외되어야 함**

- Tier D (탐색적, donor-level 지지 없음)로 강등 또는
- Supplementary only로 이동 + 명확한 caveats

원고 §3.10 Table에서 DLG1 행 삭제 또는 "Data insufficient for donor-level analysis (max 34 reads/donor in OPC; DLG1-201 not detected in any individual donor)" 각주 추가.

BISECT 메커니즘 분석 (PDZ domain gain)은 별도 섹션에서 가설로 제시 가능하되, pooled chi-sq 통계와 함께 제시하면 안 됨.
