# NatComm Submission-Prep Plan (natcomm_v0.md)

> **상태: 계획만. 실행 보류.** manuscript.md 규칙 준수 — 압축은 **발견이 안정된 뒤/제출 직전**에만, 그리고 **삭제가 아니라 본문→supplementary 이동**(정보 손실 금지). 이 문서는 그때 실행할 keep/compress/move 명세.
> 작성 2026-07-16. 근거: 당일 전체 구조·분량 감사(본문 20,677단어 = NC ~5,000 한도의 4×).

## 압축 판단 기준 = frame #2
현재 title/Abstract = "**maps the limits of PLM isoform resolution + AD Complex I**"(경계 측정 + 적용). 이 논지에 하중을 지지 *않는* frame #1 material("PRISM이 여러 유용한 일을 한다")이 우선 압축 대상. positioning이 압축 판단 기준을 제공한다.

## 분량 예산 (현재 → 목표)
| 블록 | 현재(w) | 목표(w) | 비율 |
|---|---|---|---|
| Abstract | 1,103 | ~200 | 저널 하드리밋 |
| Introduction | 701 | ~550 | +경계 예고, trim |
| Results | 13,104 | ~4,000 | 핵심 재편 |
| Discussion | 5,769 | ~1,300 | tightening |
| **본문 합계** | **20,677** | **~6,000** | ~3.4× 축소 |
| Methods | 7,076 | 유지(별도 한도) | — |
| Figure Legends | 6,458 | trim ~4,000 | — |
| Suppl Notes | 6,247 | **증가**(본문서 이동분 수용) | — |

(NC Article 본문 유연 상한 ~8,000까지 있으나, 방어적으로 ~6,000 목표. 최종은 editor 재량.)

## 섹션별 keep / compress / move
| 섹션 | 현재(w) | 조치 | 목표 | 근거(frame #2) |
|---|---|---|---|---|
| Abstract | 1,103 | **COMPRESS** | 200 | 하드리밋. thesis 3문장+핵심수치만 |
| Introduction | 701 | KEEP+**ADD** | 550 | **경계 예고 1문장 추가**(현재 frame#1 setup만). trim |
| §1 encode isoform features | 720 | KEEP-trim | 500 | core(encoding 주장) |
| §2 gains over domain tools | 581 | **MERGE→§1** | (0) | frame#1 SOTA. 0.108 대비만 §1에 흡수 |
| §3 mean-pooling bottleneck | 391 | KEEP | 350 | core(문제 진단) |
| §4 δ_layer | 776 | COMPRESS | 500 | core 도구지만 T_ψ ablation 등 detail→S_arch |
| §4b within-gene metrics | 665 | **KEEP** | 600 | **CORE novelty축**. 손대지 말 것 |
| §4c anatomical dissection | 464 | COMPRESS | 200 | 요약만 본문, detail→S_dissection(이미 존재) |
| **NEW: 경계 전용 섹션** | — | **CREATE** | 500 | 아래 "구조적 이동" 참조 |
| §5 non-overlapping space | **4,153** | **HEAVY COMPRESS+MOVE** | 600 | **최대 tangential.** 아래 참조 |
| §6 AD transfer+ceiling | 2,643 | **SPLIT** | 700 | ceiling증거→경계섹션, transfer+적용만 잔류 |
| §7 BISECT Complex I | 1,446 | KEEP-trim | 900 | CORE 발견 |
| §8 replication | 1,263 | KEEP-trim | 700 | CORE 검증 |
| Discussion | 5,769 | COMPRESS | 1,300 | §203 metric-reframe core만, 나머지 trim |

## 구조적 이동 (정보 이동, 삭제 아님)

**(A) NEW Results 섹션 "The representational boundary of PLM isoform resolution" 신설** — 흩어진 headline 증거 결집(현재 §4b·§4c "what is absent"·§6 ceiling문단·Discussion "representational ceiling"·§203 centroid/macro가 분산). 결집 대상:
- double dissociation(label-oracle macro 1.000 / DR 0.500)
- centroid-sim tie(DR 0.638≈0.630) + **centroid macro 0.099**(2-axis decoupling)
- 3중 bracket(capacity+supervision+layer-select 전부 DR 못 올림)
- tissue-contrast(muscle non-domain 46.5% vs brain 30.2%, splice-size 메커니즘)
- **non-domain encoded(disorder 0.79)+미포화(L30 +0.058)=label限 not representation限**
→ 이 섹션이 title "maps the limits"의 본문 앵커. Intro 경계 예고가 이걸 가리키게.

**(B) §5(4,153w) → 본문 ~600, 나머지 supplementary 이동:**
- "beyond-gene-label GO acquisition + Table 1" → **Supplementary**(cool capability, thesis 아님)
- "functional specialization architecture" → **Supplementary**
- 11-method SOTA 벤치마크 full → 이미 Table S2 존재; 본문은 **핵심 baseline 3개만**(gene-mean oracle 0.803, centroid tie, BLAST oracle 0.861) 잔류
- "within-gene pos_bias full GO" → Methods/Supp로
- 본문 잔류: InterProScan 비-중복(92.3%) 요지 1문단만(complementary annotation 논거)

**(C) §6 SPLIT:** ceiling 증거(3-arm bracket, tissue-contrast, non-domain)→경계섹션(A)로. §6 본문엔 zero-shot transfer 사실(0.647, oracle 0.795) + AD 배포 bridge만 잔류.

## 압축서 반드시 살릴 findings (do-not-lose)
double dissociation 1.000/0.500 · centroid tie(DR)+macro 0.099 · DR 0.630/0.775 · 3중 bracket · tissue-contrast 46.5%/30.2% · non-domain encoded 0.79+미포화 · NDUFAF5–NDUFS7 enzyme-substrate(Tier A-DR) · DOCK11(p_adj 0.004) · NDUFS8 localization · KIF21B 외부 복제. (전부 canonical 수치는 [[feedback-manuscript-satellite-sync]] 참조)

## 실행 전제조건 (언제 실행하나)
1. **발견 안정** — 세션마다 새 통합이 멈춘 뒤(지금은 계속 추가 중 → 보류 정당).
2. 실행 시 **backup 먼저**(natcomm_v0.md_backup_{date}).
3. 이동분은 **삭제 아닌 supplementary 붙여넣기** 확인.
4. 압축 후 [[approach-manuscript-refactor-safety]] 4축 감사(참조/자족/dangling/수치 일관) + 위성 파일 동기화([[feedback-manuscript-satellite-sync]]).
5. 승인 필수(구조 재편 = 규칙상 명시 승인).
