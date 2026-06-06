# Gene-level Label Noise 검증 분석
**분석일**: 2026-06-01  
**목적**: DLG1 translation score 0.889 및 IFT122 muscle contraction score 0.825가 gene-level label memorization인지 확인

---

## 요약 (결론 선행)

| 의심 케이스 | 검증 결과 | 판정 |
|------------|----------|------|
| DLG1 → translation (GO:0006412) score 0.889 | GO:0006412는 v15d_bp_clean 모델에 존재하지 않는 GO term | **허위 경보 — 잘못된 보고서** |
| IFT122 → muscle contraction (GO:0006936) score 0.825 | GO:0006936는 v15d_bp_clean 모델에 존재하지 않는 GO term | **허위 경보 — 잘못된 보고서** |

**결론: 의심된 두 score는 PRISM v15d_bp_clean (production model)에서 온 것이 아닙니다. 이 점수들은 이전 보고서(`interpro2go_vs_prism_experiment.md`)에서 잘못 기술된 것이며, gene-level label noise 문제가 실제로는 존재하지 않습니다.**

---

## Step 1: PRISM v15d_bp_clean의 실제 GO Term Set

`hMuscle/model/v15d_bp_clean.py`에 정의된 18개 GO terms:

```python
GO_TERMS = {
    'GO:0007204': 'Ca2+ signaling',
    'GO:0045214': 'Sarcomere organization',
    'GO:0006941': 'Muscle contraction',      ← GO:0006936이 아님
    'GO:0006914': 'Autophagy',
    'GO:0043161': 'Proteasome-UPS',
    'GO:0007519': 'Skeletal muscle dev',
    'GO:0042692': 'Muscle cell diff',
    'GO:0055074': 'Ca2+ homeostasis',
    'GO:0007005': 'Mitochondrion org',
    'GO:0007517': 'Muscle organ dev',
    'GO:0032006': 'TOR signaling',
    'GO:0030048': 'Actin-based movement',
    'GO:0006096': 'Glycolysis',
    'GO:0007268': 'Synaptic transmission',
    'GO:0007018': 'MT-based movement',
    'GO:0031175': 'Neuron proj development',
    'GO:0030182': 'Neuron diff',
    'GO:0000226': 'MT cytoskeleton org',
}
```

**GO:0006412 (translation)와 GO:0006936 (muscle contraction)는 이 모델에 존재하지 않습니다.**

GO:0006936은 이전 DIFFUSE benchmark 모델(`benchmark_diffuse_dataset2.py`)에서 사용되던 GO term입니다. v15d_bp_clean은 GO:0006941을 사용합니다.

---

## Step 2: GO Annotation 원본 파일에서 DLG1, IFT122 확인

### 검색 파일
- `hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt` (v15d_bp_clean이 실제로 사용)
- `hMuscle/data/raw_data/data/annotations/human_annotations_ncbi_bp.txt`
- `hMuscle/data/raw_data/data/annotations/swissprot_annotations.txt`

### DLG1 결과
| GO Term | 설명 | human_annotations_unified_bp | human_annotations_ncbi_bp |
|---------|------|-------------------------------|--------------------------|
| GO:0006412 | Translation | **NOT FOUND** | NOT FOUND |
| GO:0006936 | Muscle contraction (old) | NOT FOUND | NOT FOUND |
| GO:0006941 | Muscle contraction (v15d) | NOT FOUND | NOT FOUND |
| **GO:0007268** | **Synaptic transmission** | **FOUND** | FOUND |
| GO:0000226 | MT cytoskeleton org | FOUND | NOT FOUND |

**DLG1의 v15d_bp_clean training label**: GO:0007268 (Synaptic transmission) = **positive**, GO:0006941 (Muscle contraction) = **negative**

### IFT122 결과
| GO Term | 설명 | human_annotations_unified_bp | human_annotations_ncbi_bp |
|---------|------|-------------------------------|--------------------------|
| GO:0006412 | Translation | **NOT FOUND** | NOT FOUND |
| GO:0006936 | Muscle contraction (old) | NOT FOUND | NOT FOUND |
| GO:0006941 | Muscle contraction (v15d) | NOT FOUND | NOT FOUND |
| GO:0007268 | Synaptic transmission | NOT FOUND | NOT FOUND |
| **GO:0007018** | **MT-based movement** | **FOUND** | NOT FOUND |

**IFT122의 v15d_bp_clean training label**: GO:0007018 (MT-based movement) = **positive** (생물학적으로 타당: IFT = intraflagellar transport, 마이크로튜불 기반 이동)

---

## Step 3: PRISM Label Generation 메커니즘 (gene-level propagation 확인)

`v15d_bp_clean.py`의 `load_labels` 함수:

```python
def load_labels(go_term):
    pos = set()
    with open(f'{ANNOT_DIR}/human_annotations_unified_bp.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and go_term in parts[1:]:
                pos.add(parts[0])
    y_tr = np.array([1 if s in pos else 0 for s in tr_sym], dtype=np.float32)
    y_te = np.array([1 if s in pos else 0 for s in te_sym], dtype=np.float32)
    return y_tr, y_te
```

- `tr_sym`, `te_sym`: 각 이소폼의 **gene symbol** 리스트
- Gene symbol이 positive set에 있으면 해당 유전자의 **모든 이소폼**이 label=1
- 이것은 gene-level annotation propagation이 맞음

**그러나** 이것이 "noise"가 되려면 annotation이 생물학적으로 틀려야 합니다:
- DLG1 → GO:0007268 (synaptic transmission) = 생물학적으로 타당 (MAGUK scaffold, PDZ 도메인)
- IFT122 → GO:0007018 (MT-based movement) = 생물학적으로 타당 (intraflagellar transport)

---

## Step 4: 실제 PRISM v15d 예측값 (brain score matrix 기준)

### DLG1 (GO:0007268, Synaptic transmission)
- `tr319500` (NNIC, 187aa, L27+MAGUK_N_PEST, NO PDZ): **score = 0.033**
- Canonical DLG1-201 (906aa, 3 PDZ domains): **score = 0.818–0.927**
- Delta = 0.857 (논문에 보고된 값과 일치)
- 이소폼-특이적 예측: PDZ domain 보유 여부에 따른 기능 차이를 포착

### IFT122 (GO:0007268, Synaptic transmission — BISECT max-delta GO term)
- BISECT cases_input.csv: `IFT122, diffuse_delta=0.9538, go=Synaptic transmission`
- CT isoform (ENST00000691964, 1163aa, WD40/eIF2A): synaptic transmission score 높음
- AD isoform (ENST00000688527, 647aa, Clathrin/TPR): synaptic transmission score 낮음

**IFT122는 GO:0006936이 아닌 GO:0007268에서 Δ=0.9538로 BISECT에 통과함.**

---

## Step 5: interpro2go_vs_prism_experiment.md 오류 분석

해당 보고서의 오류 목록:
1. `KIF21B Δ=0.855, GO:0006936`: 실제 BISECT에서 KIF21B의 delta는 brain GO terms 중 GO:0007268/GO:0031175에서 발생 (GO:0006936은 v15d에 없음)
2. `IFT122 Δ=0.825, GO:0006936`: 실제 Δ=0.9538, GO:0007268 (Synaptic transmission)
3. `DLG1 translation 0.889`: 실제 v15d에 GO:0006412 없음; DLG1의 실제 score는 GO:0007268에서 canonical=0.818–0.927, tr319500=0.033

이 보고서는 이전 DIFFUSE model (GO:0006936 사용)의 수치와 현재 PRISM v15d_bp_clean (GO:0006941 사용)의 수치를 혼동한 것으로 보입니다. 또는 이 보고서 자체가 실제 계산 없이 추론으로 작성된 것일 수 있습니다.

---

## 최종 판정

### Q1: DLG1에 GO:0006412 (translation) annotation이 있는가?
**NO** — 모든 annotation 파일에서 확인됨. v15d_bp_clean에 GO:0006412가 없으므로 이 점수 자체가 존재하지 않음.

### Q2: IFT122에 GO:0006936 (muscle contraction) annotation이 있는가?
**NO** — 모든 annotation 파일에서 확인됨. v15d_bp_clean에 GO:0006936이 없으므로 이 점수 자체가 존재하지 않음.

### Q3: DLG1의 모든 이소폼이 translation score > 0.5인가? (gene-level bias 신호)
**해당 없음** — GO:0006412 (translation)은 v15d_bp_clean에 존재하지 않음. DLG1의 실제 positive label은 GO:0007268 (synaptic transmission)이며, PRISM은 이소폼별로 차별화된 score를 예측함 (tr319500=0.033 vs canonical=0.818–0.927).

### Q4: IFT122의 모든 이소폼이 muscle contraction score > 0.5인가?
**해당 없음** — GO:0006936 (muscle contraction)은 v15d_bp_clean에 존재하지 않음. IFT122의 실제 positive label은 GO:0007018 (MT-based movement)이며, BISECT max-delta GO term은 GO:0007268 (synaptic transmission, Δ=0.9538).

### Q5: 결론 — gene-level label noise인가, 이소폼-특이적 예측인가?

**Gene-level label noise가 아닙니다.** 두 이유:

1. **GO term 자체가 없음**: DLG1 translation 0.889와 IFT122 muscle contraction 0.825는 PRISM v15d_bp_clean 모델에 존재하지 않는 GO terms (GO:0006412, GO:0006936)를 언급한 잘못된 보고서의 수치입니다.

2. **실제 예측은 이소폼-특이적**: 
   - DLG1의 실제 GO:0007268 score: tr319500=0.033 vs canonical DLG1=0.818–0.927 — PDZ 도메인 보유 여부를 정확히 반영
   - IFT122의 실제 BISECT delta=0.9538 (GO:0007268) — WD40/eIF2A vs Clathrin/TPR 도메인 차이를 반영
   - 두 경우 모두 gene-level 단일 값이 아닌 이소폼별 분화된 예측값 존재

---

## 추가 주의사항: Gene-level Propagation의 실제 한계

Gene-level label propagation이 이론적으로는 편향을 유발할 수 있습니다. v15d_bp_clean의 `load_labels`는 gene symbol 단위로 label을 부여합니다. 이 한계는:
- DLG1 모든 이소폼이 GO:0007268 = 1로 학습됨
- IFT122 모든 이소폼이 GO:0007018 = 1로 학습됨

그러나 **PRISM의 실제 예측은 이소폼별로 차별화됨** (DLG1 사례에서 입증). 이는 ESM-2 embedding이 gene-level label에도 불구하고 서열 특이성을 인코딩하고 있음을 시사합니다.

---

## 검증에 사용된 파일
- `hMuscle/data/raw_data/data/annotations/human_annotations_unified_bp.txt` — v15d_bp_clean 학습 레이블 소스
- `hMuscle/data/raw_data/data/annotations/human_annotations_ncbi_bp.txt` — 교차 확인
- `hMuscle/data/raw_data/data/annotations/swissprot_annotations.txt` — 교차 확인
- `hMuscle/model/v15d_bp_clean.py` — GO term 정의 및 label generation 코드
- `reports/v15d_brain_eval/brain_eval_20260519_2125.json` — brain model GO term 순서 확인
- `reports/v15_bp_clean/cross_go_18go_20260519_1914.json` — muscle model GO term 순서 확인
- `Final_analysis/pipeline_bioanalysis/cases_input.csv` — BISECT 실제 input delta 및 GO terms
- `reports/v16_brain_switch/brain_switches_20260519_1429.tsv` — IFT122/KIF21B brain switch results

---

*분석 수행: 2026-06-01 | Gene-level label noise 검증*
