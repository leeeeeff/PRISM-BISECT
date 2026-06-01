# Challenge: 핵심 주장 3가지에 대한 최강 반론

**작성일**: 2026-04-10  
**목적**: NMI/Nature Methods Reviewer 관점에서 현재 기여 주장의 취약점 선제 파악  
**대상 주장**:
1. Gene-level shortcut 해결이 우리만의 기여인가
2. Cross-gene negative mining [R2.1]이 기존 문헌과 실질적으로 다른가
3. 4단계 커리큘럼이 이 문제에 필수인가

---

## Challenge 1: "Gene-level shortcut 해결이 우리만의 기여인가"

### 반론 ①: 기존 문헌이 이미 동일 설계를 다룬다

단백질 수준 metric learning에서 이미 동일한 구조:

- **ProtTucker** (Littmann et al., 2021, NAR): 서로 다른 protein family 간 negative sampling → within-family negative 제외. Cross-gene negative [R2.1]과 구조적으로 동일.
- **COLLAPSE** (Alcaide et al., 2022): protein superfamily-aware contrastive learning. Within-superfamily negative 제외.
- **SimCSE** (Gao et al., 2021): in-batch hard negative에서 동일 class 제외 — NLP에서의 동일 패턴.

**심사자 공격 예상**:
> "Cross-gene negative mining은 contrastive learning의 표준 설계인 'class-aware negative sampling'의 특수 케이스다. 새로운 기여가 아니다."

### 반론 ②: "Gene-level shortcut"은 기존 개념의 재명명일 수 있다

IRM (Invariant Risk Minimization, Arjovsky et al., 2019):
> "모델이 environment-specific spurious correlation에 의존한다"

이 연구의 "gene-level shortcut":
> "모델이 isoform-specific feature 대신 gene-level feature에 의존한다"

두 문제의 구조가 동일하다. "왜 IRM으로 해결이 안 되는가"를 먼저 보여야 독자성이 성립한다.  
**현재 IRM/DRO 비교 실험 없음.**

### 반론 ③: Gene-level shortcut이 실제로 존재한다는 직접 증거가 없다

```
현재 증거: DIFFUSE CRF 제거 후 성능 향상
해석: "CRF가 gene-level shortcut을 만들었다"
대안 해석: CRF 자체의 수렴 불안정 때문일 수 있음
```

**누락된 증거**:
- Gene-level feature만 쓴 모델 vs isoform-level feature만 쓴 모델 성능 비교 없음
- Attention/gradient attribution으로 모델이 실제로 gene-level feature에 집중하는지 미확인
- "Cross-gene negative를 제거하면 성능이 하락한다"는 ablation 없음

**핵심 취약점**: Shortcut의 존재를 가정하고 solution을 만들었는데, shortcut 존재 자체가 미증명.

### 이 주장의 생존 조건

| 필요 실험 | 기대 결과 |
|---------|---------|
| `no_cross_gene_negative` ablation | AUPRC 유의미하게 하락해야 함 |
| Gradient attribution (GradCAM/Integrated Gradients) | Gene-level feature에 집중하는 패턴 확인 |
| IRM 비교 실험 | IRM보다 우리 방법이 더 효과적임 입증 |

---

## Challenge 2: "Cross-gene negative mining이 기존 문헌과 실질적으로 다른가"

### 반론 ①: Standard triplet learning의 올바른 구현일 뿐이다

Metric learning 교과서적 원칙:
> "Anchor와 같은 class의 sample은 negative로 쓰지 않는다"

"Gene = class"로 보면 cross-gene negative는 **표준 triplet learning의 기본 설정**이다.

**심사자 공격 예상**:
> "Cross-gene negative를 쓴 것은 contribution이 아니라 당연한 것이다. 이것을 기여로 주장하는 것은 오히려 기존 방법을 제대로 구현하지 않았음을 자백하는 것이다."

### 반론 ②: 더 단순하고 이론적으로 우월한 대안이 존재한다

**Supervised Contrastive Learning (SupCon, Khosla et al., 2020)**:
```
L_SupCon = -∑_i (1/|P(i)|) ∑_{p∈P(i)} log(exp(z_i·z_p/τ) / ∑_{a∈A(i)} exp(z_i·z_a/τ))
```

SupCon 장점:
- 자동으로 같은 GO term = positive, 나머지 = negative 처리
- Cross-gene 제약을 별도 구현할 필요 없음
- Batch 내 모든 positive pair 활용 → 정보 효율성 높음
- 단백질 기능 예측 적용 논문 다수 존재 (ProtST, 2023)

**반론 핵심**: 현재 triplet + cross-gene 설계보다 SupCon이 더 단순하고 이론적으로 우월하다. SupCon을 쓰지 않은 이유를 설명하지 않으면 심사에서 공격받는다.

### 반론 ③: Cross-gene negative의 생물학적 가정이 틀릴 수 있다

**현재 가정**: "같은 gene의 isoform은 negative로 쓰면 안 된다"  
**이유**: 같은 gene의 isoform이 다른 기능을 가질 수 있음

**반전 사례**:
- MYH1/2/4/7은 **다른 gene** (MyHC family의 각기 다른 멤버) → 이들 간 negative는 합법적
- 같은 gene 내 isoform (BRCA1-α/β, BCL-X_L/S): 실제로 기능이 **반대**인 경우도 있음 → negative로 쓰는 것이 더 적절할 수도 있음
- Splicing isoform switch: 동일 gene의 두 isoform이 서로 반대 기능을 가지는 생물학적 메커니즘

**가장 강한 반론**: Cross-gene 제약이 일부 생물학적으로 중요한 negative pair를 차단하고 있을 수 있다. 이것이 GO:0006941 성능 저하의 원인일 수도 있다.

---

## Challenge 3: "4단계 커리큘럼이 이 문제에 필수인가"

### 반론 ①: 단순 end-to-end 학습이 더 나을 수 있다

```
현재 파이프라인: Phase 1 (triplet) → Phase 1.5 (focal, frozen) → Phase 2 (joint)
단순 대안:       Phase 1 제거, 처음부터 joint focal+triplet (Phase 2만)
```

이 대안을 한 번도 시도하지 않았다. GO:0003774 Ep7 phase transition이 Phase 1 덕분인지, Phase 2 자체의 momentum 누적 때문인지 구분하는 실험이 없다.

**심사자 공격**:
> "4단계 커리큘럼이 필요하다는 주장은 ablation 없이는 사후 합리화(post-hoc rationalization)다."

### 반론 ②: Phase 1.5의 존재 이유가 불명확하다

```
Phase 1.5 역할: encoder freeze → head를 GO term 방향으로 초기화
Phase 2 상황:   전체 unfreeze → head + encoder 모두 다시 학습
```

Phase 2에서 어차피 전체를 다시 학습하면 Phase 1.5에서 맞춰둔 head 방향이 얼마나 유지되는가? Phase 2 learning rate가 충분히 크면 Phase 1.5 효과가 사라진다. **Phase 1.5 제거 실험 없음.**

### 반론 ③: 커리큘럼 순서의 인과성이 증명되지 않았다

```
관찰: Phase 1 후 → Phase 2에서 GO:0003774 Ep7 AUPRC 급등
해석: "Phase 1이 Phase 2의 발판을 만들었다"
대안: "Phase 2 자체의 학습 역학"
```

**가장 강한 반론**:
> "Phase 1이 실패한 GO:0006941 (margin_sat=5.9%)에서 Phase 2도 실패했다.  
> 이것은 Phase 1이 Phase 2에 필수라는 증거가 아니라,  
> 둘 다 동일한 근본 원인(label sparsity, isoform indistinguishability)에 의해 실패한다는 증거다.  
> 커리큘럼의 인과성이 아니라 **데이터 구조의 문제**다."

### 반론 ④: Curriculum learning의 이론적 보장이 없다

Curriculum learning (Bengio et al., 2009)은 경험적 방법이지 이론적 보장이 없다. 특히:
- 이 데이터의 적절한 커리큘럼 순서가 Phase 1→2인지 다른 순서인지 이론적 근거가 없음
- 동일한 GO term을 다른 순서로 학습했을 때 결과가 달라질 수 있음
- Phase 1이 "easy → hard" 순서인가? Triplet이 classification보다 쉬운 task라는 근거 없음

---

## 종합 판정

| 주장 | Challenge 결과 | 현재 생존 가능성 |
|------|-------------|----------------|
| Gene-level shortcut이 우리 기여 | ❌ 기존 IRM/contrastive와 구조 동일, shortcut 존재 미증명 | **낮음** |
| Cross-gene negative가 독자적 | ❌ Standard triplet의 올바른 구현, SupCon이 더 단순한 대안 존재 | **낮음** |
| 4단계 커리큘럼이 필수 | ❌ ablation 전무, Phase 1.5 효과 불명확, 인과성 미증명 | **낮음** |

---

## 즉시 필요한 ablation 실험 목록 (우선순위순)

```
Priority 1 (Main claim 방어):
  [ ] no_cross_gene_negative — 같은 gene negative 허용 시 AUPRC
  [ ] gene_only_features — gene-level embedding만 사용 시 AUPRC
  [ ] gradient_attribution — gene vs isoform feature 의존도 시각화

Priority 2 (Curriculum 방어):
  [ ] no_phase1 — 처음부터 joint focal+triplet만
  [ ] no_phase1_5 — Phase 1.5 (linear probing) 제거

Priority 3 (Alternative 방어):
  [ ] supcон_replacement — triplet → SupCon으로 교체
  [ ] irm_comparison — IRM loss 적용 비교
```

---

## 대응 전략

### Cross-gene negative → SupCon에 대한 방어

SupCon이 이론적으로 우월하다는 반론에 대한 대응:

1. **계산 비용**: SupCon은 batch 내 전체 positive pair → sparse GO term에서 batch당 positive가 0~2개인 경우 효과 없음. Triplet은 이 상황에서도 upsample로 강제 학습 가능.
2. **Cross-gene 제약의 생물학적 의미**: GO term이 같더라도 다른 gene의 isoform이 negative인 것은 생물학적으로 의미 있음. Gene family 간 paralogue는 유사하지만 다른 기능 → hard negative로 활용.
3. **SupCon 비교 실험 추가**: 반론을 막으려면 SupCon 실험을 직접 하고 우리 방법이 우월함을 보이는 것이 최선.

### 커리큘럼 필요성에 대한 방어

Phase 1이 Phase 2 성능에 인과적으로 기여한다는 증거를 만들어야 함:
- Phase 1 epoch 수를 변수로 두고 Phase 2 최종 AUPRC와의 관계 플롯
- Phase 1 margin_sat과 Phase 2 최종 AUPRC의 상관관계 (5개 GO term 데이터로 가능)
