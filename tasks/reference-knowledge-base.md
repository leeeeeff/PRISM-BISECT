# Research Reference Knowledge Base
# 9-Axis Mathematical & Algorithmic Foundation
# for Isoform Function Prediction Research
#
# 사용법: 각 섹션이 .claude/rules/ 파일로 직접 변환 가능
# 판단 기준: 새 알고리즘/구조 제안 시 해당 축의 수식 조건을 먼저 만족해야 함

════════════════════════════════════════════════════════════
AXIS 1: DATA SPARSITY / MODE COLLAPSE
════════════════════════════════════════════════════════════

## 핵심 문제 정의
이진 분류 y∈{0,1}에서 클래스 비율 r = N_pos/N_neg << 1일 때,
표준 CE loss는 majority class gradient가 지배:
  ∂L/∂θ ≈ (1-r) × ∂L_neg/∂θ  →  minority class 신호 소멸

## 핵심 레퍼런스 & 수식

### [R1.1] Focal Loss (Lin et al., ICCV 2017)
FL(p_t) = -α_t(1 - p_t)^γ · log(p_t)
- 핵심: (1-p_t)^γ 가 easy negative의 gradient를 γ승으로 억제
- γ=0이면 CE와 동일, γ=2가 실험적 최적
- 판단 기준: γ 변경 시 반드시 easy/hard sample 비율 변화로 정당화
- ⚠️ 주의: γ 너무 크면 hard negative에 과집중 → noisy label 환경에서 악화

### [R1.2] Class-Balanced Loss (Cui et al., CVPR 2019)
w_i = (1 - β) / (1 - β^{n_i}),  β = (N-1)/N
- 유효 샘플 수(effective number) 개념 도입
- 단순 inverse frequency보다 수학적으로 엄밀한 reweighting
- 판단 기준: α 설정 시 inverse frequency 대신 이 공식 사용 권장

### [R1.3] Label Smoothing (Szegedy et al., CVPR 2016)
y_smooth = (1-ε)·y_hard + ε/K
- 과도한 confidence 방지, calibration 개선
- sparse class에서 0/1 hard label의 gradient saturation 완화
- 판단 기준: mode collapse 의심 시 ε=0.1로 테스트

### [R1.4] LDAM Loss (Cao et al., NeurIPS 2019)
Δ_j = C / n_j^{1/4}  (클래스 j의 margin)
- 이론적 margin: 클래스 빈도의 -1/4승에 비례
- Focal Loss보다 이론적 근거 강함
- 판단 기준: Focal Loss로 mode collapse 해결 안 될 때 대안

## Mode Collapse 진단 수식
Collapse score = std(prediction_distribution) / mean(prediction_distribution)
→ 낮을수록 특정 클래스로 쏠림. 임계값: < 0.3이면 collapse 의심

## Rules for Code
- γ 변경 시 항상 log: "γ={값}, hard sample 비율={%}, 근거={R1.x}"
- α 설정은 R1.2 공식 우선, inverse frequency는 baseline에서만
- Collapse score를 evaluation.py에 기본 출력으로 추가 권장


════════════════════════════════════════════════════════════
AXIS 2: GENE-LEVEL DOMINANCE / SHORTCUT LEARNING
════════════════════════════════════════════════════════════

## 핵심 문제 정의
모델이 isoform-specific signal f_iso 대신 gene-level signal f_gene을 학습:
  P(y | isoform) ≈ P(y | gene)  →  isoform-specific prediction 불가
수학적으로는 spurious correlation: f_gene과 y의 MI가 f_iso와 y의 MI보다 클 때 발생

## 핵심 레퍼런스 & 수식

### [R2.1] Shortcut Learning (Geirhos et al., Nature Machine Intelligence 2020)
- 핵심: 모델은 항상 "가장 쉬운" 예측 경로를 선택
- gene-level feature는 항상 isoform-level보다 더 많은 데이터로 학습됨
- 판단 기준: 새 feature 추가 시 "이게 gene-level shortcut을 더 강화하는가?" 먼저 질문

### [R2.2] Invariant Risk Minimization (Arjovsky et al., 2019)
min_{Φ} Σ_e R^e(Φ) s.t. Φ ∈ argmin_Φ̄ R^e(Φ̄)  ∀e
- 환경(gene family)이 달라도 일정한 표현 학습
- gene family를 "environment"로 정의하면 IRM 직접 적용 가능
- 판단 기준: gene-level 편향이 지속될 때 IRM loss term 추가 검토

### [R2.3] Disentangled Representation (Bengio et al., 2013; β-VAE)
L = E[log p(x|z)] - β·KL(q(z|x) || p(z))
- z를 gene-level z_g와 isoform-level z_i로 명시적 분리
- β > 1이면 더 강한 disentanglement
- 판단 기준: embedding을 gene/isoform 두 subspace로 분리하는 구조 제안 시 β-VAE 수식 인용

### [R2.4] Gradient Reversal Layer (Ganin et al., ICML 2015)
L = L_task - λ·L_domain  (domain = gene identity)
- gene identity를 domain으로 보고 adversarial하게 제거
- λ 스케줄링: 초기 0 → 점진적 증가 (학습 안정화)
- 판단 기준: gene-level bias 직접 제거 시 가장 명확한 수학적 근거

## Gene-Bias 탐지 수식
Bias score = 1 - (H(y|isoform_id) / H(y|gene_id))
→ 0에 가까울수록 gene-level만으로 예측 가능 = 심각한 편향
임계값: > 0.3이어야 isoform-specific learning이 일어나고 있다고 볼 수 있음

## Rules for Code
- gene context 사용 시 반드시 gating 또는 adversarial: R2.3 또는 R2.4
- 직접 concatenation 금지 (R2.1 위반)
- 새 구조 제안 시 Bias score 계산 후 기록


════════════════════════════════════════════════════════════
AXIS 3: METRIC LEARNING / CONTRASTIVE LEARNING
════════════════════════════════════════════════════════════

## 핵심 문제 정의
기능적으로 유사한 isoform은 embedding space에서 가까워야 하고
기능적으로 다른 isoform은 멀어야 함.
Triplet Loss가 이를 구현하지만 수학적 함정이 많음.

## 핵심 레퍼런스 & 수식

### [R3.1] Triplet Loss (Schroff et al., FaceNet, CVPR 2015)
L = Σ max(||f(a)-f(p)||² - ||f(a)-f(n)||² + α, 0)
- α: margin, f: embedding function
- ⚠️ 함정 1: random triplet mining → 대부분 easy triplet, gradient ≈ 0
- ⚠️ 함정 2: intra-gene negative 사용 시 gene-level signal 강화
- 판단 기준: mining 전략이 반드시 명시되어야 함

### [R3.2] Hard Negative Mining (Hermans et al., 2017)
Batch Hard strategy:
  p* = argmax_p ||f(a)-f(p)||  (hardest positive)
  n* = argmin_n ||f(a)-f(n)||  (hardest negative)
- Random mining보다 훨씬 강한 gradient 신호
- ⚠️ 너무 hard한 negative는 noisy label 위험
- 판단 기준: triplet loss 개선 시 mining 전략 변경이 첫 번째 옵션

### [R3.3] Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
L = Σ_i (-1/|P(i)|) · Σ_{p∈P(i)} log[exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ)]
- 같은 GO term = positive set P(i), temperature τ로 스케일
- Triplet보다 수학적으로 더 안정적 (모든 positive 동시 활용)
- 판단 기준: triplet mining 불안정 시 SupCon으로 교체 검토

### [R3.4] Uniformity & Alignment (Wang & Isola, ICML 2020)
L_align = E_{(x,y)~p_pos} ||f(x)-f(y)||²
L_uniform = log E_{(x,y)~p_data} exp(-2||f(x)-f(y)||²)
- 좋은 embedding = alignment(같은 class 가깝게) + uniformity(공간 균등 분포)
- 판단 기준: embedding 품질 평가 시 이 두 지표로 측정

## Triplet Collapse 진단
Triplet active ratio = |{triplets: loss > 0}| / |total triplets|
→ < 5%이면 대부분 easy triplet → mining 전략 변경 필요

## Rules for Code
- Negative는 반드시 cross-gene (intra-gene 금지): R3.1 위반 방지
- Mining 전략 변경 시 R3.2 먼저, 그래도 불안정이면 R3.3
- Embedding 평가 시 R3.4 두 지표 함께 보고


════════════════════════════════════════════════════════════
AXIS 4: MULTIMODAL FUSION
════════════════════════════════════════════════════════════

## 핵심 문제 정의
ESM embedding (z_seq), PPI network (z_ppi), Cell localization (z_loc)을
어떻게 합치는가에 따라 수학적으로 완전히 다른 모델이 됨.
단순 concatenation은 가장 위험한 선택.

## 핵심 레퍼런스 & 수식

### [R4.1] Modality Dominance 문제 (Wang et al., CVPR 2020)
- 학습률이 같으면 가장 쉬운 modality가 나머지를 suppression
- gradient norm 비교: ||∂L/∂z_seq|| >> ||∂L/∂z_ppi|| 이면 PPI 정보 소멸
- 판단 기준: 각 modality의 gradient norm을 주기적으로 로깅해야 함

### [R4.2] Attention-based Fusion (Transformer cross-attention)
Fusion(Q,K,V) = softmax(QK^T/√d_k)V
- Q: query modality, K,V: other modalities
- isoform embedding을 Q로, gene context를 K,V로 → gene info를 gating
- 판단 기준: fusion 구조 변경 시 attention이 gene-bias 방지하는지 확인

### [R4.3] Multimodal Information Bottleneck (Federici et al., NeurIPS 2020)
min I(Z;X) s.t. I(Z;Y) ≥ I_c
- 각 modality에서 task에 필요한 최소 정보만 추출
- gene-level 불필요 정보를 수학적으로 제거하는 근거
- 판단 기준: modality별 feature dimension 결정 시 IB 관점에서 정당화

### [R4.4] Gradient Modulation (OGM-GE, Peng et al., CVPR 2022)
ρ_m = tanh(ρ̄_m),  gradient_m ← gradient_m × (1 - ρ_m)
- 각 modality의 기여도(ρ)에 따라 gradient를 동적으로 조절
- dominant modality의 과학습 방지
- 판단 기준: ESM이 PPI/CellLoc을 압도한다고 판단되면 R4.4 적용

## Fusion Quality 진단
Modality contribution = |prediction_change| when modality m is masked
→ 한 modality를 0으로 만들었을 때 성능 변화량
→ 세 modality의 기여도가 너무 편향되면 fusion 재설계 필요

## Rules for Code
- 단순 concatenation은 baseline에서만 허용
- 새 fusion 설계 시 R4.1 gradient norm 체크 먼저
- Attention fusion 사용 시 isoform을 Q로 고정 (gene dominance 방지)


════════════════════════════════════════════════════════════
AXIS 5: FEW-SHOT / META-LEARNING (PFN 수학적 근거)
════════════════════════════════════════════════════════════

## 핵심 문제 정의
GO term당 positive isoform이 < 50개인 환경.
PFN이 이 환경에서 왜 좋은가, 그리고 어떤 조건에서 실패하는가.

## 핵심 레퍼런스 & 수식

### [R5.1] Prior-data Fitted Networks (Müller et al., NeurIPS 2021)
P(y|x, D_train) = ∫ P(y|x,θ) P(θ|D_train) dθ
- Bayesian posterior를 transformer가 암묵적으로 근사
- 핵심 가정: test distribution ∈ prior distribution
- ⚠️ 위반 조건: novel isoform이 prior에 없는 구조면 성능 급락
- 판단 기준: 새 데이터 추가 시 prior coverage 확인 필수

### [R5.2] MAML (Finn et al., ICML 2017)
θ* = θ - α·∇_θ L_task(θ)  (inner loop)
θ ← θ - β·∇_θ Σ L_task(θ*)  (outer loop)
- 몇 번의 gradient step으로 새 task에 적응
- PFN 실패 시 대안: GO term을 task로 정의하면 직접 적용 가능
- 판단 기준: novel GO term에 성능 급락 시 MAML fine-tuning 검토

### [R5.3] Prototypical Networks (Snell et al., NeurIPS 2017)
c_k = (1/|S_k|) Σ_{(x,y)∈S_k} f_θ(x)  (prototype)
P(y=k|x) ∝ exp(-d(f_θ(x), c_k))
- 각 GO term의 prototype vector로 분류
- sparse class에서 PFN보다 직관적이고 안정적
- 판단 기준: 5-shot 이하 GO term 성능이 특히 나쁘면 ProtoNet 비교 실험

### [R5.4] In-Context Learning Theory (Akyürek et al., ICLR 2023)
- Transformer가 gradient descent를 implicit하게 수행함을 증명
- PFN의 이론적 근거: support set이 implicit training data
- 핵심 조건: support set diversity가 충분해야 함
- 판단 기준: support set 구성 시 GO term당 최소 5개, 다양한 gene family에서

## PFN 성능 진단
Support set quality = mean(pairwise distance in embedding space of support)
→ 너무 낮으면 support set이 다양하지 않음 → 성능 저하 원인

## Rules for Code
- PFN support set은 gene family별 stratified sampling 필수 (R5.4)
- novel isoform 테스트 시 prior coverage 먼저 확인 (R5.1)
- 5-shot 이하 클래스에서 PFN 실패 시 R5.3 ProtoNet 비교


════════════════════════════════════════════════════════════
AXIS 6: REPRESENTATION LEARNING
════════════════════════════════════════════════════════════

## 핵심 문제 정의
embedding z ∈ R^d 가 언제 "의미있는" 구조를 가지는가.
"의미있다" = 기능적으로 유사한 isoform이 기하학적으로도 가까운 상태.

## 핵심 레퍼런스 & 수식

### [R6.1] Representation Quality: Linear Probing
acc_linear = performance of linear classifier on frozen embeddings
- embedding의 task-relevant information 양을 측정
- 판단 기준: 새 embedding 구조 제안 시 linear probing 먼저 비교

### [R6.2] Isotropy of Embedding Space (Mu & Viswanath, 2018)
IsoScore = (1/d) Σ_i λ_i / max_i(λ_i)  (λ: singular values of Z)
→ 1에 가까울수록 모든 방향 균등 활용 = 좋은 embedding
→ 0에 가까우면 특정 방향에 집중 = collapsed embedding
- 판단 기준: embedding collapse 의심 시 IsoScore 계산

### [R6.3] Dimensional Collapse (Hua et al., NeurIPS 2021)
rank(Z) << d 이면 dimensional collapse
→ Z의 effective rank = exp(H(singular value distribution))
- 판단 기준: triplet loss 학습 후 embedding rank 확인

### [R6.4] Mutual Information Maximization (Linsker, 1988; MINE)
I(X;Y) = E_p(x,y)[log(p(x,y)/p(x)p(y))]
MINE 추정: I(X;Z) ≥ E_T[·] - log(E_T̄[·])
- embedding z가 input x의 task-relevant 정보를 최대한 보존하는지
- 판단 기준: new feature 추가 시 MI 증가 여부로 정당화

## Embedding Quality 진단 체크리스트
1. IsoScore > 0.3 (collapse 방지)
2. effective rank > d/4 (dimensional collapse 방지)
3. Linear probing accuracy > random baseline × 2
4. Intra-class distance < Inter-class distance (by GO term)

## Rules for Code
- 새 embedding 구조는 R6.1 linear probing으로 먼저 검증
- embedding dimension 변경 시 R6.2 IsoScore 확인
- triplet loss 변경 후 R6.3 rank 확인


════════════════════════════════════════════════════════════
AXIS 7: GENERALIZATION / OUT-OF-DISTRIBUTION
════════════════════════════════════════════════════════════

## 핵심 문제 정의
학습 데이터에 없는 novel isoform에 대해 모델이 얼마나 신뢰할 수 있는가.
이것이 곧 "Novel Case Discovery" 능력의 수학적 기반.

## 핵심 레퍼런스 & 수식

### [R7.1] Bias-Variance Tradeoff in Distribution Shift
E[L_test] = E[L_train] + covariate_shift_term + label_shift_term
- novel isoform = covariate shift (input distribution 변화)
- gene family별 다른 GO term 빈도 = label shift
- 판단 기준: novel isoform 성능 저하 원인을 두 term 중 어느 쪽인지 분리

### [R7.2] Domain Generalization (Blanchard et al., 2011)
min_θ max_{P∈P_test} E_{(x,y)~P}[L(θ;x,y)]
- worst-case gene family에서도 잘 작동하도록 minimax
- 판단 기준: 특정 gene family에서만 성능 급락이면 R7.2 적용

### [R7.3] Spectral Normalization (Miyato et al., ICLR 2018)
W_SN = W / σ(W)  (σ: largest singular value)
- Lipschitz constraint → OOD에서 부드러운 extrapolation
- 판단 기준: novel isoform에서 예측이 극단적으로 치우칠 때 SN 추가

### [R7.4] Deep Ensemble for OOD (Lakshminarayanan et al., NeurIPS 2017)
p(y|x) = (1/M) Σ_m p_m(y|x)
- M개의 독립 모델 앙상블 → OOD에서 disagreement = uncertainty 신호
- 판단 기준: novel case 탐지에 가장 실용적인 방법

## OOD 탐지 수식
OOD score = -max_k p(y=k|x)  (maximum softmax probability)
→ 낮을수록 OOD 가능성 높음 (= novel case 후보)
더 정교한 방법: Mahalanobis distance from class centroids

## Rules for Code
- novel isoform 평가 시 반드시 OOD score 함께 보고 (R7.4)
- 특정 gene family 성능 급락 시 R7.1로 shift 원인 분리 먼저
- SN은 OOD 안정화 목적으로만 사용 (성능 향상 목적 X)


════════════════════════════════════════════════════════════
AXIS 8: UNCERTAINTY QUANTIFICATION
════════════════════════════════════════════════════════════

## 핵심 문제 정의
"이 isoform의 기능 예측이 얼마나 확실한가"를 수치화.
Novel case = 모델이 불확실한 isoform = high uncertainty region.

## 핵심 레퍼런스 & 수식

### [R8.1] Epistemic vs Aleatoric Uncertainty (Kendall & Gal, NeurIPS 2017)
Var[y] = E_θ[Var[y|x,θ]] + Var_θ[E[y|x,θ]]
          aleatoric (데이터)   epistemic (모델)
- Novel case 탐지에는 epistemic uncertainty가 핵심
- 판단 기준: uncertainty를 novel case 지표로 쓸 때 epistemic만 사용

### [R8.2] Monte Carlo Dropout (Gal & Ghahramani, ICML 2016)
p(y|x) ≈ (1/T) Σ_t p(y|x, θ_t)  (θ_t: dropout mask t)
uncertainty = Var_t[p(y|x,θ_t)]
- inference 시 dropout ON으로 T회 forward pass
- 계산 비용: T=30이면 충분
- 판단 기준: novel case 탐지 파이프라인에 가장 쉽게 추가 가능

### [R8.3] Conformal Prediction (Angelopoulos & Bates, 2021)
P(y_test ∈ C(x_test)) ≥ 1-α
- 예측 집합 C(x)를 보장된 coverage로 구성
- sparse label 환경에서 point prediction 대신 prediction set 제공
- 판단 기준: 단일 GO term 예측보다 "이 isoform은 이 GO term set에 속할 가능성 90%" 형태로 보고 가능

### [R8.4] Energy-based OOD Detection (Liu et al., NeurIPS 2020)
E(x;f) = -log Σ_k exp(f_k(x))
→ in-distribution: low energy, OOD: high energy
- softmax 기반 방법보다 이론적으로 우월
- 판단 기준: OOD score로 MSP(max softmax prob) 대신 에너지 사용 권장

## Novel Case Detection Pipeline
1. 모든 isoform에 대해 E(x;f) 계산 (R8.4)
2. high energy (> threshold) → OOD 후보
3. MC Dropout으로 epistemic uncertainty 확인 (R8.2)
4. high uncertainty + high energy → Novel Case 후보 목록

## Rules for Code
- novel case 탐지 시 MSP 대신 energy score 사용 (R8.4)
- MC Dropout: inference 시 T=30, dropout_rate=0.1
- 최종 novel case 보고는 energy + uncertainty 두 지표 모두


════════════════════════════════════════════════════════════
AXIS 9: EVALUATION VALIDITY
════════════════════════════════════════════════════════════

## 핵심 문제 정의
Sparse label 환경에서 어떤 지표를 믿을 수 있는가.
잘못된 지표로 평가하면 실제로 나쁜 모델이 좋아 보일 수 있음.

## 핵심 레퍼런스 & 수식

### [R9.1] Precision-Recall vs ROC (Davis & Goadrich, ICML 2006)
- 핵심 정리: imbalanced dataset에서 AUROC는 misleading
- AUPRC (Area Under PR Curve)가 더 신뢰할 수 있는 지표
- 이유: ROC는 TN이 많으면 FPR이 낮아 보여 성능 과대평가
- 판단 기준: sparse GO term (positive < 50)에서는 AUPRC를 primary metric으로

### [R9.2] F1 Score의 한계 (수식적 분석)
F1 = 2PR/(P+R)
- threshold에 민감: 같은 모델도 threshold 변화에 따라 F1 크게 달라짐
- 판단 기준: F1 비교 시 반드시 같은 threshold 조건 명시

### [R9.3] Calibration Error (Guo et al., ICML 2017)
ECE = Σ_m (|B_m|/n) |acc(B_m) - conf(B_m)|
- 예측 확률 p=0.8이면 실제로 80%의 확률로 맞아야 함
- 판단 기준: novel case 탐지에 uncertainty 활용 시 calibration 먼저 확인

### [R9.4] Statistical Significance (McNemar's Test)
χ² = (b-c)² / (b+c)
- 두 모델의 성능 차이가 통계적으로 유의한지 검정
- DIFFUSE 대비 개선 주장 시 반드시 McNemar test 또는 bootstrap CI 포함
- 판단 기준: delta F1 < 0.05이면 통계적 유의성 없을 가능성 높음

### [R9.5] Multi-label Evaluation (GO term은 multi-label)
Micro-F1: 전체 (sample, label) 쌍에서 계산 → majority label에 편향
Macro-F1: 각 label F1의 평균 → sparse label에 더 공정
- 판단 기준: overall 성능 보고 시 Macro-F1 사용, sparse 클래스 따로 보고

## Evaluation Protocol (모든 실험 공통)
1. Primary metrics (sparse GO term): AUPRC + Macro-F1
2. Secondary metrics: Micro-F1 + AUROC (참고용)
3. 통계 검정: bootstrap CI (n=1000) 또는 McNemar test
4. 반드시 DIFFUSE baseline과 동일 조건에서 비교

## Rules for Code
- sparse class (positive < 50): AUPRC primary, AUROC secondary (R9.1)
- F1 비교 시 threshold 조건 항상 명시 (R9.2)
- DIFFUSE 대비 개선 주장 시 R9.4 통계 검정 필수
- overall 성능은 Macro-F1로 보고 (R9.5)


════════════════════════════════════════════════════════════
CROSS-AXIS DECISION FRAMEWORK
════════════════════════════════════════════════════════════

## 새 알고리즘/구조 제안 시 필수 통과 체크포인트

### Level 1: 문제 진단 (제안 전)
□ Collapse score 계산 → Axis 1
□ Bias score 계산 → Axis 2
□ Triplet active ratio 계산 → Axis 3
□ Modality gradient norm 비교 → Axis 4

### Level 2: 수학적 정당화 (제안 시)
□ 어느 Axis의 어떤 레퍼런스 [Rx.y]를 근거로 하는가?
□ 핵심 수식이 현재 코드에 어떻게 매핑되는가?
□ 해당 방법의 ⚠️ 주의사항을 확인했는가?
□ 기존 방법([Rx.y])으로 먼저 해결 시도했는가?

### Level 3: 실험 설계 검증 (실험 전)
□ Evaluation metric이 Axis 9 프로토콜을 따르는가?
□ 통계 검정 방법이 명시되어 있는가?
□ Ablation이 one-at-a-time 원칙을 따르는가?

### Level 4: 결과 해석 (실험 후)
□ 개선이 특정 Axis의 문제 해결로 설명 가능한가?
□ 예상치 못한 결과는 어느 Axis에서 설명되는가?
□ Nature Methods에서 "왜 이 방법인가?" 질문에 [Rx.y]로 답할 수 있는가?

## 레퍼런스 우선순위 (같은 문제에 여러 방법이 있을 때)
1. 수학적 이론 보장이 있는 것 우선
2. 같은 도메인(bioinformatics/proteomics) 적용 사례 있는 것 우선
3. 계산 비용이 낮은 것 우선
4. ablation으로 기여 입증 가능한 것 우선
