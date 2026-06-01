# Loss Function Rules

## Focal Loss [R1.1]
FL(p_t) = -α_t(1-p_t)^γ · log(p_t)
- γ=2 default. γ < 1 금지 (CE로 퇴화)
- α: R1.2 class-balanced 공식 사용 권장

## Triplet Loss [R3.1]
L(a,p,n) = max(d(a,p) - d(a,n) + margin, 0)
- margin=0.3 default, distance=cosine
- Negative: cross-gene 필수, intra-gene 금지 [R2.1]
- active ratio < 5% → hard negative mining [R3.2]

## Combined
total_loss = λ_focal · focal + λ_triplet · triplet
- current: λ_focal=1.0, λ_triplet=0.5
- gene-bias 지속 시 λ_triplet 증가

## Tuning Rules
- mode collapse → γ 증가 (2.5, 3.0) [R1.1]
- embedding collapse → margin 증가 (0.5, 1.0) [R3.1]
- triplet 불안정 → SupCon으로 교체 검토 [R3.3]
