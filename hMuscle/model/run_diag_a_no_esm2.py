"""
Diagnostic A: ESM-2 비활성화 실험
NO_ESM2=1 환경변수로 v7c를 실행 — ESM-2 gating mask를 전부 0으로 설정.

목적: Type-B GO term에서 ESM-2가 적극적으로 해롭다는 가설 검증.
     ESM-2 제거 시 AUPRC가 v5-5 수준(GO:0007204≈0.507)으로 회복되면
     → ESM-2의 evolutionary bias가 Type-B에서 noise로 작용 (가능성 1 확인)
     ESM-2 제거 후 변화 없으면
     → ESM-2가 문제가 아님 → Phase 1 contrastive 한계 (가능성 3: data 상한)

진단 B 결과 요약 (실행 전 참고):
- GO:0007204 Phase1 centroid_cos=0.813 (낮을수록 좋음) → 분리 불량
- GO:0003774 Phase1 centroid_cos=0.438 → 분리 양호
- Phase1 in-sample LinAUPRC: GO:0007204=0.199, GO:0003774=0.477

비교:
- v7c 결과   (ESM-2 ON):  GO:0007204=0.122, GO:0006941=0.085, GO:0003774=0.420
- v7c-noesm2 (ESM-2 OFF): ???

실행:
  conda activate isoform_env
  cd ~/sw1686/DIFFUSE/hMuscle/model
  nohup python run_diag_a_no_esm2.py > ../logs_isoform/diag_a_runner_$(date +%Y%m%d_%H%M).log 2>&1 &

로그 경로: ../logs_isoform/diag_a_noesm2_{GO}_{DATE}.log
"""

import subprocess
import os
import sys
from datetime import datetime

# 진단 대상 GO terms — Type-B 2개 + Type-A 1개 (대조군)
# GO:0003774가 ESM-2 제거 후 하락하면 → ESM-2는 Type-A에서 필수 (대조군 역할 확인)
GO_TERMS_COLON = [
    'GO:0007204',  # Type-B, AUPRC=0.122 (핵심 케이스)
    'GO:0006941',  # Type-B, AUPRC=0.085
    'GO:0003774',  # Type-A, AUPRC=0.420 (대조군)
]

date_str     = datetime.now().strftime('%Y%m%d_%H%M')
model_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'v7c_integrated_full_model.py')
python_exe   = sys.executable

env = os.environ.copy()
env['NO_ESM2'] = '1'

# GPU 설정: 두 GPU 중 하나 사용 (0번 기본)
env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')

print("=" * 60)
print("Diagnostic A: ESM-2 비활성화 실험 (NO_ESM2=1)")
print(f"GO terms: {GO_TERMS_COLON}")
print(f"Python: {python_exe}")
print(f"Model: {model_script}")
print(f"GPU: {env['CUDA_VISIBLE_DEVICES']}")
print("=" * 60)

# 각 GO term을 순차 실행 (Phase 1이 오래 걸리므로 순차가 안전)
results = {}
for go_term in GO_TERMS_COLON:
    safe_go  = go_term.replace(':', '_')
    go_log   = f'../logs_isoform/diag_a_noesm2_{safe_go}_{date_str}.log'
    abs_log  = os.path.join(os.path.dirname(model_script), go_log)

    print(f"\n[Diag-A] {go_term} — NO_ESM2=1 ...")
    print(f"  Log → {abs_log}")

    cmd = [python_exe, model_script, go_term]

    with open(abs_log, 'w') as f:
        proc = subprocess.run(
            cmd, env=env,
            cwd=os.path.dirname(model_script),
            stdout=f, stderr=subprocess.STDOUT
        )

    print(f"  Exit code: {proc.returncode}")
    print(f"  Final result:")
    subprocess.run(
        ['grep', '-E', r'\[Final\]', abs_log],
        check=False
    )
    results[go_term] = abs_log

print("\n" + "="*60)
print("DIAGNOSTIC A COMPLETE")
print("="*60)
print("v7c (ESM-2 ON)  : GO:0007204=0.122 | GO:0006941=0.085 | GO:0003774=0.420")
print("v7c (ESM-2 OFF) : 위 [Final] 라인 확인")
print("\n판단 기준:")
print("  GO:0007204 OFF > ON → ESM-2가 Type-B에서 해롭다 → 전략 전환")
print("  GO:0007204 OFF ≈ ON → ESM-2가 문제 아님 → data 상한 검토")
print("  GO:0003774 OFF < ON → ESM-2가 Type-A에서 필수 (예상, 대조군 확인)")

