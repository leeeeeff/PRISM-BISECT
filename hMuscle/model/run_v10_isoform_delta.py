#!/usr/bin/env python3
"""
run_v10_isoform_delta.py — v10_isoform_delta_model.py 실행 래퍼
GPU:1 사용, 로그 자동 저장

사용:
  conda activate isoform_env
  python hMuscle/model/run_v10_isoform_delta.py [go_term|all]
  nohup python hMuscle/model/run_v10_isoform_delta.py all > logs_isoform/v10_delta_YYYYMMDD_HHMM.log 2>&1 &
"""
import subprocess, sys, os, time

go_arg = sys.argv[1] if len(sys.argv) > 1 else 'all'

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')

cmd = [
    'conda', 'run', '-n', 'isoform_env',
    'python', 'hMuscle/model/v10_isoform_delta_model.py', go_arg
]

log_path = f'logs_isoform/v10_delta_{time.strftime("%Y%m%d_%H%M")}.log'
print(f"Running: {' '.join(cmd)}")
print(f"Log: {log_path}")

with open(log_path, 'w') as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
        log.write(line)
        log.flush()
    proc.wait()

print(f"\nExit code: {proc.returncode}")
print(f"Log saved: {log_path}")
