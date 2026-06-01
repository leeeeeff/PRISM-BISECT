#!/usr/bin/env python3
"""
post_hmmscan_pipeline.py
=========================
hmmscan 완료 후 자동 실행:
  1. build_train_domain_from_hmmscan.py  → train_domain_delta_hmmscan.npy
  2. v10_isoform_delta_model.py all      → v10-Abs with hmmscan features 재측정
"""
import subprocess, sys, os, time

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
BASE = os.getcwd()
LOG_DIR = os.path.join(BASE, '..', 'logs_isoform')

def run(cmd, log_path):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        proc.wait()
    return proc.returncode

ts = time.strftime('%Y%m%d_%H%M')

# Step 1: hmmscan 결과 → train domain matrix
rc = run(
    ['conda', 'run', '-n', 'isoform_env',
     'python', 'preprocessing/build_train_domain_from_hmmscan.py'],
    os.path.join(LOG_DIR, f'train_domain_hmmscan_{ts}.log')
)
if rc != 0:
    print(f"[ERROR] build_train_domain_from_hmmscan.py failed (rc={rc})")
    sys.exit(1)

print("\n[OK] train_domain_delta_hmmscan.npy generated")

# Step 2: v10-Abs 재실행 (hmmscan feature 사용)
rc = run(
    ['conda', 'run', '-n', 'isoform_env',
     'python', 'model/v10_isoform_delta_model.py', 'all'],
    os.path.join(LOG_DIR, f'v10_delta_hmmscan_{ts}.log')
)
if rc != 0:
    print(f"[ERROR] v10_isoform_delta_model.py failed (rc={rc})")
    sys.exit(1)

print(f"\n[DONE] v10-Abs with hmmscan features complete. Log: v10_delta_hmmscan_{ts}.log")
