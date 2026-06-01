#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_GPU_v7b.py — v7b 2-GPU 병렬 실행 스크립트

v7b 변경 (v6g 대비):
  [v7b-1] Phase 1: Type-A=Triplet | Type-B=Proto-kN (Gap Statistic k 선택)
  [v7b-2] Phase 1→2: Prototype Warm Init (prediction_out 가중치 초기화)

acceptance criterion (Exp 2b):
  Macro-AUPRC > v6g(0.465) + 0.02 = 0.485
  Type-B 개선 필수: ≥ 2/3 Type-B GO term에서 AUPRC 증가

실행:
  cd hMuscle/model/
  conda activate isoform_env
  python run_GPU_v7b.py
"""
from datetime import datetime
import os
import subprocess
import time
import threading
import numpy as np

MODEL_SCRIPT = "v7b_integrated_full_model.py"
GO_LIST_FILE = 'v6_go_list.txt'
LOG_DIR      = '../logs_isoform/'
GPU_LIST     = [0, 1]

ESM2_FILES = [
    '../data/esm2_train_human.npy',
    '../data/esm2_train_human_mask.npy',
    '../data/esm2_train_swissprot.npy',
    '../data/esm2_train_swissprot_mask.npy',
    '../data/esm2_embeddings_t30_150M.npy',
    '../data/esm2_mask.npy',
]
SEQ_FILE = 'my_sequence_matrix_fixed.npy'


def check_required_files():
    ok = True
    missing = [f for f in ESM2_FILES if not os.path.exists(f)]
    if missing:
        print("[ERROR] ESM-2 파일 없음:")
        for f in missing:
            print("  ", f)
        ok = False
    else:
        print("[Check] ESM-2 파일 확인 완료")
        for f in ESM2_FILES:
            arr = np.load(f)
            print("  {} {}".format(f.split('/')[-1], arr.shape))
    if not os.path.exists(SEQ_FILE):
        print("[ERROR] sequence matrix 없음: {}".format(SEQ_FILE))
        ok = False
    else:
        arr = np.load(SEQ_FILE)
        print("[Check] {} {} dtype={}".format(SEQ_FILE, arr.shape, arr.dtype))
    return ok


def run_model_on_gpu(go_terms, gpu_id):
    total = len(go_terms)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=" * 60)
    print(" [GPU {}] v7b — {} GO terms".format(gpu_id, total))
    print("=" * 60)

    for i, go_term in enumerate(go_terms):
        go_term = go_term.strip()
        if not go_term:
            continue

        safe_go    = go_term.replace(':', '_')
        log_folder = os.path.join(LOG_DIR, safe_go)
        os.makedirs(log_folder, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = os.path.join(log_folder,
                                'v7b_{}_{}_Full.log'.format(safe_go, date_str))

        print("[GPU {}] [{}/{}] Processing: {}".format(gpu_id, i + 1, total, go_term))
        print("         Log: {}".format(log_file))

        cmd = "CUDA_VISIBLE_DEVICES={} python -u {} {} > {} 2>&1".format(
            gpu_id, MODEL_SCRIPT, go_term, log_file)

        start_time = time.time()
        try:
            ret = subprocess.call(cmd, shell=True)
            elapsed = time.time() - start_time
            status = "[Done]" if ret == 0 else "[Error] ret={}".format(ret)
            print("[GPU {}]   {} {:.0f}s".format(gpu_id, status, elapsed))
        except Exception as e:
            print("[GPU {}]   [Fail] {}".format(gpu_id, str(e)))

        print("-" * 60)
        time.sleep(2)


if __name__ == '__main__':
    print("=" * 60)
    print(" v7b Runner")
    print(" [v7b-1] Phase 1: Type-A=Triplet | Type-B=Proto-kN")
    print(" [v7b-2] Phase 1→2: Prototype Warm Init")
    print(" acceptance: Macro-AUPRC > 0.485 (v6g+0.02)")
    print("=" * 60)

    if not check_required_files():
        exit(1)

    if not os.path.exists(GO_LIST_FILE):
        print("Error: {} not found.".format(GO_LIST_FILE))
        exit(1)

    with open(GO_LIST_FILE, 'r') as f:
        go_terms = [line.strip() for line in f if line.strip()]

    if not go_terms:
        print("No GO terms found.")
        exit(1)

    total  = len(go_terms)
    half   = (total + 1) // 2
    gpu0_terms = go_terms[:half]
    gpu1_terms = go_terms[half:]

    print("\n GO terms: {}".format(total))
    print(" GPU 0 → {}".format(gpu0_terms))
    print(" GPU 1 → {}".format(gpu1_terms))
    print("=" * 60)

    threads = []
    for gpu_id, terms in zip(GPU_LIST, [gpu0_terms, gpu1_terms]):
        if not terms:
            continue
        t = threading.Thread(target=run_model_on_gpu, args=(terms, gpu_id))
        threads.append(t)

    wall_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_total = time.time() - wall_start

    print("\n" + "=" * 60)
    print(" All tasks finished. Wall time: {:.0f}s ({:.1f}min)".format(
        wall_total, wall_total / 60))
    print(" Results: hMuscle/results_isoform/GO_*/v7b_integrated_*/")
    print(" Logs:    {}GO_*/v7b_*_Full.log".format(LOG_DIR))
    print("=" * 60)
