#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_GPU_v6e.py — v6e (v6d + adaptive Phase2 + no-LP) 실행 스크립트

v6e 핵심 변경:
  - [v6e-1] Phase 0 GO term Type 자동 분류 (sep_ratio 기준 A/B)
  - [v6e-2] Phase 2 adaptive: Type B lr=0.0001/clipnorm=0.5/patience=3
  - [v6e-3] Label Propagation 제거 (alpha=0.0 고정)
  - [v6e-4] Coverage 정규화 clip(n_pos*4/256, 10, 80)

실행:
  cd hMuscle/model/
  python run_GPU_v6e.py
"""
from datetime import datetime
import os
import subprocess
import time
import threading
import numpy as np

VERSION_TAG  = 'v6e'
MODEL_SCRIPT = "v6e_integrated_full_model.py"
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
    missing_esm2 = [f for f in ESM2_FILES if not os.path.exists(f)]
    if missing_esm2:
        print("[ERROR] 다음 ESM-2 파일이 없습니다:")
        for f in missing_esm2:
            print("  ", f)
        ok = False
    else:
        print("[Check] ESM-2 파일 확인 완료")
        for f in ESM2_FILES:
            arr = np.load(f)
            print("  {} {}".format(f.split('/')[-1], arr.shape))

    if not os.path.exists(SEQ_FILE):
        print("[ERROR] sequence matrix 없음: {}".format(SEQ_FILE))
        print("  → 생성 필요: preprocessing/build_sequence_matrix.py")
        ok = False
    else:
        arr = np.load(SEQ_FILE)
        print("[Check] {} {} dtype={}".format(SEQ_FILE.split('/')[-1], arr.shape, arr.dtype))

    return ok


def run_model_on_gpu(go_terms, gpu_id):
    total = len(go_terms)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=" * 56)
    print(" [GPU {}] {} — {} GO terms".format(gpu_id, VERSION_TAG, total))
    print("=" * 56)

    for i, go_term in enumerate(go_terms):
        go_term = go_term.strip()
        if not go_term:
            continue

        safe_go    = go_term.replace(':', '_')
        log_folder = os.path.join(LOG_DIR, safe_go)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = os.path.join(log_folder,
                                '{}_{}_{}_Full.log'.format(VERSION_TAG, safe_go, date_str))

        print("[GPU {}] [{}/{}] Processing: {}".format(gpu_id, i + 1, total, go_term))
        print("         Log: {}".format(log_file))

        cmd = "CUDA_VISIBLE_DEVICES={} python -u {} {} > {} 2>&1".format(
            gpu_id, MODEL_SCRIPT, go_term, log_file)

        start_time = time.time()
        try:
            ret = subprocess.call(cmd, shell=True)
            elapsed = time.time() - start_time
            if ret == 0:
                print("[GPU {}]   [Done] {:.0f}s".format(gpu_id, elapsed))
            else:
                print("[GPU {}]   [Error] ret={} — check log".format(gpu_id, ret))
        except Exception as e:
            print("[GPU {}]   [Fail] {}".format(gpu_id, str(e)))

        print("-" * 56)
        time.sleep(2)


if __name__ == '__main__':
    print("=" * 56)
    print(" v6e (adaptive Phase2 + no-LP) GPU Runner")
    print("=" * 56)

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

    total = len(go_terms)
    half  = (total + 1) // 2
    gpu0_terms = go_terms[:half]
    gpu1_terms = go_terms[half:]

    print("\n GO terms: {}".format(total))
    print(" GPU 0 → {}".format(gpu0_terms))
    print(" GPU 1 → {}".format(gpu1_terms))
    print("=" * 56)
    print("\n v6e 변경 사항:")
    print("  [v6e-1] Phase 0 Type 분류 (sep_ratio < 1.15 → Type-B)")
    print("  [v6e-2] Type-B Phase2: lr=0.0001 clipnorm=0.5 patience=3")
    print("  [v6e-3] LP 제거 (alpha=0.0 고정)")
    print("  [v6e-4] Coverage clip(n*4/256, 10, 80)")
    print("=" * 56)

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

    wall_elapsed = time.time() - wall_start
    print("\nAll tasks finished. Wall time: {:.0f}s ({:.1f}min)".format(
        wall_elapsed, wall_elapsed / 60))
    print("Results: hMuscle/results_isoform/GO_*/")
    print("Logs:    {}".format(LOG_DIR))
