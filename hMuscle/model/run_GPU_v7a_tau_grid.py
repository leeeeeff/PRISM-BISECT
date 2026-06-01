#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_GPU_v7a_tau_grid.py — v7a τ 그리드 탐색 실행 스크립트

목적:
  v7a (SupCon) 기본 τ=0.1 결과 분석 후 최적 τ 탐색.

  관찰 (v7a τ=0.1):
    - Type-A (GO:0006096): AUPRC 0.823→0.650 (Phase 1 sep=3.39 과집중 → Phase 2 gradient 소실)
    - Type-B (GO:0006941): 0.121→0.165 (개선, 낮은 sep에서 SupCon 효과)
    - Type-A Phase 2 max_epochs=15: 수렴 미완료 (epoch=15에서 여전히 상승)

  탐색 변수:
    - τ ∈ [0.2, 0.3]: 덜 집중적인 임베딩 → Phase 2 gradient 복구
    - Phase 2 max_epochs: 25 (모든 Type에 통일)

  실행 순서:
    1) τ=0.2 (5 GO terms, GPU 0+1 병렬)
    2) τ=0.3 (5 GO terms, GPU 0+1 병렬)

실행:
  cd hMuscle/model/
  python run_GPU_v7a_tau_grid.py
"""
from datetime import datetime
import os
import subprocess
import time
import threading
import numpy as np

MODEL_SCRIPT = "v7a_integrated_full_model.py"
GO_LIST_FILE = 'v6_go_list.txt'
LOG_DIR      = '../logs_isoform/'
GPU_LIST     = [0, 1]

# 탐색할 τ 값 목록
TAU_GRID = [0.2, 0.3]

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
    if not os.path.exists(SEQ_FILE):
        print("[ERROR] sequence matrix 없음: {}".format(SEQ_FILE))
        ok = False
    else:
        arr = np.load(SEQ_FILE)
        print("[Check] {} {} dtype={}".format(SEQ_FILE, arr.shape, arr.dtype))
    return ok


def run_model_on_gpu(go_terms, gpu_id, tau):
    """특정 τ로 GO term 목록을 순차 실행."""
    tau_str = "tau{:03d}".format(int(tau * 100))
    version_tag = "v7a_{}".format(tau_str)
    total = len(go_terms)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=" * 60)
    print(" [GPU {}] {} ({} GO terms)".format(gpu_id, version_tag, total))
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
                                '{}_{}_{}_Full.log'.format(version_tag, safe_go, date_str))

        print("[GPU {}] [{}/{}] {} τ={}".format(gpu_id, i + 1, total, go_term, tau))
        print("         Log: {}".format(log_file))

        # τ를 3번째 인자로 전달
        cmd = "CUDA_VISIBLE_DEVICES={} python -u {} {} {} > {} 2>&1".format(
            gpu_id, MODEL_SCRIPT, go_term, tau, log_file)

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


def run_tau(tau, go_terms):
    """τ 하나에 대해 전체 GO term을 2-GPU 병렬 실행. 완료까지 블로킹."""
    tau_str = "tau{:03d}".format(int(tau * 100))
    total = len(go_terms)
    half  = (total + 1) // 2
    gpu0_terms = go_terms[:half]
    gpu1_terms = go_terms[half:]

    print("\n" + "=" * 60)
    print(" τ={} 시작 — GO terms: {}".format(tau, total))
    print(" GPU 0 → {}".format(gpu0_terms))
    print(" GPU 1 → {}".format(gpu1_terms))
    print("=" * 60)

    threads = []
    for gpu_id, terms in zip(GPU_LIST, [gpu0_terms, gpu1_terms]):
        if not terms:
            continue
        t = threading.Thread(target=run_model_on_gpu, args=(terms, gpu_id, tau))
        threads.append(t)

    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t_start

    print("\n[τ={}] 완료. Wall time: {:.0f}s ({:.1f}min)".format(tau, elapsed, elapsed / 60))


if __name__ == '__main__':
    print("=" * 60)
    print(" v7a τ Grid Runner")
    print(" τ 탐색: {}".format(TAU_GRID))
    print(" 변경: Phase 2 max_epochs=25 (모든 Type 통일)")
    print(" 기준: v7a τ=0.1 Macro-AUPRC ← baseline")
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

    print("\n탐색 τ 값: {}".format(TAU_GRID))
    print("GO terms: {}".format(go_terms))

    wall_start = time.time()

    for tau in TAU_GRID:
        run_tau(tau, go_terms)

    wall_total = time.time() - wall_start
    print("\n" + "=" * 60)
    print(" 모든 τ 실험 완료. 총 Wall time: {:.0f}s ({:.1f}h)".format(
        wall_total, wall_total / 3600))
    print(" 결과: hMuscle/results_isoform/GO_*/v7a_tau*/")
    print(" 로그: {}GO_*/v7a_tau*_Full.log".format(LOG_DIR))
    print("=" * 60)
