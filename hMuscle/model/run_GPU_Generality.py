from datetime import datetime
# -*- coding: utf-8 -*-
"""
run_GPU_Generality.py
---------------------
v5-5 일반성 검증 실행 스크립트.
Tier-1 GO term (GO:0006941, GO:0030017, GO:0003774, GO:0007204)을
기존 4개 GO term 결과와 독립적인 디렉토리에 저장.

결과:  hMuscle/results_isoform/GO_XXXXXXX/v5-5_integrated_YYYYMMDD/
로그:  hMuscle/logs_isoform/GO_XXXXXXX/v5-5_GO_XXXXXXX_YYYYMMDD_Full.log
"""
import os
import subprocess
import time
import threading

# [설정]
GO_LIST_FILE = 'generality_go_list.txt'   # 기존 switch_go_list.txt와 분리
LOG_DIR      = '../logs_isoform/'
VERSION_TAG  = 'v5-5'
MODEL_SCRIPT = 'v5-5_integrated_full_model.py'
GPU_LIST     = [0, 1]


def run_model_on_gpu(go_terms, gpu_id):
    total = len(go_terms)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=====================================================")
    print(" [GPU {}] v5-5 Generality Check — {} GO terms".format(gpu_id, total))
    print("=====================================================")

    for i, go_term in enumerate(go_terms):
        go_term = go_term.strip()
        if not go_term:
            continue

        safe_go  = go_term.replace(':', '_')
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
                print("[GPU {}]   [Error] code={}, check log".format(gpu_id, ret))
        except Exception as e:
            print("[GPU {}]   [Fail] {}".format(gpu_id, str(e)))

        print("-----------------------------------------------------")
        time.sleep(2)


if __name__ == '__main__':
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

    print("=====================================================")
    print(" v5-5 Generality Check — Dual GPU")
    print(" Total GO terms: {}".format(total))
    print(" GPU 0 -> {}".format(gpu0_terms))
    print(" GPU 1 -> {}".format(gpu1_terms))
    print("=====================================================")

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
    print("Results: hMuscle/results_isoform/GO_{0006941,0030017,0003774,0007204}/")
    print("Logs:    {}".format(LOG_DIR))
