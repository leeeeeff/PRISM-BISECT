from datetime import datetime
# -*- coding: utf-8 -*-
import os
import subprocess
import time
import threading

# [설정]
GO_LIST_FILE = 'GO_terms/switch_go_list.txt'
LOG_DIR = '../logs_isoform/'
VERSION_TAG = 'v5-5'
MODEL_SCRIPT = 'v5-5_integrated_full_model.py'
GPU_LIST = [0, 1]  # 사용할 GPU ID 목록

def run_model_on_gpu(go_terms, gpu_id):
    """
    배정된 GO term 리스트를 지정된 GPU에서 순차적으로 실행합니다.
    """
    total = len(go_terms)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=====================================================")
    print(" [GPU {}] Starting {} Sequential Batch Processing".format(gpu_id, VERSION_TAG))
    print(" [GPU {}] Total GO terms: {}".format(gpu_id, total))
    print("=====================================================")

    for i, go_term in enumerate(go_terms):
        go_term = go_term.strip()
        if not go_term:
            continue

        safe_go = go_term.replace(':', '_')
        log_folder = os.path.join(LOG_DIR, safe_go)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_folder, f'{VERSION_TAG}_{safe_go}_{date_str}_Full.log')

        print("[GPU {}] [{}/{}] Now Processing ({}): {}".format(gpu_id, VERSION_TAG, i + 1, total, go_term))
        print("         Log File: {}".format(log_file))

        cmd = "CUDA_VISIBLE_DEVICES={} python -u {} {} > {} 2>&1".format(
            gpu_id, MODEL_SCRIPT, go_term, log_file)

        start_time = time.time()
        try:
            ret = subprocess.call(cmd, shell=True)
            elapsed = time.time() - start_time
            if ret == 0:
                print("[GPU {}]       [Done] Duration: {:.2f}s".format(gpu_id, elapsed))
            else:
                print("[GPU {}]       [Error] Failed with return code {}. Check log.".format(gpu_id, ret))
        except Exception as e:
            print("[GPU {}]       [Fail] Execution failed: {}".format(gpu_id, str(e)))

        print("-----------------------------------------------------")
        time.sleep(2)


if __name__ == '__main__':
    if not os.path.exists(GO_LIST_FILE):
        print("Error: {} file not found.".format(GO_LIST_FILE))
        exit(1)

    with open(GO_LIST_FILE, 'r') as f:
        go_terms = [line.strip() for line in f if line.strip()]

    if not go_terms:
        print("No GO terms found in the file.")
        exit(1)

    total = len(go_terms)
    half = (total + 1) // 2  # Python 3 정수 나눗셈
    gpu0_terms = go_terms[:half]
    gpu1_terms = go_terms[half:]

    print("=====================================================")
    print(" Dual GPU Batch Processing")
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
    print("\nAll tasks finished. Total wall time: {:.2f}s ({:.1f}min)".format(
        wall_elapsed, wall_elapsed / 60))
    print("Check the logs in {}".format(LOG_DIR))
