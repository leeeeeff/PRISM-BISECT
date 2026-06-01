from datetime import datetime
# -*- coding: utf-8 -*-
# run_GPU_v9-1.py — v9-1 실험 런너
#
# v9-1 전략:
#   Type-B: Phase 1 → Phase 3 (Augmented LR: emb + splicing_delta_v2, Phase 2 skip)
#   Type-A: Phase 1 → Phase 2 (v8b 동일) → sigmoid head
#   + GENCODE canonical domain_delta_v2
#   + Continuous splicing_delta_v2 (150dim)
# 5개 GO term 두 GPU 분산

import os
import subprocess
import time
import threading

GO_LIST_FILE = 'v8_go_list.txt'
LOG_DIR      = '../logs_isoform/'
VERSION_TAG  = 'v9-1'
MODEL_SCRIPT = 'v9-1_integrated_full_model.py'
GPU_LIST     = [0, 1]

def run_model_on_gpu(go_terms, gpu_id):
    total = len(go_terms)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=" * 53)
    print(" [GPU {}] v9-1 (Augmented LR + GENCODE canonical) — {} GO terms".format(gpu_id, total))
    print("=" * 53)

    for i, go_term in enumerate(go_terms):
        go_term = go_term.strip()
        if not go_term:
            continue

        safe_go    = go_term.replace(':', '_')
        log_folder = os.path.join(LOG_DIR, safe_go)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = os.path.join(log_folder, '{}_{}_{}_{}_Full.log'.format(
            VERSION_TAG, safe_go, date_str, gpu_id))

        print("[GPU {}] [{}/{}] {} ...".format(gpu_id, i + 1, total, go_term))
        print("         Log: {}".format(log_file))

        cmd = "CUDA_VISIBLE_DEVICES={} python -u {} {} > {} 2>&1".format(
            gpu_id, MODEL_SCRIPT, go_term, log_file)

        start_time = time.time()
        try:
            ret = subprocess.call(cmd, shell=True)
            elapsed = time.time() - start_time
            if ret == 0:
                print("[GPU {}]   [Done] {:.1f}min".format(gpu_id, elapsed / 60))
            else:
                print("[GPU {}]   [Error] ret={} — check log".format(gpu_id, ret))
        except Exception as e:
            print("[GPU {}]   [Fail] {}".format(gpu_id, str(e)))

        print("-" * 53)
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

    print("=" * 53)
    print(" v9-1 (Augmented LR + GENCODE canonical + splicing_delta_v2)")
    print(" Total GO terms: {}".format(total))
    print(" GPU 0 -> {}".format(gpu0_terms))
    print(" GPU 1 -> {}".format(gpu1_terms))
    print("=" * 53)

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
    print("\nAll tasks finished. Total: {:.1f}min".format(wall_elapsed / 60))
    print("Logs: {}".format(LOG_DIR))
