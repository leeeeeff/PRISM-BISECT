from datetime import datetime
# -*- coding: utf-8 -*-
# run_GPU_v8-D256.py — v8b-D256 실험 런너
# SwissProt 유지 + embedding 64→256 확장 (2×2 ablation의 마지막 셀)
#
# 2×2 ablation:
#   v8b      (SP✓, dim=64):  Macro=0.357  [완료]
#   P3       (SP✗, dim=64):  Macro=0.298  [완료]
#   P3-256   (SP✗, dim=256): [진행 중]
#   v8b-D256 (SP✓, dim=256): [본 실험]

import os, subprocess, time, threading

GO_LIST_FILE = 'v8_go_list.txt'
LOG_DIR      = '../logs_isoform/'
VERSION_TAG  = 'v8b-D256'
MODEL_SCRIPT = 'v8b-D256_integrated_full_model.py'
GPU_LIST     = [0, 1]

def run_model_on_gpu(go_terms, gpu_id):
    total = len(go_terms)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("=" * 65)
    print(" [GPU {}] v8b-D256 (SwissProt✓ + dim256) — {} GO terms".format(gpu_id, total))
    print("=" * 65)

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

        print("-" * 65)
        time.sleep(2)


if __name__ == '__main__':
    if not os.path.exists(GO_LIST_FILE):
        print("Error: {} not found.".format(GO_LIST_FILE))
        exit(1)

    with open(GO_LIST_FILE) as f:
        go_terms = [l.strip() for l in f if l.strip()]

    total      = len(go_terms)
    half       = (total + 1) // 2
    gpu0_terms = go_terms[:half]
    gpu1_terms = go_terms[half:]

    print("=" * 65)
    print(" v8b-D256 — SwissProt✓ + embedding 64→256 — Dual GPU")
    print(" Total GO terms: {}".format(total))
    print(" GPU 0 -> {}".format(gpu0_terms))
    print(" GPU 1 -> {}".format(gpu1_terms))
    print(" 2×2 ablation 비교:")
    print("   v8b      (SP✓, 64d):  Macro=0.357  [완료]")
    print("   P3       (SP✗, 64d):  Macro=0.298  [완료]")
    print("   P3-256   (SP✗, 256d): [진행 중]")
    print("   v8b-D256 (SP✓, 256d): [본 실험]  ← 기대 Macro≈0.45+")
    print("   ESM-2 LR (640d,human): Macro=0.561 ← 목표 상한")
    print("=" * 65)

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
