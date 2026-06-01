# -*- coding: utf-8 -*-
import multiprocessing
import subprocess
import os
import time

# [설정]
GO_LIST_FILE = 'GO_terms/switch_go_list.txt'  # GO term 리스트 파일
NUM_PROCESSES = 1               # 동시에 실행할 개수 (노드 수)
LOG_DIR = '../logs/'                 # 로그 저장 폴더

def run_model(go_term):
    """
    개별 모델을 실행하는 함수 (하나의 프로세스가 됨)
    """
    go_term = go_term.strip()
    if not go_term: return

    # 로그 파일 경로 설정
    safe_go = go_term.replace(':', '_')
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except:
            pass # 동시에 만들다 에러나는 경우 무시
    
    log_folder = os.path.join(LOG_DIR, safe_go)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_file = os.path.join(log_folder, "{}_Focal.log".format(safe_go))
    
    print "[Start] {} (Log: {})".format(go_term, log_file)
    
    # 명령어 생성
    # -u 옵션: 파이썬 출력을 버퍼링 없이 바로 찍기
    # > ... 2>&1: 화면에 출력되는 모든 내용을 로그 파일로 저장
    cmd = "python -u focal_model.py {} > {} 2>&1".format(go_term, log_file)
    
    # 실행
    try:
        ret = subprocess.call(cmd, shell=True)
        if ret == 0:
            print "[Done]  {}".format(go_term)
        else:
            print "[Error] {} failed. Check {}".format(go_term, log_file)
    except Exception as e:
        print "[Fail]  {} execution failed: {}".format(go_term, str(e))

if __name__ == '__main__':
    # 1. GO term 리스트 읽기
    if not os.path.exists(GO_LIST_FILE):
        print "Error: {} file not found.".format(GO_LIST_FILE)
        exit(1)
        
    with open(GO_LIST_FILE, 'r') as f:
        go_terms = [line.strip() for line in f if line.strip()]

    print "Total GO terms to analyze: {}".format(len(go_terms))
    print "Parallel processes: {}".format(NUM_PROCESSES)
    
    # 2. 병렬 처리 시작
    # Pool을 사용하여 프로세스를 생성
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    
    # map 함수가 리스트의 각 항목을 run_model 함수에 할당하여 병렬 실행
    pool.map(run_model, go_terms)
    
    # 3. 종료 대기
    pool.close()
    pool.join()
    
    print "\nAll tasks finished."
