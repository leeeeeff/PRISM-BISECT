# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

# [설정] 경로
RAW_DATA_DIR = '../../data/raw_data/data/'

# 입력 파일들
TRAIN_GENE_LIST = os.path.join(RAW_DATA_DIR, 'id_lists/train_gene_list.npy')
MY_GENE_LIST = '../../model/my_gene_list_fixed.npy'
MY_COEXP_NET = '../../results/co-expression_net/coexp_net.npy' # (36748 x 36748)

# 출력 파일명
EXPANDED_NET_FILE = "../../results/co-expression_net/coexp_net_expanded.npy"

def expand_network():
    print("Loading Lists and Network...")
    try:
        # 1. 리스트 로드
        train_genes = np.load(TRAIN_GENE_LIST)
        my_genes = np.load(MY_GENE_LIST)
        
        n_train = len(train_genes)
        n_test = len(my_genes)
        n_total = n_train + n_test
        
        print(" - Train Genes: {}".format(n_train))
        print(" - Test Genes:  {}".format(n_test))
        print(" - Total Target: {}".format(n_total))
        
        # 2. 내 네트워크 로드
        my_net = np.load(MY_COEXP_NET)
        print(" - My Network Shape: {}".format(my_net.shape))
        
        if my_net.shape[0] != n_test:
            print("[Error] My Network size ({}) does not match My Gene List ({})!".format(my_net.shape[0], n_test))
            print("Please check if you generated coexp_net using the FIXED gene list.")
            # 만약 크기가 안 맞으면 여기서 강제로 맞춰줄 수도 있지만, 일단은 경고.
            # (여기서는 일단 진행하도록 패딩 로직 수행)

    except Exception as e:
        print("Error loading files: " + str(e))
        sys.exit(1)

    print("\nCreating Expanded Network ({} x {})...".format(n_total, n_total))
    
    # 3. 빈 거대 행렬 생성 (0으로 초기화)
    # 메모리가 부족할 수 있으니 float16이나 float32 사용 권장
    expanded_net = np.zeros((n_total, n_total), dtype=np.float32)
    
    # 4. 내 네트워크(Test 부분)를 우측 하단에 끼워넣기
    # 구조:
    # [ Train-Train(0) | Train-Test(0) ]
    # [ Test-Train(0)  | Test-Test(My) ]
    
    # 실제 데이터가 들어갈 범위 계산
    # 학습 데이터 개수만큼 건너뛰고 시작
    start_idx = n_train
    end_idx = n_train + my_net.shape[0] # 내 네트워크 크기만큼
    
    print("Inserting user network into index [{} : {}]...".format(start_idx, end_idx))
    
    expanded_net[start_idx:end_idx, start_idx:end_idx] = my_net

    # 5. 저장
    print("Saving to {}...".format(EXPANDED_NET_FILE))
    np.save(EXPANDED_NET_FILE, expanded_net)
    print("Done! Update 'model_fixed.py' to use this file.")

if __name__ == "__main__":
    expand_network()
