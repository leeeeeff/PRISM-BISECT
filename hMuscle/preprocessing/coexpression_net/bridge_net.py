# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

# [설정] 경로
RAW_DATA_DIR = '../../data/raw_data/data/'

# 입력 파일들
TRAIN_GENE_LIST = os.path.join(RAW_DATA_DIR, 'id_lists/train_gene_list.npy')
TRAIN_COEXP_NET = '../../data/raw_data/data/co-expression_net/coexp_net_unified.npy' # (Original Train Net)

MY_GENE_LIST = '../../model/my_gene_list_fixed.npy'
MY_COEXP_NET = '../../results/co-expression_net/coexp_net.npy' # (User Bambu Net)

# 출력 파일명
BRIDGED_NET_FILE = "../../results/co-expression_net/coexp_net_bridged.npy"

def bridge_network():
    print "Loading Data..."
    try:
        # 1. ID 리스트 로드
        train_genes = np.load(TRAIN_GENE_LIST) # 예: ['ENSG...1', 'ENSG...2']
        my_genes = np.load(MY_GENE_LIST)       # 예: ['ENSG...1', 'ENSG...3']
        
        # 2. 네트워크 로드
        # 주의: TRAIN_COEXP_NET은 원본 패키지에 포함된 파일이어야 함 (Train x Train)
        # 만약 원본 파일 경로가 다르면 수정 필요
        try:
            train_net = np.load(TRAIN_COEXP_NET)
        except:
            # 만약 원본 네트워크 파일이 없다면, 사용자 폴더에 복사해뒀던 것을 사용하거나 경로 확인
            print "[Check] Cannot find original Train Network. Assuming purely diagonal connection..."
            train_net = np.eye(len(train_genes))

        my_net = np.load(MY_COEXP_NET)
        
        n_train = len(train_genes)
        n_test = len(my_genes)
        n_total = n_train + n_test
        
        print " - Train: {} x {}".format(n_train, n_train)
        print " - Test:  {} x {}".format(n_test, n_test)

    except Exception as e:
        print "Error loading files: " + str(e)
        sys.exit(1)

    print "\nBuilding Bridged Network..."
    # 전체 매트릭스 초기화 (0으로)
    full_net = np.zeros((n_total, n_total), dtype=np.float32)

    # ---------------------------------------------------------
    # 1. [Train-Train] 구역 채우기 (좌측 상단)
    # ---------------------------------------------------------
    # 원래 훈련 데이터의 네트워크 정보를 그대로 넣습니다.
    # (만약 train_net 크기가 안 맞으면 에러 날 수 있으니 체크)
    if train_net.shape[0] == n_train:
        full_net[:n_train, :n_train] = train_net
    else:
        print "[Warning] Train network size mismatch! Skipping Train-Train block."

    # ---------------------------------------------------------
    # 2. [Test-Test] 구역 채우기 (우측 하단)
    # ---------------------------------------------------------
    # 사용자 데이터의 네트워크 정보를 넣습니다.
    full_net[n_train:, n_train:] = my_net

    # ---------------------------------------------------------
    # 3. [Train-Test] 다리 놓기 (핵심!)
    # ---------------------------------------------------------
    print "Bridging connections based on Gene ID match..."
    
    # Train 유전자들의 {GeneID: Index} 맵 생성 (검색 속도 향상)
    train_gene_map = {}
    for idx, gid in enumerate(train_genes):
        # 버전(.xx) 제거하고 매칭할지, 포함해서 할지 결정. 보통 유전자 레벨 매칭은 버전 제거가 안전.
        # 여기서는 정확한 매칭을 우선 시도.
        train_gene_map[gid] = idx
        
    # Test 유전자를 하나씩 순회하며 Train에 있는지 확인
    match_count = 0
    
    for i, test_gid in enumerate(my_genes):
        # 내 유전자가 훈련 데이터에도 있는가?
        if test_gid in train_gene_map:
            train_idx = train_gene_map[test_gid]
            
            # [논리]
            # Test_Isoform(i)은 Train_Gene(train_idx)과 같은 유전자다.
            # 따라서 Train_Gene이 가진 친구 관계(Row)를 Test_Isoform에게 복사해준다.
            
            # Train 쪽의 연결 정보를 가져옴 (Train_Gene <-> Other Train Genes)
            connections = train_net[train_idx, :]
            
            # 1. Test(Row) <-> Train(Col) 연결 (좌측 하단)
            full_net[n_train + i, :n_train] = connections
            
            # 2. Train(Row) <-> Test(Col) 연결 (우측 상단 - 대칭)
            full_net[:n_train, n_train + i] = connections
            
            match_count += 1
            
    print "  -> {} / {} user genes matched with training genes.".format(match_count, n_test)
    
    # 4. 저장
    print "Saving to {}...".format(BRIDGED_NET_FILE)
    np.save(BRIDGED_NET_FILE, full_net)
    print "Done! Please update 'model_fixed.py' to use this file."

if __name__ == "__main__":
    bridge_network()
