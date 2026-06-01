# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

# =========================================================
# [설정] 파일 경로
# =========================================================

EXPRESSION_FILE = '../../data/bambu_data/CPM_transcript.txt'
ID_LIST_FILE = '../../data/domain/domain_list.txt'
OUTPUT_FILE = '../../results/co-expression_net/coexp_net.npy'
POWER_BETA = 6

def build_network():
    print "--- [Co-expression Network Builder v3] ---"

    # 1. 기준 ID 로드 (꼬리표 제거)
    target_ids = []
    original_map = {} # 나중에 확인용
    
    print "Loading target IDs from: " + ID_LIST_FILE
    with open(ID_LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_id = line.split()[0]
                # [핵심 수정] .p1, .p2 등 TransDecoder 접미사 제거
                clean_id = raw_id.split('.p')[0]
                
                target_ids.append(clean_id)
                original_map[clean_id] = raw_id
    
    print "Target Isoforms: " + str(len(target_ids))
    if len(target_ids) > 0:
        print "Example ID (Raw): " + original_map[target_ids[0]]
        print "Example ID (Clean): " + target_ids[0]

    # 2. 발현량 데이터 로드
    print "Loading Expression Data from: " + EXPRESSION_FILE
    if not os.path.exists(EXPRESSION_FILE):
        print "Error: File not found!"
        sys.exit(1)

    exp_dic = {}
    success_count = 0
    
    with open(EXPRESSION_FILE, 'r') as f:
        header = f.readline() 
        for line in f:
            parts = line.strip().split() 
            if len(parts) < 3: continue
            
            # col 0: TXNAME (ID)
            # col 2~: Data
            tx_id = parts[0]
            
            try:
                vals = [float(x) for x in parts[2:]]
                exp_dic[tx_id] = vals
                success_count += 1
            except ValueError:
                continue
    
    print "Successfully loaded " + str(success_count) + " lines from CPM file."

    # 3. 매트릭스 정렬 (Alignment)
    print "Aligning matrix..."
    matrix = []
    missing_cnt = 0
    
    # 샘플 개수 확인
    if len(exp_dic) > 0:
        n_samples = len(exp_dic.values()[0])
    else:
        n_samples = 0
        
    for tid in target_ids:
        # 정확히 일치하는지 확인
        if tid in exp_dic:
            matrix.append(exp_dic[tid])
        else:
            # 못 찾으면 0으로 채움
            # print "Missing: " + tid # (디버깅용)
            matrix.append([0.0] * n_samples)
            missing_cnt += 1

    final_matrix = np.array(matrix)
    print "Aligned Matrix Shape: " + str(final_matrix.shape)
    
    if missing_cnt > 0:
        print "Warning: " + str(missing_cnt) + " IDs were missing (filled with 0)."
        if missing_cnt == len(target_ids):
             print "CRITICAL ERROR: Still no matches! Check ID format manually."
             sys.exit(1)
    else:
        print "Perfect! All IDs matched."

    # 4. 상관계수 계산
    print "Calculating Correlation..."
    cor_net = np.corrcoef(final_matrix)
    cor_net = np.nan_to_num(cor_net)

    # 5. Soft Thresholding
    print "Applying Soft Thresholding..."
    np.fill_diagonal(cor_net, 0)
    cor_net = np.power(np.abs(cor_net), POWER_BETA)

    # 6. 저장
    print "Saving to " + OUTPUT_FILE
    np.save(OUTPUT_FILE, cor_net)
    print "Done!"

if __name__ == "__main__":
    build_network()
