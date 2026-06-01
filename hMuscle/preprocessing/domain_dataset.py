# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

# =========================================================
# [설정] 파일 경로
# =========================================================
# 1. 도메인 번역 사전 (Mapping File)
MAPPING_FILE = '/app/DIFFUSE/data/raw_data/domain_data/domain_id_mapping.txt'

# 2. 내 도메인 리스트 (Input)
INPUT_FILE = '../data/domain/domain_list.txt'

# 3. 결과 저장 경로 (Output)
OUTPUT_FILE = '../data//domain/domain_matrix.npy'

# 4. 모델 고정 길이 (Titin 때문에 생긴 251)
MAX_LEN = 251

def load_mapping():
    """도메인 이름 -> 숫자 ID 매핑 로드"""
    print "Loading domain mapping dictionary..."
    domain_dic = {}
    try:
        with open(MAPPING_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # 포맷: [숫자ID] [탭] [도메인이름]
                parts = line.split('\t')
                val = int(parts[0])
                key = parts[1]
                domain_dic[key] = val
        print " -> Loaded " + str(len(domain_dic)) + " mappings."
        return domain_dic
    except Exception as e:
        print "Error loading mapping file: " + str(e)
        sys.exit(1)

def convert_and_pad(domain_dic):
    """텍스트 도메인을 숫자로 변환하고 패딩 적용"""
    print "Converting domains to numbers..."
    
    matrix_list = []
    ids = []
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            seq_id = parts[0]
            ids.append(seq_id)
            
            # 도메인이 있는 경우
            if len(parts) > 1 and parts[1].strip():
                dom_names = parts[1].split()
                # 사전에 있는 것만 숫자로 변환
                dom_ints = []
                for d in dom_names:
                    if d in domain_dic:
                        dom_ints.append(domain_dic[d])
                    else:
                        print "Warning: Unknown domain '" + d + "' in " + seq_id
                
                # 원본 코드 로직: 마지막에 0 더했다가 빼는 부분은 
                # 데이터 파싱 상의 이슈 회피용이므로, 여기선 깔끔하게 숫자 리스트만 취함
                
            else:
                # 도메인 없는 경우
                dom_ints = []
            
            matrix_list.append(dom_ints)

    # Padding (Pre-padding: 앞을 0으로 채움)
    # Target Shape: (494, 251)
    final_matrix = np.zeros((len(matrix_list), MAX_LEN), dtype=int)
    
    for i, row in enumerate(matrix_list):
        if len(row) == 0:
            continue # 0으로 채워진 상태 그대로 유지
            
        # 251보다 길면 자르고, 짧으면 앞에 0을 채움
        # Keras pad_sequences(padding='pre')와 동일한 로직 수동 구현
        length = len(row)
        if length > MAX_LEN:
             # 너무 길면 뒤쪽 251개만 가져옴 (혹은 앞쪽, 모델 학습 방식에 따라 다름)
             # 보통 뒷부분이 중요할 수 있으나 여기선 앞에서부터 자름
             final_matrix[i, :] = row[:MAX_LEN]
        else:
             # 앞부분을 비우고 뒷부분에 채워넣음 (Pre-padding)
             # 예: [0, 0, ..., 0, dom1, dom2]
             final_matrix[i, -length:] = row
             
    return final_matrix, ids

def main():
    # 1. 매핑 로드
    domain_dic = load_mapping()
    
    # 2. 변환 및 패딩
    final_matrix, ids = convert_and_pad(domain_dic)
    
    # 3. 결과 확인
    print "\n=== Result Summary ==="
    print "Total Sequences: " + str(len(ids))
    print "Matrix Shape: " + str(final_matrix.shape)
    
    # 예시 출력 (첫 번째 데이터)
    print "\nFirst Row Example (Pre-padded):"
    print final_matrix[0]
    
    # 4. 저장
    print "\nSaving to " + OUTPUT_FILE
    np.save(OUTPUT_FILE, final_matrix)
    print "Done!"

if __name__ == "__main__":
    main()
