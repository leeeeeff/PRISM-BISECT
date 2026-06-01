# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

# =========================================================
# [설정] 경로 수정 (현재 폴더: preprocessing/domain 기준)
# =========================================================

# 1. 도메인 매핑 파일들이 있는 폴더 (상대 경로)
# hMuscle/preprocessing/domain -> hMuscle/data/raw_data/data/raw_data/domain_data
RAW_DATA_DIR = '../../data/raw_data/data/raw_data/domain_data/'

# 2. 파일 이름 지정
MAPPING_FILE = os.path.join(RAW_DATA_DIR, 'domain_id_mapping.txt')
TRANSLATION_FILE_1 = os.path.join(RAW_DATA_DIR, 'human_CDD_query_results.txt')
TRANSLATION_FILE_2 = os.path.join(RAW_DATA_DIR, 'swissprot_CDD_query_results.txt')

# 3. 내 도메인 리스트 (Input)
# (상위 폴더인 preprocessing에 있다면 ../domain_list.txt 입니다)
INPUT_FILE = '../../data/domain/domain_list.txt'

# 4. 결과 저장 경로 (Output)
# (hMuscle/data 폴더에 저장)
OUTPUT_FILE = '../../data/domain/domain_matrix.npy'

# 5. 모델 고정 길이
MAX_LEN = 251

def load_translation_map():
    print "Build Translation Dictionary (Pfam -> PSSM_ID)..."
    acc_to_pssm = {}
    
    files = [TRANSLATION_FILE_1, TRANSLATION_FILE_2]
    
    for fpath in files:
        if not os.path.exists(fpath):
            print "Warning: Translation file not found -> " + fpath
            continue
            
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) < 8: continue
                
                pssm_id = parts[2]
                accession = parts[7]
                
                if accession.startswith('pfam'):
                    acc_to_pssm[accession] = pssm_id

    print " -> Learned " + str(len(acc_to_pssm)) + " translations."
    return acc_to_pssm

def load_index_map():
    print "Loading Model Index Mapping..."
    pssm_to_idx = {}
    
    if not os.path.exists(MAPPING_FILE):
        print "Error: Mapping file not found -> " + MAPPING_FILE
        sys.exit(1)
        
    with open(MAPPING_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            idx = int(parts[0])
            pssm_id = parts[1]
            pssm_to_idx[pssm_id] = idx
            
    print " -> Loaded " + str(len(pssm_to_idx)) + " model indices."
    return pssm_to_idx

def convert_and_pad(acc_to_pssm, pssm_to_idx):
    print "Converting User Domains..."
    
    matrix_list = []
    ids = []
    
    if not os.path.exists(INPUT_FILE):
        print "Error: Input file not found -> " + INPUT_FILE
        sys.exit(1)

    with open(INPUT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split('\t')
            seq_id = parts[0]
            ids.append(seq_id)
            
            dom_ints = []
            if len(parts) > 1 and parts[1].strip():
                dom_names = parts[1].split()
                
                for pf_id in dom_names:
                    query_acc = pf_id.replace('PF', 'pfam')
                    
                    if query_acc in acc_to_pssm:
                        pssm = acc_to_pssm[query_acc]
                        if pssm in pssm_to_idx:
                            dom_ints.append(pssm_to_idx[pssm])
            
            matrix_list.append(dom_ints)

    final_matrix = np.zeros((len(matrix_list), MAX_LEN), dtype=int)
    
    for i, row in enumerate(matrix_list):
        if len(row) == 0: continue
        length = len(row)
        if length > MAX_LEN:
             final_matrix[i, :] = row[:MAX_LEN]
        else:
             final_matrix[i, -length:] = row
             
    return final_matrix, ids

def main():
    acc_to_pssm = load_translation_map()
    pssm_to_idx = load_index_map()
    final_matrix, ids = convert_and_pad(acc_to_pssm, pssm_to_idx)
    
    print "\n=== Result Summary ==="
    print "Processed Sequences: " + str(len(ids))
    print "Matrix Shape: " + str(final_matrix.shape)
    
    non_zero_rows = np.count_nonzero(np.sum(final_matrix, axis=1))
    print "Sequences with valid mapped domains: " + str(non_zero_rows)
    
    # 저장 경로 폴더가 없으면 생성
    out_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(OUTPUT_FILE, final_matrix)
    print "Saved to " + OUTPUT_FILE

if __name__ == "__main__":
    main()
