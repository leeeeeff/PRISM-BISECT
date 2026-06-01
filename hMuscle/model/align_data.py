# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

# [설정] 파일 경로 (사용자 환경에 맞게 수정)
MY_SEQ_FILE = "../results/amino_seq/protein_sequences.npy"
MY_DOMAIN_FILE = "../results/domain/domain_matrix.npy"
MY_GENE_LIST = "../data/test_set/gene_list.npy"
MY_ISO_LIST = "../data/test_set/isoform_list.npy"

# 결과 저장 파일명
FIXED_SEQ_FILE = "my_sequence_matrix_fixed.npy"
FIXED_GENE_LIST = "my_gene_list_fixed.npy"
FIXED_ISO_LIST = "my_isoform_list_fixed.npy"

def align_data():
    print("Loading Data...")
    try:
        seq_data = np.load(MY_SEQ_FILE)
        dom_data = np.load(MY_DOMAIN_FILE)
        gene_list = np.load(MY_GENE_LIST)
        iso_list = np.load(MY_ISO_LIST)
    except Exception as e:
        print("Error loading files: " + str(e))
        sys.exit(1)

    print("Original Shapes:")
    print(" - Sequence: {}".format(seq_data.shape))
    print(" - Domain:   {}".format(dom_data.shape))
    print(" - ID List:  {}".format(len(iso_list)))

    n_seq = seq_data.shape[0]
    n_dom = dom_data.shape[0]

    if n_seq == n_dom:
        print("\n[OK] Sizes already match! No fix needed.")
        return

    print("\n[Mismatch Detected] Seq: {} vs Dom: {}".format(n_seq, n_dom))
    
    # ---------------------------------------------------------
    # [중요] 도메인 매트릭스가 더 짧은 경우 (누락된 경우)
    # ---------------------------------------------------------
    if n_seq > n_dom:
        diff = n_seq - n_dom
        print("The Domain Matrix is shorter by {} rows.".format(diff))
        print("Assuming the missing rows are at the end (or corresponding IDs are extra).")
        print("Trimming Sequence and ID lists to match Domain Matrix size...")
        
        # 앞에서부터 도메인 개수만큼만 자름 (주의: 순서가 1:1로 맞다고 가정)
        # 만약 순서가 섞여 있다면 이 방법은 위험하지만, 현재로선 최선입니다.
        
        seq_data_fixed = seq_data[:n_dom]
        gene_list_fixed = gene_list[:n_dom]
        iso_list_fixed = iso_list[:n_dom]
        
        # 저장
        np.save(FIXED_SEQ_FILE, seq_data_fixed)
        np.save(FIXED_GENE_LIST, gene_list_fixed)
        np.save(FIXED_ISO_LIST, iso_list_fixed)
        
        print("\n[Fixed] New files created:")
        print(" - " + FIXED_SEQ_FILE)
        print(" - " + FIXED_GENE_LIST)
        print(" - " + FIXED_ISO_LIST)
        print("Please update 'model_fixed.py' to use these FIXED files.")

    else:
        print("Error: Domain matrix is larger than Sequence matrix? This is unexpected.")

if __name__ == "__main__":
    align_data()
