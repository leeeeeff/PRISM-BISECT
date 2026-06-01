# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from evaluation import calculate_metrics

# Annotation 파일 경로
HUMAN_ANNOTATION = '../data/raw_data/data/annotations/human_annotations.txt'
SWISSPROT_ANNOTATION = '../data/raw_data/data/annotations/swissprot_annotations.txt'
# [중요] 매핑 파일 경로 (util.py와 동일한 경로 사용)
MAPPING_FILE = '../data/raw_data/data/id_lists/ensembl_to_symbol.txt'

# --------------------------------------------------------------------------
# [NEW] ID 매핑 로더 (util.py의 기능을 가져옴)
# --------------------------------------------------------------------------
def load_mapping_dict(mapping_file_path):
    mapping = {}
    if not os.path.exists(mapping_file_path):
        print "[Warning] Mapping file not found at", mapping_file_path
        return mapping
    
    with open(mapping_file_path, 'r') as f:
        # 헤더가 있다면 건너뛰기 (파일 형식에 따라 다름)
        # next(f, None) 
        
        for line in f:
            parts = line.strip().split()
            # 포맷: [GeneID, Ver, TransID, Ver, Symbol] (util.py 기준)
            if len(parts) >= 5:
                ensembl_id = parts[0]   # ENSG...
                symbol = parts[4]       # GeneName
                mapping[ensembl_id] = symbol
                
                # 혹시 버전(.xx)이 붙은 ID가 들어올 경우를 대비해 Clean ID도 매핑
                clean_id = ensembl_id.split('.')[0]
                mapping[clean_id] = symbol
                
    print "[Info] Loaded %d ID mappings (Ensembl -> Symbol)" % len(mapping)
    return mapping

# 전역 변수로 매핑 로드
ID_MAP = load_mapping_dict(MAPPING_FILE)

def load_ground_truth(selected_go):
    positive_set = set()
    if not os.path.exists(HUMAN_ANNOTATION):
        print "Error: Annotation file not found at", HUMAN_ANNOTATION
        sys.exit(1)

    with open(HUMAN_ANNOTATION, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            if len(line) < 2: continue
            gene_id = line[0]
            go_terms = line[1:]
            if selected_go in go_terms:
                positive_set.add(gene_id)
                
    if os.path.exists(SWISSPROT_ANNOTATION):
        with open(SWISSPROT_ANNOTATION, 'r') as fr:
            for line in fr:
                line = line.strip().split('\t')
                if len(line) < 2: continue
                prot_id = line[0]
                go_terms = line[1:]
                if selected_go in go_terms:
                    positive_set.add(prot_id)
    return positive_set

def main():
    if len(sys.argv) < 3:
        print "Usage: python calc_perf.py <GO_TERM> <RESULT_FILE>"
        sys.exit(1)
        
    selected_go = sys.argv[1]
    result_file = sys.argv[2]
    
    print "========================================================"
    print " Evaluating Model Performance"
    print " GO Term: " + selected_go
    print "========================================================"
    
    print "[1/3] Loading Ground Truth Labels..."
    positive_set = load_ground_truth(selected_go)
    print "  - Found %d positive annotations" % len(positive_set)
    
    print "[2/3] Loading Prediction Results..."
    try:
        if not os.path.exists(result_file):
            print "Error: File not found:", result_file
            sys.exit(1)
        df = pd.read_csv(result_file, sep='\t', names=['GeneID', 'IsoformID', 'Score'])
    except Exception as e:
        print "Error loading result file: ", e
        sys.exit(1)
        
    print "[3/3] Calculating Metrics (with ID Mapping)..."
    
    # ---------------------------------------------------------
    # [핵심 수정] 매핑 사전을 이용하여 ID 변환 후 정답 확인
    # ---------------------------------------------------------
    def check_label_with_map(row):
        # 1. 원본 ID (Ensembl)
        raw_gene = str(row['GeneID'])
        clean_gene = raw_gene.split('.')[0]
        
        # 2. 매핑 시도 (Ensembl -> Symbol)
        # 매핑되면 Symbol을, 안 되면 원래 ID(Ensembl)를 씀
        mapped_gene_1 = ID_MAP.get(raw_gene, raw_gene)
        mapped_gene_2 = ID_MAP.get(clean_gene, clean_gene)
        
        # 3. 정답 확인 (Symbol로 비교)
        if (mapped_gene_1 in positive_set) or (mapped_gene_2 in positive_set):
            return 1
            
        # 4. 혹시 정답지가 Ensembl ID일 수도 있으니 원본도 확인
        if (raw_gene in positive_set) or (clean_gene in positive_set):
            return 1
            
        return 0

    df['Label'] = df.apply(check_label_with_map, axis=1)
    
    positive_matches = df['Label'].sum()
    print "  - Matches found in result file: %d" % positive_matches
    
    if positive_matches == 0:
        print "[Warning] Still 0 matches. Check if 'ensembl_to_symbol.txt' exists and is correct."

    metrics = calculate_metrics(df, k_list=[1, 3])
    
    print "\n" + "#"*30
    print "       FINAL REPORT       "
    print "#"*30
    print " AUROC            : %.4f" % metrics['AUROC']
    print " AUPRC            : %.4f" % metrics['AUPRC']
    print "------------------------------"
    print " Gene-wise Top-1  : %.4f" % metrics['Gene_Top-1_Acc']
    print " Gene-wise Top-3  : %.4f" % metrics['Gene_Top-3_Acc']
    print "#"*30 + "\n"

if __name__ == "__main__":
    main()
