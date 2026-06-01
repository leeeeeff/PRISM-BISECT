# -*- coding: utf-8 -*-
import numpy as np
import sys
import os

# =========================================================
# [설정] 파일 경로 (사용자 환경에 맞게 수정 필수!)
# =========================================================
# 1. 족보 파일 (Bambu 결과)
COUNTS_FILE = "../../data/bambu_data/counts_transcript.txt"

# 2. 기준 파일 (도메인/시퀀스 행렬을 만들 때 썼던 최종 pep 파일)
INPUT_PEP = "../../data/top30k_isoforms.pep" 

# 3. 저장할 파일 경로
OUTPUT_GENE_LIST = "../../data/test_set/gene_list.npy"
OUTPUT_ISO_LIST = "../../data/test_set/isoform_list.npy"

def generate_id_lists():
    print("Step 1: Building Transcript-Gene Map from {}...".format(COUNTS_FILE))
    
    tx_to_gene = {}
    
    try:
        with open(COUNTS_FILE, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                # Bambu output: Col 0 = TXNAME, Col 1 = GENEID
                tx_id = parts[0]
                gene_id = parts[1]
                
                tx_to_gene[tx_id] = gene_id
                
    except IOError:
        print("Error: Count file not found.")
        sys.exit(1)
        
    print("  -> Loaded map for {} transcripts.".format(len(tx_to_gene)))

    # -----------------------------------------------------
    print("\nStep 2: Extracting IDs from PEP file (Preserving Order)...")
    
    ordered_gene_ids = []
    ordered_iso_ids = []
    
    missing_count = 0
    
    try:
        with open(INPUT_PEP, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    # PEP Header Parsing
                    # 예: >BambuTx123.p1 type=...
                    header_content = line.strip()[1:] # '>' 제거
                    full_id = header_content.split()[0] # 공백 전까지만 (ID 부분)
                    
                    # .p1, .p2 등 접미사 제거 (BambuTx123.p1 -> BambuTx123)
                    clean_id = full_id.rsplit('.p', 1)[0]
                    
                    # 매칭되는 Gene ID 찾기
                    if clean_id in tx_to_gene:
                        gene_id = tx_to_gene[clean_id]
                        ordered_iso_ids.append(clean_id)
                        ordered_gene_ids.append(gene_id)
                    else:
                        # 혹시 매칭 안 되면 원본 ID로 재시도
                        if full_id in tx_to_gene:
                            gene_id = tx_to_gene[full_id]
                            ordered_iso_ids.append(full_id)
                            ordered_gene_ids.append(gene_id)
                        else:
                            # 그래도 없으면 에러 혹은 'Unknown' 처리
                            print("Warning: No Gene ID found for " + clean_id)
                            missing_count += 1
                            ordered_iso_ids.append(clean_id)
                            ordered_gene_ids.append("Unknown_Gene")
                            
    except IOError:
        print("Error: PEP file not found at " + INPUT_PEP)
        sys.exit(1)

    # -----------------------------------------------------
    print("\nStep 3: Saving to .npy files...")
    
    # Numpy Array로 변환
    np_gene_ids = np.array(ordered_gene_ids)
    np_iso_ids = np.array(ordered_iso_ids)
    
    # 저장
    np.save(OUTPUT_GENE_LIST, np_gene_ids)
    np.save(OUTPUT_ISO_LIST, np_iso_ids)
    
    print("-" * 30)
    print("Done!")
    print("Final Count: {}".format(len(np_iso_ids)))
    print("Saved Gene IDs to: {}".format(OUTPUT_GENE_LIST))
    print("Saved Isoform IDs to: {}".format(OUTPUT_ISO_LIST))
    
    if missing_count > 0:
        print("Warning: {} IDs could not be mapped to a gene.".format(missing_count))

if __name__ == "__main__":
    generate_id_lists()
