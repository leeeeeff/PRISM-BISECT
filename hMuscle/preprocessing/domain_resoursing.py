# -*- coding: utf-8 -*-
import os
import sys
import time
import urllib
import urllib2

# =========================================================
# [설정] 경로 확인 (사용자님 환경 기준)
# =========================================================
INPUT_PEP = '../data/transcripts.fasta.transdecoder.pep'
# 결과 파일 (이 파일 뒤에 이어씁니다!)
OUTPUT_TXT = '../data/domain/domain_list.txt' 

# EBI API 설정
API_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Content-Type': 'application/x-www-form-urlencoded'
}

def get_existing_ids(output_file):
    """이미 완료된 ID 목록을 가져옴"""
    existing = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # ID는 첫 번째 컬럼
                    existing.add(line.split('\t')[0])
    return existing

def parse_fasta(file_path):
    sequences = {}
    current_id = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    current_id = line.split()[0].replace('>', '')
                    sequences[current_id] = []
                else:
                    if current_id:
                        sequences[current_id].append(line)
        
        for k in sequences:
            full_seq = "".join(sequences[k])
            sequences[k] = full_seq.replace('*', '') # Stop codon 제거
        return sequences
    except IOError:
        print "Error: Input file not found -> " + file_path
        sys.exit(1)

def run_job(email, title, sequence):
    params = {
        'email': email,
        'title': title,
        'sequence': sequence,
        'goterms': 'false',
        'pathways': 'false',
        'appl': 'PfamA'
    }
    data = urllib.urlencode(params)
    req = urllib2.Request(API_URL + '/run', data, headers=HEADERS)
    try:
        response = urllib2.urlopen(req)
        return response.read()
    except Exception as e:
        print "  Error submitting: " + str(e)
        return None

def check_status(job_id):
    try:
        req = urllib2.Request(API_URL + '/status/' + job_id, headers=HEADERS)
        response = urllib2.urlopen(req)
        return response.read()
    except:
        return "ERROR"

def get_result(job_id):
    try:
        req = urllib2.Request(API_URL + '/result/' + job_id + '/tsv', headers=HEADERS)
        response = urllib2.urlopen(req)
        return response.read()
    except:
        return ""

def main():
    print "--- [Resume Mode] Processing Missing Sequences ---"
    
    # 1. 이미 처리된 ID 확인
    finished_ids = get_existing_ids(OUTPUT_TXT)
    print "Currently Finished: " + str(len(finished_ids)) + " IDs"

    # 2. 전체 서열 읽기
    all_seqs = parse_fasta(INPUT_PEP)
    print "Total Input: " + str(len(all_seqs)) + " IDs"
    
    # 3. 누락된 ID만 선별 (Target)
    target_ids = []
    for seq_id in all_seqs:
        if seq_id not in finished_ids:
            target_ids.append(seq_id)
            
    print "\n>>> Target Missing IDs: " + str(len(target_ids)) + " <<<"
    if len(target_ids) == 0:
        print "All done! Nothing to process."
        return

    # 4. 누락된 것만 실행
    for i, seq_id in enumerate(target_ids):
        sequence = all_seqs[seq_id]
        print "[{}/{}] Retrying: {}".format(i+1, len(target_ids), seq_id)
        
        job_id = run_job("interpro_user@gmail.com", seq_id, sequence)
        
        if not job_id:
            print "  -> Submission Failed again."
            continue
            
        while True:
            status = check_status(job_id)
            if status == 'FINISHED': break
            if status in ['ERROR', 'FAILURE', 'NOT_FOUND']:
                print "  -> Job Failed."
                break
            time.sleep(3) 

        if status != 'FINISHED': continue

        tsv_text = get_result(job_id)
        domains = set()
        for line in tsv_text.split('\n'):
            cols = line.split('\t')
            if len(cols) > 4:
                acc = cols[4] 
                if acc.startswith('PF'): 
                    domains.add(acc)
        
        domain_list = list(domains)
        print "  -> Found " + str(len(domain_list)) + " domains."

        # [중요] 기존 파일 뒤에 이어쓰기 ('a' mode)
        with open(OUTPUT_TXT, 'a') as f:
            line = seq_id + "\t" + " ".join(domain_list) + "\n"
            f.write(line)
        
        time.sleep(1)

    print "\nRecovery Complete! Check the file line count again."

if __name__ == "__main__":
    main()
