# -*- coding: utf-8 -*-
import os
import sys
import time
import urllib
import urllib2
from multiprocessing import Pool  # 병렬 처리를 위한 핵심 라이브러리

# [설정]
INPUT_PEP = '../data/top30k_isoforms.pep'
OUTPUT_TXT = '../data/domain/domain_list.txt'
API_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

# [중요] EBI 서버 차단 방지를 위해 동시 접속은 20개까지만 허용
MAX_WORKERS = 20 

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

        # 리스트 데이터 정리
        final_data = []
        for k in sequences:
            full_seq = "".join(sequences[k])
            clean_seq = full_seq.replace('*', '') # Stop codon 제거
            final_data.append((k, clean_seq)) # 튜플 형태로 저장 (ID, Sequence)

        return final_data
    except IOError:
        print "Error: Cannot read file " + file_path
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
    req = urllib2.Request(API_URL + '/run', data)
    try:
        response = urllib2.urlopen(req)
        return response.read()
    except urllib2.URLError as e:
        return None

def check_status(job_id):
    try:
        response = urllib2.urlopen(API_URL + '/status/' + job_id)
        return response.read()
    except:
        return "ERROR"

def get_result(job_id):
    try:
        response = urllib2.urlopen(API_URL + '/result/' + job_id + '/tsv')
        return response.read()
    except:
        return ""

# [핵심] 하나의 시퀀스를 처리하는 작업자 함수
def worker(item):
    seq_id, sequence = item
    
    # 로그가 너무 많이 찍히면 정신없으니, 가끔씩만 찍거나 생략
    # print "Processing: " + seq_id 
    
    # 1) 제출
    job_id = run_job("user@example.com", seq_id, sequence)
    if not job_id:
        return None

    # 2) 대기 (Polling)
    while True:
        status = check_status(job_id)
        if status == 'FINISHED':
            break
        elif status in ['ERROR', 'FAILURE', 'NOT_FOUND']:
            return None
        time.sleep(5) # 서버 부하 방지용 대기

    # 3) 결과 파싱
    tsv_text = get_result(job_id)
    domains = set()
    for line in tsv_text.split('\n'):
        cols = line.split('\t')
        if len(cols) > 4:
            acc = cols[4]
            if acc.startswith('PF'):
                domains.add(acc)
    
    # 결과 반환 (ID, 도메인리스트)
    if len(domains) > 0:
        print "[Done] " + seq_id + " -> Found " + str(len(domains))
        return (seq_id, list(domains))
    else:
        # 도메인 없어도 결과는 남김
        return (seq_id, [])

def main():
    print "--- [Parallel Scanner] InterProScan (Workers: {}) ---".format(MAX_WORKERS)

    if not os.path.exists(INPUT_PEP):
        print "Error: Input file not found."
        sys.exit(1)

    # 데이터 로딩
    data_list = parse_fasta(INPUT_PEP)
    total = len(data_list)
    print "Total Sequences to analyze: " + str(total)
    print "Starting parallel processing... (This may take a while)"

    # [병렬 처리 시작]
    # Pool을 만들어 20명이 동시에 작업하게 함
    pool = Pool(processes=MAX_WORKERS)
    
    # map 함수가 리스트의 각 아이템을 worker들에게 나눠줌
    results = pool.map(worker, data_list)
    
    pool.close()
    pool.join()

    # 결과 저장
    print "Saving results to " + OUTPUT_TXT
    with open(OUTPUT_TXT, 'w') as f:
        count = 0
        for res in results:
            if res is None: continue # 실패한 건 건너뜀
            
            seq_id, doms = res
            # 도메인이 없어도 기록할지, 있으면 기록할지 결정 (여기선 다 기록)
            line = seq_id + "\t" + " ".join(doms) + "\n"
            f.write(line)
            count += 1

    print "Done! Successfully processed {} sequences.".format(count)

if __name__ == "__main__":
    main()
