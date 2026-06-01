# -*- coding: utf-8 -*-
import urllib
import urllib2
import time

# P53 단백질 서열
TEST_SEQ_ID = "TEST_P53"
TEST_SEQ = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

API_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

# [중요] 브라우저인 척 하는 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Content-Type': 'application/x-www-form-urlencoded'
}

def run_test():
    print "--- Test: Checking if InterProScan works (Email Fixed) ---"
    
    # 1. Job 제출
    params = {
        'email': 'interpro_user@gmail.com', # [수정됨] 실제 존재할 법한 이메일로 변경
        'title': TEST_SEQ_ID,
        'sequence': TEST_SEQ,
        'goterms': 'false',
        'pathways': 'false',
        'appl': 'PfamA' 
    }
    
    data = urllib.urlencode(params)
    
    # 헤더 포함 요청
    req = urllib2.Request(API_URL + '/run', data, headers=HEADERS)
    
    try:
        response = urllib2.urlopen(req)
        job_id = response.read()
        print "Job ID: " + job_id
    except urllib2.HTTPError as e:
        print "HTTP Error: " + str(e.code) + " " + str(e.reason)
        print "Response: " + e.read() 
        return
    except Exception as e:
        print "Error: " + str(e)
        return

    # 2. 대기
    while True:
        try:
            req_status = urllib2.Request(API_URL + '/status/' + job_id, headers=HEADERS)
            status = urllib2.urlopen(req_status).read()
            print "Status: " + status
            
            if status == 'FINISHED': break
            if status == 'FAILURE' or status == 'NOT_FOUND': return
            time.sleep(5)
        except Exception as e:
            print "Wait error: " + str(e)
            time.sleep(5)

    # 3. 결과 확인
    try:
        req_result = urllib2.Request(API_URL + '/result/' + job_id + '/tsv', headers=HEADERS)
        tsv = urllib2.urlopen(req_result).read()
        
        print "\n[Result Data Sample]"
        print tsv[:500] + "..." 
        
        if "PF" in tsv: 
            print "\n>>> SUCCESS! The code is working correctly. <<<"
        else:
            print "\n>>> WARNING! No Pfam domain found."
            
    except Exception as e:
        print "Error getting result: " + str(e)

if __name__ == "__main__":
    run_test()
