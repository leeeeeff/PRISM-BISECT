# -*- coding: utf-8 -*-
import csv
import sys
import math

# [설정] 결과 파일 경로
RESULT_FILE = "GO_0030049_prediction_scores.txt"
TOP_PERCENTILE = 0.05 # 상위 5%

def load_data():
    print "Loading results..."
    data = {} # (GeneID, IsoformID) -> Max Score
    
    try:
        with open(RESULT_FILE, 'rb') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3: continue
                gene, iso, score_str = row[0], row[1], row[2]
                try:
                    score = float(score_str)
                except:
                    continue
                
                # 중복 제거 (최대 점수 유지)
                key = (gene, iso)
                if key in data:
                    if score > data[key]:
                        data[key] = score
                else:
                    data[key] = score
                    
    except Exception as e:
        print "Error reading file: " + str(e)
        sys.exit(1)
        
    print "Cleaned Count: {}".format(len(data))
    return data

def save_csv(filename, header, rows):
    try:
        with open(filename, 'w') as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                line = ",".join([str(x) for x in row])
                f.write(line + "\n")
        print "Saved -> " + filename
    except Exception as e:
        print "Error saving " + filename + ": " + str(e)

def analyze():
    # 1. 데이터 로드
    raw_data = load_data()
    
    # 리스트로 변환 [(Gene, Iso, Score), ...]
    all_rows = []
    for (gene, iso), score in raw_data.items():
        all_rows.append((gene, iso, score))
    
    # 점수 기준 내림차순 정렬
    all_rows.sort(key=lambda x: x[2], reverse=True)
    
    # 2. 상위 5% 추출
    count = len(all_rows)
    top_count = int(count * TOP_PERCENTILE)
    if top_count < 1: top_count = 1
    
    top_candidates = all_rows[:top_count]
    print "\nTop {}% Cutoff Score: {:.4f}".format(int(TOP_PERCENTILE*100), top_candidates[-1][2])
    
    # 저장: Analysis_Top_Candidates.csv
    save_csv("/GO_0030049/Analysis_Top_Candidates.csv", 
             ["GeneID", "IsoformID", "Score"], 
             top_candidates)
             
    # 3. Isoform Switching 분석
    print "\nAnalyzing Isoform Switching..."
    
    # 유전자별로 점수 모으기
    gene_map = {} # GeneID -> [(IsoID, Score), ...]
    for gene, iso, score in all_rows:
        if gene not in gene_map:
            gene_map[gene] = []
        gene_map[gene].append((iso, score))
        
    # 점수 차이(Diff) 계산
    switching_rows = []
    for gene, items in gene_map.items():
        if len(items) < 2: continue
        
        # items는 이미 정렬되어 있지 않으므로 점수 기준 정렬
        items.sort(key=lambda x: x[1], reverse=True)
        
        best_iso, best_score = items[0]
        worst_iso, worst_score = items[-1]
        
        diff = best_score - worst_score
        
        switching_rows.append((gene, diff, best_iso, best_score, worst_iso, worst_score))
        
    # Diff 기준 내림차순 정렬
    switching_rows.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 50개만 저장
    top_switching = switching_rows[:50]
    
    # 저장: Analysis_Isoform_Switching.csv
    save_csv("/GO_0030049/Analysis_Isoform_Switching.csv", 
             ["GeneID", "Score_Diff", "High_Score_Isoform", "High_Score", "Low_Score_Isoform", "Low_Score"], 
             top_switching)

if __name__ == "__main__":
    analyze()
