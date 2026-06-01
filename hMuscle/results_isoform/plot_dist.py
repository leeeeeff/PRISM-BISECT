# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg') # 화면 없는 서버용 설정
import matplotlib.pyplot as plt
import glob
import os
import sys

def read_scores(filename):
    scores = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split('\t')
            # 포맷: ENSG...  Transcript...  Score
            if len(parts) >= 3:
                try:
                    score = float(parts[2]) # 3번째가 점수
                    scores.append(score)
                except ValueError:
                    continue 
    except Exception as e:
        print("Error reading {}: {}".format(filename, str(e)))
        
    return scores

def plot_distribution(data_dict):
    plt.figure(figsize=(10, 6))
    
    # 색상 리스트
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 히스토그램 그리기
    for i, (name, scores) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        
        # 겹쳐서 그리기 (alpha=0.4로 투명하게)
        plt.hist(scores, bins=50, range=(0, 1), alpha=0.4, 
                 label=name, color=color, edgecolor=color)

    plt.title('Score Distribution by GO Term', fontsize=15)
    plt.xlabel('Prediction Score (0.0 ~ 1.0)', fontsize=12)
    plt.ylabel('Count (Frequency)', fontsize=12)
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    
    # 0.5 기준선
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    
    # 저장
    output_name = 'score_Final_Integrated.png'
    plt.savefig(output_name)
    print("\nGraph saved: " + output_name)
    plt.close()

# --- [수정된 부분] 실행 로직 ---

# 현재 폴더(results) 아래의 'GO_'로 시작하는 폴더 안의 txt 파일을 찾음
# 패턴: ./GO_xxxxx/xxxxx_scores.txt
search_pattern = 'GO_*/GO_*_Final_Integrated_scores.txt'
files = glob.glob(search_pattern)

if not files:
    print("No score files found matching pattern: " + search_pattern)
    print("Current location: " + os.getcwd())
    print("Check if you are in the 'results' directory.")
else:
    print("Found {} files.".format(len(files)))
    
    all_data = {}
    for filename in files:
        # 라벨 추출: 파일명보다는 '폴더 이름'을 쓰는 게 더 깔끔함
        # 예: GO_0006096/GO_0006096_epoch_50_scores.txt -> GO_0006096 추출
        dir_name = os.path.dirname(filename)  # GO_0006096 (폴더 경로)
        label_name = os.path.basename(dir_name) # 폴더 이름만 추출
        
        # 만약 폴더 이름이 비어있으면(현재경로면) 파일명에서 추출
        if not label_name:
            label_name = os.path.basename(filename).split('_epoch')[0]

        print("Reading: {} (Label: {})".format(filename, label_name))
        scores = read_scores(filename)
        
        if len(scores) > 0:
            all_data[label_name] = scores
            print(" -> {} scores loaded.".format(len(scores)))
        else:
            print(" -> Empty or invalid format.")

    if all_data:
        plot_distribution(all_data)
    else:
        print("No valid data to plot.")
