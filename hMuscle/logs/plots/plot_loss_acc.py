# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg') # 서버에서 창을 띄우지 않고 파일로 저장하는 설정
import matplotlib.pyplot as plt
import re
import sys
import glob

def parse_log(filename):
    losses = []
    accuracies = []
    
    # 로그 파일 읽기
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # 정규표현식으로 loss와 acc 숫자 추출
        # 패턴: "loss: 1.2345 - acc: 0.5678" 형식 찾기
        match = re.search(r'loss: (\d+\.\d+) - acc: (\d+\.\d+)', line)
        if match:
            loss = float(match.group(1))
            acc = float(match.group(2))
            losses.append(loss)
            accuracies.append(acc)
            
    return losses, accuracies

def plot_graph(losses, accuracies, filename):
    total_epochs = 50.0  # 총 에포크 수 (소수점 계산을 위해 .0 추가)
    count = len(losses)

    epochs = [(float(i) * total_epochs / count) for i in range(count)]

    # 그래프 그리기 (2개를 나란히)
    plt.figure(figsize=(12, 5))
    
    # 1. Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    plt.title('Loss Change over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 2. Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'r-', label='Training Accuracy')
    plt.title('Accuracy Change over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # 저장하기
    output_filename = filename.replace('.log', '_L2_fitted.png')
    plt.savefig(output_filename)
    print("Graph saved: " + output_filename)
    plt.close()

# ../logs 폴더에 있는 모든 로그 파일을 찾아서 그림
log_files = glob.glob('../GO_0022900_L2_fitted.log')

if not log_files:
    print("No log files found in ../logs/")
else:
    for log_file in log_files:
        print("Processing: " + log_file)
        l, a = parse_log(log_file)
        if len(l) > 0:
            plot_graph(l, a, log_file)
        else:
            print(" -> No data found inside.")
