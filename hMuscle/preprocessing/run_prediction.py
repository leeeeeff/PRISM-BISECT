# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. CPU 모드 설정 (GPU 에러 방지)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. DIFFUSE 소스코드 경로 추가 (혹시 커스텀 레이어가 있을 경우 대비)
sys.path.append('/app/DIFFUSE/src') 

# =========================================================
# [경로 설정]
# =========================================================
# 모델이 있는 곳 (Docker 내부 경로)
MODEL_DIR = '/app/DIFFUSE/saved_models' 

# 사용자 데이터가 있는 곳 (상대 경로)
DATA_DIR = '../results/amino_seq'
SEQ_FILE = os.path.join(DATA_DIR, 'protein_sequences.npy')
ID_FILE  = os.path.join(DATA_DIR, 'protein_ids.npy')

# 결과 저장 경로
OUTPUT_FILE = '../results/final_prediction_results.csv'

def main():
    print("--- DIFFUSE Inference (Sequence Only) ---")

    # 1. 데이터 로드 확인
    if not os.path.exists(SEQ_FILE):
        print("Error: Data file missing at " + SEQ_FILE)
        sys.exit(1)

    print("Loading user data...")
    X = np.load(SEQ_FILE)
    ids = np.load(ID_FILE)
    print("Loaded " + str(len(ids)) + " sequences.")

    # 2. 사용할 모델(.h5) 찾기
    # 주의: saved_models에 있는 .h5 파일만 사용합니다. (.npy 가중치는 무시)
    model_files = glob.glob(os.path.join(MODEL_DIR, '*.h5'))
    
    if len(model_files) == 0:
        print("Error: No .h5 models found in " + MODEL_DIR)
        sys.exit(1)
        
    print("Found " + str(len(model_files)) + " models.")
    print("Models: " + str([os.path.basename(m) for m in model_files]))

    results = {} # 결과를 담을 그릇

    # 3. 모델별로 예측 수행
    for model_path in model_files:
        filename = os.path.basename(model_path)
        go_id = filename.split('_')[0] # 예: GO:0000387
        
        print("Predicting for " + go_id + "...")
        
        try:
            # compile=False 옵션은 학습 설정(loss 등)을 무시하고 구조만 가져옵니다.
            model = load_model(model_path, compile=False)
            
            # 예측 실행
            preds = model.predict(X, batch_size=32, verbose=0)
            
            # 확률값 추출 (Binary classification 가정)
            # shape이 (N, 2)이면 index 1이 Positive 확률
            if preds.shape[1] == 2:
                scores = preds[:, 1]
            else:
                scores = preds[:, 0]
            
            results[go_id] = scores
            
        except Exception as e:
            print("Warning: Failed to run " + go_id)
            print("Reason: " + str(e))

    # 4. 결과 CSV 저장
    if len(results) > 0:
        print("Saving results to " + OUTPUT_FILE)
        with open(OUTPUT_FILE, 'w') as f:
            # Header
            go_list = sorted(results.keys())
            header = "Isoform_ID," + ",".join(go_list) + "\n"
            f.write(header)
            
            # Rows
            for i in range(len(ids)):
                row = [str(ids[i])]
                for go_id in go_list:
                    val = results[go_id][i]
                    row.append("{:.4f}".format(val))
                f.write(",".join(row) + "\n")
                
        print("Done! Check ../results/final_prediction_results.csv")
    else:
        print("No predictions were made.")

if __name__ == "__main__":
    main()
