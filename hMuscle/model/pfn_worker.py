import numpy as np
import torch
import sys
import os
from tabpfn import TabPFNClassifier

def run_pfn_inference():
    # 1. 데이터 로드 (Numpy 포맷)
    try:
        x_support = np.load('temp_x_support.npy')
        y_support = np.load('temp_y_support.npy')
        x_query = np.load('temp_x_query.npy')
    except Exception as e:
        print(f"[PFN Worker Error] Failed to load temp files: {e}")
        sys.exit(1)

    # 2. TabPFN 초기화 (GPU 사용 권장)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # N_ensemble_configurations: 속도를 위해 4로 설정, 정확도를 위해 32까지 증가 가능
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=4)
    
    # 3. Support Set 등록 (Fit)
    # 데이터가 너무 많으면 메모리 에러가 날 수 있으므로 최대 2048개로 제한
    if len(x_support) > 2048:
        indices = np.random.choice(len(x_support), 2048, replace=False)
        x_support = x_support[indices]
        y_support = y_support[indices]

    classifier.fit(x_support, y_support)

    # 4. 예측 수행 (Batch 단위)
    batch_size = 1000
    preds = []
    
    for i in range(0, len(x_query), batch_size):
        batch = x_query[i : i+batch_size]
        if len(batch) == 0: break
        
        # Class 1(Positive)에 대한 확률만 추출
        p = classifier.predict_proba(batch)[:, 1]
        preds.append(p)
    
    final_preds = np.concatenate(preds)

    # 5. 결과 저장
    np.save('temp_pred_result.npy', final_preds)
    print("[PFN Worker] Inference Completed.")

if __name__ == "__main__":
    run_pfn_inference()
