# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob

# 1. Custom Layer 로드를 위한 경로 설정
# 현재 위치가 results이므로 상위의 model 폴더를 경로에 추가
current_dir = os.getcwd()
model_dir = os.path.join(os.path.dirname(current_dir), 'model')
sys.path.append(model_dir)

from keras.models import load_model, Model
# PyramidPooling 클래스 임포트 (경로: model/layer/PyramidPooling.py)
from layer.PyramidPooling import PyramidPooling

def process_all_go_terms():
    # 대상 GO term 리스트
    target_go_terms = ['GO_0006096', 'GO_0006412', 'GO_0006936', 'GO_0022900']
    
    # 공통 데이터 로드
    try:
        test_seq = np.load('../model/my_sequence_matrix_fixed.npy')
        test_dm = np.load('domain/domain_matrix.npy')
        print("Success: Common sequence and domain data loaded.")
    except Exception as e:
        print("Error loading data: " + str(e))
        return

    for go_id in target_go_terms:
        print("\n" + "="*40)
        print(">>> Processing Original Embedding for: " + go_id)
        
        # [수정] 파일명이 DNN.h5가 아닐 경우를 대비해 가장 유사한 원본 모델 탐색
        model_path = '../saved_models/' + go_id + '_DNN.h5'
        if not os.path.exists(model_path):
            # 대안 파일명 확인 (예: L2_fitted_DNN 등)
            alt_paths = glob.glob('../saved_models/' + go_id + '*DNN.h5')
            if alt_paths:
                model_path = alt_paths[0]
            else:
                print("    Skipping: Model file not found for " + go_id)
                continue

        print("    Loading model from: " + model_path)
        
        try:
            from sklearn.manifold import TSNE
            # 2. Custom Object를 지정하여 모델 로드
            model = load_model(model_path, 
                               custom_objects={'PyramidPooling': PyramidPooling}, 
                               compile=False)
            
            # 임베딩 레이어 추출 (Concatenate 직후 레이어 타겟팅)
            embedding_layer_model = Model(inputs=model.input, 
                                          outputs=model.layers[-5].output)
            
            embs = embedding_layer_model.predict([test_seq, test_dm])

            print("    Computing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42)
            res = tsne.fit_transform(embs)

            # 시각화 및 저장
            plt.figure(figsize=(10, 8))
            plt.scatter(res[:, 0], res[:, 1], alpha=0.3, s=1, c='gray')
            plt.title('Original Pure Embedding (No Triplet) - ' + go_id)
            
            output_img = os.path.join(go_id, go_id + '_original_pure_tsne.png')
            if not os.path.exists(go_id): os.makedirs(go_id)
            
            plt.savefig(output_img)
            plt.close()
            np.save(os.path.join(go_id, 'original_pure_embeddings.npy'), embs)
            print("    Success! Saved: " + output_img)

        except Exception as e:
            print("    Error in {}: {}".format(go_id, str(e)))

if __name__ == "__main__":
    process_all_go_terms()
