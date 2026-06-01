# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # GUI 에러 방지
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import glob

def analyze_single_go(go_path):
    go_id = os.path.basename(go_path)
    print(">>> [{}] Processing...".format(go_id))
    
    try:
        # 1. 경로 설정
        p2_emb_path = os.path.join(go_path, 'phase2_joint_final_embeddings.npy')
        p2_lab_path = os.path.join(go_path, 'phase2_joint_final_labels.npy')
        p2_score_path = os.path.join(go_path, 'phase2_joint_final_scores.txt')

        if not (os.path.exists(p2_emb_path) and os.path.exists(p2_score_path)):
            print("    Skipping: Required files not found.")
            return None

        # 2. 데이터 로드 및 전처리
        p2_emb = np.load(p2_emb_path)
        p2_lab = np.load(p2_lab_path).flatten()
        
        # 스코어 파일: [ID] [Label] [Score] 구조
        p2_scores_df = pd.read_csv(p2_score_path, sep='\t', header=None)
        y_true = p2_scores_df.iloc[:, -2].values.flatten()
        y_score = p2_scores_df.iloc[:, -1].values.flatten()
        y_true_binary = (y_true > 0).astype(int)

        # 3. 성능 지표 계산 (에러 방지 로직 반영)
        auc, aupr = np.nan, np.nan
        unique_classes = np.unique(y_true_binary)
        
        if len(unique_classes) > 1:
            auc = roc_auc_score(y_true_binary, y_score)
            aupr = average_precision_score(y_true_binary, y_score)
        else:
            print("    Note: Only one class ({}) present. AUC/AUPR set to NaN.".format(unique_classes[0]))

        # 4. t-SNE 시각화 (임베딩 공간 분석)
        tsne = TSNE(n_components=2, random_state=42)
        p2_tsne = tsne.fit_transform(p2_emb)

        plt.figure(figsize=(8, 6))
        # 라벨이 1개여도 산점도는 그려지도록 설정
        scatter = plt.scatter(p2_tsne[:, 0], p2_tsne[:, 1], c=p2_lab, cmap='coolwarm', alpha=0.6)
        if len(unique_classes) > 1:
            plt.colorbar(scatter, label='Label')
        
        plt.title('{} | AUC: {:.3f}'.format(go_id, auc if not np.isnan(auc) else 0.0))
        plt.savefig(os.path.join(go_path, '{}_final_tsne.png'.format(go_id)))
        plt.close()

        return {'GO_ID': go_id, 'AUC': auc, 'AUPR': aupr, 'Samples': len(y_true)}

    except Exception as e:
        print("    Error in {}: {}".format(go_id, str(e)))
        return None

if __name__ == "__main__":
    target_dirs = sorted([d for d in glob.glob('GO_*') if os.path.isdir(d)])
    print("Found {} GO directories.".format(len(target_dirs)))
    
    results_list = []
    for d in target_dirs:
        res = analyze_single_go(d)
        if res: results_list.append(res)

    if results_list:
        summary_df = pd.DataFrame(results_list)
        summary_df.to_csv('total_performance_summary.csv', index=False)
        print("\n" + "="*30)
        print("Analysis Complete.")
        print(summary_df)
    else:
        print("No results to summarize.")
