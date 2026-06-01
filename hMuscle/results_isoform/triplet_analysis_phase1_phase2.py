# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import glob

def compare_phases():
    # 1. 분석 대상 GO 폴더 탐색
    go_dirs = sorted([d for d in glob.glob('GO_*') if os.path.isdir(d)])
    
    if not go_dirs:
        print("No GO directories found. Run this in the 'results' folder.")
        return

    for go_path in go_dirs:
        go_id = os.path.basename(go_path)
        print("\n" + "="*40)
        print(">>> Comparing Phase 1 vs Phase 2 for: " + go_id)
        
        # 파일 경로 설정
        p1_emb_path = os.path.join(go_path, 'phase1_triplet_only_embeddings.npy')
        p2_emb_path = os.path.join(go_path, 'phase2_joint_final_embeddings.npy')
        label_path = os.path.join(go_path, 'phase2_joint_final_labels.npy')

        # 파일 존재 여부 확인
        if not (os.path.exists(p1_emb_path) and os.path.exists(p2_emb_path)):
            print("    Skipping: Embedding files (p1 or p2) missing in " + go_id)
            continue

        try:
            # 데이터 로드
            p1_emb = np.load(p1_emb_path)
            p2_emb = np.load(p2_emb_path)
            labels = np.load(label_path).flatten()

            # 2. t-SNE 연산 (비교를 위해 동일한 random_state 사용)
            tsne = TSNE(n_components=2, random_state=42)
            
            print("    Computing t-SNE for Phase 1...")
            p1_tsne = tsne.fit_transform(p1_emb)
            
            print("    Computing t-SNE for Phase 2...")
            p2_tsne = tsne.fit_transform(p2_emb)

            # 3. 시각화 (1행 2열 구성)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Phase 1 Plot
            scatter1 = ax1.scatter(p1_tsne[:, 0], p1_tsne[:, 1], c=labels, 
                                   cmap='coolwarm', alpha=0.4, s=2)
            ax1.set_title(go_id + ' - Phase 1 (Triplet Only)')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')

            # Phase 2 Plot
            scatter2 = ax2.scatter(p2_tsne[:, 0], p2_tsne[:, 1], c=labels, 
                                   cmap='coolwarm', alpha=0.4, s=2)
            ax2.set_title(go_id + ' - Phase 2 (Joint Final)')
            ax2.set_xlabel('t-SNE 1')
            
            # 컬러바 추가 (라벨이 0, 1 섞여 있을 경우 대비)
            if len(np.unique(labels)) > 1:
                plt.colorbar(scatter2, ax=ax2, label='Functional Label')

            # 결과 저장
            output_name = os.path.join(go_path, go_id + '_phase_comparison_tsne.png')
            plt.tight_layout()
            plt.savefig(output_name, dpi=300)
            plt.close()
            
            print("    Success! Comparison map saved: " + output_name)

        except Exception as e:
            print("    Error processing {}: {}".format(go_id, str(e)))

if __name__ == "__main__":
    compare_phases()
