# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

def read_scores(filename):
    scores = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Format: [ID] [Label] [Score] -> Extract the last column
                if len(parts) >= 3:
                    try:
                        score = float(parts[-1])
                        scores.append(score)
                    except ValueError:
                        continue
    except Exception as e:
        print("Error reading {}: {}".format(filename, str(e)))
    return scores

def plot_distribution(data_dict):
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (name, scores) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        plt.hist(scores, bins=50, range=(0, 1), alpha=0.4,
                 label=name, color=color, edgecolor=color)

    plt.title('Final Joint Model Score Distribution (Phase 2)', fontsize=15)
    plt.xlabel('Prediction Score (0.0 ~ 1.0)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    output_name = 'score_distribution_Phase2_Final.png'
    plt.savefig(output_name)
    print("\n[Success] Graph saved as: " + output_name)
    plt.close()

if __name__ == "__main__":
    # Target: phase2_joint_final_scores.txt in each GO folder
    search_pattern = 'GO_*/phase2_joint_final_scores.txt'
    files = glob.glob(search_pattern)

    if not files:
        print("No files found matching: " + search_pattern)
    else:
        print("Found {} files.".format(len(files)))
        all_data = {}
        for filename in files:
            dir_name = os.path.dirname(filename)
            label_name = os.path.basename(dir_name)
            
            print("Processing: " + label_name)
            scores = read_scores(filename)
            if scores:
                all_data[label_name] = scores

        if all_data:
            plot_distribution(all_data)
        else:
            print("No valid data found.")
