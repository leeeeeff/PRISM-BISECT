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
                if len(parts) >= 3:
                    try:
                        scores.append(float(parts[2]))
                    except ValueError:
                        continue
    except Exception as e:
        print("Error reading {}: {}".format(filename, str(e)))
    return scores

def plot_distribution(data_dict, version_tag, phase_tag):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, (name, scores) in enumerate(data_dict.items()):
        color = colors[i % len(colors)]
        plt.hist(scores, bins=50, range=(0, 1), alpha=0.4,
                 label=name, color=color, edgecolor=color)
    plt.title('Score Distribution by GO Term [{}] - {}'.format(version_tag, phase_tag), fontsize=15)
    plt.xlabel('Prediction Score (0.0 ~ 1.0)', fontsize=12)
    plt.ylabel('Count (Frequency)', fontsize=12)
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    output_name = '{}_{}_score_distribution.png'.format(version_tag, phase_tag)
    plt.savefig(output_name)
    print("Graph saved: " + output_name)
    plt.close()

VERSION = 'v2_integrated'
GO_TERMS = ['GO_0006412', 'GO_0006096', 'GO_0006936', 'GO_0022900']

# phase별로 각각 플롯
phases = [
    ('phase0_initial_untrained', 'phase0'),
    ('phase1_triplet_only',      'phase1'),
    ('phase2_joint_base',        'phase2'),
]

# Final도 별도 플롯
final_data = {}
for go in GO_TERMS:
    pattern = '{}/{}_260319/{}_*_Final_Integrated_scores.txt'.format(go, VERSION, VERSION)
    files = glob.glob(pattern)
    if files:
        scores = read_scores(files[0])
        if scores:
            final_data[go] = scores
            print("Final loaded: {} ({} scores)".format(go, len(scores)))

if final_data:
    plot_distribution(final_data, VERSION, 'Final_Integrated')

# Phase별 플롯
for phase_name, phase_tag in phases:
    phase_data = {}
    for go in GO_TERMS:
        pattern = '{}/{}_260319/{}_{}_scores.txt'.format(go, VERSION, VERSION, phase_name)
        files = glob.glob(pattern)
        if files:
            scores = read_scores(files[0])
            if scores:
                phase_data[go] = scores
                print("{} loaded: {} ({} scores)".format(phase_tag, go, len(scores)))
    if phase_data:
        plot_distribution(phase_data, VERSION, phase_tag)
    else:
        print("No data found for phase: " + phase_tag)
